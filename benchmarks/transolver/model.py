"""Transolver for unstructured meshes, following the reference irregular-mesh variant.

Reference: Wu et al., "Transolver: A Fast Transformer Solver for PDEs on General
Geometries", ICML 2024 (github.com/thuml/Transolver, Physics_Attention_Irregular_Mesh).
Cells are softly assigned to learned physical slices per head, slice tokens attend
to each other, and the result is broadcast back to cells. Mesh edges are not used.
"""

import torch
from torch import nn


def _activation(name):
    if name not in ("default", "tanh"):
        raise ValueError(f"Unsupported activation: {name}")
    return nn.Tanh() if name == "tanh" else nn.GELU()


class PhysicsAttention(nn.Module):
    """Multi-head slice -> attend -> deslice attention over one graph of cells."""

    def __init__(self, hidden_dim, num_slices=32, num_heads=4, dropout=0.0):
        super().__init__()
        if num_slices < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError(
                "Positive slice/head counts and a head-divisible hidden width are required."
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.temperature = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.5))
        self.in_project_x = nn.Linear(hidden_dim, hidden_dim)
        self.in_project_fx = nn.Linear(hidden_dim, hidden_dim)
        self.in_project_slice = nn.Linear(self.head_dim, num_slices)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout)
        )

    def _heads(self, values):
        batch, cells, _ = values.shape
        return values.reshape(batch, cells, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def forward(self, x):
        # x: [batch, cells, hidden]
        features = self._heads(self.in_project_fx(x))
        assignment_input = self._heads(self.in_project_x(x))
        slice_weights = torch.softmax(
            self.in_project_slice(assignment_input) / self.temperature, dim=-1
        )  # [batch, heads, cells, slices]
        slice_norm = slice_weights.sum(dim=2).unsqueeze(-1)
        tokens = torch.einsum("bhnc,bhng->bhgc", features, slice_weights)
        tokens = tokens / (slice_norm + 1e-5)
        q, k, v = self.to_q(tokens), self.to_k(tokens), self.to_v(tokens)
        attention = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        tokens = attention @ v
        cells = torch.einsum("bhgc,bhng->bhnc", tokens, slice_weights)
        return self.to_out(cells.transpose(1, 2).flatten(2))


class TransolverBlock(nn.Module):
    """Pre-norm transformer block with Physics-Attention."""

    def __init__(
        self, hidden_dim, num_slices=32, num_heads=4, mlp_ratio=2, activation="default"
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = PhysicsAttention(hidden_dim, num_slices, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            _activation(activation),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class Transolver(nn.Module):
    """Drop-in replacement for Milestone1_GNN's forward signature."""

    def __init__(
        self,
        geom_dim=5,
        design_dim=3,
        hidden_dim=32,
        num_layers=3,
        out_dim=1,
        num_slices=32,
        num_heads=4,
        mlp_ratio=2,
        activation="default",
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(geom_dim + design_dim, hidden_dim * 2),
            _activation(activation),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.placeholder = nn.Parameter(torch.rand(hidden_dim) / hidden_dim)
        self.blocks = nn.ModuleList(
            TransolverBlock(hidden_dim, num_slices, num_heads, mlp_ratio, activation)
            for _ in range(num_layers)
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, out_dim),
        )

    def _process_graph(self, h):
        h = h.unsqueeze(0)
        for block in self.blocks:
            h = block(h)
        return h.squeeze(0)

    def forward(
        self, geom, params, edge_index=None, edge_attr=None, batch_idx=None, areas=None
    ):
        h = self.encoder(torch.cat([geom, params], dim=-1)) + self.placeholder
        if batch_idx is None or int(batch_idx.max()) == 0:
            return self.decoder(self._process_graph(h))
        output = torch.zeros_like(h)
        for graph_id in torch.unique(batch_idx):
            mask = batch_idx == graph_id
            output[mask] = self._process_graph(h[mask])
        return self.decoder(output)
