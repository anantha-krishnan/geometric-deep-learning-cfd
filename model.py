"""
The geometric deep learning architecture (Mesh Message Passing GNN).
"""

import torch
from torch import nn


class MessagePassingBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        edge_dim=0,
        use_global=False,
        attention_heads=0,
        activation="default",
        area_aggregation=False,
    ):
        super().__init__()
        self.use_global = use_global
        self.attention_heads = attention_heads
        self.area_aggregation = area_aggregation
        if attention_heads and hidden_dim % attention_heads:
            raise ValueError(
                "Hidden width must be divisible by neighbor-attention heads."
            )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh() if activation == "tanh" else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        node_in = hidden_dim * (3 if use_global else 2)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh() if activation == "tanh" else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if attention_heads:
            self.message_attention = nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
                nn.Tanh() if activation == "tanh" else nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, attention_heads),
            )

    def forward(self, h, edge_index, edge_attr=None, batch_idx=None, areas=None):
        src, dst = edge_index[0], edge_index[1]
        node_weights = None
        if self.area_aggregation:
            if areas is None:
                raise ValueError("Area-weighted aggregation requires cell areas.")
            node_weights = areas.to(h.dtype)

        # 1. Message computed across face neighbors
        parts = [h[src], h[dst]] + ([edge_attr] if edge_attr is not None else [])
        msg = self.edge_mlp(torch.cat(parts, dim=-1))

        # 2. Aggregate by sum, area quadrature, or normalized edge-aware attention.
        if self.attention_heads:
            scores = self.message_attention(torch.cat(parts, dim=-1))
            if node_weights is not None:
                scores = scores + node_weights.clamp_min(1e-30).log()[src].unsqueeze(1)
            indices = dst.unsqueeze(1).expand(-1, self.attention_heads)
            maxima = h.new_full((h.size(0), self.attention_heads), -torch.inf)
            maxima.scatter_reduce_(0, indices, scores, reduce="amax", include_self=True)
            exp_scores = (scores - maxima[dst]).exp()
            normalizer = h.new_zeros(h.size(0), self.attention_heads)
            normalizer.index_add_(0, dst, exp_scores)
            weights = exp_scores / normalizer[dst].clamp_min(1e-12)
            values = msg.reshape(
                -1, self.attention_heads, h.size(1) // self.attention_heads
            )
            aggr_heads = h.new_zeros(
                h.size(0), self.attention_heads, h.size(1) // self.attention_heads
            )
            aggr_heads.index_add_(0, dst, weights.unsqueeze(-1) * values)
            aggr = aggr_heads.reshape_as(h)
        elif node_weights is not None:
            # Neighbour-area quadrature: sum_j A_j m_ji / sum_j A_j.
            weights = node_weights[src]
            aggr = torch.zeros_like(h).index_add_(0, dst, weights.unsqueeze(1) * msg)
            total = h.new_zeros(h.size(0)).index_add_(0, dst, weights)
            aggr = aggr / total.clamp_min(1e-30).unsqueeze(1)
        else:
            aggr = torch.zeros_like(h)
            aggr.index_add_(0, dst, msg)

        # 3. State update with residual connection
        node_in = [h, aggr]
        if self.use_global:
            node_in.append(_graph_mean(h, batch_idx, node_weights)[batch_idx])
        h_next = h + self.node_mlp(torch.cat(node_in, dim=-1))
        return h_next


def _graph_mean(h, batch_idx, weights=None):
    """Per-graph mean of node states, area-weighted when ``weights`` is given."""
    num_graphs = int(batch_idx.max().item()) + 1
    if weights is None:
        weights = torch.ones_like(batch_idx, dtype=h.dtype)
    weights = weights.to(h.dtype)
    sums = h.new_zeros(num_graphs, h.size(1)).index_add_(
        0, batch_idx, weights.unsqueeze(1) * h
    )
    totals = h.new_zeros(num_graphs).index_add_(0, batch_idx, weights)
    return sums / totals.clamp_min(1e-30).unsqueeze(1)


class LatentGlobalAttention(nn.Module):
    """Exchange global information through a small, graph-isolated token set.

    The first ``area_heads`` heads read cells with a log-area key bias, so their token
    pool is a quadrature of a field integral; the remaining heads weight cells equally.
    """

    def __init__(
        self,
        hidden_dim,
        num_tokens=32,
        num_heads=4,
        activation="default",
        area_heads=0,
    ):
        super().__init__()
        if num_tokens < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError(
                "Positive token/head counts and a head-divisible hidden width are required."
            )
        if not 0 <= area_heads <= num_heads:
            raise ValueError(
                "Area-weighted heads must be between 0 and the head count."
            )
        self.area_heads = area_heads
        self.num_heads = num_heads
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        self.cell_norm = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.read_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.token_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.write_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.token_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh() if activation == "tanh" else nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.cell_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh() if activation == "tanh" else nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def _read_bias(self, areas):
        """Per-head additive key bias beta_h * log(A), shape [heads, tokens, cells]."""
        log_area = areas.clamp_min(1e-30).log()
        log_area = log_area - log_area.mean()
        beta = (torch.arange(self.num_heads, device=areas.device) < self.area_heads).to(
            log_area.dtype
        )
        bias = beta[:, None] * log_area[None, :]
        return (
            bias.unsqueeze(1)
            .expand(-1, self.tokens.size(1), -1)
            .to(self.tokens.dtype)
            .contiguous()
        )

    def _attend_graph(self, cells, areas=None):
        normalized = self.cell_norm(cells).unsqueeze(0)
        bias = self._read_bias(areas) if self.area_heads else None
        tokens = (
            self.tokens
            + self.read_attention(
                self.token_norm(self.tokens),
                normalized,
                normalized,
                attn_mask=bias,
                need_weights=False,
            )[0]
        )
        token_input = self.token_norm(tokens)
        tokens = (
            tokens
            + self.token_attention(
                token_input,
                token_input,
                token_input,
                need_weights=False,
            )[0]
        )
        tokens = tokens + self.token_ffn(tokens)
        token_input = self.token_norm(tokens)
        updates = self.write_attention(
            normalized,
            token_input,
            token_input,
            need_weights=False,
        )[0].squeeze(0)
        cells = cells + updates
        return cells + self.cell_ffn(cells)

    def forward(self, cells, batch_idx, areas=None):
        if self.area_heads and areas is None:
            raise ValueError("Area-weighted attention heads require cell areas.")
        output = torch.zeros_like(cells)
        for graph_id in torch.unique(batch_idx):
            mask = batch_idx == graph_id
            graph_areas = areas[mask] if areas is not None and self.area_heads else None
            output[mask] = self._attend_graph(cells[mask], graph_areas)
        return output


class Milestone1_GNN(nn.Module):
    def __init__(
        self,
        geom_dim=7,
        design_dim=3,
        hidden_dim=64,
        num_layers=4,
        out_dim=2,
        edge_dim=0,
        use_global=False,
        attention_tokens=0,
        attention_heads=4,
        message_attention_heads=0,
        activation="default",
        area_heads=0,
        area_aggregation=False,
    ):
        super().__init__()
        if activation not in ("default", "tanh"):
            raise ValueError(f"Unsupported activation: {activation}")
        if area_heads and not attention_tokens:
            raise ValueError("Area-weighted heads require latent attention tokens.")
        self.activation = activation
        self.area_heads = area_heads
        self.area_aggregation = area_aggregation
        if attention_tokens < 0:
            raise ValueError("Attention token count cannot be negative.")
        self.attention_tokens = attention_tokens
        self.attention_heads = attention_heads
        self.message_attention_heads = message_attention_heads
        # Input: [x, y, z, nx, ny, nz, Area] + [P_D1_r, P_D2_r, P_V]
        self.encoder = nn.Sequential(
            nn.Linear(geom_dim + design_dim, hidden_dim),
            nn.Tanh() if activation == "tanh" else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            [
                MessagePassingBlock(
                    hidden_dim,
                    edge_dim,
                    use_global,
                    message_attention_heads,
                    activation,
                    area_aggregation,
                )
                for _ in range(num_layers)
            ]
        )

        # Output: [StaticPressure, WallShearStress]
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh() if activation == "tanh" else nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.attention_layers = nn.ModuleList(
            [
                LatentGlobalAttention(
                    hidden_dim,
                    attention_tokens,
                    attention_heads,
                    activation,
                    area_heads,
                )
                for _ in range(num_layers)
            ]
            if attention_tokens
            else []
        )

    def forward(
        self, geom, params, edge_index, edge_attr=None, batch_idx=None, areas=None
    ):
        """``areas`` are physical cell measures, required by the area-weighted options."""
        x = torch.cat([geom, params], dim=-1)
        h = self.encoder(x)
        if batch_idx is None:
            batch_idx = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        for index, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_attr, batch_idx, areas)
            if self.attention_tokens:
                h = self.attention_layers[index](h, batch_idx, areas)
        return self.decoder(h)
