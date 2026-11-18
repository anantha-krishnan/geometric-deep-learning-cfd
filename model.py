"""
The geometric deep learning architecture (Mesh Message Passing GNN).
"""

import torch
from torch import nn


class MessagePassingBlock(nn.Module):
    def __init__(self, hidden_dim, edge_dim=0, use_global=False):
        super().__init__()
        self.use_global = use_global
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        node_in = hidden_dim * (3 if use_global else 2)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, edge_index, edge_attr=None, batch_idx=None):
        src, dst = edge_index[0], edge_index[1]

        # 1. Message computed across face neighbors
        parts = [h[src], h[dst]] + ([edge_attr] if edge_attr is not None else [])
        msg = self.edge_mlp(torch.cat(parts, dim=-1))

        # 2. Sum aggregation into recipient face
        aggr = torch.zeros_like(h)
        aggr.index_add_(0, dst, msg)

        # 3. State update with residual connection
        node_in = [h, aggr]
        if self.use_global:
            node_in.append(_graph_mean(h, batch_idx)[batch_idx])
        h_next = h + self.node_mlp(torch.cat(node_in, dim=-1))
        return h_next


def _graph_mean(h, batch_idx):
    num_graphs = int(batch_idx.max().item()) + 1
    sums = h.new_zeros(num_graphs, h.size(1)).index_add_(0, batch_idx, h)
    counts = h.new_zeros(num_graphs).index_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=h.dtype))
    return sums / counts.unsqueeze(1)


class Milestone1_GNN(nn.Module):
    def __init__(self, geom_dim=7, design_dim=3, hidden_dim=64, num_layers=4, out_dim=2,
                 edge_dim=0, use_global=False):
        super().__init__()
        # Input: [x, y, z, nx, ny, nz, Area] + [P_D1_r, P_D2_r, P_V]
        self.encoder = nn.Sequential(
            nn.Linear(geom_dim + design_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layers = nn.ModuleList([
            MessagePassingBlock(hidden_dim, edge_dim, use_global) for _ in range(num_layers)
        ])

        # Output: [StaticPressure, WallShearStress]
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, geom, params, edge_index, edge_attr=None, batch_idx=None):
        x = torch.cat([geom, params], dim=-1)
        h = self.encoder(x)
        if batch_idx is None:
            batch_idx = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, batch_idx)
        return self.decoder(h)