"""
The geometric deep learning architecture (Mesh Message Passing GNN).
"""

import torch
import torch.nn as nn

class MessagePassingBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, edge_index):
        src, dst = edge_index[0], edge_index[1]

        # 1. Message computed across face neighbors
        msg_in = torch.cat([h[src], h[dst]], dim=-1)
        msg = self.edge_mlp(msg_in)

        # 2. Sum aggregation into recipient face
        aggr = torch.zeros_like(h)
        aggr.index_add_(0, dst, msg)

        # 3. State update with residual connection
        h_next = h + self.node_mlp(torch.cat([h, aggr], dim=-1))
        return h_next


class Milestone1_GNN(nn.Module):
    def __init__(self, geom_dim=7, design_dim=3, hidden_dim=64, num_layers=4, out_dim=3):
        super().__init__()
        # Input: [x, y, z, nx, ny, nz, Area] + [P_D1_r, P_D2_r, P_V]
        self.encoder = nn.Sequential(
            nn.Linear(geom_dim + design_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layers = nn.ModuleList([
            MessagePassingBlock(hidden_dim) for _ in range(num_layers)
        ])

        # Output: [StaticPressure, WallShearStress, Density]
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, geom, params, edge_index):
        x = torch.cat([geom, params], dim=-1)
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.decoder(h)