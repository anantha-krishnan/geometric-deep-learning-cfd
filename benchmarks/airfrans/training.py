"""Shared reduced AirfRANS protocol for Transolver and latent-attention GNN models."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


def positional_features(points, reference_size=8):
    grid_x = torch.linspace(
        -2, 4, reference_size, device=points.device, dtype=points.dtype
    )
    grid_y = torch.linspace(
        -1.5, 1.5, reference_size, device=points.device, dtype=points.dtype
    )
    coordinates = torch.stack(
        torch.meshgrid(grid_x, grid_y, indexing="ij"), dim=-1
    ).reshape(-1, 2)
    return torch.sqrt(
        torch.square(points[:, None, :] - coordinates[None, :, :]).sum(dim=-1)
    )


def physics_loss(predictions, targets, surface):
    if not surface.any() or surface.all():
        raise ValueError(
            "Training samples must contain both surface and volume points."
        )
    squared_error = torch.square(predictions - targets)
    return squared_error[surface].mean() + squared_error[~surface].mean()


# Latent-attention GNN kinds: (neighbour aggregation, edge feature mode).
GNN_KINDS = {
    "gnn": ("sum", "none"),
    "gnn_mean": ("mean", "none"),
    "gnn_rel": ("sum", "relative"),
    "gnn_hybrid": ("sum", "hybrid"),
}
MODEL_KINDS = ("reference", "local", *GNN_KINDS)


def relative_edge_features(positions, edge_index):
    """[unit direction, log(distance / chord)]; AirfRANS coordinates are in chords."""
    displacement = positions[edge_index[1]] - positions[edge_index[0]]
    distance = displacement.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.cat([displacement / distance, distance.log()], dim=1)


def hybrid_edge_features(positions, edge_index):
    """[unit direction, log distance, absolute dx/dy] for one graph."""
    relative = relative_edge_features(positions, edge_index)
    displacement = positions[edge_index[1]] - positions[edge_index[0]]
    return torch.cat([relative, displacement], dim=1)


class BenchmarkModel(nn.Module):
    def __init__(self, kind, reference_path=None):
        super().__init__()
        self.kind = kind
        self.uses_graph = kind in GNN_KINDS
        self.edge_mode = GNN_KINDS[kind][1] if self.uses_graph else "none"
        self.relative_edges = self.edge_mode in ("relative", "hybrid")
        if kind == "local":
            from benchmarks.transolver.model import Transolver

            self.model = Transolver(
                geom_dim=71,
                design_dim=0,
                hidden_dim=256,
                num_layers=8,
                out_dim=4,
                num_slices=32,
                num_heads=8,
                mlp_ratio=2,
            )
        elif kind in GNN_KINDS:
            from model import Milestone1_GNN

            aggregation, edge_mode = GNN_KINDS[kind]
            edge_dim = {"none": 0, "relative": 3, "hybrid": 5}[edge_mode]
            self.model = Milestone1_GNN(
                geom_dim=7,
                design_dim=0,
                hidden_dim=32,
                num_layers=3,
                out_dim=4,
                edge_dim=edge_dim,
                use_global=True,
                attention_tokens=32,
                attention_heads=4,
                aggregation=aggregation,
            )
        elif kind == "reference":
            if reference_path is None:
                raise ValueError("The authors' Transolver repository path is required.")
            path = (
                Path(reference_path)
                / "Airfoil-Design-AirfRANS"
                / "models"
                / "Transolver.py"
            )
            spec = importlib.util.spec_from_file_location(
                "airfrans_reference_transolver", path
            )
            if spec is None or spec.loader is None:
                raise ValueError(f"Cannot load reference model at {path}.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.model = module.Transolver(
                n_hidden=256,
                n_layers=8,
                space_dim=7,
                fun_dim=0,
                n_head=8,
                mlp_ratio=2,
                out_dim=4,
                slice_num=32,
                unified_pos=1,
            )
        else:
            raise ValueError(f"Unknown model kind: {kind}")

    def forward(self, inputs, positions, edge_index=None):
        if self.kind == "reference":
            return self.model(SimpleNamespace(x=inputs, pos=positions))
        if self.uses_graph:
            if edge_index is None:
                raise ValueError("The AirfRANS GNN requires mesh edge indices.")
            params = inputs.new_empty((len(inputs), 0))
            batch_idx = torch.zeros(len(inputs), dtype=torch.long, device=inputs.device)
            if self.edge_mode == "relative":
                edge_attr = relative_edge_features(positions, edge_index)
            elif self.edge_mode == "hybrid":
                edge_attr = hybrid_edge_features(positions, edge_index)
            else:
                edge_attr = None
            return self.model(inputs, params, edge_index, edge_attr, batch_idx)
        inputs = torch.cat((inputs, positional_features(positions)), dim=1)
        return self.model(inputs, inputs.new_empty((len(inputs), 0)))
