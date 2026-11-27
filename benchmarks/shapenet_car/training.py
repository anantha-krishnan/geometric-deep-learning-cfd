"""ShapeNet-Car model adapters and reference field metrics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


def field_loss(predictions, targets, surface, pressure_weight=0.5):
    if not surface.any() or surface.all():
        raise ValueError("A case must contain surface and exterior points.")
    error = torch.square(predictions - targets)
    return error[:, :3].mean() + pressure_weight * error[surface, 3].mean()


def field_scores(predictions, targets, surface):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    surface = np.asarray(surface, dtype=bool)
    if predictions.shape != targets.shape or predictions.shape != (len(surface), 4):
        raise ValueError("Expected matching [nodes, 4] predictions and targets.")
    if not surface.any() or surface.all():
        raise ValueError("A case must contain surface and exterior points.")
    pressure = targets[surface, 3]
    velocity = targets[~surface, :3]
    pressure_norm = np.linalg.norm(pressure)
    velocity_norm = np.linalg.norm(velocity)
    return {
        "pressure_relative_l2": float(
            np.linalg.norm(predictions[surface, 3] - pressure) / pressure_norm
        )
        if pressure_norm
        else float("nan"),
        "velocity_relative_l2": float(
            np.linalg.norm(predictions[~surface, :3] - velocity) / velocity_norm
        )
        if velocity_norm
        else float("nan"),
    }


def hybrid_edge_features(positions, edge_index):
    displacement = positions[edge_index[1]] - positions[edge_index[0]]
    distance = displacement.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.cat((displacement / distance, distance.log(), displacement), dim=1)


class BenchmarkModel(nn.Module):
    def __init__(
        self,
        kind,
        reference_path=None,
        hidden_dim=None,
        num_layers=None,
        num_heads=None,
    ):
        super().__init__()
        self.kind = kind
        hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else (256 if kind == "reference" else 32)
        )
        num_layers = (
            num_layers if num_layers is not None else (8 if kind == "reference" else 3)
        )
        num_heads = (
            num_heads if num_heads is not None else (8 if kind == "reference" else 4)
        )
        if kind in ("gnn", "gnn_hybrid_slice"):
            from model import Milestone1_GNN

            self.model = Milestone1_GNN(
                geom_dim=7,
                design_dim=0,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                out_dim=4,
                use_global=True,
                attention_tokens=32,
                attention_heads=num_heads,
                edge_dim=7 if kind == "gnn_hybrid_slice" else 0,
                global_mixer="slice" if kind == "gnn_hybrid_slice" else "latent",
            )
        elif kind == "local_transolver":
            from benchmarks.transolver.model import Transolver

            self.model = Transolver(
                geom_dim=7,
                design_dim=0,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                out_dim=4,
                num_heads=num_heads,
                num_slices=32,
            )
        elif kind == "reference":
            if reference_path is None:
                raise ValueError("The authors' Transolver repository path is required.")
            path = (
                Path(reference_path)
                / "Car-Design-ShapeNetCar"
                / "models"
                / "Transolver.py"
            )
            spec = importlib.util.spec_from_file_location(
                "shapenet_car_reference", path
            )
            if spec is None or spec.loader is None:
                raise ValueError(f"Cannot load reference model at {path}.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.model = module.Model(
                n_hidden=hidden_dim,
                n_layers=num_layers,
                space_dim=7,
                fun_dim=0,
                n_head=num_heads,
                mlp_ratio=2,
                out_dim=4,
                slice_num=32,
                unified_pos=False,
            )
        else:
            raise ValueError(f"Unknown ShapeNet-Car model: {kind}")

    def forward(self, inputs, positions, edge_index=None):
        if self.kind == "reference":
            return self.model((SimpleNamespace(x=inputs, pos=positions), None))
        if self.kind in ("gnn", "gnn_hybrid_slice") and edge_index is None:
            raise ValueError("The GNN requires mesh edges.")
        edge_attr = (
            hybrid_edge_features(positions, edge_index)
            if self.kind == "gnn_hybrid_slice"
            else None
        )
        return self.model(
            inputs, inputs.new_empty((len(inputs), 0)), edge_index, edge_attr
        )
