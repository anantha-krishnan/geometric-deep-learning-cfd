"""Shared reduced AirfRANS training protocol for both Transolver models."""

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


class BenchmarkModel(nn.Module):
    def __init__(self, kind, reference_path=None):
        super().__init__()
        self.kind = kind
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

    def forward(self, inputs, positions):
        if self.kind == "reference":
            return self.model(SimpleNamespace(x=inputs, pos=positions))
        inputs = torch.cat((inputs, positional_features(positions)), dim=1)
        return self.model(inputs, inputs.new_empty((len(inputs), 0)))
