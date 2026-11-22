"""AirfRANS node features matching the published Transolver data loader."""

from pathlib import Path

import numpy as np
import pyvista as pv

from benchmarks.airfrans.splits import parse_case


def build_features(
    name,
    points,
    velocity,
    pressure,
    viscosity,
    distance,
    surface_points,
    surface_normals,
):
    case = parse_case(name)
    points = np.asarray(points)[:, :2]
    velocity = np.asarray(velocity)[:, :2]
    surface = velocity[:, 0] == 0
    surface_positions = np.asarray(surface_points)[:, :2]
    surface_normals = np.asarray(surface_normals)[:, :2]
    surface_lookup = {
        tuple(point): index for index, point in enumerate(surface_positions)
    }
    indices = np.array(
        [surface_lookup.get(tuple(point), -1) for point in points[surface]]
    )
    if np.any(indices < 0):
        raise ValueError(f"Surface nodes do not match the internal mesh for {name}.")
    normals = np.zeros_like(velocity)
    normals[surface] = -surface_normals[indices]
    angle = np.deg2rad(case["angle"])
    inlet = np.broadcast_to(
        np.array([np.cos(angle), np.sin(angle)]) * case["speed"], points.shape
    )
    inputs = np.column_stack((points, inlet, -np.asarray(distance), normals))
    targets = np.column_stack((velocity, pressure, viscosity))
    if (
        not np.isfinite(inputs).all()
        or not np.isfinite(targets).all()
        or not surface.any()
    ):
        raise ValueError(f"Invalid node fields or missing surface for {name}.")
    return {
        "name": name,
        "pos": points.astype(np.float32),
        "x": inputs.astype(np.float32),
        "y": targets.astype(np.float32),
        "surface": surface,
    }


def load_case(root, name):
    folder = Path(root) / name
    internal = pv.read(folder / f"{name}_internal.vtu")
    airfoil = pv.read(folder / f"{name}_aerofoil.vtp")
    return build_features(
        name,
        internal.points,
        internal.point_data["U"],
        internal.point_data["p"],
        internal.point_data["nut"],
        internal.point_data["implicit_distance"],
        airfoil.points,
        airfoil.point_data["Normals"],
    )


def fit_normalization(cases):
    statistics = {}
    for field in ("x", "y"):
        count = sum(len(case[field]) for case in cases)
        total = np.zeros(cases[0][field].shape[1], dtype=np.float64)
        for case in cases:
            total += np.asarray(case[field], dtype=np.float64).sum(axis=0)
        mean = total / count
        variance = (
            sum(
                np.square(np.asarray(case[field], dtype=np.float64) - mean).sum(axis=0)
                for case in cases
            )
            / count
        )
        statistics[f"{field}_mean"] = mean.astype(np.float32)
        statistics[f"{field}_std"] = np.sqrt(variance).astype(np.float32)
    return statistics


def normalize_case(case, statistics):
    return {
        **case,
        "x": (case["x"] - statistics["x_mean"]) / (statistics["x_std"] + 1e-8),
        "y": (case["y"] - statistics["y_mean"]) / (statistics["y_std"] + 1e-8),
    }
