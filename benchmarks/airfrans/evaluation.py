"""AirfRANS full-node inference and reference-compatible evaluation."""

import numpy as np
import torch

from benchmarks.airfrans.data import induced_subgraph


def pressure_scores(predictions, targets, surface):
    error = np.square(predictions[:, 2] - targets[:, 2])
    return {
        "volume_pressure_mse": float(error[~surface].mean()),
        "surface_pressure_mse": float(error[surface].mean()),
    }


@torch.no_grad()
def predict_full(model, case, statistics, sample_size, seed, device):
    rng = np.random.default_rng(seed)
    count = len(case["x"])
    total = np.zeros((count, 4), dtype=np.float64)
    visits = np.zeros(count, dtype=np.int32)
    rounds = 0
    model.eval()
    while np.any(visits == 0):
        indices = rng.choice(count, min(sample_size, count), replace=False)
        inputs = torch.as_tensor(case["x"][indices], device=device)
        positions = torch.as_tensor(case["pos"][indices], device=device)
        if getattr(model, "uses_graph", False):
            edge_index = induced_subgraph(case["edge_index"], indices, count).to(device)
            areas = (
                torch.as_tensor(case["areas"][indices], device=device)
                if getattr(model, "uses_areas", False)
                else None
            )
            prediction = model(inputs, positions, edge_index, areas)
        else:
            prediction = model(inputs, positions)
        total[indices] += prediction.cpu().numpy()
        visits[indices] += 1
        rounds += 1
    predictions = (total / visits[:, None]).astype(np.float32)
    predictions[case["surface"], :2] = -statistics["y_mean"][:2] / (
        statistics["y_std"][:2] + 1e-8
    )
    predictions[case["surface"], 3] = -statistics["y_mean"][3] / (
        statistics["y_std"][3] + 1e-8
    )
    return predictions, rounds


def force_coefficients(internal, airfoil, surface, speed, angle, predictions=None):
    internal = internal.copy(deep=True)
    airfoil = airfoil.copy(deep=True)
    if predictions is not None:
        internal.point_data["U"][:, :2] = predictions[:, :2]
        internal.point_data["p"] = predictions[:, 2]
        internal.point_data["nut"] = predictions[:, 3]
    surface_points = internal.points[surface, :2]
    surface_lookup = {tuple(point): index for index, point in enumerate(surface_points)}
    indices = np.array(
        [surface_lookup.get(tuple(point), -1) for point in airfoil.points[:, :2]]
    )
    if np.any(indices < 0):
        raise ValueError("Airfoil points do not match the surface nodes.")
    internal = internal.compute_derivative(scalars="U", gradient="pred_grad")
    gradient = internal.point_data["pred_grad"].reshape(-1, 3, 3)[surface, :2, :2][
        indices
    ]
    strain = 0.5 * (gradient + gradient.transpose(0, 2, 1))
    strain -= strain.trace(axis1=1, axis2=2)[:, None, None] * np.eye(2)[None] / 3
    normals = -airfoil.point_data["Normals"][:, :2]
    airfoil.point_data["wallShearStress"] = (
        2 * 1.56e-5 * strain * normals[:, None, :]
    ).sum(axis=2)
    airfoil.point_data["p"] = internal.point_data["p"][surface][indices]
    airfoil = airfoil.point_data_to_cell_data(pass_point_data=True)
    lengths = airfoil.cell_data["Length"][:, None]
    pressure_force = (
        -airfoil.cell_data["p"][:, None] * airfoil.cell_data["Normals"][:, :2] * lengths
    ).sum(axis=0)
    shear_force = (airfoil.cell_data["wallShearStress"] * lengths).sum(axis=0)
    radians = np.deg2rad(angle)
    rotation = np.array(
        [[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]]
    )
    return 2 * (rotation @ (shear_force - pressure_force)) / speed**2
