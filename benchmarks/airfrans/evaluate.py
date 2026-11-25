"""Evaluate selected reduced AirfRANS models on held-out simulations."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from scipy.stats import spearmanr

from benchmarks.airfrans.data import load_case, normalize_case
from benchmarks.airfrans.evaluation import (
    force_coefficients,
    predict_full,
    pressure_scores,
)
from benchmarks.airfrans.splits import parse_case
from benchmarks.airfrans.training import BenchmarkModel


def load_models(run_dir, reference, device):
    models = {}
    config = json.loads((run_dir / "config.json").read_text())
    for kind in config["models"]:
        checkpoint = torch.load(
            run_dir / f"{kind}.pt", map_location=device, weights_only=True
        )
        model = BenchmarkModel(kind, reference).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models[kind] = model
    return models


def plot_case(case, predictions, path):
    points = case["pos"]
    near = (
        (points[:, 0] >= -0.5) & (points[:, 0] <= 1.5) & (np.abs(points[:, 1]) <= 0.8)
    )
    indices = np.flatnonzero(near)[:: max(1, int(near.sum()) // 25000)]
    actual = case["y"][:, 2]
    model_names = {
        "reference": "Reference Transolver",
        "local": "Local Transolver",
        "gnn": "Latent-attention GNN",
        "gnn_mean": "GNN, mean aggregation",
        "gnn_rel": "GNN, relative edges",
        "gnn_hybrid": "GNN, absolute + relative hybrid",
    }
    labels = ["CFD", *[model_names.get(kind, kind.title()) for kind in predictions]]
    fields = [actual, *[values[:, 2] for values in predictions.values()]]
    errors = [np.abs(values[:, 2] - actual) for values in predictions.values()]
    low, high = np.quantile(actual[indices], [0.01, 0.99])
    error_limit = max(
        float(np.quantile(np.concatenate([error[indices] for error in errors]), 0.99)),
        1e-8,
    )
    columns = len(fields)
    figure, axes = plt.subplots(
        2, columns, figsize=(5 * columns, 7), constrained_layout=True, squeeze=False
    )
    for axis, values, title in zip(axes[0], fields, labels):
        image = axis.scatter(
            points[indices, 0],
            points[indices, 1],
            c=values[indices],
            s=1,
            cmap="coolwarm",
            vmin=low,
            vmax=high,
            rasterized=True,
        )
        axis.set_title(title)
    figure.colorbar(
        image, ax=axes[0].tolist(), label="Reduced pressure (m²/s²)", shrink=0.8
    )
    axes[1, 0].axis("off")
    for axis, values, title in zip(
        axes[1, 1:],
        errors,
        [
            f"{model_names.get(kind, kind.title())} absolute error"
            for kind in predictions
        ],
    ):
        error_image = axis.scatter(
            points[indices, 0],
            points[indices, 1],
            c=values[indices],
            s=1,
            cmap="magma",
            vmin=0,
            vmax=error_limit,
            rasterized=True,
        )
        axis.set_title(title)
    figure.colorbar(
        error_image, ax=axes[1, 1:].tolist(), label="Absolute error (m²/s²)", shrink=0.8
    )
    for axis in list(axes[0]) + list(axes[1, 1:]):
        axis.set_aspect("equal")
        axis.set_xlim(-0.5, 1.5)
        axis.set_ylim(-0.8, 0.8)
    figure.suptitle(f"Held-out AirfRANS: {case['name']}")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def evaluate_case(name, root, models, statistics, sample_size, seed, device):
    raw = load_case(root, name)
    case = normalize_case(raw, statistics)
    internal = pv.read(root / name / f"{name}_internal.vtu")
    airfoil = pv.read(root / name / f"{name}_aerofoil.vtp")
    parameters = parse_case(name)
    true_force = force_coefficients(
        internal, airfoil, raw["surface"], parameters["speed"], parameters["angle"]
    )
    predictions, rows = {}, {}
    for kind, model in models.items():
        started = time.perf_counter()
        normalized, passes = predict_full(
            model, case, statistics, sample_size, seed, device
        )
        scores = pressure_scores(normalized, case["y"], case["surface"])
        values = normalized * (statistics["y_std"] + 1e-8) + statistics["y_mean"]
        values[raw["surface"], :2] = 0
        values[raw["surface"], 3] = 0
        force = force_coefficients(
            internal,
            airfoil,
            raw["surface"],
            parameters["speed"],
            parameters["angle"],
            values,
        )
        scores.update(
            true_lift=float(true_force[1]),
            predicted_lift=float(force[1]),
            lift_relative_error=float(abs((true_force[1] - force[1]) / true_force[1])),
            inference_passes=passes,
            elapsed_seconds=time.perf_counter() - started,
        )
        rows[kind] = scores
        predictions[kind] = values
    return raw, predictions, rows


def summarize(rows):
    summary = {}
    for kind, model_rows in rows.items():
        correlation = spearmanr(
            [row["true_lift"] for row in model_rows],
            [row["predicted_lift"] for row in model_rows],
        ).statistic
        summary[kind] = {
            "test_cases": len(model_rows),
            "volume_pressure_mse": float(
                np.mean([row["volume_pressure_mse"] for row in model_rows])
            ),
            "surface_pressure_mse": float(
                np.mean([row["surface_pressure_mse"] for row in model_rows])
            ),
            "mean_lift_relative_error": float(
                np.mean([row["lift_relative_error"] for row in model_rows])
            ),
            "lift_spearman": float(correlation) if np.isfinite(correlation) else None,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("public_benchmarks/airfans_Dataset/Dataset")
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("volume_results/airfrans_reduced_20261122/splits.json"),
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("public_benchmarks/transolver_upstream")
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--points",
        type=int,
        help="Nodes per inference pass (default: training value). Other values test "
        "stability to sampling density and write to evaluation_<points>/.",
    )
    args = parser.parse_args()
    config = json.loads((args.run_dir / "config.json").read_text())
    points = args.points or config["points"]
    if points < 1:
        parser.error("--points must be positive.")
    names = json.loads(args.splits.read_text())["splits"]["test"][: args.limit]
    with np.load(args.run_dir / "normalization.npz") as saved:
        statistics = dict(saved)
    device = torch.device("cuda")
    models = load_models(args.run_dir, args.reference, device)
    output = args.run_dir / (
        "evaluation" if points == config["points"] else f"evaluation_{points}"
    )
    output.mkdir(exist_ok=False)
    rows = {kind: [] for kind in models}
    for index, name in enumerate(names):
        raw, predictions, case_rows = evaluate_case(
            name,
            args.data,
            models,
            statistics,
            points,
            config["seed"] + index,
            device,
        )
        for kind, row in case_rows.items():
            rows[kind].append({"case": name, **row})
        np.savez(
            output / f"{name}.npz",
            positions=raw["pos"],
            actual_pressure=raw["y"][:, 2],
            **{
                f"{kind}_pressure": values[:, 2] for kind, values in predictions.items()
            },
        )
        if index < 3:
            plot_case(raw, predictions, output / f"{name}.png")
        print(f"Evaluated {index + 1}/{len(names)}: {name}", flush=True)
    summary = summarize(rows)
    (output / "metrics.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
