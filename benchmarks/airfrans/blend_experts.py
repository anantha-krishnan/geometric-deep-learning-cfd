"""Offline blend of the count-weighted GNN and relative-edge GNN predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.airfrans.data import load_case
from benchmarks.airfrans.splits import parse_case


def pressure_metrics(actual, prediction, surface):
    return {
        "volume_pressure_mse": float(
            np.mean(np.square(prediction[~surface] - actual[~surface]))
        ),
        "surface_pressure_mse": float(
            np.mean(np.square(prediction[surface] - actual[surface]))
        ),
    }


def load_predictions(root, baseline_dir, relative_dir):
    names = sorted(path.stem for path in baseline_dir.glob("*.npz"))
    relative_names = {path.stem for path in relative_dir.glob("*.npz")}
    names = [name for name in names if name in relative_names]
    if not names:
        raise ValueError("The two evaluation directories have no matched cases.")

    with np.load(
        baseline_dir.parent / "normalization.npz", allow_pickle=False
    ) as saved:
        pressure_mean = float(saved["y_mean"][2])
        pressure_std = float(saved["y_std"][2])

    rows = []
    for name in names:
        with np.load(baseline_dir / f"{name}.npz", allow_pickle=False) as baseline:
            positions = baseline["positions"]
            actual = baseline["actual_pressure"]
            base_prediction = baseline["gnn_pressure"]
        with np.load(relative_dir / f"{name}.npz", allow_pickle=False) as relative:
            if not np.array_equal(positions, relative["positions"]):
                raise ValueError(f"Positions differ for {name}.")
            if not np.array_equal(actual, relative["actual_pressure"]):
                raise ValueError(f"CFD pressure differs for {name}.")
            rel_prediction = relative["gnn_rel_pressure"]
        raw = load_case(root, name)
        surface = raw["surface"]
        if len(actual) != len(surface):
            raise ValueError(f"Surface mask size differs for {name}.")
        rows.append(
            {
                "name": name,
                "angle": parse_case(name)["angle"],
                "surface": surface,
                "actual": (actual - pressure_mean) / (pressure_std + 1e-8),
                "baseline": (base_prediction - pressure_mean) / (pressure_std + 1e-8),
                "relative": (rel_prediction - pressure_mean) / (pressure_std + 1e-8),
            }
        )
    return rows


def evaluate_rows(rows, prediction_getter):
    per_case = []
    surface_errors = []
    volume_errors = []
    for row in rows:
        prediction = prediction_getter(row)
        metrics = pressure_metrics(row["actual"], prediction, row["surface"])
        per_case.append({"name": row["name"], "angle": row["angle"], **metrics})
        surface_errors.append(metrics["surface_pressure_mse"])
        volume_errors.append(metrics["volume_pressure_mse"])
    return {
        "mean_volume_pressure_mse": float(np.mean(volume_errors)),
        "median_volume_pressure_mse": float(np.median(volume_errors)),
        "mean_surface_pressure_mse": float(np.mean(surface_errors)),
        "median_surface_pressure_mse": float(np.median(surface_errors)),
        "per_case": per_case,
    }


def gate_rows(rows, threshold, low_expert, high_expert):
    def prediction(row):
        expert = low_expert if row["angle"] < threshold else high_expert
        return row[expert]

    return evaluate_rows(rows, prediction)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("public_benchmarks/airfans_Dataset/Dataset")
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "volume_results/airfrans_reduced_20261122/kaggle_gnn_latentattention32_398epoch/evaluation"
        ),
    )
    parser.add_argument(
        "--relative",
        type=Path,
        default=Path(
            "volume_results/airfrans_reduced_20261122/kaggle_gnn_rel_360epoch/evaluation"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("volume_results/airfrans_reduced_20261122/expert_blends_20261125"),
    )
    parser.add_argument("--alpha-start", type=float, default=0.1)
    parser.add_argument("--alpha-end", type=float, default=0.9)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    args = parser.parse_args()

    rows = load_predictions(args.data, args.baseline, args.relative)
    baseline = evaluate_rows(rows, lambda row: row["baseline"])
    relative = evaluate_rows(rows, lambda row: row["relative"])

    alpha_values = np.arange(args.alpha_start, args.alpha_end + 1e-9, args.alpha_step)
    blends = []
    for alpha in alpha_values:
        alpha = float(round(alpha, 10))
        result = evaluate_rows(
            rows,
            lambda row, alpha=alpha: (
                alpha * row["baseline"] + (1 - alpha) * row["relative"]
            ),
        )
        blends.append(
            {
                "alpha_baseline": alpha,
                **{key: value for key, value in result.items() if key != "per_case"},
            }
        )

    gates = []
    thresholds = np.arange(2.5, 17.6, 2.5)
    for threshold in thresholds:
        threshold = float(threshold)
        for low_expert, high_expert in (
            ("baseline", "relative"),
            ("relative", "baseline"),
        ):
            result = gate_rows(rows, threshold, low_expert, high_expert)
            gates.append(
                {
                    "threshold_angle": threshold,
                    "low_angle_expert": low_expert,
                    "high_angle_expert": high_expert,
                    **{
                        key: value for key, value in result.items() if key != "per_case"
                    },
                }
            )

    low = [row for row in rows if row["angle"] < 5.0]
    high = [row for row in rows if row["angle"] >= 5.0]
    angle_groups = {
        "threshold_angle": 5.0,
        "baseline": evaluate_rows(rows, lambda row: row["baseline"]),
        "relative": evaluate_rows(rows, lambda row: row["relative"]),
        "baseline_low_rel_high": gate_rows(rows, 5.0, "baseline", "relative"),
        "rel_low_baseline_high": gate_rows(rows, 5.0, "relative", "baseline"),
        "low_angle": {
            "count": len(low),
            "baseline": evaluate_rows(low, lambda row: row["baseline"]),
            "relative": evaluate_rows(low, lambda row: row["relative"]),
        },
        "high_angle": {
            "count": len(high),
            "baseline": evaluate_rows(high, lambda row: row["baseline"]),
            "relative": evaluate_rows(high, lambda row: row["relative"]),
        },
    }

    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "baseline": baseline,
        "relative": relative,
        "blends": blends,
        "gates": gates,
        "angle_groups": angle_groups,
    }
    (args.output / "blend_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    fields = [
        "alpha_baseline",
        "mean_volume_pressure_mse",
        "median_volume_pressure_mse",
        "mean_surface_pressure_mse",
        "median_surface_pressure_mse",
    ]
    with (args.output / "blend_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(blends)

    gate_fields = [
        "threshold_angle",
        "low_angle_expert",
        "high_angle_expert",
        "mean_volume_pressure_mse",
        "median_volume_pressure_mse",
        "mean_surface_pressure_mse",
        "median_surface_pressure_mse",
    ]
    with (args.output / "gate_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=gate_fields)
        writer.writeheader()
        writer.writerows(gates)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
