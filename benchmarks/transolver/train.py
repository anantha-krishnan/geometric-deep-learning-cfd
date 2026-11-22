"""Train the Transolver benchmark on the same split as the volume-architecture study.

Example:
    uv run python -m benchmarks.transolver.train --design Design_1

The outer fold, inner validation split, near-body loss, early stopping and metrics
are reused from compare_volume_architectures.py so the numbers are directly
comparable. Results go to volume_results/benchmarks/<name>/ and include, per held-out
design, an Actual | reference(s) | Transolver PNG with pressure and |error| rows.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import compare_volume_architectures as experiment
import train_volume as training
from benchmarks.transolver.model import Transolver
from benchmarks.transolver.plots import near_body_bounds, plot_benchmark_comparison
from dataset import load_all_starccm_volume_designs
from validate_volume import read_mesh

PROJECT_ROOT = experiment.PROJECT_ROOT
LATENT_ATTENTION_DIR = (
    PROJECT_ROOT / "volume_results" / "architecture_validation" / "latent_attention32"
)
DEFAULT_COMPARISONS = (
    f"Near-body GNN={experiment.DEFAULT_REFERENCE}",
    f"Latent attention 32={LATENT_ATTENTION_DIR}",
)


def create_transolver(architecture, device):
    return Transolver(
        geom_dim=architecture["geom_dim"],
        design_dim=3,
        hidden_dim=architecture["hidden_dim"],
        num_layers=architecture["num_layers"],
        out_dim=1,
        num_slices=architecture["num_slices"],
        num_heads=architecture["attention_heads"],
        mlp_ratio=architecture["mlp_ratio"],
        activation=architecture["activation"],
    ).to(device)


def build_architecture(args):
    architecture = {
        "name": args.name or f"transolver{args.hidden_dim}",
        "model": "transolver",
        "geom_dim": 5,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_slices": args.num_slices,
        "attention_heads": args.attention_heads,
        "mlp_ratio": args.mlp_ratio,
        "activation": args.activation,
    }
    architecture["parameter_count"] = sum(
        parameter.numel()
        for parameter in create_transolver(
            architecture, torch.device("cpu")
        ).parameters()
    )
    return architecture


def parse_comparisons(entries):
    comparisons = {}
    for entry in entries:
        label, separator, directory = entry.partition("=")
        if not separator or not label.strip() or not directory.strip():
            raise ValueError(f"Comparison must be LABEL=DIR, got {entry!r}.")
        path = Path(directory.strip())
        if not path.is_dir():
            print(f"Skipping missing comparison {label!r}: {path}", flush=True)
            continue
        comparisons[label.strip()] = path
    if not comparisons:
        raise ValueError("At least one existing comparison directory is required.")
    return comparisons


def load_prediction(directory, item):
    folder = directory / "predictions"
    path = (folder if folder.is_dir() else directory) / f"{item['design_name']}.npz"
    with np.load(path, allow_pickle=False) as saved:
        actual = saved["actual_pressure_pa"].reshape(-1)
        predicted = saved["predicted_pressure_pa"].reshape(-1)
    if not np.allclose(actual, item["pressure"].numpy().reshape(-1)):
        raise ValueError(f"{path} does not match the held-out pressure field.")
    return predicted


def attach_scalers(checkpoint_path, scalers):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for name, scaler in zip(
        ("geom_scaler", "params_scaler", "pressure_scaler"), scalers
    ):
        checkpoint[name] = {"mean": scaler.mean.cpu(), "std": scaler.std.cpu()}
    torch.save(checkpoint, checkpoint_path)


def plot_all_designs(test_data, comparisons, architecture, output_dir, data_dir):
    """Write the multi-model PNG per design and return per-model metric rows."""
    rows = {label: [] for label in [*comparisons, architecture["name"]]}
    for item in test_data:
        predictions = {
            label: load_prediction(directory, item)
            for label, directory in comparisons.items()
        }
        with np.load(output_dir / f"{item['design_name']}.npz") as saved:
            predictions[architecture["name"]] = saved["predicted_pressure_pa"]
        metrics = {
            label: training.measure_design(
                torch.from_numpy(values).reshape(-1, 1), item
            )
            for label, values in predictions.items()
        }
        for label, row in metrics.items():
            rows[label].append(row)
        polygons, _, actual = read_mesh(
            data_dir / item["design_name"] / "Volume_Dataset.simh"
        )
        plot_benchmark_comparison(
            polygons,
            actual,
            predictions,
            metrics,
            f"{item['design_name']} | held-out outer-fold design | "
            f"{architecture['name']} ({architecture['parameter_count']:,} parameters, "
            f"epoch {architecture['best_epoch']})",
            output_dir / f"{item['design_name']}_benchmark_comparison.png",
            near_body_bounds(
                item["geom"][:, :2].numpy(), item["wall_distance"].numpy()
            ),
        )
    return rows


def write_summary(rows, architecture, timing, output_dir):
    summaries = []
    for label, model_rows in rows.items():
        summary = {"model": label, **training.summarize_metrics(model_rows)}
        if label == architecture["name"]:
            summary.update(
                parameter_count=architecture["parameter_count"],
                best_epoch=architecture["best_epoch"],
                **timing,
            )
        summaries.append(summary)
    fields = list(dict.fromkeys(key for summary in summaries for key in summary))
    with (output_dir / "benchmark_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps({"architecture": architecture, "models": summaries}, indent=2) + "\n"
    )
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default="Design_1")
    parser.add_argument(
        "--reference-dir", type=Path, default=experiment.DEFAULT_REFERENCE
    )
    parser.add_argument(
        "--compare",
        action="append",
        metavar="LABEL=DIR",
        help="Prediction directory to plot alongside Transolver (repeatable).",
    )
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / training.DATA_DIR
    )
    parser.add_argument("--name")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-slices", type=int, default=32)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--activation", choices=("default", "tanh"), default="default")
    parser.add_argument("--inner-fraction", type=float, default=0.2)
    parser.add_argument("--inner-seed", type=int, default=2026)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--minimum-epochs", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--stratify-inner",
        action="store_true",
        help="Pick one inner-validation design per body-length stratum.",
    )
    args = parser.parse_args()
    positive = (
        args.hidden_dim, args.num_layers, args.num_slices, args.attention_heads,
        args.mlp_ratio, args.max_epochs, args.minimum_epochs,
        args.validation_interval, args.patience, args.threads,
    )  # fmt: skip
    if min(positive) < 1 or args.hidden_dim % args.attention_heads:
        parser.error("Sizes must be positive and heads must divide the hidden width.")
    try:
        args.comparisons = parse_comparisons(args.compare or DEFAULT_COMPARISONS)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = build_architecture(args)
    output_dir = args.results_dir or (
        PROJECT_ROOT / "volume_results" / "benchmarks" / architecture["name"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data = training.prepare_data(
        load_all_starccm_volume_designs(args.data_dir, include_physics=True),
        "geometry",
    )
    outer_train, outer_test, outer_fold = experiment.outer_fold_from_report(
        data, args.reference_dir / "metrics.json", args.design
    )
    inner_train, inner_validation = experiment.split_inner_validation(
        outer_train, args.inner_fraction, args.inner_seed, args.stratify_inner
    )
    architecture["stratified_inner_split"] = args.stratify_inner
    architecture["inner_validation_designs"] = [
        item["design_name"] for item in inner_validation
    ]
    scalers = training.fit_scalers(inner_train)
    print(
        f"{architecture['name']}: {architecture['parameter_count']:,} parameters on {device}"
    )
    started = time.perf_counter()
    result = experiment.train_with_validation(
        training.normalize_data(inner_train, scalers),
        training.normalize_data(inner_validation, scalers),
        device,
        architecture,
        max_epochs=args.max_epochs,
        validation_interval=args.validation_interval,
        patience=args.patience,
        minimum_epochs=args.minimum_epochs,
        seed=args.inner_seed,
        checkpoint_path=output_dir / "best_validation.pt",
        history_path=output_dir / "validation_history.json",
        model_factory=create_transolver,
    )
    elapsed = time.perf_counter() - started
    timing = {
        "training_seconds": elapsed,
        "seconds_per_epoch": elapsed / len(result["history"]),
    }
    attach_scalers(output_dir / "best_validation.pt", scalers)
    experiment.plot_validation_history(
        output_dir / "validation_history.json",
        output_dir / "validation_curve.png",
        architecture["name"],
    )
    architecture["best_epoch"] = result["best_epoch"]
    reference_label, reference_dir = next(iter(args.comparisons.items()))
    experiment.evaluate_architecture(
        result["model"].cpu(),
        outer_test,
        scalers,
        reference_dir,
        output_dir,
        args.data_dir,
        architecture,
        outer_fold,
        reference_label,
    )
    rows = plot_all_designs(
        outer_test, args.comparisons, architecture, output_dir, args.data_dir
    )
    for summary in write_summary(rows, architecture, timing, output_dir):
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
