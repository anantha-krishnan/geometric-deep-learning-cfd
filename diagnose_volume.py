"""Diagnose volume fitting and loss weighting without a new full CV sweep.

uv run python diagnose_volume.py --stage overfit
uv run python diagnose_volume.py --stage fold
uv run python diagnose_volume.py --stage plots
uv run python diagnose_volume.py --stage attention --attention-tokens 32 --attention-heads 4
uv run python diagnose_volume.py --stage attention-plots

The overfit stage is a reconstruction, not validation. The fold stage reuses
saved area-loss predictions and trains one matched near-body-loss model. Neither
stage changes the production checkpoint or the original cross-validation data.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

import train_volume as training
from dataset import load_all_starccm_volume_designs
from validate_volume import plot_comparison, read_mesh

PROJECT_ROOT = Path(__file__).resolve().parent


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def export_run(model, scalers, data, config, history, output_dir, data_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    training.save_checkpoint(model, scalers, output_dir / "model.pt", config)
    rows = []
    predictions = {}
    for item in training.normalize_data(data, scalers):
        predicted = training.predict_pressure(
            model, item, scalers[2], torch.device("cpu")
        )
        actual = item["pressure"].numpy().reshape(-1)
        values = predicted.numpy().reshape(-1)
        row = {
            **training.measure_design(predicted, item),
            "actual_min_pa": float(actual.min()),
            "actual_max_pa": float(actual.max()),
            "predicted_min_pa": float(values.min()),
            "predicted_max_pa": float(values.max()),
        }
        rows.append(row)
        predictions[item["design_name"]] = values
        np.savez_compressed(
            output_dir / f"{item['design_name']}.npz",
            geom=item["geom"].numpy(),
            actual_pressure_pa=actual,
            predicted_pressure_pa=values,
        )
        polygons, _, _ = read_mesh(
            data_dir / item["design_name"] / "Volume_Dataset.simh"
        )
        figure = plot_comparison(
            polygons,
            actual,
            values,
            {
                "design_name": item["design_name"],
                "variant": f"geometry / {config['loss']}"
                + (
                    f" / attention({config['attention_tokens']})"
                    if config.get("attention_tokens", 0)
                    else ""
                ),
                "fold": config.get("fold", 0),
                "evaluation": config["evaluation"],
                "reconstruction": config["reconstruction"],
            },
        )
        figure.savefig(
            output_dir / f"{item['design_name']}.png", dpi=200, facecolor="white"
        )
        plt.close(figure)
    report = {
        "config": config,
        "history": history,
        "summary": training.summarize_metrics(rows),
        "designs": rows,
    }
    write_json(output_dir / "metrics.json", report)
    print(json.dumps(report["summary"]), flush=True)
    return report, predictions


def run_overfit(data, args):
    item = next(item for item in data if item["design_name"] == args.design)
    scalers = training.fit_scalers([item])
    history = []
    print(
        f"Single-design reconstruction: {args.design}; {args.overfit_epochs} optimizer steps.",
        flush=True,
    )
    model = training.train_model(
        training.normalize_data([item], scalers),
        torch.device("cpu"),
        args.overfit_epochs,
        seed=training.SEED,
        history=history,
        progress_every=200,
    )
    config = {
        "variant": "geometry",
        "loss": "area",
        "seed": training.SEED,
        "density": training.DENSITY,
        "reference_pressure": training.REFERENCE_PRESSURE,
        "epochs": args.overfit_epochs,
        "training_designs": [args.design],
        "evaluation": "Single-design reconstruction (NOT validation)",
        "reconstruction": True,
    }
    return export_run(
        model,
        scalers,
        [item],
        config,
        history,
        args.output_dir / "overfit_area",
        args.data_dir,
    )


def plot_loss_comparison(
    polygons,
    actual,
    area_prediction,
    balanced_prediction,
    name,
    fold,
    output,
    metrics=None,
    panel_titles=None,
    experiment_note=None,
):
    values = (actual, area_prediction, balanced_prediction)
    norm = Normalize(
        min(field.min() for field in values), max(field.max() for field in values)
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(20, 5.5), layout="constrained", sharex=True, sharey=True
    )
    coords = np.concatenate(polygons)
    titles = panel_titles or (
        "Actual CFD",
        "Area Loss (Saved Baseline)",
        "Near-Body-Balanced Loss",
    )
    for index, (axis, field, title) in enumerate(zip(axes, values, titles)):
        collection = PolyCollection(
            polygons,
            array=field,
            norm=norm,
            cmap="turbo",
            edgecolors="none",
            antialiased=False,
        )
        axis.add_collection(collection)
        axis.set(
            xlim=(coords[:, 0].min(), coords[:, 0].max()),
            ylim=(coords[:, 1].min(), coords[:, 1].max()),
            xlabel="x [m]",
        )
        axis.set_aspect("equal", adjustable="box")
        if metrics is not None and index:
            row = metrics[index - 1]
            near = row.get("near_wall_rmse_pa")
            near_text = f"{near:.2f} Pa" if near is not None else "N/A"
            title += f"\nRMSE: {row['rmse_pa']:.2f} Pa | Near-body RMSE: {near_text}"
        axis.set_title(title, fontsize=11)
    axes[0].set_ylabel("y [m]")
    figure.colorbar(
        collection,
        ax=axes,
        orientation="horizontal",
        label="Pressure [Pa]",
        shrink=0.65,
        aspect=60,
        pad=0.1,
    )
    figure.suptitle(
        f"{name} | Held-Out Fold {fold} | "
        + (experiment_note or "Same Architecture, Seed, Training Designs and 40 Epochs")
    )
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    print(f"Comparison image: {Path(output).resolve()}", flush=True)
    return figure


def render_saved_comparisons(baseline_dir, output, data_dir):
    baseline = json.loads((baseline_dir / "metrics.json").read_text())
    report = json.loads((output / "metrics.json").read_text())
    baseline_rows = {row["design_name"]: row for row in baseline["designs"]}
    prediction_dir = (
        baseline_dir / "predictions"
        if (baseline_dir / "predictions").is_dir()
        else baseline_dir
    )
    attention = report["config"].get("attention_tokens", 0)
    for row in report["designs"]:
        name = row["design_name"]
        with (
            np.load(prediction_dir / f"{name}.npz", allow_pickle=False) as saved,
            np.load(output / f"{name}.npz", allow_pickle=False) as diagnostic,
        ):
            actual = diagnostic["actual_pressure_pa"].reshape(-1)
            if not np.allclose(saved["actual_pressure_pa"].reshape(-1), actual):
                raise ValueError("Cached CFD fields no longer match diagnostic data.")
            area_prediction = saved["predicted_pressure_pa"].reshape(-1)
            balanced_prediction = diagnostic["predicted_pressure_pa"].reshape(-1)
        polygons, _, mesh_actual = read_mesh(data_dir / name / "Volume_Dataset.simh")
        if not np.allclose(mesh_actual, actual):
            raise ValueError("Diagnostic predictions no longer match the CFD mesh.")
        plot_loss_comparison(
            polygons,
            actual,
            area_prediction,
            balanced_prediction,
            name,
            report["config"]["fold"],
            output / f"{name}_{'attention' if attention else 'loss'}_comparison.png",
            metrics=[baseline_rows[name], row],
            panel_titles=(
                "Actual CFD",
                "Near-Body GNN (No Attention)",
                "GNN + Latent Attention",
            )
            if attention
            else None,
            experiment_note=(
                f"Same seed, loss, training designs and 40 epochs | {attention} tokens, "
                f"{report['config']['attention_heads']} heads; architectures differ"
            )
            if attention
            else None,
        )


def attention_reference(args, fold):
    path = args.reference_dir or args.output_dir / "fold_near_body"
    report = json.loads((path / "metrics.json").read_text())
    config = report["config"]
    if (
        config.get("loss") != "near-body"
        or config.get("epochs") != 40
        or config.get("seed") != fold["seed"]
        or config.get("fold") != fold["fold"]
        or set(config.get("training_designs", [])) != set(fold["training_designs"])
        or set(config.get("held_out_designs", [])) != set(fold["held_out_designs"])
        or config.get("near_body_mix") != 0.5
        or config.get("attention_tokens", 0)
    ):
        raise ValueError(
            "The cached no-attention reference must match this fold, loss, seed and 40-epoch schedule."
        )
    return path, report


def run_fold(data, args):
    baseline = json.loads((args.baseline_dir / "metrics.json").read_text())
    fold = next(
        row for row in baseline["folds"] if args.design in row["held_out_designs"]
    )
    if (
        baseline["config"]["variant"] != "geometry"
        or baseline["config"]["epochs"] != 40
    ):
        raise ValueError(
            "The cached comparison must be the geometry variant trained for 40 epochs."
        )
    training_names = set(fold["training_designs"])
    held_out_names = set(fold["held_out_designs"])
    if training_names & held_out_names:
        raise ValueError("Training and held-out design membership overlaps.")
    training_data = [item for item in data if item["design_name"] in training_names]
    held_out = [item for item in data if item["design_name"] in held_out_names]
    if len(training_data) != len(training_names) or len(held_out) != len(
        held_out_names
    ):
        raise ValueError("Data does not match the saved fold membership.")
    scalers = training.fit_scalers(training_data)
    history = []
    attention = args.stage == "attention"
    tokens = args.attention_tokens if attention else 0
    reference_dir, reference = (
        attention_reference(args, fold) if attention else (args.baseline_dir, baseline)
    )
    print(
        f"Training {'attention + ' if attention else ''}near-body loss on fold {fold['fold']} only; cached reference is NOT rerun.",
        flush=True,
    )
    model = training.train_model(
        training.normalize_data(training_data, scalers),
        torch.device("cpu"),
        40,
        seed=fold["seed"],
        loss_mode="near-body",
        history=history,
        progress_every=10,
        attention_tokens=tokens,
        attention_heads=args.attention_heads,
    )
    config = {
        **baseline["config"],
        **fold,
        "loss": "near-body",
        "near_body_mix": 0.5,
        "evaluation": f"Held-out fold {fold['fold']}",
        "reconstruction": False,
        "cached_area_results": str(args.baseline_dir.resolve()),
        "cached_reference_results": str(reference_dir.resolve()),
        "attention_tokens": tokens,
        "attention_heads": args.attention_heads,
    }
    output = args.output_dir / ("fold_attention" if attention else "fold_near_body")
    report, _ = export_run(
        model, scalers, held_out, config, history, output, args.data_dir
    )
    baseline_rows = [
        row for row in reference["designs"] if row["design_name"] in held_out_names
    ]
    comparison = {
        "no_attention" if attention else "area": training.summarize_metrics(
            baseline_rows
        ),
        "attention" if attention else "near_body": report["summary"],
        "scope": "One fixed fold only, not a full CV result.",
    }
    write_json(
        args.output_dir
        / ("attention_comparison.json" if attention else "loss_comparison.json"),
        comparison,
    )
    render_saved_comparisons(reference_dir, output, args.data_dir)
    print(
        "ATTENTION COMPARISON" if attention else "LOSS COMPARISON",
        json.dumps(comparison),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("overfit", "fold", "plots", "attention", "attention-plots", "all"),
        default="all",
    )
    parser.add_argument("--design", default="Design_1")
    parser.add_argument("--overfit-epochs", type=int, default=1200)
    parser.add_argument("--attention-tokens", type=int, default=32)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Cached no-attention near-body fold directory.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / training.DATA_DIR
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=PROJECT_ROOT / "volume_results" / "geometry",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "volume_results" / "diagnosis"
    )
    args = parser.parse_args()
    if args.stage == "attention-plots":
        output = args.output_dir / "fold_attention"
        report = json.loads((output / "metrics.json").read_text())
        render_saved_comparisons(
            Path(report["config"]["cached_reference_results"]), output, args.data_dir
        )
        return
    if args.stage == "plots":
        render_saved_comparisons(
            args.baseline_dir, args.output_dir / "fold_near_body", args.data_dir
        )
        return
    if args.stage == "attention" and (
        args.attention_tokens < 1
        or args.attention_heads < 1
        or training.HIDDEN_DIM % args.attention_heads
    ):
        parser.error(
            "Attention requires positive tokens and heads dividing the hidden width."
        )
    torch.set_num_threads(1)
    raw = load_all_starccm_volume_designs(args.data_dir, include_physics=True)
    data = training.prepare_data(raw, "geometry")
    if (
        args.design not in {item["design_name"] for item in data}
        or args.overfit_epochs < 1
    ):
        parser.error("Select an available design and a positive overfit epoch count.")
    if args.stage in ("overfit", "all"):
        run_overfit(data, args)
    if args.stage in ("fold", "attention", "all"):
        run_fold(data, args)


if __name__ == "__main__":
    main()
