"""Compare volume GNN architectures with inner validation and a fixed outer fold.

Example:
    uv run python compare_volume_architectures.py --design Design_1

The existing six-design outer fold is used only after model selection. These
particular designs have been inspected in earlier experiments, so this is not a
pristine final test set. Checkpoints use only inner-training scalers and the best
inner-validation checkpoint. Each architecture writes metrics and comparison PNGs.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import train_volume as training
from dataset import CFDVolumeDataset, load_all_starccm_volume_designs
from invariance import feature_names, reference_length
from validate_volume import read_mesh

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = PROJECT_ROOT / "volume_results" / "diagnosis" / "fold_near_body"


def split_inner_validation(data, fraction=0.2, seed=2026, stratify=False):
    """Seeded inner split; ``stratify`` draws one validation design per body-length stratum."""
    if len(data) < 4 or not 0 < fraction < 1:
        raise ValueError(
            "An inner split needs at least four designs and a fraction in (0, 1)."
        )
    validation_count = min(len(data) - 2, max(1, round(len(data) * fraction)))
    generator = torch.Generator().manual_seed(seed)
    if stratify:
        lengths = [reference_length(item) for item in data]
        if any(length is None for length in lengths):
            raise ValueError("Stratified splitting needs body bounds on every design.")
        ranked = sorted(range(len(data)), key=lambda index: (lengths[index], index))
        strata = np.array_split(np.asarray(ranked), validation_count)
        validation_indices = {
            int(stratum[torch.randint(len(stratum), (1,), generator=generator)])
            for stratum in strata
        }
    else:
        order = torch.randperm(len(data), generator=generator).tolist()
        validation_indices = set(order[:validation_count])
    return (
        [item for index, item in enumerate(data) if index not in validation_indices],
        [item for index, item in enumerate(data) if index in validation_indices],
    )


def outer_fold_from_report(data, report_path, design_name):
    report = json.loads(Path(report_path).read_text())
    if report.get("folds"):
        fold = next(
            row for row in report["folds"] if design_name in row["held_out_designs"]
        )
    else:
        config = report.get("config", {})
        fold = {
            "fold": config.get("fold"),
            "training_designs": config.get("training_designs", []),
            "held_out_designs": config.get("held_out_designs", []),
            "seed": config.get("seed", training.SEED),
        }
        if not fold["held_out_designs"]:
            raise ValueError("Reference report contains no outer-fold membership.")
        if design_name not in fold["held_out_designs"]:
            raise ValueError(f"Design {design_name} is not in this reference fold.")
    train_names, test_names = (
        set(fold["training_designs"]),
        set(fold["held_out_designs"]),
    )
    if train_names & test_names:
        raise ValueError("Outer training and test designs overlap.")
    if train_names | test_names != {item["design_name"] for item in data}:
        raise ValueError("Outer fold membership does not match the loaded designs.")
    return (
        [item for item in data if item["design_name"] in train_names],
        [item for item in data if item["design_name"] in test_names],
        fold,
    )


def create_architecture_model(architecture, device):
    return training.create_model(
        device,
        geom_dim=architecture.get("geom_dim", 5),
        edge_dim=architecture.get("edge_dim", 4),
        attention_tokens=architecture.get("attention_tokens", 0),
        attention_heads=architecture.get("attention_heads", 4),
        message_attention_heads=architecture.get("message_attention_heads", 0),
        hidden_dim=architecture.get("hidden_dim", training.HIDDEN_DIM),
        num_layers=architecture.get("num_layers", training.NUM_LAYERS),
        activation=architecture.get("activation", "default"),
        design_dim=architecture.get("design_dim", 3),
        aggregation=architecture.get("aggregation", "sum"),
        global_mixer=architecture.get("global_mixer", "latent"),
    )


def architecture_choices(device=None, geom_dim=5, edge_dim=4, design_dim=3):
    device = device or torch.device("cpu")
    common = {
        "geom_dim": geom_dim,
        "edge_dim": edge_dim,
        "design_dim": design_dim,
        "num_layers": 3,
        "attention_heads": 4,
        "message_attention_heads": 0,
    }
    choices = [
        {**common, "name": "gnn32", "hidden_dim": 32, "attention_tokens": 0},
        {
            **common,
            "name": "latent_attention32",
            "hidden_dim": 32,
            "attention_tokens": 32,
        },
        {
            **common,
            "name": "neighbor_attention32",
            "hidden_dim": 32,
            "attention_tokens": 0,
            "message_attention_heads": 4,
        },
    ]
    reference_count = sum(
        parameter.numel()
        for parameter in create_architecture_model(choices[1], device).parameters()
    )
    candidate_widths = range(40, 81)
    matched = min(
        candidate_widths,
        key=lambda width: abs(
            sum(
                parameter.numel()
                for parameter in create_architecture_model(
                    {
                        **common,
                        "name": "candidate",
                        "hidden_dim": width,
                        "attention_tokens": 0,
                    },
                    device,
                ).parameters()
            )
            - reference_count
        ),
    )
    choices.insert(
        1,
        {
            **common,
            "name": "gnn_parameter_matched",
            "hidden_dim": matched,
            "attention_tokens": 0,
        },
    )
    latent_reference = next(
        choice for choice in choices if choice["name"] == "latent_attention32"
    )
    target_parameter_count = sum(
        parameter.numel()
        for parameter in create_architecture_model(
            latent_reference, device
        ).parameters()
    )
    neighbor_common = {
        **common,
        "name": "neighbor_attention_candidate",
        "attention_tokens": 0,
        "message_attention_heads": 4,
    }
    neighbor_width = min(
        range(40, 129, 4),
        key=lambda width: abs(
            sum(
                parameter.numel()
                for parameter in create_architecture_model(
                    {**neighbor_common, "hidden_dim": width}, device
                ).parameters()
            )
            - target_parameter_count
        ),
    )
    choices.append(
        {
            **neighbor_common,
            "name": "neighbor_attention_parameter_matched",
            "hidden_dim": neighbor_width,
        }
    )
    neighbor_reference = next(
        choice for choice in choices if choice["name"] == "neighbor_attention32"
    )
    neighbor_matched_reference = next(
        choice
        for choice in choices
        if choice["name"] == "neighbor_attention_parameter_matched"
    )
    choices.extend(
        [
            {
                **latent_reference,
                "name": "latent_attention32_tanh",
                "activation": "tanh",
            },
            {
                **neighbor_reference,
                "name": "neighbor_attention32_tanh",
                "activation": "tanh",
            },
            {
                **neighbor_matched_reference,
                "name": "neighbor_attention_parameter_matched_tanh",
                "activation": "tanh",
            },
            {
                **latent_reference,
                "name": "latent_attention32_mean",
                "aggregation": "mean",
            },
            {
                **latent_reference,
                "name": "slice_attention32",
                "global_mixer": "slice",
            },
        ]
    )
    for choice in choices:
        choice["parameter_count"] = sum(
            parameter.numel()
            for parameter in create_architecture_model(choice, device).parameters()
        )
    return choices


def rank_architectures(rows, profiles=None):
    """Rank models by explicit metric priorities; lower normalized regret is better."""
    metrics = (
        "mean_design_rmse_pa",
        "mean_near_wall_rmse_pa",
        "worst_design_rmse_pa",
    )
    profiles = profiles or {
        "balanced": {
            "mean_design_rmse_pa": 0.4,
            "mean_near_wall_rmse_pa": 0.4,
            "worst_design_rmse_pa": 0.2,
        },
        "whole_domain": {
            "mean_design_rmse_pa": 0.6,
            "mean_near_wall_rmse_pa": 0.2,
            "worst_design_rmse_pa": 0.2,
        },
        "near_body_only": {"mean_near_wall_rmse_pa": 1.0},
    }
    if not rows:
        raise ValueError("At least one architecture result is required.")
    for row in rows:
        missing = [key for key in metrics if key not in row or row[key] is None]
        if missing:
            raise ValueError(
                f"Architecture {row.get('architecture', '<unnamed>')} has missing ranking metrics: {missing}"
            )
        if any(
            not math.isfinite(float(row[key])) or float(row[key]) <= 0
            for key in metrics
        ):
            raise ValueError("Ranking metrics must be finite and positive.")
    criteria = {
        key: {
            name: rank + 1
            for rank, name in enumerate(
                row["architecture"]
                for row in sorted(
                    rows, key=lambda item: (item[key], item["architecture"])
                )
            )
        }
        for key in metrics
    }
    output = {}
    best = {key: min(float(row[key]) for row in rows) for key in metrics}
    for profile_name, weights in profiles.items():
        if not weights or any(key not in metrics for key in weights):
            raise ValueError(
                f"Invalid metric weights for ranking profile {profile_name}."
            )
        if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise ValueError("Ranking weights must be finite and nonnegative.")
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("Each ranking profile needs positive total weight.")
        ranked = []
        for row in rows:
            regret = sum(
                (weight / total_weight) * (float(row[key]) / best[key] - 1)
                for key, weight in weights.items()
            )
            ranked.append(
                {
                    **row,
                    "profile": profile_name,
                    "weighted_relative_regret": regret,
                    "criterion_ranks": {
                        key: criteria[key][row["architecture"]] for key in metrics
                    },
                }
            )
        output[profile_name] = sorted(
            ranked,
            key=lambda row: (row["weighted_relative_regret"], row["architecture"]),
        )
    return output


def save_architecture_rankings(summary_paths, output_path):
    rows = []
    for path in summary_paths:
        report = json.loads(Path(path).read_text())
        rows.extend(report["architectures"])
    names = [row["architecture"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Architecture summaries contain duplicate model names.")
    rankings = rank_architectures(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rankings, indent=2) + "\n")
    return rankings


def forward_item(model, item, device):
    geom = item["geom_norm"].to(device)
    params = item["params_norm"].to(device).unsqueeze(0).expand(geom.shape[0], -1)
    edge_attr = item["edge_attr"].to(device)
    return model(geom, params, item["edge_index"].to(device), edge_attr)


@torch.no_grad()
def validation_loss(model, validation_data, device, mix=0.5):
    model.eval()
    losses = []
    for item in validation_data:
        predicted = forward_item(model, item, device)
        distance, threshold = training.near_wall_distance(item)
        losses.append(
            training.near_body_weighted_mse(
                predicted,
                item["pressure_norm"].to(device),
                item["geom"][:, 3].to(device),
                distance.to(device),
                torch.zeros(len(item["pressure"]), dtype=torch.long, device=device),
                1,
                mix,
                threshold,
            ).item()
        )
    return statistics.mean(losses)


def train_with_validation(
    train_data,
    validation_data,
    device,
    architecture,
    *,
    max_epochs=1000,
    validation_interval=10,
    patience=8,
    minimum_epochs=100,
    seed=2026,
    checkpoint_path,
    history_path,
    scalers=None,
    model_factory=create_architecture_model,
):
    if (
        not train_data
        or not validation_data
        or max_epochs < 1
        or validation_interval < 1
        or patience < 1
    ):
        raise ValueError(
            "Training/validation sets and positive epoch settings are required."
        )
    torch.manual_seed(seed)
    model = model_factory(architecture, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.LEARNING_RATE,
        weight_decay=training.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=training.LEARNING_RATE * 1e-2
    )
    loader = DataLoader(
        CFDVolumeDataset(train_data),
        batch_size=1,
        shuffle=True,
        collate_fn=training.collate_cfd_volume_graphs,
    )
    history = []
    best_loss, best_epoch, stale_checks = float("inf"), 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in loader:
            geom, params = (
                batch["geom_norm"].to(device),
                batch["params_norm"].to(device),
            )
            batch_idx, areas = (
                batch["batch_idx"].to(device),
                batch["cell_areas"].to(device),
            )
            target, edges = (
                batch["pressure_norm"].to(device),
                batch["edge_index"].to(device),
            )
            edge_attr = batch["edge_attr"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(geom, params, edges, edge_attr, batch_idx=batch_idx)
            distance, threshold = training.near_wall_distance(batch)
            loss = training.near_body_weighted_mse(
                prediction,
                target,
                areas,
                distance.to(device),
                batch_idx,
                batch["num_graphs"],
                threshold=threshold,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": statistics.mean(epoch_losses),
            "validation_loss": None,
        }
        if epoch % validation_interval == 0 or epoch == max_epochs:
            score = validation_loss(model, validation_data, device)
            record["validation_loss"] = score
            if score < best_loss:
                best_loss, best_epoch, stale_checks = score, epoch, 0
                if scalers is not None:
                    training.save_checkpoint(
                        model,
                        scalers,
                        checkpoint_path,
                        {
                            **architecture,
                            "seed": seed,
                            "selection_designs": [
                                item["design_name"] for item in validation_data
                            ],
                        },
                    )
                    saved_checkpoint = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=True
                    )
                else:
                    saved_checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "architecture": architecture,
                    }
                saved_checkpoint.update(
                    {
                        "best_epoch": epoch,
                        "validation_loss": score,
                        "selection_metric": score,
                        "selection_designs": [
                            item["design_name"] for item in validation_data
                        ],
                        "seed": seed,
                    }
                )
                torch.save(saved_checkpoint, checkpoint_path)
            else:
                stale_checks += 1
            print(
                f"{architecture['name']} epoch {epoch}/{max_epochs} | train {record['train_loss']:.6g} | val {score:.6g} | best {best_loss:.6g} @ {best_epoch}",
                flush=True,
            )
            if epoch >= minimum_epochs and stale_checks >= patience:
                history.append(record)
                break
        history.append(record)
    if best_epoch == 0:
        raise RuntimeError("No best validation checkpoint was produced.")
    best = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    Path(history_path).write_text(
        json.dumps(
            {
                "architecture": architecture,
                "max_epochs": max_epochs,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "patience": patience,
                "validation_interval": validation_interval,
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
    }


def plot_comparison(
    polygons,
    actual,
    reference,
    prediction,
    rows,
    architecture,
    outer_fold,
    output,
    reference_label="Near-body GNN reference",
):
    fields = (actual, reference, prediction)
    low = min(float(values.min()) for values in fields)
    high = max(float(values.max()) for values in fields)
    norm = plt.Normalize(low, high)
    figure, axes = plt.subplots(
        1, 3, figsize=(19, 5.7), layout="constrained", sharex=True, sharey=True
    )
    coords = np.concatenate(polygons)
    titles = (
        "Actual CFD",
        reference_label,
        f"{architecture['name']} | best inner-validation epoch",
    )
    for axis, values, title, metrics in zip(
        axes, fields, titles, (None, rows[0], rows[1])
    ):
        from matplotlib.collections import PolyCollection

        collection = PolyCollection(
            polygons,
            array=values,
            cmap="turbo",
            norm=norm,
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
        if metrics is not None:
            title += f"\nRMSE {metrics['rmse_pa']:.2f} Pa | near-body {metrics.get('near_wall_rmse_pa', float('nan')):.2f} Pa"
        axis.set_title(title, fontsize=10)
    axes[0].set_ylabel("y [m]")
    figure.colorbar(
        collection,
        ax=axes,
        orientation="horizontal",
        label="Pressure [Pa]",
        shrink=0.68,
        aspect=55,
        pad=0.09,
    )
    figure.suptitle(
        f"{architecture['name']} | outer fold {outer_fold['fold']} | "
        f"best inner-validation epoch {architecture['best_epoch']} | "
        f"parameters {architecture['parameter_count']:,}"
    )
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def evaluate_architecture(
    model,
    test_data,
    scalers,
    reference_dir,
    output_dir,
    data_dir,
    architecture,
    outer_fold,
    reference_label="Near-body GNN reference",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    rows = []
    for item in training.normalize_data(test_data, scalers):
        prediction = training.predict_pressure(
            model, item, scalers[2], torch.device("cpu")
        )
        row = training.measure_design(prediction, item)
        rows.append(row)
        reference_prediction_dir = (
            reference_dir / "predictions"
            if (reference_dir / "predictions").is_dir()
            else reference_dir
        )
        reference_path = reference_prediction_dir / f"{item['design_name']}.npz"
        with np.load(reference_path, allow_pickle=False) as saved:
            actual = saved["actual_pressure_pa"].reshape(-1)
            reference = saved["predicted_pressure_pa"].reshape(-1)
        if not np.allclose(actual, item["pressure"].numpy().reshape(-1)):
            raise ValueError(
                "Reference predictions do not match current held-out pressure."
            )
        np.savez_compressed(
            output_dir / f"{item['design_name']}.npz",
            actual_pressure_pa=actual,
            reference_pressure_pa=reference,
            predicted_pressure_pa=prediction.numpy().reshape(-1),
        )
        polygons, _, mesh_actual = read_mesh(
            data_dir / item["design_name"] / "Volume_Dataset.simh"
        )
        if not np.allclose(actual, mesh_actual):
            raise ValueError("Saved held-out pressure does not match mesh export.")
        comparison_rows = [
            training.measure_design(torch.from_numpy(reference).reshape(-1, 1), item),
            row,
        ]
        plot_comparison(
            polygons,
            actual,
            reference,
            prediction.numpy().reshape(-1),
            comparison_rows,
            architecture,
            outer_fold,
            output_dir / f"{item['design_name']}_comparison.png",
            reference_label,
        )
    return rows


def plot_validation_history(history_path, output_path, architecture_name):
    report = json.loads(Path(history_path).read_text())
    epochs = [row["epoch"] for row in report["history"]]
    train_loss = [row["train_loss"] for row in report["history"]]
    validation = [
        (row["epoch"], row["validation_loss"])
        for row in report["history"]
        if row["validation_loss"] is not None
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axis.plot(epochs, train_loss, label="Training loss", linewidth=1.5)
    if validation:
        axis.plot(
            [row[0] for row in validation],
            [row[1] for row in validation],
            marker="o",
            markersize=3,
            label="Inner-validation loss",
        )
    axis.axvline(
        report["best_epoch"],
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Selected epoch {report['best_epoch']}",
    )
    axis.set(
        title=f"{architecture_name} | Checkpoint Selection",
        xlabel="Epoch",
        ylabel="Near-body-balanced normalized loss",
    )
    axis.legend()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        default="Design_1",
        help="Select the cached outer fold containing this design.",
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--comparison-dir", type=Path)
    parser.add_argument("--comparison-label", default="Near-body GNN reference")
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Default: volume_results/architecture_validation[_invariant].",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / training.DATA_DIR
    )
    parser.add_argument("--inner-fraction", type=float, default=0.2)
    parser.add_argument("--inner-seed", type=int, default=2026)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--minimum-epochs", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Validation checks without improvement before early stopping.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=[
            "gnn32",
            "gnn_parameter_matched",
            "latent_attention32",
            "neighbor_attention32",
            "neighbor_attention_parameter_matched",
        ],
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--variant",
        choices=("geometry", "continuous", *training.VARIANT_FRAMES),
        default="geometry",
        help="invariant: body-scaled positions; two_scale: domain-scaled positions "
        "with body-scaled wall distance and blockage inputs; continuous: positions and "
        "wall distance without cell size, edges as direction and log(distance / L_ref).",
    )
    parser.add_argument(
        "--stratify-inner",
        action="store_true",
        help="Pick one inner-validation design per body-length stratum.",
    )
    args = parser.parse_args()
    if (
        args.max_epochs < 1
        or args.minimum_epochs < 1
        or args.validation_interval < 1
        or args.patience < 1
    ):
        parser.error("Epoch limits, validation interval and patience must be positive.")
    if args.results_dir is None:
        suffix = "" if args.variant == "geometry" else f"_{args.variant}"
        args.results_dir = (
            PROJECT_ROOT / "volume_results" / f"architecture_validation{suffix}"
        )
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_all_starccm_volume_designs(args.data_dir, include_physics=True)
    data = training.prepare_data(raw, args.variant)
    outer_train, outer_test, outer_fold = outer_fold_from_report(
        data, args.reference_dir / "metrics.json", args.design
    )
    inner_train, inner_validation = split_inner_validation(
        outer_train, args.inner_fraction, args.inner_seed, args.stratify_inner
    )
    scalers = training.fit_scalers(inner_train)
    normalized_train = training.normalize_data(inner_train, scalers)
    normalized_validation = training.normalize_data(inner_validation, scalers)
    choices = {
        choice["name"]: choice
        for choice in architecture_choices(
            device,
            geom_dim=normalized_train[0]["geom_norm"].shape[1],
            edge_dim=normalized_train[0]["edge_attr"].shape[1],
            design_dim=normalized_train[0]["params_norm"].shape[0],
        )
    }
    unknown = set(args.architectures) - set(choices)
    if unknown:
        parser.error(f"Unknown architecture names: {sorted(unknown)}")
    config = {
        "variant": args.variant,
        "seed": args.inner_seed,
        "stratified_inner_split": args.stratify_inner,
        "inner_training_designs": [item["design_name"] for item in inner_train],
        "inner_validation_designs": [item["design_name"] for item in inner_validation],
        "outer_training_designs": [item["design_name"] for item in outer_train],
        "outer_test_designs": [item["design_name"] for item in outer_test],
        "outer_fold": outer_fold["fold"],
        "max_epochs": args.max_epochs,
        "minimum_epochs": args.minimum_epochs,
        "validation_interval": args.validation_interval,
        "patience": args.patience,
        "loss": "near-body",
        "loss_mix": 0.5,
        "device": str(device),
        "outer_test_previously_inspected": True,
        "comparison_dir": str(args.comparison_dir or args.reference_dir),
        "comparison_label": args.comparison_label,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "split.json").write_text(json.dumps(config, indent=2) + "\n")
    all_summaries = []
    for name in args.architectures:
        architecture = dict(choices[name])
        if args.variant in training.VARIANT_FRAMES:
            architecture["feature_names"] = feature_names(
                data[0], training.VARIANT_FRAMES[args.variant]
            )
        elif args.variant == "continuous":
            architecture["feature_names"] = {
                "node": ["x", "y", "wall_distance"],
                "design": ["P_D1_r", "P_D2_r", "P_V"],
                "edge": ["direction_x", "direction_y", "log_distance_over_L"],
            }
        architecture_dir = args.results_dir / name
        architecture_dir.mkdir(parents=True, exist_ok=True)
        result = train_with_validation(
            normalized_train,
            normalized_validation,
            device,
            architecture,
            max_epochs=args.max_epochs,
            validation_interval=args.validation_interval,
            patience=args.patience,
            minimum_epochs=args.minimum_epochs,
            seed=args.inner_seed,
            checkpoint_path=architecture_dir / "best_validation.pt",
            history_path=architecture_dir / "validation_history.json",
            scalers=scalers,
        )
        plot_validation_history(
            architecture_dir / "validation_history.json",
            architecture_dir / "validation_curve.png",
            name,
        )
        architecture["best_epoch"] = result["best_epoch"]
        rows = evaluate_architecture(
            result["model"],
            outer_test,
            scalers,
            args.comparison_dir or args.reference_dir,
            architecture_dir,
            args.data_dir,
            architecture,
            outer_fold,
            args.comparison_label,
        )
        summary = training.summarize_metrics(rows)
        summary.update(
            {
                "architecture": name,
                "parameter_count": architecture["parameter_count"],
                "best_epoch": result["best_epoch"],
                "best_validation_loss": result["best_validation_loss"],
            }
        )
        all_summaries.append(summary)
        (architecture_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "architecture": architecture,
                    "summary": summary,
                    "designs": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(summary), flush=True)
    (args.results_dir / "architecture_summary.json").write_text(
        json.dumps({"experiment": config, "architectures": all_summaries}, indent=2)
        + "\n"
    )
    with (args.results_dir / "architecture_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_summaries[0]))
        writer.writeheader()
        writer.writerows(all_summaries)


if __name__ == "__main__":
    main()
