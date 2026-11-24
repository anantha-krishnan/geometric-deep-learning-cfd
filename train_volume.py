import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset import (
    CFDVolumeDataset,
    RunningStandardizer,
    collate_cfd_volume_graphs,
    load_all_starccm_volume_designs,
)
from invariance import (
    NEAR_WALL_RELATIVE,
    continuous_inputs,
    feature_names,
    make_invariant,
    reference_length,
)
from model import Milestone1_GNN

DATA_DIR = r"AIROMMinimizingDragAroundAnAerodynamicBody_input/aero_body_3D_AI_ROM/Training_Adaptive_Sampling"
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "milestone1_volume_pressure_v2.pt"
BATCH_SIZE = 1
HIDDEN_DIM = 32
NUM_LAYERS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_FOLDS = 5
CV_EPOCHS = 40
FINAL_EPOCHS = CV_EPOCHS
SEED = 0
DENSITY = 1.18415
REFERENCE_PRESSURE = 0.0
NEAR_WALL_THRESHOLD = 0.05
VARIANTS = (
    "baseline",
    "coefficient",
    "geometry",
    "invariant",
    "two_scale",
    "continuous",
)
# Position frame of each invariant variant (see invariance.make_invariant).
VARIANT_FRAMES = {"invariant": "body", "two_scale": "domain"}


def prepare_data(data, variant, density=DENSITY, reference_pressure=REFERENCE_PRESSURE):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown volume model variant: {variant}")
    if (
        not math.isfinite(density)
        or density <= 0
        or not math.isfinite(reference_pressure)
    ):
        raise ValueError(
            "Density must be positive and reference pressure must be finite."
        )
    prepared = []
    for raw in data:
        item = dict(raw)
        if raw["geom"].shape[1] > 4:
            item["wall_distance"] = raw["geom"][:, 4]
        if variant in VARIANT_FRAMES:
            item = make_invariant(item, VARIANT_FRAMES[variant])
        elif variant == "continuous":
            if "wall_distance" not in item:
                raise ValueError("Continuous variant requires wall distance.")
            item = continuous_inputs(item)
        elif variant != "geometry":
            item["geom"] = raw["geom"][:, :4]
            item.pop("edge_attr", None)
        elif raw["geom"].shape[1] != 5 or "edge_attr" not in raw:
            raise ValueError(
                "Geometry variant requires wall distance and edge attributes."
            )
        scale = 1.0
        reference = 0.0
        if variant != "baseline":
            speed = float(raw["params"][2])
            if not math.isfinite(speed) or speed <= 0:
                raise ValueError(
                    "Coefficient training requires a positive inlet speed."
                )
            scale = 0.5 * density * speed**2
            reference = reference_pressure
        item["pressure_scale"] = scale
        item["pressure_reference"] = reference
        item["target"] = (raw["pressure"] - reference) / scale
        prepared.append(item)
    return prepared


def area_weighted_mse(prediction, target, cell_areas):
    weights = cell_areas / cell_areas.mean().clamp_min(1e-12)
    return (weights.unsqueeze(1) * (prediction - target).square()).sum() / weights.sum()


def near_wall_distance(source):
    """Wall distance and near-body threshold, body-relative for invariant inputs."""
    if "wall_distance_ref" in source:
        return source["wall_distance_ref"], NEAR_WALL_RELATIVE
    return source["wall_distance"], NEAR_WALL_THRESHOLD


def model_inputs(item):
    return item.get("node_features", item["geom"]), item.get(
        "design_features", item["params"]
    )


def near_body_weighted_mse(
    prediction,
    target,
    cell_areas,
    wall_distance,
    batch_idx,
    num_graphs,
    mix=0.5,
    threshold=0.05,
):
    if not 0 <= mix <= 1 or threshold <= 0:
        raise ValueError(
            "Near-body mix must be in [0, 1] and threshold must be positive."
        )
    near_areas = cell_areas * (wall_distance <= threshold)
    domain_sum = cell_areas.new_zeros(num_graphs).index_add_(0, batch_idx, cell_areas)
    near_sum = cell_areas.new_zeros(num_graphs).index_add_(0, batch_idx, near_areas)
    domain_weights = cell_areas / domain_sum[batch_idx].clamp_min(1e-12)
    near_weights = torch.where(
        near_sum[batch_idx] > 0,
        near_areas / near_sum[batch_idx].clamp_min(1e-12),
        domain_weights,
    )
    weights = (1 - mix) * domain_weights + mix * near_weights
    return (weights * (prediction - target).square().mean(dim=1)).sum() / num_graphs


def fit_scalers(train_data):
    if not train_data:
        raise ValueError("Scaler fitting requires at least one design.")
    geom_scaler = RunningStandardizer()
    geom_scaler.fit([model_inputs(item)[0] for item in train_data])
    params_scaler = RunningStandardizer()
    if len(train_data) == 1:
        params_scaler.mean = model_inputs(train_data[0])[1].unsqueeze(0).clone()
        params_scaler.std = torch.ones_like(params_scaler.mean)
    else:
        params_scaler.fit([model_inputs(item)[1].unsqueeze(0) for item in train_data])
    pressure_scaler = RunningStandardizer()
    pressure_scaler.fit([item.get("target", item["pressure"]) for item in train_data])
    return geom_scaler, params_scaler, pressure_scaler


def normalize_data(data, scalers):
    geom_scaler, params_scaler, pressure_scaler = scalers
    normalized = []
    for item in data:
        nodes, design = model_inputs(item)
        normalized.append(
            {
                **item,
                "geom_norm": geom_scaler.transform(nodes),
                "params_norm": params_scaler.transform(design.unsqueeze(0)).squeeze(0),
                "pressure_norm": pressure_scaler.transform(
                    item.get("target", item["pressure"])
                ),
            }
        )
    return normalized


def create_model(
    device,
    geom_dim=4,
    edge_dim=0,
    *,
    attention_tokens=0,
    attention_heads=4,
    message_attention_heads=0,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    activation="default",
    design_dim=3,
    aggregation="sum",
):
    return Milestone1_GNN(
        geom_dim=geom_dim,
        design_dim=design_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=1,
        edge_dim=edge_dim,
        use_global=True,
        attention_tokens=attention_tokens,
        attention_heads=attention_heads,
        message_attention_heads=message_attention_heads,
        activation=activation,
        aggregation=aggregation,
    ).to(device)


def train_model(
    train_data,
    device,
    epochs,
    seed=SEED,
    *,
    loss_mode="area",
    near_body_mix=0.5,
    near_body_threshold=None,
    history=None,
    progress_every=0,
    attention_tokens=0,
    attention_heads=4,
    message_attention_heads=0,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
):
    if epochs < 1 or not train_data:
        raise ValueError("Training requires positive epochs and at least one design.")
    if loss_mode not in ("area", "near-body"):
        raise ValueError(f"Unknown training loss: {loss_mode}")
    torch.manual_seed(seed)
    loader = DataLoader(
        CFDVolumeDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_cfd_volume_graphs,
        generator=torch.Generator().manual_seed(seed),
    )
    edge_dim = (
        train_data[0]["edge_attr"].shape[1] if "edge_attr" in train_data[0] else 0
    )
    model = create_model(
        device,
        train_data[0]["geom_norm"].shape[1],
        edge_dim,
        attention_tokens=attention_tokens,
        attention_heads=attention_heads,
        message_attention_heads=message_attention_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        design_dim=train_data[0]["params_norm"].shape[0],
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=LEARNING_RATE * 1e-2
    )

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            geom = batch["geom_norm"].to(device)
            params = batch["params_norm"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_idx = batch["batch_idx"].to(device)
            areas = batch["cell_areas"].to(device)
            target = batch["pressure_norm"].to(device)

            optimizer.zero_grad()
            edge_attr = batch["edge_attr"].to(device) if "edge_attr" in batch else None
            prediction = model(geom, params, edge_index, edge_attr, batch_idx=batch_idx)
            if loss_mode == "near-body":
                distance, threshold = near_wall_distance(batch)
                loss = near_body_weighted_mse(
                    prediction,
                    target,
                    areas,
                    distance.to(device),
                    batch_idx,
                    batch["num_graphs"],
                    near_body_mix,
                    threshold if near_body_threshold is None else near_body_threshold,
                )
            else:
                loss = area_weighted_mse(prediction, target, areas)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite volume training loss.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        mean_loss = statistics.mean(losses)
        if history is not None:
            history.append({"epoch": epoch + 1, "training_loss": mean_loss})
        if progress_every and (
            (epoch + 1) % progress_every == 0 or epoch == epochs - 1
        ):
            print(
                f"Epoch {epoch + 1}/{epochs} | {loss_mode} training loss {mean_loss:.6g}",
                flush=True,
            )
    return model


@torch.no_grad()
def predict_pressure(model, item, pressure_scaler, device):
    model.eval()
    geom = item["geom_norm"].to(device)
    params = item["params_norm"].to(device).unsqueeze(0).expand(geom.size(0), -1)
    edge_attr = item["edge_attr"].to(device) if "edge_attr" in item else None
    prediction = model(geom, params, item["edge_index"].to(device), edge_attr)
    target = pressure_scaler.inverse_transform(prediction.cpu())
    return target * item.get("pressure_scale", 1.0) + item.get(
        "pressure_reference", 0.0
    )


def evaluate_design(prediction, item):
    error = prediction - item["pressure"]
    areas = item["geom"][:, 3]
    rmse = torch.sqrt((areas * error.squeeze(1).square()).sum() / areas.sum())
    mae = (areas * error.squeeze(1).abs()).sum() / areas.sum()
    return rmse.item(), mae.item()


def measure_design(prediction, item):
    target = item["pressure"].detach().cpu().double().flatten()
    predicted = prediction.detach().cpu().double().flatten()
    areas = item["geom"][:, 3].cpu().double()
    if predicted.shape != target.shape or not torch.isfinite(predicted).all():
        raise ValueError(
            "Predictions must be finite and match the pressure field shape."
        )
    error = predicted - target
    area = areas.sum()
    squared_error = (areas * error.square()).sum()
    target_mean = (areas * target).sum() / area
    variance = (areas * (target - target_mean).square()).sum()
    result = {
        "design_name": item["design_name"],
        "rmse_pa": math.sqrt(float(squared_error / area)),
        "mae_pa": float((areas * error.abs()).sum() / area),
        "max_abs_error_pa": float(error.abs().max()),
        "r2": float(1 - squared_error / variance) if variance > 1e-12 else None,
        "area": float(area),
        "squared_error_integral": float(squared_error),
        "target_mean_pa": float(target_mean),
        "target_variance_integral": float(variance),
    }
    if "wall_distance" in item:
        near_wall = item["wall_distance"].cpu() <= NEAR_WALL_THRESHOLD
        result["near_wall_rmse_pa"] = _masked_rmse(areas, error, near_wall)
        length = reference_length(item)
        if length is not None:
            relative = item["wall_distance"].cpu() / length <= NEAR_WALL_RELATIVE
            result["near_wall_rel_rmse_pa"] = _masked_rmse(areas, error, relative)
    return result


def _masked_rmse(areas, error, mask):
    if not mask.any():
        return None
    return math.sqrt(
        float((areas[mask] * error[mask].square()).sum() / areas[mask].sum())
    )


def summarize_metrics(rows):
    total_area = sum(row["area"] for row in rows)
    mean = sum(row["area"] * row["target_mean_pa"] for row in rows) / total_area
    variance = sum(
        row["target_variance_integral"]
        + row["area"] * (row["target_mean_pa"] - mean) ** 2
        for row in rows
    )
    squared_error = sum(row["squared_error_integral"] for row in rows)
    scores = [row["r2"] for row in rows if row["r2"] is not None]
    near_wall = [
        row["near_wall_rmse_pa"]
        for row in rows
        if row.get("near_wall_rmse_pa") is not None
    ]
    near_wall_rel = [
        row["near_wall_rel_rmse_pa"]
        for row in rows
        if row.get("near_wall_rel_rmse_pa") is not None
    ]
    return {
        "num_designs": len(rows),
        "mean_design_rmse_pa": statistics.mean(row["rmse_pa"] for row in rows),
        "median_design_rmse_pa": statistics.median(row["rmse_pa"] for row in rows),
        "worst_design_rmse_pa": max(row["rmse_pa"] for row in rows),
        "pooled_rmse_pa": math.sqrt(squared_error / total_area),
        "pooled_r2": 1 - squared_error / variance if variance > 1e-12 else None,
        "mean_design_r2": statistics.mean(scores) if scores else None,
        "min_design_r2": min(scores) if scores else None,
        "mean_near_wall_rmse_pa": statistics.mean(near_wall) if near_wall else None,
        "mean_near_wall_rel_rmse_pa": statistics.mean(near_wall_rel)
        if near_wall_rel
        else None,
    }


def write_results(rows, folds, config, results_dir):
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    report = {
        "config": config,
        "folds": folds,
        "summary": summarize_metrics(rows),
        "designs": rows,
    }
    (path / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (path / "per_design_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cross_validate(data, device, epochs, results_dir=None, run_config=None):
    if len(data) < NUM_FOLDS or epochs < 1:
        raise ValueError(
            f"Cross-validation requires positive epochs and at least {NUM_FOLDS} designs."
        )
    config = {**(run_config or {}), "epochs": epochs, "num_folds": NUM_FOLDS}
    seed = config.get("seed", SEED)
    order = torch.randperm(
        len(data), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    folds = [order[index::NUM_FOLDS] for index in range(NUM_FOLDS)]
    rows = []
    membership = []
    for fold_index, held_out in enumerate(folds):
        held_out_set = set(held_out)
        training = [
            item for index, item in enumerate(data) if index not in held_out_set
        ]
        fold_seed = seed + fold_index
        membership.append(
            {
                "fold": fold_index + 1,
                "seed": fold_seed,
                "training_designs": [item["design_name"] for item in training],
                "held_out_designs": [data[index]["design_name"] for index in held_out],
            }
        )
        scalers = fit_scalers(training)
        model = train_model(
            normalize_data(training, scalers),
            device,
            epochs,
            seed=fold_seed,
            attention_tokens=config.get("attention_tokens", 0),
            attention_heads=config.get("attention_heads", 4),
            message_attention_heads=config.get("message_attention_heads", 0),
            hidden_dim=config.get("hidden_dim", HIDDEN_DIM),
            num_layers=config.get("num_layers", NUM_LAYERS),
        )
        fold_rows = []
        for item in normalize_data([data[index] for index in held_out], scalers):
            prediction = predict_pressure(model, item, scalers[2], device)
            row = {"fold": fold_index + 1, **measure_design(prediction, item)}
            fold_rows.append(row)
            rows.append(row)
            if results_dir is not None:
                path = Path(results_dir) / "predictions"
                path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path / f"{item['design_name']}.npz",
                    geom=item["geom"].cpu().numpy(),
                    wall_distance=item.get("wall_distance", torch.empty(0))
                    .cpu()
                    .numpy(),
                    actual_pressure_pa=item["pressure"].cpu().numpy(),
                    predicted_pressure_pa=prediction.cpu().numpy(),
                    fold=np.array(fold_index + 1),
                )
        print(
            f"Fold {fold_index + 1}/{NUM_FOLDS} | {json.dumps(summarize_metrics(fold_rows))}",
            flush=True,
        )
    print(f"Out-of-fold metrics: {json.dumps(summarize_metrics(rows))}", flush=True)
    if results_dir is not None:
        write_results(rows, membership, config, results_dir)
    return rows


def save_checkpoint(model, scalers, path, run_config=None):
    geom_scaler, params_scaler, pressure_scaler = scalers
    hidden_dim = model.encoder[0].out_features
    geom_dim = geom_scaler.mean.shape[1]
    edge_dim = model.layers[0].edge_mlp[0].in_features - 2 * hidden_dim
    names = (run_config or {}).get("feature_names")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "geom_dim": geom_dim,
                "design_dim": params_scaler.mean.shape[1],
                "hidden_dim": hidden_dim,
                "num_layers": len(model.layers),
                "out_dim": 1,
                "edge_dim": edge_dim,
                "use_global": True,
                "attention_tokens": model.attention_tokens,
                "attention_heads": model.attention_heads,
                "message_attention_heads": model.message_attention_heads,
                "activation": model.activation,
                "aggregation": model.aggregation,
            },
            "format_version": 2,
            "run_config": run_config or {"variant": "baseline"},
            "geom_scaler": {
                "mean": geom_scaler.mean.cpu(),
                "std": geom_scaler.std.cpu(),
            },
            "params_scaler": {
                "mean": params_scaler.mean.cpu(),
                "std": params_scaler.std.cpu(),
            },
            "pressure_scaler": {
                "mean": pressure_scaler.mean.cpu(),
                "std": pressure_scaler.std.cpu(),
            },
            "input_features": names["node"]
            if names
            else ["x", "y", "z", "cell_area"]
            + (["wall_distance"] if geom_dim == 5 else []),
            "design_features": names["design"]
            if names
            else ["P_D1_r", "P_D2_r", "P_V"],
            "edge_features": names["edge"]
            if names
            else [
                "direction_x",
                "direction_y",
                "relative_spacing",
                "log_area_ratio",
            ]
            if edge_dim
            else [],
            "output_features": ["pressure"],
        },
        path,
    )


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Milestone1_GNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    scalers = []
    for name in ("geom_scaler", "params_scaler", "pressure_scaler"):
        scaler = RunningStandardizer()
        scaler.mean = checkpoint[name]["mean"]
        scaler.std = checkpoint[name]["std"]
        scalers.append(scaler)
    return (
        model.eval(),
        tuple(scalers),
        checkpoint.get("run_config", {"variant": "baseline"}),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a cell-centred volume pressure GNN."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--variant", choices=VARIANTS, default="geometry")
    parser.add_argument(
        "--epochs",
        type=int,
        help=f"Epochs for both CV and final training (default {CV_EPOCHS}).",
    )
    parser.add_argument(
        "--cv-epochs", type=int, help="Compatibility alias for --epochs."
    )
    parser.add_argument(
        "--final-epochs", type=int, help="Compatibility alias for --epochs."
    )
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--density", type=float, default=DENSITY)
    parser.add_argument("--reference-pressure", type=float, default=REFERENCE_PRESSURE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--attention-tokens",
        type=int,
        default=0,
        help="Global latent tokens per layer; 0 preserves the original GNN.",
    )
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--message-attention-heads", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-cv", action="store_true")
    mode.add_argument("--cv-only", action="store_true")
    args = parser.parse_args()
    supplied_epochs = {
        value
        for value in (args.epochs, args.cv_epochs, args.final_epochs)
        if value is not None
    }
    if len(supplied_epochs) > 1:
        parser.error(
            "CV and final training must use the same epoch count; use --epochs."
        )
    epochs = next(iter(supplied_epochs), CV_EPOCHS)
    if epochs < 1 or args.threads < 1:
        parser.error("Epochs and threads must be positive.")
    if (
        args.attention_tokens < 0
        or args.attention_heads < 1
        or (args.attention_tokens and HIDDEN_DIM % args.attention_heads)
    ):
        parser.error(
            "Tokens must be nonnegative and attention heads must divide hidden width."
        )
    if args.attention_tokens and args.output == OUTPUT_PATH:
        args.output = PROJECT_ROOT / "milestone1_volume_pressure_attention.pt"
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Hardware backend: {device}; variant: {args.variant}; epochs: {epochs}",
        flush=True,
    )
    raw_data = load_all_starccm_volume_designs(args.data_dir, include_physics=True)
    try:
        data = prepare_data(
            raw_data, args.variant, args.density, args.reference_pressure
        )
        if len(data) < 2:
            raise ValueError("Training requires at least two designs.")
    except ValueError as error:
        parser.error(str(error))
    config = {
        "variant": args.variant,
        "density": args.density,
        "reference_pressure": args.reference_pressure,
        "seed": args.seed,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "near_wall_threshold_m": NEAR_WALL_THRESHOLD,
        "near_wall_threshold_body_lengths": NEAR_WALL_RELATIVE,
        "design_names": [item["design_name"] for item in data],
        "cv_performed": not args.skip_cv,
        "attention_tokens": args.attention_tokens,
        "attention_heads": args.attention_heads,
        "message_attention_heads": args.message_attention_heads,
    }
    if args.variant in VARIANT_FRAMES:
        config["feature_names"] = feature_names(data[0], VARIANT_FRAMES[args.variant])
    result_name = f"{args.variant}_attention" if args.attention_tokens else args.variant
    results_dir = args.results_dir or PROJECT_ROOT / "volume_results" / result_name
    if not args.skip_cv:
        cross_validate(data, device, epochs, results_dir, config)
    if args.cv_only:
        return
    print("\n--- Training final volume model on all designs ---")
    scalers = fit_scalers(data)
    normalized = normalize_data(data, scalers)
    model = train_model(
        normalized,
        device,
        epochs,
        seed=args.seed,
        attention_tokens=args.attention_tokens,
        attention_heads=args.attention_heads,
        message_attention_heads=args.message_attention_heads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, scalers, args.output, config)
    print(f"Saved volume checkpoint: {args.output}")


if __name__ == "__main__":
    main()
