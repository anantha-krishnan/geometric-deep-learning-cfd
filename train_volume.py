import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset import (
    CFDVolumeDataset,
    RunningStandardizer,
    collate_cfd_volume_graphs,
    load_all_starccm_volume_designs,
)
from model import Milestone1_GNN

DATA_DIR = r"AIROMMinimizingDragAroundAnAerodynamicBody_input/aero_body_3D_AI_ROM/Training_Adaptive_Sampling"
OUTPUT_PATH = "milestone1_volume_pressure_model.pt"
BATCH_SIZE = 1
HIDDEN_DIM = 32
NUM_LAYERS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_FOLDS = 5
CV_EPOCHS = 40
FINAL_EPOCHS = 120
SEED = 0


def area_weighted_mse(prediction, target, cell_areas):
    weights = cell_areas / cell_areas.mean().clamp_min(1e-12)
    return (weights.unsqueeze(1) * (prediction - target).square()).sum() / weights.sum()


def fit_scalers(train_data):
    geom_scaler = RunningStandardizer()
    geom_scaler.fit([item["geom"] for item in train_data])
    params_scaler = RunningStandardizer()
    params_scaler.fit([item["params"].unsqueeze(0) for item in train_data])
    pressure_scaler = RunningStandardizer()
    pressure_scaler.fit([item["pressure"] for item in train_data])
    return geom_scaler, params_scaler, pressure_scaler


def normalize_data(data, scalers):
    geom_scaler, params_scaler, pressure_scaler = scalers
    normalized = []
    for item in data:
        normalized.append(
            {
                **item,
                "geom_norm": geom_scaler.transform(item["geom"]),
                "params_norm": params_scaler.transform(
                    item["params"].unsqueeze(0)
                ).squeeze(0),
                "pressure_norm": pressure_scaler.transform(item["pressure"]),
            }
        )
    return normalized


def create_model(device):
    return Milestone1_GNN(
        geom_dim=4,
        design_dim=3,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        out_dim=1,
        use_global=True,
    ).to(device)


def train_model(train_data, device, epochs):
    loader = DataLoader(
        CFDVolumeDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_cfd_volume_graphs,
    )
    model = create_model(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=LEARNING_RATE * 1e-2
    )

    for _ in range(epochs):
        model.train()
        for batch in loader:
            geom = batch["geom_norm"].to(device)
            params = batch["params_norm"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_idx = batch["batch_idx"].to(device)
            areas = batch["cell_areas"].to(device)
            target = batch["pressure_norm"].to(device)

            optimizer.zero_grad()
            prediction = model(geom, params, edge_index, batch_idx=batch_idx)
            loss = area_weighted_mse(prediction, target, areas)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


@torch.no_grad()
def predict_pressure(model, item, pressure_scaler, device):
    model.eval()
    geom = item["geom_norm"].to(device)
    params = item["params_norm"].to(device).unsqueeze(0).expand(geom.size(0), -1)
    prediction = model(geom, params, item["edge_index"].to(device))
    return pressure_scaler.inverse_transform(prediction.cpu())


def evaluate_design(prediction, item):
    error = prediction - item["pressure"]
    areas = item["geom"][:, 3]
    rmse = torch.sqrt((areas * error.squeeze(1).square()).sum() / areas.sum())
    mae = (areas * error.squeeze(1).abs()).sum() / areas.sum()
    return rmse.item(), mae.item()


def cross_validate(data, device, epochs):
    order = torch.randperm(
        len(data), generator=torch.Generator().manual_seed(SEED)
    ).tolist()
    folds = [order[index::NUM_FOLDS] for index in range(NUM_FOLDS)]
    design_metrics = []
    weighted_squared_error = 0.0
    total_area = 0.0
    weighted_target_sum = 0.0
    weighted_target_squared_sum = 0.0

    for fold_index, held_out in enumerate(folds):
        held_out_set = set(held_out)
        train_indices = [i for i in range(len(data)) if i not in held_out_set]
        scalers = fit_scalers([data[i] for i in train_indices])
        normalized = normalize_data(data, scalers)
        model = train_model([normalized[i] for i in train_indices], device, epochs)

        fold_metrics = []
        for index in held_out:
            item = normalized[index]
            prediction = predict_pressure(model, item, scalers[2], device)
            rmse, mae = evaluate_design(prediction, item)
            fold_metrics.append(rmse)
            design_metrics.append((item["design_name"], rmse, mae))
            error = prediction.squeeze(1) - item["pressure"].squeeze(1)
            areas = item["geom"][:, 3]
            weighted_squared_error += (areas * error.square()).sum().item()
            total_area += areas.sum().item()
            weighted_target_sum += (areas * item["pressure"].squeeze(1)).sum().item()
            weighted_target_squared_sum += (
                (areas * item["pressure"].squeeze(1).square()).sum().item()
            )

        print(
            f"Fold {fold_index + 1}/{NUM_FOLDS} | "
            f"area-weighted pressure RMSE {sum(fold_metrics) / len(fold_metrics):.5g}"
        )

    mean_target = weighted_target_sum / total_area
    total_variance = weighted_target_squared_sum - total_area * mean_target**2
    weighted_r2 = 1.0 - weighted_squared_error / max(total_variance, 1e-12)
    rmses = torch.tensor([metric[1] for metric in design_metrics])
    print(
        f"Out-of-fold area-weighted RMSE: {rmses.mean():.5g} mean per design | "
        f"{rmses.median():.5g} median | weighted R2 {weighted_r2:.5f}"
    )
    return design_metrics


def save_checkpoint(model, scalers, path):
    geom_scaler, params_scaler, pressure_scaler = scalers
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "geom_dim": 4,
                "design_dim": 3,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "out_dim": 1,
                "use_global": True,
            },
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
            "input_features": ["x", "y", "z", "cell_area"],
            "output_features": ["pressure"],
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a cell-centred volume pressure GNN."
    )
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--cv-epochs", type=int, default=CV_EPOCHS)
    parser.add_argument("--final-epochs", type=int, default=FINAL_EPOCHS)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware backend: {device}; batch size: {BATCH_SIZE}")
    data = load_all_starccm_volume_designs(args.data_dir)
    print(
        f"Loaded {len(data)} designs / {sum(len(item['pressure']) for item in data)} cells"
    )

    if not args.skip_cv:
        print(f"\n--- {NUM_FOLDS}-fold volume pressure cross-validation ---")
        cross_validate(data, device, args.cv_epochs)

    print("\n--- Training final volume model on all designs ---")
    scalers = fit_scalers(data)
    normalized = normalize_data(data, scalers)
    model = train_model(normalized, device, args.final_epochs)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_checkpoint(model, scalers, args.output)
    print(f"Saved volume checkpoint: {args.output}")


if __name__ == "__main__":
    main()
