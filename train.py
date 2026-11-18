import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import (
    load_all_starccm_designs,
    RunningStandardizer,
    CFDSurfaceDataset,
    collate_cfd_graphs
)
from model import Milestone1_GNN
from physics import compute_design_drag

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Point to Training_Adaptive_Sampling directory
DATA_DIR = r"AIROMMinimizingDragAroundAnAerodynamicBody_input/aero_body_3D_AI_ROM/Training_Adaptive_Sampling"

BATCH_SIZE = 4
HIDDEN_DIM = 64
NUM_LAYERS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware backend: {device}")

    # 1. Load Data
    raw_data = load_all_starccm_designs(DATA_DIR)
    num_total = len(raw_data)
    if num_total < 5:
        raise ValueError(f"Found only {num_total} designs. Need at least 5 designs to run train/val/test splits.")

    # 2. Train / Validation / Test Splits (80% / 10% / 10%)
    n_train = max(1, int(num_total * 0.8))
    n_val   = max(1, int(num_total * 0.1))

    train_data = raw_data[:n_train]
    val_data   = raw_data[n_train:n_train + n_val]
    test_data  = raw_data[n_train + n_val:]
    if len(test_data) == 0:
        test_data = val_data  # fallback if total designs is small

    print(f"Split sizes -> Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # 3. Fit Standardizers on Training Data ONLY
    geom_scaler = RunningStandardizer()
    geom_scaler.fit([d['geom'] for d in train_data])

    params_scaler = RunningStandardizer()
    params_scaler.fit([d['params'].unsqueeze(0) for d in train_data])

    fields_scaler = RunningStandardizer()
    fields_scaler.fit([d['fields'] for d in train_data])

    # Apply normalization
    for split in [train_data, val_data, test_data]:
        for item in split:
            item['geom_norm'] = geom_scaler.transform(item['geom'])
            item['params_norm'] = params_scaler.transform(item['params'].unsqueeze(0)).squeeze(0)
            item['fields_norm'] = fields_scaler.transform(item['fields'])

    # 4. DataLoader
    train_norm_list = [{
        'geom': d['geom_norm'],
        'params': d['params_norm'],
        'fields': d['fields_norm'],
        'edge_index': d['edge_index']
    } for d in train_data]

    train_loader = DataLoader(
        CFDSurfaceDataset(train_norm_list),
        batch_size=min(BATCH_SIZE, len(train_norm_list)),
        shuffle=True,
        collate_fn=collate_cfd_graphs
    )

    # 5. Initialize Model, Loss, Optimizer
    model = Milestone1_GNN(
        geom_dim=7,
        design_dim=3,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        out_dim=3
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = nn.MSELoss()

    best_drag_err = float('inf')
    best_model_weights = None

    print("\n--- Starting Model Training ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.perf_counter()
        # ---------------- TRAINING ----------------
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            geom = batch['geom'].to(device)
            params = batch['params'].to(device)
            edge_index = batch['edge_index'].to(device)
            target = batch['fields'].to(device)

            optimizer.zero_grad()
            pred = model(geom, params, edge_index)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch['num_graphs']

        train_loss /= len(train_data)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        val_drag_errors = []

        with torch.no_grad():
            for item in val_data:
                geom = item['geom_norm'].to(device)
                params = item['params_norm'].to(device).unsqueeze(0).repeat(geom.size(0), 1)
                edge_index = item['edge_index'].to(device)
                target = item['fields_norm'].to(device)

                pred_norm = model(geom, params, edge_index)
                loss = criterion(pred_norm, target)
                val_loss += loss.item()

                # Reconstruct physical fields & compute drag
                pred_phys = fields_scaler.inverse_transform(pred_norm.cpu())
                drag_predicted = compute_design_drag(pred_phys, item['geom'])
                drag_official  = item['true_drag']

                rel_err = abs(drag_predicted.item() - drag_official) / abs(drag_official) * 100.0
                val_drag_errors.append(rel_err)

        val_loss /= len(val_data)
        mean_drag_err = sum(val_drag_errors) / len(val_drag_errors)

        scheduler.step(val_loss)

        # Track drag accuracy directly:
        if mean_drag_err < best_drag_err:
            best_drag_err = mean_drag_err
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, "milestone1_best_drag_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.perf_counter() - epoch_start
            print(f"Epoch {epoch:03d}/{NUM_EPOCHS:03d} | "
                  f"Train Loss: {train_loss:.5f} | "
                  f"Val Loss: {val_loss:.5f} | "
                  f"Mean Drag Error: {mean_drag_err:.2f}% | "
                f"Time: {elapsed:.2f}s | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # ==============================================================================
    # 6. FINAL TEST BENCHMARK (Comparing against STAR-CCM+ Log Values)
    # ==============================================================================
    print("\n--- Final Test Set Evaluation (Unseen Designs) ---")
    model.load_state_dict(best_model_weights)
    model.eval()

    with torch.no_grad():
        for item in test_data:
            geom = item['geom_norm'].to(device)
            params = item['params_norm'].to(device).unsqueeze(0).repeat(geom.size(0), 1)
            edge_index = item['edge_index'].to(device)

            pred_norm = model(geom, params, edge_index)
            pred_phys = fields_scaler.inverse_transform(pred_norm.cpu())

            pred_drag = compute_design_drag(pred_phys, item['geom']).item()
            true_drag = item['true_drag']
            error_pct = abs(pred_drag - true_drag) / abs(true_drag) * 100.0

            print(f"[{item['design_name']}] STAR-CCM+ Report Drag: {true_drag:.4f} N | "
                  f"GNN Integrated Drag: {pred_drag:.4f} N | "
                  f"Error: {error_pct:.2f}%")

if __name__ == "__main__":
    main()