
import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset import (
    CFDSurfaceDataset,
    RunningStandardizer,
    collate_cfd_graphs_with_physics,
    load_all_starccm_designs,
)
from invariance import make_invariant
from model import Milestone1_GNN
from physics import compute_batch_drag, compute_design_drag

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = r"AIROMMinimizingDragAroundAnAerodynamicBody_input/aero_body_3D_AI_ROM/Training_Adaptive_Sampling"

BATCH_SIZE = 4
HIDDEN_DIM = 64
NUM_LAYERS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 600
# Fraction of epochs over which the drag loss ramps from 0 to LAMBDA_DRAG
DRAG_WARMUP = 0.2

# Physics Loss Weight: balances field MSE and relative drag error
LAMBDA_DRAG = 0.5


NUM_FOLDS = 5
SEED = 0
# 0 = plain field MSE, 1 = fully weighted by each face's drag influence
DRAG_WEIGHT_MIX = 0.0
A_REF = 0.005
EDGE_FEATURES = False
GLOBAL_CONTEXT = True
# Body-scaled positions + normals, no face length on nodes, Re proxy in design inputs.
INVARIANT_INPUTS = False


def fit_scalers(train_data):
    geom_scaler = RunningStandardizer()
    geom_scaler.fit([d.get('node_features', d['geom']) for d in train_data])
    params_scaler = RunningStandardizer()
    params_scaler.fit([d.get('design_features', d['params']).unsqueeze(0) for d in train_data])
    coeffs_scaler = RunningStandardizer()
    coeffs_scaler.fit([d['coeffs'] for d in train_data])
    return geom_scaler, params_scaler, coeffs_scaler


def normalize(data, scalers):
    geom_scaler, params_scaler, coeffs_scaler = scalers
    return [{
        **item,
        'geom_norm': geom_scaler.transform(item.get('node_features', item['geom'])),
        'params_norm': params_scaler.transform(
            item.get('design_features', item['params']).unsqueeze(0)
        ).squeeze(0),
        'fields_norm': coeffs_scaler.transform(item['coeffs']),
    } for item in data]


def train_model(train_data, coeffs_scaler, device):
    loader = DataLoader(
        CFDSurfaceDataset(train_data),
        batch_size=min(BATCH_SIZE, len(train_data)),
        shuffle=True,
        collate_fn=collate_cfd_graphs_with_physics
    )
    model = Milestone1_GNN(
        geom_dim=train_data[0]['geom_norm'].shape[1],
        design_dim=train_data[0]['params_norm'].shape[0],
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, out_dim=2,
        edge_dim=train_data[0]['edge_attr'].shape[1] if EDGE_FEATURES else 0,
        use_global=GLOBAL_CONTEXT,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * 1e-2)

    for epoch in range(NUM_EPOCHS):
        lambda_drag = LAMBDA_DRAG * min(1.0, epoch / max(1, DRAG_WARMUP * NUM_EPOCHS))
        model.train()
        for batch in loader:
            batch_idx = batch['batch_idx'].to(device)
            optimizer.zero_grad()
            pred_norm = model(
                batch['geom_norm'].to(device), batch['params_norm'].to(device), batch['edge_index'].to(device),
                batch['edge_attr'].to(device) if EDGE_FEATURES else None, batch_idx
            )
            weights = (1 - DRAG_WEIGHT_MIX) + DRAG_WEIGHT_MIX * batch['face_weights'].to(device)
            loss_field = torch.mean(weights * (pred_norm - batch['fields_norm'].to(device)) ** 2)
            pred_drags = compute_batch_drag(
                pred_norm, batch['geom_raw'].to(device), batch_idx, batch['num_graphs'],
                coeffs_scaler, batch['q_dyn'].to(device)
            )
            true_drags = batch['true_drags'].to(device)
            # Relative error keeps slender (4 N) and bluff (120 N) designs equally weighted.
            loss_drag = torch.mean(((pred_drags - true_drags) / true_drags) ** 2)
            loss = loss_field + lambda_drag * loss_drag
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


@torch.no_grad()
def predict_drag(model, item, coeffs_scaler, device):
    model.eval()
    geom_norm = item['geom_norm'].to(device)
    params_norm = item['params_norm'].to(device).unsqueeze(0).repeat(geom_norm.size(0), 1)
    edge_attr = item['edge_attr'].to(device) if EDGE_FEATURES else None
    pred_norm = model(geom_norm, params_norm, item['edge_index'].to(device), edge_attr)
    pred_phys = coeffs_scaler.inverse_transform(pred_norm.cpu()) * item['q_dyn']
    return compute_design_drag(pred_phys, item['geom']).item()


def cross_validate(raw_data, device):
    order = torch.randperm(len(raw_data), generator=torch.Generator().manual_seed(SEED)).tolist()
    folds = [order[k::NUM_FOLDS] for k in range(NUM_FOLDS)]
    results = []
    for k, held_out in enumerate(folds):
        held = set(held_out)
        scalers = fit_scalers([raw_data[i] for i in range(len(raw_data)) if i not in held])
        data = normalize(raw_data, scalers)
        model = train_model([data[i] for i in range(len(data)) if i not in held], scalers[2], device)
        for i in held_out:
            pred = predict_drag(model, data[i], scalers[2], device)
            results.append((data[i]['design_name'], data[i]['true_drag'], pred, data[i]['q_dyn']))
        fold_errs = [abs(r[2] - r[1]) / r[1] * 100 for r in results[-len(held_out):]]
        print(f"Fold {k + 1}/{NUM_FOLDS} | mean drag err {sum(fold_errs) / len(fold_errs):.2f}%")
    return results


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware backend: {device}")
    raw_data = load_all_starccm_designs(DATA_DIR)
    if INVARIANT_INPUTS:
        raw_data = [make_invariant(item) for item in raw_data]

    print(f"\n--- {NUM_FOLDS}-fold cross-validation ({len(raw_data)} designs) ---")
    results = cross_validate(raw_data, device)
    errs = torch.tensor([abs(r[2] - r[1]) / r[1] * 100 for r in results])
    for name, true, pred, q in sorted(results, key=lambda r: int(r[0].split('_')[1])):
        print(f"[{name}] STAR-CCM+ Drag: {true:8.4f} N (Cd {true / (q * A_REF):.4f}) | "
              f"GNN Drag: {pred:8.4f} N (Cd {pred / (q * A_REF):.4f}) | "
              f"Error: {abs(pred - true) / true * 100:6.2f}%")
    print(f"\nOut-of-fold drag error: mean {errs.mean():.2f}% | median {errs.median():.2f}% | "
          f"max {errs.max():.2f}%")

    print("\n--- Training final model on all designs ---")
    scalers = fit_scalers(raw_data)
    model = train_model(normalize(raw_data, scalers), scalers[2], device)
    torch.save(model.state_dict(), "milestone1_best_physics_model.pt")
    print("Saved milestone1_best_physics_model.pt")


if __name__ == "__main__":
    main()
