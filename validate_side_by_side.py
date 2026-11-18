"""Compare STAR-CCM+ surface fields with the current physics-trained GNN.

Run from the project directory:
    uv run python validate_side_by_side.py --design Design_1
    uv run python validate_side_by_side.py --design Design_1 --holdout --no-show

The default checkpoint was trained on ALL designs. Its plots are reconstructions,
not unseen validation. --holdout trains a fresh model using train.py's settings,
excluding the selected design from both training and scaler fitting. This may
take as long as one cross-validation fold; it does not overwrite the checkpoint.

Legacy checkpoints contain weights only. Reconstruction therefore rebuilds the
scalers from --data-dir, which must contain the original, unchanged training data.
Only pressure and streamwise wall shear stress are predicted, not density.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import load_all_starccm_designs
from model import Milestone1_GNN
from physics import compute_design_drag
from train import DATA_DIR, SEED, fit_scalers, normalize, train_model

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "milestone1_best_physics_model.pt"


def load_trained_model(model_path, device):
    """Restore the architecture encoded in a two-output physics model state dict."""
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if state["decoder.2.weight"].shape[0] != 2:
        raise ValueError("Expected a two-field checkpoint from the current train.py.")
    hidden_dim = state["encoder.0.weight"].shape[0]
    layer_ids = {key.split(".")[1] for key in state if key.startswith("layers.")}
    model = Milestone1_GNN(
        geom_dim=7,
        design_dim=3,
        hidden_dim=hidden_dim,
        num_layers=len(layer_ids),
        out_dim=2,
        edge_dim=state["layers.0.edge_mlp.0.weight"].shape[1] - 2 * hidden_dim,
        use_global=state["layers.0.node_mlp.0.weight"].shape[1] == 3 * hidden_dim,
    ).to(device)
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def predict_fields(model, item, scalers, device):
    geom_scaler, params_scaler, coeffs_scaler = scalers
    geom = geom_scaler.transform(item["geom"]).to(device)
    params = params_scaler.transform(item["params"].unsqueeze(0)).to(device)
    edge_dim = (
        model.layers[0].edge_mlp[0].in_features - 2 * model.encoder[0].out_features
    )
    edge_attr = item["edge_attr"].to(device) if edge_dim else None
    model.eval()
    predicted = model(
        geom,
        params.expand(geom.size(0), -1),
        item["edge_index"].to(device),
        edge_attr,
    )
    return coeffs_scaler.inverse_transform(predicted.cpu()) * item["q_dyn"]


def prepare_comparison(raw_data, design_name, model_path, device, holdout=False):
    item = next(
        (entry for entry in raw_data if entry["design_name"] == design_name), None
    )
    if item is None:
        raise ValueError(f"Design not found: {design_name}")
    training = (
        [entry for entry in raw_data if entry is not item] if holdout else raw_data
    )
    if len(training) < 2:
        raise ValueError(
            "Scaler fitting requires two training designs; holdout requires at least three designs."
        )
    scalers = fit_scalers(training)
    if holdout:
        print(
            f"Training a fresh model on {len(training)} designs, excluding {design_name}."
        )
        torch.manual_seed(SEED)
        model = train_model(normalize(training, scalers), scalers[2], device)
        label = "Held-out validation"
    else:
        print("WARNING: Training-data reconstruction, not unseen-design validation.")
        print(
            "Rebuilding scalers: data must match the checkpoint's original training set."
        )
        model = load_trained_model(model_path, device)
        label = "Training-data reconstruction"
    return item, predict_fields(model, item, scalers, device), label


def plot_comparison(item, predicted, label):
    actual = item["fields"].cpu()
    predicted = predicted.detach().cpu()
    coordinates = item["geom"][:, :2].cpu().numpy()
    drag_pred = compute_design_drag(predicted, item["geom"].cpu()).item()
    drag_report = item["true_drag"]
    drag_surface = compute_design_drag(actual, item["geom"].cpu()).item()
    relative_error = (
        f"{abs(drag_pred - drag_report) / abs(drag_report) * 100:.2f}%"
        if drag_report
        else "N/A (zero reference drag)"
    )
    print(f"\n{label}: {item['design_name']}")
    print(f"STAR-CCM+ reported drag: {drag_report:.4f} N")
    print(f"Integrated CFD surface drag: {drag_surface:.4f} N")
    print(f"AI ROM drag: {drag_pred:.4f} N | Report-relative error: {relative_error}")

    figure, axes = plt.subplots(2, 3, figsize=(18, 8), layout="constrained")
    field_names = ("Static pressure [Pa]", "Streamwise wall shear stress [Pa]")
    for row, field_name in enumerate(field_names):
        actual_values = actual[:, row].numpy()
        predicted_values = predicted[:, row].numpy()
        error = abs(predicted_values - actual_values)
        minimum = min(actual_values.min(), predicted_values.min())
        maximum = max(actual_values.max(), predicted_values.max())
        panels = (
            (actual_values, "Actual CFD", "viridis", minimum, maximum),
            (predicted_values, "AI ROM prediction", "viridis", minimum, maximum),
            (error, "Absolute error", "magma", 0, max(float(error.max()), 1e-12)),
        )
        for column, (values, title, cmap, lower, upper) in enumerate(panels):
            axis = axes[row, column]
            scatter = axis.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                c=values,
                cmap=cmap,
                vmin=lower,
                vmax=upper,
                s=12,
                linewidths=0,
            )
            axis.set_title(f"{title}: {field_name}", fontsize=10)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]")
            figure.colorbar(scatter, ax=axis, label="Pa", fraction=0.046, pad=0.04)
        print(
            f"{field_name} | MAE: {error.mean():.4g} | RMSE: {(error**2).mean() ** 0.5:.4g}"
        )

    figure.suptitle(
        f"{label}: {item['design_name']}\n"
        f"Reported CFD drag = {drag_report:.4f} N | AI ROM = {drag_pred:.4f} N | "
        f"Error = {relative_error}",
        fontsize=13,
    )
    return figure


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / DATA_DIR)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Physics-trained state dict (ignored with --holdout).",
    )
    parser.add_argument(
        "--design", help="Design folder name; defaults to the first loaded design."
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Retrain without the selected design for genuine unseen validation.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "validation_results"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the PNG without opening an interactive plot.",
    )
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    if not args.data_dir.is_dir():
        parser.error(f"Data directory not found: {args.data_dir}")
    if not args.holdout and not args.model.is_file():
        parser.error(f"Checkpoint not found: {args.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware backend: {device}")
    raw_data = load_all_starccm_designs(args.data_dir)
    if not raw_data:
        parser.error("No usable designs found in the data directory.")
    design_name = args.design or raw_data[0]["design_name"]
    try:
        item, predicted, label = prepare_comparison(
            raw_data, design_name, args.model, device, args.holdout
        )
    except ValueError as error:
        parser.error(str(error))
    figure = plot_comparison(item, predicted, label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode = "holdout" if args.holdout else "reconstruction"
    output = args.output_dir / f"validation_side_by_side_{design_name}_{mode}.png"
    figure.savefig(output, dpi=300)
    print(f"Comparison plot saved to: {output}")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
