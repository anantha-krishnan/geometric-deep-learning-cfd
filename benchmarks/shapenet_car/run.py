"""Train and evaluate the hybrid slice GNN on reduced ShapeNet-Car."""

import argparse
import json
import random
import time
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.airfrans.data import fit_normalization, normalize_case
from benchmarks.shapenet_car.data import load_case
from benchmarks.shapenet_car.training import BenchmarkModel, field_loss, field_scores
from model import DOMAIN_KINDS


def resolve_split_paths(root, splits, mapping):
    result = {}
    seen = set()
    for split in ("train", "validation", "test"):
        result[split] = []
        for original in splits[split]:
            parts = PurePosixPath(mapping[original]).parts
            expected = "test_data" if split == "test" else "training_data"
            if (
                original in seen
                or len(parts) != 3
                or parts[0] != expected
                or parts[1] not in {f"param{index}" for index in range(9)}
                or not parts[2].replace("_", "").isalnum()
                or parts[2] != PurePosixPath(original).name
            ):
                raise ValueError(f"Invalid or duplicate {split} mapping: {original}")
            seen.add(original)
            result[split].append((root / expected, "/".join(parts[1:])))
    return result


def tensors(case, device):
    return (
        torch.as_tensor(case["x"], device=device),
        torch.as_tensor(case["pos"], device=device),
        torch.as_tensor(case["y"], device=device),
        torch.as_tensor(case["surface"], device=device),
        case["edge_index"].to(device),
        torch.as_tensor(case["distance_ratio"], device=device),
    )


def validation_loss(model, cases, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for case in cases:
            inputs, positions, targets, surface, edges, ratio = tensors(case, device)
            losses.append(
                field_loss(
                    model(inputs, positions, edges, (surface, ratio)), targets, surface
                ).item()
            )
    return float(np.mean(losses))


def train(training, validation, args, device, output):
    model = BenchmarkModel(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        total_steps=(len(training) + 1) * args.epochs,
        final_div_factor=1000.0,
    )
    rng = np.random.default_rng(args.seed)
    history = []
    best_loss = float("inf")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for index in rng.permutation(len(training)):
            inputs, positions, targets, surface, edges, ratio = tensors(
                training[index], device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = field_loss(
                model(inputs, positions, edges, (surface, ratio)), targets, surface
            )
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        row = {"epoch": epoch, "training_loss": float(np.mean(losses))}
        if (epoch - 1) % args.validation_interval == 0 or epoch == args.epochs:
            row["validation_loss"] = validation_loss(model, validation, device)
        row["elapsed_seconds"] = time.perf_counter() - started
        history.append(row)
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        }
        torch.save(checkpoint, output / "final.pt")
        if row.get("validation_loss", float("inf")) < best_loss:
            best_loss = row["validation_loss"]
            torch.save(checkpoint, output / "best_validation.pt")
        print(json.dumps(row), flush=True)
    return model


def plot_pressure(case, prediction, destination, label="GNN hybrid slice"):
    mask = case["surface"]
    positions = case["pos"][mask]
    actual = case["y"][mask, 3]
    predicted = prediction[mask, 3]
    low, high = float(actual.min()), float(actual.max())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, axes_indices in enumerate(((0, 1), (0, 2))):
        for column, (values, title) in enumerate(((actual, "CFD"), (predicted, label))):
            image = axes[row, column].scatter(
                positions[:, axes_indices[0]],
                positions[:, axes_indices[1]],
                c=values,
                s=2,
                cmap="coolwarm",
                vmin=low,
                vmax=high,
                rasterized=True,
            )
            axes[row, column].set_title(title)
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xlabel("xyz"[axes_indices[0]])
            axes[row, column].set_ylabel("xyz"[axes_indices[1]])
    fig.colorbar(image, ax=axes, label="Surface pressure (dataset units)")
    fig.suptitle(f"Held-out ShapeNet-Car: {case['name']}")
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def evaluate(model, paths, statistics, device, output):
    output.mkdir(exist_ok=False)
    model.eval()
    rows = []
    for root, name in paths:
        raw = load_case(root, name)
        case = normalize_case(raw, statistics)
        inputs, positions, _, surface, edges, ratio = tensors(case, device)
        with torch.no_grad():
            prediction = model(inputs, positions, edges, (surface, ratio)).cpu().numpy()
        prediction = prediction * (statistics["y_std"] + 1e-8) + statistics["y_mean"]
        row = {"case": name, **field_scores(prediction, raw["y"], raw["surface"])}
        rows.append(row)
        stem = name.split("/")[-1]
        np.savez_compressed(
            output / f"{stem}.npz",
            positions=raw["pos"],
            surface=raw["surface"],
            actual=raw["y"],
            prediction=prediction,
        )
        plot_pressure(raw, prediction, output / f"{stem}.png", model.kind)
        print(json.dumps(row), flush=True)
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("velocity_relative_l2", "pressure_relative_l2")
    }
    (output / "metrics.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--model",
        choices=("gnn_hybrid_slice", *DOMAIN_KINDS),
        default="gnn_hybrid_slice",
    )
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use two train, one validation, and one test case.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.validation_interval < 1:
        parser.error("Epochs and validation interval must be positive.")
    splits = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    mapping = json.loads((args.data / "case_mapping.json").read_text(encoding="utf-8"))
    paths = resolve_split_paths(args.data, splits, mapping)
    if args.pilot:
        paths = {
            key: values[: 2 if key == "train" else 1] for key, values in paths.items()
        }
    if any(not values for values in paths.values()):
        parser.error("All splits must contain at least one case.")
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {len(paths['train'])} training meshes on {device}.", flush=True)
    started = time.perf_counter()
    training = [load_case(root, name) for root, name in paths["train"]]
    statistics = fit_normalization(training)
    training = [normalize_case(case, statistics) for case in training]
    validation = [
        normalize_case(load_case(root, name), statistics)
        for root, name in paths["validation"]
    ]
    print(
        f"Loaded training and validation in {time.perf_counter() - started:.1f}s.",
        flush=True,
    )
    np.savez(args.output / "normalization.npz", **statistics)
    config = {
        "epochs": args.epochs,
        "validation_interval": args.validation_interval,
        "seed": args.seed,
        "pilot": args.pilot,
        "splits": {key: [name for _, name in values] for key, values in paths.items()},
        "original_splits": {
            key: splits[key][: len(values)] for key, values in paths.items()
        },
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": str(torch.__version__),
        "model": args.model,
        "checkpoint_evaluated": "final.pt",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model = train(training, validation, args, device, args.output)
    del training, validation
    evaluate(model, paths["test"], statistics, device, args.output / "evaluation")


if __name__ == "__main__":
    main()
