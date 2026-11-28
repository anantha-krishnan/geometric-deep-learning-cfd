"""Train Transolver or the latent-attention GNN on a reduced AirfRANS split."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.airfrans.data import (
    fit_normalization,
    induced_subgraph,
    load_case,
    normalize_case,
)
from benchmarks.airfrans.training import MODEL_KINDS, BenchmarkModel, physics_loss


def load_training_data(root, splits):
    started = time.perf_counter()
    training = [load_case(root, name) for name in splits["train"]]
    statistics = fit_normalization(training)
    training = [normalize_case(case, statistics) for case in training]
    validation = [
        normalize_case(load_case(root, name), statistics)
        for name in splits["validation"]
    ]
    print(
        f"Loaded {len(training)} training and {len(validation)} validation cases in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return training, validation, statistics


def sample_case(case, size, rng, device, include_edges=False, include_domain=False):
    if len(case["x"]) < size:
        raise ValueError(f"{case['name']} has fewer than {size} nodes.")
    indices = rng.choice(len(case["x"]), size, replace=False)
    edge_index = None
    if include_edges:
        if "edge_index" not in case:
            raise ValueError(f"{case['name']} has no mesh edge index.")
        edge_index = induced_subgraph(case["edge_index"], indices, len(case["x"]))
        edge_index = edge_index.to(device)
    sample = (
        torch.as_tensor(case["x"][indices], device=device),
        torch.as_tensor(case["pos"][indices], device=device),
        torch.as_tensor(case["y"][indices], device=device),
        torch.as_tensor(case["surface"][indices], device=device),
        edge_index,
    )
    if include_domain:
        return (
            *sample,
            (
                sample[3],
                torch.as_tensor(case["distance_ratio"][indices], device=device),
            ),
        )
    return sample


def validation_loss(model, cases, sample_size, seed, device):
    rng = np.random.default_rng(seed)
    model.eval()
    losses = []
    with torch.no_grad():
        for case in cases:
            inputs, positions, targets, surface, edge_index, domain = sample_case(
                case,
                sample_size,
                rng,
                device,
                include_edges=model.uses_graph,
                include_domain=True,
            )
            losses.append(
                physics_loss(
                    model(inputs, positions, edge_index, domain), targets, surface
                ).item()
            )
    return float(np.mean(losses))


def train_model(kind, training, validation, statistics, args, device, output):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = BenchmarkModel(kind, args.reference).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, total_steps=(len(training) + 1) * args.epochs
    )
    history = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for index in rng.permutation(len(training)):
            inputs, positions, targets, surface, edge_index, domain = sample_case(
                training[index],
                args.points,
                rng,
                device,
                include_edges=model.uses_graph,
                include_domain=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = physics_loss(
                model(inputs, positions, edge_index, domain), targets, surface
            )
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite {kind} loss at epoch {epoch + 1}.")
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        row = {"epoch": epoch + 1, "training_loss": float(np.mean(losses))}
        if (
            epoch == 0
            or (epoch + 1) % args.validation_interval == 0
            or epoch + 1 == args.epochs
        ):
            row["validation_loss"] = validation_loss(
                model, validation, args.points, args.seed + 1, device
            )
        row["elapsed_seconds"] = time.perf_counter() - started
        history.append(row)
        print(f"{kind} {row}", flush=True)
        checkpoint = {
            "kind": kind,
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "history": history,
            "points": args.points,
            "seed": args.seed,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
        }
        torch.save(checkpoint, output / f"{kind}.pt")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("public_benchmarks/airfans_Dataset/Dataset")
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("volume_results/airfrans_reduced_20261122/splits.json"),
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("public_benchmarks/transolver_upstream")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--points", type=int, default=32000)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KINDS,
        default=["reference"],
        help="Implementation(s) to train; defaults to the paper reference.",
    )
    args = parser.parse_args()
    if min(args.epochs, args.points, args.validation_interval) <= 0:
        parser.error("Epochs, point count and validation interval must be positive.")
    if not torch.cuda.is_available():
        parser.error("The reference model requires CUDA.")
    splits = json.loads(args.splits.read_text())["splits"]
    if args.pilot:
        splits = {"train": splits["train"][:2], "validation": splits["validation"][:1]}
    args.output.mkdir(parents=True, exist_ok=False)
    training, validation, statistics = load_training_data(args.data, splits)
    np.savez(args.output / "normalization.npz", **statistics)
    config = {
        "epochs": args.epochs,
        "points": args.points,
        "seed": args.seed,
        "pilot": args.pilot,
        "split_path": str(args.splits.resolve()),
        "reference_path": str(args.reference.resolve()),
        "torch_version": str(torch.__version__),
        "device": torch.cuda.get_device_name(),
        "models": args.models,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for kind in args.models:
        model = train_model(
            kind,
            training,
            validation,
            statistics,
            args,
            torch.device("cuda"),
            args.output,
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
