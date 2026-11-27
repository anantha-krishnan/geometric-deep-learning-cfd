"""Evaluate a saved upstream Transolver on the reduced ShapeNet-Car split."""

import argparse
import hashlib
import json
import sys
import types
import urllib.request
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from benchmarks.airfrans.data import normalize_case
from benchmarks.shapenet_car.data import load_case
from benchmarks.shapenet_car.run import resolve_split_paths
from benchmarks.shapenet_car.training import field_scores

UPSTREAM_REVISION = "75e0f67643806a81cd1d3f6adc88dd8c02416fe7"
SOURCE_URL = (
    "https://raw.githubusercontent.com/thuml/Transolver/"
    f"{UPSTREAM_REVISION}/Car-Design-ShapeNetCar/models/Transolver.py"
)
SOURCE_SHA256 = "9dbea9edae30b0d8484a458b4bdb9b33427e9fda815c8011d64f19fb50e14ffa"


def normalization_from_log(log):
    values = log["coef_norm"]
    if len(values) != 4 or [len(value) for value in values] != [7, 7, 4, 4]:
        raise ValueError("Invalid upstream normalization coefficients.")
    return {
        key: np.asarray(value, dtype=np.float32)
        for key, value in zip(("x_mean", "x_std", "y_mean", "y_std"), values)
    }


def load_reference(checkpoint, device):
    package = types.ModuleType("models")
    package.__path__ = []
    module = types.ModuleType("models.Transolver")
    sys.modules["models"] = package
    sys.modules[module.__name__] = module
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Pinned upstream source hash does not match.")
    exec(compile(source, SOURCE_URL, "exec"), module.__dict__)  # noqa: S102
    allowed = [
        module.Model,
        module.MLP,
        module.Transolver_block,
        module.Physics_Attention_Irregular_Mesh,
        set,
        OrderedDict,
        nn.GELU,
        nn.Softmax,
        nn.ModuleList,
        nn.Sequential,
        nn.Dropout,
        nn.Linear,
        nn.LayerNorm,
    ]
    with torch.serialization.safe_globals(allowed):
        return torch.load(checkpoint, map_location=device, weights_only=True).eval()


def plot_pressure(case, prediction, destination):
    surface = case["surface"]
    points = case["pos"][surface]
    actual = case["y"][surface, 3]
    predicted = prediction[surface, 3]
    errors = np.abs(predicted - actual)
    low, high = float(actual.min()), float(actual.max())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, axes_indices in enumerate(((0, 1), (0, 2))):
        for column, (values, title) in enumerate(
            (
                (actual, "CFD"),
                (predicted, "Upstream Transolver"),
                (errors, "Absolute error"),
            )
        ):
            kwargs = (
                {"cmap": "coolwarm", "vmin": low, "vmax": high}
                if column < 2
                else {"cmap": "magma", "vmin": 0, "vmax": float(errors.max())}
            )
            image = axes[row, column].scatter(
                points[:, axes_indices[0]],
                points[:, axes_indices[1]],
                c=values,
                s=2,
                rasterized=True,
                **kwargs,
            )
            axes[row, column].set_title(title)
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xlabel("xyz"[axes_indices[0]])
            axes[row, column].set_ylabel("xyz"[axes_indices[1]])
            fig.colorbar(image, ax=axes[row, column], label="Pressure (dataset units)")
    fig.suptitle(f"Held-out ShapeNet-Car: {case['name']}")
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def evaluate(model, paths, statistics, device, output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with torch.no_grad():
        for root, name in paths:
            raw = load_case(root, name)
            case = normalize_case(raw, statistics)
            inputs = torch.as_tensor(case["x"], device=device)
            positions = torch.as_tensor(case["pos"], device=device)
            values = (
                model((SimpleNamespace(x=inputs, pos=positions), None)).cpu().numpy()
            )
            prediction = values * (statistics["y_std"] + 1e-8) + statistics["y_mean"]
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
            plot_pressure(raw, prediction, output / f"{stem}.png")
            print(json.dumps(row), flush=True)
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("velocity_relative_l2", "pressure_relative_l2")
    }
    (output / "metrics.json").write_text(
        json.dumps(
            {"summary": summary, "cases": rows, "upstream_revision": UPSTREAM_REVISION},
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if args.output.exists():
        parser.error("Output directory already exists.")
    splits = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    mapping = json.loads((args.data / "case_mapping.json").read_text(encoding="utf-8"))
    paths = resolve_split_paths(args.data, splits, mapping)["test"][: args.limit]
    if not paths:
        parser.error("The test split is empty.")
    log = json.loads((args.run / "log_150.json").read_text(encoding="utf-8"))
    statistics = normalization_from_log(log)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_reference(args.run / "model_150.pth", device)
    print(
        f"Loaded {sum(parameter.numel() for parameter in model.parameters())} parameters on {device}.",
        flush=True,
    )
    evaluate(model, paths, statistics, device, args.output)


if __name__ == "__main__":
    main()
