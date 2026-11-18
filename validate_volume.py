"""Render held-out full-domain pressures on the original STAR-CCM+ cell polygons.

Run: uv run python validate_volume.py --variant all --design Design_1 --no-show
Saved cross-validation predictions are used; no training or inference is needed.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

from train_volume import DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parent
VARIANTS = ("baseline", "coefficient", "geometry")


def build_cell_polygons(coords, cell_edges):
    polygons = []
    for cell, edges in enumerate(cell_edges):
        neighbors = {}
        for start, end in edges:
            neighbors.setdefault(int(start), []).append(int(end))
            neighbors.setdefault(int(end), []).append(int(start))
        if len(neighbors) < 3 or any(len(values) != 2 for values in neighbors.values()):
            raise ValueError(f"Cell {cell} does not have a closed polygon.")
        first = next(iter(neighbors))
        ordered = [first]
        visited = {first}
        previous = None
        current = first
        while True:
            adjacent = neighbors[current]
            following = adjacent[0] if adjacent[0] != previous else adjacent[1]
            if following == first:
                break
            if following in visited:
                raise ValueError(f"Cell {cell} has an invalid polygon loop.")
            ordered.append(following)
            visited.add(following)
            previous, current = current, following
        if len(ordered) != len(neighbors):
            raise ValueError(f"Cell {cell} has disconnected polygon loops.")
        polygons.append(coords[ordered, :2])
    return polygons


def read_mesh(path):
    with h5py.File(path, "r") as handle:
        mesh = handle["Meshes/FvMesh 1/Regions/Domain"]
        coords = np.asarray(mesh["Coord"])
        if not np.allclose(coords[:, 2], coords[0, 2]):
            raise ValueError("Only planar XY meshes are supported.")
        cells = handle["States/ROM/Regions/Domain/CellData"]
        centroids = np.asarray(cells["CentroidFieldFunction"])
        pressure = np.asarray(cells["Pressure"]).reshape(-1)
        cell_edges = [[] for _ in range(len(centroids))]
        for group in [mesh, *mesh["Boundaries"].values()]:
            if np.any(np.asarray(group["VertexList/VertexList_lengths"]) != 2):
                raise ValueError("Expected two vertices per face in a 2D export.")
            edges = np.asarray(group["VertexList/VertexList"]).reshape(-1, 2)
            owners = np.asarray(group["FaceCellIndex"])
            if len(edges) != len(owners):
                raise ValueError("Face connectivity and vertex counts differ.")
            for pair, edge in zip(owners, edges):
                cell_edges[int(pair[0])].append(edge)
                if group.name == mesh.name:
                    cell_edges[int(pair[1])].append(edge)
    return build_cell_polygons(coords, cell_edges), centroids, pressure


def load_validation(results_dir, data_dir, design_name):
    report = json.loads((results_dir / "metrics.json").read_text())
    with np.load(
        results_dir / "predictions" / f"{design_name}.npz", allow_pickle=False
    ) as saved:
        actual = saved["actual_pressure_pa"].reshape(-1)
        predicted = saved["predicted_pressure_pa"].reshape(-1)
        geom = saved["geom"]
        fold_number = int(saved["fold"].item())
    fold = next(entry for entry in report["folds"] if entry["fold"] == fold_number)
    if (
        design_name not in fold["held_out_designs"]
        or design_name in fold["training_designs"]
    ):
        raise ValueError("The selected prediction is not a held-out validation design.")
    polygons, centroids, mesh_pressure = read_mesh(
        data_dir / design_name / "Volume_Dataset.simh"
    )
    if (
        actual.shape != predicted.shape
        or actual.shape != mesh_pressure.shape
        or geom.shape[0] != len(polygons)
        or not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError(
            "Mesh and prediction arrays have incompatible shapes or non-finite fields."
        )
    if not np.allclose(actual, mesh_pressure, rtol=1e-5, atol=1e-5) or not np.allclose(
        geom[:, :3], centroids, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("Saved predictions do not match the current CFD export.")
    metadata = {
        "design_name": design_name,
        "variant": report["config"]["variant"],
        "fold": fold_number,
    }
    return polygons, actual, predicted, metadata


def plot_comparison(polygons, actual, predicted, metadata, pressure_range=None):
    areas = np.array(
        [
            0.5
            * abs(
                np.dot(points[:, 0], np.roll(points[:, 1], -1))
                - np.dot(points[:, 1], np.roll(points[:, 0], -1))
            )
            for points in polygons
        ]
    )
    error = predicted - actual
    rmse = np.sqrt(np.average(error**2, weights=areas))
    variance = np.average(
        (actual - np.average(actual, weights=areas)) ** 2, weights=areas
    )
    score = f"{1 - rmse**2 / variance:.4f}" if variance > 1e-12 else "N/A"
    lower, upper = pressure_range or (
        min(actual.min(), predicted.min()),
        max(actual.max(), predicted.max()),
    )
    if lower == upper:
        lower, upper = lower - 0.5, upper + 0.5
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Pressure limits must be finite and increasing.")
    norm = Normalize(vmin=lower, vmax=upper)
    figure, axes = plt.subplots(
        1, 2, figsize=(16, 5.8), layout="constrained", sharex=True, sharey=True
    )
    coords = np.concatenate(polygons)
    evaluation = metadata.get("evaluation", f"Held-out fold {metadata['fold']}")
    prediction_title = (
        "AI ROM | Training Reconstruction"
        if metadata.get("reconstruction", False)
        else "AI ROM | Held-Out Prediction"
    )
    for axis, values, title in zip(
        axes, (actual, predicted), ("SIM | Actual CFD", prediction_title)
    ):
        collection = PolyCollection(
            polygons,
            array=values,
            cmap="turbo",
            norm=norm,
            edgecolors="none",
            antialiased=False,
        )
        axis.add_collection(collection)
        axis.set_xlim(coords[:, 0].min(), coords[:, 0].max())
        axis.set_ylim(coords[:, 1].min(), coords[:, 1].max())
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title, fontsize=13, loc="left", pad=12)
        axis.set_xlabel("x [m]")
        axis.set_facecolor("white")
    axes[0].set_ylabel("y [m]")
    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())
    extend = (
        "both"
        if minimum < lower and maximum > upper
        else "min"
        if minimum < lower
        else "max"
        if maximum > upper
        else "neither"
    )
    figure.colorbar(
        collection,
        ax=axes,
        orientation="horizontal",
        label="Pressure [Pa]",
        shrink=0.7,
        pad=0.08,
        aspect=55,
        extend=extend,
    )
    figure.suptitle(
        f"Volume Pressure Comparison | {metadata['design_name']} | {metadata['variant']}\n"
        f"{evaluation} | Area-weighted RMSE: {rmse:.3f} Pa | R-squared: {score}",
        fontsize=14,
    )
    print(
        f"{metadata['variant']} / {metadata['design_name']}: {evaluation} | RMSE {rmse:.4f} Pa | R-squared {score}"
    )
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default="Design_1")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument(
        "--results-root", type=Path, default=PROJECT_ROOT / "volume_results"
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "validation_results" / "volume",
    )
    parser.add_argument(
        "--pressure-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Explicit colour limits; colourbar arrows mark clipped values.",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    try:
        comparisons = [
            load_validation(args.results_root / variant, args.data_dir, args.design)
            for variant in variants
        ]
        pressure_range = args.pressure_range or (
            min(
                min(actual.min(), predicted.min())
                for _, actual, predicted, _ in comparisons
            ),
            max(
                max(actual.max(), predicted.max())
                for _, actual, predicted, _ in comparisons
            ),
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for polygons, actual, predicted, metadata in comparisons:
            figure = plot_comparison(
                polygons, actual, predicted, metadata, pressure_range
            )
            output = (
                args.output_dir
                / f"validation_volume_{args.design}_{metadata['variant']}.png"
            )
            figure.savefig(output, dpi=300, facecolor="white")
            print(f"Saved: {output}")
            if args.no_show:
                plt.close(figure)
        print(
            f"Shared pressure scale: [{pressure_range[0]:.4g}, {pressure_range[1]:.4g}] Pa"
        )
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.error(str(error))
    if not args.no_show:
        plt.show()
        plt.close("all")


if __name__ == "__main__":
    main()
