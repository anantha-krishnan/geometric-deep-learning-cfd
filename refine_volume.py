"""Resolution-stability test: predict on a 2x refined copy of each held-out mesh.

Example:
    uv run python refine_volume.py --model "GNN geometry" CHECKPOINT geometry

Every polygon cell is split into one quadrilateral per vertex (vertex, edge midpoint,
cell centre, previous edge midpoint), which halves the spacing of quad cells. No CFD
exists on the refined mesh, so child predictions are area-averaged back onto their
parent cells and compared with the CFD and with the prediction on the original mesh.
Wall distance at the new cells is the exact distance to the WALL boundary segments.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import train_volume as training
from benchmarks.transolver.plots import _draw, near_body_bounds
from dataset import (
    BODY_BOUNDARY,
    compute_volume_edge_attr,
    load_all_starccm_volume_designs,
)
from validate_volume import cell_vertex_loops

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT = (
    PROJECT_ROOT
    / "volume_results"
    / "gpu_latent_attention_geometry_20261122"
    / "split.json"
)


def read_volume_topology(path):
    """Vertex coordinates, ordered cell loops, interior faces and WALL segments."""
    with h5py.File(path, "r") as handle:
        mesh = handle["Meshes/FvMesh 1/Regions/Domain"]
        coords = np.asarray(mesh["Coord"], dtype=np.float64)
        num_cells = len(handle["States/ROM/Regions/Domain/CellData/Pressure"])
        cell_edges = [[] for _ in range(num_cells)]
        interior_cells = np.asarray(mesh["FaceCellIndex"], dtype=np.int64)
        interior_faces = np.asarray(mesh["VertexList/VertexList"], dtype=np.int64)
        interior_faces = interior_faces.reshape(-1, 2)
        for (owner, neighbor), face in zip(interior_cells, interior_faces):
            cell_edges[owner].append(face)
            cell_edges[neighbor].append(face)
        wall = None
        for name, boundary in mesh["Boundaries"].items():
            faces = np.asarray(boundary["VertexList/VertexList"], dtype=np.int64)
            faces = faces.reshape(-1, 2)
            owners = np.asarray(boundary["FaceCellIndex"], dtype=np.int64)[:, 0]
            for owner, face in zip(owners, faces):
                cell_edges[owner].append(face)
            if name == BODY_BOUNDARY:
                wall = coords[faces, :2]
    if wall is None:
        raise ValueError(f"{path} has no {BODY_BOUNDARY} boundary.")
    return {
        "coords": coords,
        "loops": cell_vertex_loops(cell_edges),
        "interior_cells": interior_cells,
        "interior_faces": interior_faces,
        "wall_segments": wall,
    }


def polygon_area_centroid(points):
    """Shoelace area and area centroid of polygons given as [..., vertices, 2]."""
    x, y = points[..., 0], points[..., 1]
    cross = x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y
    signed = 0.5 * cross.sum(axis=-1)
    cx = ((x + np.roll(x, -1, axis=-1)) * cross).sum(axis=-1) / (6 * signed)
    cy = ((y + np.roll(y, -1, axis=-1)) * cross).sum(axis=-1) / (6 * signed)
    return np.abs(signed), np.stack([cx, cy], axis=-1)


def wall_distance(points, segments, chunk=4096):
    """Exact Euclidean distance from points [N, 2] to line segments [S, 2, 2]."""
    start, direction = segments[:, 0], segments[:, 1] - segments[:, 0]
    length2 = np.maximum((direction**2).sum(axis=1), 1e-30)
    result = np.empty(len(points))
    for low in range(0, len(points), chunk):
        block = points[low : low + chunk, None, :]
        t = np.clip(((block - start) * direction).sum(axis=2) / length2, 0.0, 1.0)
        nearest = start + t[..., None] * direction
        result[low : low + chunk] = np.sqrt(((block - nearest) ** 2).sum(axis=2)).min(1)
    return result


def refine_topology(topology):
    """Split each cell into per-vertex quads; return child geometry and adjacency."""
    coords = topology["coords"][:, :2]
    loops = topology["loops"]
    base = np.cumsum([0] + [len(loop) for loop in loops])
    local = [{vertex: index for index, vertex in enumerate(loop)} for loop in loops]
    polygons, parents, edges = [], [], []
    for cell, loop in enumerate(loops):
        points = coords[loop]
        centre = points.mean(axis=0)
        following = np.roll(points, -1, axis=0)
        previous = np.roll(points, 1, axis=0)
        polygons.extend(
            np.stack(
                [
                    points,
                    0.5 * (points + following),
                    np.broadcast_to(centre, points.shape),
                    0.5 * (points + previous),
                ],
                axis=1,
            )
        )
        parents.extend([cell] * len(loop))
        children = base[cell] + np.arange(len(loop))
        edges.append(np.stack([children, np.roll(children, -1)]))
    for (owner, neighbor), face in zip(
        topology["interior_cells"], topology["interior_faces"]
    ):
        for vertex in face:
            edges.append(
                np.array(
                    [
                        [base[owner] + local[owner][vertex]],
                        [base[neighbor] + local[neighbor][vertex]],
                    ]
                )
            )
    polygons = np.asarray(polygons)
    areas, centroids = polygon_area_centroid(polygons)
    edge_index = np.concatenate(edges, axis=1)
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    return {
        "polygons": polygons,
        "areas": areas,
        "centroids": centroids,
        "parent": np.asarray(parents),
        "edge_index": torch.as_tensor(np.unique(edge_index, axis=1)),
        "wall_distance": wall_distance(centroids, topology["wall_segments"]),
    }


def refined_item(item, refined):
    """Raw volume item on the refined mesh, ready for ``prepare_data``."""
    count = len(refined["areas"])
    geom = torch.tensor(
        np.column_stack(
            [
                refined["centroids"],
                np.full(count, float(item["geom"][0, 2])),
                refined["areas"],
                refined["wall_distance"],
            ]
        ),
        dtype=torch.float32,
    )
    parent = torch.as_tensor(refined["parent"])
    return {
        **{key: item[key] for key in ("params", "design_name", "mesh_kind")},
        **{key: item[key] for key in ("body_bounds", "domain_bounds")},
        "geom": geom,
        "pressure": item["pressure"][parent],
        "edge_index": refined["edge_index"],
        "edge_attr": compute_volume_edge_attr(geom, refined["edge_index"]),
    }


def check_refinement(item, topology, refined):
    """Children must tile their parents exactly; returns the wall-distance check."""
    parent_area = np.bincount(refined["parent"], weights=refined["areas"])
    cell_area = item["geom"][:, 3].double().numpy()
    if not np.allclose(parent_area, cell_area, rtol=1e-4):
        raise ValueError(f"{item['design_name']}: children do not tile parent cells.")
    exported = item["geom"][:, 4].double().numpy()
    geometric = wall_distance(
        item["geom"][:, :2].double().numpy(), topology["wall_segments"]
    )
    return float(np.max(np.abs(exported - geometric)))


def parent_average(values, refined, num_cells):
    weights = refined["areas"]
    totals = np.bincount(
        refined["parent"], weights=weights * values, minlength=num_cells
    )
    return totals / np.bincount(refined["parent"], weights=weights, minlength=num_cells)


def predict_item(model, item, variant, scalers):
    prepared = training.prepare_data([item], variant)
    normalized = training.normalize_data(prepared, scalers)[0]
    prediction = training.predict_pressure(
        model, normalized, scalers[2], torch.device("cpu")
    )
    return prediction.numpy().reshape(-1), prepared[0]


def plot_refinement(
    topology, refined, actual, original, fine, coarse, rows, title, output
):
    parents = [topology["coords"][loop, :2] for loop in topology["loops"]]
    children = list(refined["polygons"])
    wall = topology["wall_segments"].reshape(-1, 2)
    zoom = near_body_bounds(wall, np.zeros(len(wall)), threshold=1.0, padding=1.5)
    low, high = np.percentile(np.concatenate([actual, original, fine]), [0.5, 99.5])
    norm = plt.Normalize(low, high)
    difference = coarse - original
    limit = max(float(np.percentile(np.abs(difference), 99)), 1e-9)
    titles = (
        "Actual CFD (original mesh)",
        f"Original mesh | RMSE {rows['original']['rmse_pa']:.2f} Pa",
        f"2x refined mesh | parent-averaged RMSE {rows['refined']['rmse_pa']:.2f} Pa",
        f"Refined (parent-averaged) - original | {rows['consistency_rmse_pa']:.2f} Pa",
    )
    figure, axes = plt.subplots(2, 4, figsize=(22, 9), layout="constrained")
    for row, bounds in enumerate((None, zoom)):
        pressure = _draw(
            axes[row, 0], parents, actual, "turbo", norm, titles[0], bounds
        )
        _draw(axes[row, 1], parents, original, "turbo", norm, titles[1], bounds)
        _draw(axes[row, 2], children, fine, "turbo", norm, titles[2], bounds)
        change = _draw(
            axes[row, 3], parents, difference, "coolwarm",
            plt.Normalize(-limit, limit), titles[3], bounds,
        )  # fmt: skip
        if row == 0:
            figure.colorbar(pressure, ax=axes[:, :3], label="Pressure [Pa]", shrink=0.6)
            figure.colorbar(change, ax=axes[:, 3], label="Difference [Pa]", shrink=0.6)
    figure.suptitle(title)
    figure.savefig(output, dpi=150, facecolor="white")
    plt.close(figure)


def evaluate_model(label, checkpoint, variant, designs, data_dir, output):
    model, scalers, _ = training.load_checkpoint(checkpoint, torch.device("cpu"))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    metrics = {"original": [], "refined": [], "consistency": []}
    for item in designs:
        path = data_dir / item["design_name"] / "Volume_Dataset.simh"
        topology = read_volume_topology(path)
        refined = refine_topology(topology)
        wall_error = check_refinement(item, topology, refined)
        original, prepared = predict_item(model, item, variant, scalers)
        fine, _ = predict_item(model, refined_item(item, refined), variant, scalers)
        coarse = parent_average(fine, refined, len(original))
        areas = item["geom"][:, 3].double().numpy()
        on_original = training.measure_design(
            torch.from_numpy(original).unsqueeze(1), prepared
        )
        on_refined = training.measure_design(
            torch.from_numpy(coarse).unsqueeze(1), prepared
        )
        consistency = float(
            np.sqrt(np.average((coarse - original) ** 2, weights=areas))
        )
        metrics["original"].append(on_original)
        metrics["refined"].append(on_refined)
        metrics["consistency"].append(consistency)
        row = {
            "design_name": item["design_name"],
            "cells": len(original),
            "refined_cells": len(fine),
            "max_wall_distance_check_m": wall_error,
            "original": on_original,
            "refined": on_refined,
            "consistency_rmse_pa": consistency,
        }
        rows.append(row)
        plot_refinement(
            topology, refined, item["pressure"].numpy().reshape(-1), original, fine,
            coarse, row, f"{label} | {item['design_name']} | held-out, original vs 2x refined mesh",
            output / f"{item['design_name']}_refinement.png",
        )  # fmt: skip
        print(
            f"{label} {item['design_name']}: RMSE {on_original['rmse_pa']:.3f} -> "
            f"{on_refined['rmse_pa']:.3f} Pa, near-wall "
            f"{on_original['near_wall_rmse_pa']:.3f} -> {on_refined['near_wall_rmse_pa']:.3f} Pa, "
            f"consistency {consistency:.3f} Pa, wall-distance check {wall_error:.2e} m",
            flush=True,
        )
    summary = {
        "model": label,
        "checkpoint": str(checkpoint),
        "variant": variant,
        **{
            f"{mesh}_{metric}": float(np.mean([row[metric] for row in metrics[mesh]]))
            for mesh in ("original", "refined")
            for metric in ("rmse_pa", "near_wall_rmse_pa")
        },
        "mean_consistency_rmse_pa": float(np.mean(metrics["consistency"])),
    }
    (output / "refinement_metrics.json").write_text(
        json.dumps({"summary": summary, "designs": rows}, indent=2) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", nargs=3, action="append", required=True,
        metavar=("LABEL", "CHECKPOINT", "VARIANT"),
    )  # fmt: skip
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / training.DATA_DIR
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    names = set(json.loads(args.split.read_text())["outer_test_designs"])
    designs = [
        item
        for item in load_all_starccm_volume_designs(args.data_dir, include_physics=True)
        if item["design_name"] in names
    ]
    summaries = [
        evaluate_model(label, Path(checkpoint), variant, designs, args.data_dir,
                       args.output / label.replace(" ", "_"))
        for label, checkpoint, variant in args.model
    ]  # fmt: skip
    (args.output / "refinement_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
