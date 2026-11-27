"""Side-by-side pressure and error maps for Transolver against reference models."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


def _draw(axis, polygons, values, cmap, norm, title, bounds=None):
    collection = PolyCollection(
        polygons,
        array=values,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        antialiased=False,
    )
    axis.add_collection(collection)
    if bounds is None:
        coords = np.concatenate(polygons)
        bounds = (
            coords[:, 0].min(),
            coords[:, 0].max(),
            coords[:, 1].min(),
            coords[:, 1].max(),
        )
    axis.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="x [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title, fontsize=9)
    return collection


def near_body_bounds(centroids, wall_distance, threshold=0.1, padding=0.5):
    """Axis limits (xmin, xmax, ymin, ymax) around cells within ``threshold`` of the wall."""
    near = centroids[wall_distance <= threshold, :2]
    if not len(near):
        raise ValueError("No cells lie within the near-body threshold.")
    low, high = near.min(axis=0), near.max(axis=0)
    margin = (high - low) * padding
    return (
        low[0] - margin[0],
        high[0] + margin[0],
        low[1] - margin[1],
        high[1] + margin[1],
    )


def plot_benchmark_comparison(
    polygons, actual, predictions, metrics, title, output, zoom
):
    """Plot actual CFD and each prediction with one pressure and one error scale.

    Rows: full-domain pressure, full-domain |error|, near-body pressure (``zoom``).
    ``predictions`` and ``metrics`` are dicts keyed by model label, in display order.
    """
    labels = list(predictions)
    pressure_norm = plt.Normalize(
        min(float(actual.min()), *(float(p.min()) for p in predictions.values())),
        max(float(actual.max()), *(float(p.max()) for p in predictions.values())),
    )
    errors = {label: np.abs(predictions[label] - actual) for label in labels}
    # Clip at the 99th percentile so a few extreme cells do not wash out the maps.
    error_high = max(float(np.percentile(error, 99)) for error in errors.values())
    error_norm = plt.Normalize(0.0, max(error_high, 1e-12))
    columns = len(labels) + 1
    figure, axes = plt.subplots(
        3,
        columns,
        figsize=(5.2 * columns, 10),
        layout="constrained",
        squeeze=False,
        gridspec_kw={"height_ratios": (1, 1, 2)},
    )
    pressure = _draw(axes[0, 0], polygons, actual, "turbo", pressure_norm, "Actual CFD")
    axes[1, 0].set_axis_off()
    _draw(
        axes[2, 0],
        polygons,
        actual,
        "turbo",
        pressure_norm,
        "Actual CFD | near body",
        zoom,
    )
    for column, label in enumerate(labels, start=1):
        row = metrics[label]
        near = row.get("near_wall_rmse_pa")
        caption = f"RMSE {row['rmse_pa']:.2f} Pa | near-body " + (
            f"{near:.2f} Pa" if near is not None else "n/a"
        )
        _draw(
            axes[0, column],
            polygons,
            predictions[label],
            "turbo",
            pressure_norm,
            f"{label}\n{caption}",
        )
        error = _draw(
            axes[1, column],
            polygons,
            errors[label],
            "magma",
            error_norm,
            f"{label} | |error|",
        )
        _draw(
            axes[2, column],
            polygons,
            predictions[label],
            "turbo",
            pressure_norm,
            f"{label} | near body | near-body RMSE "
            + (f"{near:.2f} Pa" if near is not None else "n/a"),
            zoom,
        )
    for row_axes in axes:
        row_axes[0].set_ylabel("y [m]")
    figure.colorbar(pressure, ax=axes[0, :], label="Pressure [Pa]")
    figure.colorbar(error, ax=axes[1, :], label="|Error| [Pa] (p99 clip)")
    figure.colorbar(pressure, ax=axes[2, :], label="Pressure [Pa]", shrink=0.8)
    figure.suptitle(title)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def _draw_scatter(axis, positions, values, cmap, norm, title, bounds=None):
    """Draw one AirfRANS point cloud with one shared scale."""
    collection = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=values,
        cmap=cmap,
        norm=norm,
        s=1.2,
        edgecolors="none",
        rasterized=True,
    )
    if bounds is None:
        axis.set(
            xlim=(positions[:, 0].min(), positions[:, 0].max()),
            ylim=(positions[:, 1].min(), positions[:, 1].max()),
            xlabel="x [m]",
        )
    else:
        axis.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="x [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title, fontsize=9)
    return collection


def plot_airfrans_comparison(positions, actual, predictions, title, output, zoom=None):
    """Plot CFD and paired Transolver predictions with full-domain and zoomed views."""
    labels = list(predictions)
    pressure_norm = plt.Normalize(
        min(float(actual.min()), *(float(p.min()) for p in predictions.values())),
        max(float(actual.max()), *(float(p.max()) for p in predictions.values())),
    )
    errors = {label: np.abs(predictions[label] - actual) for label in labels}
    error_high = max(float(np.percentile(error, 99)) for error in errors.values())
    error_norm = plt.Normalize(0.0, max(error_high, 1e-12))
    if zoom is None:
        zoom = (-0.35, 1.35, -0.45, 0.45)

    columns = len(labels) + 1
    figure, axes = plt.subplots(
        3,
        columns,
        figsize=(5.2 * columns, 10),
        layout="constrained",
        squeeze=False,
        gridspec_kw={"height_ratios": (1, 1, 2)},
    )
    pressure = _draw_scatter(
        axes[0, 0], positions, actual, "turbo", pressure_norm, "Actual CFD"
    )
    axes[1, 0].set_axis_off()
    _draw_scatter(
        axes[2, 0],
        positions,
        actual,
        "turbo",
        pressure_norm,
        "Actual CFD | near airfoil",
        zoom,
    )

    for column, label in enumerate(labels, start=1):
        _draw_scatter(
            axes[0, column],
            positions,
            predictions[label],
            "turbo",
            pressure_norm,
            f"{label}",
        )
        error = _draw_scatter(
            axes[1, column],
            positions,
            errors[label],
            "magma",
            error_norm,
            f"{label} | |error|",
        )
        _draw_scatter(
            axes[2, column],
            positions,
            predictions[label],
            "turbo",
            pressure_norm,
            f"{label} | near airfoil",
            zoom,
        )

    for row_axes in axes:
        row_axes[0].set_ylabel("y [m]")
    figure.colorbar(pressure, ax=axes[0, :], label="Pressure [dataset units]")
    figure.colorbar(
        error, ax=axes[1, :], label="Absolute pressure error [dataset units] (p99 clip)"
    )
    figure.colorbar(
        pressure, ax=axes[2, :], label="Pressure [dataset units]", shrink=0.8
    )
    figure.suptitle(title)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
