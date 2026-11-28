"""Plot matching held-out ShapeNet-Car reference predictions."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def check_matching_cases(reference, candidate):
    for key in ("positions", "surface", "actual"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Reference and candidate differ in {key}.")


def plot_comparison(
    reference,
    candidate,
    name,
    output,
    additional=None,
    labels=("Full Transolver", "Reduced Transolver"),
):
    surface = reference["surface"]
    points = reference["positions"][surface]
    actual = reference["actual"][surface, 3]
    full = reference["prediction"][surface, 3]
    small = candidate["prediction"][surface, 3]
    low, high = float(actual.min()), float(actual.max())
    predictions = [(full, labels[0]), (small, labels[1])]
    if additional is not None:
        check_matching_cases(reference, additional)
        predictions.append((additional["prediction"][surface, 3], "Hybrid slice GNN"))
    error_limit = max(
        *(float(np.max(np.abs(values - actual))) for values, _ in predictions),
        1e-8,
    )
    norm = np.linalg.norm(actual)
    fields = [(actual, "CFD")]
    fields.extend(
        (
            values,
            f"{label}\nrelative L2: {np.linalg.norm(values - actual) / norm:.3f}"
            if norm
            else label,
        )
        for values, label in predictions
    )
    fields.extend(
        (np.abs(values - actual), f"{label} absolute error")
        for values, label in predictions
    )
    pressure_columns = len(predictions) + 1
    fig, axes = plt.subplots(
        2, len(fields), figsize=(4 * len(fields), 8), constrained_layout=True
    )
    for row, dimensions in enumerate(((0, 1), (0, 2))):
        for column, (values, title) in enumerate(fields):
            kwargs = (
                {"cmap": "coolwarm", "vmin": low, "vmax": high}
                if column < pressure_columns
                else {"cmap": "magma", "vmin": 0, "vmax": error_limit}
            )
            image = axes[row, column].scatter(
                points[:, dimensions[0]],
                points[:, dimensions[1]],
                c=values,
                s=2,
                rasterized=True,
                **kwargs,
            )
            axes[row, column].set_title(title)
            axes[row, column].set_xlabel("xyz"[dimensions[0]])
            axes[row, column].set_ylabel("xyz"[dimensions[1]])
            axes[row, column].set_aspect("equal")
            fig.colorbar(image, ax=axes[row, column], label="Pressure (dataset units)")
    fig.suptitle(f"Held-out ShapeNet-Car: {name}")
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--additional", type=Path)
    parser.add_argument(
        "--labels", nargs=2, default=("Full Transolver", "Reduced Transolver")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_files = {path.name: path for path in args.reference.glob("*.npz")}
    candidate_files = {path.name: path for path in args.candidate.glob("*.npz")}
    if not reference_files or reference_files.keys() != candidate_files.keys():
        parser.error(
            "The reference and candidate case sets must match and be nonempty."
        )
    additional_files = (
        {path.name: path for path in args.additional.glob("*.npz")}
        if args.additional is not None
        else None
    )
    if (
        additional_files is not None
        and reference_files.keys() != additional_files.keys()
    ):
        parser.error("The additional case set must match the references.")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in sorted(reference_files):
        with ExitStack() as stack:
            reference = stack.enter_context(np.load(reference_files[name]))
            candidate = stack.enter_context(np.load(candidate_files[name]))
            additional = (
                stack.enter_context(np.load(additional_files[name]))
                if additional_files is not None
                else None
            )
            check_matching_cases(reference, candidate)
            plot_comparison(
                reference,
                candidate,
                Path(name).stem,
                args.output / f"{Path(name).stem}.png",
                additional=additional,
                labels=args.labels,
            )
        print(name, flush=True)


if __name__ == "__main__":
    main()
