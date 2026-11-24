"""Create paired upstream/local pressure comparison images."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from benchmarks.transolver.plots import plot_airfrans_comparison


def plot_case(reference_npz, local_npz, output, title):
    with np.load(reference_npz, allow_pickle=False) as reference:
        positions = reference["positions"]
        actual = reference["actual_pressure"]
        upstream = reference["reference_pressure"]
    with np.load(local_npz, allow_pickle=False) as local:
        mine = local["local_pressure"]

    plot_airfrans_comparison(
        positions,
        actual,
        {
            "Authors' Transolver": upstream,
            "Your Transolver": mine,
        },
        title,
        output,
        zoom=(-0.35, 1.35, -0.45, 0.45),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_run", type=Path)
    parser.add_argument("local_run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    reference_files = {
        path.stem: path for path in (args.reference_run / "evaluation").glob("*.npz")
    }
    local_files = {
        path.stem: path for path in (args.local_run / "evaluation").glob("*.npz")
    }
    names = sorted(set(reference_files) & set(local_files))
    if not names:
        raise ValueError("The two runs have no matching held-out NPZ files.")

    args.output.mkdir(parents=True, exist_ok=False)
    metrics = ["31.812", "85.752", "90.415", "92.85", "61.728", "67.783"]
    selected = []
    for name in names:
        if any(token in name for token in metrics):
            selected.append(name)
    if not selected:
        selected = names[:6]

    for name in selected:
        plot_case(
            reference_files[name],
            local_files[name],
            args.output / f"{name}_paired_comparison.png",
            name.replace("airFoil2D_SST_", "AirfRANS case "),
        )
    print(f"Wrote {len(selected)} paired comparisons to {args.output}")


if __name__ == "__main__":
    main()
