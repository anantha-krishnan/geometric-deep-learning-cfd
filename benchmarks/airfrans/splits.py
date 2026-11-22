"""Select reproducible subsets of the original AirfRANS full-data split."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp


def parse_case(name):
    parts = name.split("_")
    if parts[:2] != ["airFoil2D", "SST"] or len(parts) not in (7, 8):
        raise ValueError(f"Invalid AirfRANS case name: {name}")
    shape = [float(value) for value in parts[4:]]
    return {
        "family": 4 if len(shape) == 3 else 5,
        "speed": float(parts[2]),
        "angle": float(parts[3]),
        "shape": shape,
        "thickness": shape[-1],
    }


def _features(cases):
    records = [parse_case(name) for name in cases]
    columns = []
    for family in (4, 5):
        mask = np.array([record["family"] == family for record in records])
        columns.append(mask.astype(float))
        for shape_index in range(3 if family == 4 else 4):
            values = np.array(
                [
                    record["shape"][shape_index] if active else 0
                    for record, active in zip(records, mask)
                ]
            )
            if mask.any():
                columns.extend(_quantile_columns(values, mask))
    for name in ("speed", "angle", "thickness"):
        values = np.array([record[name] for record in records])
        columns.extend(_quantile_columns(values, np.ones(len(cases), dtype=bool)))
    speeds = np.array([record["speed"] for record in records])
    angles = np.array([record["angle"] for record in records])
    speed_bins = np.searchsorted(np.quantile(speeds, [1 / 3, 2 / 3]), speeds)
    angle_bins = np.searchsorted(np.quantile(angles, [1 / 3, 2 / 3]), angles)
    columns.extend(
        ((speed_bins == speed_bin) & (angle_bins == angle_bin)).astype(float)
        for speed_bin in range(3)
        for angle_bin in range(3)
    )
    return np.column_stack(columns)


def _quantile_columns(values, mask):
    boundaries = np.unique(np.quantile(values[mask], [0.2, 0.4, 0.6, 0.8]))
    bins = np.searchsorted(boundaries, values)
    return [
        (mask & (bins == index)).astype(float) for index in range(len(boundaries) + 1)
    ]


def _select(cases, count, rng):
    if not 0 < count <= len(cases):
        raise ValueError(
            "Subset sizes must be positive and no larger than their source split."
        )
    features = _features(cases)
    target = features.mean(axis=0)
    order = rng.permutation(len(cases))
    selected, remaining = order[:count].copy(), order[count:].copy()
    if not len(remaining):
        return [cases[index] for index in selected]
    current = features[selected].mean(axis=0)
    score = np.square(current - target).sum()
    for _ in range(5000):
        selected_slot = rng.integers(len(selected))
        remaining_slot = rng.integers(len(remaining))
        candidate = (
            current
            + (features[remaining[remaining_slot]] - features[selected[selected_slot]])
            / count
        )
        candidate_score = np.square(candidate - target).sum()
        if candidate_score < score:
            current, score = candidate, candidate_score
            selected[selected_slot], remaining[remaining_slot] = (
                remaining[remaining_slot],
                selected[selected_slot],
            )
    return [cases[index] for index in sorted(selected)]


def build_reduced_splits(
    manifest, train_count=96, validation_count=16, test_count=40, seed=2026
):
    full_train = manifest["full_train"]
    full_test = manifest["full_test"]
    if len(set(full_train)) != len(full_train) or len(set(full_test)) != len(full_test):
        raise ValueError("The source manifest contains duplicate cases.")
    if set(full_train) & set(full_test):
        raise ValueError("The source training and test sets overlap.")
    validation_size = int(0.1 * len(full_train))
    if not validation_size:
        raise ValueError("The original training split is too small for validation.")
    rng = np.random.default_rng(seed)
    return {
        "train": _select(full_train[:-validation_size], train_count, rng),
        "validation": _select(full_train[-validation_size:], validation_count, rng),
        "test": _select(full_test, test_count, rng),
    }


def distribution_report(source, subset):
    source_records = [parse_case(name) for name in source]
    subset_records = [parse_case(name) for name in subset]
    result = {}
    for name in ("speed", "angle", "thickness"):
        original = np.array([record[name] for record in source_records])
        reduced = np.array([record[name] for record in subset_records])
        result[name] = {
            "source_quantiles": np.quantile(original, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            "subset_quantiles": np.quantile(reduced, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            "ks_distance": float(ks_2samp(original, reduced).statistic),
        }
    result["family_fraction"] = {
        str(family): {
            "source": sum(record["family"] == family for record in source_records)
            / len(source),
            "subset": sum(record["family"] == family for record in subset_records)
            / len(subset),
        }
        for family in (4, 5)
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train", type=int, default=96)
    parser.add_argument("--validation", type=int, default=16)
    parser.add_argument("--test", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    splits = build_reduced_splits(
        manifest, args.train, args.validation, args.test, args.seed
    )
    cutoff = int(0.1 * len(manifest["full_train"]))
    sources = {
        "train": manifest["full_train"][:-cutoff],
        "validation": manifest["full_train"][-cutoff:],
        "test": manifest["full_test"],
    }
    result = {
        "seed": args.seed,
        "source_manifest": str(args.manifest.resolve()),
        "splits": splits,
        "distribution": {
            name: distribution_report(sources[name], cases)
            for name, cases in splits.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
