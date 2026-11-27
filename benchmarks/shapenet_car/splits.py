"""Discover the nine original ShapeNet-Car CFD folds."""

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
import vtk
from scipy.stats import ks_2samp

REQUIRED_FILES = {"quadpress_smpl.vtk", "hexvelo_smpl.vtk", "cd.txt"}
GEOMETRY_FEATURES = (
    "extent_x",
    "extent_y",
    "extent_z",
    "surface_points",
    "volume_points",
)


def _archive_cases(path, fold):
    cases = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if member.isfile() and len(parts) == 3 and parts[0] == fold:
                cases.setdefault(parts[1], set()).add(parts[2])
    return cases


def _directory_cases(path):
    return {
        folder.name: {file.name for file in folder.iterdir() if file.is_file()}
        for folder in path.iterdir()
        if folder.is_dir()
    }


def discover_folds(root):
    root = Path(root)
    folds = []
    for index in range(9):
        fold = f"param{index}"
        directory = root / fold
        archive = root / f"{fold}.tar.gz"
        if directory.is_dir():
            cases = _directory_cases(directory)
        elif archive.is_file():
            cases = _archive_cases(archive, fold)
        else:
            raise FileNotFoundError(f"Missing ShapeNet-Car fold: {fold}")
        if not cases:
            raise ValueError(f"No cases found in {fold}.")
        cases = {
            name: files
            for name, files in cases.items()
            if files & {"quadpress_smpl.vtk", "hexvelo_smpl.vtk"}
        }
        if not cases:
            raise ValueError(f"No CFD cases found in {fold}.")
        for name, files in cases.items():
            missing = REQUIRED_FILES - files
            if missing:
                raise ValueError(
                    f"Incomplete case {fold}/{name}: missing {sorted(missing)}"
                )
        folds.append([f"{fold}/{name}" for name in sorted(cases)])
    return folds


def split_fold(folds, fold_id):
    if len(folds) != 9 or not 0 <= fold_id < 9:
        raise ValueError("Expected nine folds and a held-out fold between 0 and 8.")
    return (
        [case for index, fold in enumerate(folds) if index != fold_id for case in fold],
        list(folds[fold_id]),
    )


def _select_representative(cases, features, count, rng):
    if not 0 < count <= len(cases):
        raise ValueError("Subset size must be positive and fit its source pool.")
    columns = []
    for feature in GEOMETRY_FEATURES:
        values = np.asarray([features[case][feature] for case in cases], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {feature} in source features.")
        boundaries = np.unique(np.quantile(values, [0.2, 0.4, 0.6, 0.8]))
        bins = np.searchsorted(boundaries, values)
        columns.extend(
            (bins == index).astype(float) for index in range(len(boundaries) + 1)
        )
    for fold in sorted({case.split("/")[0] for case in cases}):
        columns.append(2 * np.asarray([case.startswith(fold + "/") for case in cases]))
    matrix = np.column_stack(columns)
    target = matrix.mean(axis=0)
    order = rng.permutation(len(cases))
    selected, remaining = order[:count].copy(), order[count:].copy()
    if len(remaining):
        current = matrix[selected].mean(axis=0)
        score = np.square(current - target).sum()
        for _ in range(5000):
            selected_slot = rng.integers(len(selected))
            remaining_slot = rng.integers(len(remaining))
            candidate = (
                current
                + (matrix[remaining[remaining_slot]] - matrix[selected[selected_slot]])
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
    folds,
    features,
    train_count=96,
    validation_count=16,
    test_count=40,
    seed=2026,
    fold_id=0,
):
    train_pool, test_pool = split_fold(folds, fold_id)
    all_cases = train_pool + test_pool
    if len(set(all_cases)) != len(all_cases):
        raise ValueError("Source folds contain duplicate cases.")
    if train_count + validation_count > len(train_pool):
        raise ValueError("Training and validation exceed the available source pool.")
    rng = np.random.default_rng(seed)
    validation = _select_representative(train_pool, features, validation_count, rng)
    validation_set = set(validation)
    remaining = [case for case in train_pool if case not in validation_set]
    return {
        "train": _select_representative(remaining, features, train_count, rng),
        "validation": validation,
        "test": _select_representative(test_pool, features, test_count, rng),
    }


def _read_geometry(content, filename):
    reader = vtk.vtkUnstructuredGridReader()
    reader.ReadFromInputStringOn()
    reader.SetInputString(content)
    reader.Update()
    grid = reader.GetOutput()
    points = grid.GetNumberOfPoints()
    if not points:
        raise ValueError(f"Empty VTK mesh: {filename}")
    if filename == "hexvelo_smpl.vtk":
        return {"volume_points": points}
    bounds = grid.GetBounds()
    return {
        "surface_points": points,
        "extent_x": bounds[1] - bounds[0],
        "extent_y": bounds[3] - bounds[2],
        "extent_z": bounds[5] - bounds[4],
    }


def read_geometry_features(root, folds):
    root = Path(root)
    features = {case: {} for fold in folds for case in fold}
    filenames = {"hexvelo_smpl.vtk", "quadpress_smpl.vtk"}
    for index, fold in enumerate(folds):
        directory = root / f"param{index}"
        if directory.is_dir():
            for case in fold:
                for filename in filenames:
                    features[case].update(
                        _read_geometry((root / case / filename).read_bytes(), filename)
                    )
        else:
            with tarfile.open(root / f"param{index}.tar.gz", "r|gz") as archive:
                for member in archive:
                    parts = PurePosixPath(member.name).parts
                    if (
                        not member.isfile()
                        or len(parts) != 3
                        or parts[-1] not in filenames
                    ):
                        continue
                    case = "/".join(parts[:2])
                    if case not in features:
                        continue
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Cannot read {member.name}")
                    features[case].update(_read_geometry(stream.read(), parts[-1]))
        print(f"Read geometry for param{index}: {len(fold)} cases", flush=True)
    for case, values in features.items():
        if set(values) != set(GEOMETRY_FEATURES):
            raise ValueError(f"Incomplete geometry features for {case}")
    return features


def distribution_report(source, subset, features):
    report = {}
    for name in GEOMETRY_FEATURES:
        original = np.asarray([features[case][name] for case in source])
        selected = np.asarray([features[case][name] for case in subset])
        report[name] = {
            "source_quantiles": np.quantile(original, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            "subset_quantiles": np.quantile(selected, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            "ks_distance": float(ks_2samp(original, selected).statistic),
        }
    report["fold_fraction"] = {
        fold: {
            "source": sum(case.startswith(fold + "/") for case in source) / len(source),
            "subset": sum(case.startswith(fold + "/") for case in subset) / len(subset),
        }
        for fold in sorted({case.split("/")[0] for case in source})
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train", type=int, default=96)
    parser.add_argument("--validation", type=int, default=16)
    parser.add_argument("--test", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    folds = discover_folds(args.root)
    features = read_geometry_features(args.root, folds)
    splits = build_reduced_splits(
        folds, features, args.train, args.validation, args.test, args.seed, args.fold
    )
    train_pool, test_pool = split_fold(folds, args.fold)
    sources = {"train": train_pool, "validation": train_pool, "test": test_pool}
    result = {
        "seed": args.seed,
        "held_out_fold": args.fold,
        "source_counts": {"train_validation": len(train_pool), "test": len(test_pool)},
        "splits": splits,
        "distribution": {
            name: distribution_report(sources[name], cases, features)
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
