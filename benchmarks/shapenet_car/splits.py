"""Discover the nine original ShapeNet-Car CFD folds."""

import tarfile
from pathlib import Path, PurePosixPath

REQUIRED_FILES = {"quadpress_smpl.vtk", "hexvelo_smpl.vtk", "cd.txt"}


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
