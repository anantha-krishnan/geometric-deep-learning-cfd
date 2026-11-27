"""Materialize a reduced ShapeNet-Car split for the upstream benchmark."""

import argparse
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from benchmarks.shapenet_car.splits import REQUIRED_FILES


def create_reduced_dataset(root, splits, output):
    root = Path(root)
    output = Path(output)
    selected = {}
    for split in ("train", "validation", "test"):
        for case in splits[split]:
            parts = PurePosixPath(case).parts
            if (
                len(parts) != 2
                or parts[0] not in {f"param{index}" for index in range(9)}
                or not parts[1].replace("_", "").isalnum()
                or case in selected
            ):
                raise ValueError(f"Invalid or duplicate case: {case}")
            if split in ("train", "validation") and parts[0] == "param0":
                raise ValueError(
                    "Training and validation must not use the held-out fold."
                )
            if split == "test" and parts[0] != "param0":
                raise ValueError("Test cases must come from the held-out fold.")
            folder = "test_data" if split == "test" else "training_data"
            fold = "param0" if split == "validation" else parts[0]
            selected[case] = Path(folder) / fold / parts[1]
    if len(set(selected.values())) != len(selected):
        raise ValueError("Selected cases collide in the upstream directory layout.")
    output.mkdir(parents=True, exist_ok=False)
    for index in range(9):
        (output / "training_data" / f"param{index}").mkdir(parents=True)
    copied = {case: set() for case in selected}
    for index in range(9):
        fold = f"param{index}"
        cases = [case for case in selected if case.startswith(fold + "/")]
        directory = root / fold
        if directory.is_dir():
            for case in cases:
                target = output / selected[case]
                target.mkdir(parents=True, exist_ok=True)
                for filename in REQUIRED_FILES:
                    shutil.copyfile(root / case / filename, target / filename)
                    copied[case].add(filename)
        elif cases:
            with tarfile.open(root / f"{fold}.tar.gz", "r|gz") as archive:
                for member in archive:
                    parts = PurePosixPath(member.name).parts
                    if (
                        not member.isfile()
                        or len(parts) != 3
                        or parts[-1] not in REQUIRED_FILES
                    ):
                        continue
                    case = "/".join(parts[:2])
                    if case not in selected:
                        continue
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Cannot read {member.name}")
                    target = output / selected[case]
                    target.mkdir(parents=True, exist_ok=True)
                    with (target / parts[-1]).open("wb") as handle:
                        shutil.copyfileobj(stream, handle)
                    copied[case].add(parts[-1])
        print(f"Copied {len(cases)} cases from {fold}", flush=True)
    for case, files in copied.items():
        if files != REQUIRED_FILES:
            raise ValueError(
                f"Missing files for {case}: {sorted(REQUIRED_FILES - files)}"
            )
    (output / "case_mapping.json").write_text(
        json.dumps({case: path.as_posix() for case, path in selected.items()}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("splits", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.splits.read_text(encoding="utf-8"))
    if manifest["held_out_fold"] != 0:
        parser.error("The upstream layout currently requires held-out fold 0.")
    create_reduced_dataset(args.root, manifest["splits"], args.output)


if __name__ == "__main__":
    main()
