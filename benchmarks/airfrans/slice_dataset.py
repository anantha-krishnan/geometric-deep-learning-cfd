"""Materialize the reduced AirfRANS splits with only model-required mesh files."""

import argparse
import json
import shutil
from pathlib import Path


def create_reduced_dataset(source_root, splits_path, output_root):
    source_root = Path(source_root)
    splits_path = Path(splits_path)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")

    source_manifest = json.loads((source_root / "manifest.json").read_text())
    selected = json.loads(splits_path.read_text())
    split_names = selected["splits"]
    train = split_names["train"]
    validation = split_names["validation"]
    test = split_names["test"]
    all_names = train + validation + test
    if len(set(all_names)) != len(all_names):
        raise ValueError(
            "Reduced training, validation and test splits must be disjoint."
        )
    if not set(train + validation) <= set(source_manifest["full_train"]):
        raise ValueError("Reduced training/validation contains a non-training case.")
    if not set(test) <= set(source_manifest["full_test"]):
        raise ValueError("Reduced test set contains a non-test case.")

    required_files = {}
    for name in all_names:
        files = [
            source_root / name / f"{name}_internal.vtu",
            source_root / name / f"{name}_aerofoil.vtp",
        ]
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing required files for {name}: {missing}")
        required_files[name] = files

    output_root.mkdir(parents=True)
    for name, files in required_files.items():
        target = output_root / name
        target.mkdir()
        for path in files:
            shutil.copy2(path, target / path.name)

    train_validation = set(train + validation)
    reduced_manifest = {
        "full_train": [
            name for name in source_manifest["full_train"] if name in train_validation
        ],
        "full_test": [
            name for name in source_manifest["full_test"] if name in set(test)
        ],
        "reduction": {
            "source": "AirfRANS full-data manifest",
            "seed": selected.get("seed"),
            "train_count": len(train),
            "validation_count": len(validation),
            "test_count": len(test),
            "included_files": ["*_internal.vtu", "*_aerofoil.vtp"],
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(reduced_manifest, indent=2) + "\n"
    )
    selected["source_manifest"] = "AirfRANS original full-data manifest.json"
    (output_root / "splits.json").write_text(json.dumps(selected, indent=2) + "\n")
    (output_root / "README.txt").write_text(
        "Reduced AirfRANS full-data benchmark\n"
        f"Cases: {len(train)} train, {len(validation)} validation, {len(test)} test.\n"
        "Use the included splits.json as-is; do not regenerate splits from this reduced manifest.\n"
        "Only internal volume and airfoil surface meshes required by the benchmark are included.\n"
        "AirfRANS data license: ODbL-1.0; retain attribution and share derivatives under ODbL.\n"
        "Source: https://github.com/Extrality/AirfRANS\n"
    )
    return {
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "files": 2 * len(all_names),
        "output": str(output_root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = create_reduced_dataset(args.data, args.splits, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
