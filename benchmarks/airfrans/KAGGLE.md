# Kaggle AirfRANS Run

This runs the authors' Transolver implementation on 96 training cases, 16 validation cases and 40 held-out test cases. Training samples 32,000 mesh nodes per step.

## Kaggle datasets

Upload two separate Kaggle Datasets. Do not zip either one.

The code Dataset must have this structure at its root:

```text
benchmarks/__init__.py
benchmarks/airfrans/__init__.py
benchmarks/airfrans/data.py
benchmarks/airfrans/evaluation.py
benchmarks/airfrans/evaluate.py
benchmarks/airfrans/requirements-kaggle.txt
benchmarks/airfrans/splits.py
benchmarks/airfrans/train.py
benchmarks/airfrans/training.py
```

The data Dataset must have `manifest.json`, `splits.json`, and the selected case folders at its root. Each case folder must contain its `*_internal.vtu` and `*_aerofoil.vtp` files. Upload the contents of the reduced dataset directory so its mounted root is the data root.

Attach both Datasets to a Kaggle Notebook, enable a GPU accelerator, and enable Internet to clone the Transolver reference implementation. The example below assumes the Dataset slugs are `airfrans-code` and `airfrans-reduced-96-16-40`.

```python
from pathlib import Path
import os
import subprocess
import sys

CODE_ROOT = Path("/kaggle/input/airfrans-code")
DATA_ROOT = Path("/kaggle/input/airfrans-reduced-96-16-40")
REFERENCE_ROOT = Path("/kaggle/working/Transolver")
RUN_ROOT = Path("/kaggle/working/airfrans_run")

assert (DATA_ROOT / "splits.json").is_file(), f"Missing splits.json in {DATA_ROOT}"
assert (DATA_ROOT / "manifest.json").is_file(), f"Missing manifest.json in {DATA_ROOT}"
assert (CODE_ROOT / "benchmarks/airfrans/train.py").is_file()

env = os.environ.copy()
env["PYTHONPATH"] = str(CODE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r",
     str(CODE_ROOT / "benchmarks/airfrans/requirements-kaggle.txt")],
    check=True,
    env=env,
)
subprocess.run(
    ["git", "clone", "-q", "--depth", "1",
     "https://github.com/thuml/Transolver.git", str(REFERENCE_ROOT)],
    check=True,
)

subprocess.run(
    [sys.executable, "-m", "benchmarks.airfrans.train",
     "--data", str(DATA_ROOT),
     "--splits", str(DATA_ROOT / "splits.json"),
     "--reference", str(REFERENCE_ROOT),
     "--output", str(RUN_ROOT),
     "--models", "reference",
     "--epochs", "30",
     "--points", "32000",
     "--validation-interval", "5",
     "--seed", "2026"],
    check=True,
    env=env,
)
subprocess.run(
    [sys.executable, "-m", "benchmarks.airfrans.evaluate",
     "--data", str(DATA_ROOT),
     "--splits", str(DATA_ROOT / "splits.json"),
     "--reference", str(REFERENCE_ROOT),
     "--run-dir", str(RUN_ROOT)],
    check=True,
    env=env,
)
```

If Internet is disabled, attach the Transolver repository as a third Kaggle Dataset and set `REFERENCE_ROOT` to its mounted path. Keep the included `splits.json` unchanged.

The output directory must be new. The script saves the latest completed epoch but does not yet implement optimizer-state resume. Download or save `/kaggle/working/airfrans_run` as notebook output after the run.

## Run the latent-attention GNN

The repository's `latentattention32` GNN uses the same 96/16/40 case split. It builds directed mesh edges from each internal VTU, trains on induced 32,000-node subgraphs, and predicts all four AirfRANS fields. This model does not need the upstream Transolver checkout. Install the requirements first, then use:

```python
from pathlib import Path
import os
import subprocess
import sys

CODE_ROOT = Path("/kaggle/input/datasets/ananthakrishnan86/gdl-codebase")
DATA_ROOT = Path("/kaggle/input/datasets/salitlokhande8/airfrans-dataset-zip/Dataset")
SPLITS_PATH = CODE_ROOT / "benchmarks/airfrans/kaggle_splits_96_16_40.json"
RUN_ROOT = Path("/kaggle/working/airfrans_gnn_latentattention32_30epoch")

env = os.environ.copy()
env["PYTHONPATH"] = str(CODE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

subprocess.run(
    [
        sys.executable, "-m", "benchmarks.airfrans.train",
        "--data", str(DATA_ROOT),
        "--splits", str(SPLITS_PATH),
        "--output", str(RUN_ROOT),
        "--models", "gnn",
        "--epochs", "30",
        "--points", "32000",
        "--validation-interval", "5",
        "--seed", "2026",
    ],
    check=True,
    env=env,
)
subprocess.run(
    [
        sys.executable, "-m", "benchmarks.airfrans.evaluate",
        "--data", str(DATA_ROOT),
        "--splits", str(SPLITS_PATH),
        "--run-dir", str(RUN_ROOT),
    ],
    check=True,
    env=env,
)
```

## Outputs and interpretation

- `airfrans_run/reference.pt`: latest completed checkpoint, training history, seed, point count and peak GPU memory.
- `airfrans_run/normalization.npz` and `config.json`: training-only normalization and run configuration.
- `airfrans_run/evaluation/metrics.json`: per-case and aggregate volume/surface pressure MSE, lift relative error and lift Spearman correlation.
- `airfrans_run/evaluation/*.png`: first three held-out CFD, prediction and absolute-error comparisons on shared scales.

The reduced test score is a screening result, not a direct reproduction of the paper's 200-case result. The paper's volume and surface pressure metrics are normalized MSE (the authors corrected the paper's relative-L2 label). Preserve the split JSON and configuration with reported scores.
