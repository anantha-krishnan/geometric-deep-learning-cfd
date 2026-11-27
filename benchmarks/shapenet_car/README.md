# ShapeNet-Car CFD benchmark foundation

This uses the Umetani and Bickel CFD dataset and the [Transolver car benchmark](https://github.com/thuml/Transolver/tree/main/Car-Design-ShapeNetCar). The downloaded `mlcfd_data/training_data` directory contains nine `param*.tar.gz` folds. Both these archives and extracted `param0` through `param8` directories are supported. Reading individual cases from compressed archives is slow; extract the folds for repeated training. Keep the original data and check its applicable terms before redistribution.

Discover folds and load a case from the project root:

```python
from pathlib import Path
from benchmarks.shapenet_car.data import load_case
from benchmarks.shapenet_car.splits import discover_folds, split_fold

root = Path("public_benchmarks/mlcfd_shapenet_data/mlcfd_data/training_data")
folds = discover_folds(root)
train_names, held_out_names = split_fold(folds, fold_id=0)
case = load_case(root, held_out_names[0])
```

The upstream fold protocol holds out one of nine folds. It does not provide a separate validation partition; select one from the training folds when tuning, and retain the held-out fold for final evaluation. The downloaded archive has 889 usable cases, with distinct shape IDs.

Each case contains `pos` (3D coordinates), `x` (coordinates, nearest-surface distance, and direction or surface normal), `y` (three velocity components and surface pressure), `surface`, `edge_index` and `true_cd`. Exterior pressure is a placeholder, not a CFD label. Graph edges connect every pair of vertices in each surface quad and volume hex, including self-edges, matching the upstream loader. Use the existing `benchmarks.airfrans.data.fit_normalization` and `normalize_case` utilities, fitting only on training cases. Their statistics follow the reference's all-node convention, including the exterior pressure placeholders.

`benchmarks.shapenet_car.training` supplies `BenchmarkModel` adapters for `gnn`, `local_transolver`, and `reference`, plus `field_loss` and `field_scores`. The reference adapter takes `reference_path` pointing to a clone of the Transolver repository. Its defaults match the authors' car model (width 256, eight layers and heads); our local models default to width 32, three layers and four heads. Training loss uses velocity at all nodes and pressure on surface nodes, with pressure weight 0.5. Field scores expect **denormalized** predictions and targets and calculate relative L2 errors for exterior velocity and surface pressure. Zero target norm produces NaN.

This is a data and model foundation, not a complete training or drag-evaluation runner. `true_cd` is the provided coefficient; the upstream drag metric computes drag from surface predictions using additional geometry and physics. Do not treat the field metrics as a drag comparison or claim published parity without matching the complete protocol and evaluation.