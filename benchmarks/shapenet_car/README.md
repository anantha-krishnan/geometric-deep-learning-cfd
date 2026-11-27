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

## Reduced upstream benchmark

`reduced_splits_96_16_40.json` fixes 96 training, 16 validation, and 40 held-out test cases. The original fold 0 is reserved for test; training and validation use folds 1-8, with 12 and 2 cases respectively from each. Selection balances the marginal surface bounding-box dimensions and mesh point counts as well as fold membership. It does not match the joint shape distribution, flow fields or drag, and it does not guarantee coverage of extreme shapes. The JSON reports quantiles and KS distances against the corresponding source pools.

Regenerate the manifest and materialize the selected files from the project root:

```powershell
uv run python -m benchmarks.shapenet_car.splits public_benchmarks/mlcfd_shapenet_data/mlcfd_data/training_data benchmarks/shapenet_car/reduced_splits_new.json
uv run python -m benchmarks.shapenet_car.slice_dataset public_benchmarks/mlcfd_shapenet_data/mlcfd_data/training_data benchmarks/shapenet_car/reduced_splits_96_16_40.json public_benchmarks/shapenet_car_reduced_96_16_40
```

The output directory must not already exist. The upstream loader requires all nine `training_data/param*` directories: the materializer places the 16 validation cars in `training_data/param0`, keeps the training cars in `param1` through `param8`, and places the original test-fold cars separately in `test_data/param0`. `case_mapping.json` records original names. Do not use the test cases for model selection.

Ten distinct selected cars took 53.73 seconds to preprocess and 3.175 seconds for ten upstream optimizer steps on an RTX PRO 4000 Blackwell laptop GPU, with 3.615 GiB peak allocated GPU memory. The upstream eight-layer, width-256 model, full 32,186-node meshes, loss and optimizer were used. A 150-epoch run has an estimated 76 minutes of optimizer work and roughly 10 minutes of preprocessing, plus validation, checkpointing and startup. This extrapolates one short run and is not an end-to-end guarantee. The upstream script saves only after the final epoch and does not provide resume; a premature session termination loses the model.

### Kaggle notebook

Upload the contents of `public_benchmarks/shapenet_car_reduced_96_16_40` as a Kaggle Dataset. Attach it, enable Internet and a GPU, and update `DATA_ROOT` in the ShapeNet-Car cell of [gdl-transolver.ipynb](../../gdl-transolver.ipynb) to the mounted dataset root. `training_data` must be immediately inside that root. The cell runs the upstream model with the mesh-edge option (Transolver does not consume these edges). A CUDA-compatible GPU and enough host memory and writable disk for preprocessing are required.

The saved model and training log appear under `Transolver/Car-Design-ShapeNetCar/metrics/Transolver/0/150_0.5/`. The command does not evaluate the 40 held-out cars; the upstream drag evaluator needs additional geometry inputs and a separate evaluation setup. Kaggle GPU type, session limits, installation and preprocessing times differ from the local measurement.

For a saved upstream checkpoint and `log_150.json`, evaluate the held-out fields with:

```powershell
uv run python -m benchmarks.shapenet_car.evaluate_reference --data public_benchmarks/shapenet_car_reduced_96_16_40 --splits benchmarks/shapenet_car/reduced_splits_96_16_40.json --run "volume_results/shapenet_car_reduced/full_ transolver_150epoch/150_0.5" --output volume_results/shapenet_car_reduced/reference_evaluation
```

The output directory must not exist. The evaluator downloads the exact pinned upstream model source, checks its SHA-256, and uses an allowlist to load the checkpoint. Network access is required. It restores training normalization from the log and reports mean per-case relative L2 surface-pressure and exterior-velocity errors, with prediction arrays and surface-pressure images. This is not a drag evaluation or a reproduction of the paper's full-data result.

The pinned upstream script swaps the names assigned to returned pressure and velocity losses when reporting `train_loss` and `val_loss`. Consequently those log values are pressure MSE plus 0.5 velocity MSE, whereas the optimized training objective is velocity MSE plus 0.5 pressure MSE. Do not compare logged losses directly to the hybrid runner's validation objective.

## Hybrid slice GNN

The final notebook cell runs `gnn_hybrid_slice` on the same reduced split. Upload the updated repository code as the Kaggle code dataset, attach it and the reduced car dataset, and enable a GPU. Check `CODE_ROOT` and `DATA_ROOT` in the cell. Run this cell in a separate Kaggle session from the upstream run. It refuses to overwrite an existing output directory.

The model has three width-32 message-passing layers and 32-slice, four-head global attention. It uses full meshes and seven 3D edge features: direction, log distance, and displacement. Self-edges have zero direction and a clamped log distance. Loss and Adam/OneCycleLR settings follow the pinned upstream car script; training-only normalization uses the same all-node convention. The GNN is a different architecture, uses mesh connectivity, and has an explicit seed; the upstream notebook does not set one. Results are not a controlled architecture-only comparison or a reproduction of the full benchmark.

After 150 epochs the runner evaluates the **final** checkpoint on the 40 held-out test cases; `best_validation.pt` is retained separately and is not evaluated automatically. It saves the normalization, split identities, configuration, per-epoch history and checkpoints, plus per-case actual/predicted four-field arrays, shared-scale surface-pressure images and relative L2 surface-pressure and exterior-velocity scores. Exterior pressure has no CFD labels; drag and whole-domain pressure errors are not available. The upstream training notebook does not evaluate its test set, so a head-to-head comparison requires separate inference on the same 40 cars and the same metric implementation. Do not select checkpoints or tune using these test cases.

For a local real-data pipeline check, run:

```powershell
uv run python -m benchmarks.shapenet_car.run --data public_benchmarks/shapenet_car_reduced_96_16_40 --splits benchmarks/shapenet_car/reduced_splits_96_16_40.json --output volume_results/shapenet_pilot --epochs 1 --validation-interval 1 --pilot
```

The pilot uses two training, one validation and one test case and is not accuracy evidence. Checkpoints include optimizer and scheduler state but the runner does not implement resume. No Kaggle end-to-end runtime has been measured; full-mesh GNN message passing is substantially different from the upstream Transolver timing probe.