# Rules for This Log

1. Record the rationale for each experiment and material implementation choice, including assumptions, supporting evidence, alternatives, and uncertainty.
2. Record decisions being made, their scope, and what would justify revisiting them.
3. Record problems encountered, failed checks, regressions, and disappointing results; do not retain only successes.
4. Record the rationale and decisions used to address each problem, and whether the attempted remedy actually worked.
5. Explicitly record compromises, options avoided, and ideas deferred for future investigation.
6. Maintain a researcher's diary: include dates, reproducible commands, data splits, model and loss settings, artifact locations, numerical results, visual comparisons, and limitations.
7. Distinguish hypotheses from measurements, reconstruction from held-out validation, internal model-selection data from external test data, and single-fold experiments from full cross-validation.
8. Append new findings rather than silently rewriting earlier conclusions. Clearly identify retrospective summaries and corrections. Never claim an experiment was run unless its output was inspected.
9. Generate and link side-by-side pressure PNGs for model experiments. Use common pressure scales for competing predictions and report near-body as well as whole-domain errors.

## 2026-11-18 - Retrospective Baseline and Data Audit

This entry summarizes work completed before this diary was created.

- Scope: planar 2D volume-pressure surrogate; surface training remains separate.
- Data: 30 STAR-CCM+ exports, 279,177 fluid cells, 1,415,224 directed adjacency edges. Cell areas reconstructed from closed face loops agree with an independent face-triangle calculation to approximately 6e-8 relative error.
- Physical settings verified across all designs: density 1.18415 kg/m^3 and outlet gauge pressure 0 Pa. Coefficient target is `(pressure - reference_pressure) / (0.5 * density * inlet_speed**2)`.
- Initial volume GNN: hidden width 32, three message-passing layers, global mean pooling, batch size 1, AdamW, initial learning rate 1e-3, cosine decay, 40 epochs per comparison. No depth increase was tested.
- Original report was broadly reproducible, but pooled R-squared hid weak individual designs. PyTorch's lower-middle median was replaced with the conventional median. CV and final fitting now use a matching epoch count.
- The saved final original model is not a held-out test model. No improved production checkpoint has been selected or promoted.

## 2026-11-18 - Coefficients and Geometric Features

Hypotheses: coefficient targets reduce speed-dependent target scaling; wall distance and dimensionless edge geometry provide useful physical context without adding layers.

Matched five-fold, 40-epoch results, with explicit fold seeds and identical design membership:

| Variant | Mean design RMSE [Pa] | Mean near-body RMSE [Pa] | Pooled R-squared |
| --- | ---: | ---: | ---: |
| Raw pressure baseline | 6.62093 | 27.14712 | 0.96337 |
| Pressure coefficient | 5.71040 | 24.90254 | 0.97416 |
| Coefficient + wall distance + edge geometry | 5.62595 | 25.27573 | 0.97445 |

- Edge features: 2D direction, spacing divided by local cell length scale, and log neighbour-area ratio. Near-body metrics use a 0.05 m wall-distance threshold.
- Artifacts: `volume_results/{baseline,coefficient,geometry}/metrics.json` and `predictions/`. Reproduce the geometry study with `uv run python train_volume.py --variant geometry --epochs 40 --cv-only --results-dir volume_results/geometry`.
- Problem: held-out field plots were much less convincing than the pooled metrics. Coefficient scaling is useful numerically, but none of these results established a visually accurate ROM across designs.
- Correction: aggregate error reductions were initially described too positively before inspecting the field images. Subsequent comparisons must include spatial evidence.
- Images: `validation_results/volume/validation_volume_Design_1_{baseline,coefficient,geometry}.png`. Regenerate without training: `uv run python validate_volume.py --variant all --design Design_1 --no-show`.

## 2026-11-18 - Fitting Capacity and Loss Weighting

- Single-design area-loss fit: `uv run python diagnose_volume.py --stage overfit`. Design_1, 1,200 optimizer steps, unchanged GNN width/depth; RMSE 0.24152 Pa, near-body RMSE 1.94463 Pa, R-squared 0.998598. This is reconstruction, NOT unseen-design validation. Image: [single-design reconstruction](volume_results/diagnosis/overfit_area/Design_1.png).
- Problem encountered: the shared standardizer computes a sample standard deviation, which is undefined for one parameter row. Resolved locally in volume scaler fitting by assigning unit parameter scales for a single-design diagnosis; surface scaler behavior was not changed.
- Interpretation: this network can represent this individual field. This does not prove sufficient capacity for all designs jointly or good generalization.
- Fixed held-out fold 2: 24 training designs, six test designs; compared cached area-loss predictions with a new model using `0.5 * whole_domain_MSE + 0.5 * near_body_MSE`, each term normalized by its represented area. Same architecture, seed and 40 epochs. Near-body membership depends only on geometry, not target pressure.
- Command: `uv run python diagnose_volume.py --stage fold`. The area-loss baseline was reused, not retrained.
- Result: mean near-body RMSE 29.59732 -> 14.70449 Pa, but mean whole-domain RMSE 4.99687 -> 5.40228 Pa. This is a trade-off on ONE fold, not a full CV improvement claim.
- Artifacts: `volume_results/diagnosis/loss_comparison.json` and `fold_near_body/`. Image: [CFD / original loss / near-body loss](volume_results/diagnosis/fold_near_body/Design_1_loss_comparison.png). Refresh all six comparison images without training: `uv run python diagnose_volume.py --stage plots`.
- Defaults were not silently changed to the experimental loss. Production checkpoint promotion is deferred.

## 2026-11-18 - Reading the Siemens Method

- Verified documentation describes transformer-based operator learning with multi-head self-attention, rather than our local message-passing GNN.
- Tutorial volume settings: 1,000 epochs and 300 field-model spatial sample points. Default internal split is 80% training / 20% validation, not all 30 designs receiving gradient updates. The reference supports best-validation, best-training, or last-checkpoint selection.
- External test: 16 NEW Latin-hypercube designs in a narrower domain: P_D1_r [0.1, 0.2], P_D2_r [1.5, 2.0], inlet speed [8, 10] m/s. Training ranges are [0.05, 0.5], [0.75, 2.25], and [7.5, 12.5] m/s respectively.
- The PressureDrop response is predicted by a separate scalar response model. Its table values are not pressure-field RMSE.
- Consequence: our 40-epoch GNN studies are preliminary experiments, not faithful reproductions of the Siemens method. Exact contributions of architecture, training duration, sampling, and test domain have not been isolated.
- Sources: [training tutorial](https://docs.sw.siemens.com/documentation/external/PL20260405388167672/en-US/userManual/starccmp_userguide_html/STARCCMP/GUID-10849E7A-F3FA-46D8-B235-4BEC2D1A35D1.html), [validation tutorial](https://docs.sw.siemens.com/documentation/external/PL20260405388167672/en-US/userManual/starccmp_userguide_html/STARCCMP/GUID-6D7259FC-8DE6-44B4-888A-F79EB900F686.html), [model reference](https://docs.sw.siemens.com/documentation/external/PL20260405388167672/en-US/userManual/starccmp_userguide_html/STARCCMP/GUID-46DD49AA-2B46-4856-B4F5-8DB38AA2EF1E.html), [attention theory](https://docs.sw.siemens.com/documentation/external/PL20260405388167672/en-US/userManual/starccmp_userguide_html/STARCCMP/GUID-CE30031D-7370-4F29-9304-56D5A9D19120.html).

## 2026-11-18 - Optional Global Attention: Initial Decision

Hypothesis: add richer global communication while preserving local mesh connectivity, rather than replacing the existing GNN immediately.

- Decision: an OPTIONAL latent-token attention block after each local GNN layer; initial configuration 32 tokens, four heads, width 32, three local layers. Cells update tokens through cross-attention, tokens communicate through self-attention, and cells read global information back through cross-attention. Residual connections and pre-layer normalization stabilize the updates.
- Isolation requirement: process each design's attention separately using `batch_idx`; information must never mix between designs in a batch. Verify permutation equivariance, graph isolation, gradients, checkpoint round trips, and compatibility when attention is disabled.
- Preserve existing surface and volume defaults and old checkpoint layouts when attention is disabled. Save attention settings in new checkpoints. Do not overwrite or promote the original production checkpoint.

### Explicit Compromise: Full Cell-to-Cell Attention Deferred

For N cells, full self-attention involves N^2 pairwise scores per head. This becomes expensive for larger designs. We are deliberately compromising by using M learned latent tokens, with attention-score work approximately O(N*M + M^2) per block rather than O(N^2), where M is much smaller than N. Feature width, number of heads, projections and feed-forward layers add further computation.

Example: at N=50,000, full attention considers 2.5 billion pairs PER HEAD. With M=32, the two cell/token cross-attention directions plus token self-attention involve about 3.2 million + 1,024 pairs per head. Optimized attention kernels can reduce memory materialization, but exact dense attention still has quadratic pairwise computation.

The token bottleneck can discard spatial detail; it is NOT equivalent to unrestricted all-cell attention, and it is NOT an exact reproduction of PhysicsAI. Token count is also not the same setting as Siemens' field-model sample-point count.

Future options explicitly retained:

- Full cell-to-cell self-attention on a small/subsampled design, then a memory/runtime feasibility study on larger designs.
- More latent tokens (for example 64 or 128), spatially anchored tokens, and sensitivity to initialization.
- Local neighbour attention (GATv2), which alone does not remove limited graph-hop communication.
- A separate transformer/operator model, rather than a hybrid GNN.
- Longer training with internal-validation checkpoint selection and early stopping. Do not use the outer six test designs to select epochs or hyperparameters.
- More width/depth only after separating architecture and training-budget effects.

Status: implementation and verification in progress. No accuracy improvement from attention has been established yet.

## 2026-11-18 - Attention Implementation and Preflight Checks

- Implemented `LatentGlobalAttention` using PyTorch `nn.MultiheadAttention`: cells-to-token cross-attention, token self-attention, token feed-forward update, and token-to-cell cross-attention plus a cell feed-forward update. The API does not request returned attention matrices (`need_weights=False`); internal memory behavior depends on the selected backend. This does not change the stated pairwise complexity.
- Optional attention follows each local message-passing layer. `attention_tokens=0` preserves the old path and state-dictionary keys. Graphs are processed separately, including when batch membership is non-contiguous.
- Tests verified graph isolation, node permutation equivariance, gradients from one cell to a distant cell, gradients through all attention parameters, disabled-mode compatibility, and attention checkpoint round trips.
- Preflight: all 28 regression tests passed; Ruff and type checking passed on the changed attention files. Both the original volume checkpoint and the cached near-body GNN checkpoint still load with attention disabled.
- Parameter counts: no-attention geometry GNN 25,089; 32-token/four-head hybrid 92,097. This comparison changes both global communication and parameter count. It cannot isolate attention from increased capacity or computation.
- Initial experiment protocol: same fold 2, same 24 training designs and six held-out designs, same seed, same near-body-balanced loss and same 40 epochs. Reuse the cached no-attention near-body model; train only the hybrid. This is exploratory testing on an already inspected fold, not an untouched external benchmark or evidence of convergence.
- Command: `uv run python diagnose_volume.py --stage attention --attention-tokens 32 --attention-heads 4`.
- Planned outputs: `volume_results/diagnosis/fold_attention/model.pt`, training history and per-design metrics in `metrics.json`, pressure arrays, individual CFD/prediction PNGs, and six three-panel `*_attention_comparison.png` files.
- Plot-only command: `uv run python diagnose_volume.py --stage attention-plots`.
- General volume training also accepts `--attention-tokens` and `--attention-heads`; defaults remain attention disabled. Default attention checkpoint/results paths are separate from the no-attention paths.
- Accuracy result: pending the bounded experiment below; no checkpoint has been promoted.

## 2026-11-18 - First Latent-Attention Experiment: Mixed Result

Completed the planned single-fold command once. The existing no-attention reference was read from disk, not retrained. Architecture: three local layers, width 32, 32 latent tokens and four heads after each local layer. The loss, seed, training membership, 40 epochs, physical target transform, and held-out designs matched the cached near-body reference.

| Metric across six held-out designs | No attention | Latent attention |
| --- | ---: | ---: |
| Mean design RMSE [Pa] | 5.40228 | 6.09530 |
| Mean near-body RMSE [Pa] | 14.70449 | 13.88705 |
| Worst design RMSE [Pa] | 7.76731 | 8.29175 |
| Pooled R-squared | 0.86815 | 0.83418 |
| Mean design R-squared | 0.69648 | 0.64307 |

Attention-model per-design measurements:

| Design | Whole-domain RMSE [Pa] | Near-body RMSE [Pa] |
| --- | ---: | ---: |
| Design_1 | 4.67695 | 16.12968 |
| Design_3 | 7.08289 | 13.73343 |
| Design_4 | 5.15327 | 5.43038 |
| Design_13 | 5.05603 | 10.87659 |
| Design_14 | 6.31088 | 22.31008 |
| Design_26 | 8.29175 | 14.84212 |

- Observation: mean near-body error improved by approximately 5.6%, but mean whole-domain error worsened by approximately 12.8%. Whole-domain error improved on only two of six designs. This does not establish an overall improvement.
- Visual inspection: [Design_1 comparison](volume_results/diagnosis/fold_attention/Design_1_attention_comparison.png) shows somewhat improved localization, but a broader pressure pattern than CFD. [Design_26 comparison](volume_results/diagnosis/fold_attention/Design_26_attention_comparison.png) shows an example of a worsened result; do not present only the improved case.
- All six comparison PNGs were generated with shared per-design pressure scales and whole-domain/near-body metric annotations. The numerical report is [attention comparison](volume_results/diagnosis/attention_comparison.json).
- Artifact verification: reloaded `volume_results/diagnosis/fold_attention/model.pt` and reproduced all six saved pressure arrays within rtol=1e-5 and atol=1e-5. Legacy volume checkpoints were also checked before the run.
- Problems and safeguards: batched attention could leak between designs, so the module explicitly isolates graphs and regression tests perturb one design while checking another remains unchanged. Plot titles initially assumed identical architectures; generalized them before the experiment to state that architectures differ. Checkpoint metadata initially lacked attention settings; added and tested token/head round trips.
- Decision: keep attention optional and OFF by default. Preserve both checkpoints and all negative evidence. Do not promote this exploratory hybrid or claim it matches PhysicsAI.
- Limitations: one seed, one already-inspected fold, larger parameter count, no matched-FLOP control, no internal-validation-based epoch selection, and a short fixed 40-epoch schedule. The six held-out designs were not used for gradient updates, but repeated research comparisons on them limit their status as a pristine test set.
- Next candidates, not executed: internal-validation convergence study; parameter-matched wider no-attention GNN; alternate token counts/spatial tokens; full-attention feasibility check on small/subsampled meshes. Full N-by-N attention remains deliberately deferred, not rejected on accuracy grounds.
- Verification at completion: 28 tests pass; Ruff and type checks passed on changed Python files. Baseline defaults, GNN local depth, original checkpoints, and unrelated worktree changes were preserved.

## 2026-11-19 - Nested Validation and Architecture Comparison Plan

- Outer split is the cached fold containing Design_1: 24 outer-training designs and six outer-test designs. These six were already inspected in prior experiments and are not pristine; keep that limitation explicit.
- Inner split: deterministic 80/20 split of the outer-training set (19 inner-training, five inner-validation for this fold). Fit every scaler on the 19 inner-training designs only. Never use outer-test pressure to select epochs or checkpoints.
- Checkpoint selection: near-body-balanced normalized validation loss, evaluated every 10 epochs; maximum 1,000 epochs; minimum 100; stop after eight validation checks without improvement. Save the best validation checkpoint, including its inner-training scalers, and plot training/validation curves.
- Architecture candidates: original three-layer width-32 GNN; width-62 GNN with 91,389 parameters (parameter-count matched to 32-token latent attention, 92,097 parameters); three-layer width-32 latent-attention GNN (32 tokens/four heads); edge-aware four-head neighbor-attention GNN (width32, 32,109 parameters).
- All candidates use the same fixed outer/inner design membership, seed, geometry features, coefficient targets, near-body loss, optimizer, max epoch budget and checkpoint-selection rule. Parameter matching only controls approximate parameter count, not FLOPs or runtime.
- After selection on inner validation, evaluate each selected checkpoint on the six outer designs and create per-design three-panel CFD / cached GNN reference / new architecture pressure PNGs, as well as validation-loss curves and JSON/CSV metrics. Do not select a winner from these already inspected outer results.
- Reproducible command: `uv run python compare_volume_architectures.py --design Design_1 --max-epochs 1000 --minimum-epochs 100 --validation-interval 10 --patience 8`.
- Smoke-tested all four architectures for one epoch on real data; artifacts were written under `volume_results/architecture_smoke/` and are not accuracy evidence.
- Full run status: pending.

## 2026-11-19 - Inner-Validation Architecture Study Completed

Command: `uv run python compare_volume_architectures.py --design Design_1 --max-epochs 1000 --minimum-epochs 100 --validation-interval 10 --patience 8`.

- Outer fold 2 was kept fixed: 24 outer-training designs, six outer evaluation designs. A seeded 80/20 split of the 24 produced 19 inner-training and five inner-validation designs. Scalers were fit only on the 19 inner-training designs. Inner-validation loss selected each saved checkpoint; the six outer designs were evaluated only after selection in code.
- All four models used the same design split, 0.5 whole-domain / 0.5 near-body loss, 1,000 maximum epochs, AdamW settings, validation cadence, and early stopping. Each selected its own best validation checkpoint. GNN depth remained three layers.

| Architecture | Parameters | Best epoch | Mean outer RMSE [Pa] | Mean near-body RMSE [Pa] | Mean design R-squared |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original GNN, width 32 | 25,089 | 50 | 4.96591 | 19.92754 | 0.67773 |
| Parameter-matched GNN, width 62 | 91,389 | 170 | 3.24815 | 14.04803 | 0.78345 |
| Latent attention, width 32 / 32 tokens / 4 heads | 92,097 | 160 | 2.98384 | 11.96292 | 0.86628 |
| Neighbour attention, width 32 / 4 heads | 32,109 | 180 | 3.42721 | 11.95167 | 0.81460 |

- On this fold the parameter-matched GNN substantially improved over width32, so capacity matters. Latent attention improved mean whole-domain and near-body metrics over both GNNs, but its result is not parameter/FLOP matched to the small baseline; the width62 GNN is its parameter-count control. Neighbour attention had the best mean near-body RMSE by a negligible 0.010 Pa versus latent attention, while its whole-domain RMSE was worse by 0.443 Pa.
- Comparison images for all six outer designs are in `volume_results/architecture_validation/{gnn32,gnn_parameter_matched,latent_attention32,neighbor_attention32}/Design_*_comparison.png`. They show CFD, the cached near-body reference, and the selected architecture with a common per-design pressure scale and whole/near-body errors.
- Validation curves: each architecture directory has `validation_curve.png` and `validation_history.json`. The overall metrics and per-design reports are in `architecture_summary.json`, `architecture_summary.csv`, and each model's `metrics.json`.
- All four best-validation checkpoints were reloaded successfully; saved epochs and scalers match the run reports. Checkpoints are `best_validation.pt` in each architecture directory.
- Verification: 33 tests passed; Ruff, ty, and `git diff --check` passed. A one-epoch end-to-end smoke test also completed for all variants before the full run.
- Problems faced: the first smoke run exposed two artifact-layout assumptions (cached fold membership stored in `config` rather than `folds`, and direct NPZs rather than a `predictions/` subfolder); the runner was updated to accept both layouts and the smoke run then completed. A gradient test also caught unused neighbor-attention value parameters; the implementation now uses learned edge messages as attention values.
- Important limitation: this was one fold and one seed, and these six outer designs were previously inspected in earlier experiments. Although they were not used for epoch selection in this run, using them repeatedly to compare architectures means this is exploratory evidence, NOT an unbiased final test. Do not select/promote a production model based solely on these results. Confirm with new DOE designs or repeated nested folds/seeds.
- Decision: retain all checkpoints and images; leave defaults and production artifacts unchanged. Tentatively prioritize latent attention and parameter-matched GNN for repeated validation; keep neighbour attention as a near-body-focused alternative. Next verification should be repeated outer folds/seeds or the tutorial-style new DOE set, with visual comparison every run.

## 2026-11-19 - Explicit Model Ranking and Parameter-Matched Neighbour Attention

- Added `rank_volume_models.py` and `rank_architectures()` to rank existing summaries under three explicit profiles: balanced (40% mean whole-domain RMSE, 40% mean near-body RMSE, 20% worst-design RMSE), whole-domain priority (60/20/20), and near-body only. The score is a weighted sum of relative regret versus the best candidate per metric; the report also includes each metric's rank. The weights express priorities, not a statistically calibrated utility or confidence measure.
- Existing four-model rankings: latent attention wins balanced and whole-domain profiles; neighbour_attention32 wins near-body-only by 0.011 Pa versus latent attention (effectively tied). This answers the user's visual preference: neighbour attention is top-ranked when near-body field structure is the deciding criterion, not when equally balancing whole-domain accuracy and worst-design robustness.
- Added `neighbor_attention_parameter_matched`: hidden width 56, four heads, 95,157 parameters, versus 92,097 for latent_attention32 and 91,389 for the parameter-matched plain GNN. The closest head-divisible hidden width is 3.32% above the latent-attention parameter count.
- Same nested split, seed 2026, 19 inner-train / 5 inner-validation / 6 outer evaluation designs, near-body-balanced loss and early-stopping budget. Best epoch 120; inner validation loss 0.0201964.

| Model | Parameters | Mean RMSE [Pa] | Mean near-body RMSE [Pa] | Worst-design RMSE [Pa] |
| --- | ---: | ---: | ---: | ---: |
| neighbor_attention32 | 32,109 | 3.42721 | 11.95167 | 5.16791 |
| neighbor_attention_parameter_matched | 95,157 | 3.47246 | 12.87265 | 5.63895 |

- Parameter matching did not improve this neighbour-attention variant on this fold. Width-32 neighbour attention remains preferable to its width-56 parameter-matched version in these measurements; do not conclude that parameter count itself caused the difference.
- Added five-model ranking output: `volume_results/architecture_validation/rankings_with_matched_neighbor.json`. Matched-neighbour checkpoint, metrics, and six pressure images are in `volume_results/architecture_neighbor_matched/neighbor_attention_parameter_matched/`.
- Visual spot-check: [Design_26 parameter-matched neighbour attention comparison](volume_results/architecture_neighbor_matched/neighbor_attention_parameter_matched/Design_26_comparison.png). It follows the broad pressure field but has higher reported errors than width32 neighbour attention on that case.
- Test/quality: 35 tests pass; Ruff and ty pass for ranking files. A test initially set an unrealistically strict 3% width-match threshold; nearest legal width differs by 3.32%, so the test was adjusted to 5% and the rationale documented.
- Interpretation caveat remains: all six outer designs were previously inspected; this is exploratory ranking, not unbiased model selection. Keep the chosen architecture provisional until new DOE designs or repeated nested splits/seeds are run.

## 2026-11-19 - Latent Attention Tanh Experiment: Protocol

- Hypothesis: replacing the latent model's ReLU and GELU activations with tanh could improve pressure-field accuracy; this is a hypothesis, not a measured improvement. Attention softmax is unchanged. The original model and its default activation behavior remain unchanged.
- Match the prior latent-attention experiment: 92,097 parameters, 32 tokens, four heads, three local layers, seed 2026, 19 inner-training / 5 inner-validation / 6 previously inspected outer designs, 0.5 near-body loss mix, CPU, and inner-validation checkpoint selection. Maximum 1,000 epochs, minimum 100, check every 10, patience eight checks. The outer designs are not used to select checkpoints.
- Command: `uv run python compare_volume_architectures.py --design Design_1 --architectures latent_attention32_tanh --max-epochs 1000 --minimum-epochs 100 --validation-interval 10 --patience 8 --results-dir volume_results/architecture_tanh --comparison-dir volume_results/architecture_validation/latent_attention32 --comparison-label "Original latent attention (ReLU/GELU)"`.
- Outputs belong under `volume_results/architecture_tanh/latent_attention32_tanh/`; comparison images use CFD, the saved original latent model and tanh on a common pressure scale. New checkpoints record the activation; old checkpoints still load with their original activation.
- One-epoch real-data smoke test completed under `volume_results/architecture_tanh_smoke/`; it checks the pipeline only, not accuracy. The full run is in progress. A new focused test first failed because the variant did not exist, then passed after implementation; 36 regression tests, Ruff and ty passed.
- Deferred: test tanh on the two neighbour-attention variants one at a time; repeat nested folds/seeds or evaluate new DOE designs before selecting a production model. Further options from the earlier plan include token-count and spatial-token studies, a small-mesh full-attention feasibility check, and matched-compute controls.

## 2026-11-19 - Tanh Run Interruption and Restart

- The initial full run stopped for an unknown reason before writing final metrics, validation history, or comparison images. Its saved best inner-validation checkpoint was at epoch 70 with loss 0.0232954; the last completed epoch is unknown. This is not a completed comparison.
- Preserved the partial checkpoint under `volume_results/architecture_tanh/latent_attention32_tanh/`. It does not contain optimizer or scheduler state, so an exact continuation is not available.
- Restarted the same experiment from epoch 1 in the visible VS Code task `CFD: latent attention tanh experiment`, using `volume_results/architecture_tanh_restart/` for new artifacts. The task runs the protocol above with this changed results directory and unbuffered Python output. No neighbour variant was started.

## 2026-11-19 - Latent Attention Tanh Results

- The restarted run completed successfully. Training stopped at epoch 340 after eight validation checks without improvement; the selected checkpoint was epoch 260. The original ReLU/GELU model selected epoch 160. Both use the same split, seed, architecture dimensions, 92,097 parameters, loss and checkpoint selection protocol described above.

| Metric across six outer designs | Original ReLU/GELU | Tanh |
| --- | ---: | ---: |
| Mean design RMSE [Pa] | 2.98384 | 2.92379 |
| Mean near-body RMSE [Pa] | 11.96292 | 9.34826 |
| Worst design RMSE [Pa] | 4.05897 | 3.96726 |
| Pooled R-squared | 0.95836 | 0.96003 |
| Best inner-validation loss | 0.013031 | 0.013986 |

- Mean near-body RMSE improved by 21.9%, with improvements on all six designs; mean whole-domain RMSE improved by 2.0%, despite worsening on four of six. The original achieved a lower inner-validation loss; differences in selected epochs and optimization are part of this comparison, not evidence that tanh universally generalizes better.

| Design | Original / tanh whole RMSE [Pa] | Original / tanh near-body RMSE [Pa] |
| --- | ---: | ---: |
| Design_1 | 3.812 / 3.967 | 21.343 / 17.992 |
| Design_3 | 3.209 / 3.496 | 7.551 / 6.366 |
| Design_4 | 3.058 / 1.595 | 8.160 / 2.371 |
| Design_13 | 1.608 / 2.223 | 5.935 / 5.091 |
| Design_14 | 4.059 / 3.669 | 24.524 / 21.164 |
| Design_26 | 2.158 / 2.593 | 4.265 / 3.106 |

- Inspected common-scale plots: [Design_4](volume_results/architecture_tanh_restart/latent_attention32_tanh/Design_4_comparison.png) shows an improved pressure field, while [Design_13](volume_results/architecture_tanh_restart/latent_attention32_tanh/Design_13_comparison.png) and [Design_1](volume_results/architecture_tanh_restart/latent_attention32_tanh/Design_1_comparison.png) show the whole-domain trade-off. Other plots: [Design_3](volume_results/architecture_tanh_restart/latent_attention32_tanh/Design_3_comparison.png), [Design_14](volume_results/architecture_tanh_restart/latent_attention32_tanh/Design_14_comparison.png), [Design_26](volume_results/architecture_tanh_restart/latent_attention32_tanh/Design_26_comparison.png). Each contains CFD, original latent attention and tanh.
- Artifacts: [metrics](volume_results/architecture_tanh_restart/latent_attention32_tanh/metrics.json), [validation curve](volume_results/architecture_tanh_restart/latent_attention32_tanh/validation_curve.png), [checkpoint](volume_results/architecture_tanh_restart/latent_attention32_tanh/best_validation.pt), and saved prediction arrays in the same directory.
- Limitations: one seed and one previously inspected outer fold, not independent external testing. No checkpoint is promoted. The two neighbour-tanh experiments and repeated nested folds or new DOE designs remain deferred.

## 2026-11-20 - Width-32 Neighbour Attention Tanh: In Progress

- First of the two planned neighbour tanh experiments. Variant: width 32, four edge-attention heads, three local layers; tanh replaces ReLU in the message/update MLPs and LeakyReLU in the edge-score MLP. This activation scope is explicit because retaining LeakyReLU would make the neighbour variant only partially tanh.
- Match the prior fold-2 protocol: seed 2026, 19 inner-training / 5 inner-validation / 6 previously inspected outer designs, near-body-balanced loss, 1,000 epoch maximum, minimum 100, validate every 10 epochs, patience eight. The saved neighbour baseline has 32,109 parameters and selected epoch 180. Tanh has the same parameter count.
- Command: `uv run python -u compare_volume_architectures.py --design Design_1 --architectures neighbor_attention32_tanh --max-epochs 1000 --minimum-epochs 100 --validation-interval 10 --patience 8 --results-dir volume_results/architecture_neighbor_tanh --comparison-dir volume_results/architecture_validation/neighbor_attention32 --comparison-label "Original neighbor attention (ReLU/LeakyReLU)"`.
- Outputs: `volume_results/architecture_neighbor_tanh/neighbor_attention32_tanh/`, including selected checkpoint, metrics, history, validation curve, saved predictions and six CFD/baseline/tanh comparison images on shared pressure scales.
- Focused tanh architecture and checkpoint round-trip test passed; Ruff and ty passed before launch. Training is running directly in the terminal; dataset loading completed, and the first epoch status has not printed yet. Results pending. Do not start the parameter-matched neighbour tanh run until this experiment is reviewed.
- Progress check on 2026-11-20: process remained active at epoch 120/1,000. Best validation loss was 0.0163645 at epoch 80; checkpoint exists. Outer-test metrics and comparison images are not available yet.
- Completed at epoch 160; best inner-validation checkpoint was epoch 80 (loss 0.0163645). Original neighbour attention selected epoch 180 (loss 0.0159981). Same fold, seed, split, loss and 32,109 parameters; no training was performed on the six outer designs during selection.

| Metric across six outer designs | Original ReLU/LeakyReLU | Tanh |
| --- | ---: | ---: |
| Mean design RMSE [Pa] | 3.42721 | 3.39515 |
| Mean near-body RMSE [Pa] | 11.95167 | 12.76458 |
| Worst design RMSE [Pa] | 5.16791 | 4.34711 |
| Pooled R-squared | 0.94505 | 0.94844 |

- Mean whole-domain RMSE improved 0.94%; mean near-body RMSE worsened 6.8%. Whole-domain RMSE improved on four of six designs; near-body RMSE improved on three. Design_14 near-body RMSE regressed 24.59 -> 32.90 Pa, visible in the [Design_14 comparison](volume_results/architecture_neighbor_tanh/neighbor_attention32_tanh/Design_14_comparison.png). The [Design_4 comparison](volume_results/architecture_neighbor_tanh/neighbor_attention32_tanh/Design_4_comparison.png) shows improved near-body error but worse whole-domain error; [Design_1](volume_results/architecture_neighbor_tanh/neighbor_attention32_tanh/Design_1_comparison.png) improves both metrics slightly. Other shared-scale images: [Design_3](volume_results/architecture_neighbor_tanh/neighbor_attention32_tanh/Design_3_comparison.png), [Design_13](volume_results/architecture_neighbor_tanh/neighbor_attention32_tanh/Design_13_comparison.png), [Design_26](volume_results/architecture_neighbor_tanh/neighbor_attention32_tanh/Design_26_comparison.png).
- Summary and per-design metrics: `volume_results/architecture_neighbor_tanh/neighbor_attention32_tanh/metrics.json`. Interpretation is exploratory: one seed and a previously inspected outer fold. No model was promoted.
- Next, run the parameter-matched width-56 neighbour tanh variant alone, compared with `volume_results/architecture_neighbor_matched/neighbor_attention_parameter_matched/`. Keep it sequential; do not combine this result with the width-32 run as though it isolated activation from width/capacity.

## 2026-11-20 - Parameter-Matched Neighbour Attention Tanh: In Progress

- Started after the width-32 neighbour-tanh run completed and was reviewed. This is the width-56, four-head edge-attention model (95,157 parameters), compared directly with its saved ReLU/LeakyReLU model. It also uses tanh for the edge-score nonlinearity.
- Same fold-2 split, seed 2026, 19/5 inner split, near-body-balanced loss, validation-selected checkpoint and 1,000/100/10/8 epoch settings. This model differs from width-32 in hidden width and parameter count; compare each only against its own activation baseline.
- Command: `uv run python -u compare_volume_architectures.py --design Design_1 --architectures neighbor_attention_parameter_matched_tanh --max-epochs 1000 --minimum-epochs 100 --validation-interval 10 --patience 8 --results-dir volume_results/architecture_neighbor_matched_tanh --comparison-dir volume_results/architecture_neighbor_matched/neighbor_attention_parameter_matched --comparison-label "Original parameter-matched neighbor attention (ReLU/LeakyReLU)"`.
- Dataset loading completed; training is active. Results pending. Do not start another experiment concurrently.
- Completed at epoch 190; best inner-validation checkpoint was epoch 110 (loss 0.0177920). The original width-56 model selected epoch 120 (loss 0.0201964). Both have 95,157 parameters and used the same fold-2 split, seed, loss and checkpoint selection protocol.

| Metric across six outer designs | Original ReLU/LeakyReLU | Tanh |
| --- | ---: | ---: |
| Mean design RMSE [Pa] | 3.47246 | 4.79624 |
| Mean near-body RMSE [Pa] | 12.87265 | 11.22219 |
| Worst design RMSE [Pa] | 5.63895 | 6.62190 |
| Pooled R-squared | 0.93914 | 0.89169 |

- Mean whole-domain RMSE worsened 38.1%, and tanh was worse on all six outer designs. Worst-design RMSE increased 17.4%. Mean near-body RMSE improved 12.8%, but only Design_1 and Design_14 improved individually; the other four worsened. This is a strong whole-domain regression in exchange for an aggregate near-body gain, not an overall improvement.

| Design | Original / tanh whole RMSE [Pa] | Original / tanh near-body RMSE [Pa] |
| --- | ---: | ---: |
| Design_1 | 5.639 / 6.271 | 27.929 / 17.726 |
| Design_3 | 2.928 / 4.326 | 8.968 / 9.715 |
| Design_4 | 2.173 / 2.641 | 3.404 / 6.159 |
| Design_13 | 1.844 / 3.325 | 5.945 / 6.457 |
| Design_14 | 5.142 / 6.622 | 26.718 / 22.378 |
| Design_26 | 3.109 / 5.593 | 4.272 / 4.898 |

- Inspected common-pressure-scale comparisons: [Design_1](volume_results/architecture_neighbor_matched_tanh/neighbor_attention_parameter_matched_tanh/Design_1_comparison.png) and [Design_14](volume_results/architecture_neighbor_matched_tanh/neighbor_attention_parameter_matched_tanh/Design_14_comparison.png) show the near-body improvements; [Design_13](volume_results/architecture_neighbor_matched_tanh/neighbor_attention_parameter_matched_tanh/Design_13_comparison.png) shows a representative whole-domain regression. Other images: [Design_3](volume_results/architecture_neighbor_matched_tanh/neighbor_attention_parameter_matched_tanh/Design_3_comparison.png), [Design_4](volume_results/architecture_neighbor_matched_tanh/neighbor_attention_parameter_matched_tanh/Design_4_comparison.png), [Design_26](volume_results/architecture_neighbor_matched_tanh/neighbor_attention_parameter_matched_tanh/Design_26_comparison.png). Metrics and checkpoint: `volume_results/architecture_neighbor_matched_tanh/neighbor_attention_parameter_matched_tanh/`.
- Limitation: this is one seed and a previously inspected outer fold. Do not promote the tanh variant; confirm only on new DOE designs or repeated nested splits if further evidence is warranted.

## 2026-11-20 - IEEE Write-Up and Retrospective Analysis (No Training)

- Wrote `paper/main.tex` (IEEEtran) covering all work to date. No LaTeX toolchain is installed locally, so it has not been compiled. Figures are read from the repository via `\graphicspath{{../}}`. The parameter-matched neighbour tanh run is reported as in progress, without numbers.
- New measurement (pooled over all 30 designs): cells with wall distance <= 0.05 m make up 67.6% of cells but only 1.73% of fluid area. This quantifies why the area loss neglected the near-body region.
- New finding: the 19 inner-training designs span P_D1_r in [0.114, 0.5], so none has P_D1_r < 0.1. All five designs with P_D1_r of about 0.05 fall outside inner training: Design_1 and Design_14 are outer test designs, and Design_9, Design_15 and Design_21 are inner-validation designs. The two outer designs that are worst for every model are therefore parametric extrapolations, and checkpoint selection mostly measures extrapolation. In the 40-epoch fold-2 runs, Design_9, Design_15 and Design_21 were in training. Revisit with stratified or space-filling inner splits.
- Learning-rate confound: in the nested runs, cosine T_max is 1,000 but every run stopped by epoch 340. The learning rate at the selected epochs stayed between about 7.4e-4 and 9.9e-4, so checkpoints were not annealed. Inner-validation losses also fluctuate by about 3x between checks.
- Seven-model regret ranking was computed in memory with `rank_architectures()`, with no file written. `latent_attention32_tanh` has zero regret in all three profiles because it is best on mean, near-body and worst-design RMSE. This remains exploratory evidence from one seed on a previously inspected fold.
- Observation: the current `eda.py` reads `fields[:, 2]` (density), which was removed from the surface loader. The existing EDA figures predate that change, so the script must be updated before they can be regenerated.

## 2026-11-22 - Mesh Scaling and Near-Wall Weighting Discussion

- Decision: leave the current implementation and loss unchanged for now. This entry records the discussion; no Reynolds-based loss or new experiment was implemented for it.
- The opt-in `invariant` volume variant derives the streamwise reference length `L_ref` from the body `WALL` boundary, not the fluid-domain bounding box. Coordinates are centered on the body and scaled isotropically by `L_ref`; volume nodes also use `d_wall / L_ref` and its logarithm. Surface inputs use scaled positions and normals. Physical cell size is omitted from node features, while edges include direction, spacing relative to mean local cell size, and a log neighbor-measure ratio. Body aspect ratios and `log(speed * L_ref)` are design inputs; the latter is a Reynolds proxy only when viscosity is fixed, not a dimensionless Reynolds number.
- The observed body lengths range from approximately 0.052 to 0.659 m, with a median of 0.329 m across all 30 designs. These data are planar 2D; the feature calculations support synthetic 3D inputs, but a real 3D volume loader and real remeshing study remain deferred. Dimensionless inputs alone do not guarantee mesh-refinement invariance, aerodynamic similarity, or accurate boundary layers.
- The invariant near-body loss uses the manually chosen fraction `0.15`: a cell is in the band when `d_wall / L_ref <= 0.15`. The body length and resulting physical band are computed per design, but the fraction is not inferred automatically. It was chosen because `0.05 / 0.329` is approximately 0.15, preserving the old 0.05 m band at the dataset median only. This is a near-body weighting heuristic, not a measured boundary-layer thickness. The median was computed using all designs, including held-out designs; avoid deriving future tuning choices from the full dataset.
- The architecture experiment's existing loss mixes area-normalized whole-domain and near-body squared errors with equal weight; if a graph has no cells in the band, its near-body term falls back to the whole-domain weights. The legacy `near_wall_rmse_pa` uses 0.05 m, while `near_wall_rel_rmse_pa` uses 0.15 body lengths when a reference length is available. These are different reporting regions and must be labeled accordingly. The chosen band affects loss and metrics, not mesh adjacency.
- Considered making the relative threshold configurable and selecting it using training and inner-validation designs only. Keep a fixed reporting definition across compared models; deriving a new fraction from `0.05 / median(L_ref)` merely recreates the old physical-distance heuristic and does not determine the boundary layer. No CLI option or threshold-selection procedure was added in this discussion.
- Considered a smooth near-wall term using `Re_L = speed * L_ref / nu`, `delta = 0.37 * L_ref * Re_L**(-0.2)`, and `exp(-d_wall / delta)`. The turbulent flat-plate correlation is a candidate length-scale heuristic, not an exact thickness for these geometries or separated flows; its applicability and kinematic viscosity require verification. A smooth profile alone does not guarantee more stable training or discretization invariance.
- If tested later, normalize the area-weighted exponential term separately per graph and mix it with the whole-domain term. Simply using `1 + exp(-d_wall / delta)` changes the effective near-body contribution and is not equivalent to the current 50/50 loss. Retain the existing 0.05 m and 0.15 `L_ref` metrics and optionally add a clearly identified `d_wall <= delta` metric, with viscosity and correlation recorded. Compare using identical splits and common-scale CFD/reference/new prediction images; model selection must use inner validation, not the repeatedly inspected outer designs.
- The invariant latent-attention comparison was previously launched using the same outer fold and inner split, but its final output was not inspected for this entry. Its loss region differs from the original geometry-input run, so any eventual comparison confounds feature and loss changes. No improvement or final run status is asserted here.

## 2026-11-22 - Fresh Invariant Latent-Attention Comparison

- On request, checked for background CFD training before restarting; none remained active. Existing editor formatter services and prior results were left untouched. Restarted the ReLU/GELU `latent_attention32` with invariant features and seed 2026 in a separate directory, using the same 19 inner-training, five inner-validation and six previously inspected outer designs as the earlier study.
- Command: `uv run python -u compare_volume_architectures.py --design Design_1 --variant invariant --architectures latent_attention32 --max-epochs 1000 --minimum-epochs 100 --validation-interval 10 --patience 8 --results-dir volume_results/invariant_restart_20261122 --comparison-dir volume_results/architecture_validation/latent_attention32 --comparison-label "Non-invariant ReLU/GELU"`.
- The run stopped after epoch 120; inner validation selected epoch 40 at loss 0.0370654. The resulting metrics reproduce the prior invariant run exactly with the same seed and configuration; this is a reproducibility rerun, not an independent seed.

| Held-out model | Mean design RMSE [Pa] | Mean near-wall RMSE, 0.05 m [Pa] | Mean near-wall RMSE, 0.15 L_ref [Pa] | Worst design RMSE [Pa] |
| --- | ---: | ---: | ---: | ---: |
| Non-invariant latent attention, tanh | 2.924 | 9.348 | 13.100 | 3.967 |
| Non-invariant latent attention, ReLU/GELU | 2.984 | 11.963 | 18.360 | 4.059 |
| Transolver | 2.493 | 10.618 | 17.322 | 3.640 |
| Invariant latent attention, ReLU/GELU | 5.349 | 14.811 | 15.933 | 10.014 |

- Relative-band metrics were recomputed for all saved predictions using the same held-out geometries. On this fold, the invariant model is worse in whole-domain and 0.05 m mean near-wall RMSE than all three references, but its mean 0.15 L_ref near-wall RMSE is better than the non-invariant ReLU/GELU and Transolver results. Different regions can lead to different rankings.
- Common-scale held-out comparisons show CFD, non-invariant tanh, non-invariant ReLU/GELU, Transolver and invariant ReLU/GELU, including near-body zooms and absolute-error maps: [Design_1](volume_results/invariant_restart_20261122/latent_attention32/Design_1_benchmark_comparison.png) and [Design_14](volume_results/invariant_restart_20261122/latent_attention32/Design_14_benchmark_comparison.png). The invariant prediction underestimates the near-body pressure pattern in both; Design_14 also shows a broad far-field discrepancy. All six images and [summary metrics](volume_results/invariant_restart_20261122/latent_attention32/benchmark_summary.csv) are saved with the selected checkpoint.
- This comparison changes both features and the near-wall loss band: invariant training uses 0.15 L_ref, while the non-invariant runs use 0.05 m. It does not isolate feature scaling, establish remeshing invariance, or establish which change caused the regression. The tanh reference also changes activation, and Transolver changes architecture. One seed and a repeatedly inspected outer fold remain exploratory; no model is promoted.

## 2026-11-22 - Diagnosis of the Invariant Regression (No Training)

- Measured: inner-training bodies span L_ref 0.149-0.659 m. The five P_D1_r = 0.05 designs have L_ref 0.052-0.082 m, so their body-scaled inputs reach x/L 30-48 versus at most 16.7 in training. Cells outside the training range of at least one node feature cover 70-88% of fluid area for these designs (Design_1 72.9%, Design_14 86.5%; inner validation Design_9 88.1%, Design_15 70.3%, Design_21 83.7%). All other validation and test designs have 0% out of range.
- Physical coordinates never extrapolate because every design shares the fixed 3 m x 1 m channel. Body scaling converts the existing parametric extrapolation (no small bodies in inner training) into a large input-space extrapolation. Design_1 and Design_14 are also the two worst invariant designs.
- Physics: the problem is not scale-invariant in L_ref alone. The channel, inlet, side walls and 0 Pa outlet are fixed in metres, not scaled with the body, so blockage and distance to outlet/walls (e.g. H_channel / L_ref) are missing dimensionless groups. The model cannot locate the domain boundaries after scaling. This is consistent with the broad far-field error seen on Design_14, but has not been tested in isolation.
- Training curve: training loss fell from 0.090 to about 0.005, while inner-validation loss oscillated between 0.037 and 0.095 after epoch 40. Three of the five validation designs are the out-of-range small bodies. This is a generalization/extrapolation gap, not underfitting; more epochs are not expected to help. The unannealed cosine learning rate remains a separate confound.
- Candidate remedies, not yet run: keep domain-referenced coordinates (or distances to channel walls/outlet over L_ref) alongside body-scaled wall distance; compress far-field body-scaled coordinates (for example signed log1p); and stratify the inner split so small bodies appear in training. Each should be tested separately with the same split and loss band.

## 2026-11-22 - Two-Scale Inputs and Stratified Inner Split: Protocol

- Implemented two remedies, both opt-in so earlier results stay reproducible. (1) `--stratify-inner`: sort the 24 outer-training designs by L_ref, split them into five contiguous strata, and draw one inner-validation design per stratum with the seed. The outer six test designs are unchanged (they already hold the small bodies Design_1 and Design_14), so outer results remain comparable with earlier runs. (2) Variant `two_scale`: positions mapped to [0, 1] over the fluid-domain bounding box (read from the mesh, not hardcoded as 3 m x 1 m), so inlet, side walls and the 0 Pa outlet stay at fixed input values; wall distance remains d/L_ref plus its logarithm; design inputs add body extent / domain extent (blockage) to the existing Reynolds proxy and aspect ratio. Edges are unchanged from `invariant`.
- Stratified split for fold 2: inner validation Design_5, Design_9, Design_17, Design_19, Design_20; small bodies Design_15 and Design_21 now in inner training (previously all three small bodies were in validation). The selected validation set differs from earlier runs, so new checkpoints are not directly comparable to old inner-validation losses.
- Known residual extrapolation: linear d/L_ref still reaches about 32 on small bodies; stratification brings such values into training but the feature is unbounded.
- Protocol: same outer fold, seed 2026, 1,000/100/10/8 epoch settings. Run sequentially on the stratified split: (a) non-invariant `geometry` latent_attention32, as the matched baseline; (b) `two_scale` latent_attention32; (c) Transolver (geometry inputs). Loss band still differs: 0.05 m for (a) and (c), 0.15 L_ref for (b). Outputs: `volume_results/stratified_20261122/{geometry,two_scale,transolver32}/`. The Transolver run writes four-panel common-scale comparisons (CFD, both latent models, Transolver).
- Verification before launch: 51 tests pass, including bounded domain-frame positions, blockage value, unchanged edges/wall-distance features between frames, and deterministic stratification; Ruff and ty pass. Results pending.
- The first background launch stopped silently during stage (a), leaving only a checkpoint; no metrics were produced. All three stages were rerun from scratch in the foreground.

## 2026-11-22 - Two-Scale and Stratified-Split Results

Same outer fold 2 (six previously inspected designs), stratified inner split (validation Design_5, Design_9, Design_17, Design_19, Design_20), seed 2026, 1,000/100/10/8 epoch settings, CPU.

| Model (stratified split) | Params | Best epoch | Mean RMSE [Pa] | Worst RMSE [Pa] | Near-wall 0.05 m [Pa] | Near-wall 0.15 L_ref [Pa] | Pooled R-squared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Non-invariant latent attention (geometry) | 92,097 | 140 | 3.316 | 4.520 | 10.868 | 13.636 | 0.9489 |
| Two-scale latent attention | 92,161 | 280 | 2.374 | 2.818 | 8.223 | 7.216 | 0.9744 |
| Transolver (geometry inputs) | 27,757 | 250 | 1.387 | 1.835 | 2.774 | 4.899 | 0.9913 |

| Design | Geometry / two-scale whole RMSE [Pa] | Geometry / two-scale near-wall 0.05 m [Pa] |
| --- | ---: | ---: |
| Design_1 | 3.28 / 2.82 | 13.47 / 13.79 |
| Design_3 | 4.47 / 2.63 | 14.61 / 7.34 |
| Design_4 | 2.46 / 2.20 | 6.87 / 4.65 |
| Design_13 | 1.84 / 1.21 | 8.45 / 3.41 |
| Design_14 | 3.32 / 2.82 | 14.47 / 14.85 |
| Design_26 | 4.52 / 2.57 | 7.34 / 5.30 |

- Two-scale versus its matched baseline: whole-domain RMSE improved on all six designs (mean -28%, worst -38%). The 0.05 m near-wall RMSE improved on four designs but slightly worsened on the two small-body designs (Design_1, Design_14). The body-scaled `invariant` failure is resolved on this split: worst design 10.01 -> 2.82 Pa.
- Transolver is the best model on every metric, by a wide margin near the wall (2.77 versus 8.22 Pa). Transolver also improved strongly versus its earlier random-split run (mean 2.49 -> 1.39 Pa, near-wall 10.62 -> 2.77 Pa), so the stratified split is a large effect for all models, likely because small bodies now appear in training.
- Visual inspection, common pressure scales: [Design_1](volume_results/stratified_20261122/transolver32/Design_1_benchmark_comparison.png) and [Design_14](volume_results/stratified_20261122/transolver32/Design_14_benchmark_comparison.png). Transolver reproduces the stagnation and wake regions smoothly. Both latent-attention models show a speckled, patchy pressure region ahead of the body on the small bodies; the non-invariant model also shows a broad error lobe behind the body that two-scale reduces. Other designs: `volume_results/stratified_20261122/transolver32/Design_*_benchmark_comparison.png`. Metrics: `benchmark_summary.csv` there, and each model's `metrics.json`.
- Confounds and limitations: (1) the near-wall loss band differs (0.15 L_ref for two-scale, 0.05 m for the other two), so the two-scale gain mixes input and loss changes; (2) Transolver uses geometry inputs and a different architecture with fewer parameters; (3) the geometry baseline on the stratified split (3.32 Pa) is worse than on the random split (2.98 Pa), showing split-to-split variation of a similar order to some differences; (4) one seed, one previously inspected outer fold; inner-validation losses are not comparable with earlier splits. No model is promoted.
- Next candidates: Transolver with two-scale inputs (separates inputs from architecture); two-scale with the 0.05 m loss band (separates inputs from loss); repeated seeds or folds before drawing conclusions.

## 2026-11-22 - Decision: Step Away from Pure GNN; Transolver Improvement Plan

- Decision: stop investing in pure message-passing GNN variants. On the stratified split, Transolver (27,757 parameters) beats the best GNN (two-scale latent attention, 92,161 parameters) by 42% in mean RMSE (1.39 vs 2.37 Pa) and about 3x near the wall (2.77 vs 8.22 Pa, 0.05 m band). The hypothesis that face-to-face message passing would win near the wall was not supported. The two largest gains so far (stratified split, two-scale inputs) are architecture-agnostic and can be carried over. GNN code and checkpoints stay as frozen references.
- Caveat: one seed, one previously inspected outer fold; the geometry baseline moved by about 0.3 Pa between splits alone. A 3x near-wall gap is unlikely to be noise, but confirm with seeds before promoting.
- Plan, implemented one step at a time on the same stratified fold-2 split, each compared against the previous step with common-scale CFD/prediction images:
  1. Two-scale inputs for Transolver. It currently receives raw cell area as a node input, which is mesh-dependent; two-scale removes it and adds blockage and the Reynolds proxy. Separates the input effect from the architecture effect.
  2. Area-weighted slicing. Slice tokens are currently unweighted averages over cells, so refined regions (boundary layer) dominate the tokens and the result depends on mesh density. Weighting each cell by its area turns the slice average into a quadrature of a field integral, which is the discretisation-invariant form.
  3. Learning-rate schedule. Every nested run stopped at epoch 250-350 with cosine T_max = 1,000, so the learning rate never annealed. Tie the schedule to the realistic training length or use plateau-based decay; this affects all models.
  4. Seed check (2-3 seeds, about 14 minutes each), then capacity (64 slices, width 64) and the near-wall band question (0.05 m vs 0.15 L_ref).
  5. Optional hybrid: one local message-passing layer inside Transolver, only if near-wall detail is still the weak spot after steps 1-3.
- Once Transolver becomes the main line, consider moving it out of `benchmarks/`.

### Physics in the Loss: Current State and Limits

- The loss is purely data-driven: area-normalized squared error on the pressure coefficient, mixed 50/50 between whole domain and near-body band. No momentum or continuity term is used.
- The volume export contains only cell centroids, wall distance and pressure. Momentum and continuity residuals need the velocity field (and viscosity); the solver is Spalart-Allmaras RANS per the run logs, so a faithful residual would also need eddy viscosity. A Navier-Stokes residual therefore requires exporting velocity (and ideally turbulent viscosity) from STAR-CCM+ first; it cannot be added with the current data.
- Weaker pressure-only constraints are possible now: (a) the 0 Pa outlet condition; (b) near-zero wall-normal pressure gradient at the body (boundary-layer approximation); (c) a global channel balance against the logged `PressureDrop`, and a pressure-drag consistency check against the logged `Drag` (which also includes shear, so only partial). These are candidate regularizers, not momentum conservation, and are deferred until after steps 1-3.

## 2026-11-22 - Discussion: Conservation Losses, Turbulence Closure, Multi-Scale Models (Planning, No Code)

### 1. Conservation: re-export from STAR-CCM+ (user action)

- Decision: the user will re-export the 30 designs from STAR-CCM+ with the fields needed for continuity and momentum residuals. Until then, continue Transolver steps 1-3, which do not depend on the new data.
- Export checklist (same `Volume_Dataset.simh` layout, same mesh, cell-centred unless noted; exact STAR-CCM+ field names to be confirmed in the tool):
  - Velocity vector (u, v; w is zero in 2D), pressure (already present), density (to confirm it is constant).
  - Molecular dynamic viscosity, turbulent (eddy) viscosity, and the Spalart-Allmaras transported variable (modified diffusivity, nu-tilde). Eddy viscosity is required for the viscous/Reynolds-stress flux; nu-tilde is needed for any closure work.
  - Optional but useful: velocity gradient tensor and vorticity as exported by STAR (lets us check our own gradient reconstruction), turbulent viscosity ratio.
  - Interior-face mass flux, if exportable: lets us verify discrete continuity of the truth data on our reconstructed mesh.
  - Boundary face values on IN, OUT, TOP, BOTTOM and WALL: velocity and pressure; on WALL also wall shear stress and y+.
  - Solver metadata: boundary types (are TOP/BOTTOM slip or no-slip?), inlet velocity and turbulence conditions (intensity, viscosity ratio or nu-tilde), outlet type, discretization schemes, final residual levels. Needed to build boundary fluxes consistently.
- Implementation plan once data arrives:
  - A. Loader reads the new fields; build a differentiable finite-volume operator from the existing face connectivity (face area vectors, owner/neighbour cells, boundary faces). Use the integral (finite-volume) form, not pointwise autodiff residuals: it is consistent with the solver, works on any mesh, and is discretisation-consistent, which fits the invariance goal.
  - B. First measure the residual of the CFD truth itself under our operator. Our reconstruction will not exactly match STAR-CCM+'s discretization (gradient reconstruction, interpolation, upwinding), so truth residuals will not be zero, especially in thin near-wall cells. Physics losses must be measured relative to this baseline, not driven to zero.
  - C. Extend Transolver outputs to non-dimensional u/V, v/V, p/q and log(nu_t/nu). Loss = data term + lambda_c * continuity + lambda_m * momentum, non-dimensionalized by q * L_ref (or channel scales), ramped in after a data-only warm-up. Loss-term balancing is a known failure mode (gradient pathologies), so use adaptive weighting or careful normalization and log each term.
  - D. Evaluate whether conservation improves pressure RMSE, near-wall error and especially the small-body/extrapolation designs, not just residual levels. Physics losses are expected to help most where data are sparse.
- Risks: near-wall residuals dominated by viscous terms on high-aspect-ratio cells; boundary-flux treatment errors; predicting four fields with the same capacity may cost pressure accuracy.

### 2. Generalised turbulence closure (research track)

- Important limitation: all training data are Spalart-Allmaras RANS. A model trained only on SA outputs can at best reproduce SA's closure; it cannot learn a better closure. Improving the closure itself requires higher-fidelity targets (LES/DNS, experiments, or at least a more general RANS model such as a Reynolds-stress model) for at least some cases.
- Realistic staged path:
  - Stage 1 (possible with the re-export): learn the SA eddy viscosity (or nu-tilde) as a cell-local function of non-dimensional, Galilean- and rotation-invariant flow features: wall-distance Reynolds number, strain/rotation magnitude ratio (Q-criterion based), normalized streamwise pressure gradient, viscosity ratio, and so on. Test whether a cell-local model trained on some geometries reproduces SA's nu_t on held-out geometries. This tests "cell-local, geometry-independent" learning cheaply and reuses the invariance work (non-dimensional inputs, no cell size, no absolute position).
  - Stage 2 (needs higher-fidelity data): field inversion + machine learning: infer a correction field to SA that matches high-fidelity data, then learn that correction from local invariant features. Alternatively, predict Reynolds-stress anisotropy using an invariant tensor basis (tensor-basis neural network).
  - Stage 3: embed the learned closure in a solver and check a posteriori consistency. Models that look good on frozen (a priori) data can make the RANS system ill-conditioned or unstable.
- Generalization levers: strictly local, non-dimensional, invariant inputs; realizability constraints on Reynolds stresses; training across diverse flows (attached, separated, wake); explicit out-of-distribution detection on input features. The x/L extrapolation failure in this log is a concrete example of why input-distribution checks matter.
- This is a separate problem from the field surrogate. The surrogate predicts the converged CFD field directly; a closure model is used inside a solver. They can share features and data, and Stage 1 can be trained alongside the conservation work.

### 3. Multi-scale networks, attention and physics losses

- Transolver slices already act as a learned coarse scale (soft clustering into physical states). Multi-scale options to test after steps 1-3: (a) hierarchical slices (coarse and fine slice counts in different layers); (b) a local operator next to global slice attention (plan step 5); (c) a mesh-coarsening U-Net-style hierarchy (multi-scale message passing). Prefer (a)/(b) first because they keep the Transolver code base small.
- Physics losses on neural operators: physics-informed operator training (data + PDE residual) is the natural fit for plan section 1. Pointwise-autodiff PINN residuals are a poor fit for cell-averaged unstructured data.
- Paper list: the user referenced a list of papers under these headings, but it was not included in the message. Reading list from memory, to be verified before citing in `paper/main.tex`:
  - Multi-scale / attention / operators: Wu et al. 2024 (Transolver, ICML); Pfaff et al. 2021 (MeshGraphNets, ICLR); Fortunato et al. 2022 (MultiScale MeshGraphNets); Li et al. 2021 (Fourier Neural Operator, ICLR); Li et al. Geo-FNO (operators on general geometries).
  - Physics losses: Raissi, Perdikaris and Karniadakis 2019 (PINNs, JCP); Li et al. physics-informed neural operator (PINO); Wang, Teng and Perdikaris 2021 (gradient pathologies in PINNs, SIAM J. Sci. Comput.); Krishnapriyan et al. 2021 (PINN failure modes, NeurIPS).
  - Turbulence closures: Pope 1975 (general effective-viscosity hypothesis / tensor basis, JFM); Ling, Kurzawski and Templeton 2016 (tensor-basis neural network, JFM); Parish and Duraisamy 2016 (field inversion and machine learning, JCP); Weatheritt and Sandberg 2016 (gene-expression programming for algebraic stress models, JCP); Wu, Xiao and Paterson 2018 (physics-informed ML for RANS, Phys. Rev. Fluids); Wu et al. 2019 (ill-conditioning of RANS with data-driven Reynolds stress, JFM); Duraisamy, Iaccarino and Xiao 2019 (Turbulence modeling in the age of data, Annu. Rev. Fluid Mech.); Duraisamy 2021 (perspectives on ML-augmented RANS/LES, Phys. Rev. Fluids).

### Updated roadmap (one step at a time, same split, images each step)

1. Transolver + two-scale inputs.
2. Area-weighted slicing.
3. Learning-rate schedule fix.
4. Seeds, then capacity and near-wall band.
5. Optional local + global hybrid / hierarchical slices.
6. Conservation losses (after the STAR-CCM+ re-export): truth-residual baseline, multi-field outputs, continuity then momentum.
7. Closure track Stage 1: cell-local learned SA eddy viscosity on invariant features, tested across held-out geometries. Stages 2-3 only with higher-fidelity data.

## 2026-11-22 - Public Benchmarks, Head-to-Head Table and Paper-Update Rule

### Why public benchmarks

- Results on our 30 private STAR-CCM+ designs cannot be verified by reviewers against established baselines; on their own they read as an engineering case study. Final claims must be made on standardized community benchmarks, with each benchmark's official splits and metrics, alongside published state-of-the-art numbers.
- Candidate benchmarks (details from memory; confirm data format, splits, metrics and licences before use):
  - **AirfRANS** (Bonnet et al., NeurIPS 2022 Datasets): 2D incompressible RANS around airfoils with Spalart-Allmaras, volume fields (u, v, p, nu_t) and standardized splits. Closest match to our current 2D SA RANS volume setup, including the conservation and closure tracks; Transolver reports results on it. Recommended first public benchmark.
  - **DrivAerNet / DrivAerNet++**: realistic 3D car geometries with CFD fields and drag coefficients; standard for automotive aerodynamic ML.
  - **3D Ahmed body**: classic automotive bluff-body benchmark for wake and separation.
  - **ShapeNet Car** (Umetani and Bickel 2018 dataset as used by GNOT and Transolver): 3D surface pressure and velocity around car shapes; Transolver reports results on it.
  - **Darcy flow** (FNO benchmark): structured-grid PDE benchmark, not aerodynamic; useful only as a sanity check that our Transolver implementation reproduces published operator-learning numbers.
- Prerequisites: a real 3D volume/surface loader (deferred so far) and GPU training; DrivAerNet and Ahmed meshes are far larger than our 2D designs. CPU-only training will not scale.
- Metrics caveat: public benchmarks usually report normalized metrics (relative L2, MSE on normalized fields, drag-coefficient error, R2), not pressure in Pa. Report each benchmark in its own official metric; keep Pa only for our private dataset.
- Head-to-head protocol: use published numbers only when the protocol (split, preprocessing, metric) is identical; otherwise rerun the baseline under our protocol and say so. Baselines to include (as published for each benchmark): Transolver (and Transolver++ where available), GNOT, MeshGraphNets, Geo-FNO/FNO, PointNet / RegDGCNN (DrivAerNet baselines), GINO.

### Running comparison table (private dataset, stratified fold-2 split)

Reference row: the current Transolver implementation (`benchmarks/transolver`). Every new addition from the roadmap is trained on the same split and appended as a new row; never overwrite earlier rows.

| Model | Whole-domain pressure RMSE (Pa) | Near-wall RMSE, 0.05 m (Pa) | CD error (%) | Memory (GB) | Inference time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transolver, geometry inputs (benchmark) | 1.387 | 2.774 | n/m | n/m | n/m |
| Transolver, geometry inputs (GPU rerun) | 1.110 | 2.025 | n/m | n/m | n/m |
| Two-scale latent-attention GNN (reference) | 2.374 | 8.223 | n/m | n/m | n/m |
| Non-invariant latent-attention GNN (reference) | 3.316 | 10.868 | n/m | n/m | n/m |
| Latent-attention GNN, geometry inputs (GPU rerun) | 2.412 | 6.843 | n/m | n/m | n/m |
| Latent GNN, 400-epoch annealed schedule (2026-11-24) | 4.150 | 10.224 | n/m | n/m | n/m |
| Latent GNN + area-weighted read, 2/4 heads, 400-epoch schedule | 2.877 | 10.356 | n/m | n/m | n/m |
| Latent GNN + area read + area aggregation/graph mean, 400-epoch schedule | 3.093 | 10.516 | n/m | n/m | n/m |
| Latent GNN + area-weighted read, T_max 1,000 early stop | 4.933 | 12.222 | n/m | n/m | n/m |
| Latent GNN + area read + aggregation, T_max 1,000 early stop | 3.210 | 10.504 | n/m | n/m | n/m |
| Latent GNN, continuous inputs (x, y, d; direction + log dist/L edges), sum | 1.417 | 4.019 | n/m | n/m | n/m |
| Latent GNN, continuous inputs, mean aggregation | 1.914 | 4.895 | n/m | n/m | n/m |

- n/m = not measured yet. Measurement rules for future rows: CD error from predicted pressure integrated over the body against the logged `Cd`; until wall shear is exported this is pressure drag only and must be labelled as such. Memory = peak training memory per design (state CPU RSS or GPU peak allocated). Inference time = mean forward time per design after warm-up, with hardware and thread count recorded. Backfill these three columns for existing rows before claiming any efficiency advantage.

### 2026-11-22 - Transolver GPU rerun

- Repeated the geometry-input Transolver on the same stratified fold and seed using the NVIDIA RTX PRO 4000 Blackwell laptop GPU. Training stopped at epoch 680, with the best validation checkpoint at epoch 600. The previous CPU run stopped at epoch 330, with its best checkpoint at epoch 250.
- Across six previously inspected held-out designs, mean RMSE was 1.110 Pa and near-wall RMSE was 2.025 Pa, versus 1.387 Pa and 2.774 Pa for the CPU reference. Training took 124.2 s for 680 epochs (0.183 s/epoch), versus 823.3 s for 330 epochs (2.495 s/epoch) in the earlier CPU run. These are single runs with different stopping points, PyTorch builds and execution devices, not a controlled accuracy or performance benchmark.
- Common-scale actual CFD, CPU reference and GPU predictions: [Design_1](volume_results/gpu_transolver_full_20261122/Design_1_benchmark_comparison.png) and [Design_14](volume_results/gpu_transolver_full_20261122/Design_14_benchmark_comparison.png). The remaining held-out comparisons and checkpoint are in `volume_results/gpu_transolver_full_20261122/`.

### 2026-11-22 - Latent-attention GPU rerun

- Repeated the geometry-input latent-attention model on the same stratified fold and seed using CUDA. Early stopping ended training at epoch 650; the best checkpoint was at epoch 570. The earlier CPU run's best checkpoint was at epoch 140.
- Across six previously inspected held-out designs, mean RMSE was 2.412 Pa and near-wall RMSE was 6.843 Pa, versus 3.316 Pa and 10.868 Pa for the CPU latent reference and 1.110 Pa and 2.025 Pa for the GPU Transolver. The CPU and GPU results are single runs with different checkpoints and PyTorch builds; they do not establish that GPU training improves accuracy.
- CPU-only evaluation initially failed because the trained model remained on CUDA. Evaluation was fixed to move the model to CPU and the held-out results were recovered from the saved checkpoint without retraining. Common-scale CFD, CPU latent, GPU Transolver and GPU latent predictions: [Design_1](volume_results/gpu_latent_attention_geometry_20261122/latent_attention32/Design_1_benchmark_comparison.png) and [Design_14](volume_results/gpu_latent_attention_geometry_20261122/latent_attention32/Design_14_benchmark_comparison.png).

### Paper-update rule

- After every completed experiment, update `paper/main.tex` in the same session, even roughly: add the new row to the head-to-head table, one or two sentences of interpretation, and the image reference. The narrative (from GNN baseline to invariance failure, to two-scale fix, to Transolver as the main line, to conservation and public benchmarks) matters more than polish; polishing comes later.

## 2026-11-22 - AirfRANS Reduced-Split Reference Transolver

- Protocol: authors' released Transolver implementation at upstream revision `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`; stratified subset of the official full-data split (96 train, 16 validation, 40 held-out test); 32,000 randomly sampled nodes per step; 398 epochs; Adam/OneCycleLR, max LR 0.001; seed 2026. Data normalization was fit on the 96 selected training cases. Run on NVIDIA RTX PRO 4000 Blackwell laptop GPU, peak allocated memory 3.61 GiB, training time 4,683 s (78.1 min). Checkpoint: `volume_results/airfrans_reduced_20261122/laptop_reference_398epoch/reference.pt`.
- Held-out metrics on 40 cases: normalized volume pressure MSE 0.0850, surface pressure MSE 0.6920, mean per-case relative lift error 1.675, lift Spearman 0.9413. Per-case pressure-MSE medians were 0.0276 (volume) and 0.0222 (surface). The worst surface case, at 14.684 degrees angle of attack, had MSE 23.0785 and dominated the surface mean; excluding that one case gives 0.1179, still not a parity claim. Shared-scale visualization: `volume_results/airfrans_reduced_20261122/laptop_reference_398epoch/evaluation/worst_surface_case.png`.
- Interpretation: 398 epochs fit the laptop budget, but this subset result is far from the paper's reported normalized pressure MSE values (0.0037 volume, 0.0142 surface) and lift scores. It is not a controlled paper reproduction: only 96 selected cases received gradient updates versus 720 in the authors' full-data training setup, with 40 rather than 200 test cases. The high-angle outlier merits a separate case-level check. Keep this as an exploratory reduced-data baseline; do not claim parity or promote the checkpoint.

## 2026-11-23 - Local Transolver vs Authors' Reference on AirfRANS

- Protocol: identical reduced split, 398 epochs, 32,000 points, seed 2026 and evaluation data. Authors' upstream implementation: best validation loss 0.1840 at epoch 380, final 0.1840 at epoch 398, 3.61 GiB peak. Local implementation: best validation loss 0.2197 at epoch 360, final 0.2252 at epoch 398, 3.64 GiB peak. Both took about 78 minutes for training.
- Held-out 40-case comparison: upstream volume/surface pressure MSE 0.0850/0.6920 and relative lift error/Spearman 1.6751/0.9413; local volume/surface pressure MSE 0.1906/0.9237 and relative lift error/Spearman 0.7813/0.8161. Paired by case, local improved lift-relative error on 20 cases and worsened it on 20, while upstream improved pressure MSE in 21/40 volume cases and 24/40 surface cases.
- Interpretation: the upstream implementation remains the better pressure surrogate on this reduced experiment. The local implementation's lower mean lift-relative error comes without lift-rank improvement and is not enough to promote it. Main differences include the upstream direct final head, weight initialization and positional-grid encoding; bring the local implementation to the upstream architecture before changing slices, capacity or loss design. No architecture is promoted.

## 2026-11-23 - Kaggle 398-Epoch Latent-Attention GNN on AirfRANS

- Run: same 96/16/40 reduced split and seed 2026, 32,000 sampled nodes per training step, 398 epochs on Kaggle Tesla T4. Output: `volume_results/airfrans_reduced_20261122/kaggle_gnn_latentattention32_398epoch/airfrans_gnn_latentattention32_30epoch/`. The checkpoint confirms a latent-attention GNN with three layers, hidden width 32, 32 tokens and four attention heads.
- Double-check: the saved `gnn.pt` reports checkpoint epoch 398 and 398 history rows. All 40 unique test NPZ files are present. Recomputing the JSON aggregates from those 40 case rows gives exactly volume MSE 0.056089426844846454, surface MSE 0.14480829813110177, mean relative lift error 0.8545636161846121 and lift Spearman 0.9465290806754223.
- Mean-versus-median caveat: pressure-MSE medians are 0.00986049 volume and 0.00634663 surface, versus the means above. The surface mean is raised by concentrated high-angle outliers; one 14.684-degree case has surface MSE 3.4092, while the upstream reference has 23.0785 and the local Transolver has 21.7196 there.
- Paired held-out comparison: versus the upstream reference, GNN volume/surface MSE improves on 36/40 and 35/40 cases respectively and lowers mean pressure MSE to 0.0561/0.1448. Versus local Transolver, it improves volume/surface MSE on 32/40 and 34/40 cases. It is worse than local Transolver on mean lift-relative error (0.8546 vs 0.7813) but has better lift Spearman (0.9465 vs 0.8161).
- Interpretation: this is materially better pressure performance and visually promising, but it is a final-epoch 398-epoch comparison, not a documented validation-selected checkpoint. The current run may be using a bad or unrebalanced split. It must be repeated with the same best-validation epoch-selection protocol used by the private GNN studies, at least two seeds, and an explicit common pressure/lift metric recomputation. No promotion yet.

## 2026-11-23 - Retrospective Matched AirfRANS Analysis

- Scope and correction: this is an analysis of the existing runs, not a new training experiment. The saved split lists and all normalisation arrays match exactly for the reference and GNN; all 40 test case names, saved positions and actual pressure fields match. The split is shared by the compared models, regardless of whether it is representative of the full dataset. Checkpoint history shows the GNN's lowest **recorded** validation loss was 0.10948 at epoch 398; the reference's was 0.18370 at epoch 380, but only its final epoch 398 was saved for evaluation. This clarifies the preceding entry's checkpoint-selection caveat, not a separately performed best-checkpoint evaluation.
- The saved checkpoints contain 91,780 GNN and 5,996,868 reference parameters. Both used 96 train / 16 validation / 40 test cases, seed 2026, 32,000 sampled nodes, 398 epochs and the same four-field surface-plus-volume MSE loss. The reference ran on the laptop GPU and the GNN on a Kaggle T4; do not compare recorded runtime or memory as controlled efficiency results. The GNN uses default activations, not the private-study tanh variant.
- Reference vs GNN mean standardised pressure MSE: volume 0.0850112 -> 0.0560894 (34.0% reduction, 36/40 case wins); surface 0.691963 -> 0.144808 (79.1%, 35/40 wins). Median surface MSE: 0.0221527 -> 0.00634663. Mean lift relative error: 1.67510 -> 0.854564; median: 0.244345 -> 0.123761; lift Spearman: 0.941276 -> 0.946529. Small true lift makes per-case relative errors unstable; the 14.684-degree case has lift relative error 0.00863 -> 0.36945 despite better GNN pressure.
- Outlier sensitivity: the 14.684-degree case has reference/GNN surface MSE 23.0785/3.4092. Excluding it changes the mean surface MSE to 0.117948/0.061106, a 48% GNN reduction; mean volume MSE becomes 0.058835/0.035275. Excluding this case is a sensitivity analysis, not a replacement for the full-set result.
- Spatial analysis: for each case, compute node-wise pressure MSE from the saved unstandardised fields in each region and average the case MSEs equally. Chord `c` is the streamwise surface extent and `x/c` is measured from the minimum surface x. Relative to the reference, GNN mean MSE fell 79% on the leading surface (`x/c < 0.1`, 34/40 wins), 16% on the middle surface (`0.1 <= x/c < 0.7`, 31/40), 54% on the trailing surface (`x/c >= 0.7`, 36/40), 31% at non-surface nodes within `0.02c` of the wall (35/40), and 71% in the non-surface wake box (`1 <= x/c <= 1.5`, `abs(y) < 0.2c`, 34/40). The leading-surface reduction is 46% without the high-angle case. These boxes overlap, use node rather than area weighting, and are not official benchmark regions.
- Separate retrospective relative-L2 calculation: average each case's unstandardised `||p-p_hat||_2 / ||p||_2` over cases, separately for volume and surface: reference 0.6001/0.4983, GNN 0.4351/0.2774. Relative L2 over the vector of all 40 lift coefficients is 0.4789/0.2151. The published paper describes relative L2, while its released evaluation code computes field MSE; this aggregation is explicitly defined here and is not established to reproduce the published table. Saved NPZs only include pressure, so velocity, viscosity and a full force-metric recomputation require renewed inference.
- Images with shared CFD/model pressure and error scales: [high-angle outlier](volume_results/airfrans_reduced_20261122/gnn_reference_analysis_20261123/airFoil2D_SST_90.415_14.684_4.356_6.209_11.299_cfd_reference_gnn.png), [another held-out case](volume_results/airfrans_reduced_20261122/gnn_reference_analysis_20261123/airFoil2D_SST_67.783_-2.041_4.431_3.865_18.25_cfd_reference_gnn.png), and [GNN surface-error regression](volume_results/airfrans_reduced_20261122/gnn_reference_analysis_20261123/airFoil2D_SST_85.752_6.341_0.382_6.715_8.448_cfd_reference_gnn.png). Error colours clip at the 99th percentile in the plotted near-airfoil region.
- Hypotheses, not demonstrated causes: local mesh messages plus global tokens, or the smaller model's capacity, could improve reduced-data generalisation. On one sampled high-angle mesh, 42.2% of the 32,000 sampled nodes had no retained mesh neighbour, so graph propagation alone cannot be assumed responsible. Repeat seeds, evaluate validation-selected checkpoints for both models, run graph/attention and capacity ablations, and retain all four predicted fields before promoting a model or claiming an architectural cause. The reduced subset is not directly comparable with the paper's larger-data results.

## 2026-11-24 - Area-Weighted Latent Attention and Message Passing: Implementation and Protocol

### Rationale (physics)

- Incompressible pressure solves a Poisson problem, so p(x) = integral of G(x, y) s(y) dA(y) over the domain. Global mixing should therefore approximate an area integral. The existing cell-to-token read is softmax over cells with one vote per cell, so it approximates the integral of (w v rho_mesh) dA, where rho_mesh is cell density. Near-wall cells are 67.6% of cells but 1.73% of area, so tokens mostly summarise the boundary layer and change when the mesh is refined. The count-based graph mean in every message-passing block has the same bias.
- Adding log A_i to a read key gives softmax(s_i + log A_i), proportional to A_i exp(s_i): a quadrature of an integral over the area. Refinement then leaves the read unchanged. The token-to-cell write and token self-attention involve no cell measure and are unchanged.
- Hypothesis only: discretisation-consistent global mixing improves generalisation across designs with different mesh grading. It has not been measured.

### Decisions

- Mixed heads, not pure area weighting. Of the four read heads, two have beta = 1 (integral heads, bias log A) and two have beta = 0 (cell-count heads, current behaviour). Pure area weighting would make tokens need logit differences of order log(A_far / A_wall) to resolve the near-wall band, which risks starving the region we care most about. beta is fixed, not learned, so the effect is interpretable; a learnable beta is deferred.
- Message passing, option (a): neighbour-area quadrature, sum_j A_j m_ji / sum_j A_j, replacing the sum. In the neighbour-attention path the same option adds log A_j to the edge scores. Option (c): the graph-mean context becomes an area-weighted mean. Both are controlled by one flag, `area_aggregation`. Caveat: (a) turns a sum into a weighted mean, so node-degree information, including whether a cell touches a boundary, is lost; wall distance remains an input.
- Deferred, option (b): finite-volume-style face-length weights l_f / d_ij. Face lengths are available in the loader (interior face vertices) but are not stored.
- Areas are the physical cell areas `geom[:, 3]`. Their units do not matter: softmax is shift-invariant in log A, and the weighted means are ratios.
- New options are off by default. State-dictionary keys and parameter count (92,097) are unchanged, and old checkpoints load with the options off. New checkpoints record `area_heads` and `area_aggregation`.
- Learning rate: added `--schedule-epochs`. It decays with a cosine to 1% of the initial rate over that length, then holds the minimum (PyTorch's `CosineAnnealingLR` would rise again). Without the option, the old schedule (T_max = max epochs) is used unchanged, so earlier runs stay reproducible.

### Verification

- Exact discretisation test: split one cell into three copies with A/3 each, duplicating its faces. With all heads area-weighted and area aggregation on, every original cell's output and each copy's output match the unrefined graph. The count-weighted model changes under the same refinement, so the test is sensitive. Also tested: head biases, the requirement for areas, checkpoint round trips, parameter-count equality, and the schedule holding its minimum. 73 tests pass; Ruff and ty pass on changed files. A two-epoch real-data GPU smoke run completed; its artifacts were deleted and it is not accuracy evidence.

### Protocol

- Same stratified fold-2 split as the 2026-11-22 GPU runs: 19 inner-training designs, inner validation Design_5/9/17/19/20, six previously inspected outer designs. Seed 2026, geometry inputs, 0.05 m near-body-balanced loss, CUDA.
- 400 epochs with a 400-epoch schedule, so every run anneals fully. Early stopping is disabled (minimum epochs = 400); the best inner-validation checkpoint is selected every 10 epochs.
- Three sequential arms with identical settings: `latent_attention32` (matched baseline under the new schedule; the earlier GPU row used T_max = 1,000, so it is not a matched control), `latent_attention32_area_read` (area-weighted read only), and `latent_attention32_area` (read + (a) + (c)). This separates the schedule effect from the attention effect and the read effect from the aggregation effect.
- Command: `uv run python -u compare_volume_architectures.py --design Design_1 --stratify-inner --architectures latent_attention32 latent_attention32_area_read latent_attention32_area --max-epochs 400 --schedule-epochs 400 --minimum-epochs 400 --validation-interval 10 --patience 8 --results-dir volume_results/area_weighted_20261124 --comparison-dir volume_results/gpu_latent_attention_geometry_20261122/latent_attention32 --comparison-label "Latent attention (GPU, T_max=1000)"`.
- Status: started; results pending.

## 2026-11-24 - Area-Weighted Latent Attention: Results (Mixed, Dominated by Training Variance)

Both protocols use the stratified fold-2 split, seed 2026, geometry inputs and CUDA. The baseline row is the 2026-11-22 GPU run, not rerun under the T_max = 1,000 protocol.

| Model | Schedule | Best epoch | Best inner val. loss | Mean RMSE [Pa] | Worst RMSE [Pa] | Near-wall 0.05 m [Pa] | Pooled R-squared |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| latent_attention32 (2026-11-22 GPU) | T_max 1,000, early stop (650) | 570 | 0.00394 | 2.412 | 3.286 | 6.843 | 0.9730 |
| latent_attention32 | 400, annealed, no early stop | 170 | 0.00643 | 4.150 | 6.968 | 10.224 | 0.9151 |
| latent_attention32_area_read | 400, annealed, no early stop | 230 | 0.00352 | 2.877 | 5.278 | 10.356 | 0.9579 |
| latent_attention32_area | 400, annealed, no early stop | 350 | 0.00473 | 3.093 | 4.170 | 10.516 | 0.9554 |
| latent_attention32_area_read | T_max 1,000, early stop | 110 | 0.00581 | 4.933 | 6.392 | 12.222 | 0.8899 |
| latent_attention32_area | T_max 1,000, early stop | 150 | 0.00623 | 3.210 | 4.854 | 10.504 | 0.9516 |

- **Learning-rate fix not supported.** With the full 400-epoch anneal, the matched baseline overfits: training loss 0.00048 against inner-validation 0.0077 at epoch 400; best checkpoint at 170. Its held-out mean RMSE (4.15 Pa) is much worse than the unannealed T_max = 1,000 GPU run (2.41 Pa). The earlier "unannealed learning rate" confound may have acted as regularisation. This is one seed; the 400-epoch length is the only tested anneal.
- **Within the 400-epoch protocol**, both area arms beat the matched baseline on mean whole-domain RMSE: area read 4.15 -> 2.88 Pa, better on 5/6 designs; full area 3.09 Pa with the lowest worst-design RMSE of the three, 4.17 Pa. Area read also had the lowest inner-validation loss of any latent run so far (0.00352). Mean near-wall RMSE was unchanged (10.2-10.5 Pa). On the small-body extrapolation design Design_14, area read doubled the near-wall error (13.75 -> 25.55 Pa); see [Design_14 area read](volume_results/area_weighted_20261124/latent_attention32_area_read/Design_14_comparison.png). The stagnation peak is underpredicted and the suction region is distorted. Full area (18.45 Pa) also underpredicts the peak but keeps the suction-region shape: [Design_14 full area](volume_results/area_weighted_20261124/latent_attention32_area/Design_14_comparison.png).
- **Under the T_max = 1,000 early-stopping protocol**, both area arms stopped early (best epochs 110 and 150, against 570 for the baseline) and are worse than the baseline on every metric. Validation and training losses are noisy: for example, a training-loss spike to 0.016 at epoch 200 in the full-area arm. Patience-based stopping then selects whichever noisy minimum came first.
- **Interpretation:** no area-weighted arm beats the best existing latent run (2.41 / 6.84 Pa), and no claim of improvement is made. Run-to-run differences of a factor of about 2 in mean RMSE come from the schedule and stopping rule alone. This is larger than any area effect, so single-seed comparisons cannot resolve the hypothesis. The near-wall band was never helped: the cell-count heads did not protect it, and the integral heads may hurt small bodies.
- Artifacts: `volume_results/area_weighted_20261124/` (400-epoch arms) and `volume_results/area_weighted_tmax1000_20261124/`. Each architecture directory contains a checkpoint, metrics, validation curve, NPZ predictions and six CFD / GPU-baseline / new-model comparison PNGs on shared pressure scales.
- Next, before any further architecture change: (1) at least three seeds for baseline and area read under one fixed protocol; (2) reduce optimisation noise, for example gradient clipping, a validation-loss moving average for selection, or a longer patience; (3) only then a learnable beta or option (b) face-length weights. Do not promote any area variant.
- Follow-up decision (same day): the `--schedule-epochs` option and `cosine_schedule()` helper were removed, because the annealed schedule was not useful. All runs return to the original `CosineAnnealingLR` with T_max = max epochs and patience-based early stopping. The 400-epoch artifacts are kept as negative evidence but can no longer be reproduced with the current CLI (they used the removed option with 400 schedule epochs).

## 2026-11-24 - Area-Weighted GNN on AirfRANS: Implementation (Kaggle Run Pending)

- AirfRANS fields are vertex (point) data on planar VTK quads, so cell areas are converted to lumped nodal areas: each quad's shoelace area is shared equally among its vertices. On one real case the nodal areas sum exactly to the mesh area (19.2378); the smallest is 6e-12 and none are zero. Surface nodes are 0.56% of nodes but only 1.1e-7 of the area. The two integral heads therefore effectively ignore the airfoil surface; the two cell-count heads still see it. The surface-pressure metric may expose this.
- Random 32,000-node sampling is uniform over nodes. Area weights over a uniform subset give an unbiased quadrature up to a constant, which cancels in the softmax and weighted means. Area-weighted reads should therefore be consistent between training subsets and the multi-pass full-mesh inference. This is a design argument, not a measurement. Isolated sampled nodes (about 42%) still receive no neighbour messages.
- New `BenchmarkModel` kinds: `gnn_area_read` (2/4 area read heads) and `gnn_area` (also area-weighted aggregation and graph mean). Otherwise they are identical to the 398-epoch `gnn` run: width 32, 3 layers, 32 tokens, 4 heads, 91,780 parameters, Adam + OneCycleLR, 32,000 points, seed 2026, 96/16/40 split. As before, the evaluated checkpoint is the final epoch, not a validation-selected one.
- Kaggle: re-upload the code dataset including `model.py` and `benchmarks/`, then run the new notebook cell in `gdl-transolver.ipynb` (`--models gnn_area_read gnn_area --epochs 398`). Results pending; compare against `volume_results/airfrans_reduced_20261122/kaggle_gnn_latentattention32_398epoch/`.

## 2026-11-24 - Area-Weighted GNN on AirfRANS: Results (Mixed)

- Only `gnn_area` was run (2/4 area read heads + area aggregation + area graph mean); `gnn_area_read` has not been run. Kaggle Tesla T4, torch 2.11.0+cu130 (same build and GPU as the `gnn` run), 398 epochs, final-epoch checkpoint, 5,679 s training. Artifacts: `volume_results/airfrans_reduced_20261122/kaggle_gnn_latentattention32_area_398epoch/`.
- Final losses are higher than the count-weighted `gnn`: training 0.111 vs 0.095, validation (16 cases) 0.120 vs 0.109.

| Held-out metric (40 cases) | GNN | GNN, area | Cases improved | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: |
| Mean volume pressure MSE | 0.0561 | 0.0543 | 20/40 | 0.65 |
| Median volume pressure MSE | 0.00986 | 0.01255 | | |
| Mean surface pressure MSE | 0.1448 | 0.0608 | 15/40 | 0.35 |
| Median surface pressure MSE | 0.00635 | 0.01335 | | |
| Mean lift relative error | 0.855 | 0.646 | 16/40 | 0.25 |
| Median lift relative error | 0.124 | 0.190 | | |
| Lift Spearman | 0.947 | 0.892 | | |

- **Means improve, typical cases worsen.** The mean gains come from a few hard cases. On the 14.684-degree outlier, surface MSE drops 3.41 -> 0.87 and volume MSE 0.87 -> 0.46. On 85.752_6.341, surface MSE drops 0.616 -> 0.030. Excluding the outlier, mean volume MSE worsens (0.0353 -> 0.0441) while mean surface MSE still improves (0.0611 -> 0.0401, mostly from the 85.752 case). Geometric-mean per-case ratios (area / count-weighted) are 1.05 for volume, 1.20 for surface and 1.23 for lift error. Low-angle cases (AoA < 5 degrees, n = 18) are hurt most: volume 1.16, surface 1.34; at AoA >= 5 degrees (n = 22) the ratios are 0.96 and 1.10. None of the paired differences is significant.
- **Visual evidence (consistent with the physics hypothesis, not proof).** On the outlier, the count-weighted GNN has a broad, smooth suction-side offset extending far into the field. This is a global, low-frequency error, the kind an area-integral read should fix, and the area model largely removes it. The leading-edge suction peak remains underpredicted by both. Image: [outlier](volume_results/airfrans_reduced_20261122/gnn_area_analysis_20261124/airFoil2D_SST_90.415_14.684_4.356_6.209_11.299_gnn_vs_area.png). Others on the same shared scales: [85.752 gain](volume_results/airfrans_reduced_20261122/gnn_area_analysis_20261124/airFoil2D_SST_85.752_6.341_0.382_6.715_8.448_gnn_vs_area.png), [52.518 volume regression](volume_results/airfrans_reduced_20261122/gnn_area_analysis_20261124/airFoil2D_SST_52.518_12.114_1.135_6.268_0.0_5.676_gnn_vs_area.png), [56.39 lift regression](volume_results/airfrans_reduced_20261122/gnn_area_analysis_20261124/airFoil2D_SST_56.39_-3.195_4.964_2.726_18.605_gnn_vs_area.png).
- **Hypothesis for the typical-case regression:** surface nodes carry about 1e-7 of the area, and area-weighted neighbour aggregation also down-weights fine near-wall neighbours relative to coarse ones. Both changes take near-wall detail away from local messages, which matters most for the attached low-angle flows. Separating read from aggregation (`gnn_area_read`) would test this.
- Decision: no promotion. Next: run `gnn_area_read` alone (separates read from aggregation), then a second seed for `gnn` and the better area variant before drawing conclusions.

## 2026-11-24 - AirfRANS `gnn_area_read`: Clear Regression

- Run: area-weighted read only (2/4 heads); aggregation and graph mean are count-based. Kaggle T4, torch 2.11.0+cu130, 398 epochs, seed 2026, final-epoch checkpoint, same split. Artifacts: `volume_results/airfrans_reduced_20261122/airfrans_gnn_area_read_398epoch/`. Final training/validation loss 0.110/0.124, against 0.095/0.109 for `gnn`.

| Held-out metric (40 cases) | GNN | Area read only | Area read + aggregation |
| --- | ---: | ---: | ---: |
| Mean volume pressure MSE | 0.0561 | 0.0774 | 0.0543 |
| Median volume pressure MSE | 0.0099 | 0.0134 | 0.0126 |
| Mean surface pressure MSE | 0.1448 | 0.8420 | 0.0608 |
| Median surface pressure MSE | 0.0063 | 0.0103 | 0.0134 |
| Mean lift relative error | 0.855 | 0.654 | 0.646 |
| Lift Spearman | 0.947 | 0.923 | 0.892 |
| Cases better than GNN, volume / surface | | 14 / 7 | 20 / 15 |
| Per-case geometric-mean MSE ratio vs GNN, volume / surface | | 1.31 / 1.51 | 1.05 / 1.20 |
| Paired Wilcoxon p, volume / surface | | 0.0007 / 0.0002 | 0.65 / 0.35 |

- Area read only is significantly worse than the count-weighted GNN on both pressure metrics, at low and high angles of attack alike (volume ratio 1.30 / 1.32). On the 14.684-degree outlier its surface MSE is 29.9, against 3.41 for `gnn` and 0.87 for `gnn_area`. Mean lift error is lower but the median is worse (0.124 -> 0.204); per case the ratio is about 1.0.
- Visual check of the outlier on shared scales ([image](volume_results/airfrans_reduced_20261122/gnn_area_analysis_20261124/airFoil2D_SST_90.415_14.684_4.356_6.209_11.299_three_gnn.png)): read-only reduces part of the broad suction-side offset but has large leading-edge errors. Read + aggregation shows the smallest far-field error.
- Correction to the previous entry's interpretation: the area-weighted read is not what helped the outlier in `gnn_area`. Within this pair of runs, the gain must come from area-weighted aggregation and/or the area-weighted graph mean, or their combination with the read. The earlier explanation (the integral read fixes global offsets) is not supported. The earlier hypothesis that aggregation hurts typical cases is also contradicted: read-only is worse on typical cases than read + aggregation.
- Private-data context: the 400-epoch private runs had suggested the reverse (read-only best mean RMSE). Single-seed results disagree across datasets, so neither ordering is established.
- Decision: drop the area-weighted read as a standalone option on AirfRANS; no promotion. If area weighting is pursued, test aggregation + graph mean without the read (a variant not yet implemented), with at least two seeds against `gnn`.

## 2026-11-24 - Log-Area Weighting Removed; Continuous Inputs, Mean Aggregation and a 2x Refinement Test

### Decisions

- Raw log-area weighting was removed from the code completely: `area_heads`, `area_aggregation`, the `gnn_area*` AirfRANS kinds and nodal dual areas. The entries above are its record. Its checkpoints remain on disk but no longer load with the current model code. The original count-weighted checkpoints still load; verified for the AirfRANS `gnn` checkpoint and the private GPU geometry checkpoint.
- New option `aggregation="mean"`: neighbour messages are averaged (sum / count) instead of summed, a self-normalising pooling independent of node degree and sampling density. The latent read (softmax) and the graph mean were already self-normalising. Default stays `sum`; parameter count is unchanged.
- New private-data variant `continuous`: node inputs are [x, y, wall distance] in metres; raw cell area is dropped from node inputs. Design inputs are unchanged. Edge inputs are [unit direction, log(distance / L_ref)] with L_ref the body length, so edges carry no cell measures. Near-wall loss and metrics keep the 0.05 m band, so the run is comparable with the geometry baseline. Caveat recorded before testing: log(distance / L_ref) is scale-invariant but not refinement-invariant (it shifts by -log 2 under 2x refinement).
- AirfRANS kinds: `gnn_mean` (mean aggregation), `gnn_rel` ([direction, log distance] edges; coordinates are in chords) and `gnn_mean_rel`. `benchmarks.airfrans.evaluate --points N` re-evaluates a trained run at another sampling density (output `evaluation_N/`).
- New `refine_volume.py`: each polygon cell is split into one quad per vertex (vertex, edge midpoints, vertex-mean centre), i.e. 2x refinement for quads. Children tile their parents (checked against loader areas). Adjacency comes from siblings and from shared half-faces across parent faces. Wall distance is the exact distance to the WALL segments. No CFD exists on the refined mesh, so child predictions are area-averaged onto parents and compared with the CFD and with the original-mesh prediction ("consistency").
- Wall-distance control: on Design_1 the exported STAR-CCM+ wall distance and the geometric distance differ by median 6.8e-5 m (max 9.5e-4 m; near-wall median 1.9% relative). Predicting on the original mesh with geometric wall distance changes mean RMSE by at most 0.002 Pa (geometry 2.412 -> 2.414, continuous 1.417 -> 1.417). The refinement differences below are therefore not a wall-distance artefact.
- Verification: 76 tests pass (mean-aggregation degree invariance and checkpoints, continuous features, refinement tiling and adjacency, new AirfRANS kinds); Ruff and ty pass.

### Private data: stratified fold 2, seed 2026, CUDA, T_max 1,000 with early stopping (same protocol as the 2.41 Pa GPU baseline)

Command: `uv run python -u compare_volume_architectures.py --design Design_1 --stratify-inner --variant continuous --architectures latent_attention32 latent_attention32_mean --max-epochs 1000 --minimum-epochs 100 --validation-interval 10 --patience 8 --results-dir volume_results/continuous_20261124 --comparison-dir volume_results/gpu_latent_attention_geometry_20261122/latent_attention32 --comparison-label "Latent attention, geometry inputs (GPU)"`.

| Model (latent attention, width 32) | Params | Best epoch | Inner val. loss | Mean RMSE [Pa] | Worst [Pa] | Near-wall 0.05 m [Pa] | Pooled R-squared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Geometry inputs, sum (2026-11-22 GPU) | 92,097 | 570 | 0.00394 | 2.412 | 3.286 | 6.843 | 0.9730 |
| Continuous inputs, sum | 91,937 | 550 | 0.00296 | **1.417** | **2.595** | **4.019** | 0.9899 |
| Continuous inputs, mean | 91,937 | 490 | 0.00200 | 1.914 | 3.536 | 4.895 | 0.9798 |

- Continuous inputs with sum aggregation are the best GNN so far: whole-domain RMSE -41% and near-wall RMSE -41% against the geometry baseline. Mean RMSE is better on all six outer designs (e.g. Design_14 3.29 -> 1.57 Pa; smallest gain Design_26 2.98 -> 2.60 Pa). It approaches the GPU Transolver (1.11 / 2.02 Pa) but does not match it. The change combines dropping cell area from node inputs, dropping z and replacing the edge features, so it does not isolate which of these mattered.
- Mean aggregation reached the lowest inner-validation loss but is worse on the outer designs than sum (1.91 vs 1.42 Pa; worst design Design_14 at 3.54 Pa). Inner-validation and outer rankings disagree, which is consistent with the known single-seed variance.
- Visual: [Design_14 comparison](volume_results/continuous_20261124/latent_attention32/Design_14_comparison.png). The continuous model removes most of the broad upstream offset seen in the geometry model and has a sharper stagnation region. Other designs are in `volume_results/continuous_20261124/{latent_attention32,latent_attention32_mean}/`.

### 2x refinement test (held-out outer designs; no retraining)

Command: `uv run python -u refine_volume.py --model NAME CHECKPOINT VARIANT --output volume_results/refinement_20261124`.

| Model | Original mesh RMSE / near-wall [Pa] | Refined mesh (parent-averaged) RMSE / near-wall [Pa] | Consistency RMSE [Pa] |
| --- | ---: | ---: | ---: |
| Geometry, sum | 2.412 / 6.843 | 2.428 / 8.795 | 2.133 |
| Continuous, sum | 1.417 / 4.019 | 1.973 / 8.197 | 2.061 |
| Continuous, mean | 1.914 / 4.895 | 3.059 / 7.715 | 2.670 |

- The expectation that local message passing plus latent tokens is strongly resolution-stable is **not supported**. For every model, refined-minus-original prediction differences (2.1-2.7 Pa RMSE) are as large as the model's own error against CFD. Near-wall error roughly doubles for the continuous models (4.0 -> 8.2 Pa). The geometry model's whole-domain RMSE is barely changed, but its near-wall error also rises (6.8 -> 8.8 Pa).
- Pattern ([continuous Design_3](volume_results/refinement_20261124/continuous_sum/Design_3_refinement.png)): a smooth +5-10 Pa offset upstream, a -5 Pa offset downstream, and suction peaks deepened by about 20-25 Pa at the body shoulders. The refined field remains smooth and plausible but is shifted.
- Interpretation (hypotheses, not tested): uniform splitting preserves cell-count proportions, so count-weighted token reads and graph means should change little. The likely causes are local. (1) Edge inputs shift with spacing: log(distance / L_ref) by -0.69 in the continuous model; cell area falls 4x in the geometry model's node inputs. (2) Three message-passing hops cover half the physical distance on the refined mesh. (3) Children near the wall are thinner than any cell seen in training. Mean aggregation does not help here, because interior quad degree is about 4 on both meshes.
- Limitations: one seed, one outer fold of previously inspected designs, a synthetic refinement (vertex-mean centres, hanging-node-free quads, no refined CFD), and no retraining on mixed resolutions.

### AirfRANS sampling-density check of the existing `gnn` (sum) checkpoint, inference only, local laptop GPU

| Nodes per inference pass | Volume pressure MSE | Surface pressure MSE | Mean lift rel. error | Lift Spearman |
| --- | ---: | ---: | ---: | ---: |
| 16,000 | 0.0553 | 0.1404 | 0.882 | 0.947 |
| 32,000 (training density; Kaggle evaluation) | 0.0561 | 0.1448 | 0.855 | 0.947 |
| 64,000 | 0.0618 | 0.1629 | 0.817 | 0.944 |

- The sum-aggregation GNN is fairly stable to halving the sampling density but degrades by about 10% at double density. A plausible cause is that more nodes keep their mesh neighbours, so summed messages grow (degree shift). `gnn_mean` is designed to remove this. Artifacts: `evaluation_16000/` and `evaluation_64000/` in `volume_results/airfrans_reduced_20261122/kaggle_gnn_latentattention32_398epoch/`.

### Next

- Kaggle: `gnn_mean` and `gnn_mean_rel`, evaluated at 16k / 32k / 64k (notebook cell 5).
- Private: test refinement robustness directly, e.g. edge inputs that are refinement-invariant (distance / local cell size) combined with continuous nodes, or training on mixed native and refined meshes. Repeat seeds before promoting the continuous variant.