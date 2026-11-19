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