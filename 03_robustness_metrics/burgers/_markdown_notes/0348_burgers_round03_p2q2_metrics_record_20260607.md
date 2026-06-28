# Burgers Round03 P2Q2 Metrics Record - 2026-06-07

## Status

Created at `2026-06-07T21:35:23Z` to consolidate the p=2, q=2 before/after perturbation metrics discussed in chat into a single Markdown record.
No new training and no new attack optimization were run for this record; values below are recomputed from existing saved traces and summaries.

## Source Files

- Main summary: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/summary.json`
- Sample manifest: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/sample_manifest.json`
- loss1 trace: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss1/attack_traces.npz`
- loss1 CSV: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss1/attack_loss_curves.csv`
- loss2 trace: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss2/attack_traces.npz`
- loss2 CSV: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss2/attack_loss_curves.csv`
- loss3 trace: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss3/attack_traces.npz`
- loss3 CSV: `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss3/attack_loss_curves.csv`
- Combined plotting script: `tools/plot_burgers_round03_p2q2_combined_attack_panels.py`
- Image-only bundle: `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607`

## Attack And Environment Settings

| Field | Value |
|---|---|
| Device | `cuda` |
| GPU | `Tesla V100-SXM2-32GB` |
| PyTorch | `2.8.0+cu126` |
| CUDA runtime | `12.6` |
| epsilon_rms | `0.12` |
| alpha_rms | `0.012` |
| attack_steps | `100` |
| frame_every | `2` |

## Model Sources

| Model | Checkpoint | Trace source |
|---|---|---|
| baseline | `/workspace/NeuralOperatorRobustness2/1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt` | shared baseline trace from `loss1/attack_traces.npz`, verified against loss2/loss3 baseline traces |
| loss1 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt` | `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss1/attack_traces.npz` |
| loss2 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt` | `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss2/attack_traces.npz` |
| loss3 | `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt` | `forensics/burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607/loss3/attack_traces.npz` |

## Sample Sources

| Sample | Split | Dataset | Index | Source path |
|---:|---|---|---:|---|
| S1 | `test` | `test_original_gaussian_corr0p03` | 23 | `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt` |
| S2 | `generalization` | `burgers_near_gaussian_corr0p1` | 17 | `generalization_datasets_rmse_1p5_3x_all_ns50/burgers/burgers_near_gaussian_corr0p1.pt` |
| S3 | `generalization` | `burgers_target_gaussian_corr0p2` | 42 | `generalization_datasets_rmse_1p5_3x_all_ns50/burgers/burgers_target_gaussian_corr0p2.pt` |
| S4 | `generalization` | `burgers_target_matern_corr0p6_nu3` | 73 | `generalization_datasets_rmse_1p5_3x_all_ns50/burgers/burgers_target_matern_corr0p6_nu3.pt` |
| S5 | `generalization` | `burgers_mid_matern_corr1_nu4` | 29 | `generalization_datasets_rmse_1p5_3x_all_ns50/burgers/burgers_mid_matern_corr1_nu4.pt` |
| S6 | `generalization` | `burgers_far_sawtooth_add_scale0p3_shift0` | 11 | `generalization_datasets_rmse_1p5_3x_all_ns50/burgers/burgers_far_sawtooth_add_scale0p3_shift0.pt` |

## Metric Definitions

Observed quantities use the pointwise error vector `e = model - solver`.

- Per-sample `MSE = mean(e^2)`.
- Per-sample `RMS = sqrt(mean(e^2))`.
- `step100 mean MSE` is `mean_i(MSE_i)` over the six samples.
- `mean model-solver RMS` is `mean_i(sqrt(MSE_i))` over the six samples.
- Because square root is nonlinear, `mean_i(sqrt(MSE_i))` is not equal to `sqrt(mean_i(MSE_i))`.
- `MSE` emphasizes large spikes more strongly; `RMS` has the same unit as the output curve and is closer to visual oscillation amplitude.

## Step 0 Before Perturbation

| Model | Mean MSE | Mean model-solver RMS | Global max abs diff | Mean delta RMS |
|---|---:|---:|---:|---:|
| baseline | 0.0002251479 | 0.0134999231 | 0.3224229813 | 0 |
| loss1 | 7.5847369772e-06 | 0.0022291869 | 0.0617683977 | 0 |
| loss2 | 8.8161114036e-06 | 0.0026218866 | 0.0741456747 | 0 |
| loss3 | 3.5604502045e-05 | 0.0055780425 | 0.1446411014 | 0 |

Observed step0 ranking by mean MSE: loss1 best, loss2 second, loss3 third, baseline worst.

## Step 100 After Perturbation

| Model | Mean MSE | Mean model-solver RMS | Global max abs diff | Mean delta RMS | Ratio vs baseline MSE | Ratio vs loss1 MSE | Ratio vs loss2 MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0265074968 | 0.1535285264 | 1.1547194719 | 0.1199999973 | 1 | 4.211615952 | 2.8600235597 |
| loss1 | 0.0062939017 | 0.070637323 | 1.1724832058 | 0.1199999973 | 0.2374385536 | 1 | 0.6790798573 |
| loss2 | 0.0092682792 | 0.0884283185 | 1.32924366 | 0.1200000048 | 0.3496474694 | 1.4725808596 | 1 |
| loss3 | 0.0029296223 | 0.0525338911 | 0.8558747768 | 0.1200000048 | 0.110520519 | 0.4654699808 | 0.3160912882 |

Observed step100 mean MSE ordering: loss3 best, loss1 second, loss2 third, baseline worst.
Observed step100 mean model-solver RMS ordering: loss3 best, loss1 second, loss2 third, baseline worst.

## Per-Sample Step100 MSE

| Sample | Split | Dataset | Baseline | Loss1 | Loss2 | Loss3 | Loss3/Baseline | Loss3/Loss1 | Loss3/Loss2 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | `test` | `test_original_gaussian_corr0p03` | 0.0015216579 | 0.0008656711 | 0.0012630873 | 0.001285064 | 0.8445157204 | 1.4844713769 | 1.0173992476 |
| S2 | `generalization` | `burgers_near_gaussian_corr0p1` | 0.0199295618 | 0.0011172956 | 0.008796053 | 0.0019985959 | 0.1002829825 | 1.7887799068 | 0.2272150806 |
| S3 | `generalization` | `burgers_target_gaussian_corr0p2` | 0.0376577862 | 0.0148793422 | 0.0227846354 | 0.0030307998 | 0.0804826864 | 0.2036917869 | 0.1330194554 |
| S4 | `generalization` | `burgers_target_matern_corr0p6_nu3` | 0.0316044874 | 0.0023961016 | 0.0024834401 | 0.0019656883 | 0.0621964938 | 0.8203693494 | 0.7915183099 |
| S5 | `generalization` | `burgers_mid_matern_corr1_nu4` | 0.0377890579 | 0.0120920241 | 0.0120940534 | 0.0034781636 | 0.0920415541 | 0.2876411422 | 0.2875928768 |
| S6 | `generalization` | `burgers_far_sawtooth_add_scale0p3_shift0` | 0.0305424295 | 0.006412975 | 0.0081884079 | 0.0058194231 | 0.1905356978 | 0.9074451564 | 0.7106904319 |

## Per-Sample Step100 Ranking

| Sample | Ranking, lower MSE is better |
|---:|---|
| S1 | loss1 (0.0008656711) < loss2 (0.0012630873) < loss3 (0.001285064) < baseline (0.0015216579) |
| S2 | loss1 (0.0011172956) < loss3 (0.0019985959) < loss2 (0.008796053) < baseline (0.0199295618) |
| S3 | loss3 (0.0030307998) < loss1 (0.0148793422) < loss2 (0.0227846354) < baseline (0.0376577862) |
| S4 | loss3 (0.0019656883) < loss1 (0.0023961016) < loss2 (0.0024834401) < baseline (0.0316044874) |
| S5 | loss3 (0.0034781636) < loss1 (0.0120920241) < loss2 (0.0120940534) < baseline (0.0377890579) |
| S6 | loss3 (0.0058194231) < loss1 (0.006412975) < loss2 (0.0081884079) < baseline (0.0305424295) |

## Loss3 Versus Baseline Step100 Amplitude Details

| Sample | Baseline diff RMS | Loss3 diff RMS | Loss3/Baseline RMS | Baseline max abs diff | Loss3 max abs diff | Loss3/Baseline max abs |
|---:|---:|---:|---:|---:|---:|---:|
| S1 | 0.0390084349 | 0.0358477905 | 0.9189753618 | 0.3100869954 | 0.3558213413 | 1.1474887583 |
| S2 | 0.141172111 | 0.0447056592 | 0.3166748646 | 0.5180555582 | 0.4093650579 | 0.7901952821 |
| S3 | 0.1940561384 | 0.0550526977 | 0.2836946984 | 1.1547194719 | 0.8243585825 | 0.7139037684 |
| S4 | 0.1777765006 | 0.0443360843 | 0.2493922657 | 0.931992054 | 0.6191914678 | 0.6643741919 |
| S5 | 0.1943940818 | 0.0589759573 | 0.3033835021 | 1.1412867308 | 0.8558747768 | 0.749920904 |
| S6 | 0.1747639179 | 0.0762851462 | 0.4365039825 | 0.7418056726 | 0.5802185535 | 0.7821705535 |

## Step0 Per-Sample Ranking

| Sample | Ranking, lower MSE is better |
|---:|---|
| S1 | loss1 (6.3427279429e-07) < loss2 (1.8351767039e-06) < loss3 (1.0902240319e-05) < baseline (3.0033435905e-05) |
| S2 | loss1 (1.1188972167e-06) < loss2 (1.8734363039e-06) < loss3 (8.4002658696e-06) < baseline (8.0644182162e-05) |
| S3 | loss1 (4.3265458771e-06) < loss2 (1.9833812985e-05) < loss3 (5.9536836488e-05) < baseline (0.0001514929) |
| S4 | loss2 (1.0805983948e-06) < loss1 (1.5218040517e-06) < loss3 (4.9559323088e-05) < baseline (0.0001018981) |
| S5 | loss1 (7.0215569394e-06) < loss2 (1.3113822206e-05) < loss3 (6.5939435444e-05) < baseline (0.0003817074) |
| S6 | loss2 (1.5159822397e-05) < loss3 (1.9288920157e-05) < loss1 (3.0885345041e-05) < baseline (0.0006051113) |

## Plot Outputs And Bundle Hygiene

| File | Dimensions | Size bytes | RGB stddev |
|---|---:|---:|---:|
| `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_loss1_loss2_loss3_step000_before_perturbation_four_column.png` | `7200x2528` | 936109 | `(24.01, 23.09, 23.64)` |
| `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_loss1_loss2_loss3_step100_after_perturbation_four_column.png` | `7200x2528` | 1382635 | `(26.70, 25.62, 26.41)` |
| `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_loss1_loss2_loss3_before_after_overlay_four_column.png` | `7200x2528` | 1514216 | `(26.52, 25.40, 26.23)` |
| `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss1_before_after_overlay_two_column.png` | `3600x2528` | 886335 | `(30.96, 25.78, 25.06)` |
| `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss2_before_after_overlay_two_column.png` | `3600x2528` | 880707 | `(27.21, 26.09, 27.27)` |
| `visualizations/burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607/comparison_dense/round03_p2q2_baseline_vs_loss3_before_after_overlay_two_column.png` | `3600x2528` | 856341 | `(27.22, 27.38, 25.72)` |

Observed bundle PNG count: `59`.
Observed bundle non-PNG count: `0`.

## Overlay Alpha Settings

Observed pre-correction overlay issue: before-perturbation main curves used alpha values `0.26`, `0.28`, and `0.30`, while after curves used approximately `0.92..0.94`; before curves were too faint.

Current correction: before-perturbation dashed main curves use `alpha=0.80`; low-alpha fill bands remain light, such as `alpha=0.045`, so they do not cover the after-step100 solid curves.

Current before-line script evidence:

- `ax.plot(grid, tr["delta"][before_idx, row], color=color, lw=1.05, alpha=0.80, ls="--", label="before")`
- `ax.plot(grid, before_x, color=color, lw=1.05, alpha=0.80, ls="--", label="before adv")`
- `ax.plot(grid, before_solver, color=SOLVER_COLOR, lw=1.05, alpha=0.80, ls="--", label="solver before")`
- `ax.plot(grid, before_model, color=color, lw=1.05, alpha=0.80, ls="--", label="model before")`
- `ax.plot(grid, before_diff, color=color, lw=1.05, alpha=0.80, ls="--", label="before")`

## Inference

- Before perturbation, loss1/loss2 have lower clean model-solver MSE than loss3 on this six-sample p2q2 probe.
- After 100 p2q2 attack steps under the same `delta_rms≈0.12` budget, loss3 has the best overall attacked-output stability by mean MSE and mean model-solver RMS.
- Loss3 wins S3-S6 by final MSE; loss1 wins S1-S2; baseline is worst on all six final attacked samples.
- The visual statement should therefore be phrased as an all-model robustness comparison, not only as loss3 versus baseline.

