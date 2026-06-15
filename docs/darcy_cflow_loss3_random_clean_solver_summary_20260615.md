# Darcy cflow loss3 / random-clean / random-solver statistical summary

Date: 2026-06-15.

## Important Correction

The attack and SVD/Jacobian tables referenced here are preflight/smoke
diagnostics from `outputs/darcy_sir20_timematched_full_20260614_smoke_initial`.
Here `1ep` = one epoch: the checkpoints use path pattern
`darcy_sir20_smoke_*_1ep`, and the attack table uses `attack_steps=1`. They are
not final time-matched robustness results. Only the clean 52-dataset tables
should be used for final clean predictive/generalization conclusions.

For final clean generalization, `loss3` is better than `physics` on all 50
generalization datasets for Relative L2, RMSE, and data MSE. See
`docs/darcy_cflow_smoke_attack_metric_correction_20260615.md`.

## Evidence

Observed from the organized release tables:

- `outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_clean_52dataset_metric_long_ranked.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_attack_metric_long_ranked.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_svd_jacobian_metric_long_ranked.csv`
- generated summary folder:
  `outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_random_clean_solver_stat_summary_20260615/`
- generated report:
  `outputs/darcy_cflow_timematched_organized_release_20260614/reports/LOSS3_RANDOM_STAT_SUMMARY_20260615.md`

## Main Finding

The complete clean 52-dataset evaluation strongly supports a broad `loss3`
advantage. On the 50 generalization datasets:

| metric | loss3 mean | random clean mean | random solver mean | loss3 first-place count |
|---|---:|---:|---:|---:|
| Relative L2 | 0.056154 | 0.082881 | 0.100569 | 47/50 |
| RMSE | 0.0005946 | 0.0008840 | 0.0010673 | 47/50 |
| Accuracy score | 94.698 | 92.393 | 90.900 | 47/50 |

Pairwise, `loss3` beats `random_clean` on 48/50 generalization datasets for
Relative L2 and RMSE, and beats `random_solver` on 50/50. It also beats
`physics` on 50/50 clean generalization datasets for Relative L2 and RMSE.

## Attack Robustness

The available attack table is marked as partial smoke coverage
(`2` samples per dataset), so it is evidence but not the requested full
50-sample-per-dataset attack sweep.

On the 100 generalization attack samples in this table:

| metric | best aggregate model | loss3 mean | random clean mean | random solver mean |
|---|---|---:|---:|---:|
| loss increase | physics | 4.03e-08 | 6.48e-08 | 1.84e-07 |
| relative increase | loss3 | 0.2268 | 0.2836 | 0.2984 |
| adversarial loss | physics | 2.29e-07 | 3.26e-07 | 8.79e-07 |

Interpretation:

- `loss3` is not the global winner on all aggregate attack metrics because
  `physics` has the smallest mean adversarial loss and mean absolute loss
  increase.
- `loss3` is still clearly stronger than `random_clean`, `random_solver`,
  `loss1`, `loss2`, and `baseline` on the available attack table.
- The extra 15 curated heatmaps are a selected `loss3 advantage` subset, not the
  global attack average.

## Random Clean vs Random Solver

Observed from clean generalization Relative L2, `random_clean` beats
`random_solver` on every family and every one of the 50 generalization datasets.

| family | n | random clean mean Rel L2 | random solver mean Rel L2 |
|---|---:|---:|---:|
| bandpass_grf | 6 | 0.07318 | 0.09083 |
| blocky_tiles | 6 | 0.09258 | 0.10861 |
| cellular_blobs | 6 | 0.10008 | 0.11064 |
| highpass_grf | 6 | 0.04945 | 0.08275 |
| matern_fine | 7 | 0.06866 | 0.08934 |
| matern_smooth | 7 | 0.08742 | 0.09153 |
| rectangles | 6 | 0.09307 | 0.11119 |
| wave_mix | 6 | 0.10022 | 0.12305 |

On the available generalization attack table:

- `random_clean` beats `random_solver` on adversarial loss in 100/100 samples.
- `random_clean` beats `random_solver` on absolute loss increase in 100/100
  samples.
- `random_clean` beats `random_solver` on relative increase in 55/100 samples.

## Mechanistic Interpretation

Observed result: `random_clean` outperforming `random_solver` is not a plotting
artifact and not isolated to one dataset family.

Inference from the training setup: `random_clean` likely behaves as input-space
augmentation with stable clean labels. It exposes the model to randomized
coefficients while keeping the target distribution anchored to the original clean
data. `random_solver` recomputes solver targets for random coefficients, which is
more physically consistent for the perturbed input, but it also changes the
target distribution and may emphasize off-manifold solver responses. In this
Darcy setting, that extra solver-target variability appears to hurt clean
generalization more than it helps robustness.

This is an inference from observed metrics and the training objective. It has
not been isolated by a separate ablation.

## Coverage Caveats

- Clean metrics are complete for train/test/50 generalization datasets in the
  organized release.
- Attack metrics in the organized release are partial smoke coverage, not the
  full requested 50-sample-per-dataset sweep.
- The visible SVD/Jacobian long table covers train/test entries in this release,
  so it should be used as mechanism diagnostics rather than a complete
  generalization conclusion.
