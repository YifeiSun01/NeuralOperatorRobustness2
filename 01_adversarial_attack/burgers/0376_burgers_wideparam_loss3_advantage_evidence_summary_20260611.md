# Burgers Wide-Parameter Loss3 Advantage Evidence Summary - 2026-06-11

This note consolidates the current evidence that the Burgers `loss3` adversarially trained model has a broad advantage over `loss1` and `loss2` on the latest wide-parameter generalization dataset. The evidence combines clean generalization loss, finite-budget accelerated attack growth, and full `1024 x 1024` Jacobian/SVD robustness metrics.

## Bottom Line

- The current evidence strongly supports a clear and broad `loss3` advantage on the latest generalization dataset.
- Clean generalization is decisive: `loss3` is best on `50/50` datasets and reduces mean RMSE by `42.69%` versus `loss1` and `45.95%` versus `loss2`.
- Finite-budget robustness is also strong: on the unified 25-sample attack batch, `loss3` has the lowest final attacked MSE on `23/25` samples among `loss1/loss2/loss3`, with attack-growth reductions of about `47.5%` versus `loss1` and `45.5%` versus `loss2`.
- Full SVD evidence is currently complete for the reused 3 generalization samples. In those samples, `loss3` reduces `||J_model - J_solver||_2` by about `49%` versus `loss1/loss2` and has the highest model-solver singular-vector/subspace similarity.
- The caveat is important: the 25-sample full SVD run is still in progress. The 25-sample attack result is complete; the full SVD conclusion will become stronger once the remaining 22 new samples finish.

## Data Sources

- Latest dataset root: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers`.
- Clean inference summary: `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611/summary.json`.
- Clean per-dataset table: `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611/per_dataset_clean_metrics.csv`.
- 25-sample attack table: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/attack_step_metrics.csv`.
- 3-sample full SVD detailed table: `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611/detailed_spectrum_similarity_20260611/model_mean_metrics.csv`.
- 3-sample detailed SVD report: `docs/burgers_wideparam_full1024_svd_attack3_detailed_spectrum_similarity_20260611.md`.

## Clean Generalization: 50 Datasets

Dataset count: `50`. Total samples: `10000`.

| model | RMSE mean | MSE mean | relative L2 mean |
|---|---|---|---|
| baseline | 0.0299017 | 0.00122221 | 0.0577373 |
| loss1 | 0.0210265 | 0.000582192 | 0.0411709 |
| loss2 | 0.0222924 | 0.000647666 | 0.0435424 |
| loss3 | 0.0120496 | 0.000197775 | 0.0236437 |

### Loss3 Pairwise Advantage

| metric | comparison | loss3 mean | comparison mean | reduction pct | win count | paired p | Cohen dz |
|---|---|---|---|---|---|---|---|
| rmse | baseline | 0.0120496 | 0.0299017 | 59.7028 | 50/50 | 5.656e-21 | -2.2104 |
| mse | baseline | 0.000197775 | 0.00122221 | 83.8183 | 50/50 | 1.774e-09 | -1.01477 |
| relative_l2 | baseline | 0.0236437 | 0.0577373 | 59.0494 | 50/50 | 2.157e-18 | -1.90455 |
| rmse | loss1 | 0.0120496 | 0.0210265 | 42.6935 | 50/50 | 4.920e-22 | -2.34484 |
| mse | loss1 | 0.000197775 | 0.000582192 | 66.0293 | 50/50 | 8.955e-13 | -1.32215 |
| relative_l2 | loss1 | 0.0236437 | 0.0411709 | 42.5716 | 50/50 | 1.034e-17 | -1.82849 |
| rmse | loss2 | 0.0120496 | 0.0222924 | 45.9477 | 50/50 | 5.728e-24 | -2.60483 |
| mse | loss2 | 0.000197775 | 0.000647666 | 69.4635 | 50/50 | 2.161e-13 | -1.3818 |
| relative_l2 | loss2 | 0.0236437 | 0.0435424 | 45.6995 | 50/50 | 8.813e-20 | -2.06551 |

### Family-Level Clean RMSE

| family | n datasets | baseline RMSE | loss1 RMSE | loss2 RMSE | loss3 RMSE | loss3 RMSE reduction vs baseline pct | loss3 best among adv |
|---|---|---|---|---|---|---|---|
| gaussian | 12 | 0.025196 | 0.0175656 | 0.0191801 | 0.010312 | 60.0015 | 12/12 |
| matern | 10 | 0.0418597 | 0.0238712 | 0.0255952 | 0.0152266 | 63.4114 | 10/10 |
| powerlaw_fourier | 14 | 0.0289448 | 0.0211705 | 0.0222152 | 0.0127957 | 56.4249 | 14/14 |
| sawtooth | 1 | 0.0159441 | 0.0159191 | 0.0128519 | 0.00457624 | 71.2981 | 1/1 |
| sine_mixture | 12 | 0.0263261 | 0.0222694 | 0.0230909 | 0.0110846 | 57.7988 | 12/12 |
| square_wave | 1 | 0.0370513 | 0.0222884 | 0.0275533 | 0.0097359 | 73.7232 | 1/1 |

## Finite-Budget Attack Robustness: 25 Fixed Samples

Attack setting: `15` P2Q2 RMS-L2 steps, `epsilon_rms=0.12`, `alpha_rms=0.012`. The attack table has `25 x 4 x 16 = 1600` rows. All final attacks reached `delta_rms=0.12`.

| model | initial MSE | final attacked MSE | attack growth | delta RMS | boundary ratio |
|---|---|---|---|---|---|
| baseline | 0.00109457 | 0.0102588 | 0.00916425 | 0.12 | 1 |
| loss1 | 0.000823095 | 0.00666104 | 0.00583795 | 0.12 | 1 |
| loss2 | 0.000855755 | 0.00647917 | 0.00562341 | 0.12 | 1 |
| loss3 | 0.000243923 | 0.00330713 | 0.00306321 | 0.12 | 1 |

### Loss3 Attack Reduction

| metric | comparison | loss3 | comparison value | reduction pct |
|---|---|---|---|---|
| final attacked MSE | baseline | 0.00330713 | 0.0102588 | 67.763 |
| attack growth | baseline | 0.00306321 | 0.00916425 | 66.5744 |
| final attacked MSE | loss1 | 0.00330713 | 0.00666104 | 50.3511 |
| attack growth | loss1 | 0.00306321 | 0.00583795 | 47.5294 |
| final attacked MSE | loss2 | 0.00330713 | 0.00647917 | 48.9575 |
| attack growth | loss2 | 0.00306321 | 0.00562341 | 45.5276 |

### Final-Attacked-MSE Winner Count Among Trained Models

| model | wins |
|---|---|
| loss1 | 1/25 |
| loss2 | 1/25 |
| loss3 | 23/25 |

## Full 1024 x 1024 SVD Robustness: 3 Completed Generalization Samples

This is the complete full-Jacobian/full-SVD result for the reused 3 generalization samples. It computes `J_solver`, each `J_model`, and `J_error = J_model - J_solver`; no block projection, no randomized SVD, and no top-k approximation were used.

| model | error sigma1 | model sigma1 | solver sigma1 | right v1 cos | left u1 cos | right top10 mean | left top10 mean | err v1-delta cos | final attacked MSE | attack growth |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 2.01134 | 2.82197 | 3.24638 | 0.940287 | 0.840368 | 0.887814 | 0.798228 | 0.313126 | 0.0162244 | 0.0154669 |
| loss1 | 2.08314 | 3.20474 | 3.24638 | 0.986033 | 0.939893 | 0.939044 | 0.847184 | 0.154677 | 0.00778604 | 0.00703813 |
| loss2 | 2.09002 | 3.12797 | 3.24638 | 0.981972 | 0.971943 | 0.936032 | 0.847439 | 0.106322 | 0.00635045 | 0.00558603 |
| loss3 | 1.06347 | 3.2908 | 3.24638 | 0.992065 | 0.992449 | 0.967363 | 0.946596 | 0.0406373 | 0.00408682 | 0.00392136 |

### Loss3 SVD/Attack Reduction On The 3 Full-SVD Samples

| metric | comparison | loss3 | comparison value | reduction fraction |
|---|---|---|---|---|
| error_spectral_norm_sv1 | baseline | 1.06347 | 2.01134 | 0.471261 |
| error_spectral_norm_sv1 | loss1 | 1.06347 | 2.08314 | 0.489484 |
| error_spectral_norm_sv1 | loss2 | 1.06347 | 2.09002 | 0.491165 |
| attack_final_mse | baseline | 0.00408682 | 0.0162244 | 0.748107 |
| attack_final_mse | loss1 | 0.00408682 | 0.00778604 | 0.475109 |
| attack_final_mse | loss2 | 0.00408682 | 0.00635045 | 0.356452 |
| attack_growth_abs | baseline | 0.00392136 | 0.0154669 | 0.746468 |
| attack_growth_abs | loss1 | 0.00392136 | 0.00703813 | 0.44284 |
| attack_growth_abs | loss2 | 0.00392136 | 0.00558603 | 0.298005 |

## Interpretation

The robustness story is consistent across the two robustness metrics. The finite-budget attack directly measures how much the loss can grow inside the epsilon ball; on the current 25-sample batch, `loss3` has the smallest final attacked loss and smallest absolute growth by a large margin. The full SVD metric measures local linear error sensitivity through `||J_model - J_solver||_2`; on the completed 3-sample full-SVD subset, `loss3` has roughly half the error spectral norm of `loss1/loss2` and much higher singular-vector/subspace alignment with the solver.

This means `loss3` is not merely lowering clean MSE. It appears to learn a model whose local input-output Jacobian is closer to the true solver Jacobian, and that local alignment agrees with the lower adversarial attack loss growth observed in the finite-budget attack.

## Current Caveat And Next Confirmation

The 25-sample full SVD job is still running. At the time of this note, the 25-sample attack table is complete, while the full SVD table has completed the reused 3 generalization samples and is computing the additional samples. Once the 25-sample full SVD finishes, the decisive follow-up check is whether `loss3` remains lowest in `error_spectral_norm` across most or all of the 25 samples and whether the SVD-vs-attack correlation remains positive.

