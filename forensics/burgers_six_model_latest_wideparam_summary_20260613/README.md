# Burgers Six-Model Latest Wideparam Summary, 2026-06-13

This report records the latest available six-model results for the wide-parameter
loss3-targeted Burgers generalization set:

- `baseline`
- `loss1`
- `loss2`
- `loss3`
- `random_clean_y`
- `random_solver_y`

The latest 50 generalization datasets are:
`burgers_widevis_l3target_d00` through `burgers_widevis_l3target_d49`.

## Output Tables

- `clean_generalization_50dataset_six_models.csv`
- `robustness_25sample_svd_attack_six_models.csv`
- `per_sample_25sample_svd_attack_six_models.csv`

## 1. Clean Generalization on 50 Datasets

Lower is better.

| rank | model | datasets | samples | RMSE mean | MSE mean | relative L2 mean |
|---:|---|---:|---:|---:|---:|---:|
| 1 | loss3 | 50 | 10000 | 0.012050 | 0.0001978 | 0.023644 |
| 2 | random_solver_y | 50 | 10000 | 0.019972 | 0.0005776 | 0.038941 |
| 3 | loss1 | 50 | 10000 | 0.021027 | 0.0005822 | 0.041171 |
| 4 | loss2 | 50 | 10000 | 0.022292 | 0.0006477 | 0.043542 |
| 5 | baseline | 50 | 10000 | 0.029902 | 0.0012222 | 0.057737 |
| 6 | random_clean_y | 50 | 10000 | 0.088542 | 0.0090267 | 0.168778 |

Clean-loss conclusion:

- `loss3` is the best clean generalization model.
- `random_solver_y` is competitive with `loss1` and `loss2`, but does not beat
  `loss3`.
- `random_clean_y` is clearly bad on clean generalization.

## 2. Robustness/SVD 25-Sample Summary

This table uses the latest wideparam/loss3-targeted 25-sample SVD/attack set.
The old four models and the two random models use the same sample manifest:

`forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/sample_manifest.csv`

and

`forensics/burgers_random_field_final_models_full_suite_20260613/jacobian_svd/sample_manifest.csv`

The manifests match: 25 samples total, including 21 generalization samples.

Lower is better for attack loss, loss increase, `J_error` spectral norm, and
`||J_error delta||`.

| model | n | gen n | attack init MSE | attack final MSE | attack increase | delta RMS | error spectral norm | error Fro norm | error effective rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 25 | 21 | 0.001095 | 0.010259 | 0.009164 | 0.120000 | 2.367033 | 3.574622 | 10.284617 |
| loss1 | 25 | 21 | 0.000823 | 0.006661 | 0.005838 | 0.120000 | 1.702471 | 2.781139 | 18.136217 |
| loss2 | 25 | 21 | 0.000856 | 0.006479 | 0.005623 | 0.120000 | 1.855107 | 2.887809 | 11.930577 |
| loss3 | 25 | 21 | 0.000244 | 0.003307 | 0.003063 | 0.120000 | 1.271533 | 1.898531 | 6.373320 |
| random_clean_y | 25 | 21 | 0.009031 | 0.045632 | 0.036601 | 0.117188 | 3.828716 | 6.759260 | 6.943157 |
| random_solver_y | 25 | 21 | 0.000744 | 0.007667 | 0.006923 | 0.120000 | 1.735561 | 2.674979 | 7.198486 |

Robustness/SVD conclusion on this 25-sample set:

- `loss3` is best on attack final MSE and attack increase.
- `random_solver_y` is much better than `random_clean_y`.
- `random_solver_y` is close to `loss1`/`loss2` in error-Jacobian spectral norm,
  but worse than `loss3` on attack final MSE/increase.
- `random_clean_y` is the worst by attack loss increase and error-Jacobian norm.

## 3. Direction and Linearized Error Quantities

The old four-model run stores `||A^T b||` as `bias_gradient_norm`.
The random-model run stores `J_error @ delta` as `j_error_delta_l2` and
`j_error_delta_rms`.

| model | attack-delta/top-error-SV abs cos | attack-delta/outward abs cos | attack-delta/affine-eps abs cos | mean `||A^T b||` | median `||A^T b||` | mean `||J_error delta||_2` | mean `J_error delta` RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.304880 | 0.793450 | 0.340490 | 0.462951 | 0.331235 |  |  |
| loss1 | 0.209058 | 0.803056 | 0.226577 | 0.315854 | 0.173628 |  |  |
| loss2 | 0.196515 | 0.798714 | 0.208314 | 0.343340 | 0.190082 |  |  |
| loss3 | 0.091452 | 0.726458 | 0.100182 | 0.111950 | 0.080228 |  |  |
| random_clean_y | 0.399197 |  |  |  |  | 9.017980 | 0.281812 |
| random_solver_y | 0.277103 |  |  |  |  | 2.528930 | 0.079029 |

Direction conclusion:

- For the old four models, `loss3` has the smallest `||A^T b||` and the smallest
  attack-delta/top-error-singular-vector alignment.
- For the random models, `random_solver_y` has much smaller `J_error @ delta`
  than `random_clean_y`.
- The stored old-four direction metric and the stored random-model direction
  metric are related local-linear diagnostics, but they are not the exact same
  scalar: old four stores `J_error^T error`-style `||A^T b||`; random stores
  `J_error @ attack_delta`.

## 4. Singular Values and Singular Vectors

The scalar summaries above include:

- `error_spectral_norm`: top singular value of `J_error`.
- `error_fro_norm`: Frobenius norm of `J_error`.
- `error_effective_rank`: effective rank of the error Jacobian.
- `attack-delta/top-error-SV abs cos`: alignment between the attack delta and
  the top right singular vector of `J_error`.

The full singular vectors and singular values are stored in the per-sample NPZ
files:

- Old four models:
  `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/sample_*/`
- Random models:
  `forensics/burgers_random_field_final_models_full_suite_20260613/jacobian_svd/sample_*/`

The compact random singular-value table is:

`forensics/burgers_random_field_final_models_full_suite_20260613/jacobian_svd/jacobian_svd_summary.csv`

The old-four compact direction/SVD table is:

`forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611/biased_direction_metrics.csv`

## 5. Delta and Attack Trace Files

Old four models:

- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/attack_traces/baseline_attack_trace.npz`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/attack_traces/loss1_attack_trace.npz`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/attack_traces/loss2_attack_trace.npz`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/attack_traces/loss3_attack_trace.npz`

Random models:

- `forensics/burgers_random_field_final_models_full_suite_20260613/p2q2_attack/random_clean_y/final_delta_by_sample.npz`
- `forensics/burgers_random_field_final_models_full_suite_20260613/p2q2_attack/random_solver_y/final_delta_by_sample.npz`
- `forensics/burgers_random_field_final_models_full_suite_20260613/p2q2_attack/random_clean_y/losses_and_delta_rms_by_sample.npz`
- `forensics/burgers_random_field_final_models_full_suite_20260613/p2q2_attack/random_solver_y/losses_and_delta_rms_by_sample.npz`

## 6. Source Files Used

- `forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611/per_dataset_clean_metrics.csv`
- `forensics/burgers_random_field_final_models_full_suite_20260613/clean_loss/per_dataset_clean_metrics.csv`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_biased_local_direction_20260611/biased_direction_metrics.csv`
- `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/attack_step_metrics.csv`
- `forensics/burgers_random_field_final_models_full_suite_20260613/jacobian_svd/jacobian_svd_summary.csv`
- `forensics/burgers_random_field_final_models_full_suite_20260613/jacobian_svd/j_error_times_attack_delta.csv`
- `forensics/burgers_random_field_final_models_full_suite_20260613/p2q2_attack/*/losses_and_delta_rms_by_sample.npz`

## 7. Random-Model Metric Similarity

For the two random models, attack loss and delta magnitude are available on the
full 52-dataset attack suite: 10,200 attacked samples. Jacobian/SVD direction
quantities are available on the 25-sample SVD subset.

### Full 10,200-Sample Delta Magnitude vs Loss Increase

| model | scope | corr(delta RMS, loss increase) Pearson | Spearman | mean delta RMS | mean loss increase |
|---|---|---:|---:|---:|---:|
| random_clean_y | all | 0.441166 | 0.336513 | 0.116832 | 0.034855 |
| random_clean_y | generalization | 0.442088 | 0.336626 | 0.116876 | 0.034781 |
| random_solver_y | all | 0.002824 | -0.005465 | 0.119997 | 0.007641 |
| random_solver_y | generalization | 0.003037 | 0.000149 | 0.119997 | 0.007761 |

Interpretation:

- For `random_clean_y`, larger final delta magnitude has a moderate relationship
  with loss increase.
- For `random_solver_y`, final delta RMS is almost fixed at the attack radius, so
  delta magnitude by itself explains almost none of the loss increase.

### Full 10,200-Sample Cross-Model Similarity

Same input sample, compare `random_clean_y` attack delta with `random_solver_y`
attack delta.

| scope | n | mean cosine | mean abs cosine | mean angle | mean abs angle |
|---|---:|---:|---:|---:|---:|
| all | 10200 | 0.105515 | 0.154642 | 83.787 deg | 80.958 deg |
| generalization | 10000 | 0.106857 | 0.155469 | 83.708 deg | 80.908 deg |
| test | 150 | 0.050973 | 0.115755 | 87.040 deg | 83.314 deg |
| train | 50 | 0.000678 | 0.106066 | 89.954 deg | 83.893 deg |

Interpretation:

- The two random-model attacks choose very different delta directions.
- Even though they attack the same samples, their delta directions are nearly
  orthogonal on average.

Loss-increase similarity between the two random models:

| scope | n | Pearson | Spearman |
|---|---:|---:|---:|
| all | 10200 | 0.468520 | 0.549069 |
| generalization | 10000 | 0.478459 | 0.576990 |
| test | 150 | 0.184692 | 0.062806 |
| train | 50 | -0.065761 | -0.009844 |

Interpretation:

- Across the 50 generalization datasets, samples that are worse for
  `random_clean_y` tend also to be worse for `random_solver_y`, but only
  moderately.
- The attack directions are much less similar than the scalar loss increases.

### 25-Sample Jacobian/SVD Direction Similarity

| model | x metric | Pearson with loss increase | Spearman with loss increase | x mean |
|---|---|---:|---:|---:|
| random_clean_y | `||J_error delta||_2` | 0.533878 | 0.601739 | 9.017980 |
| random_clean_y | `J_error delta` RMS | 0.533878 | 0.601739 | 0.281812 |
| random_clean_y | top error singular value | 0.378647 | 0.434783 | 3.789070 |
| random_clean_y | abs cos(delta, top error right singular vector) | -0.037997 | -0.082609 | 0.399197 |
| random_solver_y | `||J_error delta||_2` | 0.603287 | 0.715652 | 2.528930 |
| random_solver_y | `J_error delta` RMS | 0.603287 | 0.715652 | 0.079029 |
| random_solver_y | top error singular value | 0.638577 | 0.772174 | 1.792690 |
| random_solver_y | abs cos(delta, top error right singular vector) | -0.238355 | -0.478261 | 0.277103 |

Angle between attack delta and the top right singular vector of `J_error`:

| model | mean abs cos | mean angle |
|---|---:|---:|
| random_clean_y | 0.399197 | 65.668 deg |
| random_solver_y | 0.277103 | 73.123 deg |

Interpretation:

- For the random models, the best predictor of attack loss increase among these
  local quantities is not raw delta size. It is `J_error delta`.
- `random_solver_y` has much smaller `J_error delta` than `random_clean_y`, and
  correspondingly much smaller attack loss increase.
- The attack delta is not simply aligned with the top singular vector. Alignment
  to the top singular vector alone is weak or even negatively correlated with
  loss increase in this 25-sample subset.
