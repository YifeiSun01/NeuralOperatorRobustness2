# Corrected Full Summary: Six Burgers Models on Latest Wideparam Data

Date: 2026-06-13

Models:

- `baseline`
- `loss1`
- `loss2`
- `loss3`
- `random_clean_y`
- `random_solver_y`

Data universe:

- Latest wide-parameter/loss3-targeted generalization set:
  `burgers_widevis_l3target_d00` ... `burgers_widevis_l3target_d49`.
- Clean generalization metrics below are over the 50 generalization datasets
  and 10,000 samples.
- Robustness/SVD metrics below are over the matched latest 25-sample SVD/attack
  manifest: 21 generalization samples, 2 train samples, 2 test samples.

Important definition:

- Training-time random delta is not adversarial.
- The delta discussed in robustness is the post-training adversarial attack
  delta, computed after the model checkpoint is fixed.
- Delta is a vector. Delta RMS/magnitude is mostly an attack-radius/constraint
  check. The mechanism quantities are loss increase, `J_error`, singular
  values/vectors, `J_error @ delta`, and `J_error^T error` / `A^T b`.

## 1. Clean Generalization

Lower is better.

| rank | model | RMSE | relative L2 | MSE |
|---:|---|---:|---:|---:|
| 1 | loss3 | 0.012050 | 0.023644 | 0.0001978 |
| 2 | random_solver_y | 0.019972 | 0.038941 | 0.0005776 |
| 3 | loss1 | 0.021027 | 0.041171 | 0.0005822 |
| 4 | loss2 | 0.022292 | 0.043542 | 0.0006477 |
| 5 | baseline | 0.029902 | 0.057737 | 0.0012222 |
| 6 | random_clean_y | 0.088542 | 0.168778 | 0.0090267 |

Clean conclusion:

- `loss3` is best.
- `random_solver_y` is strong and slightly better than `loss1`/`loss2` on clean
  generalization, but it does not beat `loss3`.
- `random_clean_y` fails on generalization.

## 2. Robustness Overview

The robustness part has three business indicators:

1. Post-training adversarial attack result: attack delta and attack loss
   increase.
2. Error-Jacobian/SVD size: singular values, spectral norm, Frobenius norm,
   effective rank.
3. Direction and vector mechanism: singular vectors/subspaces, similarity to the
   solver Jacobian, \(A^\top b\), and \(J_{\mathrm{error}}\delta\).

All robustness tables below use the matched latest wideparam 25-sample
SVD/attack manifest unless otherwise stated.

## 3. Robustness Indicator 1: Attack Delta and Loss Increase

Lower attack final MSE and lower attack increase are better.

| rank by increase | model | initial MSE | final attack MSE | attack increase | final delta RMS |
|---:|---|---:|---:|---:|---:|
| 1 | loss3 | 0.000244 | 0.003307 | 0.003063 | 0.120000 |
| 2 | loss2 | 0.000856 | 0.006479 | 0.005623 | 0.120000 |
| 3 | loss1 | 0.000823 | 0.006661 | 0.005838 | 0.120000 |
| 4 | random_solver_y | 0.000744 | 0.007667 | 0.006923 | 0.120000 |
| 5 | baseline | 0.001095 | 0.010259 | 0.009164 | 0.120000 |
| 6 | random_clean_y | 0.009031 | 0.045632 | 0.036601 | 0.117188 |

Attack conclusion:

- `loss3` is best.
- `random_solver_y` is much better than `baseline` and far better than
  `random_clean_y`, but it is worse than `loss1`/`loss2`/`loss3` on attack
  increase in this 25-sample set.
- `random_clean_y` is the worst by a large margin.

The raw adversarial attack delta vectors are stored in:

- Old four models:
  `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/attack_traces/`
- Random models:
  `forensics/burgers_random_field_final_models_full_suite_20260613/p2q2_attack/*/final_delta_by_sample.npz`

## 4. Robustness Indicator 2: Error-Jacobian/SVD Scalars

Here `J_error = J_model - J_solver`. Lower error spectral norm and Frobenius norm
are better.

| model | error spectral norm | error Fro norm | error effective rank | model spectral norm | solver spectral norm |
|---|---:|---:|---:|---:|---:|
| loss3 | 1.271533 | 1.898531 | 6.373320 | 3.604760 | 3.754830 |
| loss1 | 1.702471 | 2.781139 | 18.136217 | 3.631900 | 3.754830 |
| random_solver_y | 1.735561 | 2.674979 | 7.198486 | 3.653871 | 3.754830 |
| loss2 | 1.855107 | 2.887809 | 11.930577 | 3.609040 | 3.754830 |
| baseline | 2.367033 | 3.574622 | 10.284617 | 3.455550 | 3.754830 |
| random_clean_y | 3.828716 | 6.759260 | 6.943157 | 2.525394 | 3.754830 |

Jacobian/SVD conclusion:

- `loss3` has the smallest error-Jacobian spectral norm.
- `random_solver_y` is in the same range as `loss1`/`loss2`, and much better
  than baseline.
- `random_clean_y` has the largest error-Jacobian norm.

These scalar SVD quantities are:

- `error spectral norm`: top singular value of \(J_{\mathrm{error}}\).
- `error Fro norm`: square root of the sum of all squared singular values.
- `error effective rank`: entropy rank of the squared singular-value
  distribution.

The full singular values and singular vectors are stored in the per-sample SVD
NPZ files:

- Old four models:
  `forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/sample_*/`
- Random models:
  `forensics/burgers_random_field_final_models_full_suite_20260613/jacobian_svd/sample_*/`

## 5. Robustness Indicator 3a: Singular Vectors/Subspaces vs Solver

This compares the model Jacobian singular vectors/subspaces against the solver
Jacobian singular vectors/subspaces on the same input sample.

Higher means the model Jacobian geometry is more similar to the solver Jacobian.
For top-\(k\) subspace values, \(1\) means identical subspaces and \(0\) means
orthogonal subspaces.

| model | top1 right cos | top1 left cos | top5 right subspace | top10 right subspace | top20 right subspace | top5 left subspace | top10 left subspace | top20 left subspace |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.853861 | 0.692770 | 0.902259 | 0.886490 | 0.731655 | 0.773360 | 0.799476 | 0.800560 |
| loss1 | 0.937746 | 0.841553 | 0.924231 | 0.911366 | 0.764206 | 0.818937 | 0.845780 | 0.859059 |
| loss2 | 0.940075 | 0.833315 | 0.924449 | 0.904067 | 0.760543 | 0.821315 | 0.838075 | 0.868047 |
| loss3 | 0.960089 | 0.913083 | 0.936441 | 0.957945 | 0.886612 | 0.897940 | 0.933519 | 0.943287 |
| random_clean_y | 0.283088 | 0.191025 | 0.541534 | 0.634241 | 0.656611 | 0.414371 | 0.540846 | 0.675828 |
| random_solver_y | 0.917669 | 0.820936 | 0.919859 | 0.911324 | 0.802382 | 0.824118 | 0.847724 | 0.890080 |

Subspace conclusion:

- `loss3` is the closest to the solver Jacobian geometry.
- `random_solver_y` is close to `loss1`/`loss2` and much closer to solver than
  `baseline` in several top-\(k\) subspace measures.
- `random_clean_y` is geometrically very far from solver, especially in top-1
  left/right singular vectors.

Correlation with attack loss increase:

| scope | metric | Pearson | Spearman |
|---|---|---:|---:|
| old four | top20 right subspace similarity to solver | -0.519958 | -0.697438 |
| old four | top20 left subspace similarity to solver | -0.562687 | -0.727249 |
| random two | top20 right subspace similarity to solver | -0.779752 | -0.826074 |
| random two | top20 left subspace similarity to solver | -0.904947 | -0.914622 |

The sign is negative because higher similarity to the solver corresponds to
lower attack loss increase.

## 6. Robustness Indicator 3b: \(A^\top b\), \(J_{\mathrm{error}}\delta\), and Direction Quantities

For the old four models, the stored local-gradient quantity is
`bias_gradient_norm`, which corresponds to the `||A^T b||` / `||J_error^T error||`
style quantity.

For the random models, the stored matched quantity is `||J_error @ delta||`.

These are not the same algebraic scalar, but both are local linearized
error-response diagnostics.

| model | `||A^T b||` mean | `||A^T b||` median | `||J_error delta||_2` mean | `J_error delta` RMS mean |
|---|---:|---:|---:|---:|
| baseline | 0.462951 | 0.331235 |  |  |
| loss1 | 0.315854 | 0.173628 |  |  |
| loss2 | 0.343340 | 0.190082 |  |  |
| loss3 | 0.111950 | 0.080228 |  |  |
| random_clean_y |  |  | 9.017980 | 0.281812 |
| random_solver_y |  |  | 2.528930 | 0.079029 |

Direction/vector conclusion:

- Among old four models, `loss3` has the smallest `||A^T b||`.
- Among the random models, `random_solver_y` has much smaller
  `||J_error @ delta||` than `random_clean_y`.
- This matches the attack-loss result: `random_solver_y` is far more robust than
  `random_clean_y`.

## 7. Delta/Singular-Vector Angle Similarity

Angle between post-training adversarial attack delta and the top right singular
vector of `J_error`.

Lower abs-cos means closer to orthogonal. Higher angle means closer to
orthogonal.

| model | mean abs cos(delta, top error SV) | median abs cos | mean angle | median angle |
|---|---:|---:|---:|---:|
| baseline | 0.304880 | 0.194071 | 70.727 deg | 78.810 deg |
| loss1 | 0.209058 | 0.125627 | 77.154 deg | 82.783 deg |
| loss2 | 0.196515 | 0.116110 | 77.772 deg | 83.332 deg |
| loss3 | 0.091452 | 0.078159 | 84.739 deg | 85.517 deg |
| random_clean_y | 0.399197 |  | 65.668 deg | 62.820 deg |
| random_solver_y | 0.277103 |  | 73.123 deg | 76.080 deg |

Angle conclusion:

- `loss3` attack deltas are least aligned with the top error singular vector.
- `random_clean_y` has the strongest delta/top-error-SV alignment among these
  six, and it is also the least robust.
- Alignment with the top singular vector alone is not the whole mechanism, but
  it is a useful direction diagnostic.

## 8. Correlation: Old Four Models

These correlations use the old four models over the matched 25 samples, 100
model-sample pairs total. Target is attack loss increase.

| x metric | Pearson with attack increase | Spearman with attack increase |
|---|---:|---:|
| `||A^T b||` / `bias_gradient_norm` | 0.745308 | 0.850705 |
| `J_error` spectral norm | 0.637190 | 0.777810 |
| SVD local gain at epsilon | 0.576744 | 0.780198 |
| abs cos(delta, top error SV) | 0.151217 | 0.148107 |

Old-four correlation conclusion:

- `||A^T b||` is the strongest predictor of attack loss increase.
- `J_error` spectral norm is also strongly correlated.
- Top singular-vector alignment by itself is weak as a scalar predictor.

## 9. Correlation: Random Models

### 9.1 Full 52-Dataset Attack Suite

This uses all 10,200 attacked samples for the two random models.

Loss-increase similarity between `random_clean_y` and `random_solver_y`:

| scope | n | Pearson | Spearman |
|---|---:|---:|---:|
| all | 10200 | 0.468520 | 0.549069 |
| generalization | 10000 | 0.478459 | 0.576990 |
| test | 150 | 0.184692 | 0.062806 |
| train | 50 | -0.065761 | -0.009844 |

Delta-vector direction similarity between `random_clean_y` and
`random_solver_y`:

| scope | n | mean cosine | mean abs cosine | mean angle | mean abs angle |
|---|---:|---:|---:|---:|---:|
| all | 10200 | 0.105515 | 0.154642 | 83.787 deg | 80.958 deg |
| generalization | 10000 | 0.106857 | 0.155469 | 83.708 deg | 80.908 deg |
| test | 150 | 0.050973 | 0.115755 | 87.040 deg | 83.314 deg |
| train | 50 | 0.000678 | 0.106066 | 89.954 deg | 83.893 deg |

Full-suite random conclusion:

- The two random models have moderately similar scalar loss-increase patterns on
  the 50 generalization datasets.
- Their attack delta directions are nearly orthogonal on average.
- Therefore, the scalar vulnerability pattern is more similar than the actual
  attack direction vectors.

### 9.2 Random Models on 25-Sample SVD Subset

Target is attack loss increase.

| model | x metric | Pearson | Spearman |
|---|---|---:|---:|
| random_clean_y | `J_error` spectral norm | 0.361258 | 0.416923 |
| random_clean_y | `||J_error delta||_2` | 0.533878 | 0.601739 |
| random_clean_y | top error singular value | 0.378647 | 0.434783 |
| random_clean_y | abs cos(delta, top error SV) | -0.037997 | -0.082609 |
| random_solver_y | `J_error` spectral norm | 0.654890 | 0.796923 |
| random_solver_y | `||J_error delta||_2` | 0.603287 | 0.715652 |
| random_solver_y | top error singular value | 0.638577 | 0.772174 |
| random_solver_y | abs cos(delta, top error SV) | -0.238355 | -0.478261 |

Random-model correlation conclusion:

- `||J_error @ delta||` is much more informative than raw delta magnitude.
- For `random_solver_y`, `J_error` spectral norm and top singular value are also
  strongly correlated with attack loss increase.
- Delta/top-singular-vector alignment alone is weak or negative, so it should not
  be used alone as the mechanism.

## 10. Final Interpretation

Model ranking:

- Best overall: `loss3`.
- Best random-field method: `random_solver_y`.
- Failed random-field method: `random_clean_y`.

Metric interpretation:

- Clean RMSE/relative L2 and robustness agree on the main story:
  `loss3` is strongest, `random_clean_y` is weakest.
- The most useful robustness mechanism metrics are:
  attack loss increase, `J_error` spectral norm/top singular value,
  `||A^T b||`, and `||J_error @ delta||`.
- Delta is a vector. Its raw magnitude/RMS is mostly a constraint check because
  the attack is radius-limited. Direction and Jacobian response matter more than
  magnitude.
- Direction similarity matters, but top-singular-vector alignment alone does not
  fully explain loss increase. The local response `J_error @ delta` is more
  directly tied to attack loss growth.

## 11. Files Written

- `clean_generalization_50dataset_six_models.csv`
- `robustness_25sample_svd_attack_six_models.csv`
- `per_sample_25sample_svd_attack_six_models.csv`
- `six_model_25sample_model_solver_subspace_similarity.csv`
- `random_25sample_model_solver_subspace_similarity.csv`
- `model_solver_subspace_similarity_correlations.csv`
- `old4_25sample_metric_correlations.csv`
- `old4_25sample_direction_angle_summary.csv`
- `random_25sample_svd_metric_correlations.csv`
- `random_25sample_direction_angle_summary.csv`
- `random_full10200_cross_model_delta_and_loss_similarity.csv`
- `random_full10200_delta_magnitude_correlations_low_priority.csv`

## 12. Detailed By-Dataset Appendix

The full per-dataset/per-sample detailed appendix has now been written here:

- `forensics/burgers_six_model_latest_wideparam_summary_20260613/detailed_by_dataset_and_sample_20260613.md`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/clean_52dataset_six_models_detailed.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/random_models_attack_52dataset_detailed.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/robustness_25sample_manifest.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/robustness_25sample_six_models_detailed_with_subspace.csv`

It contains all 52 clean datasets explicitly, plus the full 52-dataset random-model attack table and the six-model 25-sample SVD/attack mechanism tables.


## Frobenius Norm Correction 2026-06-13

I checked the generating code and the saved NPZ metadata after the Frobenius-norm question. The old-four models (`baseline`, `loss1`, `loss2`, `loss3`) used full dense `1024 x 1024` Jacobians and full SVDs (`svd_method: full_np_linalg_svd`, `uses_topk_svd: false`). Their `error_fro_norm` is the true Frobenius norm from all singular values.

The random models (`random_clean_y`, `random_solver_y`) were run with `svd_method: topk`, `svd_top_k: 20`. Therefore the originally reported random `fro_norm` in `jacobian_svd_summary.csv` is `sqrt(sum of top-20 singular values squared)`, not the full Frobenius norm. Because the full `1024 x 1024` Jacobian matrices were saved in the random NPZ files, I recomputed the true Frobenius norm directly from matrix entries.

Corrected error-Frobenius means on the 25-sample SVD/attack set:

| model | original reported Fro basis | original reported mean | comparable/true Fro mean | top20 fraction | effective-rank basis |
|---|---|---:|---:|---:|---|
| baseline | full_svd_all_singular_values | 3.574618 | 3.574618 |  | full_svd_all_singular_values |
| loss1 | full_svd_all_singular_values | 2.781142 | 2.781142 |  | full_svd_all_singular_values |
| loss2 | full_svd_all_singular_values | 2.887806 | 2.887806 |  | full_svd_all_singular_values |
| loss3 | full_svd_all_singular_values | 1.898529 | 1.898529 |  | full_svd_all_singular_values |
| random_clean_y | true_full_jacobian_entry_fro_for_comparable; original_report_was_top20_proxy | 6.759260 | 6.776453 | 0.997367 | top20_only_not_full_effective_rank |
| random_solver_y | true_full_jacobian_entry_fro_for_comparable; original_report_was_top20_proxy | 2.674979 | 2.681731 | 0.995400 | top20_only_not_full_effective_rank |

Files written for this correction:

- `forensics/burgers_six_model_latest_wideparam_summary_20260613/random_models_svd_top20_vs_true_fro_from_jacobian.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/robustness_25sample_six_models_detailed_corrected_fro_basis.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/robustness_25sample_svd_attack_six_models_corrected_fro_basis.csv`

The random-model spectral norm/top singular value remains the top singular value from the top-k SVD. The random-model effective rank remains top20-only and should not be interpreted as full effective rank unless a full SVD is recomputed.

## 12. Combined A^T b and J_error Delta Table

This table combines the old-four `A^T b`-style scalar and the random-model `J_error @ delta` scalar in one place. They are not the same algebraic quantity, so the table keeps both columns and identifies the stored quantity type per model.

| model | quantity_type | n | attack_inc | err_spec | err_fro_comparable | top20_R_solver | top20_L_solver | A^T b mean | A^T b median | Jerr_delta_L2 mean | Jerr_delta_L2 median | Jerr_delta_RMS mean | Jerr_delta_RMS median | abs_cos_delta_top_errSV | blank_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | A^T b / bias_gradient_norm | 25 | 0.009164 | 2.367025 | 3.574618 | 0.731655 | 0.800560 | 0.462951 | 0.331235 |  |  |  |  | 0.304880 | random J_error_delta not computed for old-four table |
| loss1 | A^T b / bias_gradient_norm | 25 | 0.005838 | 1.702470 | 2.781142 | 0.764206 | 0.859059 | 0.315854 | 0.173628 |  |  |  |  | 0.209058 | random J_error_delta not computed for old-four table |
| loss2 | A^T b / bias_gradient_norm | 25 | 0.005623 | 1.855112 | 2.887806 | 0.760543 | 0.868047 | 0.343340 | 0.190082 |  |  |  |  | 0.196515 | random J_error_delta not computed for old-four table |
| loss3 | A^T b / bias_gradient_norm | 25 | 0.003063 | 1.271530 | 1.898529 | 0.886612 | 0.943287 | 0.111950 | 0.080228 |  |  |  |  | 0.091452 | random J_error_delta not computed for old-four table |
| random_clean_y | J_error @ adversarial_delta | 25 | 0.036601 | 3.828716 | 6.776453 | 0.656611 | 0.675828 |  |  | 9.017979 | 8.926605 | 0.281812 | 0.278956 | 0.399197 | A^T b not computed/stored for random table |
| random_solver_y | J_error @ adversarial_delta | 25 | 0.006923 | 1.735561 | 2.681731 | 0.802382 | 0.890080 |  |  | 2.528932 | 2.343514 | 0.079029 | 0.073235 | 0.277103 | A^T b not computed/stored for random table |

CSV: `forensics/burgers_six_model_latest_wideparam_summary_20260613/combined_atb_jerror_delta_six_models_25sample.csv`

## 13. Unified Bias-Gradient Table: J_error^T clean error

This is the corrected same-math table. For all six models the main scalar is `||J_error^T error||`, where `error = model(x) - solver(x)`. The old-four rows were already stored as `bias_gradient_norm`; the random rows were newly computed from the saved full `J_error` matrices and clean residuals. The `||J_error delta||` column is kept only as a separate adversarial-delta diagnostic, not as the bias-gradient quantity.

| model | n | attack_inc | clean_mse | ||error|| | ||Jerr^T error|| mean | ||Jerr^T error|| median | Jerr^T error RMS | err_spec | err_fro | top20_R_solver | top20_L_solver | abs_cos(delta,Jerr^Terror) | ||Jerr delta|| separate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 25 | 0.009164 | 0.001095 | 0.914193 | 0.462951 | 0.331235 | 0.014467 | 2.367025 | 3.574618 | 0.731655 | 0.800560 | 0.793450 |  |
| loss1 | 25 | 0.005838 | 0.000823 | 0.714167 | 0.315854 | 0.173628 | 0.009870 | 1.702470 | 2.781142 | 0.764206 | 0.859059 | 0.803056 |  |
| loss2 | 25 | 0.005623 | 0.000856 | 0.742491 | 0.343340 | 0.190082 | 0.010729 | 1.855112 | 2.887806 | 0.760543 | 0.868047 | 0.798714 |  |
| loss3 | 25 | 0.003063 | 0.000244 | 0.411851 | 0.111950 | 0.080228 | 0.003498 | 1.271530 | 1.898529 | 0.886612 | 0.943287 | 0.726458 |  |
| random_clean_y | 25 | 0.036601 | 0.008996 | 2.760806 | 3.838714 | 3.508727 | 0.119960 | 3.828716 | 6.776453 | 0.656611 | 0.675828 | 0.709841 | 9.017979 |
| random_solver_y | 25 | 0.006923 | 0.000744 | 0.676364 | 0.304904 | 0.165154 | 0.009528 | 1.735561 | 2.681731 | 0.802382 | 0.890080 | 0.730414 | 2.528932 |

CSV files:

- `forensics/burgers_six_model_latest_wideparam_summary_20260613/combined_jerror_transpose_error_six_models_25sample.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/robustness_25sample_six_models_detailed_with_jerror_transpose_error.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/random_models_jerror_transpose_error_25sample.csv`
