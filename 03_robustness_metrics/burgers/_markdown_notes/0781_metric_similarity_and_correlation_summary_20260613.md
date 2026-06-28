# Metric Similarity and Correlation Summary: Burgers Six Models

Date: 2026-06-13

This report uses the corrected same-math bias-gradient quantity `||J_error^T error||`, where `error = model(x) - solver(x)`. The separate `||J_error delta||` is kept only as an adversarial-delta diagnostic.

## Files

- `forensics/burgers_six_model_latest_wideparam_summary_20260613/robustness_25sample_six_models_similarity_enriched_rows.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/metric_correlations_with_attack_corrected_jerrT_25sample.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/metric_pairwise_correlations_corrected_jerrT_25sample.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/per_sample_model_rank_similarity_corrected_jerrT_25sample.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/per_sample_model_rank_similarity_summary_corrected_jerrT_25sample.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/model_level_metric_means_corrected_jerrT_25sample.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/model_level_rank_similarity_corrected_jerrT_25sample.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/direction_angle_similarity_summary_corrected_jerrT_25sample.csv`
- `forensics/burgers_six_model_latest_wideparam_summary_20260613/subspace_similarity_summary_corrected_jerrT_25sample.csv`

## 1. Correlation With Attack Loss Increase

Positive correlation means larger metric tends to mean larger attack loss increase. For solver-subspace similarity itself, negative correlation is good: higher similarity to solver tends to lower attack increase.

| scope | metric | n | Pearson | Spearman |
| --- | --- | --- | --- | --- |
| all_six_150 | error_spectral_norm | 150 | 0.693765 | 0.831203 |
| old_four_100 | error_spectral_norm | 100 | 0.637190 | 0.777810 |
| random_two_50 | error_spectral_norm | 50 | 0.769753 | 0.825882 |
| all_six_150 | error_fro_norm_comparable | 150 | 0.803927 | 0.872713 |
| old_four_100 | error_fro_norm_comparable | 100 | 0.666342 | 0.809961 |
| random_two_50 | error_fro_norm_comparable | 50 | 0.859156 | 0.896279 |
| all_six_150 | bias_gradient_norm | 150 | 0.899726 | 0.897556 |
| old_four_100 | bias_gradient_norm | 100 | 0.745308 | 0.850705 |
| random_two_50 | bias_gradient_norm | 50 | 0.884352 | 0.911357 |
| all_six_150 | model_solver_top20_right_subspace_mean_cos | 150 | -0.591691 | -0.737457 |
| old_four_100 | model_solver_top20_right_subspace_mean_cos | 100 | -0.519958 | -0.697438 |
| random_two_50 | model_solver_top20_right_subspace_mean_cos | 50 | -0.779752 | -0.826074 |
| all_six_150 | model_solver_top20_left_subspace_mean_cos | 150 | -0.778807 | -0.815167 |
| old_four_100 | model_solver_top20_left_subspace_mean_cos | 100 | -0.562687 | -0.727249 |
| random_two_50 | model_solver_top20_left_subspace_mean_cos | 50 | -0.904947 | -0.914622 |
| all_six_150 | solver_subspace_mismatch_top20_right | 150 | 0.591691 | 0.737457 |
| old_four_100 | solver_subspace_mismatch_top20_right | 100 | 0.519958 | 0.697438 |
| random_two_50 | solver_subspace_mismatch_top20_right | 50 | 0.779752 | 0.826074 |
| all_six_150 | solver_subspace_mismatch_top20_left | 150 | 0.778807 | 0.815167 |
| old_four_100 | solver_subspace_mismatch_top20_left | 100 | 0.562687 | 0.727249 |
| random_two_50 | solver_subspace_mismatch_top20_left | 50 | 0.904947 | 0.914622 |
| all_six_150 | delta_top_error_sv_abs_cos | 148 | 0.267085 | 0.190618 |
| old_four_100 | delta_top_error_sv_abs_cos | 100 | 0.151217 | 0.148107 |
| random_two_50 | delta_top_error_sv_abs_cos | 48 | 0.174126 | 0.094768 |
| all_six_150 | attack_delta_outward_abs_cos | 148 | -0.081523 | 0.098913 |
| old_four_100 | attack_delta_outward_abs_cos | 100 | 0.299544 | 0.333045 |
| random_two_50 | attack_delta_outward_abs_cos | 48 | -0.022105 | 0.041902 |
| all_six_150 | svd_outward_abs_cos | 150 | 0.372118 | 0.272878 |
| old_four_100 | svd_outward_abs_cos | 100 | 0.166199 | 0.154743 |
| random_two_50 | svd_outward_abs_cos | 50 | 0.387274 | 0.339736 |
| all_six_150 | j_error_delta_l2 | 48 | 0.882571 | 0.900456 |
| old_four_100 | j_error_delta_l2 | 0 |  |  |
| random_two_50 | j_error_delta_l2 | 48 | 0.882571 | 0.900456 |

## 2. Per-Sample Model-Ranking Similarity

For each of the 25 samples, this ranks the six models by attack increase and compares that ranking with the ranking from each diagnostic metric. Higher mean Spearman means the metric orders the six models similarly to attack loss increase on the same sample.

| metric | samples | mean Spearman | median Spearman | mean Pearson |
| --- | --- | --- | --- | --- |
| clean_residual_mse_recomputed | 25 | 0.725714 | 0.771429 | 0.919826 |
| error_spectral_norm | 25 | 0.593143 | 0.657143 | 0.737390 |
| error_fro_norm_comparable | 25 | 0.682286 | 0.714286 | 0.880517 |
| bias_gradient_norm | 25 | 0.769143 | 0.828571 | 0.976523 |
| solver_subspace_mismatch_top20_right | 25 | 0.684571 | 0.828571 | 0.744491 |
| solver_subspace_mismatch_top20_left | 25 | 0.741714 | 0.771429 | 0.874134 |
| attack_delta_outward_abs_cos | 25 | 0.123429 | 0.085714 | -0.147918 |
| svd_outward_abs_cos | 25 | 0.261714 | 0.200000 | 0.465592 |

## 3. Model-Level Means

| model | attack inc | clean MSE | err spec | err Fro | ||Jerr^T err|| | top20 R sim | top20 L sim | cos(delta,Jerr^Terr) | cos(topSV,Jerr^Terr) | ||Jerr delta|| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.009164 | 0.001095 | 2.367025 | 3.574618 | 0.462951 | 0.731655 | 0.800560 | 0.793450 | 0.304069 |  |
| loss1 | 0.005838 | 0.000823 | 1.702470 | 2.781142 | 0.315854 | 0.764206 | 0.859059 | 0.803056 | 0.216143 |  |
| loss2 | 0.005623 | 0.000856 | 1.855112 | 2.887806 | 0.343340 | 0.760543 | 0.868047 | 0.798714 | 0.254186 |  |
| loss3 | 0.003063 | 0.000244 | 1.271530 | 1.898529 | 0.111950 | 0.886612 | 0.943287 | 0.726458 | 0.185637 |  |
| random_clean_y | 0.036601 | 0.008996 | 3.828716 | 6.776453 | 3.838714 | 0.656611 | 0.675828 | 0.709841 | 0.524951 | 9.017979 |
| random_solver_y | 0.006923 | 0.000744 | 1.735561 | 2.681731 | 0.304904 | 0.802382 | 0.890080 | 0.730414 | 0.244652 | 2.528932 |

## 4. Direction and Angle Similarity

| model | cos(delta,topSV) | angle | cos(delta,Jerr^Terr) | angle | cos(topSV,Jerr^Terr) | angle |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.304880 | 70.726607 | 0.793450 | 36.206155 | 0.304069 | 71.350218 |
| loss1 | 0.209058 | 77.154344 | 0.803056 | 35.932676 | 0.216143 | 76.646766 |
| loss2 | 0.196515 | 77.771737 | 0.798714 | 35.929811 | 0.254186 | 73.688382 |
| loss3 | 0.091452 | 84.739283 | 0.726458 | 42.714136 | 0.185637 | 79.115331 |
| random_clean_y | 0.399197 | 65.668325 | 0.709841 | 43.055979 | 0.524951 | 56.283452 |
| random_solver_y | 0.277103 | 73.122483 | 0.730414 | 42.257089 | 0.244652 | 75.390965 |

## 5. Model-Solver Subspace Similarity

| model | top1 R | top1 L | top5 R | top5 L | top20 R | top20 L |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.853861 | 0.692770 | 0.902259 | 0.773360 | 0.731655 | 0.800560 |
| loss1 | 0.937746 | 0.841553 | 0.924231 | 0.818937 | 0.764206 | 0.859059 |
| loss2 | 0.940075 | 0.833315 | 0.924449 | 0.821315 | 0.760543 | 0.868047 |
| loss3 | 0.960089 | 0.913083 | 0.936441 | 0.897940 | 0.886612 | 0.943287 |
| random_clean_y | 0.283088 | 0.191025 | 0.541534 | 0.414371 | 0.656611 | 0.675828 |
| random_solver_y | 0.917669 | 0.820936 | 0.919859 | 0.824118 | 0.802382 | 0.890080 |

## 6. Short Interpretation

- `||J_error^T error||` is the strongest same-math mechanism scalar across the old-four models and remains very informative when the random models are added.
- Solver-subspace similarity is inversely related to attack increase: models whose Jacobian subspaces look more like the solver tend to be more robust.
- Direction alignment alone is not enough. `cos(delta, top error SV)` is weak compared with `||J_error^T error||`, `J_error` norm, and solver-subspace mismatch.
- `random_clean_y` is consistently bad: high clean error, high `J_error` norm, high `||J_error^T error||`, low solver-subspace similarity, and high attack increase.
- `random_solver_y` aligns much better with the solver geometry and has much smaller `||J_error^T error||`; it is close to `loss1/loss2` but still behind `loss3`.
