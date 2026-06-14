# Burgers Metric Coverage, Correlation, And Similarity Inventory, 20260614

Generated: 2026-06-14T21:53:35+00:00

This report answers whether the requested Burgers clean/attack/robustness/SVD/Jacobian scalar metrics, scalar significance tests, scalar correlations, and vector cosine/angle/subspace similarities are present in the final audit bundle.

## Short Answer

Yes: the requested quantities are present locally, but they are split across several tables. Scalar quantities have best-model, runner-up, mean/std/median, paired t/Wilcoxon/FDR significance where paired model coverage is available. Scalar correlations have Pearson/Spearman and q-values in the ranked correlation tables. Vector quantities are summarized as cosine similarity and angle in degrees, and top-k subspace similarities are ranked; these vector quantities are diagnostic and are not always valid as a single 'best model quality' claim.

## Coverage Summary

| category | covered | primary_tables | best_and_runner_up | significance | correlations | vector_similarity_or_angle | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 52x6 clean RMSE/RelativeL2/MSE | 1 | clean_52; clean_52dataset_metric_long_ranked; metric_model_summary | 1 | 1 | not central; scalar correlations are in metric_pairwise_correlations_sorted | 0 | Clean all-52 and gen-50 first-vs-second significance is in first_vs_second_key. |
| 52x6 strict attack final loss / loss increase / delta | 1 | attack_strict_52; attack_52dataset_metric_long_ranked; first_vs_second_key | 1 | 1 | 25-sample attack correlations in correlations_with_attack_sorted; random full10200 delta/loss correlations are recovered separately | attack delta angle/cosine in direction_angles | Attack loss increase and final loss: loss3 wins every 52/52 row; delta RMS is process/constraint, not quality evidence. |
| 25-sample robustness scalar norms | 1 | robustness_25; robustness_25sample_metric_long_ranked; metric_model_summary | 1 | 1 | metric_pairwise_correlations_sorted; correlations_with_attack_sorted | some scalar rows are derived from vector comparisons; raw vector angles summarized separately | Includes clean residual, Frobenius, spectral norm, J_error^T error norm/RMS, J_error delta diagnostics. |
| top-k singular values / SVD spectrum | 1 | top20_svd_ranked; topk_svd_ranked; top100_svd_ranked; top100_svd_tests | 1 | 1 | correlations_with_attack_sorted and svd_attack_correlations in recovered historical bundle | top singular vector cosine with attack delta in direction_angles | Top1 mean-best is loss3 but can be nonsignificant in all-25; top5+ and top100 supplement are significant. |
| top-k model/solver subspace similarity | 1 | subspace_summary; top100_subspace_ranked; top50_top100_subspace_tests | 1 | partial | model_solver_subspace_similarity_correlations and correlations_with_attack_sorted | 1 | Higher is better. Some top50/top100 rows are random_solver_y best; this is diagnostic, not clean/attack quality proof. |
| vector direction cosine/angle | 1 | direction_angles; old4/random direction angle summaries | not treated as quality best | not generally used as best-model evidence | random_affine_direction_correlations and metric correlations | 1 | Includes attack delta vs top error SV, attack delta vs outward, SVD outward; mean/median cosine and mean/median angle in degrees. |
| scalar-scalar correlations | 1 | metric_pairwise_correlations_sorted; correlations_with_attack_sorted | 0 | correlation p/q present in ranked correlation tables | 1 | 0 | Includes Pearson/Spearman and BH/FDR q for ranked correlation tables. |

## Counts

| item | value |
| --- | --- |
| metric_model_summary_rows | 1522 |
| metric_best_summary_rows | 321 |
| best_vs_other_significance_rows | 1201 |
| pairwise_scalar_correlation_rows | 220 |
| correlations_with_attack_rows | 56 |
| direction_angle_model_rows | 6 |
| subspace_summary_model_rows | 6 |
| first_vs_second_key_rows | 10 |
| six_model_evidence_best_counts | {"loss1": 4, "loss3": 119, "random_solver_y": 45} |

## Direction / Angle Summary

| model | n | delta_top_error_sv_abs_cos_mean | delta_top_error_sv_abs_cos_angle_deg_mean | attack_delta_outward_abs_cos_mean | attack_delta_outward_abs_cos_angle_deg_mean | svd_outward_abs_cos_mean | svd_outward_abs_cos_angle_deg_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 25 | 0.30488 | 70.7266 | 0.79345 | 36.2062 | 0.304069 | 71.3502 |
| loss1 | 25 | 0.209058 | 77.1543 | 0.803056 | 35.9327 | 0.216143 | 76.6468 |
| loss2 | 25 | 0.196515 | 77.7717 | 0.798714 | 35.9298 | 0.254186 | 73.6884 |
| loss3 | 25 | 0.0914521 | 84.7393 | 0.726458 | 42.7141 | 0.185637 | 79.1153 |
| random_clean_y | 25 | 0.399197 | 65.6683 | 0.709841 | 43.056 | 0.524951 | 56.2835 |
| random_solver_y | 25 | 0.277103 | 73.1225 | 0.730414 | 42.2571 | 0.244652 | 75.391 |

## Model-Solver Subspace Summary

| model | model_solver_top1_right_abs_cos | model_solver_top1_left_abs_cos | model_solver_top5_right_subspace_mean_cos | model_solver_top5_left_subspace_mean_cos | model_solver_top20_right_subspace_mean_cos | model_solver_top20_left_subspace_mean_cos |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.853861 | 0.69277 | 0.902259 | 0.77336 | 0.731655 | 0.80056 |
| loss1 | 0.937746 | 0.841553 | 0.924231 | 0.818937 | 0.764206 | 0.859059 |
| loss2 | 0.940075 | 0.833315 | 0.924449 | 0.821315 | 0.760543 | 0.868047 |
| loss3 | 0.960089 | 0.913083 | 0.936441 | 0.89794 | 0.886612 | 0.943287 |
| random_clean_y | 0.283088 | 0.191025 | 0.541534 | 0.414371 | 0.656611 | 0.675828 |
| random_solver_y | 0.917669 | 0.820936 | 0.919859 | 0.824118 | 0.802382 | 0.89008 |

## Top Correlations With Attack

| scope | metric | metric_label | meaning | target | n | pearson | spearman | abs_pearson | abs_spearman | pearson_p_approx | spearman_p_approx | pearson_q_bh_fdr | spearman_q_bh_fdr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_two_50 | solver_subspace_mismatch_top20_left | 1 - top20 left subspace similarity | higher_bad | attack_loss_increase | 50 | 0.904947 | 0.914622 | 0.904947 | 0.914622 | 1.945e-19 | 1.654e-20 | 6.292e-19 | 3.498e-20 |
| random_two_50 | model_solver_top20_left_subspace_mean_cos | model-solver top20 left subspace similarity | higher_good | attack_loss_increase | 50 | -0.904947 | -0.914622 | 0.904947 | 0.914622 | 1.945e-19 | 1.654e-20 | 6.292e-19 | 3.498e-20 |
| random_two_50 | bias_gradient_rms | RMS(J_error^T error) | higher_bad | attack_loss_increase | 50 | 0.884352 | 0.911357 | 0.884352 | 0.911357 | 1.695e-17 | 3.920e-20 | 4.440e-17 | 7.701e-20 |
| random_two_50 | bias_gradient_norm | ||J_error^T error|| | higher_bad | attack_loss_increase | 50 | 0.884352 | 0.911357 | 0.884352 | 0.911357 | 1.695e-17 | 3.920e-20 | 4.440e-17 | 7.701e-20 |
| all_six_150 | j_error_delta_l2 | ||J_error delta|| separate diagnostic | higher_bad_random_only | attack_loss_increase | 48 | 0.882571 | 0.900456 | 0.882571 | 0.900456 | 1.107e-16 | 3.018e-18 | 2.647e-16 | 5.355e-18 |
| random_two_50 | j_error_delta_l2 | ||J_error delta|| separate diagnostic | higher_bad_random_only | attack_loss_increase | 48 | 0.882571 | 0.900456 | 0.882571 | 0.900456 | 1.107e-16 | 3.018e-18 | 2.647e-16 | 5.355e-18 |
| all_six_150 | bias_gradient_rms | RMS(J_error^T error) | higher_bad | attack_loss_increase | 150 | 0.899726 | 0.897556 | 0.899726 | 0.897556 | 3.735e-55 | 1.677e-54 | 1.027e-53 | 4.613e-53 |
| all_six_150 | bias_gradient_norm | ||J_error^T error|| | higher_bad | attack_loss_increase | 150 | 0.899726 | 0.897556 | 0.899726 | 0.897556 | 3.735e-55 | 1.677e-54 | 1.027e-53 | 4.613e-53 |
| random_two_50 | error_fro_norm_comparable | J_error Frobenius norm | higher_bad | attack_loss_increase | 50 | 0.859156 | 0.896279 | 0.859156 | 0.896279 | 1.430e-15 | 1.429e-18 | 3.146e-15 | 2.710e-18 |
| generalization_all_six_126 | j_error_delta_l2 | ||J_error delta|| separate diagnostic | higher_bad_random_only | attack_loss_increase | 42 | 0.862165 | 0.879588 | 0.862165 | 0.879588 | 2.222e-13 | 1.760e-14 | 4.073e-13 | 2.420e-14 |
| generalization_all_six_126 | bias_gradient_norm | ||J_error^T error|| | higher_bad | attack_loss_increase | 126 | 0.925377 | 0.875762 | 0.925377 | 0.875762 | 4.421e-54 | 4.918e-41 | 6.079e-53 | 5.410e-40 |
| generalization_all_six_126 | bias_gradient_rms | RMS(J_error^T error) | higher_bad | attack_loss_increase | 126 | 0.925377 | 0.875762 | 0.925377 | 0.875762 | 4.421e-54 | 4.918e-41 | 6.079e-53 | 5.410e-40 |

## Top Scalar Pairwise Correlations

| scope | metric_a | metric_b | n | pearson | spearman | abs_pearson | abs_spearman | pearson_p_approx | spearman_p_approx | pearson_q_bh_fdr | spearman_q_bh_fdr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_six_150 | error_fro_norm_comparable | bias_gradient_norm | 150 | 0.782006 | 0.979306 | 0.782006 | 0.979306 | 3.414e-32 | 1.380e-104 | 3.983e-31 | 2.898e-102 |
| old_four_100 | error_fro_norm_comparable | bias_gradient_norm | 100 | 0.913012 | 0.97883 | 0.913012 | 0.97883 | 6.049e-40 | 2.505e-69 | 1.270e-38 | 5.260e-68 |
| old_four_100 | clean_residual_mse_recomputed | error_fro_norm_comparable | 100 | 0.828601 | 0.978506 | 0.828601 | 0.978506 | 1.973e-26 | 5.232e-69 | 1.535e-25 | 9.989e-68 |
| old_four_100 | clean_residual_mse_recomputed | bias_gradient_norm | 100 | 0.886126 | 0.977366 | 0.886126 | 0.977366 | 1.676e-34 | 6.409e-68 | 2.200e-33 | 1.122e-66 |
| generalization_all_six_126 | error_fro_norm_comparable | bias_gradient_norm | 126 | 0.791242 | 0.974857 | 0.791242 | 0.974857 | 2.934e-28 | 1.033e-82 | 2.801e-27 | 5.424e-81 |
| generalization_all_six_126 | clean_residual_mse_recomputed | bias_gradient_norm | 126 | 0.914032 | 0.974035 | 0.914032 | 0.974035 | 2.007e-50 | 7.403e-82 | 6.020e-49 | 3.109e-80 |
| old_four_100 | error_spectral_norm | error_fro_norm_comparable | 100 | 0.96724 | 0.973285 | 0.96724 | 0.973285 | 3.720e-60 | 1.958e-64 | 2.604e-58 | 3.163e-63 |
| all_six_150 | error_spectral_norm | error_fro_norm_comparable | 150 | 0.957447 | 0.973126 | 0.957447 | 0.973126 | 9.133e-82 | 2.754e-96 | 1.918e-79 | 2.891e-94 |
| generalization_all_six_126 | clean_residual_mse_recomputed | error_fro_norm_comparable | 126 | 0.714559 | 0.971696 | 0.714559 | 0.971696 | 5.709e-21 | 1.451e-79 | 3.425e-20 | 5.079e-78 |
| all_six_150 | clean_residual_mse_recomputed | bias_gradient_norm | 150 | 0.914162 | 0.966633 | 0.914162 | 0.966633 | 6.506e-60 | 1.956e-89 | 3.416e-58 | 1.369e-87 |
| generalization_all_six_126 | error_spectral_norm | error_fro_norm_comparable | 126 | 0.943436 | 0.963363 | 0.943436 | 0.963363 | 2.679e-61 | 9.992e-73 | 2.813e-59 | 2.623e-71 |
| random_two_50 | clean_residual_mse_recomputed | bias_gradient_norm | 50 | 0.89407 | 0.95928 | 0.89407 | 0.95928 | 2.309e-18 | 5.274e-28 | 1.155e-17 | 3.164e-27 |

## Output CSVs

- `data/metric_coverage_inventory_20260614/artifact_inventory.csv`
- `data/metric_coverage_inventory_20260614/metric_coverage_summary.csv`
- `data/metric_coverage_inventory_20260614/metric_coverage_counts.csv`
- `data/metric_coverage_inventory_20260614/top_correlations_with_attack.csv`
- `data/metric_coverage_inventory_20260614/top_scalar_pairwise_correlations.csv`
- `data/metric_coverage_inventory_20260614/direction_angle_compact.csv`
- `data/metric_coverage_inventory_20260614/subspace_similarity_compact.csv`
