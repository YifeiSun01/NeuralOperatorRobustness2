# Darcy Complete 7-Model Statistics - 2026-06-13

This report fills the previously missing baseline robustness rows and the full 7-model attack20 delta metrics.

## Outputs

- `dataset52_model7_clean_attack20_delta`: `analysis_outputs/darcy_complete_7model_statistics_20260613/dataset52_model7_clean_attack20_delta.csv`
- `split_model7_clean_attack20_delta_summary`: `analysis_outputs/darcy_complete_7model_statistics_20260613/split_model7_clean_attack20_delta_summary.csv`
- `robustness_metric25_model7_sample_rows`: `analysis_outputs/darcy_complete_7model_statistics_20260613/robustness_metric25_model7_sample_rows.csv`
- `robustness_metric25_model7_summary`: `analysis_outputs/darcy_complete_7model_statistics_20260613/robustness_metric25_model7_summary.csv`
- `robustness_metric25_model7_correlations`: `analysis_outputs/darcy_complete_7model_statistics_20260613/robustness_metric25_model7_correlations.csv`
- `robustness_jacobian5_model7_sample_rows`: `analysis_outputs/darcy_complete_7model_statistics_20260613/robustness_jacobian5_model7_sample_rows.csv`
- `robustness_jacobian5_model7_summary`: `analysis_outputs/darcy_complete_7model_statistics_20260613/robustness_jacobian5_model7_summary.csv`
- `wallclock_diagnostics`: `analysis_outputs/darcy_complete_7model_statistics_20260613/wallclock_diagnostics_6training_methods.csv`
- `winner_summary`: `analysis_outputs/darcy_complete_7model_statistics_20260613/winner_summary_by_metric.csv`
- `coverage`: `analysis_outputs/darcy_complete_7model_statistics_20260613/metric_coverage_by_model.csv`
- `manifest`: `analysis_outputs/darcy_complete_7model_statistics_20260613/manifest.json`

## Coverage

| model | clean_dataset_rows | attack20_dataset_rows | attack20_sample_rows | metric25_sample_rows | jacobian5_sample_rows |
| --- | --- | --- | --- | --- | --- |
| baseline | 52 | 52 | 2600 | 25 | 5 |
| loss1 | 52 | 52 | 2600 | 25 | 5 |
| loss2 | 52 | 52 | 2600 | 25 | 5 |
| loss3 | 52 | 52 | 2600 | 25 | 5 |
| physics_loss | 52 | 52 | 2600 | 25 | 5 |
| random_clean_y | 52 | 52 | 2600 | 25 | 5 |
| random_solver_y | 52 | 52 | 2600 | 25 | 5 |

## Generalization: 52-Dataset Clean And Attack20

| model | clean_dataset_count | clean_mean_rmse | clean_mean_relative_l2 | attack20_sample_count | mean_attack_loss_gain | mean_delta_l2_rms | mean_delta_flip_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss3 | 50 | 0.000649839 | 0.0614064 | 2500 | 2.73073e-06 | 3.64266 | 0.165879 |
| loss2 | 50 | 0.000770274 | 0.072153 | 2500 | 4.06879e-06 | 3.69897 | 0.17126 |
| random_clean_y | 50 | 0.000788917 | 0.0739331 | 2500 | 3.12722e-06 | 3.64668 | 0.166509 |
| physics_loss | 50 | 0.000933881 | 0.0878201 | 2500 | 4.14508e-06 | 3.68569 | 0.170073 |
| loss1 | 50 | 0.000936259 | 0.0879285 | 2500 | 4.85208e-06 | 3.75468 | 0.176491 |
| baseline | 50 | 0.000972351 | 0.0913369 | 2500 | 4.55227e-06 | 3.8379 | 0.18402 |
| random_solver_y | 50 | 0.00101952 | 0.0959063 | 2500 | 4.59724e-06 | 3.72366 | 0.173627 |

## Metric25 Robustness Summary

| model | sample_rows | mean_attack_loss_gain | mean_jt_error_l2_norm | mean_jt_error_l2_norm_sq | mean_binary_first_order_gain | mean_sigma_max | mean_top_left_error_alignment_abs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss3 | 25 | 2.75651e-06 | 0.000105769 | 1.33657e-08 | 1.32177e-06 | 0.00207277 | 0.674547 |
| random_clean_y | 25 | 2.94164e-06 | 9.85303e-05 | 1.21563e-08 | 1.25688e-06 | 0.00181992 | 0.711743 |
| loss2 | 25 | 4.03097e-06 | 0.000116678 | 1.63285e-08 | 1.42332e-06 | 0.00172255 | 0.775131 |
| physics_loss | 25 | 4.13983e-06 | 0.000132158 | 2.02508e-08 | 1.68308e-06 | 0.00186073 | 0.812583 |
| baseline | 25 | 4.53758e-06 | 0.000116517 | 1.60225e-08 | 1.46798e-06 | 0.00169889 | 0.7462 |
| random_solver_y | 25 | 4.54417e-06 | 0.000136492 | 2.16105e-08 | 1.73765e-06 | 0.00184049 | 0.816195 |
| loss1 | 25 | 4.78855e-06 | 0.000139188 | 2.23911e-08 | 1.70468e-06 | 0.00179944 | 0.805529 |

## Jacobian5 Summary

| model | sample_rows | mean_relative_l2 | mean_jt_error_l2_norm | mean_jt_error_l2_norm_sq | mean_j_error_l2_norm | mean_spectral_norm_top_sigma |
| --- | --- | --- | --- | --- | --- | --- |
| random_clean_y | 5 | 0.0806844 | 0.000117679 | 1.75566e-08 | 4.29411e-05 | 0.00184299 |
| loss3 | 5 | 0.0785856 | 0.00012645 | 1.85496e-08 | 5.04957e-05 | 0.00210832 |
| baseline | 5 | 0.0966342 | 0.000133847 | 2.1466e-08 | 5.41212e-05 | 0.00173231 |
| loss2 | 5 | 0.0943854 | 0.000134626 | 2.24368e-08 | 5.52805e-05 | 0.00175943 |
| physics_loss | 5 | 0.0982516 | 0.000153299 | 2.75102e-08 | 6.23465e-05 | 0.00188965 |
| random_solver_y | 5 | 0.10239 | 0.00015803 | 2.93647e-08 | 6.02202e-05 | 0.00186718 |
| loss1 | 5 | 0.106734 | 0.000160822 | 3.01177e-08 | 6.57173e-05 | 0.00182829 |

## Metric25 Correlations With Attack Loss Gain

| scope | metric | label | n | pearson_r | spearman_rho |
| --- | --- | --- | --- | --- | --- |
| within_sample_centered | relative_l2 | relative L2 | 175 | 0.679605 | 0.767143 |
| within_sample_centered | clean_loss_before_attack | clean loss | 175 | 0.445828 | 0.722857 |
| within_sample_centered | jt_error_l2_norm | ||J^T e|| | 175 | 0.572887 | 0.61 |
| within_sample_centered | jt_error_l2_norm_sq | ||J^T e||^2 | 175 | 0.40812 | 0.61 |
| within_sample_centered | one_power_sigma | sigma_max | 175 | -0.497475 | -0.61 |
| within_sample_centered | sigma_power_sq_times_error_l2_sq | sigma_max^2 * ||e||^2 | 175 | 0.379771 | 0.571429 |
| within_sample_centered | sigma_power_times_error_l2 | sigma_max * ||e|| | 175 | 0.507274 | 0.571429 |
| within_sample_centered | j_error_l2_norm | ||J e|| | 175 | 0.481949 | 0.548571 |
| within_sample_centered | binary_first_order_mse_gain_positive_topk | binary first-order gain | 175 | 0.538184 | 0.547143 |
| within_sample_centered | jt_error_linf | ||J^T e||_inf | 175 | 0.531327 | 0.538571 |
| within_sample_centered | top_left_error_alignment_abs | |<e/||e||, u1>| | 175 | 0.655217 | 0.372857 |

## Wall-Clock Diagnostics

| model | epochs | true_elapsed_minutes | true_minutes_per_epoch | attack_or_random_source_minutes | optimizer_minutes | evaluation_pass_minutes | eval_split_naive_sum_minutes_do_not_use |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss1 | 1000 | 77.0066 | 0.0770066 | 8.28735 | 3.83411 | 62.9468 | 251.787 |
| loss2 | 1026 | 77.0585 | 0.0751057 | 7.09245 | 3.9191 | 63.9069 | 255.628 |
| loss3 | 1011 | 77.3917 | 0.0765496 | 7.78557 | 3.8629 | 63.6371 | 254.548 |
| physics_loss | 1040 | 76.6887 | 0.0737392 | 6.14546 | 3.9847 | 64.8039 | 259.216 |
| random_clean_y | 1100 | 303.769 | 0.276153 | 5.87139 | 9.28342 | 283.652 | 1134.61 |
| random_solver_y | 1100 | 306.838 | 0.278944 | 8.54434 | 10.0082 | 282.927 | 1131.71 |

Wall-clock note: use `true_elapsed_minutes` from each run's `summary.json`. The naive `eval_split_summary` sum repeats the same evaluation pass across ALL/train/test/generalization rows, so it is shown only as a diagnostic.

## Winners

| metric_system | scope | metric | best_model | best_value | ranking |
| --- | --- | --- | --- | --- | --- |
| generalization_clean_52datasets | generalization | clean_mean_rmse | loss3 | 0.000649839 | loss3,loss2,random_clean_y,physics_loss,loss1,baseline,random_solver_y |
| generalization_clean_52datasets | generalization | clean_mean_relative_l2 | loss3 | 0.0614064 | loss3,loss2,random_clean_y,physics_loss,loss1,baseline,random_solver_y |
| attack20_52datasets_50samples | generalization | mean_attack_loss_gain | loss3 | 2.73073e-06 | loss3,random_clean_y,loss2,physics_loss,baseline,random_solver_y,loss1 |
| attack20_52datasets_50samples | generalization | mean_delta_l2_rms | loss3 | 3.64266 | loss3,random_clean_y,physics_loss,loss2,random_solver_y,loss1,baseline |
| attack20_52datasets_50samples | generalization | mean_delta_flip_fraction | loss3 | 0.165879 | loss3,random_clean_y,physics_loss,loss2,random_solver_y,loss1,baseline |
| metric25_jacobian_attack_proxy | 25_generalization_samples | mean_attack_loss_gain | loss3 | 2.75651e-06 | loss3,random_clean_y,loss2,physics_loss,baseline,random_solver_y,loss1 |
| metric25_jacobian_attack_proxy | 25_generalization_samples | mean_jt_error_l2_norm | random_clean_y | 9.85303e-05 | random_clean_y,loss3,baseline,loss2,physics_loss,random_solver_y,loss1 |
| metric25_jacobian_attack_proxy | 25_generalization_samples | mean_jt_error_l2_norm_sq | random_clean_y | 1.21563e-08 | random_clean_y,loss3,baseline,loss2,physics_loss,random_solver_y,loss1 |
| metric25_jacobian_attack_proxy | 25_generalization_samples | mean_binary_first_order_gain | random_clean_y | 1.25688e-06 | random_clean_y,loss3,loss2,baseline,physics_loss,loss1,random_solver_y |
| metric25_jacobian_attack_proxy | 25_generalization_samples | mean_sigma_max | baseline | 0.00169889 | baseline,loss2,loss1,random_clean_y,random_solver_y,physics_loss,loss3 |
| metric25_jacobian_attack_proxy | 25_generalization_samples | mean_top_left_error_alignment_abs | loss3 | 0.674547 | loss3,random_clean_y,baseline,loss2,loss1,physics_loss,random_solver_y |
| jacobian5_probe | 5_generalization_samples | mean_relative_l2 | loss3 | 0.0785856 | loss3,random_clean_y,loss2,baseline,physics_loss,random_solver_y,loss1 |
| jacobian5_probe | 5_generalization_samples | mean_jt_error_l2_norm | random_clean_y | 0.000117679 | random_clean_y,loss3,baseline,loss2,physics_loss,random_solver_y,loss1 |
| jacobian5_probe | 5_generalization_samples | mean_jt_error_l2_norm_sq | random_clean_y | 1.75566e-08 | random_clean_y,loss3,baseline,loss2,physics_loss,random_solver_y,loss1 |
| jacobian5_probe | 5_generalization_samples | mean_j_error_l2_norm | random_clean_y | 4.29411e-05 | random_clean_y,loss3,baseline,loss2,random_solver_y,physics_loss,loss1 |
| jacobian5_probe | 5_generalization_samples | mean_spectral_norm_top_sigma | baseline | 0.00173231 | baseline,loss2,loss1,random_clean_y,random_solver_y,physics_loss,loss3 |

