# Darcy All-Model Full Statistics - 2026-06-13

This report aggregates the requested Darcy statistics across 52 datasets and 7 models.

Important: clean metrics are complete for all 7 models. Existing robustness artifacts cover 6 models; baseline robustness was not computed in the previous attack/Jacobian runs, so those baseline robustness fields are marked missing instead of being fabricated.

## Output Tables

- `dataset52_model7_clean_attack20`: `analysis_outputs/darcy_all_model_full_statistics_20260613/dataset52_model7_clean_attack20.csv`
- `split_model7_clean_attack20_summary`: `analysis_outputs/darcy_all_model_full_statistics_20260613/split_model7_clean_attack20_summary.csv`
- `metric25_model7_summary`: `analysis_outputs/darcy_all_model_full_statistics_20260613/robustness_metric25_model7_summary.csv`
- `metric25_model6_sample_rows`: `analysis_outputs/darcy_all_model_full_statistics_20260613/robustness_metric25_model6_sample_rows.csv`
- `metric25_correlations`: `analysis_outputs/darcy_all_model_full_statistics_20260613/robustness_metric25_correlations.csv`
- `jacobian5_model7_summary`: `analysis_outputs/darcy_all_model_full_statistics_20260613/robustness_jacobian5_model7_summary.csv`
- `winner_summary`: `analysis_outputs/darcy_all_model_full_statistics_20260613/winner_summary_by_metric.csv`
- `coverage`: `analysis_outputs/darcy_all_model_full_statistics_20260613/metric_coverage_by_model.csv`
- `manifest`: `analysis_outputs/darcy_all_model_full_statistics_20260613/manifest.json`

## Coverage

| model | clean_52_dataset_metrics | attack20_52_dataset_metrics | metric25_sample_rows | jacobian5_model_mean_available |
| --- | --- | --- | --- | --- |
| baseline | 52 | 0 | 0 | False |
| loss1 | 52 | 52 | 25 | True |
| loss2 | 52 | 52 | 25 | True |
| loss3 | 52 | 52 | 25 | True |
| physics_loss | 52 | 52 | 25 | True |
| random_clean_y | 52 | 52 | 25 | True |
| random_solver_y | 52 | 52 | 25 | True |

## Generalization Clean Metrics, 7 Models

| model | clean_dataset_count | clean_mean_relative_l2 | clean_mean_rmse | attack20_available |
| --- | --- | --- | --- | --- |
| loss3 | 50 | 0.0614064 | 0.000649839 | True |
| loss2 | 50 | 0.072153 | 0.000770274 | True |
| random_clean_y | 50 | 0.0739331 | 0.000788917 | True |
| physics_loss | 50 | 0.0878201 | 0.000933881 | True |
| loss1 | 50 | 0.0879285 | 0.000936259 | True |
| baseline | 50 | 0.0913369 | 0.000972351 | False |
| random_solver_y | 50 | 0.0959063 | 0.00101952 | True |

## Attack20 Generalization Robustness

| model | attack20_dataset_count | attack20_sample_count | mean_clean_loss | mean_attack_loss_gain | mean_adv_loss | mean_attack_loss_gain_relative | attack20_available | attack20_missing_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loss3 | 50 | 2500 | 4.76911e-07 | 2.73073e-06 | 3.20764e-06 | 11.5878 | True |  |
| random_clean_y | 50 | 2500 | 5.26291e-07 | 3.12722e-06 | 3.65351e-06 | 13.6122 | True |  |
| loss2 | 50 | 2500 | 7.15598e-07 | 4.06879e-06 | 4.78439e-06 | 15.5635 | True |  |
| physics_loss | 50 | 2500 | 7.90779e-07 | 4.14508e-06 | 4.93586e-06 | 15.7411 | True |  |
| random_solver_y | 50 | 2500 | 8.61425e-07 | 4.59724e-06 | 5.45866e-06 | 15.9069 | True |  |
| loss1 | 50 | 2500 | 9.7439e-07 | 4.85208e-06 | 5.82647e-06 | 15.5734 | True |  |
| baseline |  |  |  |  |  |  | False | not_computed_in_existing_artifacts |

## Metric25 Robustness Proxy Summary

| model | mean_attack_gain | mean_jt_error | mean_binary_first_order | mean_sigma | metric_available | missing_reason |
| --- | --- | --- | --- | --- | --- | --- |
| loss3 | 2.75651e-06 | 0.000105769 | 1.32177e-06 | 0.00207277 | True |  |
| random_clean_y | 2.94164e-06 | 9.85303e-05 | 1.25688e-06 | 0.00181992 | True |  |
| loss2 | 4.03097e-06 | 0.000116678 | 1.42332e-06 | 0.00172255 | True |  |
| physics_loss | 4.13983e-06 | 0.000132158 | 1.68308e-06 | 0.00186073 | True |  |
| random_solver_y | 4.54417e-06 | 0.000136492 | 1.73765e-06 | 0.00184049 | True |  |
| loss1 | 4.78855e-06 | 0.000139188 | 1.70468e-06 | 0.00179944 | True |  |
| baseline |  |  |  |  | False | not_computed_in_existing_artifacts |

## Jacobian5 Probe Summary

| model | mean_relative_l2 | mean_jt_error_l2_norm | mean_jt_error_l2_norm_sq | mean_j_error_l2_norm | mean_spectral_norm_top_sigma | metric_available | missing_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_clean_y | 0.0806844 | 0.000117679 | 1.75566e-08 | 4.29411e-05 | 0.00184299 | True |  |
| loss3 | 0.0785856 | 0.00012645 | 1.85496e-08 | 5.04957e-05 | 0.00210832 | True |  |
| loss2 | 0.0943854 | 0.000134626 | 2.24368e-08 | 5.52805e-05 | 0.00175943 | True |  |
| physics_loss | 0.0982516 | 0.000153299 | 2.75102e-08 | 6.23465e-05 | 0.00188965 | True |  |
| random_solver_y | 0.10239 | 0.00015803 | 2.93647e-08 | 6.02202e-05 | 0.00186718 | True |  |
| loss1 | 0.106734 | 0.000160822 | 3.01177e-08 | 6.57173e-05 | 0.00182829 | True |  |
| baseline |  |  |  |  |  | False | not_computed_in_existing_artifacts |

## Winner Summary

| metric_system | scope | metric | best_model | best_value | model_count_available | ranking |
| --- | --- | --- | --- | --- | --- | --- |
| attack20_52datasets | all | mean_attack_loss_gain | loss3 | 2.69124e-06 | 6 | loss3,random_clean_y,loss2,physics_loss,random_solver_y,loss1 |
| attack20_52datasets | all | mean_adv_loss | loss3 | 3.15314e-06 | 6 | loss3,random_clean_y,loss2,physics_loss,random_solver_y,loss1 |
| attack20_52datasets | all | mean_attack_loss_gain_relative | loss3 | 12.2067 | 6 | loss3,random_clean_y,loss2,random_solver_y,loss1,physics_loss |
| attack20_52datasets | train | mean_attack_loss_gain | random_clean_y | 3.54745e-07 | 6 | random_clean_y,loss2,random_solver_y,loss3,loss1,physics_loss |
| attack20_52datasets | train | mean_adv_loss | random_clean_y | 3.94627e-07 | 6 | random_clean_y,loss2,random_solver_y,loss3,loss1,physics_loss |
| attack20_52datasets | train | mean_attack_loss_gain_relative | random_clean_y | 11.9579 | 6 | random_clean_y,loss2,loss3,random_solver_y,loss1,physics_loss |
| attack20_52datasets | test | mean_attack_loss_gain | random_clean_y | 3.22537e-07 | 6 | random_clean_y,loss2,random_solver_y,loss3,physics_loss,loss1 |
| attack20_52datasets | test | mean_adv_loss | random_clean_y | 3.68863e-07 | 6 | random_clean_y,loss2,random_solver_y,loss3,physics_loss,loss1 |
| attack20_52datasets | test | mean_attack_loss_gain_relative | random_clean_y | 8.25922 | 6 | random_clean_y,loss2,loss3,random_solver_y,loss1,physics_loss |
| generalization_clean | generalization | clean_mean_relative_l2 | loss3 | 0.0614064 | 7 | loss3,loss2,random_clean_y,physics_loss,loss1,baseline,random_solver_y |
| generalization_clean | generalization | clean_mean_rmse | loss3 | 0.000649839 | 7 | loss3,loss2,random_clean_y,physics_loss,loss1,baseline,random_solver_y |
| attack20_52datasets | generalization | mean_attack_loss_gain | loss3 | 2.73073e-06 | 6 | loss3,random_clean_y,loss2,physics_loss,random_solver_y,loss1 |
| attack20_52datasets | generalization | mean_adv_loss | loss3 | 3.20764e-06 | 6 | loss3,random_clean_y,loss2,physics_loss,random_solver_y,loss1 |
| attack20_52datasets | generalization | mean_attack_loss_gain_relative | loss3 | 11.5878 | 6 | loss3,random_clean_y,loss2,loss1,physics_loss,random_solver_y |
| metric25_attack_jacobian_proxy | 25_generalization_samples | mean_attack_gain | loss3 | 2.75651e-06 | 6 | loss3,random_clean_y,loss2,physics_loss,random_solver_y,loss1 |
| metric25_attack_jacobian_proxy | 25_generalization_samples | mean_jt_error | random_clean_y | 9.85303e-05 | 6 | random_clean_y,loss3,loss2,physics_loss,random_solver_y,loss1 |
| metric25_attack_jacobian_proxy | 25_generalization_samples | mean_binary_first_order | random_clean_y | 1.25688e-06 | 6 | random_clean_y,loss3,loss2,physics_loss,loss1,random_solver_y |
| metric25_attack_jacobian_proxy | 25_generalization_samples | mean_sigma | loss2 | 0.00172255 | 6 | loss2,loss1,random_clean_y,random_solver_y,physics_loss,loss3 |
| jacobian5_probe | 5_generalization_samples | mean_relative_l2 | loss3 | 0.0785856 | 6 | loss3,random_clean_y,loss2,physics_loss,random_solver_y,loss1 |
| jacobian5_probe | 5_generalization_samples | mean_jt_error_l2_norm | random_clean_y | 0.000117679 | 6 | random_clean_y,loss3,loss2,physics_loss,random_solver_y,loss1 |
| jacobian5_probe | 5_generalization_samples | mean_jt_error_l2_norm_sq | random_clean_y | 1.75566e-08 | 6 | random_clean_y,loss3,loss2,physics_loss,random_solver_y,loss1 |
| jacobian5_probe | 5_generalization_samples | mean_j_error_l2_norm | random_clean_y | 4.29411e-05 | 6 | random_clean_y,loss3,loss2,random_solver_y,physics_loss,loss1 |
| jacobian5_probe | 5_generalization_samples | mean_spectral_norm_top_sigma | loss2 | 0.00175943 | 6 | loss2,loss1,random_clean_y,random_solver_y,physics_loss,loss3 |

