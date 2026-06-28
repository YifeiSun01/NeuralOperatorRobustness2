# Darcy CFlow/SIR20 Final Robustness Records - 20260615

这个目录是整理包，放在 timematched organized release 的 `data/` 里面。只整理已有结果，不重跑训练、不重跑 attack、不画图。

## Sanity Checks

- 只使用 `generalization_datasets_darcy_binary_loss3targeted_20260611` 的 generalization 结果。
- CSV source tables 中未发现 `lossdrop50_selected` 或 `generalization_datasets_darcy_lossdrop50_selected_20260607`。
- 未使用 smoke / partial_smoke 结果；`provenance.json` 里的 `smoke_token_found=false` 只是审计字段。
- attack source 和 residual aligned source 的 `attack_steps` 全是 `50`。
- 统一七个显示名：baseline, loss1, loss2, loss3, Physics Loss, random clean, random solver。

## Source Tables Copied

| source | copied_to | rows | sha256_16 | status |
| --- | --- | --- | --- | --- |
| outputs/darcy_cflow_final_robustness_20260615/data/robustness_attack_52datasets_samples.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/robustness_attack_52datasets_samples.csv | 18200 | abbc0758d8d43efc | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/svd_jacobian_metrics.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/svd_jacobian_metrics.csv | 175 | ddd4c441b4136a74 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/attack50_summary_by_dataset_model.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/attack50_summary_by_dataset_model.csv | 364 | e844f2f8b0ed77f2 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/attack50_summary_by_model_split.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/attack50_summary_by_model_split.csv | 21 | 1f324ab14bf8f3b3 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/winner_summary_by_metric.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/winner_summary_by_metric.csv | 26 | d1de3f260d75ab92 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/svd_scalar_correlations.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/svd_scalar_correlations.csv | 56 | c9230ffa23374f8f | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/attack50_52dataset_7model_mean_std.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/attack50_52dataset_7model_mean_std.csv | 364 | 83f571dd7bd97904 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/attack50_by_model_split_mean_std.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/attack50_by_model_split_mean_std.csv | 21 | 19bb9a0f3e748a66 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/provenance.json | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/provenance.json |  | b6fa503d54589d47 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_attack_summary_20260615.md | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/residual_jacobian_attack_summary_20260615.md |  | f00e0838a52c9506 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_svd_25samples_7models.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/residual_jacobian_svd_25samples_7models.csv | 175 | cd8c945280a04b57 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_svd_by_model.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/residual_jacobian_svd_by_model.csv | 7 | eb5775ba4085d61a | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_attack_aligned_25samples_7models.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/residual_jacobian_attack_aligned_25samples_7models.csv | 175 | f9cd8ddbd17bdc25 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_attack_by_model_25samples_20260615.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/residual_jacobian_attack_by_model_25samples_20260615.csv | 7 | 6930c250898621b8 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_attack_correlations_20260615.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/residual_jacobian_attack_correlations_20260615.csv | 27 | 4966f3621883248b | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/residual_jacobian_attack_winner_counts_20260615.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/residual_jacobian_attack_winner_counts_20260615.csv | 126 | c0e55d166b551b6e | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/model_solver_block2_subspace_similarity_by_model.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/model_solver_block2_subspace_similarity_by_model.csv | 7 | 97cd5a481591abb0 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/model_solver_block2_subspace_similarity_25samples_7models.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/model_solver_block2_subspace_similarity_25samples_7models.csv | 175 | 37880e025496a058 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_jacobian_25sample_7model_mean_std.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/svd_jacobian_25sample_7model_mean_std.csv | 7 | e69df977237ce449 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_block2_top10_singular_values_by_model_mean_std.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/svd_block2_top10_singular_values_by_model_mean_std.csv | 7 | 8edf9e1666e61232 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_lossincrease_jt_spectral_correlations_20260615.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/svd_lossincrease_jt_spectral_correlations_20260615.csv | 54 | 796e818dc7e032f3 | copied |
| outputs/darcy_cflow_final_robustness_20260615/data/final_metric_mean_std_20260615/svd_vector_cosine_angle_summary_20260615.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/svd_vector_cosine_angle_summary_20260615.csv | 80 | b70cd9bc00f8b6de | copied |
| outputs/darcy_cflow_final_robustness_20260615/reports/per_dataset_and_fixed25_tables_20260615.md | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/per_dataset_and_fixed25_tables_20260615.md |  | cdfb273a70271eb6 | copied |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_advantage_metric_tables_20260615/clean_52dataset_7model_rmse_relative_l2_long.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/clean_52dataset_7model_rmse_relative_l2_long.csv | 728 | d6fc94654e420e76 | copied |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_advantage_metric_tables_20260615/clean_52dataset_7model_rmse_wide.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/clean_52dataset_7model_rmse_wide.csv | 52 | 8ca523917445870f | copied |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/loss3_advantage_metric_tables_20260615/clean_52dataset_7model_relative_l2_wide.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/clean_52dataset_7model_relative_l2_wide.csv | 52 | 87921f37adf32e8a | copied |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/clean_per_sample_ttests_20260615/clean_per_sample_rmse_relative_l2_52datasets_7models.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/clean_per_sample_rmse_relative_l2_52datasets_7models.csv | 20860 | 23ad8d5ead8a072e | copied |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/clean_per_sample_ttests_20260615/clean_rmse_relative_l2_per_dataset_first_vs_second_ttests.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/clean_rmse_relative_l2_per_dataset_first_vs_second_ttests.csv | 104 | 003ae47c26d0eb65 | copied |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/clean_per_sample_ttests_20260615/clean_rmse_relative_l2_per_dataset_ttest_summary.csv | outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/source_tables/clean_rmse_relative_l2_per_dataset_ttest_summary.csv | 4 | abef85664667f5a2 | copied |

## Derived CSV Tables

| file | rows |
| --- | --- |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/attack50_52dataset_adv_loss_mean_wide_7models.csv | 52 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/attack50_52dataset_clean_loss_mean_wide_7models.csv | 52 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/attack50_52dataset_delta_l2_rms_mean_wide_7models.csv | 52 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/attack50_52dataset_delta_linf_mean_wide_7models.csv | 52 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/attack50_52dataset_loss_increase_mean_wide_7models.csv | 52 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/attack50_52dataset_relative_increase_mean_wide_7models.csv | 52 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/attack50_by_model_mean_std_all52_and_generalization50.csv | 14 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/clean_52dataset_rmse_relative_l2_by_dataset_7models.csv | 52 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/clean_by_model_mean_std_all52_and_generalization50.csv | 14 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/fixed25_adv_loss_wide_7models.csv | 25 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/fixed25_clean_loss_wide_7models.csv | 25 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/fixed25_loss_increase_wide_7models.csv | 25 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/fixed25_model_solver_topk_subspace_similarity_long_25samples_7models.csv | 175 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/fixed25_model_solver_topk_subspace_winners_by_sample.csv | 25 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/fixed25_relative_increase_wide_7models.csv | 25 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/fixed25_residual_attack_metrics_long_25samples_7models.csv | 175 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/fixed25_residual_block2_sigma1_wide_7models.csv | 25 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/fixed25_residual_error_l2_norm_wide_7models.csv | 25 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/fixed25_residual_jt_error_l2_norm_wide_7models.csv | 25 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/model_only_jacobian_svd_by_model_with_top10.csv | 7 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/residual_jacobian_by_model_mean_std.csv | 7 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/residual_vector_angle_summary_by_model.csv | 7 |
| outputs/darcy_cflow_timematched_organized_release_20260614/data/final_robustness_records_20260615/derived_tables/robustness_scalar_correlations_model_only_and_residual.csv | 14 |

## Key Tables Preview

### Clean by-model mean/std
| scope | model | rmse_mean | rmse_std | relative_l2_mean | relative_l2_std |
| --- | --- | --- | --- | --- | --- |
| all52 | baseline | 0.000940493 | 0.000351787 | 0.088625 | 0.028059 |
| all52 | loss1 | 0.000906385 | 0.00033568 | 0.085434 | 0.026658 |
| all52 | loss2 | 0.000842542 | 0.000320871 | 0.079345 | 0.02559 |
| all52 | loss3 | 0.000580133 | 0.00018268 | 0.055213 | 0.014064 |
| all52 | Physics Loss | 0.000908782 | 0.000324241 | 0.08575 | 0.025651 |
| all52 | random clean | 0.000855774 | 0.000334309 | 0.080533 | 0.02683 |
| all52 | random solver | 0.001032 | 0.000343886 | 0.097516 | 0.027027 |
| generalization50 | baseline | 0.000972351 | 0.000319174 | 0.091337 | 0.024984 |
| generalization50 | loss1 | 0.000936259 | 0.000305897 | 0.087929 | 0.023958 |
| generalization50 | loss2 | 0.000870154 | 0.000294811 | 0.081638 | 0.023274 |
| generalization50 | loss3 | 0.000594572 | 0.000170592 | 0.056154 | 0.013427 |
| generalization50 | Physics Loss | 0.000938999 | 0.000291751 | 0.088293 | 0.022619 |
| generalization50 | random clean | 0.000883968 | 0.00030851 | 0.082881 | 0.024531 |
| generalization50 | random solver | 0.001067 | 0.000299545 | 0.100569 | 0.022613 |

### Residual Jacobian by-model mean/std
| method_display | clean_loss_mean | clean_loss_std | adv_loss_mean | adv_loss_std | loss_increase_mean | loss_increase_std | relative_increase_mean | relative_increase_std | residual_sigma1_mean | residual_sigma1_std | residual_error_l2_mean | residual_error_l2_std | residual_jt_error_norm_mean | residual_jt_error_norm_std | residual_top10_vs_delta_angle_mean | residual_jt_vs_delta_angle_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Physics Loss | 4.51213e-07 | 4.00238e-07 | 4.16471e-06 | 1.62722e-06 | 3.7135e-06 | 1.45613e-06 | 16.216225 | 18.84153 | 0.002042 | 0.000538095 | 0.050834 | 0.026534 | 0.000112514 | 8.2329e-05 | 76.822197 | 81.329298 |
| baseline | 9.10929e-07 | 7.56133e-07 | 9.85639e-06 | 3.54243e-06 | 8.94546e-06 | 3.28015e-06 | 16.851448 | 12.231108 | 0.002464 | 0.000623642 | 0.073329 | 0.035419 | 0.000193139 | 0.000128443 | 76.041564 | 83.401386 |
| loss1 | 4.62444e-07 | 4.43906e-07 | 4.11181e-06 | 1.79743e-06 | 3.64937e-06 | 1.63877e-06 | 14.050288 | 15.518157 | 0.002071 | 0.00054934 | 0.051334 | 0.027118 | 0.000114247 | 8.76933e-05 | 75.280478 | 80.053342 |
| loss2 | 4.02127e-07 | 3.62922e-07 | 3.79503e-06 | 1.44943e-06 | 3.39291e-06 | 1.31872e-06 | 17.97678 | 20.99886 | 0.002077 | 0.000536368 | 0.048147 | 0.024733 | 0.000106839 | 8.05705e-05 | 77.059007 | 81.330894 |
| loss3 | 1.75215e-07 | 1.19519e-07 | 1.89139e-06 | 8.78326e-07 | 1.71618e-06 | 8.54329e-07 | 17.566485 | 16.897701 | 0.001702 | 0.000431903 | 0.033421 | 0.012457 | 4.72575e-05 | 3.35266e-05 | 71.497027 | 80.269652 |
| random clean | 4.44798e-07 | 4.35741e-07 | 3.89773e-06 | 1.69129e-06 | 3.45293e-06 | 1.55279e-06 | 16.081716 | 21.553389 | 0.002143 | 0.000564429 | 0.050325 | 0.026634 | 0.000113273 | 9.13984e-05 | 76.231923 | 81.073916 |
| random solver | 6.07172e-07 | 4.97758e-07 | 5.04655e-06 | 1.76251e-06 | 4.43937e-06 | 1.54137e-06 | 19.589689 | 32.707558 | 0.002009 | 0.000528757 | 0.0594 | 0.029904 | 0.000131033 | 8.93149e-05 | 76.89835 | 81.44938 |

### Robustness scalar correlations
| operator | scope | n | x | y | pearson_r | spearman_rho |
| --- | --- | --- | --- | --- | --- | --- |
| model_only_J_model | all_25_samples_all7_models | 175 | attack_loss_increase | jt_error_l2_norm | 0.382521 | 0.429304 |
| model_only_J_model | all_25_samples_all7_models | 175 | attack_loss_increase | block2_sigma1 | -0.319092 | -0.215947 |
| model_only_J_model | all_25_samples_all7_models | 175 | jt_error_l2_norm | block2_sigma1 | 0.357818 | 0.408939 |
| model_only_J_model | generalization21_all7_models | 147 | attack_loss_increase | jt_error_l2_norm | 0.064911 | 0.09526 |
| model_only_J_model | generalization21_all7_models | 147 | attack_loss_increase | block2_sigma1 | -0.816533 | -0.726549 |
| model_only_J_model | generalization21_all7_models | 147 | jt_error_l2_norm | block2_sigma1 | 0.058778 | 0.133322 |
| residual_J_model_minus_J_solver | all_25_samples_x_7_models | 175 | loss_increase | residual_jt_error_l2_norm | 0.504257 | 0.502891 |
| residual_J_model_minus_J_solver | all_25_samples_x_7_models | 175 | loss_increase | residual_block2_sigma1 | 0.561659 | 0.438184 |
| residual_J_model_minus_J_solver | all_25_samples_x_7_models | 175 | loss_increase | residual_error_l2_norm | 0.534601 | 0.495813 |
| residual_J_model_minus_J_solver | all_25_samples_x_7_models | 175 | residual_jt_error_l2_norm | residual_block2_sigma1 | 0.909114 | 0.945159 |
| residual_J_model_minus_J_solver | generalization_21_samples_x_7_models | 147 | loss_increase | residual_jt_error_l2_norm | 0.353294 | 0.20962 |
| residual_J_model_minus_J_solver | generalization_21_samples_x_7_models | 147 | loss_increase | residual_block2_sigma1 | 0.346862 | 0.114519 |
| residual_J_model_minus_J_solver | generalization_21_samples_x_7_models | 147 | loss_increase | residual_error_l2_norm | 0.345888 | 0.201369 |
| residual_J_model_minus_J_solver | generalization_21_samples_x_7_models | 147 | residual_jt_error_l2_norm | residual_block2_sigma1 | 0.918346 | 0.916724 |

## Epsilon Sweep Addendum - 20260615

Additional attack-budget sweep records are stored in this same release bundle.

| file/directory | contents |
| --- | --- |
| `epsilon_sweep_summary.md` | Markdown summary for epsilon 0.1x, 0.2x, 0.5x, 1x, 5x |
| `darcy_cflow_epsilon_sweep_full_record_20260615.md` | Full Markdown record for all 8 epsilon budgets, including Loss3 checks, scalar correlations, vector similarities, and source paths |
| `epsilon_sweep_residual_correlations.csv` | Pearson/Spearman correlations between attack loss increase and residual scalar metrics |
| `epsilon_sweep_by_model_mean_std.csv` | By-model attack mean/std across epsilon budgets |
| `epsilon_sweep_vector_angles_rows.csv` | Row-level angles between new attack deltas and residual vectors/subspaces |
| `epsilon_sweep_raw_attack_sources_20260615/` | Raw rerun attack CSVs, delta NPZ files, and logs for 0.1x, 0.2x, 0.5x, and 5x |

The 1x epsilon result in the sweep comes from the existing final attack50 source
table, filtered to the fixed 25 samples x 7 models. The accidental interrupted
`eps_1x` rerun is documented in `epsilon_sweep_raw_attack_sources_20260615/logs`
but is not used in any final table.

## Note

没有复制大型 `.npz` 向量/雅可比数组目录；本包保存的是上面报告使用的 CSV/MD/JSON 原始表和派生明细 CSV。epsilon sweep addendum 额外复制了 attack delta NPZ，因为体量很小。如果需要连大型 Jacobian/vector `.npz` 一起复制，可以再单独做一个 heavy bundle。
