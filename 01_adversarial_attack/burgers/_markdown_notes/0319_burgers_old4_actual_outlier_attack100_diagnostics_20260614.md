# Burgers Actual Old4 Outlier Re-Attack100 Diagnostics

This report uses the actual old4 source manifest, not the relabeled mixed-52 dataset names.
The selected initial conditions are the worst old `loss3_epoch1500` 20-step cases from the older `generalization_datasets/burgers` source, then re-attacked with the latest loss1/loss2/loss3/random_solver_y checkpoints for 100 steps.

## Selected Samples

| sample | rank | dataset | idx | old loss3 e1500 20-step increase |
|---|---:|---|---:|---:|
| FarCenteredScale2p5ShiftM0p25_idx156 | 1 | `burgers_far_centered_scale_shift_scale2p5_shiftm0p25` | 156 | 0.721661 |
| FarSignCenteredScale1Shift0_idx4 | 2 | `burgers_far_sign_centered_scale1_shift0` | 4 | 0.694259 |
| FarPositiveShiftScale1Shift1_idx41 | 3 | `burgers_far_positive_shift_scale1_shift1` | 41 | 0.683243 |
| FarCenteredScale2Shift0_idx125 | 4 | `burgers_far_centered_scale_shift_scale2_shift0` | 125 | 0.378037 |
| FarNegativeShiftScale1Shift1_idx17 | 5 | `burgers_far_negative_shift_scale1_shift1` | 17 | 0.327694 |
| FarCenteredScale1ShiftM0p4_idx23 | 6 | `burgers_far_centered_scale_shift_scale1_shiftm0p4` | 23 | 0.305076 |

## Latest Attack100 Loss3 vs Random Solver

| sample | loss3 inc | random solver inc | loss3 - random solver | winner |
|---|---:|---:|---:|---|
| FarCenteredScale2p5ShiftM0p25_idx156 | 0.517902 | 0.754974 | -0.237072 | loss3 |
| FarSignCenteredScale1Shift0_idx4 | 0.774578 | 0.698812 | 0.0757658 | random_solver_y |
| FarPositiveShiftScale1Shift1_idx41 | 0.592461 | 0.492625 | 0.0998366 | random_solver_y |
| FarCenteredScale2Shift0_idx125 | 0.279291 | 0.42586 | -0.146569 | loss3 |
| FarNegativeShiftScale1Shift1_idx17 | 0.263815 | 0.327248 | -0.0634333 | loss3 |
| FarCenteredScale1ShiftM0p4_idx23 | 0.235381 | 0.310206 | -0.0748254 | loss3 |

## Model Means On These Samples

| model | mean initial | mean final | mean increase | median increase | wins lower increase |
|---|---:|---:|---:|---:|---:|
| loss1 | 0.762272 | 1.34365 | 0.581382 | 0.52488 | 0 |
| loss2 | 0.452974 | 0.843258 | 0.390284 | 0.34746 | 2 |
| loss3 | 0.393952 | 0.837856 | 0.443905 | 0.398596 | 4 |
| random_solver_y | 0.689156 | 1.19078 | 0.501621 | 0.459242 | 0 |

## Local Diagnostics Means

| model | clean residual MSE | J^T error norm | J_error delta L2 | spectral norm estimate | attack-delta/top-SV abs cos | attack-delta/J^T-error abs cos |
|---|---:|---:|---:|---:|---:|---:|
| loss1 | 0.762272 | 67.797 | 17.9895 | 8.18774 | 0.253615 | 0.9462 |
| loss2 | 0.452974 | 44.0573 | 15.7133 | 7.81024 | 0.247078 | 0.94431 |
| loss3 | 0.393952 | 39.892 | 14.7939 | 7.59546 | 0.220505 | 0.948319 |
| random_solver_y | 0.689156 | 55.9101 | 15.3641 | 7.55062 | 0.224739 | 0.979322 |

## Correlations With Attack100 Increase

| x metric | n | Pearson r | Spearman r |
|---|---:|---:|---:|
| clean_residual_mse | 24 | 0.8036 | 0.8383 |
| clean_residual_l2 | 24 | 0.8239 | 0.8383 |
| bias_gradient_norm_jt_error | 24 | 0.8817 | 0.8861 |
| j_error_delta_l2 | 24 | 0.6848 | 0.6565 |
| linearized_endpoint_increase_mse | 24 | 0.8592 | 0.8174 |
| linearized_residual_movement_mse | 24 | 0.6912 | 0.6565 |
| error_spectral_norm_power_iter | 24 | 0.5526 | 0.4322 |
| attack_delta_top_error_right_abs_cos | 24 | 0.4758 | 0.4017 |
| attack_delta_bias_gradient_abs_cos | 24 | 0.06808 | 0.2078 |

## Figures

Preferred descriptive-label versions:

- `visualizations/burgers_old4_actual_outliers_latest_attack100_dense_bundle_20260614/comparison_dense/outlier_group_actual_old4/burgers_old4_actual_outliers_latest_loss1_loss2_loss3_random_solver_y_attack100_four_column_log_y_descriptive_labels.png`
- `visualizations/burgers_old4_actual_outliers_latest_attack100_dense_bundle_20260614/comparison_dense/outlier_group_actual_old4/burgers_old4_actual_outliers_latest_loss1_loss2_loss3_random_solver_y_attack100_four_column_linear_y_descriptive_labels.png`

Original compact-label versions:

- `visualizations/burgers_old4_actual_outliers_latest_attack100_dense_bundle_20260614/comparison_dense/outlier_group_actual_old4/burgers_old4_actual_outliers_latest_loss1_loss2_loss3_random_solver_y_attack100_four_column_log_y.png`
- `visualizations/burgers_old4_actual_outliers_latest_attack100_dense_bundle_20260614/comparison_dense/outlier_group_actual_old4/burgers_old4_actual_outliers_latest_loss1_loss2_loss3_random_solver_y_attack100_four_column_linear_y.png`

## Caveat

The local singular value/vector here is a power-iteration estimate for the exact selected samples. It is not a full top20/top100 SVD table.
