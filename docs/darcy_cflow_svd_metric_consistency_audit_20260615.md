# Darcy CFlow SVD Metric Consistency Audit - 2026-06-15

Source SVD table: `outputs/darcy_cflow_final_robustness_20260615/data/svd_jacobian_metrics.csv`
Source attack table: `outputs/darcy_cflow_final_robustness_20260615/data/robustness_attack_52datasets_samples.csv`

Issue count: `0`

## Checks
| check | ok | detail |
|---|---:|---|
| svd_row_count_175 | True | rows=175 |
| attack_row_count_18200 | True | rows=18200 |
| svd_methods_7 | True | ['baseline', 'loss1', 'loss2', 'loss3', 'physics_loss', 'random_clean', 'random_solver'] |
| attack_methods_7 | True | ['baseline', 'loss1', 'loss2', 'loss3', 'physics_loss', 'random_clean', 'random_solver'] |
| attack_steps_all_50 | True | [50] |
| old_20260607_absent | True |  |
| smoke_absent_in_svd_attack_sample | True |  |
| generalization_dataset_count_50 | True | gen_count=50 |
| generalization_prefix_20260611 | True |  |
| svd_split_counts_each_model_21_2_2 | True | split          generalization  test  train<br>method                                    <br>baseline                   21     2      2<br>loss1                      21     2      2<br>loss2                      21     2      2<br>loss3                      21     2      2<br>physics_loss               21     2      2<br>random_clean               21     2      2<br>random_solver              21     2      2 |
| svd_attack_join_no_missing | True | missing=0 |
| svd_attack_join_losses_match | True | max_diffs={'clean_loss': 0.0, 'attack_loss_increase': 0.0, 'attack_relative_increase': 0.0} |
| vector_npz_no_missing | True | missing_npz=0 |
| vector_npz_required_keys_present | True |  |
| vector_npz_identity_matches_csv | True |  |
| vector_npz_recomputed_metrics_match | True | max_abs={'error_l2_norm': 6.559748771950424e-09, 'jt_error_l2_norm': 1.218386280537817e-11, 'block2_sigma1': 9.90960785651751e-17, 'cos_singular_jt_error': 1.1102230246251565e-16, 'angle_singular_jt_error_deg': 2.842170943040401e-14, 'corr_singular_jt_error': 1.1102230246251565e-16, 'cos_singular_attack_delta': 9.71445146547012e-17, 'angle_singular_attack_delta_deg': 2.842170943040401e-14, 'corr_singular_attack_delta': 9.71445146547012e-17, 'cos_jt_error_attack_delta': 9.71445146547012e-17, 'angle_jt_error_attack_delta_deg': 1.4210854715202004e-14, 'corr_jt_error_attack_delta': 9.71445146547012e-17, 'topk_subspace_cos_jt_error': 4.133121328520062e-07, 'topk_subspace_angle_jt_error_deg': 0.00017188175659033078, 'topk_subspace_cos_attack_delta': 1.078055975045622e-07, 'topk_subspace_angle_attack_delta_deg': 7.82013179190244e-06}; tolerance_large={} |
| vector_attack_delta_matches_delta_npz | True | delta_missing=0, max_delta_diff=0.0 |

## Core Correlations Recomputed
| subset | x | y | n | Pearson | Spearman |
|---|---|---|---:|---:|---:|
| all175 | attack_loss_increase | jt_error_l2_norm | 175 | 0.382521 | 0.429304 |
| all175 | attack_loss_increase | block2_sigma1 | 175 | -0.319092 | -0.215947 |
| all175 | jt_error_l2_norm | block2_sigma1 | 175 | 0.357818 | 0.408939 |
| gen147 | attack_loss_increase | jt_error_l2_norm | 147 | 0.0649105 | 0.0952595 |
| gen147 | attack_loss_increase | block2_sigma1 | 147 | -0.816533 | -0.726549 |
| gen147 | jt_error_l2_norm | block2_sigma1 | 147 | 0.058778 | 0.133322 |

## Lower-Is-Better Mean Winners
| metric | winner |
|---|---|
| attack_loss_increase | loss3 |
| error_l2_norm | loss3 |
| jt_error_l2_norm | loss3 |
| block2_sigma1 | baseline |
| sigma_input_right | baseline |

## Method Means
| method | attack_loss_increase | error_l2_norm | jt_error_l2_norm | block2_sigma1 | sigma_input_right |
|---|---:|---:|---:|---:|---:|
| Physics Loss | 3.7135e-06 | 0.0610387 | 0.000103396 | 0.0018175 | 0.0018555 |
| baseline | 8.94546e-06 | 0.0838341 | 9.29277e-05 | 0.00124975 | 0.00127159 |
| loss1 | 3.64937e-06 | 0.0609317 | 9.9221e-05 | 0.0017809 | 0.00182142 |
| loss2 | 3.39291e-06 | 0.0576766 | 9.22537e-05 | 0.00171212 | 0.00175079 |
| loss3 | 1.71618e-06 | 0.0388067 | 6.1568e-05 | 0.0022889 | 0.00235389 |
| random clean | 3.45293e-06 | 0.0593247 | 9.11171e-05 | 0.0016882 | 0.00172493 |
| random solver | 4.43937e-06 | 0.069976 | 0.00012357 | 0.00185964 | 0.00189828 |