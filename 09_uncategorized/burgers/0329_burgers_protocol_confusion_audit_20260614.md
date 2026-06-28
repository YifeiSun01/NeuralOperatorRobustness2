# Burgers Protocol Confusion Audit, 20260614

Generated: 2026-06-14T14:34:19+00:00

This audit reads existing artifacts only. It does not rerun training, attacks, Jacobian, SVD, or plotting.

## Plain Conclusion

- The confusing part is real: the recovered six-model full-52 attack table is a mixed historical/current table. Its old4 rows use the historical first/master attack run, while random rows use the current solver7860/clean8000 run.
- For generalization rows in that recovered table, old4 attack_dataset_id values do not match the current widevis d00-d49 labels. Therefore it must not be used to claim current loss3 e1000 is worse/better than current random_solver_y e7860 on strict full-52 attack.
- The dense latest image-only traces are a same-sample latest direct subset. On that subset, loss3 is better than random_solver_y for attack increase.
- The historical old4 outlier attack100 rerun checks the actual old outlier initial conditions with latest checkpoints. It does not show a broad current random_solver advantage over latest loss3 on those outlier samples.
- The strict current full-52 widevis attack is now available. Loss3 attack increase mean is 0.00382075; random_solver_y is 0.00816719. Loss3 is lower on all 52 dataset rows in that strict table.

## Source Metadata

| old4_config_exists | old4_model_order | old4_gen_root | random_config_exists | random_models | random_steps | random_dataset_count | random_sample_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true | baseline,loss1_epoch8000,loss2_epoch2000,loss3_epoch1500 | /workspace/NeuralOperatorRobustness2/generalization_datasets/burgers | true | random_clean_y,random_solver_y | 20 | 52 | 1.020e+04 |

## Recovered Full-52 Attack Protocol Summary

| model | source_family | protocol_status | rows | generalization_rows | attack_id_matches_current_widevis_rows | mismatch_rows | attack_loss_increase_mean | attack_loss_increase_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | old4_historical_first_master | mismatched_old4_historical_dataset_labeled_as_widevis | 50 | 50 | 0 | 50 | 0.0435716 | 0.0245695 |
| baseline | old4_historical_first_master | old4_train_test_historical_but_same_named_split | 2 | 0 | 2 | 0 | 0.00710497 | 0.00710497 |
| loss1 | old4_historical_first_master | mismatched_old4_historical_dataset_labeled_as_widevis | 50 | 50 | 0 | 50 | 0.0504078 | 0.00286733 |
| loss1 | old4_historical_first_master | old4_train_test_historical_but_same_named_split | 2 | 0 | 2 | 0 | 0.00136473 | 0.00136473 |
| loss2 | old4_historical_first_master | mismatched_old4_historical_dataset_labeled_as_widevis | 50 | 50 | 0 | 50 | 0.0381912 | 0.00395536 |
| loss2 | old4_historical_first_master | old4_train_test_historical_but_same_named_split | 2 | 0 | 2 | 0 | 0.00158665 | 0.00158665 |
| loss3 | old4_historical_first_master | mismatched_old4_historical_dataset_labeled_as_widevis | 50 | 50 | 0 | 50 | 0.0417181 | 0.00182951 |
| loss3 | old4_historical_first_master | old4_train_test_historical_but_same_named_split | 2 | 0 | 2 | 0 | 0.000852392 | 0.000852392 |
| random_clean_y | random_current_widevis | current_random_widevis_attack | 52 | 50 | 52 | 0 | 0.0371569 | 0.035876 |
| random_solver_y | random_current_widevis | current_random_widevis_attack | 52 | 50 | 52 | 0 | 0.00816719 | 0.00656109 |

## Loss3 Historical Mismatch Examples

| dataset_index | display_label | attack_dataset_id | random_dataset_id | attack_loss_increase_mean | final_loss_mean |
| --- | --- | --- | --- | --- | --- |
| 8 | Gaussian GRF corr=0.012; range [-0.1,1.1] | burgers_far_centered_scale_shift_scale2p5_shiftm0p25 | burgers_widevis_l3target_d06 | 0.317523 | 0.693572 |
| 20 | Matern GRF c=0.04, nu=2.5; range [0,1.5] | burgers_far_sign_centered_scale1_shift0 | burgers_widevis_l3target_d18 | 0.294018 | 0.627714 |
| 14 | Matern GRF c=0.055, nu=4; range [-0.3,1.3] | burgers_far_positive_shift_scale1_shift1 | burgers_widevis_l3target_d12 | 0.272614 | 0.443162 |
| 11 | Gaussian GRF corr=0.009; range [-0.5,1.5] | burgers_far_negative_shift_scale1_shift1 | burgers_widevis_l3target_d09 | 0.231982 | 0.416364 |
| 4 | Gaussian GRF corr=0.012; range [0.15,1.25] | burgers_far_centered_scale_shift_scale1_shiftm0p4 | burgers_widevis_l3target_d02 | 0.201735 | 0.33587 |
| 6 | Gaussian GRF corr=0.009; range [-0.3,1.3] | burgers_far_centered_scale_shift_scale2_shift0 | burgers_widevis_l3target_d04 | 0.161495 | 0.242684 |
| 5 | Gaussian GRF corr=0.012; range [0,1.2] | burgers_far_centered_scale_shift_scale1p5_shift0 | burgers_widevis_l3target_d03 | 0.112776 | 0.148351 |
| 7 | Gaussian GRF corr=0.009; range [0.15,1.25] | burgers_far_centered_scale_shift_scale2p5_shift0p25 | burgers_widevis_l3target_d05 | 0.10566 | 0.148635 |
| 13 | Gaussian GRF corr=0.012; range [0.05,1.05] | burgers_far_positive_shift_scale1_shift0p5 | burgers_widevis_l3target_d11 | 0.0842605 | 0.0973704 |
| 10 | Gaussian GRF corr=0.012; range [-0.5,1.5] | burgers_far_negative_shift_scale1_shift0p5 | burgers_widevis_l3target_d08 | 0.0677855 | 0.0763347 |
| 21 | Matern GRF c=0.08, nu=2.5; range [-0.5,1.5] | burgers_far_zero_mean_scale1_shift0 | burgers_widevis_l3target_d19 | 0.0665702 | 0.0742838 |
| 19 | Matern GRF c=0.04, nu=2.5; range [-0.65,1.35] | burgers_far_sign_centered_scale0p5_shift0p25 | burgers_widevis_l3target_d17 | 0.0399745 | 0.0426682 |

## Dense Latest Direct Subset

Loss3 vs random_solver_y sample rows: loss3 wins 36/36. Mean attack increase: loss3=0.00368543, random_solver_y=0.00718798.

Overall best-model counts among all six models:

| best_model | sample_rows |
| --- | --- |
| loss3 | 31 |
| baseline | 3 |
| loss1 | 1 |
| loss2 | 1 |

| model | n | initial_loss_mean | final_loss_mean | attack_increase_mean | attack_increase_median |
| --- | --- | --- | --- | --- | --- |
| baseline | 36 | 0.00128263 | 0.0161868 | 0.0149042 | 0.0125772 |
| loss1 | 36 | 0.000397147 | 0.00679478 | 0.00639764 | 0.00659401 |
| loss2 | 36 | 0.000429287 | 0.00662348 | 0.0061942 | 0.00671136 |
| loss3 | 36 | 0.000138127 | 0.00382355 | 0.00368543 | 0.00309229 |
| random_clean_y | 36 | 0.011527 | 0.0578839 | 0.0463569 | 0.0439837 |
| random_solver_y | 36 | 0.000366534 | 0.00755451 | 0.00718798 | 0.0064737 |

## Historical Outlier Initial Conditions Retested With Latest Checkpoints

Latest loss3 beats random_solver_y on 4/6 historical outlier samples.

Overall best-model counts among the four retested models:

| best_model | sample_rows |
| --- | --- |
| loss3 | 4 |
| loss2 | 2 |

| model | n | initial_loss_mean | final_loss_mean | attack100_increase_mean | attack100_increase_median |
| --- | --- | --- | --- | --- | --- |
| loss1 | 6 | 0.762272 | 1.34365 | 0.581382 | 0.52488 |
| loss2 | 6 | 0.452974 | 0.843258 | 0.390284 | 0.34746 |
| loss3 | 6 | 0.393952 | 0.837856 | 0.443905 | 0.398596 |
| random_solver_y | 6 | 0.689156 | 1.19078 | 0.501621 | 0.459242 |

## Strict Latest Full-52 Attack

| model | datasets | sample_count_sum | initial_loss_mean | final_loss_mean | attack_loss_increase_mean | attack_loss_increase_median | final_delta_rms_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 52 | 1.020e+04 | 0.0011785 | 0.0118478 | 0.0106693 | 0.00841407 | 0.12 |
| loss1 | 52 | 1.020e+04 | 0.000562629 | 0.00715168 | 0.00658905 | 0.00566032 | 0.12 |
| loss2 | 52 | 1.020e+04 | 0.000623494 | 0.00722935 | 0.00660586 | 0.00571937 | 0.12 |
| loss3 | 52 | 1.020e+04 | 0.000181205 | 0.00400196 | 0.00382075 | 0.00298269 | 0.119982 |
| random_clean_y | 52 | 1.020e+04 | 0.00985418 | 0.0470111 | 0.0371569 | 0.035876 | 0.117134 |
| random_solver_y | 52 | 1.020e+04 | 0.000488038 | 0.00865523 | 0.00816719 | 0.00656109 | 0.119998 |

## Metric Table Protocol Counts After Fix

| counts_as_evidence | model_coverage_class | protocol_comparability_class | strict_latest_protocol | counts_in_six_model_evidence_claim | metric_rows |
| --- | --- | --- | --- | --- | --- |
| false | old4_only | strict_latest_same_sample_25 | true | false | 16 |
| false | random_only | partial_random_only_supplement | false | false | 11 |
| false | random_only | strict_latest_same_sample_25 | true | false | 12 |
| false | six_model_common | strict_latest_same_dataset | true | false | 4 |
| false | six_model_common | strict_latest_same_sample_25 | true | false | 20 |
| true | old4_only | strict_latest_same_sample_25 | true | false | 40 |
| true | random_only | partial_random_only_supplement | false | false | 14 |
| true | random_only | strict_latest_same_sample_25 | true | false | 36 |
| true | six_model_common | strict_latest_same_dataset | true | true | 24 |
| true | six_model_common | strict_latest_same_sample_25 | true | true | 144 |

## Evidence Status Table

| artifact | status | what_it_can_support | what_it_cannot_support | action |
| --- | --- | --- | --- | --- |
| clean_52dataset_six_models_selected_worktime.csv | strict_latest_six_model_clean | Clean RMSE/Relative L2/MSE comparison on current 52 datasets. | Adversarial robustness under attack. | Use for clean/generalization claims. |
| attack_52dataset_six_models_recovered_full_long.csv | mixed_historical_current_reference | Historical old4 attack behavior and current random attack behavior separately. | Strict current loss3 e1000 vs current random_solver_y e7860 full-52 attack claim. | Keep but exclude from strict latest six-model evidence. |
| attack_52dataset_six_models_selected_worktime_long.csv | partial_selected_worktime_attack | Baseline/random_clean_y/random_solver_y attack table only. | Loss1/loss2/loss3 attack ranking. | Do not use for six-model attack ranking. |
| dense image-only group00-group05 traces | strict_latest_direct_subset | Same-sample visual attack comparison on 36 displayed sample rows. | Full 52-dataset average unless full attack is run. | Use as latest direct subset evidence. |
| robustness_25sample_six_models_selected_worktime.csv and SVD tables | strict_latest_six_model_local_25sample | Local residual/Jacobian/SVD/error-operator evidence on fixed 25 samples. | Full 52-dataset attack loss ranking by itself. | Use evidence metrics only; keep cosines/angles as diagnostics. |
| old4 actual outlier latest attack100 diagnostics | strict_latest_on_old_outlier_initials_subset | Whether latest loss3 still fails badly on the historical outlier initial conditions. | Current full widevis 52-dataset mean. | Use as outlier sanity check. |
| strict latest full-52 widevis attack for old4 latest checkpoints | available_strict_latest_six_model_attack | The cleanest current full-52 attack answer. |  | Use attack_52dataset_six_models_strict_latest_widevis_long.csv for strict attack claims. |

## Output CSVs

- `data/protocol_confusion_audit_20260614/recovered_attack_protocol_row_audit.csv`: 312 rows, 16 columns
- `data/protocol_confusion_audit_20260614/recovered_attack_protocol_summary.csv`: 10 rows, 12 columns
- `data/protocol_confusion_audit_20260614/recovered_attack_loss3_mismatch_examples.csv`: 12 rows, 16 columns
- `data/protocol_confusion_audit_20260614/clean_52_protocol_audit.csv`: 52 rows, 10 columns
- `data/protocol_confusion_audit_20260614/dense_latest_attack_direct_summary_by_model.csv`: 6 rows, 6 columns
- `data/protocol_confusion_audit_20260614/dense_latest_attack_best_model_counts.csv`: 4 rows, 2 columns
- `data/protocol_confusion_audit_20260614/dense_latest_attack_winners_by_sample.csv`: 36 rows, 16 columns
- `data/protocol_confusion_audit_20260614/dense_latest_attack_loss3_vs_random_solver_by_sample.csv`: 36 rows, 12 columns
- `data/protocol_confusion_audit_20260614/old_outlier_attack100_summary_by_model.csv`: 4 rows, 6 columns
- `data/protocol_confusion_audit_20260614/old_outlier_attack100_best_model_counts.csv`: 2 rows, 2 columns
- `data/protocol_confusion_audit_20260614/old_outlier_attack100_winners_by_sample.csv`: 6 rows, 8 columns
- `data/protocol_confusion_audit_20260614/old_outlier_attack100_loss3_vs_random_solver_diagnostics.csv`: 6 rows, 19 columns
- `data/protocol_confusion_audit_20260614/strict_latest_attack52_model_summary.csv`: 6 rows, 9 columns
- `data/protocol_confusion_audit_20260614/metric_best_summary_protocol_counts.csv`: 10 rows, 6 columns
- `data/protocol_confusion_audit_20260614/evidence_status_table.csv`: 7 rows, 5 columns
