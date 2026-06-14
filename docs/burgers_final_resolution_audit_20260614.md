# Burgers Final Resolution Audit, 20260614

Generated: 2026-06-14T14:56:57+00:00

This report is the final sweep of the confusing Burgers conclusions after the strict latest full-52 attack repair. It reads existing artifacts only.

## Bottom Line

- The earlier contradiction is resolved: it came from a mixed historical/current recovered attack table.
- The strict current full-52 attack is now same-manifest and same-protocol across all six models.
- On the main comparable quality evidence, loss3 is the stable best model.
- It is not correct to claim loss3 wins every single recorded scalar; several single-split, small-n local, and diagnostic rows favor another model. Those do not overturn the main result.

## Strict Attack Summary

| model | datasets | sample_count_sum | initial_loss_mean | final_loss_mean | attack_loss_increase_mean | attack_loss_increase_median | final_delta_rms_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 52 | 1.020e+04 | 0.0011785 | 0.0118478 | 0.0106693 | 0.00841407 | 0.12 |
| loss1 | 52 | 1.020e+04 | 0.000562629 | 0.00715168 | 0.00658905 | 0.00566032 | 0.12 |
| loss2 | 52 | 1.020e+04 | 0.000623494 | 0.00722935 | 0.00660586 | 0.00571937 | 0.12 |
| loss3 | 52 | 1.020e+04 | 0.000181205 | 0.00400196 | 0.00382075 | 0.00298269 | 0.119982 |
| random_clean_y | 52 | 1.020e+04 | 0.00985418 | 0.0470111 | 0.0371569 | 0.035876 | 0.117134 |
| random_solver_y | 52 | 1.020e+04 | 0.000488038 | 0.00865523 | 0.00816719 | 0.00656109 | 0.119998 |

## Protocol Validation

| check | ok | old4_value | random_value | meaning |
| --- | --- | --- | --- | --- |
| config_steps_matches | 1 | 20 | 20 | Strict attack protocol must match across old4 and random models. |
| config_epsilon_rms_matches | 1 | 0.12 | 0.12 | Strict attack protocol must match across old4 and random models. |
| config_alpha_rms_matches | 1 | 0.012 | 0.012 | Strict attack protocol must match across old4 and random models. |
| config_sample_count_matches | 1 | 1.020e+04 | 1.020e+04 | Strict attack protocol must match across old4 and random models. |
| config_dataset_count_matches | 1 | 52 | 52 | Strict attack protocol must match across old4 and random models. |
| manifest_length_matches | 1 | 1.020e+04 | 1.020e+04 | All six models must be evaluated on the same 10200 sample rows. |
| manifest_global_sample_id_matches | 1 | all_equal | all_equal | Same-sample comparison check. |
| manifest_split_matches | 1 | all_equal | all_equal | Same-sample comparison check. |
| manifest_dataset_id_matches | 1 | all_equal | all_equal | Same-sample comparison check. |
| manifest_source_index_matches | 1 | all_equal | all_equal | Same-sample comparison check. |
| manifest_dataset_sample_offset_matches | 1 | all_equal | all_equal | Same-sample comparison check. |

## Question Resolution

| question | status | answer | evidence_path | remaining_caveat |
| --- | --- | --- | --- | --- |
| Was the earlier full-52 attack conclusion mixed-source? | resolved | Yes. The recovered table mixed historical old4 attack rows with current random rows, and old4 generalization rows did not match current widevis labels. | outputs/.../data/protocol_confusion_audit_20260614/recovered_attack_protocol_summary.csv | Recovered table is retained only as historical reference. |
| Is the strict current full-52 attack now available? | resolved | Yes. Old4 latest checkpoints were rerun on the same 10200-row manifest as the current random suite. | outputs/.../data/attack_52dataset_six_models_strict_latest_widevis_long.csv | No caveat blocking the full-52 attack conclusion. |
| Does loss3 beat random_solver_y on strict full-52 attack? | resolved | Yes. Loss3 attack increase mean=0.00382075; random_solver_y=0.00816719; loss3 wins 52/52 dataset rows. | docs/burgers_strict_latest_attack52_20260614.md | This is 20-step P2Q2 under epsilon RMS 0.12, not a different attack budget. |
| Does dense image-only visual evidence agree? | resolved | Yes. Loss3 beats random_solver_y on 36/36 dense displayed sample rows. | outputs/.../data/protocol_confusion_audit_20260614/dense_latest_attack_loss3_vs_random_solver_by_sample.csv | Dense panels are a visual subset, not the full-52 aggregate. |
| Are diagnostic/process metrics still mixed into quality claims? | resolved | No. Ranked tables now tag metric_role, protocol_comparability_class, strict_latest_protocol, and counts_in_six_model_evidence_claim. | outputs/.../data/ranked_metric_tables_20260614/metric_role_definitions.csv | Cosines, angles, delta norms, and process quantities remain recorded but are not quality-proof counts. |
| Does the final bundle pass integrity checks? | resolved | Yes. full bundle overall_pass=True, parse failures=0, shape failures=0, copy failures=0. | outputs/.../data/full_data_bundle_integrity_audit_20260614/full_data_bundle_integrity_summary.json | External GitHub/R2 sync still needs credentials in environment variables. |

## Strict Evidence Best Counts

| family | strict_evidence_rows | loss3_best_rows | random_solver_y_best_rows | other_best_rows | loss3_best_fraction |
| --- | --- | --- | --- | --- | --- |
| attack_robustness_52dataset | 12 | 10 | 2 | 0 | 0.833333 |
| clean_generalization | 12 | 6 | 6 | 0 | 0.5 |
| model_solver_subspace_top100_supplement | 10 | 7 | 3 | 0 | 0.7 |
| robustness_svd_jacobian_25sample | 92 | 54 | 34 | 4 | 0.586957 |
| svd_error_spectrum | 29 | 29 | 0 | 0 | 1 |
| svd_error_spectrum_top100_supplement | 13 | 13 | 0 | 0 | 1 |

## Loss3 Significance Summary

| family | loss3_mean_best_rows | loss3_sig_vs_all_rows | significant_pair_count | comparison_count |
| --- | --- | --- | --- | --- |
| attack_robustness_52dataset | 10 | 6 | 30 | 50 |
| clean_generalization | 6 | 6 | 30 | 30 |
| model_solver_subspace_top100_supplement | 7 | 5 | 30 | 35 |
| robustness_svd_jacobian_25sample | 54 | 40 | 217 | 270 |
| svd_error_spectrum | 29 | 26 | 142 | 145 |
| svd_error_spectrum_top100_supplement | 13 | 11 | 63 | 65 |

## Non-Loss3 Best Rows: What They Mean

These rows are kept visible because hiding them would be another way to get confused. Most are single train/test split rows, small-n local rows, or subspace-similarity rows rather than the main full-52 quality conclusions.

| family | scope | metric | best_model | mean | runner_up_model | runner_up_mean | best_vs_runner_significant_q05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attack_robustness_52dataset | attack_test_1dataset | initial_loss_mean | random_solver_y | 5.701e-07 | loss1 | 9.827e-07 | 0 |
| attack_robustness_52dataset | attack_train_1dataset | initial_loss_mean | random_solver_y | 4.340e-07 | loss1 | 8.670e-07 | 0 |
| clean_generalization | clean_test_1dataset | mse | random_solver_y | 5.701e-07 | loss1 | 7.774e-07 | 0 |
| clean_generalization | clean_test_1dataset | relative_l2 | random_solver_y | 0.00137934 | loss1 | 0.00161192 | 0 |
| clean_generalization | clean_test_1dataset | rmse | random_solver_y | 0.000728259 | loss1 | 0.000836263 | 0 |
| clean_generalization | clean_train_1dataset | mse | random_solver_y | 4.415e-07 | loss1 | 5.990e-07 | 0 |
| clean_generalization | clean_train_1dataset | relative_l2 | random_solver_y | 0.0012313 | loss1 | 0.00144669 | 0 |
| clean_generalization | clean_train_1dataset | rmse | random_solver_y | 0.000642213 | loss1 | 0.000746925 | 0 |
| model_solver_subspace_top100_supplement | model_solver_subspace_25sample | model_solver_top100_left_subspace_mean_cos | random_solver_y | 0.57089 | loss3 | 0.558577 | 0 |
| model_solver_subspace_top100_supplement | model_solver_subspace_25sample | model_solver_top100_right_subspace_mean_cos | random_solver_y | 0.460405 | loss3 | 0.449667 | 1 |
| model_solver_subspace_top100_supplement | model_solver_subspace_25sample | model_solver_top50_left_subspace_mean_cos | random_solver_y | 0.754114 | loss3 | 0.733096 | 1 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | atb_norm | random_solver_y | 0.000878852 | loss1 | 0.00363038 | 1 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | attack_initial_mse | random_solver_y | 3.435e-07 | loss1 | 5.613e-07 | 0 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | bias_gradient_norm | random_solver_y | 0.000878852 | loss1 | 0.00363038 | 1 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | bias_gradient_rms | random_solver_y | 2.746e-05 | loss1 | 0.000113449 | 1 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | clean_residual_mse_recomputed | random_solver_y | 3.434e-07 | loss1 | 5.658e-07 | 0 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | clean_residual_norm_l2 | random_solver_y | 0.0186678 | loss1 | 0.0237928 | 0 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | error_fro_norm | random_solver_y | 0.22832 | loss1 | 0.627941 | 1 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | error_fro_norm_comparable | random_solver_y | 0.22832 | loss1 | 0.627941 | 1 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | error_fro_norm_original_reported | random_solver_y | 0.222019 | loss1 | 0.627941 | 1 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | error_spectral_norm | random_solver_y | 0.0926756 | loss1 | 0.29811 | 1 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | j_error_transpose_error_l2 | random_solver_y | 0.000878852 | loss1 | 0.00363038 | 1 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | j_error_transpose_error_rms | random_solver_y | 2.746e-05 | loss1 | 0.000113449 | 1 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | model_solver_top10_left_subspace_mean_cos | random_solver_y | 0.999085 | loss1 | 0.998748 | 0 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | model_solver_top1_left_abs_cos | random_solver_y | 0.999893 | loss1 | 0.999875 | 0 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | model_solver_top20_left_subspace_mean_cos | random_solver_y | 0.995273 | loss3 | 0.994801 | 0 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | model_solver_top5_left_subspace_mean_cos | random_solver_y | 0.999899 | loss1 | 0.999843 | 0 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | model_solver_top5_right_subspace_mean_cos | random_solver_y | 0.999852 | loss3 | 0.999808 | 1 |
| robustness_svd_jacobian_25sample | robustness_test_2sample | top_error_sv_value | random_solver_y | 0.0926756 | loss1 | 0.29811 | 1 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | atb_norm | random_solver_y | 0.00129057 | loss1 | 0.00175122 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | attack_initial_mse | random_solver_y | 3.963e-07 | loss1 | 4.923e-07 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | bias_gradient_norm | random_solver_y | 0.00129057 | loss1 | 0.00175122 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | bias_gradient_rms | random_solver_y | 4.033e-05 | loss1 | 5.473e-05 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | clean_residual_mse_recomputed | loss1 | 4.905e-07 | random_solver_y | 6.285e-07 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | clean_residual_norm_l2 | loss1 | 0.022054 | random_solver_y | 0.0249168 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | error_fro_norm | random_solver_y | 0.254056 | loss1 | 0.618462 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | error_fro_norm_comparable | random_solver_y | 0.254056 | loss1 | 0.618462 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | error_fro_norm_original_reported | random_solver_y | 0.247931 | loss1 | 0.618462 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | error_spectral_norm | random_solver_y | 0.129125 | loss1 | 0.256461 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | j_error_transpose_error_l2 | random_solver_y | 0.00129057 | loss1 | 0.00175122 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | j_error_transpose_error_rms | random_solver_y | 4.033e-05 | loss1 | 5.473e-05 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | model_solver_top10_left_subspace_mean_cos | random_solver_y | 0.999715 | loss1 | 0.999579 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | model_solver_top10_right_subspace_mean_cos | random_solver_y | 0.999505 | loss3 | 0.998891 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | model_solver_top1_left_abs_cos | loss1 | 0.99992 | random_solver_y | 0.999661 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | model_solver_top1_right_abs_cos | random_solver_y | 0.99991 | loss3 | 0.999846 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | model_solver_top20_left_subspace_mean_cos | random_solver_y | 0.994034 | loss3 | 0.991508 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | model_solver_top5_left_subspace_mean_cos | loss1 | 0.999809 | random_solver_y | 0.999746 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | model_solver_top5_right_subspace_mean_cos | random_solver_y | 0.999849 | loss3 | 0.999804 | 0 |
| robustness_svd_jacobian_25sample | robustness_train_2sample | top_error_sv_value | random_solver_y | 0.129125 | loss1 | 0.256461 | 0 |

## Final Claim Boundaries

| claim | verdict | support | boundary |
| --- | --- | --- | --- |
| Clean/generalization accuracy | loss3 strongly supported | Loss3 is best for all-52 and 50-generalization RMSE/Relative L2/MSE; significance tests pass for those main clean scopes. | Train/test single split rows are n=1 and random_solver_y can be lower there; they are not the generalization conclusion. |
| Strict full-52 attack robustness | loss3 strongly supported | Loss3 is best on full-52 attack increase/final loss and on 52/52 dataset rows; loss3 vs random_solver_y is significant. | Applies to current P2Q2 20-step epsilon RMS 0.12 protocol. |
| SVD/error-operator spectrum | loss3 strongly supported | Loss3 is best on all error-spectrum rows in strict evidence summaries; most rank/top-k rows are significant. | Rank1/top1 variants may be mean-best but not always q<0.05 against every other model. |
| 25-sample robustness/Jacobian/SVD local metrics | loss3 overall strongest but not every row | Loss3 has the most best rows in strict local evidence. | Some train/test 2-sample and model-solver subspace rows favor random_solver_y or loss1; small n rows should not override full-52 and spectrum conclusions. |
| Global statement 'loss3 is better on every numeric field' | not a valid claim | Some single-split clean, local subspace, and diagnostic rows are not loss3-best. | The valid strong claim is metric-family specific: main quality evidence strongly favors loss3. |

## Output CSVs

- `data/final_resolution_audit_20260614/protocol_validation.csv`
- `data/final_resolution_audit_20260614/question_resolution_table.csv`
- `data/final_resolution_audit_20260614/evidence_family_summary.csv`
- `data/final_resolution_audit_20260614/loss3_significance_family_summary.csv`
- `data/final_resolution_audit_20260614/non_loss3_best_strict_evidence_rows.csv`
- `data/final_resolution_audit_20260614/final_claim_boundaries.csv`
