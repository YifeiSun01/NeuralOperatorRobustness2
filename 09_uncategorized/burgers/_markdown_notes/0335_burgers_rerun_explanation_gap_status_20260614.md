# Burgers Rerun Explanation And Gap Status, 20260614

Generated: 2026-06-14T14:56:57+00:00

This report answers four concrete questions: what was rerun, why the old conclusion was wrong, why the new conclusion is valid, and what is still missing.

## Short Answer

- The repair did **not** rerun training.
- The repair did **not** rerun random_solver_y or random_clean_y.
- The repair reran exactly the missing fair attack piece: latest old4 models (`baseline`, `loss1`, `loss2`, `loss3`) on the same current 52-dataset / 10200-sample widevis manifest used by `random_solver_y=7860` and `random_clean_y=8000`.
- The old six-model full-52 attack table was not a fair current comparison because old4 rows came from historical first-master data while random rows came from the current solver7860/clean8000 suite.
- The current strict table is fair because protocol and manifest checks pass across old4 and random models.

## Rerun Summary

| item | status | answer | evidence |
| --- | --- | --- | --- |
| what_was_rerun | done | Only the latest old4 full-52 P2Q2 attack was rerun: baseline/loss1/loss2/loss3 on the current widevis 10200-row manifest. | forensics/burgers_latest_old4_widevis_full52_p2q2_20step_20260614 |
| what_was_not_rerun | intentional | No self-training, no random model rerun, no Jacobian/SVD rerun, and no new dense visual attack except the already completed targeted visual/diagnostic pieces. | docs/burgers_final_resolution_audit_20260614.md |
| why_old_table_was_wrong | resolved | The recovered six-model full-52 attack table mixed historical old4 rows with current random rows; old4 generalization rows were not the current widevis datasets although they had current-style labels. | data/protocol_confusion_audit_20260614/recovered_attack_protocol_summary.csv |
| why_new_table_is_valid | resolved | The strict latest table uses one current manifest and one current protocol across all six models: 52 datasets, 10200 samples, P2Q2 20 steps, epsilon RMS 0.12, alpha RMS 0.012. | data/final_resolution_audit_20260614/protocol_validation.csv |
| main_attack_result | resolved | Loss3 attack increase mean 0.00382075; random_solver_y 0.00816719; random_clean_y 0.03715695; loss3 beats random_solver_y 52/52 dataset rows. | data/strict_latest_attack52_20260614/strict_latest_attack52_winners_by_dataset.csv |
| bundle_integrity | pass | overall_pass=True; parse=0; shape=0; ranked=0; summary=0; docs=0; copy=0. | data/full_data_bundle_integrity_audit_20260614/full_data_bundle_integrity_summary.json |

## Why The Previous Table Was Wrong

The previous recovered table was useful as a provenance artifact, but not valid for the current six-model claim. In the generalization rows, old4 models had historical attack dataset IDs, while random models had current widevis dataset IDs. That is exactly the source of the contradictory statement.

| model | source | source_family | protocol_status | rows | mismatch_rows | strict_latest_same_dataset_comparable_rows | attack_loss_increase_mean | attack_loss_increase_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | r2_first_master_old4_52dataset_20step | old4_historical_first_master | mismatched_old4_historical_dataset_labeled_as_widevis | 50 | 50 | 0 | 0.0435716 | 0.0245695 |
| loss1 | r2_first_master_old4_52dataset_20step | old4_historical_first_master | mismatched_old4_historical_dataset_labeled_as_widevis | 50 | 50 | 0 | 0.0504078 | 0.00286733 |
| loss2 | r2_first_master_old4_52dataset_20step | old4_historical_first_master | mismatched_old4_historical_dataset_labeled_as_widevis | 50 | 50 | 0 | 0.0381912 | 0.00395536 |
| loss3 | r2_first_master_old4_52dataset_20step | old4_historical_first_master | mismatched_old4_historical_dataset_labeled_as_widevis | 50 | 50 | 0 | 0.0417181 | 0.00182951 |

## Old Mixed Table Versus Strict Latest Table

This table shows why the old mean was misleading for current `loss3`: the old mixed table and the strict latest table are different protocol/provenance objects.

| model | source | datasets | attack_loss_increase_mean_old_mixed | attack_loss_increase_median_old_mixed | attack_loss_increase_mean_strict_latest | attack_loss_increase_median_strict_latest | old_minus_strict_attack_increase_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | r2_first_master_old4_52dataset_20step | 52 | 0.042169 | 0.0215512 | 0.0106693 | 0.00841407 | 0.0314998 |
| loss1 | r2_first_master_old4_52dataset_20step | 52 | 0.0485215 | 0.00284595 | 0.00658905 | 0.00566032 | 0.0419325 |
| loss2 | r2_first_master_old4_52dataset_20step | 52 | 0.0367834 | 0.0035991 | 0.00660586 | 0.00571937 | 0.0301775 |
| loss3 | r2_first_master_old4_52dataset_20step | 52 | 0.0401464 | 0.00165531 | 0.00382075 | 0.00298269 | 0.0363256 |
| random_clean_y | solver7860_clean8000_random_52dataset_current | 52 | 0.0371569 | 0.035876 | 0.0371569 | 0.035876 | 0 |
| random_solver_y | solver7860_clean8000_random_52dataset_current | 52 | 0.00816719 | 0.00656109 | 0.00816719 | 0.00656109 | 6.592e-17 |

## Why The New Table Is Correct

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

Strict latest attack winner count by dataset:

{
  "loss3": 52
}

## Remaining Gaps

| area | status | blocking | detail |
| --- | --- | --- | --- |
| strict_current_full52_attack | complete | 0 | The missing fair attack comparison is now complete and is the source used by ranked attack tables. |
| clean_generalization_tables | complete | 0 | 52-dataset clean table and 50-generalization summaries exist and pass recompute checks. |
| robustness_svd_jacobian_25sample | complete_with_declared_limits | 0 | Six-model 25-sample tables and SVD/error spectrum summaries are present; small-n local rows are not used as global proof. |
| diagnostic_metric_interpretation | complete_with_boundaries | 0 | Cosines, angles, delta norms, and process quantities remain recorded but are not counted as model-quality proof. |
| figures_and_organized_release | complete_local | 0 | Organized release has 2245 files and missing=0 after refresh. |
| github_push | blocked_by_missing_credential | 1 | Current shell has no GitHub credential; local vast-ai is ahead of origin/vast-ai by 2 commits. |
| r2_upload_latest_final_files | blocked_by_missing_environment_credentials | 1 | Current shell has no R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_ENDPOINT, so newest final-resolution files are local only until env vars are set. |

## Claim Boundaries

| claim | verdict | support | boundary |
| --- | --- | --- | --- |
| Clean/generalization accuracy | loss3 strongly supported | Loss3 is best for all-52 and 50-generalization RMSE/Relative L2/MSE; significance tests pass for those main clean scopes. | Train/test single split rows are n=1 and random_solver_y can be lower there; they are not the generalization conclusion. |
| Strict full-52 attack robustness | loss3 strongly supported | Loss3 is best on full-52 attack increase/final loss and on 52/52 dataset rows; loss3 vs random_solver_y is significant. | Applies to current P2Q2 20-step epsilon RMS 0.12 protocol. |
| SVD/error-operator spectrum | loss3 strongly supported | Loss3 is best on all error-spectrum rows in strict evidence summaries; most rank/top-k rows are significant. | Rank1/top1 variants may be mean-best but not always q<0.05 against every other model. |
| 25-sample robustness/Jacobian/SVD local metrics | loss3 overall strongest but not every row | Loss3 has the most best rows in strict local evidence. | Some train/test 2-sample and model-solver subspace rows favor random_solver_y or loss1; small n rows should not override full-52 and spectrum conclusions. |
| Global statement 'loss3 is better on every numeric field' | not a valid claim | Some single-split clean, local subspace, and diagnostic rows are not loss3-best. | The valid strong claim is metric-family specific: main quality evidence strongly favors loss3. |

## Bottom Line

Locally, there is no remaining analysis blocker for the Burgers conclusion. The valid strong claim is:

`loss3` is the best model on the main comparable current evidence: clean/generalization accuracy, strict full-52 attack robustness, and SVD/error-operator spectrum.

The invalid overstatement is:

`loss3` is best on every single recorded scalar. Some single-split, small-n local, or diagnostic rows favor another model, and those rows are preserved instead of hidden.

External sync is the only current blocking item: GitHub/R2 credentials are not present in the active shell, so the newest local commits/results cannot be pushed/uploaded from here until those environment variables exist.

## Output CSVs

- `data/rerun_explanation_gap_status_20260614/rerun_summary.csv`
- `data/rerun_explanation_gap_status_20260614/remaining_gap_status.csv`
- `data/rerun_explanation_gap_status_20260614/old_mixed_vs_strict_latest_attack_summary.csv`
- `data/rerun_explanation_gap_status_20260614/old_mixed_bad_source_rows.csv`
- `data/rerun_explanation_gap_status_20260614/strict_protocol_validation_copy.csv`
- `data/rerun_explanation_gap_status_20260614/final_claim_boundaries_copy.csv`
