# Burgers Full Data Bundle Integrity Audit, 20260614

This audit validates the generated Burgers solver7860/clean8000 data bundle.
It does not rerun training, attacks, Jacobian, SVD, or plotting.

## Verdict

- Overall pass: **True**
- Parsed files checked: **1827**
- Parse failures: **0**
- Expected-shape failures: **0**
- Clean best-field failures: **0**
- Clean generalization summary failures: **0**
- Ranked long-table failures: **0**
- Best-summary failures: **0**
- Model-summary mean/n failures: **0**
- Documentation caveat failures: **0**
- Organized-copy failures: **0**

## Important Protocol Caveats Kept Explicit

- The strict selected-worktime 52-dataset attack table contains baseline,
  random_clean_y, and random_solver_y only.
- The recovered six-model 52-dataset attack table is mixed source:
  baseline/loss1/loss2/loss3 are from the old full-52 20-step run, while
  random_clean_y/random_solver_y are from the solver7860/clean8000 random suite.
- The strict latest six-model 52-dataset attack table is now available and the
  ranked attack table uses `attack_52dataset_six_models_strict_latest_widevis_long.csv`.
- The 54 robustness metrics are not all six-model-common; some are old4-only
  or random-only and should not be used as one undifferentiated "best model"
  proof.

## Expected Shape Checks

| check | ok | expected | actual | details |
| --- | --- | --- | --- | --- |
| clean_52_rows | True | 52 | 52 |  |
| clean_52_split_counts | True | {"generalization": 50, "test": 1, "train": 1} | {"generalization": 50, "test": 1, "train": 1} |  |
| clean_summary_models | True | ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"] | ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"] |  |
| attack_selected_rows | True | 156 | 156 |  |
| attack_selected_models | True | ["baseline", "random_clean_y", "random_solver_y"] | ["baseline", "random_clean_y", "random_solver_y"] |  |
| attack_selected_counts_per_model | True | {"baseline": 52, "random_clean_y": 52, "random_solver_y": 52} | {"baseline": 52, "random_clean_y": 52, "random_solver_y": 52} |  |
| attack_recovered_rows | True | 312 | 312 |  |
| attack_recovered_models | True | ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"] | ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"] |  |
| attack_recovered_counts_per_model | True | {"baseline": 52, "loss1": 52, "loss2": 52, "loss3": 52, "random_clean_y": 52, "random_solver_y": 52} | {"baseline": 52, "loss1": 52, "loss2": 52, "loss3": 52, "random_clean_y": 52, "random_solver_y": 52} |  |
| attack_recovered_has_mixed_source_caveat | True | "old4 one source and random2 one different source" | {"('r2_first_master_old4_52dataset_20step', 'baseline')": 52, "('r2_first_master_old4_52dataset_20step', 'loss1')": 52, "('r2_first_master_old4_52dataset_20step', 'loss2')": 52, "('r2_first_master_old4_52dataset_20step', 'loss3')": 52, "('solver7860_clean8000_random_52dataset_current', 'random_clean_y')": 52, "('solver7860_clean8000_random_52dataset_current', 'random_solver_y')": 52} |  |
| attack_strict_latest_rows | True | 312 | 312 |  |
| attack_strict_latest_models | True | ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"] | ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"] |  |
| attack_strict_latest_counts_per_model | True | {"baseline": 52, "loss1": 52, "loss2": 52, "loss3": 52, "random_clean_y": 52, "random_solver_y": 52} | {"baseline": 52, "loss1": 52, "loss2": 52, "loss3": 52, "random_clean_y": 52, "random_solver_y": 52} |  |
| attack_ranked_source_is_strict_latest | True | "ranked attack table uses strict latest source" | ["attack_52dataset_six_models_strict_latest_widevis_long.csv"] |  |
| robustness_25sample_rows | True | 150 | 150 |  |
| robustness_25sample_model_counts | True | {"baseline": 25, "loss1": 25, "loss2": 25, "loss3": 25, "random_clean_y": 25, "random_solver_y": 25} | {"baseline": 25, "loss1": 25, "loss2": 25, "loss3": 25, "random_clean_y": 25, "random_solver_y": 25} |  |
| robustness_25sample_unique_samples | True | 25 | 25 |  |
| robustness_every_sample_has_six_models | True | "every sample_id has 6 models" | {"6": 25} |  |
| robustness_sample_split_counts | True | {"generalization": 21, "test": 2, "train": 2} | {"generalization": 21, "test": 2, "train": 2} |  |
| ranked_clean_rows | True | 936 | 936 |  |
| ranked_attack_rows | True | 1248 | 1248 |  |
| ranked_robustness_rows | True | 8100 | 8100 |  |
| ranked_svd_top20_rows | True | 3000 | 3000 |  |
| ranked_svd_topk_rows | True | 1200 | 1200 |  |
| ranked_svd_top100_supplement_rows | True | 15000 | 15000 |  |
| ranked_svd_topk_top100_supplement_rows | True | 1800 | 1800 |  |
| ranked_model_solver_subspace_top100_supplement_rows | True | 1500 | 1500 |  |
| ranked_random_affine_direction_supplement_rows | True | 1250 | 1250 |  |
| singular_reference_rows | True | 3500 | 3500 |  |
| singular_reference_models | True | ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y", "solver"] | ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y", "solver"] |  |
| best_summary_rows | True | 321 | 321 |  |
| model_summary_rows | True | 1522 | 1522 |  |
| best_vs_other_tests_rows | True | 1201 | 1201 |  |
| loss3_vs_other_tests_rows | True | 1128 | 1128 |  |

## Failures

### Parse Failures

_No rows._

### Shape Failures

_No rows._

### Clean Best Field Failures

_No rows._

### Clean Summary Failures

_No rows._

### Ranked Long Table Failures

_No rows._

### Best Summary Failures

_No rows._

### Model Summary Failures

_No rows._

### Documentation Caveat Failures

_No rows._

### Organized Copy Failures

_No rows._

## Output Files

- `data/full_data_bundle_integrity_audit_20260614/file_parse_audit.csv`
- `data/full_data_bundle_integrity_audit_20260614/expected_table_shape_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/clean_best_field_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/clean_generalization_summary_recompute_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/ranked_long_table_consistency_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/best_summary_consistency_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/model_summary_from_long_recompute_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/docs_caveat_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/organized_release_copy_checks.csv`
- `data/full_data_bundle_integrity_audit_20260614/full_data_bundle_integrity_summary.json`
