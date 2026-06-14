# Burgers Metric Integrity Audit, 20260614

This audit rechecks the generated Burgers solver7860/clean8000 metric tables
without rerunning training, attack, Jacobian, SVD, or plotting jobs.

## Check Summary

| check | value |
| --- | --- |
| mean_recompute_rows | 1522 |
| mean_recompute_failures | 0 |
| best_recompute_rows | 321 |
| best_model_failures | 0 |
| best_mean_failures | 0 |
| robustness_metric_class_counts | {"old4_only": 14, "random_only": 12, "six_model_common": 28} |
| selected_attack_models | ["baseline", "random_clean_y", "random_solver_y"] |
| recovered_attack_models | ["baseline", "loss1", "loss2", "loss3", "random_clean_y", "random_solver_y"] |

## Attack-52 Source Audit

The selected-worktime attack table has only baseline and the two random models.
The recovered full table used by the ranked appendix is a mixed source table:
old-four rows come from the old full-52 20-step run, while random rows come from
the current solver7860/clean8000 random full suite.

| table | source | model | rows | dataset_count | sample_count_sum | attack_increase_mean | final_loss_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selected_worktime_original | old4_existing | baseline | 52 | 52 | 10200 | 0.042169 | 0.070382 |
| selected_worktime_original | selected_random_worktime | random_clean_y | 52 | 52 | 10200 | 0.037157 | 0.047011 |
| selected_worktime_original | selected_random_worktime | random_solver_y | 52 | 52 | 10200 | 0.008167 | 0.008655 |
| recovered_full_ranked_source | r2_first_master_old4_52dataset_20step | baseline | 52 | 52 | 10200 | 0.042169 | 0.070382 |
| recovered_full_ranked_source | r2_first_master_old4_52dataset_20step | loss1 | 52 | 52 | 10200 | 0.048522 | 0.085428 |
| recovered_full_ranked_source | r2_first_master_old4_52dataset_20step | loss2 | 52 | 52 | 10200 | 0.036783 | 0.061439 |
| recovered_full_ranked_source | r2_first_master_old4_52dataset_20step | loss3 | 52 | 52 | 10200 | 0.040146 | 0.066960 |
| recovered_full_ranked_source | solver7860_clean8000_random_52dataset_current | random_clean_y | 52 | 52 | 10200 | 0.037157 | 0.047011 |
| recovered_full_ranked_source | solver7860_clean8000_random_52dataset_current | random_solver_y | 52 | 52 | 10200 | 0.008167 | 0.008655 |

## Robustness Metric Comparability

The 54 robustness metrics are not all six-model-common metrics.

| comparability_class | metric_count |
| --- | --- |
| old4_only | 14 |
| random_only | 12 |
| six_model_common | 28 |

Interpretation:

- `six_model_common`: all six models have at least one valid value, although
  random attack trace fields may have 24 rather than 25 valid samples.
- `old4_only`: only baseline/loss1/loss2/loss3 have values.
- `random_only`: only random_clean_y/random_solver_y have values.

## Dense Trace Pooled Means

These are the latest dense visual trace means, pooled by group means over
group00 through group05.

| model | initial_loss_mean | final_loss_mean | attack_increase_mean |
| --- | --- | --- | --- |
| baseline | 0.001283 | 0.016187 | 0.014904 |
| loss1 | 3.971e-04 | 0.006795 | 0.006398 |
| loss2 | 4.293e-04 | 0.006623 | 0.006194 |
| loss3 | 1.381e-04 | 0.003824 | 0.003685 |
| random_clean_y | 0.011527 | 0.057884 | 0.046357 |
| random_solver_y | 3.665e-04 | 0.007555 | 0.007188 |

## Main Corrections

- Do not claim that `random_solver_y` beats latest `loss3 e1000` on a strict
  full-52 attack protocol. That same-protocol artifact was not found locally.
- The recovered attack-52 ranked rows are historical mixed evidence; they remain
  useful only with that caveat.
- Do not interpret all 54 robustness metrics as equal model-quality rankings.
  Split them by comparability class first.
- The previous random top50/top100 SVD export gap is now resolved by the
  20260614 supplement derived from stored random-model Jacobian matrices.
- The random-model affine/local-gain fields are now recorded as a supplement
  derived from existing checkpoints and stored Jacobians; attack-delta cosine
  has n=24 per random model because sample_id=4 is outside the saved attack
  manifest.
- For latest direct evidence, use the 25-sample robustness/SVD table and dense
  visual trace set for `loss3 e1000`, plus the clean 52-dataset table for clean
  generalization.
