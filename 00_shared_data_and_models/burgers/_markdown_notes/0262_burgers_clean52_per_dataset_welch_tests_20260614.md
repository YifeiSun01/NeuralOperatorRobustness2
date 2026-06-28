# Burgers Clean 52 Per-Dataset Welch Tests, 20260614

Generated: 2026-06-14T22:01:58+00:00

This report answers the per-dataset clean-metric question for the 52-dataset six-model clean table. It uses existing per-dataset mean/std/n summaries and does not rerun evaluation.

## Important Method Boundary

These are Welch tests from summary statistics, not paired per-sample tests. A unified six-model per-sample clean table is not present in the final bundle. The old4 clean source has per-dataset mean/std/sem and old4 internal paired tests; the random-model clean source has per-sample and per-dataset summaries. The six-model strict common clean comparison can therefore be tested per dataset only with summary-stat Welch tests unless clean per-sample values are recomputed or recovered for all six models.

## What Was Tested

- Dataset rows: 52.
- Metrics: RMSE, Relative L2, MSE.
- Total per-dataset tests: 156.
- Direction: lower is better.
- Comparison: `loss3` versus the best non-loss3 model for the same dataset and metric. When `loss3` is best, that comparison is the runner-up.
- Multiple-testing correction: BH/FDR over all 156 clean Welch tests.

## Metric Summary

| metric | dataset_rows | loss3_best_rows | loss3_significantly_lower_rows | mean_advantage | min_advantage | max_advantage |
| --- | --- | --- | --- | --- | --- | --- |
| rmse | 52 | 48 | 48 | 0.00542914 | -0.00364563 | 0.0135326 |
| relative_l2 | 52 | 48 | 48 | 0.010658 | -0.00694575 | 0.0373 |
| mse | 52 | 49 | 48 | 0.000267732 | -1.993e-05 | 0.000883177 |

## Best-Model Counts

| metric | best_model | dataset_rows |
| --- | --- | --- |
| mse | loss3 | 49 |
| mse | random_solver_y | 3 |
| relative_l2 | loss3 | 48 |
| relative_l2 | random_solver_y | 4 |
| rmse | loss3 | 48 |
| rmse | random_solver_y | 4 |

## Split Summary

| metric | split | dataset_rows | loss3_best_rows | loss3_significantly_lower_rows |
| --- | --- | --- | --- | --- |
| rmse | train | 1 | 0 | 0 |
| relative_l2 | train | 1 | 0 | 0 |
| mse | train | 1 | 0 | 0 |
| rmse | test | 1 | 0 | 0 |
| relative_l2 | test | 1 | 0 | 0 |
| mse | test | 1 | 0 | 0 |
| rmse | generalization | 50 | 48 | 48 |
| relative_l2 | generalization | 50 | 48 | 48 |
| mse | generalization | 50 | 49 | 48 |

## Rows With Smallest Loss3 Advantage

| metric | dataset_order | split | dataset_id | best_model | second_model | loss3_rank | comparison_model | loss3_mean | comparison_mean | mean_advantage_comparison_minus_loss3 | welch_q_loss3_lower_bh_fdr_all156 | loss3_significantly_lower_q05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mse | 2 | test | test_original_gaussian_corr0p03 | random_solver_y | loss1 | 4 | random_solver_y | 2.050e-05 | 5.701e-07 | -1.993e-05 | 1 | 0 |
| mse | 1 | train | train_original_gaussian_corr0p03 | random_solver_y | loss1 | 5 | random_solver_y | 1.938e-05 | 4.415e-07 | -1.894e-05 | 1 | 0 |
| mse | 48 | generalization | burgers_widevis_l3target_d45 | random_solver_y | loss3 | 2 | random_solver_y | 1.978e-05 | 8.573e-06 | -1.120e-05 | 1 | 0 |
| mse | 14 | generalization | burgers_widevis_l3target_d11 | loss3 | random_solver_y | 1 | random_solver_y | 2.602e-05 | 2.785e-05 | 1.840e-06 | 0.20285 | 0 |
| mse | 40 | generalization | burgers_widevis_l3target_d37 | loss3 | random_solver_y | 1 | random_solver_y | 7.438e-05 | 9.598e-05 | 2.159e-05 | 5.192e-29 | 1 |
| mse | 18 | generalization | burgers_widevis_l3target_d15 | loss3 | random_solver_y | 1 | random_solver_y | 5.619e-05 | 8.262e-05 | 2.643e-05 | 2.550e-06 | 1 |
| relative_l2 | 2 | test | test_original_gaussian_corr0p03 | random_solver_y | loss1 | 4 | random_solver_y | 0.00832509 | 0.00137934 | -0.00694575 | 1 | 0 |
| relative_l2 | 1 | train | train_original_gaussian_corr0p03 | random_solver_y | loss1 | 5 | random_solver_y | 0.00811692 | 0.0012313 | -0.00688562 | 1 | 0 |
| relative_l2 | 48 | generalization | burgers_widevis_l3target_d45 | random_solver_y | loss3 | 2 | random_solver_y | 0.00874176 | 0.00554387 | -0.00319789 | 1 | 0 |
| relative_l2 | 14 | generalization | burgers_widevis_l3target_d11 | random_solver_y | loss3 | 2 | random_solver_y | 0.00895354 | 0.00880972 | -0.000143826 | 0.707948 | 0 |
| relative_l2 | 40 | generalization | burgers_widevis_l3target_d37 | loss3 | random_solver_y | 1 | random_solver_y | 0.0114676 | 0.0129543 | 0.00148663 | 4.367e-30 | 1 |
| relative_l2 | 18 | generalization | burgers_widevis_l3target_d15 | loss3 | random_solver_y | 1 | random_solver_y | 0.0101431 | 0.0118 | 0.00165693 | 2.272e-06 | 1 |
| rmse | 2 | test | test_original_gaussian_corr0p03 | random_solver_y | loss1 | 4 | random_solver_y | 0.00437389 | 0.000728259 | -0.00364563 | 1 | 0 |
| rmse | 1 | train | train_original_gaussian_corr0p03 | random_solver_y | loss1 | 5 | random_solver_y | 0.00425008 | 0.000642213 | -0.00360787 | 1 | 0 |
| rmse | 48 | generalization | burgers_widevis_l3target_d45 | random_solver_y | loss3 | 2 | random_solver_y | 0.00426401 | 0.00266482 | -0.00159919 | 1 | 0 |
| rmse | 14 | generalization | burgers_widevis_l3target_d11 | random_solver_y | loss3 | 2 | random_solver_y | 0.00496369 | 0.00483687 | -0.000126826 | 0.817395 | 0 |
| rmse | 40 | generalization | burgers_widevis_l3target_d37 | loss3 | random_solver_y | 1 | random_solver_y | 0.00861603 | 0.00973211 | 0.00111608 | 3.032e-30 | 1 |
| rmse | 18 | generalization | burgers_widevis_l3target_d15 | loss3 | random_solver_y | 1 | random_solver_y | 0.00726109 | 0.00844501 | 0.00118391 | 1.058e-05 | 1 |

## Output CSVs

- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/clean52_per_dataset_welch_tests_20260614/clean52_per_dataset_loss3_vs_best_other_welch_tests.csv`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/clean52_per_dataset_welch_tests_20260614/clean52_per_dataset_loss3_vs_best_other_welch_tests_compact.csv`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/clean52_per_dataset_welch_tests_20260614/clean52_per_dataset_welch_metric_summary.csv`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/clean52_per_dataset_welch_tests_20260614/clean52_per_dataset_welch_split_summary.csv`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/clean52_per_dataset_welch_tests_20260614/clean52_per_dataset_best_model_counts.csv`
