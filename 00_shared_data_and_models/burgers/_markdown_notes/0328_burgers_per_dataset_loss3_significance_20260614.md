# Burgers Per-Dataset Loss3 Significance, 20260614

Generated: 2026-06-14T21:57:39+00:00

This report answers the per-dataset version of the significance question. It uses the strict latest same-manifest P2Q2 attack NPZ files and runs paired tests inside each dataset row.

## What Was Tested

- Paired unit: one attack-manifest sample inside one dataset.
- Dataset rows: 52.
- Metrics per dataset: `initial_loss`, `final_loss`, and `attack_loss_increase`.
- Total tests: 156 rows.
- Comparison: `loss3` versus the best non-loss3 model for that same dataset and metric. When `loss3` is best, that comparison model is the runner-up.
- Tests: one-sided paired t-test, one-sided Wilcoxon signed-rank test, BH/FDR q-values over all 156 rows, and 20000 paired bootstrap resamples for the mean advantage.

Positive advantage means the comparison model has larger loss than `loss3`, so `loss3` is better.

## Metric Summary

| metric | dataset_rows | loss3_best_rows | loss3_rank1_rows | loss3_significantly_better_rows | loss3_significantly_worse_rows | mean_advantage | min_advantage | max_advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial_loss | 52 | 49 | 49 | 49 | 2 | 0.000280059 | -5.658e-06 | 0.000931606 |
| final_loss | 52 | 52 | 52 | 52 | 0 | 0.00287222 | 0.000659185 | 0.00652441 |
| attack_loss_increase | 52 | 52 | 52 | 52 | 0 | 0.00246923 | 0.00021181 | 0.00559281 |

## Split Summary

| metric | split | dataset_rows | loss3_best_rows | loss3_significantly_better_rows | loss3_significantly_worse_rows |
| --- | --- | --- | --- | --- | --- |
| initial_loss | train | 1 | 0 | 0 | 1 |
| final_loss | train | 1 | 1 | 1 | 0 |
| attack_loss_increase | train | 1 | 1 | 1 | 0 |
| initial_loss | test | 1 | 0 | 0 | 1 |
| final_loss | test | 1 | 1 | 1 | 0 |
| attack_loss_increase | test | 1 | 1 | 1 | 0 |
| initial_loss | generalization | 50 | 49 | 49 | 0 |
| final_loss | generalization | 50 | 50 | 50 | 0 |
| attack_loss_increase | generalization | 50 | 50 | 50 | 0 |

## Attack Loss Increase Rows With Smallest Loss3 Advantage

| dataset_order | split | dataset_id | n_pairs | best_model | second_model | comparison_model | loss3_mean | comparison_mean | mean_advantage_comparison_minus_loss3 | paired_t_q_loss3_better_bh_fdr_all156 | bootstrap_mean_advantage_ci95_low | bootstrap_mean_advantage_ci95_high | loss3_significantly_better_q05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 52 | generalization | burgers_widevis_l3target_d49 | 200 | loss3 | loss1 | loss1 | 0.00185251 | 0.00206432 | 0.00021181 | 1.586e-36 | 0.000185412 | 0.000238142 | 1 |
| 1 | train | train_original_gaussian_corr0p03_first50 | 50 | loss3 | loss1 | loss1 | 0.000606907 | 0.00127111 | 0.000664203 | 1.750e-16 | 0.000557374 | 0.000774255 | 1 |
| 2 | test | test_original_gaussian_corr0p03 | 150 | loss3 | loss1 | loss1 | 0.000623062 | 0.00141068 | 0.000787622 | 8.544e-40 | 0.000705594 | 0.000873121 | 1 |
| 48 | generalization | burgers_widevis_l3target_d45 | 200 | loss3 | loss1 | loss1 | 0.000927731 | 0.00178314 | 0.00085541 | 5.044e-53 | 0.000778446 | 0.000935176 | 1 |
| 46 | generalization | burgers_widevis_l3target_d43 | 200 | loss3 | baseline | baseline | 0.00236195 | 0.00335209 | 0.000990138 | 3.014e-79 | 0.000927655 | 0.00104982 | 1 |
| 41 | generalization | burgers_widevis_l3target_d38 | 200 | loss3 | baseline | baseline | 0.00284717 | 0.00386073 | 0.00101356 | 3.535e-72 | 0.00094449 | 0.00108223 | 1 |
| 50 | generalization | burgers_widevis_l3target_d47 | 200 | loss3 | loss2 | loss2 | 0.0020146 | 0.00312414 | 0.00110954 | 4.885e-179 | 0.00108974 | 0.00112914 | 1 |
| 43 | generalization | burgers_widevis_l3target_d40 | 200 | loss3 | loss2 | loss2 | 0.00127702 | 0.00244324 | 0.00116622 | 8.532e-141 | 0.00113309 | 0.00119878 | 1 |
| 47 | generalization | burgers_widevis_l3target_d44 | 200 | loss3 | loss2 | loss2 | 0.00213922 | 0.00336902 | 0.00122981 | 2.683e-151 | 0.00119961 | 0.00126043 | 1 |
| 42 | generalization | burgers_widevis_l3target_d39 | 200 | loss3 | loss2 | loss2 | 0.00152547 | 0.00278601 | 0.00126054 | 2.462e-149 | 0.00122829 | 0.00129237 | 1 |

## Interpretation

- Attack loss increase: `loss3` is rank 1 on 52/52 dataset rows and significantly better after all-156 BH/FDR plus bootstrap on 52/52 rows.
- Attack final loss: `loss3` is rank 1 on 52/52 dataset rows and significantly better on 52/52 rows.
- Initial clean loss on the attack manifest: `loss3` is rank 1 on 49/52 dataset rows and significantly better on 49/52 rows. This is clean MSE on the attack manifest, not the separate full clean RMSE/Relative L2 table.

## Output CSVs

- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/per_dataset_loss3_significance_20260614/per_dataset_loss3_vs_best_other_significance.csv`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/per_dataset_loss3_significance_20260614/per_dataset_loss3_vs_best_other_significance_compact.csv`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/per_dataset_loss3_significance_20260614/per_dataset_loss3_significance_metric_summary.csv`
- `outputs/burgers_timematched_solver7860_clean8000_audit_20260614/data/per_dataset_loss3_significance_20260614/per_dataset_loss3_significance_split_summary.csv`
