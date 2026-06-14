# Burgers First-Vs-Second Significance, 20260614

Generated: 2026-06-14T21:47:40+00:00

This report checks whether `loss3`, when it is the mean-best model, is significantly lower than the actual runner-up across paired dataset rows. It reads existing clean and strict latest attack tables only.

Methods:
- Paired unit: dataset row.
- Clean scopes: all 52 rows and 50 generalization rows.
- Attack scopes: strict latest all 52 rows and 50 generalization rows.
- Direction: lower is better.
- Tests: paired one-sided t-test, one-sided Wilcoxon signed-rank test, and 100000 paired bootstrap resamples of the mean improvement.
- Improvement is `runner_up - loss3`; positive means `loss3` is lower/better.

## Key Result

For the requested clean loss/error metrics and attack loss-increase metrics, `loss3` is the mean-best model and is significantly lower than the actual runner-up. The bootstrap 95% CI for the paired mean improvement is above zero in every listed clean and attack row.

## Compact Table

| family | scope | metric | n_dataset_rows | mean_best_model | runner_up_model | loss3_row_wins_vs_runner | loss3_mean | loss3_std | runner_up_mean | runner_up_std | paired_improvement_mean_runner_minus_loss3 | paired_improvement_std | paired_t_p_one_sided | wilcoxon_p_one_sided | bootstrap_mean_improvement_ci95_low | bootstrap_mean_improvement_ci95_high | bootstrap_p_mean_improvement_le_0 | significant_t_p05 | significant_bootstrap_ci_excludes_0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | all_52 | mse | 52 | loss3 | random_solver_y | 49 | 0.000190935 | 0.000178737 | 0.000488038 | 0.000498568 | 0.000297103 | 0.000327243 | 1.414e-08 | 2.957e-10 | 0.000212535 | 0.000388249 | 1.000e-05 | 1 | 1 |
| clean | generalization_50 | mse | 50 | loss3 | random_solver_y | 49 | 0.000197775 | 0.000178912 | 0.000507539 | 0.000498624 | 0.000309765 | 0.000327423 | 1.004e-08 | 2.665e-15 | 0.000223087 | 0.000401595 | 1.000e-05 | 1 | 1 |
| clean | all_52 | rmse | 52 | loss3 | random_solver_y | 48 | 0.011752 | 0.00541205 | 0.0174937 | 0.00986755 | 0.00574178 | 0.00511702 | 5.226e-11 | 5.353e-09 | 0.00438757 | 0.00714612 | 1.000e-05 | 1 | 1 |
| clean | generalization_50 | rmse | 50 | loss3 | random_solver_y | 48 | 0.0120496 | 0.00530433 | 0.0181661 | 0.0094525 | 0.00611652 | 0.00485045 | 3.924e-12 | 3.819e-14 | 0.00481607 | 0.007477 | 1.000e-05 | 1 | 1 |
| clean | all_52 | relative_l2 | 52 | loss3 | random_solver_y | 48 | 0.0230506 | 0.0132093 | 0.034197 | 0.022643 | 0.0111464 | 0.0103768 | 1.816e-10 | 5.074e-09 | 0.00841893 | 0.0139774 | 1.000e-05 | 1 | 1 |
| clean | generalization_50 | relative_l2 | 50 | loss3 | random_solver_y | 48 | 0.0236437 | 0.0131252 | 0.0355127 | 0.0220841 | 0.0118689 | 0.00991079 | 1.854e-11 | 4.885e-14 | 0.00922388 | 0.0146792 | 1.000e-05 | 1 | 1 |
| attack | all_52 | attack_loss_increase_mean | 52 | loss3 | loss1 | 52 | 0.00382075 | 0.00255594 | 0.00658905 | 0.0037323 | 0.0027683 | 0.0012507 | 8.893e-22 | 1.752e-10 | 0.00243581 | 0.00311029 | 1.000e-05 | 1 | 1 |
| attack | generalization_50 | attack_loss_increase_mean | 50 | loss3 | loss1 | 50 | 0.00394899 | 0.00252254 | 0.00679898 | 0.00365092 | 0.00284999 | 0.00120452 | 3.376e-22 | 8.882e-16 | 0.0025193 | 0.00318597 | 1.000e-05 | 1 | 1 |
| attack | all_52 | final_loss_mean | 52 | loss3 | loss1 | 52 | 0.00400196 | 0.00271406 | 0.00715168 | 0.00412037 | 0.00314972 | 0.00148477 | 5.374e-21 | 1.752e-10 | 0.00275531 | 0.0035541 | 1.000e-05 | 1 | 1 |
| attack | generalization_50 | final_loss_mean | 50 | loss3 | loss1 | 50 | 0.0041372 | 0.00267984 | 0.00738407 | 0.00402956 | 0.00324688 | 0.00142966 | 1.856e-21 | 8.882e-16 | 0.00286172 | 0.00364401 | 1.000e-05 | 1 | 1 |

## Output CSVs

- `data/first_vs_second_significance_20260614/first_vs_second_key_metric_significance.csv`
- `data/first_vs_second_significance_20260614/first_vs_second_key_metric_significance_compact.csv`
