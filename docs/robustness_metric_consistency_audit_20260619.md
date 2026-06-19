# Robustness Metric Consistency Audit

Generated: 2026-06-19T13:55:55+00:00

This audit checks the finite-budget robustness metrics behind the Burgers and Darcy log-log plots.

Metrics checked:
- `adv_loss_mean`: final attacked loss.
- `absolute_loss_increase`: `adv_loss_mean - clean_loss_mean` for Burgers, recorded `loss_increase_mean` for Darcy.
- `relative_increase_mean`: mean per-sample relative increase when present.
- `batch_ratio_percent`: `100 * (adv_loss_mean / clean_loss_mean - 1)`.

Key finding:
- Burgers generalization curves are monotone in the compact sweep, but the relative/percent metrics invert the cross-model ranking because clean-loss denominators differ strongly. Absolute increase and final attacked loss rank `loss3` best.
- Darcy generalization curves are not monotone for several methods and metrics in the original budget sweep. The source attack is `binary_darcy_replace_attack`, which recomputes an exact `k = round(epsilon_fraction * n_pix)` replacement set for each budget rather than preserving a nested lower-budget solution. Larger budget therefore means more flipped pixels, not guaranteed larger loss under this heuristic.
- Darcy `delta_l2_rms_mean` is monotone increasing for every method in the generalization sweep, while final attacked loss and loss increase are not monotone for most methods. This supports the interpretation that the budget/delta grows, but the exact-k replacement heuristic does not produce a nested best-loss envelope.

Max-epsilon generalization metrics:

| problem | method | epsilon | clean_loss_mean | adv_loss_mean | absolute_loss_increase | relative_increase_mean | batch_ratio_percent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| burgers | loss3 | 0.12 | 0.000118892 | 0.00270727 | 0.00258837 | 46.627 | 2177.09 |
| burgers | loss2 | 0.12 | 0.000525876 | 0.00575644 | 0.00523056 | 16.3865 | 994.637 |
| burgers | loss1 | 0.12 | 0.000533297 | 0.00585409 | 0.0053208 | 17.6725 | 997.717 |
| burgers | random_solver_y | 0.12 | 0.000401236 | 0.00618193 | 0.0057807 | 89.6309 | 1440.72 |
| burgers | baseline | 0.12 | 0.000542865 | 0.0069502 | 0.00640733 | 11.1661 | 1180.28 |
| burgers | random_clean_y | 0.12 | 0.0119638 | 0.037956 | 0.0259922 | 2.22598 | 217.257 |
| darcy | loss3 | 0.075 | 3.56097e-07 | 1.03144e-06 | 6.7534e-07 | 2.40522 | 189.651 |
| darcy | loss2 | 0.075 | 7.42728e-07 | 5.98901e-06 | 5.24629e-06 | 19.5904 | 706.353 |
| darcy | physics_loss | 0.075 | 1.29572e-06 | 9.36069e-06 | 8.06497e-06 | 16.2215 | 622.434 |
| darcy | random_solver | 0.075 | 1.31874e-06 | 9.46199e-06 | 8.14326e-06 | 15.1245 | 617.504 |
| darcy | random_clean | 0.075 | 1.13975e-06 | 1.00794e-05 | 8.93964e-06 | 24.8956 | 784.352 |
| darcy | baseline | 0.075 | 1.49282e-06 | 1.12049e-05 | 9.71213e-06 | 18.2135 | 650.59 |
| darcy | loss1 | 0.075 | 1.75568e-06 | 1.21764e-05 | 1.04208e-05 | 14.5901 | 593.547 |

Mean rank by metric over epsilon grid; lower is better:

| problem | metric | method | mean_rank_lower_better |
| --- | --- | --- | --- |
| burgers | adv_loss_mean | loss3 | 1 |
| burgers | adv_loss_mean | random_solver_y | 2.66667 |
| burgers | adv_loss_mean | loss2 | 2.73333 |
| burgers | adv_loss_mean | loss1 | 3.6 |
| burgers | adv_loss_mean | baseline | 5 |
| burgers | adv_loss_mean | random_clean_y | 6 |
| burgers | absolute_loss_increase | loss3 | 1 |
| burgers | absolute_loss_increase | loss1 | 3 |
| burgers | absolute_loss_increase | random_solver_y | 3.06667 |
| burgers | absolute_loss_increase | loss2 | 3.26667 |
| burgers | absolute_loss_increase | baseline | 4.66667 |
| burgers | absolute_loss_increase | random_clean_y | 6 |
| burgers | relative_increase_mean | random_clean_y | 1 |
| burgers | relative_increase_mean | baseline | 2 |
| burgers | relative_increase_mean | loss2 | 3 |
| burgers | relative_increase_mean | loss1 | 4.06667 |
| burgers | relative_increase_mean | loss3 | 4.93333 |
| burgers | relative_increase_mean | random_solver_y | 6 |
| burgers | batch_ratio_percent | random_clean_y | 1 |
| burgers | batch_ratio_percent | loss1 | 2.26667 |
| burgers | batch_ratio_percent | loss2 | 3.2 |
| burgers | batch_ratio_percent | baseline | 3.53333 |
| burgers | batch_ratio_percent | random_solver_y | 5 |
| burgers | batch_ratio_percent | loss3 | 6 |
| darcy | adv_loss_mean | loss3 | 1 |
| darcy | adv_loss_mean | loss2 | 2 |
| darcy | adv_loss_mean | physics_loss | 3.14286 |
| darcy | adv_loss_mean | random_solver | 4.28571 |
| darcy | adv_loss_mean | random_clean | 4.57143 |
| darcy | adv_loss_mean | baseline | 6 |
| darcy | adv_loss_mean | loss1 | 7 |
| darcy | absolute_loss_increase | loss3 | 1 |
| darcy | absolute_loss_increase | loss2 | 2 |
| darcy | absolute_loss_increase | physics_loss | 3 |
| darcy | absolute_loss_increase | random_solver | 4.14286 |
| darcy | absolute_loss_increase | random_clean | 4.85714 |
| darcy | absolute_loss_increase | baseline | 6 |
| darcy | absolute_loss_increase | loss1 | 7 |
| darcy | relative_increase_mean | loss3 | 1 |
| darcy | relative_increase_mean | loss1 | 2.42857 |
| darcy | relative_increase_mean | random_solver | 2.85714 |
| darcy | relative_increase_mean | physics_loss | 3.71429 |
| darcy | relative_increase_mean | baseline | 5.14286 |
| darcy | relative_increase_mean | loss2 | 5.85714 |
| darcy | relative_increase_mean | random_clean | 7 |
| darcy | batch_ratio_percent | loss3 | 1 |
| darcy | batch_ratio_percent | loss1 | 2.57143 |
| darcy | batch_ratio_percent | physics_loss | 2.71429 |
| darcy | batch_ratio_percent | random_solver | 4 |
| darcy | batch_ratio_percent | baseline | 5.28571 |
| darcy | batch_ratio_percent | loss2 | 5.42857 |
| darcy | batch_ratio_percent | random_clean | 7 |

Non-monotone segments:

| problem | metric | method | epsilon_from | epsilon_to | value_from | value_to | drop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| darcy | adv_loss_mean | baseline | 0.05 | 0.075 | 1.12704e-05 | 1.12049e-05 | 6.54672e-08 |
| darcy | adv_loss_mean | loss1 | 0.0375 | 0.04375 | 1.27602e-05 | 1.27085e-05 | 5.17403e-08 |
| darcy | adv_loss_mean | loss1 | 0.04375 | 0.05 | 1.27085e-05 | 1.26309e-05 | 7.76104e-08 |
| darcy | adv_loss_mean | loss1 | 0.05 | 0.075 | 1.26309e-05 | 1.21764e-05 | 4.54426e-07 |
| darcy | adv_loss_mean | loss3 | 0.0125 | 0.025 | 1.26322e-06 | 1.18337e-06 | 7.98485e-08 |
| darcy | adv_loss_mean | loss3 | 0.025 | 0.0375 | 1.18337e-06 | 1.13082e-06 | 5.25491e-08 |
| darcy | adv_loss_mean | loss3 | 0.0375 | 0.04375 | 1.13082e-06 | 1.11205e-06 | 1.87707e-08 |
| darcy | adv_loss_mean | loss3 | 0.04375 | 0.05 | 1.11205e-06 | 1.09572e-06 | 1.63295e-08 |
| darcy | adv_loss_mean | loss3 | 0.05 | 0.075 | 1.09572e-06 | 1.03144e-06 | 6.42826e-08 |
| darcy | adv_loss_mean | physics_loss | 0.05 | 0.075 | 9.42983e-06 | 9.36069e-06 | 6.91439e-08 |
| darcy | adv_loss_mean | random_clean | 0.05 | 0.075 | 1.01158e-05 | 1.00794e-05 | 3.64504e-08 |
| darcy | adv_loss_mean | random_solver | 0.0375 | 0.04375 | 9.7643e-06 | 9.73721e-06 | 2.70894e-08 |
| darcy | adv_loss_mean | random_solver | 0.04375 | 0.05 | 9.73721e-06 | 9.7e-06 | 3.72086e-08 |
| darcy | adv_loss_mean | random_solver | 0.05 | 0.075 | 9.7e-06 | 9.46199e-06 | 2.38007e-07 |
| darcy | absolute_loss_increase | baseline | 0.05 | 0.075 | 9.77759e-06 | 9.71213e-06 | 6.54672e-08 |
| darcy | absolute_loss_increase | loss1 | 0.0375 | 0.04375 | 1.10045e-05 | 1.09528e-05 | 5.17403e-08 |
| darcy | absolute_loss_increase | loss1 | 0.04375 | 0.05 | 1.09528e-05 | 1.08752e-05 | 7.76104e-08 |
| darcy | absolute_loss_increase | loss1 | 0.05 | 0.075 | 1.08752e-05 | 1.04208e-05 | 4.54426e-07 |
| darcy | absolute_loss_increase | loss3 | 0.0125 | 0.025 | 9.07121e-07 | 8.27272e-07 | 7.98485e-08 |
| darcy | absolute_loss_increase | loss3 | 0.025 | 0.0375 | 8.27272e-07 | 7.74723e-07 | 5.25491e-08 |
| darcy | absolute_loss_increase | loss3 | 0.0375 | 0.04375 | 7.74723e-07 | 7.55953e-07 | 1.87707e-08 |
| darcy | absolute_loss_increase | loss3 | 0.04375 | 0.05 | 7.55953e-07 | 7.39623e-07 | 1.63295e-08 |
| darcy | absolute_loss_increase | loss3 | 0.05 | 0.075 | 7.39623e-07 | 6.7534e-07 | 6.42826e-08 |
| darcy | absolute_loss_increase | physics_loss | 0.05 | 0.075 | 8.13412e-06 | 8.06497e-06 | 6.91439e-08 |
| darcy | absolute_loss_increase | random_clean | 0.05 | 0.075 | 8.97609e-06 | 8.93964e-06 | 3.64504e-08 |
| darcy | absolute_loss_increase | random_solver | 0.0375 | 0.04375 | 8.44556e-06 | 8.41847e-06 | 2.70894e-08 |
| darcy | absolute_loss_increase | random_solver | 0.04375 | 0.05 | 8.41847e-06 | 8.38127e-06 | 3.72086e-08 |
| darcy | absolute_loss_increase | random_solver | 0.05 | 0.075 | 8.38127e-06 | 8.14326e-06 | 2.38007e-07 |
| darcy | relative_increase_mean | baseline | 0.05 | 0.075 | 18.3242 | 18.2135 | 0.110673 |
| darcy | relative_increase_mean | loss1 | 0.0375 | 0.04375 | 15.258 | 15.1897 | 0.0683758 |
| darcy | relative_increase_mean | loss1 | 0.04375 | 0.05 | 15.1897 | 15.0671 | 0.122615 |
| darcy | relative_increase_mean | loss1 | 0.05 | 0.075 | 15.0671 | 14.5901 | 0.476988 |
| darcy | relative_increase_mean | loss3 | 0.0125 | 0.025 | 3.0889 | 2.85228 | 0.236617 |
| darcy | relative_increase_mean | loss3 | 0.025 | 0.0375 | 2.85228 | 2.7077 | 0.144583 |
| darcy | relative_increase_mean | loss3 | 0.0375 | 0.04375 | 2.7077 | 2.6594 | 0.0482949 |
| darcy | relative_increase_mean | loss3 | 0.04375 | 0.05 | 2.6594 | 2.61089 | 0.0485105 |
| darcy | relative_increase_mean | loss3 | 0.05 | 0.075 | 2.61089 | 2.40522 | 0.205677 |
| darcy | relative_increase_mean | physics_loss | 0.05 | 0.075 | 16.2287 | 16.2215 | 0.00721571 |
| darcy | relative_increase_mean | random_solver | 0.0375 | 0.04375 | 15.3823 | 15.3103 | 0.07193 |
| darcy | relative_increase_mean | random_solver | 0.04375 | 0.05 | 15.3103 | 15.2927 | 0.0176255 |
| darcy | relative_increase_mean | random_solver | 0.05 | 0.075 | 15.2927 | 15.1245 | 0.168229 |
| darcy | batch_ratio_percent | baseline | 0.05 | 0.075 | 654.975 | 650.59 | 4.38548 |
| darcy | batch_ratio_percent | loss1 | 0.0375 | 0.04375 | 626.798 | 623.851 | 2.94703 |
| darcy | batch_ratio_percent | loss1 | 0.04375 | 0.05 | 623.851 | 619.43 | 4.42054 |
| darcy | batch_ratio_percent | loss1 | 0.05 | 0.075 | 619.43 | 593.547 | 25.8832 |
| darcy | batch_ratio_percent | loss3 | 0.0125 | 0.025 | 254.74 | 232.317 | 22.4233 |
| darcy | batch_ratio_percent | loss3 | 0.025 | 0.0375 | 232.317 | 217.56 | 14.757 |
| darcy | batch_ratio_percent | loss3 | 0.0375 | 0.04375 | 217.56 | 212.289 | 5.27123 |
| darcy | batch_ratio_percent | loss3 | 0.04375 | 0.05 | 212.289 | 207.703 | 4.5857 |
| darcy | batch_ratio_percent | loss3 | 0.05 | 0.075 | 207.703 | 189.651 | 18.052 |
| darcy | batch_ratio_percent | physics_loss | 0.05 | 0.075 | 627.77 | 622.434 | 5.33635 |
| darcy | batch_ratio_percent | random_clean | 0.05 | 0.075 | 787.55 | 784.352 | 3.19811 |
| darcy | batch_ratio_percent | random_solver | 0.0375 | 0.04375 | 640.428 | 638.374 | 2.05419 |
| darcy | batch_ratio_percent | random_solver | 0.04375 | 0.05 | 638.374 | 635.553 | 2.82154 |
| darcy | batch_ratio_percent | random_solver | 0.05 | 0.075 | 635.553 | 617.504 | 18.0481 |

Darcy delta-vs-loss monotonicity:

| problem | method | delta_l2_monotone | adv_loss_monotone | loss_increase_monotone | delta_l2_first | delta_l2_last |
| --- | --- | --- | --- | --- | --- | --- |
| darcy | baseline | True | False | False | 3.10424 | 4.49145 |
| darcy | loss1 | True | False | False | 3.10577 | 4.45228 |
| darcy | loss2 | True | True | True | 2.88927 | 4.2769 |
| darcy | loss3 | True | False | False | 2.98717 | 4.58769 |
| darcy | physics_loss | True | False | False | 3.03274 | 4.36433 |
| darcy | random_clean | True | False | False | 3.05457 | 4.48239 |
| darcy | random_solver | True | False | False | 3.06813 | 4.40012 |

Output CSVs:
- `analysis_outputs/robustness_metric_consistency_20260619/generalization_max_epsilon_metrics.csv`
- `analysis_outputs/robustness_metric_consistency_20260619/generalization_mean_ranks_by_metric.csv`
- `analysis_outputs/robustness_metric_consistency_20260619/generalization_baseline_comparison_at_max_epsilon.csv`
- `analysis_outputs/robustness_metric_consistency_20260619/darcy_generalization_delta_vs_loss_monotonicity.csv`
- `analysis_outputs/robustness_metric_consistency_20260619/generalization_nonmonotone_segments.csv`
