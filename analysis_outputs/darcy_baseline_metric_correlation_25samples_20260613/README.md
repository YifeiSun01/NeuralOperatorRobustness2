# Darcy Metric-Attack Correlation (20260613_7model_complete_missing_metrics)

- Created: 2026-06-13T21:57:15+00:00
- Samples: 25 generated Darcy samples.
- Models: baseline.
- Rows: 25 model-sample pairs.
- Attack: shared binary loss3 attack, steps=20, epsilon_fraction=0.025.
- Sigma metric: block/2 top singular vector lifted to full grid, followed by one full-space power-refinement step.
- Binary first-order gain uses the top-k feasible binary flips scored by `delta^T J^T e`, scaled to MSE units.

## Overall Correlation Ranking By Absolute Spearman

| rank | metric | Pearson | Spearman | R2 | partial Pearson controlling clean loss |
|---:|---|---:|---:|---:|---:|
| 1 | `clean_loss_before_attack` | -0.283 | -0.661 | 0.080 | nan |
| 2 | `sigma_power_times_error_l2` | -0.143 | -0.612 | 0.020 | 0.448 |
| 3 | `sigma_power_sq_times_error_l2_sq` | -0.249 | -0.612 | 0.062 | 0.105 |
| 4 | `relative_l2` | -0.110 | -0.592 | 0.012 | 0.434 |
| 5 | `jt_error_l2_norm` | -0.114 | -0.590 | 0.013 | 0.521 |
| 6 | `jt_error_l2_norm_sq` | -0.234 | -0.590 | 0.055 | 0.125 |
| 7 | `jt_error_linf` | -0.096 | -0.588 | 0.009 | 0.375 |
| 8 | `top1_weighted_error_energy` | -0.232 | -0.588 | 0.054 | 0.104 |
| 9 | `binary_first_order_half_sse_gain_topk` | -0.110 | -0.587 | 0.012 | 0.382 |
| 10 | `binary_first_order_mse_gain_topk` | -0.110 | -0.587 | 0.012 | 0.382 |
| 11 | `binary_first_order_mse_gain_positive_topk` | -0.110 | -0.587 | 0.012 | 0.382 |
| 12 | `error_l2_norm` | -0.134 | -0.583 | 0.018 | 0.509 |

## Main Files

- metrics: `analysis_outputs/darcy_baseline_metric_correlation_25samples_20260613/metrics_by_model_sample.csv`
- correlations overall/by-model: `analysis_outputs/darcy_baseline_metric_correlation_25samples_20260613/correlations.csv`
- selected samples: `analysis_outputs/darcy_baseline_metric_correlation_25samples_20260613/selected_samples.csv`
- figures: `visualizations/darcy_baseline_metric_correlation_25samples_20260613`

## Interpretation Guide

If `||J^T e||`, `||J^T e||^2`, or binary first-order gain correlate more strongly with attack gain than `sigma_max`, then the attack is governed more by error-aligned sensitivity than by worst-case input-output sensitivity alone.
If `sigma_max * ||e||` improves over `sigma_max`, that means the clean error magnitude matters, but it still may be weaker than the actual aligned gradient `J^T e`.
