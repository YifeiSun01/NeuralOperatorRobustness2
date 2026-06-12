# Darcy Metric-Attack Correlation (20260612_25samples_loss3attack20_corr_chunk64)

- Created: 2026-06-12T11:13:03+00:00
- Samples: 25 generated Darcy samples.
- Models: loss1, loss2, loss3, physics.
- Rows: 100 model-sample pairs.
- Attack: shared binary loss3 attack, steps=20, epsilon_fraction=0.025.
- Sigma metric: block/2 top singular vector lifted to full grid, followed by one full-space power-refinement step.
- Binary first-order gain uses the top-k feasible binary flips scored by `delta^T J^T e`, scaled to MSE units.

## Overall Correlation Ranking By Absolute Spearman

| rank | metric | Pearson | Spearman | R2 | partial Pearson controlling clean loss |
|---:|---|---:|---:|---:|---:|
| 1 | `one_power_sigma` | -0.539 | -0.604 | 0.290 | -0.523 |
| 2 | `lifted_jv_sigma` | -0.528 | -0.599 | 0.279 | -0.510 |
| 3 | `block2_sigma1` | -0.528 | -0.599 | 0.279 | -0.509 |
| 4 | `sigma_power_times_error_l2` | -0.202 | -0.328 | 0.041 | -0.107 |
| 5 | `sigma_power_sq_times_error_l2_sq` | -0.242 | -0.328 | 0.059 | -0.237 |
| 6 | `clean_loss_before_attack` | -0.175 | -0.266 | 0.031 | nan |
| 7 | `jt_error_linf` | -0.136 | -0.262 | 0.019 | 0.044 |
| 8 | `binary_first_order_half_sse_gain_topk` | -0.131 | -0.256 | 0.017 | 0.061 |
| 9 | `binary_first_order_mse_gain_topk` | -0.131 | -0.256 | 0.017 | 0.061 |
| 10 | `binary_first_order_mse_gain_positive_topk` | -0.131 | -0.256 | 0.017 | 0.061 |
| 11 | `jt_error_l2_norm` | -0.121 | -0.248 | 0.015 | 0.114 |
| 12 | `jt_error_l2_norm_sq` | -0.184 | -0.248 | 0.034 | -0.056 |

## Main Files

- metrics: `analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/metrics_by_model_sample.csv`
- correlations overall/by-model: `analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/correlations.csv`
- selected samples: `analysis_outputs/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64/selected_samples.csv`
- figures: `visualizations/darcy_metric_correlation_25samples_20260612_25samples_loss3attack20_corr_chunk64`

## Interpretation Guide

If `||J^T e||`, `||J^T e||^2`, or binary first-order gain correlate more strongly with attack gain than `sigma_max`, then the attack is governed more by error-aligned sensitivity than by worst-case input-output sensitivity alone.
If `sigma_max * ||e||` improves over `sigma_max`, that means the clean error magnitude matters, but it still may be weaker than the actual aligned gradient `J^T e`.
