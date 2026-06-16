# Darcy Metric-Attack Correlation (20260612_10samples_loss3attack20_corr_chunk64)

- Created: 2026-06-12T08:23:38+00:00
- Samples: 10 generated Darcy samples.
- Models: loss1, loss2, loss3, physics.
- Rows: 40 model-sample pairs.
- Attack: shared binary loss3 attack, steps=20, epsilon_fraction=0.025.
- Sigma metric: block/2 top singular vector lifted to full grid, followed by one full-space power-refinement step.
- Binary first-order gain uses the top-k feasible binary flips scored by `delta^T J^T e`, scaled to MSE units.

## Overall Correlation Ranking By Absolute Spearman

| rank | metric | Pearson | Spearman | R2 | partial Pearson controlling clean loss |
|---:|---|---:|---:|---:|---:|
| 1 | `one_power_sigma` | -0.440 | -0.592 | 0.193 | -0.409 |
| 2 | `block2_sigma1` | -0.422 | -0.584 | 0.178 | -0.390 |
| 3 | `lifted_jv_sigma` | -0.422 | -0.584 | 0.178 | -0.390 |
| 4 | `clean_loss_before_attack` | -0.176 | -0.323 | 0.031 | nan |
| 5 | `sigma_power_times_error_l2` | -0.156 | -0.304 | 0.024 | 0.051 |
| 6 | `sigma_power_sq_times_error_l2_sq` | -0.208 | -0.304 | 0.043 | -0.152 |
| 7 | `jt_error_linf` | -0.123 | -0.274 | 0.015 | 0.138 |
| 8 | `binary_first_order_half_sse_gain_topk` | -0.120 | -0.267 | 0.014 | 0.152 |
| 9 | `binary_first_order_mse_gain_topk` | -0.120 | -0.267 | 0.014 | 0.152 |
| 10 | `binary_first_order_mse_gain_positive_topk` | -0.120 | -0.267 | 0.014 | 0.152 |
| 11 | `top_left_error_alignment_abs` | 0.511 | 0.258 | 0.262 | 0.658 |
| 12 | `jt_error_l2_norm` | -0.096 | -0.221 | 0.009 | 0.269 |

## Main Files

- metrics: `analysis_outputs/darcy_metric_correlation_10samples_20260612_10samples_loss3attack20_corr_chunk64/metrics_by_model_sample.csv`
- correlations overall/by-model: `analysis_outputs/darcy_metric_correlation_10samples_20260612_10samples_loss3attack20_corr_chunk64/correlations.csv`
- selected samples: `analysis_outputs/darcy_metric_correlation_10samples_20260612_10samples_loss3attack20_corr_chunk64/selected_samples.csv`
- figures: `visualizations/darcy_metric_correlation_10samples_20260612_10samples_loss3attack20_corr_chunk64`

## Interpretation Guide

If `||J^T e||`, `||J^T e||^2`, or binary first-order gain correlate more strongly with attack gain than `sigma_max`, then the attack is governed more by error-aligned sensitivity than by worst-case input-output sensitivity alone.
If `sigma_max * ||e||` improves over `sigma_max`, that means the clean error magnitude matters, but it still may be weaker than the actual aligned gradient `J^T e`.
