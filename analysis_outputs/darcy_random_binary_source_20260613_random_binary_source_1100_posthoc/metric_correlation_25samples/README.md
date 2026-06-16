# Darcy Metric-Attack Correlation (20260613_random_binary_source_1100_posthoc)

- Created: 2026-06-13T08:34:39+00:00
- Samples: 25 generated Darcy samples.
- Models: random_fixed_y, random_solver_y.
- Rows: 50 model-sample pairs.
- Attack: shared binary loss3 attack, steps=20, epsilon_fraction=0.025.
- Sigma metric: block/2 top singular vector lifted to full grid, followed by one full-space power-refinement step.
- Binary first-order gain uses the top-k feasible binary flips scored by `delta^T J^T e`, scaled to MSE units.

## Overall Correlation Ranking By Absolute Spearman

| rank | metric | Pearson | Spearman | R2 | partial Pearson controlling clean loss |
|---:|---|---:|---:|---:|---:|
| 1 | `top_left_error_alignment_abs` | 0.559 | 0.298 | 0.312 | 0.629 |
| 2 | `block2_sigma1` | -0.024 | -0.176 | 0.001 | -0.027 |
| 3 | `lifted_jv_sigma` | -0.024 | -0.175 | 0.001 | -0.028 |
| 4 | `one_power_sigma` | -0.033 | -0.155 | 0.001 | -0.038 |
| 5 | `clean_loss_before_attack` | -0.001 | -0.123 | 0.000 | nan |
| 6 | `sigma_power_times_error_l2` | 0.103 | -0.081 | 0.011 | 0.329 |
| 7 | `sigma_power_sq_times_error_l2_sq` | 0.023 | -0.081 | 0.001 | 0.098 |
| 8 | `error_l2_norm` | 0.135 | -0.057 | 0.018 | 0.426 |
| 9 | `error_l2_norm_sq` | 0.050 | -0.057 | 0.003 | 0.228 |
| 10 | `jt_error_linf` | 0.135 | -0.024 | 0.018 | 0.321 |
| 11 | `binary_first_order_half_sse_gain_topk` | 0.145 | -0.021 | 0.021 | 0.358 |
| 12 | `binary_first_order_mse_gain_topk` | 0.145 | -0.021 | 0.021 | 0.358 |

## Main Files

- metrics: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/metric_correlation_25samples/metrics_by_model_sample.csv`
- correlations overall/by-model: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/metric_correlation_25samples/correlations.csv`
- selected samples: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/metric_correlation_25samples/selected_samples.csv`
- figures: `visualizations/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/metric_correlation_25samples`

## Interpretation Guide

If `||J^T e||`, `||J^T e||^2`, or binary first-order gain correlate more strongly with attack gain than `sigma_max`, then the attack is governed more by error-aligned sensitivity than by worst-case input-output sensitivity alone.
If `sigma_max * ||e||` improves over `sigma_max`, that means the clean error magnitude matters, but it still may be weaker than the actual aligned gradient `J^T e`.
