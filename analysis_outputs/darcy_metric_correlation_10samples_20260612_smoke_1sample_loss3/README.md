# Darcy Metric-Attack Correlation (20260612_smoke_1sample_loss3)

- Created: 2026-06-12T08:09:55+00:00
- Samples: 1 generated Darcy samples.
- Models: loss3.
- Rows: 1 model-sample pairs.
- Attack: shared binary loss3 attack, steps=20, epsilon_fraction=0.025.
- Sigma metric: block/2 top singular vector lifted to full grid, followed by one full-space power-refinement step.
- Binary first-order gain uses the top-k feasible binary flips scored by `delta^T J^T e`, scaled to MSE units.

## Overall Correlation Ranking By Absolute Spearman

| rank | metric | Pearson | Spearman | R2 | partial Pearson controlling clean loss |
|---:|---|---:|---:|---:|---:|
| 1 | `clean_loss_before_attack` | nan | nan | nan | nan |
| 2 | `error_l2_norm` | nan | nan | nan | nan |
| 3 | `error_l2_norm_sq` | nan | nan | nan | nan |
| 4 | `relative_l2` | nan | nan | nan | nan |
| 5 | `jt_error_l2_norm` | nan | nan | nan | nan |
| 6 | `jt_error_l2_norm_sq` | nan | nan | nan | nan |
| 7 | `jt_error_linf` | nan | nan | nan | nan |
| 8 | `binary_first_order_half_sse_gain_topk` | nan | nan | nan | nan |
| 9 | `binary_first_order_mse_gain_topk` | nan | nan | nan | nan |
| 10 | `binary_first_order_mse_gain_positive_topk` | nan | nan | nan | nan |
| 11 | `block2_sigma1` | nan | nan | nan | nan |
| 12 | `lifted_jv_sigma` | nan | nan | nan | nan |

## Main Files

- metrics: `analysis_outputs/darcy_metric_correlation_10samples_20260612_smoke_1sample_loss3/metrics_by_model_sample.csv`
- correlations overall/by-model: `analysis_outputs/darcy_metric_correlation_10samples_20260612_smoke_1sample_loss3/correlations.csv`
- selected samples: `analysis_outputs/darcy_metric_correlation_10samples_20260612_smoke_1sample_loss3/selected_samples.csv`
- figures: `visualizations/darcy_metric_correlation_10samples_20260612_smoke_1sample_loss3`

## Interpretation Guide

If `||J^T e||`, `||J^T e||^2`, or binary first-order gain correlate more strongly with attack gain than `sigma_max`, then the attack is governed more by error-aligned sensitivity than by worst-case input-output sensitivity alone.
If `sigma_max * ||e||` improves over `sigma_max`, that means the clean error magnitude matters, but it still may be weaker than the actual aligned gradient `J^T e`.
