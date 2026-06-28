# Loss3 Alpha/Epsilon Core-Four Sweep Result - 2026-05-19

Status: generated from completed local sweep outputs.

## Source Data

Observed from local files:

- Sweep root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520`
- Analysis root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_alpha_epsilon_core4_analysis_baseline_giftrace_20260520`
- Completed setting roots analyzed: `1`
- Per-setting source: `per_step_metrics.csv`, `per_sample_step_metrics.csv`, `manifest.json`, and `config.json`.

Methods compared:

- `raw_add`
- `raw_replace`
- `steepest_add`
- `steepest_replace`

Metrics used:

- final `loss3_q` mean/std/nonfinite count
- step to reach 90% and 95% of the best final mean loss within the same epsilon/alpha setting
- `delta_pnorm_mean`, `boundary_ratio_mean`, final boundary gap, and first step to 95%/99% boundary
- per-sample first step to 95%/99% boundary, including the max step needed for all reached samples
- final high-frequency energy ratio, spectral centroid, first-derivative L2, and total variation

## Method Rollup

| method | setting_count | mean_final_loss3_q | median_step_to_95pct_setting_best | median_step_to_boundary_ratio_mean_0.99 | median_all_samples_boundary_0.99_step | mean_final_boundary_ratio | mean_final_high_frequency_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_add | 1 | 3 | 123 | 43 | 56 | 1 | 4.041e-08 |
| raw_replace | 1 | 3.062 | 10 | 1 | 1 | 1 | 3.035e-09 |
| steepest_add | 1 | 3.065 | 50 | 11 | 14 | 1 | 4.958e-08 |
| steepest_replace | 1 | 3.062 | 10 | 1 | 1 | 1 | 3.035e-09 |

## Boundary Arrival Preview

| epsilon | alpha | method | nominal_l2_steepest_boundary_steps | final_delta_pnorm_mean | final_boundary_ratio_mean | step_to_boundary_ratio_mean_0.99 | sample_step_to_0.99_boundary_max | sample_step_to_0.99_boundary_not_reached_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.4 | raw_add | 10 | 4 | 1 | 43 | 56 | 0 |
| 4 | 0.4 | raw_replace | 10 | 4 | 1 | 1 | 1 | 0 |
| 4 | 0.4 | steepest_add | 10 | 4 | 1 | 11 | 14 | 0 |
| 4 | 0.4 | steepest_replace | 10 | 4 | 1 | 1 | 1 | 0 |

## Per-Setting Winners

| epsilon | alpha | criterion | winning_method | winning_value |
| --- | --- | --- | --- | --- |
| 4 | 0.4 | largest_final_loss | steepest_add | 3.065 |
| 4 | 0.4 | fastest_to_95pct_loss | raw_replace | 10 |
| 4 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | lowest_high_frequency_ratio | raw_replace | 3.035e-09 |
| 4 | 0.4 | lowest_first_derivative_l2 | raw_replace | 0.161 |
| 4 | 0.4 | lowest_total_variation | raw_replace | 2.002 |

## Evidence And Interpretation

Observed from the generated CSV tables above: final-loss, boundary-arrival speed, and smoothness winners are computed within each fixed `(epsilon, alpha, p, q)` setting.

Inference from these summaries should be made within the fixed `p=2,q=2` scope unless additional P/Q roots are added and analyzed.

Boundary arrival uses `boundary_ratio = ||delta||_p / epsilon`. A 99% threshold is used as the main practical boundary test to avoid numerical roundoff around exactly `1.0`.

The smoothness metrics are numerical proxies for whether the final perturbation is high-frequency or sharp. They should be read together with the saved delta trajectory/heatmap figures under each setting root.
