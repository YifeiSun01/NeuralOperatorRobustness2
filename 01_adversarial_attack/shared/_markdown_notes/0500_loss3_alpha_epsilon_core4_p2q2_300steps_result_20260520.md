# Loss3 Alpha/Epsilon Core-Four Sweep Result - 2026-05-19

Status: generated from completed local sweep outputs.

## Source Data

Observed from local files:

- Sweep root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520`
- Analysis root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520`
- Completed setting roots analyzed: `20`
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
| raw_add | 20 | 5.659 | 94 | 42.5 | 49.5 | 1 | 4.46e-06 |
| raw_replace | 20 | 5.747 | 7 | 1 | 1 | 1 | 5.06e-09 |
| steepest_add | 20 | 5.827 | 36 | 11 | 12 | 1 | 4.337e-06 |
| steepest_replace | 20 | 5.747 | 7 | 1 | 1 | 1 | 5.06e-09 |

## Boundary Arrival Preview

| epsilon | alpha | method | nominal_l2_steepest_boundary_steps | final_delta_pnorm_mean | final_boundary_ratio_mean | step_to_boundary_ratio_mean_0.99 | sample_step_to_0.99_boundary_max | sample_step_to_0.99_boundary_not_reached_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 0.6 | raw_add | 20 | 12 | 1 | 86 | 102 | 0 |
| 12 | 0.6 | raw_replace | 20 | 12 | 1 | 1 | 1 | 0 |
| 12 | 0.6 | steepest_add | 20 | 12 | 1 | 25 | 29 | 0 |
| 12 | 0.6 | steepest_replace | 20 | 12 | 1 | 1 | 1 | 0 |
| 12 | 1.2 | raw_add | 10 | 12 | 1 | 44 | 51 | 0 |
| 12 | 1.2 | raw_replace | 10 | 12 | 1 | 1 | 1 | 0 |
| 12 | 1.2 | steepest_add | 10 | 12 | 1 | 13 | 17 | 0 |
| 12 | 1.2 | steepest_replace | 10 | 12 | 1 | 1 | 1 | 0 |
| 16 | 1.6 | raw_add | 10 | 16 | 1 | 46 | 55 | 0 |
| 16 | 1.6 | raw_replace | 10 | 16 | 1 | 1 | 1 | 0 |
| 16 | 1.6 | steepest_add | 10 | 16 | 1 | 14 | 17 | 0 |
| 16 | 1.6 | steepest_replace | 10 | 16 | 1 | 1 | 1 | 0 |
| 16 | 3.2 | raw_add | 5 | 16 | 1 | 23 | 28 | 0 |
| 16 | 3.2 | raw_replace | 5 | 16 | 1 | 1 | 1 | 0 |
| 16 | 3.2 | steepest_add | 5 | 16 | 1 | 8 | 11 | 0 |
| 16 | 3.2 | steepest_replace | 5 | 16 | 1 | 1 | 1 | 0 |
| 1 | 0.1 | raw_add | 10 | 1 | 1 | 57 | 72 | 0 |
| 1 | 0.1 | raw_replace | 10 | 1 | 1 | 1 | 1 | 0 |
| 1 | 0.1 | steepest_add | 10 | 1 | 1 | 11 | 12 | 0 |
| 1 | 0.1 | steepest_replace | 10 | 1 | 1 | 1 | 1 | 0 |
| 1 | 0.2 | raw_add | 5 | 1 | 1 | 29 | 36 | 0 |
| 1 | 0.2 | raw_replace | 5 | 1 | 1 | 1 | 1 | 0 |
| 1 | 0.2 | steepest_add | 5 | 1 | 1 | 6 | 6 | 0 |
| 1 | 0.2 | steepest_replace | 5 | 1 | 1 | 1 | 1 | 0 |
| 2 | 0.2 | raw_add | 10 | 2 | 1 | 49 | 59 | 0 |
| 2 | 0.2 | raw_replace | 10 | 2 | 1 | 1 | 1 | 0 |
| 2 | 0.2 | steepest_add | 10 | 2 | 1 | 11 | 12 | 0 |
| 2 | 0.2 | steepest_replace | 10 | 2 | 1 | 1 | 1 | 0 |
| 2 | 0.4 | raw_add | 5 | 2 | 1 | 25 | 30 | 0 |
| 2 | 0.4 | raw_replace | 5 | 2 | 1 | 1 | 1 | 0 |
| 2 | 0.4 | steepest_add | 5 | 2 | 1 | 6 | 8 | 0 |
| 2 | 0.4 | steepest_replace | 5 | 2 | 1 | 1 | 1 | 0 |
| 2 | 0.8 | raw_add | 2.5 | 2 | 1 | 13 | 16 | 0 |
| 2 | 0.8 | raw_replace | 2.5 | 2 | 1 | 1 | 1 | 0 |
| 2 | 0.8 | steepest_add | 2.5 | 2 | 1 | 3 | 5 | 0 |
| 2 | 0.8 | steepest_replace | 2.5 | 2 | 1 | 1 | 1 | 0 |
| 4 | 0.2 | raw_add | 20 | 4 | 1 | 85 | 111 | 0 |
| 4 | 0.2 | raw_replace | 20 | 4 | 1 | 1 | 1 | 0 |
| 4 | 0.2 | steepest_add | 20 | 4 | 1 | 22 | 25 | 0 |
| 4 | 0.2 | steepest_replace | 20 | 4 | 1 | 1 | 1 | 0 |
| 4 | 0.4 | raw_add | 10 | 4 | 1 | 43 | 56 | 0 |
| 4 | 0.4 | raw_replace | 10 | 4 | 1 | 1 | 1 | 0 |
| 4 | 0.4 | steepest_add | 10 | 4 | 1 | 11 | 14 | 0 |
| 4 | 0.4 | steepest_replace | 10 | 4 | 1 | 1 | 1 | 0 |
| 4 | 0.8 | raw_add | 5 | 4 | 1 | 22 | 28 | 0 |
| 4 | 0.8 | raw_replace | 5 | 4 | 1 | 1 | 1 | 0 |
| 4 | 0.8 | steepest_add | 5 | 4 | 1 | 6 | 9 | 0 |
| 4 | 0.8 | steepest_replace | 5 | 4 | 1 | 1 | 1 | 0 |
| 4 | 1.2 | raw_add | 3.333 | 4 | 1 | 15 | 19 | 0 |
| 4 | 1.2 | raw_replace | 3.333 | 4 | 1 | 1 | 1 | 0 |
| 4 | 1.2 | steepest_add | 3.333 | 4 | 1 | 5 | 7 | 0 |
| 4 | 1.2 | steepest_replace | 3.333 | 4 | 1 | 1 | 1 | 0 |
| 4 | 1.6 | raw_add | 2.5 | 4 | 1 | 12 | 15 | 0 |
| 4 | 1.6 | raw_replace | 2.5 | 4 | 1 | 1 | 1 | 0 |
| 4 | 1.6 | steepest_add | 2.5 | 4 | 1 | 4 | 5 | 0 |
| 4 | 1.6 | steepest_replace | 2.5 | 4 | 1 | 1 | 1 | 0 |
| 8 | 0.2 | raw_add | 40 | 8 | 1 | 164 | 189 | 0 |
| 8 | 0.2 | raw_replace | 40 | 8 | 1 | 1 | 1 | 0 |
| 8 | 0.2 | steepest_add | 40 | 8 | 1 | 46 | 51 | 0 |
| 8 | 0.2 | steepest_replace | 40 | 8 | 1 | 1 | 1 | 0 |
| 8 | 0.3 | raw_add | 26.67 | 8 | 1 | 110 | 126 | 0 |
| 8 | 0.3 | raw_replace | 26.67 | 8 | 1 | 1 | 1 | 0 |
| 8 | 0.3 | steepest_add | 26.67 | 8 | 1 | 31 | 34 | 0 |
| 8 | 0.3 | steepest_replace | 26.67 | 8 | 1 | 1 | 1 | 0 |
| 8 | 0.4 | raw_add | 20 | 8 | 1 | 83 | 95 | 0 |
| 8 | 0.4 | raw_replace | 20 | 8 | 1 | 1 | 1 | 0 |
| 8 | 0.4 | steepest_add | 20 | 8 | 1 | 23 | 26 | 0 |
| 8 | 0.4 | steepest_replace | 20 | 8 | 1 | 1 | 1 | 0 |
| 8 | 0.8 | raw_add | 10 | 8 | 1 | 42 | 48 | 0 |
| 8 | 0.8 | raw_replace | 10 | 8 | 1 | 1 | 1 | 0 |
| 8 | 0.8 | steepest_add | 10 | 8 | 1 | 12 | 15 | 0 |
| 8 | 0.8 | steepest_replace | 10 | 8 | 1 | 1 | 1 | 0 |
| 8 | 1.6 | raw_add | 5 | 8 | 1 | 21 | 24 | 0 |
| 8 | 1.6 | raw_replace | 5 | 8 | 1 | 1 | 1 | 0 |
| 8 | 1.6 | steepest_add | 5 | 8 | 1 | 7 | 8 | 0 |
| 8 | 1.6 | steepest_replace | 5 | 8 | 1 | 1 | 1 | 0 |
| 8 | 2.4 | raw_add | 3.333 | 8 | 1 | 15 | 17 | 0 |
| 8 | 2.4 | raw_replace | 3.333 | 8 | 1 | 1 | 1 | 0 |
| 8 | 2.4 | steepest_add | 3.333 | 8 | 1 | 5 | 6 | 0 |
| 8 | 2.4 | steepest_replace | 3.333 | 8 | 1 | 1 | 1 | 0 |

## Per-Setting Winners

| epsilon | alpha | criterion | winning_method | winning_value |
| --- | --- | --- | --- | --- |
| 12 | 0.6 | largest_final_loss | steepest_add | 11.01 |
| 12 | 0.6 | fastest_to_95pct_loss | raw_replace | 4 |
| 12 | 0.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 12 | 0.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 12 | 0.6 | lowest_high_frequency_ratio | raw_replace | 7.688e-10 |
| 12 | 0.6 | lowest_first_derivative_l2 | raw_replace | 0.296 |
| 12 | 0.6 | lowest_total_variation | raw_replace | 3.651 |
| 12 | 1.2 | largest_final_loss | steepest_add | 11.15 |
| 12 | 1.2 | fastest_to_95pct_loss | raw_replace | 4 |
| 12 | 1.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 12 | 1.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 12 | 1.2 | lowest_high_frequency_ratio | raw_replace | 7.688e-10 |
| 12 | 1.2 | lowest_first_derivative_l2 | raw_replace | 0.296 |
| 12 | 1.2 | lowest_total_variation | raw_replace | 3.651 |
| 16 | 1.6 | largest_final_loss | steepest_add | 15.73 |
| 16 | 1.6 | fastest_to_95pct_loss | raw_replace | 5 |
| 16 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 16 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 16 | 1.6 | lowest_high_frequency_ratio | raw_replace | 6.34e-10 |
| 16 | 1.6 | lowest_first_derivative_l2 | raw_replace | 0.3749 |
| 16 | 1.6 | lowest_total_variation | raw_replace | 4.45 |
| 16 | 3.2 | largest_final_loss | steepest_add | 15.76 |
| 16 | 3.2 | fastest_to_95pct_loss | raw_replace | 5 |
| 16 | 3.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 16 | 3.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 16 | 3.2 | lowest_high_frequency_ratio | raw_replace | 6.34e-10 |
| 16 | 3.2 | lowest_first_derivative_l2 | raw_replace | 0.3749 |
| 16 | 3.2 | lowest_total_variation | raw_replace | 4.45 |
| 1 | 0.1 | largest_final_loss | steepest_add | 0.7516 |
| 1 | 0.1 | fastest_to_95pct_loss | raw_replace | 3 |
| 1 | 0.1 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | lowest_high_frequency_ratio | raw_replace | 2.617e-08 |
| 1 | 0.1 | lowest_first_derivative_l2 | raw_replace | 0.07976 |
| 1 | 0.1 | lowest_total_variation | raw_replace | 1.335 |
| 1 | 0.2 | largest_final_loss | steepest_add | 0.7518 |
| 1 | 0.2 | fastest_to_95pct_loss | raw_replace | 3 |
| 1 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | lowest_high_frequency_ratio | raw_replace | 2.617e-08 |
| 1 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.07976 |
| 1 | 0.2 | lowest_total_variation | raw_replace | 1.335 |
| 2 | 0.2 | largest_final_loss | steepest_add | 1.452 |
| 2 | 0.2 | fastest_to_95pct_loss | steepest_add | 34 |
| 2 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.2 | lowest_high_frequency_ratio | raw_replace | 8.75e-09 |
| 2 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.1346 |
| 2 | 0.2 | lowest_total_variation | raw_replace | 2.025 |
| 2 | 0.4 | largest_final_loss | steepest_add | 1.455 |
| 2 | 0.4 | fastest_to_95pct_loss | steepest_add | 19 |
| 2 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.4 | lowest_high_frequency_ratio | raw_replace | 8.75e-09 |
| 2 | 0.4 | lowest_first_derivative_l2 | raw_replace | 0.1346 |
| 2 | 0.4 | lowest_total_variation | raw_replace | 2.025 |
| 2 | 0.8 | largest_final_loss | raw_add | 1.455 |
| 2 | 0.8 | fastest_to_95pct_loss | steepest_add | 12 |
| 2 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.8 | lowest_high_frequency_ratio | raw_replace | 8.75e-09 |
| 2 | 0.8 | lowest_first_derivative_l2 | raw_replace | 0.1346 |
| 2 | 0.8 | lowest_total_variation | raw_replace | 2.025 |
| 4 | 0.2 | largest_final_loss | raw_replace | 3.062 |
| 4 | 0.2 | fastest_to_95pct_loss | raw_replace | 10 |
| 4 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.2 | lowest_high_frequency_ratio | raw_replace | 3.035e-09 |
| 4 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.161 |
| 4 | 0.2 | lowest_total_variation | raw_replace | 2.002 |
| 4 | 0.4 | largest_final_loss | steepest_add | 3.065 |
| 4 | 0.4 | fastest_to_95pct_loss | raw_replace | 10 |
| 4 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | lowest_high_frequency_ratio | raw_replace | 3.035e-09 |
| 4 | 0.4 | lowest_first_derivative_l2 | raw_replace | 0.161 |
| 4 | 0.4 | lowest_total_variation | raw_replace | 2.002 |
| 4 | 0.8 | largest_final_loss | raw_replace | 3.062 |
| 4 | 0.8 | fastest_to_95pct_loss | raw_replace | 10 |
| 4 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.8 | lowest_high_frequency_ratio | raw_replace | 3.035e-09 |
| 4 | 0.8 | lowest_first_derivative_l2 | raw_replace | 0.161 |
| 4 | 0.8 | lowest_total_variation | raw_replace | 2.002 |
| 4 | 1.2 | largest_final_loss | steepest_add | 3.083 |
| 4 | 1.2 | fastest_to_95pct_loss | raw_replace | 12 |
| 4 | 1.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 1.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 1.2 | lowest_high_frequency_ratio | raw_replace | 3.035e-09 |
| 4 | 1.2 | lowest_first_derivative_l2 | raw_replace | 0.161 |
| 4 | 1.2 | lowest_total_variation | raw_replace | 2.002 |
| 4 | 1.6 | largest_final_loss | steepest_add | 3.161 |
| 4 | 1.6 | fastest_to_95pct_loss | raw_replace | 16 |
| 4 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 1.6 | lowest_high_frequency_ratio | raw_replace | 3.035e-09 |
| 4 | 1.6 | lowest_first_derivative_l2 | raw_replace | 0.161 |
| 4 | 1.6 | lowest_total_variation | raw_replace | 2.002 |
| 8 | 0.2 | largest_final_loss | raw_replace | 6.808 |
| 8 | 0.2 | fastest_to_95pct_loss | raw_replace | 5 |
| 8 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.2 | lowest_high_frequency_ratio | raw_replace | 7.744e-10 |
| 8 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.2179 |
| 8 | 0.2 | lowest_total_variation | raw_replace | 2.695 |
| 8 | 0.3 | largest_final_loss | steepest_add | 6.808 |
| 8 | 0.3 | fastest_to_95pct_loss | raw_replace | 5 |
| 8 | 0.3 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.3 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.3 | lowest_high_frequency_ratio | raw_replace | 7.744e-10 |
| 8 | 0.3 | lowest_first_derivative_l2 | raw_replace | 0.2179 |
| 8 | 0.3 | lowest_total_variation | raw_replace | 2.695 |
| 8 | 0.4 | largest_final_loss | steepest_add | 6.957 |
| 8 | 0.4 | fastest_to_95pct_loss | raw_replace | 6 |
| 8 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.4 | lowest_high_frequency_ratio | raw_replace | 7.744e-10 |
| 8 | 0.4 | lowest_first_derivative_l2 | raw_replace | 0.2179 |
| 8 | 0.4 | lowest_total_variation | raw_replace | 2.695 |
| 8 | 0.8 | largest_final_loss | steepest_add | 6.962 |
| 8 | 0.8 | fastest_to_95pct_loss | raw_replace | 6 |
| 8 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.8 | lowest_high_frequency_ratio | raw_replace | 7.744e-10 |
| 8 | 0.8 | lowest_first_derivative_l2 | raw_replace | 0.2179 |
| 8 | 0.8 | lowest_total_variation | raw_replace | 2.695 |
| 8 | 1.6 | largest_final_loss | steepest_add | 7.033 |
| 8 | 1.6 | fastest_to_95pct_loss | raw_replace | 8 |
| 8 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 1.6 | lowest_high_frequency_ratio | raw_replace | 7.744e-10 |
| 8 | 1.6 | lowest_first_derivative_l2 | raw_replace | 0.2179 |
| 8 | 1.6 | lowest_total_variation | raw_replace | 2.695 |
| 8 | 2.4 | largest_final_loss | steepest_add | 7.137 |
| 8 | 2.4 | fastest_to_95pct_loss | raw_replace | 28 |
| 8 | 2.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 2.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 2.4 | lowest_high_frequency_ratio | raw_replace | 7.744e-10 |
| 8 | 2.4 | lowest_first_derivative_l2 | raw_replace | 0.2179 |
| 8 | 2.4 | lowest_total_variation | raw_replace | 2.695 |

## Evidence And Interpretation

Observed from the generated CSV tables above: final-loss, boundary-arrival speed, and smoothness winners are computed within each fixed `(epsilon, alpha, p, q)` setting.

Inference from these summaries should be made within the fixed `p=2,q=2` scope unless additional P/Q roots are added and analyzed.

Boundary arrival uses `boundary_ratio = ||delta||_p / epsilon`. A 99% threshold is used as the main practical boundary test to avoid numerical roundoff around exactly `1.0`.

The smoothness metrics are numerical proxies for whether the final perturbation is high-frequency or sharp. They should be read together with the saved delta trajectory/heatmap figures under each setting root.

## 2026-05-20 Interpretation Addendum: 300-Step P2Q2 Final Loss vs Speed

Observed from `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/core4_alpha_epsilon_method_summary.csv`:

- For `epsilon=2, alpha=0.4`, `steepest_add` has the largest final mean loss (`1.4552`), followed by `raw_add` (`1.4420`), while `raw_replace`/`steepest_replace` are lower (`1.3651`). Mean 99% boundary-hit steps are `raw_add=25`, `steepest_add=6`, and replacement methods `1`.
- For `epsilon=2, alpha=0.8`, `raw_add` has the largest final mean loss (`1.4551`), followed by `steepest_add` (`1.4440`), while `raw_replace`/`steepest_replace` are lower (`1.3651`). Mean 99% boundary-hit steps are `raw_add=13`, `steepest_add=3`, and replacement methods `1`.
- For `epsilon=1, alpha=0.2`, `steepest_add` and `raw_add` finish slightly above replacement methods: `0.7518`/`0.7515` versus `0.7437`.
- For larger epsilon examples, replacement/GPI remains extremely fast and has lower final-loss sample std, but the largest 300-step final mean loss is often `steepest_add`: at `epsilon=8, alpha=0.3`, `steepest_add=6.8080` and replacement/GPI `6.8078`; at `epsilon=12, alpha=1.2`, `steepest_add=11.1459` and replacement/GPI `10.9495`; at `epsilon=16, alpha=1.6`, `steepest_add=15.7336` and replacement/GPI `15.6521`.
- In these `p=2,q=2` runs, `raw_replace` and `steepest_replace` have identical final-loss and std values to the reported precision.
- Across the 20 alpha/epsilon settings, strict largest-final-mean winners are `steepest_add` for 16 settings and `raw_add` for 1 setting; replacement/GPI tie for the largest final mean in 3 settings when ties are counted.

Inference from these observations:

- The 300-step result weakens the claim that GPI/replacement always gives the largest final loss. It supports a more precise claim: GPI/replacement reaches the boundary immediately, reaches useful loss levels very quickly, and tends to have lower across-sample variance, while sufficiently long `raw_add` or especially `steepest_add` can overtake or slightly exceed it in final mean loss for some epsilon/alpha settings.
- A good paper phrasing is therefore a Pareto-style conclusion rather than a single winner: replacement/GPI is the fastest and most stable path to a strong boundary perturbation; `steepest_add` can be better when the metric is only the final 300-step mean loss.

## 2026-05-20 Addendum: Why Boundary-Ratio Std Differs So Much

Observed from `forensics/loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520/*/per_step_metrics.csv` and `forensics/loss3_alpha_epsilon_core4_analysis_p2q2_300steps_20260520/core4_alpha_epsilon_method_summary.csv`:

- `boundary_ratio_std` is the sample standard deviation over the batch of `||delta||_p / epsilon` at a fixed optimization step. It measures how synchronized the 100 samples are in radial progress toward the epsilon boundary.
- For `epsilon=8, alpha=0.3`, max/mean `boundary_ratio_std` values are `raw_add=0.288/0.0588`, `steepest_add=0.0438/0.00211`, and `raw_replace=steepest_replace=~3e-08/~3e-08`.
- In the same setting, per-sample 99% boundary steps are much more spread for `raw_add`: mean/median/max `73.47/80/126`; for `steepest_add`: `29.44/29/34`; for replacement methods: `1/1/1`.
- Step-level evidence in `epsilon=8, alpha=0.3`: at step 20, `raw_add` has `boundary_ratio_mean=0.310` and `boundary_ratio_std=0.194`, while `steepest_add` has `boundary_ratio_mean=0.700` and `boundary_ratio_std=0.026`. At step 1, `steepest_add` has `delta_step_pnorm_mean=0.3` and `delta_step_pnorm_std=1.5e-08`, matching its normalized step rule.
- For `epsilon=16, alpha=1.6`, max/mean `boundary_ratio_std` values are `raw_add=0.295/0.0238`, `steepest_add=0.0843/0.00214`, and replacement methods `~3e-08/~3e-08`.
- For `epsilon=2, alpha=0.4`, max/mean `boundary_ratio_std` values are `raw_add=0.276/0.0137`, `steepest_add=0.0427/0.000478`, and replacement methods `~3e-08/~3e-08`.

Inference:

- Raw PGD (`raw_add`) uses the raw gradient scale, so the effective p-norm step length differs strongly across samples. Samples with larger gradient norms or better radial alignment reach the boundary much earlier than samples with smaller or more tangential gradients; this creates large `boundary_ratio_std`.
- LP steepest PGD (`steepest_add`) normalizes the steepest direction in the p-ball geometry before adding the step. Before projection dominates, the intended p-norm step is essentially `alpha` for every sample, so radial progress is close to the deterministic line `k * alpha / epsilon`; the remaining spread mostly comes from direction changes, cancellation with the current delta, and projection near the boundary.
- Replacement/GPI methods enforce the p-norm boundary immediately after the first update, so `boundary_ratio` is almost exactly 1 for every sample after step 1. Their `boundary_ratio_std` therefore collapses to floating-point noise even though their directions and losses can still change substantially along the boundary.

Practical reading:

- The tiny `boundary_ratio_std` for `steepest_add` does not mean its perturbation shapes or losses are identical across samples. It means only that the radial budget usage is synchronized.
- To study how much the method still changes after reaching the boundary, use angular change (`delta_prev_angle_degrees`), loss gain after boundary, and final delta-shape/spectral metrics rather than `boundary_ratio_std` alone.

## 2026-05-20 Addendum: Fast Angular Motion and Why Replacement/GPI Can Work Here

Observed from the generated angle dynamics figures under `forensics/loss3_alpha_epsilon_core4_visuals_p2q2_300steps_20260520/figures/dynamics_triptychs_angle_available/`:

- Replacement/GPI-style methods show much larger per-step `angle(delta_k, delta_{k-1})` than raw PGD and LP-steepest additive PGD, especially after they are already on the epsilon boundary.
- Raw PGD has the slowest angular movement overall. LP-steepest additive PGD rotates faster than raw PGD, but its angular change still decays once `||delta||_p` is near epsilon.
- This supports the interpretation that replacement/GPI is not only faster because it reaches the boundary immediately; it also moves more aggressively along the boundary after arrival.

Inference:

- There is no contradiction with the fact that the loss is not a quadratic form. The method should not be described as having the classical global guarantee of power iteration for a fixed quadratic/eigenvalue problem.
- In this code path, replacement/GPI is better understood as a repeated local linearized boundary maximization step: at each iteration it takes the current gradient or steepest direction and replaces the perturbation by a full-budget boundary perturbation in that direction. For `p=2`, this is closely related to normalized gradient ascent on the sphere/ball boundary.
- Additive PGD has angular inertia: after `delta` has large norm, the update `delta <- Proj(delta + alpha * direction)` can only rotate the existing vector gradually, especially when `alpha << epsilon`. Replacement has almost no such inertia because it can reset the full direction at every step.
- A plausible mechanism is therefore: replacement/GPI quickly identifies and tracks a high-gain direction of the local neural-operator loss landscape. The objective is nonquadratic, but along the sampled trajectory it may behave locally like a dominant-mode or strongly aligned gradient-direction map, so the power-iteration-like replacement step is empirically effective.

Practical conclusion:

- The right claim is empirical and geometric, not a classical theorem: replacement/GPI is a very aggressive boundary-direction optimizer. Its advantage appears to come from both immediate radial budget use and much faster angular motion along the boundary.
- The remaining check is whether this aggressive angular motion creates high-frequency or spike-like perturbations. That should be judged using the existing delta-shape grids, high-frequency energy, derivative/TV metrics, and trajectory GIF panels rather than angle alone.

