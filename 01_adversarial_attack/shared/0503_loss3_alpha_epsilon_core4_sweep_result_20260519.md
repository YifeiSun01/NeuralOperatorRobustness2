# Loss3 Alpha/Epsilon Core-Four Sweep Result - 2026-05-19

Status: generated from completed local sweep outputs.

## Source Data

Observed from local files:

- Sweep root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_alpha_epsilon_core4_sweep_20260519`
- Analysis root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_alpha_epsilon_core4_analysis_20260519`
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
| raw_add | 20 | 5.245 | 56 | 35.5 | 49.5 | 0.9896 | 2.975e-06 |
| raw_replace | 20 | 5.723 | 5.5 | 1 | 1 | 1 | 5.47e-09 |
| steepest_add | 20 | 5.644 | 28.5 | 11 | 12 | 1 | 4.12e-06 |
| steepest_replace | 20 | 5.723 | 5.5 | 1 | 1 | 1 | 5.47e-09 |

## Boundary Arrival Preview

| epsilon | alpha | method | nominal_l2_steepest_boundary_steps | final_delta_pnorm_mean | final_boundary_ratio_mean | step_to_boundary_ratio_mean_0.99 | sample_step_to_0.99_boundary_max | sample_step_to_0.99_boundary_not_reached_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 0.6 | raw_add | 20 | 12 | 0.9998 | 86 | 98 | 1 |
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
| 4 | 0.2 | raw_add | 20 | 3.996 | 0.999 | 85 | 98 | 1 |
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
| 8 | 0.2 | raw_add | 40 | 6.594 | 0.8242 | nan | 99 | 54 |
| 8 | 0.2 | raw_replace | 40 | 8 | 1 | 1 | 1 | 0 |
| 8 | 0.2 | steepest_add | 40 | 8 | 1 | 46 | 51 | 0 |
| 8 | 0.2 | steepest_replace | 40 | 8 | 1 | 1 | 1 | 0 |
| 8 | 0.3 | raw_add | 26.67 | 7.751 | 0.9689 | nan | 99 | 32 |
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
| 12 | 0.6 | largest_final_loss | raw_replace | 10.89 |
| 12 | 0.6 | fastest_to_95pct_loss | raw_replace | 4 |
| 12 | 0.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 12 | 0.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 12 | 0.6 | lowest_high_frequency_ratio | raw_replace | 7.441e-10 |
| 12 | 0.6 | lowest_first_derivative_l2 | raw_replace | 0.2961 |
| 12 | 0.6 | lowest_total_variation | raw_replace | 3.653 |
| 12 | 1.2 | largest_final_loss | raw_replace | 10.89 |
| 12 | 1.2 | fastest_to_95pct_loss | raw_replace | 4 |
| 12 | 1.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 12 | 1.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 12 | 1.2 | lowest_high_frequency_ratio | raw_replace | 7.441e-10 |
| 12 | 1.2 | lowest_first_derivative_l2 | raw_replace | 0.2961 |
| 12 | 1.2 | lowest_total_variation | raw_replace | 3.653 |
| 16 | 1.6 | largest_final_loss | steepest_add | 15.51 |
| 16 | 1.6 | fastest_to_95pct_loss | raw_replace | 4 |
| 16 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 16 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 16 | 1.6 | lowest_high_frequency_ratio | raw_replace | 5.855e-10 |
| 16 | 1.6 | lowest_first_derivative_l2 | raw_replace | 0.3722 |
| 16 | 1.6 | lowest_total_variation | raw_replace | 4.45 |
| 16 | 3.2 | largest_final_loss | steepest_add | 15.55 |
| 16 | 3.2 | fastest_to_95pct_loss | raw_replace | 4 |
| 16 | 3.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 16 | 3.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 16 | 3.2 | lowest_high_frequency_ratio | raw_replace | 5.855e-10 |
| 16 | 3.2 | lowest_first_derivative_l2 | raw_replace | 0.3722 |
| 16 | 3.2 | lowest_total_variation | raw_replace | 4.45 |
| 1 | 0.1 | largest_final_loss | steepest_add | 0.7505 |
| 1 | 0.1 | fastest_to_95pct_loss | raw_replace | 3 |
| 1 | 0.1 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | lowest_high_frequency_ratio | raw_replace | 2.547e-08 |
| 1 | 0.1 | lowest_first_derivative_l2 | raw_replace | 0.08014 |
| 1 | 0.1 | lowest_total_variation | raw_replace | 1.341 |
| 1 | 0.2 | largest_final_loss | steepest_add | 0.7514 |
| 1 | 0.2 | fastest_to_95pct_loss | raw_replace | 3 |
| 1 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | lowest_high_frequency_ratio | raw_replace | 2.547e-08 |
| 1 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.08014 |
| 1 | 0.2 | lowest_total_variation | raw_replace | 1.341 |
| 2 | 0.2 | largest_final_loss | steepest_add | 1.431 |
| 2 | 0.2 | fastest_to_95pct_loss | raw_replace | 17 |
| 2 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.2 | lowest_high_frequency_ratio | raw_replace | 9.501e-09 |
| 2 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.1349 |
| 2 | 0.2 | lowest_total_variation | raw_replace | 2.034 |
| 2 | 0.4 | largest_final_loss | steepest_add | 1.441 |
| 2 | 0.4 | fastest_to_95pct_loss | steepest_add | 17 |
| 2 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.4 | lowest_high_frequency_ratio | raw_replace | 9.501e-09 |
| 2 | 0.4 | lowest_first_derivative_l2 | raw_replace | 0.1349 |
| 2 | 0.4 | lowest_total_variation | raw_replace | 2.034 |
| 2 | 0.8 | largest_final_loss | raw_add | 1.441 |
| 2 | 0.8 | fastest_to_95pct_loss | steepest_add | 10 |
| 2 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.8 | lowest_high_frequency_ratio | raw_replace | 9.501e-09 |
| 2 | 0.8 | lowest_first_derivative_l2 | raw_replace | 0.1349 |
| 2 | 0.8 | lowest_total_variation | raw_replace | 2.034 |
| 4 | 0.2 | largest_final_loss | raw_replace | 3.047 |
| 4 | 0.2 | fastest_to_95pct_loss | raw_replace | 10 |
| 4 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.2 | lowest_high_frequency_ratio | raw_replace | 4.488e-09 |
| 4 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.1602 |
| 4 | 0.2 | lowest_total_variation | raw_replace | 1.985 |
| 4 | 0.4 | largest_final_loss | raw_replace | 3.047 |
| 4 | 0.4 | fastest_to_95pct_loss | raw_replace | 10 |
| 4 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | lowest_high_frequency_ratio | raw_replace | 4.488e-09 |
| 4 | 0.4 | lowest_first_derivative_l2 | raw_replace | 0.1602 |
| 4 | 0.4 | lowest_total_variation | raw_replace | 1.985 |
| 4 | 0.8 | largest_final_loss | raw_replace | 3.047 |
| 4 | 0.8 | fastest_to_95pct_loss | raw_replace | 10 |
| 4 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.8 | lowest_high_frequency_ratio | raw_replace | 4.488e-09 |
| 4 | 0.8 | lowest_first_derivative_l2 | raw_replace | 0.1602 |
| 4 | 0.8 | lowest_total_variation | raw_replace | 1.985 |
| 4 | 1.2 | largest_final_loss | raw_replace | 3.047 |
| 4 | 1.2 | fastest_to_95pct_loss | raw_replace | 10 |
| 4 | 1.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 1.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 1.2 | lowest_high_frequency_ratio | raw_replace | 4.488e-09 |
| 4 | 1.2 | lowest_first_derivative_l2 | raw_replace | 0.1602 |
| 4 | 1.2 | lowest_total_variation | raw_replace | 1.985 |
| 4 | 1.6 | largest_final_loss | steepest_add | 3.125 |
| 4 | 1.6 | fastest_to_95pct_loss | raw_replace | 12 |
| 4 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 1.6 | lowest_high_frequency_ratio | raw_replace | 4.488e-09 |
| 4 | 1.6 | lowest_first_derivative_l2 | raw_replace | 0.1602 |
| 4 | 1.6 | lowest_total_variation | raw_replace | 1.985 |
| 8 | 0.2 | largest_final_loss | raw_replace | 6.805 |
| 8 | 0.2 | fastest_to_95pct_loss | raw_replace | 5 |
| 8 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.2 | lowest_high_frequency_ratio | raw_replace | 8.099e-10 |
| 8 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.217 |
| 8 | 0.2 | lowest_total_variation | raw_replace | 2.685 |
| 8 | 0.3 | largest_final_loss | raw_replace | 6.805 |
| 8 | 0.3 | fastest_to_95pct_loss | raw_replace | 5 |
| 8 | 0.3 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.3 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.3 | lowest_high_frequency_ratio | raw_replace | 8.099e-10 |
| 8 | 0.3 | lowest_first_derivative_l2 | raw_replace | 0.217 |
| 8 | 0.3 | lowest_total_variation | raw_replace | 2.685 |
| 8 | 0.4 | largest_final_loss | raw_replace | 6.805 |
| 8 | 0.4 | fastest_to_95pct_loss | raw_replace | 5 |
| 8 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.4 | lowest_high_frequency_ratio | raw_replace | 8.099e-10 |
| 8 | 0.4 | lowest_first_derivative_l2 | raw_replace | 0.217 |
| 8 | 0.4 | lowest_total_variation | raw_replace | 2.685 |
| 8 | 0.8 | largest_final_loss | raw_replace | 6.805 |
| 8 | 0.8 | fastest_to_95pct_loss | raw_replace | 5 |
| 8 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.8 | lowest_high_frequency_ratio | raw_replace | 8.099e-10 |
| 8 | 0.8 | lowest_first_derivative_l2 | raw_replace | 0.217 |
| 8 | 0.8 | lowest_total_variation | raw_replace | 2.685 |
| 8 | 1.6 | largest_final_loss | steepest_add | 6.933 |
| 8 | 1.6 | fastest_to_95pct_loss | raw_replace | 6 |
| 8 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 1.6 | lowest_high_frequency_ratio | raw_replace | 8.099e-10 |
| 8 | 1.6 | lowest_first_derivative_l2 | raw_replace | 0.217 |
| 8 | 1.6 | lowest_total_variation | raw_replace | 2.685 |
| 8 | 2.4 | largest_final_loss | steepest_add | 7.02 |
| 8 | 2.4 | fastest_to_95pct_loss | raw_replace | 7 |
| 8 | 2.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 2.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 2.4 | lowest_high_frequency_ratio | raw_replace | 8.099e-10 |
| 8 | 2.4 | lowest_first_derivative_l2 | raw_replace | 0.217 |
| 8 | 2.4 | lowest_total_variation | raw_replace | 2.685 |

## Evidence And Interpretation

Observed from the generated CSV tables above: final-loss, boundary-arrival speed, and smoothness winners are computed within each fixed `(epsilon, alpha, p, q)` setting.

Inference from these summaries should be made within the fixed `p=2,q=2` scope unless additional P/Q roots are added and analyzed.

Boundary arrival uses `boundary_ratio = ||delta||_p / epsilon`. A 99% threshold is used as the main practical boundary test to avoid numerical roundoff around exactly `1.0`.

The smoothness metrics are numerical proxies for whether the final perturbation is high-frequency or sharp. They should be read together with the saved delta trajectory/heatmap figures under each setting root.
