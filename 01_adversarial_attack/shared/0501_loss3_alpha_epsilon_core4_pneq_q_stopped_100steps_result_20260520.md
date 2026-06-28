# Loss3 Alpha/Epsilon Core-Four Sweep Result - 2026-05-19

Status: generated from completed local sweep outputs.

## Source Data

Observed from local files:

- Sweep root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_alpha_epsilon_core4_sweep_pneq_q_100steps_20260520`
- Analysis root: `/workspace/NeuralOperatorRobustness2/forensics/loss3_alpha_epsilon_core4_analysis_pneq_q_stopped_100steps_20260520`
- Completed setting roots analyzed: `82`
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
| raw_add | 82 | 35.62 | 12.5 | 6 | 9 | 0.9279 | 0.007967 |
| raw_replace | 82 | 29.74 | 10 | 1 | 1 | 0.9993 | 0.00859 |
| steepest_add | 82 | 35.48 | 19 | 10 | 10 | 0.9929 | 0.1175 |
| steepest_replace | 82 | 32.57 | 1.5 | 1 | 1 | 0.9744 | 0.1191 |

## Boundary Arrival Preview

| epsilon | alpha | method | nominal_l2_steepest_boundary_steps | final_delta_pnorm_mean | final_boundary_ratio_mean | step_to_boundary_ratio_mean_0.99 | sample_step_to_0.99_boundary_max | sample_step_to_0.99_boundary_not_reached_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 0.6 | raw_add | 20 | 12 | 1 | 6 | 8 | 0 |
| 12 | 0.6 | raw_replace | 20 | 12 | 1 | 1 | 1 | 0 |
| 12 | 0.6 | steepest_add | 20 | 12 | 1 | 20 | 20 | 0 |
| 12 | 0.6 | steepest_replace | 20 | 12 | 1 | 1 | 1 | 0 |
| 12 | 0.6 | raw_add | 20 | 11.98 | 0.998 | 22 | 38 | 0 |
| 12 | 0.6 | raw_replace | 20 | 12 | 1 | 1 | 1 | 0 |
| 12 | 0.6 | steepest_add | 20 | 12 | 1 | 20 | 20 | 0 |
| 12 | 0.6 | steepest_replace | 20 | 12 | 1 | 1 | 1 | 0 |
| 12 | 0.6 | raw_add | 20 | 12 | 1 | 7 | 9 | 0 |
| 12 | 0.6 | raw_replace | 20 | 12 | 1 | 1 | 1 | 0 |
| 12 | 0.6 | steepest_add | 20 | 12 | 1 | 24 | 27 | 0 |
| 12 | 0.6 | steepest_replace | 20 | 12 | 1 | 1 | 1 | 0 |
| 12 | 0.6 | raw_add | 20 | 3.948 | 0.329 | nan | nan | 100 |
| 12 | 0.6 | raw_replace | 20 | 12 | 1 | 1 | 1 | 0 |
| 12 | 0.6 | steepest_add | 20 | 11.94 | 0.9952 | 92 | 97 | 7 |
| 12 | 0.6 | steepest_replace | 20 | 12 | 1 | 1 | 1 | 0 |
| 12 | 1.2 | raw_add | 10 | 12 | 1 | 4 | 4 | 0 |
| 12 | 1.2 | raw_replace | 10 | 12 | 1 | 1 | 1 | 0 |
| 12 | 1.2 | steepest_add | 10 | 12 | 1 | 10 | 10 | 0 |
| 12 | 1.2 | steepest_replace | 10 | 12 | 1 | 1 | 1 | 0 |
| 12 | 1.2 | raw_add | 10 | 11.96 | 0.9969 | 12 | 19 | 0 |
| 12 | 1.2 | raw_replace | 10 | 12 | 1 | 1 | 1 | 0 |
| 12 | 1.2 | steepest_add | 10 | 11.52 | 0.9602 | nan | 10 | 11 |
| 12 | 1.2 | steepest_replace | 10 | 12 | 1 | 1 | 1 | 0 |
| 12 | 1.2 | raw_add | 10 | 12 | 1 | 4 | 5 | 0 |
| 12 | 1.2 | raw_replace | 10 | 12 | 1 | 1 | 1 | 0 |
| 12 | 1.2 | steepest_add | 10 | 12 | 1 | 13 | 15 | 0 |
| 12 | 1.2 | steepest_replace | 10 | 12 | 1 | 1 | 1 | 0 |
| 12 | 1.2 | raw_add | 10 | 5.464 | 0.4554 | nan | nan | 100 |
| 12 | 1.2 | raw_replace | 10 | 12 | 1 | 1 | 1 | 0 |
| 12 | 1.2 | steepest_add | 10 | 11.99 | 0.9992 | 36 | 53 | 0 |
| 12 | 1.2 | steepest_replace | 10 | 12 | 1 | 1 | 1 | 0 |
| 16 | 1.6 | raw_add | 10 | 16 | 1 | 3 | 4 | 0 |
| 16 | 1.6 | raw_replace | 10 | 16 | 1 | 1 | 1 | 0 |
| 16 | 1.6 | steepest_add | 10 | 15.15 | 0.947 | nan | 10 | 29 |
| 16 | 1.6 | steepest_replace | 10 | 6.24 | 0.39 | 1 | 1 | 0 |
| 16 | 1.6 | raw_add | 10 | 15.98 | 0.999 | 12 | 20 | 0 |
| 16 | 1.6 | raw_replace | 10 | 16 | 1 | 1 | 1 | 0 |
| 16 | 1.6 | steepest_add | 10 | 14.95 | 0.9346 | nan | 12 | 15 |
| 16 | 1.6 | steepest_replace | 10 | 9.44 | 0.59 | 1 | 1 | 0 |
| 16 | 1.6 | raw_add | 10 | 16 | 1 | 4 | 5 | 0 |
| 16 | 1.6 | raw_replace | 10 | 16 | 1 | 1 | 1 | 0 |
| 16 | 1.6 | steepest_add | 10 | 16 | 1 | 15 | 20 | 0 |
| 16 | 1.6 | steepest_replace | 10 | 16 | 1 | 1 | 1 | 0 |
| 16 | 1.6 | raw_add | 10 | 6.259 | 0.3912 | nan | nan | 100 |
| 16 | 1.6 | raw_replace | 10 | 15.52 | 0.97 | 1 | 1 | 0 |
| 16 | 1.6 | steepest_add | 10 | 16 | 0.9998 | 40 | 56 | 0 |
| 16 | 1.6 | steepest_replace | 10 | 15.52 | 0.97 | 1 | 1 | 0 |
| 16 | 3.2 | raw_add | 5 | 16 | 1 | 2 | 3 | 0 |
| 16 | 3.2 | raw_replace | 5 | 16 | 1 | 1 | 1 | 0 |
| 16 | 3.2 | steepest_add | 5 | 15.01 | 0.938 | nan | 5 | 27 |
| 16 | 3.2 | steepest_replace | 5 | 6.24 | 0.39 | 1 | 1 | 0 |
| 16 | 3.2 | raw_add | 5 | 16 | 1 | 6 | 10 | 0 |
| 16 | 3.2 | raw_replace | 5 | 16 | 1 | 1 | 1 | 0 |
| 16 | 3.2 | steepest_add | 5 | 15.43 | 0.9643 | nan | 5 | 10 |
| 16 | 3.2 | steepest_replace | 5 | 9.44 | 0.59 | 1 | 1 | 0 |
| 16 | 3.2 | raw_add | 5 | 16 | 1 | 2 | 2 | 0 |
| 16 | 3.2 | raw_replace | 5 | 16 | 1 | 1 | 1 | 0 |
| 16 | 3.2 | steepest_add | 5 | 16 | 1 | 8 | 11 | 0 |
| 16 | 3.2 | steepest_replace | 5 | 16 | 1 | 1 | 1 | 0 |
| 16 | 3.2 | raw_add | 5 | 9.258 | 0.5786 | nan | 90 | 93 |
| 16 | 3.2 | raw_replace | 5 | 15.52 | 0.97 | 1 | 1 | 0 |
| 16 | 3.2 | steepest_add | 5 | 15.99 | 0.9995 | 14 | 17 | 0 |
| 16 | 3.2 | steepest_replace | 5 | 15.52 | 0.97 | 1 | 1 | 0 |
| 1 | 0.1 | raw_add | 10 | 1 | 1 | 4 | 6 | 0 |
| 1 | 0.1 | raw_replace | 10 | 1 | 1 | 1 | 1 | 0 |
| 1 | 0.1 | steepest_add | 10 | 1 | 1 | 10 | 10 | 0 |
| 1 | 0.1 | steepest_replace | 10 | 1 | 1 | 1 | 1 | 0 |
| 1 | 0.1 | raw_add | 10 | 1 | 1 | 12 | 19 | 0 |
| 1 | 0.1 | raw_replace | 10 | 1 | 1 | 1 | 1 | 0 |
| 1 | 0.1 | steepest_add | 10 | 1 | 1 | 10 | 10 | 0 |
| 1 | 0.1 | steepest_replace | 10 | 1 | 1 | 1 | 1 | 0 |
| 1 | 0.1 | raw_add | 10 | 1 | 1 | 6 | 9 | 0 |
| 1 | 0.1 | raw_replace | 10 | 1 | 1 | 1 | 1 | 0 |
| 1 | 0.1 | steepest_add | 10 | 1 | 1 | 11 | 12 | 0 |
| 1 | 0.1 | steepest_replace | 10 | 1 | 1 | 1 | 1 | 0 |
| 1 | 0.1 | raw_add | 10 | 0.9131 | 0.9131 | nan | 100 | 32 |
| 1 | 0.1 | raw_replace | 10 | 1 | 1 | 1 | 1 | 0 |
| 1 | 0.1 | steepest_add | 10 | 1 | 1 | 26 | 39 | 0 |
| 1 | 0.1 | steepest_replace | 10 | 1 | 1 | 1 | 1 | 0 |

## Per-Setting Winners

| epsilon | alpha | criterion | winning_method | winning_value |
| --- | --- | --- | --- | --- |
| 12 | 0.6 | largest_final_loss | steepest_add | 1.269 |
| 12 | 0.6 | fastest_to_95pct_loss | steepest_replace | 2 |
| 12 | 0.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 12 | 0.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 12 | 0.6 | lowest_high_frequency_ratio | raw_replace | 4.562e-08 |
| 12 | 0.6 | lowest_first_derivative_l2 | raw_replace | 0.05076 |
| 12 | 0.6 | lowest_total_variation | raw_replace | 0.8826 |
| 12 | 0.6 | largest_final_loss | steepest_add | 0.3791 |
| 12 | 0.6 | fastest_to_95pct_loss | steepest_add | 20 |
| 12 | 0.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 12 | 0.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 12 | 0.6 | lowest_high_frequency_ratio | raw_add | 0.01038 |
| 12 | 0.6 | lowest_first_derivative_l2 | raw_add | 0.1669 |
| 12 | 0.6 | lowest_total_variation | raw_replace | 1.018 |
| 12 | 0.6 | largest_final_loss | raw_add | 218.7 |
| 12 | 0.6 | fastest_to_95pct_loss | raw_add | 13 |
| 12 | 0.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 12 | 0.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 12 | 0.6 | lowest_high_frequency_ratio | raw_add | 5.867e-07 |
| 12 | 0.6 | lowest_first_derivative_l2 | raw_replace | 0.3468 |
| 12 | 0.6 | lowest_total_variation | raw_replace | 4.707 |
| 12 | 0.6 | largest_final_loss | steepest_add | 1.062 |
| 12 | 0.6 | fastest_to_95pct_loss | steepest_add | 77 |
| 12 | 0.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 12 | 0.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 12 | 0.6 | lowest_high_frequency_ratio | raw_add | 0.0058 |
| 12 | 0.6 | lowest_first_derivative_l2 | raw_add | 0.7858 |
| 12 | 0.6 | lowest_total_variation | raw_add | 6.652 |
| 12 | 1.2 | largest_final_loss | steepest_add | 1.265 |
| 12 | 1.2 | fastest_to_95pct_loss | steepest_replace | 2 |
| 12 | 1.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 12 | 1.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 12 | 1.2 | lowest_high_frequency_ratio | raw_replace | 4.562e-08 |
| 12 | 1.2 | lowest_first_derivative_l2 | raw_replace | 0.05076 |
| 12 | 1.2 | lowest_total_variation | raw_replace | 0.8826 |
| 12 | 1.2 | largest_final_loss | steepest_add | 0.3626 |
| 12 | 1.2 | fastest_to_95pct_loss | steepest_replace | 1 |
| 12 | 1.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 12 | 1.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 12 | 1.2 | lowest_high_frequency_ratio | raw_add | 0.008617 |
| 12 | 1.2 | lowest_first_derivative_l2 | raw_add | 0.1605 |
| 12 | 1.2 | lowest_total_variation | raw_replace | 1.018 |
| 12 | 1.2 | largest_final_loss | raw_add | 225.3 |
| 12 | 1.2 | fastest_to_95pct_loss | raw_add | 10 |
| 12 | 1.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 12 | 1.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 12 | 1.2 | lowest_high_frequency_ratio | raw_add | 6.747e-07 |
| 12 | 1.2 | lowest_first_derivative_l2 | raw_add | 0.3343 |
| 12 | 1.2 | lowest_total_variation | raw_add | 4.399 |
| 12 | 1.2 | largest_final_loss | steepest_add | 1.274 |
| 12 | 1.2 | fastest_to_95pct_loss | steepest_add | 82 |
| 12 | 1.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 12 | 1.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 12 | 1.2 | lowest_high_frequency_ratio | steepest_add | 0.005856 |
| 12 | 1.2 | lowest_first_derivative_l2 | raw_add | 1.143 |
| 12 | 1.2 | lowest_total_variation | raw_replace | 8.48 |
| 16 | 1.6 | largest_final_loss | steepest_add | 1.386 |
| 16 | 1.6 | fastest_to_95pct_loss | steepest_replace | 1 |
| 16 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 16 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 16 | 1.6 | lowest_high_frequency_ratio | raw_add | 3.5e-08 |
| 16 | 1.6 | lowest_first_derivative_l2 | raw_replace | 0.06723 |
| 16 | 1.6 | lowest_total_variation | raw_replace | 1.128 |
| 16 | 1.6 | largest_final_loss | steepest_add | 0.3692 |
| 16 | 1.6 | fastest_to_95pct_loss | steepest_replace | 1 |
| 16 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 16 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 16 | 1.6 | lowest_high_frequency_ratio | raw_add | 0.004666 |
| 16 | 1.6 | lowest_first_derivative_l2 | raw_add | 0.1742 |
| 16 | 1.6 | lowest_total_variation | raw_replace | 1.285 |
| 16 | 1.6 | largest_final_loss | raw_add | 337.8 |
| 16 | 1.6 | fastest_to_95pct_loss | raw_add | 8 |
| 16 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 16 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 16 | 1.6 | lowest_high_frequency_ratio | raw_add | 6.235e-07 |
| 16 | 1.6 | lowest_first_derivative_l2 | raw_add | 0.4029 |
| 16 | 1.6 | lowest_total_variation | raw_add | 4.899 |
| 16 | 1.6 | largest_final_loss | steepest_add | 1.696 |
| 16 | 1.6 | fastest_to_95pct_loss | steepest_add | 80 |
| 16 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 16 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 16 | 1.6 | lowest_high_frequency_ratio | steepest_add | 0.002879 |
| 16 | 1.6 | lowest_first_derivative_l2 | raw_add | 1.303 |
| 16 | 1.6 | lowest_total_variation | raw_replace | 10.37 |
| 16 | 3.2 | largest_final_loss | steepest_add | 1.312 |
| 16 | 3.2 | fastest_to_95pct_loss | steepest_replace | 1 |
| 16 | 3.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 16 | 3.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 16 | 3.2 | lowest_high_frequency_ratio | raw_add | 3.488e-08 |
| 16 | 3.2 | lowest_first_derivative_l2 | raw_replace | 0.06723 |
| 16 | 3.2 | lowest_total_variation | raw_replace | 1.128 |
| 16 | 3.2 | largest_final_loss | steepest_add | 0.3644 |
| 16 | 3.2 | fastest_to_95pct_loss | steepest_replace | 1 |
| 16 | 3.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 16 | 3.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 16 | 3.2 | lowest_high_frequency_ratio | raw_add | 0.007374 |
| 16 | 3.2 | lowest_first_derivative_l2 | raw_add | 0.2031 |
| 16 | 3.2 | lowest_total_variation | raw_replace | 1.285 |
| 16 | 3.2 | largest_final_loss | steepest_add | 342.4 |
| 16 | 3.2 | fastest_to_95pct_loss | raw_add | 12 |
| 16 | 3.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 16 | 3.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 16 | 3.2 | lowest_high_frequency_ratio | steepest_add | 5.79e-07 |
| 16 | 3.2 | lowest_first_derivative_l2 | raw_add | 0.4062 |
| 16 | 3.2 | lowest_total_variation | raw_add | 5.026 |
| 16 | 3.2 | largest_final_loss | steepest_add | 1.648 |
| 16 | 3.2 | fastest_to_95pct_loss | steepest_add | 30 |
| 16 | 3.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 16 | 3.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 16 | 3.2 | lowest_high_frequency_ratio | steepest_add | 0.001125 |
| 16 | 3.2 | lowest_first_derivative_l2 | steepest_add | 0.9299 |
| 16 | 3.2 | lowest_total_variation | steepest_add | 8.642 |
| 1 | 0.1 | largest_final_loss | steepest_add | 0.3374 |
| 1 | 0.1 | fastest_to_95pct_loss | steepest_replace | 1 |
| 1 | 0.1 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | lowest_high_frequency_ratio | raw_replace | 6.598e-08 |
| 1 | 0.1 | lowest_first_derivative_l2 | raw_replace | 0.004104 |
| 1 | 0.1 | lowest_total_variation | raw_replace | 0.08034 |
| 1 | 0.1 | largest_final_loss | steepest_add | 0.1114 |
| 1 | 0.1 | fastest_to_95pct_loss | steepest_replace | 1 |
| 1 | 0.1 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | lowest_high_frequency_ratio | raw_replace | 0.02656 |
| 1 | 0.1 | lowest_first_derivative_l2 | raw_replace | 0.02431 |
| 1 | 0.1 | lowest_total_variation | raw_replace | 0.09197 |
| 1 | 0.1 | largest_final_loss | raw_add | 10.65 |
| 1 | 0.1 | fastest_to_95pct_loss | raw_replace | 3 |
| 1 | 0.1 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | lowest_high_frequency_ratio | raw_replace | 2.408e-05 |
| 1 | 0.1 | lowest_first_derivative_l2 | raw_replace | 0.09114 |
| 1 | 0.1 | lowest_total_variation | raw_replace | 1.688 |
| 1 | 0.1 | largest_final_loss | steepest_add | 0.236 |
| 1 | 0.1 | fastest_to_95pct_loss | steepest_add | 15 |
| 1 | 0.1 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | lowest_high_frequency_ratio | steepest_add | 0.01467 |
| 1 | 0.1 | lowest_first_derivative_l2 | steepest_add | 0.2584 |
| 1 | 0.1 | lowest_total_variation | raw_replace | 1.433 |
| 1 | 0.1 | largest_final_loss | steepest_add | 405.5 |
| 1 | 0.1 | fastest_to_95pct_loss | steepest_add | 35 |
| 1 | 0.1 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.1 | lowest_high_frequency_ratio | raw_replace | 1.364e-06 |
| 1 | 0.1 | lowest_first_derivative_l2 | raw_replace | 0.3306 |
| 1 | 0.1 | lowest_total_variation | raw_replace | 4.577 |
| 1 | 0.2 | largest_final_loss | steepest_add | 0.3374 |
| 1 | 0.2 | fastest_to_95pct_loss | steepest_replace | 1 |
| 1 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | lowest_high_frequency_ratio | raw_replace | 6.598e-08 |
| 1 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.004104 |
| 1 | 0.2 | lowest_total_variation | raw_replace | 0.08034 |
| 1 | 0.2 | largest_final_loss | steepest_add | 0.1114 |
| 1 | 0.2 | fastest_to_95pct_loss | steepest_replace | 1 |
| 1 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | lowest_high_frequency_ratio | raw_replace | 0.02656 |
| 1 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.02431 |
| 1 | 0.2 | lowest_total_variation | raw_replace | 0.09197 |
| 1 | 0.2 | largest_final_loss | raw_add | 10.65 |
| 1 | 0.2 | fastest_to_95pct_loss | raw_add | 3 |
| 1 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | lowest_high_frequency_ratio | raw_add | 2.318e-05 |
| 1 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.09114 |
| 1 | 0.2 | lowest_total_variation | raw_replace | 1.688 |
| 1 | 0.2 | largest_final_loss | raw_add | 0.2375 |
| 1 | 0.2 | fastest_to_95pct_loss | steepest_add | 9 |
| 1 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | lowest_high_frequency_ratio | steepest_add | 0.009 |
| 1 | 0.2 | lowest_first_derivative_l2 | steepest_add | 0.2142 |
| 1 | 0.2 | lowest_total_variation | raw_replace | 1.433 |
| 1 | 0.2 | largest_final_loss | steepest_add | 456.8 |
| 1 | 0.2 | fastest_to_95pct_loss | steepest_add | 19 |
| 1 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 1 | 0.2 | lowest_high_frequency_ratio | raw_replace | 1.364e-06 |
| 1 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.3306 |
| 1 | 0.2 | lowest_total_variation | raw_replace | 4.577 |
| 2 | 0.2 | largest_final_loss | steepest_add | 0.393 |
| 2 | 0.2 | fastest_to_95pct_loss | steepest_replace | 1 |
| 2 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.2 | lowest_high_frequency_ratio | raw_add | 6.301e-08 |
| 2 | 0.2 | lowest_first_derivative_l2 | raw_add | 0.008235 |
| 2 | 0.2 | lowest_total_variation | raw_add | 0.1599 |
| 2 | 0.2 | largest_final_loss | steepest_add | 0.1509 |
| 2 | 0.2 | fastest_to_95pct_loss | steepest_replace | 1 |
| 2 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.2 | lowest_high_frequency_ratio | raw_replace | 0.02776 |
| 2 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.04926 |
| 2 | 0.2 | lowest_total_variation | raw_replace | 0.1862 |
| 2 | 0.2 | largest_final_loss | raw_add | 17.86 |
| 2 | 0.2 | fastest_to_95pct_loss | raw_add | 7 |
| 2 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.2 | lowest_high_frequency_ratio | raw_add | 9.746e-06 |
| 2 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.1674 |
| 2 | 0.2 | lowest_total_variation | raw_replace | 3.062 |
| 2 | 0.2 | largest_final_loss | steepest_add | 0.3811 |
| 2 | 0.2 | fastest_to_95pct_loss | steepest_add | 24 |
| 2 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.2 | lowest_high_frequency_ratio | steepest_add | 0.007826 |
| 2 | 0.2 | lowest_first_derivative_l2 | steepest_add | 0.4122 |
| 2 | 0.2 | lowest_total_variation | raw_replace | 2.215 |
| 2 | 0.4 | largest_final_loss | steepest_add | 0.3926 |
| 2 | 0.4 | fastest_to_95pct_loss | steepest_replace | 1 |
| 2 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.4 | lowest_high_frequency_ratio | raw_add | 6.302e-08 |
| 2 | 0.4 | lowest_first_derivative_l2 | raw_add | 0.008235 |
| 2 | 0.4 | lowest_total_variation | raw_add | 0.1599 |
| 2 | 0.4 | largest_final_loss | steepest_add | 0.1514 |
| 2 | 0.4 | fastest_to_95pct_loss | steepest_replace | 1 |
| 2 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.4 | lowest_high_frequency_ratio | raw_replace | 0.02776 |
| 2 | 0.4 | lowest_first_derivative_l2 | raw_replace | 0.04926 |
| 2 | 0.4 | lowest_total_variation | raw_replace | 0.1862 |
| 2 | 0.4 | largest_final_loss | raw_add | 18.04 |
| 2 | 0.4 | fastest_to_95pct_loss | raw_add | 5 |
| 2 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.4 | lowest_high_frequency_ratio | raw_add | 7.608e-06 |
| 2 | 0.4 | lowest_first_derivative_l2 | raw_replace | 0.1674 |
| 2 | 0.4 | lowest_total_variation | raw_replace | 3.062 |
| 2 | 0.4 | largest_final_loss | steepest_add | 0.3842 |
| 2 | 0.4 | fastest_to_95pct_loss | steepest_add | 13 |
| 2 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.4 | lowest_high_frequency_ratio | steepest_add | 0.002661 |
| 2 | 0.4 | lowest_first_derivative_l2 | steepest_add | 0.3205 |
| 2 | 0.4 | lowest_total_variation | raw_replace | 2.215 |
| 2 | 0.8 | largest_final_loss | steepest_add | 0.3913 |
| 2 | 0.8 | fastest_to_95pct_loss | steepest_replace | 1 |
| 2 | 0.8 | fastest_mean_boundary_99pct | raw_add | 1 |
| 2 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.8 | lowest_high_frequency_ratio | raw_add | 6.301e-08 |
| 2 | 0.8 | lowest_first_derivative_l2 | raw_add | 0.008235 |
| 2 | 0.8 | lowest_total_variation | raw_replace | 0.1599 |
| 2 | 0.8 | largest_final_loss | steepest_add | 0.1509 |
| 2 | 0.8 | fastest_to_95pct_loss | steepest_replace | 1 |
| 2 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.8 | lowest_high_frequency_ratio | raw_add | 0.02671 |
| 2 | 0.8 | lowest_first_derivative_l2 | raw_add | 0.0484 |
| 2 | 0.8 | lowest_total_variation | raw_replace | 0.1862 |
| 2 | 0.8 | largest_final_loss | raw_add | 18.04 |
| 2 | 0.8 | fastest_to_95pct_loss | raw_add | 4 |
| 2 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.8 | lowest_high_frequency_ratio | raw_add | 6.959e-06 |
| 2 | 0.8 | lowest_first_derivative_l2 | raw_replace | 0.1674 |
| 2 | 0.8 | lowest_total_variation | raw_replace | 3.062 |
| 2 | 0.8 | largest_final_loss | raw_add | 0.3849 |
| 2 | 0.8 | fastest_to_95pct_loss | steepest_add | 10 |
| 2 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 2 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 2 | 0.8 | lowest_high_frequency_ratio | raw_add | 0.002786 |
| 2 | 0.8 | lowest_first_derivative_l2 | raw_add | 0.2984 |
| 2 | 0.8 | lowest_total_variation | raw_replace | 2.215 |
| 4 | 0.2 | largest_final_loss | steepest_add | 0.5333 |
| 4 | 0.2 | fastest_to_95pct_loss | steepest_add | 19 |
| 4 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.2 | lowest_high_frequency_ratio | raw_add | 5.828e-08 |
| 4 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.01671 |
| 4 | 0.2 | lowest_total_variation | raw_replace | 0.3172 |
| 4 | 0.2 | largest_final_loss | steepest_add | 0.2183 |
| 4 | 0.2 | fastest_to_95pct_loss | steepest_add | 19 |
| 4 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.2 | lowest_high_frequency_ratio | raw_replace | 0.02729 |
| 4 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.0985 |
| 4 | 0.2 | lowest_total_variation | raw_replace | 0.3748 |
| 4 | 0.2 | largest_final_loss | raw_replace | 42.13 |
| 4 | 0.2 | fastest_to_95pct_loss | raw_replace | 12 |
| 4 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.2 | lowest_high_frequency_ratio | raw_add | 3.301e-06 |
| 4 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.2102 |
| 4 | 0.2 | lowest_total_variation | raw_replace | 3.424 |
| 4 | 0.2 | largest_final_loss | steepest_add | 0.5651 |
| 4 | 0.2 | fastest_to_95pct_loss | steepest_add | 63 |
| 4 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.2 | lowest_high_frequency_ratio | raw_replace | 0.007984 |
| 4 | 0.2 | lowest_first_derivative_l2 | raw_add | 0.6623 |
| 4 | 0.2 | lowest_total_variation | raw_add | 3.643 |
| 4 | 0.4 | largest_final_loss | steepest_add | 0.5321 |
| 4 | 0.4 | fastest_to_95pct_loss | steepest_add | 10 |
| 4 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | lowest_high_frequency_ratio | raw_add | 5.828e-08 |
| 4 | 0.4 | lowest_first_derivative_l2 | raw_replace | 0.01671 |
| 4 | 0.4 | lowest_total_variation | raw_replace | 0.3172 |
| 4 | 0.4 | largest_final_loss | steepest_add | 0.2202 |
| 4 | 0.4 | fastest_to_95pct_loss | steepest_add | 10 |
| 4 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | lowest_high_frequency_ratio | raw_add | 0.02663 |
| 4 | 0.4 | lowest_first_derivative_l2 | raw_add | 0.09385 |
| 4 | 0.4 | lowest_total_variation | raw_replace | 0.3748 |
| 4 | 0.4 | largest_final_loss | raw_replace | 42.13 |
| 4 | 0.4 | fastest_to_95pct_loss | raw_replace | 12 |
| 4 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | lowest_high_frequency_ratio | raw_add | 2.375e-06 |
| 4 | 0.4 | lowest_first_derivative_l2 | raw_replace | 0.2102 |
| 4 | 0.4 | lowest_total_variation | raw_replace | 3.424 |
| 4 | 0.4 | largest_final_loss | steepest_add | 0.6112 |
| 4 | 0.4 | fastest_to_95pct_loss | steepest_add | 49 |
| 4 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.4 | lowest_high_frequency_ratio | steepest_add | 0.003627 |
| 4 | 0.4 | lowest_first_derivative_l2 | steepest_add | 0.6766 |
| 4 | 0.4 | lowest_total_variation | raw_replace | 3.835 |
| 4 | 0.8 | largest_final_loss | steepest_add | 0.5281 |
| 4 | 0.8 | fastest_to_95pct_loss | steepest_add | 5 |
| 4 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.8 | lowest_high_frequency_ratio | raw_add | 5.828e-08 |
| 4 | 0.8 | lowest_first_derivative_l2 | raw_replace | 0.01671 |
| 4 | 0.8 | lowest_total_variation | raw_replace | 0.3172 |
| 4 | 0.8 | largest_final_loss | steepest_add | 0.2187 |
| 4 | 0.8 | fastest_to_95pct_loss | steepest_add | 5 |
| 4 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.8 | lowest_high_frequency_ratio | raw_add | 0.02111 |
| 4 | 0.8 | lowest_first_derivative_l2 | raw_add | 0.08054 |
| 4 | 0.8 | lowest_total_variation | raw_replace | 0.3748 |
| 4 | 0.8 | largest_final_loss | raw_replace | 42.13 |
| 4 | 0.8 | fastest_to_95pct_loss | raw_replace | 12 |
| 4 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.8 | lowest_high_frequency_ratio | raw_add | 2.821e-06 |
| 4 | 0.8 | lowest_first_derivative_l2 | raw_replace | 0.2102 |
| 4 | 0.8 | lowest_total_variation | raw_replace | 3.424 |
| 4 | 0.8 | largest_final_loss | steepest_add | 0.6364 |
| 4 | 0.8 | fastest_to_95pct_loss | steepest_add | 40 |
| 4 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 0.8 | lowest_high_frequency_ratio | steepest_add | 0.002558 |
| 4 | 0.8 | lowest_first_derivative_l2 | steepest_add | 0.6125 |
| 4 | 0.8 | lowest_total_variation | raw_replace | 3.835 |
| 4 | 1.2 | largest_final_loss | steepest_add | 0.5246 |
| 4 | 1.2 | fastest_to_95pct_loss | steepest_add | 4 |
| 4 | 1.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 1.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 1.2 | lowest_high_frequency_ratio | raw_add | 5.828e-08 |
| 4 | 1.2 | lowest_first_derivative_l2 | raw_replace | 0.01671 |
| 4 | 1.2 | lowest_total_variation | raw_replace | 0.3172 |
| 4 | 1.2 | largest_final_loss | steepest_add | 0.2153 |
| 4 | 1.2 | fastest_to_95pct_loss | steepest_replace | 1 |
| 4 | 1.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 1.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 1.2 | lowest_high_frequency_ratio | raw_add | 0.01718 |
| 4 | 1.2 | lowest_first_derivative_l2 | raw_add | 0.07563 |
| 4 | 1.2 | lowest_total_variation | raw_replace | 0.3748 |
| 4 | 1.2 | largest_final_loss | raw_replace | 42.13 |
| 4 | 1.2 | fastest_to_95pct_loss | raw_replace | 12 |
| 4 | 1.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 1.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 1.2 | lowest_high_frequency_ratio | raw_add | 3.262e-06 |
| 4 | 1.2 | lowest_first_derivative_l2 | raw_replace | 0.2102 |
| 4 | 1.2 | lowest_total_variation | raw_replace | 3.424 |
| 4 | 1.2 | largest_final_loss | steepest_add | 0.6384 |
| 4 | 1.2 | fastest_to_95pct_loss | steepest_add | 30 |
| 4 | 1.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 1.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 1.2 | lowest_high_frequency_ratio | steepest_add | 0.002442 |
| 4 | 1.2 | lowest_first_derivative_l2 | steepest_add | 0.6196 |
| 4 | 1.2 | lowest_total_variation | raw_replace | 3.835 |
| 4 | 1.6 | largest_final_loss | steepest_add | 0.5221 |
| 4 | 1.6 | fastest_to_95pct_loss | steepest_add | 3 |
| 4 | 1.6 | fastest_mean_boundary_99pct | raw_add | 1 |
| 4 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 1.6 | lowest_high_frequency_ratio | raw_add | 5.828e-08 |
| 4 | 1.6 | lowest_first_derivative_l2 | raw_replace | 0.01671 |
| 4 | 1.6 | lowest_total_variation | raw_replace | 0.3172 |
| 4 | 1.6 | largest_final_loss | steepest_add | 0.2138 |
| 4 | 1.6 | fastest_to_95pct_loss | steepest_replace | 1 |
| 4 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 1.6 | lowest_high_frequency_ratio | raw_add | 0.01858 |
| 4 | 1.6 | lowest_first_derivative_l2 | raw_add | 0.07822 |
| 4 | 1.6 | lowest_total_variation | raw_replace | 0.3748 |
| 4 | 1.6 | largest_final_loss | raw_replace | 42.13 |
| 4 | 1.6 | fastest_to_95pct_loss | raw_add | 12 |
| 4 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 1.6 | lowest_high_frequency_ratio | steepest_add | 2.847e-06 |
| 4 | 1.6 | lowest_first_derivative_l2 | raw_replace | 0.2102 |
| 4 | 1.6 | lowest_total_variation | raw_replace | 3.424 |
| 4 | 1.6 | largest_final_loss | steepest_add | 0.6441 |
| 4 | 1.6 | fastest_to_95pct_loss | steepest_add | 33 |
| 4 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 4 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 4 | 1.6 | lowest_high_frequency_ratio | steepest_add | 0.003147 |
| 4 | 1.6 | lowest_first_derivative_l2 | steepest_add | 0.662 |
| 4 | 1.6 | lowest_total_variation | raw_replace | 3.835 |
| 8 | 0.2 | largest_final_loss | steepest_add | 0.8548 |
| 8 | 0.2 | fastest_to_95pct_loss | steepest_add | 39 |
| 8 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.2 | lowest_high_frequency_ratio | raw_add | 5.137e-08 |
| 8 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.03399 |
| 8 | 0.2 | lowest_total_variation | raw_replace | 0.6179 |
| 8 | 0.2 | largest_final_loss | steepest_add | 0.3166 |
| 8 | 0.2 | fastest_to_95pct_loss | steepest_add | 38 |
| 8 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.2 | lowest_high_frequency_ratio | raw_replace | 0.0219 |
| 8 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.1735 |
| 8 | 0.2 | lowest_total_variation | raw_replace | 0.7012 |
| 8 | 0.2 | largest_final_loss | raw_add | 114.8 |
| 8 | 0.2 | fastest_to_95pct_loss | raw_replace | 6 |
| 8 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.2 | lowest_high_frequency_ratio | raw_add | 9.004e-07 |
| 8 | 0.2 | lowest_first_derivative_l2 | raw_replace | 0.2728 |
| 8 | 0.2 | lowest_total_variation | raw_replace | 4.028 |
| 8 | 0.2 | largest_final_loss | steepest_add | 0.6689 |
| 8 | 0.2 | fastest_to_95pct_loss | raw_replace | 5 |
| 8 | 0.2 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.2 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.2 | lowest_high_frequency_ratio | raw_replace | 0.01452 |
| 8 | 0.2 | lowest_first_derivative_l2 | raw_add | 0.6623 |
| 8 | 0.2 | lowest_total_variation | raw_add | 3.643 |
| 8 | 0.3 | largest_final_loss | steepest_add | 0.856 |
| 8 | 0.3 | fastest_to_95pct_loss | steepest_add | 27 |
| 8 | 0.3 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.3 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.3 | lowest_high_frequency_ratio | raw_add | 5.16e-08 |
| 8 | 0.3 | lowest_first_derivative_l2 | raw_replace | 0.03399 |
| 8 | 0.3 | lowest_total_variation | raw_replace | 0.6179 |
| 8 | 0.3 | largest_final_loss | steepest_add | 0.3171 |
| 8 | 0.3 | fastest_to_95pct_loss | steepest_add | 26 |
| 8 | 0.3 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.3 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.3 | lowest_high_frequency_ratio | raw_replace | 0.0219 |
| 8 | 0.3 | lowest_first_derivative_l2 | raw_replace | 0.1735 |
| 8 | 0.3 | lowest_total_variation | raw_replace | 0.7012 |
| 8 | 0.3 | largest_final_loss | raw_add | 116.2 |
| 8 | 0.3 | fastest_to_95pct_loss | raw_replace | 8 |
| 8 | 0.3 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.3 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.3 | lowest_high_frequency_ratio | raw_add | 8.801e-07 |
| 8 | 0.3 | lowest_first_derivative_l2 | raw_replace | 0.2728 |
| 8 | 0.3 | lowest_total_variation | raw_replace | 4.028 |
| 8 | 0.3 | largest_final_loss | steepest_add | 0.739 |
| 8 | 0.3 | fastest_to_95pct_loss | steepest_add | 78 |
| 8 | 0.3 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.3 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.3 | lowest_high_frequency_ratio | raw_add | 0.01271 |
| 8 | 0.3 | lowest_first_derivative_l2 | raw_add | 0.6928 |
| 8 | 0.3 | lowest_total_variation | raw_add | 4.664 |
| 8 | 0.4 | largest_final_loss | steepest_add | 0.8574 |
| 8 | 0.4 | fastest_to_95pct_loss | steepest_add | 20 |
| 8 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.4 | lowest_high_frequency_ratio | raw_add | 5.166e-08 |
| 8 | 0.4 | lowest_first_derivative_l2 | raw_replace | 0.03399 |
| 8 | 0.4 | lowest_total_variation | raw_replace | 0.6179 |
| 8 | 0.4 | largest_final_loss | steepest_add | 0.317 |
| 8 | 0.4 | fastest_to_95pct_loss | steepest_add | 20 |
| 8 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.4 | lowest_high_frequency_ratio | raw_add | 0.02055 |
| 8 | 0.4 | lowest_first_derivative_l2 | raw_add | 0.1515 |
| 8 | 0.4 | lowest_total_variation | raw_replace | 0.7012 |
| 8 | 0.4 | largest_final_loss | raw_add | 117.3 |
| 8 | 0.4 | fastest_to_95pct_loss | raw_replace | 8 |
| 8 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.4 | lowest_high_frequency_ratio | raw_add | 7.99e-07 |
| 8 | 0.4 | lowest_first_derivative_l2 | raw_replace | 0.2728 |
| 8 | 0.4 | lowest_total_variation | raw_replace | 4.028 |
| 8 | 0.4 | largest_final_loss | steepest_add | 0.8279 |
| 8 | 0.4 | fastest_to_95pct_loss | steepest_add | 78 |
| 8 | 0.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.4 | lowest_high_frequency_ratio | raw_add | 0.008698 |
| 8 | 0.4 | lowest_first_derivative_l2 | raw_add | 0.7144 |
| 8 | 0.4 | lowest_total_variation | raw_add | 5.411 |
| 8 | 0.8 | largest_final_loss | steepest_add | 0.8501 |
| 8 | 0.8 | fastest_to_95pct_loss | steepest_add | 10 |
| 8 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.8 | lowest_high_frequency_ratio | raw_add | 5.168e-08 |
| 8 | 0.8 | lowest_first_derivative_l2 | raw_replace | 0.03399 |
| 8 | 0.8 | lowest_total_variation | raw_replace | 0.6179 |
| 8 | 0.8 | largest_final_loss | steepest_add | 0.3111 |
| 8 | 0.8 | fastest_to_95pct_loss | steepest_add | 10 |
| 8 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.8 | lowest_high_frequency_ratio | raw_add | 0.0121 |
| 8 | 0.8 | lowest_first_derivative_l2 | raw_add | 0.1207 |
| 8 | 0.8 | lowest_total_variation | raw_replace | 0.7012 |
| 8 | 0.8 | largest_final_loss | raw_add | 120.2 |
| 8 | 0.8 | fastest_to_95pct_loss | raw_add | 11 |
| 8 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.8 | lowest_high_frequency_ratio | raw_add | 9.298e-07 |
| 8 | 0.8 | lowest_first_derivative_l2 | raw_replace | 0.2728 |
| 8 | 0.8 | lowest_total_variation | raw_replace | 4.028 |
| 8 | 0.8 | largest_final_loss | steepest_add | 0.9868 |
| 8 | 0.8 | fastest_to_95pct_loss | steepest_add | 74 |
| 8 | 0.8 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 0.8 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 0.8 | lowest_high_frequency_ratio | raw_add | 0.005375 |
| 8 | 0.8 | lowest_first_derivative_l2 | raw_add | 0.8933 |
| 8 | 0.8 | lowest_total_variation | raw_add | 7.719 |
| 8 | 1.6 | largest_final_loss | steepest_add | 0.8418 |
| 8 | 1.6 | fastest_to_95pct_loss | steepest_add | 6 |
| 8 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 1.6 | lowest_high_frequency_ratio | raw_add | 5.168e-08 |
| 8 | 1.6 | lowest_first_derivative_l2 | raw_replace | 0.03399 |
| 8 | 1.6 | lowest_total_variation | raw_replace | 0.6179 |
| 8 | 1.6 | largest_final_loss | steepest_add | 0.2969 |
| 8 | 1.6 | fastest_to_95pct_loss | steepest_replace | 1 |
| 8 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 1.6 | lowest_high_frequency_ratio | raw_add | 0.01365 |
| 8 | 1.6 | lowest_first_derivative_l2 | raw_add | 0.1249 |
| 8 | 1.6 | lowest_total_variation | raw_replace | 0.7012 |
| 8 | 1.6 | largest_final_loss | raw_add | 120.8 |
| 8 | 1.6 | fastest_to_95pct_loss | raw_add | 9 |
| 8 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 1.6 | lowest_high_frequency_ratio | steepest_add | 1.062e-06 |
| 8 | 1.6 | lowest_first_derivative_l2 | raw_replace | 0.2728 |
| 8 | 1.6 | lowest_total_variation | raw_replace | 4.028 |
| 8 | 1.6 | largest_final_loss | steepest_add | 1.058 |
| 8 | 1.6 | fastest_to_95pct_loss | steepest_add | 64 |
| 8 | 1.6 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 1.6 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 1.6 | lowest_high_frequency_ratio | steepest_add | 0.001302 |
| 8 | 1.6 | lowest_first_derivative_l2 | steepest_add | 0.7342 |
| 8 | 1.6 | lowest_total_variation | steepest_add | 6.936 |
| 8 | 2.4 | largest_final_loss | steepest_add | 0.8333 |
| 8 | 2.4 | fastest_to_95pct_loss | steepest_replace | 5 |
| 8 | 2.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 2.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 2.4 | lowest_high_frequency_ratio | raw_add | 5.167e-08 |
| 8 | 2.4 | lowest_first_derivative_l2 | raw_replace | 0.03399 |
| 8 | 2.4 | lowest_total_variation | raw_replace | 0.6179 |
| 8 | 2.4 | largest_final_loss | steepest_add | 0.284 |
| 8 | 2.4 | fastest_to_95pct_loss | steepest_replace | 1 |
| 8 | 2.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 2.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 2.4 | lowest_high_frequency_ratio | raw_add | 0.01203 |
| 8 | 2.4 | lowest_first_derivative_l2 | raw_add | 0.1289 |
| 8 | 2.4 | lowest_total_variation | raw_replace | 0.7012 |
| 8 | 2.4 | largest_final_loss | raw_add | 119.3 |
| 8 | 2.4 | fastest_to_95pct_loss | raw_add | 8 |
| 8 | 2.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 2.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 2.4 | lowest_high_frequency_ratio | steepest_add | 6.722e-07 |
| 8 | 2.4 | lowest_first_derivative_l2 | raw_add | 0.2676 |
| 8 | 2.4 | lowest_total_variation | raw_add | 3.862 |
| 8 | 2.4 | largest_final_loss | steepest_add | 1.05 |
| 8 | 2.4 | fastest_to_95pct_loss | steepest_add | 39 |
| 8 | 2.4 | fastest_mean_boundary_99pct | raw_replace | 1 |
| 8 | 2.4 | fastest_all_samples_boundary_99pct | raw_replace | 1 |
| 8 | 2.4 | lowest_high_frequency_ratio | steepest_add | 0.001103 |
| 8 | 2.4 | lowest_first_derivative_l2 | steepest_add | 0.6977 |
| 8 | 2.4 | lowest_total_variation | steepest_add | 5.739 |

## Evidence And Interpretation

Observed from the generated CSV tables above: final-loss, boundary-arrival speed, and smoothness winners are computed within each fixed `(epsilon, alpha, p, q)` setting.

Inference from these summaries should be made within the fixed `p=2,q=2` scope unless additional P/Q roots are added and analyzed.

Boundary arrival uses `boundary_ratio = ||delta||_p / epsilon`. A 99% threshold is used as the main practical boundary test to avoid numerical roundoff around exactly `1.0`.

The smoothness metrics are numerical proxies for whether the final perturbation is high-frequency or sharp. They should be read together with the saved delta trajectory/heatmap figures under each setting root.
