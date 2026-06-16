# Darcy Random-Source Posthoc Analysis (20260613_random_binary_source_1100_posthoc)

- Created: 2026-06-13T08:34:56+00:00
- Models: `random_fixed_y`, `random_solver_y`.
- These are evaluation/analysis runs only; no additional training is performed here.

## Artifacts

- `clean_dataset_csv`: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/clean_eval_by_model_dataset.csv`
- `clean_split_csv`: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/clean_eval_by_model_split.csv`
- `attack20_out_dir`: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/attack20_52datasets_50samples`
- `attack20_viz_dir`: `visualizations/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/attack20_52datasets_50samples`
- `metric25_out_dir`: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/metric_correlation_25samples`
- `metric25_viz_dir`: `visualizations/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/metric_correlation_25samples`
- `jacobian5_out_dir`: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/jacobian_probe_5gen`
- `jacobian5_viz_dir`: `visualizations/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/jacobian_probe_5gen`
- `elapsed_seconds`: `874.433`

## Clean Eval Means

| model | split | mean relative L2 | median relative L2 | mean RMSE |
|---|---|---:|---:|---:|
| random_fixed_y | all | 0.0721128 | 0.0715691 | 0.000765648 |
| random_fixed_y | train | 0.024404 | 0.024404 | 0.000168725 |
| random_fixed_y | test | 0.0288059 | 0.0288059 | 0.000199131 |
| random_fixed_y | generalization | 0.0739331 | 0.0737779 | 0.000788917 |
| random_solver_y | all | 0.0930441 | 0.0988787 | 0.000986022 |
| random_solver_y | train | 0.0168455 | 0.0168455 | 0.000116467 |
| random_solver_y | test | 0.0261327 | 0.0261327 | 0.000180652 |
| random_solver_y | generalization | 0.0959063 | 0.0995798 | 0.00101952 |

## Attack20 Means

| model | split | clean loss | attack gain | attacked loss |
|---|---|---:|---:|---:|
| random_fixed_y | all | 5.07706e-07 | 3.01997e-06 | 3.52767e-06 |
| random_fixed_y | train | 3.98826e-08 | 3.54745e-07 | 3.94627e-07 |
| random_fixed_y | test | 4.63255e-08 | 3.22537e-07 | 3.68863e-07 |
| random_fixed_y | generalization | 5.26291e-07 | 3.12722e-06 | 3.65351e-06 |
| random_solver_y | all | 8.29141e-07 | 4.47295e-06 | 5.30209e-06 |
| random_solver_y | train | 1.89613e-08 | 1.27662e-06 | 1.29558e-06 |
| random_solver_y | test | 2.51169e-08 | 1.4551e-06 | 1.48021e-06 |
| random_solver_y | generalization | 8.61425e-07 | 4.59724e-06 | 5.45866e-06 |

## 5-Sample Jacobian Probe Means

| model | rel L2 | ||J^T e|| | ||J^T e||^2 | top sigma |
|---|---:|---:|---:|---:|
| random_fixed_y | 0.0806844 | 0.000117679 | 1.75566e-08 | 0.00184299 |
| random_solver_y | 0.10239 | 0.00015803 | 2.93647e-08 | 0.00186718 |
