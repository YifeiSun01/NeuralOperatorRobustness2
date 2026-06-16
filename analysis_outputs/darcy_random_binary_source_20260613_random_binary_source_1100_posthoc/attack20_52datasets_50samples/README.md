# Darcy 52-dataset 20-step attack benchmark (20260613_random_binary_source_1100_posthoc)

- Created: 2026-06-13T08:27:41+00:00
- Attack objective: shared `loss3` solver-consistent MSE for all four models.
- Datasets: train + test + generated generalization, max datasets=all.
- Samples per dataset: 50, sample policy `random`, seed `20260612`.
- Attack steps: 20, epsilon fraction `0.025`.

Lowest mean absolute loss growth overall: `random_fixed_y` (3.01997e-06).
Lowest relative growth from means overall: `random_solver_y` (5.39468).

## Overall

| model | clean | attacked | abs gain | rel gain from means | mean sample rel gain | samples |
|---|---:|---:|---:|---:|---:|---:|
| random_fixed_y | 5.07706e-07 | 3.52767e-06 | 3.01997e-06 | 5.94826 | 13.4774 | 2600 |
| random_solver_y | 8.29141e-07 | 5.30209e-06 | 4.47295e-06 | 5.39468 | 18.0445 | 2600 |

## Split Summary

| model | split | clean | attacked | abs gain | rel gain from means | samples |
|---|---|---:|---:|---:|---:|---:|
| random_fixed_y | train | 3.98826e-08 | 3.94627e-07 | 3.54745e-07 | 8.89472 | 50 |
| random_fixed_y | test | 4.63255e-08 | 3.68863e-07 | 3.22537e-07 | 6.96241 | 50 |
| random_fixed_y | generalization | 5.26291e-07 | 3.65351e-06 | 3.12722e-06 | 5.942 | 2500 |
| random_solver_y | train | 1.89613e-08 | 1.29558e-06 | 1.27662e-06 | 67.3276 | 50 |
| random_solver_y | test | 2.51169e-08 | 1.48021e-06 | 1.4551e-06 | 57.9329 | 50 |
| random_solver_y | generalization | 8.61425e-07 | 5.45866e-06 | 4.59724e-06 | 5.33678 | 2500 |

## Outputs

- Samples CSV: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/attack20_52datasets_50samples/all_samples.csv`
- Dataset summary CSV: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/attack20_52datasets_50samples/summary_by_dataset_model.csv`
- Model summary CSV: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/attack20_52datasets_50samples/summary_by_model_split.csv`
- Selected sample manifest: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/attack20_52datasets_50samples/selected_samples_manifest.csv`
- Figures: `visualizations/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/attack20_52datasets_50samples`
