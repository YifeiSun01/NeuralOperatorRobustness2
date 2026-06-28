# Darcy Jacobian Probe (20260613_random_binary_source_1100_posthoc)

- Created: 2026-06-13T08:34:56+00:00
- Samples: 5 generated Darcy generalization samples, sample_index=0.
- Models: random_fixed_y, random_solver_y.
- `J^T error` is the gradient of `0.5 * ||model(x) - solver(x)||_2^2` with respect to the input coefficient field.
- Top singular value is estimated by `12` power iterations using JVP/VJP, not by materializing the full 7225 x 7225 Jacobian.

## Model Means

| model | mean rel L2 | mean ||J^T e|| | mean ||J^T e||^2 | mean ||J e|| | mean top sigma | max top sigma |
|---|---:|---:|---:|---:|---:|---:|
| random_fixed_y | 0.0806844 | 0.000117679 | 1.75566e-08 | 4.29411e-05 | 0.00184299 | 0.00203821 |
| random_solver_y | 0.10239 | 0.00015803 | 2.93647e-08 | 6.02202e-05 | 0.00186718 | 0.00207804 |

## Outputs

- Summary: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/jacobian_probe_5gen/summary.csv`
- Model summary: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/jacobian_probe_5gen/summary_by_model.csv`
- Selected samples: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/jacobian_probe_5gen/selected_samples.csv`
- Saved vectors: `analysis_outputs/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/jacobian_probe_5gen/vectors`
- Figures: `visualizations/darcy_random_binary_source_20260613_random_binary_source_1100_posthoc/jacobian_probe_5gen`
