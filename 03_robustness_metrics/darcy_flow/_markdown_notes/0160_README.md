# Darcy Jacobian Probe (20260613_7model_complete_missing_metrics)

- Created: 2026-06-13T22:01:28+00:00
- Samples: 5 generated Darcy generalization samples, sample_index=0.
- Models: baseline.
- `J^T error` is the gradient of `0.5 * ||model(x) - solver(x)||_2^2` with respect to the input coefficient field.
- Top singular value is estimated by `12` power iterations using JVP/VJP, not by materializing the full 7225 x 7225 Jacobian.

## Model Means

| model | mean rel L2 | mean ||J^T e|| | mean ||J^T e||^2 | mean ||J e|| | mean top sigma | max top sigma |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.0966342 | 0.000133847 | 2.1466e-08 | 5.41212e-05 | 0.00173231 | 0.00192314 |

## Outputs

- Summary: `analysis_outputs/darcy_baseline_jacobian_probe_5gen_20260613/summary.csv`
- Model summary: `analysis_outputs/darcy_baseline_jacobian_probe_5gen_20260613/summary_by_model.csv`
- Selected samples: `analysis_outputs/darcy_baseline_jacobian_probe_5gen_20260613/selected_samples.csv`
- Saved vectors: `analysis_outputs/darcy_baseline_jacobian_probe_5gen_20260613/vectors`
- Figures: `visualizations/darcy_baseline_jacobian_probe_5gen_20260613`
