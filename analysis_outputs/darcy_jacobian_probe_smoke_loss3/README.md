# Darcy Jacobian Probe (smoke_jacobian_loss3)

- Created: 2026-06-12T06:51:23+00:00
- Samples: 5 generated Darcy generalization samples, sample_index=0.
- Models: loss1, loss2, loss3, fixed (fixed = physics-loss model checkpoint).
- `J^T error` is the gradient of `0.5 * ||model(x) - solver(x)||_2^2` with respect to the input coefficient field.
- Top singular value is estimated by `3` power iterations using JVP/VJP, not by materializing the full 7225 x 7225 Jacobian.

## Model Means

| model | mean rel L2 | mean ||J^T e|| | mean ||J^T e||^2 | mean ||J e|| | mean top sigma | max top sigma |
|---|---:|---:|---:|---:|---:|---:|
| loss3 | 0.0785856 | 0.00012645 | 1.85496e-08 | 5.04957e-05 | 0.0020827 | 0.00231292 |

## Outputs

- Summary: `analysis_outputs/darcy_jacobian_probe_smoke_loss3/summary.csv`
- Model summary: `analysis_outputs/darcy_jacobian_probe_smoke_loss3/summary_by_model.csv`
- Selected samples: `analysis_outputs/darcy_jacobian_probe_smoke_loss3/selected_samples.csv`
- Saved vectors: `analysis_outputs/darcy_jacobian_probe_smoke_loss3/vectors`
- Figures: `visualizations/darcy_jacobian_probe_smoke_loss3`
