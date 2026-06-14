# Darcy Jacobian Probe (20260612_jacobian_5gen_1000c)

- Created: 2026-06-12T06:52:56+00:00
- Samples: 5 generated Darcy generalization samples, sample_index=0.
- Models: loss1, loss2, loss3, fixed (fixed = physics-loss model checkpoint).
- `J^T error` is the gradient of `0.5 * ||model(x) - solver(x)||_2^2` with respect to the input coefficient field.
- Top singular value is estimated by `12` power iterations using JVP/VJP, not by materializing the full 7225 x 7225 Jacobian.

## Model Means

| model | mean rel L2 | mean ||J^T e|| | mean ||J^T e||^2 | mean ||J e|| | mean top sigma | max top sigma |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 0.0982516 | 0.000153299 | 2.75102e-08 | 6.23465e-05 | 0.00188965 | 0.00209871 |
| loss1 | 0.106734 | 0.000160822 | 3.01177e-08 | 6.57173e-05 | 0.00182829 | 0.00201389 |
| loss2 | 0.0943854 | 0.000134626 | 2.24368e-08 | 5.52805e-05 | 0.00175943 | 0.00198079 |
| loss3 | 0.0785856 | 0.00012645 | 1.85496e-08 | 5.04957e-05 | 0.00210832 | 0.00231467 |

## Outputs

- Summary: `analysis_outputs/darcy_jacobian_probe_5gen_20260612_jacobian_5gen_1000c/summary.csv`
- Model summary: `analysis_outputs/darcy_jacobian_probe_5gen_20260612_jacobian_5gen_1000c/summary_by_model.csv`
- Selected samples: `analysis_outputs/darcy_jacobian_probe_5gen_20260612_jacobian_5gen_1000c/selected_samples.csv`
- Saved vectors: `analysis_outputs/darcy_jacobian_probe_5gen_20260612_jacobian_5gen_1000c/vectors`
- Figures: `visualizations/darcy_jacobian_probe_5gen_20260612_jacobian_5gen_1000c`
