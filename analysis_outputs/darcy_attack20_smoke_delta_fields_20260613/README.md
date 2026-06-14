# Darcy 52-dataset 20-step attack benchmark (smoke_delta_fields_20260613)

- Created: 2026-06-13T21:27:04+00:00
- Attack objective: shared `loss3` solver-consistent MSE for all four models.
- Datasets: train + test + generated generalization, max datasets=1.
- Samples per dataset: 2, sample policy `first`, seed `20260612`.
- Attack steps: 1, epsilon fraction `0.025`.

Lowest mean absolute loss growth overall: `loss3` (4.63125e-08).
Lowest relative growth from means overall: `loss3` (0.723873).

## Overall

| model | clean | attacked | abs gain | rel gain from means | mean sample rel gain | delta RMS | flip frac | samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| loss3 | 6.39787e-08 | 1.10291e-07 | 4.63125e-08 | 0.723873 | 0.670586 | 1.4245 | 0.0250519 | 2 |

## Split Summary

| model | split | clean | attacked | abs gain | rel gain from means | samples |
|---|---|---:|---:|---:|---:|---:|
| loss3 | train | 6.39787e-08 | 1.10291e-07 | 4.63125e-08 | 0.723873 | 2 |

## Outputs

- Samples CSV: `analysis_outputs/darcy_attack20_smoke_delta_fields_20260613/all_samples.csv`
- Dataset summary CSV: `analysis_outputs/darcy_attack20_smoke_delta_fields_20260613/summary_by_dataset_model.csv`
- Model summary CSV: `analysis_outputs/darcy_attack20_smoke_delta_fields_20260613/summary_by_model_split.csv`
- Selected sample manifest: `analysis_outputs/darcy_attack20_smoke_delta_fields_20260613/selected_samples_manifest.csv`
- Figures: `visualizations/darcy_attack20_smoke_delta_fields_20260613`
