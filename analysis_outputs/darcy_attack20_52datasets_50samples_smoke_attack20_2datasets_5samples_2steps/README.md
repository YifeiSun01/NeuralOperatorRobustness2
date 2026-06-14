# Darcy 52-dataset 20-step attack benchmark (smoke_attack20_2datasets_5samples_2steps)

- Created: 2026-06-12T06:16:23+00:00
- Attack objective: shared `loss3` solver-consistent MSE for all four models.
- Datasets: train + test + generated generalization, max datasets=2.
- Samples per dataset: 5, sample policy `random`, seed `20260612`.
- Attack steps: 2, epsilon fraction `0.025`.

Lowest mean absolute loss growth overall: `loss1` (7.23215e-08).
Lowest relative growth from means overall: `loss3` (1.40939).

## Overall

| model | clean | attacked | abs gain | rel gain from means | mean sample rel gain | samples |
|---|---:|---:|---:|---:|---:|---:|
| loss1 | 1.96765e-08 | 9.1998e-08 | 7.23215e-08 | 3.67553 | 3.61284 | 10 |
| loss2 | 3.16679e-08 | 1.22237e-07 | 9.05689e-08 | 2.85995 | 3.77747 | 10 |
| loss3 | 7.39789e-08 | 1.78244e-07 | 1.04265e-07 | 1.40939 | 1.47872 | 10 |
| physics | 2.04668e-08 | 1.13498e-07 | 9.30308e-08 | 4.54545 | 5.12383 | 10 |

## Split Summary

| model | split | clean | attacked | abs gain | rel gain from means | samples |
|---|---|---:|---:|---:|---:|---:|
| loss1 | train | 1.89544e-08 | 5.65563e-08 | 3.76019e-08 | 1.9838 | 5 |
| loss1 | test | 2.03986e-08 | 1.2744e-07 | 1.07041e-07 | 5.24749 | 5 |
| loss2 | train | 2.78425e-08 | 1.22473e-07 | 9.46302e-08 | 3.39877 | 5 |
| loss2 | test | 3.54934e-08 | 1.22001e-07 | 8.65076e-08 | 2.43728 | 5 |
| loss3 | train | 6.44966e-08 | 1.69657e-07 | 1.0516e-07 | 1.63048 | 5 |
| loss3 | test | 8.34613e-08 | 1.86831e-07 | 1.0337e-07 | 1.23854 | 5 |
| physics | train | 1.75669e-08 | 1.04728e-07 | 8.71613e-08 | 4.96168 | 5 |
| physics | test | 2.33667e-08 | 1.22267e-07 | 9.89003e-08 | 4.23253 | 5 |

## Outputs

- Samples CSV: `analysis_outputs/darcy_attack20_52datasets_50samples_smoke_attack20_2datasets_5samples_2steps/all_samples.csv`
- Dataset summary CSV: `analysis_outputs/darcy_attack20_52datasets_50samples_smoke_attack20_2datasets_5samples_2steps/summary_by_dataset_model.csv`
- Model summary CSV: `analysis_outputs/darcy_attack20_52datasets_50samples_smoke_attack20_2datasets_5samples_2steps/summary_by_model_split.csv`
- Selected sample manifest: `analysis_outputs/darcy_attack20_52datasets_50samples_smoke_attack20_2datasets_5samples_2steps/selected_samples_manifest.csv`
- Figures: `visualizations/darcy_attack20_52datasets_50samples_smoke_attack20_2datasets_5samples_2steps`
