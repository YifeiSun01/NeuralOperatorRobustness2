# Darcy 52-dataset 20-step attack benchmark (probe_attack20_2datasets_50samples_loss3)

- Created: 2026-06-12T06:17:18+00:00
- Attack objective: shared `loss3` solver-consistent MSE for all four models.
- Datasets: train + test + generated generalization, max datasets=2.
- Samples per dataset: 50, sample policy `random`, seed `20260612`.
- Attack steps: 20, epsilon fraction `0.025`.

Lowest mean absolute loss growth overall: `loss3` (1.70403e-06).
Lowest relative growth from means overall: `loss3` (19.6703).

## Overall

| model | clean | attacked | abs gain | rel gain from means | mean sample rel gain | samples |
|---|---:|---:|---:|---:|---:|---:|
| loss3 | 8.66298e-08 | 1.79066e-06 | 1.70403e-06 | 19.6703 | 27.6793 | 100 |

## Split Summary

| model | split | clean | attacked | abs gain | rel gain from means | samples |
|---|---|---:|---:|---:|---:|---:|
| loss3 | train | 8.28652e-08 | 1.76033e-06 | 1.67747e-06 | 20.2434 | 50 |
| loss3 | test | 9.03944e-08 | 1.82099e-06 | 1.73059e-06 | 19.1449 | 50 |

## Outputs

- Samples CSV: `analysis_outputs/darcy_attack20_52datasets_50samples_probe_attack20_2datasets_50samples_loss3/all_samples.csv`
- Dataset summary CSV: `analysis_outputs/darcy_attack20_52datasets_50samples_probe_attack20_2datasets_50samples_loss3/summary_by_dataset_model.csv`
- Model summary CSV: `analysis_outputs/darcy_attack20_52datasets_50samples_probe_attack20_2datasets_50samples_loss3/summary_by_model_split.csv`
- Selected sample manifest: `analysis_outputs/darcy_attack20_52datasets_50samples_probe_attack20_2datasets_50samples_loss3/selected_samples_manifest.csv`
- Figures: `visualizations/darcy_attack20_52datasets_50samples_probe_attack20_2datasets_50samples_loss3`
