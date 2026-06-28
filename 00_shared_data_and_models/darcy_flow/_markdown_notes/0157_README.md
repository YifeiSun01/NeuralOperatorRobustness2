# Darcy 52-dataset 20-step attack benchmark (20260612_attack20_1000c_52datasets_50samples)

- Created: 2026-06-12T06:32:37+00:00
- Attack objective: shared `loss3` solver-consistent MSE for all four models.
- Datasets: train + test + generated generalization, max datasets=all.
- Samples per dataset: 50, sample policy `random`, seed `20260612`.
- Attack steps: 20, epsilon fraction `0.025`.

Lowest mean absolute loss growth overall: `loss3` (2.69124e-06).
Lowest relative growth from means overall: `loss1` (5.09028).

## Overall

| model | clean | attacked | abs gain | rel gain from means | mean sample rel gain | samples |
|---|---:|---:|---:|---:|---:|---:|
| loss1 | 9.38239e-07 | 5.71414e-06 | 4.7759e-06 | 5.09028 | 18.9199 | 2600 |
| loss2 | 6.89173e-07 | 4.6215e-06 | 3.93232e-06 | 5.70586 | 15.8011 | 2600 |
| loss3 | 4.619e-07 | 3.15314e-06 | 2.69124e-06 | 5.82646 | 12.2067 | 2600 |
| physics | 7.61504e-07 | 4.85337e-06 | 4.09187e-06 | 5.3734 | 20.1133 | 2600 |

## Split Summary

| model | split | clean | attacked | abs gain | rel gain from means | samples |
|---|---|---:|---:|---:|---:|---:|
| loss1 | train | 3.16258e-08 | 2.77397e-06 | 2.74234e-06 | 86.7121 | 50 |
| loss1 | test | 3.73152e-08 | 3.03757e-06 | 3.00025e-06 | 80.4029 | 50 |
| loss1 | generalization | 9.7439e-07 | 5.82647e-06 | 4.85208e-06 | 4.97961 | 2500 |
| loss2 | train | 2.7771e-08 | 3.98165e-07 | 3.70394e-07 | 13.3374 | 50 |
| loss2 | test | 2.93319e-08 | 7.00166e-07 | 6.70834e-07 | 22.8704 | 50 |
| loss2 | generalization | 7.15598e-07 | 4.78439e-06 | 4.06879e-06 | 5.68586 | 2500 |
| loss3 | train | 8.28652e-08 | 1.76033e-06 | 1.67747e-06 | 20.2434 | 50 |
| loss3 | test | 9.03944e-08 | 1.82099e-06 | 1.73059e-06 | 19.1449 | 50 |
| loss3 | generalization | 4.76911e-07 | 3.20764e-06 | 2.73073e-06 | 5.72588 | 2500 |
| physics | train | 2.57631e-08 | 2.8389e-06 | 2.81314e-06 | 109.193 | 50 |
| physics | test | 3.35102e-08 | 2.74336e-06 | 2.70985e-06 | 80.8665 | 50 |
| physics | generalization | 7.90779e-07 | 4.93586e-06 | 4.14508e-06 | 5.24177 | 2500 |

## Outputs

- Samples CSV: `analysis_outputs/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples/all_samples.csv`
- Dataset summary CSV: `analysis_outputs/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples/summary_by_dataset_model.csv`
- Model summary CSV: `analysis_outputs/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples/summary_by_model_split.csv`
- Selected sample manifest: `analysis_outputs/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples/selected_samples_manifest.csv`
- Figures: `visualizations/darcy_attack20_52datasets_50samples_20260612_attack20_1000c_52datasets_50samples`
