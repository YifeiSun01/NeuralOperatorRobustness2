# Darcy/C-flow Soft vs Binary TAG Audit - 2026-06-09

## Question

The intended Darcy/C-flow TAG setting may require binary coefficient fields, matching the original GRF-threshold Darcy benchmark where the coefficient has only low/high values.

## Observed Evidence

Current visualization and weak-TAG examples used files under:

- `generalization_datasets_darcy_lossdrop50_selected_20260607/darcy/`

The selected files are named `darcy_lossdrop_pool_soft_*`, for example:

- `darcy_lossdrop_pool_soft_l4_h10_b12_10.pt`
- `darcy_lossdrop_pool_soft_l4_h10_b14_07.pt`

Direct tensor inspection of selected samples showed these are not binary fields:

| file | sample | min | max | rounded unique values |
|---|---:|---:|---:|---:|
| `darcy_lossdrop_pool_soft_l4_h10_b12_10.pt` | 25 | `4.00037145614624` | `9.99985122680664` | `7197` |
| `darcy_lossdrop_pool_soft_l4_h10_b14_07.pt` | 30 | `4.000173568725586` | `9.993141174316406` | `7196` |

The original trained Darcy benchmark configs include binary coefficient metadata, for example:

- `2D_Darcy_FNO2d/saved_models/2D/darcy_flow_other30training_20260608/config.json`
- dataset path contains `binary3-12`
- metadata says thresholded GRF with low/high coefficient and `soft_coefficients: false`

The current TAG attack implementation is still a binary-replace attack:

- Source: `tools/adversarial_training.py`
- Function: `binary_darcy_replace_attack`
- It chooses pixels by gradient score and replaces each selected value by that sample's min or max.

## Conclusion

Observed from the local files: the large loss3-advantage visualizations and recent weak-TAG metrics were generated on `soft` Darcy coefficient fields, not on the original binary GRF-threshold Darcy coefficient fields.

This explains why the plotted `delta` has many red/blue intensities: for soft fields, `delta = replaced_value - original_value`, and `original_value` varies continuously.

Inference: if the intended PhD/Darcy TAG experiment must preserve the binary coefficient-field setting, the recent soft-field TAG plots and metrics should be treated as a separate soft-coefficient experiment, not as evidence for binary Darcy TAG robustness.

## Required Follow-Up

Run a corrected binary-coefficient TAG audit using binary Darcy generalization datasets, or explicitly threshold the selected soft fields before attack if the purpose is only a controlled visualization. The preferred correction is to use true binary Darcy generalization files so that the data distribution matches the binary-trained Darcy benchmark.
