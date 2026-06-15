# Darcy CFlow Final Delta FFT Polished-Report Addendum - 2026-06-15

Status: complete.

## Why It Was Missing

The current organized-release polished report was rebuilt as a clean-evaluation
report. Its original README explicitly said final-model attack heatmaps were not
included.

The older Burgers-style Darcy reports had a figure named like:

```text
polished_checkpoint_style_delta_fft_raw_heatmap_25epoch_smoothed_lines_*.png
```

That older figure requires per-epoch `attack_probe_samples.csv` plus per-epoch
attack-probe NPZ files. The final organized release does not contain that
epoch-wise attack-probe history for every final model.

## What Was Added

A final-checkpoint Delta FFT addendum was generated from the completed final
attack20 delta NPZ files:

```text
outputs/darcy_cflow_final_robustness_20260615_full_attack20_svd25/data/robustness_deltas/
```

This does not rerun attack or training. It reads stored final deltas and computes
FFT diagnostics on the 50 lossdrop generalization datasets.

Implementation:

```text
tools/add_darcy_cflow_final_delta_fft_polished_report_20260615.py
```

Manifest:

```text
outputs/darcy_cflow_timematched_organized_release_20260614/manifests/final_delta_fft_polished_report_20260615.json
```

## Generated Figures

Top-level overview:

```text
outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/final_delta_fft/polished_final_delta_fft_log_magnitude_grid_all7.png
outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/final_delta_fft/polished_final_delta_radial_spectrum_all7.png
```

Per-model figures were added under:

```text
outputs/darcy_cflow_timematched_organized_release_20260614/figures/polished_report/<method>/
```

Each model has:

```text
polished_final_delta_fft_log_magnitude_heatmap_<method>.png
polished_final_delta_fft_dataset_radial_heatmap_<method>.png
```

The seven methods covered are:

- baseline
- loss1
- loss2
- loss3
- Physics Loss
- random clean
- random solver

## Generated Data

```text
outputs/darcy_cflow_timematched_organized_release_20260614/data/polished_report/final_delta_fft/final_delta_fft_dataset_metrics.csv
outputs/darcy_cflow_timematched_organized_release_20260614/data/polished_report/final_delta_fft/final_delta_fft_summary_by_model.csv
```

Summary values:

| method | datasets | samples | high-frequency ratio mean | spectral centroid mean |
|---|---:|---:|---:|---:|
| baseline | 50 | 2400 | 0.111549 | 0.151099 |
| loss3 | 50 | 2400 | 0.120269 | 0.181955 |
| Physics Loss | 50 | 2400 | 0.173449 | 0.223407 |
| random solver | 50 | 2400 | 0.170962 | 0.226435 |
| loss2 | 50 | 2400 | 0.182039 | 0.227218 |
| random clean | 50 | 2400 | 0.185718 | 0.230360 |
| loss1 | 50 | 2400 | 0.189292 | 0.236244 |

## Interpretation Boundary

These figures are final-checkpoint FFT diagnostics. They should not be described
as epoch-wise attack-probe FFT histories. The old epoch-wise figure can only be
recreated for runs that saved `attack_probe_samples.csv` and per-epoch probe NPZ
files.
