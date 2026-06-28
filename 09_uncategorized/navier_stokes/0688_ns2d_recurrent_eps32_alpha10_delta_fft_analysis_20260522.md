# NS2D Recurrent FNO eps32 alpha10 Final Delta Fourier Analysis - 2026-05-22

## Status

Generated CPU-only Fourier analysis for all completed `epsilon=32`, `alpha=10` baseline final perturbations.

## Observed Evidence

- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Source arrays: each method's saved `final_state_outputs.npz`, field `final_delta` with 10 samples.
- Analysis script: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_delta_fft_analysis.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_delta_fft_analysis_20260522`.
- Summary CSV: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_delta_fft_analysis_20260522/eps32_alpha10_final_delta_fft_metrics_summary.csv`.
- Report JSON: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_delta_fft_analysis_20260522/eps32_alpha10_final_delta_fft_analysis_report.json`.
- The command used `CUDA_VISIBLE_DEVICES=''`; the script imports NumPy/matplotlib only and does not use GPU, torch, JAX, model, or solver.
- Follow-up GPU query reported `NVIDIA A100-SXM4-80GB, 47197, 81920, 99`, so the long-running attack remained active.

## Generated Plots

- 2D sample-0 FFT log-magnitude grid: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_delta_fft_analysis_20260522/eps32_alpha10_final_delta_fft_log_magnitude_sample0_grid.png`.
- Batch-mean radial FFT power profiles: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_delta_fft_analysis_20260522/eps32_alpha10_final_delta_radial_fft_profiles.png`.
- Low/mid/high frequency band metric heatmaps: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_delta_fft_analysis_20260522/eps32_alpha10_final_delta_fft_band_metrics_heatmaps.png`.

## Metric Definitions

The script uses normalized radial frequency `rho`, where `rho=0` is DC and `rho=1` is the diagonal Nyquist corner. DC is excluded from the band fractions and spectral centroid.

- Low frequency: `0 < rho <= 0.15`.
- Mid frequency: `0.15 < rho <= 0.35`.
- High frequency: `rho > 0.35`.
- Spectral centroid: power-weighted mean `rho` after excluding DC.
- High/low ratio: high-band power divided by low-band power.

## Aggregate By Loss Type

Observed from `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_delta_fft_analysis_20260522/eps32_alpha10_final_delta_fft_metrics_summary.csv`:

| Loss group | Mean low frac | Mean mid frac | Mean high frac | Mean spectral centroid | Mean high/low ratio |
|---|---:|---:|---:|---:|---:|
| `loss1` | 0.999710 | 0.000290 | 8.48e-08 | 0.013960 | 8.48e-08 |
| `loss2` | 0.999977 | 0.000021 | 2.12e-06 | 0.011957 | 2.12e-06 |
| `loss3` | 0.998162 | 0.001837 | 1.01e-06 | 0.019692 | 1.02e-06 |

## Interpretation

The user's visual observation is mostly supported, with one nuance:

- All perturbations are still overwhelmingly low-frequency by total power; even `loss3` has more than `99.5%` of power in the low-frequency band under this banding.
- `loss2` is the most low-frequency / smooth case by spectral centroid and mid-band power. Its mean spectral centroid is about `0.01196`, the lowest among the groups.
- `loss1` is also low-frequency, with spectral centroid around `0.01396`.
- `loss3` is the relatively higher-frequency/curvier case: it has much larger mid-band fraction (`0.001837` mean) and the highest spectral centroid (`0.01969` mean). This matches the visual impression that `loss3` final deltas have more bending/structure.
- The strict high band `rho > 0.35` is tiny for all methods. What visually appears as "higher frequency" is mainly a shift from very low frequencies into low-mid/mid radial frequencies, rather than a large Nyquist-scale high-frequency component.

## Highest Frequency-Shifted Cases

Highest spectral/high-mid cases from the CSV include:

| Block | Method | Low frac | Mid frac | High frac | Spectral centroid |
|---|---|---:|---:|---:|---:|
| `loss3/all_w` | `raw_replace` | 0.995275 | 0.004722 | 3.04e-06 | 0.027451 |
| `loss3/all_w` | `steepest_replace` | 0.995275 | 0.004722 | 3.04e-06 | 0.027451 |
| `loss3/d1_5_w6_9_target_w` | `raw_replace` | 0.995731 | 0.004266 | 2.92e-06 | 0.025889 |
| `loss3/d1_5_w6_9_target_w` | `steepest_replace` | 0.995731 | 0.004266 | 2.92e-06 | 0.025889 |
| `loss3/all_w` | `steepest_add` | 0.996853 | 0.003146 | 1.48e-06 | 0.026301 |

## Inference

For the completed `epsilon=32`, `alpha=10` baseline pair, the final-delta Fourier analysis supports the working hypothesis that `loss2` perturbations are especially smooth/low-frequency, while `loss3` perturbations shift more energy into mid-frequency structure. This is a relative frequency shift, not a large absolute high-frequency explosion.

## Follow-up Visual Observation

Observed from the final-delta grids and the FFT plots: `loss3` combined with LP steepest PGD, recorded as `steepest_add`, produces some of the most visually distinct perturbation structures among the completed baseline cases. This is especially clear in the `loss3/all_w` and related W-heavy mode settings, where the final delta has stronger spatially structured variation than the smoother `loss2` perturbations.

Inference from the plots and CSV metrics: LP steepest PGD is not merely increasing the perturbation norm; it is also selecting a different spatial direction in perturbation space. In the completed `epsilon=32`, `alpha=10` run, that direction appears more effective for producing large final-output/model-solver discrepancies in the 2D recurrent NS attack than the replacement/GPI-style methods in several `loss3` cases.
