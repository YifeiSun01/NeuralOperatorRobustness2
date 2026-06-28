# NS2D Recurrent FNO Loss1 Loss Curves - 2026-05-22

## Status

Generated CPU-only loss curves for the completed baseline `epsilon=32, alpha=10`, `loss1/all_w` attack block.

## Observed Evidence

- Source loss root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1`.
- Source files: each method's `per_step_metrics.csv`.
- Plotting script: `2D_NS_FNO2d_recurrent/visualizations/plot_attack_loss_curves.py`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_loss_curves_pair_outer_eps32_alpha10_20260522`.
- Step-curve PNG: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_loss_curves_pair_outer_eps32_alpha10_20260522/loss1_loss_curve_overview_with_std.png`.
- Wall-time PNG: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_loss_curves_pair_outer_eps32_alpha10_20260522/loss1_loss_curve_wall_time_with_std.png`.
- Summary CSV: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_loss_curves_pair_outer_eps32_alpha10_20260522/loss1_loss_curve_summary_with_std.csv`.
- Report JSON: `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_loss_curves_pair_outer_eps32_alpha10_20260522/loss1_loss_curve_report_with_std.json`.
- The plotting command used `CUDA_VISIBLE_DEVICES=''`; the script imports NumPy/matplotlib only and does not touch GPU.

## Plot Contents

The overview plot contains four panels with shaded `mean +/- std` bands across the 10 attacked samples:

- `loss1_mean` objective curve with `loss1_std` shading.
- `true_loss_mean` curve with `true_loss_std` shading.
- `boundary_ratio_mean = ||delta||_p / epsilon` with std derived from `delta_p_std / epsilon`.
- `true_loss_mean - true_loss_mean@k0` with `true_loss_std` shading.

## Summary Table

Observed from `2D_NS_FNO2d_recurrent/visualizations/loss1_attack_loss_curves_pair_outer_eps32_alpha10_20260522/loss1_loss_curve_summary_with_std.csv`:

| Method | 100% boundary step | Final objective | Final objective std | Initial true | Final true | True increase | Final true std | True ratio | Final boundary | Runtime min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw_add` | 1 | 445.810025 | 30.303622 | 68.491732 | 93.041568 | 24.549836 | 42.799631 | 1.358435 | 1.000000 | 18.601783 |
| `raw_replace` | 61 | 401.017053 | 50.599833 | 68.491732 | 98.326353 | 29.834621 | 47.266085 | 1.435594 | 1.000000 | 18.519913 |
| `steepest_add` | 12 | 519.401230 | 25.633891 | 68.491732 | 158.283305 | 89.791573 | 43.656279 | 2.310984 | 0.997810 | 18.525295 |
| `steepest_replace` | 61 | 401.017053 | 50.599833 | 68.491732 | 98.326353 | 29.834621 | 47.266085 | 1.435594 | 1.000000 | 18.545024 |

## Inference

For this completed `loss1` baseline block, `steepest_add` gives the clearest true-loss growth by the end of 100 steps. `raw_add` reaches the boundary immediately in the mean curve, but its final true-loss increase is much smaller than `steepest_add`. The standard deviation bands are large, so individual samples vary substantially and should be checked against the saved final-state panels.
