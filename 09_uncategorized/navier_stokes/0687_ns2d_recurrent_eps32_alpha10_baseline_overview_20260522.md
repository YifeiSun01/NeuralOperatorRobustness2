# NS2D Recurrent FNO eps32 alpha10 Baseline Overview - 2026-05-22

## Status

Generated CPU-only overview plots for the completed baseline attack pair `epsilon=32`, `alpha=10`.

## Observed Evidence

- Source pair root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10`.
- Output directory: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522`.
- Plotting script: `2D_NS_FNO2d_recurrent/visualizations/plot_eps32_alpha10_baseline_overview.py`.
- Report JSON: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522/eps32_alpha10_baseline_overview_report.json`.
- Summary CSV: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522/eps32_alpha10_baseline_final_metric_summary.csv`.
- The plotting command used `CUDA_VISIBLE_DEVICES=''`; it reads saved CSV/NPZ files and does not import torch/JAX or rerun model/solver.
- Follow-up GPU query after plotting reported `NVIDIA A100-SXM4-80GB, 47197, 81920, 99`, so the long-running attack remained active.

## Generated Plots

- Loss/objective curves with standard-deviation bands: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522/eps32_alpha10_baseline_loss_curves_with_std.png`.
- Delta norm and boundary-ratio curves with standard-deviation bands: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522/eps32_alpha10_baseline_delta_norm_boundary_with_std.png`.
- Sample-0 angle diagnostics: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522/eps32_alpha10_baseline_angle_curves_sample0.png`.
- Sample-0 final perturbation grid: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522/eps32_alpha10_baseline_final_delta_sample0_grid.png`.
- Final metric heatmap: `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522/eps32_alpha10_baseline_final_metrics_heatmap.png`.

## Notes On Angle Plots

The angle curves are based on `step_sample_trace.npz`, which records only `step_sample_position=0` for each attack method. Therefore:

- `angle(delta, direction)` is sample-0 only.
- `angle(delta_k, delta_(k-1))` is sample-0 only.
- Loss/objective/delta-norm curves are batch summaries across the 10 attacked samples and include standard-deviation shading.

## Best Method By Block

Observed from `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522/eps32_alpha10_baseline_final_metric_summary.csv`:

| Block | Best method by true-loss increase | True-loss increase | Final true loss | Final boundary ratio |
|---|---|---:|---:|---:|
| `loss1/all_w` | `steepest_add` | 89.791573 | 158.283305 | 0.997810 |
| `loss2/all_a_target_w` | `steepest_add` | 13.779233 | 82.271230 | 1.000000 |
| `loss3/all_w` | `steepest_add` | 236.100975 | 304.592972 | 1.000000 |
| `loss3/all_d_target_w` | `steepest_add` | 147.921621 | 216.413618 | 1.000000 |
| `loss3/w1_5_d6_9_target_w` | `steepest_add` | 139.366045 | 207.858042 | 1.000000 |
| `loss3/d1_5_w6_9_target_w` | `steepest_add` | 220.735313 | 289.227310 | 1.000000 |
| `loss3/a1_5_d6_9_target_w` | `steepest_add` | 108.555902 | 177.047899 | 1.000000 |

## Overall Top True-Loss Increases

| Rank | Block | Method | True-loss increase | Final true loss | Runtime min |
|---:|---|---|---:|---:|---:|
| 1 | `loss3/all_w` | `steepest_add` | 236.100975 | 304.592972 | 30.316820 |
| 2 | `loss3/d1_5_w6_9_target_w` | `steepest_add` | 220.735313 | 289.227310 | 30.967611 |
| 3 | `loss3/all_d_target_w` | `steepest_add` | 147.921621 | 216.413618 | 30.457681 |
| 4 | `loss3/w1_5_d6_9_target_w` | `steepest_add` | 139.366045 | 207.858042 | 30.775962 |
| 5 | `loss3/a1_5_d6_9_target_w` | `steepest_add` | 108.555902 | 177.047899 | 30.668232 |

## Inference

For the completed `epsilon=32`, `alpha=10` baseline pair, `steepest_add` is the best method by final true-loss increase in every completed block. The largest observed effect is `loss3/all_w` with `steepest_add`. The perturbation grid provides a visual check of the final sample-0 perturbation patterns, while the angle curves show how sample-0 perturbation directions rotate relative to the current update direction.

## Revision: v2 Plot Corrections

Generated an updated output directory:
`2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522_v2`.

Changes in v2:

- The true-loss column in the loss-curve overview now uses one shared y-axis range across all rows, so the blocks are visually comparable.
- The objective and true-loss curves now mark the first crossing of `25%`, `50%`, `75%`, and `100%` of the epsilon boundary with open markers.
- The delta norm and boundary-ratio plots also mark the same `25%`, `50%`, `75%`, and `100%` boundary crossing points.
- The boundary marker shapes are: circle for `25%`, triangle for `50%`, square for `75%`, and X for `100%`.

Updated v2 plot files:

- `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522_v2/eps32_alpha10_baseline_loss_curves_with_std_shared_true_y_thresholds.png`.
- `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522_v2/eps32_alpha10_baseline_delta_norm_boundary_with_std_thresholds.png`.
- `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522_v2/eps32_alpha10_baseline_angle_curves_sample0.png`.
- `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522_v2/eps32_alpha10_baseline_final_delta_sample0_grid.png`.
- `2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522_v2/eps32_alpha10_baseline_final_metrics_heatmap.png`.

Clarification of the angle diagnostic:

- `angle(delta, direction)` is the angle between the current perturbation `delta_k` and the current update direction `s_k` for the recorded sample. Here `s_k` is the method-specific update direction derived from the gradient. Small values mean the current update direction points along the existing perturbation; values near `90` degrees mean it is mostly rotating sideways; values near `180` degrees mean it points against the current perturbation.
- `angle(delta_k, delta_(k-1))` is the rotation angle of the perturbation itself between consecutive saved steps.
- Both angle plots are sample-0 diagnostics only because the attack run saved per-step trace only for `step_sample_position=0`.

## Revision: Perturbed Final Output Heatmaps

Generated additional sample-0 heatmap grids from saved `final_state_outputs.npz` files. These are CPU-only plots and do not rerun the model or solver.

Output directory:
`2D_NS_FNO2d_recurrent/visualizations/eps32_alpha10_baseline_overview_20260522_v2`.

Generated files:

- `eps32_alpha10_baseline_adv_model_final_sample0_grid.png`: perturbed final FNO output `F(x + delta)`.
- `eps32_alpha10_baseline_adv_solver_final_sample0_grid.png`: perturbed final solver output `G(x + delta)`.
- `eps32_alpha10_baseline_adv_model_minus_solver_sample0_grid.png`: signed difference `F(x + delta) - G(x + delta)`.
- `eps32_alpha10_baseline_abs_adv_model_minus_solver_sample0_grid.png`: absolute difference `|F(x + delta) - G(x + delta)|`.
- `eps32_alpha10_baseline_adv_final_output_diff_sample0_summary.csv`: numeric summary for sample position 0.
- `eps32_alpha10_baseline_adv_final_output_diff_sample0_report.json`: source/report metadata.

Each subplot title records:

- dataset index,
- adversarial true loss,
- true-loss increase from clean,
- true-loss ratio,
- L2 norm of `F(x + delta) - G(x + delta)`.

Top sample-0 true-loss increases from the generated summary:

| Block | Method | Adv true loss | True-loss increase | Ratio |
|---|---|---:|---:|---:|
| `loss3/all_w` | `steepest_add` | 263.904083 | 229.079956 | 7.578197 |
| `loss3/d1_5_w6_9_target_w` | `steepest_add` | 257.590118 | 222.765991 | 7.396887 |
| `loss3/w1_5_d6_9_target_w` | `steepest_add` | 145.218689 | 110.394562 | 4.170060 |
| `loss3/all_d_target_w` | `steepest_add` | 145.115677 | 110.291550 | 4.167102 |
| `loss1/all_w` | `steepest_add` | 133.663818 | 98.839691 | 3.838253 |

Note: these values are for `sample_position=0`; they should not be confused with the batch mean/std curves.
