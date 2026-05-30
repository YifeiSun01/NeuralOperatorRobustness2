# NS2D Heatmaps With Delta Spectra And Loss Curves - 2026-05-24

Observed from saved `final_state_outputs.npz`, `per_step_metrics.csv`, and `per_sample_step_metrics.csv` only. No model inference, solver rollout, attack update, PyTorch, JAX, or GPU work was run.

Each PNG has a 2 x 2 top line-plot block. The loss curves use the same final all-W true-loss metric for every attack target.

## Outputs

- Summary CSV: [row_loss_summary.csv](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/row_loss_summary.csv)
- Manifest JSON: [manifest.json](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/manifest.json)
- PNG directory: `docs/ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/`

## Rendered Figures

| Epsilon / Alpha | Optimizer | attack rows | highlighted final true-loss row | highlighted true loss | PNG |
|---|---|---:|---|---:|---|
| Epsilon = 160, Alpha = 50 | Steepest Add | 3 | Loss 2 / all A -> W | 312.9 | [eps160_alpha50_steepest_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps160_alpha50_steepest_add_spectrum_loss_curves_dataset0.png) |
| Epsilon = 32, Alpha = 10 | Raw Add | 7 | Loss 3 / all W | 93.87 | [eps32_alpha10_raw_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps32_alpha10_raw_add_spectrum_loss_curves_dataset0.png) |
| Epsilon = 32, Alpha = 10 | Raw Replace | 7 | Loss 3 / W steps 1-5, D steps 6-9 -> W | 94.71 | [eps32_alpha10_raw_replace_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps32_alpha10_raw_replace_spectrum_loss_curves_dataset0.png) |
| Epsilon = 32, Alpha = 10 | Steepest Add | 7 | Loss 3 / all W | 263.9 | [eps32_alpha10_steepest_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps32_alpha10_steepest_add_spectrum_loss_curves_dataset0.png) |
| Epsilon = 32, Alpha = 10 | Steepest Replace | 7 | Loss 3 / W steps 1-5, D steps 6-9 -> W | 94.71 | [eps32_alpha10_steepest_replace_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps32_alpha10_steepest_replace_spectrum_loss_curves_dataset0.png) |
| Epsilon = 16, Alpha = 5 | Raw Add | 3 | Loss 3 / all W | 74.67 | [eps16_alpha5_raw_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps16_alpha5_raw_add_spectrum_loss_curves_dataset0.png) |
| Epsilon = 16, Alpha = 5 | Raw Replace | 3 | Loss 3 / all W | 47.45 | [eps16_alpha5_raw_replace_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps16_alpha5_raw_replace_spectrum_loss_curves_dataset0.png) |
| Epsilon = 16, Alpha = 5 | Steepest Add | 3 | Loss 3 / all W | 123 | [eps16_alpha5_steepest_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps16_alpha5_steepest_add_spectrum_loss_curves_dataset0.png) |
| Epsilon = 16, Alpha = 5 | Steepest Replace | 3 | Loss 3 / all W | 47.45 | [eps16_alpha5_steepest_replace_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps16_alpha5_steepest_replace_spectrum_loss_curves_dataset0.png) |
| Epsilon = 8, Alpha = 2.5 | Raw Add | 7 | Loss 3 / all W | 65.07 | [eps8_alpha2p5_raw_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps8_alpha2p5_raw_add_spectrum_loss_curves_dataset0.png) |
| Epsilon = 8, Alpha = 2.5 | Raw Replace | 7 | Loss 3 / D steps 1-5, W steps 6-9 -> W | 55.25 | [eps8_alpha2p5_raw_replace_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps8_alpha2p5_raw_replace_spectrum_loss_curves_dataset0.png) |
| Epsilon = 8, Alpha = 2.5 | Steepest Add | 7 | Loss 3 / all W | 94.82 | [eps8_alpha2p5_steepest_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps8_alpha2p5_steepest_add_spectrum_loss_curves_dataset0.png) |
| Epsilon = 8, Alpha = 2.5 | Steepest Replace | 7 | Loss 3 / D steps 1-5, W steps 6-9 -> W | 55.25 | [eps8_alpha2p5_steepest_replace_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps8_alpha2p5_steepest_replace_spectrum_loss_curves_dataset0.png) |
| Epsilon = 4, Alpha = 1.25 | Raw Add | 3 | Loss 3 / all W | 59.5 | [eps4_alpha1p25_raw_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps4_alpha1p25_raw_add_spectrum_loss_curves_dataset0.png) |
| Epsilon = 4, Alpha = 1.25 | Raw Replace | 3 | Loss 3 / all W | 47.32 | [eps4_alpha1p25_raw_replace_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps4_alpha1p25_raw_replace_spectrum_loss_curves_dataset0.png) |
| Epsilon = 4, Alpha = 1.25 | Steepest Add | 3 | Loss 3 / all W | 59.56 | [eps4_alpha1p25_steepest_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps4_alpha1p25_steepest_add_spectrum_loss_curves_dataset0.png) |
| Epsilon = 4, Alpha = 1.25 | Steepest Replace | 3 | Loss 3 / all W | 47.32 | [eps4_alpha1p25_steepest_replace_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps4_alpha1p25_steepest_replace_spectrum_loss_curves_dataset0.png) |
| Epsilon = 2, Alpha = 0.625 | Raw Add | 3 | Loss 3 / all W | 45.31 | [eps2_alpha0p625_raw_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps2_alpha0p625_raw_add_spectrum_loss_curves_dataset0.png) |
| Epsilon = 2, Alpha = 0.625 | Raw Replace | 3 | Loss 3 / all W | 45.31 | [eps2_alpha0p625_raw_replace_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps2_alpha0p625_raw_replace_spectrum_loss_curves_dataset0.png) |
| Epsilon = 2, Alpha = 0.625 | Steepest Add | 3 | Loss 3 / all W | 45.31 | [eps2_alpha0p625_steepest_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps2_alpha0p625_steepest_add_spectrum_loss_curves_dataset0.png) |
| Epsilon = 2, Alpha = 0.625 | Steepest Replace | 3 | Loss 3 / all W | 45.31 | [eps2_alpha0p625_steepest_replace_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps2_alpha0p625_steepest_replace_spectrum_loss_curves_dataset0.png) |
| Epsilon = 1, Alpha = 0.3125 | Raw Add | 3 | Loss 3 / all W | 39.81 | [eps1_alpha0p3125_raw_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps1_alpha0p3125_raw_add_spectrum_loss_curves_dataset0.png) |
| Epsilon = 1, Alpha = 0.3125 | Raw Replace | 3 | Loss 3 / all W | 39.81 | [eps1_alpha0p3125_raw_replace_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps1_alpha0p3125_raw_replace_spectrum_loss_curves_dataset0.png) |
| Epsilon = 1, Alpha = 0.3125 | Steepest Add | 3 | Loss 3 / all W | 39.81 | [eps1_alpha0p3125_steepest_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps1_alpha0p3125_steepest_add_spectrum_loss_curves_dataset0.png) |
| Epsilon = 1, Alpha = 0.3125 | Steepest Replace | 3 | Loss 3 / all W | 39.81 | [eps1_alpha0p3125_steepest_replace_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps1_alpha0p3125_steepest_replace_spectrum_loss_curves_dataset0.png) |

## Missing Expected Canonical Rows

- Epsilon = 160, Alpha = 50, Steepest Add: missing Loss 3 / all W
