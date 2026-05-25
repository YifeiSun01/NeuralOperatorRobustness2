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
| Epsilon = 32, Alpha = 10 | Steepest Add | 1 | Loss 3 / all W | 263.9 | [eps32_alpha10_steepest_add_spectrum_loss_curves_dataset0.png](ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524/eps32_alpha10_steepest_add_spectrum_loss_curves_dataset0.png) |

## Missing Expected Canonical Rows

- Epsilon = 32, Alpha = 10, Steepest Add: missing Loss 1 / all W
- Epsilon = 32, Alpha = 10, Steepest Add: missing Loss 2 / all A -> W
