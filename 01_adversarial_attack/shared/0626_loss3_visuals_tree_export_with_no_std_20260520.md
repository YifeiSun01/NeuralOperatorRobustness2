# Loss3 Tree-Style Figure Export With No-Std Curves - 2026-05-20

Export folder: `forensics/loss3_visuals_tree_export_20260520_with_no_std`

This export reorganizes the previous flat image export into a navigable folder tree. It contains image files only (`.png` and `.gif`) plus directories; no Markdown, CSV, JSON, or numeric artifacts are stored inside the export folder.

## Top-Level Layout

- `visuals/`: loss/boundary/angle curves, heatmaps, representative samples, delta grids, and dynamics triptychs.
- `similarity/`: final-delta pairwise similarity heatmaps and rollup matrices.
- `gpi_early_step_comparison/`: GPI early-step delta/loss/cosine figures.
- `baseline_giftrace/`: baseline trajectory curves, delta GIFs, and condition GIFs.

Within each visual group, curve figures are split into `with_std` and `no_std` subfolders where applicable.

## Counts

| group | image count |
|---|---:|
| `baseline_giftrace/main_run_figures` | 44 |
| `baseline_giftrace/trajectory_condition_gifs` | 16 |
| `gpi_early_step_comparison` | 6 |
| `similarity/p2q2/100step` | 6 |
| `similarity/p2q2/300step` | 6 |
| `similarity/pneq/p1_q2/100step` | 6 |
| `similarity/pneq/p1_qinf/100step` | 6 |
| `similarity/pneq/p2_q1/100step` | 6 |
| `similarity/pneq/p2_qinf/100step` | 6 |
| `similarity/pneq/pinf_q1_partial/100step` | 6 |
| `visuals/p2q2/100step` | 46 |
| `visuals/p2q2/300step` | 71 |
| `visuals/pneq/p1_q2/100step` | 68 |
| `visuals/pneq/p1_qinf/100step` | 68 |
| `visuals/pneq/p2_q1/100step` | 68 |
| `visuals/pneq/p2_qinf/100step` | 68 |
| `visuals/pneq/pinf_q1_partial/100step` | 24 |

Total images: **521**.
Images with `_no_std` in filename: **158**.
