# Loss3 Core4/PQ Landscape Probe - 2026-05-20

Status: completed small GPU evaluation pilot. It reused existing core4 deltas and did not rerun optimizers.

## Scope

- Setting roots: `1`
- Sample indices: `[0, 1, 2, 3, 4]`
- Output directory: `analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/burgers_landscape_ridge_probe`
- Row counts: `{'ray': 1700, 'arc': 255, 'slice': 250, 'curvature': 20}`

GPU runtime was verified and recorded in `manifest.json` before evaluating the landscape points.

## Tables and Figures

- Ray profile table: `analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/burgers_landscape_ridge_probe/tables/ray_profile.csv`
- Boundary arc table: `analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/burgers_landscape_ridge_probe/tables/boundary_arc.csv`
- 2D slice grid: `analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/burgers_landscape_ridge_probe/tables/slice_2d_grid.csv`
- Curvature table: `analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/burgers_landscape_ridge_probe/tables/curvature_finite_difference.csv`
- Figures: `analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/burgers_landscape_ridge_probe/figures`

## Endpoint Ray Loss Along Final Delta Directions

| pq | method | direction_kind | trajectory_step | radius_fraction | loss3_q_mean | loss3_q_std | delta_l2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p2q2 | raw_add | final |  | 1 | 4.295 | 2.163 | 8 |
| p2q2 | raw_replace | final |  | 1 | 6.115 | 1.1 | 8 |
| p2q2 | steepest_add | final |  | 1 | 6.392 | 2.085 | 8 |
| p2q2 | steepest_replace | final |  | 1 | 6.115 | 1.1 | 8 |

## Boundary Arc Aggregate Snapshot

| pq | pair | s | loss3_q_mean | loss3_q_std |
| --- | --- | --- | --- | --- |
| p2q2 | raw_add__steepest_add | 0 | 4.295 | 2.163 |
| p2q2 | raw_add__steepest_add | 0.0625 | 4.35 | 2.179 |
| p2q2 | raw_add__steepest_add | 0.125 | 4.421 | 2.185 |
| p2q2 | raw_add__steepest_add | 0.1875 | 4.476 | 2.207 |
| p2q2 | raw_add__steepest_add | 0.25 | 4.55 | 2.215 |
| p2q2 | raw_add__steepest_add | 0.3125 | 4.74 | 2.124 |
| p2q2 | raw_add__steepest_add | 0.375 | 4.958 | 2.032 |
| p2q2 | raw_add__steepest_add | 0.4375 | 5.169 | 1.97 |
| p2q2 | raw_add__steepest_add | 0.5 | 5.369 | 1.933 |
| p2q2 | raw_add__steepest_add | 0.5625 | 5.554 | 1.921 |
| p2q2 | raw_add__steepest_add | 0.625 | 5.722 | 1.929 |
| p2q2 | raw_add__steepest_add | 0.6875 | 5.875 | 1.948 |
| p2q2 | raw_add__steepest_add | 0.75 | 6.011 | 1.974 |
| p2q2 | raw_add__steepest_add | 0.8125 | 6.129 | 2.002 |
| p2q2 | raw_add__steepest_add | 0.875 | 6.231 | 2.031 |
| p2q2 | raw_add__steepest_add | 0.9375 | 6.319 | 2.059 |
| p2q2 | raw_add__steepest_add | 1 | 6.392 | 2.085 |
| p2q2 | steepest_replace__raw_add | 0 | 6.115 | 1.1 |
| p2q2 | steepest_replace__raw_add | 0.0625 | 6.142 | 1.108 |
| p2q2 | steepest_replace__raw_add | 0.125 | 6.144 | 1.129 |
| p2q2 | steepest_replace__raw_add | 0.1875 | 6.113 | 1.165 |
| p2q2 | steepest_replace__raw_add | 0.25 | 6.04 | 1.221 |
| p2q2 | steepest_replace__raw_add | 0.3125 | 5.914 | 1.304 |
| p2q2 | steepest_replace__raw_add | 0.375 | 5.728 | 1.421 |
| p2q2 | steepest_replace__raw_add | 0.4375 | 5.478 | 1.59 |
| p2q2 | steepest_replace__raw_add | 0.5 | 5.238 | 1.772 |
| p2q2 | steepest_replace__raw_add | 0.5625 | 5.017 | 1.946 |
| p2q2 | steepest_replace__raw_add | 0.625 | 4.731 | 2.189 |
| p2q2 | steepest_replace__raw_add | 0.6875 | 4.5 | 2.348 |
| p2q2 | steepest_replace__raw_add | 0.75 | 4.462 | 2.277 |
| p2q2 | steepest_replace__raw_add | 0.8125 | 4.415 | 2.243 |
| p2q2 | steepest_replace__raw_add | 0.875 | 4.364 | 2.225 |
| p2q2 | steepest_replace__raw_add | 0.9375 | 4.321 | 2.202 |
| p2q2 | steepest_replace__raw_add | 1 | 4.295 | 2.163 |
| p2q2 | steepest_replace__steepest_add | 0 | 6.115 | 1.1 |
| p2q2 | steepest_replace__steepest_add | 0.0625 | 6.162 | 1.108 |
| p2q2 | steepest_replace__steepest_add | 0.125 | 6.195 | 1.129 |
| p2q2 | steepest_replace__steepest_add | 0.1875 | 6.211 | 1.166 |
| p2q2 | steepest_replace__steepest_add | 0.25 | 6.205 | 1.223 |
| p2q2 | steepest_replace__steepest_add | 0.3125 | 6.17 | 1.306 |

## Curvature Aggregate Snapshot

| pq | center_method | direction | curvature_second_diff_mean | curvature_second_diff_std | loss3_q_center_mean |
| --- | --- | --- | --- | --- | --- |
| p2q2 | steepest_add | radial | -0.01973 | 0.01589 | 6.392 |
| p2q2 | steepest_add | toward_steepest_replace | -0.07838 | 0.07047 | 6.392 |
| p2q2 | steepest_replace | radial | -0.02564 | 0.089 | 6.115 |
| p2q2 | steepest_replace | toward_steepest_add | 0.01083 | 0.0598 | 6.115 |

## Interpretation Notes

- This is the first small landscape probe tied to the current core4/PQ deltas rather than the older PGD-only path experiments.
- Ray profiles test whether early/final GPI directions are already strong along the full radius.
- p=2 boundary arcs test whether method final deltas are connected by a high-loss ridge on the L2 boundary.
- 2D slices and finite-difference curvature are pilot-scale; use them as directional evidence, not final statistics.
