# Loss3 Core4/PQ Landscape Probe - 2026-05-20

Status: completed small GPU evaluation pilot. It reused existing core4 deltas and did not rerun optimizers.

## Scope

- Setting roots: `1`
- Sample indices: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- Output directory: `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20`
- Row counts: `{'ray': 6800, 'arc': 1020, 'slice': 1000, 'curvature': 80}`

GPU runtime was verified and recorded in `manifest.json` before evaluating the landscape points.

## Tables and Figures

- Ray profile table: `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20/tables/ray_profile.csv`
- Boundary arc table: `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20/tables/boundary_arc.csv`
- 2D slice grid: `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20/tables/slice_2d_grid.csv`
- Curvature table: `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20/tables/curvature_finite_difference.csv`
- Figures: `analysis_outputs/mechanism_20260622/full_mechanism_validation/burgers_landscape_ridge_probe_N20/figures`

## Endpoint Ray Loss Along Final Delta Directions

| pq | method | direction_kind | trajectory_step | radius_fraction | loss3_q_mean | loss3_q_std | delta_l2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p2q2 | raw_add | final |  | 1 | 5.358 | 2.648 | 8 |
| p2q2 | raw_replace | final |  | 1 | 7.069 | 1.446 | 8 |
| p2q2 | steepest_add | final |  | 1 | 6.339 | 2.351 | 8 |
| p2q2 | steepest_replace | final |  | 1 | 7.069 | 1.446 | 8 |

## Boundary Arc Aggregate Snapshot

| pq | pair | s | loss3_q_mean | loss3_q_std |
| --- | --- | --- | --- | --- |
| p2q2 | raw_add__steepest_add | 0 | 5.358 | 2.648 |
| p2q2 | raw_add__steepest_add | 0.0625 | 5.395 | 2.637 |
| p2q2 | raw_add__steepest_add | 0.125 | 5.438 | 2.62 |
| p2q2 | raw_add__steepest_add | 0.1875 | 5.481 | 2.607 |
| p2q2 | raw_add__steepest_add | 0.25 | 5.531 | 2.589 |
| p2q2 | raw_add__steepest_add | 0.3125 | 5.613 | 2.538 |
| p2q2 | raw_add__steepest_add | 0.375 | 5.704 | 2.487 |
| p2q2 | raw_add__steepest_add | 0.4375 | 5.793 | 2.445 |
| p2q2 | raw_add__steepest_add | 0.5 | 5.881 | 2.412 |
| p2q2 | raw_add__steepest_add | 0.5625 | 5.963 | 2.387 |
| p2q2 | raw_add__steepest_add | 0.625 | 6.04 | 2.369 |
| p2q2 | raw_add__steepest_add | 0.6875 | 6.111 | 2.357 |
| p2q2 | raw_add__steepest_add | 0.75 | 6.173 | 2.349 |
| p2q2 | raw_add__steepest_add | 0.8125 | 6.226 | 2.346 |
| p2q2 | raw_add__steepest_add | 0.875 | 6.272 | 2.346 |
| p2q2 | raw_add__steepest_add | 0.9375 | 6.309 | 2.348 |
| p2q2 | raw_add__steepest_add | 1 | 6.339 | 2.351 |
| p2q2 | steepest_replace__raw_add | 0 | 7.069 | 1.446 |
| p2q2 | steepest_replace__raw_add | 0.0625 | 7.11 | 1.455 |
| p2q2 | steepest_replace__raw_add | 0.125 | 7.127 | 1.466 |
| p2q2 | steepest_replace__raw_add | 0.1875 | 7.114 | 1.482 |
| p2q2 | steepest_replace__raw_add | 0.25 | 7.063 | 1.505 |
| p2q2 | steepest_replace__raw_add | 0.3125 | 6.963 | 1.544 |
| p2q2 | steepest_replace__raw_add | 0.375 | 6.808 | 1.612 |
| p2q2 | steepest_replace__raw_add | 0.4375 | 6.602 | 1.723 |
| p2q2 | steepest_replace__raw_add | 0.5 | 6.354 | 1.868 |
| p2q2 | steepest_replace__raw_add | 0.5625 | 6.048 | 2.079 |
| p2q2 | steepest_replace__raw_add | 0.625 | 5.781 | 2.292 |
| p2q2 | steepest_replace__raw_add | 0.6875 | 5.538 | 2.506 |
| p2q2 | steepest_replace__raw_add | 0.75 | 5.356 | 2.669 |
| p2q2 | steepest_replace__raw_add | 0.8125 | 5.294 | 2.734 |
| p2q2 | steepest_replace__raw_add | 0.875 | 5.343 | 2.679 |
| p2q2 | steepest_replace__raw_add | 0.9375 | 5.362 | 2.657 |
| p2q2 | steepest_replace__raw_add | 1 | 5.358 | 2.648 |
| p2q2 | steepest_replace__steepest_add | 0 | 7.069 | 1.446 |
| p2q2 | steepest_replace__steepest_add | 0.0625 | 7.119 | 1.448 |
| p2q2 | steepest_replace__steepest_add | 0.125 | 7.151 | 1.451 |
| p2q2 | steepest_replace__steepest_add | 0.1875 | 7.162 | 1.454 |
| p2q2 | steepest_replace__steepest_add | 0.25 | 7.145 | 1.461 |
| p2q2 | steepest_replace__steepest_add | 0.3125 | 7.092 | 1.479 |

## Curvature Aggregate Snapshot

| pq | center_method | direction | curvature_second_diff_mean | curvature_second_diff_std | loss3_q_center_mean |
| --- | --- | --- | --- | --- | --- |
| p2q2 | steepest_add | radial | -0.02553 | 0.02799 | 6.339 |
| p2q2 | steepest_add | toward_steepest_replace | -0.1597 | 0.1314 | 6.339 |
| p2q2 | steepest_replace | radial | -0.02147 | 0.05245 | 7.069 |
| p2q2 | steepest_replace | toward_steepest_add | -0.01151 | 0.04829 | 7.069 |

## Interpretation Notes

- This is the first small landscape probe tied to the current core4/PQ deltas rather than the older PGD-only path experiments.
- Ray profiles test whether early/final GPI directions are already strong along the full radius.
- p=2 boundary arcs test whether method final deltas are connected by a high-loss ridge on the L2 boundary.
- 2D slices and finite-difference curvature are pilot-scale; use them as directional evidence, not final statistics.
