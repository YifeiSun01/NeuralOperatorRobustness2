# Loss3 Core4/PQ Landscape Probe - 2026-05-20

Status: completed small GPU evaluation pilot. It reused existing core4 deltas and did not rerun optimizers.

## Scope

- Setting roots: `4`
- Sample indices: `[0, 7, 20, 40, 47, 63, 80, 99]`
- Output directory: `forensics/loss3_core4_pq_landscape_probe_full_20260521`
- Row counts: `{'ray': 3968, 'arc': 2952, 'slice': 7744, 'curvature': 128}`

GPU runtime was verified and recorded in `manifest.json` before evaluating the landscape points.

## Tables and Figures

- Ray profile table: `forensics/loss3_core4_pq_landscape_probe_full_20260521/tables/ray_profile.csv`
- Boundary arc table: `forensics/loss3_core4_pq_landscape_probe_full_20260521/tables/boundary_arc.csv`
- 2D slice grid: `forensics/loss3_core4_pq_landscape_probe_full_20260521/tables/slice_2d_grid.csv`
- Curvature table: `forensics/loss3_core4_pq_landscape_probe_full_20260521/tables/curvature_finite_difference.csv`
- Figures: `forensics/loss3_core4_pq_landscape_probe_full_20260521/figures`

## Endpoint Ray Loss Along Final Delta Directions

| pq | method | direction_kind | trajectory_step | radius_fraction | loss3_q_mean | loss3_q_std | delta_l2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p1qinf | raw_add | final |  | 1 | 0.09395 | 0.05422 | 0.2244 |
| p1qinf | raw_replace | final |  | 1 | 0.08898 | 0.05392 | 0.2195 |
| p1qinf | steepest_add | final |  | 1 | 0.202 | 0.09285 | 3.254 |
| p1qinf | steepest_replace | final |  | 1 | 0.2089 | 0.08537 | 4 |
| p2q1 | raw_add | final |  | 1 | 41.75 | 13.5 | 4 |
| p2q1 | raw_replace | final |  | 1 | 46.04 | 9.513 | 4 |
| p2q1 | steepest_add | final |  | 1 | 42.84 | 12.12 | 4 |
| p2q1 | steepest_replace | final |  | 1 | 46.04 | 9.513 | 4 |
| p2q2 | raw_add | final |  | 1 | 3.473 | 0.8732 | 4 |
| p2q2 | raw_replace | final |  | 1 | 3.562 | 0.7922 | 4 |
| p2q2 | steepest_add | final |  | 1 | 3.474 | 0.8725 | 4 |
| p2q2 | steepest_replace | final |  | 1 | 3.562 | 0.7922 | 4 |
| p2qinf | raw_add | final |  | 1 | 0.4856 | 0.1604 | 4 |
| p2qinf | raw_replace | final |  | 1 | 0.416 | 0.1608 | 4 |
| p2qinf | steepest_add | final |  | 1 | 0.6101 | 0.175 | 4 |
| p2qinf | steepest_replace | final |  | 1 | 0.416 | 0.1608 | 4 |

## Boundary Arc Aggregate Snapshot

| pq | pair | s | loss3_q_mean | loss3_q_std |
| --- | --- | --- | --- | --- |
| p2q1 | raw_add__steepest_add | 0 | 41.75 | 13.5 |
| p2q1 | raw_add__steepest_add | 0.025 | 41.76 | 13.5 |
| p2q1 | raw_add__steepest_add | 0.05 | 41.76 | 13.49 |
| p2q1 | raw_add__steepest_add | 0.075 | 41.76 | 13.48 |
| p2q1 | raw_add__steepest_add | 0.1 | 41.75 | 13.48 |
| p2q1 | raw_add__steepest_add | 0.125 | 41.74 | 13.48 |
| p2q1 | raw_add__steepest_add | 0.15 | 41.74 | 13.47 |
| p2q1 | raw_add__steepest_add | 0.175 | 41.74 | 13.44 |
| p2q1 | raw_add__steepest_add | 0.2 | 41.74 | 13.41 |
| p2q1 | raw_add__steepest_add | 0.225 | 41.75 | 13.38 |
| p2q1 | raw_add__steepest_add | 0.25 | 41.75 | 13.35 |
| p2q1 | raw_add__steepest_add | 0.275 | 41.76 | 13.3 |
| p2q1 | raw_add__steepest_add | 0.3 | 41.78 | 13.25 |
| p2q1 | raw_add__steepest_add | 0.325 | 41.81 | 13.19 |
| p2q1 | raw_add__steepest_add | 0.35 | 41.84 | 13.12 |
| p2q1 | raw_add__steepest_add | 0.375 | 41.88 | 13.05 |
| p2q1 | raw_add__steepest_add | 0.4 | 41.93 | 12.98 |
| p2q1 | raw_add__steepest_add | 0.425 | 41.99 | 12.91 |
| p2q1 | raw_add__steepest_add | 0.45 | 42.05 | 12.84 |
| p2q1 | raw_add__steepest_add | 0.475 | 42.11 | 12.77 |
| p2q1 | raw_add__steepest_add | 0.5 | 42.16 | 12.71 |
| p2q1 | raw_add__steepest_add | 0.525 | 42.21 | 12.65 |
| p2q1 | raw_add__steepest_add | 0.55 | 42.26 | 12.6 |
| p2q1 | raw_add__steepest_add | 0.575 | 42.31 | 12.55 |
| p2q1 | raw_add__steepest_add | 0.6 | 42.36 | 12.51 |
| p2q1 | raw_add__steepest_add | 0.625 | 42.41 | 12.47 |
| p2q1 | raw_add__steepest_add | 0.65 | 42.46 | 12.43 |
| p2q1 | raw_add__steepest_add | 0.675 | 42.51 | 12.39 |
| p2q1 | raw_add__steepest_add | 0.7 | 42.56 | 12.35 |
| p2q1 | raw_add__steepest_add | 0.725 | 42.6 | 12.32 |
| p2q1 | raw_add__steepest_add | 0.75 | 42.64 | 12.29 |
| p2q1 | raw_add__steepest_add | 0.775 | 42.68 | 12.26 |
| p2q1 | raw_add__steepest_add | 0.8 | 42.72 | 12.24 |
| p2q1 | raw_add__steepest_add | 0.825 | 42.75 | 12.22 |
| p2q1 | raw_add__steepest_add | 0.85 | 42.77 | 12.2 |
| p2q1 | raw_add__steepest_add | 0.875 | 42.79 | 12.18 |
| p2q1 | raw_add__steepest_add | 0.9 | 42.81 | 12.17 |
| p2q1 | raw_add__steepest_add | 0.925 | 42.83 | 12.15 |
| p2q1 | raw_add__steepest_add | 0.95 | 42.84 | 12.14 |
| p2q1 | raw_add__steepest_add | 0.975 | 42.84 | 12.13 |

## Curvature Aggregate Snapshot

| pq | center_method | direction | curvature_second_diff_mean | curvature_second_diff_std | loss3_q_center_mean |
| --- | --- | --- | --- | --- | --- |
| p1qinf | steepest_add | radial | -0.01233 | 0.01032 | 0.202 |
| p1qinf | steepest_add | toward_steepest_replace | -0.01736 | 0.02998 | 0.202 |
| p1qinf | steepest_replace | radial | -0.01379 | 0.01848 | 0.2089 |
| p1qinf | steepest_replace | toward_steepest_add | -0.007208 | 0.009585 | 0.2089 |
| p2q1 | steepest_add | radial | 1.422 | 2.513 | 42.84 |
| p2q1 | steepest_add | toward_steepest_replace | -0.8178 | 3.244 | 42.84 |
| p2q1 | steepest_replace | radial | 2.386 | 1.81 | 46.04 |
| p2q1 | steepest_replace | toward_steepest_add | 0.5355 | 1.754 | 46.04 |
| p2q2 | steepest_add | radial | -0.1392 | 0.1687 | 3.474 |
| p2q2 | steepest_add | toward_steepest_replace | -0.1199 | 0.1643 | 3.474 |
| p2q2 | steepest_replace | radial | -0.114 | 0.1259 | 3.562 |
| p2q2 | steepest_replace | toward_steepest_add | -0.1207 | 0.2143 | 3.562 |
| p2qinf | steepest_add | radial | -0.08751 | 0.1414 | 0.6101 |
| p2qinf | steepest_add | toward_steepest_replace | -0.5483 | 0.8896 | 0.6101 |
| p2qinf | steepest_replace | radial | 0.06924 | 0.2165 | 0.416 |
| p2qinf | steepest_replace | toward_steepest_add | -0.03269 | 0.194 | 0.416 |

## Interpretation Notes

- This is the first small landscape probe tied to the current core4/PQ deltas rather than the older PGD-only path experiments.
- Ray profiles test whether early/final GPI directions are already strong along the full radius.
- p=2 boundary arcs test whether method final deltas are connected by a high-loss ridge on the L2 boundary.
- 2D slices and finite-difference curvature are pilot-scale; use them as directional evidence, not final statistics.
