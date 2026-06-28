# Loss3 Core4/PQ Landscape Probe - 2026-05-20

Status: completed small GPU evaluation pilot. It reused existing core4 deltas and did not rerun optimizers.

## Scope

- Setting roots: `4`
- Sample indices: `[0, 7, 40, 47]`
- Output directory: `forensics/loss3_core4_pq_landscape_probe_20260520`
- Row counts: `{'ray': 8000, 'arc': 756, 'slice': 1568, 'curvature': 64}`

GPU runtime was verified and recorded in `manifest.json` before evaluating the landscape points.

## Tables and Figures

- Ray profile table: `forensics/loss3_core4_pq_landscape_probe_20260520/tables/ray_profile.csv`
- Boundary arc table: `forensics/loss3_core4_pq_landscape_probe_20260520/tables/boundary_arc.csv`
- 2D slice grid: `forensics/loss3_core4_pq_landscape_probe_20260520/tables/slice_2d_grid.csv`
- Curvature table: `forensics/loss3_core4_pq_landscape_probe_20260520/tables/curvature_finite_difference.csv`
- Figures: `forensics/loss3_core4_pq_landscape_probe_20260520/figures`

## Endpoint Ray Loss Along Final Delta Directions

| pq | method | direction_kind | trajectory_step | radius_fraction | loss3_q_mean | loss3_q_std | delta_l2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p1qinf | raw_add | final |  | 1 | 0.07813 | 0.01953 | 0.2191 |
| p1qinf | raw_replace | final |  | 1 | 0.07433 | 0.01832 | 0.2159 |
| p1qinf | steepest_add | final |  | 1 | 0.1309 | 0.01956 | 3.287 |
| p1qinf | steepest_replace | final |  | 1 | 0.1708 | 0.06264 | 4 |
| p2q1 | raw_add | final |  | 1 | 37.3 | 13.18 | 4 |
| p2q1 | raw_replace | final |  | 1 | 46.69 | 4.491 | 4 |
| p2q1 | steepest_add | final |  | 1 | 41.03 | 10.05 | 4 |
| p2q1 | steepest_replace | final |  | 1 | 46.69 | 4.491 | 4 |
| p2q2 | raw_add | final |  | 1 | 3.629 | 0.3209 | 4 |
| p2q2 | raw_replace | final |  | 1 | 3.634 | 0.5027 | 4 |
| p2q2 | steepest_add | final |  | 1 | 3.629 | 0.3208 | 4 |
| p2q2 | steepest_replace | final |  | 1 | 3.634 | 0.5027 | 4 |
| p2qinf | raw_add | final |  | 1 | 0.4196 | 0.08222 | 4 |
| p2qinf | raw_replace | final |  | 1 | 0.3249 | 0.1718 | 4 |
| p2qinf | steepest_add | final |  | 1 | 0.6106 | 0.1517 | 4 |
| p2qinf | steepest_replace | final |  | 1 | 0.3249 | 0.1718 | 4 |

## Boundary Arc Aggregate Snapshot

| pq | pair | s | loss3_q_mean | loss3_q_std |
| --- | --- | --- | --- | --- |
| p2q1 | raw_add__steepest_add | 0 | 37.3 | 13.18 |
| p2q1 | raw_add__steepest_add | 0.05 | 37.32 | 13.15 |
| p2q1 | raw_add__steepest_add | 0.1 | 37.36 | 13.12 |
| p2q1 | raw_add__steepest_add | 0.15 | 37.41 | 13.07 |
| p2q1 | raw_add__steepest_add | 0.2 | 37.54 | 12.93 |
| p2q1 | raw_add__steepest_add | 0.25 | 37.71 | 12.74 |
| p2q1 | raw_add__steepest_add | 0.3 | 37.94 | 12.49 |
| p2q1 | raw_add__steepest_add | 0.35 | 38.22 | 12.2 |
| p2q1 | raw_add__steepest_add | 0.4 | 38.55 | 11.87 |
| p2q1 | raw_add__steepest_add | 0.45 | 38.89 | 11.55 |
| p2q1 | raw_add__steepest_add | 0.5 | 39.22 | 11.26 |
| p2q1 | raw_add__steepest_add | 0.55 | 39.54 | 10.99 |
| p2q1 | raw_add__steepest_add | 0.6 | 39.84 | 10.76 |
| p2q1 | raw_add__steepest_add | 0.65 | 40.1 | 10.57 |
| p2q1 | raw_add__steepest_add | 0.7 | 40.34 | 10.42 |
| p2q1 | raw_add__steepest_add | 0.75 | 40.55 | 10.3 |
| p2q1 | raw_add__steepest_add | 0.8 | 40.71 | 10.21 |
| p2q1 | raw_add__steepest_add | 0.85 | 40.84 | 10.14 |
| p2q1 | raw_add__steepest_add | 0.9 | 40.94 | 10.1 |
| p2q1 | raw_add__steepest_add | 0.95 | 41 | 10.07 |
| p2q1 | raw_add__steepest_add | 1 | 41.03 | 10.05 |
| p2q1 | steepest_replace__raw_add | 0 | 46.69 | 4.491 |
| p2q1 | steepest_replace__raw_add | 0.05 | 46.82 | 4.498 |
| p2q1 | steepest_replace__raw_add | 0.1 | 46.86 | 4.522 |
| p2q1 | steepest_replace__raw_add | 0.15 | 46.79 | 4.593 |
| p2q1 | steepest_replace__raw_add | 0.2 | 46.59 | 4.735 |
| p2q1 | steepest_replace__raw_add | 0.25 | 46.24 | 4.969 |
| p2q1 | steepest_replace__raw_add | 0.3 | 45.82 | 5.195 |
| p2q1 | steepest_replace__raw_add | 0.35 | 45.32 | 5.442 |
| p2q1 | steepest_replace__raw_add | 0.4 | 44.69 | 5.857 |
| p2q1 | steepest_replace__raw_add | 0.45 | 43.9 | 6.476 |
| p2q1 | steepest_replace__raw_add | 0.5 | 42.96 | 7.295 |
| p2q1 | steepest_replace__raw_add | 0.55 | 41.91 | 8.278 |
| p2q1 | steepest_replace__raw_add | 0.6 | 40.77 | 9.391 |
| p2q1 | steepest_replace__raw_add | 0.65 | 39.6 | 10.56 |
| p2q1 | steepest_replace__raw_add | 0.7 | 38.46 | 11.73 |
| p2q1 | steepest_replace__raw_add | 0.75 | 37.43 | 12.8 |
| p2q1 | steepest_replace__raw_add | 0.8 | 36.71 | 13.58 |
| p2q1 | steepest_replace__raw_add | 0.85 | 36.44 | 13.9 |
| p2q1 | steepest_replace__raw_add | 0.9 | 36.78 | 13.61 |

## Curvature Aggregate Snapshot

| pq | center_method | direction | curvature_second_diff_mean | curvature_second_diff_std | loss3_q_center_mean |
| --- | --- | --- | --- | --- | --- |
| p1qinf | steepest_add | radial | -0.006063 | 0.004855 | 0.1309 |
| p1qinf | steepest_add | toward_steepest_replace | -0.008034 | 0.008036 | 0.1309 |
| p1qinf | steepest_replace | radial | -0.008992 | 0.01186 | 0.1708 |
| p1qinf | steepest_replace | toward_steepest_add | -0.004035 | 0.007431 | 0.1708 |
| p2q1 | steepest_add | radial | 0.4525 | 1.269 | 41.03 |
| p2q1 | steepest_add | toward_steepest_replace | -1.724 | 2.723 | 41.03 |
| p2q1 | steepest_replace | radial | 1.457 | 0.8981 | 46.69 |
| p2q1 | steepest_replace | toward_steepest_add | -0.2962 | 0.911 | 46.69 |
| p2q2 | steepest_add | radial | -0.2174 | 0.2081 | 3.629 |
| p2q2 | steepest_add | toward_steepest_replace | -0.1865 | 0.08967 | 3.629 |
| p2q2 | steepest_replace | radial | -0.1878 | 0.1373 | 3.634 |
| p2q2 | steepest_replace | toward_steepest_add | -0.1587 | 0.09574 | 3.634 |
| p2qinf | steepest_add | radial | -0.137 | 0.08114 | 0.6106 |
| p2qinf | steepest_add | toward_steepest_replace | -0.5731 | 0.5338 | 0.6106 |
| p2qinf | steepest_replace | radial | 0.2297 | 0.1818 | 0.3249 |
| p2qinf | steepest_replace | toward_steepest_add | 0.04452 | 0.1009 | 0.3249 |

## Interpretation Notes

- This is the first small landscape probe tied to the current core4/PQ deltas rather than the older PGD-only path experiments.
- Ray profiles test whether early/final GPI directions are already strong along the full radius.
- p=2 boundary arcs test whether method final deltas are connected by a high-loss ridge on the L2 boundary.
- 2D slices and finite-difference curvature are pilot-scale; use them as directional evidence, not final statistics.

## Observed Conclusions From This Pilot

Observed from `ray_profile_aggregate.csv`:

- In baseline `p2q2`, replacement/GPI early directions become strong very quickly. At radius `epsilon`, mean `loss3_q` is `1.481` for step 1, `3.494` for step 5, `3.664` for step 10, `3.679` for step 20, and `3.634` for the final direction. This directly supports the claim that the useful direction is mostly found by about 5-10 steps in this baseline setting.
- Endpoint final-direction winners differ by PQ. In `p2q2`, all final directions are almost tied (`3.629` to `3.634`). In `p2q1`, replacement wins (`46.69` vs `41.03` for steepest_add and `37.30` for raw_add). In `p2qinf`, `steepest_add` wins (`0.611`) while replacement is lower (`0.325`). In `p1qinf`, `steepest_replace` is largest on this small sample (`0.171`) but the geometry is spike-prone from the earlier concentration metrics.

Observed from `boundary_arc_aggregate.csv`:

- The p2q2 boundary arcs are high-loss connected. The lowest mean loss along replacement-to-additive arcs is about `98.8%` of the weaker endpoint. That supports the broad-ridge/same-region interpretation for p2q2.
- p2q1 is also mostly connected, with arc minima around `97.7-97.8%` of the weaker endpoint for replacement-to-additive arcs.
- p2qinf is less clean: replacement-to-raw_add has a lower arc dip, around `93.9%` of the weaker endpoint. This is consistent with q=inf making the landscape less like one smooth shared ridge.

Observed from `slice_2d_grid.csv` postprocessing:

- The fitted affine-plane residual score is small for `p2q2` (`~0.028`) and `p2q1` (`~0.025`) on this pilot, but much larger for `p2qinf` (`~0.223`). This supports the idea that q=inf creates a less planar / less smooth local slice in the checked planes.
- `p1qinf` has small scalar plane residual in this tiny plane (`~0.037`) but earlier concentration metrics show spike-like deltas, so this pilot slice alone should not be used to claim p1qinf is benign.

Observed from `curvature_aggregate.csv`:

- Curvature is pilot-scale and noisy with only four samples, but `p2q1` has the largest finite-difference curvature magnitudes among checked centers/directions, while p2q2 is moderate and p2qinf has mixed signs/magnitudes.
- These curvature values should be treated as local diagnostics, not global rankings.

Inference:

- The existing older ray/curvature experiments were not wasted; they already proved local-to-global nonlinear behavior along old PGD paths. This new pilot is the missing bridge to the current core4/PQ story.
- The dominant-ridge story is best supported for `p2q2` and partly `p2q1`. It should be qualified for `p2qinf` and `p1qinf`.

