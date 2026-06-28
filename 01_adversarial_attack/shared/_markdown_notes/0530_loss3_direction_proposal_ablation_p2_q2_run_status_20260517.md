# Loss3 Direction-Proposal Ablation P2/Q2 Run Status - 2026-05-17

Status checked at `2026-05-17 22:24:07 UTC`.

## Process Status

Observed running process:

```text
PID 37171
elapsed 13:43
CPU 98.0%
RSS 2280672 KiB
```

Observed GPU status:

```text
GPU utilization: 97%
GPU memory used: 10744 MiB / 32768 MiB
```

Output root:

```text
/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2
```

Log path:

```text
/workspace/NeuralOperatorRobustness2/logs/loss3_direction_proposal_p2_q2_batch100.log
```

## Progress

Observed from the log and completed `summary.json` files:

- Completed methods: 7 / 11.
- Current method: `power_add__pure_jvp_vjp`.
- Remaining after current method: `power_replace__pure_jvp_vjp`,
  `power_add__affine_jvp_vjp`, `power_replace__affine_jvp_vjp`.

Completed method runtimes:

| Method | Runtime seconds |
|---|---:|
| `raw_add` | 106.171 |
| `unit_raw_add` | 105.574 |
| `raw_replace` | 105.539 |
| `steepest_add` | 105.140 |
| `steepest_replace` | 105.701 |
| `power_add__objective_gradient` | 105.258 |
| `power_replace__objective_gradient` | 105.374 |

Completed runtime sum: `738.757` seconds.  Completed-method mean: `105.537`
seconds.

## Estimate

Inference from completed method times plus the earlier tiny smoke:

- Ordinary/objective-gradient methods are taking about `105.5` seconds each.
- q-aware JVP/VJP methods should be planned at about `180-210` seconds each.
- The current q-aware method started around `22:22:58 UTC` and had been running
  about `69` seconds at the status check.

Estimated remaining time from `22:24:07 UTC`:

```text
optimizer remaining: about 11-14 minutes
final aggregation/static plots: about 2-4 minutes
reasonable total remaining: about 13-18 minutes
```

Estimated finish window:

```text
2026-05-17 22:37-22:42 UTC
```

This estimate assumes no method failure and no unexpected slowdown during final
static figure generation.  GIF rendering is not enabled for this run.

## Completion Update - 2026-05-17 22:38:08 UTC

Observed from `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/manifest.json`: status is `completed`, method count is
`11`, and the run finished at `2026-05-17 22:38:08 UTC`.

Observed from method `summary.json` files:

- Method runtime sum: `1537.89` seconds.
- Mean method runtime: `139.81` seconds.
- Median method runtime: `105.70` seconds.
- Ordinary/objective-gradient methods: about `105-106` seconds each.
- q-aware JVP/VJP methods: about `199-200` seconds each.

Observed output completeness:

- `per_step_metrics.csv`: `1111` data rows.
- `per_sample_step_metrics.csv`: `111100` data rows.
- `pq_geometry_summary.csv`: `11` data rows.
- Figures under `figures/`: `56` files.
- GIF files: `0`, as intended for this no-GIF run.

Next work: analyze the completed p=2/q=2 results and generate the scientific
comparison summary before launching additional p/q pairs.

## Visualization Update - 2026-05-17

Observed/generated visualizations for the completed `p=2,q=2` run:

- Loss progression and boundary ratio:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/loss_progression_by_method/loss3_q_and_boundary_ratio_methods_p2_q2.png`.
- Existing mean curves:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/curves/loss3_q_mean_vs_step.png`,
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/curves/boundary_ratio_mean_vs_step.png`, and roughness curves
  under `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/curves/`.
- Per-step delta grids for dataset indices `0`, `7`, `40`, and `47`:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/delta_grids/`.
- Delta time-space heatmaps by method/sample:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/delta_heatmaps/`.
- Newly generated final-input overlays showing clean `x`, `x + final_delta`,
  and final `delta` for dataset indices `0`, `7`, `40`, and `47`:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/final_input_delta_overlays/`.
- Newly generated final-delta overlays across methods:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/final_delta_overlaid/`.

Important scope note: only `p=2,q=2` has been run so far.  No `(p=inf,q=2)`,
`(p=2,q=inf)`, or `(p=inf,q=inf)` figures exist yet.  Requested trajectory
index `115` was outside the batch `0..99`, so trajectory visualizations exist
for `0`, `7`, `40`, and `47`; index `115` requires a separate targeted run or
a batch that includes that index.

After this update, the run has `65` figure files and `0` GIF files.

## Actual-Loss-Only Visualization Update - 2026-05-17

Observed from existing metrics file:
`/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/per_step_metrics.csv`.

Generated a new actual-loss-only comparison figure:
`/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517/fno_nu0p001_eps8_alpha0p3_batch100_steps100_p2_q2/figures/true_loss_progression/actual_loss3_q_mean_methods_p2_q2_clean_20260517.png`.

This new figure uses the observed `loss3_q_mean` values along the actual
optimizer iterates `delta_k`. It does not use boundary-normalized loss or
project the direction to the boundary for plotting.

Preservation note: existing figures were not deleted or overwritten. New
visualizations should continue to be written to new directories or new file
names unless the user explicitly asks to replace an existing plot.

