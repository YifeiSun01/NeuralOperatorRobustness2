# Loss3 Core All-PQ Four Figures - 2026-05-18

Status: completed post-processing visualization. No optimizer experiment was run
for this step.

Output folder:

`/workspace/NeuralOperatorRobustness2/forensics/loss3_core_all_pq_four_figures_20260518`

The folder intentionally contains only four PNG files:

1. `01_actual_loss_all_pq_core4_mean_std.png`
   - 3x3 grid over all P/Q pairs.
   - Four core methods: `raw_add`, `raw_replace`, `steepest_add`,
     `steepest_replace`.
   - Line is mean actual `loss3_q`; shading is `+/- 1` sample std over 100
     samples.

2. `02_boundary_ratio_all_pq_core4_mean_std.png`
   - 3x3 grid over all P/Q pairs.
   - Shows `||delta||_p / epsilon` with mean line and `+/- 1` std shading.
   - This is a constraint-use diagnostic, not a smoothness metric.

3. `03_final_metrics_all_pq_core4_heatmaps.png`
   - Final-step heatmaps for actual loss, high-frequency energy ratio,
     first-derivative L2, and total variation.
   - Rows are P/Q pairs; columns are the four core methods.
   - For high-frequency ratio, first-derivative L2, and total variation, smaller
     means smoother / less high-frequency.

4. `04_sample040_final_delta_all_pq_core4.png`
   - Sample `40` final delta shapes across all P/Q pairs and the four core
     methods.
   - Each P/Q row shares y-limits across methods, so method shapes are comparable
     within a fixed P/Q pair.

Source data:

- Original P/Q runs:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517`
- Raw-replace backfill runs:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`

Observed evidence:

- The original P/Q runs provide `raw_add`, `steepest_add`, and
  `steepest_replace` for all P/Q pairs.
- `p=2,q=2` already contains `raw_replace` in the original run.
- The raw-replace backfill provides `raw_replace` for the other eight P/Q pairs.

Inference:

The four figures are strict four-method plots for every P/Q pair because
`raw_replace` is now sourced from actual backfill runs rather than inferred from
`steepest_replace`.
