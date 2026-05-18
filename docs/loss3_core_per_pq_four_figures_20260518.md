# Loss3 Core Per-PQ Four-Figure Folders - 2026-05-18

Status: completed post-processing visualization. No optimizer experiment was run
for this step.

Output root:

`/workspace/NeuralOperatorRobustness2/forensics/loss3_core_per_pq_four_figures_20260518`

Structure:

- The output root contains 9 P/Q subfolders.
- Each P/Q subfolder contains exactly 4 PNG files and no data files.
- P/Q subfolders: `p1_q1`, `p1_q2`, `p1_qinf`, `p2_q1`, `p2_q2`, `p2_qinf`,
  `pinf_q1`, `pinf_q2`, and `pinf_qinf`.

Each P/Q folder contains:

1. `01_actual_loss_core4_mean_std.png`
   - Four core methods: `raw_add`, `raw_replace`, `steepest_add`,
     `steepest_replace`.
   - Actual `loss3_q` mean curve with `+/- 1` sample std shading over 100
     samples.

2. `02_delta_quality_core4_metrics.png`
   - Wide/flat layout matching the revised `p=2,q=2` style.
   - Boundary ratio, high-frequency energy ratio, first-derivative L2, and total
     variation.
   - Left panels: mean curves with `+/- 1` sample std shading.
   - Right panels: final-step mean `+/-` sample std summaries.

3. `03_final_delta_core4_selected_samples.png`
   - Final delta shapes for selected samples `0`, `7`, `40`, and `47`.
   - Each sample row shares y-limits across methods so shape differences are
     comparable within the same sample.

4. `04_final_smoothness_core4_mean_std.png`
   - Final-step smoothness summary bars: high-frequency energy ratio, spectral
     centroid, first-derivative L2, and total variation.
   - Mean `+/-` sample std over 100 samples.

Source data:

- Original P/Q runs:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_20260517`
- Raw-replace backfill runs:
  `/workspace/NeuralOperatorRobustness2/forensics/loss3_optimizer_direction_proposal_ablation_raw_replace_backfill_20260518`

Observed file check:

- `9` P/Q subfolders.
- `4` PNG files per P/Q subfolder.
- `36` total PNG files.

Inference:

These are strict four-core-method figures for every P/Q pair because
`raw_replace` is sourced from actual backfill runs where it was missing in the
original queue.
