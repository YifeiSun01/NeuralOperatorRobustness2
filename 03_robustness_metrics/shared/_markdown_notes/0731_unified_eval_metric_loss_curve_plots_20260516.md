# Unified Evaluation-Metric Loss Curve Plots - 2026-05-16

Status: completed from existing saved trajectory data. No attack was rerun.

## Purpose

The original batch loss-curve plots show each attack trajectory's own optimized
objective on the y-axis. That is useful for checking whether the optimizer
improved the objective it was assigned, but it does not answer questions like:

- If every attack trajectory is evaluated by `loss3_original`, which one
  actually creates the largest solver-relative endpoint error?
- If every trajectory is evaluated by `loss3_increment_ratio`, which one has
  the best ratio behavior over the optimization path?

The saved batch attack outputs already record all nine evaluation metrics at
every step:

- `loss1_original`, `loss1_increment_ratio`, `loss1_regularized`
- `loss2_original`, `loss2_increment_ratio`, `loss2_regularized`
- `loss3_original`, `loss3_increment_ratio`, `loss3_regularized`

Therefore the curves can be redrawn from the existing `loss_stats.csv` and
`loss_values.npz` files without running any solver/model attack again.

## Data Source

The complete 27-run FNO `nu=0.001` historical result directory was restored
from R2:

`results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/`

This setting contains:

- 3 optimized losses: `loss1`, `loss2`, `loss3`
- 3 objective variants: `original`, `increment_ratio`, `regularized`
- 3 optimization methods: `pgd`, `lp_steepest_pgd`, `generalized_power`

That gives 27 attack trajectories.

## New Plotting Script

`tools/plot_batch_eval_metric_matrix_curves.py`

The script fixes the y-axis to one recorded evaluation key and then plots all
27 trajectories in a 3-by-3 matrix:

- rows: optimized loss (`loss1`, `loss2`, `loss3`)
- columns: optimized objective variant (`original`, `increment_ratio`,
  `regularized`)
- curves inside each panel: PGD, LP-steepest PGD, generalized power iteration

Example command:

```bash
adv_robust/bin/python tools/plot_batch_eval_metric_matrix_curves.py \
  --root results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary \
  --eval-keys loss3_original loss3_increment_ratio loss3_regularized \
  --dataset-indices 0
```

To generate all nine shared-y-axis evaluation metrics:

```bash
adv_robust/bin/python tools/plot_batch_eval_metric_matrix_curves.py \
  --root results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary \
  --eval-keys all \
  --dataset-indices 0
```

## Generated Figures

Batch mean/std plots:

`results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_curves/png/`

Single-index plots for dataset index `0`:

`results/three_loss_batch100_full_loss3_delta_rerun_20260514_fno_eps8_alpha0p3_final_boundary/figures/eval_metric_curves/index_png/`

The most directly relevant files for the loss3 story are:

- `loss3_original_all_27_mean_std.png`
- `loss3_increment_ratio_all_27_mean_std.png`
- `loss3_regularized_all_27_mean_std.png`
- `loss3_original_all_27_index0.png`
- `loss3_increment_ratio_all_27_index0.png`
- `loss3_regularized_all_27_index0.png`

## Interpretation Note

These plots separate optimization target from evaluation metric. A trajectory
may optimize `loss1_increment_ratio` or `loss2_regularized`, but the new plots
can still evaluate that same trajectory by `loss3_original` at every step.
This is the correct view when the scientific question is whether a surrogate
objective actually creates large solver-relative error.
