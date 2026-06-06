# Burgers Round03 Dense Plot Outputs

Date: 2026-06-05 UTC.

## What Was Preserved

The existing sparse comparison plots were left in place and were not overwritten:

- `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605/`

No extra copy of the sparse plots is kept.

## New Dense Outputs

New every-epoch and every-5-epoch comparison plots were generated under:

- `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/`

The dense source tables are:

- `round03_dense_epoch_metrics_every1.csv`: 8013 lines including header.
- `round03_dense_epoch_metrics_every5.csv`: 1613 lines including header.

The dense wall-clock x-axis is reconstructed from `train_steps.csv` plus `evaluation_passes.csv`, with checkpoint epochs pinned to `checkpoints.csv` `wall_elapsed_seconds`.

## Plot Sets

For both RMSE and relative L2, the following dense plot families exist:

- same-epoch, every 1 epoch, train/test/generalization panels.
- wall-clock, every 1 epoch, train/test/generalization panels.
- same-epoch, every 5 epochs, train/test/generalization panels.
- wall-clock, every 5 epochs, train/test/generalization panels.

The train/test/generalization versions contain the test/generalization view, so separate two-panel test/generalization duplicates were removed.

## Verification

Observed verification after cleanup:

- The dense directory contains 8 PNG files: RMSE and relative L2, same-epoch and wall-clock, every 1 and every 5 epochs, all with train/test/generalization panels.
- The two dense CSV files and dense manifest are present.
- The extra sparse-copy directory was removed.

## Source Script

- `tools/plot_burgers_round03_dense_training_comparison.py`
