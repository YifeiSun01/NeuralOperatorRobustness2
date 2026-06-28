# Burgers Round03 Stitched Extended Plots - 2026-06-06

Status: corrected and complete for the requested post-training plot refresh.

This plotting correction presents the later training epochs as one continuous training history, not as separate displayed runs. The user-facing plots now show loss1 as epochs 0..3000 and loss2/loss3 as epochs 0..1000.

Observed issue:
- Earlier plotting created separate visualization directories whose names included the resume run names. That was the wrong presentation for the requested figures.
- The requested figure semantics are single stitched histories: extend the original loss1/loss2/loss3 curves, do not show a separate displayed run.

Fix applied:
- Added `tools/plot_burgers_round03_stitched_single_run_visualizations.py` to read a base run plus its later run and redraw the original single-run plot directory as one stitched curve.
- Updated `tools/plot_burgers_round03_loss123_continuation_dense_comparison.py` display text so comparison titles say extended training, and the merged CSV uses `run_role` values `original` and `extended`.
- Overwrote the original single-run plot directories:
  - `visualizations/burgers_loss3_selective_round03_loss1_3000ep_long_20260605_plots` now has max eval/attack epoch `3000`.
  - `visualizations/burgers_loss3_selective_round03_loss2_1000ep_long_20260605_plots` now has max eval/attack epoch `1000`.
  - `visualizations/burgers_loss3_selective_round03_loss3_1000ep_long_20260605_plots` now has max eval/attack epoch `1000`.
- Overwrote the old comparison directories:
  - `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605`
  - `visualizations/burgers_loss3_selective_round03_long_training_comparison_20260605`
- Removed the incorrect separate visualization directories under `visualizations/` whose names contained `continue`/`continuation`.

Exact source runs:
- loss1 base: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605`
- loss1 later run: `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue1000to3000_20260605`
- loss2 base: `adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605`
- loss2 later run: `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue500to1000_20260605`
- loss3 base: `adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605`
- loss3 later run: `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue500to1000_20260605`

Observed verification:
- Single-run manifests report max epochs: loss1 `3000`, loss2 `1000`, loss3 `1000`.
- Parsed `visualizations/burgers_loss3_selective_round03_long_training_comparison_dense_20260605/round03_dense_epoch_metrics_every1.csv`: max epoch by loss is loss1 `3000`, loss2 `1000`, loss3 `1000`.
- The same dense CSV has `run_role` values `original` and `extended`; it no longer labels the plotted second segment as a separate continuation role.
- PIL spot checks found nonblank PNGs for old comparison outputs and the three original single-run RMSE heatmaps.

Final generated-generalization metrics shown by the refreshed plots:
- loss1 final epoch3000 generated RMSE `0.0343239`, relative L2 `0.0615363`; best generated RMSE `0.032945` at epoch2673.
- loss2 final epoch1000 generated RMSE `0.0365082`, relative L2 `0.0654749`; best generated RMSE `0.0355695` at epoch948.
- loss3 final epoch1000 generated RMSE `0.0220984`, relative L2 `0.0396222`; best generated RMSE `0.0193611` at epoch893.

Inference from observed evidence:
- The user-facing figures now implement the intended view: original curves extended forward, not separate displayed runs.
- loss3 remains the strongest on round03 generated-OOD generalization after the later training epochs.

Remaining work:
- Keep generated PNG/CSV artifacts out of Git unless explicitly requested.
- Same-wall/time-matched SVD remains cancelled and was not rerun.
