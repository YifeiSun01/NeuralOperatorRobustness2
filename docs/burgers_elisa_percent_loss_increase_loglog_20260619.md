# Burgers Elisa Percent Loss Increase Log-Log Curves

Generated: 2026-06-19T13:32:08+00:00

Observed from a fresh GPU P2Q2 RMS-L2 attack sweep on the restored Burgers final checkpoints.

Definition used for the plotted y-axis:

`percent loss increase = 100 * (adv_loss_mean / clean_loss_mean - 1)`

Artifacts:
- Summary CSV: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/percent_loss_increase_loglog/budget_sweep_loss_increase_summary.csv`
- Sample detail CSV: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/percent_loss_increase_loglog/budget_sweep_loss_increase_samples.csv`
- Best-envelope CSV: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/percent_loss_increase_loglog/epsilon_vs_percent_loss_increase_best_envelope.csv`
- Generalization-only CSV: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/percent_loss_increase_loglog/epsilon_vs_percent_loss_increase_generalization.csv`
- Best-envelope PNG: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/figures/percent_loss_increase_loglog/epsilon_vs_percent_loss_increase_best_envelope_loglog.png`
- Generalization-only PNG: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/figures/percent_loss_increase_loglog/epsilon_vs_percent_loss_increase_generalization_loglog.png`

Key settings:
- Models: `baseline, loss1, loss2, loss3, random_clean_y, random_solver_y`
- Budgets: `0.005, 0.0075, 0.01, 0.0125, 0.0175, 0.02, 0.025, 0.03, 0.0375, 0.04375, 0.05, 0.0625, 0.075, 0.1, 0.12`
- Attack steps: `10`
- Sample count: `12` with split counts `{'train': 4, 'test': 4, 'generalization': 4}`
- GPU: `Tesla V100-SXM2-32GB`

Best-envelope maxima by method:

| method_display | max_percent_loss_increase | epsilon_at_max | split_at_max |
| --- | --- | --- | --- |
| Burgers baseline | 7093.35 | 0.12 | train |
| Burgers loss1 | 158282 | 0.12 | test |
| Burgers loss2 | 55060.5 | 0.12 | train |
| Burgers loss3 | 5982.77 | 0.12 | train |
| Burgers random clean | 282202 | 0.12 | train |
| Burgers random solver | 333524 | 0.12 | train |

Generalization-only maxima by method:

| method_display | max_percent_loss_increase | epsilon_at_max | split_at_max |
| --- | --- | --- | --- |
| Burgers baseline | 1180.28 | 0.12 | generalization |
| Burgers loss1 | 997.717 | 0.12 | generalization |
| Burgers loss2 | 994.637 | 0.12 | generalization |
| Burgers loss3 | 2177.09 | 0.12 | generalization |
| Burgers random clean | 217.257 | 0.12 | generalization |
| Burgers random solver | 1440.72 | 0.12 | generalization |

Notes:
- The best-envelope plot takes the largest split-level percent increase among train/test/generalization for each model and epsilon.
- Several best-envelope points are amplified by very small clean-loss denominators on train/test splits, especially for random-field checkpoints; read the generalization-only plot alongside the envelope plot.
- Even on the generalization-only plot, percent loss increase should not be read as a standalone cross-model robustness ranking for Burgers. The companion loss-scale sanity plots show that absolute loss increase and final adversarial loss rank `loss3` best on this compact generalization sweep: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/figures/loss_scale_sanity/`.
- This run is a compact real sweep, not the full 52-dataset x 50-sample suite.
