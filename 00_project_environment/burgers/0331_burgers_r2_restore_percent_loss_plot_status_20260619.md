# Burgers R2 Restore And Percent-Loss Plot Status - 2026-06-19

## Request

Restore the Burgers models from the R2 bucket for the Elisa/Burgers follow-up
and generate the same epsilon-vs-percent-loss-increase log-log plot used for
the Darcy/SIR20 analysis.

## Status

Completed on this instance after R2/S3 access was provided by the user. The
local `R2` rclone remote was configured without writing credentials into this
report, the target final checkpoints were restored, and a fresh GPU attack
sweep was run for the requested log-log curves.

## Restored Checkpoints

Verified local final checkpoints:

- `adversarial_training_runs/burgers_wideparam_loss1_8000ep_retrain_20260611/burgers/checkpoints/burgers_epoch8000_step008000.pt`
- `adversarial_training_runs/burgers_wideparam_loss2_2000ep_retrain_20260611/burgers/checkpoints/burgers_epoch2000_step002000.pt`
- `adversarial_training_runs/burgers_wideparam_loss3_1000ep_retrain_20260611/burgers/checkpoints/burgers_epoch1000_step001000.pt`
- `adversarial_training_runs/burgers_wideparam_random_field_clean_y_8000ep_continue_20260613/burgers/checkpoints/burgers_epoch8000_step008000.pt`
- `adversarial_training_runs/burgers_wideparam_random_field_solver_y_7860ep_continue_20260613/burgers/checkpoints/burgers_epoch7860_step007860.pt`

The baseline checkpoint used for comparison was already local:

- `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`

Restore helper:

- `tools/restore_burgers_elisa_models_from_r2_20260619.sh`

## Plot Artifacts

Fresh attack/plot runner:

- `tools/run_burgers_elisa_percent_loss_increase_loglog_20260619.py`

Output root:

- `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/`

Figures:

- Best-envelope PNG: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/figures/percent_loss_increase_loglog/epsilon_vs_percent_loss_increase_best_envelope_loglog.png`
- Generalization-only PNG: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/figures/percent_loss_increase_loglog/epsilon_vs_percent_loss_increase_generalization_loglog.png`

Data and reports:

- Summary CSV: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/percent_loss_increase_loglog/budget_sweep_loss_increase_summary.csv`
- Sample detail CSV: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/percent_loss_increase_loglog/budget_sweep_loss_increase_samples.csv`
- Best-envelope curve CSV: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/percent_loss_increase_loglog/epsilon_vs_percent_loss_increase_best_envelope.csv`
- Generalization-only curve CSV: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/percent_loss_increase_loglog/epsilon_vs_percent_loss_increase_generalization.csv`
- Bundle report: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/reports/percent_loss_increase_loglog.md`
- Dedicated plot note: `docs/burgers_elisa_percent_loss_increase_loglog_20260619.md`

## Attack And Plot Settings

- Models: `baseline`, `loss1`, `loss2`, `loss3`, `random_clean_y`,
  `random_solver_y`.
- Budgets: `0.005, 0.0075, 0.01, 0.0125, 0.0175, 0.02, 0.025, 0.03, 0.0375,
  0.04375, 0.05, 0.0625, 0.075, 0.1, 0.12`.
- Attack: P2Q2 RMS-L2 solver-loss attack.
- Attack steps: `10`.
- Plotted y-axis:
  `percent loss increase = 100 * (adv_loss_mean / clean_loss_mean - 1)`.
- Sample count: compact real sweep with `12` samples total (`4` train, `4`
  test, `4` generalization), not the full 52-dataset x 50-sample suite.
- GPU: `Tesla V100-SXM2-32GB`.

## Observed Result

Best-envelope maxima by method:

| method | max percent loss increase | epsilon | split |
| --- | ---: | ---: | --- |
| baseline | 7093.35 | 0.12 | train |
| loss1 | 158282 | 0.12 | test |
| loss2 | 55060.5 | 0.12 | train |
| loss3 | 5982.77 | 0.12 | train |
| random clean | 282202 | 0.12 | train |
| random solver | 333524 | 0.12 | train |

Generalization-only maxima by method:

| method | max percent loss increase | epsilon | split |
| --- | ---: | ---: | --- |
| baseline | 1180.28 | 0.12 | generalization |
| loss1 | 997.717 | 0.12 | generalization |
| loss2 | 994.637 | 0.12 | generalization |
| loss3 | 2177.09 | 0.12 | generalization |
| random clean | 217.257 | 0.12 | generalization |
| random solver | 1440.72 | 0.12 | generalization |

The best-envelope plot is intentionally a worst-split envelope over train,
test, and generalization. Several curves have very small clean-loss
denominators on train/test, so the generalization-only plot should be read
alongside the envelope plot.
