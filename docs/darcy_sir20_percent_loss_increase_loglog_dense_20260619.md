# Darcy/SIR20 Percent Loss Increase Log-Log Curves

Observed from the original completed Darcy/SIR20 budget sweep plus a new train/test extra-budget attack sweep run on 2026-06-19. The extra sweep uses the same final checkpoint manifest, `20` attack steps, and `20` samples per train/test split, but it does not rerun the full 50-dataset generalization sweep.

Definition used for the plotted y-axis:

`percent loss increase = 100 * (adv_loss_mean / clean_loss_mean - 1)`

The best-envelope plot takes the largest plotted percent increase among the available split summaries for the same model and epsilon. The generalization-only plot uses only the original full `generalization` split, so it remains the full-data sparse-grid reference rather than a small train/test-only dense overlay.

Artifacts:
- Source summary CSVs:
  - `outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/budget_sweep_loss_increase_summary.csv`
  - `outputs/darcy_sir20_dense_extra_budgets_train_test_20260619/data/budget_sweep_loss_increase_summary.csv`
- Best-envelope CSV: `outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/percent_loss_increase_loglog/dense_percent_loss_increase_best_envelope.csv`
- Generalization-only CSV: `outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/data/percent_loss_increase_loglog/dense_percent_loss_increase_generalization.csv`
- Best-envelope PNG: `outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/figures/percent_loss_increase_loglog/dense_epsilon_vs_percent_loss_increase_best_envelope_loglog.png`
- Best-envelope PDF: `outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/figures/percent_loss_increase_loglog/dense_epsilon_vs_percent_loss_increase_best_envelope_loglog.pdf`
- Generalization-only PNG: `outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/figures/percent_loss_increase_loglog/dense_epsilon_vs_percent_loss_increase_generalization_loglog.png`
- Generalization-only PDF: `outputs/darcy_sir20_timematched_full_serial_double_budget_full_delta_budget_records_20260617/figures/percent_loss_increase_loglog/dense_epsilon_vs_percent_loss_increase_generalization_loglog.pdf`

Best-envelope maxima by method:

| method | max percent loss increase | epsilon at max | source split |
|---|---:|---:|---|
| baseline | 14458 | 0.0625 | train |
| loss1 | 59915.7 | 0.075 | train |
| loss2 | 23025.7 | 0.075 | train |
| loss3 | 2610.25 | 0.075 | train |
| Physics Loss | 88936 | 0.075 | train |
| random clean | 4266.15 | 0.0625 | test |
| random solver | 32007.9 | 0.075 | train |

Generalization-only maxima by method:

| method | max percent loss increase | epsilon at max |
|---|---:|---:|
| baseline | 654.975 | 0.05 |
| loss1 | 626.798 | 0.0375 |
| loss2 | 706.353 | 0.075 |
| loss3 | 254.74 | 0.0125 |
| Physics Loss | 627.77 | 0.05 |
| random clean | 787.55 | 0.05 |
| random solver | 640.428 | 0.0375 |
