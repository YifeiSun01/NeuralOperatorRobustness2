# Burgers Elisa Loss-Scale Sanity Plots

Generated: 2026-06-19T13:45:40+00:00

These plots reuse the completed Burgers Elisa attack sweep CSV. No attack was rerun.

Purpose: check whether the percent-loss plot is ranking true robustness or mostly reflecting clean-loss denominators.

Artifacts:
- Source summary CSV: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/percent_loss_increase_loglog/budget_sweep_loss_increase_summary.csv`
- Generalization sanity CSV: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/loss_scale_sanity/generalization_absolute_loss_sanity.csv`
- Mean-rank CSV: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/data/loss_scale_sanity/generalization_mean_ranks_by_metric.csv`
- Absolute loss increase PNG: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/figures/loss_scale_sanity/epsilon_vs_absolute_loss_increase_generalization_loglog.png`
- Final adversarial loss PNG: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/figures/loss_scale_sanity/epsilon_vs_final_adversarial_loss_generalization_loglog.png`
- Clean loss PNG: `outputs/burgers_elisa_percent_loss_increase_loglog_20260619/figures/loss_scale_sanity/epsilon_vs_clean_loss_generalization_loglog.png`

Generalization rows at epsilon `0.12` sorted by absolute loss increase:

| method | clean_loss_mean | adv_loss_mean | abs_loss_increase | percent_loss_increase |
| --- | --- | --- | --- | --- |
| loss3 | 0.000118892 | 0.00270727 | 0.00258837 | 2177.09 |
| loss2 | 0.000525876 | 0.00575644 | 0.00523056 | 994.637 |
| loss1 | 0.000533297 | 0.00585409 | 0.0053208 | 997.717 |
| random_solver_y | 0.000401236 | 0.00618193 | 0.0057807 | 1440.72 |
| baseline | 0.000542865 | 0.0069502 | 0.00640733 | 1180.28 |
| random_clean_y | 0.0119638 | 0.037956 | 0.0259922 | 217.257 |

Mean ranks over all epsilon values; lower is better:

| metric | method | mean_rank_lower_better |
| --- | --- | --- |
| abs_loss_increase | loss3 | 1 |
| abs_loss_increase | loss1 | 3 |
| abs_loss_increase | random_solver_y | 3.06667 |
| abs_loss_increase | loss2 | 3.26667 |
| abs_loss_increase | baseline | 4.66667 |
| abs_loss_increase | random_clean_y | 6 |
| adv_loss_mean | loss3 | 1 |
| adv_loss_mean | random_solver_y | 2.66667 |
| adv_loss_mean | loss2 | 2.73333 |
| adv_loss_mean | loss1 | 3.6 |
| adv_loss_mean | baseline | 5 |
| adv_loss_mean | random_clean_y | 6 |
| percent_loss_increase | random_clean_y | 1 |
| percent_loss_increase | loss1 | 2.26667 |
| percent_loss_increase | loss2 | 3.2 |
| percent_loss_increase | baseline | 3.53333 |
| percent_loss_increase | random_solver_y | 5 |
| percent_loss_increase | loss3 | 6 |

Interpretation:
- The percent-loss ranking is strongly affected by the clean-loss denominator.
- Absolute loss increase and final adversarial loss both rank `loss3` best on this compact generalization sweep.
- Therefore the existing percent-loss figure is arithmetically consistent with the CSV, but it is not a good standalone cross-model robustness ranking for Burgers.
