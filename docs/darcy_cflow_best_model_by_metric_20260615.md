# Darcy cflow best model by metric

Date: 2026-06-15.

## Important Correction

The attack and SVD/Jacobian rows in this note come from the organized-release
smoke diagnostic tables, not from final time-matched robustness runs. The source
attack table uses `darcy_sir20_smoke_*_1ep` checkpoints and `attack_steps=1`.
Therefore, the attack/SVD "best model" rows below must not be used as final
model rankings.

For final clean generalization, `loss3` beats `physics` on all 50 generalization
datasets for Relative L2, RMSE, and data MSE. See
`docs/darcy_cflow_smoke_attack_metric_correction_20260615.md`.

## Evidence

Observed from:

- `outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_clean_52dataset_metric_long_ranked.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_attack_metric_long_ranked.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_svd_jacobian_metric_long_ranked.csv`

Generated tables:

- `outputs/darcy_cflow_timematched_organized_release_20260614/data/best_model_by_metric_20260615/best_model_by_metric_all_scopes.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/data/best_model_by_metric_20260615/focused_best_model_by_metric.csv`
- `outputs/darcy_cflow_timematched_organized_release_20260614/reports/BEST_MODEL_BY_METRIC_20260615.md`

## Clean Generalization

For the 50 generalization datasets, `loss3` is the best model on the main clean
quality metrics.

| metric | direction | best by mean | mean | first-place count |
|---|---|---:|---:|---:|
| Relative L2 | lower is better | loss3 | 0.056154 | 47/50 |
| RMSE | lower is better | loss3 | 0.0005946 | 47/50 |
| MAE | lower is better | loss3 | 0.0004694 | 47/50 |
| data MSE | lower is better | loss3 | 3.820e-07 | 47/50 |
| accuracy score | higher is better | loss3 | 94.698 | 47/50 |
| invalid value fraction | lower is better | tied at 0 | 0 | all tied |

## Surrogate Attack Generalization

The available attack table is partial smoke coverage, not the complete requested
50-sample-per-dataset attack sweep.

| metric | direction | best by mean | mean | first-place count |
|---|---|---:|---:|---:|
| clean_loss | lower is better | physics | 1.358e-07 | 93/100 |
| adv_loss | lower is better | physics | 1.722e-07 | 91/100 |
| loss_increase | lower is better | physics | 3.644e-08 | 72/100 |
| relative_increase | lower is better | loss3 | 0.2268 | 44/100 |
| delta_l2_rms | lower is better | physics | 0.8611 | 57/100 |
| delta_linf | lower is better | physics | 5.919 | 76/100 |

## Visible SVD/Jacobian Diagnostics

These rows are mechanism diagnostics from the visible small SVD/Jacobian table.
They are not a complete generalization conclusion. "Best" here means best under
the chosen diagnostic direction.

| metric | direction | best by mean |
|---|---|---:|
| clean_loss | lower is better | baseline |
| attack_loss_increase | lower is better | random_solver |
| attack_relative_increase | lower is better | random_clean |
| error_l2_norm | lower is better | baseline |
| jt_error_l2_norm | lower is better | baseline |
| sigma_input_right | lower local sensitivity | loss2 |
| cos_singular_jt_error | higher alignment | physics |
| angle_singular_jt_error_deg | lower angle | loss3 |
| corr_singular_jt_error | higher correlation | baseline |
| cos_singular_attack_delta | higher alignment | random_solver |
| angle_singular_attack_delta_deg | lower angle | random_solver |
| corr_singular_attack_delta | higher correlation | random_solver |
| cos_jt_error_attack_delta | higher alignment | loss3 |
| angle_jt_error_attack_delta_deg | lower angle | loss3 |
| corr_jt_error_attack_delta | higher correlation | physics |

## Interpretation

- If the question is clean predictive/generalization quality, `loss3` is the
  clear best model.
- If the question is the current partial-smoke surrogate attack table, `physics`
  is best on absolute post-attack loss and absolute loss increase, while `loss3`
  is best on relative increase.
- If the question is SVD/Jacobian diagnostics, there is no single overall winner:
  the best model depends on the diagnostic quantity, and the visible sample count
  is too small to rank global model quality.
