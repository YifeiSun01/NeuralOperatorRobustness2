# Darcy cflow smoke attack metric correction

Date: 2026-06-15.

Terminology: `1ep` = one epoch. Preflight/smoke diagnostics are small
pipeline-validation artifacts, not final model results.

## Correction

The organized-release `cflow_attack_metric_long_ranked.csv` and
`cflow_svd_jacobian_metric_long_ranked.csv` tables must not be interpreted as
final time-matched model robustness results.

Observed from
`outputs/darcy_cflow_timematched_organized_release_20260614/data/source_tables/robustness_attack_52datasets_samples.csv`:

- `attack_steps` is `1`.
- The six trained-method checkpoints are from
  `outputs/darcy_sir20_timematched_full_20260614_smoke_initial/...`.
- The method checkpoints are one-epoch preflight checkpoints with path pattern
  `darcy_sir20_smoke_*_1ep/.../darcy_epoch001_step000004.pt` (`1ep` means one
  epoch).

Therefore, the attack `clean_loss`, `adv_loss`, `loss_increase`, and
`relative_increase` tables are preflight diagnostics, not final-model results.

## Why This Matters

The user's final generalization plots and the final clean metric table use the
long/time-matched models. In those final clean metrics, `loss3` is clearly better
than `physics`.

Observed from
`outputs/darcy_cflow_timematched_organized_release_20260614/data/cflow_clean_52dataset_metric_long_ranked.csv`,
on the 50 final generalization datasets:

| metric | physics better count | loss3 better count | physics mean | loss3 mean |
|---|---:|---:|---:|---:|
| Relative L2 | 0/50 | 50/50 | 0.088293 | 0.056154 |
| RMSE | 0/50 | 50/50 | 0.0009390 | 0.0005946 |
| data MSE | 0/50 | 50/50 | 9.651e-07 | 3.820e-07 |
| accuracy score | 0/50 | 50/50 | 91.926 | 94.698 |

So the apparent statement that "physics clean loss is smaller than loss3" is
not true for the final generalization curves. It came from the smoke attack
source table, which used one-epoch preflight checkpoints and a different attack
objective table.

## Correct Interpretation

- For final clean predictive/generalization performance: `loss3` is better than
  `physics` across the 50 generalization datasets.
- For the current organized-release attack/SVD tables: treat them only as
  smoke diagnostics and coverage placeholders.
- A valid final-model robustness comparison requires recomputing the attack and
  SVD/Jacobian tables using the final time-matched checkpoints, not the
  `smoke_initial` checkpoints.
