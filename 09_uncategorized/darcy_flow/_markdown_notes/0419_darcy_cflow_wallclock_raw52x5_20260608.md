# Darcy/C-flow raw 52x5 wall-clock table - 2026-06-08

Status: complete. This is the corrected raw-data view: 52 datasets (`train`, `test`, and 50 separate `generalization` datasets) times 5 models (`baseline`, `loss1`, `loss2`, `loss3`, `physics`).

Comparison axis: same wall-clock budget for the four self-training models. Epochs differ by objective and are not forced to match.

## Outputs

- Long raw table, 260 rows: `adversarial_training_runs/darcy_cflow_wallclock_raw52x5_20260608/darcy_cflow_raw52x5_wallclock_long.csv`
- Wide raw table, 52 rows: `adversarial_training_runs/darcy_cflow_wallclock_raw52x5_20260608/darcy_cflow_raw52x5_wallclock_wide.csv`
- Split summary: `adversarial_training_runs/darcy_cflow_wallclock_raw52x5_20260608/darcy_cflow_raw52x5_wallclock_split_summary.csv`
- Manifest: `adversarial_training_runs/darcy_cflow_wallclock_raw52x5_20260608/manifest.json`

## Same Wall-Clock Generalization Summary

| model | elapsed min | final epoch | gen rel-L2 mean | gen RMSE mean | gen rel-L2 drop vs baseline |
|---|---:|---:|---:|---:|---:|
| loss1 | 82.93 | 683 | 0.0914804 | 0.000560149398 | 2.03% |
| loss2 | 82.90 | 733 | 0.07687 | 0.000470695561 | 17.68% |
| loss3 | 82.78 | 500 | 0.0502322 | 0.000307583678 | 46.21% |
| physics | 82.91 | 778 | 0.0636878 | 0.000389987228 | 31.80% |

## Notes

- Baseline is the common epoch-0 row, stored once per dataset rather than duplicated per objective.
- The 50 generalization datasets are separate rows in both raw tables; they are not averaged away before export.
- Split summaries are derived from the raw table and should be treated as summaries, not the original data.
- Observed ranking by same wall-clock generalization relative-L2: loss3, physics, loss2, loss1.
