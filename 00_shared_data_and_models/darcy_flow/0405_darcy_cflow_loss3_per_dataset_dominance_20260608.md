# Darcy/C-flow loss3 per-dataset dominance check - 2026-06-08

Status: inspected the corrected raw table `adversarial_training_runs/darcy_cflow_wallclock_raw52x5_20260608/darcy_cflow_raw52x5_wallclock_wide.csv`.

## Result

For the 50 separate generalization datasets, loss3 is best on every dataset for all three inspected error metrics: relative-L2, RMSE, and MAE.

For the full 52-dataset set (`train`, `test`, and 50 generalization datasets), loss3 is not best on all datasets. It is worst on the train and test datasets for relative-L2, RMSE, and MAE.

## Counts

| metric | split | datasets | loss3 best vs all five models |
|---|---|---:|---:|
| relative-L2 | train | 1 | 0 |
| relative-L2 | test | 1 | 0 |
| relative-L2 | generalization | 50 | 50 |
| RMSE | train | 1 | 0 |
| RMSE | test | 1 | 0 |
| RMSE | generalization | 50 | 50 |
| MAE | train | 1 | 0 |
| MAE | test | 1 | 0 |
| MAE | generalization | 50 | 50 |

## Train/test ordering by relative-L2

Train relative-L2 order, lower is better:

1. loss1: `0.0239629`
2. physics: `0.0289417`
3. loss2: `0.0340981`
4. baseline: `0.0355534`
5. loss3: `0.0632418`

Test relative-L2 order, lower is better:

1. loss1: `0.0339170`
2. physics: `0.0375648`
3. loss2: `0.0401999`
4. baseline: `0.0447192`
5. loss3: `0.0644120`

## Interpretation

- Correct statement for generalization only: loss3 is uniformly best across all 50 generalization datasets at the same wall-clock budget.
- Correct statement for all 52 datasets: loss3 is best on 50/52 datasets and worst on train/test.
- This explains the aggregate behavior: loss3 gives the strongest off-distribution/generalization improvement, but it sacrifices in-distribution train/test performance.
