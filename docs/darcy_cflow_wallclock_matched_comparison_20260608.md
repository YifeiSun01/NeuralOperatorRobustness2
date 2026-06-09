# Darcy/C-flow wall-clock matched comparison - 2026-06-08

Status: corrected wall-clock-time interpretation after inspecting local artifacts. The fair comparison is the final row of each time-matched run, because all four visible runs used the same target wall time and finished within about 9 seconds of each other. Epoch 650 is only a diagnostic point for loss1/loss2/physics, not the comparison axis.

## Observed source artifacts

- loss1: `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/darcy/summary.json` and `eval_split_summary.csv`
- loss2: `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/darcy/summary.json` and `eval_split_summary.csv`
- loss3: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/summary.json` and `eval_split_summary.csv`
- physics: `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy/summary.json` and `eval_split_summary.csv`
- source driver log for single-GPU loss1/loss2: `adversarial_training_runs/darcy_loss12_single_gpu_time_matched_20260608_logs/driver.log`

Baseline generalization row for all objectives: RMSE `0.000571721582`, relative-L2 `0.0933799985`, accuracy score `91.4600`.

## Same Wall-Clock Final Rows

| objective | elapsed min | final epoch | gen RMSE | gen rel-L2 | gen accuracy | gen rel-L2 drop vs baseline | train rel-L2 drop | test rel-L2 drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| loss1 | 82.93 | 683 | 0.000560149 | 0.0914804 | 91.6192 | 2.03% | 32.60% | 24.16% |
| loss2 | 82.90 | 733 | 0.000470696 | 0.0768700 | 92.8622 | 17.68% | 4.09% | 10.11% |
| loss3 | 82.78 | 500 | 0.000307584 | 0.0502322 | 95.2172 | 46.21% | -77.88% | -44.04% |
| physics | 82.91 | 778 | 0.000389987 | 0.0636878 | 94.0128 | 31.80% | 18.60% | 16.00% |

## Interpretation

- The wall-clock matched ranking on the 50-dataset generalization split is: loss3 strongest, physics second, loss2 third, loss1 weakest.
- loss1 and loss2 reached around/above 650 epochs because their attack objectives are faster than loss3. Physics reached 778 epochs in the same wall-clock window, so comparing physics at epoch 650 would underuse its allowed time.
- loss3 only reached 500 epochs because that path uses the attacked solver-output objective and solver-gradient path, but at equal wall-clock it still gives the best generalization split.
- loss3's train/test rows worsen relative to the baseline, while its generalization split improves the most. Physics is the steadier comparator across train/test/generalization.
- Therefore the correct fair statement is not "compare everyone at epoch 650". It is "compare each objective at the same wall-clock budget, using its final time-matched checkpoint".

## Corrected Raw Table

The raw data table that keeps `train`, `test`, and all 50 generalization datasets separate is recorded at `adversarial_training_runs/darcy_cflow_wallclock_raw52x5_20260608/darcy_cflow_raw52x5_wallclock_long.csv`. It has 260 rows: 52 datasets times 5 models (`baseline`, `loss1`, `loss2`, `loss3`, `physics`). The wide 52-row version is `adversarial_training_runs/darcy_cflow_wallclock_raw52x5_20260608/darcy_cflow_raw52x5_wallclock_wide.csv`.

## Diagnostic Epoch-650 Note

At epoch 650, loss1 had a small positive generalization gain, loss2 was worse than baseline, and physics was already positive. These epoch-650 rows are useful for understanding trajectories, but they are not the fair wall-clock comparison.
