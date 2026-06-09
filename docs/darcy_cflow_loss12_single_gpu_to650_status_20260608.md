# Darcy/C-flow loss1/loss2 single-GPU to-650 status - 2026-06-08

Status: inspected local artifacts on 2026-06-08 UTC. The dedicated `continue_to650` launcher has code/plan records but no local run logs or stitched outputs. The clean single-GPU sequential time-matched rerun did complete and naturally exceeded epoch 650: loss1 reached epoch 683 and loss2 reached epoch 733.

## Observed source artifacts

- loss1 run: `adversarial_training_runs/darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608/`
- loss2 run: `adversarial_training_runs/darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608/`
- loss3 comparison run: `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/`
- physics comparison run: `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/`
- loss1/loss2 driver log: `adversarial_training_runs/darcy_loss12_single_gpu_time_matched_20260608_logs/driver.log`
- exact metrics: each run's `darcy/eval_split_summary.csv` and `darcy/summary.json`

## Generalization split

Baseline for all rows: generalization RMSE `0.000571721582`, relative-L2 `0.0933799985`, accuracy score `91.4600`.

| objective | checked epoch | final epoch | minutes | gen RMSE | gen rel-L2 | gen accuracy | rel-L2 drop vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| loss1 | 650 | 683 | 82.93 | 0.000539846 | 0.0881645 | 91.8984 | 5.59% |
| loss1 final | 683 | 683 | 82.93 | 0.000560149 | 0.0914804 | 91.6192 | 2.03% |
| loss2 | 650 | 733 | 82.90 | 0.000607934 | 0.0992872 | 90.9686 | -6.33% |
| loss2 final | 733 | 733 | 82.90 | 0.000470696 | 0.0768700 | 92.8622 | 17.68% |
| loss3 final | 500 | 500 | 82.78 | 0.000307584 | 0.0502322 | 95.2172 | 46.21% |
| physics 650 | 650 | 778 | 82.91 | 0.000432595 | 0.0706467 | 93.4018 | 24.34% |
| physics final | 778 | 778 | 82.91 | 0.000389987 | 0.0636878 | 94.0128 | 31.80% |

## Train/test/generalization at epoch 650

| objective | split | RMSE | rel-L2 | accuracy | rel-L2 drop vs baseline |
|---|---|---:|---:|---:|---:|
| loss1 | train | 0.000164513 | 0.0244742 | 97.611 | 31.16% |
| loss1 | test | 0.000235876 | 0.0345724 | 96.658 | 22.69% |
| loss1 | generalization | 0.000539846 | 0.0881645 | 91.898 | 5.59% |
| loss2 | train | 0.000176364 | 0.0262372 | 97.443 | 26.20% |
| loss2 | test | 0.000229937 | 0.0337019 | 96.740 | 24.64% |
| loss2 | generalization | 0.000607934 | 0.0992872 | 90.969 | -6.33% |
| physics | train | 0.000158397 | 0.0235643 | 97.698 | 33.72% |
| physics | test | 0.000231556 | 0.0339391 | 96.717 | 24.11% |
| physics | generalization | 0.000432595 | 0.0706467 | 93.402 | 24.34% |

## Observed interpretation

- loss1 is fine on train/test by epoch 650, but its generalization gain is small. It briefly had a much better generalization best at epoch 2, then drifted upward.
- loss2 is not good exactly at epoch 650 on the generalization split, but by final epoch 733 it recovers to a clear positive generalization gain.
- loss3 remains the strongest of the visible local Darcy/C-flow results on the 50-dataset generalization split, despite only running to epoch 500.
- physics is the most stable non-loss3 comparator here: it is already positive at epoch 650 and improves further by epoch 778.

## Missing local evidence

- No local `adversarial_training_runs/darcy_loss12_continue_to650_20260608_logs/` directory was found.
- No local `adversarial_training_runs/darcy_lossdrop50_loss12_to650_loss3_physics_summary_20260608/` stitched combined output was found.
- No R2 upload evidence for the to-650 continuation was found in the local artifact scan.
