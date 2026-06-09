# Darcy Time-Matched Self-Training Summary

Observed from local run artifacts produced by `tools/adversarial_training.py`.

## Runs

| objective | epochs | steps | minutes | stop | gen RMSE baseline -> final | gen relL2 baseline -> final | peak CUDA MB |
|---|---:|---:|---:|---|---:|---:|---:|
| loss2 | 240 | 960 | 83.01 | max_wall_seconds_reached | 0.000571722 -> 0.000665826 (-16.46%) | 0.09338 -> 0.108746 (-16.46%) | 9328.7 |

## Files

- `adversarial_training_runs/darcy_lossdrop50_loss2_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_time_matched_summary.csv`
- `adversarial_training_runs/darcy_lossdrop50_loss2_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_loss2_time_matched_loss3wall_20260608/rmse_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_loss2_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_loss2_time_matched_loss3wall_20260608/relative_l2_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_loss2_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_loss2_time_matched_loss3wall_20260608/attack_objective_and_solver_mse_curves.png`
