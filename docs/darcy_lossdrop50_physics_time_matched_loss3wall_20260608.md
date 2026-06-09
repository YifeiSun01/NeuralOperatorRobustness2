# Darcy Time-Matched Self-Training Summary

Observed from local run artifacts produced by `tools/adversarial_training.py`.

## Runs

| objective | epochs | steps | minutes | stop | gen RMSE baseline -> final | gen relL2 baseline -> final | peak CUDA MB |
|---|---:|---:|---:|---|---:|---:|---:|
| physics | 778 | 3112 | 82.91 | max_wall_seconds_reached | 0.000571722 -> 0.000389987 (31.79%) | 0.09338 -> 0.0636878 (31.80%) | 9331.4 |

## Files

- `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_time_matched_summary.csv`
- `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/rmse_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/relative_l2_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/darcy_time_matched_summary/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/attack_objective_and_solver_mse_curves.png`
