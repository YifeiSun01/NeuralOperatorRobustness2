# Darcy Time-Matched Self-Training Summary

Observed from local run artifacts produced by `tools/adversarial_training.py`.

## Runs

| objective | epochs | steps | minutes | stop | gen RMSE baseline -> final | gen relL2 baseline -> final | peak CUDA MB |
|---|---:|---:|---:|---|---:|---:|---:|
| loss3 | 20 | 80 | 2.46 | completed_configured_epochs | 0.0005662 -> 0.00136231 (-140.60%) | 0.0929271 -> 0.223521 (-140.53%) | 9327.1 |

## Files

- `adversarial_training_runs/darcy_stockloss3other20training_20260608/darcy_time_matched_summary/darcy_time_matched_summary.csv`
- `adversarial_training_runs/darcy_stockloss3other20training_20260608/darcy_time_matched_summary/darcy_stockloss3other20training_20260608/rmse_split_curves.png`
- `adversarial_training_runs/darcy_stockloss3other20training_20260608/darcy_time_matched_summary/darcy_stockloss3other20training_20260608/relative_l2_split_curves.png`
- `adversarial_training_runs/darcy_stockloss3other20training_20260608/darcy_time_matched_summary/darcy_stockloss3other20training_20260608/attack_objective_and_solver_mse_curves.png`
