# Darcy Time-Matched Self-Training Summary

Observed from local run artifacts produced by `tools/adversarial_training.py`.

## Runs

| objective | epochs | steps | minutes | stop | gen RMSE baseline -> final | gen relL2 baseline -> final | peak CUDA MB |
|---|---:|---:|---:|---|---:|---:|---:|
| loss1 | 5 | 20 | 0.65 | completed_configured_epochs | 0.000571722 -> 0.000447101 (21.80%) | 0.09338 -> 0.0730285 (21.79%) | 10131.2 |

## Files

- `adversarial_training_runs/darcy_lossdrop50_loss1_timing5ep_20260608/darcy_time_matched_summary/darcy_time_matched_summary.csv`
- `adversarial_training_runs/darcy_lossdrop50_loss1_timing5ep_20260608/darcy_time_matched_summary/darcy_lossdrop50_loss1_timing5ep_20260608/rmse_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_loss1_timing5ep_20260608/darcy_time_matched_summary/darcy_lossdrop50_loss1_timing5ep_20260608/relative_l2_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_loss1_timing5ep_20260608/darcy_time_matched_summary/darcy_lossdrop50_loss1_timing5ep_20260608/attack_objective_and_solver_mse_curves.png`
