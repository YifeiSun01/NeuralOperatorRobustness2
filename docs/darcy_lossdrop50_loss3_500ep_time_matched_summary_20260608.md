# Darcy Time-Matched Self-Training Summary

Observed from local run artifacts produced by `tools/adversarial_training.py`.

## Runs

| objective | epochs | steps | minutes | stop | gen RMSE baseline -> final | gen relL2 baseline -> final | peak CUDA MB |
|---|---:|---:|---:|---|---:|---:|---:|
| loss3 | 500 | 2000 | 82.78 |  | 0.000571722 -> 0.000307584 (46.20%) | 0.09338 -> 0.0502322 (46.21%) | 9335.8 |

## Files

- `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy_time_matched_summary/darcy_time_matched_summary.csv`
- `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy_time_matched_summary/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/rmse_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy_time_matched_summary/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/relative_l2_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy_time_matched_summary/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/attack_objective_and_solver_mse_curves.png`
