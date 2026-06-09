# Darcy Time-Matched Self-Training Summary

Observed from local run artifacts produced by `tools/adversarial_training.py`.

## Runs

| objective | epochs | steps | minutes | stop | gen RMSE baseline -> final | gen relL2 baseline -> final | peak CUDA MB |
|---|---:|---:|---:|---|---:|---:|---:|
| loss1 | 352 | 1408 | 82.97 | max_wall_seconds_reached | 0.000571722 -> 0.000552445 (3.37%) | 0.09338 -> 0.0902246 (3.38%) | 10131.2 |
| loss2 | 240 | 960 | 83.01 | max_wall_seconds_reached | 0.000571722 -> 0.000665826 (-16.46%) | 0.09338 -> 0.108746 (-16.46%) | 9328.7 |
| loss3 | 500 | 2000 | 82.78 |  | 0.000571722 -> 0.000307584 (46.20%) | 0.09338 -> 0.0502322 (46.21%) | 9335.8 |
| physics | 778 | 3112 | 82.91 | max_wall_seconds_reached | 0.000571722 -> 0.000389987 (31.79%) | 0.09338 -> 0.0636878 (31.80%) | 9331.4 |

## Files

- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/darcy_time_matched_summary.csv`
- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/darcy_lossdrop50_loss1_time_matched_loss3wall_20260608/rmse_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/darcy_lossdrop50_loss1_time_matched_loss3wall_20260608/relative_l2_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/darcy_lossdrop50_loss1_time_matched_loss3wall_20260608/attack_objective_and_solver_mse_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/darcy_lossdrop50_loss2_time_matched_loss3wall_20260608/rmse_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/darcy_lossdrop50_loss2_time_matched_loss3wall_20260608/relative_l2_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/darcy_lossdrop50_loss2_time_matched_loss3wall_20260608/attack_objective_and_solver_mse_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/rmse_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/relative_l2_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/attack_objective_and_solver_mse_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/rmse_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/relative_l2_split_curves.png`
- `adversarial_training_runs/darcy_lossdrop50_time_matched_loss123_physics_summary_20260608/darcy_lossdrop50_physics_time_matched_loss3wall_20260608/attack_objective_and_solver_mse_curves.png`


## Concurrency Caveat - 2026-06-08

Observed from driver logs: loss1 started at `2026-06-08T02:06:58Z` and loss2 started at `2026-06-08T02:07:00Z`, so they ran concurrently on the same V100. Both completed around `2026-06-08T03:30Z`. Physics started later at `2026-06-08T03:31:04Z`, after loss1/loss2 had finished. Loss3 was the earlier standalone 500-epoch baseline run.

Observed from `attack_epoch_summary.csv`: loss1 records `attack_uses_solver_forward=0` and `attack_uses_solver_backward=0`; loss2 records `attack_uses_solver_forward=1` and `attack_uses_solver_backward=0`; physics records `0/0`. The Darcy loss3 code path uses attacked solver output and enables solver backward when solver target gradients are enabled.

Inference: the lower completed epoch counts for loss1 and loss2 are very likely caused primarily by running them together and sharing GPU resources, not because loss1/loss2 are intrinsically slower than loss3. Therefore the final-checkpoint loss1/loss2 rows should be treated as concurrency-contaminated and not as an official fair single-GPU time-matched comparison against standalone loss3/physics. A clean comparison should rerun loss1 and loss2 sequentially on an otherwise idle GPU with the same target wall time.
