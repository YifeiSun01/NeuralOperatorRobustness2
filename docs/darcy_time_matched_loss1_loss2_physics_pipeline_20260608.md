# Darcy Flow Time-Matched Self-Training Pipeline - 2026-06-08

Status: code implemented, timing probes completed, official loss1/loss2 runs launched and monitored for about ten minutes. Physics is queued behind loss1/loss2 to avoid GPU OOM.

## Source Changes

Observed local source files:

- `tools/adversarial_training.py`
- `tools/run_darcy_lossdrop50_time_matched_objective_20260608.sh`
- `tools/summarize_darcy_time_matched_runs_20260608.py`

Key implementation details:

- Darcy Flow attack objective is now configurable with `--darcy-attack-loss-objective loss1|loss2|loss3|physics`.
- `loss1` attack objective is `MSE(model(a_adv), model(a_clean).detach())`.
- `loss2` attack objective is `MSE(model(a_adv), solver(a_clean).detach())`.
- `loss3` attack objective remains `MSE(model(a_adv), solver(a_adv))`.
- `physics`/`loss4` attack objective is the Darcy PDE residual plus boundary penalty, using the same finite-difference operator as the existing physics-loss attack code.
- Optimizer training target remains solver-consistent self-training: `model(a_adv) -> solver(a_adv).detach()` for all objectives.
- Darcy `loss1` gets a recorded random initial binary flip mask by default, because the exact clean start has zero loss1 gradient.
- The training script now supports `--max-wall-seconds`, stopping gracefully after the first completed epoch/evaluation that reaches the wall-clock budget.

## GPU Preflight

Observed before official runs:

- GPU: Tesla V100-SXM2-32GB.
- `nvidia-smi` before timing showed 0 MiB used.
- PyTorch: `2.8.0+cu126`, CUDA `12.6`, device capability `(7, 0)`, arch list includes `sm_70`.
- JAX: `0.10.0`, backend `gpu`, device `cuda:0`.
- PyTorch and JAX matrix-multiply sanity checks both ran on GPU.

## Loss3 Wall-Clock Target

Observed from `adversarial_training_runs/darcy_lossdrop50_loss3_500ep_fromscreen_20260607/darcy/summary.json`:

- Loss3 500-epoch elapsed seconds: `4966.925741452724`.
- Time-matched loss1/loss2/physics runs use this value as `--max-wall-seconds`.

## Timing Probes

Observed from 5-epoch timing runs with full train/test/50-generalization evaluation:

| objective | timing run | seconds | sec/epoch | estimated time-matched epochs | peak allocated MB | reserved MB |
|---|---|---:|---:|---:|---:|---:|
| loss1 | `adversarial_training_runs/darcy_lossdrop50_loss1_timing5ep_20260608` | `39.21667575277388` | `7.843335150554776` | `633` | `10131.2085` | `12404.0` |
| loss2 | `adversarial_training_runs/darcy_lossdrop50_loss2_timing5ep_20260608` | `39.598233016207814` | `7.919646603241563` | `627` | `9328.6514` | `11288.0` |
| physics | `adversarial_training_runs/darcy_lossdrop50_physics_timing5ep_20260608` | `34.802722845226526` | `6.960544569045306` | `714` | `9331.3574` | `11288.0` |

Inference from timing/memory:

- Three concurrent full runs would probably be unsafe on the 32GB V100 because the timing runs reserve about 11-12.4GB each.
- Two concurrent runs are reasonable; observed official loss1+loss2 concurrency stabilized around 24.7GB.
- Physics is queued to start after loss1/loss2 finish.

## Official Runs

Started in tmux:

| objective | tmux session | run directory | mode |
|---|---|---|---|
| loss1 | `darcy_loss1_timematch_20260608` | `adversarial_training_runs/darcy_lossdrop50_loss1_time_matched_loss3wall_20260608` | running |
| loss2 | `darcy_loss2_timematch_20260608` | `adversarial_training_runs/darcy_lossdrop50_loss2_time_matched_loss3wall_20260608` | running |
| physics | `darcy_physics_timematch_waiter_20260608` | `adversarial_training_runs/darcy_lossdrop50_physics_time_matched_loss3wall_20260608` | queued behind loss1/loss2 |

Observed at 2026-06-08T02:16:49Z after about ten minutes of monitoring:

- GPU memory: `24684 MiB / 32768 MiB`, GPU utilization `100%`, temperature `61C`.
- loss1 last eval epoch: `41`, global step `164`, generalization RMSE `0.0005696446592135116`, relative L2 `0.09304014607419886`.
- loss2 last eval epoch: `27`, global step `108`; last train epoch `28`, step `111`; generalization RMSE `0.0005717207608290261`, relative L2 `0.09338007305526914`.
- No OOM, no session exit, no traceback observed in run logs during the monitoring window.

## Upload And Git Notes

- The launcher supports R2 upload after completion when `AUTO_UPLOAD_R2=1` and either `R2_RCLONE_CONFIG_FILE` or `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY` are available.
- During this turn, `/tmp/neural_operator_r2_auto_rclone.conf` was not present and no `R2_*`/`RCLONE_CONFIG*` environment variables were present, so R2 upload was not armed for the already-running tmux jobs.
- Generated large run artifacts should remain out of Git and go to R2 after completion.
- Lightweight source and Markdown records are intended for GitHub branch `vast-ai`.
- Local Git commit `a8ed7db` was created, but `git push origin vast-ai` failed because the shell has no GitHub credential helper/token available (`could not read Username for https://github.com`).

Generated at 2026-06-08T02:18:51Z.
