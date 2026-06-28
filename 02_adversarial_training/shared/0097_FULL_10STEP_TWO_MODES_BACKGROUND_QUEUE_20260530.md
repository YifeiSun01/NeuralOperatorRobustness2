# Full 10-Step Two-Mode Background Queue

Date: 2026-05-30 UTC

This launches the six requested adversarial-training cases as one sequential GPU queue, to avoid concurrent GPU memory collisions on the single V100.

Cases:

| order | task | training data mode | run name |
|---:|---|---|---|
| 1 | Burgers | adv-only | `full10_burgers_adv_only_20260530` |
| 2 | Burgers | clean-plus-adv | `full10_burgers_clean_plus_adv_20260530` |
| 3 | Darcy/C-flow | adv-only | `full10_darcy_adv_only_20260530` |
| 4 | Darcy/C-flow | clean-plus-adv | `full10_darcy_clean_plus_adv_20260530` |
| 5 | NS2D | adv-only | `full10_ns2d_adv_only_20260530` |
| 6 | NS2D | clean-plus-adv | `full10_ns2d_clean_plus_adv_20260530` |

Common settings:

- `--label-mode solver`
- `--eval-every-fraction 0.2`, so each run records baseline plus 20/40/60/80/100 percent progress evaluations.
- `--eval-max-samples 50`
- `--max-generalization-eval 50`

Attack settings:

- Burgers: 10-step, batch 1350, optimizer microbatch 32, `burgers_solver_remat=chunk`, chunk steps 50.
- Darcy/C-flow: 10-step, batch 448, optimizer microbatch 32.
- NS2D: 10-step, alpha ratio 0.2, batch 6, optimizer microbatch 1, `ns2d_solver_remat=chunk`, chunk steps 20.

Launcher:

- `adversarial_training_runs/run_full_adv_training_queue_20260530.sh`

Queue logs/status:

- `adversarial_training_runs/full_10step_two_modes_queue_20260530/queue.log`
- `adversarial_training_runs/full_10step_two_modes_queue_20260530/queue_status.csv`
- per-case logs in `adversarial_training_runs/full_10step_two_modes_queue_20260530/logs/`

## 10-Minute Startup Monitor

Started successfully after one empty first launch attempt caused by the `nohup` redirection directory not existing before shell redirection. The empty attempt was archived at:

- `adversarial_training_runs/full_10step_two_modes_queue_20260530_attempt1_empty_start`

Detached active queue:

- launcher PID: `2778630`
- current Python PID: `2778635`
- current case: `burgers_adv_only`

At about 5 minutes after the successful start:

- process alive: yes
- GPU memory: about `12890 MiB / 32768 MiB`
- GPU utilization: about `87%`
- latest written training step: `global_step=6`
- observed step time: about `52-54 s/step`
- CUDA peak allocated in CSV: about `11008.9 MiB`
- target invalid fraction: `0.0`
- no OOM or traceback observed

At about 10 minutes after the successful start:

- process alive: yes
- GPU memory: about `12890 MiB / 32768 MiB`
- latest written training step: `global_step=12`
- `train_steps.csv`: 12 data rows plus header
- `optimizer_steps.csv`: 516 optimizer microbatch rows plus header
- `memory.csv`: 12 data rows plus header
- latest progress fraction: `0.024`
- latest observed step time: about `52.86 s`
- latest attack time: about `51.77 s`
- latest train time: about `1.08 s`
- latest CUDA peak allocated in CSV: about `11008.9 MiB`
- no OOM, traceback, NaN invalid target, or queue failure observed

The first scheduled 20% eval/checkpoint for this run will occur at `global_step=100` because this run has `total_steps=500` and `eval_every_fraction=0.2`. The code also wrote baseline eval rows at startup in `eval_metrics.csv`.
