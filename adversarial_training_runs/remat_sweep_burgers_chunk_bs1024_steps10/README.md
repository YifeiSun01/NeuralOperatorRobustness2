# Adversarial Training Run

- Run directory: `adversarial_training_runs/remat_sweep_burgers_chunk_bs1024_steps10`
- Tasks: `burgers`
- Smoke: `False`
- Label mode: `solver`

Each task subdirectory contains `train_steps.csv`, `attack_batches.csv`, `optimizer_steps.csv`, `eval_metrics.csv`, `memory.csv`, checkpoints, `data_range_summary.json`, and `summary.json`.
Default training now uses the full original train split for every epoch; pass `--<task>-train-max N` only for debugging caps, or `0` for full.

Default attack policy:
- Burgers: short L-infinity fast-replace attack; default attack batch covers the full 1350-sample train split, optimizer microbatch is 300, with per-sample epsilon jitter and alpha=epsilon*ratio.
- Darcy: binary steepest-replace flips coefficient pixels; default attack batch is 384 because 512/1200 OOM on the 31.7GB GPU, optimizer microbatch is 80, with jittered flip budgets.
- NS2D: L-infinity add attack on the initial vorticity frame; attack batch and optimizer batch stay 1:1 by default.

Evaluation is clean evaluation on train/test/generated datasets at baseline and at scheduled progress fractions.
