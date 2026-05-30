# Adversarial Training Run

- Run directory: `adversarial_training_runs/full_adversarial_training_20260530_v3_large_batches`
- Tasks: `burgers,darcy,ns2d`
- Smoke: `False`
- Label mode: `clean`

Each task subdirectory contains `train_steps.csv`, `attack_batches.csv`, `eval_metrics.csv`, `memory.csv`, checkpoints, `data_range_summary.json`, and `summary.json`.

Default attack policy:
- Burgers: short L-infinity fast-replace attack with random start and epsilon jitter.
- Darcy: binary steepest-replace flips a jittered fraction of coefficient pixels, with noisy top-k selection for diversity.
- NS2D: short L-infinity add attack on the first input frame by default, with alpha proportional to epsilon.

Evaluation is clean evaluation on train/test/generated datasets at baseline and at scheduled progress fractions.
