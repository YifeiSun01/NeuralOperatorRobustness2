# Darcy/SIR20 Serial Double-Budget Run

This run updates the Darcy/SIR20 serial training pipeline for the six training methods:

- `loss1`
- `loss2`
- `loss3`
- `physics_loss`
- `random_clean`
- `random_solver`

## Training Change

For Darcy binary adversarial training, the per-batch epsilon budget factor is now sampled as:

```text
uniform(0.25, 1.75)
```

The base Darcy epsilon fraction remains `0.025`, so each batch uses a fresh budget multiplier against that base fraction. This is intentionally batch-level randomization, which is finer than epoch-level randomization.

## Checkpoints

The full pipeline keeps the original time-matched epoch count for each method, then trains to twice that count.

For each of the six methods, the launcher writes:

- one checkpoint at the original epoch count
- one checkpoint at the doubled epoch count

The final run therefore produces 12 trained checkpoints, plus the unchanged baseline checkpoint for evaluation.

The launcher writes two manifests:

- `training_checkpoints_full.json`: baseline plus six final checkpoints, used for training-curve visualization.
- `training_checkpoints_full_all_saved.json`: baseline plus the 12 saved original/final checkpoints, used for final evaluation, advanced serial attack, Jacobian analysis, and SVD analysis.

## Pipeline

Run:

```bash
MODE=full tools/run_darcy_sir20_timematched_full_20260614.sh
```

By default, the script:

- calibrates the six methods
- trains with doubled epochs and batch-level epsilon jitter
- evaluates all saved checkpoints on the 52 Darcy datasets
- runs the advanced serial loss3 attack
- computes Jacobian/SVD diagnostics
- generates figures
- uploads the output bundle to R2
- commits source and Markdown changes and pushes to `vast-ai-darcy-flow`

Runtime credentials for R2 and GitHub must be supplied through environment variables or an out-of-repository askpass/config file. They must not be committed.
