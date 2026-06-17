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

The base Darcy epsilon fraction remains `0.025`, so every training attack batch refreshes the budget multiplier against that base fraction. In the current binary Darcy attack implementation the multiplier is sampled per sample inside the batch, so it is at least batch-level randomization and finer than epoch-level randomization.

The driver defaults to `DARCY_TRAIN_MAX=64`, `DARCY_BATCH=64`, and `OPT_BATCH=32`, matching the older long Darcy scripts where one epoch corresponds to one optimizer-update step.

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
- generates figures, including training loss, attack loss gain, split-mean MSE/RMSE/Relative-L2 curves, and all 50 generalization-dataset MSE/RMSE/Relative-L2 curves split into `part01` and `part02`
- records fixed training attack probes every epoch by default, including `x_clean`, `x_adv`, `delta`, targets, per-sample attack losses, and delta spectral statistics, then generates delta heatmaps, FFT spectrum heatmaps, radial spectra, and spectral-stat curves
- uploads the output bundle to R2
- commits source and Markdown changes and pushes to `vast-ai-darcy-flow`

Runtime credentials for R2 and GitHub must be supplied through environment variables or an out-of-repository askpass/config file. They must not be committed.
