# Burgers Random-Field Noise Training Plan - 2026-06-12

## Purpose

This adds two non-adversarial random-delta training baselines for the current Burgers wide-parameter generalization dataset. The point is to compare the adversarial loss1/loss2/loss3 training runs against a simpler data-augmentation strategy where the perturbation is sampled randomly instead of optimized by an attack.

## Training Modes

Both modes sample a fresh perturbation every epoch and every training batch:

- `random_clean_y`: train on `x + delta` while keeping the original clean target `y_clean` fixed.
- `random_solver_y`: train on `x + delta` and recompute the target with the Burgers solver, `solver(x + delta)`.

The first mode is intentionally non-physical because the target is held fixed after changing the input. The second mode is solver-consistent, but it is still not adversarial because `delta` is not chosen by maximizing any loss.

## Random Delta Generator

The new perturbation path is enabled with:

```bash
--training-perturbation-mode random-field
```

For Burgers, the sampled `delta` is a 1D random field generated in Fourier space. For every sample in the batch, the code randomly chooses one of:

- Gaussian / RBF kernel
- Matern kernel

Default parameter choices:

- Gaussian correlation length: `0.015,0.03,0.06,0.12,0.24`
- Matern correlation length: `0.015,0.03,0.06,0.12,0.24`
- Matern smoothness `nu`: `1.2,2.2,3.2,4.2,5.2`
- domain extent: `2.0`

The field is mean-centered and normalized so that the per-sample RMS of `delta` matches the epsilon budget. The launcher default is intentionally capped so the random-field RMS energy never exceeds 5% of the clean sample range:

```text
epsilon = per_sample_range(x) * burgers_epsilon_fraction * jitter
```

Default epsilon settings in the launcher:

- `burgers_epsilon_fraction = 0.04`
- jitter low/high: `0.75,1.25`
- resulting per-sample RMS delta budget: `3%` to `5%` of the clean sample range, mean about `4%`

Optional clamps are available but off by default so that the RMS delta budget remains exact:

```bash
--random-field-clip-x-min VALUE
--random-field-clip-x-max VALUE
```

## Implementation Files

- `tools/adversarial_training.py`
  - Adds `--training-perturbation-mode random-field`.
  - Adds `--random-field-target-mode clean-y|solver-y`.
  - Records random-field family counts, sampled length ranges, Matern `nu` ranges, epsilon stats, delta RMS/Linf, FFT probe stats, train/eval CSVs, checkpoints, and summaries using the same output layout as adversarial training.

- `tools/run_burgers_wideparam_random_field_training_20260612.sh`
  - Runs both 2000 epoch baselines.
  - Defaults to serial execution to avoid fighting the current long GPU job.
  - Set `RUN_RANDOM_PARALLEL=1` only when the GPU has enough free memory and no other major job is running.

- `tools/plot_burgers_wideparam_loss123_retrain_20260611.py`
  - Backward-compatible: old loss1/loss2/loss3 usage is unchanged.
  - New generic `--run label=path` support lets the random-field launcher reuse the same merged CSV and polished variable-epoch visualizations.

## Planned Output Directories

Training outputs:

- `adversarial_training_runs/burgers_wideparam_random_field_clean_y_2000ep_20260612/`
- `adversarial_training_runs/burgers_wideparam_random_field_solver_y_2000ep_20260612/`
- `adversarial_training_runs/burgers_wideparam_random_field_training_20260612_logs/`

Visualization/report outputs:

- `visualizations/burgers_wideparam_random_field_training_20260612/`
- `docs/burgers_wideparam_random_field_training_report_20260612.md`

Preflight outputs:

- `forensics/burgers_wideparam_random_field_training_preflight_20260612/`

## Commands

Dry run only:

```bash
DRY_RUN=1 bash tools/run_burgers_wideparam_random_field_training_20260612.sh
```

Full run after the current loss1/loss2 adversarial retrain is done:

```bash
bash tools/run_burgers_wideparam_random_field_training_20260612.sh
```

Run only one target mode:

```bash
RUN_RANDOM_SOLVER_Y=0 bash tools/run_burgers_wideparam_random_field_training_20260612.sh
RUN_RANDOM_CLEAN_Y=0 bash tools/run_burgers_wideparam_random_field_training_20260612.sh
```

Override random-field parameters:

```bash
RANDOM_FIELD_GAUSSIAN_CORR=0.01,0.025,0.05,0.1,0.2 RANDOM_FIELD_MATERN_CORR=0.01,0.025,0.05,0.1,0.2 RANDOM_FIELD_MATERN_NU=1.2,2.2,3.2,4.2,5.2 bash tools/run_burgers_wideparam_random_field_training_20260612.sh
```

## Notes

These two runs are baselines, not replacements for loss1/loss2/loss3 adversarial training. They test whether random local augmentation can improve clean generalization or robustness. The expected limitation is that random deltas do not actively search difficult directions, so they may help local smoothness around train/test data but may not match adversarial training on the wider generalization datasets or on attack-driven robustness metrics.
