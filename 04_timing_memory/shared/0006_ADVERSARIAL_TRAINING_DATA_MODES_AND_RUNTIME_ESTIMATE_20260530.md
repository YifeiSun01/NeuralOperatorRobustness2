# Adversarial Training Data Modes And Runtime Estimate

Date: 2026-05-30 UTC

This note records the new adversarial-training data modes and a runtime estimate for training/eval under the corrected 10-step full-solver attack policy.

## Why This Was Added

Concern: if adversarial training uses only attacked samples for many epochs, clean train/test performance may degrade even if generalization or robustness improves.

To test this, adversarial training now supports two data modes:

| mode | training pairs used per attack batch | purpose |
|---|---|---|
| `adv-only` | `model(x_adv) -> solver(x_adv)` | pure adversarial/self-training on attacked data |
| `clean-plus-adv` | `model(x_clean) -> solver(x_clean)` plus `model(x_adv) -> solver(x_adv)` | preserve clean train/test behavior while also training on attacked data |

`clean-plus-adv` doubles the training examples produced by each attack batch. The attack itself is still re-generated each epoch/batch, so attacked examples are not identical across epochs.

## Solver-Label Rule

Both modes use full solver labels.

For Burgers and Darcy/C-flow:

```text
clean: model(x_clean) -> solver(x_clean)
adv:   model(x_adv)   -> solver(x_adv)
```

For NS2D:

```text
clean: x0_clean -> solver rollout -> (x_seq_clean, y_seq_clean)
adv:   x0_adv   -> solver rollout -> (x_seq_adv,   y_seq_adv)
```

The code rejects `--training-data-mode clean-plus-adv` unless `--label-mode solver` is used. This prevents the invalid old ablation target `x_adv -> y_clean`.

## Code Changes

Main file:

- `tools/adversarial_training.py`

Added CLI option:

```bash
--training-data-mode adv-only
--training-data-mode clean-plus-adv
```

Added helper behavior:

- `clean_solver_training_pair(...)`: regenerates clean labels with the full solver.
- `combine_training_pairs(...)`: returns either adv-only data or concatenated clean+adv data.

Logged in `train_steps.csv`:

- `training_data_mode`
- `clean_train_samples`
- `adv_train_samples`
- `effective_train_samples`
- `clean_plus_adv_multiplier`

Smoke verification:

```text
batch_size = 2
training_data_mode = clean-plus-adv
clean_train_samples = 2
adv_train_samples = 2
effective_train_samples = 4
clean_plus_adv_multiplier = 2.0
```

Smoke run:

- `adversarial_training_runs/smoke_clean_plus_adv_burgers_20260530_r2`

## Corrected Attack Policy

Use the corrected 10-step policy, not the earlier 1-step/3-step comparison.

| task | attack | steps | alpha/remat |
|---|---|---:|---|
| NS2D | `steepest_add_linf` | 10 | `alpha_ratio=0.2`, `ns2d_solver_remat=chunk`, `chunk_steps=20` |
| Burgers | `steepest_replace_linf` | 10 | `burgers_solver_remat=chunk`, `chunk_steps=50` |
| Darcy/C-flow | `binary_steepest_replace` | 10 | no explicit rollout remat; batch tuned |

Corrected throughput record:

- `adversarial_training_runs/CORRECTED_10STEP_REMAT_CHECKPOINT_RECORD_20260530.md`
- `adversarial_training_runs/corrected_10step_remat_checkpoint_throughput_20260530.csv`

## Recommended Batch Settings

| task | recommended batch | optimizer microbatch | reason |
|---|---:|---:|---|
| NS2D | 6 | 1 | batch 8/10 OOM |
| Burgers | 1350 | 32 | full train batch passes and has best tested throughput |
| Darcy/C-flow | 448 | 32 | highest tested throughput; 480 passes but slightly slower; 496/512 OOM |

## Runtime Estimate Assumptions

These are estimates from one-batch measured timings, not full completed 500-epoch runs.

Assumptions:

- Epochs: `500`
- Eval schedule: baseline plus 5 scheduled eval passes, from `eval_every_fraction=0.2`, so `6` eval passes total.
- Eval/val means clean evaluation on train/test/generalization datasets. The current code does not have a separate validation split beyond the clean eval datasets.
- Eval dataset count per pass: `52` per task (`train`, `test`, plus 50 generalization datasets).
- `clean-plus-adv` attack time is about the same as `adv-only`, but optimizer training sees twice as many samples and the clean half also regenerates solver labels.
- For `clean-plus-adv`, the extra clean solver-label generation is estimated as a small fraction of attack time. This is an estimate; exact cost should be measured in a full run.

Measured one-batch timings used:

| task | batch | attack sec | optimizer train sec | full step sec |
|---|---:|---:|---:|---:|
| Burgers | 1350 | 53.075 | 1.418 | 54.498 |
| Darcy/C-flow | 448 | 14.522 | 2.060 | 16.594 |
| NS2D | 6 | 431.943 | 2.537 | 434.494 |

## Estimated Full Runtime By Task

### `adv-only`

| task | full train samples | batch | batches/epoch | total attack batches | train time | eval/val time | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Burgers | 1350 | 1350 | 1 | 500 | 7.57 h | ~0.004 h | ~7.57 h |
| Darcy/C-flow | 1200 | 448 | 3 | 1500 | 6.91 h | ~0.004 h | ~6.92 h |
| NS2D | 50 | 6 | 9 | 4500 | 543.12 h | ~0.008 h | ~543.13 h |

All three tasks together:

```text
training: ~557.60 h
eval/val: ~0.016 h
total:    ~557.62 h  (~23.23 days)
```

### `clean-plus-adv`

| task | full train samples | attack batch | effective training samples/batch | total attack batches | train time | eval/val time | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Burgers | 1350 | 1350 | 2700 | 500 | 8.36 h | ~0.004 h | ~8.36 h |
| Darcy/C-flow | 1200 | 448 | 896 | 1500 | 8.25 h | ~0.004 h | ~8.25 h |
| NS2D | 50 | 6 | 12 | 4500 | 611.06 h | ~0.008 h | ~611.07 h |

All three tasks together:

```text
training: ~627.67 h
eval/val: ~0.016 h
total:    ~627.69 h  (~26.15 days)
```

## Interpretation

- The runtime is dominated by NS2D full-solver 10-step attack.
- Eval/val is tiny compared with training because eval is clean model evaluation, not adversarial attack generation.
- `clean-plus-adv` is expected to help preserve original clean train/test performance because every update still includes clean solver-consistent data.
- `clean-plus-adv` costs more mainly because optimizer training sees twice as many examples and clean solver labels are regenerated.
- The attack remains stochastic across epochs because random starts, epsilon jitter, score noise, and epoch shuffling remain active.

## Example Commands

Adv-only:

```bash
adv_robust/bin/python tools/adversarial_training.py   --tasks burgers,darcy,ns2d   --training-data-mode adv-only   --label-mode solver
```

Clean plus adversarial:

```bash
adv_robust/bin/python tools/adversarial_training.py   --tasks burgers,darcy,ns2d   --training-data-mode clean-plus-adv   --label-mode solver
```

Correct 10-step task-specific overrides should also be supplied for production runs, for example:

```bash
--burgers-attack-steps 10 --burgers-batch-size 1350 --burgers-optimizer-batch-size 32 --burgers-solver-remat chunk --burgers-solver-remat-chunk-steps 50
--darcy-attack-steps 10 --darcy-batch-size 448 --darcy-optimizer-batch-size 32
--ns2d-attack-steps 10 --ns2d-alpha-ratio 0.2 --ns2d-batch-size 6 --ns2d-optimizer-batch-size 1 --ns2d-solver-remat chunk --ns2d-solver-remat-chunk-steps 20
```
