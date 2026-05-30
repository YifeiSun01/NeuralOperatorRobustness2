# Solver-Label Adversarial Training Fix - 2026-05-30

The previous v5 clean-label run is invalid because it trained `model(x_adv) -> y_clean`. The script now defaults to solver-label training and blocks clean-label unless `--allow-clean-label` is explicitly passed.

## Code Changes

- `tools/adversarial_training.py` now returns an `AttackBatchResult(x_train, y_train, info)` from every attack path.
- Burgers target: `solver_burgers(x_adv)`.
- Darcy target: JAX Darcy solver on attacked coefficient field.
- NS2D target: attack the initial vorticity frame, then regenerate input frames `0..9` and target frames `10..19` by NS solver rollout from the attacked initial state.
- Training loop now optimizes `finite_mse(model(x_train_adv), y_train_adv)`.
- `--label-mode` default is `solver`; `--label-mode clean` raises unless `--allow-clean-label` is supplied.

## Smoke Tests

All smoke tests used one training sample, one batch, one epoch, no generalization evaluation, and exact solver target gradients.

| task | run dir | attack time | train time | peak CUDA allocated | target source | notes |
|---|---:|---:|---:|---:|---|---|
| Burgers | `adversarial_training_runs/smoke_solver_label_burgers_exact_fixcheck` | 12.73 s | 0.14 s | 78 MB | `solver` | passed |
| Darcy | `adversarial_training_runs/smoke_solver_label_darcy_exact_fixcheck` | 2.78 s | 0.19 s | 4.52 GB | `solver` | passed |
| NS2D | `adversarial_training_runs/smoke_solver_label_ns2d_exact_fixcheck` | 52.18 s | 0.48 s | 26.88 GB | `solver_rollout_from_attacked_initial_state` | passed; target absmax 3.10 |

## Practical Warning

The corrected NS2D exact solver-label attack is much more expensive than the invalid clean-label path. A single NS2D sample and one attack step took about 52 s and peaked near 26.9 GB, because the 20-second PDE rollout is now inside the attack graph.

## Exact Solver-Gradient Correction

The script now allows only exact full solver-gradient adversarial training. The attack objective is:

```text
maximize MSE(model(x_adv), solver(x_adv))
```

Gradients flow back through both `model(x_adv)` and `solver(x_adv)` to the attacked initial condition/coefficient. There is no command-line switch for changing this objective; if a batch is too large, the batch size must be reduced instead of changing the physics/gradient objective.

## Exact Batch Smoke Results

These runs use solver labels and exact solver-target gradients. Attack batch and optimizer batch are decoupled: first generate `x_adv/y_adv` for the attack batch, then train through optimizer microbatches.

| task | run dir | attack batch | optimizer batch | attack time | train time | peak CUDA allocated | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Burgers | `adversarial_training_runs/smoke_solver_exact_burgers_batch256_opt32` | 256 | 32 | 10.22 s | 0.31 s | 15.07 GB | passed, 8 optimizer microbatches |
| Darcy | `adversarial_training_runs/smoke_solver_exact_darcy_batch256_opt64` | 256 | 64 | 4.13 s | 1.00 s | 17.22 GB | passed, 4 optimizer microbatches |
| NS2D | `adversarial_training_runs/smoke_solver_label_ns2d_exact_fixcheck` | 1 | 1 | 52.18 s | 0.48 s | 26.88 GB | passed; target absmax 3.10 |
| NS2D | `adversarial_training_runs/smoke_solver_exact_ns2d_batch2_step1` | 2 | 1 | OOM | OOM | 31.73 GB used | failed; exact batch 2 is too large |

## Current Defaults

- Burgers: attack batch 256, optimizer batch 32, exact solver-target gradients.
- Darcy: attack batch 256, optimizer batch 32, exact solver-target gradients. Darcy was tested with optimizer batch 64; batch 32 is a smaller, safer optimizer microbatch.
- NS2D: attack batch 1, optimizer batch 1, exact solver rollout gradients. Batch 2 OOMed on the 31.7GB GPU.

New per-task CLI overrides remain available:

- `--burgers-optimizer-batch-size`
- `--darcy-optimizer-batch-size`
- `--ns2d-optimizer-batch-size`

Validation after the correction:

- `adv_robust/bin/python -m py_compile tools/adversarial_training.py` passed.
- `adversarial_training_runs/smoke_full_solver_gradient_field_burgers` confirms the training path runs after removing the objective switch and logs `full_solver_gradient=True`.
