# NS2D Loss3 Self-Training Epoch Timing

Date: 2026-06-08 UTC

## Goal

Measure how long one Navier-Stokes 2D `loss3` solver-label self-training epoch
takes on the current V100 GPU path.

## Code Path

Observed from `tools/adversarial_training.py`:

- NS2D `loss3` self-training perturbs the initial vorticity frame `x0`.
- For each attack step it rolls out the differentiable NS solver from `x0_adv`.
- The attack loss is `MSE(model(x_seq_adv), y_seq_adv)` and backpropagates
  through the NS solver to `x0_adv`.
- The training pair is then regenerated from the attacked initial state without
  gradient, and the recurrent FNO is updated on that attacked solver pair.

## Timing Harness

Added files:

- `tools/time_ns2d_loss3_self_training_epoch_20260608.py`
- `tools/run_ns2d_loss3_self_training_epoch_timing_20260608.sh`
- `tools/run_ns2d_loss3_epoch_then_batch_probe_20260608.sh`

Default timing settings:

- `num_samples=50`
- `batch_size=6`
- `optimizer_batch_size=1`
- `attack_steps=10`
- `attack_method=fast_add_linf`
- `epsilon_fraction=0.035`
- `alpha_ratio=0.2`
- `ns2d_solver_remat=chunk`
- `chunk_steps=20`

## Prior Estimate

Observed from `adversarial_training_runs/RUNTIME_BLOWUP_DIAGNOSIS_20260530.md`:

- NS2D corrected setting used `50` train samples, batch `6`, `9` batches per
  epoch, and measured step time `434.494 s/batch`.
- That gives `3910.445 s/epoch`, about `65.17 min/epoch`, for the 50-sample
  NS2D self-training timing/evaluation口径.

Observed elsewhere in the ledger/docs:

- The original real NS2D train tensor was `N1150`, shape
  `x=(1150,256,256)`, `y=(1150,256,256,21)`.

Inference:

- If the same per-batch time were applied to all `1150` training samples at
  batch `6`, one full-data epoch would be about `192 * 434.494 s = 23.17 h`.
- The queued 2026-06-08 timing run will directly measure the current code path
  for the default `50`-sample epoch and write an `1150`-sample extrapolation.

## Current Local Constraints

Observed locally on 2026-06-08:

- A Darcy training driver is active, so the NS2D epoch timing is queued and waits
  for exclusive GPU access.
- The retained NS2D checkpoint and real NS2D `.pt` train/generalization files are
  not visible in the current working tree. If still missing when the timing job
  starts, the timing harness will use synthetic `256x256` initial vorticity and a
  random same-architecture model, clearly marked as runtime-only evidence.

## Outputs

Queued output directory:

`forensics/ns2d_loss3_self_training_epoch_timing_20260608/`

Expected files:

- `gpu_preflight.log`
- `logs/wait_for_idle.log`
- `logs/epoch_timing.log`
- `epoch_train_batches.csv`
- `epoch_summary.json`

## Queue Status

Observed on 2026-06-08 UTC:

- tmux session: `ns2d_loss3_epoch_then_batch_20260608`
- Current state: waiting for the active Darcy driver/training process to finish.
- The timing run records GPU preflight before waiting and will start only when no
  matching active training process is present and free GPU memory is at least
  `30000 MiB`.
- After the one-epoch timing finishes, the same tmux chain continues to the
  previously requested batch-size probe.
