# NS2D Loss3 Adversarial Training Batch Probe

Date: 2026-06-08 UTC

## Question

Prepare Navier-Stokes 2D adversarial training with the solver-gradient `loss3`
objective, and determine the largest practical attack batch size on the current
single V100 path.

## Observed From Source

- `tools/adversarial_training.py` dispatches NS2D solver-label attacks to
  `ns2d_solver_consistent_attack`, not to the generic model-input solver target.
- The attack variable is the first vorticity frame `x0 = xb[..., 0]`.
- For each attack step, the code computes:
  `x_seq_adv, y_seq_adv = ns2d_solver_pair_from_initial(x0_adv, for_attack=True)`,
  `pred = model(x_seq_adv)`, and `loss = MSE(pred, y_seq_adv)`.
- That loss backpropagates through both the recurrent FNO and the differentiable
  NS rollout into `x0_adv`. This is the NS2D analogue of `loss3`.
- After the attack, the training pair is regenerated without gradient from the
  attacked initial state, so optimizer training uses
  `model(x_train_from_x0_adv) -> solver_y_from_x0_adv`.
- The solver path uses `NavierStokesVorticity2DZongyiETDRK4` with `dt=0.005`,
  `t_final=20`, `diffusivity=1e-5`, and `ns2d_solver_remat=chunk` /
  `chunk_steps=20` by default.

## Prior Evidence

Observed from `adversarial_training_runs/CORRECTED_10STEP_REMAT_CHECKPOINT_RECORD_20260530.md`:

- Hardware/path: the corrected records are for the single 32GB V100 path, not
  an A100/B200 large-memory path.
- Corrected NS2D policy: `fast_add_linf`, `attack_steps=10`,
  `alpha_ratio=0.2`, `ns2d_solver_remat=chunk`, `chunk_steps=20`.
- Batch `6` passed with peak allocated memory `26.96 GiB`.
- Batch `8` and `10` failed OOM.

Observed from `NS2D_GENERALIZATION_AND_REMAT_RECORD_20260530.md`:

- For the direct full-gradient NS2D attack sweep, `chunk` remat batch `6`
  passed and batch `7` OOMed.

Inference from these records:

- The best known production starting point for NS2D loss3 adversarial training
  on this V100 is `ns2d_batch_size=6`, `optimizer_batch_size=1`.
- Batch `7` is worth one confirmation attempt for the current exact code path,
  but batch `8+` should be expected to OOM.
- Older or separate larger-batch notes from A100/B200-era runs must not be mixed
  into this V100 recommendation. They may be useful only after a fresh probe on
  that exact GPU/software path.

## Files Added

- `tools/probe_ns2d_loss3_adversarial_training_batch_20260608.py`
- `tools/run_ns2d_loss3_batch_probe_20260608.sh`
- `tools/run_ns2d_loss3_adversarial_training_20260608.sh`

The probe runs the complete expensive loss3 kernel: differentiable NS rollout,
loss3 backward to `x0_adv`, no-grad attacked solver-pair regeneration, and a
real optimizer update through the recurrent FNO.

## Current Local Constraints

Observed locally on 2026-06-08:

- A Darcy adversarial-training process is active in tmux, so NS2D probing must
  wait for exclusive GPU access.
- The retained NS2D checkpoint path recorded by the training script is missing
  from this working tree:
  `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/best.pt`.
- The default NS2D train/test `.pt` files and the 50 NS2D generalization files
  are also not visible locally. They may need to be restored from git/R2/another
  machine before an official full training run.

Because of the missing local checkpoint/data, the batch probe supports a clearly
marked random same-architecture model for memory-capacity testing only. It must
not be interpreted as a scientific metric run. The production launcher still
uses `tools/adversarial_training.py`, which will fail preflight if the real data
and checkpoint are absent.

## Recommended Training Command

Use:

```bash
tools/run_ns2d_loss3_adversarial_training_20260608.sh
```

This sets:

- `--tasks ns2d`
- `--training-data-mode adv-only`
- `--label-mode solver`
- `--ns2d-attack-method fast_add_linf`
- `--ns2d-attack-steps 10`
- `--ns2d-batch-size 6`
- `--ns2d-optimizer-batch-size 1`
- `--ns2d-epsilon-fraction 0.035`
- `--ns2d-alpha-ratio 0.2`
- `--ns2d-solver-remat chunk`
- `--ns2d-solver-remat-chunk-steps 20`

## Next Run

Run the probe after the GPU is idle:

```bash
tools/run_ns2d_loss3_batch_probe_20260608.sh
```

Default candidates are `4 5 6 7 8`; the script stops at the first OOM and writes
JSON/CSV under:

`forensics/ns2d_loss3_adversarial_training_batch_probe_20260608/`
