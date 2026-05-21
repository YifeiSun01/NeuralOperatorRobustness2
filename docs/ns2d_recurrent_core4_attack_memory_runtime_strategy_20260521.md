# 2D NS Recurrent Core4 Attack Memory/Runtime Strategy

Date: 2026-05-21

## Hardware And Software

Observed GPU path for the attack probes:

- Machine GPU: `NVIDIA A100-SXM4-80GB`.
- `nvidia-smi` capacity shown as `81920 MiB`; PyTorch total memory reported about `79.25 GiB`.
- PyTorch: `2.8.0+cu126`; CUDA runtime: `12.6`; supported arch list includes `sm_80`.
- JAX: `0.10.0`; backend `gpu`; device `CudaDevice(id=0)`.
- JAX allocation defaults in the attack wrapper: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`.

Source path under analysis:

- `2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py`
- Critical solver checkpoint line: `step_once_checked = jax.checkpoint(step_once)` inside `make_ns_rollout_function()`.

## What Was Actually Tested

All batch-size and speed probes below use the expensive path:

- `LOSS_TYPES=loss3`
- `METHODS=raw_add`
- `MODE_SPEC=all_w`
- `P_ORDER=2`, `Q_ORDER=2`
- `EPSILON=32`, `ALPHA=1`
- Solver rollout to zero-based target frame `19`
- `fixed_step=0.005`, so `200` solver micro-steps per simulated second
- For target frame `19`, each sample uses `3800` solver micro-steps per rollout
- Solver trace confirms `requires_grad=True` on the attack update rollout

This is the right path to test for upper-bound memory pressure because it backpropagates through the differentiable solver target.

## Batch-Size/OOM Results

Observed successful probes:

| Attack batch size | Max allocated | Max reserved | One-step runtime including compile/final eval | Output root |
|---:|---:|---:|---:|---|
| 10 | `45.75 GB` | `46.79 GB` | `22.90s` | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215124_UTC` |
| 12 | `54.70 GB` | `55.86 GB` | `24.86s` | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215421_UTC` |
| 14 | `63.66 GB` | `64.93 GB` | `26.79s` | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215525_UTC` |

Observed failed probes:

| Attack batch size | Failure | Output root |
|---:|---|---|
| 15 | JAX backward / DLPack OOM while trying to allocate about `14.79 GiB` | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215632_UTC` |
| 16 | JAX backward / DLPack OOM while trying to allocate about `15.78 GiB` | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215325_UTC` |
| 20 | PyTorch recurrent FNO FFT OOM at `torch.fft.rfft2`; process had about `79.04 GiB` in use and only about `201 MiB` free | `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215231_UTC` |

Observed post-OOM behavior:

- After failed processes exited, `nvidia-smi` returned to `0 MiB / 81920 MiB`.

## Memory Growth Versus Runtime Growth

Observed passing one-step probes show:

| Change | Reserved memory increase | Runtime increase | Sample throughput change, using one-step runtime |
|---|---:|---:|---:|
| batch `10 -> 12` | `46.79 GB -> 55.86 GB`, about `+19.4%` | `22.90s -> 24.86s`, about `+8.6%` | `0.437 -> 0.483` sample-updates/s, about `+10.5%` |
| batch `12 -> 14` | `55.86 GB -> 64.93 GB`, about `+16.2%` | `24.86s -> 26.79s`, about `+7.8%` | `0.483 -> 0.522` sample-updates/s, about `+8.1%` |

Inference from these probes:

- Increasing batch size from `10` to `14` improves sample throughput because runtime grows slower than batch size.
- Memory grows almost linearly and much faster than wall time. The marginal memory cost is roughly `4.5 GB` reserved per additional sample in this path.
- `batch=14` is throughput-friendly but close to the edge; `batch=15` fails because JAX backward needs an additional large allocation.

## Checkpoint/Rematerialization Speed Cost

Observed source:

- The solver uses `jax.checkpoint(step_once)` inside the micro-step scan.
- This reduces stored activations and allows larger batches, but backward recomputes parts of the solver trajectory.

Observed speed benchmark:

- Output root: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215942_UTC`
- Settings: `ATTACK_BATCH_SIZE=12`, `STEPS=5`, `loss3/raw_add/all_w`.
- Peak memory: max allocated `54.71 GB`, max reserved `55.86 GB`.
- First update, including JAX compile/warmup: `20.53s`.
- Warm update steps after compile: about `15.0s` per step for batch size `12`.
- Final no-backward evaluation: about `4.1s`.
- Estimated `100`-step runtime for one `loss3/all_w/raw_add` batch of 12 samples: about `25.2 minutes`.

Comparison to the user's recalled B200 run:

- Recalled B200 behavior: about `7s/step` for batch size `7`, roughly `1.0` sample-update/s.
- Current A100 checkpointed behavior: about `15s/step` for batch size `12`, roughly `0.8` sample-update/s.
- Inference: per optimizer step this A100/checkpointed path is about `2.1x` slower than the recalled B200 number, but because it processes `12` samples instead of `7`, per-sample throughput is closer to about `25%` worse, not `2x` worse.

This supports the user's suspicion: checkpointing probably reduces memory enough to allow bigger batches, but it also slows each update. Whether it is worth it depends on sample-throughput, not only batch size.

## Current Recommendation For Maximum Attacks Per Unit Time

For the currently implemented checkpointed solver path:

- Recommended practical full-run batch size: `ATTACK_BATCH_SIZE=12`.
- Aggressive but observed passing value: `ATTACK_BATCH_SIZE=14`.
- Unsafe values for this path: `15+`.

Why `12` is the default recommendation:

- It uses about `55.86 GB` reserved, leaving real headroom for fragmentation, JAX/PyTorch caches, different methods, and longer runs.
- It has observed warm update speed around `15s/step`.
- It avoids the narrow cliff where `14` passes but `15` fails.

When to use `14`:

- Use `14` only for short, monitored runs when maximizing immediate throughput matters more than stability.
- Watch `batch_memory.csv` and stop if reserved memory creeps higher than the one-step probe.

Avoid starting with `15` or `20`:

- Both have already failed on this A100 setup for the same expensive path.

## How To Improve Further

The actual optimization objective is sample-updates per second, not maximum batch size alone. The next useful experiment is to expose solver checkpointing as a mode and compare throughput directly:

1. Current mode: checkpoint every solver micro-step with `jax.checkpoint(step_once)`.
   - Pros: lower activation memory and larger batch.
   - Cons: recomputation slows backward.

2. No-rematerialization mode: call `step_once` directly without `jax.checkpoint`.
   - Pros: should reduce backward recomputation and possibly lower seconds/step.
   - Cons: will store more activations and likely require a smaller batch, possibly close to the user's remembered `batch=6-7` range or lower.

3. Coarser checkpoint mode: rematerialize larger chunks instead of every micro-step if JAX/exponax structure permits it.
   - Goal: find a middle point between memory and speed.
   - This is likely the best long-term direction, but it requires code changes and a new benchmark.

Benchmark plan for throughput tuning:

- Add a CLI flag such as `--solver-remat micro|none`.
- For `micro`, use the current implementation and benchmark `batch=12` and `batch=14` for `STEPS=20`.
- For `none`, find the largest passing batch size, then benchmark `STEPS=20`.
- Compare `sample_updates_per_second = batch_size * update_steps / elapsed_update_seconds`.
- Choose the configuration with the highest sample-updates/s, not the largest batch.

## Tactical Run Plan

To avoid wasting hours:

1. Screen quickly with `STEPS=20`, `ATTACK_BATCH_SIZE=12`, and only the expensive/important combinations.
2. Use `loss1` and `loss2` as cheaper preliminary signals when possible because they do not require `G(x+delta)` target solve in the current code path.
3. Reserve full `STEPS=100` runs for the combinations that show meaningful growth.
4. Run the most expensive path, `loss3/all_w`, with `ATTACK_BATCH_SIZE=12` unless there is active monitoring and a reason to risk `14`.
5. Add and benchmark a no-rematerialization solver mode before assuming the current checkpointed path is throughput-optimal.

## Bottom Line

Observed evidence says the current checkpointed solver path is stable and memory-efficient enough to run batch `12` comfortably and batch `14` aggressively. It is not obviously optimal for throughput. The user's concern is valid: checkpointing almost certainly trades away speed. The next best engineering step is a controlled benchmark of checkpointed versus non-checkpointed solver modes using the same `loss3/all_w` attack path and comparing sample-updates per second.
