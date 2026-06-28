# 2D NS Recurrent Core4 Attack Remat/Checkpoint Comparison

Date: 2026-05-21

All results below are for the expensive proof path:

- `loss3 + raw_add + MODE_SPEC=all_w`
- `EPSILON=32`, `ALPHA=1`, `p=q=2`
- Solver target frame index `19` with `fixed_step=0.005`, so `3800` micro-steps per target rollout
- A100-SXM4-80GB, PyTorch `2.8.0+cu126`, JAX `0.10.0` GPU backend

## Remat Modes

- `none`: no explicit `jax.checkpoint` in the solver scan. This is fastest per step but stores far more solver activations, so max batch is tiny.
- `micro`: checkpoint/rematerialize every solver micro-step. This was the original implementation.
- `chunk=20`: checkpoint/rematerialize every 20 micro-steps, i.e. every `0.1s` of simulated time.
- `chunk=100`: checkpoint/rematerialize every 100 micro-steps, i.e. every `0.5s` of simulated time.
- `second`: checkpoint/rematerialize one simulated second at a time, i.e. 200 micro-steps per block.

## Observed Upper Bounds And Speed

| Mode | Largest observed passing batch | First failing batch | Peak reserved at max observed pass | Warm step time | Warm sample-throughput | Notes |
|---|---:|---:|---:|---:|---:|---|
| `none` | `2` | `3` | `11.19 GB` | `~7.76s/step` at batch 2 | `~0.26 sample-step/s` | Fast per step but poor throughput because batch is too small. |
| `micro` | `14` | `15` | `64.93 GB` at batch 14 | `~15.0s/update` at batch 12 | `~0.80 sample-update/s` | Original mode. Good throughput, safer full-run batch is 12. |
| `chunk=20` | `17` | `18` | `78.74 GB` decimal / `73.34 GiB` reserved at batch 17 | `~19.41s/update` at batch 17 | `~0.88 sample-update/s` | Best observed sample throughput so far, but it is close to the A100 memory ceiling. |
| `chunk=100` | `15` observed passing | not tested above 15 | `69.67 GB` at batch 15 | one-step `~28.8s` at batch 15; no 5-step warm run yet | not fully measured | Similar to `chunk=20`/`second` on one-step probes. |
| `second` | `15` | `16` | `69.67 GB` at batch 15 | `~18.68s/update` at batch 15 | `~0.80 sample-update/s` | Larger batch than `micro`, but per-update time rises; throughput roughly ties `micro`. |

## Key Evidence Paths

- `none`, batch 2, 5 steps: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_222441_UTC`
- `micro`, batch 12, 5 steps: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_215942_UTC`
- `second`, batch 15, 5 steps: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_222217_UTC`
- `chunk=20`, batch 17, 5 steps: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_225102_UTC`
- `chunk=20`, batch 18, one-step OOM: batch 17 passed at `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_224009_UTC`; the follow-up batch 18 probe failed with OOM.
- `chunk=100`, batch 15, one step: `2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/mode_wwwwwwwwww_p2_q2_20260521_222706_UTC`

## Interpretation

Observed evidence supports the user's concern that checkpoint/rematerialization changes the speed/memory tradeoff. However, no-checkpoint is not best for total throughput on this A100 setup because it only reaches batch 2. The best observed throughput is roughly tied between:

- `micro`, batch 12: about `12 / 15.0 = 0.80` sample-updates/s.
- `second`, batch 15: about `15 / 18.68 = 0.80` sample-updates/s.
- `chunk=20`, batch 17: about `17 / 19.41 = 0.88` sample-updates/s.

`chunk=20` is now the best observed throughput setting, but only by about `9%` over `micro`/`second`, while using much more memory. It should be treated as an aggressive setting rather than the default safe setting.

Timing rows must be read carefully. In `per_step_metrics.csv`, row `k=0` already includes one forward/backward/update proposal from zero perturbation. The final row `k=steps` is a no-backward final evaluation, which is why the last interval is much shorter. For the batch 17 `chunk=20` run, the observed timings were:

- first update at `24.24s`
- warm update intervals about `19.41s`
- final no-backward evaluation interval about `4.99s`
- total 5-update run time about `107.41s`

That explains why one-step wall times, warm update times, and final average times appeared inconsistent when compared directly.

## Practical Recommendation

- If starting now and prioritizing reliability: use `SOLVER_REMAT=micro`, `ATTACK_BATCH_SIZE=12`.
- If willing to run aggressively for highest observed throughput: use `SOLVER_REMAT=chunk`, `SOLVER_REMAT_CHUNK_STEPS=20`, `ATTACK_BATCH_SIZE=17`, and monitor memory closely.
- If wanting a middle ground: use `SOLVER_REMAT=second`, `ATTACK_BATCH_SIZE=15`.
- Avoid `SOLVER_REMAT=none` for full sweeps despite its faster per-step timing, because its max batch is too small and throughput is poor.

## Memory Terminology Clarification

- `allocated` means memory currently held by live PyTorch tensors, plus memory visible to PyTorch's allocator accounting.
- `reserved` means memory PyTorch has reserved from CUDA for its caching allocator. Reserved memory can be larger than allocated because PyTorch keeps blocks for reuse instead of returning them immediately.
- These two numbers do not fully describe JAX/XLA's internal buffer planning. A run can show a modest PyTorch reserved peak and still fail when the next JAX/XLA backward buffer asks CUDA for one additional large contiguous allocation.
- OOM is decided by whether the next requested allocation fits, not by whether the previous peak looks far below 80 GB.
- Decimal GB and binary GiB differ, and tools report them differently. On this machine `nvidia-smi` reports `81920 MiB` total, i.e. 80 GiB-class memory; PyTorch reported total memory around `79.25 GiB`. The benchmark table mixes human-readable decimal GB labels and binary GiB conversions, so compare bytes or the same unit when close to the limit.
- The `none` remat mode with batch 2 reserved only about `11.19 GB`, but batch 3 failed because the no-remat solver backward needed a much larger activation/buffer allocation, not because memory scales linearly from batch 2 to batch 3.
- The `micro` and `chunk` modes change the computational graph and saved activations, so their memory curves are not directly comparable to `none` by simple linear extrapolation.

Observed evidence: `SOLVER_REMAT=none`, batch 2, 5 steps passed with about `11.19 GB` reserved, while batch 3 failed with a requested allocation around `46.91 GiB`. This is normal for a solver-backward graph near a rematerialization boundary: the next batch size can trigger a qualitatively different XLA buffer plan.
