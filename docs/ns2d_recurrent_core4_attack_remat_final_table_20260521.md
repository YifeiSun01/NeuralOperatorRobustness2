# NS2D Core4 Attack Remat Final Table

Date: 2026-05-21

This table summarizes existing probes only. No new GPU run was launched.

Target rollout settings: `fixed_step=0.005`, `200` micro-steps per second, target frame index `19`, so the differentiable target solver rollout contains `3800` micro-steps.

## Important Interpretation

The `recompute coverage` column below means explicit `jax.checkpoint` coverage in our code. It is not a directly measured wall-clock percentage, and it is not a simple slowdown multiplier.

- `none`: no explicit checkpoint in our code, so explicit recompute coverage is `0%`. JAX/XLA may still internally rematerialize some operations, but that is compiler-controlled and not directly specified by us.
- `micro`, `chunk=20`, `chunk=100`, and `second`: all 3800 solver micro-steps are inside checkpointed regions, so explicit recompute coverage is conceptually `~100%` of the differentiable solver rollout. They differ in checkpoint granularity, not in whether the rollout is covered.

## Summary Table

| Mode | Explicit recompute coverage | Checkpoint unit | Units over 3800-step target rollout | Largest observed passing batch | First observed failing batch | Peak reserved at largest pass | Step time at largest pass | Throughput at largest pass | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `none` | `0%` explicit | no explicit checkpoint | N/A | `2` | `3` | `10.42 GiB` / `11.19 GB` | `7.76s/update` at batch 2 | `0.258 sample-updates/s` | Fast per step, poor throughput, OOM cliff at batch 3. |
| `micro/original` | `~100%` explicit | `1` micro-step | `3800` checkpointed units | `14` | `15` | `60.47 GiB` / `64.93 GB` at batch 14 | `15.01s/update` measured at batch 12; no 5-step timing at batch 14 | `0.799 sample-updates/s` at batch 12 | Reliable recommendation: batch 12; batch 14 passed one-step but less headroom. |
| `chunk=20` | `~100%` explicit | `20` micro-steps = `0.1s` | `190` checkpointed chunks | `17` | `18` | `73.34 GiB` / `78.74 GB` at batch 17 | `19.41s/update` at batch 17 | `0.876 sample-updates/s` | Highest observed throughput, tight memory headroom. |
| `chunk=100` | `~100%` explicit | `100` micro-steps = `0.5s` | `38` checkpointed chunks | `15` observed | not tested above `15` | `64.89 GiB` / `69.67 GB` at batch 15 | only one-step probe: first update `23.57s`; no 5-step steady timing | not ranked | Memory looks like `second`/`chunk=20`, but steady throughput is unconfirmed. |
| `second` | `~100%` explicit | `200` micro-steps = `1.0s` | `19` checkpointed chunks | `15` | `16` | `64.89 GiB` / `69.67 GB` at batch 15 | `18.68s/update` at batch 15 | `0.803 sample-updates/s` | Middle-ground aggressive setting. |

## What This Means

The checkpointed modes all recompute roughly the solver forward region during backward, but the chunk size changes how many boundaries XLA sees and how it plans buffers. That is why time and memory are not proportional to the number of chunks.

Observed memory inside the passing region is approximately linear for checkpointed modes:

```text
peak_reserved ~= base + 4.2 GiB * attack_batch_size
```

Observed time is sublinear in batch size. Larger batches amortize fixed overhead, so throughput improves until OOM.

Practical choice:

- Safest full sweep: `micro/original`, batch `12`.
- Best observed throughput: `chunk=20`, batch `17`.
- Middle ground: `second`, batch `15`.
- `chunk=100`: needs a 5-step steady benchmark before being recommended.
- Avoid `none` for full sweeps because max batch is only `2`.
