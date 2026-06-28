# NS2D Core4 Attack Remat Scaling Summary

Date: 2026-05-21

This document summarizes existing GPU benchmark outputs only. No new long attack sweep was launched for this summary. Plots and CSVs were generated on CPU from existing `summary.json`, `batch_memory.csv`, and `per_step_metrics.csv` files.

## Source Files

- Main numeric table: `docs/ns2d_recurrent_core4_attack_remat_scaling_20260521.csv`
- Linear memory fits: `docs/ns2d_recurrent_core4_attack_remat_scaling_fits_20260521.csv`
- Memory plot: `docs/figures/ns2d_core4_attack_remat_memory_vs_batch_20260521.png`
- Time plot: `docs/figures/ns2d_core4_attack_remat_time_vs_batch_20260521.png`
- Throughput plot: `docs/figures/ns2d_core4_attack_remat_throughput_vs_batch_20260521.png`

## Memory Scaling

Observed passing points are almost linear within each remat mode's passing range. In binary units, the fitted slope is about `4.2-4.3 GiB` per additional attack sample for the checkpointed/rematerialized modes.

| Mode | Passing batches used for memory trend | Approx peak reserved fit | Largest observed pass | First observed fail | Interpretation |
|---|---:|---:|---:|---:|---|
| `none` | `1, 2` | `0.55 + 4.94 * batch GiB` | `2` | `3` | Locally small and fast, but no-remat backward hits a large XLA buffer jump at batch 3. |
| `micro/original` | `1, 6, 10, 12, 14` | `1.59 + 4.21 * batch GiB` | `14` | `15` | Very linear over passing points; original safe setting is batch 12. |
| `second` | `12, 14, 15` | `0.64 + 4.28 * batch GiB` | `15` | `16` | Similar memory slope to micro/chunk, slightly larger usable batch. |
| `chunk=20` | `12, 15, 16, 17` | `1.01 + 4.25 * batch GiB` | `17` | `18` | Best observed throughput, but batch 17 is close to the memory ceiling. |
| `chunk=100` | `12, 15` | `0.56 + 4.29 * batch GiB` | `15` observed | not bounded above 15 | Memory looks similar to `second`/`chunk=20` at sampled points; fewer tests were run. |

Observed peak reserved examples:

| Mode | Batch | Peak reserved |
|---|---:|---:|
| `none` | `2` | `10.42 GiB` / `11.19 GB` |
| `micro/original` | `12` | `52.02 GiB` / `55.86 GB` |
| `micro/original` | `14` | `60.47 GiB` / `64.93 GB` |
| `second` | `15` | `64.89 GiB` / `69.67 GB` |
| `chunk=20` | `17` | `73.34 GiB` / `78.74 GB` |


## Remat Granularity And Recompute

With `fixed_step=0.005`, the solver has `200` micro-steps per physical second. The current default target frame index is `19`, so the expensive target rollout has `19 * 200 = 3800` solver micro-steps.

| Mode | Checkpoint/remat unit | Units over 19s target rollout | What is saved conceptually | Recompute implication |
|---|---:|---:|---|---|
| `none` | no explicit `jax.checkpoint` | not chunked by our code | XLA is free to keep many backward activations/buffers | Lowest explicit recompute, but can require huge backward memory; batch 3 OOMed. |
| `micro/original` | 1 micro-step | 3800 units | very fine-grained step boundaries | More recomputation during backward, lower activation storage; original reliable mode. |
| `chunk=20` | 20 micro-steps = 0.1s | 190 chunks | chunk boundaries every 20 micro-steps | Recomputes inside 20-step chunks during backward; best observed throughput at batch 17. |
| `chunk=100` | 100 micro-steps = 0.5s | 38 chunks | chunk boundaries every 100 micro-steps | Similar sampled memory to `chunk=20`/`second`; only one-step probes were run, so steady throughput is not confirmed. |
| `second` | 200 micro-steps = 1.0s | 19 chunks | one boundary per physical second | Coarser remat than chunk=100; batch 15 steady probe passed. |

The important point: checkpointing usually causes approximately one extra forward-style recomputation of the checkpointed region during backward. It does not mean `chunk=20` is 20x slower or `second` is 200x slower. Larger chunks save fewer internal boundaries but can let XLA use a different fused buffer plan. That is why the time increase is not proportional to the apparent memory change.

`chunk=100` status: tested at batch 12 and batch 15 for one-step memory. It showed `52.02 GiB` reserved at batch 12 and `64.89 GiB` reserved at batch 15, essentially the same sampled memory curve as `second` and `chunk=20`. It does not yet have a 5-step steady timing probe, so it is not ranked for final throughput.

## Time And Throughput Scaling

Stable timing is only available from 5-step probes. One-step probes include first-update/JIT/allocator overhead and should not be used alone for throughput conclusions.

| Mode | Batch | Warm update time | Throughput |
|---|---:|---:|---:|
| `none` | `2` | `7.76s/update` | `0.258 sample-updates/s` |
| `micro/original` | `12` | `15.01s/update` | `0.799 sample-updates/s` |
| `second` | `12` | `15.88s/update` | `0.756 sample-updates/s` |
| `second` | `15` | `18.68s/update` | `0.803 sample-updates/s` |
| `chunk=20` | `12` | `16.04s/update` | `0.748 sample-updates/s` |
| `chunk=20` | `17` | `19.41s/update` | `0.876 sample-updates/s` |

Inference from observed timing:

- Time per update grows sublinearly with batch in the tested range. For example, `chunk=20` goes from batch 12 to batch 17: batch increases by `41.7%`, while warm update time increases from `16.04s` to `19.41s`, about `21.0%`.
- Because time grows slower than batch, throughput improves with larger batch until memory becomes the limiting factor.
- `none` is fastest per update, but its max batch is too small, so its throughput is poor.
- `chunk=20`, batch 17 is the best observed throughput setting, but it leaves little memory headroom.

## Why This Is Not Purely Linear

Within passing points, peak reserved memory is almost linear: roughly a fixed base plus about `4.2 GiB` per attack sample for checkpointed modes.

However, the OOM boundary is not linear. Reasons:

- PyTorch `reserved` is allocator-cache accounting, not the whole CUDA/JAX/XLA memory story.
- JAX/XLA can choose a different buffer plan when batch size changes.
- The next allocation may need one large contiguous buffer; OOM is decided by whether that request fits, not by the previous run's peak reserved number.
- Remat/checkpoint settings change the graph and the saved activations, so different modes can have different OOM cliffs even if their passing-point slopes look similar.

The clearest example is `none`: batch 2 passed with only `10.42 GiB` reserved, but batch 3 failed because no-remat backward attempted a much larger allocation. That is a discontinuity from the XLA backward buffer plan, not a contradiction in the memory table.

## Practical Reading

- If you want reliable full sweeps: use `micro/original`, batch `12`.
- If you want highest observed throughput and can monitor closely: use `chunk=20`, batch `17`.
- If you want a middle ground: use `second`, batch `15`.
- `chunk=100` has memory probes at batch `12` and `15`, but no 5-step steady speed probe yet, so it is not the current recommendation.
- Do not choose based on per-update seconds alone. Choose based on sample-updates per second and memory headroom.
