# Adversarial Training Remat / Batch Throughput Sweep

Date: 2026-05-30 UTC

Goal: tune NS2D, Burgers, and Darcy/C-flow adversarial training so that, under the 32GB V100 memory limit, the attack batch size is as large as possible while maximizing samples attacked per second.

## Code Changes

Changed files:

- `solvers.py`
  - Added PyTorch checkpoint/rematerialization support to `solve_burgers_final_batch_with_solver`.
  - Added PyTorch checkpoint/rematerialization support to `solve_ns_zongyi_rollout_batch_with_solver`.
  - Supported modes:
    - Burgers: `none`, `micro`/`step`, `chunk`, `all`/`full`
    - NS2D: `none`, `micro`/`step`, `chunk`, `second`
- `tools/adversarial_training.py`
  - Added CLI options:
    - `--burgers-solver-remat`
    - `--burgers-solver-remat-chunk-steps`
    - `--ns2d-solver-remat`
    - `--ns2d-solver-remat-chunk-steps`
  - Logged attack throughput fields:
    - `attack_sec_per_sample`
    - `attack_samples_per_sec`
    - `step_sec_per_sample`
    - `step_samples_per_sec`

Darcy/C-flow note: the current Darcy/C-flow attack uses a JAX CG solve with implicit differentiation rather than a long explicit time rollout. There is no analogous time-step rematerialization knob in this path, so it was tuned by attack batch size and optimizer microbatch size.

Summary CSV:

- `adversarial_training_runs/adversarial_training_remat_throughput_sweep_20260530_summary.csv`

## NS2D

Adversarial training setup:

- Task: `ns2d`
- Attack method: `fast_add_linf`
- Attack steps: `5`
- Solver remat: `chunk`
- Chunk steps: `20`
- Optimizer microbatch: `1`

Results:

| batch | remat | status | attack sec | sec/sample | attack samples/sec | peak allocated | reserved |
|---:|---|---|---:|---:|---:|---:|---:|
| 6 | chunk | pass | 223.209 | 37.201 | 0.026881 | 26.96 GiB | 29.14 GiB |
| 8 | chunk | OOM | - | - | - | - | - |
| 10 | chunk | OOM | - | - | - | - | - |

OOM details:

- `batch=8` failed during FNO forward with only about `338.50 MiB` free and a requested `480 MiB` allocation.
- `batch=10` failed during FNO forward with only about `36.50 MiB` free and a requested `152 MiB` allocation.

Conclusion:

- For this adversarial training entry point, `batch=8` and `batch=10` are not more cost-effective because they do not fit in memory.
- Best tested NS2D setting: `batch_size=6`, `solver_remat=chunk`, `solver_remat_chunk_steps=20`.
- This matches the earlier dedicated NS2D full-gradient attack conclusion: batch 6 is the practical top point on this 32GB V100 for the tested setup.

## Burgers

Adversarial training setup:

- Task: `burgers`
- Attack method: `fast_replace_linf`
- Attack steps: `3`
- Optimizer microbatch: `32`

Passed configurations, sorted by attack throughput:

| batch | remat | status | attack sec | sec/sample | attack samples/sec | peak allocated | reserved |
|---:|---|---|---:|---:|---:|---:|---:|
| 768 | chunk | pass | 17.941 | 0.0234 | 42.807 | 6.10 GiB | 6.94 GiB |
| 384 | none | pass | 11.626 | 0.0303 | 33.029 | 22.56 GiB | 23.00 GiB |
| 512 | chunk | pass | 17.124 | 0.0334 | 29.899 | 4.07 GiB | 4.65 GiB |
| 512 | all | pass | 18.478 | 0.0361 | 27.709 | 25.49 GiB | 27.88 GiB |
| 384 | chunk | pass | 16.956 | 0.0442 | 22.646 | 3.06 GiB | 3.51 GiB |
| 256 | none | pass | 15.056 | 0.0588 | 17.003 | 14.72 GiB | 15.03 GiB |

Failed configurations:

- `batch=512`, `remat=none`: OOM
- `batch=768`, `remat=none`: OOM
- `batch=768`, `remat=all`: OOM

Conclusion:

- Checkpointing helps Burgers a lot: `none bs512` OOM, but `chunk bs512` passes, and `chunk bs768` passes.
- Best tested Burgers setting: `batch_size=768`, `burgers_solver_remat=chunk`, `burgers_solver_remat_chunk_steps=50`.
- `chunk` is better than `all` here. Full-rollout checkpointing can still OOM during recomputation, while chunked checkpointing fits and gives the highest throughput.

## Darcy / C-flow

Adversarial training setup:

- Task: `darcy`
- Attack method: `binary_steepest_replace`
- Attack steps: `1`
- Optimizer microbatch: `32`
- Solver path: JAX CG / implicit differentiation, no rollout remat knob

Results:

| batch | remat | status | attack sec | sec/sample | attack samples/sec | peak allocated | reserved |
|---:|---|---|---:|---:|---:|---:|---:|
| 384 | n/a | pass | 4.174 | 0.0109 | 91.994 | 24.49 GiB | 30.11 GiB |
| 256 | n/a | pass | 3.729 | 0.0146 | 68.649 | 16.82 GiB | 20.38 GiB |
| 128 | n/a | pass | 3.173 | 0.0248 | 40.339 | 9.15 GiB | 11.06 GiB |
| 512 | n/a | OOM | - | - | - | - | - |

Conclusion:

- Darcy/C-flow throughput improves strongly with batch size until memory runs out.
- Best tested Darcy/C-flow setting: `batch_size=384`, `optimizer_batch_size=32`.
- `batch=512` OOMs during FNO forward, so 384 is the best tested safe point.

## Final Recommended Settings

| task | recommended attack batch | remat/checkpoint setting | optimizer microbatch | reason |
|---|---:|---|---:|---|
| NS2D | 6 | `ns2d_solver_remat=chunk`, `chunk_steps=20` | 1 | batch 8/10 OOM; batch 6 is fastest tested safe point |
| Burgers | 768 | `burgers_solver_remat=chunk`, `chunk_steps=50` | 32 | highest tested throughput; larger no-remat/all-remat variants OOM |
| Darcy/C-flow | 384 | n/a | 32 | highest tested throughput; batch 512 OOM |

Main interpretation:

- Yes, larger batch can be more cost-effective because time does not increase linearly with batch size.
- But the optimum is bounded by actual OOM behavior.
- For NS2D, batch 8/10 are not better because they do not fit.
- For Burgers, chunk checkpointing makes a much larger batch feasible and improves throughput.
- For Darcy/C-flow, batch scaling helps up to 384, then 512 OOMs.
