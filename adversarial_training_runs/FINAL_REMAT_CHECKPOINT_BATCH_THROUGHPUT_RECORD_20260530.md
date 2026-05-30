# Final Rematerialization / Checkpoint Batch Throughput Record

Date: 2026-05-30 UTC

This file records the measured rematerialization/checkpoint settings, batch sizes, memory, time, per-sample cost, and unit output for the attack/adversarial-training experiments.

Definitions:

- `unit output` = `attack_samples_per_sec`, i.e. how many samples can be attacked per second.
- `sec/sample` = attack or warm-update time divided by batch size.
- `GiB/sample` = observed peak memory divided by batch size. This is a rough amortized number because model weights, caches, and allocator reserve are shared fixed costs.
- `OOM` means the configuration was actually attempted and failed from GPU memory pressure.

## Final Recommendation

| task / experiment | best setting tested | batch | unit output | why |
|---|---|---:|---:|---|
| NS2D dedicated full-gradient attack | `solver_remat=chunk`, `chunk_steps=20` | 6 | 0.413 samples/s warm | Highest tested warm throughput; batch 7 OOM |
| NS2D adversarial training | `ns2d_solver_remat=chunk`, `chunk_steps=20` | 6 | 0.0269 samples/s | batch 8 and 10 OOM |
| Burgers adversarial training | `burgers_solver_remat=chunk`, `chunk_steps=50` | 768 | 42.807 samples/s | Highest tested throughput; no-remat/all-remat larger variants OOM |
| Darcy/C-flow adversarial training | no rollout remat knob, batch tuning | 384 | 91.994 samples/s | Highest tested throughput; batch 512 OOM |

## NS2D Dedicated Full-Gradient Attack Sweep

Setup: 2D NS, V100 32GB, `loss3/raw_add`, `p=2`, `q=2`, 3 attack steps, 3800 PDE micro-steps per sample/update. This is the earlier direct attack runner sweep.

| config | remat method | batch | status | peak GiB/batch | GiB/sample | warm sec/batch | sec/sample | unit output samples/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `chunk_bs5_steps3` | `chunk` | 5 | passed | 25.18 | 5.04 | 13.844 | 2.769 | 0.361 |
| `chunk_bs6_steps3` | `chunk` | 6 | passed | 28.98 | 4.83 | 14.522 | 2.420 | 0.413 |
| `chunk_bs7_steps3` | `chunk` | 7 | failed OOM | 31.64 | 4.52 | - | - | - |
| `micro_bs4_steps3` | `micro` | 4 | passed | 24.64 | 6.16 | 11.377 | 2.844 | 0.352 |
| `micro_bs5_steps3` | `micro` | 5 | passed | 31.38 | 6.28 | 13.111 | 2.622 | 0.381 |
| `none_bs2_steps3` | `none` | 2 | failed OOM | 10.89 | 5.45 | - | - | - |
| `second_bs5_steps3` | `second` | 5 | passed | 28.18 | 5.64 | 13.890 | 2.778 | 0.360 |
| `second_bs6_steps3` | `second` | 6 | failed OOM | 27.97 | 4.66 | - | - | - |
| `second_bs7_steps3` | `second` | 7 | failed OOM | 31.64 | 4.52 | - | - | - |

NS2D dedicated attack conclusion:

- `none` failed even at batch 1/2 in follow-up checks, so rematerialization is required for this full-gradient attack.
- Among successful configurations, `chunk bs6` is best: `0.413 samples/s` warm throughput.
- Batch 7 OOMed in the direct attack sweep, so 8/10 would not be expected to fit there.

## NS2D Adversarial Training

Setup: adversarial training entry point, `fast_add_linf`, 5 attack steps, optimizer microbatch 1. Tested specifically whether batch 8/10 beat batch 6.

| run | remat/checkpoint method | chunk steps | batch | status | attack sec/batch | sec/sample | peak allocated GiB | reserved GiB | GiB/sample | unit output samples/s |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `remat_sweep_ns2d_chunk_bs6_steps5` | `chunk` | 20 | 6 | passed | 223.209 | 37.2014 | 26.96 | 29.14 | 4.493 | 0.027 |
| `remat_sweep_ns2d_chunk_bs8_steps5` | `chunk` | - | 8 | failed | - | - | - | - | - | - |
| `remat_sweep_ns2d_chunk_bs10_steps5` | `chunk` | - | 10 | failed | - | - | - | - | - | - |

Best measured NS2D Adversarial Training: `remat_sweep_ns2d_chunk_bs6_steps5` with batch `6`, method `chunk`, unit output `0.026881 samples/s`.

## Burgers Adversarial Training

Setup: adversarial training entry point, `fast_replace_linf`, 3 attack steps, optimizer microbatch 32. Tested no remat, chunk remat, and full/all remat.

| run | remat/checkpoint method | chunk steps | batch | status | attack sec/batch | sec/sample | peak allocated GiB | reserved GiB | GiB/sample | unit output samples/s |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `remat_sweep_burgers_chunk_bs768_steps3` | `chunk` | 50 | 768 | passed | 17.941 | 0.0234 | 6.10 | 6.94 | 0.008 | 42.807 |
| `remat_sweep_burgers_none_bs384_steps3` | `none` | 50 | 384 | passed | 11.626 | 0.0303 | 22.56 | 23.00 | 0.059 | 33.029 |
| `remat_sweep_burgers_chunk_bs512_steps3` | `chunk` | 50 | 512 | passed | 17.124 | 0.0334 | 4.07 | 4.65 | 0.008 | 29.899 |
| `remat_sweep_burgers_all_bs512_steps3` | `all` | 50 | 512 | passed | 18.478 | 0.0361 | 25.49 | 27.88 | 0.050 | 27.709 |
| `remat_sweep_burgers_chunk_bs384_steps3` | `chunk` | 50 | 384 | passed | 16.956 | 0.0442 | 3.06 | 3.51 | 0.008 | 22.646 |
| `remat_sweep_burgers_none_bs256_steps3_r2` | `none` | 20 | 256 | passed | 15.056 | 0.0588 | 14.72 | 15.03 | 0.057 | 17.003 |
| `remat_sweep_burgers_none_bs256_steps3` | `none` | - | 256 | failed | - | - | - | - | - | - |
| `remat_sweep_burgers_none_bs512_steps3` | `none` | - | 512 | failed | - | - | - | - | - | - |
| `remat_sweep_burgers_none_bs512_steps3_r2` | `none` | - | 512 | failed | - | - | - | - | - | - |
| `remat_sweep_burgers_all_bs768_steps3` | `all` | - | 768 | failed | - | - | - | - | - | - |
| `remat_sweep_burgers_none_bs768_steps3` | `none` | - | 768 | failed | - | - | - | - | - | - |
| `remat_sweep_burgers_none_bs768_steps3_r2` | `none` | - | 768 | failed | - | - | - | - | - | - |

Best measured Burgers Adversarial Training: `remat_sweep_burgers_chunk_bs768_steps3` with batch `768`, method `chunk`, unit output `42.807019 samples/s`.

## Darcy / C-flow Adversarial Training

Setup: adversarial training entry point, `binary_steepest_replace`, 1 attack step, optimizer microbatch 32. Darcy/C-flow uses JAX CG implicit differentiation, so there is no explicit rollout remat method; we tune batch size.

| run | remat/checkpoint method | chunk steps | batch | status | attack sec/batch | sec/sample | peak allocated GiB | reserved GiB | GiB/sample | unit output samples/s |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `remat_sweep_darcy_bs384_steps1` | `n/a` | - | 384 | passed | 4.174 | 0.0109 | 24.49 | 30.11 | 0.064 | 91.994 |
| `remat_sweep_darcy_bs256_steps1` | `n/a` | - | 256 | passed | 3.729 | 0.0146 | 16.82 | 20.38 | 0.066 | 68.649 |
| `remat_sweep_darcy_bs128_steps1` | `n/a` | - | 128 | passed | 3.173 | 0.0248 | 9.15 | 11.06 | 0.071 | 40.339 |
| `remat_sweep_darcy_bs512_steps1` | `n/a` | - | 512 | failed | - | - | - | - | - | - |

Best measured Darcy / C-flow Adversarial Training: `remat_sweep_darcy_bs384_steps1` with batch `384`, method `n/a`, unit output `91.994284 samples/s`.


## Comparability Check: Why Darcy/C-flow Looked Faster Than Burgers

The first cross-task table compared each task's current default/best adversarial-training attack, not an equal number of attack steps or equal attack objective. That makes the raw `samples/sec` useful for practical scheduling, but not a pure PDE-dimensionality benchmark.

Key difference:

- Burgers best row above used `fast_replace_linf` with `attack_steps=3`. Each step calls the differentiable Burgers solver target and then backpropagates through model + solver.
- Darcy/C-flow used `binary_steepest_replace` with `attack_steps=1`. It computes one gradient score field, chooses pixels by top-k, and then forms one attacked training target.

Follow-up check:

| task | method | batch | attack steps | attack sec/batch | sec/sample | unit output samples/s |
|---|---|---:|---:|---:|---:|---:|
| Burgers | `chunk`, chunk steps 50 | 768 | 1 | 7.382 | 0.00961 | 104.039 |
| Burgers | `chunk`, chunk steps 50 | 768 | 3 | 17.941 | 0.02336 | 42.807 |
| Darcy/C-flow | n/a | 384 | 1 | 4.174 | 0.01087 | 91.994 |

After making Burgers a 1-step attack, Burgers is faster than Darcy/C-flow in unit output: `104.039 samples/s` versus `91.994 samples/s`. So the earlier cross-task ordering was not evidence that the 2D Darcy/C-flow PDE is intrinsically cheaper than 1D Burgers; it mainly reflected different attack algorithms and step counts.

Darcy/C-flow remat note:

- The current Darcy/C-flow path is not an explicit time rollout. It calls a JAX-jitted batched conjugate-gradient elliptic solve and JAX differentiates it by implicit differentiation through another linear solve.
- Because there is no saved sequence of time-step activations like NS2D/Burgers rollouts, the same `chunk/micro/second` rematerialization knob does not naturally exist.
- We can still tune memory/throughput by batch size, JAX memory settings, CG tolerance/maxiter, and possibly by rewriting the Darcy solver with an explicit checkpointed iterative loop, but that would be a different solver implementation and may change numerical behavior/performance.

## Overall Interpretation

- The experiments support the non-linear batch/time/memory observation: increasing batch often improves per-sample time because fixed overhead and GPU parallelism are amortized.
- This only helps until the configuration hits the memory wall. The practical optimum is the largest/highest-throughput configuration that actually passes.
- NS2D is already at the edge at batch 6; batch 8 and 10 OOM in adversarial training.
- Burgers benefits the most from chunk checkpointing: chunk remat makes batch 768 pass and gives the best throughput.
- Darcy/C-flow has no explicit rollout remat in the current code path; batch 384 gives the best measured output, while batch 512 OOMs.

