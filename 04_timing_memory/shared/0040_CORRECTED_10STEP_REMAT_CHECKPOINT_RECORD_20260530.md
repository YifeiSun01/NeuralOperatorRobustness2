# Corrected 10-Step Remat / Checkpoint Throughput Record

Date: 2026-05-30 UTC

This record replaces the earlier 1-step/3-step comparison. The corrected attack policy is: about 10 optimization steps; for add-style attacks, `alpha = epsilon / 5` so the perturbation reaches the epsilon boundary around step 5 and then continues optimizing on/near the boundary for the remaining steps.

Corrected policy:

- NS2D: `steepest_add_linf`, `attack_steps=10`, `alpha_ratio=0.2`, `ns2d_solver_remat=chunk`, `chunk_steps=20`.
- Burgers: `steepest_replace_linf`, `attack_steps=10`, `burgers_solver_remat=chunk`, `chunk_steps=50`.
- Darcy/C-flow: `binary_steepest_replace`, `attack_steps=10`. This path uses JAX CG implicit differentiation and has no explicit rollout remat knob; it is tuned by batch size.

Summary CSV: `adversarial_training_runs/corrected_10step_remat_checkpoint_throughput_20260530.csv`

## Final Corrected Recommendation

| task | best tested setting | batch | attack sec/batch | sec/sample | samples/s | peak GiB | status |
|---|---|---:|---:|---:|---:|---:|---|
| ns2d | `steepest_add_linf, remat=chunk, chunk=20` | 6 | 431.943 | 71.9905 | 0.014 | 26.96 | best passed |
| burgers | `steepest_replace_linf, remat=chunk, chunk=50` | 1350 | 53.075 | 0.0393 | 25.436 | 10.73 | best passed |
| darcy_cflow | `binary_steepest_replace, remat=n/a` | 448 | 14.522 | 0.0324 | 30.850 | 24.81 | best passed |

## NS2D 10-Step Steepest Add

| run | batch | status | remat | chunk | alpha ratio | attack sec/batch | sec/sample | samples/s | peak GiB | reserved GiB | GiB/sample | boundary ratio | loss gain |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `remat_sweep_ns2d_chunk_bs6_steps10_alpha0p2` | 6 | passed | `chunk` | 20 | 0.2 | 431.943 | 71.9905 | 0.014 | 26.96 | 29.14 | 4.4929 | 0.873 | 0.031094 |
| `remat_sweep_ns2d_chunk_bs8_steps10_alpha0p2` | 8 | failed | `chunk` | 20 | 0.2 | - | - | - | - | - | - | - | - |
| `remat_sweep_ns2d_chunk_bs10_steps10_alpha0p2` | 10 | failed | `chunk` | 20 | 0.2 | - | - | - | - | - | - | - | - |

## Burgers 10-Step Steepest Replace

| run | batch | status | remat | chunk | alpha ratio | attack sec/batch | sec/sample | samples/s | peak GiB | reserved GiB | GiB/sample | boundary ratio | loss gain |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `remat_sweep_burgers_chunk_bs1350_steps10` | 1350 | passed | `chunk` | 50 | n/a | 53.075 | 0.0393 | 25.436 | 10.73 | 12.19 | 0.0079 | 1.000 | 0.000322 |
| `remat_sweep_burgers_chunk_bs1280_steps10` | 1280 | passed | `chunk` | 50 | n/a | 50.349 | 0.0393 | 25.422 | 10.17 | 11.56 | 0.0079 | 1.000 | 0.000321 |
| `remat_sweep_burgers_chunk_bs1024_steps10` | 1024 | passed | `chunk` | 50 | n/a | 49.301 | 0.0481 | 20.770 | 8.13 | 9.25 | 0.0079 | 1.000 | 0.000325 |
| `remat_sweep_burgers_chunk_bs768_steps10` | 768 | passed | `chunk` | 50 | n/a | 50.119 | 0.0653 | 15.324 | 6.10 | 6.94 | 0.0079 | 1.000 | 0.000322 |

## Darcy/C-flow 10-Step Binary Steepest Replace

| run | batch | status | remat | chunk | alpha ratio | attack sec/batch | sec/sample | samples/s | peak GiB | reserved GiB | GiB/sample | boundary ratio | loss gain |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `remat_sweep_darcy_bs448_steps10` | 448 | passed | `n/a` | - | n/a | 14.522 | 0.0324 | 30.850 | 24.81 | 28.57 | 0.0554 | 6.128 | 0.000000 |
| `remat_sweep_darcy_bs480_steps10` | 480 | passed | `n/a` | - | n/a | 15.649 | 0.0326 | 30.673 | 26.48 | 30.52 | 0.0552 | 6.136 | 0.000000 |
| `remat_sweep_darcy_bs384_steps10` | 384 | passed | `n/a` | - | n/a | 12.773 | 0.0333 | 30.062 | 21.48 | 24.65 | 0.0559 | 6.114 | 0.000000 |
| `remat_sweep_darcy_bs256_steps10` | 256 | passed | `n/a` | - | n/a | 9.486 | 0.0371 | 26.987 | 14.82 | 16.74 | 0.0579 | 6.083 | 0.000000 |
| `remat_sweep_darcy_bs496_steps10` | 496 | failed | `n/a` | - | n/a | - | - | - | - | - | - | - | - |
| `remat_sweep_darcy_bs512_steps10` | 512 | failed | `n/a` | - | n/a | - | - | - | - | - | - | - | - |

## Interpretation

- The earlier 1-step/3-step comparison was not the right experimental protocol. The corrected table above uses 10-step attacks.
- NS2D still tops out at batch 6. Batch 8 and 10 OOM even with chunk remat and alpha ratio 0.2.
- Burgers benefits strongly from chunk checkpointing and can run the full available train batch of 1350 samples for this one-batch sweep. This is the highest tested throughput.
- Darcy/C-flow can run batch 480 for 10-step binary steepest replace, but batch 496 and 512 OOM. The highest unit-time output is batch 448; batch 480 is the largest safe tested batch but is slightly slower per sample.
- Darcy/C-flow still has no explicit `micro/chunk/second` remat because the current solver path is JAX CG implicit differentiation, not an explicit time rollout. A checkpointed Darcy variant would require rewriting the solver loop and revalidating numerical behavior.

