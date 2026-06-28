# NS2D Rematerialization Throughput Check

Date: 2026-05-30 UTC

Setup:
- GPU: Tesla V100-SXM2-32GB
- Attack: 2D NS recurrent core4, `loss3`, `raw_add`, `p=2`, `q=2`
- Grid/input: `256x256`
- Attack steps: `3`
- PDE micro-steps per sample/update: `3800`
- XLA: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`

Main result:
- `solver_remat=none` failed at `batch_size=2` in the original sweep.
- A follow-up baseline run with `solver_remat=none`, `batch_size=1` also failed with OOM while trying to allocate `15.64 GiB`.
- Therefore, under this exact V100 attack configuration, no-remat is not a viable baseline even for batch 1. Checkpoint/rematerialization is required to run the full differentiable attack.

Passed configurations from `v100_remat_throughput_summary.csv`:

| remat | batch | nvidia-smi peak GiB | torch max allocated GB | warm update sec | warm samples/sec | measured samples/sec |
|---|---:|---:|---:|---:|---:|---:|
| micro | 4 | 24.64 | 18.89 | 11.377 | 0.352 | 0.280 |
| micro | 5 | 31.38 | 23.54 | 13.111 | 0.381 | 0.308 |
| second | 5 | 28.18 | 23.54 | 13.890 | 0.360 | 0.297 |
| chunk | 5 | 25.18 | 23.54 | 13.844 | 0.361 | 0.296 |
| chunk | 6 | 28.98 | 27.86 | 14.522 | 0.413 | 0.342 |

Best observed throughput:
- Best warm throughput: `chunk`, `batch_size=6`, `0.413 samples/sec`
- Best measured end-to-end throughput including first compile/update and final eval: `chunk`, `batch_size=6`, `0.342 samples/sec`

Relative to `micro`, `batch_size=4`:

| config | peak memory ratio | warm time ratio | batch ratio | warm throughput ratio |
|---|---:|---:|---:|---:|
| chunk bs5 | 1.02x | 1.22x | 1.25x | 1.03x |
| chunk bs6 | 1.18x | 1.28x | 1.50x | 1.18x |
| micro bs5 | 1.27x | 1.15x | 1.25x | 1.08x |
| second bs5 | 1.14x | 1.22x | 1.25x | 1.02x |

Conclusion:
- Yes, the batch gain and time penalty are not proportional.
- In the passing remat configurations, increasing batch from 4 to 6 raises warm update time only `1.28x`, while batch rises `1.50x`, giving `1.18x` higher warm throughput.
- The best observed setting is `solver_remat=chunk`, `solver_remat_chunk_steps=20`, `batch_size=6`.
- Stronger or differently placed remat is not automatically better: `chunk bs7`, `second bs6/bs7`, and no-remat all OOM in this sweep.
