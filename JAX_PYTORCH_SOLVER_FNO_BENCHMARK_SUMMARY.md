# JAX vs PyTorch: Solver and FNO Timing/Memory Benchmark Summary

This note summarizes the benchmark results discussed in this repo for:

- Frameworks: PyTorch and JAX
- Problems: 1D Burgers and 2D Navier-Stokes
- Computation types: numerical solver, FNO model inference, FNO model training, inverse/backward-through-solver
- Measurements: forward time, backward time, optimizer/update time, total time, GPU memory, and scaling with `t_final`

## Source Files

Solver forward-only:

```text
benchmark_results/torch_vs_exponax_solver_comparison_full_tfinal_multi10.csv
benchmark_results/torch_vs_exponax_solver_comparison_ns_t2_multi10.csv
benchmark_results/torch_vs_exponax_solver_comparison_ns_t5_multi10.csv
benchmark_results/torch_vs_exponax_solver_comparison_ns_t10_multi10.csv
```

Solver inverse/backward:

```text
benchmark_results/inverse_solver_demo_100steps_full_forward_params_manual/summary.json
benchmark_results/inverse_solver_demo_100steps_full_forward_params_manual/metrics_all.csv
benchmark_results/inverse_solver_demo_ns_t2_100steps_manual/metrics_all.csv
benchmark_results/inverse_solver_demo_ns_t5_100steps_manual/metrics_all.csv
benchmark_results/inverse_solver_demo_ns_t10_100steps_manual/metrics_all.csv
```

FNO inference:

```text
benchmark_results/fno_inference_burgers_trainbatch64_0507/summary.json
benchmark_results/fno_inference_burgers_trainbatch64_0507/inference_metrics.csv
benchmark_results/fno_inference_ns_trainbatch8_0507/summary.json
benchmark_results/fno_inference_ns_trainbatch8_0507/inference_metrics.csv
```

FNO training:

```text
1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/training_logs/memory_phases_pytorch.csv
1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/training_logs/memory_phases_jax_real_imag.csv
fno_training_runs/ns_m12_w20_profile_500/ns_real_initial_laxmap/ns_2d/memory/memory_phases_pytorch.csv
fno_training_runs/ns_m12_w20_profile_500/ns_real_initial_laxmap/ns_2d/memory/memory_phases_jax_real_imag.csv
```

Run-level GPU monitor logs:

```text
run_logs/training_fno1d_burgers/
run_logs/training_fno2d_ns/
run_logs/fno_inference_burgers_trainbatch64_0507/
run_logs/fno_inference_ns_trainbatch8_0507/
run_logs/inverse_solver_full_forward_params_100step_manual/
run_logs/inverse_solver_ns_t2_100step_manual/
run_logs/inverse_solver_ns_t5_100step_manual/
run_logs/inverse_solver_ns_t10_100step_manual/
```

## Measurement Notes

- PyTorch memory is often reported as `torch_peak_allocated_mib`, which is the PyTorch allocator peak. This is relatively clean for tensor memory.
- JAX memory is usually reported from `nvidia-smi` global GPU memory because there is no direct equivalent allocator statistic in these logs.
- Therefore PyTorch and JAX memory are not always identical measurement scopes. Still, the large trends are stable.
- JAX backward timing in training/inverse uses `value_and_grad`, which includes forward evaluation. Where noted, adjusted JAX backward is:

```text
adjusted_backward = value_and_grad_time - forward_time
```

- JAX first-run timings include JIT compilation and should be separated from steady-state timings.

## Checkpoints

FNO1d Burgers:

```text
PyTorch: 1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt
JAX:     1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/jax_real_imag_fno1d_500.pkl
```

FNO2d NS:

```text
PyTorch: fno_training_runs/ns_m12_w20_profile_500/ns_real_initial_laxmap/ns_2d/checkpoints/fno2d_pytorch.pt
JAX:     fno_training_runs/ns_m12_w20_profile_500/ns_real_initial_laxmap/ns_2d/checkpoints/fno2d_jax_real_imag.pkl
```

## FNO Inference, Same Batch Size As Training

For fair comparison with training:

- Burgers FNO inference was rerun with `batch_size=64`.
- NS FNO inference was rerun with `batch_size=8`.

### Burgers FNO Inference, Batch 64

| Framework | First Run | Steady Forward | Global Peak | PyTorch Alloc Peak | Relative L2 |
|---|---:|---:|---:|---:|---:|
| PyTorch | 0.2031 s | 0.003645 s | 747 MiB | 140.2 MiB | 0.01806 |
| JAX real/imag | 3.2866 s | 0.001957 s | 717 MiB | n/a | 0.01702 |

Steady-state ratio:

```text
JAX is about 1.86x faster than PyTorch for Burgers FNO inference.
```

Visualization:

```text
benchmark_results/fno_inference_burgers_trainbatch64_0507/visualizations/burgers_1d_batch64_first5_comparison.png
```

### NS FNO Inference, Batch 8

| Framework | First Run | Steady Forward | Global Peak | PyTorch Alloc Peak | Relative L2 |
|---|---:|---:|---:|---:|---:|
| PyTorch | 0.2966 s | 0.06334 s | 1241 MiB | 534.7 MiB | 0.12831 |
| JAX real/imag | 7.6469 s | 0.05402 s | 1533 MiB | n/a | 0.12870 |

Steady-state ratio:

```text
JAX is about 1.17x faster than PyTorch for NS FNO inference.
```

Visualization:

```text
benchmark_results/fno_inference_ns_trainbatch8_0507/visualizations/ns_2d_batch8_first5_comparison.png
```

For NS, the FNO input is not a single initial condition. The recurrent FNO uses:

```text
input:  y[..., 0:10]
target: y[..., 10:20]
```

The visualization panel named `initial last` means the last frame of the input window, `y[..., 9]`, not the original first frame `y[..., 0]`.

## FNO Training vs FNO Inference

### Burgers FNO, Batch 64

| Framework | Inference Forward | Train Forward | Train Backward | Optimizer | Forward / Inference | Backward / Inference |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch | 0.003645 s | 0.002747 s | 0.003920 s | 0.000977 s | 0.75x | 1.08x |
| JAX real/imag | 0.001957 s | 0.002002 s | 0.005371 s raw | 0.001996 s | 1.02x | 2.74x raw |
| JAX real/imag adjusted | 0.001957 s | 0.002002 s | 0.003370 s adjusted | 0.001996 s | 1.02x | 1.72x adjusted |

Memory:

| Framework | Inference Memory | Train Forward Memory | Train Backward Memory | Forward / Inference | Backward / Inference |
|---|---:|---:|---:|---:|---:|
| PyTorch | 140.2 MiB | 505.8 MiB | 505.8 MiB | 3.61x | 3.61x |
| JAX real/imag | 717.0 MiB | 2733.0 MiB | 2733.0 MiB | 3.81x | 3.81x |

### NS FNO, Batch 8

| Framework | Inference Forward | Train Forward | Train Backward | Optimizer | Forward / Inference | Backward / Inference |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch | 0.06334 s | 0.05882 s | 0.10854 s | 0.00291 s | 0.93x | 1.71x |
| JAX real/imag | 0.05402 s | 0.06118 s | 0.14163 s raw | 0.00348 s | 1.13x | 2.62x raw |
| JAX real/imag adjusted | 0.05402 s | 0.06118 s | 0.08045 s adjusted | 0.00348 s | 1.13x | 1.49x adjusted |

Memory:

| Framework | Inference Memory | Train Forward Memory | Train Backward Memory | Forward / Inference | Backward / Inference |
|---|---:|---:|---:|---:|---:|
| PyTorch | 534.7 MiB | 11664.6 MiB | 11724.0 MiB | 21.82x | 21.93x |
| JAX real/imag | 1533.0 MiB | 30570.3 MiB | 30583.0 MiB | 19.94x | 19.95x |

Conclusion:

```text
FNO training takes roughly 1-3x the inference time for one batch.
FNO training memory can be much larger than inference memory.
Burgers: about 3.6-3.8x inference memory.
NS: about 20-22x inference memory.
```

## FNO Training Total Time

| Problem | Framework | Total Training Time | Test Relative L2 | Peak Memory |
|---|---:|---:|---:|---:|
| Burgers | PyTorch | 80.94 s | 0.01752 | 1407 MiB global whole-run, 505.8 MiB profiled train |
| Burgers | JAX real/imag | 89.16 s | 0.01598 | 2735 MiB global whole-run, 2733 MiB profiled train |
| NS | PyTorch | 14618.47 s, 4h03m38s | 0.15366 | 13987 MiB global whole-run, 11724 MiB profiled train |
| NS | JAX real/imag | 12113.28 s, 3h21m53s | 0.15161 | 30651 MiB global whole-run, 30597 MiB profiled train |

Overall:

```text
NS FNO training: JAX was faster but used much more memory.
Burgers FNO training: PyTorch was slightly faster in total time, while JAX got slightly lower test relative L2.
```

## Solver Forward-Only

### Burgers, `nx=1024`, `t_final=1`, `dt=0.001`

| Framework | First Run | Steady Forward | Memory |
|---|---:|---:|---:|
| PyTorch solver | 0.7496 s | 0.6394 s | 0.45 MiB PyTorch alloc, 8 MiB global delta |
| JAX/Exponax solver | 0.5726 s | 0.2711 s | about 19-20 MiB global delta |

### NS, `nx=256`, `t_final=20`, `dt=0.005`

| Framework | First Run | Steady Forward | Memory |
|---|---:|---:|---:|
| PyTorch solver | 5.8473 s | 5.5949 s | 33.82 MiB PyTorch alloc, 64 MiB global delta |
| JAX/Exponax solver | 4.5333 s | 3.6890 s | about 45.6 MiB global delta |

Conclusion:

```text
Solver pure forward generally uses less memory than FNO inference.
FNO inference is much faster but not more memory-efficient in pure forward.
```

## Solver Inverse/Backward, Full-Size Run

### Burgers, `nx=1024`, 100 inverse steps

| Framework | Total Time | Forward / Step | Backward / Step | Optimizer / Step | Final Loss |
|---|---:|---:|---:|---:|---:|
| PyTorch solver | 241.93 s | 0.894 s | 1.384 s | 0.00138 s | 0.03762 |
| JAX/Exponax solver | 51.24 s | 0.137 s | 0.142 s adjusted | 0.00095 s | 0.03762 |

Memory:

| Framework | Forward Memory | Backward Memory | Global Peak |
|---|---:|---:|---:|
| PyTorch solver | 53.55 MiB alloc | 53.53 MiB alloc | 565 MiB global |
| JAX/Exponax solver | n/a | n/a | 621 MiB global |

### NS, `nx=256`, `t_final=20`, 100 inverse steps

| Framework | Total Time | Forward / Step | Backward / Step | Optimizer / Step | Final Loss |
|---|---:|---:|---:|---:|---:|
| PyTorch solver | 1938.77 s, 32m19s | 7.929 s | 11.127 s | 0.00743 s | 1.06545 |
| JAX/Exponax solver | 546.80 s, 9m07s | 1.836 s | 1.665 s adjusted | 0.00118 s | 0.83047 |

Memory:

| Framework | Forward Memory | Backward Memory | Global Peak |
|---|---:|---:|---:|
| PyTorch solver | 21554.7 MiB, 21.05 GiB | 21550.9 MiB, 21.05 GiB | 24613 MiB, 24.04 GiB |
| JAX/Exponax solver | n/a | n/a | 57423 MiB, 56.08 GiB |

Conclusion:

```text
Solver inverse/backward is dramatically slower than FNO backward.
For NS t=20, solver inverse/backward also uses more memory than FNO training.
```

## NS Solver Scaling With `t_final`

For NS, `dt=0.005`, so:

```text
t_final=2  -> 400 solver steps
t_final=5  -> 1000 solver steps
t_final=10 -> 2000 solver steps
t_final=20 -> 4000 solver steps
```

### NS Solver Inverse Timing

| t_final | Steps | Framework | Forward / Step | Backward / Step | Optimizer / Step |
|---:|---:|---|---:|---:|---:|
| 2 | 400 | PyTorch | 0.678 s | 0.882 s | 0.0013 s |
| 2 | 400 | JAX | 0.190 s | 0.176 s adjusted | 0.0010 s |
| 5 | 1000 | PyTorch | 1.750 s | 2.368 s | 0.0017 s |
| 5 | 1000 | JAX | 0.462 s | 0.420 s adjusted | 0.0010 s |
| 10 | 2000 | PyTorch | 3.697 s | 5.084 s | 0.0027 s |
| 10 | 2000 | JAX | 0.925 s | 0.826 s adjusted | 0.0010 s |
| 20 | 4000 | PyTorch | 7.929 s | 11.127 s | 0.0074 s |
| 20 | 4000 | JAX | 1.836 s | 1.665 s adjusted | 0.0012 s |

The time is close to linear in `t_final`.

### NS Solver Inverse Total Time

| t_final | PyTorch Total | JAX Total |
|---:|---:|---:|
| 2 | 167.9 s | 70.9 s |
| 5 | 427.4 s | 145.4 s |
| 10 | 898.4 s | 280.9 s |
| 20 | 1938.8 s | 546.8 s |

Observation:

```text
JAX total inverse time is much lower than PyTorch.
The solver inverse total time grows strongly with t_final.
```

### NS Solver Inverse Memory, GiB

FNO NS training backward memory:

```text
PyTorch FNO training backward: 11724 MiB = 11.45 GiB
JAX FNO training backward:     30583 MiB = 29.87 GiB
```

Solver backward memory:

| t_final | Steps | PyTorch Solver Backward | vs PyTorch FNO Training | JAX Solver Backward | vs JAX FNO Training |
|---:|---:|---:|---:|---:|---:|
| 2 | 400 | 2.12 GiB | 0.19x | 7.00 GiB | 0.23x |
| 5 | 1000 | 5.27 GiB | 0.46x | 14.50 GiB | 0.49x |
| 10 | 2000 | 10.53 GiB | 0.92x | 28.36 GiB | 0.95x |
| 20 | 4000 | 21.05 GiB | 1.84x | 56.08 GiB | 1.88x |

Observation:

```text
Short t_final: solver backward uses less memory than FNO training.
Around t_final=10: solver backward and FNO training memory are similar.
At t_final=20: solver backward uses about 1.8-1.9x FNO training memory.
```

## Solver Backward Time vs FNO Backward Time

NS FNO training backward, batch 8:

```text
PyTorch FNO backward: 0.10854 s / batch8
JAX FNO adjusted backward: 0.08045 s / batch8
```

NS solver backward compared to FNO backward:

| t_final | Steps | Framework | Solver Backward | Solver / FNO Batch Backward | Solver / FNO Per-Sample Backward |
|---:|---:|---|---:|---:|---:|
| 2 | 400 | PyTorch | 0.882 s | 8.12x | 65.0x |
| 2 | 400 | JAX | 0.176 s | 2.19x | 17.5x |
| 5 | 1000 | PyTorch | 2.368 s | 21.81x | 174.5x |
| 5 | 1000 | JAX | 0.420 s | 5.22x | 41.7x |
| 10 | 2000 | PyTorch | 5.084 s | 46.84x | 374.8x |
| 10 | 2000 | JAX | 0.826 s | 10.27x | 82.2x |
| 20 | 4000 | PyTorch | 11.127 s | 102.52x | 820.2x |
| 20 | 4000 | JAX | 1.665 s | 20.70x | 165.6x |

Observation:

```text
Solver backward time grows with the number of solver time steps.
FNO backward time is fixed by the neural network depth and batch size.
Therefore the solver backward / FNO backward time ratio grows rapidly with t_final.
```

## FNO vs Solver, Forward Inference

For fair timing, FNO batch timings are converted to per-sample time.

### Burgers

| Framework | Solver Forward / Sample | FNO Inference / Sample | Solver / FNO |
|---|---:|---:|---:|
| PyTorch | 0.639 s | 0.000057 s | 11227x |
| JAX | 0.271 s | 0.000031 s | 8865x |

### NS, `t_final=20`

| Framework | Solver Forward / Sample | FNO Inference / Sample | Solver / FNO |
|---|---:|---:|---:|
| PyTorch | 5.595 s | 0.00792 s | 707x |
| JAX | 3.689 s | 0.00675 s | 546x |

Observation:

```text
FNO is a very fast surrogate for forward inference.
The speedup is hundreds to thousands of times depending on problem and framework.
```

## FNO vs Solver Memory

### Pure Forward

| Problem | Framework | Solver Forward Memory | FNO Inference Memory | Comment |
|---|---|---:|---:|---|
| Burgers | PyTorch | 0.45 MiB alloc | 140.2 MiB alloc | Solver much smaller |
| Burgers | JAX | about 20 MiB global delta | 717 MiB global | Solver much smaller |
| NS | PyTorch | 33.8 MiB alloc | 534.7 MiB alloc | Solver much smaller |
| NS | JAX | about 46 MiB global delta | 1533 MiB global | Solver much smaller |

Observation:

```text
FNO forward inference is much faster than solver forward, but it is not more memory efficient.
For pure forward, solver memory is usually lower.
```

### Backward/Inverse

| Problem | Framework | Solver Inverse/Backward Memory | FNO Training/Backward Memory | Solver / FNO |
|---|---|---:|---:|---:|
| Burgers | PyTorch | 53.5 MiB | 505.8 MiB | 0.11x |
| Burgers | JAX | 621 MiB global | 2733 MiB global | 0.23x |
| NS, t=20 | PyTorch | 21.05 GiB | 11.45 GiB | 1.84x |
| NS, t=20 | JAX | 56.08 GiB | 29.87 GiB | 1.88x |

Observation:

```text
Burgers is small, so solver inverse still uses less memory than FNO training.
NS t=20 is large, so solver inverse uses about 1.8-1.9x FNO training memory.
```

## JAX vs PyTorch Summary

### Time

Across most steady-state computations:

```text
JAX is faster than PyTorch.
```

Strong examples:

```text
NS solver inverse t=20:
PyTorch forward 7.929 s, backward 11.127 s
JAX forward     1.836 s, backward 1.665 s

NS FNO inference batch8:
PyTorch 0.06334 s
JAX     0.05402 s

Burgers FNO inference batch64:
PyTorch 0.003645 s
JAX     0.001957 s
```

Exceptions and caveats:

```text
JAX first run includes JIT compilation and can be much slower.
Burgers FNO training total time was slightly faster in PyTorch than JAX in this run.
```

### Memory

General trend:

```text
JAX usually uses more GPU memory than PyTorch, especially for training and solver inverse/backward.
```

Examples:

```text
NS FNO training:
PyTorch about 11.45 GiB
JAX about 29.87 GiB

NS solver inverse t=20:
PyTorch about 21.05 GiB
JAX about 56.08 GiB
```

Caveat:

```text
JAX memory is global nvidia-smi memory, while PyTorch often has allocator-level statistics.
The exact numbers are not perfectly identical scopes, but the large trend is clear.
```

## Effect of `t_final`

### Solver

Solver forward/backward time is strongly affected by `t_final` because more physical time means more solver steps.

For NS with `dt=0.005`:

```text
t_final=2  -> 400 steps
t_final=5  -> 1000 steps
t_final=10 -> 2000 steps
t_final=20 -> 4000 steps
```

Solver inverse time and memory both grow strongly with `t_final`.

### FNO

FNO inference/training time is not directly controlled by solver `t_final` once the model architecture and input/output window are fixed.

For the NS FNO used here:

```text
input window:  Tin=10
output window: Tout=10
architecture: fixed recurrent FNO2d
```

So FNO cost is mostly controlled by:

```text
batch size
grid size
model width/modes/layers
number of recurrent output frames
framework implementation
```

It is not linearly tied to physical `t_final` in the way the numerical solver is.

## Main Conclusions

1. FNO forward inference is much faster than numerical solver forward.
   - Burgers: thousands to ten-thousands times faster per sample.
   - NS t=20: hundreds of times faster per sample.

2. FNO forward inference is not necessarily more memory efficient.
   - Pure solver forward often uses much less memory than FNO inference.
   - FNO advantage is primarily speed, not pure-forward memory.

3. FNO training uses much more memory than FNO inference.
   - Burgers: about 3.6-3.8x inference memory.
   - NS: about 20-22x inference memory.

4. Solver backward/inverse time grows strongly with `t_final`.
   - NS solver backward becomes dramatically slower than FNO backward as `t_final` increases.
   - At t=20, PyTorch solver backward is about 102x the FNO batch backward and about 820x the FNO per-sample backward.
   - At t=20, JAX solver backward is about 21x the FNO batch backward and about 166x the FNO per-sample backward.

5. Solver backward/inverse memory also grows strongly with `t_final`.
   - At t=2 and t=5, NS solver backward uses less memory than FNO training.
   - Around t=10, solver and FNO training memory are similar.
   - At t=20, solver backward uses about 1.8-1.9x FNO training memory.

6. JAX generally trades memory for speed.
   - JAX is often faster in steady state.
   - JAX often uses more GPU memory, especially in training and inverse/backward.
   - JAX first run must be separated from steady-state timing because of JIT compilation.

7. Burgers and NS behave differently.
   - Burgers solver states are small, so solver memory remains low even for inverse.
   - NS has large grid and many time steps, so solver inverse/backward can become very expensive in both time and memory.

