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

## Forward vs Backward/Training Ratios

This section directly compares the cost of:

```text
FNO inference forward
FNO training forward/backward
solver pure forward
solver inverse forward/backward
```

The main question is whether backward/inverse cost grows with `t_final`, and whether the behavior differs between FNO and solver.

### FNO: Training vs Inference

For FNO, the physical `t_final` is not the main scaling variable once the architecture is fixed.

For the trained models used here:

```text
Burgers FNO:
input: one initial field
output: one final field
batch size for fair inference/training comparison: 64

NS recurrent FNO:
input: 10 frames, y[..., 0:10]
output: 10 frames, y[..., 10:20]
batch size for fair inference/training comparison: 8
```

FNO timing ratios:

| Problem | Framework | Inference Forward | Train Forward | Train Backward | Backward / Inference |
|---|---|---:|---:|---:|---:|
| Burgers | PyTorch | 0.003645 s | 0.002747 s | 0.003920 s | 1.08x |
| Burgers | JAX | 0.001957 s | 0.002002 s | 0.003370 s adjusted | 1.72x |
| NS | PyTorch | 0.06334 s | 0.05882 s | 0.10854 s | 1.71x |
| NS | JAX | 0.05402 s | 0.06118 s | 0.08045 s adjusted | 1.49x |

FNO memory ratios:

| Problem | Framework | Inference Memory | Train Forward Memory | Train Backward Memory | Backward / Inference |
|---|---|---:|---:|---:|---:|
| Burgers | PyTorch | 140.2 MiB | 505.8 MiB | 505.8 MiB | 3.61x |
| Burgers | JAX | 717.0 MiB | 2733.0 MiB | 2733.0 MiB | 3.81x |
| NS | PyTorch | 534.7 MiB | 11664.6 MiB | 11724.0 MiB | 21.93x |
| NS | JAX | 1533.0 MiB | 30570.3 MiB | 30583.0 MiB | 19.95x |

FNO pattern:

```text
FNO backward time is usually only about 1.5-1.7x inference forward time.
FNO training memory is much larger than inference memory.
Burgers FNO training memory is about 3.6-3.8x inference.
NS FNO training memory is about 20-22x inference.
FNO cost is mostly controlled by architecture, grid size, batch size, and output window, not directly by solver t_final.
```

### Solver: Inverse/Backward vs Pure Forward

For the numerical solver, `t_final` directly controls the number of solver time steps. With fixed `dt`, larger `t_final` means more steps, so both time and backward memory grow.

For NS:

```text
dt = 0.005
t_final=2  -> 400 steps
t_final=5  -> 1000 steps
t_final=10 -> 2000 steps
t_final=20 -> 4000 steps
```

NS solver timing ratios, inverse/backward compared to pure forward:

| t_final | Steps | Framework | Pure Forward | Inverse Forward | Inverse Backward | Inv Fwd / Pure | Inv Bwd / Pure |
|---:|---:|---|---:|---:|---:|---:|---:|
| 2 | 400 | PyTorch | 0.510 s | 0.678 s | 0.882 s | 1.33x | 1.73x |
| 2 | 400 | JAX | 0.183 s | 0.190 s | 0.176 s | 1.04x | 0.96x |
| 5 | 1000 | PyTorch | 1.328 s | 1.750 s | 2.368 s | 1.32x | 1.78x |
| 5 | 1000 | JAX | 0.455 s | 0.462 s | 0.420 s | 1.01x | 0.92x |
| 10 | 2000 | PyTorch | 2.684 s | 3.697 s | 5.084 s | 1.38x | 1.89x |
| 10 | 2000 | JAX | 0.912 s | 0.925 s | 0.826 s | 1.01x | 0.91x |
| 20 | 4000 | PyTorch | 5.595 s | 7.929 s | 11.127 s | 1.42x | 1.99x |
| 20 | 4000 | JAX | 3.689 s* | 1.836 s | 1.665 s | 0.50x* | 0.45x* |

`*` The NS `t_final=20` JAX pure-forward result came from the forward-only script that returned a fuller time sequence, while inverse JAX only returns the final state. Therefore the t=20 JAX pure-forward ratio is not exactly the same measurement scope. The `t=2/5/10` trend is cleaner.

Time pattern:

```text
Solver forward and backward time grow approximately linearly with t_final.
PyTorch solver inverse backward is about 1.7-2.0x pure forward.
JAX adjusted solver backward is roughly comparable to pure forward, often slightly below or around 1x in these logs.
Total solver inverse step time is roughly forward + backward + small optimizer/update cost.
```

NS solver memory ratios, inverse/backward compared to pure forward:

| t_final | Steps | Framework | Pure Forward Memory | Inverse Backward Memory | Inverse / Pure |
|---:|---:|---|---:|---:|---:|
| 2 | 400 | PyTorch | 29-34 MiB alloc scale | 2170 MiB | about 70x |
| 2 | 400 | JAX | about 13.6 MiB global delta | 7173 MiB global | about 296x using logged deltas |
| 5 | 1000 | PyTorch | about 31 MiB alloc | 5394 MiB | about 174x |
| 5 | 1000 | JAX | about 13.6 MiB global delta | 14849 MiB global | about 589x using logged deltas |
| 10 | 2000 | PyTorch | about 31 MiB alloc | 10779 MiB | about 344x |
| 10 | 2000 | JAX | about 12.8 MiB global delta | 29037 MiB global | about 1174x using logged deltas |
| 20 | 4000 | PyTorch | 33.8 MiB alloc | 21551 MiB | about 637x |
| 20 | 4000 | JAX | about 45.6 MiB global delta | 57423 MiB global | about 713x using logged deltas |

Memory pattern:

```text
Pure solver forward memory is small and changes weakly with t_final.
Solver inverse/backward memory grows strongly with the number of time steps.
The backward/forward memory ratio can grow from tens of times to hundreds of times.
This is because forward-only can overwrite intermediate states, while backward/inverse must store or reconstruct the time-step history needed for gradients.
```

### Solver Backward vs FNO Backward

NS FNO backward, batch 8:

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

This is the cleanest time comparison:

```text
FNO backward is fixed-depth neural-network backward.
Solver backward is backward-through-time over hundreds to thousands of solver steps.
Therefore solver backward becomes increasingly slower than FNO backward as t_final increases.
```

### Solver Backward Memory vs FNO Backward Memory

NS FNO training backward memory:

```text
PyTorch FNO backward: 11724 MiB = 11.45 GiB
JAX FNO backward:     30583 MiB = 29.87 GiB
```

NS solver backward memory compared to FNO backward memory:

| t_final | Steps | Framework | Solver Backward Memory | FNO Backward Memory | Solver / FNO |
|---:|---:|---|---:|---:|---:|
| 2 | 400 | PyTorch | 2.12 GiB | 11.45 GiB | 0.19x |
| 2 | 400 | JAX | 7.00 GiB | 29.87 GiB | 0.23x |
| 5 | 1000 | PyTorch | 5.27 GiB | 11.45 GiB | 0.46x |
| 5 | 1000 | JAX | 14.50 GiB | 29.87 GiB | 0.49x |
| 10 | 2000 | PyTorch | 10.53 GiB | 11.45 GiB | 0.92x |
| 10 | 2000 | JAX | 28.36 GiB | 29.87 GiB | 0.95x |
| 20 | 4000 | PyTorch | 21.05 GiB | 11.45 GiB | 1.84x |
| 20 | 4000 | JAX | 56.08 GiB | 29.87 GiB | 1.88x |

This gives the clearest memory scaling result:

```text
For short NS horizons, solver backward uses less memory than FNO training.
At around t_final=10, solver backward and FNO training are similar.
At t_final=20, solver backward uses about 1.8-1.9x FNO training memory.
So solver backward memory grows with t_final, while FNO training memory stays roughly fixed for a fixed model architecture.
```

### Burgers Difference

Burgers is much smaller than NS. The solver state is 1D and cheap.

For Burgers:

| Framework | Solver Forward | Solver Backward | FNO Inference | FNO Training Backward |
|---|---:|---:|---:|---:|
| PyTorch time | 0.639 s | 1.384 s | 0.003645 s / batch64 | 0.003920 s / batch64 |
| JAX time | 0.271 s | 0.142 s adjusted | 0.001957 s / batch64 | 0.003370 s adjusted / batch64 |
| PyTorch memory | 0.45 MiB forward, 53.5 MiB inverse | 53.5 MiB | 140.2 MiB inference | 505.8 MiB training |
| JAX memory | about 20 MiB forward delta, 621 MiB inverse | 621 MiB | 717 MiB inference | 2733 MiB training |

Burgers pattern:

```text
FNO is vastly faster than the Burgers solver.
But Burgers solver memory remains small, even for inverse/backward.
So for Burgers, FNO does not provide a memory advantage; it provides a speed advantage.
```

### Overall Pattern

```text
FNO:
  forward/backward time is tied mostly to network architecture and batch size.
  memory increases strongly from inference to training because training stores activations, gradients, and optimizer state.
  cost is not directly linear in solver t_final once the architecture and output window are fixed.

Solver:
  forward/backward time grows approximately linearly with t_final at fixed dt.
  pure forward memory stays relatively small because intermediate states can be overwritten.
  inverse/backward memory grows strongly with t_final because gradient computation needs time-step history or equivalent saved/recomputed states.

NS:
  t_final has a large effect on solver inverse time and memory.
  FNO cost stays roughly fixed for the fixed Tin/Tout architecture.

Burgers:
  solver memory is small because the state is small.
  FNO is much faster but uses more memory than the tiny 1D solver.
```

## Four-Dimensional Forward/Backward Comparison

This section reorganizes the results by the four dimensions used in discussion:

```text
Problem:   Burgers vs NS
Object:    Solver vs FNO
Framework: PyTorch vs JAX
Metric:    Time vs Memory
```

The comparison of interest is:

```text
Solver: inverse forward/backward compared to pure solver forward
FNO: training forward/backward compared to pure FNO inference
```

### NS Solver: Inverse vs Pure Forward Time

| t_final | Steps | Framework | Pure Forward | Inverse Forward | Inverse Backward | Inv Fwd / Pure | Inv Bwd / Pure |
|---:|---:|---|---:|---:|---:|---:|---:|
| 2s | 400 | PyTorch | 0.510 s | 0.678 s | 0.882 s | 1.33x | 1.73x |
| 2s | 400 | JAX | 0.183 s | 0.190 s | 0.176 s | 1.04x | 0.96x |
| 5s | 1000 | PyTorch | 1.328 s | 1.750 s | 2.368 s | 1.32x | 1.78x |
| 5s | 1000 | JAX | 0.455 s | 0.462 s | 0.420 s | 1.01x | 0.92x |
| 10s | 2000 | PyTorch | 2.684 s | 3.697 s | 5.084 s | 1.38x | 1.89x |
| 10s | 2000 | JAX | 0.912 s | 0.925 s | 0.826 s | 1.01x | 0.91x |
| 20s | 4000 | PyTorch | 5.595 s | 7.929 s | 11.127 s | 1.42x | 1.99x |
| 20s | 4000 | JAX | 3.689 s* | 1.836 s | 1.665 s | 0.50x* | 0.45x* |

The `20s` JAX pure-forward value has a slightly different output-scope from the inverse script, so the `2s/5s/10s` JAX rows are cleaner for ratio interpretation.

Key point:

```text
Solver time grows with t_final because the number of solver steps grows.
PyTorch solver inverse backward is roughly 1.7-2.0x pure solver forward.
JAX adjusted solver backward is around the same scale as pure solver forward in the clean t=2/5/10 rows.
```

### NS Solver: Inverse/Backward Memory vs Pure Forward Memory

This is the comparison that shows the memory effect most clearly.

| t_final | Steps | Framework | Pure Forward Memory | Inverse/Backward Memory | Backward / Pure Forward Memory |
|---:|---:|---|---:|---:|---:|
| 2s | 400 | PyTorch | 29.3 MiB | 2170 MiB | 74x |
| 2s | 400 | JAX | 13.6 MiB delta | 4148 MiB delta | 296x |
| 5s | 1000 | PyTorch | 31.0 MiB | 5394 MiB | 174x |
| 5s | 1000 | JAX | 13.6 MiB delta | 8244 MiB delta | 589x |
| 10s | 2000 | PyTorch | 31.3 MiB | 10779 MiB | 344x |
| 10s | 2000 | JAX | 12.8 MiB delta | 16436 MiB delta | 1174x |
| 20s | 4000 | PyTorch | 33.8 MiB | 21551 MiB | 637x |
| 20s | 4000 | JAX | 45.6 MiB delta | 32810 MiB delta | 713x |

Key point:

```text
NS solver pure forward memory stays small and changes weakly with t_final.
NS solver inverse/backward memory grows strongly with t_final.
Therefore backward / pure-forward memory ratio can grow from tens of times to hundreds or over one thousand times.
```

Mechanism:

```text
Pure forward:
  keep current state, advance to next state, discard old state.

Backward/inverse:
  need saved or reconstructable time-step history for gradients.
  more time steps means more memory pressure.
```

### Burgers Solver: Inverse vs Pure Forward

| Framework | Pure Forward Time | Inverse Forward Time | Inverse Backward Time | Backward / Pure Time | Pure Forward Memory | Inverse Backward Memory | Backward / Pure Memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch | 0.639 s | 0.894 s | 1.384 s | 2.17x | 0.45 MiB | 53.5 MiB | 119x |
| JAX | 0.271 s | 0.137 s | 0.142 s adjusted | 0.52x | 19.2 MiB delta | 46 MiB delta | 2.4x |

Key point:

```text
Burgers is a small 1D problem.
The solver backward can still be slower than pure forward, but the absolute memory is small.
FNO is much faster, but FNO memory is not lower than the tiny Burgers solver.
```

### FNO: Training/Backward vs Inference

| Problem | Framework | Inference Forward | Train Forward | Train Backward | Backward / Inference Time | Inference Memory | Train Backward Memory | Backward / Inference Memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Burgers | PyTorch | 0.00365 s | 0.00275 s | 0.00392 s | 1.08x | 140.2 MiB | 505.8 MiB | 3.61x |
| Burgers | JAX | 0.00196 s | 0.00200 s | 0.00337 s adjusted | 1.72x | 717 MiB | 2733 MiB | 3.81x |
| NS | PyTorch | 0.06334 s | 0.05882 s | 0.10854 s | 1.71x | 534.7 MiB | 11724 MiB | 21.93x |
| NS | JAX | 0.05402 s | 0.06118 s | 0.08045 s adjusted | 1.49x | 1533 MiB | 30583 MiB | 19.95x |

Key point:

```text
FNO training/backward time is only about 1-2x FNO inference time.
FNO training memory is much larger than inference memory.
Burgers FNO training memory is about 3.6-3.8x inference memory.
NS FNO training memory is about 20-22x inference memory.
```

### Combined Pattern

| Object | Time Compared To Forward | Memory Compared To Forward | Dependence On t_final |
|---|---|---|---|
| FNO | Training/backward is about 1-2x inference forward | Training memory is much larger than inference | Not directly controlled by solver `t_final`; mostly architecture, batch, grid, output window |
| Solver | Backward time is often 1-2x pure forward, but absolute time grows with t_final | Backward memory can be tens to hundreds or over 1000x pure forward | Strong dependence at fixed `dt`; more `t_final` means more time steps |
| Burgers solver | Much slower than FNO but memory remains small | Backward memory increases but absolute memory is small | Small 1D state, weaker memory pressure |
| NS solver | Much slower than FNO and grows with t_final | Backward memory grows strongly with t_final | Strong scaling with time-step count |

One-sentence summary:

```text
FNO forward/backward cost is relatively stable for a fixed architecture, while solver forward/backward cost grows with physical integration time; the strongest effect is NS solver inverse/backward memory, which grows from tens of times to hundreds or over 1000x pure-forward memory as t_final increases.
```

## Linear Scaling With `t_final` and Fixed Memory Overhead

For the NS solver, the `t_final=2/5/10/20` experiments were fit with a simple linear model:

```text
metric = intercept + slope * t_final
```

The goal was to check whether solver time and memory are truly scaling with physical integration time, and whether there is a fixed overhead term.

### Solver Time Scaling

| Metric | Framework | Values at t=2/5/10/20 | Intercept | Slope per physical second | R² |
|---|---|---:|---:|---:|---:|
| inverse forward time | PyTorch | 0.678 / 1.750 / 3.697 / 7.929 s | -0.234 s | 0.405 s/s | 0.99901 |
| inverse backward time | PyTorch | 0.882 / 2.368 / 5.084 / 11.127 s | -0.434 s | 0.573 s/s | 0.99857 |
| inverse forward time | JAX | 0.190 / 0.462 / 0.925 / 1.836 s | 0.0067 s | 0.0915 s/s | 0.99999 |
| inverse backward time | JAX | 0.176 / 0.420 / 0.826 / 1.665 s | 0.0060 s | 0.0828 s/s | 0.99993 |

Time conclusion:

```text
Solver forward time is approximately linear in t_final.
Solver backward time is also approximately linear in t_final.
This is expected because fixed dt means t_final directly determines the number of solver steps.
```

The negative PyTorch intercepts should not be interpreted physically as negative fixed cost. They just indicate small measurement noise or slight nonlinearity in four measured points. The important point is that R² is essentially 1.

### Solver Memory Scaling

| Metric | Framework | Values at t=2/5/10/20 | Intercept | Slope per physical second | R² |
|---|---|---:|---:|---:|---:|
| inverse/backward memory | PyTorch | 2.12 / 5.27 / 10.53 / 21.05 GiB | 0.013 GiB | 1.052 GiB/s | 1.00000 |
| inverse/backward memory | JAX | 7.00 / 14.50 / 28.36 / 56.08 GiB | 1.14 GiB | 2.74 GiB/s | 0.99977 |
| pure forward memory | PyTorch | 0.0286 / 0.0303 / 0.0306 / 0.0330 GiB | 0.0286 GiB | 0.00022 GiB/s | 0.94292 |
| pure forward memory | JAX | 0.0133 / 0.0133 / 0.0125 / 0.0445 GiB | 0.0043 GiB | 0.0018 GiB/s | 0.80823 |

Memory conclusion:

```text
Solver backward memory is approximately linear in t_final.
Solver pure forward memory is almost independent of t_final.
```

Interpretation:

```text
Pure forward:
  The solver mostly keeps the current state and temporary work buffers.
  Old time states can be discarded.
  Therefore memory is dominated by a fixed/small working set.

Backward/inverse:
  Gradient computation needs saved or reconstructable information across time steps.
  More physical time means more time steps.
  Therefore memory grows approximately linearly with t_final.
```

This explains why the ratio:

```text
solver backward memory / solver pure forward memory
```

becomes very large as `t_final` increases. The forward memory baseline stays small and mostly fixed, while backward memory grows with the number of time steps.

### Four Core Scaling Conclusions

| Quantity | Relation to `t_final` | Reason |
|---|---|---|
| Solver forward time | approximately linear growth | more time steps must be advanced |
| Solver backward time | approximately linear growth | gradients must propagate through more time steps |
| Solver backward memory | approximately linear growth | more saved/reconstructed time-step history |
| Solver forward memory | almost no growth | only current state and small work buffers are needed |

Short version:

```text
Time:
  both solver forward and solver backward scale approximately linearly with t_final.

Memory:
  solver backward memory scales approximately linearly with t_final.
  solver forward memory is mostly fixed and weakly dependent on t_final.
```

## Backward Through a Spectral PDE Solver: Memory, Jacobians, and Saved Intermediates

### Problem Setting

The PDE solver can be viewed as a long time-unrolled computation graph. Given an initial condition, the solver repeatedly advances the state by a small time step `dt` until it reaches `t_final`. The loss is computed at the final condition, and the gradient is propagated back to the initial condition.

The goal is not to train the solver parameters. The goal is to compute:

```text
dL / du0
```

or, for the 2D Navier-Stokes vorticity solver:

```text
dL / dOmega0
```

For example, with:

```text
t_final = 20
dt = 0.005
```

the number of time steps is:

```text
N = 20 / 0.005 = 4000
```

Therefore, differentiating through the solver is equivalent to backpropagating through a 4000-layer time-unrolled network.

### Forward Map

One solver step can be written abstractly as:

```text
u_{n+1} = Phi_dt(u_n)
```

The full rollout is:

```text
u_N = Phi_dt o Phi_dt o ... o Phi_dt(u_0)
```

The loss is:

```text
L = L(u_N)
```

For the vorticity-form Navier-Stokes solver:

```text
Omega_{n+1} = Phi_dt(Omega_n)
```

and the desired gradient is:

```text
dL / dOmega0
```

### Backward Recurrence

Define the adjoint variable:

```text
lambda_n = dL / du_n
```

At the final state:

```text
lambda_N = dL / du_N
```

For one time step, define the local Jacobian:

```text
J_n = d u_{n+1} / d u_n = D Phi_dt(u_n)
```

Reverse-mode differentiation propagates:

```text
lambda_n = J_n^T lambda_{n+1}
```

Therefore:

```text
dL / du0 = lambda_0
         = J_0^T J_1^T ... J_{N-1}^T lambda_N
```

Equivalently, the full forward Jacobian is:

```text
d u_N / d u_0 = J_{N-1} J_{N-2} ... J_1 J_0
```

and:

```text
dL / du0 = (d u_N / d u_0)^T dL/du_N
```

The transposed product appears in reverse order during backpropagation.

### The Jacobian Is Not Built Explicitly

Although the math is written using `J_n`, PyTorch and JAX do not explicitly construct the full Jacobian matrix.

For a `256 x 256` state, the flattened dimension is:

```text
256 * 256 = 65536
```

A single dense Jacobian would have shape:

```text
65536 x 65536
```

This is far too large to materialize. In practice, the frameworks compute vector-Jacobian products:

```text
J_n^T lambda_{n+1}
```

The backward pass therefore evolves:

```text
lambda_N -> lambda_{N-1} -> ... -> lambda_0
```

without ever forming the full Jacobian matrix.

### Why Backward Needs Forward States or Residuals

The local Jacobian is evaluated at the forward state:

```text
J_n = D Phi_dt(u_n)
```

If the solver step were linear:

```text
u_{n+1} = A u_n
```

then:

```text
J_n = A
```

and the Jacobian would not depend on the current state. But Burgers and Navier-Stokes are nonlinear. Their nonlinear terms include expressions such as:

```text
u * u_x
u * Omega_x + v * Omega_y
```

Thus:

```text
J_n = J(u_n)
```

or:

```text
J_n = J(Omega_n)
```

This means that the backward step at time `n` needs information from the forward pass at time `n`. If those values were saved during forward, backward can use them directly. If they were not saved, they must be recomputed. This is the core tradeoff behind checkpointing, rematerialization, and recomputation.

### Simple Euler Example

For a simple explicit Euler step:

```text
u_{n+1} = u_n + dt * f(u_n)
```

the local Jacobian is:

```text
J_n = I + dt * df(u_n)/du_n
```

The backward update is:

```text
lambda_n = (I + dt * df(u_n)/du_n)^T lambda_{n+1}
```

This explicitly depends on the forward state `u_n`.

### Burgers Nonlinearity

The 1D Burgers equation is:

```text
u_t + u u_x = nu u_xx
```

or:

```text
u_t = -u u_x + nu u_xx
```

After spatial discretization:

```text
f(u) = -u * D_x u + nu * D_xx u
```

For an Euler step:

```text
u_{n+1} = u_n + dt * (-u_n * D_x u_n + nu * D_xx u_n)
```

For a perturbation `delta u_n`:

```text
delta u_{n+1}
  = delta u_n
  + dt * (
      -delta u_n * D_x u_n
      -u_n * D_x delta u_n
      +nu * D_xx delta u_n
    )
```

Therefore, the local Jacobian-vector product needs both:

```text
u_n
D_x u_n
```

This is why backward needs either saved forward intermediates or recomputed forward intermediates.

### Navier-Stokes Vorticity Solver Step

The 2D Navier-Stokes solver uses the vorticity state:

```text
Omega_n
```

In the PyTorch code, one step is:

```python
omega_hat = self.fft(omega)
omega_next_hat = self.integrator.step_fourier(omega_hat)
omega_next = self.ifft(omega_next_hat)
```

Mathematically:

```text
z_n = F(Omega_n)
z_{n+1} = Psi_dt(z_n)
Omega_{n+1} = F^{-1}(z_{n+1})
```

Therefore:

```text
Omega_{n+1} = F^{-1}(Psi_dt(F(Omega_n)))
```

The local Jacobian has the form:

```text
J_n = D F^{-1} * D Psi_dt(z_n) * D F
```

Because FFT and inverse FFT are linear maps, the complicated part is:

```text
D Psi_dt(z_n)
```

### ETDRK4 Forward Structure

Let:

```text
z = z_n = F(Omega_n)
```

The ETDRK4 step uses four nonlinear evaluations:

```text
K0 = N(z)
A  = E_half z + q K0
K1 = N(A)
B  = E_half z + q K1
K2 = N(B)
C  = E_half A + q (2 K2 - K0)
K3 = N(C)
```

The final spectral state is:

```text
z_plus = E z + f1 K0 + 2 f2 (K1 + K2) + f3 K3
```

This corresponds to the code in `ETDRK4.step_fourier`.

### ETDRK4 Jacobian-Vector Product

For a perturbation:

```text
z -> z + delta z
```

the Jacobian-vector product is obtained by linearizing every stage:

```text
delta K0 = DN(z) delta z
delta A  = E_half delta z + q delta K0
delta K1 = DN(A) delta A
delta B  = E_half delta z + q delta K1
delta K2 = DN(B) delta B
delta C  = E_half delta A + q (2 delta K2 - delta K0)
delta K3 = DN(C) delta C
```

and:

```text
delta z_plus
  = E delta z
  + f1 delta K0
  + 2 f2 (delta K1 + delta K2)
  + f3 delta K3
```

This is the ETDRK4 Jacobian-vector product. Reverse-mode AD computes the corresponding vector-Jacobian products without explicitly forming the Jacobian.

### Navier-Stokes Nonlinear Function

Given Fourier-space vorticity:

```text
z = hat(Omega)
```

the nonlinear function performs:

```text
z_d = M z
psi_hat = Laplace^{-1} z_d
u_hat = D_y psi_hat
v_hat = -D_x psi_hat
Omega_x_hat = D_x z_d
Omega_y_hat = D_y z_d
```

Then it transforms to physical space:

```text
u       = F^{-1}(M u_hat)
v       = F^{-1}(M v_hat)
Omega_x = F^{-1}(M Omega_x_hat)
Omega_y = F^{-1}(M Omega_y_hat)
```

The convection term is:

```text
C = u * Omega_x + v * Omega_y
```

and:

```text
N(z) = -alpha * F(C) + forcing_hat
```

Since the forcing term does not depend on `z`, its derivative is zero.

For a perturbation `delta z`, the linearized nonlinear function is:

```text
DN(z) delta z
  = -alpha * F(
      delta u * Omega_x
      + u * delta Omega_x
      + delta v * Omega_y
      + v * delta Omega_y
    )
```

This expression shows why the backward pass needs forward quantities such as:

```text
u, v, Omega_x, Omega_y
```

These are not trainable parameters, but they are residuals needed to compute the gradient.

### How Many Large Intermediates Exist Per Step

One call to `nonlinear_fun` contains many large tensors, including:

```text
z_d
psi_hat
u_hat
v_hat
Omega_x_hat
Omega_y_hat
u
v
Omega_x
Omega_y
C
F(C)
N(z)
```

This is roughly 12 to 13 full-field or spectral-field sized tensors.

ETDRK4 calls the nonlinear function four times:

```text
N(z), N(A), N(B), N(C)
```

Therefore, the nonlinear part alone can involve roughly:

```text
4 * (12 to 13) = 48 to 52
```

large tensor-scale intermediates per time step.

The ETDRK4 stage values add more large tensors:

```text
z, K0, A, K1, B, K2, C, K3, z_plus
```

This does not mean that every Python variable is necessarily saved as a persistent activation. Autograd saves only the residuals required by backward, and frameworks may reuse buffers or avoid saving constants. However, the step contains enough state-dependent operations that the saved-residual count can be large.

### Why the Minimal Estimate Is Too Small

If we only stored one `256 x 256` float32 field per time step, the memory would be:

```text
4000 * 256 * 256 * 4 bytes
  = 1,048,576,000 bytes
  approximately 1.05 GB
```

This estimate assumes only one `Omega_n` is stored per step. It is a lower bound, not a realistic estimate for autograd through ETDRK4 spectral Navier-Stokes.

A more realistic scaling is:

```text
memory ~= N * H * W * 4 bytes * K
```

where `K` is the number of full-field-equivalent residuals saved per step.

### Residual Count Inferred From Benchmarks

For the NS solver at:

```text
t_final = 20
dt = 0.005
N = 4000
```

the measured backward memory was approximately:

```text
PyTorch: 21.05 GiB
JAX:     56.08 GiB
```

For PyTorch:

```text
21.05 * 1024 / 4000 = 5.39 MiB per step
```

A single `256 x 256` float32 field is:

```text
256 * 256 * 4 bytes = 0.25 MiB
```

So PyTorch stores roughly:

```text
5.39 / 0.25 = 21.6
```

full-field-equivalent tensors per step.

For JAX:

```text
56.08 * 1024 / 4000 = 14.36 MiB per step
14.36 / 0.25 = 57.4
```

So JAX stores roughly:

```text
57 full-field-equivalent tensors per step
```

This agrees with the qualitative code-level estimate that each ETDRK4 spectral NS step can involve tens of large residuals.

### Why JAX Uses More Memory Here

The measured JAX memory is higher than PyTorch for this benchmark. Several factors can contribute:

```text
1. Different memory accounting.
   PyTorch allocator peak and nvidia-smi global memory are not identical measurements.

2. XLA may keep more buffers for speed.
   Compiled buffers, fusion temporaries, residual buffers, and preallocated pools can raise memory.

3. JAX may save more residuals or choose a different rematerialization strategy by default.
```

The benchmark trend was consistent:

```text
JAX is usually faster.
JAX usually uses more GPU memory.
```

For NS solver inverse at `t_final=20`, this appears as:

```text
PyTorch backward memory: about 21.05 GiB
JAX backward memory:     about 56.08 GiB
```

### Why Pure Forward Memory Is Small

Pure forward inference does not need gradients. It can repeatedly update:

```text
Omega_n -> Omega_{n+1}
```

and discard old states. Therefore, pure forward only needs:

```text
current state
next state
FFT/IFFT temporary buffers
small solver work arrays
framework overhead
```

It does not need the full time history. This is why pure forward memory is nearly independent of `t_final`.

Backward/inverse is different. It needs:

```text
J_0^T J_1^T ... J_{N-1}^T lambda_N
```

and each local vector-Jacobian product needs the corresponding forward residuals from that time step. Therefore, backward memory grows with the number of time steps unless rematerialization or checkpointing is used.

### Are 20 GiB and 50 GiB Reasonable?

Yes. The measured numbers are reasonable for this solver.

The rough scaling is:

```text
memory ~= 4000 * (20 to 60) * 256 * 256 * 4 bytes
```

This lands in the tens-of-GiB range.

The benchmark-derived estimates are:

```text
PyTorch: about 20+ full-field-equivalent tensors per step -> about 21 GiB
JAX:     about 50+ full-field-equivalent tensors per step -> about 56 GiB
```

Therefore, the large memory usage is not an anomaly. It reflects the fact that the computation is differentiating through thousands of spectral ETDRK4 time steps, each with many nonlinear intermediates.

### Does Every Variable Have To Be Saved?

Not literally. PyTorch and JAX do not necessarily preserve every Python variable name as a distinct persistent tensor. They save the residuals required by the backward rules.

Constants such as the following generally do not need to be saved as per-step activations:

```text
D_x, D_y, inverse Laplacian, dealiasing mask
E, E_half, q, f1, f2, f3
```

They are fixed solver operators or coefficients.

State-dependent quantities are different. Examples include:

```text
z, K0, A, K1, B, K2, C, K3, z_plus
u, v, Omega_x, Omega_y, convection, F(convection)
```

These are tied to the current time step and are needed, directly or indirectly, to compute:

```text
J_n^T lambda_{n+1}
```

### Checkpointing, Rematerialization, and Recomputation

There is no free way to avoid the information requirement. The backward pass needs the forward trajectory information in one of two forms:

```text
1. saved in memory
2. recomputed during backward
```

Default reverse-mode AD usually saves many residuals:

```text
forward:
  save residuals

backward:
  reuse saved residuals
```

This is memory-heavy but faster.

Checkpointing/rematerialization changes the strategy:

```text
forward:
  save only selected checkpoints

backward:
  recompute local forward segments from the nearest checkpoint
```

For example, one could save every 100th state:

```text
Omega_0, Omega_100, Omega_200, ..., Omega_4000
```

When backward needs states inside the interval `2600..2700`, it recomputes that local segment from `Omega_2600`.

More advanced schemes, such as Revolve/binomial checkpointing, choose checkpoint schedules more intelligently to reduce recomputation cost for a given memory budget.

### Discrete Adjoint vs Continuous Adjoint

If the goal is the exact gradient of the actual discrete solver output:

```text
d L(discrete_solver(u0)) / d u0
```

then the appropriate options are:

```text
default reverse-mode AD
checkpointed reverse-mode AD
hand-written discrete adjoint
```

These compute gradients of the actual discrete computation.

Continuous adjoint methods can reduce memory, but they solve an adjoint equation at the continuous-equation level and then discretize it. This may not match the exact gradient of the discrete solver implementation. For adversarial attacks and inverse problems where the gradient should correspond exactly to the implemented solver, the safer choice is:

```text
checkpointed reverse-mode AD or a hand-written discrete adjoint
```

### Final Interpretation

The key interpretation is:

```text
Pure forward:
  Memory is small because old time states can be discarded.
  Memory is dominated by the current state, work buffers, FFT buffers, and framework overhead.

Backward/inverse:
  Memory is large because gradient computation needs the forward residuals for many time steps.
  With default reverse-mode AD, memory grows approximately linearly with the number of time steps.

Checkpoint/remat:
  Memory can be reduced, but the missing states/intermediates must be recomputed during backward.
  This trades extra runtime for lower GPU memory.
```

In one sentence:

```text
The 20 GiB / 50 GiB memory usage is reasonable because the solver is not merely saving 4000 single Omega fields; it is saving or accounting for the residuals needed to differentiate through 4000 complex ETDRK4 spectral Navier-Stokes steps, each of which contains several stages, FFT/IFFT operations, nonlinear products, and state-dependent intermediates.
```

## Trainable Parameters vs Intermediate Computational Values

A useful way to compare FNO models and PDE solvers is to separate three different concepts:

```text
1. Trainable parameters
2. Fixed solver operators / coefficients
3. Intermediate computational values generated during forward/backward
```

FNO has trainable parameters. A PDE solver mostly does not. However, the solver can still generate and process a very large number of intermediate values because it advances the state through many small time steps.

Therefore, the better comparison is not only:

```text
How many trainable parameters does the model have?
```

but also:

```text
How many scalar values are generated, transformed, or needed internally to map the input to the output?
```

This quantity can be described as:

```text
intermediate scalar values
forward computational state volume
time-unrolled intermediate values
```

These are not trainable parameters, but they strongly affect runtime, memory traffic, and backward memory.

### FNO Trainable Parameter Counts

The trained FNO configurations used in the benchmarks are:

```text
Burgers FNO1d:
  nx = 1024
  modes = 16
  width = 64
  layers = 4

NS FNO2d:
  nx = 256
  input channels = 10
  modes = 12 x 12
  width = 20
  layers = 4
```

The PyTorch model parameter counts are:

| Model | Layers | Tensor parameter count | Real-scalar equivalent | Parameter memory |
|---|---:|---:|---:|---:|
| Burgers FNO1d | 4 | 320,705 | 582,849 | 2.22 MiB |
| Burgers FNO1d | 1 | 86,657 | 152,193 | 0.58 MiB |
| NS FNO2d | 4 | 467,861 | 928,661 | 3.54 MiB |
| NS FNO2d | 1 | 118,481 | 233,681 | 0.89 MiB |

The "real-scalar equivalent" column counts each complex Fourier parameter as two real scalar values.

### Solver Time-Unrolled State Counts

The solver does not have comparable trainable parameters, but it has a large time-unrolled trajectory. If we only count one physical state per time step, the raw state volume is:

```text
steps * grid_size
```

For the solver cases:

| Solver case | Grid state size | Steps | Raw trajectory grid values |
|---|---:|---:|---:|
| Burgers, `nx=1024`, `t=1`, `dt=0.001` | 1,024 | 1,000 | 1,024,000 |
| NS, `nx=256`, `t=2`, `dt=0.005` | 65,536 | 400 | 26,214,400 |
| NS, `nx=256`, `t=5`, `dt=0.005` | 65,536 | 1,000 | 65,536,000 |
| NS, `nx=256`, `t=10`, `dt=0.005` | 65,536 | 2,000 | 131,072,000 |
| NS, `nx=256`, `t=20`, `dt=0.005` | 65,536 | 4,000 | 262,144,000 |

This table only counts one state per time step. It is a lower bound. The actual solver step also creates spectral states, ETDRK4 stage values, nonlinear terms, FFT/IFFT buffers, velocity fields, derivatives, and convection terms.

### Forward Intermediate Values: FNO vs Solver

The more relevant comparison is the number of intermediate scalar values involved in the forward computation.

For the NS FNO with:

```text
nx = 256
input channels = 10
width = 20
modes = 12 x 12
layers = 4
```

the input contains:

```text
256 * 256 * 10 = 655,360 values
```

After lifting to width 20:

```text
256 * 256 * 20 = 1,310,720 values
```

Each FNO layer contains:

```text
FFT over width channels
low-mode spectral multiplication
inverse FFT
MLP branch
1x1 convolution branch
residual add
GELU, except after the final layer
```

A rough operation-level accounting gives the NS FNO forward pass roughly:

```text
50M - 60M intermediate scalar-equivalent values
```

For the NS spectral ETDRK4 solver at:

```text
nx = 256
t_final = 20
dt = 0.005
steps = 4000
```

the raw state trajectory alone is:

```text
4000 * 256 * 256 = 262,144,000 values
```

But each ETDRK4 step includes:

```text
4 nonlinear_fun evaluations
FFT / IFFT operations
velocity reconstruction
Omega_x / Omega_y derivatives
convection terms
ETDRK4 stage variables K0, A, K1, B, K2, C, K3
spectral coefficient multiplications
```

A rough code-level estimate gives one NS solver step about:

```text
~4.1M real-scalar-equivalent intermediate values per step
```

Across 4000 steps:

```text
4.1M * 4000 ~= 16.6B real-scalar-equivalent intermediate values
```

So a useful forward-volume comparison is:

| Case | Approximate forward intermediate values |
|---|---:|
| NS FNO2d, 4 layers | ~58M |
| NS solver, `t=20`, `dt=0.005` | ~16.6B |

This gives:

```text
NS solver forward intermediate value volume / NS FNO forward intermediate value volume
  ~= 16.6B / 58M
  ~= 280x
```

For Burgers, using the trained FNO setup and the dataset solver setup:

| Case | Approximate forward intermediate values |
|---|---:|
| Burgers FNO1d, 4 layers | ~2.6M |
| Burgers solver, `t=1`, `dt=0.001` | ~40M |
| Burgers solver, `t=20`, `dt=0.001` | ~800M |

Therefore:

```text
Burgers solver t=1 / Burgers FNO ~= 15x
Burgers solver t=20 / Burgers FNO ~= 300x
```

These numbers are approximate. They are intended to give the correct order of magnitude, not an exact FLOP count or exact memory allocation count.

### Interpretation

FNO behaves like a learned operator:

```text
u_final ~= F_theta(u_initial)
```

The complexity is stored partly in trainable parameters `theta` and partly in a small number of neural-network layers. For NS:

```text
NS FNO trainable real-scalar equivalent parameters: ~0.93M
NS FNO forward intermediate scalar-equivalent values: ~58M
```

The PDE solver behaves like a physical time-stepper:

```text
u_0 -> u_1 -> u_2 -> ... -> u_N
```

The complexity is not in learned parameters. It is in the long trajectory and in the numerical work inside each time step. For NS at `t=20`:

```text
NS solver trainable parameters: essentially none
NS solver raw trajectory states: ~262M values
NS solver estimated forward intermediate values: ~16.6B values
```

So the correct conceptual contrast is:

```text
FNO:
  More trainable parameters.
  Shorter computation depth.
  Intermediate values scale mainly with grid size, width, modes, and number of FNO layers.

Solver:
  Almost no trainable parameters.
  Very long time-unrolled computation.
  Intermediate values scale mainly with grid size, step complexity, and t_final / dt.
```

This explains why solver pure forward can have small peak memory but still perform an enormous amount of numerical work. It also explains why solver backward is expensive: the backward pass needs the trajectory information or must recompute it.

Short version:

```text
FNO complexity mainly lives in learned parameters and a shallow learned computation graph.
Solver complexity mainly lives in the long physical trajectory and the large number of intermediate numerical values generated across time steps.
```
