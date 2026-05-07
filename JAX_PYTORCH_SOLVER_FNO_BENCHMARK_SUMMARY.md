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
