# Attack Framework Matrix Observations

This note summarizes the debugging observations from the Burgers 1D
Solver-Model-Framework PGD attack matrix.

## Experiment Context

Script:

- `tools/attack_framework_matrix.py`

Current Burgers result directory:

- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d`

Four attack combinations:

- `torch_solver_torch_model`
- `torch_solver_jax_model`
- `jax_solver_torch_model`
- `jax_solver_jax_model`

Main Burgers attack parameters used:

- norm: `inf`
- epsilon: `0.5`
- alpha: `0.05`
- steps: `100`
- sample index: `0`
- solver dtype: `float64` for both PyTorch solver and JAX solver

## Important Result: Solver Backend Is Not The Main Difference

For the same initial condition `x0` at PGD step 0, the PyTorch solver and JAX
solver outputs are essentially identical:

```text
PyTorch solver - JAX solver, step 0:
L2       = 1.641251e-07
Linf     = 1.005702e-08
RMSE     = 5.128908e-09
mean_abs = 4.459015e-09
```

The 10-sample sanity check confirmed the same behavior:

```text
torch_solver - jax_solver over 10 initial conditions:
L2   mean/std = 7.74e-07 / 1.33e-06
Linf mean/std = 4.91e-08 / 8.59e-08
RMSE mean/std = 2.42e-08 / 4.17e-08
```

So, for Burgers 1D, the solver implementations are numerically very close.
The solver backend does not explain the large grouping seen in the attack loss
curves.

Relevant output files:

- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d/step0_backend_output_differences.csv`
- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d/step0_backend_output_differences.png`
- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d/backend_output_sanity_10/summary_pairwise_differences.csv`
- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d/backend_output_sanity_10/per_sample_pairwise_differences.csv`

## Critical Observation: Model Backend / Checkpoint Is The Main Difference

For the same `x0` at PGD step 0, the PyTorch model and JAX real/imag model
already differ noticeably:

```text
PyTorch model - JAX model, step 0:
L2       = 5.310195e-01
Linf     = 1.623930e-01
RMSE     = 1.659436e-02
mean_abs = 9.413907e-03
```

The 10-sample sanity check shows this is stable across multiple initial
conditions, not a one-sample accident:

```text
torch_model - jax_model over 10 initial conditions:
L2   mean/std = 0.4083 / 0.0661
Linf mean/std = 0.1019 / 0.0315
RMSE mean/std = 0.01276 / 0.00206
```

This explains why the PGD attack loss curves group primarily by model backend:

```text
torch_solver_torch_model final loss = 0.095115
jax_solver_torch_model   final loss = 0.095001

torch_solver_jax_model   final loss = 0.057274
jax_solver_jax_model     final loss = 0.056977
```

The solver choice changes the final loss only slightly. The model choice changes
the final loss much more.

## Why This Is Not A Contradiction With Training Metrics

The original training run did show that PyTorch and JAX models both achieved
similar test accuracy against the same ground truth.

Attack-ready model directory:

- `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500`

Source training run:

- `fno_training_runs/burgers_profile_500_four_way`

Training comparison plots:

- `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/plots/loss_compare_all_frameworks.png`
- `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/plots/inference_compare_all_frameworks.png`

Final train/test metrics:

```text
PyTorch:
train RMSE        = 0.008987
test RMSE         = 0.009547
train relative L2 = 0.016820
test relative L2  = 0.017519

JAX real/imag:
train RMSE        = 0.008276
test RMSE         = 0.008574
train relative L2 = 0.015557
test relative L2  = 0.015980
```

However, the archived inference summary also recorded that the JAX prediction
and PyTorch prediction are not pointwise identical:

```text
jax_real_imag prediction vs pytorch prediction:
RMSE        = 0.013500
MAE         = 0.008378
max_abs     = 0.162356
relative L2 = 0.024476
```

This matches the attack sanity check almost exactly:

```text
PyTorch model - JAX model at attack step 0:
RMSE = 0.01659
Linf = 0.16239
```

Therefore:

- Both models can be good approximations of the PDE target.
- But the two independently trained checkpoints are not the same function.
- The attack loss `||model(x_adv) - solver(x_adv)||` is sensitive to this
  model-output difference.
- As a result, the attack matrix using native PyTorch checkpoint vs native JAX
  checkpoint is not a pure framework comparison.

## Fairness Implication

The current attack matrix compares:

```text
PyTorch-trained model in PyTorch
vs
JAX-trained model in JAX
```

This mixes two factors:

- framework implementation
- trained checkpoint / learned function

For timing and memory benchmarking, this may still be useful if interpreted
carefully. But for a clean framework comparison of attack behavior, it is not
fully fair.

For a cleaner framework-only comparison, use the same model weights in both
frameworks. The attack-ready archive includes:

- `checkpoints/jax_real_imag_as_pytorch_fno1d_500.pt`

The README reports that this converted PyTorch checkpoint matches native JAX
real/imag inference closely:

```text
converted PyTorch vs native JAX real/imag:
max_abs diff = 0.003465116
relative L2  = 0.000368576
```

This converted checkpoint is better suited for a same-weight framework sanity
check.

## Visualization Updates

The Burgers GIF visualization was revised so each frame shows:

- per-combo initial condition before/after perturbation
- per-combo final solver condition before/after perturbation
- per-combo perturbation `delta = x_adv - x0`
- overlay comparisons across all four methods
- differences relative to the reference combo

Generated files:

- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d/burgers_1d_attack_all_combos.gif`
- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d/burgers_1d_attack_first_frame.png`
- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d/burgers_1d_attack_final_frame.png`
- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d/loss_progression_all_combos.png`
- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d/loss_progression_small_multiples.png`
- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d/combo_divergence_vs_reference.png`
- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d/conversion_overhead_all_combos.png`
- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d/runtime_memory_all_combos.png`

## Runtime And Memory Interpretation

The current script records:

- model forward time
- solver forward time
- loss evaluation time
- backward / gradient time
- projection time
- total PGD step time
- PyTorch allocated/reserved/peak memory when applicable
- global GPU memory sampled through `nvidia-smi`
- conversion overhead for mixed PyTorch/JAX paths

For JAX memory, there is no PyTorch-like allocator API available in this script,
so the main JAX memory signal is global GPU memory from `nvidia-smi`. Because
each combo is run in a separate subprocess, cross-combo memory contamination is
reduced. If other processes are using the same GPU, global GPU memory can still
be affected.

### Complete Native-Checkpoint Timing Summary

The following numbers are from:

- `benchmark_results/attack_framework_matrix_all_float64/burgers_1d/metrics_all_combos.csv`

They use the native PyTorch checkpoint for the PyTorch model backend and the
native JAX real/imag checkpoint for the JAX model backend.

The table below includes all timed PGD update steps, including step 0 JIT/CUDA
warmup effects:

```text
jax_solver_torch_model
total 100-step timed loop = 87.412 s
avg step                  = 0.874 s
avg forward total         = 0.160 s
avg model forward         = 0.00462 s
avg solver forward        = 0.155 s
avg backward / grad       = 0.319 s
avg projection            = 0.00045 s
peak global GPU memory    = 669 MiB
final attack loss         = 0.095001

jax_solver_jax_model
total 100-step timed loop = 95.487 s
avg step                  = 0.955 s
avg forward total         = 0.169 s
avg model forward         = 0.01461 s
avg solver forward        = 0.153 s
avg backward / value_grad = 0.329 s
avg projection            = 0.00052 s
peak global GPU memory    = 747 MiB
final attack loss         = 0.056977

torch_solver_torch_model
total 100-step timed loop = 210.174 s
avg step                  = 2.102 s
avg forward total         = 0.733 s
avg model forward         = 0.00796 s
avg solver forward        = 0.725 s
avg backward / grad       = 0.936 s
avg projection            = 0.00039 s
peak global GPU memory    = 703 MiB
final attack loss         = 0.095115

torch_solver_jax_model
total 100-step timed loop = 224.909 s
avg step                  = 2.249 s
avg forward total         = 0.768 s
avg model forward         = 0.02142 s
avg solver forward        = 0.746 s
avg backward / grad       = 0.984 s
avg projection            = 0.00041 s
peak global GPU memory    = 807 MiB
final attack loss         = 0.057274
```

The timing summary CSV created during analysis is:

- `benchmark_results/attack_timing_summary_burgers.csv`

### Main Runtime Conclusion

For Burgers 1D, runtime is mainly determined by the solver backend.

Holding model fixed and changing solver:

```text
Torch model:
torch solver -> jax solver: 2.102 s/step -> 0.874 s/step
difference = 1.228 s/step

JAX model:
torch solver -> jax solver: 2.249 s/step -> 0.955 s/step
difference = 1.294 s/step
```

Holding solver fixed and changing model:

```text
JAX solver:
torch model -> jax model: 0.874 s/step -> 0.955 s/step
difference = 0.081 s/step

Torch solver:
torch model -> jax model: 2.102 s/step -> 2.249 s/step
difference = 0.147 s/step
```

Therefore, the solver backend effect on runtime is roughly 8x to 15x larger
than the model backend effect.

The fastest and lowest-memory combination in this native-checkpoint Burgers run
was:

```text
jax_solver_torch_model:
avg step = 0.874 s
peak global GPU memory = 669 MiB
```

The slowest and highest-memory combination was:

```text
torch_solver_jax_model:
avg step = 2.249 s
peak global GPU memory = 807 MiB
```

### Memory Pattern

Peak global GPU memory in the native-checkpoint Burgers run:

```text
jax_solver_torch_model  = 669 MiB
torch_solver_torch_model = 703 MiB
jax_solver_jax_model    = 747 MiB
torch_solver_jax_model  = 807 MiB
```

The JAX model backend increases memory relative to the Torch model backend:

```text
fixed JAX solver:
torch model -> jax model: 669 MiB -> 747 MiB
difference = 78 MiB

fixed Torch solver:
torch model -> jax model: 703 MiB -> 807 MiB
difference = 104 MiB
```

Runtime is dominated by solver backend. Memory is affected by both solver and
model backend, with the JAX model backend increasing memory in this run.

### Tensor Conversion Overhead

The mixed PyTorch/JAX combinations explicitly record DLPack bridge overhead.
Across all 100 timed steps, pure tensor conversion overhead was very small:

```text
jax_solver_torch_model:
forward torch->jax total  = 0.0336 s
forward jax->torch total  = 0.0255 s
backward torch->jax total = 0.0402 s
backward jax->torch total = 0.0268 s

torch_solver_jax_model:
forward torch->jax total  = 0.0337 s
forward jax->torch total  = 0.0131 s
backward torch->jax total = 0.0442 s
backward jax->torch total = 0.0132 s
```

Per step, these transfers are about `0.0001-0.0004 s`, so they are negligible
relative to the total step times of `0.8-2.2 s`.

Important distinction:

```text
Tensor conversion overhead is tiny.
JAX VJP/backward computation can still be significant.
```

For example:

```text
jax_solver_torch_model:
JAX VJP total over 100 steps = 31.548 s
JAX VJP mean per step        = 0.315 s

torch_solver_jax_model:
JAX VJP total over 100 steps = 3.017 s
JAX VJP mean per step        = 0.030 s
```

So the mixed-framework cost is not mainly memory transfer. It is mostly the
actual differentiated JAX computation when the JAX component participates in
backpropagation.

### JAX JIT / Warmup Behavior

The original timing averages above include step 0. Step 0 contains JAX JIT
compilation and CUDA warmup effects.

JIT means **Just-In-Time compilation**. In JAX, a `jax.jit`-compiled function is
not compiled once globally when the Python file is written, like a normal
ahead-of-time C/C++ build. Instead, JAX compiles the function when it is first
called with concrete input shapes, dtypes, and static arguments.

Conceptually:

```text
first call:
trace Python/JAX function -> build computation graph / jaxpr -> compile with XLA
for the actual input shape and dtype -> execute

later calls with the same shape/dtype/static args:
reuse cached compiled XLA executable -> execute directly
```

This is why it is called "just in time": compilation happens just before the
compiled code is needed for execution. The compiler has enough information about
the function, input shapes, dtypes, and target hardware to optimize the
computation. XLA can fuse operations, optimize memory movement, and generate
hardware-specific kernels.

The tradeoff is:

```text
first call can be slow because it includes compilation;
repeated calls can be much faster because they reuse the compiled executable.
```

The JIT effect is very clear in the per-step metrics.

For `jax_solver_jax_model`:

```text
total step:
step0     = 5.133 s
late mean = 0.913 s
step0 / late mean = 5.62x

forward total:
step0     = 1.814 s
late mean = 0.153 s
step0 / late mean = 11.88x

model forward:
step0     = 1.406 s
late mean = 0.00055 s
step0 / late mean = 2562x

backward / value_and_grad:
step0     = 2.546 s
late mean = 0.307 s
step0 / late mean = 8.30x
```

For `torch_solver_jax_model`, the JAX model also shows a strong first-step
compile effect:

```text
total step:
step0     = 5.702 s
late mean = 2.198 s
step0 / late mean = 2.59x

JAX model forward:
step0     = 1.463 s
late mean = 0.00106 s
step0 / late mean = 1380x

JAX model VJP:
step0     = 1.869 s
late mean = 0.00068 s
step0 / late mean = 2737x
```

PyTorch-only also has a first-step warmup effect, but it is smaller and does not
have the same compile/cache signature:

```text
torch_solver_torch_model:
total step step0 = 3.499 s
late mean        = 2.086 s
step0 / late     = 1.68x

forward total step0 = 1.426 s
late mean           = 0.729 s
step0 / late        = 1.96x

solver forward step0 = 0.912 s
late mean            = 0.726 s
step0 / late         = 1.26x

backward step0 = 1.053 s
late mean      = 0.934 s
step0 / late   = 1.13x
```

PyTorch can still have first-step overhead from CUDA context initialization,
memory allocator setup, cuDNN/cuBLAS/FFT/kernel warmup, and first-use GPU
runtime costs. But this is not the same as JAX tracing and XLA compilation, and
the spike is much smaller in this experiment.

In short:

```text
PyTorch first step is somewhat slower than later steps.
JAX first step is often much slower because it includes JIT compilation.
After compilation, JAX reuses cached executables and can be substantially faster.
```

### Runtime With And Without JIT/Warmup

Including step 0:

```text
jax_solver_torch_model   = 0.874 s/step
jax_solver_jax_model     = 0.955 s/step
torch_solver_torch_model = 2.102 s/step
torch_solver_jax_model   = 2.249 s/step
```

Dropping step 0:

```text
jax_solver_torch_model   = 0.865 s/step
jax_solver_jax_model     = 0.913 s/step
torch_solver_torch_model = 2.088 s/step
torch_solver_jax_model   = 2.214 s/step
```

Dropping the first 10 steps, as a steady-state estimate:

```text
jax_solver_torch_model   = 0.862 s/step
jax_solver_jax_model     = 0.913 s/step
torch_solver_torch_model = 2.086 s/step
torch_solver_jax_model   = 2.198 s/step
```

Solver speed ratios:

```text
Including step 0:
torch_solver_torch_model / jax_solver_torch_model = 2.40x
torch_solver_jax_model   / jax_solver_jax_model   = 2.36x

Dropping step 0:
torch_solver_torch_model / jax_solver_torch_model = 2.41x
torch_solver_jax_model   / jax_solver_jax_model   = 2.43x

Dropping first 10 steps:
torch_solver_torch_model / jax_solver_torch_model = 2.42x
torch_solver_jax_model   / jax_solver_jax_model   = 2.41x
```

Conclusion:

```text
JAX step 0 is slower because of JIT compilation.
After removing JIT/warmup, JAX solver's advantage becomes slightly more obvious.
The main conclusion does not change: JAX solver is about 2.4x faster than the
PyTorch solver in this Burgers PGD loop.
```

This means a one-step or very short run can make JAX look worse than it really
is for repeated workloads. For a PGD attack with many repeated steps and fixed
shapes, JAX pays the compilation cost early and then benefits from cached,
optimized execution. In this Burgers attack, even when step 0 is included, the
JAX solver combinations are already much faster than the PyTorch solver
combinations; after removing the first-step JIT/warmup cost, the JAX solver
advantage is slightly stronger and cleaner.

### Direct First-Step Vs Steady-State Comparison

The clearest way to interpret the timing is to separate two questions:

```text
Question 1: If I only look at the very first PGD step, is JAX slower?
Question 2: After the first step, when JIT/cache/warmup effects are mostly gone,
            is JAX faster for the repeated PGD loop?
```

For the pure-backend comparison:

```text
First step only:
jax_solver_jax_model     = 5.133 s
torch_solver_torch_model = 3.499 s

Pure JAX first step / pure PyTorch first step = 1.47x slower
```

So yes: if the comparison is literally only the first step, the pure JAX
combination is slower here, because the first JAX step includes JIT compilation.

For the repeated steady-state part:

```text
Late-step mean:
jax_solver_jax_model     = 0.913 s/step
torch_solver_torch_model = 2.086 s/step

Pure PyTorch late step / pure JAX late step = 2.29x
```

So after the first-step compilation cost, pure JAX is about `2.29x` faster than
pure PyTorch in this Burgers PGD workload.

Holding the model backend fixed gives the cleaner solver comparison.

With the Torch model fixed:

```text
First step:
jax_solver_torch_model   = 1.824 s
torch_solver_torch_model = 3.499 s

torch_solver_torch_model / jax_solver_torch_model = 1.92x

Late-step mean:
jax_solver_torch_model   = 0.862 s/step
torch_solver_torch_model = 2.086 s/step

torch_solver_torch_model / jax_solver_torch_model = 2.42x
```

With the JAX model fixed:

```text
First step:
jax_solver_jax_model   = 5.133 s
torch_solver_jax_model = 5.702 s

torch_solver_jax_model / jax_solver_jax_model = 1.11x

Late-step mean:
jax_solver_jax_model   = 0.913 s/step
torch_solver_jax_model = 2.198 s/step

torch_solver_jax_model / jax_solver_jax_model = 2.41x
```

The conclusion is:

```text
The first step is not a fair representation of long PGD runtime for JAX,
because it includes JIT compilation.

PyTorch also has a first-step warmup effect, but it is much smaller.

For many repeated fixed-shape PGD steps, the JAX solver advantage is the
important long-run behavior. In this run, the steady-state JAX solver path is
about 2.4x faster than the PyTorch solver path.
```

### Same-JAX-Weights Run Status

A second version was added to use the same JAX real/imag model weights for both
model backends:

```text
--burgers-model-weight-source jax_real_imag
```

This uses:

```text
JAX model backend:
checkpoints/jax_real_imag_fno1d_500.pkl

PyTorch model backend:
checkpoints/jax_real_imag_as_pytorch_fno1d_500.pt
```

At the time of this note, the result directory:

- `benchmark_results/attack_framework_matrix_burgers_idx0_same_jax_weights_float64/burgers_1d`

contained one completed combo:

```text
torch_solver_torch_model
total 100-step timed loop = 211.425 s
avg step                  = 2.114 s
avg forward total         = 0.711 s
avg model forward         = 0.00419 s
avg solver forward        = 0.706 s
avg backward / grad       = 0.949 s
avg projection            = 0.00036 s
peak global GPU memory    = 703 MiB
final attack loss         = 0.057004
```

This is important because the native PyTorch model version had final loss
`0.095115`, while the same-JAX-weights PyTorch backend has final loss
`0.057004`, close to the native JAX-model group. This supports the hypothesis
that the original final-loss grouping was caused by model checkpoint/function
difference, not solver backend difference.

### Same-JAX-Weights Timing And Memory Comparison

After running all four Burgers combos with `--burgers-model-weight-source
jax_real_imag`, the loss curves became almost identical across the four
framework combinations. This is the expected fairness check: once the PyTorch
model backend and JAX model backend use the same JAX real/imag weights, the
large native-weight loss gap disappears.

The timing and memory pattern, however, stayed essentially the same as the
native-weight run.

Native-weight run:

```text
combo                     avg step   late mean   peak GPU   final loss
jax_solver_torch_model    0.871 s    0.859 s     669 MiB    0.095001
jax_solver_jax_model      0.952 s    0.909 s     747 MiB    0.056977
torch_solver_torch_model  2.089 s    2.072 s     703 MiB    0.095115
torch_solver_jax_model    2.238 s    2.186 s     807 MiB    0.057274
```

Same-JAX-weights run:

```text
combo                     avg step   late mean   peak GPU   final loss
jax_solver_torch_model    0.890 s    0.866 s     669 MiB    0.057001
jax_solver_jax_model      0.949 s    0.903 s     747 MiB    0.056982
torch_solver_torch_model  2.104 s    2.101 s     703 MiB    0.057004
torch_solver_jax_model    2.185 s    2.145 s     807 MiB    0.057274
```

The timed 100-step PGD loop should be interpreted per combo. Each row below is
one independent 100-step attack run, not an average and not the sum across
combos.

```text
native-weight run:
combo                     100-step timed loop
jax_solver_torch_model    87.944 s
jax_solver_jax_model      96.102 s
torch_solver_torch_model  210.943 s
torch_solver_jax_model    226.017 s

same-JAX-weights run:
combo                     100-step timed loop
jax_solver_torch_model    89.874 s
jax_solver_jax_model      95.830 s
torch_solver_torch_model  212.547 s
torch_solver_jax_model    220.645 s
```

The per-combo runtime pattern stayed essentially unchanged after switching the
PyTorch model backend to the converted JAX real/imag weights. The big change was
loss fairness, not runtime.

The speed ratios also stayed very close.

Native-weight run:

```text
torch_solver_torch_model / jax_solver_torch_model:
avg step = 2.40x
late     = 2.41x

torch_solver_jax_model / jax_solver_jax_model:
avg step = 2.35x
late     = 2.40x
```

Same-JAX-weights run:

```text
torch_solver_torch_model / jax_solver_torch_model:
avg step = 2.37x
late     = 2.43x

torch_solver_jax_model / jax_solver_jax_model:
avg step = 2.30x
late     = 2.38x
```

Memory ordering did not change:

```text
smallest peak memory: jax_solver_torch_model   = 669 MiB
largest peak memory:  torch_solver_jax_model   = 807 MiB

jax_solver_jax_model       = 747 MiB
torch_solver_torch_model   = 703 MiB
```

The interpretation is:

```text
Changing the model weights to the same JAX real/imag checkpoint fixes the loss
fairness problem.

It does not materially change the runtime or memory conclusion.

Runtime is still dominated by the solver backend: JAX solver is about 2.3x to
2.4x faster than the PyTorch solver in steady repeated PGD steps.

Memory ordering is also unchanged: JAX solver + Torch model is still the
smallest, and Torch solver + JAX model is still the largest.

The mixed JAX/PyTorch conversion overhead remains negligible compared with the
forward/backward solver cost.
```

### Comparison With Earlier Solver/FNO Benchmarks

The current PGD matrix memory pattern should not be read as a universal claim
that "JAX always uses less memory." Earlier standalone benchmarks used different
measurement scopes and different workloads:

- model-only inference
- model training forward/backward/optimizer
- solver-only forward
- solver inverse/backward optimization
- full mixed model-vs-solver PGD attack

The earlier benchmark summary is:

- `JAX_PYTORCH_SOLVER_FNO_BENCHMARK_SUMMARY.md`

Important source files:

```text
FNO inference:
benchmark_results/fno_inference_burgers_trainbatch64_0507/summary.json
benchmark_results/fno_inference_ns_trainbatch8_0507/summary.json

FNO training:
fno_training_runs/burgers_profile_500_four_way/metrics_summary.csv
fno_training_runs/ns_m12_w20_profile_500/metrics_summary.csv

Solver forward-only:
benchmark_results/torch_vs_exponax_solver_comparison_full_tfinal_multi10.csv
benchmark_results/torch_vs_exponax_solver_comparison_ns_t2_multi10.csv
benchmark_results/torch_vs_exponax_solver_comparison_ns_t5_multi10.csv
benchmark_results/torch_vs_exponax_solver_comparison_ns_t10_multi10.csv

Solver inverse/backward:
benchmark_results/inverse_solver_demo_100steps_full_forward_params_manual/summary.json
benchmark_results/inverse_solver_demo_100steps_full_forward_params_manual/metrics_all.csv
```

The older FNO training data does support the memory pattern remembered earlier:
JAX model training used much more GPU memory than PyTorch model training.

```text
Burgers FNO training, batch 64:
framework        train forward peak   train backward peak   whole-run peak
PyTorch          505.8 MiB            505.8 MiB             1407 MiB
JAX real/imag    2733 MiB             2733 MiB              2735 MiB

NS FNO training, batch 8:
framework        train forward peak   train backward peak   whole-run peak
PyTorch          11666 MiB            11724 MiB             13987 MiB
JAX real/imag    30597 MiB            30597 MiB             30651 MiB
```

For model-only inference, the picture was mixed:

```text
Burgers FNO inference, batch 64:
framework        steady forward       global peak
PyTorch          0.003645 s           747 MiB global
JAX real/imag    0.001957 s           717 MiB global

NS FNO inference, batch 8:
framework        steady forward       global peak
PyTorch          0.06334 s            1241 MiB global
JAX real/imag    0.05402 s            1533 MiB global
```

So for NS inference, JAX used more global GPU memory. For Burgers isolated
inference, the global peak was slightly lower for JAX, although PyTorch's own
allocator peak was only about `140 MiB`. This is a measurement-scope issue:
PyTorch allocator memory and global `nvidia-smi` memory are not the same thing.

For solver-only forward, the older data was also not uniformly "JAX uses more
memory":

```text
Burgers solver forward, nx=1024, t_final=1:
framework        steady forward       global delta
PyTorch solver   0.639 s              8 MiB
JAX/Exponax      0.271 s              19.2 MiB

NS solver forward, nx=256, t_final=20:
framework        steady forward       global delta
PyTorch solver   5.595 s              64 MiB
JAX/Exponax      3.689 s              45.6 MiB

NS solver forward, nx=256, t_final=2:
framework        steady forward       global delta
PyTorch solver   0.510 s              60 MiB
JAX/Exponax      0.183 s              13.6 MiB

NS solver forward, nx=256, t_final=5:
framework        steady forward       global delta
PyTorch solver   1.328 s              60 MiB
JAX/Exponax      0.455 s              13.6 MiB

NS solver forward, nx=256, t_final=10:
framework        steady forward       global delta
PyTorch solver   2.684 s              62 MiB
JAX/Exponax      0.912 s              12.8 MiB
```

For solver inverse/backward, the old full-size benchmark showed:

```text
Burgers inverse/backward, nx=1024, 100 steps:
framework        total time   forward/step   backward/step   global end/peak-ish
PyTorch solver   241.9 s      0.894 s        1.384 s         565 MiB
JAX/Exponax      51.2 s       0.137 s        0.142 s         621 MiB

NS inverse/backward, nx=256, t_final=20, 100 steps:
framework        total time   forward/step   backward/step   global end/peak-ish
PyTorch solver   1938.8 s     7.929 s        11.127 s        24613 MiB
JAX/Exponax      546.8 s      1.836 s        1.665 s         57423 MiB
```

This old inverse/backward benchmark strongly supports the remembered NS pattern:
JAX/Exponax was much faster but used much more memory for the large NS inverse
problem.

The current Burgers PGD attack matrix is a different workload:

```text
single Burgers sample
model-vs-solver attack loss
four mixed backend combinations
each combo isolated in a subprocess
global GPU memory sampled during the full PGD step
```

In that current workload, the observed peak memory is:

```text
jax_solver_torch_model    = 669 MiB
torch_solver_torch_model  = 703 MiB
jax_solver_jax_model      = 747 MiB
torch_solver_jax_model    = 807 MiB
```

So the current PGD result does not contradict the old training/inverse-memory
observations. It says something narrower:

```text
For this small Burgers mixed PGD attack, the JAX model backend increases memory,
but the JAX solver backend does not increase the total measured peak relative
to the PyTorch solver backend.

Earlier model training and large NS inverse/backward experiments still show
that JAX can use much more GPU memory, especially when JAX owns the full
training graph or a large differentiated solver graph.
```

### Planned NS 25-Step PGD Matrix

The next planned run is the same four-way solver/model framework matrix for
`ns_2d`, but with only 25 PGD steps to keep runtime manageable. This run should
use native model checkpoints; no JAX-to-PyTorch model weight copying is needed.

Chosen NS attack parameters:

```text
case          = ns_2d
steps         = 25
sample index  = 1
norm          = 2
epsilon       = 32768
alpha         = 5
torch dtype   = float64
jax dtype     = float64
device        = cuda
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 adv_robust/bin/python tools/attack_framework_matrix.py \
  --cases ns_2d \
  --combos all \
  --steps 25 \
  --sample-index 1 \
  --ns-norm 2 \
  --ns-epsilon 32768 \
  --ns-alpha 5 \
  --ns-torch-solver-dtype float64 \
  --ns-jax-solver-dtype float64 \
  --device cuda \
  --gpu-sample-interval 0.02 \
  --output-root benchmark_results/attack_framework_matrix_ns_idx1_25steps_native_float64
```

Expected recorded metrics are the same as the Burgers matrix:

```text
per-step attack loss and true loss
forward total time
model forward time
solver forward time
loss evaluation time
backward / gradient time
projection time
total PGD step time
global GPU memory before/after/peak
phase-level GPU peaks
PyTorch allocated/reserved/peak memory where applicable
mixed-framework conversion times:
  forward torch -> jax
  forward jax -> torch
  backward torch -> jax
  backward jax -> torch
  JAX VJP compute time
delta norm and delta/gradient divergence versus reference combo
```

For memory cleanliness, this command should be run as a fresh Linux process.
The script runs the four combos sequentially in isolated subprocesses when
`--combos all` is used, which reduces cross-combo GPU allocator contamination.
The result directory should contain one `metrics.csv` and one `summary.json` per
combo, plus combined comparison tables and visualizations.

### Completed NS 25-Step Matrix Versus Burgers

The NS 25-step run has now been completed and should be compared against the
same-weight Burgers matrix, not the older native-checkpoint Burgers run. The
important high-level result is that Burgers and NS do not have the same memory
ordering.

For Burgers 1D with same JAX real/imag weights, runtime is mainly solver
backend dominated and memory is small:

```text
combo                     late step   active avg   peak GPU
torch_solver_torch_model  2.107 s     2.114 s      703 MiB  = 0.687 GiB
torch_solver_jax_model    2.169 s     2.199 s      807 MiB  = 0.788 GiB
jax_solver_torch_model    0.882 s     0.895 s      669 MiB  = 0.653 GiB
jax_solver_jax_model      0.910 s     0.952 s      747 MiB  = 0.729 GiB
```

Burgers late-step speedups from using the JAX solver:

```text
same Torch model: torch_solver_torch_model / jax_solver_torch_model = 2.39x
same JAX model:   torch_solver_jax_model   / jax_solver_jax_model   = 2.38x

solver forward only, same Torch model: 4.61x faster with JAX solver
solver forward only, same JAX model:   4.80x faster with JAX solver
backward, same Torch model:            3.01x faster with JAX solver
backward, same JAX model:              3.16x faster with JAX solver
```

For NS 2D, the solver backend dominates both runtime and memory:

```text
combo                     total 25-step wall time   late step   active avg   peak GPU
torch_solver_torch_model  292.91 s = 4.88 min      9.852 s     9.882 s      26349 MiB = 25.73 GiB
torch_solver_jax_model    315.60 s = 5.26 min      10.171 s    10.783 s     27035 MiB = 26.40 GiB
jax_solver_torch_model    111.07 s = 1.85 min      4.084 s     4.194 s      35127 MiB = 34.30 GiB
jax_solver_jax_model      125.53 s = 2.09 min      3.998 s     4.732 s      33591 MiB = 32.80 GiB
```

By total 25-step wall-clock time, `jax_solver_torch_model` is the fastest NS
combination in this run. The two JAX-solver runs are much faster than the two
Torch-solver runs, but `jax_solver_torch_model` finishes first despite having
the highest peak memory.

NS late-step speedups from using the JAX solver:

```text
same Torch model: torch_solver_torch_model / jax_solver_torch_model = 2.41x
same JAX model:   torch_solver_jax_model   / jax_solver_jax_model   = 2.54x

solver forward only, same Torch model: 5.22x faster with JAX solver
solver forward only, same JAX model:   5.27x faster with JAX solver
backward, same Torch model:            1.71x faster with JAX solver
backward, same JAX model:              1.87x faster with JAX solver
```

This means the current NS result is not following the Burgers memory pattern.
The observed memory ordering is:

```text
NS:
jax_solver_torch_model  highest memory, 34.30 GiB
jax_solver_jax_model    second,         32.80 GiB
torch_solver_jax_model  third,          26.40 GiB
torch_solver_torch_model lowest,        25.73 GiB

Burgers same-weight:
torch_solver_jax_model  highest memory, 0.788 GiB
jax_solver_jax_model    second,         0.729 GiB
torch_solver_torch_model third,         0.687 GiB
jax_solver_torch_model  lowest memory,  0.653 GiB
```

The difference is explainable by the workload. Burgers is a small 1D problem, so
the solver memory footprint is not the main signal; model backend, runtime
context, and mixed-framework overhead can change the ordering. NS is a large
2D spectral rollout with `nx=256`, `t_out=10`, and `dt=0.005`, so each solver
call contains about 2000 micro-steps. In that regime, the JAX/Exponax solver
itself becomes the memory-dominant object. XLA buffer planning, FFT/spectral
work buffers, scan/VJP buffers, and JAX runtime cache make the JAX solver paths
substantially larger in global `nvidia-smi` memory.

The mixed `jax_solver_torch_model` path is the largest in NS because it combines
the large JAX solver allocation with the PyTorch model runtime and allocator
state. The pure `jax_solver_jax_model` path is slightly lower, likely because
XLA can manage the model and solver buffers inside one JAX runtime instead of
coexisting with the PyTorch CUDA allocator. The gap between the two JAX-solver
NS runs is about:

```text
35127 MiB - 33591 MiB = 1536 MiB = 1.50 GiB
```

The two Torch-solver NS runs are much closer:

```text
27035 MiB - 26349 MiB = 686 MiB = 0.67 GiB
```

That smaller gap should not be overinterpreted; the robust conclusion is that
NS JAX-solver paths use about 6.4-8.6 GiB more global GPU memory than the
Torch-solver paths in this PGD workload.

### NS Timing Breakdown

For NS, step 0 includes warmup, first-use kernels, and for JAX paths, tracing
and XLA compilation. The more representative steady-state signal is the late
mean over steps 1-24. The final step 25 has no backward/update and is excluded
from the late mean.

Step 0 timing:

```text
combo                     total     forward   model fwd  solver fwd  loss      backward   projection
torch_solver_torch_model  10.596 s  6.608 s   0.182 s    6.081 s     0.344 s   3.584 s    0.009 s
torch_solver_jax_model    25.465 s  11.448 s  5.538 s    5.910 s     0.001 s   13.670 s   0.007 s
jax_solver_torch_model    6.839 s   2.093 s   0.183 s    1.889 s     0.021 s   3.716 s    0.018 s
jax_solver_jax_model      22.357 s  7.213 s   5.475 s    1.600 s     0.138 s   14.169 s   0.458 s
```

Late-step timing, steps 1-24:

```text
combo                     total     forward   model fwd  solver fwd  loss      backward   projection
torch_solver_torch_model  9.852 s   5.575 s   0.032 s    5.543 s     0.000 s   3.640 s    0.000 s
torch_solver_jax_model    10.171 s  5.683 s   0.182 s    5.501 s     0.000 s   3.892 s    0.000 s
jax_solver_torch_model    4.084 s   1.125 s   0.062 s    1.062 s     0.000 s   2.131 s    0.000 s
jax_solver_jax_model      3.998 s   1.051 s   0.008 s    1.043 s     0.000 s   2.079 s    0.000 s
```

Late-step phase percentages, using each combo's own total PGD step time:

```text
combo                     forward share  solver fwd share  backward share
torch_solver_torch_model  56.6%          56.3%             36.9%
torch_solver_jax_model    55.9%          54.1%             38.3%
jax_solver_torch_model    27.5%          26.0%             52.2%
jax_solver_jax_model      26.3%          26.1%             52.0%
```

The phase percentages do not sum to exactly 100% because the measured total step
also includes runtime cleanup, synchronization, CSV/array bookkeeping, GPU
memory sampling, and Python overhead. The meaningful comparison is still clear:
Torch solver paths spend most late-step time in forward solver rollout, while
JAX solver paths make the forward rollout much faster and shift the dominant
share toward the backward/value-and-gradient computation.

The first-step overhead is especially visible for paths that include a JAX
model:

```text
combo                     step 0 / late-step total
torch_solver_torch_model  1.08x
torch_solver_jax_model    2.50x
jax_solver_torch_model    1.67x
jax_solver_jax_model      5.59x
```

This is why first-step timing should not be used as the main runtime comparison
for JAX. Use the late-step means for steady-state PGD runtime.

If the warmup boundary is pushed later, the JAX-solver advantage becomes a
little clearer. The following ratios compare Torch solver time divided by JAX
solver time, with the model backend held fixed:

```text
NS, same Torch model
step range   total step   forward total   solver forward   backward
0 only       1.55x        3.16x           3.22x            0.96x
0-24         2.36x        4.83x           5.08x            1.66x
1-24         2.41x        4.96x           5.22x            1.71x
2-24         2.43x        5.03x           5.31x            1.75x
3-24         2.43x        5.02x           5.31x            1.75x

NS, same JAX model
step range   total step   forward total   solver forward   backward
0 only       1.14x        1.59x           3.69x            0.96x
0-24         2.28x        4.56x           5.18x            1.67x
1-24         2.54x        5.41x           5.27x            1.87x
2-24         2.41x        5.24x           5.27x            1.72x
3-24         2.41x        5.24x           5.27x            1.72x
```

So even when step 0 is included, the JAX solver is faster than the Torch solver
for NS. Excluding step 0 mostly removes JIT/first-use overhead and shows the
steady-state solver-forward gap: about `5.3x` in favor of the JAX solver.

### Why JAX Solver Is Faster Here

The runtime comparison in this matrix should be described precisely as:

```text
JAX solver   = JIT/XLA-compiled solver
Torch solver = PyTorch eager solver in this script
```

This does not mean PyTorch is theoretically unable to compile. Modern PyTorch
has compilation options such as `torch.compile`, TorchScript, CUDA graphs, and
custom fused kernels. However, the current Torch solver path in this script is
not using an equivalent whole-solver graph compiler. It is running as a PyTorch
eager implementation.

JAX behaves differently because the solver functions are wrapped in `jax.jit`.
On the first call for a given input shape and dtype, JAX traces the Python
function, lowers it to XLA, compiles it, and executes it. Later calls reuse the
compiled executable. This explains both observations:

```text
step 0 includes tracing/compilation/first-use overhead
steps after warmup are much faster and more stable
```

For the NS solver, this matters a lot. The JAX/Exponax implementation gives XLA
a large structured computation containing the spectral rollout. XLA can optimize
across the computation using graph-level transformations such as fusion,
scheduling, and buffer assignment. This can reduce Python overhead, reduce
kernel-launch overhead, and avoid materializing some intermediates in HBM.

The current PyTorch solver path does not get the same whole-graph treatment, so
it pays more eager execution overhead. Therefore the rigorous conclusion is:

```text
In this experiment, the JAX solver is faster mainly because it is compiled by
JAX/XLA, while the Torch solver is the eager baseline.
```

It would be a different experiment to compare this JAX solver against a
`torch.compile`-compiled Torch solver. That would test compiler/runtime quality
more directly. The present matrix tests the framework paths as implemented here.

### What Model / Graph Compilation Is For

In deep learning frameworks, model or graph compilation is usually introduced
for two broad engineering goals:

```text
1. reduce runtime
2. improve memory behavior
```

The primary advertised goal is usually runtime speed. Compilers try to reduce
Python/eager overhead, reduce GPU kernel launch overhead, fuse multiple tensor
operations into fewer kernels, optimize the forward and backward graphs, and
reduce unnecessary intermediate tensor reads/writes. When this works, GPU
utilization improves and wall-clock time goes down.

Memory optimization is also important, but it is not guaranteed to mean lower
peak `nvidia-smi` memory in every workload. A compiler may reduce memory by
reusing buffers, eliminating dead intermediates, fusing kernels, or avoiding
materialization of temporary tensors. But it may also increase memory by keeping
extra buffers, using larger workspaces, caching compiled executables, or choosing
a speed-oriented schedule that stores more intermediates.

Therefore compilation should be understood as an optimization tradeoff, not a
guarantee:

```text
compile may make code faster and use less memory
compile may make code faster and use more memory
compile may make code slower and use less memory
compile may make code slower and use more memory
```

The current experiments show several of these cases:

```text
JAX/XLA solver:
runtime is much faster, especially steady-state solver forward.
For NS, peak global GPU memory is much higher.

Burgers PyTorch torch.compile step solver:
runtime is slower.
peak GPU memory is slightly lower.
loss trajectory is essentially unchanged.

NS PyTorch torch.compile step solver:
runtime is slower.
peak GPU memory is much lower, about 4 GiB lower.
loss trajectory changes substantially.
```

So the right conclusion is not that compilation always improves performance.
The right conclusion is that compilation changes the execution strategy. It must
be benchmarked for each workload, especially for nonstandard workloads such as
complex-valued FFT/spectral PDE solvers.

### Burgers PyTorch Solver TorchInductor Compile Result

We tested a PyTorch solver compile variant for Burgers using:

```text
torch.compile(..., backend="inductor")
```

The first attempt compiled the whole `burgers_torch_solver` wrapper. That was
not a good compile boundary: it included solver construction, Fourier/ETDRK4
constant construction, the full 1000-step rollout loop, complex FFT operations,
and backward through that computation. This could compile for a very long time
or be killed by the container. The implementation was then changed to a more
reasonable step-level compile:

```text
construct the Burgers solver object once
compile only solver.step
keep the outer 1000-step rollout as a Python loop
```

The resulting log line should be:

```text
[torch-compile] compiling burgers_torch_solver_step
```

not the older:

```text
[torch-compile] compiling burgers_torch_solver
```

The completed step-level compile results are in:

```text
benchmark_results/attack_framework_matrix_burgers_torch_compile_solver/burgers_1d
```

Additional comparison plots against the original same-weight Burgers baseline
are in:

```text
benchmark_results/attack_framework_matrix_burgers_torch_compile_solver/burgers_1d/compare_with_baseline
```

Generated comparison plots:

```text
loss_baseline_vs_torch_compile.png
total_step_time_baseline_vs_torch_compile.png
forward_time_baseline_vs_torch_compile.png
solver_forward_time_baseline_vs_torch_compile.png
backward_time_baseline_vs_torch_compile.png
gpu_memory_baseline_vs_torch_compile.png
summary_active_total_step.png
summary_late_total_step.png
summary_peak_memory.png
summary_baseline_vs_torch_compile.csv
```

The loss curves overlap with the original eager PyTorch solver curves. This
means the compile experiment did not change the attack trajectory numerically in
this Burgers run.

However, the performance got worse. For `torch_solver_torch_model`:

```text
metric                 eager PyTorch solver   torch.compile step solver   ratio / change
late total step        2.107 s                3.949 s                     compile is 1.87x slower
late solver forward    0.707 s                1.467 s                     compile is 2.08x slower
late backward          0.949 s                1.898 s                     compile is 2.00x slower
peak GPU memory        703 MiB                657 MiB                     compile uses 46 MiB less
final loss             0.057004               0.057004                    same
```

For `torch_solver_jax_model`:

```text
metric                 eager PyTorch solver   torch.compile step solver   ratio / change
late total step        2.169 s                3.985 s                     compile is 1.84x slower
late solver forward    0.727 s                1.473 s                     compile is 2.02x slower
late backward          0.970 s                1.915 s                     compile is 1.97x slower
peak GPU memory        807 MiB                761 MiB                     compile uses 46 MiB less
final loss             0.057274               0.057274                    same
```

Compared with the JAX solver baselines, the compiled PyTorch solver is much
slower:

```text
same Torch model:
compiled PyTorch solver total step / JAX solver total step = 3.949 / 0.882 = 4.48x slower
compiled PyTorch solver solver fwd / JAX solver solver fwd = 1.467 / 0.153 = 9.57x slower

same JAX model:
compiled PyTorch solver total step / JAX solver total step = 3.985 / 0.910 = 4.38x slower
compiled PyTorch solver solver fwd / JAX solver solver fwd = 1.473 / 0.152 = 9.72x slower
```

So the Burgers result is:

```text
torch.compile did not speed up the PyTorch spectral solver.
It made forward, backward, and total step time slower.
It slightly reduced peak global GPU memory by about 46 MiB.
```

This is consistent with the TorchInductor warning observed during the run:

```text
Torchinductor does not support code generation for complex operators.
Performance may be worse than eager.
```

The warning does not mean the Python syntax is wrong, and it does not mean the
experiment must crash. It means the default `torch.compile` path through
TorchDynamo and TorchInductor does not have mature native code generation for
ordinary PyTorch complex tensors.

The relevant compilation pipeline is:

```text
torch.compile
  -> TorchDynamo captures Python/PyTorch graph regions
  -> backend="inductor" sends those graph regions to TorchInductor
  -> Inductor tries to optimize and generate code, often Triton/C++ kernels
  -> unsupported regions may graph-break or fall back to eager PyTorch
```

Fallback means that an operation or subgraph is not compiled by Inductor and is
instead executed using normal PyTorch eager machinery. A mixed compiled/eager
execution can still produce correct outputs, but it can be slower than pure
eager because it pays compilation, graph boundary, dispatch, and fallback
overheads without getting enough fused-kernel benefit.

This matters for this project because Burgers/FNO-style spectral code uses
operations such as:

```text
torch.fft.rfft
torch.fft.irfft
torch.fft.rfft2 / irfft2 in NS
native complex tensors
complex multiplication
.real / .imag
possibly torch.complex-style construction in related FNO code
```

A related PyTorch issue, `torch.compile and complex numbers #125718`, shows the
same warning on a minimal complex example. In that discussion, the basic
compiled forward can run, but compiling autograd through the complex expression
can fail. A PyTorch maintainer summarized the state bluntly as complex support
not working properly in that path. The proposed long-term direction involved a
more compiler-friendly complex tensor representation, not merely changing one
user-side compile flag.

Another related issue, `torch.compile inductor crash: F.pad followed by stft
#179807`, involved STFT/FFT with `return_complex=True`. Eager execution worked,
but `torch.compile` entered the Inductor path, emitted the same complex-operator
warning, and failed around real-to-complex FFT handling. That specific crash may
have been fixed in later PyTorch commits, but it illustrates the broader point:

```text
torch.compile + Inductor + FFT/STFT/native complex tensors is a fragile area.
It may warn, fall back, compile slowly, run slower than eager, or fail depending
on the exact graph and PyTorch version.
```

Therefore the Burgers compile result should not be interpreted as "PyTorch can
never compile PDE solvers." It should be interpreted more narrowly:

```text
For this spectral Burgers solver, with native complex FFT-heavy PyTorch code,
the current torch.compile/Inductor path is not beneficial.
The eager PyTorch solver is faster than the compiled-step PyTorch solver.
The JAX/XLA solver remains much faster than both PyTorch variants.
```

Practical engineering implications:

```text
Do not assume torch.compile will speed up FNO/spectral/FFT-heavy code.
Benchmark eager and compiled variants separately.
Keep FFT/native-complex-heavy solver pieces eager unless compile is proven useful.
Compile only real-valued subgraphs when possible.
For a more ambitious PyTorch compile attempt, rewrite complex math in real/imag
form and compile smaller kernels, but FFT calls may still rely on library paths
and may not fuse like ordinary real tensor arithmetic.
```

### NS PyTorch Solver TorchInductor Compile Result

The same step-level PyTorch solver compile experiment was also run for NS:

```text
construct the NS solver object once
compile only solver.step
keep the outer integer-second / micro-step rollout as Python loops
```

The expected log line is:

```text
[torch-compile] compiling ns_torch_solver_step
```

The completed results are in:

```text
benchmark_results/attack_framework_matrix_ns_torch_compile_solver/ns_2d
```

Additional comparison plots against the original NS eager/JAX baseline are in:

```text
benchmark_results/attack_framework_matrix_ns_torch_compile_solver/ns_2d/compare_with_baseline
```

Generated comparison plots:

```text
loss_baseline_vs_torch_compile.png
total_step_time_baseline_vs_torch_compile.png
forward_time_baseline_vs_torch_compile.png
solver_forward_time_baseline_vs_torch_compile.png
backward_time_baseline_vs_torch_compile.png
gpu_memory_baseline_vs_torch_compile.png
summary_active_total_step.png
summary_late_total_step.png
summary_peak_memory.png
summary_final_loss.png
summary_baseline_vs_torch_compile.csv
```

Unlike Burgers, the NS loss curves do not overlap after PyTorch solver compile.
The compiled Torch-solver attacks grow more slowly and end at a substantially
lower attack loss:

```text
combo                     eager final loss   torch.compile final loss
torch_solver_torch_model  3.0201             2.3415
torch_solver_jax_model    3.0061             2.3356
```

So the NS compile run is not behaviorally equivalent to the eager PyTorch solver
PGD attack. The compile run is slower and lower-memory, but it also follows a
different attack trajectory.

For `torch_solver_torch_model`:

```text
metric                 eager PyTorch solver   torch.compile step solver   ratio / change
late total step        9.852 s                14.205 s                    compile is 1.44x slower
late solver forward    5.543 s                6.967 s                     compile is 1.26x slower
late backward          3.640 s                6.812 s                     compile is 1.87x slower
peak GPU memory        26349 MiB = 25.73 GiB 22257 MiB = 21.74 GiB       compile uses 4092 MiB less
final loss             3.0201                 2.3415                      compile attack loss is lower
```

For `torch_solver_jax_model`:

```text
metric                 eager PyTorch solver   torch.compile step solver   ratio / change
late total step        10.171 s               14.533 s                    compile is 1.43x slower
late solver forward    5.501 s                6.862 s                     compile is 1.25x slower
late backward          3.892 s                6.982 s                     compile is 1.79x slower
peak GPU memory        27035 MiB = 26.40 GiB 22915 MiB = 22.38 GiB       compile uses 4120 MiB less
final loss             3.0061                 2.3356                      compile attack loss is lower
```

Compared with the JAX solver baselines, the compiled PyTorch solver is still
much slower:

```text
same Torch model:
compiled PyTorch solver total step / JAX solver total step = 14.205 / 4.084 = 3.48x slower

same JAX model:
compiled PyTorch solver total step / JAX solver total step = 14.533 / 3.998 = 3.64x slower
```

The time/memory tradeoff is therefore:

```text
Burgers compile:
time gets worse
memory gets slightly better
loss trajectory stays essentially identical

NS compile:
time gets worse
memory gets much better, about 4 GiB lower
loss trajectory changes substantially and final loss is much lower
```

Because the NS loss trajectory changed, additional heatmap comparisons were
generated to inspect the attack inputs and outputs directly:

```text
torch_solver_torch_model_eager_vs_torch_compile_heatmaps.png
torch_solver_torch_model_torch_compile_minus_eager_heatmaps.png
torch_solver_jax_model_eager_vs_torch_compile_heatmaps.png
torch_solver_jax_model_torch_compile_minus_eager_heatmaps.png
```

The side-by-side heatmaps compare:

```text
input x0 last frame
attacked input x_adv
input perturbation
solver final from x0
solver final from x_adv
solver output change
model - solver at x_adv
```

The final attacked inputs and outputs differ strongly between eager and
compiled PyTorch solver attacks:

```text
torch_solver_torch_model:
x_adv compile - eager L2       = 157.74
x_adv compile - eager Linf     = 0.948
solver final diff L2           = 967.49
solver final diff Linf         = 6.058
model-solver diff L2           = 1152.22
model-solver diff Linf         = 8.608

torch_solver_jax_model:
x_adv compile - eager L2       = 157.39
x_adv compile - eager Linf     = 0.929
solver final diff L2           = 956.10
solver final diff Linf         = 6.023
model-solver diff L2           = 1128.68
model-solver diff Linf         = 8.720
```

This supports the visual observation: the compiled PyTorch solver attack does
not just run slower; it finds a different perturbation. The likely mechanism is
that compile/fallback/autograd segmentation changes the gradient path for this
long complex FFT-heavy differentiated solver. PGD is highly gradient-sensitive,
so even small gradient differences can lead to a different attack trajectory
over 25 steps. In NS the long 2D rollout appears sensitive enough that this
effect is visible in the loss curves and heatmaps. In Burgers the same compile
experiment did not visibly change the loss trajectory.

The NS compile result should therefore be interpreted cautiously:

```text
It is useful as a performance/memory experiment.
It is not behaviorally interchangeable with the eager PyTorch solver attack.
Before using it for robustness conclusions, compare eager and compile forward
outputs and input gradients at the same fixed input.
```

### Why Memory Ordering Changes Between Burgers And NS

The memory result should also not be summarized as "JAX always uses less memory"
or "JAX always uses more memory." The allocator and workload details matter.

PyTorch uses a CUDA caching allocator. It requests GPU memory from CUDA, keeps
freed blocks in a cache, and reuses them to avoid slow synchronization-heavy
allocation/deallocation. Therefore `nvidia-smi` memory for PyTorch includes:

```text
live tensors
PyTorch reserved/cached allocator blocks
CUDA context and library workspaces
```

JAX/XLA also uses a GPU memory allocator and memory reuse strategy. By default,
JAX can preallocate a large fraction of GPU memory at first use. In this script
that default behavior is disabled with:

```python
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
```

So the current measurements are not simply caused by JAX's default 75% GPU
preallocation. Even with preallocation disabled, JAX/XLA still allocates and
reuses memory for compiled executables, temporary buffers, FFT/spectral
workspaces, and autodiff residuals.

The important distinction is that XLA sees a compiled computation and performs
static buffer planning. This can improve performance, but it does not guarantee
the lowest possible `nvidia-smi` peak. For differentiated computations,
reverse-mode autodiff may also save forward-pass residuals for the backward
pass. JAX has `jax.checkpoint` / `jax.remat` to trade extra recomputation for
lower memory, but this script does not manually rematerialize the solver.

This explains the observed reversal:

```text
Burgers:
1D, small solver, small rollout.
The JAX solver compiled buffers are small.
Memory ordering is dominated by model backend/runtime/mixed-framework effects.
Result: jax_solver_torch_model is the lowest-memory combo.

NS:
2D, nx=256, long spectral rollout, about 2000 micro-steps per solver call.
The JAX solver compiled graph, FFT/spectral temporaries, scan/VJP buffers, and
autodiff residuals become the dominant memory object.
Result: JAX-solver paths use much more memory; jax_solver_torch_model is the
highest-memory combo.
```

The mixed `jax_solver_torch_model` path is especially expensive in NS memory
because it combines the large JAX solver allocation with the PyTorch model
runtime and CUDA caching allocator. The pure `jax_solver_jax_model` path still
uses the large JAX solver allocation, but it avoids coexisting with the PyTorch
model allocator, and XLA may manage model and solver buffers within one runtime.

Therefore the best summary is:

```text
Runtime:
JAX solver is faster here mainly because it is JIT/XLA compiled.
The current Torch solver is an eager PyTorch baseline.

Memory:
Burgers is too small for the JAX solver memory footprint to dominate.
NS is large enough that the JAX solver memory footprint dominates.
The apparent memory rule changes because the dominant memory term changes.
```

### NS Loss And Visualization Notes

The NS loss progressions are nearly identical in shape across the four
framework combinations. The final losses are:

```text
torch_solver_torch_model  3.020094
jax_solver_torch_model    3.020150
torch_solver_jax_model    3.006072
jax_solver_jax_model      3.006116
```

The remaining small split is by model backend, not solver backend. Torch-model
paths end around `3.020`, while JAX-model paths end around `3.006`. The solver
choice changes final loss only in the fourth decimal place here.

The NS GIF/first/final-frame visualization was expanded to show the actual
perturbation and output differences. Each combo row now shows:

```text
input x0 last frame
input x_adv last frame
input perturbation
solver final from x0
solver final from x_adv
solver output change
model output change
model - solver at x_adv
```

Updated files:

```text
benchmark_results/attack_framework_matrix_ns_idx1_25steps_native_float64/ns_2d/ns_2d_attack_all_combos.gif
benchmark_results/attack_framework_matrix_ns_idx1_25steps_native_float64/ns_2d/ns_2d_attack_first_frame.png
benchmark_results/attack_framework_matrix_ns_idx1_25steps_native_float64/ns_2d/ns_2d_attack_final_frame.png
```

## Current Conclusion

The most important finding is:

```text
Burgers and NS both show that the JAX solver gives a clear steady-state runtime
speedup in this PGD attack matrix.

However, their memory ordering is different.
Burgers is small enough that model/runtime/backend-mixing effects dominate the
small memory differences.
NS is large enough that the JAX solver itself dominates memory, so both
JAX-solver paths use substantially more global GPU memory than Torch-solver
paths.

For NS, the JAX solver is about 2.4-2.5x faster per late PGD step, but costs
about 6.4-8.6 GiB more peak global GPU memory.
```

Therefore the current conclusion should not be "JAX always uses less memory" or
"mixed JAX/Torch always has the same memory ordering." The better conclusion is:

```text
Runtime benefit from the JAX solver is stable across Burgers and NS.
Memory behavior is PDE/workload dependent.
For small Burgers, memory ordering is weak and backend-mixing/model effects can
dominate.
For large NS, JAX solver memory dominates and reverses the Burgers ordering.
```
