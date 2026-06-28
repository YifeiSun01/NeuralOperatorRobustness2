# Burgers FNO PyTorch vs JAX Current Comparison

This report summarizes the current 1D Burgers FNO runs:

- Profile run: `fno_training_runs/burgers_profile_500_fair`
- No-profile run: `fno_training_runs/burgers_no_profile_500_fair`
- Dataset: `nu=0.001`, train `1350`, test `150`
- Model: `modes=16`, `width=64`, `num_layers=4`, `epochs=500`, `batch_size=64`
- Both configs show `frameworks=pytorch,jax` and `align_jax_init_with_pytorch=true`.

## Total Time And Final Accuracy

| run | framework | total seconds | epoch/s | train rel L2 | test rel L2 | test MSE |
|---|---:|---:|---:|---:|---:|---:|
| profile | PyTorch | 77.45 | 6.46 | 0.01682 | 0.01752 | 9.114e-5 |
| profile | JAX | 96.57 | 5.18 | 0.02082 | 0.02379 | 1.673e-4 |
| no-profile | PyTorch | 82.04 | 6.09 | 0.01682 | 0.01752 | 9.114e-5 |
| no-profile | JAX | 86.24 | 5.80 | 0.01936 | 0.02268 | 1.567e-4 |

PyTorch is more accurate in both runs. JAX is close but consistently higher-loss on this setup.

## Loss Curve Check

Profile run selected test relative L2:

| epoch | PyTorch | JAX | JAX / PyTorch |
|---:|---:|---:|---:|
| 1 | 0.35120 | 0.35608 | 1.01 |
| 5 | 0.10155 | 0.30207 | 2.97 |
| 10 | 0.04906 | 0.18433 | 3.76 |
| 50 | 0.02630 | 0.07373 | 2.80 |
| 100 | 0.01748 | 0.04924 | 2.82 |
| 200 | 0.01500 | 0.03596 | 2.40 |
| 500 | 0.01752 | 0.02379 | 1.36 |

The visual impression is correct: PyTorch loss drops much faster early, and stays lower for most of training.

## Profile Memory

| framework | whole-run sampled peak | profiled train peak | inference peak |
|---|---:|---:|---:|
| PyTorch | 1407 MiB | 505.84 MiB | 36.28 MiB |
| JAX | 2733 MiB | 2729 MiB | 2733 MiB |

PyTorch uses `torch.cuda` allocator statistics for detailed peaks. JAX memory is from `nvidia-smi`, so it reports process/GPU occupancy rather than per-op allocator peak.

## Forward And Backward Timing

The first profiled row includes compilation/warmup effects. Steady-state numbers below exclude the first profiled row.

| framework | forward | backward | optimizer |
|---|---:|---:|---:|
| PyTorch | 2.43 ms | 3.36 ms | 0.72 ms |
| JAX | 1.94 ms | 5.27 ms | 2.03 ms |

JAX forward is slightly faster in the steady-state profiled batch samples, but the raw JAX `backward` row is not a pure reverse pass. In the current code, PyTorch `backward` times `loss.backward()` after an already-recorded forward, while JAX `backward` times `jax.value_and_grad(loss_fn)`, which recomputes the loss forward and then computes gradients. A rough pure-JAX-reverse estimate is therefore `5.27 ms - 1.94 ms = 3.33 ms`, very close to PyTorch's `3.36 ms`.

The JAX optimizer row is a separately jitted Adam/PyTree state update plus synchronization. The no-profile training path still uses the fused `train_step`; the detailed profiling rows are measurement probes and should be read as phase-level approximations, not as an XLA internal timeline.

## Memory And Timing Semantics

PyTorch timing and memory are more directly phase-specific:

- `forward`: `model(xb)` and loss computation, bracketed by `torch.cuda.synchronize()`.
- `backward`: `loss.backward()`, bracketed by `torch.cuda.synchronize()`.
- `optimizer_step`: `optimizer.step()`, bracketed by `torch.cuda.synchronize()`.
- memory: `torch.cuda.memory_allocated/reserved/max_memory_allocated/max_memory_reserved` after resetting the peak counter for each phase.

JAX timing is synchronized with `jax.block_until_ready()`, so the wall-clock phase times are real for the staged functions being measured. However, JAX memory is only from `nvidia-smi`, not a JAX allocator-level phase counter. It reflects how much GPU memory the JAX/XLA process/runtime is holding, not how much each internal op temporarily needs. This is why JAX `whole peak`, `train peak`, and `inference peak` are nearly the same.

## Inference Sample Comparison

Profile run, 5 held-out samples:

| sample | PyTorch vs JAX rel L2 | PyTorch vs target rel L2 | JAX vs target rel L2 |
|---:|---:|---:|---:|
| 0 | 0.03316 | 0.01672 | 0.02460 |
| 1 | 0.02586 | 0.01669 | 0.01345 |
| 2 | 0.02985 | 0.01531 | 0.02117 |
| 3 | 0.01993 | 0.01376 | 0.01218 |
| 4 | 0.02983 | 0.02105 | 0.02328 |

Aggregate profile inference:

- PyTorch vs JAX prediction rel L2: `0.02835`
- PyTorch vs target rel L2: `0.01650`
- JAX vs target rel L2: `0.01966`
- PyTorch vs JAX max abs: `0.15473`

The two frameworks are using the same input and target arrays exactly, but trained predictions are not identical.

## Interpretation

The current profile run is internally consistent: it used `FRAMEWORKS=pytorch,jax`, `align_jax_init_with_pytorch=true`, and the corrected profiling path where true JAX parameter updates go through the same `train_step` as no-profile mode.

The remaining PyTorch/JAX differences are real training-dynamics differences in this implementation, not a stale JAX-only profile initialization issue. Likely contributors include framework-specific FFT/complex arithmetic, Adam implementation details, XLA vs CUDA kernel numerics, and small floating-point differences accumulating over 500 epochs.

The no-profile run was created before the latest profile cleanup, but its no-profile training path is not the part that was fixed. For the strictest possible profile-vs-no-profile table, rerunning no-profile once with the current code would remove that historical caveat; for current memory and forward/backward conclusions, the profile run is the relevant source.

## Gradient Audit

Gradient audit output:

- `gradient_audit/fno1d_burgers_steps_0_9_40_50_100_110/gradient_summary.csv`
- `gradient_audit/fno1d_burgers_steps_0_9_40_50_100_110/gradient_blocks.csv`
- `gradient_audit/fno1d_burgers_steps_0_9_40_50_100_110/plots/`

This audit follows each framework's own training trajectory after the shared initialization. Therefore early steps compare almost identical parameters, while later steps show the accumulated divergence of the two training paths.

| step range | mean grad rel L2 | max grad rel L2 | mean objective abs diff | min cosine | mean cosine |
|---|---:|---:|---:|---:|---:|
| 0-9 | 0.0184 | 0.1583 | 0.000740 | 0.9885 | 0.9988 |
| 40-50 | 0.5298 | 1.0162 | 0.08335 | 0.2513 | 0.8168 |
| 100-110 | 3.0711 | 14.5508 | 0.17571 | -0.5950 | -0.1342 |

The early gradients are almost identical in aggregate: step 0 has gradient relative L2 `2.41e-4` and cosine `0.99999997`. The largest early block-level differences are already concentrated in the complex spectral convolution weights (`conv_layers.*.weights1`). By steps `40-50`, the training trajectories are visibly separated. By steps `100-110`, several gradient vectors have negative cosine similarity, meaning the two optimizers are often moving in substantially different directions.
