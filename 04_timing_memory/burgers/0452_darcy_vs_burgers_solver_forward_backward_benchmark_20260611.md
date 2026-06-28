# Darcy vs Burgers solver forward/backward benchmark - 2026-06-11

Goal: verify whether the current Darcy-flow solver used by adversarial training is much faster than the current Burgers solver, both in forward solve and in backward differentiation through the solver.

## Setup

- Machine: Tesla V100-SXM2-32GB.
- PyTorch: `2.8.0+cu126`; JAX: `0.10.0`, GPU backend.
- Benchmark script: `tools/benchmark_darcy_vs_burgers_solver_runtime_20260611.py`.
- Main output: `analysis_outputs/darcy_vs_burgers_solver_forward_backward_benchmark_20260611/solver_runtime_summary.json`.
- Larger Burgers batch check: `analysis_outputs/darcy_vs_burgers_solver_forward_backward_benchmark_20260611_large_burgers_batches/solver_runtime_summary.json`.

This is a solver-only benchmark: no neural operator forward pass, no optimizer step, no train/test evaluation, and no checkpoint I/O.

## Solver settings

Darcy:

- Input coefficients from the actual binary Darcy train set: `2D_Darcy_FNO2d/datasets/grf_darcy_20260528_N1500/train/dim2d_darcy_nx85_N1200_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_train.pt`.
- Coefficient values are exactly `[3.0, 12.0]`.
- Runtime solver input is `85 x 85`, matching the model/adversarial-training input.
- Solver: JAX batched CG, `tol=1e-5`, differentiated by JAX VJP/implicit differentiation.

Burgers:

- Solver: project `Burgers1DETDRK4`.
- Input size `N=1024`, `dt=0.001`, `t_final=1.0`, so `1000` ETDRK4 steps.
- `diffusivity=1e-3`, non-conservative form, dealiasing `2/3`.
- Local Burgers dataset was not needed for this timing-only check, so smooth deterministic random periodic inputs were used.
- `remat_mode=none`, matching the default solver path in `tools/adversarial_training.py`.

For each batch, the benchmark records:

- `forward_no_grad`: solver forward without autograd graph.
- `forward_with_grad`: solver forward while building a graph/VJP path.
- `backward_only`: only the time for `loss.backward()` after forward has finished.

Darcy's first call includes JAX compilation, so the table below uses the warmed-up means.

## Same-batch results

| batch | Darcy forward_with_grad | Darcy backward_only | Darcy backward/forward | Burgers forward_with_grad | Burgers backward_only | Burgers backward/forward | Burgers/Darcy forward | Burgers/Darcy backward |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00948s | 0.01978s | 2.09x | 0.85522s | 1.37897s | 1.61x | 90.17x | 69.73x |
| 4 | 0.01253s | 0.02399s | 1.92x | 0.85916s | 1.36196s | 1.59x | 68.60x | 56.78x |
| 8 | 0.01361s | 0.02587s | 1.90x | 0.86495s | 1.37320s | 1.59x | 63.54x | 53.08x |

## Larger-batch check

| solver | batch | forward_with_grad | backward_only | backward/forward |
|---|---:|---:|---:|---:|
| Darcy | 64 | 0.02317s | 0.04253s | 1.84x |
| Burgers | 16 | 0.84689s | 1.35751s | 1.60x |
| Burgers | 32 | 0.91488s | 1.43250s | 1.57x |

## Interpretation

The current Darcy solver is indeed much faster than the Burgers solver in this code path. On equal batches `1/4/8`, Burgers forward-with-grad is about `64x-90x` slower, and Burgers backward-only is about `53x-70x` slower.

The user's memory that backward is often about twice the forward time is basically correct for Darcy after warm-up: Darcy `backward_only / forward_with_grad` is `1.84x-2.09x` across the measured batches. If using `forward_no_grad` as the forward baseline, Darcy backward is also about `1.9x-2.0x`.

Burgers backward-only is `1.56x-1.61x` of its graph-building forward in this benchmark, but it is roughly `1.9x` of the no-grad forward. The combined forward-plus-backward cost is about `3.1x` of a no-grad forward for Burgers.

The main reason the total solver times differ so much is structural: the Darcy path is one steady elliptic CG solve on an `85 x 85` grid with a batched JAX solver and VJP, while Burgers is a time rollout of `1000` ETDRK4 steps at `N=1024`, with FFT-heavy operations repeated at every step.
