# Batched Data Generation Summary

## 1D Burgers

Script:

```bash
1D_Burgers/data_generation/generate_burgers_exponax_batched.py
```

The 1D Burgers generator does not read an existing dataset. It samples GRF
initial conditions and solves with Exponax.

Benchmark on A100 with `N=128`, `nu=0.0005`, `t_final=1`, `nx=1024`:

| batch size | sec/sample | max diff vs batch=1 |
|---:|---:|---:|
| 1 | 0.1439 | 0 |
| 16 | 0.0114 | 0 |
| 64 | 0.0047 | 0 |
| 128 | 0.0046 | 0 |

Peak GPU memory in the monitored run was about `458 MiB`. For this case,
batching is both faster and numerically identical in the tested setting.

Recommended full run:

```bash
RUN_NAME=burgers_full GPU_MONITOR_INTERVAL=5 \
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
bash tools/run_with_gpu_monitor.sh \
adv_robust/bin/python -u 1D_Burgers/data_generation/generate_burgers_exponax_batched.py \
  --num-records 1500 \
  --nu 0.01,0.0005 \
  --batch-size 256 \
  --benchmark \
  --benchmark-samples 64 \
  --benchmark-batch-sizes 1,16,64,128,256
```

Default output:

```bash
1D_Burgers/datasets/1D/Burgers/batched_exponax/
```

## 2D Navier-Stokes Real Initial

Script:

```bash
2D_NS_FNO2d_recurrent/data_generation/generate_ns_real_initial_batched.py
```

The 2D NS generator reads only `x` from:

```bash
2D_NS_FNO2d_recurrent/datasets/source_zongyi_real_initial/
```

Then it spectrally upsamples each initial condition from `64x64` to `256x256`
and solves with Exponax. It saves only integer-second frames, not every tiny
internal PDE step.

The script has three batch/reproducibility knobs:

- `--solver-batch-size`: how many samples enter one JAX/Exponax rollout.
- `--batch-size`: how many samples are processed before copying results back to
  CPU. The split is still saved as one complete `.pt` file.
- `--solver-mode`: `vmap` is faster; `lax-map` keeps the same numerical path as
  sequential `batch_size=1` in the tests below.

The solver batch size affects speed, memory, and numerical reproducibility. The
processing batch size affects peak CPU/GPU transfer size and progress cadence,
but not how many final `.pt` files are written.

Fast `vmap` benchmark on A100 with `N=32`, `t_final=20`, `fixed_step=0.005`,
reference `batch_size=1`:

| solver batch size | sec/sample | global max abs vs batch=1 | global RMSE | mean relative RMSE |
|---:|---:|---:|---:|---:|
| 1 | 1.9429 | 0 | 0 | 0 |
| 4 | 0.5758 | 0.000593 | 1.35e-5 | 1.13e-5 |
| 8 | 0.3707 | 0.000593 | 1.35e-5 | 1.13e-5 |
| 16 | 0.2570 | 0.071441 | 0.001755 | 0.001479 |
| 32 | 0.2112 | 0.071441 | 0.001755 | 0.001479 |

Peak GPU memory in the monitored `vmap` comparison run was about `1.6 GiB`.
Memory is not the bottleneck on an A100 80GB for this configuration; numerical
agreement with the sequential solver is the main tradeoff.

Exact `lax-map` benchmark on A100 with `N=32`, `t_final=20`, `fixed_step=0.005`,
reference `batch_size=1`:

| solver batch size | sec/sample | global max abs vs batch=1 | global RMSE | mean relative RMSE |
|---:|---:|---:|---:|---:|
| 1 | 1.9369 | 0 | 0 | 0 |
| 8 | 1.8451 | 0 | 0 | 0 |
| 16 | 1.8449 | 0 | 0 | 0 |
| 32 | 1.8444 | 0 | 0 | 0 |

Peak GPU memory in the monitored `lax-map` comparison run was about `1.6 GiB`.
This mode preserved exact agreement with the sequential solver for the tested
samples, but it only gave a small speedup.

Recommended exact full run. This is also the script default now:

```bash
RUN_NAME=ns_real_initial_full_laxmap GPU_MONITOR_INTERVAL=5 \
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
bash tools/run_with_gpu_monitor.sh \
adv_robust/bin/python -u 2D_NS_FNO2d_recurrent/data_generation/generate_ns_real_initial_batched.py \
  --splits test,train \
  --batch-size 64 \
  --solver-batch-size 32 \
  --solver-mode lax-map \
  --fixed-step 0.005 \
  --t-final 20
```

Faster optional full run, if small `vmap` differences are acceptable:

```bash
RUN_NAME=ns_real_initial_full_vmap GPU_MONITOR_INTERVAL=5 \
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
bash tools/run_with_gpu_monitor.sh \
adv_robust/bin/python -u 2D_NS_FNO2d_recurrent/data_generation/generate_ns_real_initial_batched.py \
  --splits test,train \
  --batch-size 64 \
  --solver-batch-size 8 \
  --solver-mode vmap \
  --fixed-step 0.005 \
  --t-final 20
```

Default output:

```bash
2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/
```

Each split is saved as one complete `.pt` file:

```bash
2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt
2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt
```

Detailed comparison report:

```bash
NS_BATCH_SIZE_COMPARISON.md
NS_BATCH_SIZE_COMPARISON_LAXMAP_N32.md
benchmark_results/ns_batch_compare_vmap_vs_batch1_N32_A100.json
benchmark_results/ns_batch_compare_laxmap_vs_batch1_N32_A100.json
```
