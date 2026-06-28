# A100 Batched Generation Results

These results were measured on `NVIDIA A100-SXM4-80GB` with
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and `CUDA_VISIBLE_DEVICES=0`.

## 1D Burgers

Script:

```bash
1D_Burgers/data_generation/generate_burgers_exponax_batched.py
```

Benchmark setup: `N=128`, `nx=1024`, `nu=0.0005`, `t_final=1`.

| batch size | sec/sample | max diff vs batch=1 |
|---:|---:|---:|
| 1 | 0.1439 | 0 |
| 16 | 0.0114 | 0 |
| 64 | 0.0047 | 0 |
| 128 | 0.0046 | 0 |

Peak monitored GPU memory was about `458 MiB`.

## 2D Navier-Stokes, `solver_mode=vmap`

Script:

```bash
2D_NS_FNO2d_recurrent/data_generation/generate_ns_real_initial_batched.py
```

Benchmark setup: `N=32`, `target_size=256`, `t_final=20`,
`fixed_step=0.005`, source initial conditions from dataset key `x`.

| solver batch size | sec/sample | global max abs vs batch=1 | global RMSE | mean relative RMSE |
|---:|---:|---:|---:|---:|
| 1 | 1.9429 | 0 | 0 | 0 |
| 4 | 0.5758 | 0.000593 | 1.35e-5 | 1.13e-5 |
| 8 | 0.3707 | 0.000593 | 1.35e-5 | 1.13e-5 |
| 16 | 0.2570 | 0.071441 | 0.001755 | 0.001479 |
| 32 | 0.2112 | 0.071441 | 0.001755 | 0.001479 |

Peak monitored GPU memory was about `1.6 GiB`. `vmap` is fast, but it can
change the floating-point execution path enough to differ from sequential
`batch_size=1`, especially at solver batch sizes `16` and `32`.

Machine-readable result:

```bash
benchmark_results/ns_batch_compare_vmap_vs_batch1_N32_A100.json
```

## 2D Navier-Stokes, `solver_mode=lax-map`

Benchmark setup: `N=32`, `target_size=256`, `t_final=20`,
`fixed_step=0.005`, source initial conditions from dataset key `x`.

| solver batch size | sec/sample | global max abs vs batch=1 | global RMSE | mean relative RMSE |
|---:|---:|---:|---:|---:|
| 1 | 1.9369 | 0 | 0 | 0 |
| 8 | 1.8451 | 0 | 0 | 0 |
| 16 | 1.8449 | 0 | 0 | 0 |
| 32 | 1.8444 | 0 | 0 | 0 |

Peak monitored GPU memory was about `1.6 GiB`. `lax-map` kept exact agreement
with sequential `batch_size=1` in this test, but it only gave a small speedup.

Machine-readable result:

```bash
benchmark_results/ns_batch_compare_laxmap_vs_batch1_N32_A100.json
```

