# NS2D dictionary solver batch-size probe - 2026-05-22

## What was tested

- GPU: NVIDIA A100-SXM4-80GB, empty at start and after the probe.
- Environment: PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, JAX backend `gpu`.
- Code path: `tools/benchmark_ns_dictionary_solver_batch_size.py` imports `make_ns_rollout_fn` from `2D_NS_FNO2d_recurrent/data_generation/generate_ns_dictionary_batched.py`.
- Solver: `ex.stepper._navier_stokes.NavierStokesVorticityZongyi(2, ...)`, so this is the 2D Navier-Stokes vorticity solver, not Burgers.
- Probe settings: `nx=256`, `t_final=20`, `fixed_step=0.01`, `nu=1e-5`, `solver_mode=vmap`, GRF initial conditions.
- This was a forward solver probe only. It did not generate the full dictionary and did not upload to R2.

## Observed table

| solver batch | status | finite / total | nonfinite | seconds | seconds/sample | JAX peak in-use GiB | JAX peak pool GiB | Torch peak reserved GiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | `ok` | 10 / 10 | 0 | 2.926 | 0.2926 | 0.252 | 0.283 | 0.100 |
| 20 | `ok` | 20 / 20 | 0 | 3.662 | 0.1831 | 0.350 | 0.533 | 0.197 |
| 32 | `nonfinite` | 31 / 32 | 1 | 4.604 | 0.1439 | 0.658 | 1.033 | 0.314 |
| 48 | `nonfinite` | 46 / 48 | 2 | 5.885 | 0.1226 | 0.737 | 1.033 | 0.471 |
| 64 | `nonfinite` | not recorded | not recorded | 7.185 | 0.1123 | 0.955 | 2.033 | 0.627 |
| 96 | `nonfinite` | not recorded | not recorded | 11.188 | 0.1165 | 1.471 | 2.033 | 0.939 |
| 128 | `nonfinite` | not recorded | not recorded | 14.829 | 0.1159 | 1.893 | 4.033 | 1.252 |
| 192 | `nonfinite` | not recorded | not recorded | 20.735 | 0.1080 | 2.940 | 4.033 | 1.877 |
| 256 | `nonfinite` | 242 / 256 | 14 | 26.680 | 0.1042 | 3.768 | 8.033 | 2.502 |
| 384 | `nonfinite` | 365 / 384 | 19 | 38.441 | 0.1001 | 5.752 | 8.033 | 3.732 |
| 512 | `nonfinite` | 485 / 512 | 27 | 50.343 | 0.0983 | 7.518 | 16.033 | 4.982 |
| 768 | `nonfinite` | 715 / 768 | 53 | 73.858 | 0.0962 | 11.268 | 16.033 | 7.463 |

## Interpretation

- `solver_batch_size=10` and `20` passed fully finite at `dt=0.01`.
- `solver_batch_size=32` and above did not fail from memory; they produced some nonfinite samples at `dt=0.01`. For example, batch 32 had `31/32` finite and batch 48 had `46/48` finite.
- `solver_batch_size=512` completed a 2D NS forward solve in `50.343s`, with `485/512` finite and `27/512` nonfinite. JAX peak in-use memory was about `7.52 GiB`, with peak pool about `16.03 GiB`.
- `solver_batch_size=768` also completed before the probe was stopped, with `715/768` finite and `53/768` nonfinite. JAX peak in-use memory was about `11.27 GiB`, with peak pool about `16.03 GiB`.
- Therefore the current limit in these probes is numerical stability at `dt=0.01`, not A100 memory. The nonfinite samples are the ones that need fallback to `dt=0.0005`.

## Runtime estimate for N=2000

- A pure first-pass `dt=0.01` run with `solver_batch_size=512` would need about four large chunks for 2000 samples. Using the observed `50.343s` per 512-sample chunk gives about `3.3 minutes` for the first pass. This is a rough estimate because JAX compilation/cache behavior differs between isolated probes and one long run.
- The observed nonfinite rate around batch 512 was `27/512 = 5.3%`, so for 2000 samples a rough expectation is about `100` fallback samples.
- Fallback at `dt=0.0005` has 20x more solver micro-steps than `dt=0.01`, but it applies only to the nonfinite subset. A practical compute estimate is roughly `8-15 minutes` total solver time if the fallback rate stays near 5%.
- Add CPU copy / `torch.save` for an `11-12 GB` `.pt` file and R2 upload. Planning estimate for full generate+save+upload is about `15-30 minutes`, assuming network speed similar to the prior R2 upload. If many more samples fall back to `dt=0.0005`, it can be longer.

## Recommendation

- For a conservative production dictionary run, use `--solver-batch-size 20`; it was fully finite in the probe and very safe in memory.
- For speed, `--solver-batch-size 512` is reasonable on this A100 because memory use is far below 80GB; rely on the generator fallback path to rerun nonfinite samples at `dt=0.0005`.
- A good production command is `--batch-size 512 --solver-batch-size 512 --solver-mode vmap --step-options 0.01,0.0005`.
