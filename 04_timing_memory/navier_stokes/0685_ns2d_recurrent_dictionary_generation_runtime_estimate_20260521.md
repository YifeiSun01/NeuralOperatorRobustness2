# NS2D recurrent dictionary generation runtime estimate - 2026-05-21

## Source inspected

- Generator script: `2D_NS_FNO2d_recurrent/data_generation/VT_NS_gen_all_frame_dict.py`
- Wrapper script: `2D_NS_FNO2d_recurrent/data_generation/VT_NS_gen_all_frame_dict.sh`
- Historical log: `2D_NS_FNO2d_recurrent/data_generation/logs_gen/VT_NS_gen_all_frame_dict_7837625.out`
- Local dictionary directory checked: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary`

## Observed from source

- The dictionary generator creates `N=2000` samples.
- Output name is expected to be:
  `dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`
- Each sample stores `x` plus `y` frames with `ntimepoints=tfinal+1=21` at `256 x 256`.
- The script loops over samples one by one inside one batch of 2000.
- For each sample, it tries solver step sizes in this order by default:
  `0.01`, `0.005`, `0.001`, `0.0005`, `0.0001`.
- The first stable step size is used and saved.
- The generator now accepts `NS_DICT_STEP_OPTIONS`; for the requested fast run use `NS_DICT_STEP_OPTIONS=0.01,0.0005`, meaning try `dt=0.01` first and jump directly to `dt=0.0005` if that sample is unstable.
- The generator now accepts `NS_DICT_BATCH_SIZE`; for the requested single-file run use `NS_DICT_BATCH_SIZE=2000`. In this script, batch size controls output file batching, while the solver still loops over samples one at a time.
- The wrapper requested a B200 GPU with a 30 hour wall time, but that is a scheduler limit, not the observed runtime.

## Observed from historical full dictionary log

- Hardware in the log: `NVIDIA B200`, 178.36 GiB GPU memory.
- Environment in the log: PyTorch `2.7.0+cu128`, JAX `0.6.0`, JAX backend `gpu`.
- Full run progress: `2000/2000` samples completed in `2:03:34`.
- Average observed rate: about `3.71 s/sample`.
- Step-size counts:
  - `1875/2000` samples were stable at solver step `0.01`.
  - `125/2000` samples needed fallback to solver step `0.005`.
- No failed samples were observed in that log.
- The log ended with a successful save message for the expected dictionary `.pt` file.

## Current local state observed

- Current GPU query at this inspection reported `NVIDIA A100-SXM4-80GB`.
- The local dictionary directory exists, but the actual `N2000 ... all_frames.pt` dictionary file is not present there.
- Current local dictionary directory size is only about `904K`, consistent with helper files/logs rather than the full tensor dataset.

## Size estimate

The full dictionary tensor payload is roughly:

- `y`: `2000 x 256 x 256 x 21 x float32` = about `10.25 GiB`.
- `x`: `2000 x 256 x 256 x float32` = about `0.49 GiB`.
- Expected `.pt` file size with metadata/serialization overhead: roughly `11-12 GB`.

## Runtime estimate

Observed B200 runtime for the complete `N=2000` dictionary was about `2.06 hours`.

Inference for the current A100:

- If the current A100 run behaves close to the B200 log and most samples use `dt=0.01`, budget roughly `3-5 hours`; this is conservative because the historical evidence is from a faster B200-class GPU and a different software stack.
- If many more samples fall back to `dt=0.0005`, budget can grow substantially beyond the B200 `2:03:34` observation.
- Disk save/upload overhead for an `11-12 GB` `.pt` file should be minutes, not hours, unless the filesystem or R2 upload is slow.

## Recommendation

For a production dictionary run, use the existing dictionary script only after verifying the GPU environment with the repository GPU-only rule. For the requested dictionary, use `NS_DICT_STEP_OPTIONS=0.01,0.0005`: most stable samples keep the fast `dt=0.01`; only unstable samples jump to the safer `dt=0.0005`.


## True Batched Solver Option

Observed from source: `2D_NS_FNO2d_recurrent/data_generation/generate_ns_dictionary_batched.py` has been added for the dictionary case. Unlike the legacy script, its `--solver-batch-size` controls how many samples enter one JAX/Exponax rollout.

Observed from existing real-initial batched generation records: `solver_batch_size=8`, `solver_mode=lax-map`, and `dt=0.005` produced rollout times around `1.99 s/sample` over 1200 generated samples.

Inference: for the dictionary on the current A100, `--solver-batch-size 20 --solver-mode vmap --step-options 0.01,0.0005` should be meaningfully faster than the legacy one-sample solver loop if most samples pass at `dt=0.01`. A reasonable planning range is about `45-90 minutes` plus save/upload time. If many samples fall back to `dt=0.0005`, runtime can be longer.

Caution: `NS_DICT_BATCH_SIZE` in the legacy generator is only output batching and will not accelerate the solver. The speedup requires the new true solver batch option.
