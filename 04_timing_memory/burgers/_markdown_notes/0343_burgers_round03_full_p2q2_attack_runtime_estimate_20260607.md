# Burgers Round03 Full P2Q2 Attack Runtime Estimate - 2026-06-07

## Status

This is a runtime and output-schema estimate only. No full 52-dataset x 4-model attack has been started.

## Observed GPU Evidence

- `nvidia-smi`: Tesla V100-SXM2-32GB, 0 MiB used, no running GPU processes at `2026-06-07T21:41:44Z`.
- PyTorch: `2.8.0+cu126`.
- CUDA runtime: `12.6`.
- CUDA available: `True`.
- Device capability: `(7, 0)`.
- PyTorch arch list includes `sm_70`.
- CUDA matmul sanity check returned `256.0`.

## Model Set

| Model | Checkpoint |
|---|---|
| baseline | `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt` |
| loss1 epoch8000 | `adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt` |
| loss2 epoch2000 | `adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt` |
| loss3 epoch1500 | `adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt` |

## Dataset Scope

| Split/group | Dataset count | Samples per dataset | Samples used |
|---|---:|---:|---:|
| train | 1 | 1350 available | 50 selected |
| test | 1 | 150 | 150 |
| generalization | 50 | 200 each | 10000 |
| total per model | 52 | mixed | 10200 |
| total across 4 models | 208 model-dataset jobs | mixed | 40800 model-sample attacks |

Observed sources:

- Train/test split root: `1D_Burgers/datasets/1D/Burgers/batched_exponax_splits/dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45`.
- Generalization root: `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`, 50 `.pt` files.
- Data shape: train/test/generalization files use tensors `x` and `y` with spatial length `1024`.

## Existing Attack Logic

The current p=2, q=2 code in `tools/plot_burgers_p2q2_baseline_vs_epoch1000_attack_gif.py` uses:

- `EPSILON_RMS = 0.12`
- `ALPHA_RMS = 0.012`
- `ATTACK_STEPS = 100`
- differentiable Burgers solver target via `burgers_solver_target`
- per-step update: normalized RMS-L2 gradient step followed by RMS-L2 projection

The existing visualization helper records trajectories every 2 steps. The proposed full run should not do that. It should record only:

- initial loss at delta=0
- final loss after the chosen number of attack steps
- final perturbation `delta`
- per-dataset mean FFT spectrum of final `delta`

## Benchmark Results

Benchmarks used the same differentiable solver/model attack path and included final FFT computation in the final record timing. They used the baseline model and representative generalization samples.

| Batch size | Init record sec | Step times sec | Mean step sec | Final record sec | Peak allocated GiB |
|---:|---:|---|---:|---:|---:|
| 50 | 0.7601 | 3.5749, 3.0535, 3.0184 | 3.2156 | 1.1426 | 2.8239 |
| 100 | 1.1087 | 3.8998, 2.8770, 3.0859 | 3.2876 | 1.1086 | 5.6343 |
| 200 | 1.1341 | 4.9727, 3.0898, 2.9657 | 3.6761 | 1.1179 | 11.2144 |
| 400 | 0.8652 | 2.9432, 2.4381, 2.4303 | 2.6039 | 0.8345 | 22.4426 |
| 500 | 1.1171 | 3.2007, 2.2750, 2.2734 | 2.5830 | 0.7567 | 28.5935 |

Observed inference from benchmark:

- Batch 500 is stable on the V100 and uses about 87% of allocated GPU memory capacity, leaving limited but workable headroom.
- Batch 400 is safer at about 68% allocated memory and still much faster than per-dataset batch 200.
- Batch 200 maps naturally to each 200-sample generalization dataset but underuses the GPU and is much slower overall if every dataset is attacked separately.

## Runtime Estimate

The estimate assumes all samples are concatenated with a manifest and attacked in large batches, then split back by dataset for recording. This preserves per-dataset outputs and FFT means while avoiding one GPU launch group per dataset.

| Attack steps | Batch 500 grouped estimate | Batch 400 grouped estimate | Strict per-dataset batch estimate |
|---:|---:|---:|---:|
| 10 | 0.61 h | 0.78 h | 1.99 h |
| 20 | 1.14 h | 1.49 h | 3.74 h |
| 50 | 2.76 h | 3.62 h | 8.99 h |
| 100 | 5.46 h | 7.16 h | 17.74 h |

Practical wall-clock recommendation:

- Use batch 500 if the machine stays otherwise idle: expect about `5.5 h` ideal for 100 steps, more realistically `6-7 h` with I/O, checkpoint loading, metadata writing, and safety overhead.
- Use batch 400 if you want more memory headroom: expect about `7.2 h` ideal, more realistically `8-9 h`.
- Avoid strict per-dataset batch 200 for the full 100-step run unless isolation is more important than time; it is likely around `18 h` ideal and can exceed that with overhead.

## Proposed Output Schema

One final run can save a compact result root such as:

`forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_YYYYMMDD/`

Recommended files:

- `manifest.json`: one row per sample, with `global_sample_id`, `split`, `dataset_id`, `source_path`, `source_index`, and train/test/generalization grouping.
- `config.json`: models, checkpoints, attack settings, selected train indices, batch size, timing, GPU preflight.
- `summary.csv`: one row per model/dataset with mean initial loss, mean final loss, final/initial ratio, mean delta RMS, and FFT band ratios.
- `fft_power_mean_by_dataset.npz`: arrays keyed by model, shape `[52, 513]` for rFFT power modes `0..512` averaged over samples inside each dataset.
- `losses_by_sample.npz`: per-model arrays `initial_loss` and `final_loss`, shape `[10200]` each.
- `delta_by_model/*.npz`: per-model final delta arrays, shape `[10200, 1024]`, or split by dataset if easier to inspect.

Approximate storage if final delta is float32:

- Delta arrays: `40800 * 1024 * 4 bytes = 167.1 MB` decimal, about `159.4 MiB`.
- Initial/final loss arrays: less than `1 MB`.
- FFT dataset means: `4 * 52 * 513 * 4 bytes`, about `0.43 MB`.
- Metadata and CSVs are negligible.

## FFT Plan

For each final perturbation `delta`:

1. Compute `rfft(delta, dim=-1)` over the 1024 spatial grid.
2. Compute power spectrum, e.g. `abs(rfft(delta))^2 / 1024`.
3. For each model and dataset, average power over all samples in that dataset.
4. Save `fft_power_mean_by_dataset[model, dataset, mode]` for modes `0..512`.
5. Also save derived low/high frequency summaries, for example:
   - low band power fraction: modes `1..32`
   - mid band power fraction: modes `33..128`
   - high band power fraction: modes `129..512`
   - spectral centroid

This directly addresses the hypothesis that self-training makes the adversarial `delta` more high-frequency. The mean FFT spectrum should be computed per dataset, while final delta remains stored per sample.

## Recommendation

Run a smaller pilot first before committing to the 100-step full run:

- Recommended pilot: all 52 datasets x 4 models, batch 500, `20` attack steps. Estimated ideal runtime: about `1.15 h`; practical runtime likely `1.3-1.6 h`.
- If the FFT already clearly shows frequency shift at 20 steps, decide whether 100 steps is necessary.
- If the final-loss growth at 20 steps is too small, use 50 steps as the next pilot; estimated ideal runtime is about `2.76 h`.

The full 100-step batch-500 run is feasible on this V100, but it is not a quick run; plan around `6-7 h` if the machine remains idle.

## Resume From 20-Step Pilot

Observed implementation update on 2026-06-07: `tools/run_burgers_round03_full_p2q2_finalonly_attack.py` now supports continuing an attack from a previous final-only run via:

- `--resume-from-root`
- `--resume-start-steps`
- `--target-total-steps`

Prepared launchers:

- `tools/run_burgers_round03_full_p2q2_resume_20to40_20260607.sh`
- `tools/run_burgers_round03_full_p2q2_resume_20to60_20260607.sh`

Observed safeguards: both launchers check that the 20-step source root exists, that `summary.json` exists, and that all four model directories contain `final_delta_by_sample.npz` plus `losses_and_delta_rms_by_sample.npz`. If any required 20-step result is missing, the launcher exits instead of silently restarting or producing a mixed run.

Resume behavior:

- The saved 20-step `final_delta_by_sample.npz` is loaded as the initial perturbation for the resumed run.
- A 20-to-40 run performs `20` additional attack steps.
- A 20-to-60 run performs `40` additional attack steps.
- The original step0 loss is preserved.
- The 20-step resume-start loss, resume-start model-solver RMS, and resume-start delta RMS are recorded.
- Final loss, final model-solver RMS, final delta, per-dataset mean FFT power, low/mid/high band fractions, and spectral centroid are recomputed for the 40-step or 60-step target.

Inference: because the current 20-step pilot uses roughly `28.6 GiB` peak allocated and about `30.1 GiB` total GPU memory, the resume launchers should not be started while the 20-step job is still active. They are intended to run after the pilot completes and all four saved delta arrays are present.
