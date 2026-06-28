# NS2D Recurrent Training Dataset Generation Time And Stability - 2026-05-22

Status: inspected the local train/test datasets used by the recurrent FNO2d NS training run.

## Source Files

Observed local dataset files:

- Train: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt`.
  - Size: `6632250121` bytes.
  - Modified: `2026-05-21 05:26:21 UTC`.
  - Shape: `x=[1150,256,256]`, `y=[1150,256,256,21]`.
- Test: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt`.
  - Size: `288362621` bytes.
  - Modified: `2026-05-21 04:48:06 UTC`.
  - Shape: `x=[50,256,256]`, `y=[50,256,256,21]`.
- Generation timing source: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/generation_summary.json`.
- Stability scan report: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/training_dataset_stability_report_20260522.json`.
- Per-sample stability CSV: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/training_dataset_stability_per_sample_20260522.csv`.

## Generation Time

Observed from `generation_summary.json` and dataset metadata:

- Dataset generation settings:
  - `fixed_step=0.005`.
  - `processing_batch_size=32`.
  - `solver_batch_size=8`.
  - `solver_mode=lax-map`.
  - `tfinal=20` with 21 saved frames.
- Test split:
  - `50` samples in `2` batches.
  - Recorded rollout time: `104.30499046598561s`.
  - Recorded upsample time: `0.20982681098394096s`.
  - Recorded rollout+upsample time: `104.51481727696955s`, about `1m44.5s`.
  - Rollout time per sample: `2.086099809319712s`.
- Train split:
  - `1150` samples in `36` batches.
  - Recorded rollout time: `2278.595210202737s`.
  - Recorded upsample time: `0.31126681552268565s`.
  - Recorded rollout+upsample time: `2278.9064770182595s`, about `37m58.9s`.
  - Rollout time per sample: `1.9813871393067277s`.
- Combined train+test recorded rollout+upsample time: `2383.421294295229s`, about `39m43.4s`.

Inference:

- These are the recorded per-batch generation timings from the current local dataset metadata. They may not include any small unrecorded process startup or final filesystem overhead outside the stored timing records.
- A separate older `.out` log under `data_generation/logs_gen/VT_NS_gen_all_frame_6765843.out` is from a 2025 B200 job that was canceled and does not fully describe the current local train/test `.pt` files.

## Stability Scan

Observed from the CPU-only scan of the actual local train/test `.pt` files:

- Train split:
  - `finite_count=1150`; no non-finite samples observed.
  - Counts with `max_abs > 10`: `0`.
  - Counts with `max_abs > 50`: `0`.
  - Counts with `max_abs > 100`: `0`.
  - Counts with `max_abs > 1000`: `0`.
  - `max_abs` quantiles: min `2.6480345726013184`, median `3.1202988624572754`, 90% `3.370944929122925`, 95% `3.4458749890327454`, 99% `3.602002828121185`, max `3.8361892700195312`.
  - RMS max: `1.2642641067504883`.
  - Top max-abs sample: index `1111`, `max_abs=3.8361892700195312`, `rms=1.1656326055526733`.
- Test split:
  - `finite_count=50`; no non-finite samples observed.
  - Counts with `max_abs > 10`: `0`.
  - Counts with `max_abs > 50`: `0`.
  - Counts with `max_abs > 100`: `0`.
  - Counts with `max_abs > 1000`: `0`.
  - `max_abs` quantiles: min `2.8046298027038574`, median `3.117771029472351`, 90% `3.3514097213745115`, 95% `3.408313274383545`, 99% `3.453973386287689`, max `3.4770100116729736`.
  - RMS max: `1.258671522140503`.
  - Top max-abs sample: index `31`, `max_abs=3.4770100116729736`, `rms=1.0448931455612183`.

Inference:

- The train/test datasets used for the recurrent FNO2d model do not show the explosion pattern seen earlier in the dictionary dataset.
- There are no observed values in the tens, hundreds, or thousands. The maximum absolute value in train is about `3.84`; in test it is about `3.48`.
- On this evidence, the model training dataset looks numerically stable under the explicit large-value check.
