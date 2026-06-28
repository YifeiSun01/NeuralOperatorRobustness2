# NS2D Dictionary Batch-500 Generation And GIF Visualization - 2026-05-22

Status: completed local dictionary generation and CPU GIF visualization.

## Observed Evidence

- UTC run date: 2026-05-22.
- GPU verification before launch:
  - PyTorch: `2.8.0+cu126`.
  - CUDA runtime: `12.6`.
  - PyTorch CUDA available: `True`.
  - GPU: `NVIDIA A100-SXM4-80GB`, compute capability `(8, 0)`.
  - PyTorch arch list included `sm_80`.
  - JAX backend: `gpu`.
  - JAX device: `CudaDevice(id=0)`.
- Generation command was run from `2D_NS_FNO2d_recurrent` with:
  - `--nsamples 2000`.
  - `--batch-size 500`.
  - `--solver-batch-size 500`.
  - `--solver-mode vmap`.
  - `--step-options 0.01,0.0005`.
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`.
- During the run, `nvidia-smi` observed GPU utilization at `99%` and GPU memory around `17477 MiB / 81920 MiB`.
- Source log: `2D_NS_FNO2d_recurrent/data_generation/logs_gen/generate_ns_dictionary_batched_b500_20260522_001856_UTC.out`.
- Log timing from tqdm:
  - Batch 1/4: `02:34` elapsed.
  - Batch 2/4: `05:37` elapsed.
  - Batch 3/4: `08:15` elapsed.
  - Batch 4/4: `11:07` elapsed.
  - Average displayed: `166.91s/batch`.
- The generation script initially failed only at final JSON summary writing because a numpy ndarray in metadata was not JSON serializable. The `.pt` data file had already been written before this failure.
- The script was fixed to serialize numpy arrays, numpy scalars, tensors, paths, lists, tuples, and dictionaries through `json_ready()`.
- A replacement summary was written during CPU GIF visualization.

## Output Files

- Dictionary data:
  - `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`
  - Exact size: `11534339833` bytes, about `10.74 GiB`.
- Summary JSON:
  - `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/generation_summary_dictionary_batched.json`.
- Shapes observed from summary:
  - `x_shape = [2000, 256, 256]`.
  - `y_shape = [2000, 256, 256, 21]`.
- Step usage observed from metadata:
  - `dt=0.01`: `1859` samples.
  - `dt=0.0005`: `141` fallback samples.
- GIF output directory:
  - `2D_NS_FNO2d_recurrent/visualizations/dictionary_batched_20260522/`.
- GIFs generated with CPU-only visualization and `coolwarm` colormap:
  - `dictionary_sample_0000_coolwarm_21frames.gif`.
  - `dictionary_sample_0499_coolwarm_21frames.gif`.
  - `dictionary_sample_0999_coolwarm_21frames.gif`.
  - `dictionary_sample_1499_coolwarm_21frames.gif`.
  - `dictionary_sample_1999_coolwarm_21frames.gif`.

## Inference

- `batch-size=500` and `solver-batch-size=500` completed all 2000 requested NS2D dictionary samples locally without OOM.
- Actual solver generation was not the initially hoped `3-4` minutes; observed solver loop time was about `11m07s`, plus load/save/summary/GIF overhead.
- The low GPU memory use is consistent with forward-only dictionary generation: no attack backward graph is retained.
- The `141/2000` fallback count means about `7.05%` of samples were not stable at `dt=0.01` and were rerun with `dt=0.0005`.

## Remaining Work

- This turn did not upload the 10.74 GiB dictionary to R2.
- The `.pt` dictionary and GIFs are generated artifacts and should stay out of git unless explicitly requested.
- If this dictionary will be used by attack modes with `A`, verify the attack loader points to the generated `.pt` path above.


## Stability Scan Update

Observed evidence from `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/stability_report_dictionary_batched.json`:

- A CPU-only scan was run after generation with `2D_NS_FNO2d_recurrent/data_generation/validate_ns_dictionary_stability.py`.
- All 2000 samples are finite: `finite_count=2000`, `nonfinite_count=0`.
- Robust outlier scan flagged 59 samples with multiplier `20.0`.
- Robust thresholds used by the report:
  - `max_abs_threshold = 6.987165093421936`.
  - `max_rms_threshold = 2.1462059020996094`.
- Important top outliers:
  - index `566`: `max_abs=237200850944.0`, `rms=538037440.0`.
  - index `1998`: `max_abs=193768416.0`, `rms=456879.09375`.
  - index `377`: `max_abs=9266.5751953125`, `rms=30.226783752441406`.
  - index `1114`: `max_abs=5169.54150390625`, `rms=18.982328414916992`.
- The top-outlier CSV is `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/stability_report_dictionary_batched_top_outliers.csv`.

Inference:

- The user's concern is valid: checking only NaN/Inf is insufficient because finite but physically exploded trajectories can remain in the dictionary.
- The generated dictionary should be treated as not fully clean until the 59 robust outliers are regenerated or the full dataset is regenerated with stricter stability thresholds.
- `generate_ns_dictionary_batched.py` now supports optional `--max-abs-threshold` and `--max-rms-threshold`; samples exceeding those thresholds are treated as unstable and rerun with the next `--step-options` value.


## In-Place Repair Update

Observed evidence from `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames_inplace_repair_report.json`:

- The canonical dictionary file was repaired in place:
  - `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`.
- Repair criterion: non-finite samples or `max_abs > 10`.
- Targeted unstable sample count: `56`.
- Rerun step options: `0.0005,0.0001`.
- All 56 targeted samples were accepted with `dt=0.0005`; no sample needed `dt=0.0001`.
- Repair runtime: `254.23267521499656s` total, with `134.64052360795904s` spent in the `dt=0.0005` rerun step.
- Failed repair count: `0`.

Observed evidence from post-repair scan `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/stability_report_dictionary_batched_after_inplace_repair_maxabs10.json`:

- `finite_count=2000`.
- `nonfinite_indices=[]`.
- User-threshold flagged count for `max_abs > 10`: `0`.
- Post-repair `max_abs` quantiles:
  - min `2.5911293029785156`.
  - median `2.959254026412964`.
  - 90% `3.1659043073654174`.
  - 95% `3.2432757735252378`.
  - 99% `3.4812101626396177`.
  - max `9.996569633483887`.
- Post-repair `rms` max: `1.2635504007339478`.

Inference:

- The obviously exploded finite trajectories were successfully replaced in the canonical dictionary file.
- The dictionary now passes the explicit `max_abs <= 10` stability requirement used for this repair.
- Five robust outliers remain under the older adaptive IQR rule, but they are below the explicit large-value threshold: top `max_abs` values are `9.9966`, `9.4071`, `8.8296`, `6.9561`, and `6.8228`.


## In-Place Repair Update: Max Abs 5

Observed evidence from `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames_inplace_repair_maxabs5_report.json`:

- The prior R2 upload was stopped before completing the large `.pt` transfer.
- The canonical dictionary file was repaired in place again:
  - `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`.
- Repair criterion: non-finite samples or `max_abs > 5`.
- Targeted unstable sample count: `12`.
- Targeted indices: `76, 790, 820, 845, 971, 1057, 1078, 1150, 1287, 1670, 1704, 1923`.
- Rerun step options: `0.0001` only.
- All 12 targeted samples were accepted with `dt=0.0001`.
- Repair runtime: `432.052681391011s` total, with `311.43955161608756s` spent in the `dt=0.0001` rerun step.
- Failed repair count: `0`.

Observed evidence from post-repair scan `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/stability_report_dictionary_batched_after_inplace_repair_maxabs5.json`:

- `finite_count=2000`.
- `nonfinite_indices=[]`.
- User-threshold flagged count for `max_abs > 5`: `0`.
- Robust-IQR flagged count: `0`.
- Post-repair `max_abs` quantiles:
  - min `2.5911293029785156`.
  - median `2.9590262174606323`.
  - 90% `3.1636674404144287`.
  - 95% `3.2328158378601075`.
  - 99% `3.3662182211875913`.
  - max `4.837541103363037`.
- Post-repair RMS max: `1.2635504007339478`.
- Top post-repair `max_abs` samples are indices `819` with `4.837541103363037` and `1694` with `4.811870574951172`.

Inference:

- The stricter max-abs threshold now matches the train/test dataset scale more closely: most values are around 3, and no sample exceeds 5.
- The canonical dictionary file now satisfies `max_abs <= 5` and has no non-finite samples under the CPU-only scan.
- The R2 bucket should not be treated as having the final corrected `.pt` until the max-abs-5 repaired file is uploaded again.


## In-Place Repair Update: Max Abs 4

Observed evidence from `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames_inplace_repair_maxabs4_report.json`:

- Pre-repair scan of the max-abs-5 dictionary found 4 samples with `max_abs > 4`.
- Targeted indices: `263, 819, 1357, 1694`.
- Targeted values before this repair:
  - index `263`: `max_abs=4.079109191894531`, `rms=1.1367237567901611`.
  - index `819`: `max_abs=4.837541103363037`, `rms=1.1424120664596558`.
  - index `1357`: `max_abs=4.065290451049805`, `rms=1.127825140953064`.
  - index `1694`: `max_abs=4.811870574951172`, `rms=1.161211609840393`.
- The canonical dictionary file was repaired in place again:
  - `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`.
- Repair criterion: non-finite samples or `max_abs > 4`.
- Rerun step options: `0.00005` only.
- All 4 targeted samples were accepted with `dt=0.00005`.
- Repair runtime: `386.4510918520391s` total, with `270.24317298003007s` spent in the `dt=0.00005` rerun step.
- Failed repair count: `0`.

Observed evidence from post-repair scan `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/stability_report_dictionary_batched_after_inplace_repair_maxabs4.json`:

- `finite_count=2000`.
- `nonfinite_indices=[]`.
- User-threshold flagged count for `max_abs > 4`: `0`.
- Robust-IQR flagged count: `0`.
- Post-repair `max_abs` quantiles:
  - min `2.5911293029785156`.
  - median `2.9590262174606323`.
  - 90% `3.1619613409042358`.
  - 95% `3.229175364971161`.
  - 99% `3.357116301059723`.
  - max `3.980586528778076`.
- Post-repair RMS max: `1.2635504007339478`.
- Top post-repair `max_abs` sample is index `1721` with `3.980586528778076`.

Inference:

- The stricter `max_abs <= 4` policy now passes on the canonical dictionary file.
- This scale is now close to the observed train/test dataset maxima: train max `3.8361892700195312`, test max `3.4770100116729736`.
- The final corrected dictionary still needs a fresh R2 upload if it will be used remotely.


## R2 Upload Update: Max Abs 4 Final Dictionary

Observed evidence from R2 verification on 2026-05-22 UTC:

- Final canonical dictionary uploaded to R2 path:
  - `r2://neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dictionary/dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt`.
- R2-reported size for the dictionary: `11534341969` bytes.
- Local size for the same dictionary: `11534341969` bytes.
- Upload elapsed time reported by rclone for the large `.pt`: `4m27.1s`.
- Final maxabs4 reports uploaded to the same R2 directory:
  - `dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames_inplace_repair_maxabs4_report.json`, R2 size `2807` bytes.
  - `stability_report_dictionary_batched_after_inplace_repair_maxabs4.json`, R2 size `5350` bytes.
  - `stability_report_dictionary_batched_after_inplace_repair_maxabs4_top_outliers.csv`, R2 size `4369` bytes.
- No `rclone`, repair, or stability scan process remained running after upload verification.

Inference:

- The R2 bucket now contains the final max-abs-4 repaired dictionary file and matching repair/stability reports.
- Older maxabs10 reports are still present in the R2 directory from the earlier interrupted upload, but the final intended dataset object is the canonical `.pt` listed above plus the maxabs4 reports.
