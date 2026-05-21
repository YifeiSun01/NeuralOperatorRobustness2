# 2D NS Recurrent FNO2d Training Monitor - 2026-05-21

Status: inspected active training run after the first three completed epochs; training was still running.

## Observed Evidence

- Active process: PID `23159`, command `adv_robust/bin/python -u 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py ...`.
- Output directory: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_071254_UTC`.
- Train log source: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_071254_UTC/train_log.csv`.
- Progress source: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_071254_UTC/progress_latest.json` and `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_071254_UTC/progress.jsonl`.
- Dataset paths:
  - Train: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt`.
  - Test: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt`.
- GPU snapshot at `2026-05-21 07:36:38 UTC`: `NVIDIA A100-SXM4-80GB`, GPU utilization `68%`, memory utilization `38%`, memory used `23113 MiB / 81920 MiB`, power draw `189.15 W`.
- Latest progress after the first three epochs showed epoch `4/500`, batch `10/288`, run elapsed `1423.15s`, and train-relative-L2-so-far `0.242269`.

## First Three Epochs

| epoch | seconds | elapsed total | ETA reported | estimated total | train rel L2 | test rel L2 | train score | test score |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 475.279 | 7m55s | 65h52m44s | 66h00m39s | 0.41788557 | 0.28925265 | 0.58211443 | 0.71074735 |
| 2 | 444.870 | 15m25s | 63h38m37s | 63h54m02s | 0.27541957 | 0.26146268 | 0.72458043 | 0.73853732 |
| 3 | 476.025 | 23m26s | 64h15m00s | 64h38m25s | 0.25393027 | 0.27035083 | 0.74606973 | 0.72964917 |

## Inference

- The run is active and has progressed into epoch 4.
- The first-three-epoch mean runtime is about `465.39s` per epoch, or about `7m45s` per epoch.
- Extrapolating 500 epochs from the first three epochs gives roughly `64.6h` total wall time, about `2.7 days`, assuming runtime stays similar.
- The estimate includes evaluation every epoch and R2 record sync overhead because those were enabled during the observed run.

## Remaining Work

- Continue monitoring `train_log.csv` because the early epoch timing may shift once caches, allocator behavior, and R2 sync overhead stabilize.
- Verify final model upload under the configured R2 prefix after epoch 500 completes.


## Runtime Estimate Update - 2026-05-21 07:38 UTC

Observed evidence:
- Active process PID `23159` was still running in `adv_robust/bin/python`.
- Completed epochs available in `train_log.csv`: epochs 1-3 only.
- Epoch times: `475.2787s`, `444.8704s`, `476.0252s`; mean `465.39s` per epoch.
- Latest `progress_latest.json` showed epoch `4/500`, batch `70/288`, run elapsed `1520.47s`, and `total_train_batch_eta` `64h41m39s`.
- GPU snapshot at `2026-05-21 07:38:35 UTC`: A100 memory used `23113 MiB / 81920 MiB`; instantaneous utilization sample `2%`, likely a sampling point during synchronization/eval/logging because the progress file continued updating.

Inference:
- Current 500-epoch runtime estimate remains about `64.6h` total, roughly `2.7 days`, with the current settings (`eval_every=1`, `r2_sync_every_epochs=1`, batch size 4).
- The estimate is based on early epochs and should be refreshed after 10-20 epochs.


## A100 Runtime Bottleneck Check - 2026-05-21 07:43 UTC

Observed evidence:
- Active PID `23159` remained in `adv_robust/bin/python`.
- Latest inspected progress from `progress.jsonl` showed epoch `4/500`, batch `240/288`, run elapsed `1798.87s`, and total train-batch ETA `64h40m37s`.
- `nvidia-smi dmon -s pucm -c 8` sampled bursts at `100%` SM utilization, with intervening `0%` samples during synchronization/logging points; framebuffer memory stayed around `23113 MB / 81920 MB`.
- Active command settings include `--batch-size 4`, `--eval-batch-size 4`, `--eval-every 1`, and `--r2-sync-every-epochs 1`.
- Current wrapper defaults imply `ceil(1150/4)=288` train batches per epoch.
- `2D_NS_FNO2d_recurrent/models/FNO2d.py` autoregressively calls the FNO model 10 times per batch for `T_out=10`, `step=1`.
- The older local script `2D_NS_FNO2d_recurrent/training_models/trainFNO2d_unnormalized.py` used `batch_size = 20` and `ntrain = 1000`, which is roughly 50 train batches per epoch.

Inference:
- No data generation, solver loop, or CPU fallback was observed in the active training loop.
- The active run is GPU-backed, but the A100 is not memory-saturated at about 23 GB used out of 80 GB.
- The most important runtime explanation is the current small batch size combined with 10-step recurrent unrolling: about 288 batches per epoch times 10 FNO calls per batch.
- Evaluation and R2 record sync every epoch add overhead, but the training batch count is the dominant visible issue.

Next action:
- Benchmark a larger batch size before committing to another 500-epoch run, preferably `BATCH_SIZE=8`, then `12` or `16` if memory remains safe.
- For a long official run, consider less frequent full evaluation and R2 record sync, such as every 5 epochs, if per-epoch monitoring is not required.


## Fourth Epoch Runtime Refresh - 2026-05-21 07:48 UTC

Observed evidence:
- Epoch 4 completed in `487.730s` (`8m08s`) according to `train_log.csv`.
- Mean epoch time after four epochs is `473.832s`; estimated total is `65h24m59s`.
- Latest progress after epoch 4 showed epoch `5/500`, batch `110/288`, and total train-batch ETA `65h23m55s`.
- Current GPU snapshot showed `100%` GPU utilization, `23113 MiB / 81920 MiB` memory used, and `189.23 W / 400 W` power draw.

Inference:
- Current settings still extrapolate to about `65h` total.
- A100 execution is active, but memory headroom is large; larger batch-size benchmarking remains the main optimization candidate.


## GPU Path And Utilization Verification - 2026-05-21 07:50 UTC

Observed evidence:
- The run GPU verification file records `cuda:0`, `NVIDIA A100-SXM4-80GB`, `sm_80`, PyTorch `2.8.0+cu126`, CUDA `12.6`, and a passing CUDA sanity matmul.
- The saved config records `amp: false`, `batch_size: 4`, `num_workers: 0`, `ntrain: 1150`, `t_out: 10`, and `step: 1`.
- `nvidia-smi dmon -s pucmt -c 15` showed bursty GPU use: samples included `100%`, `97%`, and `73%` SM utilization, but many samples were `0%`; memory stayed around `23113 MB / 81920 MB`.
- Latest progress showed epoch `5/500`, batch `190/288`, and total train-batch ETA `65h17m43s`.

Inference:
- The active run is using the A100 GPU, but the workload is not continuously saturating it.
- The current small batch size and frequent synchronization/evaluation/logging make this a poor A100 throughput configuration.


## A100 Fast Batch Benchmark And Default Update - 2026-05-21 08:00 UTC

Observed evidence:
- The slow `batch_size=4` run was stopped after reaching epoch `5/500`, batch `260/288`, with total train-batch ETA around `65h13m12s`.
- FP32 larger-batch benchmark on the real train dataset and A100:
  - `batch_size=10`: peak `46.71 GiB`, train-only epoch estimate `106.2s`.
  - `batch_size=12`: peak `55.52 GiB`, train-only epoch estimate `103.1s`.
  - `batch_size=14`: peak `64.33 GiB`, train-only epoch estimate `89.4s`.
  - `batch_size=16`: peak `72.19 GiB`, train-only epoch estimate `83.9s`.
  - `batch_size=20`: CUDA OOM.
- AMP failed because the spectral convolution uses complex FFT tensors and the CUDA `ComplexHalf` einsum path is not implemented in the installed PyTorch.
- Defaults changed to `BATCH_SIZE=16`, `EVAL_BATCH_SIZE=16`, `EVAL_EVERY=5`, and `R2_SYNC_EVERY_EPOCHS=0`; final R2 upload remains enabled when `R2_UPLOAD=1`.

Inference:
- `batch_size=16` is the fastest tested setting that fits the A100 in FP32.
- Under 10 hours is not yet evidenced for 500 epochs with this exact model; the current best tested setting suggests roughly 12+ hours before full-run overhead.


## Batch Size 18 OOM Test - 2026-05-21 08:10 UTC

Observed evidence:
- Before testing, no active 2D FNO training process was found and A100 memory was `0 MiB / 81920 MiB`.
- Plain FP32 `batch_size=18` failed with CUDA OOM, with the process using about `79.12 GiB`.
- A fresh retry with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` also failed with CUDA OOM, with the process using about `78.31 GiB`.

Inference:
- `batch_size=18` is not safe for this exact recurrent FNO2d setup.
- `batch_size=16` remains the largest tested safe batch size; `batch_size=14` is the safer fallback.


## Batch Size 17 Test - 2026-05-21 08:12 UTC

Observed evidence:
- Plain FP32 `batch_size=17` completed one batch, then OOMed with process memory around `78.28 GiB`.
- With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `batch_size=17` completed four timed batches, peak memory `76.53 GiB`, and reported `68` batches per epoch with a short-run train-only estimate of `93.3s`.

Inference:
- `batch_size=17` is possible only with expandable CUDA segments in the short test, but it is close to the A100 memory limit.
- `batch_size=16` remains safer for official training.


## AMP Compatibility Retest - 2026-05-21 08:18 UTC

Observed evidence:
- FP32/no autocast forward-backward completed successfully on a real recurrent FNO2d batch.
- FP16/default AMP autocast failed at the spectral convolution `einsum` with `baddbmm_cuda not implemented for ComplexHalf`.
- BF16 autocast failed at `torch.fft.rfft2(x)` with `Unsupported dtype BFloat16`.

Inference:
- Full-model AMP is not compatible with the current FNO2d FFT/spectral-convolution implementation.
- `AMP=0` remains required for official runs unless selective mixed precision is implemented and tested.


## Batch Size 16 Full-Epoch Stress Test - 2026-05-21 08:25 UTC

Observed evidence:
- `batch_size=16` completed all `72/72` batches for one full epoch with no CUDA OOM.
- Peak memory rose to `72.19 GiB` and stayed there through the end of the epoch.
- Mean measured GPU compute section was `1.170s` per batch, but full script wall time was `401.1s` (`6.69min`).

Inference:
- `batch_size=16` is much more stable than `batch_size=17` for memory.
- Further speedup likely requires reducing host-side data loading/collation/transfer overhead, not only changing batch size.


## Data Loading Bottleneck Analysis - 2026-05-21 08:30 UTC

Observed evidence:
- The trainer loads data on CPU, slices each sample in `NSTrajectoryDataset.__getitem__()`, calls `.contiguous()` for x and y per sample, collates batches through DataLoader, and then transfers each batch to GPU.
- The DataLoader is recreated inside every epoch.
- Batch-size-16 stress testing measured about `1.170s` for the synchronized GPU compute section per batch, but `401.1s` wall time for one 72-batch epoch.

Inference:
- The overhead is likely dominated by host-side slicing/copying/collation/transfer waits rather than model compute alone.
- Precomputed contiguous tensors, persistent loading, and optional GPU-resident batches are the main next optimization candidates.


## GPU-Resident Data Path Benchmark And Trainer Integration - 2026-05-21 08:42 UTC

Observed evidence:
- Standalone GPU-resident benchmark with train `x/y` on GPU:
  - `batch_size=16`: `72/72` batches, `87.1s` (`1.45min`), peak `77.80 GiB`.
  - `batch_size=10`: `115/115` batches, `99.2s` (`1.65min`), peak `52.32 GiB`.
- Keeping both train and test tensors resident on GPU OOMed in the official trainer, so the implementation was adjusted to keep only train tensors on GPU and evaluate test data from CPU batches.
- Official one-epoch wrapper validation completed with `DATA_RESIDENCY=gpu`, `BATCH_SIZE=16`, `EVAL_BATCH_SIZE=16`, and `R2_UPLOAD=0`.
- Official epoch 1 took `100.266s` (`1m40s`) including epoch evaluation; full command total was `2m31s` including final evaluation and checkpoint save.

Inference:
- GPU-resident train data removes most of the host-side DataLoader bottleneck and brings the epoch time down from `6.69min` to about `1.7min` in the official trainer.
- `batch_size=16` is fast but close to the memory ceiling; `batch_size=10` is much safer and still fast.


## 10-12 Hour Run Configuration - 2026-05-21 08:55 UTC

Observed evidence:
- Optimized `BATCH_SIZE=16` official one-epoch timing was `100.266s` including epoch evaluation; standalone train-only GPU-resident timing was `87.1s`.
- Optimized `BATCH_SIZE=4` official one-epoch timing was `152.087s` including epoch evaluation; full one-epoch command including final eval/save was `3m22s`.
- Defaults were changed to `EPOCHS=450` while keeping `BATCH_SIZE=16`, `EVAL_BATCH_SIZE=16`, `EVAL_EVERY=10`, `DATA_RESIDENCY=gpu`, and `R2_SYNC_EVERY_EPOCHS=0`.
- Best checkpoint logic already saves `checkpoints/best.pt` on new best evaluated test loss, and final R2 upload copies the whole output directory.

Inference:
- `450` epochs is the best current fit for a 10-12 hour optimized run; `1000` epochs would likely run much longer.
- Optimized `BATCH_SIZE=4` is estimated around `17-19h` for 450 epochs, so it is not the preferred setting for the 10-12 hour target.
