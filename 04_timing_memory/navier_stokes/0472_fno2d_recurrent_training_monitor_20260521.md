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

## Active 500-Epoch Run First-Three-Epoch Timing - 2026-05-21 09:08 UTC

Observed evidence:
- Active training process observed as PID `58013`, running `adv_robust/bin/python -u 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`.
- Output directory: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC`.
- Configured settings observed from the process command and output files: `EPOCHS=500`, `BATCH_SIZE=16`, `EVAL_BATCH_SIZE=16`, `EVAL_EVERY=10`, `DATA_RESIDENCY=gpu`, `AMP=0`, `R2_UPLOAD=1`, `R2_SYNC_EVERY_EPOCHS=0`.
- `nvidia-smi` at the monitoring point reported `NVIDIA A100-SXM4-80GB`, about `81121 MiB / 81920 MiB` used, `100%` GPU utilization, and `67-75%` memory-controller utilization.
- `gpu_verification.txt` for the run records PyTorch `2.8.0+cu126`, CUDA `12.6`, GPU `NVIDIA A100-SXM4-80GB`, compute capability `sm_80`, and a passing CUDA sanity matmul.
- Observed from `train_log.csv`:
  - Epoch 1: `93.884s` (`1m34s`), with train relative L2 mean `0.610978`, test relative L2 mean `0.430839`, and test MSE `0.365911`.
  - Epoch 2: `83.839s` (`1m24s`), with train relative L2 mean `0.347183`; no test evaluation on this epoch.
  - Epoch 3: `83.837s` (`1m24s`), with train relative L2 mean `0.279608`; no test evaluation on this epoch.
- At the same monitoring point, `progress_latest.json` showed the run already in epoch `4/500`, batch `30/72`.

Inference:
- The first three logged epochs give a mean of about `87.19s/epoch`; the trainer's own current estimate from `train_log.csv` is total runtime `12h06m36s` with about `12h02m12s` remaining after epoch 3.
- Using epochs 2-3 as the train-only baseline gives about `83.84s` per non-evaluation epoch. Epoch 1 was about `10.05s` slower because it included test evaluation. With evaluation at epoch 1, every 10 epochs, and the final epoch, a practical training-only plus periodic-evaluation estimate is about `11h45m-12h10m`, before final R2 upload variability.
- The active run is using the intended optimized high-memory A100 path. It is not the earlier slow `batch_size=4`/CPU-collation path.

Remaining work:
- Continue monitoring around epoch 10 because that is the next scheduled test evaluation and will show the steady-state evaluation overhead.
- Watch for OOM because the run intentionally uses most of the A100 80GB memory.
- Confirm final R2 upload and the presence of `checkpoints/best.pt` after the run completes.

## Active Run Status Update - 2026-05-21 09:12 UTC

Observed evidence:
- `train_log.csv` reached epoch 6 while the run was still active.
- Epoch 4: `83.826s`, train relative L2 mean `0.256466`.
- Epoch 5: `83.882s`, train relative L2 mean `0.242491`.
- Epoch 6: `83.918s`, train relative L2 mean `0.232962`.
- The trainer estimate after epoch 6 was total runtime `11h39m03s`, with `11h30m27s` remaining.
- `progress_latest.json` showed epoch `7/500`, batch `40/72`.
- `nvidia-smi` still showed `NVIDIA A100-SXM4-80GB`, about `81121 MiB / 81920 MiB` used, and `100%` GPU utilization.

Inference:
- The non-evaluation epochs are stable around `83.8-83.9s`.
- Current best estimate is roughly `11h40m-12h` before final R2 upload variability, with epoch 10 still worth checking because it is the next scheduled evaluation epoch.

## Active Run Epochs 7-10 Timing Update - 2026-05-21 09:18 UTC

Observed evidence:
- `train_log.csv` reached epoch 10 in output directory `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC`.
- Epoch 7: `83.929s`, train relative L2 mean `0.225809`; no test evaluation on this epoch.
- Epoch 8: `83.859s`, train relative L2 mean `0.218148`; no test evaluation on this epoch.
- Epoch 9: `83.889s`, train relative L2 mean `0.211692`; no test evaluation on this epoch.
- Epoch 10: `99.226s`, train relative L2 mean `0.206796`, test relative L2 mean `0.225781`, test MSE `0.102908`.
- After epoch 10, the trainer reported estimated total runtime `12h04m39s` and ETA `11h50m12s`.
- `progress_latest.json` showed the run in epoch `11/500`, batch `30/72`.
- `nvidia-smi` still showed A100 utilization at `100%` with about `81121 MiB / 81920 MiB` allocated.

Inference:
- Non-evaluation epochs remain stable at about `83.9s` (`1m24s`).
- The scheduled epoch-10 evaluation added roughly `15s` over a normal training epoch.
- Current practical estimate remains about `12h` total before final R2 upload variability.

## Active Run Epoch 11 Quick Check - 2026-05-21 09:19 UTC

Observed evidence:
- Epoch 11 completed in `83.894s`, train relative L2 mean `0.203593`.
- The run advanced to epoch `12/500`, batch `1/72`.

Inference:
- The run returned to normal non-evaluation timing after the longer scheduled epoch-10 evaluation.

## Active Run GPU Memory Headroom Check - 2026-05-21 09:20 UTC

Observed evidence:
- `nvidia-smi` reported `81121 MiB` used, `32 MiB` free, `81920 MiB` total on `NVIDIA A100-SXM4-80GB`, with `100%` GPU utilization.
- Per-process GPU memory query showed the active Python compute process using `81112 MiB`.
- `progress_latest.json` showed epoch `12/500` complete, elapsed `17m18s`, estimated total `12h04m38s`, and ETA `11h47m21s`.
- `train_log.csv` showed epoch 12 completed in `83.978s`.
- Code inspection found scalar logging via `float(...detach().cpu())`, `@torch.no_grad()` evaluation, and no obvious accumulation of GPU tensors or retained computation graphs in the epoch loop.

Inference:
- The current memory report is about `79.22 GiB / 80 GiB`; the configuration is very aggressive.
- The stable high value is consistent with PyTorch's CUDA caching allocator, not currently evidence of a growing memory leak.
- Because epoch 10 evaluation already completed and later epochs use the same shapes, gradual leak-driven OOM is not evidenced, but the tiny headroom leaves nonzero OOM risk.

## Active Run GPU Memory Trend Check - 2026-05-21 09:24 UTC

Observed evidence:
- Four consecutive `nvidia-smi` samples at about 10-second intervals reported `81121 MiB` used, `32 MiB` free, `81920 MiB` total, and `100%` GPU utilization.
- `progress_latest.json` showed epoch `15/500`, batch `50/72`.
- `train_log.csv` had reached epoch 14; epochs 11-14 remained normal non-evaluation epochs around `83.9-84.0s`.

Inference:
- No upward GPU-memory trend was observed.
- The stable number is consistent with PyTorch holding its peak cached allocation, not with an evidenced per-epoch memory leak.

## Epoch-25 Checkpoint Save Overhead Measurement - 2026-05-21 09:41 UTC

Observed evidence:
- `checkpoints/latest.pt` appeared at epoch 25 with size about `2.7G`; `checkpoints/best.pt` also exists at about `2.7G`.
- Checkpoint directory total size became about `5.3G`.
- Epoch 25 logged `seconds=84.0115s`, `elapsed_total_seconds=2147.7380s`; this log row is written before checkpoint saving.
- Epoch 26 logged `seconds=84.0250s`, `elapsed_total_seconds=2234.5126s`.
- Inferred checkpoint/save gap: about `2.75s`.

Inference:
- The observed latest-checkpoint write took only a few seconds on this machine, but it is still avoidable repeated multi-GB disk I/O.
- Future runs should use `SAVE_EVERY=0`; that default has already been changed.

## Active 500-Epoch Runtime Projection After 28 Epochs - 2026-05-21 09:43 UTC

Observed evidence:
- Completed epoch 28 with `elapsed_total_seconds=2402.501s` (`40m03s`).
- Recent stable non-evaluation epoch mean: about `83.982s`.
- Evaluation overhead: about `15.468s` over a normal epoch.
- Epoch-25 checkpoint overhead: about `2.75s`.

Inference:
- Remaining training time after epoch 28 is about `11h14m-11h16m`.
- Projected total training time is about `11h54m-11h56m`.
- Projected training completion from `2026-05-21 09:43:11 UTC` is about `2026-05-21 20:57-20:59 UTC`, before final R2 upload variability.

## Active Run Health Check At Epoch 86 - 2026-05-21 11:05 UTC

Observed evidence:
- Active process PID `58013` had `ps` elapsed time `02:03:43`, state `Rl+`.
- `progress_latest.json` showed epoch `86/500`, batch `40/72`, completed train batches `6160/36000`, run elapsed `7329.845s`, and train-batch ETA `9h51m47s`.
- Latest completed epoch in `train_log.csv` was epoch 85 with elapsed total `2h01m23s`; recent non-evaluation epochs remained around `83.9-84.0s`.
- Epoch 80 scheduled evaluation completed with test relative L2 mean `0.146012` and test MSE `0.0452241`.
- `nvidia-smi` reported `81121 MiB / 81920 MiB` used and `100%` GPU utilization.

Inference:
- No evidence of failure, OOM, stall, or runaway memory growth was observed.
- Approximate finish remains around `2026-05-21 20:57 UTC`, before final R2 upload variability.

## Active Run Progress Check At Epoch 449 - 2026-05-21 19:43 UTC

Observed evidence:
- Active process PID `58013` had `ps` elapsed time `10:41:51`.
- `progress_latest.json` showed epoch `449/500`, batch `30/72`, completed train batches `32286/36000`, run elapsed `38412.897s`, and train-batch ETA `1h13m39s`.
- Current total train-batch completion was `89.6833%`.
- Latest completed epoch in `train_log.csv` was epoch 448 with elapsed total `10h39m38s`; recent non-evaluation epochs remained around `83.94-83.96s`.
- `nvidia-smi` reported `81121 MiB / 81920 MiB` used and `100%` GPU utilization.

Inference:
- No evidence of failure, OOM, stall, or runaway memory growth was observed.
- The run remains on track to finish training around `2026-05-21 20:57 UTC`, before final checkpoint/R2 upload variability.

## Active Run Latest Train/Test Loss Check - 2026-05-21 20:04 UTC

Observed evidence:
- `progress_latest.json` showed epoch `463/500`, batch `50/72`, completed train batches `33314/36000`, run elapsed `39650.645s`, and train-batch ETA `53m17s`.
- Latest completed epoch in `train_log.csv` was epoch 462.
- Epoch 462 train relative L2 mean was `0.05333483872206315`; train MSE was `0.0057479435699465484`.
- Latest completed test evaluation was epoch 460.
- Epoch 460 test relative L2 mean was `0.08675776571035385`; test MSE was `0.0179227876663208`.
- Current in-progress batch loss was `0.9232521057128906` at epoch 463 batch 50.

Inference:
- Test metrics update only every 10 epochs in this active old-argument run; NaN test fields on non-evaluation epochs are expected.

## Active Run Loss Trend Check - 2026-05-21 20:06 UTC

Observed evidence:
- Latest completed epoch was epoch 464; active progress was epoch `465/500`, batch `20/72`.
- Train relative L2 mean decreased from `0.610978` at epoch 1 to `0.053272` at epoch 464, about `91.28%` lower.
- Train MSE decreased from `0.791622` at epoch 1 to `0.005734` at epoch 464.
- Test relative L2 mean decreased from `0.430839` at epoch 1 to `0.086758` at epoch 460, about `79.86%` lower.
- Test MSE decreased from `0.365911` at epoch 1 to `0.017923` at epoch 460.
- Recent test relative L2 means from epoch 410 to 460 were `0.087922`, `0.087647`, `0.087481`, `0.087220`, `0.086976`, `0.086758`.

Inference:
- The train and test curves are clearly decreasing.
- Late-stage improvement is slower and close to plateau, but the latest evaluated test loss remains the best observed test relative L2 so far.

## Late-run status at 2026-05-21T20:53:07Z

Observed from `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/progress.jsonl`:

- Active process: PID `58013`, `train_fno2d_recurrent_cli.py`, `EPOCHS=500`, `BATCH_SIZE=16`, `DATA_RESIDENCY=gpu`, `EVAL_EVERY=10`, `SAVE_EVERY=25`, `R2_UPLOAD=1`.
- Latest progress record: epoch `498/500`, batch `10/72`, completed train batches `35794/36000`, timestamp `2026-05-21T20:53:07+00:00`.
- Latest progress ETA: `4m05s` for remaining train batches. Inference: local training should finish in about `4-5` minutes from that timestamp, with final evaluation/save/R2 upload possibly adding extra time.
- Last completed epoch: epoch `497`, timestamp `2026-05-21T20:52:55+00:00`, train relative L2 mean `0.05292977390081986`, train MSE `0.005659206264206897`.
- Latest test evaluation: epoch `490`, timestamp `2026-05-21T20:43:05+00:00`, test relative L2 mean `0.08654208987951278`, test MSE `0.0178410891443491`, test score `0.9134579101204872`.
- Recent test evaluations show a continued slow decrease: epoch 460 `0.08675776571035385`, epoch 470 `0.08668057322502136`, epoch 480 `0.08656245350837707`, epoch 490 `0.08654208987951278`.
- Next test evaluation should occur at epoch `500` because the active process was started with `--eval-every 10`.
