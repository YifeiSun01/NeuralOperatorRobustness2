# 2D NS Recurrent FNO2d Runtime Optimization Summary - 2026-05-21

## Current Recommended Run

Observed target machine:
- GPU: NVIDIA A100-SXM4-80GB
- PyTorch: 2.8.0+cu126
- CUDA runtime in PyTorch: 12.6
- Compute capability: sm_80
- Official trainer: `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`
- Wrapper: `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh`

Current defaults after optimization:
- `EPOCHS=500`
- `BATCH_SIZE=16`
- `EVAL_BATCH_SIZE=16`
- `EVAL_EVERY=10`
- `DATA_RESIDENCY=gpu`
- `R2_SYNC_EVERY_EPOCHS=0`
- `AMP=0`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

The wrapper defaults now keep the training tensors resident on GPU, while test data remains CPU-backed and is copied only during evaluation. This avoids the earlier slow CPU/DataLoader path while preserving enough GPU memory for `BATCH_SIZE=16`.

## What Went Wrong Earlier

The early CLI was not a line-for-line copy of the original `trainFNO2d_unnormalized.py`. It used the same model classes, but its initial data path was slower:
- Original script precomputed `x_train`, `y_train`, `x_test`, and `y_test` once, then used `TensorDataset`.
- Early CLI used a custom Dataset that sliced each sample in `__getitem__()` and called `.contiguous()` per sample before DataLoader collation.
- Early CLI also initially used `BATCH_SIZE=4`, causing `ceil(1150 / 4) = 288` training batches per epoch.

Inference: the earlier 6-8 minute epoch timings were caused mainly by conservative batch size plus inefficient data handling, not by replacing the FNO model. The current optimized path fixes this by precomputing train tensors and keeping them on GPU.

## Model Path

The trainer uses the repository model implementation:
- `2D_NS_FNO2d_recurrent/models/FNO2d.py`
- `FNO2d`
- `RecurrentPredictor`
- Recurrent setup: `T_in=10`, `T_out=10`, `step=1`

Each training batch performs 10 autoregressive FNO steps because `T_out=10` and `step=1`.

## Runtime Evidence

Observed slow baseline with old data path and `BATCH_SIZE=4`:
- Epoch 1: `475.279s` (`7m55s`)
- Epoch 2: `444.870s` (`7m25s`)
- Epoch 3: `476.025s` (`7m56s`)
- Epoch 4: `487.730s` (`8m08s`)
- First-four-epoch estimate: about `65h` for 500 epochs.

Observed optimized `DATA_RESIDENCY=gpu`, `BATCH_SIZE=16`:
- Standalone train-only full epoch: `87.1s` (`1.45min`), peak memory `77.80 GiB`.
- Official one-epoch validation with epoch eval: `100.266s` (`1m40s`).
- Full one-epoch command including final eval/save: `2m31s`.

Observed optimized `DATA_RESIDENCY=gpu`, `BATCH_SIZE=10`:
- Standalone full epoch: `99.2s` (`1.65min`), peak memory `52.32 GiB`.

Observed optimized `DATA_RESIDENCY=gpu`, `BATCH_SIZE=4`:
- Official one-epoch validation with epoch eval: `152.087s` (`2m32s`).
- Full one-epoch command including final eval/save: `3m22s`.
- Estimated 500 epochs: roughly `21h` using the one-epoch validation, before final upload details.

Estimated optimized `BATCH_SIZE=16`, `EPOCHS=500`:
- Using `100.266s/epoch`: about `13.9h` plus final evaluation/checkpoint/R2 upload.
- Using train-only `87.1s/epoch`: about `12.1h` plus periodic eval/final work.
- Practical expectation: around `12-14h`, not the earlier `~65h`.

## Memory Evidence

Batch-size tests on A100 80GB:
- `BATCH_SIZE=10`: safe, peak about `52.32 GiB` in GPU-resident optimized benchmark.
- `BATCH_SIZE=16`: passed full official validation after keeping only train tensors on GPU; peak in standalone optimized benchmark about `77.80 GiB`.
- `BATCH_SIZE=17`: only ran with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in short test, peak about `76.53 GiB`; too close to the memory ceiling for official use.
- `BATCH_SIZE=18`: OOM.
- `BATCH_SIZE=20`: OOM for the current `modes=64`, `width=60`, recurrent setup.

Recommendation: use `BATCH_SIZE=16` for speed, fall back to `BATCH_SIZE=10` or `14` if OOM occurs.

## AMP Status

Full-model AMP is disabled intentionally.

Observed tests:
- FP32/no autocast: passed.
- Default FP16 AMP: failed at spectral convolution with `baddbmm_cuda not implemented for ComplexHalf`.
- BF16 autocast: failed at `torch.fft.rfft2` with `Unsupported dtype BFloat16`.

Inference: the FFT/spectral convolution path is not compatible with global AMP in this PyTorch/CUDA setup. A selective mixed-precision implementation would need to keep FFT/spectral complex operations in FP32/complex64 and validate numerical behavior.

## Checkpoints And R2 Upload

The trainer saves:
- `checkpoints/best.pt`: whenever evaluated test relative L2 reaches a new finite minimum.
- `checkpoints/latest.pt`: every `SAVE_EVERY` epochs and at the end.
- `checkpoints/final.pt`: final model/optimizer/scheduler state.
- legacy `.pth` model state file.
- `train_log.csv`, `progress.jsonl`, `progress_latest.json`, `results.json`.

With `R2_UPLOAD=1`, the final upload copies the whole output directory to R2, so `checkpoints/best.pt` is included. Periodic small-record sync is off by default with `R2_SYNC_EVERY_EPOCHS=0`.

## Command Template

Run from a persistent shell:

```bash
cd /workspace/NeuralOperatorRobustness2

export R2_ACCESS_KEY_ID='YOUR_R2_ACCESS_KEY_ID'
export R2_SECRET_ACCESS_KEY='YOUR_R2_SECRET_ACCESS_KEY'

PYTHON_BIN=adv_robust/bin/python DATA_RESIDENCY=gpu EPOCHS=500 BATCH_SIZE=16 EVAL_BATCH_SIZE=16 EVAL_EVERY=10 R2_SYNC_EVERY_EPOCHS=0 PROGRESS_EVERY=10 R2_UPLOAD=1 AMP=0 ./2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh
```

Do not put R2 secrets into repository files.

## Active 500-Epoch Run Timing Update - 2026-05-21 09:08 UTC

Observed evidence:
- Active `DATA_RESIDENCY=gpu`, `BATCH_SIZE=16`, `EPOCHS=500` output directory: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC`.
- Epoch 1: `93.884s` (`1m34s`), including test evaluation.
- Epoch 2: `83.839s` (`1m24s`), train only.
- Epoch 3: `83.837s` (`1m24s`), train only.
- The trainer's estimate after epoch 3 was total runtime `12h06m36s`, with about `12h02m12s` remaining.

Inference:
- The current practical expectation for this active 500-epoch run is about `11h45m-12h10m` before final R2 upload variability.
- This confirms the optimized run is much closer to the expected A100 timing than the earlier slow `batch_size=4`/CPU-collation run.

## Active Run Status Update - 2026-05-21 09:12 UTC

Observed evidence:
- Epochs 2-6 were all approximately `83.8-83.9s` non-evaluation epochs.
- After epoch 6, the trainer estimated total runtime `11h39m03s`, with `11h30m27s` remaining.
- The run was active in epoch `7/500`, batch `40/72`, with A100 utilization still at `100%`.

Inference:
- The active long run is stable so far and is trending closer to `11.5-12h` than the earlier conservative `12-14h` estimate.

## Active Run Epochs 7-10 Timing Update - 2026-05-21 09:18 UTC

Observed evidence:
- Epoch 7: `83.929s`.
- Epoch 8: `83.859s`.
- Epoch 9: `83.889s`.
- Epoch 10: `99.226s`, with scheduled test evaluation; test relative L2 mean was `0.225781`.
- The trainer estimated total runtime `12h04m39s` after epoch 10.

Inference:
- Normal epochs are stable around `83.9s`; scheduled evaluation epochs are currently around `99s`.
- A 500-epoch run with evaluation every 10 epochs is still tracking around `12h` total before final R2 upload variability.

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
