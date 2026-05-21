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
