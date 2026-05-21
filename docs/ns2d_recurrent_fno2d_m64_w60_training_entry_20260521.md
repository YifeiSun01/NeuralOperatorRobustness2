# 2D NS Recurrent FNO2d 64/64/60 Training Entry

Date: 2026-05-21

## Status

Prepared a GPU-only PyTorch recurrent FNO2d training entry for the requested
`modes1=64`, `modes2=64`, `width=60` setup. Full training was not launched in
this turn.

## Observed Evidence

- Model code exists at
  `2D_NS_FNO2d_recurrent/models/FNO2d.py`. It defines `FNO2d(modes1, modes2,
  width, num_layers=4, in_channels=10)` and `RecurrentPredictor`, which rolls
  one-step FNO2d predictions autoregressively to `T_out`.
- Legacy training code exists at
  `2D_NS_FNO2d_recurrent/training_models/trainFNO2d_unnormalized.py`, but it is
  hard-coded to `modes=96`, `width=80`, fixed paths, and `.cuda()` usage.
- A historical log for the requested `modes=64`, `width=60`, `epochs=500`,
  `Tin=10`, `T=10` model exists at
  `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_width60_epochs500_Tin10_T10/NS_2d_FNO_log_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.txt`.
  That log records `235989181` model parameters and final legacy summed
  relative L2 values at epoch 499: train `66.15601575`, test `10.51145339`.
- No `.pt` dataset files are currently present under
  `2D_NS_FNO2d_recurrent/datasets/` in this working tree; the observed count was
  `0`.
- No `.pth` checkpoints are currently present under
  `2D_NS_FNO2d_recurrent/saved_models/` in this working tree; the observed
  count was `0`.
- Dataset generation code exists at
  `2D_NS_FNO2d_recurrent/data_generation/generate_ns_real_initial_batched.py`
  and `2D_NS_FNO2d_recurrent/data_generation/VT_NS_gen_all_frame.py`. The newer
  generator expects source initial conditions under
  `2D_NS_FNO2d_recurrent/datasets/source_zongyi_real_initial/`, which is not
  present locally.
- R2 bucket lookup on 2026-05-21 found the selected machine-sync prefix, but no
  2D NS `.pt` dataset objects. The recursive R2 scan found `5482`
  data/checkpoint-suffix objects, `32` `.pt` objects, `0` `.pth` objects, `0`
  data objects under `2D_NS_FNO2d_recurrent/datasets/`, and `0` `.pth`
  checkpoint objects under `2D_NS_FNO2d_recurrent/`.
- The only R2 object matching 2D NS/FNO2d keywords with a data/checkpoint suffix
  was
  `machine-sync/NeuralOperatorRobustness2-selected/fno_training_runs/ns_m12_w20_profile_500/ns_real_initial_laxmap/ns_2d/checkpoints/fno2d_pytorch.pt`
  with size `3730297` bytes. This is a small `modes=12,width=20` training-suite
  checkpoint, not the requested 2D NS training dataset and not the 64/64/60
  model.
- The R2 dataset/data suffix objects found in the selected prefix were all
  1D Burgers `.pt` files, not 2D Navier-Stokes data.

## New Training Files

- `2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`
- `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh`

The new Python entry refuses CPU fallback, runs `nvidia-smi`, records PyTorch
version/CUDA/device/compute capability/CUDA arch list, checks a CUDA matmul, and
writes `gpu_verification.txt` before training.

## Default Command

```bash
bash 2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh
```

The wrapper defaults to:

- `modes1=64`
- `modes2=64`
- `width=60`
- `epochs=500`
- `ntrain=1000`
- `ntest=100`
- `batch_size=4`
- `eval_batch_size=4`
- `T_in=10`
- `T_out=10`
- `target_size=256`
- `lr=0.001`
- `weight_decay=0.0001`

If the generated dataset is stored at a non-default path, pass it explicitly:

```bash
bash 2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh \
  --train-path /path/to/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pt
```

For a separate test file:

```bash
bash 2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh \
  --train-path /path/to/train.pt \
  --test-path /path/to/test.pt \
  --ntest 50
```

## Outputs

Each run writes under
`2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch*/`
unless `--output-root` or `--run-name` is overridden.

Expected files include:

- `gpu_verification.txt`
- `config.json`
- `dataset_info.json`
- `train_log.csv`
- `NS_2d_FNO_log_trainedby_<dataset>.txt`
- `NS_2d_FNO_model_trainedby_<dataset>.pth`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `checkpoints/final.pt`
- `results.json`

## Verification Performed

- `adv_robust/bin/python -m py_compile 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py`
- `adv_robust/bin/python 2D_NS_FNO2d_recurrent/training_models/train_fno2d_recurrent_cli.py --help`
- `2D_NS_FNO2d_recurrent/training_models/run_fno2d_recurrent_m64_w60.sh --help`

## Remaining Work

- Put or generate the required `.pt` data files locally before launching full
  training.
- The checked R2 selected machine-sync prefix does not currently provide the
  required 2D NS `.pt` training data; use the generator or locate another
  storage prefix/machine copy.
- Run the command above on the GPU machine. The script will stop immediately if
  CUDA, `nvidia-smi`, or the PyTorch CUDA architecture check fails.

## 2026-05-21 Dataset Completion And Training Launch

Observed from local files after generation completed:

- Test output: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt`, size `288362621` bytes, with `x=(50,256,256)` and `y=(50,256,256,21)`.
- Train output: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt`, size `6632250121` bytes, with `x=(1150,256,256)` and `y=(1150,256,256,21)`.
- Generation summary: `2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/generation_summary.json`, with `38` timing records.

Observed from `generation_summary.json`:

- Test split: `2` batches, `50` samples, rollout seconds `104.30499046598561`.
- Train split: `36` batches, `1150` samples, rollout seconds `2278.595210202737`.

Observed from training launch:

- Dry run passed and reported `118024381` trainable parameters by PyTorch count.
- Detached training PID: `314121`.
- Training log: `run_logs/train_fno2d_recurrent_m64_w60_20260521_0530.log`.
- Output directory: `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_real_initial_laxmap_v100/`.
- Initial training GPU status: V100-SXM2-32GB, GPU utilization `100%`, memory used `22662 MiB / 32768 MiB`.


## 2026-05-21 Training Status Check 05:34 UTC

Observed from process, GPU, and log inspection:

- Data generation is complete; both train and test 256x256 `.pt` files exist.
- Training is not complete. Detached training PID `314121` was still running with elapsed time `02:54`.
- GPU status at `2026-05-21 05:34:08 UTC`: utilization `91%`, memory used `22662 MiB / 32768 MiB`, power draw `261.45 W`.
- Training log still showed startup/epoch `0/500`; no final `results.json` or completed training metrics were observed yet.

Inference:

- The run has moved past dataset generation into active GPU training.
- The current evidence supports "generation finished, training still running", not "full model training finished".


## 2026-05-21 Progress Logging Patch

Observed from the active training process and log:

- The already-running PID `314121` was started before the progress patch and only emits tqdm-style epoch-level updates in the redirected log.
- The log showed epoch `1/500` completed in about `217.06s`, with tqdm estimating roughly `30:05:15` for the full 500-epoch run at that early rate.

Code change made for future launches/restarts:

- Added `--progress-every` to `train_fno2d_recurrent_cli.py`, defaulting to `10` training batches.
- Added flushed `[progress]` log lines during each epoch with epoch, batch, samples, loss, running train relative L2, learning rate, epoch ETA, and total train-batch ETA.
- Added `progress.jsonl` for append-only machine-readable progress events.
- Added `progress_latest.json` for the most recent progress snapshot.
- Added `[eval]` and `[epoch]` log lines around evaluation and epoch completion.

Inference:

- The patch solves progress visibility for new runs or for a restart of this run.
- The current PID cannot load the modified Python source while it is already executing; getting batch-level progress for this exact run requires stopping and restarting it.


## 2026-05-21 Frame Semantics Check

Observed from local generated `.pt` files:

- Train file: `x=(1150,256,256)`, `y=(1150,256,256,21)`, dtype `torch.float32`.
- Test file: `x=(50,256,256)`, `y=(50,256,256,21)`, dtype `torch.float32`.
- For both train and test, `max_abs(x - y[...,0]) = 0.0`, so `y` frame index `0` is the real initial condition.
- `NSTrajectoryDataset.__getitem__` returns `x = sample[..., :t_in]` and `y = sample[..., t_in:t_in+t_out]`.
- With defaults `t_in=10` and `t_out=10`, training uses stored frame indices `0..9` as input and `10..19` as supervised targets; stored frame index `20` is not used.
- `RecurrentPredictor.forward` predicts one frame at a time and rolls the input window with `torch.cat([x[..., step:], y_pred], dim=-1)`, so the model is autoregressive/recurrent over the 10 output frames.

Inference:

- In one-based stored-frame language, the current training setup is input frames `1..10`, predict supervised frames `11..20`, and does not supervise against stored frame `21`.


## 2026-05-21 - 2D NS FNO2d Runtime And Hardware Utilization Check 05:43 UTC

Status: inspected active recurrent FNO2d 64/64/60 training runtime and GPU utilization.

Observed evidence:
- Active training PID `314121`, elapsed time `12:05`, CPU time `00:30:00`, process RSS about `7.84 GB`.
- GPU snapshot at `2026-05-21 05:43:20 UTC`: Tesla V100-SXM2-32GB, GPU utilization `100%`, memory utilization `82%`, memory used `22662 MiB / 32768 MiB`, power draw `222.73 W`, SM clock `1530 MHz`, memory clock `877 MHz`.
- `nvidia-smi dmon` samples showed SM utilization mostly `88-100%`, memory utilization `76-84%`, and power draw `213-262 W`.
- `train_log.csv` contained epochs 1-3 with seconds `213.07864790898748`, `211.67835453199223`, and `213.18633831501938`.
- The redirected tqdm log showed epoch `3/500` around `10:51`, consistent with about `212-217s` per epoch including evaluation.

Inference:
- The V100 is substantially occupied by the training run; this is a GPU-limited run rather than an obviously CPU-idle bottleneck.
- At the observed early rate, the 500-epoch run is roughly `29.5-30.0` hours total on the current V100 setup, assuming the per-epoch time remains stable.
- Larger/newer GPUs such as A100/B200 should reduce runtime, but exact speedup must be measured because this FNO workload includes FFT/complex spectral operations and autoregressive recurrent steps, not only dense tensor-core GEMMs.

Remaining work:
- Continue monitoring actual epoch times from `train_log.csv`.
- If restarting for progress logging, use the patched `--progress-every` command and record the new PID/log path.


## 2026-05-21 - 2D NS FNO2d V100 Speed-Up Assessment

Status: assessed whether the active V100 training run can be made substantially faster without changing the requested model/data semantics.

Observed evidence:
- Active run is using a Tesla V100-SXM2-32GB with GPU utilization around `91%` at the snapshot and previous `nvidia-smi dmon` samples mostly `88-100%` SM utilization.
- Memory used remains about `22662 MiB / 32768 MiB`, so there is memory headroom, but the GPU compute path is already highly occupied.
- First three recorded epoch times are approximately `213.08s`, `211.68s`, and `213.19s`, implying about `29.5` hours for 500 epochs if stable.
- The current process was launched without `--amp`; the trainer supports `--amp`, but this must be benchmarked because the FNO implementation uses FFT and complex spectral multiplication.

Inference:
- The current V100 run is primarily GPU-limited rather than obviously blocked by CPU or data loading.
- Increasing batch size may improve overhead modestly but is unlikely by itself to reduce `~30h` to `~10h` because SM/memory utilization is already high.
- Realistic V100-side speedups without changing the model/data are likely from `--amp`, less frequent evaluation, and perhaps a slightly larger batch if stable; a full 3x speedup on V100 is uncertain and should not be assumed without a benchmark.
- Reaching roughly 10 hours is much more plausible on a faster GPU such as A100/B200, or by reducing the training budget such as epochs/early stopping.

Remaining work:
- If the user wants to optimize on this V100, stop the current run and launch a short AMP/eval-frequency benchmark before committing to a 500-epoch run.


## 2026-05-21 - 2D NS Training Stopped And Dataset Backed Up To R2

Status: stopped the active V100 recurrent FNO2d training run and backed up the generated 256x256 real-initial 2D NS dataset to R2.

Observed evidence:
- User requested stopping the active training process before backup.
- Training process PID `314121` was terminated at approximately `2026-05-21 05:48 UTC`.
- GPU snapshot after stopping showed utilization `0%` and memory used `0 MiB / 32768 MiB`.
- R2 destination prefix: `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/real_initial_laxmap_single/`.
- R2 listing after copy showed:
  - `generation_summary.json`, size `21943` bytes.
  - `test/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt`, size `288362621` bytes.
  - `train/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt`, size `6632250121` bytes.

Inference:
- The generated real-initial 256x256 train/test dataset and generation summary are backed up on R2 under the selected machine-sync prefix.
- The stopped training run did not complete; its partial local run outputs remain local and were not treated as final training results.

Remaining work:
- Push code and experiment records to GitHub branch `vast-ai`.
- Keep large dataset/checkpoint artifacts out of git unless explicitly requested.
