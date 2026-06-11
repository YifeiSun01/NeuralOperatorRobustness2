# Vast.ai Darcy m64/w60 Checkpoint Load - 2026-06-11

Status: complete.

Observed from local setup:
- Repository cloned to `/workspace/NeuralOperatorRobustness2` on branch `vast-ai`.
- `adv_robust` was created with `tools/setup_adv_robust_gpu_env.py --python python3`.
- GPU verification passed with PyTorch `2.8.0+cu126`, CUDA `12.6`, V100 compute capability `sm_70`, and JAX backend `gpu`.
- `pip check` reported no broken requirements.

Observed from R2 listing:
- Source prefix: `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/2D_Darcy_FNO2d/saved_models/2D`.
- Selected checkpoint directory: `darcy_N1500_nx85_m64_w60_e500_20260528`.
- Downloaded local files:
  - `2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/best.pt`
  - `2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/config.json`
  - `2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/train_log.csv`

Observed from downloaded config:
- Resolution: `85`.
- Modes: `64`.
- Width: `60`.
- Layers: `4`.
- Padding: `0`.
- Model parameters recorded by training config: `235988641`.
- Training data: `N=1200`, test data: `N=300`, binary Darcy coefficients `3/12`, alpha `2.0`, tau `3.0`, seed `45`.

Observed from GPU load smoke test:
- Checkpoint loaded strictly into `FNO2d(modes1=64, modes2=64, width=60, num_layers=4, in_channels=1, out_channels=1, padding=0)`.
- Missing keys: `0`.
- Unexpected keys: `0`.
- Parameter count from loaded model: `235988641`.
- Random GPU input shape `(1, 85, 85, 1)` produced output shape `(1, 85, 85, 1)`.
- Output finite check: `True`.
- CUDA memory allocated after load and one forward pass: about `908.42 MiB`.

Inference from the smoke test:
- The requested trained Darcy Flow PyTorch FNO checkpoint is present locally and loadable on the current V100 GPU without CPU fallback.
- No training was run in this setup step; this only restored and validated the previously trained model.

Remaining work:
- If this checkpoint is used for attacks/evaluation, download the matching datasets from R2 or verify they already exist locally.
- `/workspace` is not backed by a persistent Vast volume on this instance, so local downloads should be synced back to durable storage if they are modified or regenerated.
