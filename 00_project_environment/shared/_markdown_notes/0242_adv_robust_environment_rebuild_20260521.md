# adv_robust Environment Rebuild - 2026-05-21

Status: rebuilt and verified with the repository setup script.

Observed evidence:
- Existing `adv_robust` was unusable: `adv_robust/bin/python` was a broken symlink path and `tools/setup_adv_robust_gpu_env.py --verify-only` failed with `Too many levels of symbolic links` for `adv_robust/bin/python3.12`.
- The broken directory was preserved as `adv_robust_broken_20260521_venv_symlink`.
- Rebuild command used: `tools/setup_adv_robust_gpu_env.py`.
- Active GPU from `nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader`: `NVIDIA A100-SXM4-80GB`, compute capability `8.0`, required PyTorch arch `sm_80`.
- Installed environment verification reported:
  - Python: `/workspace/NeuralOperatorRobustness2/adv_robust/bin/python`
  - PyTorch: `2.8.0+cu126`
  - PyTorch CUDA runtime: `12.6`
  - `torch.cuda.is_available()`: `True`
  - Torch device: `NVIDIA A100-SXM4-80GB`
  - Torch capability: `(8, 0)` / `sm_80`
  - Torch arch list included `sm_80`
  - PyTorch CUDA matmul succeeded.
  - JAX backend: `gpu`
  - JAX GPU devices: `[CudaDevice(id=0)]`
  - JAX GPU matmul succeeded.
  - `pip check`: `No broken requirements found.`

Inference:
- `adv_robust` is now usable for GPU-only PyTorch/JAX work on the current A100 machine.
- The previous venv directory was not safe to use for training because its executable entry point was broken.

Remaining work:
- Use `adv_robust/bin/python` or the shell wrappers that auto-detect `adv_robust/bin/python` for subsequent training runs.
- Keep using `tools/setup_adv_robust_gpu_env.py` rather than raw `pip install -r requirements.txt` when rebuilding this environment.
