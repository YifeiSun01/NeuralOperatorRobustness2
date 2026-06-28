# GPU-Only Experiment Policy

Date: 2026-05-16

This note records the standing execution rule for the neural-operator robustness
experiments in this project.

## Standing Rule

Official experiment runs must use GPU. Do not silently switch to CPU when CUDA
fails, when PyTorch cannot execute on the installed GPU, or when JAX cannot see a
GPU backend.

If the GPU path fails, stop the experiment and fix the environment first. A CPU
run can only be used if the user explicitly asks for a CPU-only diagnostic, and
that run must not be reported as the official experiment result.

## Current Machine Observation

Observed on 2026-05-16:

- GPU: Tesla V100-SXM2-32GB
- GPU compute capability: `sm_70`
- NVIDIA driver: 570.211.01
- System CUDA reported by `nvidia-smi`: 12.8
- Current `adv_robust` PyTorch at the time of this note: `2.11.0+cu128`
- Problem: that PyTorch wheel can see the V100 but does not include `sm_70`, so
  CUDA kernels fail with `no kernel image is available for execution on the
  device`.
- JAX backend check at the time of this note reported GPU backend and
  `CudaDevice(id=0)`.

## Required Pre-Run Checks

Before starting an official experiment, record:

- `nvidia-smi`
- Python executable path, expected to be `adv_robust/bin/python`
- PyTorch version and `torch.version.cuda`
- `torch.cuda.is_available()`
- CUDA device name
- CUDA compute capability
- `torch.cuda.get_arch_list()` and confirmation that it contains the active GPU
  architecture, currently `sm_70` for V100
- A tiny CUDA tensor operation that successfully synchronizes on GPU
- JAX `default_backend()` and `jax.devices()` if the solver/model path uses JAX

## Persistence Mechanism

This rule is persisted in two places:

- `AGENTS.md`, so future Codex sessions that read repository instructions see
  the GPU-only rule before starting experiment work.
- this document, so the human-readable project notes explain why CPU fallback is
  not acceptable for official results.

Experiment scripts should also enforce this rule in code, not only in prose.

## Resolution Applied On 2026-05-16

The `adv_robust` environment was repaired for V100 GPU execution:

- installed PyTorch: `2.8.0+cu126`
- `torch.cuda.get_arch_list()` includes `sm_70`
- PyTorch CUDA tensor matmul succeeded on `Tesla V100-SXM2-32GB`
- JAX backend: `gpu`
- JAX GPU matmul succeeded
- `pip check` reported no broken requirements

`requirements.txt` was regenerated with `--extra-index-url https://download.pytorch.org/whl/cu126` and `torch==2.8.0+cu126`, so future environment rebuilds should keep the V100-compatible PyTorch wheel instead of reinstalling the broken `torch==2.11.0+cu128` combination.

## CUDA Wheel Compatibility Reference

Detailed notes about V100/A100/H100 compute capabilities, PyTorch wheel
architecture lists, CUDA runtime versions, and the JAX/cuDNN compatibility trap
are recorded in `docs/cuda_gpu_wheel_compatibility_notes_20260516.md`. Read that
file before reinstalling PyTorch, JAX, or CUDA runtime packages.
