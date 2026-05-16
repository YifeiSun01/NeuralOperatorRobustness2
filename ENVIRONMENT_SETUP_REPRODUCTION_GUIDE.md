# Environment Setup Reproduction Guide

## 2026-05-16 GPU-Aware Setup Rule

Do not rebuild `adv_robust` by hand for official experiment work. Use the
project setup script instead:

```bash
cd /workspace/NeuralOperatorRobustness2
python3 tools/setup_adv_robust_gpu_env.py
```

For an already-created environment, verify it with:

```bash
tools/setup_adv_robust_gpu_env.py --verify-only
```

This script installs the pinned requirements and refuses to pass unless PyTorch
and JAX both execute on the visible GPU. On the current V100 machine, it also
checks that the PyTorch wheel contains `sm_70`. This avoids the previous failure
mode where `torch.cuda.is_available()` was true but the wheel could not launch
kernels on V100.

This note is the clean, reproducible setup path for the Vast AI / direct-GPU
environment used by this repository. It intentionally removes the older detours
from the first setup attempts: Slurm/HPC paths, conda paths under `/blue`,
untracked local package experiments, and installing the public upstream Exponax.

The environment was originally verified on an A100 instance on 2026-05-09.
The current GPU-aware setup path was updated and verified on 2026-05-16 for the
current V100 instance:

```text
repo root: /workspace/NeuralOperatorRobustness2
branch:    vast-ai
remote:    https://github.com/YifeiSun01/NeuralOperatorRobustness2.git
venv:      adv_robust/
GPU:       Tesla V100-SXM2-32GB
arch:      sm_70
PyTorch:   2.8.0+cu126
JAX:       gpu backend verified
```

## Non-Negotiable Exponax Detail

Do **not** install the ordinary public `exponax` package for this repo.

This project currently depends on the modified Exponax fork:

```text
https://github.com/YifeiSun01/modified_exponax.git
```

The pinned dependency in [requirements.txt](/workspace/NeuralOperatorRobustness2/requirements.txt:10) is:

```text
exponax @ git+https://github.com/YifeiSun01/modified_exponax.git@febe20b2103192654b256243eb464742616e95df
```

That fork is required for the JAX/Exponax solver code paths used by the attack
framework experiments. Installing plain `pip install exponax` can silently give
the wrong solver behavior or missing APIs.

## Clean Setup

Start from a fresh GPU machine with Git, Python 3.12, and a recent NVIDIA driver.
The current verified environment uses CUDA 12 PyTorch/JAX wheels and is
validated by `tools/setup_adv_robust_gpu_env.py` on the visible GPU before use.

```bash
git clone https://github.com/YifeiSun01/NeuralOperatorRobustness2.git
cd NeuralOperatorRobustness2
git checkout vast-ai
```

Create the local virtual environment with the GPU-aware project setup script.
The repo convention is to keep it inside the project as `adv_robust/`; it is
ignored by Git.

```bash
python3 tools/setup_adv_robust_gpu_env.py
```

The script creates `adv_robust/` if needed, installs `requirements.txt`, and
then runs strict PyTorch/JAX GPU verification. Do not treat the environment as
ready until this script passes.

If `python3.12` is not available on the machine, install Python 3.12 first or use
a Python 3.12 provider such as conda/mamba. The current environment reports:

```text
Python 3.12.13
```

## Required Environment Variables

Use these before running attack or solver benchmarks:

```bash
cd /workspace/NeuralOperatorRobustness2
source adv_robust/bin/activate

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# Important when mixing JAX and PyTorch on the same GPU.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

`XLA_PYTHON_CLIENT_PREALLOCATE=false` is especially useful here because several
experiments mix PyTorch and JAX. Without it, JAX may reserve a large fraction of
GPU memory up front, which makes PyTorch/JAX memory comparisons harder to read
and can trigger avoidable OOMs.

## Verified Package Versions

The current environment passes `pip check` with no broken requirements.

Key package versions:

```text
torch        2.8.0+cu126
torch cuda   12.6
jax          0.10.0
jaxlib       0.10.0
numpy        2.4.4
exponax      0.1.0 from YifeiSun01/modified_exponax
equinox      0.13.8
matplotlib   3.10.9
pillow       12.2.0
tqdm         4.67.3
pandas       3.0.2
scipy        1.17.1
DeepXDE      1.15.0
phiflow      3.4.0
torch2jax    0.1.0
```

The CUDA runtime libraries are installed through Python wheels in
`requirements.txt`, including `nvidia-cublas-cu12`, `nvidia-cufft-cu12`,
`nvidia-cudnn-cu12`, `triton`, and related CUDA packages.

## Smoke Tests

Run this immediately after installation. It includes `nvidia-smi`, PyTorch GPU
architecture validation, real PyTorch CUDA matmul, JAX GPU validation, real JAX
GPU matmul, and `pip check`.

```bash
tools/setup_adv_robust_gpu_env.py --verify-only
```

Additional manual checks, if needed:

```bash
nvidia-smi
```

```bash
python -m pip check
```

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "import jax; print(jax.__version__, jax.devices())"
```

```bash
python -c "import exponax, equinox, torch, jax; print('core imports ok')"
```

```bash
python -m py_compile solvers.py tools/attack_framework_matrix.py
```

A minimal attack-framework smoke run is:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
adv_robust/bin/python tools/attack_framework_matrix.py \
  --cases burgers_1d \
  --combos torch_solver_torch_model \
  --steps 1 \
  --sample-index 0 \
  --burgers-model-weight-source jax_real_imag \
  --burgers-norm inf \
  --burgers-epsilon 0.5 \
  --burgers-alpha 0.05 \
  --burgers-torch-solver-dtype float64 \
  --burgers-jax-solver-dtype float64 \
  --device cuda \
  --output-root /tmp/attack_framework_smoke \
  --max-gif-frames 2
```

This smoke run assumes the local datasets/checkpoints expected by the repo are
present. If it fails with missing checkpoint/data paths, fix the artifact layout
first; do not change package versions as the first response.

## Main Experiment Commands

Burgers baseline matrix:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
adv_robust/bin/python tools/attack_framework_matrix.py \
  --cases burgers_1d \
  --combos torch_solver_torch_model,torch_solver_jax_model,jax_solver_torch_model,jax_solver_jax_model \
  --steps 100 \
  --sample-index 0 \
  --burgers-model-weight-source jax_real_imag \
  --burgers-norm inf \
  --burgers-epsilon 0.5 \
  --burgers-alpha 0.05 \
  --burgers-torch-solver-dtype float64 \
  --burgers-jax-solver-dtype float64 \
  --device cuda \
  --gpu-sample-interval 0.02 \
  --output-root benchmark_results/attack_framework_matrix_burgers_idx0_same_jax_weights_float64 \
  --max-gif-frames 30
```

Navier-Stokes baseline matrix:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
adv_robust/bin/python tools/attack_framework_matrix.py \
  --cases ns_2d \
  --combos torch_solver_torch_model,torch_solver_jax_model,jax_solver_torch_model,jax_solver_jax_model \
  --steps 25 \
  --sample-index 1 \
  --ns-norm inf \
  --ns-epsilon 1.0 \
  --ns-alpha 0.01 \
  --ns-torch-solver-dtype float64 \
  --ns-jax-solver-dtype float64 \
  --device cuda \
  --gpu-sample-interval 0.02 \
  --output-root benchmark_results/attack_framework_matrix_ns_idx1_25steps_native_float64 \
  --max-gif-frames 30
```

Torch solver compile experiments:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
adv_robust/bin/python tools/attack_framework_matrix.py \
  --cases burgers_1d \
  --combos torch_solver_torch_model,torch_solver_jax_model \
  --steps 100 \
  --sample-index 0 \
  --burgers-model-weight-source jax_real_imag \
  --burgers-norm inf \
  --burgers-epsilon 0.5 \
  --burgers-alpha 0.05 \
  --burgers-torch-solver-dtype float64 \
  --burgers-jax-solver-dtype float64 \
  --torch-compile-solver \
  --device cuda \
  --gpu-sample-interval 0.02 \
  --output-root benchmark_results/attack_framework_matrix_burgers_torch_compile_solver \
  --max-gif-frames 30
```

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
adv_robust/bin/python tools/attack_framework_matrix.py \
  --cases ns_2d \
  --combos torch_solver_torch_model,torch_solver_jax_model \
  --steps 25 \
  --sample-index 1 \
  --ns-norm inf \
  --ns-epsilon 1.0 \
  --ns-alpha 0.01 \
  --ns-torch-solver-dtype float64 \
  --ns-jax-solver-dtype float64 \
  --torch-compile-solver \
  --device cuda \
  --gpu-sample-interval 0.02 \
  --output-root benchmark_results/attack_framework_matrix_ns_torch_compile_solver \
  --max-gif-frames 30
```

## Codex / IDE Usage Notes

Codex itself is not a Python package dependency of this repo. The working setup
is an IDE/Codex agent connected to this workspace, with shell access in:

```text
/workspace/NeuralOperatorRobustness2
```

For Codex-assisted work, the important repo-side setup is:

```bash
git checkout vast-ai
git config core.hooksPath .githooks
source adv_robust/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

The hooks are:

```text
.githooks/pre-commit -> python3 tools/check_large_files.py --mode staged
.githooks/pre-push   -> python3 tools/check_large_files.py --mode tracked
```

They exist to prevent accidentally committing/pushing very large model weights,
datasets, figures, or binary outputs.

## Git / Artifact Rules

The repo intentionally ignores local environments and heavy artifacts:

```text
adv_robust/
.venv/
run_logs/
wandb/
mlruns/
*.pt, *.pkl, *.npy, *.npz, *.mat
*.png, *.gif, *.jpg
*.mp4, *.zip, *.tar.gz
```

CSV/TXT/ERR/OUT files are selectively allowed because many benchmark summaries
are small and useful to track.

Before committing:

```bash
git status --short
python3 tools/check_large_files.py --mode staged
```

After committing on `vast-ai`, push with:

```bash
git push origin vast-ai
```

## Old Detours To Avoid

These came from earlier setup and old logs, but they should not be part of the
new reproducible path:

- Do not use old hard-coded `/blue/shiboli.fsu/...` paths.
- Do not rely on Slurm job scripts for this Vast AI environment.
- Do not install the public upstream `exponax`.
- Do not start by changing JAX/Torch versions to fix missing local data files.
- Do not track local virtualenvs, model checkpoints, PNG/GIF/NPZ outputs, or
  benchmark scratch artifacts.
- Be careful with old JAX APIs such as `jax.lib.xla_bridge.devices`; current code
  should use modern `jax.devices()` style checks.

## Practical Debug Checklist

If an experiment hangs or gets killed:

1. Check GPU processes and memory:

   ```bash
   nvidia-smi
   ```

2. Confirm the virtual environment is active and coherent:

   ```bash
   source adv_robust/bin/activate
   python -m pip check
   ```

3. Confirm Torch and JAX see the same intended GPU:

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "import jax; print(jax.devices())"
   ```

4. For mixed JAX/PyTorch runs, keep:

   ```bash
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   ```

5. For `torch.compile` runs involving FNO/FFT/complex tensors, expect possible
   long compile time, fallback, slower execution, or different numerical attack
   trajectories. The detailed observations are recorded in
   [ATTACK_FRAMEWORK_MATRIX_OBSERVATIONS.md](/workspace/NeuralOperatorRobustness2/ATTACK_FRAMEWORK_MATRIX_OBSERVATIONS.md:1).
