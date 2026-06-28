# CUDA / GPU Wheel Compatibility Notes

Date: 2026-05-16

This note records why CUDA/GPU runs can fail on different NVIDIA GPUs and what
must be checked before installing PyTorch, JAX, or CUDA runtime packages for this
project.

## Short Rule

Do not choose a PyTorch CUDA wheel only from the machine's displayed CUDA
version. Choose a wheel that satisfies all of these at the same time:

1. the NVIDIA driver can run the wheel CUDA runtime;
2. the PyTorch wheel was compiled for the GPU compute capability;
3. JAX can still use GPU after the install;
4. the installed cuDNN version is high enough for both PyTorch and JAX;
5. a real CUDA tensor operation succeeds before any official experiment starts.

For the current Vast.ai machine, the safe verified choice is:

- GPU: Tesla V100-SXM2-32GB
- compute capability: `sm_70`
- PyTorch: `torch==2.8.0+cu126`
- PyTorch index: `https://download.pytorch.org/whl/cu126`
- JAX: `jax==0.10.0`, `jaxlib==0.10.0`, `jax-cuda12-plugin==0.10.0`
- cuDNN package observed working: `nvidia-cudnn-cu12==9.10.2.21`

## What Went Wrong Before

The broken environment used:

- PyTorch: `torch==2.11.0+cu128`
- GPU: Tesla V100-SXM2-32GB
- GPU compute capability: `sm_70`

That PyTorch wheel could see the GPU, so `torch.cuda.is_available()` returned
`True`. But the wheel's compiled CUDA architecture list was:

```text
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

It did not include `sm_70`. Therefore V100 CUDA kernels failed at runtime with:

```text
CUDA error: no kernel image is available for execution on the device
```

This is the key lesson: seeing the GPU is not enough. The wheel must include the
GPU's `sm_*` architecture.

## Why V100 Is Special Here

V100 is a Volta GPU with compute capability `7.0`, written as `sm_70`.
Some newer PyTorch wheels have started dropping older architecture support, or
at least the specific wheel installed here did not include `sm_70`.

That means V100 can be more fragile than A100/H100 when blindly installing the
newest PyTorch CUDA wheel.

Common examples:

| GPU family | Example GPU | Compute capability | Required arch in PyTorch wheel |
| --- | --- | --- | --- |
| Volta | V100 | 7.0 | `sm_70` |
| Turing | T4 | 7.5 | `sm_75` |
| Ampere | A100 | 8.0 | `sm_80` |
| Ampere | RTX 3090 / A10 | 8.6 | `sm_86` |
| Hopper | H100 | 9.0 | `sm_90` |

If the active GPU is V100, always check for `sm_70`. If it is A100, check for
`sm_80`. If it is H100, check for `sm_90`.

## CUDA Version Is Not The Same As GPU Architecture

`nvidia-smi` may show something like:

```text
Driver Version: 570.211.01     CUDA Version: 12.8
```

That means the driver can support CUDA applications up to that level. It does
not mean every `cu128` PyTorch wheel will support the installed GPU.

A `cu126` wheel can be perfectly valid on a machine whose driver reports CUDA
12.8, because the newer driver can run older CUDA 12.x runtime libraries. In
this project, `torch==2.8.0+cu126` works on the CUDA 12.8 driver and includes
`sm_70`.

## The JAX / cuDNN Trap

When PyTorch was first downgraded to `torch==2.6.0+cu126`, PyTorch itself
started working on V100 because the wheel included `sm_70`.

However, that install pulled:

```text
nvidia-cudnn-cu12==9.5.1.17
```

JAX `0.10.0` was compiled expecting cuDNN `9.8.0` or newer. As a result, JAX GPU
failed with a cuDNN mismatch, even though PyTorch CUDA worked.

The working fix was to use `torch==2.8.0+cu126`, which pulled:

```text
nvidia-cudnn-cu12==9.10.2.21
```

That version satisfied JAX and still kept PyTorch V100 support.

## Current Verified Environment

Observed after repair on 2026-05-16:

```text
Python: /workspace/NeuralOperatorRobustness2/adv_robust/bin/python
PyTorch: 2.8.0+cu126
PyTorch CUDA runtime: 12.6
GPU: Tesla V100-SXM2-32GB
compute capability: (7, 0)
PyTorch arch list: ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
JAX backend: gpu
JAX devices: [CudaDevice(id=0)]
```

Both PyTorch CUDA matmul and JAX GPU matmul succeeded in this environment.


## Automated Setup Entry Point

Use this script whenever `adv_robust` is missing, stale, or needs to be rebuilt:

```bash
cd /workspace/NeuralOperatorRobustness2
python3 tools/setup_adv_robust_gpu_env.py
```

The script does the following in order:

1. reads the visible GPU and compute capability from `nvidia-smi`;
2. creates `adv_robust/` with Python 3.12 if it does not already exist;
3. checks that `requirements.txt` still contains the verified `cu126` /
   `torch==2.8.0+cu126` pins;
4. installs the full requirements file;
5. verifies that PyTorch sees CUDA and that the wheel contains the required
   `sm_*` architecture for the active GPU;
6. runs a real PyTorch CUDA matmul and synchronizes;
7. verifies that JAX backend is `gpu`;
8. runs a real JAX GPU matmul and blocks until ready;
9. runs `pip check`;
10. exits nonzero if any GPU check fails.

For an existing environment, use the fast verification path:

```bash
tools/setup_adv_robust_gpu_env.py --verify-only
```

This script is now the project default. Manual `pip install -r requirements.txt`
is allowed only when debugging, and must be followed by the same verification
script before any official experiment run.

## Required Verification Commands

Run these before official experiments.

### 1. Hardware and driver

```bash
nvidia-smi
```

### 2. PyTorch GPU architecture and real CUDA operation

```bash
adv_robust/bin/python -c 'import torch, sys; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list()); x=torch.eye(32, device="cuda"); y=x@x; torch.cuda.synchronize(); print(float(y[0,0].cpu()))'
```

The required `sm_*` value must be present in `torch.cuda.get_arch_list()`.
For V100 that means `sm_70`.

### 3. JAX GPU backend and real GPU operation

```bash
adv_robust/bin/python -c 'import jax, jax.numpy as jnp; print(jax.default_backend()); print(jax.devices()); x=jnp.eye(16); y=x@x; y.block_until_ready(); print(float(y[0,0]))'
```

The backend must be `gpu`, not `cpu`.

### 4. Dependency health

```bash
adv_robust/bin/python -m pip check
```

## Install Guidance By GPU

### V100 / `sm_70`

Use the verified project combination unless there is a strong reason to change:

```bash
adv_robust/bin/python -m pip install --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu126 'torch==2.8.0+cu126'
```

Then run all verification commands above. Do not accept the install unless
`torch.cuda.get_arch_list()` includes `sm_70` and JAX GPU still works.

Avoid for official V100 runs:

```text
torch==2.11.0+cu128
```

because the observed wheel did not include `sm_70`.

### A100 / `sm_80`

A100 usually works with newer CUDA wheels because the observed newer wheel list
included `sm_80`. Still, do not assume. Check that `torch.cuda.get_arch_list()`
contains `sm_80`, and verify JAX GPU after any PyTorch/CUDA package change.

### H100 / `sm_90`

H100 needs `sm_90`. Newer CUDA wheels usually include this, but the same rule
applies: check the actual wheel arch list and run a real CUDA operation.

### Unknown GPU

First get the compute capability:

```bash
adv_robust/bin/python -c 'import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list())'
```

Convert capability `(major, minor)` to `sm_<major><minor>`. For example,
`(7, 0)` becomes `sm_70`. The wheel must include that architecture.

## Practical Checklist Before Running Experiments

1. Activate or call `adv_robust/bin/python` directly.
2. Run `nvidia-smi`.
3. Confirm the GPU model and compute capability.
4. Confirm `torch.cuda.get_arch_list()` contains the required `sm_*`.
5. Run a real PyTorch CUDA operation and synchronize.
6. Confirm JAX backend is `gpu` if JAX solver/model is involved.
7. Run a real JAX GPU operation and `block_until_ready()`.
8. Run `pip check`.
9. Only then start the experiment.
10. Record the GPU evidence in the experiment `manifest.json` and result note.

## Do Not Repeat These Mistakes

- Do not use `torch.cuda.is_available()` as the only GPU test.
- Do not assume `cu128` is better just because `nvidia-smi` says CUDA 12.8.
- Do not install the newest PyTorch wheel on V100 without checking `sm_70`.
- Do not fix PyTorch and forget to re-test JAX GPU.
- Do not let an experiment script silently choose CPU when CUDA fails.
- Do not report a CPU run as an official experiment result.

## Project-Level Persistence

This rule is persisted through:

- `AGENTS.md`, which tells future Codex sessions not to CPU fallback and to read
  this compatibility note before environment changes;
- `docs/gpu_only_experiment_policy_20260516.md`, which states the standing
  GPU-only experiment policy;
- this file, which records the detailed CUDA/PyTorch/JAX compatibility logic;
- `requirements.txt`, which currently pins the verified V100-compatible PyTorch
  wheel combination;
- experiment scripts such as `tools/run_loss3_small_epsilon_sweep.py`, which
  refuse to run official experiments if the GPU checks fail.
