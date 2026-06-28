# Installation Guide

This code was organized for a Python research environment named `adv_robust`.
The exact package versions can be installed from the original project
requirements file when available. This branch keeps only source code and
documentation, so large environment caches and generated artifacts are not
included.

## 1. Create An Environment

PowerShell:

```powershell
cd "D:\research\UIUC research\paper\paper_branch_exports_20260628_154456\paper-code-only"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Conda alternative:

```powershell
conda create -n adv_robust python=3.12 -y
conda activate adv_robust
python -m pip install --upgrade pip setuptools wheel
```

## 2. Install Dependencies

If you are using this branch inside the original full repository, install the
original dependency file:

```powershell
python -m pip install -r requirements.txt
```

If the dependency file is not present in this code-only branch, install the
package families needed by the script group you plan to run:

```powershell
python -m pip install numpy scipy pandas matplotlib seaborn tqdm h5py pyyaml
python -m pip install torch torchvision torchaudio
python -m pip install jax jaxlib optax equinox
```

GPU builds of `torch`, `jax`, and `jaxlib` depend on the CUDA version of the
machine. Install the CUDA-specific wheels that match the target server before
running large Navier-Stokes or adversarial-training jobs.

## 3. Verify The Environment

The helper below checks the expected packages and prints environment details:

```powershell
python 00_shared/reproducibility/setup_adv_robust_gpu_env.py --verify-only
```

If you want the helper to create or update a virtual environment, inspect its
available options first:

```powershell
python 00_shared/reproducibility/setup_adv_robust_gpu_env.py --help
```

## 4. Expected External Files

Most experiment scripts require files that are intentionally not stored in this
branch:

- training datasets, usually `.h5`, `.npz`, `.pt`, or `.pkl`;
- trained FNO/DeepONet checkpoints;
- saved attack trajectories and intermediate summaries;
- generated paper figures and tables.

Set these locations through command-line arguments such as `--output_dir`,
`--burgers-test-path`, `--burgers-torch-checkpoint`, `--ns-test-path`, and
`--ns-torch-checkpoint`. See `CONFIGURATION_GUIDE.md` for the common knobs.

## 5. Running On A Cluster

Use the same command structure as the local examples, but set:

- `CUDA_VISIBLE_DEVICES` or the scheduler GPU request before launching Python;
- an output directory on a scratch or project filesystem;
- a smaller `--steps`, sample count, or dataset count for smoke tests;
- `--seed` for reproducibility when supported.

Large adversarial-training and Navier-Stokes jobs should be launched through the
cluster scheduler rather than an interactive shell.

