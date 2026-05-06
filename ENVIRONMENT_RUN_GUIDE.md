# Current GPU Environment Run Guide

This repository was originally written for a Slurm HPC cluster, but the
current workspace is a direct GPU instance, such as a VastAI rental.

Current observed workspace:

- Project root: `/workspace/NeuralOperatorRobustness2`
- Runtime style: direct single-machine GPU execution
- Current visible GPU: `GPU 0: NVIDIA A100-SXM4-80GB`
- Slurm commands: not available here (`sbatch` / `srun` are not installed)
- Conda: not available here
- Local Python environment: `adv_robust/`
- Environment record: `requirements.txt`
- Exponax: installed from the modified GitHub fork pinned in `requirements.txt`

## Key Difference From Slurm

On the original cluster, scripts requested GPUs with lines like:

```bash
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
```

On this VastAI-style instance, the GPU has already been allocated to the
container. You do not request a GPU with Slurm. You just make the visible GPU
available to Python:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Then PyTorch or JAX can use it directly.

## Standard Command To Run A Python Job

Use this as the default starting template:

```bash
cd /workspace/NeuralOperatorRobustness2
source adv_robust/bin/activate

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

python path/to/your_script.py
```

For a PyTorch script, the code should choose CUDA with something like:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
```

For a JAX script, prefer this to avoid JAX preallocating most GPU memory:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python path/to/your_jax_script.py
```

## How To Check The GPU

Before running a long job:

```bash
nvidia-smi
```

Check PyTorch:

```bash
source adv_robust/bin/activate
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Check JAX:

```bash
source adv_robust/bin/activate
XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "import jax; print(jax.devices())"
```

## Running Old Shell Scripts

Do not run old scripts with `sbatch` in this environment.

Instead, if a script is otherwise valid:

```bash
bash path/to/script.sh
```

However, most existing `.sh` files still contain HPC assumptions that need
attention.

Problematic Slurm/HPC patterns:

```bash
#SBATCH --partition=...
#SBATCH --gres=gpu:b200:1
source ~/.bashrc
conda activate adv_robust
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}
export PYTHONPATH=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO:$PYTHONPATH
PROJECT_ROOT=/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO
```

Local replacement pattern:

```bash
cd /workspace/NeuralOperatorRobustness2
source adv_robust/bin/activate

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"
```

The `#SBATCH` lines are harmless if the script is run with `bash` because they
are comments. The real problems are usually:

- `conda activate adv_robust`, because conda is not installed here
- hard-coded `/blue/...` or `/home/...` paths
- missing `SLURM_*` environment variables
- paths pointing to data/model files that do not exist in this workspace

## If You Rent A Different GPU Later

Do not assume every future instance is A100. VastAI may give you A100, H100,
H200, B200, L40S, etc.

Always check:

```bash
nvidia-smi -L
nvidia-smi
```

Usually the command still stays:

```bash
export CUDA_VISIBLE_DEVICES=0
```

because a single-GPU container normally exposes the rented GPU as local GPU 0.
If a future instance exposes multiple GPUs, choose one with:

```bash
export CUDA_VISIBLE_DEVICES=0
```

or:

```bash
export CUDA_VISIBLE_DEVICES=1
```

Inside Python, the selected GPU will still appear as `cuda:0`.

## Important Current Repository Issues

The environment is now mostly ready for PyTorch/JAX work, but old experiments
may still not run immediately.

Known issues:

- Many scripts were written for Slurm and `/blue/...` cluster paths.
- Many shell scripts activate `conda activate adv_robust`; this must be changed
  to `source adv_robust/bin/activate`.
- This workspace currently does not contain the old `.pt` datasets or `.pth`
  model checkpoints referenced by several scripts.
- Exponax is installed from `YifeiSun01/modified_exponax`, pinned in
  `requirements.txt`.
- JAX can preallocate GPU memory unless `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  is set.

## Minimal Long-Running Command Example

For a PyTorch training script:

```bash
cd /workspace/NeuralOperatorRobustness2
source adv_robust/bin/activate
export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

python -u path/to/train_script.py 2>&1 | tee train.log
```

For a JAX or mixed JAX/PyTorch script:

```bash
cd /workspace/NeuralOperatorRobustness2
source adv_robust/bin/activate
export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

python -u path/to/script.py 2>&1 | tee run.log
```

## Environment Rebuild

The virtual environment directory `adv_robust/` should stay ignored by git.
Track `requirements.txt`, not the environment directory itself.

To rebuild:

```bash
cd /workspace/NeuralOperatorRobustness2
python3 -m venv adv_robust
source adv_robust/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
