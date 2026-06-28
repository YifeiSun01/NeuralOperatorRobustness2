# Performance Monitoring Guide

This repository has partial timing logs, but it does not currently have one
consistent monitoring system for all training, evaluation, and attack jobs.

## Current Audit Summary

What exists now:

- Many training scripts record only epoch-level wall time, for example
  `time taken` per epoch.
- Some evaluation scripts write CSV rows with total seconds per dataset/model.
- Several PGD attack scripts print detailed forward/PDE/backward/update timing.
- Some scripts print `nvidia-smi` once at startup.
- Some scripts print `torch.cuda.memory_allocated()` or peak allocator memory.

What is missing:

- No consistent per-job GPU memory sampling every N seconds.
- No consistent CSV/JSONL record of GPU processes and memory usage.
- No consistent named-stage timing across training/evaluation/attack scripts.
- No reliable distinction between PyTorch allocator memory and total GPU memory.
- No centralized helper, so each script implements logging differently.

## Important Accuracy Notes

PyTorch CUDA operations are asynchronous. For accurate wall-clock timing around
GPU work, call:

```python
torch.cuda.synchronize()
```

before starting and after finishing the timed block.

PyTorch memory APIs measure the PyTorch caching allocator:

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
```

These are useful, but they are not the same as total GPU memory shown by
`nvidia-smi`. JAX, CUDA context memory, and other processes can use memory that
PyTorch does not report.

For whole-GPU memory and process-level memory, use `nvidia-smi` sampling.

## Whole Job Monitoring

Use the wrapper:

```bash
bash tools/run_with_gpu_monitor.sh python -u path/to/train_script.py
```

This creates a run directory under:

```text
run_logs/YYYYMMDD_HHMMSS/
```

Files created:

- `run.log`: stdout/stderr from the command
- `run_metadata.txt`: command, GPU, start/end metadata
- `gpu_samples.csv`: periodic whole-GPU utilization and memory samples
- `gpu_processes.csv`: periodic GPU process memory samples

Useful options:

```bash
GPU_MONITOR_INTERVAL=2 bash tools/run_with_gpu_monitor.sh python -u path/to/train.py
RUN_NAME=my_experiment bash tools/run_with_gpu_monitor.sh python -u path/to/train.py
LOG_ROOT=/workspace/logs bash tools/run_with_gpu_monitor.sh python -u path/to/train.py
```

The wrapper automatically sets:

```bash
source adv_robust/bin/activate
export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
```

## Named Stage Monitoring Inside Python

Use `tools.runtime_monitor.stage` to log key sections:

```python
from tools.runtime_monitor import stage

stage_log = "run_logs/my_run/stages.jsonl"

with stage("load_data", stage_log):
    data = torch.load(path, map_location=device)

with stage("train_epoch_000", stage_log):
    train_one_epoch()

with stage("eval_epoch_000", stage_log):
    evaluate()
```

Each JSONL row includes:

- stage name
- start/end UTC timestamps
- seconds
- process RSS memory
- PyTorch allocated/reserved/max CUDA memory
- deltas from stage start to end

For loops:

```python
for ep in range(epochs):
    with stage(f"epoch_{ep:04d}_train", stage_log):
        train_one_epoch()
    with stage(f"epoch_{ep:04d}_eval", stage_log):
        evaluate()
```

## Recommended Monitoring Levels

For every long job:

```bash
bash tools/run_with_gpu_monitor.sh python -u your_script.py
```

For important scripts, also add `stage(...)` around:

- data loading
- model construction
- one full training epoch
- evaluation
- PDE solver rollout
- forward pass
- loss computation
- backward pass
- optimizer update
- checkpoint save

## Current Code Risks Found

The existing logs are useful but not enough for detailed performance diagnosis:

- Basic training scripts only log epoch time and metrics, not GPU memory.
- Some timing uses `default_timer()` without explicit CUDA synchronization.
  Calls like `.item()` often synchronize indirectly, but this is not a clean
  timing boundary.
- Several memory logs use `memory_allocated()` differences. This can be
  negative or misleading because PyTorch reuses cached memory.
- Some code prints peak memory but does not save it to a structured file.
- `gpu_choose.py` estimates free memory as `reserved - allocated`, which is not
  true free GPU memory. Use `torch.cuda.mem_get_info()` or `nvidia-smi` instead.
- JAX memory is not reflected in PyTorch allocator stats. Use `nvidia-smi`
  sampling when JAX is involved.

## Best Current Command Template

For a PyTorch task:

```bash
cd /workspace/NeuralOperatorRobustness2
GPU_MONITOR_INTERVAL=2 RUN_NAME=my_run \
  bash tools/run_with_gpu_monitor.sh python -u path/to/script.py
```

For a JAX or mixed JAX/PyTorch task:

```bash
cd /workspace/NeuralOperatorRobustness2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
GPU_MONITOR_INTERVAL=2 RUN_NAME=my_jax_run \
  bash tools/run_with_gpu_monitor.sh python -u path/to/script.py
```
