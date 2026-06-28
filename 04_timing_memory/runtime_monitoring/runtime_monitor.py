"""Small runtime instrumentation helpers for PyTorch/JAX experiments.

Use the shell wrapper `tools/run_with_gpu_monitor.sh` for whole-job GPU
sampling. Use this module inside Python code when you need named stage timing
and PyTorch CUDA memory snapshots.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional

try:
    import psutil
except Exception:  # pragma: no cover - optional at runtime
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - optional at runtime
    torch = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def synchronize_cuda() -> None:
    if _cuda_available():
        torch.cuda.synchronize()


def memory_snapshot(device: Optional[int] = None) -> Dict[str, object]:
    """Return process RSS and PyTorch CUDA allocator stats.

    PyTorch allocator stats are not the same as total GPU memory shown by
    nvidia-smi. Use the shell wrapper for whole-GPU and per-process sampling.
    """

    snapshot: Dict[str, object] = {}

    if psutil is not None:
        proc = psutil.Process(os.getpid())
        snapshot["process_rss_mib"] = proc.memory_info().rss / 1024**2

    if not _cuda_available():
        snapshot["cuda_available"] = False
        return snapshot

    if device is None:
        device = torch.cuda.current_device()

    snapshot.update(
        {
            "cuda_available": True,
            "cuda_device": int(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "torch_allocated_mib": torch.cuda.memory_allocated(device) / 1024**2,
            "torch_reserved_mib": torch.cuda.memory_reserved(device) / 1024**2,
            "torch_max_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "torch_max_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        }
    )

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        snapshot["cuda_free_mib"] = free_bytes / 1024**2
        snapshot["cuda_total_mib"] = total_bytes / 1024**2
    except Exception:
        pass

    return snapshot


def append_jsonl(path: os.PathLike[str] | str, row: Dict[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


@contextmanager
def stage(
    name: str,
    log_path: os.PathLike[str] | str = "run_logs/stages.jsonl",
    *,
    device: Optional[int] = None,
    reset_peak: bool = True,
    sync_cuda: bool = True,
) -> Iterator[None]:
    """Record wall time and PyTorch CUDA memory for a named code block.

    Example:

        from tools.runtime_monitor import stage

        with stage("train_epoch", "run_logs/stages.jsonl"):
            train_one_epoch()
    """

    if sync_cuda:
        synchronize_cuda()
    if reset_peak and _cuda_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_wall = time.perf_counter()
    start_ts = _utc_now()
    start_mem = memory_snapshot(device)

    exc_repr: Optional[str] = None
    try:
        yield
    except Exception as exc:
        exc_repr = repr(exc)
        raise
    finally:
        if sync_cuda:
            synchronize_cuda()
        end_wall = time.perf_counter()
        end_mem = memory_snapshot(device)

        row: Dict[str, object] = {
            "name": name,
            "pid": os.getpid(),
            "start_time_utc": start_ts,
            "end_time_utc": _utc_now(),
            "seconds": end_wall - start_wall,
            "start": start_mem,
            "end": end_mem,
        }

        for key in (
            "torch_allocated_mib",
            "torch_reserved_mib",
            "process_rss_mib",
        ):
            if key in start_mem and key in end_mem:
                row[f"delta_{key}"] = end_mem[key] - start_mem[key]

        if exc_repr is not None:
            row["exception"] = exc_repr

        append_jsonl(log_path, row)
