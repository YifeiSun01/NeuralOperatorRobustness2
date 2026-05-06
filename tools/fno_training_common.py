"""Shared helpers for FNO training scripts."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def append_csv_row(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def _parse_first_number(value: str) -> float:
    value = value.strip()
    token = value.split()[0] if value else ""
    return _float_or_nan(token)


def get_gpu_memory_snapshot(pid: int | None = None) -> dict[str, float | int | str]:
    """Read coarse process/GPU memory from nvidia-smi.

    PyTorch has allocator-level APIs; JAX generally does not expose matching
    per-step counters, so this nvidia-smi snapshot gives both frameworks a
    common external memory signal.
    """

    pid = os.getpid() if pid is None else pid
    snapshot: dict[str, float | int | str] = {
        "pid": pid,
        "process_gpu_memory_mib": float("nan"),
        "gpu_memory_used_mib": float("nan"),
        "gpu_util_percent": float("nan"),
        "gpu_power_w": float("nan"),
        "gpu_temp_c": float("nan"),
        "memory_source": "nvidia-smi",
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            total = 0.0
            found = False
            for line in proc.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 2:
                    continue
                if parts[0] == str(pid):
                    total += _parse_first_number(parts[1])
                    found = True
            if found:
                snapshot["process_gpu_memory_mib"] = total

        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if gpu.returncode == 0:
            used_values = []
            util_values = []
            power_values = []
            temp_values = []
            for line in gpu.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 1:
                    used_values.append(_parse_first_number(parts[0]))
                if len(parts) >= 2:
                    util_values.append(_parse_first_number(parts[1]))
                if len(parts) >= 3:
                    power_values.append(_parse_first_number(parts[2]))
                if len(parts) >= 4:
                    temp_values.append(_parse_first_number(parts[3]))
            if used_values:
                snapshot["gpu_memory_used_mib"] = float(np.nansum(used_values))
            if util_values:
                snapshot["gpu_util_percent"] = float(np.nanmax(util_values))
            if power_values:
                snapshot["gpu_power_w"] = float(np.nansum(power_values))
            if temp_values:
                snapshot["gpu_temp_c"] = float(np.nanmax(temp_values))
    except (FileNotFoundError, subprocess.SubprocessError, TimeoutError):
        snapshot["memory_source"] = "unavailable"
    return snapshot


def synchronize_torch(device: torch.device | None = None) -> None:
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)


def reset_torch_peak_memory(device: torch.device | None = None) -> None:
    if torch.cuda.is_available() and (device is None or device.type == "cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def torch_memory_stats(device: torch.device | None = None) -> dict[str, float | str]:
    if not torch.cuda.is_available() or (device is not None and device.type != "cuda"):
        return {
            "torch_allocated_mib": float("nan"),
            "torch_reserved_mib": float("nan"),
            "torch_peak_allocated_mib": float("nan"),
            "torch_peak_reserved_mib": float("nan"),
            "memory_backend": "cpu",
        }
    return {
        "torch_allocated_mib": _bytes_to_mib(torch.cuda.memory_allocated(device)),
        "torch_reserved_mib": _bytes_to_mib(torch.cuda.memory_reserved(device)),
        "torch_peak_allocated_mib": _bytes_to_mib(torch.cuda.max_memory_allocated(device)),
        "torch_peak_reserved_mib": _bytes_to_mib(torch.cuda.max_memory_reserved(device)),
        "memory_backend": "torch.cuda",
    }


def should_profile_batch(args: Any, batch_idx: int, epoch: int | None = None) -> bool:
    if not (getattr(args, "memory_profile", True) or getattr(args, "time_profile", True)):
        return False
    epoch_limit = int(getattr(args, "memory_profile_detailed_epochs", -1))
    if epoch is not None and epoch_limit >= 0 and int(epoch) > epoch_limit:
        return False
    limit = int(getattr(args, "memory_profile_detailed_batches", 5))
    return limit < 0 or batch_idx < limit


def should_write_profile_outputs(args: Any) -> bool:
    return bool(getattr(args, "memory_profile", True) or getattr(args, "time_profile", True))


def empty_profile_summary() -> dict[str, Any]:
    return {
        "profile_outputs_enabled": False,
        "summary_json": None,
        "samples_csv": None,
        "phases_csv": None,
        "samples_plot": None,
        "phases_plot": None,
        "sampled_whole_run_peak_gpu_memory_used_mib": float("nan"),
        "sampled_whole_run_seconds": float("nan"),
        "profiled_train_peak_mib": float("nan"),
        "profiled_inference_peak_mib": float("nan"),
        "profiled_backward_to_forward_peak_ratio": float("nan"),
        "profiled_backward_to_forward_mean_time_ratio": float("nan"),
        "phases": {},
    }


def make_memory_phase_row(
    *,
    problem: str,
    framework: str,
    scope: str,
    phase: str,
    epoch: int | str,
    batch: int | str,
    batch_size: int,
    start_time: float,
    end_time: float,
    extra: dict[str, Any] | None = None,
    include_nvidia_smi: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "problem": problem,
        "framework": framework,
        "scope": scope,
        "phase": phase,
        "epoch": epoch,
        "batch": batch,
        "batch_size": batch_size,
        "timestamp": time.time(),
        "phase_seconds": end_time - start_time,
    }
    if include_nvidia_smi:
        row.update(get_gpu_memory_snapshot())
    if extra:
        row.update(extra)
    return row


class GpuMemorySampler:
    def __init__(
        self,
        *,
        problem: str,
        framework: str,
        scope: str,
        interval_seconds: float,
        enabled: bool = True,
    ):
        self.problem = problem
        self.framework = framework
        self.scope = scope
        self.interval_seconds = max(float(interval_seconds), 0.05)
        self.enabled = enabled
        self.rows: list[dict[str, Any]] = []
        self._start_monotonic = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if not self.enabled:
            return self
        self._start_monotonic = time.perf_counter()
        self.sample("start")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2.0))
        self.sample("end")
        return False

    def start(self):
        return self.__enter__()

    def stop(self):
        return self.__exit__(None, None, None)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample("interval")

    def sample(self, tag: str) -> None:
        now = time.perf_counter()
        row: dict[str, Any] = {
            "problem": self.problem,
            "framework": self.framework,
            "scope": self.scope,
            "tag": tag,
            "timestamp": time.time(),
            "elapsed_seconds": now - self._start_monotonic if self._start_monotonic else 0.0,
        }
        row.update(get_gpu_memory_snapshot())
        self.rows.append(row)


def summarize_memory_rows(sample_rows: list[dict], phase_rows: list[dict]) -> dict[str, Any]:
    def valid_values(rows: list[dict], key: str) -> list[float]:
        values = [_float_or_nan(row.get(key)) for row in rows]
        return [value for value in values if not np.isnan(value)]

    def memory_values(rows: list[dict]) -> list[float]:
        values = []
        for row in rows:
            for key in ("torch_peak_allocated_mib", "process_gpu_memory_mib", "gpu_memory_used_mib", "torch_reserved_mib"):
                value = _float_or_nan(row.get(key))
                if not np.isnan(value):
                    values.append(value)
                    break
        return values

    summary: dict[str, Any] = {
        "num_sample_rows": len(sample_rows),
        "num_phase_rows": len(phase_rows),
        "max_process_gpu_memory_mib": float("nan"),
        "max_gpu_memory_used_mib": float("nan"),
        "sampled_whole_run_peak_process_gpu_memory_mib": float("nan"),
        "sampled_whole_run_peak_gpu_memory_used_mib": float("nan"),
        "sampled_whole_run_seconds": float("nan"),
        "profiled_train_peak_mib": float("nan"),
        "profiled_inference_peak_mib": float("nan"),
        "profiled_backward_to_forward_peak_ratio": float("nan"),
        "profiled_backward_to_forward_mean_time_ratio": float("nan"),
        "scopes": {},
        "phases": {},
    }
    if sample_rows:
        process_values = valid_values(sample_rows, "process_gpu_memory_mib")
        gpu_values = valid_values(sample_rows, "gpu_memory_used_mib")
        elapsed_values = valid_values(sample_rows, "elapsed_seconds")
        if process_values:
            summary["max_process_gpu_memory_mib"] = float(np.nanmax(process_values))
            summary["sampled_whole_run_peak_process_gpu_memory_mib"] = summary["max_process_gpu_memory_mib"]
        if gpu_values:
            summary["max_gpu_memory_used_mib"] = float(np.nanmax(gpu_values))
            summary["sampled_whole_run_peak_gpu_memory_used_mib"] = summary["max_gpu_memory_used_mib"]
        if elapsed_values:
            summary["sampled_whole_run_seconds"] = float(np.nanmax(elapsed_values))

    scopes = sorted({str(row.get("scope")) for row in phase_rows})
    for scope in scopes:
        rows = [row for row in phase_rows if str(row.get("scope")) == scope]
        mem = memory_values(rows)
        seconds = valid_values(rows, "phase_seconds")
        summary["scopes"][scope] = {
            "num_rows": len(rows),
            "max_memory_mib": float(np.nanmax(mem)) if mem else float("nan"),
            "mean_seconds": float(np.nanmean(seconds)) if seconds else float("nan"),
            "max_seconds": float(np.nanmax(seconds)) if seconds else float("nan"),
            "total_seconds": float(np.nansum(seconds)) if seconds else float("nan"),
        }
    if "train" in summary["scopes"]:
        summary["profiled_train_peak_mib"] = summary["scopes"]["train"]["max_memory_mib"]
    if "inference" in summary["scopes"]:
        summary["profiled_inference_peak_mib"] = summary["scopes"]["inference"]["max_memory_mib"]

    phases = sorted({str(row.get("phase")) for row in phase_rows})
    for phase in phases:
        rows = [row for row in phase_rows if str(row.get("phase")) == phase]
        peak_candidates = memory_values(rows)
        seconds = valid_values(rows, "phase_seconds")
        summary["phases"][phase] = {
            "num_rows": len(rows),
            "max_memory_mib": float(np.nanmax(peak_candidates)) if peak_candidates else float("nan"),
            "mean_memory_mib": float(np.nanmean(peak_candidates)) if peak_candidates else float("nan"),
            "mean_seconds": float(np.nanmean(seconds)) if seconds else float("nan"),
            "max_seconds": float(np.nanmax(seconds)) if seconds else float("nan"),
            "min_seconds": float(np.nanmin(seconds)) if seconds else float("nan"),
            "total_seconds": float(np.nansum(seconds)) if seconds else float("nan"),
        }
    forward = summary["phases"].get("forward", {})
    backward = summary["phases"].get("backward", {})
    forward_peak = _float_or_nan(forward.get("max_memory_mib"))
    backward_peak = _float_or_nan(backward.get("max_memory_mib"))
    forward_time = _float_or_nan(forward.get("mean_seconds"))
    backward_time = _float_or_nan(backward.get("mean_seconds"))
    if not np.isnan(forward_peak) and forward_peak > 0 and not np.isnan(backward_peak):
        summary["profiled_backward_to_forward_peak_ratio"] = float(backward_peak / forward_peak)
    if not np.isnan(forward_time) and forward_time > 0 and not np.isnan(backward_time):
        summary["profiled_backward_to_forward_mean_time_ratio"] = float(backward_time / forward_time)
    return summary


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def array_metrics(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b
    abs_diff = np.abs(diff)
    mse = float(np.mean(abs_diff * abs_diff))
    denom = max(float(np.linalg.norm(b.reshape(-1))), eps)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(abs_diff)),
        "max_abs": float(np.max(abs_diff)),
        "relative_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
    }


def per_sample_prediction_rows(
    pytorch_pred: np.ndarray,
    jax_pred: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, float]]:
    rows = []
    for i in range(int(pytorch_pred.shape[0])):
        pt_vs_jax = array_metrics(pytorch_pred[i], jax_pred[i])
        pt_vs_target = array_metrics(pytorch_pred[i], target[i])
        jax_vs_target = array_metrics(jax_pred[i], target[i])
        rows.append(
            {
                "sample": i,
                "pytorch_vs_jax_mse": pt_vs_jax["mse"],
                "pytorch_vs_jax_rmse": pt_vs_jax["rmse"],
                "pytorch_vs_jax_mae": pt_vs_jax["mae"],
                "pytorch_vs_jax_max_abs": pt_vs_jax["max_abs"],
                "pytorch_vs_jax_relative_l2": pt_vs_jax["relative_l2"],
                "pytorch_vs_target_relative_l2": pt_vs_target["relative_l2"],
                "jax_vs_target_relative_l2": jax_vs_target["relative_l2"],
            }
        )
    return rows


def compare_named_arrays(
    pytorch_params: dict[str, np.ndarray],
    jax_params: dict[str, np.ndarray],
) -> tuple[list[dict], dict[str, float]]:
    rows = []
    total_sq = 0.0
    total_ref_sq = 0.0
    total_count = 0
    max_abs = 0.0
    for name in sorted(set(pytorch_params) | set(jax_params)):
        if name not in pytorch_params:
            rows.append({"name": name, "status": "missing_pytorch"})
            continue
        if name not in jax_params:
            rows.append({"name": name, "status": "missing_jax"})
            continue

        left = np.asarray(pytorch_params[name])
        right = np.asarray(jax_params[name])
        if left.shape != right.shape:
            rows.append(
                {
                    "name": name,
                    "status": "shape_mismatch",
                    "pytorch_shape": list(left.shape),
                    "jax_shape": list(right.shape),
                }
            )
            continue

        metrics = array_metrics(left, right)
        diff = left - right
        total_sq += float(np.sum(np.abs(diff) ** 2))
        total_ref_sq += float(np.sum(np.abs(right) ** 2))
        total_count += int(left.size)
        max_abs = max(max_abs, metrics["max_abs"])
        rows.append(
            {
                "name": name,
                "status": "ok",
                "shape": list(left.shape),
                **metrics,
            }
        )

    summary = {
        "num_parameter_blocks": len(rows),
        "num_compared_blocks": sum(1 for row in rows if row.get("status") == "ok"),
        "num_values_compared": total_count,
        "max_abs": max_abs,
        "relative_l2": float(np.sqrt(total_sq) / max(np.sqrt(total_ref_sq), 1e-12)),
    }
    return rows, summary


def torch_relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    diff_norm = torch.linalg.vector_norm(pred_flat - target_flat, dim=1)
    target_norm = torch.linalg.vector_norm(target_flat, dim=1).clamp_min(eps)
    return (diff_norm / target_norm).mean()


def torch_batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = pred - target
    return {
        "mse": float(torch.mean(diff * diff).detach().cpu()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).detach().cpu()),
        "mae": float(torch.mean(torch.abs(diff)).detach().cpu()),
        "relative_l2": float(torch_relative_l2(pred, target).detach().cpu()),
    }


def combine_weighted_metrics(parts: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    total = sum(n for n, _ in parts)
    if total == 0:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "relative_l2": float("nan")}
    mse = sum(n * m["mse"] for n, m in parts) / total
    mae = sum(n * m["mae"] for n, m in parts) / total
    rel = sum(n * m["relative_l2"] for n, m in parts) / total
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "relative_l2": rel,
    }


def plot_loss_curves(loss_rows: list[dict], output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    if not loss_rows:
        return
    epochs = [int(r["epoch"]) for r in loss_rows]
    train = [float(r["train_relative_l2"]) for r in loss_rows]
    test = [float(r["test_relative_l2"]) for r in loss_rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train, label="train relative L2")
    plt.plot(epochs, test, label="test relative L2")
    plt.xlabel("epoch")
    plt.ylabel("relative L2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_framework_loss_comparison(
    rows_by_framework: dict[str, list[dict]],
    output_path: Path,
    title: str,
) -> None:
    ensure_dir(output_path.parent)
    if not rows_by_framework:
        return
    plt.figure(figsize=(7.5, 4.8))
    plotted = False
    for framework, rows in rows_by_framework.items():
        if not rows:
            continue
        epochs = [int(r["epoch"]) for r in rows]
        train = [float(r["train_relative_l2"]) for r in rows]
        test = [float(r["test_relative_l2"]) for r in rows]
        plt.plot(epochs, train, label=f"{framework} train")
        plt.plot(epochs, test, linestyle="--", label=f"{framework} test")
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("epoch")
    plt.ylabel("relative L2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_parameter_block_differences(rows: list[dict], output_path: Path, title: str, max_bars: int = 40) -> None:
    ensure_dir(output_path.parent)
    valid_rows = [row for row in rows if row.get("status") == "ok"]
    if not valid_rows:
        return
    valid_rows = sorted(valid_rows, key=lambda row: float(row["max_abs"]), reverse=True)[:max_bars]
    labels = [str(row["name"]) for row in valid_rows]
    values = [float(row["max_abs"]) for row in valid_rows]
    plt.figure(figsize=(10, max(4.5, 0.22 * len(labels))))
    plt.barh(range(len(labels)), values)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.xlabel("max abs difference")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_gpu_memory_samples(rows: list[dict], output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    if not rows:
        return
    x = [_float_or_nan(row.get("elapsed_seconds")) for row in rows]
    process_mem = [_float_or_nan(row.get("process_gpu_memory_mib")) for row in rows]
    gpu_mem = [_float_or_nan(row.get("gpu_memory_used_mib")) for row in rows]
    plt.figure(figsize=(8, 4.6))
    if not np.all(np.isnan(process_mem)):
        plt.plot(x, process_mem, label="process GPU memory MiB")
    if not np.all(np.isnan(gpu_mem)):
        plt.plot(x, gpu_mem, alpha=0.55, label="total visible GPU memory MiB")
    plt.xlabel("elapsed seconds")
    plt.ylabel("memory MiB")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_memory_phase_peaks(rows: list[dict], output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    if not rows:
        return
    phase_order = ["forward", "backward", "optimizer_step", "train_step", "inference_forward"]
    phases = [phase for phase in phase_order if any(row.get("phase") == phase for row in rows)]
    phases += sorted({str(row.get("phase")) for row in rows if str(row.get("phase")) not in phases})
    values = []
    for phase in phases:
        phase_values = []
        for row in rows:
            if str(row.get("phase")) != phase:
                continue
            for key in ("torch_peak_allocated_mib", "process_gpu_memory_mib", "gpu_memory_used_mib", "torch_reserved_mib"):
                value = _float_or_nan(row.get(key))
                if not np.isnan(value):
                    phase_values.append(value)
                    break
        values.append(float(np.nanmax(phase_values)) if phase_values else float("nan"))
    if not values or np.all(np.isnan(values)):
        return
    plt.figure(figsize=(8, 4.6))
    plt.bar(phases, values)
    plt.ylabel("peak memory MiB")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_memory_profile_outputs(
    output_dir: Path,
    *,
    problem: str,
    framework: str,
    sample_rows: list[dict],
    phase_rows: list[dict],
) -> dict[str, Any]:
    memory_dir = ensure_dir(output_dir / "memory")
    samples_csv = memory_dir / f"gpu_samples_{framework}.csv"
    phases_csv = memory_dir / f"memory_phases_{framework}.csv"
    summary_json = memory_dir / f"memory_summary_{framework}.json"
    samples_plot = memory_dir / f"gpu_memory_curve_{framework}.png"
    phases_plot = memory_dir / f"phase_memory_peaks_{framework}.png"

    write_csv_rows(samples_csv, sample_rows)
    write_csv_rows(phases_csv, phase_rows)
    plot_gpu_memory_samples(sample_rows, samples_plot, f"{problem} {framework} GPU memory")
    plot_memory_phase_peaks(phase_rows, phases_plot, f"{problem} {framework} phase memory")
    summary = summarize_memory_rows(sample_rows, phase_rows)

    def existing_path(path: Path) -> str | None:
        return str(path) if path.exists() else None

    summary.update(
        {
            "profile_outputs_enabled": True,
            "samples_csv": existing_path(samples_csv),
            "phases_csv": existing_path(phases_csv),
            "samples_plot": existing_path(samples_plot),
            "phases_plot": existing_path(phases_plot),
        }
    )
    write_json(summary_json, summary)
    summary["summary_json"] = str(summary_json)
    return summary


def plot_1d_prediction_comparison(
    inputs: np.ndarray,
    target: np.ndarray,
    pytorch_pred: np.ndarray,
    jax_pred: np.ndarray,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    n = int(min(inputs.shape[0], target.shape[0], pytorch_pred.shape[0], jax_pred.shape[0]))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(9, max(3.0, 2.1 * n)), squeeze=False)
    x_grid = np.arange(target.shape[1])
    for i in range(n):
        ax = axes[i, 0]
        ax.plot(x_grid, inputs[i, :, 0], color="0.65", linewidth=1.0, label="input")
        ax.plot(x_grid, target[i, :, 0], color="black", linewidth=1.4, label="target")
        ax.plot(x_grid, pytorch_pred[i, :, 0], color="#1f77b4", linewidth=1.1, label="PyTorch")
        ax.plot(x_grid, jax_pred[i, :, 0], color="#d62728", linewidth=1.1, linestyle="--", label="JAX")
        ax.set_title(f"sample {i}")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_1d_prediction_multi_comparison(
    inputs: np.ndarray,
    target: np.ndarray,
    predictions_by_framework: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    if not predictions_by_framework:
        return
    n = int(min([inputs.shape[0], target.shape[0], *(pred.shape[0] for pred in predictions_by_framework.values())]))
    if n == 0:
        return
    colors = {
        "pytorch": "#1f77b4",
        "jax_complex": "#d62728",
        "jax_real_imag": "#2ca02c",
        "jax_conjugate_grad": "#9467bd",
    }
    linestyles = {
        "pytorch": "-",
        "jax_complex": "--",
        "jax_real_imag": "-.",
        "jax_conjugate_grad": ":",
    }
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3.2, 2.35 * n)), squeeze=False)
    x_grid = np.arange(target.shape[1])
    for i in range(n):
        ax = axes[i, 0]
        ax.plot(x_grid, inputs[i, :, 0], color="0.68", linewidth=1.0, label="input")
        ax.plot(x_grid, target[i, :, 0], color="black", linewidth=1.5, label="target")
        for framework, pred in predictions_by_framework.items():
            ax.plot(
                x_grid,
                pred[i, :, 0],
                color=colors.get(framework),
                linewidth=1.1,
                linestyle=linestyles.get(framework, "-"),
                label=framework,
            )
        ax.set_title(f"sample {i}")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_2d_prediction_comparison(
    target: np.ndarray,
    pytorch_pred: np.ndarray,
    jax_pred: np.ndarray,
    output_path: Path,
    *,
    frame_index: int = -1,
) -> None:
    ensure_dir(output_path.parent)
    n = int(min(target.shape[0], pytorch_pred.shape[0], jax_pred.shape[0]))
    if n == 0:
        return
    frame_index = frame_index if frame_index >= 0 else target.shape[-1] + frame_index
    fig, axes = plt.subplots(n, 4, figsize=(12, max(3.0, 2.6 * n)), squeeze=False)
    for i in range(n):
        arrays = [
            target[i, :, :, frame_index],
            pytorch_pred[i, :, :, frame_index],
            jax_pred[i, :, :, frame_index],
            np.abs(pytorch_pred[i, :, :, frame_index] - jax_pred[i, :, :, frame_index]),
        ]
        titles = ["target", "PyTorch", "JAX", "|PyTorch-JAX|"]
        for j, (arr, subtitle) in enumerate(zip(arrays, titles)):
            ax = axes[i, j]
            cmap = "magma" if j == 3 else "viridis"
            im = ax.imshow(arr, cmap=cmap, origin="lower")
            ax.set_title(f"sample {i} {subtitle}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
