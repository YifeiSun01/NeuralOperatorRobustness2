#!/usr/bin/env python3
"""Compare local PyTorch spectral solvers against modified Exponax/JAX solvers.

The script runs each solver in a separate worker process so GPU memory peaks are
not polluted by allocations from the previous framework. It records:

- output agreement between PyTorch and Exponax/JAX for the same initial states;
- cold first-run time, including JAX compilation when applicable;
- warm second-run time, after compilation/warmup;
- sampled process GPU memory peak from nvidia-smi;
- PyTorch CUDA allocator peak when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "benchmark_results" / "torch_vs_exponax_solver_comparison.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "benchmark_results" / "torch_vs_exponax_solver_comparison.md"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "benchmark_results" / "torch_vs_exponax_solver_comparison.csv"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "benchmark_results" / "torch_vs_exponax_solver_visualizations"


class NvidiaProcessMemorySampler:
    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = float(interval_seconds)
        self.pid = os.getpid()
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        global_used_mib = float("nan")
        try:
            global_out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            global_values = [float(line.strip()) for line in global_out.splitlines() if line.strip()]
            if global_values:
                global_used_mib = float(global_values[0])
        except Exception:
            pass

        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return

        used_mib = 0.0
        for line in out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                mem = float(parts[1])
            except ValueError:
                continue
            if pid == self.pid:
                used_mib = max(used_mib, mem)
        self.rows.append(
            {
                "elapsed_seconds": time.perf_counter() - self._start,
                "process_gpu_memory_mib": used_mib,
                "global_gpu_memory_used_mib": global_used_mib,
            }
        )

    def start(self) -> None:
        self._start = time.perf_counter()

        def loop() -> None:
            while not self._stop.is_set():
                self._sample_once()
                time.sleep(self.interval_seconds)
            self._sample_once()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        values = [float(row["process_gpu_memory_mib"]) for row in self.rows]
        global_values = [
            float(row["global_gpu_memory_used_mib"])
            for row in self.rows
            if not math.isnan(float(row["global_gpu_memory_used_mib"]))
        ]
        global_start = global_values[0] if global_values else float("nan")
        global_peak = float(max(global_values)) if global_values else float("nan")
        return {
            "sample_count": len(self.rows),
            "sampled_process_gpu_peak_mib": float(max(values)) if values else float("nan"),
            "sampled_global_gpu_start_mib": global_start,
            "sampled_global_gpu_peak_mib": global_peak,
            "sampled_global_gpu_peak_delta_mib": (
                float(global_peak - global_start)
                if not math.isnan(global_start) and not math.isnan(global_peak)
                else float("nan")
            ),
        }


def smooth_burgers_initial(batch: int, nx: int, domain_extent: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, domain_extent, nx, endpoint=False, dtype=np.float32)
    rows = []
    for _ in range(batch):
        row = np.zeros_like(x)
        for k in range(1, 5):
            amp = rng.normal(0.0, 0.35 / k)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            row += amp * np.sin(2.0 * np.pi * k * x / domain_extent + phase)
        rows.append(row.astype(np.float32))
    return np.stack(rows, axis=0)


def smooth_ns_initial(batch: int, nx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float32)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    rows = []
    for _ in range(batch):
        field = np.zeros((nx, nx), dtype=np.float32)
        for kx in range(1, 4):
            for ky in range(1, 4):
                amp = rng.normal(0.0, 0.18 / (kx + ky))
                phase = rng.uniform(0.0, 2.0 * np.pi)
                field += amp * np.sin(2.0 * np.pi * (kx * x + ky * y) + phase)
                field += 0.5 * amp * np.cos(2.0 * np.pi * (kx * x - ky * y) + phase)
        rows.append(field.astype(np.float32))
    return np.stack(rows, axis=0)


def synchronize_torch(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def torch_peak_stats(torch_module) -> dict[str, float]:
    if not torch_module.cuda.is_available():
        return {
            "torch_peak_allocated_mib": float("nan"),
            "torch_peak_reserved_mib": float("nan"),
        }
    return {
        "torch_peak_allocated_mib": float(torch_module.cuda.max_memory_allocated() / 1024**2),
        "torch_peak_reserved_mib": float(torch_module.cuda.max_memory_reserved() / 1024**2),
    }


def reset_torch_peak(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()


def worker_torch_burgers(args: argparse.Namespace, output_npy: Path) -> dict[str, Any]:
    import torch
    from solvers import solve_burgers_final_batch

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    u0_np = smooth_burgers_initial(args.burgers_batch, args.burgers_nx, args.burgers_domain, args.seed)
    u0 = torch.from_numpy(u0_np).to(device)

    reset_torch_peak(torch)
    sampler = NvidiaProcessMemorySampler(args.memory_sample_interval)
    sampler.start()

    t0 = time.perf_counter()
    y1 = solve_burgers_final_batch(
        u0,
        t_final=args.burgers_t_final,
        dt=args.burgers_dt,
        domain_extent=args.burgers_domain,
        diffusivity=args.burgers_nu,
        conservative=False,
        device=device,
        dtype=torch.float32,
    )
    synchronize_torch(torch)
    first_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    y2 = solve_burgers_final_batch(
        u0,
        t_final=args.burgers_t_final,
        dt=args.burgers_dt,
        domain_extent=args.burgers_domain,
        diffusivity=args.burgers_nu,
        conservative=False,
        device=device,
        dtype=torch.float32,
    )
    synchronize_torch(torch)
    second_seconds = time.perf_counter() - t0

    np.save(output_npy, y2.detach().cpu().numpy())
    memory = sampler.stop()
    memory.update(torch_peak_stats(torch))
    return {
        "case": "burgers_1d",
        "framework": "pytorch_solvers_py",
        "backend": {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "device_used": str(device),
        },
        "output_shape": list(y2.shape),
        "first_run_seconds": first_seconds,
        "second_run_seconds": second_seconds,
        "memory": memory,
    }


def worker_exponax_burgers(args: argparse.Namespace, output_npy: Path) -> dict[str, Any]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import exponax as ex

    u0_np = smooth_burgers_initial(args.burgers_batch, args.burgers_nx, args.burgers_domain, args.seed)
    steps = int(round(args.burgers_t_final / args.burgers_dt))
    stepper = ex.stepper.Burgers(
        1,
        args.burgers_domain,
        args.burgers_nx,
        args.burgers_dt,
        diffusivity=args.burgers_nu,
        convection_scale=1.0,
        order=4,
        conservative=False,
        dealiasing_fraction=2 / 3,
        num_circle_points=16,
    )
    repeat_fn = ex.repeat(stepper, steps)

    def one(u0):
        full_ic = jnp.expand_dims(u0.squeeze(), axis=0).astype(jnp.float32)
        return repeat_fn(full_ic)[0]

    solve = jax.jit(jax.vmap(one, in_axes=0, out_axes=0))
    u0 = jnp.asarray(u0_np, dtype=jnp.float32)

    sampler = NvidiaProcessMemorySampler(args.memory_sample_interval)
    sampler.start()

    t0 = time.perf_counter()
    y1 = solve(u0)
    y1.block_until_ready()
    first_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    y2 = solve(u0)
    y2.block_until_ready()
    second_seconds = time.perf_counter() - t0

    np.save(output_npy, np.asarray(y2))
    memory = sampler.stop()
    return {
        "case": "burgers_1d",
        "framework": "jax_exponax",
        "backend": {
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "output_shape": list(y2.shape),
        "first_run_seconds": first_seconds,
        "second_run_seconds": second_seconds,
        "memory": memory,
    }


def worker_torch_ns(args: argparse.Namespace, output_npy: Path) -> dict[str, Any]:
    import torch
    from solvers import solve_ns_zongyi_rollout_batch

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    u0_np = smooth_ns_initial(args.ns_batch, args.ns_nx, args.seed + 1000)
    u0 = torch.from_numpy(u0_np).to(device)

    reset_torch_peak(torch)
    sampler = NvidiaProcessMemorySampler(args.memory_sample_interval)
    sampler.start()

    def run_once():
        return solve_ns_zongyi_rollout_batch(
            u0,
            t_final=args.ns_t_final,
            fixed_step=args.ns_dt,
            domain_extent=args.ns_domain,
            diffusivity=args.ns_nu,
            return_time_last=True,
            device=device,
            dtype=torch.float32,
        )

    t0 = time.perf_counter()
    y1 = run_once()
    synchronize_torch(torch)
    first_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    y2 = run_once()
    synchronize_torch(torch)
    second_seconds = time.perf_counter() - t0

    np.save(output_npy, y2.detach().cpu().numpy())
    memory = sampler.stop()
    memory.update(torch_peak_stats(torch))
    return {
        "case": "ns_2d",
        "framework": "pytorch_solvers_py",
        "backend": {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "device_used": str(device),
        },
        "output_shape": list(y2.shape),
        "first_run_seconds": first_seconds,
        "second_run_seconds": second_seconds,
        "memory": memory,
    }


def worker_exponax_ns(args: argparse.Namespace, output_npy: Path) -> dict[str, Any]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import exponax as ex

    u0_np = smooth_ns_initial(args.ns_batch, args.ns_nx, args.seed + 1000)
    steps_per_second = int(round(1.0 / args.ns_dt))
    stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
        2,
        args.ns_domain,
        args.ns_nx,
        args.ns_dt,
        diffusivity=args.ns_nu,
        order=4,
        num_circle_points=16,
        dealiasing_fraction=2 / 3,
    )

    def one(u0):
        u = jnp.rot90(jnp.flip(u0, axis=-2), 3, axes=(-2, -1))

        def micro(carry, _):
            return stepper(carry[None, ...])[0], None

        def one_second(carry, _):
            u_next, _ = jax.lax.scan(micro, carry, None, length=steps_per_second)
            return u_next, u_next

        _, seconds = jax.lax.scan(one_second, u, None, length=args.ns_t_final)
        seq = jnp.concatenate([u[None, ...], seconds], axis=0)
        return jnp.swapaxes(seq, -1, -2)

    def batch(u0_batch):
        seq = jax.lax.map(one, u0_batch)
        return jnp.moveaxis(seq, 1, -1)

    solve = jax.jit(batch)
    u0 = jnp.asarray(u0_np, dtype=jnp.float32)

    sampler = NvidiaProcessMemorySampler(args.memory_sample_interval)
    sampler.start()

    t0 = time.perf_counter()
    y1 = solve(u0)
    y1.block_until_ready()
    first_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    y2 = solve(u0)
    y2.block_until_ready()
    second_seconds = time.perf_counter() - t0

    np.save(output_npy, np.asarray(y2))
    memory = sampler.stop()
    return {
        "case": "ns_2d",
        "framework": "jax_exponax",
        "backend": {
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "output_shape": list(y2.shape),
        "first_run_seconds": first_seconds,
        "second_run_seconds": second_seconds,
        "memory": memory,
    }


WORKERS = {
    ("burgers_1d", "pytorch_solvers_py"): worker_torch_burgers,
    ("burgers_1d", "jax_exponax"): worker_exponax_burgers,
    ("ns_2d", "pytorch_solvers_py"): worker_torch_ns,
    ("ns_2d", "jax_exponax"): worker_exponax_ns,
}


def compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    diff = left - right
    metrics = {
        "max_abs": float(np.max(np.abs(diff))),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "relative_l2": float(np.linalg.norm(diff.reshape(-1)) / max(np.linalg.norm(right.reshape(-1)), 1e-12)),
    }
    if left.ndim >= 4 and left.shape[-1] == right.shape[-1]:
        metrics["per_time_index"] = []
        for t in range(left.shape[-1]):
            metrics["per_time_index"].append({"time_index": t, **compare_arrays(left[..., t], right[..., t])})
    return metrics


def parse_seed_list(seed_text: str | None, base_seed: int, num_seeds: int) -> list[int]:
    if seed_text:
        seeds = [int(part.strip()) for part in seed_text.split(",") if part.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no valid integer seeds were parsed")
        return seeds
    return [base_seed + i for i in range(num_seeds)]


def finite_array(values: list[Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def summarize_values(values: list[Any]) -> dict[str, float | int]:
    arr = finite_array(values)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def paired_t_test(left_values: list[Any], right_values: list[Any]) -> dict[str, float | int]:
    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if left.size < 2:
        return {
            "count": int(left.size),
            "mean_left_minus_right": float(np.mean(left - right)) if left.size else float("nan"),
            "t_statistic": float("nan"),
            "p_value": float("nan"),
        }

    diff = left - right
    if np.allclose(diff, diff[0], rtol=0.0, atol=1e-12):
        if abs(float(diff[0])) <= 1e-12:
            t_statistic = 0.0
            p_value = 1.0
        else:
            t_statistic = math.copysign(float("inf"), float(diff[0]))
            p_value = 0.0
        return {
            "count": int(left.size),
            "mean_left_minus_right": float(np.mean(diff)),
            "t_statistic": t_statistic,
            "p_value": p_value,
        }

    try:
        from scipy import stats

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stats.ttest_rel(left, right, nan_policy="omit")
        return {
            "count": int(left.size),
            "mean_left_minus_right": float(np.mean(diff)),
            "t_statistic": float(result.statistic),
            "p_value": float(result.pvalue),
        }
    except Exception:
        return {
            "count": int(left.size),
            "mean_left_minus_right": float(np.mean(diff)),
            "t_statistic": float("nan"),
            "p_value": float("nan"),
        }


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return relpath(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {key: _as_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_as_jsonable(val) for val in value]
    if isinstance(value, tuple):
        return [_as_jsonable(val) for val in value]
    return value


def plot_burgers_comparison(
    initial_batch: np.ndarray,
    pt_output: np.ndarray,
    jax_output: np.ndarray,
    seed: int,
    plot_dir: Path,
    max_samples: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_dir = plot_dir / "burgers_1d"
    case_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    num_samples = min(pt_output.shape[0], max_samples)
    x = np.arange(pt_output.shape[-1])
    for sample_idx in range(num_samples):
        initial = initial_batch[sample_idx]
        pt = pt_output[sample_idx]
        jax = jax_output[sample_idx]
        diff = pt - jax

        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(x, initial, color="0.35", linewidth=1.4)
        axes[0].set_title(f"1D Burgers initial condition, seed={seed}, sample={sample_idx}")
        axes[0].set_ylabel("u0")
        axes[0].grid(alpha=0.25)

        axes[1].plot(x, pt, label="PyTorch solvers.py final", linewidth=1.6)
        axes[1].plot(x, jax, label="Exponax/JAX final", linewidth=1.2, linestyle="--")
        axes[1].set_title("Final solution")
        axes[0].set_xlabel("x index")
        axes[1].set_ylabel("u(T)")
        axes[1].legend(loc="best")
        axes[1].grid(alpha=0.25)

        axes[2].plot(x, diff, color="tab:red", linewidth=1.1)
        axes[2].set_title("Final difference: PyTorch - JAX")
        axes[2].set_xlabel("x index")
        axes[2].set_ylabel("diff")
        axes[2].grid(alpha=0.25)
        fig.tight_layout()

        path = case_dir / f"burgers_seed{seed}_sample{sample_idx}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(relpath(path))
    return paths


def plot_ns_comparison(
    pt_output: np.ndarray,
    jax_output: np.ndarray,
    seed: int,
    plot_dir: Path,
    max_samples: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_dir = plot_dir / "ns_2d"
    case_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    num_samples = min(pt_output.shape[0], max_samples)
    final_t = pt_output.shape[-1] - 1
    for sample_idx in range(num_samples):
        initial = pt_output[sample_idx, ..., 0]
        pt = pt_output[sample_idx, ..., final_t]
        jax = jax_output[sample_idx, ..., final_t]
        diff = pt - jax
        value_abs = max(float(np.max(np.abs(initial))), float(np.max(np.abs(pt))), float(np.max(np.abs(jax))), 1e-12)
        diff_abs = max(float(np.max(np.abs(diff))), 1e-12)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        images = [
            axes[0].imshow(initial, cmap="viridis", vmin=-value_abs, vmax=value_abs, origin="lower"),
            axes[1].imshow(pt, cmap="viridis", vmin=-value_abs, vmax=value_abs, origin="lower"),
            axes[2].imshow(jax, cmap="viridis", vmin=-value_abs, vmax=value_abs, origin="lower"),
            axes[3].imshow(diff, cmap="coolwarm", vmin=-diff_abs, vmax=diff_abs, origin="lower"),
        ]
        axes[0].set_title("Initial condition")
        axes[1].set_title("PyTorch final")
        axes[2].set_title("Exponax/JAX final")
        axes[3].set_title("Final PyTorch - JAX")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"2D NS final frame heatmaps, seed={seed}, sample={sample_idx}, t_index={final_t}")
        fig.colorbar(images[0], ax=axes[:3], fraction=0.046, pad=0.04)
        fig.colorbar(images[3], ax=axes[3], fraction=0.046, pad=0.04)

        path = case_dir / f"ns_seed{seed}_sample{sample_idx}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(relpath(path))
    return paths


def plot_case_comparison(
    case: str,
    pt_output: np.ndarray,
    jax_output: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    plot_dir: Path,
    max_samples: int,
) -> list[str]:
    if max_samples <= 0:
        return []
    if case == "burgers_1d":
        initial_batch = smooth_burgers_initial(
            pt_output.shape[0],
            pt_output.shape[-1],
            args.burgers_domain,
            seed,
        )
        return plot_burgers_comparison(initial_batch, pt_output, jax_output, seed, plot_dir, max_samples)
    if case == "ns_2d":
        return plot_ns_comparison(pt_output, jax_output, seed, plot_dir, max_samples)
    raise ValueError(f"Unknown case: {case}")


def build_case_statistics(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    comparison_metrics = ["max_abs", "mae", "rmse", "relative_l2"]
    run_metrics = ["first_run_seconds", "second_run_seconds"]
    memory_metrics = [
        "sampled_process_gpu_peak_mib",
        "sampled_global_gpu_peak_delta_mib",
        "sampled_global_gpu_peak_mib",
        "torch_peak_allocated_mib",
        "torch_peak_reserved_mib",
    ]

    statistics: dict[str, Any] = {
        "comparison": {},
        "runs": {},
        "paired_tests_pytorch_minus_jax": {},
    }
    for metric in comparison_metrics:
        statistics["comparison"][metric] = summarize_values(
            [condition["comparison"].get(metric) for condition in conditions]
        )

    for framework in ["pytorch_solvers_py", "jax_exponax"]:
        statistics["runs"][framework] = {}
        for metric in run_metrics:
            statistics["runs"][framework][metric] = summarize_values(
                [condition["runs"][framework].get(metric) for condition in conditions]
            )
        for metric in memory_metrics:
            statistics["runs"][framework][metric] = summarize_values(
                [
                    condition["runs"][framework].get("memory", {}).get(metric)
                    for condition in conditions
                ]
            )

    for metric in run_metrics:
        statistics["paired_tests_pytorch_minus_jax"][metric] = paired_t_test(
            [condition["runs"]["pytorch_solvers_py"].get(metric) for condition in conditions],
            [condition["runs"]["jax_exponax"].get(metric) for condition in conditions],
        )
    for metric in ["sampled_process_gpu_peak_mib", "sampled_global_gpu_peak_delta_mib", "sampled_global_gpu_peak_mib"]:
        statistics["paired_tests_pytorch_minus_jax"][metric] = paired_t_test(
            [
                condition["runs"]["pytorch_solvers_py"].get("memory", {}).get(metric)
                for condition in conditions
            ],
            [
                condition["runs"]["jax_exponax"].get("memory", {}).get(metric)
                for condition in conditions
            ],
        )

    return statistics


def run_worker(args: argparse.Namespace) -> int:
    key = (args.worker_case, args.worker_framework)
    output_npy = Path(args.worker_output_npy)
    result_json = Path(args.worker_result_json)
    result = WORKERS[key](args, output_npy)
    result_json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


def run_subprocess(args: argparse.Namespace, case: str, framework: str, tmpdir: Path, seed: int) -> tuple[dict, Path]:
    output_npy = tmpdir / f"{case}_{framework}_seed{seed}.npy"
    result_json = tmpdir / f"{case}_{framework}_seed{seed}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-case",
        case,
        "--worker-framework",
        framework,
        "--worker-output-npy",
        str(output_npy),
        "--worker-result-json",
        str(result_json),
        "--burgers-batch",
        str(args.burgers_batch),
        "--burgers-nx",
        str(args.burgers_nx),
        "--burgers-t-final",
        str(args.burgers_t_final),
        "--burgers-dt",
        str(args.burgers_dt),
        "--burgers-nu",
        str(args.burgers_nu),
        "--burgers-domain",
        str(args.burgers_domain),
        "--ns-batch",
        str(args.ns_batch),
        "--ns-nx",
        str(args.ns_nx),
        "--ns-t-final",
        str(args.ns_t_final),
        "--ns-dt",
        str(args.ns_dt),
        "--ns-nu",
        str(args.ns_nu),
        "--ns-domain",
        str(args.ns_domain),
        "--seed",
        str(seed),
        "--memory-sample-interval",
        str(args.memory_sample_interval),
    ]
    if args.cpu:
        cmd.append("--cpu")
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    return json.loads(result_json.read_text()), output_npy


def write_outputs(results: dict[str, Any], output_json: Path, output_md: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_as_jsonable(results), indent=2) + "\n")

    rows = []
    for case, case_data in results["cases"].items():
        for condition in case_data["conditions"]:
            comparison = condition["comparison"]
            for framework, run in condition["runs"].items():
                mem = run["memory"]
                rows.append(
                    {
                        "case": case,
                        "condition_index": condition["condition_index"],
                        "seed": condition["seed"],
                        "framework": framework,
                        "first_run_seconds": run["first_run_seconds"],
                        "second_run_seconds": run["second_run_seconds"],
                        "sampled_process_gpu_peak_mib": mem.get("sampled_process_gpu_peak_mib"),
                        "sampled_global_gpu_peak_delta_mib": mem.get("sampled_global_gpu_peak_delta_mib"),
                        "sampled_global_gpu_peak_mib": mem.get("sampled_global_gpu_peak_mib"),
                        "torch_peak_allocated_mib": mem.get("torch_peak_allocated_mib"),
                        "torch_peak_reserved_mib": mem.get("torch_peak_reserved_mib"),
                        "max_abs_vs_other": comparison["max_abs"],
                        "mae_vs_other": comparison["mae"],
                        "rmse_vs_other": comparison["rmse"],
                        "relative_l2_vs_other": comparison["relative_l2"],
                    }
                )
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PyTorch solvers.py vs Exponax/JAX Solver Comparison",
        "",
        "Each solver was run in a separate process. `first_run_seconds` includes JAX JIT compilation for Exponax; `second_run_seconds` is the warm run after compilation.",
        "",
        "## Settings",
        "",
        "```json",
        json.dumps(results["settings"], indent=2),
        "```",
        "",
        "## Error Summary Across Conditions",
        "",
        "| case | conditions | relative L2 mean | relative L2 std | max abs mean | max abs max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case, case_data in results["cases"].items():
        stats = case_data["statistics"]["comparison"]
        lines.append(
            "| {case} | {count} | {rel_mean:.6g} | {rel_std:.6g} | {max_mean:.6g} | {max_max:.6g} |".format(
                case=case,
                count=stats["relative_l2"]["count"],
                rel_mean=stats["relative_l2"]["mean"],
                rel_std=stats["relative_l2"]["std"],
                max_mean=stats["max_abs"]["mean"],
                max_max=stats["max_abs"]["max"],
            )
        )

    lines.extend(
        [
            "",
            "## Runtime And Memory Summary",
            "",
            "`p_value` is from a paired t-test across the repeated initial conditions. The paired test is `PyTorch - Exponax/JAX`, so a negative mean difference means PyTorch was smaller/faster.",
            "",
            "| case | metric | PyTorch mean | PyTorch std | Exponax/JAX mean | Exponax/JAX std | mean PyTorch-JAX | paired p-value |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    summary_metrics = [
        ("first_run_seconds", "cold seconds"),
        ("second_run_seconds", "warm seconds"),
        ("sampled_global_gpu_peak_delta_mib", "global GPU delta MiB"),
        ("sampled_process_gpu_peak_mib", "process GPU peak MiB"),
    ]
    for case, case_data in results["cases"].items():
        stats = case_data["statistics"]
        for metric, label in summary_metrics:
            pt_stat = stats["runs"]["pytorch_solvers_py"][metric]
            jax_stat = stats["runs"]["jax_exponax"][metric]
            test = stats["paired_tests_pytorch_minus_jax"][metric]
            lines.append(
                "| {case} | {label} | {pt_mean:.6g} | {pt_std:.6g} | {jax_mean:.6g} | {jax_std:.6g} | {diff:.6g} | {p:.6g} |".format(
                    case=case,
                    label=label,
                    pt_mean=pt_stat["mean"],
                    pt_std=pt_stat["std"],
                    jax_mean=jax_stat["mean"],
                    jax_std=jax_stat["std"],
                    diff=test["mean_left_minus_right"],
                    p=test["p_value"],
                )
            )

    lines.extend(
        [
            "",
            "## Visualization Files",
            "",
            f"- Plot directory: `{results['settings']['plot_dir']}`",
        ]
    )
    for case, case_data in results["cases"].items():
        plot_paths = []
        for condition in case_data["conditions"]:
            plot_paths.extend(condition.get("visualizations", []))
        lines.append(f"- `{case}` plots: {len(plot_paths)}")
        for path in plot_paths[:10]:
            lines.append(f"  - `{path}`")
        if len(plot_paths) > 10:
            lines.append(f"  - ... {len(plot_paths) - 10} more")

    lines.extend(
        [
            "",
            "## Full Run Details",
            "",
            "```json",
            json.dumps(_as_jsonable(results["cases"]), indent=2),
            "```",
            "",
        ]
    )
    output_md.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-case", choices=["burgers_1d", "ns_2d"])
    parser.add_argument("--worker-framework", choices=["pytorch_solvers_py", "jax_exponax"])
    parser.add_argument("--worker-output-npy")
    parser.add_argument("--worker-result-json")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--cases",
        default="burgers_1d,ns_2d",
        help="Comma-separated cases to run: burgers_1d,ns_2d.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated explicit seed list.")
    parser.add_argument("--max-visualized-samples-per-seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--memory-sample-interval", type=float, default=0.05)

    parser.add_argument("--burgers-batch", type=int, default=4)
    parser.add_argument("--burgers-nx", type=int, default=256)
    parser.add_argument("--burgers-t-final", type=float, default=0.1)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-nu", type=float, default=1e-3)

    parser.add_argument("--ns-batch", type=int, default=2)
    parser.add_argument("--ns-nx", type=int, default=64)
    parser.add_argument("--ns-t-final", type=int, default=1)
    parser.add_argument("--ns-dt", type=float, default=0.005)
    parser.add_argument("--ns-domain", type=float, default=1.0)
    parser.add_argument("--ns-nu", type=float, default=1e-5)
    args = parser.parse_args()

    if args.worker:
        return run_worker(args)

    seeds = parse_seed_list(args.seeds, args.seed, args.num_seeds)
    requested_cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    valid_cases = {"burgers_1d", "ns_2d"}
    unknown_cases = sorted(set(requested_cases) - valid_cases)
    if unknown_cases:
        raise ValueError(f"Unknown cases: {unknown_cases}")
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "seed": args.seed,
        "seeds": seeds,
        "num_conditions": len(seeds),
        "plot_dir": relpath(args.plot_dir),
        "max_visualized_samples_per_seed": args.max_visualized_samples_per_seed,
        "burgers": {
            "batch": args.burgers_batch,
            "nx": args.burgers_nx,
            "t_final": args.burgers_t_final,
            "dt": args.burgers_dt,
            "nu": args.burgers_nu,
            "domain_extent": args.burgers_domain,
        },
        "ns": {
            "batch": args.ns_batch,
            "nx": args.ns_nx,
            "t_final": args.ns_t_final,
            "dt": args.ns_dt,
            "nu": args.ns_nu,
            "domain_extent": args.ns_domain,
        },
    }
    results: dict[str, Any] = {"settings": settings, "cases": {}}
    conditions_by_case: dict[str, list[dict[str, Any]]] = {case: [] for case in requested_cases}

    with tempfile.TemporaryDirectory(prefix="torch_vs_exponax_") as tmp:
        tmpdir = Path(tmp)
        for condition_index, seed in enumerate(seeds):
            for case in requested_cases:
                case_runs = {}
                output_paths = {}
                for framework in ["pytorch_solvers_py", "jax_exponax"]:
                    print(f"[run] seed={seed} {case} {framework}", flush=True)
                    run, output_path = run_subprocess(args, case, framework, tmpdir, seed)
                    case_runs[framework] = run
                    output_paths[framework] = output_path

                pt_output = np.load(output_paths["pytorch_solvers_py"])
                jax_output = np.load(output_paths["jax_exponax"])
                comparison = compare_arrays(pt_output, jax_output)
                visualization_paths = plot_case_comparison(
                    case,
                    pt_output,
                    jax_output,
                    seed,
                    args,
                    args.plot_dir,
                    args.max_visualized_samples_per_seed,
                )
                conditions_by_case[case].append(
                    {
                        "condition_index": condition_index,
                        "seed": seed,
                        "runs": case_runs,
                        "comparison": comparison,
                        "visualizations": visualization_paths,
                    }
                )

    for case, conditions in conditions_by_case.items():
        results["cases"][case] = {
            "conditions": conditions,
            "statistics": build_case_statistics(conditions),
        }

    write_outputs(results, args.output_json, args.output_md, args.output_csv)
    print(json.dumps(_as_jsonable(results), indent=2))
    print(f"[saved] {args.output_json}")
    print(f"[saved] {args.output_md}")
    print(f"[saved] {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
