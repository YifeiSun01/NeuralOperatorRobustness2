#!/usr/bin/env python3
"""Benchmark trained FNO model inference for PyTorch and JAX checkpoints."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BURGERS_TORCH = (
    PROJECT_ROOT
    / "1D_Burgers"
    / "trained_models"
    / "attack_ready"
    / "burgers_nu0.001_fno1d_500"
    / "checkpoints"
    / "pytorch_fno1d_500.pt"
)
DEFAULT_BURGERS_JAX = DEFAULT_BURGERS_TORCH.with_name("jax_real_imag_fno1d_500.pkl")
DEFAULT_BURGERS_TEST = (
    PROJECT_ROOT
    / "1D_Burgers"
    / "datasets"
    / "1D"
    / "Burgers"
    / "batched_exponax_splits"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt"
)
DEFAULT_NS_TORCH = (
    PROJECT_ROOT
    / "fno_training_runs"
    / "ns_m12_w20_profile_500"
    / "ns_real_initial_laxmap"
    / "ns_2d"
    / "checkpoints"
    / "fno2d_pytorch.pt"
)
DEFAULT_NS_JAX = DEFAULT_NS_TORCH.with_name("fno2d_jax_real_imag.pkl")
DEFAULT_NS_TEST = (
    PROJECT_ROOT
    / "2D_NS_FNO2d_recurrent"
    / "datasets"
    / "exponax_datasets"
    / "t20"
    / "real_initial_laxmap_single"
    / "test"
    / "dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt"
)


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def global_gpu_used_mib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        vals = [float(line.strip()) for line in out.splitlines() if line.strip()]
        return vals[0] if vals else float("nan")
    except Exception:
        return float("nan")


class PhaseSampler:
    def __init__(self, interval: float):
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.samples = []
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                self.samples.append(global_gpu_used_mib())
                time.sleep(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, self.interval * 4))
        self.samples.append(global_gpu_used_mib())

    @property
    def peak(self) -> float:
        vals = [v for v in self.samples if math.isfinite(v)]
        return max(vals) if vals else float("nan")


def summarize(values: list[float]) -> dict[str, float | int | None]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {"count": 0, "mean": None, "min": None, "max": None, "total": None}
    return {
        "count": len(vals),
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "total": float(np.sum(vals)),
    }


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = pred - target
    flat_diff = diff.reshape((diff.shape[0], -1))
    flat_target = target.reshape((target.shape[0], -1))
    rel = np.linalg.norm(flat_diff, axis=1) / np.maximum(np.linalg.norm(flat_target, axis=1), 1e-12)
    mse = float(np.mean(diff * diff))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(diff))),
        "relative_l2": float(np.mean(rel)),
    }


def load_burgers_batch(path: Path, batch_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    x = data["x"].float().numpy()[..., None]
    y = data["y"].float().numpy()[..., None]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(x.shape[0], size=batch_size, replace=False))
    return x[idx].astype(np.float32), y[idx].astype(np.float32), idx.tolist()


def load_ns_batch(path: Path, batch_size: int, seed: int, t_in: int = 10, t_out: int = 10):
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    seq = data["y"].float().numpy()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(seq.shape[0], size=batch_size, replace=False))
    sample = seq[idx].astype(np.float32)
    x = sample[..., :t_in]
    y = sample[..., t_in : t_in + t_out]
    return x, y, idx.tolist()


def load_jax_params(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["params"] if isinstance(payload, dict) and "params" in payload else payload


def torch_model(case: str, checkpoint: Path, device: Any):
    import torch

    if case == "burgers_1d":
        mod = load_module("bench_fno1d_torch", PROJECT_ROOT / "1D_Burgers" / "models" / "FNO1d.py")
        model = mod.FNO1d(modes=16, width=64, num_layers=4, dtype=torch.float32).to(device)
    elif case == "ns_2d":
        mod = load_module("bench_fno2d_torch", PROJECT_ROOT / "2D_NS_FNO2d_recurrent" / "models" / "FNO2d.py")
        fno = mod.FNO2d(modes1=12, modes2=12, width=20, num_layers=4, in_channels=10).to(device)
        model = mod.RecurrentPredictor(fno, T_out=10, step=1).to(device)
    else:
        raise ValueError(case)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if case == "ns_2d":
        model.model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def jax_model(case: str, checkpoint: Path):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp

    params = jax.device_put(load_jax_params(checkpoint))
    if case == "burgers_1d":
        mod = load_module("bench_fno1d_jax_ri", PROJECT_ROOT / "1D_Burgers" / "models" / "FNO1d_jax_real_imag.py")

        @jax.jit
        def apply(x):
            return mod.fno1d_apply(params, x, modes=16, width=64, num_layers=4, dtype=jnp.float32)

    elif case == "ns_2d":
        mod = load_module("bench_fno2d_jax_ri", PROJECT_ROOT / "2D_NS_FNO2d_recurrent" / "models" / "FNO2d_jax_real_imag.py")

        @jax.jit
        def apply(x):
            return mod.recurrent_predictor_apply(
                params, x, modes1=12, modes2=12, width=20, num_layers=4, T_out=10, step=1, dtype=jnp.float32
            )

    else:
        raise ValueError(case)
    return apply


def run_torch_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = args.burgers_torch_checkpoint if args.case == "burgers_1d" else args.ns_torch_checkpoint
    data_path = args.burgers_test_path if args.case == "burgers_1d" else args.ns_test_path
    model = torch_model(args.case, checkpoint, device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    model_loaded_gpu_mib = global_gpu_used_mib()

    rows = []
    prediction_paths: dict[str, str] = {}
    for batch_size in args.batch_sizes:
        if args.case == "burgers_1d":
            x_np, y_np, indices = load_burgers_batch(data_path, batch_size, args.seed + batch_size)
        else:
            x_np, y_np, indices = load_ns_batch(data_path, batch_size, args.seed + batch_size)
        x = torch.from_numpy(x_np).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pred_last = None
        for rep in range(args.repeats + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            before = global_gpu_used_mib()
            with PhaseSampler(args.gpu_sample_interval) as sampler:
                start = time.perf_counter()
                with torch.no_grad():
                    pred = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - start
            after = global_gpu_used_mib()
            pred_last = pred.detach().cpu().numpy()
            rows.append(
                {
                    "case": args.case,
                    "framework": "pytorch",
                    "batch_size": batch_size,
                    "rep": rep,
                    "run_kind": "first_compile_or_warmup" if rep == 0 else "steady",
                    "forward_seconds": seconds,
                    "global_gpu_before_mib": before,
                    "global_gpu_after_mib": after,
                    "global_gpu_peak_mib": sampler.peak,
                    "global_gpu_peak_delta_mib": sampler.peak - before if math.isfinite(sampler.peak) and math.isfinite(before) else float("nan"),
                    "torch_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 1024**2)
                    if device.type == "cuda"
                    else float("nan"),
                    "torch_peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 1024**2)
                    if device.type == "cuda"
                    else float("nan"),
                    "sample_indices": json.dumps(indices),
                    **metrics(pred_last, y_np),
                }
            )
        pred_path = args.worker_output_json.parent / "predictions" / f"{args.case}_pytorch_batch{batch_size}.npz"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_path, input=x_np, target=y_np, pred=pred_last, indices=np.asarray(indices))
        prediction_paths[f"batch{batch_size}"] = relpath(pred_path)
    return {
        "case": args.case,
        "framework": "pytorch",
        "checkpoint": relpath(checkpoint),
        "data_path": relpath(data_path),
        "model_loaded_global_gpu_mib": model_loaded_gpu_mib,
        "prediction_paths": prediction_paths,
        "rows": rows,
    }


def run_jax_worker(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp

    checkpoint = args.burgers_jax_checkpoint if args.case == "burgers_1d" else args.ns_jax_checkpoint
    data_path = args.burgers_test_path if args.case == "burgers_1d" else args.ns_test_path
    apply = jax_model(args.case, checkpoint)
    model_loaded_gpu_mib = global_gpu_used_mib()

    rows = []
    prediction_paths: dict[str, str] = {}
    for batch_size in args.batch_sizes:
        if args.case == "burgers_1d":
            x_np, y_np, indices = load_burgers_batch(data_path, batch_size, args.seed + batch_size)
        else:
            x_np, y_np, indices = load_ns_batch(data_path, batch_size, args.seed + batch_size)
        x = jax.device_put(jnp.asarray(x_np, dtype=jnp.float32))
        x.block_until_ready()
        pred_last = None
        for rep in range(args.repeats + 1):
            before = global_gpu_used_mib()
            with PhaseSampler(args.gpu_sample_interval) as sampler:
                start = time.perf_counter()
                pred = apply(x)
                pred.block_until_ready()
                seconds = time.perf_counter() - start
            after = global_gpu_used_mib()
            pred_last = np.asarray(pred)
            rows.append(
                {
                    "case": args.case,
                    "framework": "jax_real_imag",
                    "batch_size": batch_size,
                    "rep": rep,
                    "run_kind": "first_compile_or_warmup" if rep == 0 else "steady",
                    "forward_seconds": seconds,
                    "global_gpu_before_mib": before,
                    "global_gpu_after_mib": after,
                    "global_gpu_peak_mib": sampler.peak,
                    "global_gpu_peak_delta_mib": sampler.peak - before if math.isfinite(sampler.peak) and math.isfinite(before) else float("nan"),
                    "torch_peak_allocated_mib": float("nan"),
                    "torch_peak_reserved_mib": float("nan"),
                    "sample_indices": json.dumps(indices),
                    **metrics(pred_last, y_np),
                }
            )
        pred_path = args.worker_output_json.parent / "predictions" / f"{args.case}_jax_real_imag_batch{batch_size}.npz"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_path, input=x_np, target=y_np, pred=pred_last, indices=np.asarray(indices))
        prediction_paths[f"batch{batch_size}"] = relpath(pred_path)
    return {
        "case": args.case,
        "framework": "jax_real_imag",
        "checkpoint": relpath(checkpoint),
        "data_path": relpath(data_path),
        "model_loaded_global_gpu_mib": model_loaded_gpu_mib,
        "prediction_paths": prediction_paths,
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(results: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"workers": [], "by_case_framework_batch": {}}
    for result in results:
        summary["workers"].append(
            {
                "case": result["case"],
                "framework": result["framework"],
                "checkpoint": result["checkpoint"],
                "data_path": result["data_path"],
                "model_loaded_global_gpu_mib": result["model_loaded_global_gpu_mib"],
                "prediction_paths": result.get("prediction_paths", {}),
            }
        )
    for key in sorted({(r["case"], r["framework"], int(r["batch_size"])) for r in rows}):
        case, framework, batch_size = key
        group = [r for r in rows if r["case"] == case and r["framework"] == framework and int(r["batch_size"]) == batch_size]
        first = [r for r in group if r["run_kind"] == "first_compile_or_warmup"]
        steady = [r for r in group if r["run_kind"] == "steady"]
        summary["by_case_framework_batch"][f"{case}/{framework}/batch{batch_size}"] = {
            "first_forward_seconds": first[0]["forward_seconds"] if first else None,
            "steady_forward_seconds": summarize([r["forward_seconds"] for r in steady]),
            "steady_global_gpu_peak_mib": summarize([r["global_gpu_peak_mib"] for r in steady]),
            "steady_global_gpu_peak_delta_mib": summarize([r["global_gpu_peak_delta_mib"] for r in steady]),
            "steady_torch_peak_allocated_mib": summarize([r["torch_peak_allocated_mib"] for r in steady]),
            "steady_torch_peak_reserved_mib": summarize([r["torch_peak_reserved_mib"] for r in steady]),
            "relative_l2": summarize([r["relative_l2"] for r in group]),
            "mse": summarize([r["mse"] for r in group]),
        }
    return summary


def make_visualizations(output_root: Path, results: list[dict[str, Any]], max_samples: int = 5) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipping visualizations: {exc}", flush=True)
        return []

    by_key: dict[tuple[str, str], dict[str, Path]] = {}
    for result in results:
        framework = result["framework"]
        for batch_key, raw_path in result.get("prediction_paths", {}).items():
            path = PROJECT_ROOT / raw_path
            by_key.setdefault((result["case"], batch_key), {})[framework] = path

    saved: list[str] = []
    vis_dir = output_root / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for (case, batch_key), paths in sorted(by_key.items()):
        if "pytorch" not in paths or "jax_real_imag" not in paths:
            continue
        pt = np.load(paths["pytorch"])
        jx = np.load(paths["jax_real_imag"])
        x = pt["input"]
        target = pt["target"]
        pred_pt = pt["pred"]
        pred_jx = jx["pred"]
        n = min(max_samples, x.shape[0])

        if case == "burgers_1d":
            fig, axes = plt.subplots(n, 2, figsize=(11, 2.2 * n), squeeze=False)
            grid = np.arange(x.shape[1])
            for i in range(n):
                axes[i, 0].plot(grid, x[i, :, 0], color="0.35", linewidth=1.0, label="initial")
                axes[i, 0].plot(grid, target[i, :, 0], color="black", linewidth=1.2, label="target")
                axes[i, 0].plot(grid, pred_pt[i, :, 0], color="#1f77b4", linewidth=1.0, label="PyTorch")
                axes[i, 0].plot(grid, pred_jx[i, :, 0], color="#2ca02c", linewidth=1.0, linestyle="--", label="JAX")
                axes[i, 0].set_title(f"sample {i}")
                axes[i, 0].grid(alpha=0.2)
                axes[i, 1].plot(grid, pred_pt[i, :, 0] - pred_jx[i, :, 0], color="#d62728", linewidth=1.0)
                axes[i, 1].set_title("|PyTorch - JAX| diff signed")
                axes[i, 1].grid(alpha=0.2)
            axes[0, 0].legend(loc="best", fontsize=8)
            fig.suptitle(f"Burgers FNO inference comparison, {batch_key}")
        else:
            fig, axes = plt.subplots(n, 5, figsize=(15, 3.0 * n), squeeze=False)
            for i in range(n):
                panels = [
                    ("initial last", x[i, :, :, -1]),
                    ("target final", target[i, :, :, -1]),
                    ("PyTorch final", pred_pt[i, :, :, -1]),
                    ("JAX final", pred_jx[i, :, :, -1]),
                    ("abs diff", np.abs(pred_pt[i, :, :, -1] - pred_jx[i, :, :, -1])),
                ]
                for j, (title, arr) in enumerate(panels):
                    im = axes[i, j].imshow(arr, origin="lower", cmap="viridis")
                    axes[i, j].set_title(f"sample {i} {title}")
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                    fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.02)
            fig.suptitle(f"NS FNO inference comparison, {batch_key}")
        fig.tight_layout()
        out = vis_dir / f"{case}_{batch_key}_first{n}_comparison.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        saved.append(relpath(out))
    return saved


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_text(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def worker_args(base: argparse.Namespace, case: str, framework: str, output_json: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--case",
        case,
        "--framework",
        framework,
        "--batch-sizes",
        ",".join(map(str, base.batch_sizes)),
        "--repeats",
        str(base.repeats),
        "--seed",
        str(base.seed),
        "--gpu-sample-interval",
        str(base.gpu_sample_interval),
        "--worker-output-json",
        str(output_json),
        "--burgers-torch-checkpoint",
        str(base.burgers_torch_checkpoint),
        "--burgers-jax-checkpoint",
        str(base.burgers_jax_checkpoint),
        "--ns-torch-checkpoint",
        str(base.ns_torch_checkpoint),
        "--ns-jax-checkpoint",
        str(base.ns_jax_checkpoint),
        "--burgers-test-path",
        str(base.burgers_test_path),
        "--ns-test-path",
        str(base.ns_test_path),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=["burgers_1d", "ns_2d"])
    parser.add_argument("--framework", choices=["pytorch", "jax_real_imag"])
    parser.add_argument("--cases", default="burgers_1d,ns_2d")
    parser.add_argument("--frameworks", default="pytorch,jax_real_imag")
    parser.add_argument("--batch-sizes", type=parse_csv_ints, default=[5, 10])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.005)
    parser.add_argument("--visualize-samples", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "benchmark_results" / "fno_inference_benchmark")
    parser.add_argument("--worker-output-json", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--burgers-torch-checkpoint", type=Path, default=DEFAULT_BURGERS_TORCH)
    parser.add_argument("--burgers-jax-checkpoint", type=Path, default=DEFAULT_BURGERS_JAX)
    parser.add_argument("--ns-torch-checkpoint", type=Path, default=DEFAULT_NS_TORCH)
    parser.add_argument("--ns-jax-checkpoint", type=Path, default=DEFAULT_NS_JAX)
    parser.add_argument("--burgers-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument("--ns-test-path", type=Path, default=DEFAULT_NS_TEST)
    args = parser.parse_args()

    if args.worker:
        if args.framework == "pytorch":
            result = run_torch_worker(args)
        elif args.framework == "jax_real_imag":
            result = run_jax_worker(args)
        else:
            raise ValueError(args.framework)
        if args.worker_output_json is None:
            print(json.dumps(result, indent=2))
        else:
            args.worker_output_json.parent.mkdir(parents=True, exist_ok=True)
            args.worker_output_json.write_text(json.dumps(result, indent=2) + "\n")
        return 0

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = parse_csv_text(args.cases)
    frameworks = parse_csv_text(args.frameworks)
    worker_results = []
    for case in cases:
        for framework in frameworks:
            out = output_root / "workers" / f"{case}_{framework}.json"
            cmd = worker_args(args, case, framework, out)
            env = os.environ.copy()
            env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            print("[worker]", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
            worker_results.append(json.loads(out.read_text()))

    rows = []
    for result in worker_results:
        rows.extend(result["rows"])
    csv_path = output_root / "inference_metrics.csv"
    summary_path = output_root / "summary.json"
    write_csv(csv_path, rows)
    visualizations = make_visualizations(output_root, worker_results, max_samples=args.visualize_samples)
    summary = {
        "settings": {
            "output_root": relpath(output_root),
            "cases": cases,
            "frameworks": frameworks,
            "batch_sizes": args.batch_sizes,
            "repeats": args.repeats,
            "seed": args.seed,
            "visualize_samples": args.visualize_samples,
        },
        **build_summary(worker_results, rows),
        "csv": relpath(csv_path),
        "visualizations": visualizations,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[saved] {summary_path}", flush=True)
    print(f"[saved] {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
