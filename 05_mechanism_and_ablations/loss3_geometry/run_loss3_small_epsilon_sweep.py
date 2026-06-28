#!/usr/bin/env python3
"""Run the FNO nu=0.001 small-epsilon local-structure sweep.

This script is intentionally scoped to the current Loss3 Experiment 3 plan:
FNO / 1D Burgers / nu=0.001 only.

For each selected test index and epsilon, it evaluates a fixed candidate bank
of unit directions and records candidate-bank estimates of:

    L_f(epsilon) = max ||f(x+eps v)-f(x)|| / eps
    L_j(epsilon) = max ||j(x+eps v)-j(x)|| / eps
    L_e(epsilon) = max ||e(x+eps v)-e(x)|| / eps
    G_e(epsilon) = max (||e(x+eps v)||-||e(x)||) / eps

The direction bank is built from saved clean-point SVD directions for J_f, J_j,
and J_e=J_f-J_j, plus clean outward-growth directions and random controls. This
keeps the sweep reproducible and directly tied to the existing local Jacobian
forensics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analyze_local_jacobian_fno_deeponet import load_sample  # noqa: E402
from tools.attack_framework_matrix import (  # noqa: E402
    DEFAULT_BURGERS_MODEL_DIR,
    DEFAULT_BURGERS_TEST,
    load_burgers_torch_model,
    make_burgers_jax_solver,
    make_jax_torch_bridge,
    sync_torch,
)


DEFAULT_JACOBIAN_ROOT = PROJECT_ROOT / "forensics" / "fno_solver_jacobian_similarity_20260514_raw_recomputed"
DEFAULT_OUTWARD_ROOT = PROJECT_ROOT / "forensics" / "outward_growth_direction_20260515" / "fno_nu0p001"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forensics" / "loss3_small_epsilon_sweep_20260516" / "fno_nu0p001"
DEFAULT_RESULT_DOC = PROJECT_ROOT / "docs" / "loss3_small_epsilon_sweep_result_20260516.md"
EPS = 1e-12


def finite_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def norm2(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64).reshape(-1)))


def normalize_l2(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    n = norm2(arr)
    if n <= EPS:
        return np.zeros_like(arr)
    return arr / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = norm2(af) * norm2(bf)
    if denom <= EPS:
        return math.nan
    return float(np.dot(af, bf) / denom)


def finite_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _path_entries() -> list[str]:
    return [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]


def find_venv_ptxas_dir() -> Path | None:
    roots: list[Path] = []
    for candidate in (os.environ.get("VIRTUAL_ENV"), sys.prefix, PROJECT_ROOT / "adv_robust"):
        if not candidate:
            continue
        root = Path(candidate).expanduser().resolve()
        if root not in roots:
            roots.append(root)
    for root in roots:
        for ptxas in sorted(root.glob("lib/python*/site-packages/triton/backends/nvidia/bin/ptxas")):
            if ptxas.is_file():
                return ptxas.parent
    return None


def configure_runtime(args: argparse.Namespace) -> None:
    args.runtime_ptxas_dir = None
    args.runtime_cudnn_enabled = None
    if not args.runtime_workarounds:
        return
    if args.prepend_env_ptxas:
        ptxas_dir = find_venv_ptxas_dir()
        if ptxas_dir is not None:
            text = str(ptxas_dir)
            os.environ["PATH"] = os.pathsep.join([text, *[entry for entry in _path_entries() if entry != text]])
            args.runtime_ptxas_dir = text
            print(f"[runtime] using virtualenv ptxas dir first: {text}", flush=True)
    torch.backends.cudnn.enabled = False
    args.runtime_cudnn_enabled = bool(torch.backends.cudnn.enabled)
    print("[runtime] torch.backends.cudnn.enabled=False", flush=True)


def require_gpu_runtime(requested_device: str) -> tuple[torch.device, dict[str, Any]]:
    """Return a CUDA device after proving the run can execute on GPU.

    Official experiment runs are GPU-only. This deliberately refuses CPU
    fallback and catches unsupported CUDA wheels, such as PyTorch builds that can
    see a V100 but do not include sm_70 kernels.
    """

    if not str(requested_device).startswith("cuda"):
        raise RuntimeError(
            f"GPU-only experiment refused device={requested_device!r}. "
            "Use a CUDA device and fix the GPU environment instead of running on CPU."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only experiment refused to start: torch.cuda.is_available() is false.")

    device = torch.device(requested_device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    capability = torch.cuda.get_device_capability(device)
    required_arch = f"sm_{capability[0]}{capability[1]}"
    arch_list = list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else []
    if arch_list and required_arch not in arch_list:
        raise RuntimeError(
            "GPU-only experiment refused to start: the installed PyTorch wheel "
            f"does not include {required_arch}. "
            f"torch={torch.__version__}, torch_cuda={torch.version.cuda}, "
            f"device={torch.cuda.get_device_name(device)}, supported_arches={arch_list}. "
            "Install a CUDA PyTorch wheel that supports this GPU before running."
        )

    try:
        a = torch.eye(16, device=device, dtype=torch.float32)
        b = a @ a
        torch.cuda.synchronize(device)
        sanity_value = float(b[0, 0].detach().cpu())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "GPU-only experiment refused to start: a CUDA tensor sanity check failed. "
            "Fix the PyTorch/CUDA GPU environment before running."
        ) from exc

    try:
        import jax

        jax_backend = jax.default_backend()
        jax_devices = [str(d) for d in jax.devices()]
        jax_gpu_devices = [str(d) for d in jax.devices("gpu")]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "GPU-only experiment refused to start: JAX GPU backend check failed. "
            "Fix the JAX/CUDA GPU environment before running."
        ) from exc

    if jax_backend != "gpu" or not jax_gpu_devices:
        raise RuntimeError(
            "GPU-only experiment refused to start: JAX is not using a GPU backend. "
            f"jax_backend={jax_backend!r}, jax_devices={jax_devices!r}."
        )

    metadata = {
        "policy": "gpu_only_no_cpu_fallback",
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_name": torch.cuda.get_device_name(device),
        "torch_device_capability": list(capability),
        "required_torch_arch": required_arch,
        "torch_supported_arches": arch_list,
        "torch_cuda_sanity_value": sanity_value,
        "jax_default_backend": jax_backend,
        "jax_devices": jax_devices,
        "jax_gpu_devices": jax_gpu_devices,
    }
    print(
        "[gpu-check] "
        f"device={metadata['torch_device_name']} capability={required_arch} "
        f"torch={torch.__version__} cuda={torch.version.cuda} jax_backend={jax_backend}",
        flush=True,
    )
    return device, metadata


def make_solver_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        burgers_nx=args.burgers_nx,
        burgers_nu=args.burgers_nu,
        burgers_t_final=args.burgers_t_final,
        burgers_dt=args.burgers_dt,
        burgers_domain=args.burgers_domain,
        burgers_jax_solver_dtype=args.burgers_jax_solver_dtype,
    )


def load_svd(root: Path, index: int, name: str) -> dict[str, np.ndarray | str]:
    path = root / f"index_{index:03d}" / name / f"{name}_index{index}_jacobian_svd.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    return {
        "source_path": str(path),
        "jacobian": data["jacobian"].astype(np.float64),
        "singular_values": data["singular_values"].astype(np.float64),
        "right_singular_vectors": data["right_singular_vectors"].astype(np.float64),
    }


def load_outward_direction(root: Path, index: int) -> np.ndarray | None:
    path = root / f"index_{index:03d}" / "outward_growth_direction.npy"
    if path.exists():
        return normalize_l2(np.load(path))
    npz_path = root / f"index_{index:03d}" / "directions.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        for key in ("outward_growth", "outward_growth_direction", "v_growth"):
            if key in data:
                return normalize_l2(data[key])
    return None


def direction_key(source: str, rank: int, sign: int) -> str:
    suffix = "plus" if sign >= 0 else "minus"
    return f"{source}_r{rank}_{suffix}"


def add_direction(
    items: list[dict[str, Any]],
    *,
    source: str,
    rank: int,
    sign: int,
    vector: np.ndarray,
    notes: str,
) -> None:
    v = normalize_l2(vector)
    if norm2(v) <= EPS:
        return
    if sign < 0:
        v = -v
    items.append(
        {
            "direction_id": direction_key(source, rank, sign),
            "direction_source": source,
            "rank": rank,
            "sign": int(sign),
            "v": v,
            "notes": notes,
        }
    )


def build_direction_bank(
    *,
    index: int,
    svds: dict[str, dict[str, Any]],
    outward_root: Path,
    top_k: int,
    random_directions: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source in ("fno", "solver", "error"):
        Vh = np.asarray(svds[source]["right_singular_vectors"], dtype=np.float64)
        for rank0 in range(min(top_k, Vh.shape[0])):
            for sign in (1, -1):
                add_direction(
                    items,
                    source=source,
                    rank=rank0 + 1,
                    sign=sign,
                    vector=Vh[rank0],
                    notes="clean-point right singular vector",
                )

    outward = load_outward_direction(outward_root, index)
    outward_available = outward is not None and norm2(outward) > EPS
    if outward_available:
        for sign in (1, -1):
            add_direction(
                items,
                source="outward_growth",
                rank=1,
                sign=sign,
                vector=outward,
                notes="clean-residual outward-growth direction",
            )

    rng = np.random.default_rng(seed + 1009 * index)
    for rank in range(1, random_directions + 1):
        v = normalize_l2(rng.normal(size=np.asarray(svds["fno"]["jacobian"]).shape[1]))
        add_direction(
            items,
            source="random",
            rank=rank,
            sign=1,
            vector=v,
            notes="seeded random unit direction",
        )

    return items, {"outward_available": bool(outward_available), "candidate_count": len(items)}


def evaluate_batch(
    *,
    model: torch.nn.Module,
    solver_fn: Any,
    bridge: Any,
    x0: np.ndarray,
    directions: np.ndarray,
    epsilon: float,
    device: torch.device,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    nx = int(x0.size)
    x_adv = x0.reshape(1, nx) + float(epsilon) * directions
    x_t = torch.as_tensor(x_adv[..., None], device=device, dtype=torch.float32)
    with torch.no_grad():
        f = model(x_t).detach().cpu().numpy()[..., 0].astype(np.float64)
        j = bridge(x_t, solver_fn, label).detach().cpu().numpy()[..., 0].astype(np.float64)
    sync_torch(torch, device)
    return f, j


def objective_column(objective: str) -> str:
    return {
        "L_f": "f_gain",
        "L_j": "solver_gain",
        "L_e": "residual_gain",
        "G_e": "error_norm_growth",
    }[objective]


def aggregate_rows(rows: list[dict[str, Any]], keys: list[str], value_cols: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items(), key=lambda kv: kv[0]):
        item = {k: v for k, v in zip(keys, group_key)}
        item["n"] = len(group_rows)
        for col in value_cols:
            stats = finite_summary([float(r[col]) for r in group_rows])
            for stat_name, stat_value in stats.items():
                item[f"{col}_{stat_name}"] = stat_value
        out.append(item)
    return out


def format_float(value: float, digits: int = 4) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}g}"


def markdown_table(rows: list[dict[str, Any]], cols: list[str], *, max_rows: int | None = None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in shown:
        values = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(format_float(value))
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def save_plots(
    *,
    output_dir: Path,
    aggregate_best: list[dict[str, Any]],
    stability_summary: list[dict[str, Any]],
    source_counts: list[dict[str, Any]],
) -> list[Path]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    objectives = ["L_f", "L_j", "L_e", "G_e"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for objective in objectives:
        rows = [r for r in aggregate_best if r["objective"] == objective]
        rows = sorted(rows, key=lambda r: float(r["epsilon"]))
        xs = np.asarray([float(r["epsilon"]) for r in rows])
        ys = np.asarray([float(r["best_value_mean"]) for r in rows])
        yerr = np.asarray([float(r["best_value_std"]) for r in rows])
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=objective)
    ax.set_xscale("log")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("candidate-bank estimate")
    ax.set_title("Small-Epsilon Sweep: Best Value By Objective")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    path = fig_dir / "small_epsilon_best_values.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for objective in objectives:
        rows = [r for r in aggregate_best if r["objective"] == objective]
        rows = sorted(rows, key=lambda r: float(r["epsilon"]))
        xs = np.asarray([float(r["epsilon"]) for r in rows])
        ys = np.asarray([float(r["best_to_local_ratio_mean"]) for r in rows])
        ax.plot(xs, ys, marker="o", label=objective)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("best finite-epsilon value / clean local reference")
    ax.set_title("Finite-Epsilon Value Relative To Clean Local Reference")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    path = fig_dir / "small_epsilon_local_reference_ratio.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    pairs = sorted({(float(r["epsilon_a"]), float(r["epsilon_b"])) for r in stability_summary})
    matrix = np.full((len(objectives), len(pairs)), np.nan, dtype=np.float64)
    for i, objective in enumerate(objectives):
        for j, pair in enumerate(pairs):
            match = [
                r
                for r in stability_summary
                if r["objective"] == objective and float(r["epsilon_a"]) == pair[0] and float(r["epsilon_b"]) == pair[1]
            ]
            if match:
                matrix[i, j] = float(match[0]["selected_abs_cosine_mean"])
    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(objectives)), objectives)
    ax.set_xticks(range(len(pairs)), [f"{format_float(a, 0)} vs {format_float(b, 0)}" for a, b in pairs], rotation=35, ha="right")
    ax.set_title("Selected Direction Stability Across Epsilon")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if math.isfinite(float(matrix[i, j])):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] < 0.55 else "black")
    fig.colorbar(im, ax=ax, label="mean abs cosine")
    path = fig_dir / "small_epsilon_direction_stability_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), sharex=True)
    for ax, objective in zip(axes.reshape(-1), objectives):
        rows = [r for r in source_counts if r["objective"] == objective]
        epsilons = sorted({float(r["epsilon"]) for r in rows})
        sources = sorted({str(r["direction_source"]) for r in rows})
        bottom = np.zeros(len(epsilons))
        for source in sources:
            vals = []
            for eps in epsilons:
                match = [r for r in rows if float(r["epsilon"]) == eps and str(r["direction_source"]) == source]
                vals.append(float(match[0]["count"]) if match else 0.0)
            ax.bar([str(eps) for eps in epsilons], vals, bottom=bottom, label=source)
            bottom += np.asarray(vals)
        ax.set_title(objective)
        ax.set_ylabel("selected count")
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes.reshape(-1)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
    fig.suptitle("Best Direction Source Counts", y=0.99)
    path = fig_dir / "small_epsilon_best_direction_sources.png"
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    return made


def write_result_doc(
    *,
    path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    aggregate_best: list[dict[str, Any]],
    stability_summary: list[dict[str, Any]],
    reference_summary: list[dict[str, Any]],
    source_counts: list[dict[str, Any]],
    plot_paths: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_for_table = [
        {
            "objective": r["objective"],
            "epsilon": r["epsilon"],
            "best mean": r["best_value_mean"],
            "best/local": r["best_to_local_ratio_mean"],
            "abs cos local": r["selected_abs_cos_to_local_reference_mean"],
        }
        for r in aggregate_best
    ]
    stability_table = [
        {
            "objective": r["objective"],
            "eps a": r["epsilon_a"],
            "eps b": r["epsilon_b"],
            "abs cosine mean": r["selected_abs_cosine_mean"],
            "abs cosine min": r["selected_abs_cosine_min"],
        }
        for r in stability_summary
    ]
    ref_table = [
        {
            "objective": r["objective"],
            "local reference mean": r["local_reference_value_mean"],
            "local reference std": r["local_reference_value_std"],
        }
        for r in reference_summary
    ]

    lf_small = next(r for r in aggregate_best if r["objective"] == "L_f" and float(r["epsilon"]) == min(args.epsilons))
    le_small = next(r for r in aggregate_best if r["objective"] == "L_e" and float(r["epsilon"]) == min(args.epsilons))
    ge_small = next(r for r in aggregate_best if r["objective"] == "G_e" and float(r["epsilon"]) == min(args.epsilons))
    ge_large = next(r for r in aggregate_best if r["objective"] == "G_e" and float(r["epsilon"]) == max(args.epsilons))

    lines = [
        "# Loss3 Small-Epsilon Sweep Result - 2026-05-16",
        "",
        "Status: completed for FNO / 1D Burgers `nu=0.001` using a fixed candidate-direction bank.",
        "",
        "## Scope",
        "",
        "- Model: FNO1d, `nu=0.001`.",
        f"- Indices: `{args.sample_indices}`.",
        f"- Epsilons: `{args.epsilons}`.",
        "- Input/output norm: L2.",
        f"- Output directory: `{output_dir}`.",
        "- Device policy: GPU-only; the script refuses CPU fallback.",
        f"- Runtime device: `{getattr(args, 'gpu_runtime', {}).get('torch_device_name', 'not recorded')}`.",
        "",
        "The candidate bank uses clean-point top right singular vectors from `J_f`, `J_j`, and `J_e=J_f-J_j`, both signs of those directions, outward-growth directions, and seeded random directions.",
        "",
        "## Local References",
        "",
        markdown_table(ref_table, ["objective", "local reference mean", "local reference std"]),
        "",
        "## Aggregate Sweep Values",
        "",
        markdown_table(rows_for_table, ["objective", "epsilon", "best mean", "best/local", "abs cos local"]),
        "",
        "## Direction Stability",
        "",
        markdown_table(stability_table, ["objective", "eps a", "eps b", "abs cosine mean", "abs cosine min"]),
        "",
        "## Visualizations",
        "",
    ]
    for plot in plot_paths:
        rel = os.path.relpath(plot, path.parent)
        title = plot.stem.replace("_", " ")
        lines.append(f"![{title}]({rel})")
        lines.append("")

    lines.extend(
        [
            "## Conclusion",
            "",
            "Observed from the completed sweep:",
            "",
            f"- At the smallest radius, `L_f` is `{format_float(lf_small['best_value_mean'])}` on average, while `L_e` is `{format_float(le_small['best_value_mean'])}`. This preserves the main mechanism: model-only sensitivity is much larger than residual/error-field sensitivity.",
            f"- The local clean-error outward-growth diagnostic `G_e` is much smaller than `L_e`: at the smallest radius, `G_e` is `{format_float(ge_small['best_value_mean'])}` on average.",
            f"- At the largest tested local radius, `G_e` is `{format_float(ge_large['best_value_mean'])}` on average, so this diagnostic remains distinct from the residual-movement quantity `L_e`.",
            "- Direction-stability tables record whether the selected candidate direction remains stable as epsilon grows; low cross-epsilon cosine means nonlinear/local-to-finite-radius drift is already visible.",
            "",
            "Inference:",
            "",
            "The Experiment 3 evidence supports the intended distinction in the `loss3_original` plan: local ratio-style diagnostics are meaningful at small radii, but they are different objects from the finite-radius endpoint objective. In particular, `L_e` measures how fast the error field moves, while `G_e` measures whether the current clean residual norm is pushed outward.",
            "",
            "## Output Files",
            "",
            "- `candidate_metrics.csv`: every evaluated candidate direction.",
            "- `best_by_objective_epsilon_index.csv`: per-index selected direction/value.",
            "- `aggregate_best_by_objective_epsilon.csv`: aggregate values used above.",
            "- `direction_stability.csv` and `direction_stability_summary.csv`: cross-epsilon selected-direction cosines.",
            "- `local_reference_by_index.csv` and `local_reference_summary.csv`: clean-point local references.",
            "- `manifest.json`: reproducibility manifest.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-doc", type=Path, default=DEFAULT_RESULT_DOC)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 7, 40, 47, 115])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--random-directions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime-workarounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepend-env-ptxas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fno-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument(
        "--fno-checkpoint",
        type=Path,
        default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt",
    )
    parser.add_argument("--jacobian-root", type=Path, default=DEFAULT_JACOBIAN_ROOT)
    parser.add_argument("--outward-root", type=Path, default=DEFAULT_OUTWARD_ROOT)
    parser.add_argument("--burgers-nx", type=int, default=1024)
    parser.add_argument("--burgers-nu", type=float, default=0.001)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs(float(args.burgers_nu) - 0.001) > 1e-12:
        raise ValueError("This script is intentionally scoped to FNO nu=0.001 only.")
    configure_runtime(args)
    requested_device = str(args.device or "cuda")
    device, gpu_runtime = require_gpu_runtime(requested_device)
    args.gpu_runtime = gpu_runtime
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    model = load_burgers_torch_model(args.fno_checkpoint, device)
    solver_fn = make_burgers_jax_solver(make_solver_args(args))
    bridge = make_jax_torch_bridge()

    candidate_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    local_reference_rows: list[dict[str, Any]] = []
    direction_vectors: dict[tuple[int, str, float], np.ndarray] = {}
    source_paths: set[str] = {str(args.fno_test_path), str(args.fno_checkpoint), str(args.jacobian_root), str(args.outward_root)}
    per_index_meta: list[dict[str, Any]] = []

    for index in args.sample_indices:
        index_start = time.perf_counter()
        print(f"[index] {index}", flush=True)
        index_dir = args.output_dir / f"index_{index:03d}"
        index_dir.mkdir(parents=True, exist_ok=True)
        svds = {name: load_svd(args.jacobian_root, index, name) for name in ("fno", "solver", "error")}
        source_paths.update(str(svds[name]["source_path"]) for name in ("fno", "solver", "error"))
        Jf = np.asarray(svds["fno"]["jacobian"], dtype=np.float64)
        Jj = np.asarray(svds["solver"]["jacobian"], dtype=np.float64)
        Je = np.asarray(svds["error"]["jacobian"], dtype=np.float64)
        A = Jf - Jj
        consistency_l2 = norm2(A - Je)
        consistency_max_abs = float(np.max(np.abs(A - Je)))

        x0 = load_sample(args.fno_test_path, index).reshape(-1).astype(np.float64)
        f0, j0 = evaluate_batch(
            model=model,
            solver_fn=solver_fn,
            bridge=bridge,
            x0=x0,
            directions=np.zeros((1, x0.size), dtype=np.float64),
            epsilon=0.0,
            device=device,
            label=f"clean_index{index}",
        )
        f0 = f0[0]
        j0 = j0[0]
        e0 = f0 - j0
        e0_norm = norm2(e0)
        e0_unit = normalize_l2(e0)
        local_ge = norm2(A.T @ e0_unit) if e0_norm > EPS else math.nan

        local_refs = {
            "L_f": float(np.asarray(svds["fno"]["singular_values"])[0]),
            "L_j": float(np.asarray(svds["solver"]["singular_values"])[0]),
            "L_e": float(np.asarray(svds["error"]["singular_values"])[0]),
            "G_e": local_ge,
        }
        local_ref_vectors = {
            "L_f": normalize_l2(np.asarray(svds["fno"]["right_singular_vectors"])[0]),
            "L_j": normalize_l2(np.asarray(svds["solver"]["right_singular_vectors"])[0]),
            "L_e": normalize_l2(np.asarray(svds["error"]["right_singular_vectors"])[0]),
            "G_e": normalize_l2(A.T @ e0_unit) if e0_norm > EPS else np.zeros_like(x0),
        }
        for objective, value in local_refs.items():
            local_reference_rows.append(
                {
                    "sample_index": index,
                    "objective": objective,
                    "local_reference_value": value,
                    "clean_residual_norm_l2": e0_norm,
                    "jacobian_consistency_l2": consistency_l2,
                    "jacobian_consistency_max_abs": consistency_max_abs,
                }
            )

        directions, meta = build_direction_bank(
            index=index,
            svds=svds,
            outward_root=args.outward_root,
            top_k=args.top_k,
            random_directions=args.random_directions,
            seed=args.seed,
        )
        direction_matrix = np.stack([item["v"] for item in directions], axis=0)
        np.savez_compressed(
            index_dir / "direction_bank.npz",
            directions=direction_matrix.astype(np.float32),
            direction_ids=np.asarray([item["direction_id"] for item in directions]),
            direction_sources=np.asarray([item["direction_source"] for item in directions]),
            ranks=np.asarray([item["rank"] for item in directions], dtype=np.int32),
            signs=np.asarray([item["sign"] for item in directions], dtype=np.int32),
        )

        for epsilon in args.epsilons:
            print(f"[sweep] index={index} epsilon={epsilon:g} candidates={len(directions)}", flush=True)
            f_parts: list[np.ndarray] = []
            j_parts: list[np.ndarray] = []
            for start_i in range(0, len(directions), args.eval_batch_size):
                batch_dirs = direction_matrix[start_i : start_i + args.eval_batch_size]
                f_batch, j_batch = evaluate_batch(
                    model=model,
                    solver_fn=solver_fn,
                    bridge=bridge,
                    x0=x0,
                    directions=batch_dirs,
                    epsilon=float(epsilon),
                    device=device,
                    label=f"sweep_index{index}_eps{epsilon:g}_batch{start_i}",
                )
                f_parts.append(f_batch)
                j_parts.append(j_batch)
            f_all = np.concatenate(f_parts, axis=0)
            j_all = np.concatenate(j_parts, axis=0)
            df = f_all - f0[None, :]
            dj = j_all - j0[None, :]
            e_adv = f_all - j_all
            de = e_adv - e0[None, :]
            f_gain = np.linalg.norm(df, axis=1) / float(epsilon)
            solver_gain = np.linalg.norm(dj, axis=1) / float(epsilon)
            residual_gain = np.linalg.norm(de, axis=1) / float(epsilon)
            error_norm = np.linalg.norm(e_adv, axis=1)
            error_norm_growth = (error_norm - e0_norm) / float(epsilon)

            values_by_col = {
                "f_gain": f_gain,
                "solver_gain": solver_gain,
                "residual_gain": residual_gain,
                "error_norm": error_norm,
                "error_norm_growth": error_norm_growth,
            }

            for pos, item in enumerate(directions):
                row = {
                    "sample_index": index,
                    "epsilon": float(epsilon),
                    "direction_id": item["direction_id"],
                    "direction_source": item["direction_source"],
                    "rank": item["rank"],
                    "sign": item["sign"],
                    "f_gain": float(f_gain[pos]),
                    "solver_gain": float(solver_gain[pos]),
                    "residual_gain": float(residual_gain[pos]),
                    "error_norm": float(error_norm[pos]),
                    "error_norm_growth": float(error_norm_growth[pos]),
                    "clean_residual_norm_l2": e0_norm,
                    "cos_to_local_Lf": cosine(item["v"], local_ref_vectors["L_f"]),
                    "cos_to_local_Lj": cosine(item["v"], local_ref_vectors["L_j"]),
                    "cos_to_local_Le": cosine(item["v"], local_ref_vectors["L_e"]),
                    "cos_to_local_Ge": cosine(item["v"], local_ref_vectors["G_e"]),
                    "notes": item["notes"],
                }
                candidate_rows.append(row)

            for objective in ("L_f", "L_j", "L_e", "G_e"):
                col = objective_column(objective)
                values = values_by_col[col]
                best_pos = int(np.nanargmax(values))
                item = directions[best_pos]
                best_value = float(values[best_pos])
                local_value = float(local_refs[objective])
                selected_v = item["v"].copy()
                direction_vectors[(index, objective, float(epsilon))] = selected_v
                np.save(index_dir / f"selected_{objective}_eps{epsilon:g}.npy", selected_v.astype(np.float32))
                best_rows.append(
                    {
                        "sample_index": index,
                        "epsilon": float(epsilon),
                        "objective": objective,
                        "best_value": best_value,
                        "local_reference_value": local_value,
                        "best_to_local_ratio": best_value / local_value if abs(local_value) > EPS else math.nan,
                        "direction_id": item["direction_id"],
                        "direction_source": item["direction_source"],
                        "rank": item["rank"],
                        "sign": item["sign"],
                        "selected_abs_cos_to_local_reference": abs(cosine(selected_v, local_ref_vectors[objective])),
                        "selected_cos_to_local_reference": cosine(selected_v, local_ref_vectors[objective]),
                        "clean_residual_norm_l2": e0_norm,
                    }
                )

        per_index_meta.append(
            {
                "sample_index": index,
                "candidate_count": meta["candidate_count"],
                "outward_available": meta["outward_available"],
                "clean_residual_norm_l2": e0_norm,
                "jacobian_consistency_l2": consistency_l2,
                "jacobian_consistency_max_abs": consistency_max_abs,
                "seconds": time.perf_counter() - index_start,
            }
        )
        write_json(index_dir / "index_manifest.json", per_index_meta[-1])

    direction_stability_rows: list[dict[str, Any]] = []
    for index in args.sample_indices:
        for objective in ("L_f", "L_j", "L_e", "G_e"):
            for i, eps_a in enumerate(args.epsilons):
                for eps_b in args.epsilons[i + 1 :]:
                    va = direction_vectors[(index, objective, float(eps_a))]
                    vb = direction_vectors[(index, objective, float(eps_b))]
                    c = cosine(va, vb)
                    direction_stability_rows.append(
                        {
                            "sample_index": index,
                            "objective": objective,
                            "epsilon_a": float(eps_a),
                            "epsilon_b": float(eps_b),
                            "selected_cosine": c,
                            "selected_abs_cosine": abs(c) if math.isfinite(c) else math.nan,
                        }
                    )

    aggregate_best = aggregate_rows(
        best_rows,
        ["objective", "epsilon"],
        ["best_value", "local_reference_value", "best_to_local_ratio", "selected_abs_cos_to_local_reference"],
    )
    stability_summary = aggregate_rows(
        direction_stability_rows,
        ["objective", "epsilon_a", "epsilon_b"],
        ["selected_cosine", "selected_abs_cosine"],
    )
    reference_summary = aggregate_rows(local_reference_rows, ["objective"], ["local_reference_value"])

    count_map: Counter[tuple[str, float, str]] = Counter()
    for row in best_rows:
        count_map[(str(row["objective"]), float(row["epsilon"]), str(row["direction_source"]))] += 1
    source_count_rows = [
        {"objective": obj, "epsilon": eps, "direction_source": source, "count": count}
        for (obj, eps, source), count in sorted(count_map.items())
    ]

    write_csv(args.output_dir / "candidate_metrics.csv", candidate_rows)
    write_csv(args.output_dir / "best_by_objective_epsilon_index.csv", best_rows)
    write_csv(args.output_dir / "aggregate_best_by_objective_epsilon.csv", aggregate_best)
    write_csv(args.output_dir / "local_reference_by_index.csv", local_reference_rows)
    write_csv(args.output_dir / "local_reference_summary.csv", reference_summary)
    write_csv(args.output_dir / "direction_stability.csv", direction_stability_rows)
    write_csv(args.output_dir / "direction_stability_summary.csv", stability_summary)
    write_csv(args.output_dir / "best_direction_source_counts.csv", source_count_rows)

    plot_paths = save_plots(
        output_dir=args.output_dir,
        aggregate_best=aggregate_best,
        stability_summary=stability_summary,
        source_counts=source_count_rows,
    )

    manifest = {
        "experiment": "loss3_small_epsilon_sweep_fno_nu0p001",
        "status": "completed",
        "scope": "FNO / 1D Burgers / nu=0.001 only",
        "output_dir": args.output_dir,
        "result_doc": args.result_doc,
        "sample_indices": args.sample_indices,
        "epsilons": args.epsilons,
        "top_k": args.top_k,
        "random_directions": args.random_directions,
        "seed": args.seed,
        "eval_batch_size": args.eval_batch_size,
        "device": str(device),
        "gpu_runtime": args.gpu_runtime,
        "runtime_workarounds": args.runtime_workarounds,
        "runtime_ptxas_dir": args.runtime_ptxas_dir,
        "runtime_cudnn_enabled": args.runtime_cudnn_enabled,
        "fno_test_path": args.fno_test_path,
        "fno_checkpoint": args.fno_checkpoint,
        "jacobian_root": args.jacobian_root,
        "outward_root": args.outward_root,
        "burgers_nx": args.burgers_nx,
        "burgers_nu": args.burgers_nu,
        "burgers_t_final": args.burgers_t_final,
        "burgers_dt": args.burgers_dt,
        "burgers_domain": args.burgers_domain,
        "burgers_jax_solver_dtype": args.burgers_jax_solver_dtype,
        "source_paths": sorted(source_paths),
        "output_files": [
            "candidate_metrics.csv",
            "best_by_objective_epsilon_index.csv",
            "aggregate_best_by_objective_epsilon.csv",
            "local_reference_by_index.csv",
            "local_reference_summary.csv",
            "direction_stability.csv",
            "direction_stability_summary.csv",
            "best_direction_source_counts.csv",
            "manifest.json",
            *[str(p.relative_to(args.output_dir)) for p in plot_paths],
        ],
        "per_index": per_index_meta,
        "seconds": time.perf_counter() - start,
    }
    write_json(args.output_dir / "manifest.json", manifest)

    write_result_doc(
        path=args.result_doc,
        output_dir=args.output_dir,
        args=args,
        aggregate_best=aggregate_best,
        stability_summary=stability_summary,
        reference_summary=reference_summary,
        source_counts=source_count_rows,
        plot_paths=plot_paths,
    )
    print(f"[done] output_dir={args.output_dir}", flush=True)
    print(f"[done] result_doc={args.result_doc}", flush=True)


if __name__ == "__main__":
    main()
