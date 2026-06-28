#!/usr/bin/env python3
"""Pathwise residual-Jacobian top-k/subspace rotation diagnostic.

Scope: FNO / 1D Burgers / nu=0.001 only.

This expands Experiment 6 beyond the top-1 direction.  Along

    x_t = x + t delta*

it estimates the top right singular subspace of J_e(x_t)=J_f(x_t)-J_j(x_t)
using a randomized SVD sketch.  It records top-1 angles, top-k principal angles,
top singular values, and random-probe response-sketch similarity to the clean and
previous path points.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
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
from tools.run_loss3_small_epsilon_sweep import (  # noqa: E402
    DEFAULT_JACOBIAN_ROOT,
    configure_runtime,
    finite_json,
    finite_summary,
    load_svd,
    normalize_l2,
    require_gpu_runtime,
)

DEFAULT_DELTA_NPZ = (
    PROJECT_ROOT
    / "forensics"
    / "loss3_ray_profile_corrected_20260516"
    / "fno_nu0p001_gpu_v100_batch20"
    / "deltas.npz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forensics" / "loss3_jacobian_subspace_rotation_path_20260516" / "fno_nu0p001"
DEFAULT_RESULT_DOC = PROJECT_ROOT / "docs" / "loss3_jacobian_subspace_rotation_path_fno_nu0p001_result_20260516.md"
EPS = 1e-12


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


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return math.nan
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = norm2(af) * norm2(bf)
    if denom <= EPS:
        return math.nan
    return float(np.dot(af, bf) / denom)


def angle_from_abs_cos(abs_cos: float) -> float:
    if not math.isfinite(abs_cos):
        return math.nan
    return float(np.degrees(np.arccos(np.clip(abs_cos, 0.0, 1.0))))


def fro_cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return math.nan
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom <= EPS:
        return math.nan
    return float(np.dot(af, bf) / denom)


def rel_fro_diff(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return math.nan
    af = np.asarray(a, dtype=np.float64)
    bf = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(bf))
    if denom <= EPS:
        return math.nan
    return float(np.linalg.norm(af - bf) / denom)


def subspace_metrics(current_vh: np.ndarray, reference_vh: np.ndarray, k: int) -> dict[str, float]:
    k = min(int(k), current_vh.shape[0], reference_vh.shape[0])
    if k <= 0:
        return {"min_cos": math.nan, "mean_cos": math.nan, "max_angle_deg": math.nan, "mean_angle_deg": math.nan}
    qc = np.asarray(current_vh[:k], dtype=np.float64).T
    qr = np.asarray(reference_vh[:k], dtype=np.float64).T
    _, s, _ = np.linalg.svd(qc.T @ qr, full_matrices=False)
    s = np.clip(s, 0.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return {
        "min_cos": float(np.min(s)),
        "mean_cos": float(np.mean(s)),
        "max_angle_deg": float(np.max(angles)),
        "mean_angle_deg": float(np.mean(angles)),
    }


def make_solver_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        burgers_nx=args.burgers_nx,
        burgers_nu=args.burgers_nu,
        burgers_t_final=args.burgers_t_final,
        burgers_dt=args.burgers_dt,
        burgers_domain=args.burgers_domain,
        burgers_jax_solver_dtype=args.burgers_jax_solver_dtype,
    )


def load_delta_map(path: Path, source: str) -> tuple[dict[int, np.ndarray], list[int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    if source not in data:
        raise KeyError(f"{source!r} not found in {path}; available keys: {list(data.keys())}")
    sample_indices = [int(x) for x in data["sample_indices"].tolist()]
    deltas = np.asarray(data[source], dtype=np.float32)
    return {idx: deltas[pos].reshape(-1).astype(np.float64) for pos, idx in enumerate(sample_indices)}, sample_indices


def make_path_fractions(num_points: int) -> list[float]:
    if num_points < 2:
        raise ValueError("num_path_points must be >= 2")
    return [float(x) for x in np.linspace(0.0, 1.0, int(num_points))]


def orthonormal_probes(n: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(n, int(count)))
    q, _ = np.linalg.qr(omega)
    return q[:, : int(count)].astype(np.float64)


def residual_output(model: torch.nn.Module, bridge: Any, solver_fn: Any, x: torch.Tensor, label: str) -> torch.Tensor:
    return model(x) - bridge(x, solver_fn, label)


def finite_difference_responses(
    *,
    model: torch.nn.Module,
    bridge: Any,
    solver_fn: Any,
    x_state_np: np.ndarray,
    probes: np.ndarray,
    epsilon: float,
    device: torch.device,
    label: str,
) -> np.ndarray:
    n, probe_count = probes.shape
    x0 = torch.as_tensor(x_state_np.reshape(1, n, 1), device=device, dtype=torch.float32)
    x_batch_np = x_state_np.reshape(1, n) + float(epsilon) * probes.T
    x_batch = torch.as_tensor(x_batch_np[..., None], device=device, dtype=torch.float32)
    with torch.no_grad():
        e0 = residual_output(model, bridge, solver_fn, x0, label=f"{label}_base").detach()
        eb = residual_output(model, bridge, solver_fn, x_batch, label=f"{label}_batch").detach()
    sync_torch(torch, device)
    y = ((eb - e0).detach().cpu().numpy()[..., 0].astype(np.float64) / float(epsilon)).T
    if y.shape != (n, probe_count):
        raise RuntimeError(f"Expected response sketch {(n, probe_count)}, got {y.shape}")
    return y


def residual_vjp_rows(
    *,
    model: torch.nn.Module,
    bridge: Any,
    solver_fn: Any,
    x_state_np: np.ndarray,
    q_output: np.ndarray,
    device: torch.device,
    label: str,
) -> np.ndarray:
    n, q_count = q_output.shape
    x = torch.as_tensor(x_state_np.reshape(1, n, 1), device=device, dtype=torch.float32).detach().requires_grad_(True)
    e = residual_output(model, bridge, solver_fn, x, label=f"{label}_graph").reshape(-1)
    rows = []
    for col in range(q_count):
        q = torch.as_tensor(q_output[:, col], device=device, dtype=torch.float32)
        scalar = torch.sum(e * q)
        grad = torch.autograd.grad(scalar, x, retain_graph=True, create_graph=False, allow_unused=False)[0]
        rows.append(grad.detach().cpu().numpy().reshape(-1).astype(np.float64))
    del e, x
    sync_torch(torch, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.stack(rows, axis=0)


def randomized_residual_svd(
    *,
    model: torch.nn.Module,
    bridge: Any,
    solver_fn: Any,
    x_state_np: np.ndarray,
    probes: np.ndarray,
    epsilon: float,
    max_rank: int,
    device: torch.device,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = finite_difference_responses(
        model=model,
        bridge=bridge,
        solver_fn=solver_fn,
        x_state_np=x_state_np,
        probes=probes,
        epsilon=epsilon,
        device=device,
        label=f"{label}_fd",
    )
    q, _ = np.linalg.qr(y)
    b = residual_vjp_rows(
        model=model,
        bridge=bridge,
        solver_fn=solver_fn,
        x_state_np=x_state_np,
        q_output=q,
        device=device,
        label=f"{label}_vjp",
    )
    _u_hat, s, vh = np.linalg.svd(b, full_matrices=False)
    rank = min(int(max_rank), vh.shape[0], s.shape[0])
    return s[:rank].astype(np.float64), vh[:rank].astype(np.float64), y.astype(np.float64)


def aggregate_rows(rows: list[dict[str, Any]], keys: list[str], value_cols: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        item = {k: v for k, v in zip(keys, group_key)}
        item["n"] = len(group_rows)
        for col in value_cols:
            stats = finite_summary([float(r.get(col, math.nan)) for r in group_rows])
            for stat_name, stat_value in stats.items():
                item[f"{col}_{stat_name}"] = stat_value
        out.append(item)
    return out


def plot_outputs(out_dir: Path, aggregate: list[dict[str, Any]], ks: list[int]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray([float(r["path_fraction"]) for r in aggregate], dtype=np.float64)

    def plot_col(col: str, title: str, ylabel: str, filename: str) -> None:
        mean = np.asarray([float(r.get(f"{col}_mean", math.nan)) for r in aggregate], dtype=np.float64)
        std = np.asarray([float(r.get(f"{col}_std", math.nan)) for r in aggregate], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(x, mean, marker="o", linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18)
        ax.set_title(title)
        ax.set_xlabel("path fraction t")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.28)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)

    plot_col("top1_angle_to_clean_deg", "Top-1 residual-Jacobian direction rotation", "angle to clean, deg", "top1_angle_to_clean_vs_t.png")
    plot_col("spectral_norm", "Estimated residual-Jacobian spectral norm", "sigma1", "spectral_norm_vs_t.png")
    plot_col("response_sketch_rel_diff_to_clean", "Random-probe residual-Jacobian sketch change", "relative Frobenius difference", "response_sketch_rel_diff_to_clean_vs_t.png")
    for k in ks:
        plot_col(f"subspace{k}_clean_max_angle_deg", f"Top-{k} subspace rotation from clean", "max principal angle, deg", f"top{k}_subspace_clean_max_angle_vs_t.png")


def fnum(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(v):
        return "nan"
    if abs(v) != 0 and (abs(v) < 1e-4 or abs(v) >= 1e5):
        return f"{v:.3e}"
    return f"{v:.{digits}f}".rstrip("0").rstrip(".")


def markdown_table(rows: list[dict[str, Any]], columns: list[str], labels: list[str] | None = None) -> str:
    labels = labels or columns
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fnum(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_result_doc(path: Path, args: argparse.Namespace, manifest: dict[str, Any], aggregate: list[dict[str, Any]], sample_summary: list[dict[str, Any]]) -> None:
    final = next((r for r in aggregate if abs(float(r["path_fraction"]) - 1.0) < 1e-9), aggregate[-1])
    lines = [
        "# Loss3 Jacobian Subspace Rotation Along Path Result",
        "",
        "Date: 2026-05-16 UTC",
        "",
        "Scope: FNO / 1D Burgers `nu=0.001` only.",
        "",
        "## Purpose",
        "",
        "This is the expanded Experiment 6 diagnostic. It walks along `x_t = x + t delta*` and estimates the residual Jacobian `J_e(x_t)=J_f(x_t)-J_j(x_t)` with a randomized SVD sketch. Unlike the earlier top-1-only pilot, this records top-k subspace angles, singular values, and a random-probe response sketch for broader Jacobian comparison.",
        "",
        "The `path_fraction` column is normalized progress from the clean input to the endpoint `x + delta*`; it is not physical PDE time. The actual distance traveled is recorded as `path_delta_norm_l2 = t ||delta*||_2`.",
        "",
        "## Settings",
        "",
        f"- Output directory: `{args.output_dir}`",
        f"- Delta source: `{args.delta_npz}` key `{args.delta_source}`",
        f"- Sample indices: `{args.sample_indices}`",
        f"- Path points: `{args.num_path_points}` fractions from 0 to 1",
        f"- Finite-difference epsilon: `{args.fd_epsilon:g}`",
        f"- Randomized SVD rank: `{args.max_rank}`; oversample: `{args.oversample}`; probe count: `{args.max_rank + args.oversample}`",
        f"- Subspace k values: `{args.subspace_ks}`",
        f"- Runtime seconds: `{manifest['seconds']:.1f}`",
        f"- GPU: `{manifest['gpu_runtime']['torch_device_name']}`",
        "",
        "## Main Result",
        "",
        f"At `t=1`, the mean top-1 angle to the clean top direction is `{final['top1_angle_to_clean_deg_mean']:.2f}` degrees.",
        f"At `t=1`, the mean top-4 max principal angle to the clean top-4 subspace is `{final.get('subspace4_clean_max_angle_deg_mean', math.nan):.2f}` degrees, and the mean top-8 max principal angle is `{final.get('subspace8_clean_max_angle_deg_mean', math.nan):.2f}` degrees.",
        f"At `t=1`, the mean estimated spectral norm `sigma1(J_e(x_t))` is `{final['spectral_norm_mean']:.4g}`.",
        "",
        "## Aggregate By Path Fraction",
        "",
        markdown_table(
            aggregate,
            [
                "path_fraction", "n", "path_delta_norm_l2_mean",
                "top1_angle_to_clean_deg_mean", "top1_angle_to_previous_deg_mean",
                "subspace2_clean_max_angle_deg_mean", "subspace4_clean_max_angle_deg_mean", "subspace8_clean_max_angle_deg_mean",
                "spectral_norm_mean", "sigma2_over_sigma1_mean",
                "response_sketch_cos_to_clean_mean", "response_sketch_rel_diff_to_clean_mean",
                "response_sketch_cos_to_previous_mean", "response_sketch_rel_diff_to_previous_mean",
            ],
            labels=[
                "t", "n", "mean ||t delta*||2",
                "top1 angle clean", "top1 angle prev",
                "top2 max angle", "top4 max angle", "top8 max angle",
                "sigma1", "sigma2/sigma1",
                "sketch cos clean", "sketch rel diff clean",
                "sketch cos prev", "sketch rel diff prev",
            ],
        ),
        "",
        "## Per-Sample Endpoint Summary",
        "",
        markdown_table(
            sample_summary,
            [
                "sample_index", "endpoint_top1_angle_to_clean_deg", "endpoint_subspace4_clean_max_angle_deg", "endpoint_subspace8_clean_max_angle_deg",
                "endpoint_spectral_norm", "endpoint_response_sketch_rel_diff_to_clean",
                "max_top1_angle_to_clean_deg", "mean_top1_angle_to_previous_deg_nonzero",
            ],
            labels=[
                "sample", "endpoint top1 angle", "endpoint top4 max angle", "endpoint top8 max angle",
                "endpoint sigma1", "endpoint sketch rel diff",
                "max top1 angle", "mean top1 prev angle",
            ],
        ),
        "",
        "## Files",
        "",
        "- `jacobian_subspace_rotation_by_sample_t.csv`",
        "- `jacobian_subspace_rotation_aggregate_by_t.csv`",
        "- `jacobian_subspace_rotation_summary_by_sample.csv`",
        "- `singular_values_by_sample_t.csv`",
        "- `manifest.json`",
        "- `figures/top1_angle_to_clean_vs_t.png`",
        "- `figures/spectral_norm_vs_t.png`",
        "- `figures/response_sketch_rel_diff_to_clean_vs_t.png`",
        "- `figures/top{k}_subspace_clean_max_angle_vs_t.png` for requested k values",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 7, 40, 47, 115])
    parser.add_argument("--num-path-points", type=int, default=11)
    parser.add_argument("--max-rank", type=int, default=8)
    parser.add_argument("--oversample", type=int, default=4)
    parser.add_argument("--subspace-ks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--fd-epsilon", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-doc", type=Path, default=DEFAULT_RESULT_DOC)
    parser.add_argument("--delta-npz", type=Path, default=DEFAULT_DELTA_NPZ)
    parser.add_argument("--delta-source", default="loss3_original_pgd_best")
    parser.add_argument("--jacobian-root", type=Path, default=DEFAULT_JACOBIAN_ROOT)
    parser.add_argument("--fno-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument("--fno-checkpoint", type=Path, default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt")
    parser.add_argument("--burgers-nx", type=int, default=1024)
    parser.add_argument("--burgers-nu", type=float, default=0.001)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=1e-4)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--runtime-workarounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepend-env-ptxas", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    configure_runtime(args)
    device, gpu_metadata = require_gpu_runtime(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    delta_map, delta_sample_indices = load_delta_map(args.delta_npz, args.delta_source)
    missing = [idx for idx in args.sample_indices if idx not in delta_map]
    if missing:
        raise RuntimeError(f"Missing requested sample indices in {args.delta_npz}: {missing}")

    model = load_burgers_torch_model(args.fno_checkpoint, device)
    solver_fn = make_burgers_jax_solver(make_solver_args(args))
    bridge = make_jax_torch_bridge()
    path_fractions = make_path_fractions(args.num_path_points)
    probe_count = int(args.max_rank + args.oversample)

    rows: list[dict[str, Any]] = []
    sv_rows: list[dict[str, Any]] = []
    source_paths = {str(args.fno_checkpoint), str(args.fno_test_path), str(args.delta_npz), str(args.jacobian_root)}

    for index in args.sample_indices:
        print(f"[index] {index}", flush=True)
        x0 = load_sample(args.fno_test_path, index).reshape(-1).astype(np.float64)
        delta = delta_map[index].reshape(-1).astype(np.float64)
        delta_norm = norm2(delta)
        clean_svd = load_svd(args.jacobian_root, index, "error")
        source_paths.add(str(clean_svd["source_path"]))
        clean_vh_exact = np.asarray(clean_svd["right_singular_vectors"], dtype=np.float64)[: args.max_rank]
        clean_s = np.asarray(clean_svd["singular_values"], dtype=np.float64)[: args.max_rank]
        probes = orthonormal_probes(x0.size, probe_count, args.seed + 7919 * index)
        clean_y: np.ndarray | None = None
        previous_vh: np.ndarray | None = None
        previous_y: np.ndarray | None = None

        for frac in path_fractions:
            print(f"[path] index={index} t={frac:.3f}", flush=True)
            x_state = x0 + float(frac) * delta
            y = finite_difference_responses(
                model=model,
                bridge=bridge,
                solver_fn=solver_fn,
                x_state_np=x_state,
                probes=probes,
                epsilon=args.fd_epsilon,
                device=device,
                label=f"idx{index}_t{frac:.3f}_sketch",
            )
            if abs(float(frac)) < 1e-12:
                s = clean_s.copy()
                vh = clean_vh_exact.copy()
                clean_y = y.copy()
            else:
                q, _ = np.linalg.qr(y)
                b = residual_vjp_rows(
                    model=model,
                    bridge=bridge,
                    solver_fn=solver_fn,
                    x_state_np=x_state,
                    q_output=q,
                    device=device,
                    label=f"idx{index}_t{frac:.3f}",
                )
                _u_hat, s_all, vh_all = np.linalg.svd(b, full_matrices=False)
                rank = min(args.max_rank, len(s_all), vh_all.shape[0])
                s = s_all[:rank].astype(np.float64)
                vh = vh_all[:rank].astype(np.float64)

            if previous_vh is not None and cosine(vh[0], previous_vh[0]) < 0:
                vh = vh.copy()
                vh[0] = -vh[0]

            top1_abs_cos_clean = abs(cosine(vh[0], clean_vh_exact[0]))
            top1_abs_cos_prev = abs(cosine(vh[0], previous_vh[0])) if previous_vh is not None else math.nan
            row: dict[str, Any] = {
                "sample_index": int(index),
                "path_fraction": float(frac),
                "path_delta_norm_l2": float(frac) * delta_norm,
                "delta_star_norm_l2": delta_norm,
                "fd_epsilon": float(args.fd_epsilon),
                "probe_count": probe_count,
                "top1_abs_cos_to_clean": top1_abs_cos_clean,
                "top1_angle_to_clean_deg": angle_from_abs_cos(top1_abs_cos_clean),
                "top1_abs_cos_to_previous": top1_abs_cos_prev,
                "top1_angle_to_previous_deg": angle_from_abs_cos(top1_abs_cos_prev),
                "spectral_norm": float(s[0]) if len(s) else math.nan,
                "sigma2_over_sigma1": float(s[1] / s[0]) if len(s) > 1 and abs(s[0]) > EPS else math.nan,
                "response_sketch_cos_to_clean": fro_cos(y, clean_y),
                "response_sketch_rel_diff_to_clean": rel_fro_diff(y, clean_y),
                "response_sketch_cos_to_previous": fro_cos(y, previous_y),
                "response_sketch_rel_diff_to_previous": rel_fro_diff(y, previous_y),
            }
            for k in args.subspace_ks:
                cm = subspace_metrics(vh, clean_vh_exact, k)
                pm = subspace_metrics(vh, previous_vh, k) if previous_vh is not None else {"min_cos": math.nan, "mean_cos": math.nan, "max_angle_deg": math.nan, "mean_angle_deg": math.nan}
                for name, value in cm.items():
                    row[f"subspace{k}_clean_{name}"] = value
                for name, value in pm.items():
                    row[f"subspace{k}_previous_{name}"] = value
            rows.append(row)

            sv_row = {"sample_index": int(index), "path_fraction": float(frac), "path_delta_norm_l2": float(frac) * delta_norm}
            for rank in range(args.max_rank):
                sv_row[f"sigma{rank + 1}"] = float(s[rank]) if rank < len(s) else math.nan
            sv_rows.append(sv_row)
            previous_vh = vh.copy()
            previous_y = y.copy()

    value_cols = [
        "path_delta_norm_l2",
        "top1_angle_to_clean_deg",
        "top1_angle_to_previous_deg",
        "spectral_norm",
        "sigma2_over_sigma1",
        "response_sketch_cos_to_clean",
        "response_sketch_rel_diff_to_clean",
        "response_sketch_cos_to_previous",
        "response_sketch_rel_diff_to_previous",
    ]
    for k in args.subspace_ks:
        value_cols.extend([
            f"subspace{k}_clean_max_angle_deg",
            f"subspace{k}_clean_mean_angle_deg",
            f"subspace{k}_previous_max_angle_deg",
            f"subspace{k}_previous_mean_angle_deg",
        ])
    aggregate = aggregate_rows(rows, ["path_fraction"], value_cols)

    sample_summary = []
    for index in args.sample_indices:
        group = [r for r in rows if int(r["sample_index"]) == int(index)]
        final = next(r for r in group if abs(float(r["path_fraction"]) - 1.0) < 1e-9)
        nonzero = [r for r in group if float(r["path_fraction"]) > 0]
        sample_summary.append(
            {
                "sample_index": int(index),
                "endpoint_top1_angle_to_clean_deg": final["top1_angle_to_clean_deg"],
                "endpoint_subspace4_clean_max_angle_deg": final.get("subspace4_clean_max_angle_deg", math.nan),
                "endpoint_subspace8_clean_max_angle_deg": final.get("subspace8_clean_max_angle_deg", math.nan),
                "endpoint_spectral_norm": final["spectral_norm"],
                "endpoint_response_sketch_rel_diff_to_clean": final["response_sketch_rel_diff_to_clean"],
                "max_top1_angle_to_clean_deg": max(float(r["top1_angle_to_clean_deg"]) for r in nonzero),
                "mean_top1_angle_to_previous_deg_nonzero": finite_summary([float(r["top1_angle_to_previous_deg"]) for r in nonzero])["mean"],
            }
        )

    write_csv(args.output_dir / "jacobian_subspace_rotation_by_sample_t.csv", rows)
    write_csv(args.output_dir / "jacobian_subspace_rotation_aggregate_by_t.csv", aggregate)
    write_csv(args.output_dir / "jacobian_subspace_rotation_summary_by_sample.csv", sample_summary)
    write_csv(args.output_dir / "singular_values_by_sample_t.csv", sv_rows)
    plot_outputs(args.output_dir, aggregate, args.subspace_ks)

    seconds = time.perf_counter() - start_time
    manifest = {
        "experiment": "loss3_jacobian_subspace_rotation_path",
        "status": "completed",
        "scope": "FNO / 1D Burgers / nu=0.001 only",
        "sample_indices": args.sample_indices,
        "num_path_points": args.num_path_points,
        "path_fractions": path_fractions,
        "fd_epsilon": args.fd_epsilon,
        "max_rank": args.max_rank,
        "oversample": args.oversample,
        "probe_count": probe_count,
        "subspace_ks": args.subspace_ks,
        "delta_npz": str(args.delta_npz),
        "delta_source": args.delta_source,
        "delta_sample_indices_available": delta_sample_indices,
        "output_dir": str(args.output_dir),
        "result_doc": str(args.result_doc),
        "source_paths": sorted(source_paths),
        "gpu_runtime": gpu_metadata,
        "seconds": seconds,
        "output_files": [
            "jacobian_subspace_rotation_by_sample_t.csv",
            "jacobian_subspace_rotation_aggregate_by_t.csv",
            "jacobian_subspace_rotation_summary_by_sample.csv",
            "singular_values_by_sample_t.csv",
            "manifest.json",
        ],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_result_doc(args.result_doc, args, manifest, aggregate, sample_summary)
    print(f"[done] output_dir={args.output_dir} seconds={seconds:.1f}", flush=True)
    print(f"[done] result_doc={args.result_doc}", flush=True)


if __name__ == "__main__":
    main()
