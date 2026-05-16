#!/usr/bin/env python3
"""Add the missing outward-growth direction to the FNO nu=0.001 local table.

This is a postprocessing experiment for the existing FNO-vs-solver Jacobian/SVD
artifacts. It does not recompute dense Jacobians. It loads saved `J_f`, `J_j`,
and `J_e`, computes the clean residual `b = f(x) - j(x)`, and adds the local
direction

    v_growth = normalize((J_f - J_j).T @ (b / ||b||_2)).

The script records linear response metrics for top model/solver/error singular
directions, the new outward-growth direction, its negative control, and random
directions. It also runs a small-radius nonlinear finite-difference sanity check
for the core directions.
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forensics" / "outward_growth_direction_20260515" / "fno_nu0p001"
DEFAULT_RESULT_DOC = PROJECT_ROOT / "docs" / "outward_growth_direction_result_20260515.md"
EPS = 1e-12


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


def configure_gpu_runtime(args: argparse.Namespace) -> None:
    args.runtime_ptxas_dir = None
    args.runtime_cudnn_enabled = None
    if not args.runtime_workarounds:
        print("[runtime] runtime workarounds disabled", flush=True)
        return
    if args.prepend_env_ptxas:
        ptxas_dir = find_venv_ptxas_dir()
        if ptxas_dir is None:
            print("[runtime] no virtualenv ptxas found; PATH unchanged", flush=True)
        else:
            ptxas_dir_text = str(ptxas_dir)
            entries = [entry for entry in _path_entries() if entry != ptxas_dir_text]
            os.environ["PATH"] = os.pathsep.join([ptxas_dir_text, *entries])
            args.runtime_ptxas_dir = ptxas_dir_text
            print(f"[runtime] using virtualenv ptxas dir first: {ptxas_dir_text}", flush=True)
    torch.backends.cudnn.enabled = False
    args.runtime_cudnn_enabled = bool(torch.backends.cudnn.enabled)
    print("[runtime] torch.backends.cudnn.enabled=False", flush=True)


def finite_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
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


def norm2(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64).reshape(-1)))


def normalize_l2(v: np.ndarray, *, eps: float = EPS) -> tuple[np.ndarray, float, bool]:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    n = norm2(arr)
    if n <= eps:
        return np.zeros_like(arr), n, True
    return arr / n, n, False


def cosine(a: np.ndarray, b: np.ndarray, *, eps: float = EPS) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = norm2(af) * norm2(bf)
    if denom <= eps:
        return math.nan
    return float(np.dot(af, bf) / denom)


def angle_deg(cos_value: float) -> float:
    if not math.isfinite(cos_value):
        return math.nan
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cos_value)))))


def load_svd(root: Path, index: int, name: str) -> dict[str, Any]:
    path = root / f"index_{index:03d}" / name / f"{name}_index{index}_jacobian_svd.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    return {
        "source_path": str(path),
        "J": data["jacobian"].astype(np.float64),
        "singular_values": data["singular_values"].astype(np.float64),
        "right_singular_vectors": data["right_singular_vectors"].astype(np.float64),
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


def forward_model_solver(
    model: torch.nn.Module,
    solver_fn: Any,
    bridge: Any,
    x_flat: np.ndarray,
    *,
    device: torch.device,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    nx = int(x_flat.size)
    x = torch.as_tensor(x_flat.reshape(1, nx, 1), device=device, dtype=torch.float32)
    with torch.no_grad():
        f = model(x).detach().cpu().numpy().reshape(nx).astype(np.float64)
        j = bridge(x, solver_fn, label).detach().cpu().numpy().reshape(nx).astype(np.float64)
    sync_torch(torch, device)
    return f, j


def response_metrics(
    *,
    index: int,
    direction_source: str,
    rank: int,
    v: np.ndarray,
    Jf: np.ndarray,
    Jj: np.ndarray,
    A: np.ndarray,
    u: np.ndarray,
    clean_residual_norm: float,
    eta: float,
    is_degenerate: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    yf = Jf @ v
    yj = Jj @ v
    ym = A @ v
    nf = norm2(yf)
    nj = norm2(yj)
    nm = norm2(ym)
    cos_fj = cosine(yf, yj)
    outward_component = float(np.dot(u, ym)) if norm2(u) > EPS else math.nan
    return {
        "sample_index": index,
        "direction_source": direction_source,
        "rank": rank,
        "direction_norm_l2": norm2(v),
        "fno_gain": nf,
        "solver_gain": nj,
        "mismatch_gain": nm,
        "cos_fno_solver_response": cos_fj,
        "D_f": nm / (nf + eta),
        "D_sym": nm / (nf + nj + eta),
        "outward_component": outward_component,
        "outward_component_ratio": outward_component / (nm + eta),
        "loss3_squared_derivative": 2.0 * clean_residual_norm * outward_component
        if math.isfinite(outward_component)
        else math.nan,
        "clean_residual_norm_l2": clean_residual_norm,
        "is_degenerate": bool(is_degenerate),
        "notes": notes,
    }


def build_directions(
    *,
    index: int,
    svds: dict[str, dict[str, Any]],
    A: np.ndarray,
    b: np.ndarray,
    top_k: int,
    random_directions: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directions: list[dict[str, Any]] = []
    top_k = min(top_k, A.shape[1])

    for source in ("fno", "solver", "error"):
        Vh = svds[source]["right_singular_vectors"]
        for rank in range(min(top_k, Vh.shape[0])):
            v, v_norm, degenerate = normalize_l2(Vh[rank])
            directions.append(
                {
                    "direction_source": source,
                    "rank": rank + 1,
                    "v": v,
                    "raw_norm_l2": v_norm,
                    "is_degenerate": degenerate,
                    "notes": "saved right singular vector",
                }
            )

    b_unit, b_norm, b_degenerate = normalize_l2(b)
    atb = A.T @ b_unit if not b_degenerate else np.zeros(A.shape[1], dtype=np.float64)
    v_growth, atb_norm, atb_degenerate = normalize_l2(atb)
    growth_degenerate = b_degenerate or atb_degenerate
    directions.append(
        {
            "direction_source": "outward_growth",
            "rank": 1,
            "v": v_growth,
            "raw_norm_l2": atb_norm,
            "is_degenerate": growth_degenerate,
            "notes": "normalize((J_f-J_j)^T clean_residual_unit)",
        }
    )
    directions.append(
        {
            "direction_source": "negative_outward_growth",
            "rank": 1,
            "v": -v_growth,
            "raw_norm_l2": atb_norm,
            "is_degenerate": growth_degenerate,
            "notes": "sign sanity check",
        }
    )

    rng = np.random.default_rng(seed + index)
    random_items: list[dict[str, Any]] = []
    for rank in range(random_directions):
        v, v_norm, degenerate = normalize_l2(rng.normal(size=A.shape[1]))
        item = {
            "direction_source": "random",
            "rank": rank + 1,
            "v": v,
            "raw_norm_l2": v_norm,
            "is_degenerate": degenerate,
            "notes": "fixed seeded random unit vector",
        }
        random_items.append(item)
        directions.append(item)

    if random_items:
        def random_outward(item: dict[str, Any]) -> float:
            return float(np.dot(b_unit, A @ item["v"])) if not b_degenerate else -math.inf

        def random_mismatch(item: dict[str, Any]) -> float:
            return norm2(A @ item["v"])

        best_outward = max(random_items, key=random_outward)
        best_mismatch = max(random_items, key=random_mismatch)
        directions.append(
            {
                **best_outward,
                "direction_source": "random_best_by_outward_component",
                "notes": f"selected from {random_directions} random directions; original_rank={best_outward['rank']}",
            }
        )
        directions.append(
            {
                **best_mismatch,
                "direction_source": "random_best_by_mismatch_gain",
                "notes": f"selected from {random_directions} random directions; original_rank={best_mismatch['rank']}",
            }
        )

    metadata = {
        "clean_residual_norm_l2": b_norm,
        "clean_residual_degenerate": b_degenerate,
        "At_b_unit_norm_l2": atb_norm,
        "outward_growth_degenerate": growth_degenerate,
    }
    return directions, metadata


def direction_similarity_rows(index: int, directions: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for item in directions:
        by_source.setdefault(str(item["direction_source"]), []).append(item)
    if "outward_growth" not in by_source:
        return []
    v_growth = by_source["outward_growth"][0]["v"]
    rows: list[dict[str, Any]] = []
    for source in ("fno", "solver", "error"):
        source_items = sorted(by_source.get(source, []), key=lambda r: int(r["rank"]))
        for item in source_items[:top_k]:
            c = cosine(v_growth, item["v"])
            rows.append(
                {
                    "sample_index": index,
                    "reference_direction": "outward_growth",
                    "comparison": source,
                    "rank": item["rank"],
                    "dot": c,
                    "abs_dot": abs(c) if math.isfinite(c) else math.nan,
                    "angle_deg": angle_deg(c),
                }
            )
        if source_items:
            V = np.stack([item["v"] for item in source_items[:top_k]], axis=0)
            dots = V @ v_growth
            rows.append(
                {
                    "sample_index": index,
                    "reference_direction": "outward_growth",
                    "comparison": f"{source}_top{min(top_k, len(source_items))}_subspace",
                    "rank": 0,
                    "subspace_projection_l2": norm2(dots),
                    "max_abs_dot": float(np.max(np.abs(dots))),
                    "mean_abs_dot": float(np.mean(np.abs(dots))),
                }
            )
    return rows


def should_finite_difference(item: dict[str, Any], finite_difference_top_k: int) -> bool:
    source = str(item["direction_source"])
    rank = int(item["rank"])
    if source in {"outward_growth", "negative_outward_growth"}:
        return True
    if source in {"fno", "solver", "error"} and rank <= finite_difference_top_k:
        return True
    return source in {"random_best_by_outward_component", "random_best_by_mismatch_gain"}


def finite_difference_rows(
    *,
    index: int,
    directions: list[dict[str, Any]],
    sample_flat: np.ndarray,
    f_clean: np.ndarray,
    j_clean: np.ndarray,
    model: torch.nn.Module,
    solver_fn: Any,
    bridge: Any,
    device: torch.device,
    rhos: list[float],
    finite_difference_top_k: int,
    Jf: np.ndarray,
    Jj: np.ndarray,
    A: np.ndarray,
    u: np.ndarray,
) -> list[dict[str, Any]]:
    b = f_clean - j_clean
    b_norm = norm2(b)
    rows: list[dict[str, Any]] = []
    for item in directions:
        if not should_finite_difference(item, finite_difference_top_k):
            continue
        v = item["v"]
        pred_model = norm2(Jf @ v)
        pred_solver = norm2(Jj @ v)
        pred_residual = norm2(A @ v)
        pred_outward = float(np.dot(u, A @ v)) if norm2(u) > EPS else math.nan
        for rho in rhos:
            start = time.perf_counter()
            f_pert, j_pert = forward_model_solver(
                model,
                solver_fn,
                bridge,
                sample_flat + float(rho) * v,
                device=device,
                label=f"fd_index{index}_{item['direction_source']}_{item['rank']}_rho{rho:g}",
            )
            elapsed = time.perf_counter() - start
            df = f_pert - f_clean
            dj = j_pert - j_clean
            residual = f_pert - j_pert
            residual_delta = residual - b
            rows.append(
                {
                    "sample_index": index,
                    "direction_source": item["direction_source"],
                    "rank": item["rank"],
                    "rho": float(rho),
                    "actual_loss3_original": norm2(residual),
                    "clean_loss3_original": b_norm,
                    "actual_loss3_growth_per_rho": (norm2(residual) - b_norm) / float(rho),
                    "predicted_loss3_growth_per_rho": pred_outward,
                    "loss3_growth_prediction_error": ((norm2(residual) - b_norm) / float(rho)) - pred_outward,
                    "actual_residual_movement_per_rho": norm2(residual_delta) / float(rho),
                    "predicted_residual_movement_per_rho": pred_residual,
                    "residual_movement_prediction_error": (norm2(residual_delta) / float(rho)) - pred_residual,
                    "actual_model_movement_per_rho": norm2(df) / float(rho),
                    "predicted_model_movement_per_rho": pred_model,
                    "actual_solver_movement_per_rho": norm2(dj) / float(rho),
                    "predicted_solver_movement_per_rho": pred_solver,
                    "cos_delta_f_delta_j": cosine(df, dj),
                    "seconds": elapsed,
                }
            )
    return rows


def aggregate_direction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [
        ("fno_top", lambda r: r["direction_source"] == "fno"),
        ("solver_top", lambda r: r["direction_source"] == "solver"),
        ("error_top", lambda r: r["direction_source"] == "error"),
        ("outward_growth", lambda r: r["direction_source"] == "outward_growth"),
        ("negative_outward_growth", lambda r: r["direction_source"] == "negative_outward_growth"),
        ("random", lambda r: r["direction_source"] == "random"),
        ("random_best_by_outward_component", lambda r: r["direction_source"] == "random_best_by_outward_component"),
        ("random_best_by_mismatch_gain", lambda r: r["direction_source"] == "random_best_by_mismatch_gain"),
    ]
    fields = [
        "fno_gain",
        "solver_gain",
        "mismatch_gain",
        "cos_fno_solver_response",
        "D_f",
        "D_sym",
        "outward_component",
        "outward_component_ratio",
        "loss3_squared_derivative",
    ]
    out: list[dict[str, Any]] = []
    for group_name, pred in groups:
        subset = [r for r in rows if pred(r)]
        if not subset:
            continue
        row: dict[str, Any] = {"direction_group": group_name, "n_rows": len(subset)}
        for field in fields:
            stats = finite_summary([float(r[field]) for r in subset])
            for stat, value in stats.items():
                row[f"{field}_{stat}"] = value
        out.append(row)
    return out


def aggregate_fd_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["direction_source"]), float(row["rho"]))
        grouped.setdefault(key, []).append(row)
    fields = [
        "actual_loss3_growth_per_rho",
        "predicted_loss3_growth_per_rho",
        "loss3_growth_prediction_error",
        "actual_residual_movement_per_rho",
        "predicted_residual_movement_per_rho",
        "residual_movement_prediction_error",
        "actual_model_movement_per_rho",
        "actual_solver_movement_per_rho",
        "cos_delta_f_delta_j",
        "seconds",
    ]
    out: list[dict[str, Any]] = []
    for (source, rho), subset in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        row: dict[str, Any] = {"direction_source": source, "rho": rho, "n_rows": len(subset)}
        for field in fields:
            stats = finite_summary([float(r[field]) for r in subset])
            for stat, value in stats.items():
                row[f"{field}_{stat}"] = value
        out.append(row)
    return out


def write_summary_md(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    direction_summary: list[dict[str, Any]],
    fd_summary: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# FNO nu=0.001 Outward-Growth Direction Summary",
        "",
        "Observed from this run:",
        "",
        f"- Output directory: `{out_dir}`",
        f"- Jacobian source root: `{args.jacobian_root}`",
        f"- Indices: `{args.sample_indices}`",
        f"- Top-k singular directions recorded: `{args.top_k}`",
        f"- Random directions per sample: `{args.random_directions}`",
        f"- Finite-difference radii: `{args.finite_difference_rhos}`",
        "",
        "This experiment adds the missing Experiment 2 row:",
        "",
        "```text",
        "v_growth = normalize((J_f - J_j)^T (b / ||b||_2))",
        "```",
        "",
        "## Direction Summary",
        "",
        "| direction group | n | mismatch gain mean | outward component mean | D_f mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in direction_summary:
        lines.append(
            f"| {row['direction_group']} | {row['n_rows']} | "
            f"{row.get('mismatch_gain_mean', math.nan):.6g} | "
            f"{row.get('outward_component_mean', math.nan):.6g} | "
            f"{row.get('D_f_mean', math.nan):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Finite-Difference Summary",
            "",
            "| direction source | rho | n | actual loss3 growth mean | predicted growth mean | residual movement mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fd_summary:
        lines.append(
            f"| {row['direction_source']} | {row['rho']:.1e} | {row['n_rows']} | "
            f"{row.get('actual_loss3_growth_per_rho_mean', math.nan):.6g} | "
            f"{row.get('predicted_loss3_growth_per_rho_mean', math.nan):.6g} | "
            f"{row.get('actual_residual_movement_per_rho_mean', math.nan):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `config.json`",
            "- `manifest.json`",
            "- `all_direction_response_table.csv`",
            "- `aggregate_direction_response_summary.csv`",
            "- `all_direction_similarity_table.csv`",
            "- `finite_difference_growth_table.csv`",
            "- `finite_difference_growth_summary.csv`",
            "- `index_*/clean_model_output.npy`",
            "- `index_*/clean_solver_output.npy`",
            "- `index_*/clean_residual.npy`",
            "- `index_*/outward_growth_direction.npy`",
            "",
            "Observed source paths:",
        ]
    )
    for path in manifest.get("source_paths", []):
        lines.append(f"- `{path}`")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 7, 40, 47, 115])
    parser.add_argument("--jacobian-root", type=Path, default=DEFAULT_JACOBIAN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-doc", type=Path, default=DEFAULT_RESULT_DOC)
    parser.add_argument("--write-result-doc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--random-directions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--finite-difference-rhos", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--finite-difference-top-k", type=int, default=8)
    parser.add_argument("--finite-difference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime-workarounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepend-env-ptxas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fno-test-path", type=Path, default=DEFAULT_BURGERS_TEST)
    parser.add_argument(
        "--fno-checkpoint",
        type=Path,
        default=DEFAULT_BURGERS_MODEL_DIR / "checkpoints" / "pytorch_fno1d_500.pt",
    )
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
        raise ValueError("This script is intentionally scoped to FNO nu=0.001; pass burgers-nu=0.001.")
    configure_gpu_runtime(args)
    requested_device = str(args.device or "cuda")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false; this experiment is GPU-only by default.")
    device = torch.device(requested_device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        args.output_dir / "config.json",
        {
            "sample_indices": args.sample_indices,
            "jacobian_root": args.jacobian_root,
            "output_dir": args.output_dir,
            "top_k": args.top_k,
            "random_directions": args.random_directions,
            "seed": args.seed,
            "eta": args.eta,
            "finite_difference": args.finite_difference,
            "finite_difference_rhos": args.finite_difference_rhos,
            "finite_difference_top_k": args.finite_difference_top_k,
            "device": str(device),
            "runtime_workarounds": args.runtime_workarounds,
            "prepend_env_ptxas": args.prepend_env_ptxas,
            "runtime_ptxas_dir": args.runtime_ptxas_dir,
            "runtime_cudnn_enabled": args.runtime_cudnn_enabled,
            "fno_test_path": args.fno_test_path,
            "fno_checkpoint": args.fno_checkpoint,
            "burgers_nx": args.burgers_nx,
            "burgers_nu": args.burgers_nu,
            "burgers_t_final": args.burgers_t_final,
            "burgers_dt": args.burgers_dt,
            "burgers_domain": args.burgers_domain,
            "burgers_jax_solver_dtype": args.burgers_jax_solver_dtype,
        },
    )

    model = load_burgers_torch_model(args.fno_checkpoint, device)
    model.eval()
    solver_fn = make_burgers_jax_solver(make_solver_args(args))
    bridge = make_jax_torch_bridge()

    all_response_rows: list[dict[str, Any]] = []
    all_similarity_rows: list[dict[str, Any]] = []
    all_fd_rows: list[dict[str, Any]] = []
    source_paths: list[str] = [str(args.fno_test_path), str(args.fno_checkpoint), str(args.jacobian_root)]
    per_index_metadata: list[dict[str, Any]] = []

    for index in args.sample_indices:
        print(f"[index] {index}", flush=True)
        start_index = time.perf_counter()
        index_dir = args.output_dir / f"index_{index:03d}"
        index_dir.mkdir(parents=True, exist_ok=True)

        svds = {name: load_svd(args.jacobian_root, index, name) for name in ("fno", "solver", "error")}
        source_paths.extend([svds[name]["source_path"] for name in ("fno", "solver", "error")])
        Jf = svds["fno"]["J"]
        Jj = svds["solver"]["J"]
        A = Jf - Jj
        A_saved = svds["error"]["J"]
        consistency_l2 = norm2(A - A_saved)
        consistency_max_abs = float(np.max(np.abs(A - A_saved)))

        sample = load_sample(args.fno_test_path, index)
        sample_flat = sample.reshape(-1).astype(np.float64)
        f_clean, j_clean = forward_model_solver(
            model,
            solver_fn,
            bridge,
            sample_flat,
            device=device,
            label=f"clean_solver_index{index}",
        )
        b = f_clean - j_clean
        u, b_norm, b_degenerate = normalize_l2(b)

        np.save(index_dir / "clean_input.npy", sample_flat)
        np.save(index_dir / "clean_model_output.npy", f_clean)
        np.save(index_dir / "clean_solver_output.npy", j_clean)
        np.save(index_dir / "clean_residual.npy", b)

        directions, meta = build_directions(
            index=index,
            svds=svds,
            A=A,
            b=b,
            top_k=args.top_k,
            random_directions=args.random_directions,
            seed=args.seed,
        )
        outward_item = next(item for item in directions if item["direction_source"] == "outward_growth")
        np.save(index_dir / "outward_growth_direction.npy", outward_item["v"])
        np.savez_compressed(
            index_dir / "directions.npz",
            **{f"{item['direction_source']}_rank{item['rank']}": item["v"] for item in directions},
        )

        response_rows = [
            response_metrics(
                index=index,
                direction_source=str(item["direction_source"]),
                rank=int(item["rank"]),
                v=item["v"],
                Jf=Jf,
                Jj=Jj,
                A=A,
                u=u,
                clean_residual_norm=b_norm,
                eta=args.eta,
                is_degenerate=bool(item["is_degenerate"] or b_degenerate),
                notes=str(item["notes"]),
            )
            for item in directions
        ]
        similarity_rows = direction_similarity_rows(index, directions, top_k=args.top_k)

        fd_rows: list[dict[str, Any]] = []
        if args.finite_difference:
            fd_rows = finite_difference_rows(
                index=index,
                directions=directions,
                sample_flat=sample_flat,
                f_clean=f_clean,
                j_clean=j_clean,
                model=model,
                solver_fn=solver_fn,
                bridge=bridge,
                device=device,
                rhos=[float(r) for r in args.finite_difference_rhos],
                finite_difference_top_k=args.finite_difference_top_k,
                Jf=Jf,
                Jj=Jj,
                A=A,
                u=u,
            )

        write_csv(index_dir / "direction_response_table.csv", response_rows)
        write_csv(index_dir / "direction_similarity_table.csv", similarity_rows)
        write_csv(index_dir / "finite_difference_growth_table.csv", fd_rows)
        write_json(
            index_dir / "manifest.json",
            {
                "sample_index": index,
                "jacobian_sources": {name: svds[name]["source_path"] for name in ("fno", "solver", "error")},
                "clean_input_path": str(index_dir / "clean_input.npy"),
                "clean_model_output_path": str(index_dir / "clean_model_output.npy"),
                "clean_solver_output_path": str(index_dir / "clean_solver_output.npy"),
                "clean_residual_path": str(index_dir / "clean_residual.npy"),
                "outward_growth_direction_path": str(index_dir / "outward_growth_direction.npy"),
                "clean_residual_norm_l2": b_norm,
                "clean_residual_degenerate": b_degenerate,
                "At_b_unit_norm_l2": meta["At_b_unit_norm_l2"],
                "outward_growth_degenerate": meta["outward_growth_degenerate"],
                "error_jacobian_consistency_l2": consistency_l2,
                "error_jacobian_consistency_max_abs": consistency_max_abs,
                "seconds": time.perf_counter() - start_index,
            },
        )

        all_response_rows.extend(response_rows)
        all_similarity_rows.extend(similarity_rows)
        all_fd_rows.extend(fd_rows)
        per_index_metadata.append(
            {
                "sample_index": index,
                "clean_residual_norm_l2": b_norm,
                "At_b_unit_norm_l2": meta["At_b_unit_norm_l2"],
                "outward_growth_degenerate": meta["outward_growth_degenerate"],
                "error_jacobian_consistency_l2": consistency_l2,
                "error_jacobian_consistency_max_abs": consistency_max_abs,
                "seconds": time.perf_counter() - start_index,
            }
        )
        print(
            f"[done-index] {index} clean_loss3={b_norm:.6g} Atb={meta['At_b_unit_norm_l2']:.6g} "
            f"seconds={time.perf_counter() - start_index:.1f}",
            flush=True,
        )

    direction_summary = aggregate_direction_rows(all_response_rows)
    fd_summary = aggregate_fd_rows(all_fd_rows)
    manifest = {
        "experiment": "outward_growth_direction_fno_nu0p001",
        "status": "completed",
        "source_paths": sorted(set(source_paths)),
        "output_paths": [
            str(args.output_dir / "all_direction_response_table.csv"),
            str(args.output_dir / "aggregate_direction_response_summary.csv"),
            str(args.output_dir / "all_direction_similarity_table.csv"),
            str(args.output_dir / "finite_difference_growth_table.csv"),
            str(args.output_dir / "finite_difference_growth_summary.csv"),
            str(args.output_dir / "summary.md"),
        ],
        "per_index_metadata": per_index_metadata,
    }

    write_csv(args.output_dir / "all_direction_response_table.csv", all_response_rows)
    write_csv(args.output_dir / "aggregate_direction_response_summary.csv", direction_summary)
    write_csv(args.output_dir / "all_direction_similarity_table.csv", all_similarity_rows)
    write_csv(args.output_dir / "finite_difference_growth_table.csv", all_fd_rows)
    write_csv(args.output_dir / "finite_difference_growth_summary.csv", fd_summary)
    write_json(args.output_dir / "manifest.json", manifest)
    write_summary_md(args.output_dir, args=args, direction_summary=direction_summary, fd_summary=fd_summary, manifest=manifest)

    if args.write_result_doc:
        args.result_doc.parent.mkdir(parents=True, exist_ok=True)
        args.result_doc.write_text((args.output_dir / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[done] wrote {args.output_dir}", flush=True)
    if args.write_result_doc:
        print(f"[done] wrote {args.result_doc}", flush=True)


if __name__ == "__main__":
    main()
