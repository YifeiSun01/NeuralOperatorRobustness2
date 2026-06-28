#!/usr/bin/env python3
"""Experiment 6: direction rotation along a finite-radius loss3 path.

Scope: FNO / 1D Burgers / nu=0.001 only.

For each selected sample, this script loads a direct loss3 endpoint perturbation
delta*, walks along x_t = x + t delta*, and re-estimates the local residual
movement direction

    v_e*(x_t) ~= argmax_{||v||_2 = 1} ||e(x_t + eps v) - e(x_t)||_2 / eps

where e = f - j.  The optimized small-epsilon direction is then compared with
the clean-point direction and with adjacent path points.
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
from tools.run_loss3_gradient_direction_optimization import (  # noqa: E402
    evaluate_clean_torch,
    objective_value,
    safe_float,
    torch_unit,
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forensics" / "loss3_direction_rotation_path_20260516" / "fno_nu0p001_pilot"
DEFAULT_RESULT_DOC = PROJECT_ROOT / "docs" / "loss3_direction_rotation_path_fno_nu0p001_result_20260516.md"
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


def make_solver_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        burgers_nx=args.burgers_nx,
        burgers_nu=args.burgers_nu,
        burgers_t_final=args.burgers_t_final,
        burgers_dt=args.burgers_dt,
        burgers_domain=args.burgers_domain,
        burgers_jax_solver_dtype=args.burgers_jax_solver_dtype,
    )


def parse_fractions(text: str) -> list[float]:
    values = [float(part) for part in text.replace(",", " ").split()]
    if not values:
        raise ValueError("Need at least one path fraction.")
    return values


def load_delta_map(path: Path, source: str) -> tuple[dict[int, np.ndarray], list[int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    if source not in data:
        raise KeyError(f"{source!r} not found in {path}; available keys: {list(data.keys())}")
    sample_indices = [int(x) for x in data["sample_indices"].tolist()]
    deltas = np.asarray(data[source], dtype=np.float32)
    return {idx: deltas[pos].reshape(-1).astype(np.float64) for pos, idx in enumerate(sample_indices)}, sample_indices


def direction_value_no_grad(
    *,
    model: torch.nn.Module,
    solver_fn: Any,
    bridge: Any,
    x_state: torch.Tensor,
    f_state: torch.Tensor,
    j_state: torch.Tensor,
    direction: np.ndarray,
    epsilon: float,
    label: str,
) -> float:
    v = torch.as_tensor(direction.reshape(-1), device=x_state.device, dtype=torch.float32)
    v = torch_unit(v)
    with torch.no_grad():
        value = objective_value(
            objective="L_e",
            model=model,
            solver_fn=solver_fn,
            bridge=bridge,
            x_clean=x_state,
            f0=f_state,
            j0=j_state,
            v_unit=v,
            epsilon=float(epsilon),
            label=label,
        )
    sync_torch(torch, x_state.device)
    return float(value.detach().cpu())


def optimize_one_start(
    *,
    model: torch.nn.Module,
    solver_fn: Any,
    bridge: Any,
    x_state: torch.Tensor,
    f_state: torch.Tensor,
    j_state: torch.Tensor,
    epsilon: float,
    init: np.ndarray,
    device: torch.device,
    steps: int,
    lr: float,
    save_every: int,
    label_prefix: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    v = torch.as_tensor(init.reshape(-1), device=device, dtype=torch.float32).clone().detach()
    v = torch_unit(v).detach().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=float(lr))
    trajectory: list[dict[str, Any]] = []
    initial_value = math.nan
    final_grad_norm = math.nan
    best_value = -math.inf
    best_step = 0
    best_v = torch_unit(v).detach().clone()

    for step in range(int(steps) + 1):
        opt.zero_grad(set_to_none=True)
        v_unit = torch_unit(v)
        value = objective_value(
            objective="L_e",
            model=model,
            solver_fn=solver_fn,
            bridge=bridge,
            x_clean=x_state,
            f0=f_state,
            j0=j_state,
            v_unit=v_unit,
            epsilon=float(epsilon),
            label=f"{label_prefix}_step{step:03d}",
        )
        value_float = float(value.detach().cpu())
        if step == 0:
            initial_value = value_float
        if value_float > best_value:
            best_value = value_float
            best_step = int(step)
            best_v = v_unit.detach().clone()
        if step % int(save_every) == 0 or step in {0, int(steps)}:
            trajectory.append(
                {
                    "opt_step": int(step),
                    "objective_value": value_float,
                    "best_value_so_far": best_value,
                    "v_param_norm": float(torch.linalg.vector_norm(v.detach()).cpu()),
                }
            )
        if step == int(steps):
            if v.grad is not None:
                final_grad_norm = float(torch.linalg.vector_norm(v.grad.detach()).cpu())
            break
        (-value).backward()
        if v.grad is not None:
            final_grad_norm = float(torch.linalg.vector_norm(v.grad.detach()).cpu())
        opt.step()
        with torch.no_grad():
            v.copy_(torch_unit(v))

    final_v = best_v.detach().cpu().numpy().astype(np.float64)
    final_value = float(best_value)
    return (
        {
            "initial_value": initial_value,
            "final_value": final_value,
            "improvement": final_value - initial_value,
            "final_grad_norm": final_grad_norm,
            "best_step": best_step,
        },
        trajectory,
        normalize_l2(final_v),
    )


def make_starts(
    *,
    clean_reference: np.ndarray,
    previous_direction: np.ndarray | None,
    n: int,
    rng: np.random.Generator,
    random_starts: int,
    policy: str,
) -> list[dict[str, Any]]:
    starts: list[dict[str, Any]] = []
    use_previous = previous_direction is not None and norm2(previous_direction) > EPS
    if use_previous:
        starts.append({"start_id": "previous", "start_type": "previous", "v0": normalize_l2(previous_direction)})

    if policy == "warm_only" and use_previous:
        include_clean_plus = False
        include_clean_minus = False
    elif policy == "warm_clean":
        include_clean_plus = True
        include_clean_minus = False
    else:
        include_clean_plus = True
        include_clean_minus = True

    if include_clean_plus:
        starts.append({"start_id": "clean_plus", "start_type": "clean_reference", "v0": normalize_l2(clean_reference)})
    if include_clean_minus:
        starts.append({"start_id": "clean_minus", "start_type": "clean_reference", "v0": -normalize_l2(clean_reference)})
    for k in range(int(random_starts)):
        starts.append({"start_id": f"random_{k:02d}", "start_type": "random", "v0": normalize_l2(rng.normal(size=n))})
    return starts


def aggregate_by_fraction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(float(row["path_fraction"]), []).append(row)
    value_cols = [
        "angle_to_clean_deg",
        "adjacent_angle_deg",
        "best_value",
        "clean_direction_value",
        "best_over_clean_direction_value",
        "abs_cos_to_attack_delta",
    ]
    out: list[dict[str, Any]] = []
    for frac, group in sorted(groups.items()):
        item: dict[str, Any] = {"path_fraction": frac, "n": len(group)}
        for col in value_cols:
            stats = finite_summary([float(r.get(col, math.nan)) for r in group])
            for key, value in stats.items():
                item[f"{col}_{key}"] = value
        out.append(item)
    return out


def aggregate_by_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row["sample_index"]), []).append(row)
    out: list[dict[str, Any]] = []
    for index, group in sorted(groups.items()):
        final_rows = [r for r in group if abs(float(r["path_fraction"]) - 1.0) < 1e-9]
        nonzero = [r for r in group if float(r["path_fraction"]) > 0.0]
        final = final_rows[0] if final_rows else {}
        item = {
            "sample_index": index,
            "n_path_points": len(group),
            "final_angle_to_clean_deg": final.get("angle_to_clean_deg", math.nan),
            "final_best_over_clean_direction_value": final.get("best_over_clean_direction_value", math.nan),
            "mean_angle_to_clean_deg_nonzero": finite_summary([float(r["angle_to_clean_deg"]) for r in nonzero])["mean"],
            "max_angle_to_clean_deg_nonzero": finite_summary([float(r["angle_to_clean_deg"]) for r in nonzero])["max"],
            "mean_adjacent_angle_deg_nonzero": finite_summary([float(r["adjacent_angle_deg"]) for r in nonzero])["mean"],
        }
        out.append(item)
    return out


def plot_outputs(out_dir: Path, aggregate_rows: list[dict[str, Any]], sample_rows: list[dict[str, Any]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray([r["path_fraction"] for r in aggregate_rows], dtype=np.float64)

    def plot_mean_std(y_key: str, title: str, ylabel: str, filename: str) -> None:
        mean = np.asarray([r[f"{y_key}_mean"] for r in aggregate_rows], dtype=np.float64)
        std = np.asarray([r[f"{y_key}_std"] for r in aggregate_rows], dtype=np.float64)
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

    plot_mean_std("angle_to_clean_deg", "Rotation from clean local residual direction", "angle to clean v_e*(x0), deg", "angle_to_clean_vs_t.png")
    plot_mean_std("adjacent_angle_deg", "Adjacent local residual-direction rotation", "angle to previous t, deg", "adjacent_angle_vs_t.png")
    plot_mean_std("best_over_clean_direction_value", "Local gain recovered by re-estimating direction", "best gain / clean-direction gain", "best_over_clean_gain_vs_t.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for index in sorted({int(r["sample_index"]) for r in sample_rows}):
        rows = sorted([r for r in sample_rows if int(r["sample_index"]) == index], key=lambda r: float(r["path_fraction"]))
        ax.plot(
            [r["path_fraction"] for r in rows],
            [r["angle_to_clean_deg"] for r in rows],
            marker="o",
            linewidth=1.2,
            label=f"idx {index}",
        )
    ax.set_title("Per-sample direction rotation")
    ax.set_xlabel("path fraction t")
    ax.set_ylabel("angle to clean v_e*(x0), deg")
    ax.grid(alpha=0.28)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "angle_to_clean_by_sample.png", dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict[str, Any]], keys: list[str], max_rows: int | None = None) -> str:
    use_rows = rows if max_rows is None else rows[:max_rows]
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    lines = [header, sep]
    for row in use_rows:
        vals = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append(f"{value:.4g}" if math.isfinite(value) else "nan")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_result_doc(
    *,
    path: Path,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    sample_summary: list[dict[str, Any]],
) -> None:
    final = next((r for r in aggregate_rows if abs(float(r["path_fraction"]) - 1.0) < 1e-9), aggregate_rows[-1])
    doc = [
        "# Loss3 Direction Rotation Along Path Result",
        "",
        "Date: 2026-05-16 UTC",
        "",
        "Scope: FNO / 1D Burgers `nu=0.001` only.",
        "",
        "## Run Settings",
        "",
        f"- Output directory: `{args.output_dir}`",
        f"- Delta source: `{args.delta_npz}` key `{args.delta_source}`",
        f"- Sample indices: `{args.sample_indices}`",
        f"- Path fractions: `{args.path_fractions}`",
        f"- Local direction objective: `L_e = ||e(x_t + eps v)-e(x_t)||_2 / eps`, `eps={args.local_epsilon:g}`",
        f"- Direction optimizer: Adam on unit L2 direction with best-over-steps selection, `steps={args.steps}`, `lr={args.lr}`, `random_starts={args.random_starts}`",
        "",
        "## Main Aggregate",
        "",
        markdown_table(
            aggregate_rows,
            [
                "path_fraction",
                "n",
                "angle_to_clean_deg_mean",
                "angle_to_clean_deg_std",
                "adjacent_angle_deg_mean",
                "best_over_clean_direction_value_mean",
                "abs_cos_to_attack_delta_mean",
            ],
        ),
        "",
        "## Per-Sample Summary",
        "",
        markdown_table(
            sample_summary,
            [
                "sample_index",
                "final_angle_to_clean_deg",
                "final_best_over_clean_direction_value",
                "mean_angle_to_clean_deg_nonzero",
                "max_angle_to_clean_deg_nonzero",
                "mean_adjacent_angle_deg_nonzero",
            ],
        ),
        "",
        "## Interpretation",
        "",
        (
            f"At `t=1`, the mean sign-invariant angle from the clean local residual direction is "
            f"`{final['angle_to_clean_deg_mean']:.2f}` degrees. The mean adjacent-step angle at `t=1` is "
            f"`{final['adjacent_angle_deg_mean']:.2f}` degrees."
        ),
        "",
        (
            f"The mean ratio `best local gain / clean-direction local gain` at `t=1` is "
            f"`{final['best_over_clean_direction_value_mean']:.4g}`. Values above 1 mean that re-estimating "
            "the local direction at the current path point finds more residual movement than continuing to use "
            "the clean-point direction."
        ),
        "",
        "## Output Files",
        "",
        "- `direction_rotation_by_sample_t.csv`",
        "- `direction_rotation_aggregate_by_t.csv`",
        "- `direction_rotation_summary_by_sample.csv`",
        "- `optimization_trace.csv`",
        "- `optimized_directions.npz`",
        "- `manifest.json`",
        "- `figures/angle_to_clean_vs_t.png`",
        "- `figures/adjacent_angle_vs_t.png`",
        "- `figures/best_over_clean_gain_vs_t.png`",
        "- `figures/angle_to_clean_by_sample.png`",
        "",
        "## Manifest Snippet",
        "",
        "```json",
        json.dumps(finite_json({k: manifest[k] for k in ("experiment", "status", "seconds", "sample_indices")}), indent=2),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(doc), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 7, 40, 47, 115])
    parser.add_argument("--path-fractions", default="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0")
    parser.add_argument("--local-epsilon", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--random-starts", type=int, default=1)
    parser.add_argument("--start-policy", choices=["clean_pair", "warm_clean", "warm_only"], default="clean_pair")
    parser.add_argument("--save-every", type=int, default=5)
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
    args.path_fractions = parse_fractions(str(args.path_fractions))
    start_time = time.perf_counter()
    configure_runtime(args)
    device, gpu_metadata = require_gpu_runtime(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    delta_map, delta_sample_indices = load_delta_map(args.delta_npz, args.delta_source)
    missing = [idx for idx in args.sample_indices if idx not in delta_map]
    if missing:
        raise RuntimeError(f"Missing requested sample indices in {args.delta_npz}: {missing}")

    print(f"[load] model={args.fno_checkpoint}", flush=True)
    model = load_burgers_torch_model(args.fno_checkpoint, device)
    solver_fn = make_burgers_jax_solver(make_solver_args(args))
    bridge = make_jax_torch_bridge()

    rotation_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    direction_arrays: dict[str, np.ndarray] = {}

    source_paths = {
        str(args.fno_checkpoint),
        str(args.fno_test_path),
        str(args.delta_npz),
        str(args.jacobian_root),
    }

    for index in args.sample_indices:
        print(f"[index] {index}", flush=True)
        x0 = load_sample(args.fno_test_path, index).reshape(-1).astype(np.float64)
        delta = delta_map[index].reshape(-1).astype(np.float64)
        delta_norm = norm2(delta)
        delta_unit = normalize_l2(delta)
        svd = load_svd(args.jacobian_root, index, "error")
        source_paths.add(str(svd["source_path"]))
        clean_v = normalize_l2(np.asarray(svd["right_singular_vectors"], dtype=np.float64)[0])
        clean_sigma = float(np.asarray(svd["singular_values"], dtype=np.float64)[0])
        clean_sigma2 = float(np.asarray(svd["singular_values"], dtype=np.float64)[1])
        clean_gap_ratio = clean_sigma2 / clean_sigma if abs(clean_sigma) > EPS else math.nan
        rng = np.random.default_rng(args.seed + 1009 * index)
        previous_v: np.ndarray | None = None

        for frac in args.path_fractions:
            print(f"[path] index={index} t={frac:g}", flush=True)
            x_state_np = x0 + float(frac) * delta
            x_state, f_state, j_state = evaluate_clean_torch(
                model=model,
                solver_fn=solver_fn,
                bridge=bridge,
                x0=x_state_np,
                device=device,
            )
            clean_value = direction_value_no_grad(
                model=model,
                solver_fn=solver_fn,
                bridge=bridge,
                x_state=x_state,
                f_state=f_state,
                j_state=j_state,
                direction=clean_v,
                epsilon=float(args.local_epsilon),
                label=f"idx{index}_t{frac:g}_clean_direction_value",
            )

            starts = make_starts(
                clean_reference=clean_v,
                previous_direction=previous_v,
                n=x0.size,
                rng=rng,
                random_starts=args.random_starts,
                policy=args.start_policy,
            )
            case_rows: list[dict[str, Any]] = []
            case_vectors: dict[str, np.ndarray] = {}
            for start_info in starts:
                if hasattr(bridge, "reset"):
                    bridge.reset()
                label_prefix = f"idx{index}_t{frac:g}_{start_info['start_id']}"
                run_result, traj, final_v = optimize_one_start(
                    model=model,
                    solver_fn=solver_fn,
                    bridge=bridge,
                    x_state=x_state,
                    f_state=f_state,
                    j_state=j_state,
                    epsilon=float(args.local_epsilon),
                    init=np.asarray(start_info["v0"], dtype=np.float64),
                    device=device,
                    steps=int(args.steps),
                    lr=float(args.lr),
                    save_every=int(args.save_every),
                    label_prefix=label_prefix,
                )
                base = {
                    "sample_index": int(index),
                    "path_fraction": float(frac),
                    "path_delta_norm_l2": float(frac) * delta_norm,
                    "delta_star_norm_l2": delta_norm,
                    "local_epsilon": float(args.local_epsilon),
                    "start_id": start_info["start_id"],
                    "start_type": start_info["start_type"],
                    "initial_value": run_result["initial_value"],
                    "final_value": run_result["final_value"],
                    "improvement": run_result["improvement"],
                    "final_grad_norm": run_result["final_grad_norm"],
                    "best_step": run_result.get("best_step", math.nan),
                    "final_abs_cos_to_clean": abs(cosine(final_v, clean_v)),
                    "final_abs_cos_to_previous": abs(cosine(final_v, previous_v)),
                }
                case_rows.append(base)
                case_vectors[start_info["start_id"]] = final_v
                for trow in traj:
                    trace_rows.append({**base, **trow})

            best = max(case_rows, key=lambda r: safe_float(r["final_value"]))
            best_v = case_vectors[str(best["start_id"])]
            if previous_v is not None and cosine(best_v, previous_v) < 0.0:
                best_v = -best_v
            elif previous_v is None and cosine(best_v, clean_v) < 0.0:
                best_v = -best_v

            abs_cos_clean = abs(cosine(best_v, clean_v))
            abs_cos_adjacent = abs(cosine(best_v, previous_v)) if previous_v is not None else math.nan
            abs_cos_attack = abs(cosine(best_v, delta_unit))
            start_values = sorted([r["final_value"] for r in case_rows], reverse=True)
            second_value = start_values[1] if len(start_values) > 1 else math.nan
            best_value = float(best["final_value"])
            best_over_clean = best_value / clean_value if abs(clean_value) > EPS else math.nan
            value_gap_rel = (best_value - second_value) / abs(best_value) if math.isfinite(second_value) and abs(best_value) > EPS else math.nan

            row = {
                "sample_index": int(index),
                "path_fraction": float(frac),
                "path_delta_norm_l2": float(frac) * delta_norm,
                "delta_star_norm_l2": delta_norm,
                "local_epsilon": float(args.local_epsilon),
                "best_start_id": best["start_id"],
                "best_start_type": best["start_type"],
                "best_value": best_value,
                "second_best_value": second_value,
                "best_second_relative_gap": value_gap_rel,
                "clean_direction_value": clean_value,
                "best_over_clean_direction_value": best_over_clean,
                "abs_cos_to_clean": abs_cos_clean,
                "angle_to_clean_deg": angle_from_abs_cos(abs_cos_clean),
                "abs_cos_to_adjacent": abs_cos_adjacent,
                "adjacent_angle_deg": angle_from_abs_cos(abs_cos_adjacent),
                "abs_cos_to_attack_delta": abs_cos_attack,
                "angle_to_attack_delta_deg": angle_from_abs_cos(abs_cos_attack),
                "clean_sigma1": clean_sigma,
                "clean_sigma2": clean_sigma2,
                "clean_sigma2_over_sigma1": clean_gap_ratio,
                "n_starts": len(starts),
            }
            rotation_rows.append(row)
            direction_arrays[f"index_{index:03d}_t_{float(frac):.3f}"] = best_v.astype(np.float32)
            previous_v = best_v

    aggregate_rows = aggregate_by_fraction(rotation_rows)
    sample_summary = aggregate_by_sample(rotation_rows)
    write_csv(args.output_dir / "direction_rotation_by_sample_t.csv", rotation_rows)
    write_csv(args.output_dir / "direction_rotation_aggregate_by_t.csv", aggregate_rows)
    write_csv(args.output_dir / "direction_rotation_summary_by_sample.csv", sample_summary)
    write_csv(args.output_dir / "optimization_trace.csv", trace_rows)
    np.savez_compressed(args.output_dir / "optimized_directions.npz", **direction_arrays)
    plot_outputs(args.output_dir, aggregate_rows, rotation_rows)

    seconds = time.perf_counter() - start_time
    manifest = {
        "experiment": "loss3_direction_rotation_path",
        "status": "completed",
        "scope": "FNO / 1D Burgers / nu=0.001 only",
        "sample_indices": args.sample_indices,
        "path_fractions": args.path_fractions,
        "local_epsilon": args.local_epsilon,
        "steps": args.steps,
        "lr": args.lr,
        "random_starts": args.random_starts,
        "start_policy": args.start_policy,
        "delta_npz": str(args.delta_npz),
        "delta_source": args.delta_source,
        "delta_sample_indices_available": delta_sample_indices,
        "output_dir": str(args.output_dir),
        "result_doc": str(args.result_doc),
        "source_paths": sorted(source_paths),
        "gpu_runtime": gpu_metadata,
        "seconds": seconds,
        "output_files": [
            "direction_rotation_by_sample_t.csv",
            "direction_rotation_aggregate_by_t.csv",
            "direction_rotation_summary_by_sample.csv",
            "optimization_trace.csv",
            "optimized_directions.npz",
            "manifest.json",
            "figures/angle_to_clean_vs_t.png",
            "figures/adjacent_angle_vs_t.png",
            "figures/best_over_clean_gain_vs_t.png",
            "figures/angle_to_clean_by_sample.png",
        ],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_result_doc(
        path=args.result_doc,
        args=args,
        manifest=manifest,
        aggregate_rows=aggregate_rows,
        sample_summary=sample_summary,
    )
    print(f"[done] output_dir={args.output_dir} seconds={seconds:.1f}", flush=True)
    print(f"[done] result_doc={args.result_doc}", flush=True)


if __name__ == "__main__":
    main()
