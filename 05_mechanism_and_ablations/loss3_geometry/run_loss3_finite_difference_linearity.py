#!/usr/bin/env python3
"""Symmetric finite-difference local linearity for Loss3 PGD path.

Experiment 2 from the simple path-linearity plan.  It evaluates, without
Hessians/Jacobians/SVDs, whether the scalar Loss3 and residual map e(z)=f(z)-j(z)
look more locally linear later along a saved PGD trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_three_loss_objective_attack as attack

PATH_KEYS = (
    "output_dir",
    "burgers_test_path",
    "burgers_torch_checkpoint",
    "ns_test_path",
    "ns_torch_checkpoint",
)

DEFAULT_TRAJECTORY_ROOT = (
    PROJECT_ROOT
    / "forensics"
    / "loss3_simple_path_linearity_20260517"
    / "fno_nu0p001"
    / "loss3_original_pgd_trajectories_steps100"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "forensics"
    / "loss3_simple_path_linearity_20260517"
    / "fno_nu0p001"
    / "experiment2_finite_difference_linearity_steps100"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_config_args(config_path: Path, *, device: str | None) -> SimpleNamespace:
    cfg = read_json(config_path)
    for key in PATH_KEYS:
        if key in cfg and cfg[key] is not None:
            cfg[key] = Path(cfg[key])
    cfg.setdefault("runtime_workarounds", True)
    cfg.setdefault("disable_cudnn", True)
    cfg.setdefault("prepend_env_ptxas", True)
    cfg["device"] = device if device is not None else cfg.get("device", None)
    cfg.setdefault("cpu", False)
    cfg["objective_variant"] = "original"
    cfg["loss_type"] = "loss3"
    cfg["attack_method"] = "pgd"
    cfg["ratio_denominator_epsilon"] = float(cfg.get("ratio_denominator_epsilon", 0.0))
    cfg["regularization_c"] = float(cfg.get("regularization_c", 0.0))
    cfg["frame_mode"] = cfg.get("frame_mode", "")
    args = SimpleNamespace(**cfg)
    attack.configure_runtime_workarounds(args)
    args.input_p_order = attack.parse_norm_order(args.input_p if args.input_p is not None else args.norm)
    args.output_q_order = attack.parse_norm_order(args.output_q)
    args.loss_q_order = args.output_q_order
    return args


def vector_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64).reshape(-1)))


def unit(x: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(x, dtype=np.float64)
    n = vector_norm(arr)
    if n <= 0.0 or not math.isfinite(n):
        return None
    return arr / n


def grad_loss3(problem: Any, x_adv_np: np.ndarray) -> tuple[float, np.ndarray]:
    import torch

    x = torch.as_tensor(x_adv_np[None, ...], device=problem.device, dtype=torch.float32).detach().requires_grad_(True)
    objective, _base_loss, _baseline, _delta_p, _f_delta, _target = attack.active_objective(
        problem, x, "loss3", problem.args.frame_mode
    )
    grad = torch.autograd.grad(objective, x, retain_graph=False, allow_unused=False)[0]
    attack.sync_torch(torch, problem.device)
    value = float(objective.detach().cpu())
    grad_np = grad.detach().cpu().numpy()[0].astype(np.float64)
    del x, objective, grad
    return value, grad_np


def eval_loss3_residual(problem: Any, x_np: np.ndarray) -> tuple[float, np.ndarray]:
    import torch

    with torch.no_grad():
        x = torch.as_tensor(x_np[None, ...], device=problem.device, dtype=torch.float32)
        f = problem.model_forward(x)
        g = problem.solver_forward(x, allow_grad=False)
        residual = f - g
        loss = float(torch.linalg.vector_norm(residual.detach()).cpu())
        residual_np = residual.detach().cpu().numpy()[0].astype(np.float64)
    attack.sync_torch(torch, problem.device)
    return loss, residual_np


def mean_std_min_max(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), float(arr.max())


def bin_name(radius_ratio: float) -> str:
    if radius_ratio <= 0.2:
        return "early"
    if radius_ratio < 0.6:
        return "middle"
    return "late"


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)

    metrics = [
        "loss_linearity_score",
        "residual_linearity_score",
        "loss_second_diff_abs",
        "loss_first_diff_abs",
        "residual_second_diff_norm",
        "residual_first_diff_norm",
        "loss_plus",
        "loss_center",
        "loss_minus",
        "delta_budget_ratio",
        "rho",
    ]
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        row: dict[str, Any] = {name: value for name, value in zip(keys, key)}
        row["n_points"] = len(items)
        for metric in metrics:
            m, s, mn, mx = mean_std_min_max([float(item.get(metric, float("nan"))) for item in items])
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
            row[f"{metric}_min"] = mn
            row[f"{metric}_max"] = mx
        out.append(row)
    return out


def fmt_float(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not isinstance(value, float):
        return str(value)
    if math.isnan(value):
        return "nan"
    if abs(value) >= 100 or (abs(value) < 1e-3 and value != 0.0):
        return f"{value:.3e}"
    return f"{value:.4f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt_float(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 7, 40, 47, 115])
    parser.add_argument("--sample-ks", type=int, nargs="+", default=list(range(0, 101, 5)))
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.16])
    parser.add_argument("--num-random-directions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eps-num", type=float, default=1e-12)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    source_paths: list[str] = []

    for index in args.indices:
        run_dir = args.trajectory_root / f"index_{index:03d}" / "loss3_original_pgd"
        config_path = run_dir / "config.json"
        traj_path = run_dir / "trajectory.npz"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        if not traj_path.exists():
            raise FileNotFoundError(traj_path)

        problem_args = load_config_args(config_path, device=args.device)
        problem_args.index = index
        problem = attack.AttackProblem(problem_args)
        traj = np.load(traj_path)
        ks = traj["k"].astype(int)
        x_adv = traj["x_adv"].astype(np.float64)
        delta = traj["delta"].astype(np.float64)
        available = {int(k): pos for pos, k in enumerate(ks)}
        selected_ks = [int(k) for k in args.sample_ks if int(k) in available]
        if not selected_ks:
            raise ValueError(f"No requested k values found in {traj_path}; available={ks.tolist()}")

        rng = np.random.default_rng(args.seed + 1009 * index)
        random_dirs: list[np.ndarray] = []
        for _ in range(args.num_random_directions):
            v = rng.standard_normal(size=x_adv[available[selected_ks[0]]].shape)
            u = unit(v)
            if u is None:
                raise RuntimeError("random direction unexpectedly has zero norm")
            random_dirs.append(u)

        print(f"[index] {index} selected_ks={selected_ks}", flush=True)
        source_paths.extend([str(config_path), str(traj_path)])

        center_cache: dict[int, tuple[float, np.ndarray]] = {}
        grad_cache: dict[int, tuple[float, np.ndarray]] = {}
        for k in selected_ks:
            pos = available[k]
            x_center = x_adv[pos]
            center_cache[k] = eval_loss3_residual(problem, x_center)
            grad_cache[k] = grad_loss3(problem, x_center)

        for k in selected_ks:
            pos = available[k]
            x_center = x_adv[pos]
            d = delta[pos]
            delta_norm = vector_norm(d)
            radius_ratio = delta_norm / float(problem_args.epsilon)
            center_loss, center_residual = center_cache[k]
            grad_value, grad = grad_cache[k]

            directions: list[tuple[str, np.ndarray]] = []
            grad_u = unit(grad)
            if grad_u is not None:
                directions.append(("grad", grad_u))
            radial_u = unit(d)
            if radial_u is not None:
                directions.append(("radial", radial_u))
            if k != selected_ks[0]:
                prev_k = selected_ks[selected_ks.index(k) - 1]
                prev_delta = delta[available[prev_k]]
                step_u = unit(d - prev_delta)
                if step_u is not None:
                    directions.append(("step", step_u))
            for i, u in enumerate(random_dirs):
                directions.append((f"random_{i}", u))

            for rho in args.rhos:
                for direction_type, u in directions:
                    x_plus = x_center + float(rho) * u
                    x_minus = x_center - float(rho) * u
                    loss_plus, residual_plus = eval_loss3_residual(problem, x_plus)
                    loss_minus, residual_minus = eval_loss3_residual(problem, x_minus)

                    loss_second = abs(loss_plus - 2.0 * center_loss + loss_minus)
                    loss_first = abs(loss_plus - loss_minus)
                    residual_second = vector_norm(residual_plus - 2.0 * center_residual + residual_minus)
                    residual_first = vector_norm(residual_plus - residual_minus)
                    loss_score = loss_second / (loss_first + float(args.eps_num))
                    residual_score = residual_second / (residual_first + float(args.eps_num))

                    rows.append(
                        {
                            "sample_index": index,
                            "k": k,
                            "path_bin": bin_name(radius_ratio),
                            "delta_norm_l2": delta_norm,
                            "delta_budget_ratio": radius_ratio,
                            "direction_type": direction_type,
                            "rho": float(rho),
                            "loss_linearity_score": loss_score,
                            "residual_linearity_score": residual_score,
                            "loss_second_diff_abs": loss_second,
                            "loss_first_diff_abs": loss_first,
                            "loss_plus": loss_plus,
                            "loss_center": center_loss,
                            "loss_minus": loss_minus,
                            "grad_loss3_value": grad_value,
                            "grad_norm_l2": vector_norm(grad),
                            "residual_second_diff_norm": residual_second,
                            "residual_first_diff_norm": residual_first,
                            "trajectory_path": str(traj_path),
                        }
                    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    by_k_direction = summarize(rows, ("k", "direction_type", "rho"))
    by_bin_direction = summarize(rows, ("path_bin", "direction_type", "rho"))
    by_k = summarize(rows, ("k", "rho"))
    by_bin = summarize(rows, ("path_bin", "rho"))
    by_direction = summarize(rows, ("direction_type", "rho"))

    write_csv(out_dir / "finite_difference_linearity_by_sample_k_direction.csv", rows)
    write_csv(out_dir / "finite_difference_linearity_aggregate_by_k_direction.csv", by_k_direction)
    write_csv(out_dir / "finite_difference_linearity_aggregate_by_bin_direction.csv", by_bin_direction)
    write_csv(out_dir / "finite_difference_linearity_aggregate_by_k.csv", by_k)
    write_csv(out_dir / "finite_difference_linearity_aggregate_by_bin.csv", by_bin)
    write_csv(out_dir / "finite_difference_linearity_aggregate_by_direction.csv", by_direction)

    manifest = {
        "experiment": "Experiment 2: Symmetric Finite-Difference Local Linearity Score",
        "trajectory_root": str(args.trajectory_root),
        "out_dir": str(out_dir),
        "indices": args.indices,
        "sample_ks": args.sample_ks,
        "rhos": args.rhos,
        "num_random_directions": args.num_random_directions,
        "seed": args.seed,
        "device": args.device,
        "eps_num": args.eps_num,
        "source_paths": sorted(set(source_paths)),
        "output_files": [
            str(out_dir / "finite_difference_linearity_by_sample_k_direction.csv"),
            str(out_dir / "finite_difference_linearity_aggregate_by_k_direction.csv"),
            str(out_dir / "finite_difference_linearity_aggregate_by_bin_direction.csv"),
            str(out_dir / "finite_difference_linearity_aggregate_by_k.csv"),
            str(out_dir / "finite_difference_linearity_aggregate_by_bin.csv"),
            str(out_dir / "finite_difference_linearity_aggregate_by_direction.csv"),
            str(out_dir / "summary.md"),
        ],
    }
    write_json(out_dir / "manifest.json", manifest)

    bin_columns = [
        "path_bin", "rho", "n_points",
        "loss_linearity_score_mean", "loss_linearity_score_std",
        "residual_linearity_score_mean", "residual_linearity_score_std",
        "delta_budget_ratio_mean",
    ]
    bin_direction_columns = [
        "path_bin", "direction_type", "rho", "n_points",
        "loss_linearity_score_mean", "loss_linearity_score_std",
        "residual_linearity_score_mean", "residual_linearity_score_std",
    ]
    k_columns = [
        "k", "rho", "n_points", "delta_budget_ratio_mean",
        "loss_linearity_score_mean", "loss_linearity_score_std",
        "residual_linearity_score_mean", "residual_linearity_score_std",
    ]
    lines = [
        "# Experiment 2: Symmetric Finite-Difference Local Linearity",
        "",
        "Scope: FNO Burgers `nu=0.001`, `loss3_original_pgd`, 100-step trajectory, saved every 5 steps.",
        "",
        "This experiment uses only three-point symmetric finite differences. It does not compute Hessians, dense Jacobians, or SVDs.",
        "",
        "Scores:",
        "",
        "- `loss_linearity_score = |L(z+rho u)-2L(z)+L(z-rho u)| / (|L(z+rho u)-L(z-rho u)| + eps)`",
        "- `residual_linearity_score = ||e(z+rho u)-2e(z)+e(z-rho u)|| / (||e(z+rho u)-e(z-rho u)|| + eps)`",
        "",
        "Smaller means more locally linear along that direction and radius.",
        "",
        "## Aggregate By Path Bin",
        "",
        markdown_table(by_bin, bin_columns),
        "",
        "## Aggregate By Path Bin And Direction",
        "",
        markdown_table(by_bin_direction, bin_direction_columns),
        "",
        "## Aggregate By k",
        "",
        markdown_table(by_k, k_columns),
        "",
        "## Output Files",
        "",
        "- `finite_difference_linearity_by_sample_k_direction.csv`",
        "- `finite_difference_linearity_aggregate_by_k_direction.csv`",
        "- `finite_difference_linearity_aggregate_by_bin_direction.csv`",
        "- `finite_difference_linearity_aggregate_by_k.csv`",
        "- `finite_difference_linearity_aggregate_by_bin.csv`",
        "- `finite_difference_linearity_aggregate_by_direction.csv`",
        "- `manifest.json`",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
