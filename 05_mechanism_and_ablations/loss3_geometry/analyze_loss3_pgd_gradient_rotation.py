#!/usr/bin/env python3
"""Analyze Loss3 gradient rotation along a saved PGD trajectory.

This is the first, cheap path-linearity validation experiment:

  g_k = grad_z L3(z_k),  z_k = x0 + delta_k

For each saved trajectory point, the script compares the true Loss3 gradient
against the previous saved gradient and against the clean/k0 reference.
It intentionally does not compute Hessians, dense Jacobians, or SVDs.
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
    / "loss3_original_pgd_trajectories"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "forensics"
    / "loss3_simple_path_linearity_20260517"
    / "fno_nu0p001"
    / "experiment1_pgd_gradient_rotation"
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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(af, bf) / denom)


def angle_deg_from_cos(c: float) -> float:
    if not math.isfinite(c):
        return float("nan")
    return float(math.degrees(math.acos(max(-1.0, min(1.0, c)))))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return angle_deg_from_cos(cosine(a, b))


def mean_std_min_max(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), float(arr.max())


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


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)

    metrics = [
        "loss3_value",
        "grad_norm_l2",
        "delta_norm_l2",
        "delta_budget_ratio",
        "angle_grad_to_previous_saved_deg",
        "angle_grad_to_k0_deg",
        "angle_grad_to_clean_input_deg",
        "angle_grad_to_step_deg",
        "step_delta_norm_l2",
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
    parser.add_argument("--sample-ks", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument("--device", default="cuda")
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
        x_adv = traj["x_adv"]
        delta = traj["delta"].astype(np.float64)
        x_clean = traj["x_clean"].astype(np.float64)
        available = {int(k): pos for pos, k in enumerate(ks)}
        selected_ks = [int(k) for k in args.sample_ks if int(k) in available]
        if not selected_ks:
            raise ValueError(f"No requested k values found in {traj_path}; available={ks.tolist()}")

        print(f"[index] {index} selected_ks={selected_ks}", flush=True)
        source_paths.extend([str(config_path), str(traj_path)])

        clean_value, clean_grad = grad_loss3(problem, x_clean)
        grads: dict[int, np.ndarray] = {}
        values: dict[int, float] = {}
        for k in selected_ks:
            pos = available[k]
            value, grad = grad_loss3(problem, x_adv[pos])
            values[k] = value
            grads[k] = grad

        k0 = selected_ks[0]
        k0_grad = grads[k0]
        prev_k: int | None = None
        for k in selected_ks:
            pos = available[k]
            d = delta[pos]
            grad = grads[k]
            row: dict[str, Any] = {
                "sample_index": index,
                "k": k,
                "loss3_value": values[k],
                "clean_loss3_value": clean_value,
                "grad_norm_l2": vector_norm(grad),
                "delta_norm_l2": vector_norm(d),
                "delta_budget_ratio": vector_norm(d) / float(problem_args.epsilon),
                "angle_grad_to_k0_deg": angle_deg(grad, k0_grad),
                "angle_grad_to_clean_input_deg": angle_deg(grad, clean_grad),
                "trajectory_path": str(traj_path),
            }
            if prev_k is None:
                row["previous_saved_k"] = ""
                row["angle_grad_to_previous_saved_deg"] = float("nan")
                row["step_delta_norm_l2"] = float("nan")
                row["angle_grad_to_step_deg"] = float("nan")
            else:
                prev_pos = available[prev_k]
                prev_delta = delta[prev_pos]
                step_delta = d - prev_delta
                step_norm = vector_norm(step_delta)
                row["previous_saved_k"] = prev_k
                row["angle_grad_to_previous_saved_deg"] = angle_deg(grad, grads[prev_k])
                row["step_delta_norm_l2"] = step_norm
                row["angle_grad_to_step_deg"] = angle_deg(grad, step_delta) if step_norm > 0.0 else float("nan")
            rows.append(row)
            prev_k = k

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    by_k = summarize(rows, ("k",))
    by_sample = summarize(rows, ("sample_index",))
    write_csv(out_dir / "pgd_gradient_rotation_by_sample_k.csv", rows)
    write_csv(out_dir / "pgd_gradient_rotation_aggregate_by_k.csv", by_k)
    write_csv(out_dir / "pgd_gradient_rotation_summary_by_sample.csv", by_sample)
    manifest = {
        "experiment": "Experiment 1: True PGD Gradient Rotation",
        "trajectory_root": str(args.trajectory_root),
        "out_dir": str(out_dir),
        "indices": args.indices,
        "sample_ks": args.sample_ks,
        "device": args.device,
        "source_paths": sorted(set(source_paths)),
        "output_files": [
            str(out_dir / "pgd_gradient_rotation_by_sample_k.csv"),
            str(out_dir / "pgd_gradient_rotation_aggregate_by_k.csv"),
            str(out_dir / "pgd_gradient_rotation_summary_by_sample.csv"),
            str(out_dir / "summary.md"),
        ],
    }
    write_json(out_dir / "manifest.json", manifest)

    columns = [
        "k",
        "n_points",
        "delta_budget_ratio_mean",
        "loss3_value_mean",
        "grad_norm_l2_mean",
        "angle_grad_to_previous_saved_deg_mean",
        "angle_grad_to_k0_deg_mean",
        "angle_grad_to_clean_input_deg_mean",
        "angle_grad_to_step_deg_mean",
    ]
    summary_lines = [
        "# Experiment 1: True PGD Gradient Rotation",
        "",
        "Scope: FNO Burgers `nu=0.001`, `loss3_original_pgd`, five samples, saved steps every 5.",
        "",
        "This experiment recomputes the true Loss3 gradient at each saved PGD point and compares it against the previous saved point and the clean/k0 reference. It does not compute Hessians, dense Jacobians, or SVDs.",
        "",
        "## Aggregate By k",
        "",
        markdown_table(by_k, columns),
        "",
        "## Output Files",
        "",
        "- `pgd_gradient_rotation_by_sample_k.csv`",
        "- `pgd_gradient_rotation_aggregate_by_k.csv`",
        "- `pgd_gradient_rotation_summary_by_sample.csv`",
        "- `manifest.json`",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
