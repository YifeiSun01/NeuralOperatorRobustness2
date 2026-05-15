#!/usr/bin/env python3
"""Analyze saved loss-gradient trajectories by recomputing all three loss gradients.

The trajectory experiment saves x_adv/delta/model/solver outputs at selected
steps, but it only saves the optimized loss gradient for the loss that was being
attacked. This postprocessor reloads each saved trajectory point and computes
all three exact original-objective gradients at that same point:

  grad loss1, grad loss2, grad loss3

It then reports pairwise gradient cosine/angle statistics and loss values.
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

LOSSES = ("loss1", "loss2", "loss3")
PATH_KEYS = (
    "output_dir",
    "burgers_test_path",
    "burgers_torch_checkpoint",
    "ns_test_path",
    "ns_torch_checkpoint",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_config_args(config_path: Path) -> SimpleNamespace:
    cfg = read_json(config_path)
    for key in PATH_KEYS:
        if key in cfg and cfg[key] is not None:
            cfg[key] = Path(cfg[key])
    cfg.setdefault("runtime_workarounds", True)
    cfg.setdefault("disable_cudnn", True)
    cfg.setdefault("prepend_env_ptxas", True)
    cfg.setdefault("device", None)
    cfg.setdefault("cpu", False)
    cfg["objective_variant"] = "original"
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
    return float(np.linalg.norm(x.reshape(-1)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(af, bf) / denom)


def angle_deg_from_cos(c: float) -> float:
    if not math.isfinite(c):
        return float("nan")
    return float(math.degrees(math.acos(max(-1.0, min(1.0, c)))))


def grad_and_value(problem: Any, x_adv_np: np.ndarray, loss_type: str) -> tuple[float, np.ndarray]:
    import torch

    x = torch.as_tensor(x_adv_np[None, ...], device=problem.device, dtype=torch.float32).detach().requires_grad_(True)
    objective, _base_loss, _baseline, _delta_p, _f_delta, _target = attack.active_objective(
        problem, x, loss_type, problem.args.frame_mode
    )
    objective.backward()
    attack.sync_torch(torch, problem.device)
    grad = x.grad.detach().cpu().numpy()[0].astype(np.float64)
    value = float(objective.detach().cpu())
    del x, objective
    return value, grad


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def summarize_group(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    metrics = [
        "cos_g1_g2", "angle_g1_g2_deg",
        "cos_g1_g3", "angle_g1_g3_deg",
        "cos_g2_g3", "angle_g2_g3_deg",
        "cos_delta_g1", "cos_delta_g2", "cos_delta_g3",
        "loss1_value", "loss2_value", "loss3_value",
        "grad1_norm_l2", "grad2_norm_l2", "grad3_norm_l2",
        "delta_norm_l2", "delta_budget_ratio",
    ]
    for key, items in sorted(groups.items()):
        row: dict[str, Any] = {name: value for name, value in zip(group_keys, key)}
        row["n_points"] = len(items)
        for metric in metrics:
            m, s = mean_std([float(item[metric]) for item in items])
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
        out.append(row)
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in shown:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                if math.isnan(val):
                    vals.append("nan")
                elif abs(val) >= 100 or (abs(val) < 1e-3 and val != 0):
                    vals.append(f"{val:.3e}")
                else:
                    vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-ks", nargs="+", type=int, default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument("--include-k0", action="store_true")
    args = parser.parse_args()

    result_root = args.result_root
    out_dir = args.out_dir or result_root / "gradient_direction_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_ks = list(args.sample_ks)
    if args.include_k0 and 0 not in sample_ks:
        sample_ks = [0, *sample_ks]

    index_dirs = sorted(p for p in result_root.glob("index_*") if p.is_dir())
    per_point: list[dict[str, Any]] = []

    for index_dir in index_dirs:
        index = int(index_dir.name.split("_")[1])
        config_path = index_dir / "loss1_original_pgd" / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        problem_args = load_config_args(config_path)
        problem_args.index = index
        problem = attack.AttackProblem(problem_args)
        print(f"[index] {index}", flush=True)

        for attack_loss in LOSSES:
            traj_path = index_dir / f"{attack_loss}_original_pgd" / "trajectory.npz"
            if not traj_path.exists():
                raise FileNotFoundError(traj_path)
            traj = np.load(traj_path)
            ks = traj["k"].astype(int)
            x_adv = traj["x_adv"]
            delta = traj["delta"]
            for pos, k in enumerate(ks):
                if int(k) not in sample_ks:
                    continue
                values: dict[str, float] = {}
                grads: dict[str, np.ndarray] = {}
                for loss in LOSSES:
                    value, grad = grad_and_value(problem, x_adv[pos], loss)
                    values[loss] = value
                    grads[loss] = grad

                d = delta[pos].astype(np.float64)
                delta_l2 = vector_norm(d)
                row = {
                    "index": index,
                    "attack_loss": attack_loss,
                    "k": int(k),
                    "loss1_value": values["loss1"],
                    "loss2_value": values["loss2"],
                    "loss3_value": values["loss3"],
                    "grad1_norm_l2": vector_norm(grads["loss1"]),
                    "grad2_norm_l2": vector_norm(grads["loss2"]),
                    "grad3_norm_l2": vector_norm(grads["loss3"]),
                    "delta_norm_l2": delta_l2,
                    "delta_budget_ratio": delta_l2 / float(problem_args.epsilon),
                }
                pair_names = [
                    ("g1_g2", "loss1", "loss2"),
                    ("g1_g3", "loss1", "loss3"),
                    ("g2_g3", "loss2", "loss3"),
                ]
                for name, a, b in pair_names:
                    c = cosine(grads[a], grads[b])
                    row[f"cos_{name}"] = c
                    row[f"angle_{name}_deg"] = angle_deg_from_cos(c)
                for name, loss in (("g1", "loss1"), ("g2", "loss2"), ("g3", "loss3")):
                    row[f"cos_delta_{name}"] = cosine(d, grads[loss])
                per_point.append(row)

    fieldnames = [
        "index", "attack_loss", "k",
        "loss1_value", "loss2_value", "loss3_value",
        "grad1_norm_l2", "grad2_norm_l2", "grad3_norm_l2",
        "delta_norm_l2", "delta_budget_ratio",
        "cos_g1_g2", "angle_g1_g2_deg",
        "cos_g1_g3", "angle_g1_g3_deg",
        "cos_g2_g3", "angle_g2_g3_deg",
        "cos_delta_g1", "cos_delta_g2", "cos_delta_g3",
    ]
    write_csv(out_dir / "per_point_loss_gradient_angles.csv", per_point, fieldnames)

    by_attack = summarize_group(per_point, ("attack_loss",))
    by_k = summarize_group(per_point, ("k",))
    by_attack_k = summarize_group(per_point, ("attack_loss", "k"))
    by_index_attack = summarize_group(per_point, ("index", "attack_loss"))
    final_rows = [row for row in per_point if row["k"] == max(sample_ks)]
    final_by_attack = summarize_group(final_rows, ("attack_loss",))

    write_csv(out_dir / "summary_by_attack_loss.csv", by_attack)
    write_csv(out_dir / "summary_by_k.csv", by_k)
    write_csv(out_dir / "summary_by_attack_loss_and_k.csv", by_attack_k)
    write_csv(out_dir / "summary_by_index_and_attack_loss.csv", by_index_attack)
    write_csv(out_dir / "final_k_summary_by_attack_loss.csv", final_by_attack)

    overall = summarize_group(per_point, tuple())
    overall_row = overall[0] if overall else {}
    write_json(out_dir / "summary.json", {
        "result_root": str(result_root),
        "out_dir": str(out_dir),
        "sample_ks": sample_ks,
        "n_points": len(per_point),
        "overall": overall_row,
    })

    lines = [
        "# FNO nu=0.001 Loss-Gradient Path Analysis",
        "",
        "Date: 2026-05-15 UTC",
        "",
        f"Result root: `{result_root}`",
        f"Analyzed points: `{len(per_point)}` = 5 indices x 3 attack losses x {len(sample_ks)} saved steps.",
        "",
        "This postprocess recomputes exact original-objective gradients for `loss1`, `loss2`, and `loss3` at each saved `x + delta_k` point. The trajectory files themselves save `x_adv`, `delta`, model output, solver output, and residual output; they do not save all three cross-loss gradients at every point.",
        "",
        "## Overall Gradient Angles",
        "",
        markdown_table([overall_row], [
            "n_points",
            "cos_g1_g2_mean", "angle_g1_g2_deg_mean",
            "cos_g1_g3_mean", "angle_g1_g3_deg_mean",
            "cos_g2_g3_mean", "angle_g2_g3_deg_mean",
        ]),
        "",
        "## By Attack Loss",
        "",
        markdown_table(by_attack, [
            "attack_loss", "n_points",
            "angle_g1_g2_deg_mean", "angle_g1_g3_deg_mean", "angle_g2_g3_deg_mean",
            "cos_g1_g2_mean", "cos_g1_g3_mean", "cos_g2_g3_mean",
            "loss1_value_mean", "loss2_value_mean", "loss3_value_mean",
        ]),
        "",
        "## Final Step k=max",
        "",
        markdown_table(final_by_attack, [
            "attack_loss", "n_points",
            "loss1_value_mean", "loss2_value_mean", "loss3_value_mean",
            "angle_g1_g2_deg_mean", "angle_g1_g3_deg_mean", "angle_g2_g3_deg_mean",
            "delta_budget_ratio_mean",
        ]),
        "",
        "## By Saved Step",
        "",
        markdown_table(by_k, [
            "k", "n_points",
            "angle_g1_g2_deg_mean", "angle_g1_g3_deg_mean", "angle_g2_g3_deg_mean",
            "loss1_value_mean", "loss2_value_mean", "loss3_value_mean",
            "delta_budget_ratio_mean",
        ]),
        "",
        "## Files",
        "",
        "- `per_point_loss_gradient_angles.csv`",
        "- `summary_by_attack_loss.csv`",
        "- `summary_by_k.csv`",
        "- `summary_by_attack_loss_and_k.csv`",
        "- `summary_by_index_and_attack_loss.csv`",
        "- `final_k_summary_by_attack_loss.csv`",
        "- `summary.json`",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
