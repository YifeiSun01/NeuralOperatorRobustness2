"""Compare true nonlinear endpoint and residual-movement gradients.

This script addresses the limitation of local-A diagnostics.  It does not use a
fixed clean-point Jacobian.  Instead, at each saved trajectory point x_adv it
directly differentiates the nonlinear objectives:

  endpoint: ||f(x_adv) - j(x_adv)||_2
  movement: ||(f(x_adv) - j(x_adv)) - (f(x0) - j(x0))||_2

The comparison is made at the same x_adv / delta_k, so it is an
apples-to-apples nonlinear gradient comparison.
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

DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "forensics"
    / "outward_growth_direction_20260515"
    / "fno_nu0p001"
    / "true_nonlinear_endpoint_vs_movement_gradients"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def mean_std(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), float(arr.max())


def grad_endpoint_and_movement(problem: Any, x_adv_np: np.ndarray) -> dict[str, Any]:
    import torch

    x = torch.as_tensor(x_adv_np[None, ...], device=problem.device, dtype=torch.float32).detach().requires_grad_(True)
    f_adv = problem.model_forward(x)
    g_adv = problem.solver_forward(x, allow_grad=True)
    residual_adv = f_adv - g_adv
    residual_clean = (problem.f0 - problem.g0).detach()
    residual_increment = residual_adv - residual_clean

    endpoint_loss = attack.torch_norm(residual_adv, 2.0)
    movement_loss = attack.torch_norm(residual_increment, 2.0)

    endpoint_grad = torch.autograd.grad(endpoint_loss, x, retain_graph=True)[0].detach().cpu().numpy()[0].astype(np.float64)
    if float(movement_loss.detach().cpu()) == 0.0:
        movement_grad = np.zeros_like(endpoint_grad)
    else:
        movement_grad = torch.autograd.grad(movement_loss, x)[0].detach().cpu().numpy()[0].astype(np.float64)
    attack.sync_torch(torch, problem.device)

    return {
        "endpoint_loss": float(endpoint_loss.detach().cpu()),
        "movement_loss": float(movement_loss.detach().cpu()),
        "endpoint_grad": endpoint_grad,
        "movement_grad": movement_grad,
    }


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    metrics = [
        "endpoint_movement_cos",
        "endpoint_movement_angle_deg",
        "endpoint_loss",
        "movement_loss",
        "endpoint_grad_norm_l2",
        "movement_grad_norm_l2",
        "movement_to_endpoint_grad_norm_ratio",
        "delta_norm_l2",
        "delta_budget_ratio",
    ]
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        row: dict[str, Any] = {name: value for name, value in zip(keys, key)}
        row["n_points"] = len(items)
        for metric in metrics:
            m, s, mn, mx = mean_std([float(item[metric]) for item in items])
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
            row[f"{metric}_min"] = mn
            row[f"{metric}_max"] = mx
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 7, 40, 47, 115])
    parser.add_argument("--attacks", nargs="+", default=["loss1", "loss2", "loss3"])
    parser.add_argument("--sample-ks", nargs="+", type=int, default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    source_paths: list[str] = []

    for index in args.indices:
        index_dir = args.result_root / f"index_{index:03d}"
        config_path = index_dir / "loss1_original_pgd" / "config.json"
        problem_args = load_config_args(config_path)
        problem_args.index = index
        problem = attack.AttackProblem(problem_args)
        source_paths.append(str(config_path))
        print(f"[index] {index}", flush=True)

        for attack_loss in args.attacks:
            traj_path = index_dir / f"{attack_loss}_original_pgd" / "trajectory.npz"
            traj = np.load(traj_path)
            source_paths.append(str(traj_path))
            ks = traj["k"].astype(int)
            x_adv = traj["x_adv"]
            delta = traj["delta"]
            for pos, k in enumerate(ks):
                if int(k) not in args.sample_ks:
                    continue
                grad_info = grad_endpoint_and_movement(problem, x_adv[pos])
                endpoint_grad = grad_info["endpoint_grad"]
                movement_grad = grad_info["movement_grad"]
                c = cosine(endpoint_grad, movement_grad)
                endpoint_norm = vector_norm(endpoint_grad)
                movement_norm = vector_norm(movement_grad)
                d = delta[pos].astype(np.float64)
                delta_norm = vector_norm(d)
                rows.append(
                    {
                        "index": index,
                        "attack_loss": attack_loss,
                        "k": int(k),
                        "endpoint_loss": grad_info["endpoint_loss"],
                        "movement_loss": grad_info["movement_loss"],
                        "endpoint_grad_norm_l2": endpoint_norm,
                        "movement_grad_norm_l2": movement_norm,
                        "movement_to_endpoint_grad_norm_ratio": (
                            movement_norm / endpoint_norm if endpoint_norm > 0.0 else math.nan
                        ),
                        "endpoint_movement_cos": c,
                        "endpoint_movement_angle_deg": angle_deg_from_cos(c),
                        "delta_norm_l2": delta_norm,
                        "delta_budget_ratio": delta_norm / float(problem_args.epsilon),
                        "trajectory_path": str(traj_path),
                    }
                )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "true_nonlinear_endpoint_vs_movement_gradients.csv", rows)
    write_csv(out_dir / "summary_by_attack_k.csv", summarize(rows, ("attack_loss", "k")))
    write_csv(out_dir / "summary_by_attack.csv", summarize(rows, ("attack_loss",)))
    manifest = {
        "result_root": str(args.result_root),
        "out_dir": str(out_dir),
        "indices": args.indices,
        "attacks": args.attacks,
        "sample_ks": args.sample_ks,
        "source_paths": sorted(set(source_paths)),
        "output_files": [
            str(out_dir / "true_nonlinear_endpoint_vs_movement_gradients.csv"),
            str(out_dir / "summary_by_attack_k.csv"),
            str(out_dir / "summary_by_attack.csv"),
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
