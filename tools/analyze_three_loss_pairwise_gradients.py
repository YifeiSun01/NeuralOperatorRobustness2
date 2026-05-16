"""Pairwise gradient-angle diagnostics for loss1/loss2/loss3.

This script fixes the same saved perturbation delta_k and computes the true
nonlinear autograd gradients of the three original objectives:

  L1(delta) = ||f(x + delta) - f(x)||_q
  L2(delta) = ||f(x + delta) - j(x)||_q
  L3(delta) = ||f(x + delta) - j(x + delta)||_q

It then records pairwise angles among grad_delta L1, grad_delta L2, and
grad_delta L3.  Unlike the outward-growth diagnostic, this is a native
loss1/loss2/loss3 gradient-direction comparison.
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
    / "three_loss_pairwise_gradients_20260516"
    / "fno_nu0p001"
)

LOSS_NAMES = ("loss1", "loss2", "loss3")
PAIR_NAMES = (("loss1", "loss2"), ("loss1", "loss3"), ("loss2", "loss3"))


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


def grad_three_original_losses(problem: Any, x_adv_np: np.ndarray) -> dict[str, Any]:
    import torch

    x = torch.as_tensor(x_adv_np[None, ...], device=problem.device, dtype=torch.float32).detach().requires_grad_(True)
    f_adv = problem.model_forward(x)
    g_adv = problem.solver_forward(x, allow_grad=True)
    q = problem.args.loss_q_order

    losses = {
        "loss1": attack.torch_norm(f_adv - problem.f0, q),
        "loss2": attack.torch_norm(f_adv - problem.g0, q),
        "loss3": attack.torch_norm(f_adv - g_adv, q),
    }

    grads: dict[str, np.ndarray] = {}
    values: dict[str, float] = {}
    for i, name in enumerate(LOSS_NAMES):
        retain = i < len(LOSS_NAMES) - 1
        values[name] = float(losses[name].detach().cpu())
        grad = torch.autograd.grad(losses[name], x, retain_graph=retain, allow_unused=False)[0]
        grads[name] = grad.detach().cpu().numpy()[0].astype(np.float64)
    attack.sync_torch(torch, problem.device)
    return {"loss_values": values, "grads": grads}


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)

    metrics = [
        "angle_loss1_loss2_deg",
        "angle_loss1_loss3_deg",
        "angle_loss2_loss3_deg",
        "cos_loss1_loss2",
        "cos_loss1_loss3",
        "cos_loss2_loss3",
        "loss1_value",
        "loss2_value",
        "loss3_value",
        "grad_loss1_norm_l2",
        "grad_loss2_norm_l2",
        "grad_loss3_norm_l2",
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
    parser.add_argument(
        "--delta-sources",
        nargs="+",
        default=["loss1_original_pgd", "loss2_original_pgd", "loss3_original_pgd"],
    )
    parser.add_argument("--sample-ks", nargs="+", type=int, default=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    source_paths: list[str] = []

    for index in args.indices:
        index_dir = args.result_root / f"index_{index:03d}"
        config_path = index_dir / "loss1_original_pgd" / "config.json"
        problem_args = load_config_args(config_path, device=args.device)
        problem_args.index = index
        problem = attack.AttackProblem(problem_args)
        source_paths.append(str(config_path))
        print(f"[index] {index} device={problem.device}", flush=True)

        for delta_source in args.delta_sources:
            traj_path = index_dir / delta_source / "trajectory.npz"
            traj = np.load(traj_path)
            source_paths.append(str(traj_path))
            ks = traj["k"].astype(int)
            x_adv = traj["x_adv"]
            delta = traj["delta"]
            for pos, k in enumerate(ks):
                if int(k) not in args.sample_ks:
                    continue
                info = grad_three_original_losses(problem, x_adv[pos])
                grads = info["grads"]
                loss_values = info["loss_values"]
                pair_metrics: dict[str, float] = {}
                for a, b in PAIR_NAMES:
                    c = cosine(grads[a], grads[b])
                    pair_metrics[f"cos_{a}_{b}"] = c
                    pair_metrics[f"angle_{a}_{b}_deg"] = angle_deg_from_cos(c)
                d = delta[pos].astype(np.float64)
                delta_norm = vector_norm(d)
                row = {
                    "index": index,
                    "delta_source": delta_source,
                    "k": int(k),
                    "delta_norm_l2": delta_norm,
                    "delta_budget_ratio": delta_norm / float(problem_args.epsilon),
                    "trajectory_path": str(traj_path),
                }
                for name in LOSS_NAMES:
                    row[f"{name}_value"] = loss_values[name]
                    row[f"grad_{name}_norm_l2"] = vector_norm(grads[name])
                row.update(pair_metrics)
                rows.append(row)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "pairwise_three_loss_gradients.csv", rows)
    write_csv(out_dir / "summary_by_delta_source_k.csv", summarize(rows, ("delta_source", "k")))
    write_csv(out_dir / "summary_by_k.csv", summarize(rows, ("k",)))
    write_csv(out_dir / "summary_by_delta_source.csv", summarize(rows, ("delta_source",)))
    manifest = {
        "result_root": str(args.result_root),
        "out_dir": str(out_dir),
        "indices": args.indices,
        "delta_sources": args.delta_sources,
        "sample_ks": args.sample_ks,
        "device_requested": args.device,
        "source_paths": sorted(set(source_paths)),
        "output_files": [
            str(out_dir / "pairwise_three_loss_gradients.csv"),
            str(out_dir / "summary_by_delta_source_k.csv"),
            str(out_dir / "summary_by_k.csv"),
            str(out_dir / "summary_by_delta_source.csv"),
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
