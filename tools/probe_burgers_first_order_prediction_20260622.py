#!/usr/bin/env python3
"""First-order prediction probe for Burgers replace/add trajectories.

This reads a saved ``run_loss3_direction_proposal_ablation.py`` output root,
recomputes the true Loss3 gradient at selected trajectory points, and compares
the first-order predicted gain with the actual true Loss3 gain for add,
replace, final, actual-next, and random-boundary candidate moves.
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

from tools import run_loss3_direction_proposal_ablation as ablation  # noqa: E402


DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "analysis_outputs/mechanism_20260622/replace_add_validation/raw/"
    / "burgers_eps8_alpha0p3_steps100_N5_trace"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs/mechanism_20260622/replace_add_validation/diagnostics/"
    / "burgers_first_order_prediction"
)


def as_path(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and ("/" in value or value.endswith(".pt") or value.endswith(".npz")):
        return Path(value)
    return value


def load_problem_args(config_path: Path, *, device: str) -> SimpleNamespace:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for key in (
        "out_root",
        "burgers_test_path",
        "burgers_torch_checkpoint",
        "deeponet_checkpoint",
        "deeponet_output_transform_stats",
    ):
        if key in cfg:
            cfg[key] = as_path(cfg[key])
    cfg["device"] = device
    cfg["p_order"] = ablation.parse_norm(str(cfg.get("p", cfg.get("p_order", "2"))))
    cfg["q_order"] = ablation.parse_norm(str(cfg.get("q", cfg.get("q_order", "2"))))
    if cfg.get("dataset_indices") == []:
        cfg["dataset_indices"] = None
    return SimpleNamespace(**cfg)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dot_per_sample(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    af = np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)
    bf = np.asarray(b, dtype=np.float64).reshape(b.shape[0], -1)
    return np.sum(af * bf, axis=1)


def corrcoef_safe(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if xa.size < 2 or float(np.std(xa)) == 0.0 or float(np.std(ya)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman_safe(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if xa.size < 2:
        return float("nan")
    return corrcoef_safe(rankdata(xa).tolist(), rankdata(ya).tolist())


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def eval_loss(problem: Any, x0_sel: Any, delta_np: np.ndarray) -> np.ndarray:
    import torch

    delta = torch.as_tensor(delta_np, device=problem.device, dtype=torch.float32)
    with torch.no_grad():
        losses, _f, _g, _r = ablation.loss3_norms(problem, x0_sel + delta, allow_solver_grad=False)
    ablation.sync_torch(torch, problem.device)
    return tensor_to_numpy(losses["loss3_q"]).astype(np.float64)


def grad_at(problem: Any, x0_sel: Any, delta_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    delta = torch.as_tensor(delta_np, device=problem.device, dtype=torch.float32)
    x_adv = (x0_sel + delta).detach().requires_grad_(True)
    losses, _f, _g, _r = ablation.loss3_norms(problem, x_adv, allow_solver_grad=True)
    objective = losses["loss3_q"]
    objective.sum().backward()
    grad = tensor_to_numpy(x_adv.grad.detach()).astype(np.float64)
    loss = tensor_to_numpy(objective.detach()).astype(np.float64)
    ablation.sync_torch(torch, problem.device)
    return loss, grad


def project_np(problem: Any, delta_np: np.ndarray) -> np.ndarray:
    import torch

    delta = torch.as_tensor(delta_np, device=problem.device, dtype=torch.float32)
    projected = ablation.project_delta(delta, float(problem.args.epsilon), problem.args.p_order)
    return tensor_to_numpy(projected.detach()).astype(np.float64)


def load_direction_for_k(method_dir: Path) -> dict[int, np.ndarray]:
    path = method_dir / "direction_trace_selected.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as z:
        ks = [int(x) for x in z["k"].tolist()]
        directions = np.asarray(z["direction"], dtype=np.float64)
    return {k: directions[i] for i, k in enumerate(ks)}


def random_boundary(shape: tuple[int, ...], epsilon: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(size=shape)
    flat = arr.reshape(arr.shape[0], -1)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return (epsilon * flat / norm).reshape(shape)


def stable_method_offset(method: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(method))


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        pred = [float(r["predicted_gain"]) for r in items]
        true = [float(r["true_gain"]) for r in items]
        rel = np.asarray([float(r["relative_abs_error"]) for r in items], dtype=np.float64)
        mask = np.isfinite(rel)
        row = {k: v for k, v in zip(keys, key)}
        row["n"] = len(items)
        row["pearson_pred_true"] = corrcoef_safe(pred, true)
        row["spearman_pred_true"] = spearman_safe(pred, true)
        row["predicted_gain_mean"] = float(np.nanmean(pred)) if pred else float("nan")
        row["true_gain_mean"] = float(np.nanmean(true)) if true else float("nan")
        row["relative_abs_error_mean"] = float(np.mean(rel[mask])) if np.any(mask) else float("nan")
        row["relative_abs_error_median"] = float(np.median(rel[mask])) if np.any(mask) else float("nan")
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--methods", nargs="+", default=["raw_add", "raw_replace", "steepest_add", "steepest_replace"])
    parser.add_argument("--sample-ks", nargs="+", type=int, default=[0, 1, 2, 5, 10, 20, 50, 100])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--eta", type=float, default=1e-9)
    args = parser.parse_args()

    run_root = args.run_root if args.run_root.is_absolute() else PROJECT_ROOT / args.run_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    problem_args = load_problem_args(run_root / "config.json", device=args.device)
    problem = ablation.AblationProblem(problem_args)

    rows: list[dict[str, Any]] = []
    for method in args.methods:
        method_dir = run_root / method
        traj_path = method_dir / "trajectory_samples.npz"
        if not traj_path.exists():
            print(f"[skip] missing {traj_path}", flush=True)
            continue
        with np.load(traj_path, allow_pickle=False) as z:
            ks = [int(x) for x in z["k"].tolist()]
            dataset_index = np.asarray(z["dataset_index"], dtype=np.int64)
            sample_position = np.asarray(z["sample_position"], dtype=np.int64)
            delta = np.asarray(z["delta"], dtype=np.float64)
        direction_by_k = load_direction_for_k(method_dir)
        k_to_i = {k: i for i, k in enumerate(ks)}
        x0_sel = problem.x0[sample_position.tolist()].detach()
        final_delta = delta[-1]
        rand_boundary = random_boundary(final_delta.shape, float(problem.args.epsilon), args.seed + stable_method_offset(method))

        print(f"[method] {method} samples={len(sample_position)} saved_steps={len(ks)}", flush=True)
        for k in args.sample_ks:
            if k not in k_to_i:
                continue
            i = k_to_i[k]
            current_delta = delta[i]
            current_loss, grad = grad_at(problem, x0_sel, current_delta)
            candidates: list[tuple[str, np.ndarray]] = [
                ("final_delta", final_delta),
                ("random_boundary", rand_boundary),
            ]
            if k + 1 in k_to_i:
                candidates.append(("actual_next_step", delta[k_to_i[k + 1]]))
            direction = direction_by_k.get(k)
            if direction is not None:
                candidates.append(("add_from_saved_direction", project_np(problem, current_delta + float(problem.args.alpha) * direction)))
                candidates.append(("replace_from_saved_direction", project_np(problem, float(problem.args.epsilon) * direction)))

            for candidate_label, candidate_delta in candidates:
                pred_gain = dot_per_sample(grad, candidate_delta - current_delta)
                cand_loss = eval_loss(problem, x0_sel, candidate_delta)
                true_gain = cand_loss - current_loss
                for j, ds_idx in enumerate(dataset_index.tolist()):
                    abs_err = abs(float(pred_gain[j] - true_gain[j]))
                    rows.append(
                        {
                            "method": method,
                            "candidate": candidate_label,
                            "k": k,
                            "dataset_index": int(ds_idx),
                            "sample_position": int(sample_position[j]),
                            "current_loss": float(current_loss[j]),
                            "candidate_loss": float(cand_loss[j]),
                            "predicted_gain": float(pred_gain[j]),
                            "true_gain": float(true_gain[j]),
                            "abs_error": abs_err,
                            "relative_abs_error": abs_err / (abs(float(true_gain[j])) + float(args.eta)),
                        }
                    )

    by_method_candidate = summarize(rows, ("method", "candidate"))
    by_method_candidate_k = summarize(rows, ("method", "candidate", "k"))
    by_candidate = summarize(rows, ("candidate",))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "first_order_prediction_rows.csv", rows)
    write_csv(out_dir / "first_order_prediction_by_method_candidate.csv", by_method_candidate)
    write_csv(out_dir / "first_order_prediction_by_method_candidate_k.csv", by_method_candidate_k)
    write_csv(out_dir / "first_order_prediction_by_candidate.csv", by_candidate)
    write_json(
        out_dir / "manifest.json",
        {
            "run_root": str(run_root),
            "out_dir": str(out_dir),
            "methods": args.methods,
            "sample_ks": args.sample_ks,
            "device": args.device,
            "rows": len(rows),
            "outputs": [
                "first_order_prediction_rows.csv",
                "first_order_prediction_by_method_candidate.csv",
                "first_order_prediction_by_method_candidate_k.csv",
                "first_order_prediction_by_candidate.csv",
            ],
        },
    )
    print(f"[done] rows={len(rows)} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
