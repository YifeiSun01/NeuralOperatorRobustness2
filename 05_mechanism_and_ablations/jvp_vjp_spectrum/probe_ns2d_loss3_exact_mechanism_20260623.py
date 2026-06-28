#!/usr/bin/env python3
"""Exact NS2D Loss3 mechanism probes from saved step-sample traces.

This script reuses saved NS2D core4 traces and re-evaluates the recurrent FNO
and differentiable solver at diagnostic points.  It is intentionally small-N
and GPU-only: the goal is to test first-order full-budget prediction,
multi-scale local linearity, and boundary arc width without rerunning the whole
optimizer.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NS_ATTACK_PATH = PROJECT_ROOT / "2D_NS_FNO2d_recurrent/perturbation_methods/attack_ns2d_recurrent_core4.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_ns_module():
    spec = importlib.util.spec_from_file_location("attack_ns2d_recurrent_core4", NS_ATTACK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {NS_ATTACK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ns = load_ns_module()

METHODS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
PAIR_LIST = [
    ("steepest_replace", "steepest_add"),
    ("steepest_replace", "raw_add"),
    ("raw_add", "steepest_add"),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


def find_run_manifest(root: Path) -> Path:
    candidates = sorted(root.glob("mode_*/manifest.json"))
    if (root / "manifest.json").exists():
        return root / "manifest.json"
    if not candidates:
        raise FileNotFoundError(f"No NS2D manifest under {root}")
    if len(candidates) > 1:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[0]


def path_arg(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and ("/" in value or value.endswith(".pt")):
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return value


def load_problem_args(manifest_path: Path, device: str) -> SimpleNamespace:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = dict(payload["args"])
    for key in ("checkpoint", "dictionary_path", "test_path", "out_root", "initial_delta_npz"):
        if key in raw:
            raw[key] = path_arg(raw[key])
    raw["device"] = device
    raw["p_order"] = ns.parse_norm_order(raw.get("p", raw.get("p_order", 2.0)))
    raw["q_order"] = ns.parse_norm_order(raw.get("q", raw.get("q_order", 2.0)))
    raw["mode_spec"] = ns.resolve_mode_spec(raw.get("mode_spec", "all_w"))
    raw["parameter_sweep"] = raw.get("parameter_sweep") or [{"epsilon": raw["epsilon"], "alpha": raw["alpha"], "tag": f"eps{raw['epsilon']}_alpha{raw['alpha']}"}]
    raw["sweep_is_multi"] = len(raw["parameter_sweep"]) > 1
    return SimpleNamespace(**raw)


def fnum(x: Any) -> float:
    try:
        out = float(x)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.asarray(a, dtype=np.float64).ravel() * np.asarray(b, dtype=np.float64).ravel()))


def norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64).ravel()))


def normalize_l2(a: np.ndarray) -> np.ndarray:
    n = norm(a)
    if n <= 1e-12:
        return np.zeros_like(a, dtype=np.float64)
    return np.asarray(a, dtype=np.float64) / n


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


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        pred = [fnum(r.get("predicted_gain")) for r in items]
        true = [fnum(r.get("true_gain")) for r in items]
        rel = np.asarray([fnum(r.get("relative_abs_error")) for r in items], dtype=np.float64)
        rel = rel[np.isfinite(rel)]
        row = {k: v for k, v in zip(keys, key)}
        row["n"] = len(items)
        row["pearson_pred_true"] = corrcoef_safe(pred, true)
        row["spearman_pred_true"] = spearman_safe(pred, true)
        row["predicted_gain_mean"] = float(np.nanmean(pred)) if pred else float("nan")
        row["true_gain_mean"] = float(np.nanmean(true)) if true else float("nan")
        row["relative_abs_error_mean"] = float(np.mean(rel)) if rel.size else float("nan")
        row["relative_abs_error_median"] = float(np.median(rel)) if rel.size else float("nan")
        return_row = row
        out.append(return_row)
    return out


def aggregate_mean(rows: list[dict[str, Any]], keys: tuple[str, ...], value_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = {k: v for k, v in zip(keys, key)}
        row["n"] = len(items)
        for value_key in value_keys:
            vals = np.asarray([fnum(r.get(value_key)) for r in items], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row[f"{value_key}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{value_key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out.append(row)
    return out


class ExactEvaluator:
    def __init__(self, args: SimpleNamespace):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; refusing CPU fallback.")
        self.torch = torch
        self.args = args
        self.device = torch.device("cuda")
        self.model = ns.load_recurrent_model(args, self.device)
        self.rollout_solver = ns.DifferentiableNSRollout(args.nu, args.fixed_step, args.solver_remat, args.solver_remat_chunk_steps)

    def problem_for_xclean(self, x_clean_np: np.ndarray):
        x0 = self.torch.as_tensor(x_clean_np[None, ...], device=self.device, dtype=self.torch.float32)
        return ns.NS2DRecurrentProblem(self.args, x0, clean_target=None, model=self.model, rollout_solver=self.rollout_solver, dictionary=None)

    def project(self, delta_np: np.ndarray) -> np.ndarray:
        delta = self.torch.as_tensor(delta_np[None, ...], device=self.device, dtype=self.torch.float32)
        projected = ns.project_delta(delta, float(self.args.epsilon), self.args.p_order)[0]
        return ns.tensor_to_numpy(projected.detach()).astype(np.float64)

    def eval_batch(self, problem: Any, deltas_np: list[np.ndarray], *, chunk: int = 1, residual: bool = False) -> tuple[list[float], list[np.ndarray]]:
        losses: list[float] = []
        residuals: list[np.ndarray] = []
        for start in range(0, len(deltas_np), chunk):
            batch = np.stack(deltas_np[start : start + chunk], axis=0).astype(np.float32)
            delta = self.torch.as_tensor(batch, device=self.device, dtype=self.torch.float32)
            with self.torch.no_grad():
                x_adv = problem.x0 + delta
                loss_dict, pred, target, _aux = problem.active_losses(x_adv, "loss3")
                loss = ns.tensor_to_numpy(loss_dict["loss3"].detach()).astype(np.float64)
                losses.extend(float(x) for x in loss)
                if residual:
                    res = ns.tensor_to_numpy((pred - target).detach()).astype(np.float64)
                    residuals.extend([res[i] for i in range(res.shape[0])])
            del delta, x_adv, loss_dict, pred, target
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        return losses, residuals

    def grad_at(self, problem: Any, delta_np: np.ndarray) -> tuple[float, np.ndarray]:
        delta = self.torch.as_tensor(delta_np[None, ...], device=self.device, dtype=self.torch.float32)
        x_adv = (problem.x0 + delta).detach().requires_grad_(True)
        loss_dict, pred, target, _aux = problem.active_losses(x_adv, "loss3")
        objective = loss_dict["loss3"].sum()
        objective.backward()
        loss = float(ns.tensor_to_numpy(loss_dict["loss3"].detach())[0])
        grad = ns.tensor_to_numpy(x_adv.grad.detach()[0]).astype(np.float64)
        del delta, x_adv, loss_dict, pred, target, objective
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return loss, grad


def trace_records(root: Path, methods: list[str]) -> list[dict[str, Any]]:
    paths = sorted(root.glob("mode_*/batch_*/loss3/*/step_sample_trace.npz"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        method = path.parent.name
        if method not in methods:
            continue
        with np.load(path, allow_pickle=False) as z:
            rows.append(
                {
                    "path": path,
                    "method": method,
                    "dataset_index": int(np.asarray(z["dataset_index"]).item()),
                    "sample_position": int(np.asarray(z["sample_position"]).item()),
                    "x_clean": np.asarray(z["x_clean"], dtype=np.float64),
                    "k": np.asarray(z["k"], dtype=np.int64),
                    "delta": np.asarray(z["delta"], dtype=np.float64),
                    "direction": np.asarray(z["direction"], dtype=np.float64),
                    "direction_available": np.asarray(z["direction_available"], dtype=bool),
                    "grad": np.asarray(z["grad"], dtype=np.float64),
                    "grad_available": np.asarray(z["grad_available"], dtype=bool),
                    "residual": np.asarray(z["adv_model_minus_solver"], dtype=np.float64),
                }
            )
    return rows


def random_boundary(shape: tuple[int, ...], epsilon: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(shape)
    return epsilon * normalize_l2(v)


def run_first_order(evaluator: ExactEvaluator, records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec_i, rec in enumerate(records):
        problem = evaluator.problem_for_xclean(rec["x_clean"])
        ks = [int(x) for x in rec["k"].tolist()]
        k_to_i = {k: i for i, k in enumerate(ks)}
        final_delta = rec["delta"][-1]
        rand_delta = random_boundary(final_delta.shape, float(evaluator.args.epsilon), args.seed + 1009 * rec_i)
        print(f"[ns-first-order] method={rec['method']} dataset={rec['dataset_index']}", flush=True)
        for k in args.first_order_ks:
            if k not in k_to_i:
                continue
            i = k_to_i[k]
            if not bool(rec["grad_available"][i]):
                continue
            current = rec["delta"][i]
            current_loss, grad = evaluator.grad_at(problem, current)
            candidates: list[tuple[str, np.ndarray]] = [
                ("final_delta", final_delta),
                ("random_boundary", rand_delta),
            ]
            if k + 1 in k_to_i:
                candidates.append(("actual_next_step", rec["delta"][k_to_i[k + 1]]))
            if bool(rec["direction_available"][i]):
                direction = rec["direction"][i]
                candidates.append(("add_from_saved_direction", evaluator.project(current + float(evaluator.args.alpha) * direction)))
                candidates.append(("replace_from_saved_direction", evaluator.project(float(evaluator.args.epsilon) * direction)))

            losses, _ = evaluator.eval_batch(problem, [cand for _label, cand in candidates], chunk=args.eval_chunk)
            for (label, cand), cand_loss in zip(candidates, losses):
                predicted = dot(grad, cand - current)
                true_gain = float(cand_loss - current_loss)
                rows.append(
                    {
                        "method": rec["method"],
                        "dataset_index": rec["dataset_index"],
                        "sample_position": rec["sample_position"],
                        "k": int(k),
                        "candidate": label,
                        "current_loss": float(current_loss),
                        "candidate_loss": float(cand_loss),
                        "predicted_gain": float(predicted),
                        "true_gain": true_gain,
                        "relative_abs_error": abs(float(predicted) - true_gain) / (abs(true_gain) + args.eta),
                        "current_delta_l2": norm(current),
                        "candidate_delta_l2": norm(cand),
                    }
                )
    return rows


def direction_for_label(rec: dict[str, Any], i: int, label: str, seed: int) -> np.ndarray | None:
    current = rec["delta"][i]
    if label == "grad":
        if not bool(rec["grad_available"][i]):
            return None
        return normalize_l2(rec["grad"][i])
    if label == "saved_direction":
        if not bool(rec["direction_available"][i]):
            return None
        return normalize_l2(rec["direction"][i])
    if label == "radial":
        return normalize_l2(current)
    if label == "final_delta":
        return normalize_l2(rec["delta"][-1] - current)
    if label == "random":
        rng = np.random.default_rng(seed)
        return normalize_l2(rng.standard_normal(current.shape))
    raise ValueError(label)


def run_linearity(evaluator: ExactEvaluator, records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = [r for r in records if r["method"] in set(args.linearity_methods)]
    for rec_i, rec in enumerate(selected):
        problem = evaluator.problem_for_xclean(rec["x_clean"])
        ks = [int(x) for x in rec["k"].tolist()]
        k_to_i = {k: i for i, k in enumerate(ks)}
        print(f"[ns-linearity] method={rec['method']} dataset={rec['dataset_index']}", flush=True)
        for k in args.linearity_ks:
            if k not in k_to_i:
                continue
            i = k_to_i[k]
            current = rec["delta"][i]
            current_loss = fnum(norm(rec["residual"][i]))
            current_residual = rec["residual"][i]
            for direction_label in args.linearity_directions:
                u = direction_for_label(rec, i, direction_label, args.seed + rec_i * 917 + k)
                if u is None or norm(u) <= 0:
                    continue
                for frac in args.radius_fractions:
                    rho = float(frac) * float(evaluator.args.epsilon)
                    plus = evaluator.project(current + rho * u)
                    minus = evaluator.project(current - rho * u)
                    losses, residuals = evaluator.eval_batch(problem, [plus, minus], chunk=args.eval_chunk, residual=True)
                    lp, lm = losses
                    rp, rm = residuals
                    c_phi = abs(lp - 2.0 * current_loss + lm) / (abs(lp - lm) + args.eta)
                    c_res = norm(rp - 2.0 * current_residual + rm) / (norm(rp - rm) + args.eta)
                    rows.append(
                        {
                            "method": rec["method"],
                            "dataset_index": rec["dataset_index"],
                            "k": int(k),
                            "direction": direction_label,
                            "radius_fraction": float(frac),
                            "rho": rho,
                            "current_loss": float(current_loss),
                            "plus_loss": float(lp),
                            "minus_loss": float(lm),
                            "scalar_linearity_c_phi": float(c_phi),
                            "residual_linearity_c_r": float(c_res),
                            "plus_delta_l2": norm(plus),
                            "minus_delta_l2": norm(minus),
                        }
                    )
    return rows


def run_boundary_arcs(evaluator: ExactEvaluator, records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_dataset: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        by_dataset[int(rec["dataset_index"])][rec["method"]] = rec

    rows: list[dict[str, Any]] = []
    s_values = np.linspace(0.0, 1.0, int(args.arc_points))
    for dataset_index, methods in sorted(by_dataset.items()):
        if not methods:
            continue
        any_rec = next(iter(methods.values()))
        problem = evaluator.problem_for_xclean(any_rec["x_clean"])
        print(f"[ns-boundary-arc] dataset={dataset_index}", flush=True)
        for a, b in PAIR_LIST:
            if a not in methods or b not in methods:
                continue
            da = methods[a]["delta"][-1]
            db = methods[b]["delta"][-1]
            candidates: list[np.ndarray] = []
            for s in s_values:
                v = (1.0 - float(s)) * da + float(s) * db
                candidates.append(float(evaluator.args.epsilon) * normalize_l2(v))
            losses, _ = evaluator.eval_batch(problem, candidates, chunk=args.eval_chunk)
            for s, cand, loss in zip(s_values, candidates, losses):
                rows.append(
                    {
                        "dataset_index": int(dataset_index),
                        "pair": f"{a}__{b}",
                        "s": float(s),
                        "loss3": float(loss),
                        "delta_l2": norm(cand),
                    }
                )
    return rows


def write_report(out_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# NS2D Loss3 Exact Mechanism Probe - 2026-06-23",
        "",
        f"Status: `{manifest['status']}`.",
        "",
        "This run reuses saved NS2D step-sample traces and re-evaluates exact Loss3 at diagnostic points.",
        "",
        "## Outputs",
        "",
        f"- First-order rows: `{rel(out_dir / 'first_order_rows.csv')}`",
        f"- First-order by candidate: `{rel(out_dir / 'first_order_by_candidate.csv')}`",
        f"- Linearity rows: `{rel(out_dir / 'linearity_rows.csv')}`",
        f"- Linearity aggregate: `{rel(out_dir / 'linearity_aggregate.csv')}`",
        f"- Boundary arc rows: `{rel(out_dir / 'boundary_arc.csv')}`",
        f"- Boundary arc aggregate: `{rel(out_dir / 'boundary_arc_aggregate.csv')}`",
        "",
        "Use this as small-N mechanism evidence, not the final formal curve statistic.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/ns2d/ns2d_eps32_alpha10_steps100_N3_trace")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation/ns2d_exact_mechanism")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--first-order-ks", nargs="+", type=int, default=[0, 1, 2, 5, 10, 20, 50])
    parser.add_argument("--linearity-methods", nargs="+", default=["steepest_add", "steepest_replace"])
    parser.add_argument("--linearity-ks", nargs="+", type=int, default=[1, 10, 50])
    parser.add_argument("--linearity-directions", nargs="+", default=["grad", "saved_direction", "radial", "final_delta", "random"])
    parser.add_argument("--radius-fractions", nargs="+", type=float, default=[0.01, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--arc-points", type=int, default=9)
    parser.add_argument("--eval-chunk", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--eta", type=float, default=1e-9)
    parser.add_argument("--skip-if-complete", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    manifest_path = out_dir / "manifest.json"
    if args.skip_if_complete and manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("status") == "completed":
                print(f"[skip] completed manifest exists: {manifest_path}", flush=True)
                return
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = find_run_manifest(root)
    problem_args = load_problem_args(run_manifest, args.device)
    records = trace_records(root, args.methods)
    if not records:
        raise RuntimeError(f"No traces found under {root}")

    evaluator = ExactEvaluator(problem_args)
    first_rows = run_first_order(evaluator, records, args)
    line_rows = run_linearity(evaluator, records, args)
    arc_rows = run_boundary_arcs(evaluator, records, args)

    write_csv(out_dir / "first_order_rows.csv", first_rows)
    write_csv(out_dir / "first_order_by_candidate.csv", summarize(first_rows, ("candidate",)))
    write_csv(out_dir / "first_order_by_method_candidate.csv", summarize(first_rows, ("method", "candidate")))
    write_csv(out_dir / "linearity_rows.csv", line_rows)
    write_csv(
        out_dir / "linearity_aggregate.csv",
        aggregate_mean(line_rows, ("method", "direction", "radius_fraction"), ("scalar_linearity_c_phi", "residual_linearity_c_r")),
    )
    write_csv(out_dir / "boundary_arc.csv", arc_rows)
    write_csv(out_dir / "boundary_arc_aggregate.csv", aggregate_mean(arc_rows, ("pair", "s"), ("loss3",)))

    manifest = {
        "status": "completed",
        "root": str(root),
        "run_manifest": str(run_manifest),
        "out_dir": str(out_dir),
        "methods": args.methods,
        "num_trace_records": len(records),
        "row_counts": {
            "first_order": len(first_rows),
            "linearity": len(line_rows),
            "boundary_arc": len(arc_rows),
        },
        "first_order_ks": args.first_order_ks,
        "linearity_methods": args.linearity_methods,
        "linearity_ks": args.linearity_ks,
        "radius_fractions": args.radius_fractions,
        "arc_points": args.arc_points,
    }
    write_json(manifest_path, manifest)
    write_report(out_dir, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
