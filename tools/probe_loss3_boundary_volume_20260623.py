#!/usr/bin/env python3
"""Boundary-volume and cap-volume probes for Loss3 high-loss regions.

The goal is to estimate how much of the epsilon-boundary is high-loss, and how
quickly Loss3 decays when moving away from optimizer endpoints on the boundary.
This is a direct test of the "Burgers has broad high-loss regions, NS2D has
narrow/path-dependent high-loss regions" hypothesis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import probe_burgers_first_order_prediction_20260622 as burgers_first_order
from tools import probe_ns2d_loss3_exact_mechanism_20260623 as ns_exact
from tools import run_loss3_direction_proposal_ablation as burgers_ablation


METHODS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
ENDPOINT_METHODS = ["steepest_add", "steepest_replace"]
TAUS = [0.90, 0.95, 0.99]
EPS = 1e-12


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


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def norm_l2(delta: np.ndarray) -> np.ndarray:
    flat = np.asarray(delta, dtype=np.float64).reshape(delta.shape[0], -1)
    return np.linalg.norm(flat, axis=1)


def normalize_l2(delta: np.ndarray) -> np.ndarray:
    flat = np.asarray(delta, dtype=np.float64).reshape(delta.shape[0], -1)
    n = np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), EPS)
    return (flat / n).reshape(delta.shape)


def dot_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    af = np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)
    bf = np.asarray(b, dtype=np.float64).reshape(b.shape[0], -1)
    return np.sum(af * bf, axis=1)


def random_boundary(shape: tuple[int, ...], epsilon: float, rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal(size=shape)
    return epsilon * normalize_l2(z)


def cap_boundary(center: np.ndarray, theta: float, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    u = normalize_l2(center)
    z = rng.standard_normal(size=center.shape)
    proj = dot_batch(z, u).reshape((-1,) + (1,) * (z.ndim - 1))
    v = normalize_l2(z - proj * u)
    out = epsilon * (math.cos(float(theta)) * u + math.sin(float(theta)) * v)
    return epsilon * normalize_l2(out)


def slerp_boundary(a: np.ndarray, b: np.ndarray, s: float, epsilon: float) -> np.ndarray:
    ua = normalize_l2(a)
    ub = normalize_l2(b)
    dots = np.clip(dot_batch(ua, ub), -0.999999, 0.999999)
    out = np.zeros_like(ua, dtype=np.float64)
    for i, c in enumerate(dots):
        omega = math.acos(float(c))
        denom = math.sin(omega)
        if abs(denom) < 1e-8:
            out[i] = normalize_l2(((1.0 - s) * ua[i : i + 1] + s * ub[i : i + 1]))[0]
        else:
            out[i] = (math.sin((1.0 - s) * omega) / denom) * ua[i] + (math.sin(s * omega) / denom) * ub[i]
    return epsilon * normalize_l2(out)


def aggregate_probability(rows: list[dict[str, Any]], keys: tuple[str, ...], denom_key: str, threshold_key: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = {k: v for k, v in zip(keys, key)}
        vals = np.asarray([fnum(item.get(threshold_key)) for item in items], dtype=np.float64)
        ratios = np.asarray([fnum(item.get(denom_key)) for item in items], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        ratios = ratios[np.isfinite(ratios)]
        row["n"] = len(items)
        row["probability"] = float(np.mean(vals)) if vals.size else float("nan")
        row[f"{denom_key}_mean"] = float(np.mean(ratios)) if ratios.size else float("nan")
        row[f"{denom_key}_median"] = float(np.median(ratios)) if ratios.size else float("nan")
        out.append(row)
    return out


def aggregate_connectivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["system"], row["pair"], row["dataset_index"])].append(row)
    out: list[dict[str, Any]] = []
    for (system, pair, dataset_index), items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        ratios = np.asarray([fnum(item["ratio_to_weaker_endpoint"]) for item in items], dtype=np.float64)
        ratios = ratios[np.isfinite(ratios)]
        loss = np.asarray([fnum(item["loss"]) for item in items], dtype=np.float64)
        loss = loss[np.isfinite(loss)]
        out.append(
            {
                "system": system,
                "pair": pair,
                "dataset_index": dataset_index,
                "n": len(items),
                "valley_ratio_min": float(np.min(ratios)) if ratios.size else float("nan"),
                "valley_ratio_mean": float(np.mean(ratios)) if ratios.size else float("nan"),
                "loss_min": float(np.min(loss)) if loss.size else float("nan"),
                "loss_max": float(np.max(loss)) if loss.size else float("nan"),
            }
        )
    return out


def cap_width_auc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["system"], row["endpoint_method"], row["tau"])].append(row)
    out: list[dict[str, Any]] = []
    for (system, method, tau), items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        theta_probs: list[tuple[float, float]] = []
        by_theta: dict[float, list[float]] = defaultdict(list)
        for item in items:
            by_theta[float(item["theta"])].append(fnum(item["is_high"]))
        for theta, vals in sorted(by_theta.items()):
            arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=np.float64)
            theta_probs.append((theta, float(np.mean(arr)) if arr.size else float("nan")))
        good = [(t, p) for t, p in theta_probs if math.isfinite(p)]
        theta50 = max([t for t, p in good if p >= 0.5], default=float("nan"))
        
        if len(good) >= 2:
            trapezoid = getattr(np, "trapezoid", None)
            if trapezoid is None:
                trapezoid = getattr(np, "trapz")
            auc = float(trapezoid([p for _t, p in good], [t for t, _p in good]))
        else:
            auc = float("nan")
        out.append(
            {
                "system": system,
                "endpoint_method": method,
                "tau": tau,
                "theta50": theta50,
                "auc_width": auc,
                "theta_points": len(good),
            }
        )
    return out


class BurgersBoundaryEvaluator:
    def __init__(self, root: Path, num_samples: int, device: str):
        self.root = root
        self.args = burgers_first_order.load_problem_args(root / "config.json", device=device)
        self.problem = burgers_ablation.AblationProblem(self.args)
        with np.load(root / "steepest_add" / "trajectory_samples.npz", allow_pickle=False) as z:
            self.dataset_index = np.asarray(z["dataset_index"], dtype=np.int64)[:num_samples]
            self.sample_position = np.asarray(z["sample_position"], dtype=np.int64)[:num_samples]
        self.x0_sel = self.problem.x0[self.sample_position.tolist()].detach()
        self.epsilon = float(self.problem.args.epsilon)
        self.final_by_method = self._load_finals(num_samples)
        final_losses = {m: self.eval(v) for m, v in self.final_by_method.items()}
        self.lmax = np.max(np.stack(list(final_losses.values()), axis=0), axis=0)
        self.endpoint_loss = final_losses

    def _load_finals(self, num_samples: int) -> dict[str, np.ndarray]:
        with np.load(self.root / "final_deltas.npz", allow_pickle=False) as z:
            methods = [str(x) for x in z["method"].tolist()]
            final = np.asarray(z["final_delta"], dtype=np.float64)
        out: dict[str, np.ndarray] = {}
        for method in METHODS:
            if method in methods:
                out[method] = final[methods.index(method), :num_samples]
        return out

    def eval(self, delta: np.ndarray) -> np.ndarray:
        return burgers_first_order.eval_loss(self.problem, self.x0_sel, np.asarray(delta, dtype=np.float64))


class NSBoundaryEvaluator:
    def __init__(self, root: Path, num_samples: int, device: str):
        self.root = root
        run_manifest = ns_exact.find_run_manifest(root)
        self.problem_args = ns_exact.load_problem_args(run_manifest, device)
        self.evaluator = ns_exact.ExactEvaluator(self.problem_args)
        records = ns_exact.trace_records(root, METHODS)
        by_dataset: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        for rec in records:
            by_dataset[int(rec["dataset_index"])][str(rec["method"])] = rec
        self.datasets = sorted([idx for idx, methods in by_dataset.items() if all(m in methods for m in METHODS)])[:num_samples]
        self.records = {idx: by_dataset[idx] for idx in self.datasets}
        self.epsilon = float(self.problem_args.epsilon)
        self.lmax: dict[int, float] = {}
        self.endpoint_loss: dict[int, dict[str, float]] = {}
        for idx in self.datasets:
            problem = self.problem_for(idx)
            finals = [self.records[idx][m]["delta"][-1] for m in METHODS]
            losses, _ = self.evaluator.eval_batch(problem, finals, chunk=1, residual=False)
            self.endpoint_loss[idx] = {m: float(losses[i]) for i, m in enumerate(METHODS)}
            self.lmax[idx] = float(max(losses))

    def problem_for(self, dataset_index: int) -> Any:
        return self.evaluator.problem_for_xclean(self.records[dataset_index][METHODS[0]]["x_clean"])

    def final(self, dataset_index: int, method: str) -> np.ndarray:
        return np.asarray(self.records[dataset_index][method]["delta"][-1], dtype=np.float64)

    def eval_one(self, dataset_index: int, deltas: list[np.ndarray]) -> list[float]:
        problem = self.problem_for(dataset_index)
        losses, _ = self.evaluator.eval_batch(problem, deltas, chunk=1, residual=False)
        return [float(x) for x in losses]


def add_threshold_rows(row: dict[str, Any], ratio: float, tau_values: list[float], rows: list[dict[str, Any]]) -> None:
    for tau in tau_values:
        rows.append({**row, "tau": tau, "ratio": ratio, "is_high": 1.0 if ratio >= tau else 0.0})


def run_burgers(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, list[dict[str, Any]]]:
    root = args.burgers_root if args.burgers_root.is_absolute() else PROJECT_ROOT / args.burgers_root
    ev = BurgersBoundaryEvaluator(root, args.burgers_num_samples, args.device)
    n = len(ev.dataset_index)
    shape = (n,) + tuple(next(iter(ev.final_by_method.values())).shape[1:])
    global_rows: list[dict[str, Any]] = []
    cap_rows: list[dict[str, Any]] = []
    connect_rows: list[dict[str, Any]] = []

    for draw in range(args.global_samples):
        delta = random_boundary(shape, ev.epsilon, rng)
        losses = ev.eval(delta)
        for j, dataset_index in enumerate(ev.dataset_index.tolist()):
            ratio = float(losses[j] / (ev.lmax[j] + EPS))
            add_threshold_rows(
                {
                    "system": "burgers1d",
                    "dataset_index": int(dataset_index),
                    "sample_position": int(ev.sample_position[j]),
                    "draw_index": draw,
                    "loss": float(losses[j]),
                    "lmax": float(ev.lmax[j]),
                },
                ratio,
                args.taus,
                global_rows,
            )

    for method in args.endpoint_methods:
        if method not in ev.final_by_method:
            continue
        center = ev.final_by_method[method]
        endpoint_loss = ev.endpoint_loss[method]
        for theta in args.angles:
            for draw in range(args.cap_samples):
                delta = cap_boundary(center, float(theta), ev.epsilon, rng)
                losses = ev.eval(delta)
                for j, dataset_index in enumerate(ev.dataset_index.tolist()):
                    ratio = float(losses[j] / (endpoint_loss[j] + EPS))
                    add_threshold_rows(
                        {
                            "system": "burgers1d",
                            "endpoint_method": method,
                            "dataset_index": int(dataset_index),
                            "sample_position": int(ev.sample_position[j]),
                            "theta": float(theta),
                            "draw_index": draw,
                            "loss": float(losses[j]),
                            "endpoint_loss": float(endpoint_loss[j]),
                        },
                        ratio,
                        args.taus,
                        cap_rows,
                    )

    for a, b in args.connectivity_pairs:
        if a not in ev.final_by_method or b not in ev.final_by_method:
            continue
        da = ev.final_by_method[a]
        db = ev.final_by_method[b]
        la = ev.endpoint_loss[a]
        lb = ev.endpoint_loss[b]
        for s in np.linspace(0.0, 1.0, args.connectivity_points):
            delta = slerp_boundary(da, db, float(s), ev.epsilon)
            losses = ev.eval(delta)
            for j, dataset_index in enumerate(ev.dataset_index.tolist()):
                weaker = min(float(la[j]), float(lb[j]))
                stronger = max(float(la[j]), float(lb[j]))
                connect_rows.append(
                    {
                        "system": "burgers1d",
                        "pair": f"{a}__{b}",
                        "dataset_index": int(dataset_index),
                        "sample_position": int(ev.sample_position[j]),
                        "s": float(s),
                        "loss": float(losses[j]),
                        "endpoint_a_loss": float(la[j]),
                        "endpoint_b_loss": float(lb[j]),
                        "ratio_to_weaker_endpoint": float(losses[j] / (weaker + EPS)),
                        "ratio_to_stronger_endpoint": float(losses[j] / (stronger + EPS)),
                    }
                )
    return {"global": global_rows, "cap": cap_rows, "connectivity": connect_rows}


def run_ns2d(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, list[dict[str, Any]]]:
    root = args.ns_root if args.ns_root.is_absolute() else PROJECT_ROOT / args.ns_root
    ev = NSBoundaryEvaluator(root, args.ns_num_samples, args.device)
    global_rows: list[dict[str, Any]] = []
    cap_rows: list[dict[str, Any]] = []
    connect_rows: list[dict[str, Any]] = []

    for dataset_index in ev.datasets:
        shape = (1,) + ev.final(dataset_index, METHODS[0]).shape
        for draw in range(args.global_samples):
            delta = random_boundary(shape, ev.epsilon, rng)[0]
            loss = ev.eval_one(dataset_index, [delta])[0]
            ratio = float(loss / (ev.lmax[dataset_index] + EPS))
            add_threshold_rows(
                {
                    "system": "ns2d",
                    "dataset_index": int(dataset_index),
                    "sample_position": int(ev.records[dataset_index][METHODS[0]]["sample_position"]),
                    "draw_index": draw,
                    "loss": float(loss),
                    "lmax": float(ev.lmax[dataset_index]),
                },
                ratio,
                args.taus,
                global_rows,
            )

        for method in args.endpoint_methods:
            center = ev.final(dataset_index, method)[None, ...]
            endpoint_loss = ev.endpoint_loss[dataset_index][method]
            for theta in args.angles:
                for draw in range(args.cap_samples):
                    delta = cap_boundary(center, float(theta), ev.epsilon, rng)[0]
                    loss = ev.eval_one(dataset_index, [delta])[0]
                    ratio = float(loss / (endpoint_loss + EPS))
                    add_threshold_rows(
                        {
                            "system": "ns2d",
                            "endpoint_method": method,
                            "dataset_index": int(dataset_index),
                            "sample_position": int(ev.records[dataset_index][METHODS[0]]["sample_position"]),
                            "theta": float(theta),
                            "draw_index": draw,
                            "loss": float(loss),
                            "endpoint_loss": float(endpoint_loss),
                        },
                        ratio,
                        args.taus,
                        cap_rows,
                    )

        for a, b in args.connectivity_pairs:
            da = ev.final(dataset_index, a)[None, ...]
            db = ev.final(dataset_index, b)[None, ...]
            la = ev.endpoint_loss[dataset_index][a]
            lb = ev.endpoint_loss[dataset_index][b]
            for s in np.linspace(0.0, 1.0, args.connectivity_points):
                delta = slerp_boundary(da, db, float(s), ev.epsilon)[0]
                loss = ev.eval_one(dataset_index, [delta])[0]
                weaker = min(float(la), float(lb))
                stronger = max(float(la), float(lb))
                connect_rows.append(
                    {
                        "system": "ns2d",
                        "pair": f"{a}__{b}",
                        "dataset_index": int(dataset_index),
                        "sample_position": int(ev.records[dataset_index][METHODS[0]]["sample_position"]),
                        "s": float(s),
                        "loss": float(loss),
                        "endpoint_a_loss": float(la),
                        "endpoint_b_loss": float(lb),
                        "ratio_to_weaker_endpoint": float(loss / (weaker + EPS)),
                        "ratio_to_stronger_endpoint": float(loss / (stronger + EPS)),
                    }
                )
    return {"global": global_rows, "cap": cap_rows, "connectivity": connect_rows}


def write_figures(out_dir: Path, global_summary: list[dict[str, Any]], cap_summary: list[dict[str, Any]], connectivity_summary: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    if global_summary:
        rows = [r for r in global_summary if abs(float(r["tau"]) - 0.95) < 1e-9]
        systems = sorted({str(r["system"]) for r in rows})
        vals = [np.mean([float(r["probability"]) for r in rows if r["system"] == system]) for system in systems]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.bar(systems, vals, color=["#4c78a8", "#f58518"][: len(systems)])
        ax.set_ylim(0, 1)
        ax.set_ylabel("estimated P(L >= 0.95 Lmax)")
        ax.set_title("Global Boundary High-Loss Volume")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = fig_dir / "global_boundary_p095.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(str(path))

    if cap_summary:
        rows = [r for r in cap_summary if abs(float(r["tau"]) - 0.95) < 1e-9]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for (system, method), items in sorted(defaultdict(list, { }).items()):
            pass
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[(str(r["system"]), str(r["endpoint_method"]))].append(r)
        for (system, method), items in sorted(groups.items()):
            xs = [float(r["theta"]) for r in sorted(items, key=lambda x: float(x["theta"]))]
            ys = [float(r["probability"]) for r in sorted(items, key=lambda x: float(x["theta"]))]
            ax.plot(xs, ys, marker="o", label=f"{system}:{method}", linewidth=1.8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("cap angle theta (radians)")
        ax.set_ylabel("P(L >= 0.95 endpoint loss)")
        ax.set_title("Endpoint Cap High-Loss Width")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = fig_dir / "endpoint_cap_p095_by_theta.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(str(path))

    if connectivity_summary:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        labels = [f"{r['system']}\n{r['pair']}\nidx={r['dataset_index']}" for r in connectivity_summary]
        vals = [float(r["valley_ratio_min"]) for r in connectivity_summary]
        ax.bar(range(len(vals)), vals)
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("min arc loss / weaker endpoint loss")
        ax.set_title("High-Loss Basin Connectivity")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = fig_dir / "connectivity_valley_ratio.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(str(path))

    return paths


def parse_pair(value: str) -> tuple[str, str]:
    if "__" in value:
        a, b = value.split("__", 1)
    elif "," in value:
        a, b = value.split(",", 1)
    else:
        raise argparse.ArgumentTypeError("pair must be METHOD__METHOD")
    return a, b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=["burgers", "ns2d"], default=["burgers", "ns2d"])
    parser.add_argument("--burgers-root", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation/raw/burgers_eps8_alpha0p3_steps100_N20_trace")
    parser.add_argument("--ns-root", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/ns2d/ns2d_eps32_alpha10_steps100_N3_trace")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation/boundary_volume_probe_20260623")
    parser.add_argument("--burgers-num-samples", type=int, default=5)
    parser.add_argument("--ns-num-samples", type=int, default=2)
    parser.add_argument("--global-samples", type=int, default=32)
    parser.add_argument("--cap-samples", type=int, default=8)
    parser.add_argument("--angles", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20, 0.40, 0.80])
    parser.add_argument("--taus", nargs="+", type=float, default=TAUS)
    parser.add_argument("--endpoint-methods", nargs="+", default=ENDPOINT_METHODS)
    parser.add_argument("--connectivity-pairs", nargs="+", type=parse_pair, default=[("steepest_replace", "steepest_add")])
    parser.add_argument("--connectivity-points", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    start = time.time()

    all_global: list[dict[str, Any]] = []
    all_cap: list[dict[str, Any]] = []
    all_conn: list[dict[str, Any]] = []

    # Run NS2D before Burgers. The Burgers JAX solver setup can change process
    # dtype behavior in a way that conflicts with the recurrent-NS JAX bridge.
    if "ns2d" in args.systems:
        print("[boundary-volume] start ns2d", flush=True)
        out = run_ns2d(args, rng)
        all_global.extend(out["global"])
        all_cap.extend(out["cap"])
        all_conn.extend(out["connectivity"])
        print("[boundary-volume] done ns2d", flush=True)
    if "burgers" in args.systems:
        print("[boundary-volume] start burgers", flush=True)
        out = run_burgers(args, rng)
        all_global.extend(out["global"])
        all_cap.extend(out["cap"])
        all_conn.extend(out["connectivity"])
        print("[boundary-volume] done burgers", flush=True)

    global_summary = aggregate_probability(all_global, ("system", "tau"), "ratio", "is_high")
    cap_summary = aggregate_probability(all_cap, ("system", "endpoint_method", "theta", "tau"), "ratio", "is_high")
    cap_width = cap_width_auc(all_cap)
    conn_summary = aggregate_connectivity(all_conn)

    tables = out_dir / "tables"
    write_csv(tables / "global_boundary_volume.csv", all_global)
    write_csv(tables / "global_boundary_volume_summary.csv", global_summary)
    write_csv(tables / "endpoint_cap_volume.csv", all_cap)
    write_csv(tables / "endpoint_cap_volume_summary.csv", cap_summary)
    write_csv(tables / "endpoint_cap_width_summary.csv", cap_width)
    write_csv(tables / "basin_connectivity.csv", all_conn)
    write_csv(tables / "basin_connectivity_summary.csv", conn_summary)
    figures = write_figures(out_dir, global_summary, cap_summary, conn_summary)
    manifest = {
        "status": "completed",
        "out_dir": str(out_dir),
        "systems": args.systems,
        "burgers_root": str(args.burgers_root),
        "ns_root": str(args.ns_root),
        "row_counts": {
            "global_boundary_volume": len(all_global),
            "endpoint_cap_volume": len(all_cap),
            "basin_connectivity": len(all_conn),
        },
        "summary_counts": {
            "global": len(global_summary),
            "cap": len(cap_summary),
            "cap_width": len(cap_width),
            "connectivity": len(conn_summary),
        },
        "parameters": {
            "burgers_num_samples": args.burgers_num_samples,
            "ns_num_samples": args.ns_num_samples,
            "global_samples": args.global_samples,
            "cap_samples": args.cap_samples,
            "angles": args.angles,
            "taus": args.taus,
            "endpoint_methods": args.endpoint_methods,
            "connectivity_pairs": [f"{a}__{b}" for a, b in args.connectivity_pairs],
            "connectivity_points": args.connectivity_points,
            "seed": args.seed,
        },
        "figures": figures,
        "runtime_seconds": time.time() - start,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
