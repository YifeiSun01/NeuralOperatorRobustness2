#!/usr/bin/env python3
"""Matched-angle tangent-width probe for Loss3 high-loss basins.

This is a stricter follow-up to keep/remove.  It keeps the angular distance from
an optimizer endpoint fixed and compares two perturbation families on the epsilon
boundary:

    delta(theta, v) = eps * (cos(theta) * u_endpoint + sin(theta) * v),
    v perpendicular to u_endpoint.

The only difference is where the tangent direction v lives:

    top_tangent  : tangent projection of endpoint-local Jacobian/GN top modes
    orth_tangent : random tangent directions orthogonal to those top modes

If the endpoint-local Jacobian modes genuinely support a high-loss basin, then
top_tangent deviations should preserve high loss better than matched-angle
orth_tangent deviations.
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
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")
os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.probe_loss3_boundary_volume_20260623 import (  # noqa: E402
    ENDPOINT_METHODS,
    EPS,
    METHODS,
    BurgersBoundaryEvaluator,
    NSBoundaryEvaluator,
)
from tools.probe_loss3_jacobian_mode_causal_ablation_20260623 import (  # noqa: E402
    MatrixFreeBurgers,
    compute_top_basis,
    cosine,
    dot,
    flat,
    norm,
    unit,
)
from tools.probe_ns2d_jvp_vjp_spectrum_20260623 import MatrixFreeNS2D  # noqa: E402


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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def gram_schmidt(vectors: list[np.ndarray], *, tol: float = 1e-10) -> list[np.ndarray]:
    basis: list[np.ndarray] = []
    for vec in vectors:
        v = flat(vec).copy()
        for b in basis:
            v -= dot(v, b) * b
        n = norm(v)
        if n > tol:
            basis.append(v / n)
    return basis


def tangent_basis(endpoint: np.ndarray, basis: list[np.ndarray], k: int) -> tuple[np.ndarray, list[np.ndarray]]:
    u = unit(endpoint)
    tangent_vecs: list[np.ndarray] = []
    for b in basis[: int(k)]:
        v = flat(b) - dot(b, u) * u
        tangent_vecs.append(v)
    top_tangent = gram_schmidt(tangent_vecs)
    return u, top_tangent


def sample_top_tangent(top_tangent: list[np.ndarray], rng: np.random.Generator) -> np.ndarray | None:
    if not top_tangent:
        return None
    coeff = rng.standard_normal(len(top_tangent))
    v = np.zeros_like(top_tangent[0])
    for c, b in zip(coeff, top_tangent):
        v += float(c) * b
    return unit(v)


def sample_orth_tangent(dim: int, u: np.ndarray, top_tangent: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(dim)
    full_basis = [u] + list(top_tangent)
    for b in full_basis:
        v -= dot(v, b) * b
    tries = 0
    while norm(v) <= EPS and tries < 8:
        v = rng.standard_normal(dim)
        for b in full_basis:
            v -= dot(v, b) * b
        tries += 1
    return unit(v)


def boundary_candidate(endpoint: np.ndarray, tangent: np.ndarray, theta: float, epsilon: float) -> np.ndarray:
    shape = np.asarray(endpoint).shape
    u = unit(endpoint)
    v = unit(tangent)
    cand = float(epsilon) * (math.cos(float(theta)) * u + math.sin(float(theta)) * v)
    return cand.reshape(shape)


def add_threshold_rows(row: dict[str, Any], ratio: float, taus: list[float], rows: list[dict[str, Any]]) -> None:
    for tau in taus:
        rows.append({**row, "tau": float(tau), "ratio_to_lmax": ratio, "is_high_lmax": 1.0 if ratio >= tau else 0.0})


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("system", "endpoint_method", "k", "direction_family", "theta", "tau")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        ratios = np.asarray([fnum(r["ratio_to_lmax"]) for r in items], dtype=np.float64)
        highs = np.asarray([fnum(r["is_high_lmax"]) for r in items], dtype=np.float64)
        cosines = np.asarray([fnum(r["candidate_cosine_to_endpoint"]) for r in items], dtype=np.float64)
        ratios = ratios[np.isfinite(ratios)]
        highs = highs[np.isfinite(highs)]
        cosines = cosines[np.isfinite(cosines)]
        row = {k: v for k, v in zip(keys, key)}
        row["n"] = len(items)
        row["p_high_lmax"] = float(np.mean(highs)) if highs.size else float("nan")
        row["ratio_to_lmax_mean"] = float(np.mean(ratios)) if ratios.size else float("nan")
        row["ratio_to_lmax_min"] = float(np.min(ratios)) if ratios.size else float("nan")
        row["ratio_to_lmax_max"] = float(np.max(ratios)) if ratios.size else float("nan")
        row["candidate_cosine_to_endpoint_mean"] = float(np.mean(cosines)) if cosines.size else float("nan")
        out.append(row)
    return out


def run_burgers(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, list[dict[str, Any]]]:
    root = args.burgers_root if args.burgers_root.is_absolute() else PROJECT_ROOT / args.burgers_root
    ev = BurgersBoundaryEvaluator(root, args.burgers_num_samples, args.device)
    rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []

    for j, dataset_index in enumerate(ev.dataset_index.tolist()):
        x0_one = ev.x0_sel[j : j + 1]
        matrix = MatrixFreeBurgers(ev.problem, x0_one, ev.problem.args.q_order, args.jvp_mode, args.fd_step)
        sample_position = int(ev.sample_position[j])
        for method in args.endpoint_methods:
            endpoint = np.asarray(ev.final_by_method[method][j], dtype=np.float64)
            basis, prow = compute_top_basis(
                matrix,
                None,
                endpoint,
                max_k=max(args.ks),
                power_iters=args.power_iters,
                seed=args.seed + 1009 * j + 17 * len(method),
            )
            for r in prow:
                power_rows.append({**r, "system": "burgers1d", "endpoint_method": method, "dataset_index": int(dataset_index), "sample_position": sample_position})
            for k in args.ks:
                u, top_tangent = tangent_basis(endpoint, basis, int(k))
                for theta in args.angles:
                    for draw in range(args.draws):
                        candidates: list[tuple[str, np.ndarray]] = []
                        top_v = sample_top_tangent(top_tangent, rng)
                        if top_v is not None:
                            candidates.append(("top_tangent", boundary_candidate(endpoint, top_v, theta, ev.epsilon)))
                        orth_v = sample_orth_tangent(flat(endpoint).size, u, top_tangent, rng)
                        candidates.append(("orth_tangent", boundary_candidate(endpoint, orth_v, theta, ev.epsilon)))
                        # This loop is sample-local. Use the sample-local
                        # matrix evaluator instead of the all-sample Burgers
                        # evaluator, otherwise candidate count can differ from
                        # the saved N samples in ev.x0_sel.
                        losses = [matrix.eval_loss(c) for _label, c in candidates]
                        for idx, (family, cand) in enumerate(candidates):
                            ratio = float(losses[idx] / (float(ev.lmax[j]) + EPS))
                            add_threshold_rows(
                                {
                                    "system": "burgers1d",
                                    "endpoint_method": method,
                                    "dataset_index": int(dataset_index),
                                    "sample_position": sample_position,
                                    "k": int(k),
                                    "k_eff": min(int(k), len(basis)),
                                    "top_tangent_dim": len(top_tangent),
                                    "direction_family": family,
                                    "theta": float(theta),
                                    "draw_index": int(draw),
                                    "loss": float(losses[idx]),
                                    "lmax": float(ev.lmax[j]),
                                    "candidate_cosine_to_endpoint": cosine(cand, endpoint),
                                },
                                ratio,
                                args.taus,
                                rows,
                            )
    return {"rows": rows, "power": power_rows}


def run_ns2d(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, list[dict[str, Any]]]:
    root = args.ns_root if args.ns_root.is_absolute() else PROJECT_ROOT / args.ns_root
    ev = NSBoundaryEvaluator(root, args.ns_num_samples, args.device)
    matrix = MatrixFreeNS2D(ev.evaluator, args)
    rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []

    for dataset_i, dataset_index in enumerate(ev.datasets):
        problem = ev.problem_for(dataset_index)
        sample_position = int(ev.records[dataset_index][METHODS[0]]["sample_position"])
        for method in args.endpoint_methods:
            endpoint = np.asarray(ev.final(dataset_index, method), dtype=np.float64)
            basis, prow = compute_top_basis(
                matrix,
                problem,
                endpoint,
                max_k=max(args.ks),
                power_iters=args.power_iters,
                seed=args.seed + 20011 * dataset_i + 19 * len(method),
            )
            for r in prow:
                power_rows.append({**r, "system": "ns2d", "endpoint_method": method, "dataset_index": int(dataset_index), "sample_position": sample_position})
            candidates: list[np.ndarray] = []
            metas: list[dict[str, Any]] = []
            for k in args.ks:
                u, top_tangent = tangent_basis(endpoint, basis, int(k))
                for theta in args.angles:
                    for draw in range(args.draws):
                        top_v = sample_top_tangent(top_tangent, rng)
                        if top_v is not None:
                            cand = boundary_candidate(endpoint, top_v, theta, ev.epsilon)
                            candidates.append(cand)
                            metas.append(
                                {
                                    "system": "ns2d",
                                    "endpoint_method": method,
                                    "dataset_index": int(dataset_index),
                                    "sample_position": sample_position,
                                    "k": int(k),
                                    "k_eff": min(int(k), len(basis)),
                                    "top_tangent_dim": len(top_tangent),
                                    "direction_family": "top_tangent",
                                    "theta": float(theta),
                                    "draw_index": int(draw),
                                    "lmax": float(ev.lmax[dataset_index]),
                                    "candidate_cosine_to_endpoint": cosine(cand, endpoint),
                                }
                            )
                        orth_v = sample_orth_tangent(flat(endpoint).size, u, top_tangent, rng)
                        cand = boundary_candidate(endpoint, orth_v, theta, ev.epsilon)
                        candidates.append(cand)
                        metas.append(
                            {
                                "system": "ns2d",
                                "endpoint_method": method,
                                "dataset_index": int(dataset_index),
                                "sample_position": sample_position,
                                "k": int(k),
                                "k_eff": min(int(k), len(basis)),
                                "top_tangent_dim": len(top_tangent),
                                "direction_family": "orth_tangent",
                                "theta": float(theta),
                                "draw_index": int(draw),
                                "lmax": float(ev.lmax[dataset_index]),
                                "candidate_cosine_to_endpoint": cosine(cand, endpoint),
                            }
                        )
            losses = ev.eval_one(dataset_index, candidates)
            for meta, loss in zip(metas, losses):
                ratio = float(loss / (float(meta["lmax"]) + EPS))
                add_threshold_rows({**meta, "loss": float(loss)}, ratio, args.taus, rows)
    return {"rows": rows, "power": power_rows, "jvp_mode_used": dict(matrix.jvp_mode_used)}


def write_svg(summary: list[dict[str, Any]], path: Path, *, endpoint_method: str, k: int, tau: float) -> None:
    rows = [r for r in summary if str(r["endpoint_method"]) == endpoint_method and int(r["k"]) == int(k) and abs(float(r["tau"]) - tau) < 1e-9]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 520
    left, right, top, bottom = 82, 245, 55, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    angles = sorted({float(r["theta"]) for r in rows})
    xmax = max(angles) if angles else 1.0
    colors = {
        ("burgers1d", "top_tangent"): "#16a34a",
        ("burgers1d", "orth_tangent"): "#2563eb",
        ("ns2d", "top_tangent"): "#dc2626",
        ("ns2d", "orth_tangent"): "#9333ea",
    }

    def sx(x: float) -> float:
        return left + (x / xmax) * plot_w if xmax > 0 else left

    def sy(y: float) -> float:
        return top + (1.0 - y) * plot_h

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["system"]), str(row["direction_family"]))].append(row)

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(f'<text x="{left}" y="32" font-family="Arial, sans-serif" font-size="20" font-weight="700">Matched-angle tangent width: {endpoint_method}, k={k}, tau={tau:.2f}</text>')
    svg.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>')
    for y in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = sy(float(y))
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left - 12}" y="{yy + 5:.2f}" font-family="Arial, sans-serif" font-size="13" text-anchor="end">{y:.2f}</text>')
    for angle in angles:
        xx = sx(angle)
        svg.append(f'<text x="{xx:.2f}" y="{top + plot_h + 25}" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">{angle:g}</text>')
    for key, items in sorted(groups.items()):
        pts = sorted(items, key=lambda r: float(r["theta"]))
        color = colors.get(key, "#111827")
        coords = " ".join(f'{sx(float(r["theta"])):.2f},{sy(float(r["p_high_lmax"])):.2f}' for r in pts)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
        for r in pts:
            svg.append(f'<circle cx="{sx(float(r["theta"])):.2f}" cy="{sy(float(r["p_high_lmax"])):.2f}" r="4" fill="{color}"/>')
    svg.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 22}" font-family="Arial, sans-serif" font-size="15" text-anchor="middle">angle from endpoint theta</text>')
    svg.append(f'<text x="22" y="{top + plot_h / 2:.2f}" font-family="Arial, sans-serif" font-size="15" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2:.2f})">P(loss >= tau Lmax)</text>')
    lx, ly = left + plot_w + 30, top + 25
    svg.append(f'<text x="{lx}" y="{ly}" font-family="Arial, sans-serif" font-size="15" font-weight="700">Legend</text>')
    for i, key in enumerate(sorted(groups)):
        y = ly + 30 + 28 * i
        color = colors.get(key, "#111827")
        svg.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 34}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{lx + 45}" y="{y + 5}" font-family="Arial, sans-serif" font-size="13">{key[0]} {key[1]}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=["burgers", "ns2d"], default=["burgers", "ns2d"])
    parser.add_argument("--burgers-root", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation/raw/burgers_eps8_alpha0p3_steps100_N20_trace")
    parser.add_argument("--ns-root", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/ns2d/ns2d_eps32_alpha10_steps100_N3_trace")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation/jacobian_tangent_width_20260623")
    parser.add_argument("--burgers-num-samples", type=int, default=5)
    parser.add_argument("--ns-num-samples", type=int, default=2)
    parser.add_argument("--ks", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--angles", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20, 0.40])
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--power-iters", type=int, default=2)
    parser.add_argument("--endpoint-methods", nargs="+", default=ENDPOINT_METHODS)
    parser.add_argument("--taus", nargs="+", type=float, default=[0.90, 0.95, 0.99])
    parser.add_argument("--jvp-mode", choices=["auto", "autograd", "finite_difference"], default="auto")
    parser.add_argument("--fd-step", type=float, default=1e-2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260623)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    start = time.time()
    rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    mode_used: dict[str, Any] = {}

    if "ns2d" in args.systems:
        print("[tangent-width] ns2d", flush=True)
        out = run_ns2d(args, rng)
        rows.extend(out["rows"])
        power_rows.extend(out["power"])
        mode_used["ns2d"] = out.get("jvp_mode_used", {})
    if "burgers" in args.systems:
        print("[tangent-width] burgers", flush=True)
        out = run_burgers(args, rng)
        rows.extend(out["rows"])
        power_rows.extend(out["power"])

    summary = aggregate(rows)
    tables = out_dir / "tables"
    write_csv(tables / "jacobian_tangent_power_iterations.csv", power_rows)
    write_csv(tables / "jacobian_tangent_width_samples.csv", rows)
    write_csv(tables / "jacobian_tangent_width_summary.csv", summary)
    figure_paths: list[str] = []
    svg = out_dir / "figures" / "jacobian_tangent_width_steepest_replace_k8_p095.svg"
    write_svg(summary, svg, endpoint_method="steepest_replace", k=8, tau=0.95)
    if svg.exists():
        figure_paths.append(rel(svg))

    manifest = {
        "status": "completed",
        "runtime_seconds": time.time() - start,
        "out_dir": rel(out_dir),
        "systems": args.systems,
        "parameters": {
            "burgers_num_samples": args.burgers_num_samples,
            "ns_num_samples": args.ns_num_samples,
            "ks": args.ks,
            "angles": args.angles,
            "draws": args.draws,
            "power_iters": args.power_iters,
            "endpoint_methods": args.endpoint_methods,
            "taus": args.taus,
            "jvp_mode": args.jvp_mode,
            "fd_step": args.fd_step,
            "seed": args.seed,
            "device": args.device,
        },
        "row_counts": {
            "jacobian_tangent_power_iterations": len(power_rows),
            "jacobian_tangent_width_samples": len(rows),
            "jacobian_tangent_width_summary": len(summary),
        },
        "jvp_mode_used": mode_used,
        "figures": figure_paths,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
