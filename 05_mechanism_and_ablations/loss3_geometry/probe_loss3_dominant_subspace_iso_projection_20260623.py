#!/usr/bin/env python3
"""Dominant-subspace iso-projection probe for Loss3 optimizer mechanisms.

Question tested:
    If a system's high Loss3 values are controlled by a few coherent/dominant
    directions, then keeping the endpoint projection onto that subspace while
    randomizing the orthogonal residual should preserve high loss. If the
    high-loss region is a narrow/path-dependent ridge, this randomization should
    rapidly destroy high loss.

The subspace used here is a data-driven endpoint PCA/SVD subspace built from the
saved optimizer endpoints.  This deliberately tests the optimizer-relevant
dominant directions rather than an abstract global Fourier or Jacobian basis.
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


def flat(delta: np.ndarray) -> np.ndarray:
    return np.asarray(delta, dtype=np.float64).reshape(-1)


def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vec, dtype=np.float64).reshape(-1)))


def unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = norm(vec)
    if n <= EPS:
        return np.zeros_like(vec)
    return vec / n


def project_onto(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.size == 0:
        return np.zeros_like(vec, dtype=np.float64)
    return basis @ (basis.T @ vec)


def random_orthogonal_unit(dim: int, basis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal(dim).astype(np.float64)
    if basis.size:
        z = z - project_onto(z, basis)
    z_norm = norm(z)
    tries = 0
    while z_norm <= EPS and tries < 8:
        z = rng.standard_normal(dim).astype(np.float64)
        if basis.size:
            z = z - project_onto(z, basis)
        z_norm = norm(z)
        tries += 1
    if z_norm <= EPS:
        return np.zeros(dim, dtype=np.float64)
    return z / z_norm


def endpoint_svd(vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x = np.stack([unit(v) for v in vectors], axis=0)
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    energy = s**2
    denom = float(np.sum(energy))
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for i, value in enumerate(s.tolist(), start=1):
        frac = float((value * value) / denom) if denom > 0 else float("nan")
        cumulative += frac if math.isfinite(frac) else 0.0
        rows.append(
            {
                "component": i,
                "singular_value": float(value),
                "energy_fraction": frac,
                "cumulative_energy_fraction": cumulative,
            }
        )
    return vt.T.astype(np.float64), s.astype(np.float64), rows


def make_candidate(
    delta_vec: np.ndarray,
    basis: np.ndarray,
    epsilon: float,
    beta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    d = np.asarray(delta_vec, dtype=np.float64).reshape(-1)
    d_norm = norm(d)
    p = project_onto(d, basis)
    p_norm = norm(p)
    p_energy = float((p_norm * p_norm) / max(d_norm * d_norm, EPS))
    p_norm_clamped = min(p_norm, float(epsilon))
    residual_budget = math.sqrt(max(float(epsilon) * float(epsilon) - p_norm_clamped * p_norm_clamped, 0.0))

    if beta == 0.0:
        combo = p
        raw_amp = 0.0
    else:
        q = random_orthogonal_unit(d.size, basis, rng)
        raw_amp = float(beta) * residual_budget
        combo = p + raw_amp * q

    combo_norm = norm(combo)
    if combo_norm <= EPS:
        candidate = np.zeros_like(d)
    else:
        candidate = float(epsilon) * combo / combo_norm

    candidate_p = project_onto(candidate, basis)
    candidate_p_norm = norm(candidate_p)
    cosine = float(np.dot(unit(candidate), unit(d)))
    return candidate, {
        "endpoint_norm": d_norm,
        "projection_norm": p_norm,
        "projection_energy_fraction": p_energy,
        "residual_budget": residual_budget,
        "raw_orthogonal_amplitude": raw_amp,
        "candidate_projection_energy_fraction": float(
            (candidate_p_norm * candidate_p_norm) / max(float(epsilon) * float(epsilon), EPS)
        ),
        "candidate_cosine_to_endpoint": cosine,
    }


def add_threshold_rows(row: dict[str, Any], ratio: float, taus: list[float], rows: list[dict[str, Any]]) -> None:
    for tau in taus:
        rows.append({**row, "tau": float(tau), "ratio_to_lmax": ratio, "is_high_lmax": 1.0 if ratio >= tau else 0.0})


def summarize_threshold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["system"], row["endpoint_method"], row["k"], row["k_eff"], row["beta"], row["tau"])].append(row)

    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), int(kv[0][2]), float(kv[0][4]), float(kv[0][5]))):
        system, method, k, k_eff, beta, tau = key
        ratios = np.asarray([fnum(r["ratio_to_lmax"]) for r in items], dtype=np.float64)
        highs = np.asarray([fnum(r["is_high_lmax"]) for r in items], dtype=np.float64)
        proj = np.asarray([fnum(r["projection_energy_fraction"]) for r in items], dtype=np.float64)
        cand_proj = np.asarray([fnum(r["candidate_projection_energy_fraction"]) for r in items], dtype=np.float64)
        cosines = np.asarray([fnum(r["candidate_cosine_to_endpoint"]) for r in items], dtype=np.float64)
        ratios = ratios[np.isfinite(ratios)]
        highs = highs[np.isfinite(highs)]
        proj = proj[np.isfinite(proj)]
        cand_proj = cand_proj[np.isfinite(cand_proj)]
        cosines = cosines[np.isfinite(cosines)]
        out.append(
            {
                "system": system,
                "endpoint_method": method,
                "k": int(k),
                "k_eff": int(k_eff),
                "beta": float(beta),
                "tau": float(tau),
                "n": len(items),
                "p_high_lmax": float(np.mean(highs)) if highs.size else float("nan"),
                "ratio_to_lmax_mean": float(np.mean(ratios)) if ratios.size else float("nan"),
                "ratio_to_lmax_median": float(np.median(ratios)) if ratios.size else float("nan"),
                "ratio_to_lmax_min": float(np.min(ratios)) if ratios.size else float("nan"),
                "ratio_to_lmax_max": float(np.max(ratios)) if ratios.size else float("nan"),
                "endpoint_projection_energy_mean": float(np.mean(proj)) if proj.size else float("nan"),
                "candidate_projection_energy_mean": float(np.mean(cand_proj)) if cand_proj.size else float("nan"),
                "candidate_cosine_to_endpoint_mean": float(np.mean(cosines)) if cosines.size else float("nan"),
            }
        )
    return out


def summarize_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["system"], row["endpoint_method"], row["k"], row["k_eff"])].append(fnum(row["projection_energy_fraction"]))
    out: list[dict[str, Any]] = []
    for (system, method, k, k_eff), vals in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), int(kv[0][2]))):
        arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=np.float64)
        out.append(
            {
                "system": system,
                "endpoint_method": method,
                "k": int(k),
                "k_eff": int(k_eff),
                "n": len(vals),
                "projection_energy_mean": float(np.mean(arr)) if arr.size else float("nan"),
                "projection_energy_min": float(np.min(arr)) if arr.size else float("nan"),
                "projection_energy_max": float(np.max(arr)) if arr.size else float("nan"),
            }
        )
    return out


def write_svg_beta(summary: list[dict[str, Any]], path: Path, tau: float, k_target: int) -> None:
    rows = [r for r in summary if abs(float(r["tau"]) - tau) < 1e-9 and int(r["k"]) == k_target]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 540
    left, right, top, bottom = 85, 245, 55, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    betas = sorted({float(r["beta"]) for r in rows})
    xmax = max(betas) if betas else 1.0
    colors = {
        ("burgers1d", "steepest_add"): "#2563eb",
        ("burgers1d", "steepest_replace"): "#16a34a",
        ("ns2d", "steepest_add"): "#dc2626",
        ("ns2d", "steepest_replace"): "#9333ea",
    }

    def sx(x: float) -> float:
        return left + (x / xmax) * plot_w if xmax > 0 else left

    def sy(y: float) -> float:
        return top + (1.0 - y) * plot_h

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["system"]), str(row["endpoint_method"]))].append(row)

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(f'<text x="{left}" y="32" font-family="Arial, sans-serif" font-size="21" font-weight="700">Dominant-subspace iso-projection, k={k_target}, tau={tau:.2f}</text>')
    svg.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.3"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.3"/>')
    for y in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = sy(y)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left - 12}" y="{yy + 5:.2f}" font-family="Arial, sans-serif" font-size="13" text-anchor="end">{y:.2f}</text>')
    for beta in betas:
        xx = sx(beta)
        svg.append(f'<line x1="{xx:.2f}" y1="{top + plot_h}" x2="{xx:.2f}" y2="{top + plot_h + 6}" stroke="#111827"/>')
        svg.append(f'<text x="{xx:.2f}" y="{top + plot_h + 25}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">{beta:g}</text>')
    for key, items in sorted(groups.items()):
        pts = sorted(items, key=lambda r: float(r["beta"]))
        color = colors.get(key, "#111827")
        coords = " ".join(f'{sx(float(r["beta"])):.2f},{sy(float(r["p_high_lmax"])):.2f}' for r in pts)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
        for r in pts:
            svg.append(f'<circle cx="{sx(float(r["beta"])):.2f}" cy="{sy(float(r["p_high_lmax"])):.2f}" r="4" fill="{color}"/>')
    svg.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 22}" font-family="Arial, sans-serif" font-size="15" text-anchor="middle">orthogonal residual scale beta</text>')
    svg.append(f'<text x="23" y="{top + plot_h / 2:.2f}" font-family="Arial, sans-serif" font-size="15" text-anchor="middle" transform="rotate(-90 23 {top + plot_h / 2:.2f})">P(loss >= tau Lmax)</text>')
    lx, ly = left + plot_w + 32, top + 25
    svg.append(f'<text x="{lx}" y="{ly}" font-family="Arial, sans-serif" font-size="15" font-weight="700">Legend</text>')
    for i, key in enumerate(sorted(groups)):
        y = ly + 30 + 28 * i
        color = colors.get(key, "#111827")
        label = f"{key[0]} {key[1]}"
        svg.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 34}" y2="{y}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        svg.append(f'<text x="{lx + 46}" y="{y + 5}" font-family="Arial, sans-serif" font-size="13">{label}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def collect_burgers_vectors(ev: BurgersBoundaryEvaluator, methods: list[str]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    vectors: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for method in methods:
        if method not in ev.final_by_method:
            continue
        final = ev.final_by_method[method]
        for j, dataset_index in enumerate(ev.dataset_index.tolist()):
            vectors.append(flat(final[j]))
            metas.append(
                {
                    "system": "burgers1d",
                    "method": method,
                    "dataset_index": int(dataset_index),
                    "sample_position": int(ev.sample_position[j]),
                }
            )
    return vectors, metas


def collect_ns_vectors(ev: NSBoundaryEvaluator, methods: list[str]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    vectors: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for dataset_index in ev.datasets:
        for method in methods:
            vectors.append(flat(ev.final(dataset_index, method)))
            metas.append(
                {
                    "system": "ns2d",
                    "method": method,
                    "dataset_index": int(dataset_index),
                    "sample_position": int(ev.records[dataset_index][METHODS[0]]["sample_position"]),
                }
            )
    return vectors, metas


def run_burgers(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, list[dict[str, Any]]]:
    root = args.burgers_root if args.burgers_root.is_absolute() else PROJECT_ROOT / args.burgers_root
    ev = BurgersBoundaryEvaluator(root, args.burgers_num_samples, args.device)
    vectors, _metas = collect_burgers_vectors(ev, args.subspace_methods)
    basis_full, singular_values, spectrum_rows = endpoint_svd(vectors)
    for row in spectrum_rows:
        row["system"] = "burgers1d"
        row["num_endpoint_vectors"] = len(vectors)
        row["ambient_dim"] = int(basis_full.shape[0])

    result_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    endpoint_shape = next(iter(ev.final_by_method.values())).shape[1:]
    for method in args.endpoint_methods:
        if method not in ev.final_by_method:
            continue
        finals = ev.final_by_method[method]
        for k in args.ks:
            k_eff = min(int(k), basis_full.shape[1])
            basis = basis_full[:, :k_eff]
            for j, dataset_index in enumerate(ev.dataset_index.tolist()):
                d = flat(finals[j])
                p = project_onto(d, basis)
                p_frac = float((norm(p) ** 2) / max(norm(d) ** 2, EPS))
                projection_rows.append(
                    {
                        "system": "burgers1d",
                        "endpoint_method": method,
                        "dataset_index": int(dataset_index),
                        "sample_position": int(ev.sample_position[j]),
                        "k": int(k),
                        "k_eff": int(k_eff),
                        "projection_energy_fraction": p_frac,
                    }
                )
            for beta in args.betas:
                for draw in range(args.random_draws):
                    batch = []
                    metrics_by_j = []
                    for j in range(finals.shape[0]):
                        candidate, metrics = make_candidate(flat(finals[j]), basis, ev.epsilon, float(beta), rng)
                        batch.append(candidate.reshape(endpoint_shape))
                        metrics_by_j.append(metrics)
                    losses = ev.eval(np.stack(batch, axis=0))
                    for j, dataset_index in enumerate(ev.dataset_index.tolist()):
                        ratio = float(losses[j] / (ev.lmax[j] + EPS))
                        base = {
                            "system": "burgers1d",
                            "endpoint_method": method,
                            "dataset_index": int(dataset_index),
                            "sample_position": int(ev.sample_position[j]),
                            "k": int(k),
                            "k_eff": int(k_eff),
                            "beta": float(beta),
                            "draw_index": int(draw),
                            "loss": float(losses[j]),
                            "lmax": float(ev.lmax[j]),
                            **metrics_by_j[j],
                        }
                        add_threshold_rows(base, ratio, args.taus, result_rows)
    return {"rows": result_rows, "projection": projection_rows, "spectrum": spectrum_rows}


def run_ns2d(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, list[dict[str, Any]]]:
    root = args.ns_root if args.ns_root.is_absolute() else PROJECT_ROOT / args.ns_root
    ev = NSBoundaryEvaluator(root, args.ns_num_samples, args.device)
    vectors, _metas = collect_ns_vectors(ev, args.subspace_methods)
    basis_full, singular_values, spectrum_rows = endpoint_svd(vectors)
    for row in spectrum_rows:
        row["system"] = "ns2d"
        row["num_endpoint_vectors"] = len(vectors)
        row["ambient_dim"] = int(basis_full.shape[0])

    result_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for dataset_index in ev.datasets:
        sample_position = int(ev.records[dataset_index][METHODS[0]]["sample_position"])
        for method in args.endpoint_methods:
            final = ev.final(dataset_index, method)
            endpoint_shape = final.shape
            for k in args.ks:
                k_eff = min(int(k), basis_full.shape[1])
                basis = basis_full[:, :k_eff]
                d = flat(final)
                p = project_onto(d, basis)
                p_frac = float((norm(p) ** 2) / max(norm(d) ** 2, EPS))
                projection_rows.append(
                    {
                        "system": "ns2d",
                        "endpoint_method": method,
                        "dataset_index": int(dataset_index),
                        "sample_position": sample_position,
                        "k": int(k),
                        "k_eff": int(k_eff),
                        "projection_energy_fraction": p_frac,
                    }
                )
                deltas: list[np.ndarray] = []
                metas: list[dict[str, Any]] = []
                for beta in args.betas:
                    for draw in range(args.random_draws):
                        candidate, metrics = make_candidate(d, basis, ev.epsilon, float(beta), rng)
                        deltas.append(candidate.reshape(endpoint_shape))
                        metas.append(
                            {
                                "system": "ns2d",
                                "endpoint_method": method,
                                "dataset_index": int(dataset_index),
                                "sample_position": sample_position,
                                "k": int(k),
                                "k_eff": int(k_eff),
                                "beta": float(beta),
                                "draw_index": int(draw),
                                "lmax": float(ev.lmax[dataset_index]),
                                **metrics,
                            }
                        )
                losses = ev.eval_one(dataset_index, deltas)
                for loss, meta in zip(losses, metas):
                    ratio = float(loss / (float(meta["lmax"]) + EPS))
                    add_threshold_rows({**meta, "loss": float(loss)}, ratio, args.taus, result_rows)
    return {"rows": result_rows, "projection": projection_rows, "spectrum": spectrum_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=["burgers", "ns2d"], default=["burgers", "ns2d"])
    parser.add_argument("--burgers-root", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation/raw/burgers_eps8_alpha0p3_steps100_N20_trace")
    parser.add_argument("--ns-root", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/ns2d/ns2d_eps32_alpha10_steps100_N3_trace")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation/dominant_subspace_iso_projection_20260623")
    parser.add_argument("--burgers-num-samples", type=int, default=5)
    parser.add_argument("--ns-num-samples", type=int, default=2)
    parser.add_argument("--random-draws", type=int, default=16)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--taus", nargs="+", type=float, default=[0.90, 0.95, 0.99])
    parser.add_argument("--endpoint-methods", nargs="+", default=ENDPOINT_METHODS)
    parser.add_argument("--subspace-methods", nargs="+", default=METHODS)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []

    # Keep NS first to avoid Burgers/JAX initialization altering dtype behavior
    # before the NS recurrent solver bridge is loaded.
    if "ns2d" in args.systems:
        print("[dominant-iso] start ns2d", flush=True)
        out = run_ns2d(args, rng)
        rows.extend(out["rows"])
        projection_rows.extend(out["projection"])
        spectrum_rows.extend(out["spectrum"])
        print("[dominant-iso] done ns2d", flush=True)
    if "burgers" in args.systems:
        print("[dominant-iso] start burgers", flush=True)
        out = run_burgers(args, rng)
        rows.extend(out["rows"])
        projection_rows.extend(out["projection"])
        spectrum_rows.extend(out["spectrum"])
        print("[dominant-iso] done burgers", flush=True)

    summary = summarize_threshold_rows(rows)
    projection_summary = summarize_projection(projection_rows)

    tables = out_dir / "tables"
    write_csv(tables / "dominant_subspace_spectrum.csv", spectrum_rows)
    write_csv(tables / "endpoint_projection_energy.csv", projection_rows)
    write_csv(tables / "endpoint_projection_energy_summary.csv", projection_summary)
    write_csv(tables / "iso_projection_samples.csv", rows)
    write_csv(tables / "iso_projection_summary.csv", summary)

    figure_paths: list[str] = []
    selected_k = 4 if 4 in args.ks else int(args.ks[min(len(args.ks) - 1, 0)])
    svg_path = out_dir / "figures" / f"iso_projection_p095_by_beta_k{selected_k}.svg"
    write_svg_beta(summary, svg_path, tau=0.95, k_target=selected_k)
    if svg_path.exists():
        figure_paths.append(rel(svg_path))

    manifest = {
        "status": "completed",
        "runtime_seconds": time.time() - start,
        "out_dir": rel(out_dir),
        "systems": args.systems,
        "burgers_root": rel(args.burgers_root if args.burgers_root.is_absolute() else PROJECT_ROOT / args.burgers_root),
        "ns_root": rel(args.ns_root if args.ns_root.is_absolute() else PROJECT_ROOT / args.ns_root),
        "parameters": {
            "burgers_num_samples": args.burgers_num_samples,
            "ns_num_samples": args.ns_num_samples,
            "random_draws": args.random_draws,
            "ks": args.ks,
            "betas": args.betas,
            "taus": args.taus,
            "endpoint_methods": args.endpoint_methods,
            "subspace_methods": args.subspace_methods,
            "seed": args.seed,
            "device": args.device,
        },
        "row_counts": {
            "dominant_subspace_spectrum": len(spectrum_rows),
            "endpoint_projection_energy": len(projection_rows),
            "iso_projection_samples": len(rows),
            "iso_projection_summary": len(summary),
        },
        "figures": figure_paths,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
