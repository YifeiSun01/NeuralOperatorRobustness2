#!/usr/bin/env python3
"""Analyze pairwise similarity between final perturbations from optimizer methods.

This is a post-processing script: it reads saved final_deltas.npz files from
completed loss3 direction/proposal ablation runs and writes CSV/PNG summaries.
It does not run the model, solver, or optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


CORE_METHODS = ["raw_add", "steepest_add", "steepest_replace"]


LABELS = {
    "raw_add": "PGD raw add",
    "unit_raw_add": "unit raw add",
    "raw_replace": "unit raw replace",
    "steepest_add": "Lp steepest add",
    "steepest_replace": "Lp steepest replace",
    "power_add__objective_gradient": "power obj-grad add",
    "power_replace__objective_gradient": "power obj-grad replace",
    "power_add__pure_jvp_vjp": "power JVP/VJP add",
    "power_replace__pure_jvp_vjp": "power JVP/VJP replace",
    "power_add__affine_jvp_vjp": "power affine add",
    "power_replace__affine_jvp_vjp": "power affine replace",
    "power_replace__generalized_pq": "power generalized P/Q replace",
}


def run_label(root: Path) -> str:
    name = root.name
    p = "?"
    q = "?"
    for part in name.split("_"):
        if part.startswith("p") and part not in {"p"}:
            p = part[1:]
        if part.startswith("q") and part not in {"q"}:
            q = part[1:]
    return f"p={p}, q={q}"


def read_completed(root: Path) -> bool:
    manifest = root / "manifest.json"
    if not manifest.exists():
        return False
    try:
        return json.loads(manifest.read_text()).get("status") == "completed"
    except Exception:
        return False


def finite_stats(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "finite_count": 0, "nonfinite_count": int(arr.size)}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "finite_count": int(finite.size),
        "nonfinite_count": int(arr.size - finite.size),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def matrix_to_rows(methods: list[str], matrix: np.ndarray, key: str) -> list[dict]:
    rows = []
    for i, method in enumerate(methods):
        row = {"method": method}
        for j, other in enumerate(methods):
            row[other] = float(matrix[i, j]) if np.isfinite(matrix[i, j]) else "nan"
        rows.append(row)
    return rows


def save_heatmap(path: Path, methods: list[str], matrix: np.ndarray, title: str, cbar_label: str, vmin=None, vmax=None, cmap="viridis") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(methods)
    labels = [LABELS.get(m, m) for m in methods]
    fig_w = max(7.0, 0.55 * n + 3.5)
    fig_h = max(6.0, 0.50 * n + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if np.isfinite(val):
                color = "white" if (vmin is not None and vmax is not None and val > (vmin + vmax) / 2) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_root(root: Path, tag: str) -> dict:
    data_path = root / "final_deltas.npz"
    z = np.load(data_path, allow_pickle=True)
    methods = [str(x) for x in z["method"].tolist()]
    dataset_index = z["dataset_index"].astype(int)
    delta = z["final_delta"].astype(np.float64)
    flat = delta.reshape(delta.shape[0], delta.shape[1], -1)
    norms = np.linalg.norm(flat, axis=2)

    m = len(methods)
    n = len(dataset_index)
    cosine = np.full((m, m, n), np.nan, dtype=np.float64)
    abs_cosine = np.full_like(cosine, np.nan)
    rel_l2 = np.full_like(cosine, np.nan)
    l2_diff = np.full_like(cosine, np.nan)

    long_rows = []
    summary_rows = []
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            dots = np.sum(flat[i] * flat[j], axis=1)
            denom = norms[i] * norms[j]
            valid = denom > 1e-30
            cos = np.full(n, np.nan, dtype=np.float64)
            cos[valid] = dots[valid] / denom[valid]
            cos = np.clip(cos, -1.0, 1.0)
            diff = np.linalg.norm(flat[i] - flat[j], axis=1)
            scale = 0.5 * (norms[i] + norms[j])
            rel = np.full(n, np.nan, dtype=np.float64)
            rel[scale > 1e-30] = diff[scale > 1e-30] / scale[scale > 1e-30]
            cosine[i, j] = cos
            abs_cosine[i, j] = np.abs(cos)
            l2_diff[i, j] = diff
            rel_l2[i, j] = rel
            if i < j:
                cs = finite_stats(cos)
                acs = finite_stats(np.abs(cos))
                ds = finite_stats(diff)
                rs = finite_stats(rel)
                ns_i = finite_stats(norms[i])
                ns_j = finite_stats(norms[j])
                summary_rows.append({
                    "method_a": mi,
                    "method_b": mj,
                    "sample_count": n,
                    "cosine_mean": cs["mean"],
                    "cosine_std": cs["std"],
                    "cosine_min": cs["min"],
                    "cosine_max": cs["max"],
                    "abs_cosine_mean": acs["mean"],
                    "relative_l2_distance_mean": rs["mean"],
                    "relative_l2_distance_std": rs["std"],
                    "l2_distance_mean": ds["mean"],
                    "l2_distance_std": ds["std"],
                    "norm_a_mean": ns_i["mean"],
                    "norm_b_mean": ns_j["mean"],
                    "finite_count": cs["finite_count"],
                    "nonfinite_count": cs["nonfinite_count"],
                })
                for s_idx, ds_idx in enumerate(dataset_index):
                    long_rows.append({
                        "dataset_index": int(ds_idx),
                        "method_a": mi,
                        "method_b": mj,
                        "cosine": float(cos[s_idx]) if np.isfinite(cos[s_idx]) else "nan",
                        "abs_cosine": float(abs(cos[s_idx])) if np.isfinite(cos[s_idx]) else "nan",
                        "l2_distance": float(diff[s_idx]) if np.isfinite(diff[s_idx]) else "nan",
                        "relative_l2_distance": float(rel[s_idx]) if np.isfinite(rel[s_idx]) else "nan",
                        "norm_a": float(norms[i, s_idx]),
                        "norm_b": float(norms[j, s_idx]),
                    })

    out_dir = root / "final_delta_similarity" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "final_delta_pairwise_similarity_per_sample.csv", long_rows)
    write_csv(out_dir / "final_delta_pairwise_similarity_summary.csv", summary_rows)

    mean_cos = np.nanmean(cosine, axis=2)
    std_cos = np.nanstd(cosine, axis=2)
    mean_abs_cos = np.nanmean(abs_cosine, axis=2)
    mean_rel_l2 = np.nanmean(rel_l2, axis=2)
    mean_l2 = np.nanmean(l2_diff, axis=2)

    write_csv(out_dir / "final_delta_cosine_mean_matrix.csv", matrix_to_rows(methods, mean_cos, "cosine_mean"))
    write_csv(out_dir / "final_delta_cosine_std_matrix.csv", matrix_to_rows(methods, std_cos, "cosine_std"))
    write_csv(out_dir / "final_delta_abs_cosine_mean_matrix.csv", matrix_to_rows(methods, mean_abs_cos, "abs_cosine_mean"))
    write_csv(out_dir / "final_delta_relative_l2_mean_matrix.csv", matrix_to_rows(methods, mean_rel_l2, "relative_l2_distance_mean"))
    write_csv(out_dir / "final_delta_l2_distance_mean_matrix.csv", matrix_to_rows(methods, mean_l2, "l2_distance_mean"))

    fig_dir = out_dir / "figures"
    save_heatmap(fig_dir / "final_delta_cosine_mean_heatmap_all_methods.png", methods, mean_cos, f"Final delta cosine mean, {run_label(root)}", "mean cosine", vmin=-1, vmax=1, cmap="coolwarm")
    save_heatmap(fig_dir / "final_delta_relative_l2_mean_heatmap_all_methods.png", methods, mean_rel_l2, f"Final delta relative L2 distance mean, {run_label(root)}", "mean relative L2", vmin=0, vmax=float(np.nanmax(mean_rel_l2)) if np.isfinite(mean_rel_l2).any() else 1, cmap="magma_r")

    core_indices = [methods.index(x) for x in CORE_METHODS if x in methods]
    if len(core_indices) >= 2:
        core_methods = [methods[i] for i in core_indices]
        save_heatmap(fig_dir / "final_delta_cosine_mean_heatmap_core_methods.png", core_methods, mean_cos[np.ix_(core_indices, core_indices)], f"Core final delta cosine mean, {run_label(root)}", "mean cosine", vmin=-1, vmax=1, cmap="coolwarm")
        save_heatmap(fig_dir / "final_delta_relative_l2_mean_heatmap_core_methods.png", core_methods, mean_rel_l2[np.ix_(core_indices, core_indices)], f"Core final delta relative L2 distance, {run_label(root)}", "mean relative L2", vmin=0, vmax=float(np.nanmax(mean_rel_l2[np.ix_(core_indices, core_indices)])) if np.isfinite(mean_rel_l2[np.ix_(core_indices, core_indices)]).any() else 1, cmap="magma_r")

    sorted_pairs = sorted(summary_rows, key=lambda r: (-float(r["cosine_mean"]), float(r["relative_l2_distance_mean"])))
    return {
        "root": str(root),
        "label": run_label(root),
        "out_dir": str(out_dir),
        "method_count": m,
        "sample_count": n,
        "top_pairs": sorted_pairs[:10],
        "bottom_pairs": sorted(summary_rows, key=lambda r: float(r["cosine_mean"]))[:5],
    }


def discover_roots(base: Path) -> list[Path]:
    return sorted(p.parent for p in base.glob("fno_nu0p001_eps8_alpha0p3_batch100_steps100_p*_q*/manifest.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("forensics/loss3_optimizer_direction_proposal_ablation_20260517"))
    parser.add_argument("--roots", type=Path, nargs="*", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%Sutc")
    roots = args.roots if args.roots else discover_roots(args.base)
    reports = []
    for root in roots:
        if not read_completed(root):
            print(f"[skip] {root}: manifest is not completed")
            continue
        if not (root / "final_deltas.npz").exists():
            print(f"[skip] {root}: missing final_deltas.npz")
            continue
        report = analyze_root(root, tag)
        reports.append(report)
        print(f"[ok] {report['label']} -> {report['out_dir']}")
        for pair in report["top_pairs"][:5]:
            print(f"  top {pair['method_a']} vs {pair['method_b']}: cosine={float(pair['cosine_mean']):.6f}, rel_l2={float(pair['relative_l2_distance_mean']):.6f}")

    if reports:
        summary_path = args.base / f"final_delta_similarity_summary_{tag}.json"
        summary_path.write_text(json.dumps(reports, indent=2))
        print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
