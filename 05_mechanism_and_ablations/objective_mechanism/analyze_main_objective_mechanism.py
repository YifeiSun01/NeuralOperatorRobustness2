#!/usr/bin/env python3
"""Post-hoc mechanism diagnostics for the main objective comparison.

This script reuses saved final deltas from a batch three-loss run and computes
the decomposition requested in docs/loss3_original_theory_experiment_plan.md:

  Delta f = f(x + delta) - f(x)
  Delta j = j(x + delta) - j(x)
  Delta e = Delta f - Delta j

For every final and boundary-rescaled delta it records norms, cosine between
Delta f and Delta j, and tracking-discount ratios.
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

from tools.run_batch_three_loss_loss_only import (  # noqa: E402
    LOSSES,
    METHODS,
    VARIANTS,
    BatchProblem,
    batch_norm,
    norm_name,
    parse_norm,
)


EPS = 1e-12


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_summary(values: Any) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    count = int(np.count_nonzero(finite))
    if count == 0:
        return {"mean": math.nan, "std": math.nan, "median": math.nan, "min": math.nan, "max": math.nan, "finite_count": 0}
    vals = arr[finite]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "finite_count": count,
    }


def config_to_args(config: dict[str, Any], *, device: str | None) -> SimpleNamespace:
    cfg = dict(config)
    if device is not None:
        cfg["device"] = device
    for key in ("out_root", "burgers_test_path", "burgers_torch_checkpoint", "deeponet_checkpoint", "deeponet_output_transform_stats"):
        if key in cfg and cfg[key] is not None:
            path = Path(cfg[key])
            cfg[key] = path if path.is_absolute() else PROJECT_ROOT / path
    cfg["p_order"] = parse_norm(str(cfg.get("p", cfg.get("p_order", "2"))))
    cfg["q_order"] = parse_norm(str(cfg.get("q", cfg.get("q_order", "2"))))
    cfg.setdefault("burgers_nx", 1024)
    cfg.setdefault("burgers_nu", 0.001)
    cfg.setdefault("burgers_t_final", 1.0)
    cfg.setdefault("burgers_dt", 0.001)
    cfg.setdefault("burgers_domain", 2.0)
    cfg.setdefault("burgers_jax_solver_dtype", "float64")
    cfg.setdefault("model_label", cfg.get("model_kind", "model"))
    return SimpleNamespace(**cfg)


def torch_to_np(x) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float64)


def compute_metrics_for_delta(problem: BatchProblem, delta_np: np.ndarray, *, eta: float) -> dict[str, np.ndarray]:
    import torch

    delta = torch.as_tensor(delta_np, device=problem.device, dtype=problem.x0.dtype)
    x_adv = problem.x0 + delta
    with torch.no_grad():
        f_adv = problem.model_forward(x_adv)
        j_adv = problem.solver_forward(x_adv, allow_grad=False)
        delta_f = f_adv - problem.f0
        delta_j = j_adv - problem.g0
        delta_e = delta_f - delta_j
        e0 = problem.f0 - problem.g0
        e_adv = f_adv - j_adv

        q = problem.args.q_order
        delta_norm = batch_norm(delta, problem.args.p_order)
        df_norm = batch_norm(delta_f, q)
        dj_norm = batch_norm(delta_j, q)
        de_norm = batch_norm(delta_e, q)
        e0_norm = batch_norm(e0, q)
        e_adv_norm = batch_norm(e_adv, q)

        df_flat = delta_f.reshape(delta_f.shape[0], -1)
        dj_flat = delta_j.reshape(delta_j.shape[0], -1)
        dot = (df_flat * dj_flat).sum(dim=1)
        df_l2 = torch.linalg.vector_norm(df_flat, ord=2, dim=1)
        dj_l2 = torch.linalg.vector_norm(dj_flat, ord=2, dim=1)
        cosine = dot / (df_l2 * dj_l2 + EPS)

        tracking_discount_f = de_norm / (df_norm + eta)
        tracking_discount_sym = de_norm / (df_norm + dj_norm + eta)
        error_growth = e_adv_norm - e0_norm

    return {
        "delta_pnorm": torch_to_np(delta_norm),
        "delta_f_norm": torch_to_np(df_norm),
        "delta_j_norm": torch_to_np(dj_norm),
        "delta_mismatch_norm": torch_to_np(de_norm),
        "clean_error_norm": torch_to_np(e0_norm),
        "adv_error_norm": torch_to_np(e_adv_norm),
        "error_norm_growth": torch_to_np(error_growth),
        "cos_delta_f_delta_j": torch_to_np(cosine),
        "tracking_discount_f": torch_to_np(tracking_discount_f),
        "tracking_discount_sym": torch_to_np(tracking_discount_sym),
        "delta_f_over_delta": torch_to_np(df_norm / (delta_norm + eta)),
        "delta_j_over_delta": torch_to_np(dj_norm / (delta_norm + eta)),
        "mismatch_over_delta": torch_to_np(de_norm / (delta_norm + eta)),
    }


def tag_parts(tag: str) -> tuple[str, str, str]:
    for loss in LOSSES:
        prefix = f"{loss}_"
        if not tag.startswith(prefix):
            continue
        rest = tag[len(prefix) :]
        for variant in VARIANTS:
            vprefix = f"{variant}_"
            if rest.startswith(vprefix):
                return loss, variant, rest[len(vprefix) :]
    raise ValueError(f"Cannot parse tag name: {tag}")


def analyze_root(root: Path, *, device: str | None, out_subdir: str) -> Path:
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = config_to_args(config, device=device)
    problem = BatchProblem(args)
    out_dir = root / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}

    tag_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "final_delta.npz").exists())
    if not tag_dirs:
        raise RuntimeError(f"No final_delta.npz files found under {root}")

    for tag_dir in tag_dirs:
        tag = tag_dir.name
        optimized_loss, variant, method = tag_parts(tag)
        payload = np.load(tag_dir / "final_delta.npz")
        sample_position = payload["sample_position"].astype(np.int64)
        dataset_index = payload["dataset_index"].astype(np.int64)

        for delta_kind in ("final", "boundary"):
            key = "final_delta" if delta_kind == "final" else "boundary_delta"
            if key not in payload:
                continue
            metrics = compute_metrics_for_delta(problem, payload[key], eta=float(args.eta))
            arrays[f"{tag}/{delta_kind}"] = metrics
            n = len(sample_position)
            for i in range(n):
                row: dict[str, Any] = {
                    "root": root.name,
                    "model_kind": args.model_kind,
                    "model_label": args.model_label,
                    "optimized_loss": optimized_loss,
                    "objective_variant": variant,
                    "attack_method": method,
                    "tag": tag,
                    "delta_kind": delta_kind,
                    "sample_position": int(sample_position[i]),
                    "dataset_index": int(dataset_index[i]),
                    "epsilon": float(args.epsilon),
                    "alpha": float(args.alpha),
                    "p": norm_name(args.p_order),
                    "q": norm_name(args.q_order),
                }
                for metric_name, values in metrics.items():
                    row[metric_name] = float(values[i])
                per_sample_rows.append(row)

            srow: dict[str, Any] = {
                "root": root.name,
                "model_kind": args.model_kind,
                "model_label": args.model_label,
                "optimized_loss": optimized_loss,
                "objective_variant": variant,
                "attack_method": method,
                "tag": tag,
                "delta_kind": delta_kind,
                "epsilon": float(args.epsilon),
                "alpha": float(args.alpha),
                "p": norm_name(args.p_order),
                "q": norm_name(args.q_order),
            }
            for metric_name, values in metrics.items():
                for stat_name, stat_value in finite_summary(values).items():
                    srow[f"{metric_name}_{stat_name}"] = stat_value
            summary_rows.append(srow)

    write_csv(out_dir / "mechanism_per_sample.csv", per_sample_rows)
    write_csv(out_dir / "mechanism_summary.csv", summary_rows)
    save_plots(out_dir, summary_rows, per_sample_rows, root.name)
    return out_dir


def save_plots(out_dir: Path, summary_rows: list[dict[str, Any]], per_sample_rows: list[dict[str, Any]], title_suffix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    focused = [
        row
        for row in summary_rows
        if row["delta_kind"] == "final" and row["objective_variant"] == "original" and row["attack_method"] in {"pgd", "lp_steepest_pgd", "generalized_power"}
    ]
    focused.sort(key=lambda r: (r["optimized_loss"], r["attack_method"]))
    labels = [f"{r['optimized_loss']}\n{short_method(r['attack_method'])}" for r in focused]

    metrics = [
        ("delta_f_norm_mean", r"$\|\Delta f\|$"),
        ("delta_j_norm_mean", r"$\|\Delta j\|$"),
        ("delta_mismatch_norm_mean", r"$\|\Delta f-\Delta j\|$"),
        ("adv_error_norm_mean", r"$\|e(x+\delta)\|$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (field, ylabel) in zip(axes.ravel(), metrics):
        values = [float(r[field]) for r in focused]
        ax.bar(range(len(values)), values, color=[loss_color(r["optimized_loss"]) for r in focused])
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Main objective mechanism metrics: original objectives\n{title_suffix}")
    fig.savefig(plot_dir / "original_objective_mechanism_bars.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, (field, ylabel) in zip(
        axes,
        [
            ("cos_delta_f_delta_j_mean", r"$\cos(\Delta f,\Delta j)$"),
            ("tracking_discount_f_mean", r"$D_f$"),
            ("tracking_discount_sym_mean", r"$D_{sym}$"),
        ],
    ):
        values = [float(r[field]) for r in focused]
        ax.bar(range(len(values)), values, color=[loss_color(r["optimized_loss"]) for r in focused])
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Tracking and alignment: original objectives\n{title_suffix}")
    fig.savefig(plot_dir / "original_objective_tracking_alignment.png", dpi=180)
    plt.close(fig)

    final_rows = [row for row in per_sample_rows if row["delta_kind"] == "final" and row["objective_variant"] == "original"]
    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    for loss in LOSSES:
        xs = [float(r["delta_f_norm"]) for r in final_rows if r["optimized_loss"] == loss]
        ys = [float(r["delta_mismatch_norm"]) for r in final_rows if r["optimized_loss"] == loss]
        ax.scatter(xs, ys, s=12, alpha=0.55, label=loss, color=loss_color(loss))
    ax.set_xlabel(r"$\|\Delta f\|$")
    ax.set_ylabel(r"$\|\Delta f-\Delta j\|$")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title(f"Model movement vs oracle mismatch\n{title_suffix}")
    fig.savefig(plot_dir / "delta_f_vs_mismatch_scatter.png", dpi=180)
    plt.close(fig)

    heat_rows = [row for row in summary_rows if row["delta_kind"] == "final"]
    heat_rows.sort(key=lambda r: (r["optimized_loss"], r["objective_variant"], r["attack_method"]))
    labels = [f"{r['optimized_loss']}_{r['objective_variant']}\n{short_method(r['attack_method'])}" for r in heat_rows]
    heat_fields = [
        "delta_f_norm_mean",
        "delta_j_norm_mean",
        "delta_mismatch_norm_mean",
        "adv_error_norm_mean",
        "cos_delta_f_delta_j_mean",
        "tracking_discount_f_mean",
        "tracking_discount_sym_mean",
    ]
    matrix = np.asarray([[float(r[f]) for f in heat_fields] for r in heat_rows], dtype=np.float64)
    scaled = matrix.copy()
    for j in range(scaled.shape[1]):
        col = scaled[:, j]
        finite = np.isfinite(col)
        if np.count_nonzero(finite) > 1:
            lo, hi = np.nanmin(col[finite]), np.nanmax(col[finite])
            if hi > lo:
                scaled[:, j] = (col - lo) / (hi - lo)
    fig, ax = plt.subplots(figsize=(10, max(8, 0.28 * len(labels))), constrained_layout=True)
    im = ax.imshow(scaled, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(len(heat_fields)))
    ax.set_xticklabels([f.replace("_mean", "") for f in heat_fields], rotation=35, ha="right", fontsize=8)
    ax.set_title(f"All final-delta mechanism metrics, column-normalized\n{title_suffix}")
    fig.colorbar(im, ax=ax, label="column-normalized value")
    fig.savefig(plot_dir / "all_tags_mechanism_heatmap.png", dpi=180)
    plt.close(fig)


def short_method(method: str) -> str:
    return {"pgd": "PGD", "lp_steepest_pgd": "LP", "generalized_power": "GPI"}.get(method, method)


def loss_color(loss: str) -> str:
    return {"loss1": "#2563eb", "loss2": "#d97706", "loss3": "#059669"}.get(loss, "#6b7280")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-subdir", default="mechanism_diagnostics")
    args = parser.parse_args()

    for root in args.roots:
        root = root if root.is_absolute() else PROJECT_ROOT / root
        out_dir = analyze_root(root, device=args.device, out_subdir=args.out_subdir)
        print(f"[done] {root} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
