#!/usr/bin/env python3
"""Directional curvature of scalar Loss3 along a saved PGD path.

This is Experiment D's cheap implementation: do not build the full Hessian.
Estimate tau^T H tau by centered finite differences of true Loss3 gradients:

    tau_k = (z_k - z_prev) / ||z_k - z_prev||
    tau^T H(z_k) tau ~= <grad L(z_k+h tau)-grad L(z_k-h tau), tau> / (2h)

Also records the centered scalar-loss second difference along the same tangent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
for p in (PROJECT_ROOT, TOOLS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_three_loss_objective_attack as attack  # noqa: E402
from run_loss3_finite_difference_linearity import (  # noqa: E402
    eval_loss3_residual,
    grad_loss3,
    load_config_args,
    markdown_table,
    vector_norm,
    write_csv,
    write_json,
)

DEFAULT_TRAJECTORY_ROOT = (
    PROJECT_ROOT
    / "forensics"
    / "loss3_simple_path_linearity_20260517"
    / "fno_nu0p001"
    / "loss3_original_pgd_trajectories_steps100"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "forensics"
    / "loss3_simple_path_linearity_20260517"
    / "fno_nu0p001"
    / "experiment4_directional_curvature_steps100_pilot"
)
DEFAULT_DOC = PROJECT_ROOT / "docs" / "loss3_directional_curvature_fno_nu0p001_steps100_pilot_20260517.md"


def aggregate_by_k(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row["k"]), []).append(row)
    metrics = [
        "delta_budget_ratio",
        "center_loss",
        "tangent_curvature_signed",
        "tangent_curvature_abs",
        "loss_second_diff_signed",
        "loss_second_diff_abs",
        "fd_grad_loss_seconddiff_relative_gap",
        "grad_plus_minus_angle_deg",
    ]
    out: list[dict[str, Any]] = []
    for k, items in sorted(groups.items()):
        row: dict[str, Any] = {"k": k, "n_points": len(items)}
        for metric in metrics:
            arr = np.asarray([float(item.get(metric, float("nan"))) for item in items], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                row[f"{metric}_mean"] = float(arr.mean())
                row[f"{metric}_std"] = float(arr.std(ddof=0))
                row[f"{metric}_median"] = float(np.median(arr))
            else:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_median"] = float("nan")
        out.append(row)
    return out


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom <= 0.0:
        return float("nan")
    c = float(np.dot(af, bf) / denom)
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))


def write_figures(out_dir: Path, rows_by_k: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    k = [float(r["k"]) for r in rows_by_k]
    budget = [float(r["delta_budget_ratio_mean"]) for r in rows_by_k]
    curv = [float(r["tangent_curvature_abs_mean"]) for r in rows_by_k]
    curv_std = [float(r["tangent_curvature_abs_std"]) for r in rows_by_k]
    loss_curv = [float(r["loss_second_diff_abs_mean"]) for r in rows_by_k]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.errorbar(k, curv, yerr=curv_std, marker="o", capsize=3, linewidth=1.8, label="|<g+ - g-, tau>| / 2h")
    ax.plot(k, loss_curv, marker="s", linewidth=1.3, label="|L+ - 2L0 + L-| / h^2")
    ax.set_xlabel("PGD step k")
    ax.set_ylabel("directional curvature along path tangent")
    ax.set_title("Loss3 directional curvature along PGD path")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = fig_dir / "directional_curvature_vs_k.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.errorbar(budget, curv, yerr=curv_std, marker="o", capsize=3, linewidth=1.8)
    ax.set_xlabel("mean budget ratio ||delta_k||_2 / epsilon")
    ax.set_ylabel("mean |tau^T H tau| finite-diff estimate")
    ax.set_title("Loss3 directional curvature vs radius")
    ax.grid(True, alpha=0.25)
    path = fig_dir / "directional_curvature_vs_radius.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))
    return files


def write_doc(path: Path, out_dir: Path, rows_by_k: list[dict[str, Any]], figure_files: list[str], h: float, indices: list[int]) -> None:
    cols = [
        "k",
        "n_points",
        "delta_budget_ratio_mean",
        "center_loss_mean",
        "tangent_curvature_abs_mean",
        "tangent_curvature_abs_std",
        "loss_second_diff_abs_mean",
        "fd_grad_loss_seconddiff_relative_gap_mean",
    ]
    trend = " -> ".join(f"k={int(r['k'])}: {float(r['tangent_curvature_abs_mean']):.4g}" for r in rows_by_k)
    fig_rel = [str(Path(f).relative_to(PROJECT_ROOT)) for f in figure_files]
    lines = [
        "# Loss3 Directional Curvature Along Path Pilot",
        "",
        "Date: 2026-05-17 UTC",
        "",
        "Experiment name:",
        "",
        r"\[",
        r"\textbf{Experiment D: Hessian / Curvature Along Path}",
        r"\]",
        "",
        "This pilot uses the cheap directional version of the Hessian experiment. It does not form the full Hessian and does not estimate the top eigenvalue.",
        "",
        "## Definition",
        "",
        r"At each saved PGD point \(z_k=x_0+\delta_k\), define the local path tangent",
        "",
        r"\[",
        r"\tau_k=\frac{z_k-z_{k-5}}{\|z_k-z_{k-5}\|_2}.",
        r"\]",
        "",
        "The target scalar loss is",
        "",
        r"\[",
        r"L_3(z)=\|f(z)-j(z)\|_2.",
        r"\]",
        "",
        "The intended curvature is",
        "",
        r"\[",
        r"\kappa_k=|\tau_k^\top \nabla_z^2 L_3(z_k)\tau_k|.",
        r"\]",
        "",
        "Because exact second-order differentiation through the JAX-to-Torch solver bridge needs extra validation, this pilot estimates it by centered gradient differences:",
        "",
        r"\[",
        r"\tau_k^\top H_k\tau_k \approx \frac{\langle \nabla L_3(z_k+h\tau_k)-\nabla L_3(z_k-h\tau_k),\tau_k\rangle}{2h}.",
        r"\]",
        "",
        f"This run uses `h={h}` and samples `{indices}`.",
        "",
        "The scalar-loss second difference is also recorded as a consistency check:",
        "",
        r"\[",
        r"\frac{L_3(z_k+h\tau_k)-2L_3(z_k)+L_3(z_k-h\tau_k)}{h^2}.",
        r"\]",
        "",
        "## Aggregate By k",
        "",
        markdown_table(rows_by_k, cols),
        "",
        "## Trend",
        "",
        f"Mean finite-difference tangent curvature trend: `{trend}`.",
        "",
        "This pilot is meant to answer feasibility first. If the trend is useful, it can be expanded to more samples and optional top-eigenvalue estimation.",
        "",
        "## Output Files",
        "",
        f"- output directory: `{out_dir.relative_to(PROJECT_ROOT)}`",
        "- `directional_curvature_by_sample_k.csv`",
        "- `directional_curvature_aggregate_by_k.csv`",
        "- `manifest.json`",
        "",
        "## Figures",
        "",
    ]
    for fig in fig_rel:
        lines.append(f"- `{fig}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 40, 115])
    parser.add_argument("--sample-ks", type=int, nargs="+", default=list(range(5, 101, 5)))
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    source_paths: list[str] = []
    for index in args.indices:
        run_dir = args.trajectory_root / f"index_{index:03d}" / "loss3_original_pgd"
        config_path = run_dir / "config.json"
        traj_path = run_dir / "trajectory.npz"
        problem_args = load_config_args(config_path, device=args.device)
        problem_args.index = index
        problem = attack.AttackProblem(problem_args)
        traj = np.load(traj_path)
        ks = traj["k"].astype(int)
        x_adv = traj["x_adv"].astype(np.float64)
        delta = traj["delta"].astype(np.float64)
        available = {int(k): pos for pos, k in enumerate(ks)}
        source_paths.extend([str(config_path), str(traj_path)])
        print(f"[index] {index}", flush=True)
        for k in args.sample_ks:
            if k not in available:
                continue
            prev_candidates = [kk for kk in ks.tolist() if int(kk) < int(k)]
            if not prev_candidates:
                continue
            prev_k = int(prev_candidates[-1])
            pos = available[int(k)]
            prev_pos = available[prev_k]
            z = x_adv[pos]
            step = x_adv[pos] - x_adv[prev_pos]
            step_norm = vector_norm(step)
            if step_norm <= 0.0:
                continue
            tau = step / step_norm
            h = float(args.h)
            lp, gp = grad_loss3(problem, z + h * tau)
            lm, gm = grad_loss3(problem, z - h * tau)
            l0, _ = eval_loss3_residual(problem, z)
            signed = float(np.dot((gp - gm).reshape(-1), tau.reshape(-1)) / (2.0 * h))
            loss_second = float((lp - 2.0 * l0 + lm) / (h * h))
            gap = abs(signed - loss_second) / (abs(loss_second) + 1e-12)
            row = {
                "sample_index": int(index),
                "k": int(k),
                "prev_k": int(prev_k),
                "h": h,
                "delta_norm_l2": vector_norm(delta[pos]),
                "delta_budget_ratio": vector_norm(delta[pos]) / float(problem_args.epsilon),
                "step_norm_l2": step_norm,
                "center_loss": float(l0),
                "loss_plus": float(lp),
                "loss_minus": float(lm),
                "tangent_curvature_signed": signed,
                "tangent_curvature_abs": abs(signed),
                "loss_second_diff_signed": loss_second,
                "loss_second_diff_abs": abs(loss_second),
                "fd_grad_loss_seconddiff_relative_gap": gap,
                "grad_plus_norm_l2": vector_norm(gp),
                "grad_minus_norm_l2": vector_norm(gm),
                "grad_plus_minus_angle_deg": angle_deg(gp, gm),
                "trajectory_path": str(traj_path),
            }
            rows.append(row)
            print(f"[curv] index={index} k={k} budget={row['delta_budget_ratio']:.4f} kappa={abs(signed):.4g} loss2={abs(loss_second):.4g} gap={gap:.3g}", flush=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    by_k = aggregate_by_k(rows)
    write_csv(out_dir / "directional_curvature_by_sample_k.csv", rows)
    write_csv(out_dir / "directional_curvature_aggregate_by_k.csv", by_k)
    figures: list[str] = []
    if not args.no_plots:
        figures = write_figures(out_dir, by_k)
    manifest = {
        "experiment": "Experiment D: Hessian / Curvature Along Path - directional finite-difference pilot",
        "trajectory_root": str(args.trajectory_root),
        "out_dir": str(out_dir),
        "doc_path": str(args.doc_path),
        "indices": args.indices,
        "sample_ks": args.sample_ks,
        "h": args.h,
        "device": args.device,
        "source_paths": sorted(set(source_paths)),
        "output_files": [
            str(out_dir / "directional_curvature_by_sample_k.csv"),
            str(out_dir / "directional_curvature_aggregate_by_k.csv"),
            str(out_dir / "manifest.json"),
            str(args.doc_path),
            *figures,
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    write_doc(args.doc_path, out_dir, by_k, figures, float(args.h), list(args.indices))
    print(f"[done] wrote {out_dir}", flush=True)
    print(f"[done] wrote {args.doc_path}", flush=True)


if __name__ == "__main__":
    main()
