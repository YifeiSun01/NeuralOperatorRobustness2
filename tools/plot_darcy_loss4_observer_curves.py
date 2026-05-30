#!/usr/bin/env python3
"""Plot Darcy physics loss4 as an observer metric during loss1/loss2/loss3/loss4 attacks."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DARCY = ROOT / "2D_Darcy_FNO2d"
if str(DARCY) not in sys.path:
    sys.path.insert(0, str(DARCY))

from perturbation_methods.attack_darcy_binary_physics_loss4 import per_sample_physics_components

OLD_LOSS12_BUDGET = DARCY / "perturbation_results/binary_loss12_steepest_replace_budget_completion/darcy_loss12_steepest_replace_budget_completion_nx211_N50_steps100_20260528/runs"
OLD_LOSS3_BUDGET = DARCY / "perturbation_results/binary_loss3_steepest_replace_budget_sweep/darcy_loss3_steepest_replace_budget_sweep_nx211_N50_steps100_20260528/runs"
LOSS4_BUDGET = DARCY / "perturbation_results/binary_loss4_physics_budget_sweep"
OUT = ROOT / "analysis_outputs/darcy_flow_figures_clean_bundle_with_loss4_20260529/05_loss4_observer_curves"
LOSSES = ["loss1", "loss2", "loss3", "loss4"]
COLORS = {"loss1": "#1f77b4", "loss2": "#ff7f0e", "loss3": "#2ca02c", "loss4": "#d62728"}


def first(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def trace_path(loss: str, k: int, method: str) -> Path:
    if method != "steepest_replace":
        raise ValueError("budget traces are currently available for steepest_replace")
    if loss in {"loss1", "loss2"}:
        p = first(sorted(OLD_LOSS12_BUDGET.glob(f"loss12_steepest_replace_nx211_N50_steps100_K{k}_*/loss_method_grid/{loss}/steepest_replace/step_sample_trace.npz")))
    elif loss == "loss3":
        p = first(sorted(OLD_LOSS3_BUDGET.glob(f"loss3_steepest_replace_nx211_N50_steps100_K{k}_*/loss_method_grid/loss3/steepest_replace/step_sample_trace.npz")))
    elif loss == "loss4":
        p = first(sorted(LOSS4_BUDGET.glob(f"darcy_loss4_physics_steepest_replace_nx211_N50_steps100_K{k}_*/step_sample_trace.npz")))
    else:
        p = None
    if p is None:
        raise FileNotFoundError(f"missing trace for {loss}, K={k}, method={method}")
    return p


def compute_curve(path: Path, *, bc_weight: float, physics_metric: str):
    z = np.load(path)
    rows = []
    for i, step in enumerate(np.asarray(z["steps"], dtype=int)):
        a = torch.as_tensor(np.asarray(z["A_adv"][i]), dtype=torch.float32).unsqueeze(0)
        u = torch.as_tensor(np.asarray(z["model_u"][i]), dtype=torch.float32).unsqueeze(0)
        pde, bc = per_sample_physics_components(a, u, physics_metric=physics_metric)
        pde_f = float(pde.item())
        bc_f = float(bc.item())
        loss4 = pde_f + float(bc_weight) * bc_f
        rows.append({
            "step": int(step),
            "loss4": loss4,
            "loss4_pde": pde_f,
            "loss4_bc": bc_f,
            "true_loss3": float(np.asarray(z["true_loss3"])[i]),
            "flip_count": int(np.asarray(z["flip_count"])[i]),
        })
    return rows


def write_csv(path: Path, all_rows: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["loss", "step", "loss4", "loss4_pde", "loss4_bc", "true_loss3", "flip_count", "loss4_increase", "loss4_ratio"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for loss, rows in all_rows.items():
            base = rows[0]["loss4"] if rows else 1.0
            for r in rows:
                out = dict(r)
                out["loss"] = loss
                out["loss4_increase"] = r["loss4"] - base
                out["loss4_ratio"] = r["loss4"] / max(abs(base), 1e-12)
                w.writerow(out)


def plot_curves(out_dir: Path, all_rows: dict[str, list[dict]], *, k: int, method: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0), constrained_layout=True)
    for loss, rows in all_rows.items():
        steps = np.array([r["step"] for r in rows])
        y = np.array([r["loss4"] for r in rows])
        base = y[0]
        axes[0].plot(steps, y, color=COLORS[loss], lw=2.0, label=loss)
        axes[1].plot(steps, y - base, color=COLORS[loss], lw=2.0, label=loss)
        axes[0].scatter([steps[-1]], [y[-1]], color=COLORS[loss], s=24)
        axes[1].scatter([steps[-1]], [y[-1] - base], color=COLORS[loss], s=24)
    axes[0].set_title("physics loss4 during each optimized attack")
    axes[0].set_xlabel("attack step")
    axes[0].set_ylabel("loss4 = PDE residual + BC residual")
    axes[1].set_title("physics loss4 increase from step 0")
    axes[1].set_xlabel("attack step")
    axes[1].set_ylabel("loss4 - loss4(step 0)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(f"Darcy K={k} {method}: loss4 as observer metric, sample00")
    fig.savefig(out_dir / f"darcy_K{k}_{method}_loss4_observer_curves.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0), constrained_layout=True)
    for loss, rows in all_rows.items():
        steps = np.array([r["step"] for r in rows])
        pde = np.array([r["loss4_pde"] for r in rows])
        bc = np.array([r["loss4_bc"] for r in rows])
        axes[0].plot(steps, pde, color=COLORS[loss], lw=2.0, label=loss)
        axes[1].plot(steps, bc, color=COLORS[loss], lw=2.0, label=loss)
    axes[0].set_title("PDE residual component")
    axes[0].set_xlabel("attack step")
    axes[0].set_ylabel("loss4_pde")
    axes[1].set_title("boundary residual component")
    axes[1].set_xlabel("attack step")
    axes[1].set_ylabel("loss4_bc")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(f"Darcy K={k} {method}: loss4 components, sample00")
    fig.savefig(out_dir / f"darcy_K{k}_{method}_loss4_observer_components.png", dpi=220)
    plt.close(fig)


def write_summary(path: Path, all_rows: dict[str, list[dict]], *, k: int, method: str, sources: dict[str, Path]) -> None:
    lines = [
        "# Darcy Loss4 Observer Curves", "",
        f"Setting: `K={k}`, method `{method}`, sample00 step traces.", "",
        "This evaluates the physics loss4 metric offline along attacks optimized for loss1/loss2/loss3/loss4.",
        "For every step, `A_adv` and `model_u` from `step_sample_trace.npz` are plugged into the same Darcy physics-loss function used by the loss4 attack:", "",
        "`loss4 = loss4_pde + loss4_bc` with `bc_weight=1.0` and `physics_metric=rel_l2`.", "",
        "## Final Values", "",
        "| optimized loss | step0 loss4 | final loss4 | increase | ratio | final true loss3 | source |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for loss, rows in all_rows.items():
        start = rows[0]
        final = rows[-1]
        ratio = final["loss4"] / max(abs(start["loss4"]), 1e-12)
        lines.append(
            f"| {loss} | {start['loss4']:.6g} | {final['loss4']:.6g} | {final['loss4'] - start['loss4']:.6g} | {ratio:.6g} | {final['true_loss3']:.6g} | `{sources[loss].relative_to(ROOT)}` |"
        )
    lines.extend([
        "", "## Outputs", "",
        f"- `darcy_K{k}_{method}_loss4_observer_curves.png`",
        f"- `darcy_K{k}_{method}_loss4_observer_components.png`",
        f"- `darcy_K{k}_{method}_loss4_observer_curves.csv`",
    ])
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=10920)
    ap.add_argument("--method", default="steepest_replace")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--bc-weight", type=float, default=1.0)
    ap.add_argument("--physics-metric", default="rel_l2", choices=["rel_l2", "mse"])
    args = ap.parse_args(argv)
    sources = {loss: trace_path(loss, args.k, args.method) for loss in LOSSES}
    all_rows = {loss: compute_curve(path, bc_weight=args.bc_weight, physics_metric=args.physics_metric) for loss, path in sources.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / f"darcy_K{args.k}_{args.method}_loss4_observer_curves.csv", all_rows)
    plot_curves(args.out, all_rows, k=args.k, method=args.method)
    write_summary(args.out / "README.md", all_rows, k=args.k, method=args.method, sources=sources)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
