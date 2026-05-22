#!/usr/bin/env python3
"""CPU-only final-state attack panels from saved final_state_outputs.npz.

This script does not import torch or jax and does not rerun the model/solver.
It plots the clean and adversarial final-step arrays already recorded by
attack_ns2d_recurrent_core4.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]


def sym_limits(*arrays: np.ndarray, pct: float = 99.5) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    vmax = float(np.percentile(np.abs(vals), pct))
    if vmax <= 0:
        vmax = float(np.max(np.abs(vals))) if vals.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def pos_limits(*arrays: np.ndarray, pct: float = 99.5) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmax = float(np.percentile(vals, pct))
    if vmax <= 0:
        vmax = float(np.max(vals)) if vals.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    return 0.0, vmax


def draw_panel(fig, ax, arr: np.ndarray, title: str, vlim: tuple[float, float], cmap: str = "coolwarm") -> None:
    im = ax.imshow(arr, cmap=cmap, origin="lower", vmin=vlim[0], vmax=vlim[1])
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_summary(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def plot_one(
    *,
    out_path: Path,
    loss_name: str,
    method: str,
    sample_position: int,
    dataset_index: int,
    arrays: dict[str, np.ndarray],
    sample_metric: dict[str, str] | None,
) -> dict[str, float | int | str]:
    x_clean = arrays["x_clean"]
    x_adv = arrays["x_adv"]
    delta = arrays["final_delta"]
    clean_model = arrays["clean_model_final"]
    clean_solver = arrays["clean_solver_final"]
    adv_model = arrays["adv_model_final"]
    adv_solver = arrays["adv_solver_final"]
    clean_diff = arrays["clean_model_minus_solver"]
    adv_diff = arrays["adv_model_minus_solver"]
    model_change = arrays["model_final_change"]
    solver_change = arrays["solver_final_change"]

    state_lim = sym_limits(x_clean, x_adv)
    delta_lim = sym_limits(delta, x_adv - x_clean)
    final_lim = sym_limits(clean_model, clean_solver, adv_model, adv_solver)
    clean_diff_lim = sym_limits(clean_diff)
    adv_diff_lim = sym_limits(adv_diff)
    model_change_lim = sym_limits(model_change)
    solver_change_lim = sym_limits(solver_change)
    abs_delta_lim = pos_limits(np.abs(delta))

    delta_l2 = float(np.linalg.norm(delta.reshape(-1), ord=2))
    delta_linf = float(np.max(np.abs(delta)))
    clean_err_l2 = float(np.linalg.norm(clean_diff.reshape(-1), ord=2))
    adv_err_l2 = float(np.linalg.norm(adv_diff.reshape(-1), ord=2))
    true_loss_increase = adv_err_l2 - clean_err_l2
    true_loss_ratio = adv_err_l2 / clean_err_l2 if clean_err_l2 != 0 else float("nan")
    model_change_l2 = float(np.linalg.norm(model_change.reshape(-1), ord=2))
    solver_change_l2 = float(np.linalg.norm(solver_change.reshape(-1), ord=2))

    saved_clean_loss = float(arrays["clean_true_loss"])
    saved_adv_loss = float(arrays["adv_true_loss"])
    saved_loss_increase = saved_adv_loss - saved_clean_loss
    saved_loss_ratio = saved_adv_loss / saved_clean_loss if saved_clean_loss != 0 else float("nan")

    fig, axes = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
    fig.suptitle(
        f"{loss_name}/{method} final-state panels | sample_pos={sample_position} dataset_index={dataset_index} | "
        f"delta L2={delta_l2:.4g}, Linf={delta_linf:.4g} | "
        f"clean true={saved_clean_loss:.4g}, adv true={saved_adv_loss:.4g}, "
        f"diff={saved_loss_increase:+.4g}, ratio={saved_loss_ratio:.4g}",
        fontsize=12,
    )

    draw_panel(fig, axes[0, 0], x_clean, "clean initial x", state_lim)
    draw_panel(fig, axes[0, 1], x_adv, "perturbed initial x+delta", state_lim)
    draw_panel(fig, axes[0, 2], delta, "final delta = x_adv - x_clean", delta_lim)
    draw_panel(fig, axes[0, 3], np.abs(delta), "abs(final delta)", abs_delta_lim, cmap="viridis")

    draw_panel(fig, axes[1, 0], clean_model, "clean FNO final F(x)", final_lim)
    draw_panel(fig, axes[1, 1], clean_solver, "clean solver final G(x)", final_lim)
    draw_panel(fig, axes[1, 2], clean_diff, f"clean FNO - solver\nL2 loss={clean_err_l2:.4g}", clean_diff_lim)
    draw_panel(fig, axes[1, 3], model_change, f"FNO final change F(x+delta)-F(x)\nL2={model_change_l2:.4g}", model_change_lim)

    draw_panel(fig, axes[2, 0], adv_model, "adv FNO final F(x+delta)", final_lim)
    draw_panel(fig, axes[2, 1], adv_solver, "adv solver final G(x+delta)", final_lim)
    draw_panel(fig, axes[2, 2], adv_diff, f"adv FNO - solver\nL2 loss={adv_err_l2:.4g}, diff={true_loss_increase:+.4g}", adv_diff_lim)
    draw_panel(fig, axes[2, 3], solver_change, f"solver final change G(x+delta)-G(x)\nL2={solver_change_l2:.4g}", solver_change_lim)

    extra = "difference panels use independent color ranges"
    if sample_metric:
        metric_bits = []
        for key in ("active_loss_value", "surrogate_loss_value", "true_loss_value", "delta_p", "boundary_ratio"):
            value = sample_metric.get(key)
            if value not in (None, ""):
                try:
                    metric_bits.append(f"{key}={float(value):.4g}")
                except ValueError:
                    metric_bits.append(f"{key}={value}")
        if metric_bits:
            extra = "; ".join(metric_bits) + "; " + extra
    fig.text(0.5, 0.005, extra, ha="center", va="bottom", fontsize=9)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "loss": loss_name,
        "method": method,
        "sample_position": int(sample_position),
        "dataset_index": int(dataset_index),
        "delta_l2": delta_l2,
        "delta_linf": delta_linf,
        "clean_true_loss": saved_clean_loss,
        "adv_true_loss": saved_adv_loss,
        "true_loss_increase": saved_loss_increase,
        "true_loss_ratio": saved_loss_ratio,
        "clean_model_solver_l2": clean_err_l2,
        "adv_model_solver_l2": adv_err_l2,
        "model_final_change_l2": model_change_l2,
        "solver_final_change_l2": solver_change_l2,
        "png": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-root", type=Path, required=True, help="Directory containing method subdirs for one loss.")
    parser.add_argument("--loss-name", default="loss1")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--sample-positions", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for method in args.methods:
        method_dir = args.loss_root / method
        npz_path = method_dir / "final_state_outputs.npz"
        if not npz_path.exists():
            print(f"[skip] missing {npz_path}")
            continue
        print(f"[plot] {args.loss_name}/{method}: {npz_path}")
        z = np.load(npz_path)
        dataset_indices = z["dataset_indices"].astype(int)
        sample_metrics = read_csv_rows(method_dir / "final_state_metrics.csv")
        summary = load_summary(method_dir / "summary.json")

        positions = [p for p in args.sample_positions if 0 <= p < len(dataset_indices)]
        for pos in positions:
            arrays = {key: z[key][pos] for key in z.keys() if key != "dataset_indices"}
            metric = sample_metrics[pos] if pos < len(sample_metrics) else None
            out_png = args.out_dir / f"{args.loss_name}_{method}_samplepos{pos}_idx{int(dataset_indices[pos])}_full_final_panels.png"
            rows.append(
                plot_one(
                    out_path=out_png,
                    loss_name=args.loss_name,
                    method=method,
                    sample_position=pos,
                    dataset_index=int(dataset_indices[pos]),
                    arrays=arrays,
                    sample_metric=metric,
                )
            )

        method_report = {
            "loss_root": str(args.loss_root),
            "method": method,
            "summary": summary,
            "npz": str(npz_path),
        }
        (args.out_dir / f"{args.loss_name}_{method}_source_summary.json").write_text(json.dumps(method_report, indent=2))

    report = {
        "note": "CPU-only visualization from saved final_state_outputs.npz; no torch/jax/model/solver rerun.",
        "loss_root": str(args.loss_root),
        "loss_name": args.loss_name,
        "rows": rows,
    }
    report_path = args.out_dir / f"{args.loss_name}_saved_final_state_panel_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[done] wrote {len(rows)} panels under {args.out_dir}")
    print(f"[done] report {report_path}")


if __name__ == "__main__":
    main()
