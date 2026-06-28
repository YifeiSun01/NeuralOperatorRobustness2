#!/usr/bin/env python3
"""Replot the NS FNO2d delta-RMSE bar chart with one range dropped or hidden.

This reuses the project's existing bar-plot ordering, colors, hatches, and
baseline subtraction code.  By default it drops range=(-1.5,-1) bars and
compresses the x positions so no blank gaps remain.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = (
    PROJECT_ROOT
    / "2D_NS_FNO2d_recurrent"
    / "continual_train_attack"
    / "eval_models"
    / "performance_results"
    / "combined_eval__roots_models.csv"
)
DEFAULT_PLOT_MODULE = (
    PROJECT_ROOT
    / "2D_NS_FNO2d_recurrent"
    / "continual_train_attack"
    / "eval_models"
    / "plot_results.py"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "ns_fno2d_r6_m96x96_w80_remove_range_20260615"
)
DEFAULT_MODEL = "FNO2d_r6_m96x96_w80_Tin10_T10"


def load_plot_module(path: Path):
    spec = importlib.util.spec_from_file_location("ns_plot_results", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plot module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_pair(lo: float, hi: float) -> tuple[float, float]:
    lo_f = float(lo)
    hi_f = float(hi)
    return (lo_f, hi_f) if lo_f <= hi_f else (hi_f, lo_f)


def range_matches(rt, target: tuple[float, float], atol: float = 1e-9) -> bool:
    if rt is None:
        return False
    lo, hi = canonical_pair(rt[0], rt[1])
    return abs(lo - target[0]) <= atol and abs(hi - target[1]) <= atol


def target_range_row_mask(plot_mod, pivot, target: tuple[float, float]) -> np.ndarray:
    if isinstance(pivot.index, plot_mod.pd.MultiIndex):
        idx_names = pivot.index.names
        rows = list(pivot.index)
    else:
        idx_names = [pivot.index.name or "index"]
        rows = [(v,) for v in pivot.index.values]

    mask = []
    for row in rows:
        group = plot_mod.infer_group_from_row(idx_names, row)
        rt = plot_mod.extract_range_from_row(idx_names, row)
        mask.append(group == "generalizability" and range_matches(rt, target))
    return np.asarray(mask, dtype=bool)


def first_non_baseline_column(plot_mod, pivot, model_name: str):
    if isinstance(pivot.columns, plot_mod.pd.MultiIndex):
        col_names = list(pivot.columns.names or [])
        candidates = []
        for col in pivot.columns:
            is_baseline = False
            if "model_group" in col_names:
                is_baseline = str(col[col_names.index("model_group")]) == "baseline"
            if is_baseline:
                continue
            if "model_name" in col_names:
                name = str(col[col_names.index("model_name")])
                if name != model_name:
                    continue
            candidates.append(col)
        if not candidates:
            raise RuntimeError(f"No non-baseline column found for model_name={model_name!r}")
        return candidates[0]

    for col in pivot.columns:
        if str(col).lower() != "baseline":
            return col
    raise RuntimeError("No non-baseline column found")


def plot_single(
    plot_mod,
    pivot_minus,
    model_name: str,
    out_path: Path,
    target: tuple[float, float],
    mode: str,
) -> dict[str, int]:
    hidden_original = target_range_row_mask(plot_mod, pivot_minus, target)
    visible_pivot = pivot_minus.loc[~hidden_original] if mode == "drop" else pivot_minus

    range_palette = plot_mod.build_generalizability_range_palette(pivot_minus, cmap_name="coolwarm")
    row_order, _, _, _, _, meta_by_new, _ = plot_mod.compute_row_order_and_layout(
        visible_pivot, range_palette
    )
    pivot_ord = visible_pivot.iloc[row_order, :]

    if mode == "drop":
        hide_ordered = np.zeros(len(pivot_ord), dtype=bool)
        removed_count = int(hidden_original.sum())
    else:
        hide_ordered = target_range_row_mask(plot_mod, pivot_ord, target)
        removed_count = int(hide_ordered.sum())

    col = first_non_baseline_column(plot_mod, pivot_ord, model_name)
    series = pivot_ord[col]
    y = plot_mod.pd.to_numeric(series, errors="coerce").to_numpy(dtype=float).copy()
    full_col = first_non_baseline_column(plot_mod, pivot_minus, model_name)
    y_for_limits = plot_mod.pd.to_numeric(
        pivot_minus[full_col], errors="coerce"
    ).to_numpy(dtype=float)
    if mode == "hide":
        y[hide_ordered] = np.nan

    colors, hatches = [], []
    for m in meta_by_new:
        if m["group"] == "generalizability":
            label = m["range_label"] or "range=unknown"
            colors.append(range_palette.get(label, plot_mod.get_cmap("coolwarm")(0.5)))
            hatches.append(plot_mod.KERNEL_HATCH.get(m["kernel"], None))
        else:
            colors.append(plot_mod.GROUP_BASE_COLOR.get(m["group"], plot_mod.GROUP_BASE_COLOR["other"]))
            hatches.append(None)

    fig_w = max(20, int(len(series) * 0.06))
    fig, ax = plot_mod.plt.subplots(figsize=(fig_w, 13))
    x = np.arange(len(series))

    bars = ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.0, align="center")
    for bar, hatch, hide in zip(bars, hatches, hide_ordered):
        if hide:
            bar.set_alpha(0.0)
            bar.set_linewidth(0.0)
            continue
        if hatch:
            bar.set_hatch(hatch)

    ax.set_xlim(-0.5, len(series) - 0.5)
    finite_limits = y_for_limits[np.isfinite(y_for_limits)]
    if finite_limits.size:
        lo = min(float(finite_limits.min()), 0.0)
        hi = max(float(finite_limits.max()), 0.0)
        pad = (hi - lo) * 0.05 if hi > lo else 0.05
        ax.set_ylim(lo - pad, hi + pad)
    ax.margins(x=0)
    ax.set_title(plot_mod.wrap_and_truncate_title(model_name), fontsize=14, pad=24)
    ax.set_ylabel("ΔRMSE", fontsize=12)
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.6)
    plot_mod.plt.subplots_adjust(top=0.78)

    hidden_range_label = f"range=({target[0]:g},{target[1]:g})"
    legend_elems = []
    visible_range_labels = [
        m["range_label"]
        for m, hide in zip(meta_by_new, hide_ordered)
        if m["group"] == "generalizability" and not hide and m["range_label"]
    ]
    for rl in sorted(set(visible_range_labels)):
        if rl == hidden_range_label:
            continue
        display_label = f"generalizability\n{rl}"
        color = range_palette.get(rl, plot_mod.get_cmap("coolwarm")(0.5))
        legend_elems.append(plot_mod.Patch(facecolor=color, edgecolor="none", label=display_label))

    kernels = sorted(
        {m["kernel"] for m, hide in zip(meta_by_new, hide_ordered) if m["group"] == "generalizability" and not hide},
        key=lambda k: (
            plot_mod.KERNEL_ORDER.index(k)
            if k in plot_mod.KERNEL_ORDER
            else len(plot_mod.KERNEL_ORDER)
        ),
    )
    for kernel in kernels:
        hatch = plot_mod.KERNEL_HATCH.get(kernel)
        if hatch:
            legend_elems.append(
                plot_mod.Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch=hatch,
                    linewidth=0.0,
                    label=f"kernel={kernel}",
                )
            )

    for group in ["test", "train", "expanded"]:
        if any(m["group"] == group for m in meta_by_new):
            legend_elems.append(
                plot_mod.Patch(facecolor=plot_mod.GROUP_BASE_COLOR[group], edgecolor="none", label=group)
            )

    if legend_elems:
        ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Legend")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=plot_mod.FIG_DPI, bbox_inches="tight")
    plot_mod.plt.close(fig)

    return {
        "rows_total": int(len(series)),
        "target_range_rows": removed_count,
        "visible_rows": int(len(series) - removed_count if mode == "hide" else len(series)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--plot-module", type=Path, default=DEFAULT_PLOT_MODULE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--range-low", type=float, default=-1.5)
    parser.add_argument("--range-high", type=float, default=-1.0)
    parser.add_argument("--mode", choices=["hide", "drop"], default="drop")
    parser.add_argument(
        "--families",
        nargs="+",
        default=["GRF", "LogGRF", "NegLogGRF"],
        help="Use one or more of GRF, LogGRF, NegLogGRF.",
    )
    args = parser.parse_args()

    plot_mod = load_plot_module(args.plot_module)
    pd = plot_mod.pd

    df = pd.read_csv(args.csv, low_memory=False)
    keep = (df["model_group"].astype(str) == "baseline") | (df["model_name"].astype(str) == args.model_name)
    df = df.loc[keep].copy()
    if df.empty:
        raise RuntimeError("No baseline/model rows selected")

    df = plot_mod.ensure_model_hparams(df)
    pivot_full = plot_mod.build_pivot_multi_metrics(
        df,
        values=("rmse_mean", "rmse_std", "mae_mean", "mae_std", "mape_mean", "mape_std"),
        aggfunc="mean",
    )
    pivot_minus, _ = plot_mod.subtract_baseline_rmse(pivot_full, metric="rmse_mean")

    target = canonical_pair(args.range_low, args.range_high)
    stem = f"{args.model_name}_remove_range_{target[0]:g}_{target[1]:g}_{args.mode}".replace("-", "m").replace(".", "p")
    summary_rows = []

    for family in args.families:
        sub = plot_mod.filter_pivot_by_family_including_others(pivot_minus, family)
        out_path = args.outdir / family / f"{stem}.png"
        stats = plot_single(plot_mod, sub, args.model_name, out_path, target, args.mode)
        stats["family"] = family
        stats["output"] = str(out_path)
        summary_rows.append(stats)
        print(f"[saved] {family}: {out_path} ({stats})", flush=True)

    summary = pd.DataFrame(summary_rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / f"summary_{args.mode}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[saved] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
