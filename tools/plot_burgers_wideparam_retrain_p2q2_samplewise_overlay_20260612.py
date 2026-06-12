#!/usr/bin/env python3
"""Add sample-wise attack-loss bottom panels to the wideparam retrain P2Q2 overlays."""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
COMBINED_PLOTTER = REPO / "tools" / "plot_burgers_round03_p2q2_combined_attack_panels.py"
DEFAULT_TRACE_ROOT = (
    REPO
    / "forensics"
    / "burgers_wideparam_loss123_retrain_round00_p2q2_diverse_multi_sample_attack_visuals_batched_20260612"
)
DEFAULT_VIS_ROOT = (
    REPO
    / "visualizations"
    / "burgers_wideparam_loss123_retrain_round00_p2q2_comparison_dense_diverse_multi_sample_batched_20260612"
)
DEFAULT_BUNDLE_ROOT = (
    REPO
    / "visualizations"
    / "burgers_wideparam_loss123_retrain_comparison_dense_image_only_bundle_20260612"
)

DISPLAY = {
    "baseline": "Baseline model",
    "loss1": "Wideparam retrain loss1 epoch8000",
    "loss2": "Wideparam retrain loss2 epoch2000",
    "loss3": "Wideparam retrain loss3 epoch1000",
}
SHORT_DISPLAY = {
    "baseline": "baseline",
    "loss1": "loss1",
    "loss2": "loss2",
    "loss3": "loss3",
}


def load_combined_module():
    spec = importlib.util.spec_from_file_location("round03_combined_samplewise", COMBINED_PLOTTER)
    if spec is None or spec.loader is None:
        raise ImportError(COMBINED_PLOTTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_grid(dpi: int, bottom_layout: str):
    if bottom_layout == "one_row":
        fig_height = 17.2
        nrows = 7
        height_ratios = [1, 1, 1, 1, 1, 1, 0.70]
    else:
        fig_height = 18.8
        nrows = 8
        height_ratios = [1, 1, 1, 1, 1, 1, 0.70, 0.70]
    fig = plt.figure(figsize=(45.0, fig_height), facecolor="#fbfaf7", dpi=dpi)
    gs = fig.add_gridspec(
        nrows,
        16,
        height_ratios=height_ratios,
        left=0.045,
        right=0.992,
        top=0.875,
        bottom=0.055,
        hspace=0.36,
        wspace=0.20,
    )
    return fig, gs


def _sample_loss_ylim(models: dict[str, dict[str, np.ndarray]], model_order: list[str], sample_idx: int) -> tuple[float, float]:
    vals = np.concatenate([models[m]["loss"][:, sample_idx].reshape(-1) for m in model_order])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(vals.min())
    hi = float(vals.max())
    span = max(hi - lo, 1e-8)
    return max(0.0, lo - 0.08 * span), hi + 0.12 * span


def _sample_loss_ylim_log(models: dict[str, dict[str, np.ndarray]], model_order: list[str], sample_idx: int) -> tuple[float, float]:
    vals = np.concatenate([models[m]["loss"][:, sample_idx].reshape(-1) for m in model_order])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1e-8, 1.0
    lo = max(float(vals.min()) / 1.35, 1e-10)
    hi = float(vals.max()) * 1.35
    if hi <= lo:
        hi = lo * 10.0
    return lo, hi


def render_overlay_samplewise(
    mod,
    out_path: Path,
    clean: np.ndarray,
    manifest: list[dict[str, object]],
    models: dict[str, dict[str, np.ndarray]],
    limits: dict[str, list[tuple[float, float]]],
    *,
    dpi: int = 160,
    bottom_layout: str = "two_rows",
    loss_yscale: str = "linear",
) -> None:
    mod.set_style()
    model_keys = ["baseline", "loss1", "loss2", "loss3"]
    before_idx = 0
    after_idx = int(len(models["baseline"]["step"]) - 1)
    after_step = int(models["baseline"]["step"][after_idx])
    grid = np.linspace(0.0, 1.0, clean.shape[1])
    fig, gs = make_grid(dpi, bottom_layout)
    row_label_positions: list[tuple[float, str]] = []
    group_centers: list[tuple[float, str, str]] = []

    for row in range(clean.shape[0]):
        item = manifest[row]
        first_ax_for_row = None
        for group_idx, model_key in enumerate(model_keys):
            base_col = group_idx * 4
            color = mod.COLORS[model_key]
            tr = models[model_key]

            ax = fig.add_subplot(gs[row, base_col])
            if first_ax_for_row is None:
                first_ax_for_row = ax
            ax.plot(grid, tr["delta"][before_idx, row], color=color, lw=1.05, alpha=0.80, ls="--", label="before")
            ax.plot(grid, tr["delta"][after_idx, row], color=color, lw=1.35, alpha=0.94, label="after")
            ax.axhline(0.0, color="#222222", lw=0.65, alpha=0.35)
            ax.set_ylim(*limits["delta"][row])
            if row == 0:
                ax.set_title(mod.PANEL_NAMES[0], fontsize=9.5, pad=7, fontweight="bold")
                ax.legend(loc="upper right", fontsize=6.1, handlelength=1.3)
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

            ax = fig.add_subplot(gs[row, base_col + 1])
            clean_y = clean[row]
            before_x = tr["x_adv"][before_idx, row]
            after_x = tr["x_adv"][after_idx, row]
            ax.plot(grid, clean_y, color="#111111", lw=1.20, alpha=0.78, label="clean")
            ax.plot(grid, before_x, color=color, lw=1.05, alpha=0.80, ls="--", label="before adv")
            ax.plot(grid, after_x, color=color, lw=1.35, alpha=0.94, label="after adv")
            ax.fill_between(grid, clean_y, before_x, color=color, alpha=0.045, linewidth=0)
            ax.fill_between(grid, clean_y, after_x, color=color, alpha=0.115, linewidth=0)
            ax.set_ylim(*limits["input"][row])
            if row == 0:
                ax.set_title(mod.PANEL_NAMES[1], fontsize=9.5, pad=7, fontweight="bold")
                ax.legend(loc="upper right", fontsize=5.9, handlelength=1.2)
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

            ax = fig.add_subplot(gs[row, base_col + 2])
            before_model = tr["model"][before_idx, row]
            before_solver = tr["solver"][before_idx, row]
            after_model = tr["model"][after_idx, row]
            after_solver = tr["solver"][after_idx, row]
            ax.plot(grid, before_solver, color=mod.SOLVER_COLOR, lw=1.05, alpha=0.80, ls="--", label="solver before")
            ax.plot(grid, before_model, color=color, lw=1.05, alpha=0.80, ls="--", label="model before")
            ax.plot(grid, after_solver, color=mod.SOLVER_COLOR, lw=1.45, alpha=0.92, label="solver after")
            ax.plot(grid, after_model, color=color, lw=1.35, alpha=0.94, label="model after")
            ax.fill_between(grid, before_solver, before_model, color=color, alpha=0.045, linewidth=0)
            ax.fill_between(grid, after_solver, after_model, color=color, alpha=0.135, linewidth=0)
            ax.set_ylim(*limits["output"][row])
            if row == 0:
                ax.set_title(mod.PANEL_NAMES[2], fontsize=9.5, pad=7, fontweight="bold")
                ax.legend(loc="upper right", fontsize=5.5, handlelength=1.0)
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

            ax = fig.add_subplot(gs[row, base_col + 3])
            before_diff = tr["diff"][before_idx, row]
            after_diff = tr["diff"][after_idx, row]
            ax.plot(grid, before_diff, color=color, lw=1.05, alpha=0.80, ls="--", label="before")
            ax.plot(grid, after_diff, color=color, lw=1.30, alpha=0.94, label="after")
            ax.fill_between(grid, 0.0, before_diff, color=color, alpha=0.045, linewidth=0)
            ax.fill_between(grid, 0.0, after_diff, color=color, alpha=0.135, linewidth=0)
            ax.axhline(0.0, color="#222222", lw=0.65, alpha=0.45)
            ax.set_ylim(*limits["diff"][row])
            before_loss = float(tr["loss"][before_idx, row])
            after_loss = float(tr["loss"][after_idx, row])
            before_rms = float(tr["delta_rms"][before_idx, row])
            after_rms = float(tr["delta_rms"][after_idx, row])
            if row == 0:
                ax.set_title(mod.PANEL_NAMES[3], fontsize=9.5, pad=7, fontweight="bold")
                ax.legend(loc="lower right", fontsize=6.0, handlelength=1.2)
            ax.text(
                0.025,
                0.93,
                f"pre {before_loss:.1e}/{before_rms:.3f}\naft {after_loss:.1e}/{after_rms:.3f}",
                transform=ax.transAxes,
                fontsize=5.7,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.35},
            )
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

        if first_ax_for_row is not None:
            bbox = first_ax_for_row.get_position()
            row_label_positions.append(((bbox.y0 + bbox.y1) / 2.0, mod.sample_label(row, item)))

    mod.add_row_labels(fig, row_label_positions)

    # Bottom: one attack-loss subplot per sample, with all four models overlaid.
    if bottom_layout == "one_row":
        bottom_gs = gs[6, 0:16].subgridspec(1, clean.shape[0], wspace=0.34)
    elif bottom_layout == "two_rows":
        col_spans = [(0, 5), (5, 10), (10, 16)]
    else:
        raise ValueError(f"unknown bottom_layout={bottom_layout!r}")
    for sample_idx in range(clean.shape[0]):
        if bottom_layout == "one_row":
            ax = fig.add_subplot(bottom_gs[0, sample_idx])
        else:
            row_block = 6 + sample_idx // 3
            col0, col1 = col_spans[sample_idx % 3]
            ax = fig.add_subplot(gs[row_block, col0:col1])
        for model_key in model_keys:
            tr = models[model_key]
            ax.plot(
                tr["step"],
                tr["loss"][:, sample_idx],
                color=mod.COLORS[model_key],
                lw=1.35,
                alpha=0.92,
                label=SHORT_DISPLAY[model_key],
            )
        ax.axvline(0, color="#111111", lw=0.85, alpha=0.36, ls="--")
        ax.axvline(after_step, color="#111111", lw=0.95, alpha=0.55)
        ax.set_xlim(0, after_step)
        if loss_yscale == "log":
            ax.set_yscale("log")
            ax.set_ylim(*_sample_loss_ylim_log(models, model_keys, sample_idx))
        elif loss_yscale == "linear":
            ax.set_ylim(*_sample_loss_ylim(models, model_keys, sample_idx))
        else:
            raise ValueError(f"unknown loss_yscale={loss_yscale!r}")
        item = manifest[sample_idx]
        split = str(item.get("split", ""))
        title_size = 7.6 if bottom_layout == "one_row" else 8.8
        label_size = 6.8 if bottom_layout == "one_row" else 7.7
        tick_size = 5.9 if bottom_layout == "one_row" else 6.7
        suffix = " (log)" if loss_yscale == "log" else ""
        ax.set_title(f"S{sample_idx + 1} {split}: attack loss{suffix}", fontsize=title_size, loc="left", fontweight="bold", pad=4)
        ax.set_xlabel("attack step", fontsize=label_size)
        if sample_idx == 0 or bottom_layout == "two_rows":
            ylabel = "MSE (log)" if loss_yscale == "log" else "MSE"
            ax.set_ylabel(ylabel, fontsize=label_size)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=tick_size, pad=1.2)
        if sample_idx == 0:
            ax.legend(
                ncol=2 if bottom_layout == "one_row" else 4,
                fontsize=5.7 if bottom_layout == "one_row" else 6.5,
                loc="upper left",
                handlelength=1.0,
            )

    for group_idx, model_key in enumerate(model_keys):
        x = 0.045 + (group_idx + 0.5) * (0.992 - 0.045) / 4.0
        fig.text(
            x,
            0.925,
            DISPLAY[model_key],
            ha="center",
            va="center",
            fontsize=11.0,
            fontweight="bold",
            color=mod.COLORS[model_key],
            bbox={"boxstyle": "round,pad=0.30", "facecolor": "#ffffff", "edgecolor": mod.COLORS[model_key], "alpha": 0.94},
        )
    fig.suptitle(
        "Burgers wideparam retrain p=2, q=2 RMS-L2 attack: before/after overlay with sample-wise loss curves",
        fontsize=15.0,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.956,
        "Top panels keep the before/after overlay; bottom panels compare baseline/loss1/loss2/loss3 within each initial condition.",
        ha="center",
        va="center",
        fontsize=9.0,
        color=mod.MUTED,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--vis-root", type=Path, default=DEFAULT_VIS_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--num-groups", type=int, default=5)
    args = parser.parse_args()

    mod = load_combined_module()
    mod.DISPLAY.update(DISPLAY)
    outputs = []
    bundle_outputs = []
    one_row_outputs = []
    one_row_bundle_outputs = []
    for group_id in range(args.num_groups):
        group_trace_root = args.trace_root / f"group{group_id:02d}"
        group_vis_dir = args.vis_root / "comparison_dense" / f"group{group_id:02d}"
        mod.TRACE_ROOT = group_trace_root
        mod.OUT_DIR = group_vis_dir
        mod.OUTPUT_PREFIX = f"wideparam_loss3targeted_round00_group{group_id:02d}_p2q2"
        clean, manifest, models, limits, _loss_ylim, _final_idx = mod.build_data()
        out_path = group_vis_dir / f"{mod.OUTPUT_PREFIX}_baseline_loss1_loss2_loss3_before_after_overlay_four_column_samplewise_loss.png"
        render_overlay_samplewise(mod, out_path, clean, manifest, models, limits)
        outputs.append(out_path)

        bundle_path = args.bundle_root / "comparison_dense" / f"group{group_id:02d}" / out_path.name
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, bundle_path)
        bundle_outputs.append(bundle_path)

        one_row_path = group_vis_dir / f"{mod.OUTPUT_PREFIX}_baseline_loss1_loss2_loss3_before_after_overlay_four_column_samplewise_loss_one_row.png"
        render_overlay_samplewise(mod, one_row_path, clean, manifest, models, limits, bottom_layout="one_row")
        one_row_outputs.append(one_row_path)

        one_row_bundle_path = args.bundle_root / "comparison_dense" / f"group{group_id:02d}" / one_row_path.name
        one_row_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(one_row_path, one_row_bundle_path)
        one_row_bundle_outputs.append(one_row_bundle_path)

    print("Generated sample-wise overlay PNGs:")
    for path in outputs:
        print(path)
    print("Copied into image-only bundle:")
    for path in bundle_outputs:
        print(path)
    print("Generated one-row sample-wise overlay PNGs:")
    for path in one_row_outputs:
        print(path)
    print("Copied one-row overlays into image-only bundle:")
    for path in one_row_bundle_outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
