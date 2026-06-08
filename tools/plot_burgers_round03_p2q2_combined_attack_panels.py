#!/usr/bin/env python3
"""Render combined Burgers round03 p2q2 before/after attack panels from saved traces."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TRACE_ROOT = REPO / "forensics" / "burgers_round03_baseline_vs_final_p2q2_attack_verification_20260607"
BUNDLE_ROOT = REPO / "visualizations" / "burgers_loss3_selective_round03_longtraining_comparison_dense_image_only_bundle_20260607"
OUT_DIR = BUNDLE_ROOT / "comparison_dense"

LOSS_ORDER = ["loss1", "loss2", "loss3"]
MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3"]
DISPLAY = {
    "baseline": "Baseline model",
    "loss1": "Round03 loss1 epoch8000",
    "loss2": "Round03 loss2 epoch2000",
    "loss3": "Round03 loss3 epoch1500",
}
COLORS = {
    "baseline": "#2f6f9f",
    "loss1": "#2f9b75",
    "loss2": "#d98a2b",
    "loss3": "#c35b5b",
}
SOLVER_COLOR = "#17191c"
SAMPLE_COLORS = ["#2f6f9f", "#2f9b75", "#d9a441", "#c35b5b", "#7b5fb3", "#5d6470"]
BG = "#fbfaf7"
TEXT = "#202124"
MUTED = "#62666d"
PANEL_NAMES = ["delta", "initial condition", "model and solver", "model - solver"]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#fffefa",
            "axes.edgecolor": "#b8b4aa",
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.color": "#d8d4ca",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.45,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
        }
    )


def short_dataset_label(dataset_id: str) -> str:
    replacements = [
        ("test_original_", "test "),
        ("burgers_", ""),
        ("gaussian_", "gauss "),
        ("matern_", "matern "),
        ("corr", "c"),
        ("scale", "s"),
        ("shift", "sh"),
        ("add_", "add "),
        ("target_", "target "),
        ("near_", "near "),
        ("mid_", "mid "),
        ("far_", "far "),
    ]
    out = dataset_id
    for old, new in replacements:
        out = out.replace(old, new)
    return out.replace("_", " ")


def load_trace(loss_name: str) -> dict[str, np.ndarray]:
    path = TRACE_ROOT / loss_name / "attack_traces.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(np.load(path))


def extract(z: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {
        "step": z[f"{prefix}_step"],
        "x_adv": z[f"{prefix}_x_adv"],
        "delta": z[f"{prefix}_delta"],
        "model": z[f"{prefix}_model"],
        "solver": z[f"{prefix}_solver"],
        "diff": z[f"{prefix}_diff"],
        "loss": z[f"{prefix}_loss"],
        "delta_rms": z[f"{prefix}_delta_rms"],
    }


def y_limits(vals: np.ndarray, pad: float = 0.05) -> tuple[float, float]:
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    span = max(hi - lo, 1e-6)
    return lo - pad * span, hi + pad * span


def build_data() -> tuple[np.ndarray, list[dict[str, object]], dict[str, dict[str, np.ndarray]], dict[str, list[tuple[float, float]]], tuple[float, float], int]:
    manifest = json.loads((TRACE_ROOT / "sample_manifest.json").read_text())
    raw = {loss: load_trace(loss) for loss in LOSS_ORDER}
    clean = raw["loss1"]["clean"]

    baseline = extract(raw["loss1"], "baseline")
    for loss in ["loss2", "loss3"]:
        other_clean = raw[loss]["clean"]
        if not np.allclose(clean, other_clean):
            raise RuntimeError(f"clean samples differ for {loss}")
        other_baseline = extract(raw[loss], "baseline")
        for key in ["step", "x_adv", "delta", "model", "solver", "diff", "loss", "delta_rms"]:
            if not np.allclose(baseline[key], other_baseline[key]):
                raise RuntimeError(f"baseline trace differs in {loss}:{key}")

    models = {"baseline": baseline}
    for loss in LOSS_ORDER:
        models[loss] = extract(raw[loss], "target")

    limits: dict[str, list[tuple[float, float]]] = {"input": [], "delta": [], "output": [], "diff": []}
    for i in range(clean.shape[0]):
        input_vals = np.concatenate([clean[None, i], *[models[k]["x_adv"][:, i] for k in MODEL_ORDER]], axis=0)
        delta_vals = np.concatenate([models[k]["delta"][:, i] for k in MODEL_ORDER], axis=0)
        output_vals = np.concatenate(
            [*[models[k]["model"][:, i] for k in MODEL_ORDER], *[models[k]["solver"][:, i] for k in MODEL_ORDER]],
            axis=0,
        )
        diff_vals = np.concatenate([models[k]["diff"][:, i] for k in MODEL_ORDER], axis=0)
        limits["input"].append(y_limits(input_vals))
        limits["delta"].append(y_limits(delta_vals, pad=0.12))
        limits["output"].append(y_limits(output_vals))
        max_abs = float(np.nanmax(np.abs(diff_vals)))
        limits["diff"].append((-1.08 * max_abs, 1.08 * max_abs))

    all_losses = np.concatenate([models[k]["loss"].reshape(-1) for k in MODEL_ORDER])
    lo = float(all_losses.min())
    hi = float(all_losses.max())
    span = max(hi - lo, 1e-8)
    loss_ylim = (max(0.0, lo - 0.05 * span), hi + 0.08 * span)
    final_idx = int(len(models["baseline"]["step"]) - 1)
    return clean, manifest, models, limits, loss_ylim, final_idx


def add_row_labels(fig: plt.Figure, positions: list[tuple[float, str]]) -> None:
    for y, label in positions:
        fig.text(
            0.009,
            y,
            label,
            ha="left",
            va="center",
            fontsize=6.8,
            color=TEXT,
            linespacing=1.15,
            bbox={"facecolor": BG, "edgecolor": "none", "alpha": 0.95, "pad": 0.2},
        )


def make_grid(n_models: int, dpi: int) -> tuple[plt.Figure, matplotlib.gridspec.GridSpec]:
    fig_width = 11.25 * n_models
    left = 0.082 if n_models == 2 else 0.045
    fig = plt.figure(figsize=(fig_width, 15.8), facecolor=BG, dpi=dpi)
    gs = fig.add_gridspec(
        7,
        4 * n_models,
        height_ratios=[1, 1, 1, 1, 1, 1, 0.70],
        left=left,
        right=0.992,
        top=0.875,
        bottom=0.065,
        hspace=0.30,
        wspace=0.20,
    )
    return fig, gs


def render_frame(
    out_path: Path,
    model_keys: list[str],
    frame_idx: int,
    clean: np.ndarray,
    manifest: list[dict[str, object]],
    models: dict[str, dict[str, np.ndarray]],
    limits: dict[str, list[tuple[float, float]]],
    loss_ylim: tuple[float, float],
    *,
    dpi: int = 160,
) -> None:
    set_style()
    step = int(models["baseline"]["step"][frame_idx])
    grid = np.linspace(0.0, 1.0, clean.shape[1])
    fig, gs = make_grid(len(model_keys), dpi)
    row_label_positions: list[tuple[float, str]] = []
    group_centers: list[tuple[float, str, str]] = []

    for row in range(clean.shape[0]):
        item = manifest[row]
        sample_label = "\n".join([f"S{row + 1}  {item['split']}", short_dataset_label(str(item["dataset_id"])), f"index {item['index']}"])
        first_ax_for_row = None
        for group_idx, model_key in enumerate(model_keys):
            base_col = group_idx * 4
            color = COLORS[model_key]
            tr = models[model_key]

            ax = fig.add_subplot(gs[row, base_col])
            if first_ax_for_row is None:
                first_ax_for_row = ax
            ax.plot(grid, tr["delta"][frame_idx, row], color=color, lw=1.35)
            ax.axhline(0.0, color="#222222", lw=0.65, alpha=0.35)
            ax.set_ylim(*limits["delta"][row])
            if row == 0:
                ax.set_title(PANEL_NAMES[0], fontsize=9.5, pad=7, fontweight="bold")
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

            ax = fig.add_subplot(gs[row, base_col + 1])
            clean_y = clean[row]
            attacked_y = tr["x_adv"][frame_idx, row]
            ax.plot(grid, clean_y, color="#111111", lw=1.30, label="clean")
            ax.plot(grid, attacked_y, color=color, lw=1.35, label="attacked")
            ax.fill_between(grid, clean_y, attacked_y, color=color, alpha=0.13, linewidth=0)
            ax.set_ylim(*limits["input"][row])
            if row == 0:
                ax.set_title(PANEL_NAMES[1], fontsize=9.5, pad=7, fontweight="bold")
                ax.legend(loc="upper right", fontsize=6.3, handlelength=1.4)
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

            ax = fig.add_subplot(gs[row, base_col + 2])
            model_y = tr["model"][frame_idx, row]
            solver_y = tr["solver"][frame_idx, row]
            ax.plot(grid, solver_y, color=SOLVER_COLOR, lw=1.45, label="solver")
            ax.plot(grid, model_y, color=color, lw=1.35, label="model")
            ax.fill_between(grid, solver_y, model_y, color=color, alpha=0.17, linewidth=0)
            ax.set_ylim(*limits["output"][row])
            if row == 0:
                ax.set_title(PANEL_NAMES[2], fontsize=9.5, pad=7, fontweight="bold")
                ax.legend(loc="upper right", fontsize=6.3, handlelength=1.4)
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

            ax = fig.add_subplot(gs[row, base_col + 3])
            diff_y = tr["diff"][frame_idx, row]
            ax.plot(grid, diff_y, color=color, lw=1.30)
            ax.fill_between(grid, 0.0, diff_y, color=color, alpha=0.17, linewidth=0)
            ax.axhline(0.0, color="#222222", lw=0.65, alpha=0.45)
            ax.set_ylim(*limits["diff"][row])
            loss = float(tr["loss"][frame_idx, row])
            rms = float(tr["delta_rms"][frame_idx, row])
            if row == 0:
                ax.set_title(PANEL_NAMES[3], fontsize=9.5, pad=7, fontweight="bold")
            ax.text(
                0.025,
                0.93,
                f"MSE={loss:.2e}\nRMS={rms:.3f}",
                transform=ax.transAxes,
                fontsize=6.1,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 1.5},
            )
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

        if first_ax_for_row is not None:
            bbox = first_ax_for_row.get_position()
            row_label_positions.append(((bbox.y0 + bbox.y1) / 2.0, sample_label))

    add_row_labels(fig, row_label_positions)

    for group_idx, model_key in enumerate(model_keys):
        ax = fig.add_subplot(gs[6, group_idx * 4 : (group_idx + 1) * 4])
        tr = models[model_key]
        steps = tr["step"]
        for i in range(clean.shape[0]):
            ax.plot(steps, tr["loss"][:, i], color=SAMPLE_COLORS[i], lw=1.25, alpha=0.82, label=f"S{i + 1}")
        ax.axvline(step, color="#111111", lw=1.05, alpha=0.60)
        ax.set_ylim(*loss_ylim)
        ax.set_xlim(0, int(models["baseline"]["step"][-1]))
        ax.set_title(f"{DISPLAY[model_key]} attack loss curves", fontsize=9.8, loc="left", fontweight="bold", pad=5)
        ax.set_xlabel("attack step", fontsize=8.5)
        ax.set_ylabel("MSE(model, solver)", fontsize=8.5)
        ax.legend(ncol=6, fontsize=6.5, loc="upper left", handlelength=1.2)
        ax.tick_params(labelsize=7, pad=1.5)
        bbox = ax.get_position()
        group_centers.append(((bbox.x0 + bbox.x1) / 2.0, DISPLAY[model_key], COLORS[model_key]))

    for x, label, color in group_centers:
        fig.text(
            x,
            0.925,
            label,
            ha="center",
            va="center",
            fontsize=11.0 if len(model_keys) > 2 else 12.0,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.30", "facecolor": "#ffffff", "edgecolor": color, "alpha": 0.94},
        )
    phase = "before perturbation" if frame_idx == 0 else "after perturbation"
    fig.suptitle(f"Burgers round03 p=2, q=2 RMS-L2 attack, step {step:03d}: {phase}", fontsize=15.0, fontweight="bold", y=0.985)
    fig.text(0.5, 0.956, "Baseline is shown once; loss1/loss2/loss3 panels use their corresponding final self-training models.", ha="center", va="center", fontsize=9.0, color=MUTED)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def render_overlay(
    out_path: Path,
    model_keys: list[str],
    clean: np.ndarray,
    manifest: list[dict[str, object]],
    models: dict[str, dict[str, np.ndarray]],
    limits: dict[str, list[tuple[float, float]]],
    loss_ylim: tuple[float, float],
    *,
    dpi: int = 160,
) -> None:
    set_style()
    before_idx = 0
    after_idx = int(len(models["baseline"]["step"]) - 1)
    after_step = int(models["baseline"]["step"][after_idx])
    grid = np.linspace(0.0, 1.0, clean.shape[1])
    fig, gs = make_grid(len(model_keys), dpi)
    row_label_positions: list[tuple[float, str]] = []
    group_centers: list[tuple[float, str, str]] = []

    for row in range(clean.shape[0]):
        item = manifest[row]
        sample_label = "\n".join([f"S{row + 1}  {item['split']}", short_dataset_label(str(item["dataset_id"])), f"index {item['index']}"])
        first_ax_for_row = None
        for group_idx, model_key in enumerate(model_keys):
            base_col = group_idx * 4
            color = COLORS[model_key]
            tr = models[model_key]

            ax = fig.add_subplot(gs[row, base_col])
            if first_ax_for_row is None:
                first_ax_for_row = ax
            ax.plot(grid, tr["delta"][before_idx, row], color=color, lw=1.05, alpha=0.80, ls="--", label="before")
            ax.plot(grid, tr["delta"][after_idx, row], color=color, lw=1.35, alpha=0.94, label="after")
            ax.axhline(0.0, color="#222222", lw=0.65, alpha=0.35)
            ax.set_ylim(*limits["delta"][row])
            if row == 0:
                ax.set_title(PANEL_NAMES[0], fontsize=9.5, pad=7, fontweight="bold")
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
                ax.set_title(PANEL_NAMES[1], fontsize=9.5, pad=7, fontweight="bold")
                ax.legend(loc="upper right", fontsize=5.9, handlelength=1.2)
            ax.set_xticks([])
            ax.tick_params(labelsize=6.5, pad=1.5)

            ax = fig.add_subplot(gs[row, base_col + 2])
            before_model = tr["model"][before_idx, row]
            before_solver = tr["solver"][before_idx, row]
            after_model = tr["model"][after_idx, row]
            after_solver = tr["solver"][after_idx, row]
            ax.plot(grid, before_solver, color=SOLVER_COLOR, lw=1.05, alpha=0.80, ls="--", label="solver before")
            ax.plot(grid, before_model, color=color, lw=1.05, alpha=0.80, ls="--", label="model before")
            ax.plot(grid, after_solver, color=SOLVER_COLOR, lw=1.45, alpha=0.92, label="solver after")
            ax.plot(grid, after_model, color=color, lw=1.35, alpha=0.94, label="model after")
            ax.fill_between(grid, before_solver, before_model, color=color, alpha=0.045, linewidth=0)
            ax.fill_between(grid, after_solver, after_model, color=color, alpha=0.135, linewidth=0)
            ax.set_ylim(*limits["output"][row])
            if row == 0:
                ax.set_title(PANEL_NAMES[2], fontsize=9.5, pad=7, fontweight="bold")
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
                ax.set_title(PANEL_NAMES[3], fontsize=9.5, pad=7, fontweight="bold")
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
            row_label_positions.append(((bbox.y0 + bbox.y1) / 2.0, sample_label))

    add_row_labels(fig, row_label_positions)

    for group_idx, model_key in enumerate(model_keys):
        ax = fig.add_subplot(gs[6, group_idx * 4 : (group_idx + 1) * 4])
        tr = models[model_key]
        steps = tr["step"]
        for i in range(clean.shape[0]):
            ax.plot(steps, tr["loss"][:, i], color=SAMPLE_COLORS[i], lw=1.25, alpha=0.82, label=f"S{i + 1}")
        ax.axvline(0, color="#111111", lw=0.95, alpha=0.42, ls="--")
        ax.axvline(after_step, color="#111111", lw=1.05, alpha=0.62)
        ax.set_ylim(*loss_ylim)
        ax.set_xlim(0, after_step)
        ax.set_title(f"{DISPLAY[model_key]} attack loss curves", fontsize=9.8, loc="left", fontweight="bold", pad=5)
        ax.set_xlabel("attack step", fontsize=8.5)
        ax.set_ylabel("MSE(model, solver)", fontsize=8.5)
        ax.legend(ncol=6, fontsize=6.5, loc="upper left", handlelength=1.2)
        ax.tick_params(labelsize=7, pad=1.5)
        bbox = ax.get_position()
        group_centers.append(((bbox.x0 + bbox.x1) / 2.0, DISPLAY[model_key], COLORS[model_key]))

    for x, label, color in group_centers:
        fig.text(
            x,
            0.925,
            label,
            ha="center",
            va="center",
            fontsize=11.0 if len(model_keys) > 2 else 12.0,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.30", "facecolor": "#ffffff", "edgecolor": color, "alpha": 0.94},
        )
    fig.suptitle("Burgers round03 p=2, q=2 RMS-L2 attack: before/after perturbation overlay", fontsize=15.0, fontweight="bold", y=0.985)
    fig.text(0.5, 0.956, "Dashed curves are before perturbation; solid/darker curves are after 100 attack steps.", ha="center", va="center", fontsize=9.0, color=MUTED)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean, manifest, models, limits, loss_ylim, final_idx = build_data()
    render_frame(
        OUT_DIR / "round03_p2q2_baseline_loss1_loss2_loss3_step000_before_perturbation_four_column.png",
        MODEL_ORDER,
        0,
        clean,
        manifest,
        models,
        limits,
        loss_ylim,
    )
    render_frame(
        OUT_DIR / "round03_p2q2_baseline_loss1_loss2_loss3_step100_after_perturbation_four_column.png",
        MODEL_ORDER,
        final_idx,
        clean,
        manifest,
        models,
        limits,
        loss_ylim,
    )
    render_overlay(
        OUT_DIR / "round03_p2q2_baseline_loss1_loss2_loss3_before_after_overlay_four_column.png",
        MODEL_ORDER,
        clean,
        manifest,
        models,
        limits,
        loss_ylim,
    )
    for loss in LOSS_ORDER:
        render_overlay(
            OUT_DIR / f"round03_p2q2_baseline_vs_{loss}_before_after_overlay_two_column.png",
            ["baseline", loss],
            clean,
            manifest,
            models,
            limits,
            loss_ylim,
        )
    outputs = sorted(str(p.relative_to(REPO)) for p in OUT_DIR.glob("round03_p2q2_*.png"))
    print(json.dumps({"output_dir": str(OUT_DIR), "generated_outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
