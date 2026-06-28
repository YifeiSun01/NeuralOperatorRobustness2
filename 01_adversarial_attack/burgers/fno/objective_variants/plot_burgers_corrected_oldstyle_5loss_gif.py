#!/usr/bin/env python3
"""Old-style GIF/PNG plotter for corrected Burgers multi-loss attacks."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import imageio.v3 as iio
except ModuleNotFoundError:
    iio = None


DEFAULT_METHODS = (
    "loss3",
    "loss3_stopgrad",
    "loss2_dict_N200",
    "loss2_dict_N2000",
    "loss2_dict_N20000",
    "loss2_fixed",
    "loss1",
)
MODEL_LABEL = "FNO"
PALETTE = {
    "black_deep": "#000000",
    "black_light": "#666666",
    "loss3": "#d35400",
    "loss3_light": "#f5b041",
    "loss3_stopgrad": "#1f77b4",
    "loss3_stopgrad_light": "#85c1e9",
    "loss2_dict": "#c0392b",
    "loss2_dict_light": "#f1948a",
    "loss2_dict_N200": "#e34a33",
    "loss2_dict_N200_light": "#fdbb84",
    "loss2_dict_N2000": "#b30000",
    "loss2_dict_N2000_light": "#fc9272",
    "loss2_dict_N20000": "#67000d",
    "loss2_dict_N20000_light": "#ef3b2c",
    "loss2_fixed": "#7b3294",
    "loss2_fixed_light": "#c2a5cf",
    "loss1": "#1a9850",
    "loss1_light": "#91cf60",
}
LABELS = {
    "loss1": "loss1",
    "loss2_fixed": "loss2 fixed",
    "loss2_dict": "loss2 dict",
    "loss2_dict_N200": "loss2 dict N=200",
    "loss2_dict_N2000": "loss2 dict N=2000",
    "loss2_dict_N20000": "loss2 dict N=20000",
    "loss3_stopgrad": "loss3 stopgrad",
    "loss3": "loss3",
}
FORMULAS = {
    "loss1": r"$L_1=\|f(x+\delta)-f(x)\|$",
    "loss2_fixed": r"$L_2$ fixed $=\|f(x+\delta)-g(x)\|$",
    "loss2_dict": r"$L_2$ dict $=\|f(x+\delta)-y_{\mathrm{nearest}}(x+\delta)\|$",
    "loss2_dict_N200": (
        r"$L_2$ dict N=200",
        r"$=\|f(x+\delta)-y_{\mathrm{nearest},200}(x+\delta)\|$",
    ),
    "loss2_dict_N2000": (
        r"$L_2$ dict N=2000",
        r"$=\|f(x+\delta)-y_{\mathrm{nearest},2000}(x+\delta)\|$",
    ),
    "loss2_dict_N20000": (
        r"$L_2$ dict N=20000",
        r"$=\|f(x+\delta)-y_{\mathrm{nearest},20000}$",
        r"$(x+\delta)\|$",
    ),
    "loss3_stopgrad": (
        r"$L_3$ stopgrad",
        r"$=\|f(x+\delta)-\mathrm{stopgrad}(g(x+\delta))\|$",
    ),
    "loss3": r"$L_3=\|f(x+\delta)-g(x+\delta)\|$",
}


def method_label(method: str) -> str:
    return LABELS.get(method, method)


def method_color(method: str) -> str:
    return PALETTE.get(method, PALETTE.get("loss2_dict" if method.startswith("loss2_dict") else "black_deep"))


def method_light_color(method: str) -> str:
    return PALETTE.get(f"{method}_light", PALETTE.get("loss2_dict_light" if method.startswith("loss2_dict") else "black_light"))


def method_formula(method: str, formulas: dict[str, str] | None = None) -> str:
    # Prefer mathtext strings here; runner formulas are plain text for CSV/JSON.
    return FORMULAS.get(method, method)


def method_formula_lines(method: str, formulas: dict[str, str] | None = None) -> tuple[str, ...]:
    formula = method_formula(method, formulas)
    if isinstance(formula, tuple):
        return formula
    return (formula,)


def methods_from_attack(attack_dict: dict[str, Any], step_update: dict[str, Any]) -> tuple[str, ...]:
    formulas = attack_dict.get("formulas", {})
    if formulas:
        ordered = [m for m in DEFAULT_METHODS if m in formulas]
        ordered.extend([m for m in formulas if m not in ordered])
        return tuple(ordered)
    for sk in sorted_steps(step_update):
        loss_metrics = step_update[sk].get("loss_metrics", {})
        if loss_metrics:
            ordered = [m for m in DEFAULT_METHODS if m in loss_metrics]
            ordered.extend([m for m in loss_metrics if m not in ordered])
            return tuple(ordered)
    return DEFAULT_METHODS


def sorted_steps(step_update: dict[str, Any]) -> list[str]:
    def key(name: str) -> int:
        try:
            return int(name.split("_")[-1])
        except Exception:
            return 10**9

    return sorted([k for k in step_update if k.startswith("step_")], key=key)


def arr(values: dict[str, Any], key: str) -> np.ndarray | None:
    value = values.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(-1)


def finite_minmax(items: list[np.ndarray]) -> tuple[float, float]:
    vals = [x[np.isfinite(x)] for x in items if x is not None and np.asarray(x).size]
    vals = [x for x in vals if x.size]
    if not vals:
        return -1.0, 1.0
    cat = np.concatenate(vals)
    lo = float(np.min(cat))
    hi = float(np.max(cat))
    if lo == hi:
        pad = max(1.0, abs(lo) + 1.0) * 0.05
        return lo - pad, hi + pad
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def collect_limits(step_update: dict[str, Any], methods: tuple[str, ...]):
    input_arrays: list[np.ndarray] = []
    output_arrays: list[np.ndarray] = []
    opt_losses = {method: [] for method in methods}
    true_losses = {method: [] for method in methods}
    step_indices: list[int] = []

    for sk in sorted_steps(step_update):
        idx = int(sk.split("_")[-1])
        step_indices.append(idx)
        sdict = step_update[sk]
        values = sdict.get("values", {})
        a_values = values.get("a_values", {})
        g_values = values.get("g_values", {})
        G_values = values.get("G_values", {})
        input_arrays.extend([v for v in [arr(a_values, "a")] if v is not None])
        output_arrays.extend([v for v in [arr(G_values, "G(a)"), arr(g_values, "g(a)")] if v is not None])
        for method in methods:
            for source, key in [(a_values, f"a_{method}")]:
                value = arr(source, key)
                if value is not None:
                    input_arrays.append(value)
            for source, key in [(G_values, f"G(a_{method})"), (g_values, f"g(a_{method})")]:
                value = arr(source, key)
                if value is not None:
                    output_arrays.append(value)
            opt_losses[method].append(float(sdict.get("loss_metrics", {}).get(method, np.nan)))
            true_losses[method].append(float(sdict.get("true_loss_metrics", {}).get(method, np.nan)))

    input_ylim = finite_minmax(input_arrays)
    output_ylim = finite_minmax(output_arrays)
    opt_loss_ylim = finite_minmax([np.asarray(v, dtype=float) for v in opt_losses.values()])
    true_loss_ylim = finite_minmax([np.asarray(v, dtype=float) for v in true_losses.values()])
    return input_ylim, output_ylim, true_loss_ylim, opt_loss_ylim, np.asarray(step_indices, dtype=int), opt_losses, true_losses


def load_loss_history_curves(path: Path, methods: tuple[str, ...]):
    if not path.exists():
        return None
    rows = {method: [] for method in methods}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row.get("method", "")
            if method not in rows:
                continue
            try:
                rows[method].append(
                    (
                        int(float(row["k"])),
                        float(row["optimized_loss"]),
                        float(row["true_loss"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    if not any(rows.values()):
        return None

    step_indices = sorted({k for method_rows in rows.values() for k, _, _ in method_rows})
    opt_losses = {method: [np.nan] * len(step_indices) for method in methods}
    true_losses = {method: [np.nan] * len(step_indices) for method in methods}
    pos = {k: i for i, k in enumerate(step_indices)}
    for method, method_rows in rows.items():
        for k, opt, true in method_rows:
            i = pos[k]
            opt_losses[method][i] = opt
            true_losses[method][i] = true

    true_loss_ylim = finite_minmax([np.asarray(v, dtype=float) for v in true_losses.values()])
    opt_loss_ylim = finite_minmax([np.asarray(v, dtype=float) for v in opt_losses.values()])
    return np.asarray(step_indices, dtype=int), opt_losses, true_losses, true_loss_ylim, opt_loss_ylim


def plot_inputs(ax, xgrid, a_values, input_ylim, methods: tuple[str, ...]):
    ax.clear()
    clean = arr(a_values, "a")
    if clean is not None:
        ax.plot(xgrid, clean, label="clean input", color=PALETTE["black_deep"], linewidth=1.7)
    for method in methods:
        value = arr(a_values, f"a_{method}")
        if value is not None:
            ax.plot(
                xgrid,
                value,
                label=f"perturbed input: {method_label(method)}",
                color=method_color(method),
                linewidth=1.1,
                alpha=0.4,
            )
    ax.set_title("Clean Input / Perturbed Inputs")
    ax.set_ylim(input_ylim)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=7, loc="upper right")


def fmt_loss(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1e3 or (abs(value) > 0 and abs(value) < 1e-2):
        return f"{value:.2e}"
    return f"{value:.3g}"


def plot_method_panel(ax, xgrid, method, G_values, g_values, output_ylim, current_true_loss: float, current_opt_loss: float):
    ax.clear()
    base_G = arr(G_values, "G(a)")
    base_g = arr(g_values, "g(a)")
    if base_G is not None:
        ax.plot(xgrid, base_G, label=f"{MODEL_LABEL} output clean", color=PALETTE["black_deep"], linewidth=1.35)
    if base_g is not None:
        ax.plot(xgrid, base_g, label="solver output clean", color=PALETTE["black_light"], linewidth=1.2)
    G_adv = arr(G_values, f"G(a_{method})")
    g_adv = arr(g_values, f"g(a_{method})")
    if G_adv is not None and g_adv is not None:
        ax.fill_between(
            xgrid,
            G_adv,
            g_adv,
            color=method_color(method),
            alpha=0.13,
            linewidth=0.0,
            label=f"|{MODEL_LABEL} - solver| area",
        )
    if G_adv is not None:
        ax.plot(xgrid, G_adv, label=f"{MODEL_LABEL} output perturbed", color=method_color(method), linewidth=1.55)
    if g_adv is not None:
        ax.plot(xgrid, g_adv, label="solver output perturbed", color=method_light_color(method), linewidth=1.35)
    ax.set_title(
        f"{method_label(method)}: {MODEL_LABEL} / solver\n"
        f"(true={fmt_loss(current_true_loss)}, opt={fmt_loss(current_opt_loss)})"
    )
    ax.set_ylim(output_ylim)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=7, loc="upper right")


def plot_loss_panel(ax, step_indices, curves, current_step, loss_ylim, title, methods: tuple[str, ...]):
    ax.clear()
    end = np.searchsorted(step_indices, current_step, side="right")
    x_visible = step_indices[:end]
    for method in methods:
        y = np.asarray(curves[method], dtype=float)
        y_visible = y[:end]
        ax.plot(x_visible, y_visible, label=method_label(method), color=method_color(method), linewidth=1.55)
        finite = np.where(np.isfinite(y_visible))[0]
        if finite.size:
            last = finite[-1]
            y_span = loss_ylim[1] - loss_ylim[0]
            ax.text(
                x_visible[last],
                y_visible[last] - 0.018 * y_span,
                method_label(method),
                color=method_color(method),
                fontsize=6.8,
                va="top",
                ha="right",
                clip_on=True,
            )
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("squared L2 loss")
    ax.set_ylim(loss_ylim)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=7, loc="upper left")


def plot_formula_panel(
    ax,
    methods: tuple[str, ...],
    formulas: dict[str, str] | None,
    title: str = "Optimized losses",
    show_notes: bool = False,
):
    ax.clear()
    ax.axis("off")
    y = 0.97
    if title:
        ax.text(0.0, y, title, fontsize=21.8, weight="bold", va="top")
        y -= 0.14
    for method in methods:
        lines = method_formula_lines(method, formulas)
        for line in lines:
            ax.text(0.0, y, line, fontsize=21.0, color=method_color(method), va="top")
            y -= 0.115
        y -= 0.015
    if show_notes:
        ax.text(0.0, 0.14, r"$L_1$ uses random start for $\delta$.", fontsize=18.0, color=PALETTE["loss1"], va="bottom")
        ax.text(
            0.0,
            0.02,
            r"True-loss panel uses $L_3=\|f(x+\delta)-g(x+\delta)\|$ for every method.",
            fontsize=18.0,
            color="0.25",
            va="bottom",
        )


def infer_xgrid(sdict: dict[str, Any], methods: tuple[str, ...]) -> np.ndarray:
    values = sdict.get("values", {})
    a_values = values.get("a_values", {})
    for key in ["a", *[f"a_{method}" for method in methods]]:
        value = arr(a_values, key)
        if value is not None:
            return np.arange(value.size)
    return np.arange(1024)


def render_frame(
    step_update,
    sk,
    limits,
    attack_label: str,
    title_text: str,
    methods: tuple[str, ...],
    formulas: dict[str, str] | None,
    frame_path: Path | None = None,
):
    input_ylim, output_ylim, true_loss_ylim, opt_loss_ylim, step_indices, opt_losses, true_losses = limits
    idx = int(sk.split("_")[-1])
    sdict = step_update[sk]
    values = sdict.get("values", {})
    a_values = values.get("a_values", {})
    G_values = values.get("G_values", {})
    g_values = values.get("g_values", {})
    loss_metrics = sdict.get("loss_metrics", {})
    true_loss_metrics = sdict.get("true_loss_metrics", {})
    xgrid = infer_xgrid(sdict, methods)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.30, wspace=0.18)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(4)]

    plot_inputs(axes[0], xgrid, a_values, input_ylim, methods)
    for ax, method in zip(axes[1:1 + len(methods)], methods):
        plot_method_panel(
            ax,
            xgrid,
            method,
            G_values,
            g_values,
            output_ylim,
            float(true_loss_metrics.get(method, np.nan)),
            float(loss_metrics.get(method, np.nan)),
        )
    cursor = 1 + len(methods)
    plot_loss_panel(axes[cursor], step_indices, true_losses, idx, true_loss_ylim, f"True Loss L3 Curve (to step {idx})", methods)
    cursor += 1
    plot_loss_panel(axes[cursor], step_indices, opt_losses, idx, opt_loss_ylim, f"Optimized Objective Loss Curve (to step {idx})", methods)
    cursor += 1
    split = (len(methods) + 1) // 2
    plot_formula_panel(axes[cursor], methods[:split], formulas, title="Optimized losses", show_notes=False)
    cursor += 1
    plot_formula_panel(axes[cursor], methods[split:], formulas, title="", show_notes=True)
    for ax in axes[cursor + 1:]:
        ax.axis("off")

    fig.suptitle(f"{title_text} | step={idx}", fontsize=14)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.04, top=0.935)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    if frame_path is not None:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(frame_path, dpi=150)
    plt.close(fig)
    return frame


def safe_name(value: Any) -> str:
    text = "__".join(value) if isinstance(value, (tuple, list)) else str(value)
    return "".join(c if c.isalnum() or c in "._-+=" else "_" for c in text)


def title_from_attack_key(attack_key: Any) -> str:
    parts = list(attack_key) if isinstance(attack_key, (tuple, list)) else str(attack_key).split("__")
    values: dict[str, str] = {}
    prefixes = [
        ("nu_", "nu"),
        ("domain_", "domain"),
        ("norm_", "norm"),
        ("epsilon_", "epsilon"),
        ("alpha_", "alpha"),
        ("numsteps_", "steps"),
        ("index_", "index"),
        ("tfinal_", "t_final"),
        ("dt_", "dt"),
    ]
    for part in parts:
        text = str(part)
        for prefix, key in prefixes:
            if text.startswith(prefix):
                values[key] = text[len(prefix):]
    ordered = ["nu", "domain", "norm", "epsilon", "alpha", "steps", "index", "t_final", "dt"]
    param_text = ", ".join(f"{key}={values[key]}" for key in ordered if key in values)
    title = "Burgers PGD attack with different losses"
    return f"{title} | {param_text}" if param_text else title


def make_outputs(pickle_path: Path, outdir: Path, fps: int, stride: int, cleanup_frames: bool, final_png_only: bool) -> None:
    with pickle_path.open("rb") as f:
        data = pickle.load(f)
    outdir.mkdir(parents=True, exist_ok=True)

    for attack_key, attack_dict in data.items():
        label = safe_name(attack_key)
        title_text = title_from_attack_key(attack_key)
        step_update = attack_dict.get("step_update", {})
        step_keys = sorted_steps(step_update)
        if not step_keys:
            print(f"[warn] no steps in {attack_key}; skipping")
            continue
        selected = step_keys[:: max(1, stride)]
        if selected[-1] != step_keys[-1]:
            selected.append(step_keys[-1])
        methods = methods_from_attack(attack_dict, step_update)
        formulas = attack_dict.get("formulas", {})
        limits = collect_limits(step_update, methods)
        history_curves = load_loss_history_curves(pickle_path.parent / "loss_history.csv", methods)
        if history_curves is not None:
            hist_step_indices, hist_opt_losses, hist_true_losses, hist_true_ylim, hist_opt_ylim = history_curves
            limits = (
                limits[0],
                limits[1],
                hist_true_ylim,
                hist_opt_ylim,
                hist_step_indices,
                hist_opt_losses,
                hist_true_losses,
            )

        png_path = outdir / f"{label}_final.png"
        if final_png_only:
            render_frame(step_update, step_keys[-1], limits, label, title_text, methods, formulas, png_path)
            print(f"[ok] saved {png_path}")
            continue

        if iio is None:
            raise RuntimeError("imageio is required for GIF output; use --final_png_only for PNG-only rendering.")

        frame_dir = outdir / f".{label}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for i, sk in enumerate(tqdm(selected, desc=f"Rendering {label}")):
            frame_path = frame_dir / f"frame_{i:04d}_{sk}.png"
            frames.append(render_frame(step_update, sk, limits, label, title_text, methods, formulas, frame_path))

        gif_path = outdir / f"{label}.gif"
        iio.imwrite(gif_path, frames, fps=fps)
        render_frame(step_update, step_keys[-1], limits, label, title_text, methods, formulas, png_path)
        if cleanup_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)
        print(f"[ok] saved {gif_path}")
        print(f"[ok] saved {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", type=Path, required=True, help="Runner pickle output.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for GIF/final PNG.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth step in the GIF, always keeping the final step.")
    parser.add_argument("--cleanup_frames", action="store_true")
    parser.add_argument("--final_png_only", action="store_true", help="Only render the final PNG frame; do not write GIF or intermediate frames.")
    args = parser.parse_args()
    make_outputs(args.pickle, args.outdir, args.fps, args.stride, args.cleanup_frames, args.final_png_only)


if __name__ == "__main__":
    main()
