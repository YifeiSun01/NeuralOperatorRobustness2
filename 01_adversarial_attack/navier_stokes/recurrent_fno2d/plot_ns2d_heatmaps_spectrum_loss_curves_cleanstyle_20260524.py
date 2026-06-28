#!/usr/bin/env python3
"""Clean-style NS2D heatmaps with selected/all-sample delta spectra and loss curves.

Offline visualization only: reads saved final_state_outputs.npz and metrics CSVs.
No model inference, solver rollout, attack update, PyTorch, JAX, or GPU work is run.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
BASE_SCRIPT = ROOT / "tools" / "plot_ns2d_combined_target_mode_heatmaps_base_20260524.py"
spec = importlib.util.spec_from_file_location("ns2d_current_base_readable", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

# Bump text sizes for the readable redraw. The copied base helper keeps older
# compact fonts, so override them here without touching the original base file.
base.FONT_TITLE = base.font(base.FONT_BOLD_PATH, 32)
base.FONT_SUBTITLE = base.font(base.FONT_REGULAR_PATH, 18)
base.FONT_COL = base.font(base.FONT_BOLD_PATH, 16)
base.FONT_ROW = base.font(base.FONT_BOLD_PATH, 22)
base.FONT_SMALL = base.font(base.FONT_REGULAR_PATH, 17)
base.FONT_TINY = base.font(base.FONT_REGULAR_PATH, 11)

EXTRA_EPS1_ROOT = ROOT / "2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/eps1_missing_loss2_loss3_b10_20260524_024533_UTC"
EXTRA_EPS160_ROOT = ROOT / "2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack/eps1_missing_loss2_loss3_b10_20260524_023355_UTC"
_extra_roots = [p for p in [EXTRA_EPS1_ROOT, EXTRA_EPS160_ROOT] if p.exists()]
if _extra_roots:
    base.RUN_ROOTS = list(base.RUN_ROOTS) + _extra_roots
if "eps160_alpha50" not in base.EPS_ORDER:
    base.EPS_ORDER = ["eps160_alpha50"] + list(base.EPS_ORDER)

OUT_DIR = ROOT / "docs/ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524"
REPORT_PATH = ROOT / "docs/ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524.md"
SAMPLE_POSITION = base.SAMPLE_POSITION
SPECTRUM_XMAX = 0.25
SPECTRUM_FLOOR = 1e-5
LOSS_CURVE_XMAX = 100.0
N_BINS = 128
EPS = 1e-30

EPS_PRETTY = {
    "eps160_alpha50": ("160", "50"),
    "eps32_alpha10": ("32", "10"),
    "eps16_alpha5": ("16", "5"),
    "eps8_alpha2p5": ("8", "2.5"),
    "eps4_alpha1p25": ("4", "1.25"),
    "eps2_alpha0p625": ("2", "0.625"),
    "eps1_alpha0p3125": ("1", "0.3125"),
}
METHOD_PRETTY = {
    "raw_add": "Raw Add",
    "raw_replace": "Raw Replace",
    "steepest_add": "Steepest Add",
    "steepest_replace": "Steepest Replace",
}
CANDIDATE_PRETTY = {
    "loss1/all_w": "Loss 1 / all W",
    "loss2/all_a_target_w": "Loss 2 / all A -> W",
    "loss3/all_w": "Loss 3 / all W",
    "loss3/all_d_target_w": "Loss 3 / all D -> W",
    "loss3/all_a_target_w": "Loss 3 / all A -> W",
    "loss3/w1_5_d6_9_target_w": "Loss 3 / W steps 1-5, D steps 6-9 -> W",
    "loss3/d1_5_w6_9_target_w": "Loss 3 / D steps 1-5, W steps 6-9 -> W",
    "loss3/a1_5_d6_9_target_w": "Loss 3 / A steps 1-5, D steps 6-9 -> W",
}
SHORT_LABEL = {
    "loss1/all_w": "L1 all W",
    "loss2/all_a_target_w": "L2 all A->W",
    "loss3/all_w": "L3 all W",
    "loss3/all_d_target_w": "L3 all D->W",
    "loss3/all_a_target_w": "L3 all A->W",
    "loss3/w1_5_d6_9_target_w": "L3 W1-5 D6-9",
    "loss3/d1_5_w6_9_target_w": "L3 D1-5 W6-9",
    "loss3/a1_5_d6_9_target_w": "L3 A1-5 D6-9",
}
LINE_COLORS = {
    "loss1/all_w": "#3267b1",
    "loss2/all_a_target_w": "#d28b26",
    "loss3/all_w": "#c8353e",
    "loss3/all_d_target_w": "#4c9a57",
    "loss3/all_a_target_w": "#7c5ab8",
    "loss3/w1_5_d6_9_target_w": "#2b8cbe",
    "loss3/d1_5_w6_9_target_w": "#d95f02",
    "loss3/a1_5_d6_9_target_w": "#7570b3",
}
COLUMN_SPECS = [
    ("input", "Initial / Perturbed Input", "sequential"),
    ("delta", "Perturbation Delta", "diverging"),
    ("model", "Model Output", "sequential"),
    ("solver", "Solver Output", "sequential"),
    ("diff", "Model - Solver", "diverging"),
]

@dataclass
class RenderSummary:
    eps_label: str
    method: str
    dataset_index: int
    n_attack_rows: int
    output_png: Path
    clean_loss: float
    max_candidate_label: str
    max_adv_true_loss: float
    rows: list[dict[str, Any]]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def report_rel(path: Path) -> str:
    return path.relative_to(REPORT_PATH.parent).as_posix()


def eps_text(eps_label: str) -> str:
    eps, alpha = EPS_PRETTY.get(eps_label, (eps_label, "unknown"))
    return f"Epsilon = {eps}, Alpha = {alpha}"


def method_text(method: str) -> str:
    return METHOD_PRETTY.get(method, method.replace("_", " ").title())


def label_text(label: str) -> str:
    return CANDIDATE_PRETTY.get(label, label.replace("_", " ").replace("/", " / "))


def short_label(label: str) -> str:
    return SHORT_LABEL.get(label, label.replace("loss", "L").replace("_", " "))


def loss_change_phrase(change: float) -> str:
    if not math.isfinite(change):
        return "change unavailable"
    if change > 0:
        return f"increase {base.fmt(change)}"
    if change < 0:
        return f"decrease {base.fmt(abs(change))}"
    return "no change"


def draw_lines(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], fonts: list[Any], fills: list[tuple[int, int, int]], gap: int = 4) -> None:
    x, y = xy
    for i, line in enumerate(lines):
        if not line:
            continue
        font = fonts[min(i, len(fonts) - 1)]
        fill = fills[min(i, len(fills) - 1)]
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += bbox[3] - bbox[1] + gap


def radial_grid_2d(n0: int, n1: int) -> np.ndarray:
    k0 = np.fft.fftfreq(n0) * n0
    k1 = np.fft.fftfreq(n1) * n1
    yy, xx = np.meshgrid(k0, k1, indexing="ij")
    r = np.sqrt(yy * yy + xx * xx)
    return r / (math.sqrt((n0 / 2.0) ** 2 + (n1 / 2.0) ** 2) + EPS)


def normalized_radial_amplitude(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = base.finite(delta).squeeze()
    if x.ndim != 2:
        raise ValueError(f"Expected 2D delta, got {x.shape}")
    x = x - float(x.mean())
    r = radial_grid_2d(x.shape[0], x.shape[1]).ravel()
    amp = np.abs(np.fft.fft2(x)).ravel()
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    amp_sum = np.zeros(N_BINS, dtype=np.float64)
    counts = np.zeros(N_BINS, dtype=np.float64)
    idx = np.clip(np.searchsorted(bins, r, side="right") - 1, 0, N_BINS - 1)
    np.add.at(amp_sum, idx, amp)
    np.add.at(counts, idx, 1.0)
    mean_amp = amp_sum / np.maximum(counts, 1.0)
    area = float(np.trapezoid(mean_amp, centers))
    if area > 0:
        mean_amp = mean_amp / area
    else:
        mean_amp = mean_amp / (float(np.max(mean_amp)) + EPS)
    return centers, mean_amp


def spectrum_stack(deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = []
    freq = None
    for delta in deltas:
        freq, amp = normalized_radial_amplitude(delta)
        curves.append(amp)
    stack = np.stack(curves, axis=0)
    std = np.std(stack, axis=0, ddof=1 if stack.shape[0] > 1 else 0)
    return freq, np.mean(stack, axis=0), std


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def curve_for_case(case: dict[str, Any], selected: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    method_dir = case["candidate"].path.parent
    if selected:
        rows = [r for r in read_csv_rows(method_dir / "per_sample_step_metrics.csv") if int(float(r.get("sample_position", -1))) == SAMPLE_POSITION]
        x = np.asarray([float(r["k"]) for r in rows], dtype=np.float64)
        y = np.asarray([float(r.get("true_loss", "nan")) for r in rows], dtype=np.float64)
        return x, y, None
    rows = read_csv_rows(method_dir / "per_step_metrics.csv")
    x = np.asarray([float(r["k"]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r.get("true_loss_mean", "nan")) for r in rows], dtype=np.float64)
    std = np.asarray([float(r.get("true_loss_std", "nan")) for r in rows], dtype=np.float64)
    return x, y, std


def line_limits_from_curves(curves: list[tuple[np.ndarray, np.ndarray]], *, positive: bool = False) -> tuple[float, float]:
    vals = []
    for x, y in curves:
        mask = np.isfinite(x) & np.isfinite(y)
        if positive:
            mask &= y > 0
        if mask.any():
            vals.append(y[mask])
    if not vals:
        return (1e-6, 1.0) if positive else (0.0, 1.0)
    allv = np.concatenate(vals)
    lo = float(allv.min())
    hi = float(allv.max())
    if positive:
        lo = max(lo, 1e-12)
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0) * 0.1
    lo = max(lo - 0.08 * span, 1e-12 if positive else -math.inf)
    hi = hi + 0.12 * span
    return lo, hi


def chart_line_limits(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    vals = []
    for x, y in curves:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            vals.append(y[mask])
    if not vals:
        return 0.0, 1.0
    allv = np.concatenate(vals)
    lo = float(allv.min())
    hi = float(allv.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0) * 0.1
    return lo - 0.08 * span, hi + 0.12 * span


def chart_map_points(x: np.ndarray, y: np.ndarray, box: tuple[int, int, int, int], xlim: tuple[float, float], ylim: tuple[float, float]) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = box
    xx = np.clip((x - xlim[0]) / (xlim[1] - xlim[0] + EPS), 0.0, 1.0)
    yy = np.clip((y - ylim[0]) / (ylim[1] - ylim[0] + EPS), 0.0, 1.0)
    return [(int(round(x0 + a * (x1 - x0))), int(round(y1 - b * (y1 - y0)))) for a, b in zip(xx, yy)]


def draw_chart_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: list[dict[str, Any]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    legend: bool = True,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1, y1), radius=6, fill=(255, 255, 253), outline=(184, 193, 204), width=1)
    draw.text((x0 + 12, y0 + 8), title, font=base.FONT_TINY, fill=(30, 34, 40))
    plot = (x0 + 58, y0 + 34, x1 - 16, y1 - 35)
    px0, py0, px1, py1 = plot
    for frac in (0.25, 0.5, 0.75):
        gx = int(px0 + (px1 - px0) * frac)
        gy = int(py0 + (py1 - py0) * frac)
        draw.line((gx, py0, gx, py1), fill=(226, 228, 226), width=1)
        draw.line((px0, gy, px1, gy), fill=(226, 228, 226), width=1)
    draw.rectangle(plot, outline=(67, 72, 80), width=1)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    for item in series:
        if "lower" not in item or "upper" not in item:
            continue
        x = np.asarray(item["x"], dtype=np.float64)
        lower = np.asarray(item["lower"], dtype=np.float64)
        upper = np.asarray(item["upper"], dtype=np.float64)
        pts = chart_map_points(x, upper, plot, xlim, ylim) + chart_map_points(x[::-1], lower[::-1], plot, xlim, ylim)
        color = item["color"]
        odraw.polygon(pts, fill=(*color, 34))
    canvas.alpha_composite(overlay) if canvas.mode == "RGBA" else canvas.paste(Image.alpha_composite(Image.new("RGBA", canvas.size, (0,0,0,0)), overlay).convert("RGB"), (0, 0), overlay)
    for item in series:
        x = np.asarray(item["x"], dtype=np.float64)
        y = np.asarray(item["y"], dtype=np.float64)
        pts = chart_map_points(x, y, plot, xlim, ylim)
        if len(pts) > 1:
            draw.line(pts, fill=item["color"], width=item.get("width", 2), joint="curve")
    draw.text((px0, y1 - 23), xlabel, font=base.FONT_TINY, fill=(76, 82, 91))
    draw.text((x0 + 10, y0 + 32), ylabel, font=base.FONT_TINY, fill=(76, 82, 91))
    ticks = [xlim[0], (xlim[0] + xlim[1]) / 2, xlim[1]]
    for value, xpos in zip(ticks, [px0, (px0 + px1) // 2, px1]):
        label = base.fmt(value, 3)
        bbox = draw.textbbox((0,0), label, font=base.FONT_TINY)
        tx = xpos - (bbox[2] - bbox[0]) // 2
        tx = max(px0, min(tx, px1 - (bbox[2] - bbox[0])))
        draw.text((tx, py1 + 4), label, font=base.FONT_TINY, fill=(76,82,91))
    for value, ypos in [(ylim[1], py0), (ylim[0], py1 - 10)]:
        draw.text((x0 + 9, ypos), base.fmt(value, 3), font=base.FONT_TINY, fill=(76,82,91))
    if legend:
        lx = x1 - 126
        ly = y0 + 8
        for item in series:
            label = item.get("label", "")
            if not label:
                continue
            draw.line((lx, ly + 7, lx + 20, ly + 7), fill=item["color"], width=2)
            draw.text((lx + 25, ly), label, font=base.FONT_TINY, fill=(22,26,31))
            ly += 14

def plot_top_panels(cases: list[dict[str, Any]], width: int, height: int, max_label: str) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    gap_x = 24
    gap_y = 22
    panel_w = (width - gap_x) // 2
    panel_h = (height - gap_y) // 2
    boxes = [
        (0, 0, panel_w, panel_h),
        (panel_w + gap_x, 0, width, panel_h),
        (0, panel_h + gap_y, panel_w, height),
        (panel_w + gap_x, panel_h + gap_y, width, height),
    ]

    selected_spec_lines = []
    mean_spec_lines = []
    selected_loss_lines = []
    mean_loss_lines = []
    prepared = []
    for case in cases:
        label = case["candidate"].label
        hex_color = LINE_COLORS.get(label, "#333333").lstrip("#")
        color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        is_loss2 = label == "loss2/all_a_target_w"
        is_loss3_aw = label == "loss3/all_w"
        spec_width = 5 if (is_loss2 or is_loss3_aw) else 3
        loss_width = 6 if is_loss3_aw else 4
        data = case["data"]
        f_sel, a_sel = normalized_radial_amplitude(data["final_delta"][SAMPLE_POSITION])
        f_mean, a_mean, a_std = spectrum_stack(data["final_delta"])
        k_sel, l_sel, _ = curve_for_case(case, True)
        k_mean, l_mean, l_std = curve_for_case(case, False)
        mask_sel = f_sel <= SPECTRUM_XMAX
        mask_mean = f_mean <= SPECTRUM_XMAX
        k_sel_mask = k_sel <= LOSS_CURVE_XMAX
        k_mean_mask = k_mean <= LOSS_CURVE_XMAX
        spec_sel_y = np.log10(np.maximum(a_sel[mask_sel], SPECTRUM_FLOOR))
        spec_mean_y = np.log10(np.maximum(a_mean[mask_mean], SPECTRUM_FLOOR))
        spec_mean_lo = np.log10(np.maximum(a_mean[mask_mean] - a_std[mask_mean], SPECTRUM_FLOOR))
        spec_mean_hi = np.log10(np.maximum(a_mean[mask_mean] + a_std[mask_mean], SPECTRUM_FLOOR))
        prepared.append({
            "label": short_label(label),
            "color": color,
            "spec_width": spec_width,
            "loss_width": loss_width,
            "f_sel": f_sel[mask_sel],
            "spec_sel_y": spec_sel_y,
            "f_mean": f_mean[mask_mean],
            "spec_mean_y": spec_mean_y,
            "spec_mean_lo": spec_mean_lo,
            "spec_mean_hi": spec_mean_hi,
            "k_sel": k_sel[k_sel_mask],
            "l_sel": l_sel[k_sel_mask],
            "k_mean": k_mean[k_mean_mask],
            "l_mean": l_mean[k_mean_mask],
            "l_std": l_std[k_mean_mask] if l_std is not None and len(l_std) == len(k_mean) else None,
        })
        selected_spec_lines.append((f_sel[mask_sel], spec_sel_y))
        mean_spec_lines.append((f_mean[mask_mean], spec_mean_y))
        selected_loss_lines.append((k_sel[k_sel_mask], l_sel[k_sel_mask]))
        mean_loss_lines.append((k_mean[k_mean_mask], l_mean[k_mean_mask]))

    y_sel_spec = chart_line_limits(selected_spec_lines)
    y_mean_spec = chart_line_limits(mean_spec_lines)
    y_sel_spec = (max(y_sel_spec[0], math.log10(SPECTRUM_FLOOR)), y_sel_spec[1])
    y_mean_spec = (max(y_mean_spec[0], math.log10(SPECTRUM_FLOOR)), y_mean_spec[1])
    y_sel_loss = chart_line_limits(selected_loss_lines)
    y_mean_loss = chart_line_limits(mean_loss_lines)

    draw_chart_panel(
        canvas, draw, boxes[0], "Single index delta spectrum",
        [{"x": p["f_sel"], "y": p["spec_sel_y"], "color": p["color"], "width": p["spec_width"], "label": p["label"]} for p in prepared],
        (0.0, SPECTRUM_XMAX), y_sel_spec, "freq |k| (0-0.25)", "log10 amp", True,
    )
    draw_chart_panel(
        canvas, draw, boxes[1], "All samples mean spectrum",
        [{"x": p["f_mean"], "y": p["spec_mean_y"], "lower": p["spec_mean_lo"], "upper": p["spec_mean_hi"], "color": p["color"], "width": p["spec_width"], "label": p["label"]} for p in prepared],
        (0.0, SPECTRUM_XMAX), y_mean_spec, "freq |k| (0-0.25)", "log10 amp", True,
    )
    draw_chart_panel(
        canvas, draw, boxes[2], "Single index true loss3 curve",
        [{"x": p["k_sel"], "y": p["l_sel"], "color": p["color"], "width": p["loss_width"], "label": p["label"]} for p in prepared],
        (0.0, LOSS_CURVE_XMAX), y_sel_loss, "attack step", "loss3", True,
    )
    mean_series = []
    for p in prepared:
        item = {"x": p["k_mean"], "y": p["l_mean"], "color": p["color"], "width": p["loss_width"], "label": p["label"]}
        if p["l_std"] is not None:
            item["lower"] = p["l_mean"] - p["l_std"]
            item["upper"] = p["l_mean"] + p["l_std"]
        mean_series.append(item)
    draw_chart_panel(
        canvas, draw, boxes[3], "All samples mean true loss3 curve",
        mean_series,
        (0.0, LOSS_CURVE_XMAX), y_mean_loss, "attack step", "loss3", True,
    )
    return canvas


def render_group(eps_label: str, method: str, candidates: list[Any]) -> RenderSummary:
    cases = [base.load_case(c) for c in sorted(candidates, key=base.candidate_sort_key)]
    first = cases[0]
    data0 = first["data"]
    dataset_index = first["dataset_index"]
    x_clean = base.finite(data0["x_clean"][SAMPLE_POSITION])
    clean_model = base.finite(data0["clean_model_final"][SAMPLE_POSITION])
    clean_solver = base.finite(data0["clean_solver_final"][SAMPLE_POSITION])
    clean_diff = base.finite(data0["clean_model_minus_solver"][SAMPLE_POSITION])
    clean_loss = first["clean_loss"]
    max_case = max(cases, key=lambda item: item["adv_loss"] if math.isfinite(item["adv_loss"]) else -math.inf)
    max_internal_label = max_case["candidate"].label
    max_pretty_label = label_text(max_internal_label)
    max_adv_true_loss = max_case["adv_loss"]

    rows: list[dict[str, Any]] = [{
        "kind": "clean",
        "label": "Clean baseline",
        "sub": f"Final true loss = {base.fmt(clean_loss)}",
        "sub2": "Before perturbation",
        "input": x_clean,
        "delta": np.zeros_like(x_clean),
        "model": clean_model,
        "solver": clean_solver,
        "diff": clean_diff,
        "is_max": False,
        "summary": {
            "row_type": "clean",
            "candidate_label": "Clean baseline",
            "clean_true_loss": clean_loss,
            "adv_true_loss": clean_loss,
            "true_loss_increase": 0.0,
            "true_loss_ratio": 1.0,
            "delta_l2": 0.0,
            "delta_linf": 0.0,
            "source_npz": rel(first["candidate"].path),
        },
    }]

    for case in cases:
        c = case["candidate"]
        d = case["data"]
        change = case["adv_loss"] - case["clean_loss"]
        is_max = c.label == max_internal_label
        rows.append({
            "kind": "attack",
            "label": label_text(c.label),
            "sub": f"Clean {base.fmt(case['clean_loss'])} -> Attack {base.fmt(case['adv_loss'])}; {loss_change_phrase(change)}",
            "sub2": f"Delta L2 = {base.fmt(case['delta_l2'])}; Delta Linf = {base.fmt(case['delta_linf'])}",
            "input": base.finite(d["x_adv"][SAMPLE_POSITION]),
            "delta": base.finite(d["final_delta"][SAMPLE_POSITION]),
            "model": base.finite(d["adv_model_final"][SAMPLE_POSITION]),
            "solver": base.finite(d["adv_solver_final"][SAMPLE_POSITION]),
            "diff": base.finite(d["adv_model_minus_solver"][SAMPLE_POSITION]),
            "is_max": is_max,
            "summary": {
                "row_type": "attack",
                "candidate_label": label_text(c.label),
                "internal_candidate_label": c.label,
                "loss_type": c.loss_type,
                "mode_spec": c.mode_spec,
                "method": c.method,
                "clean_true_loss": case["clean_loss"],
                "adv_true_loss": case["adv_loss"],
                "true_loss_increase": change,
                "true_loss_ratio": case["ratio"],
                "delta_l2": case["delta_l2"],
                "delta_linf": case["delta_linf"],
                "is_max_adv_true_loss": is_max,
                "source_npz": rel(c.path),
            },
        })

    input_lim = base.limits_sequential([r["input"] for r in rows])
    delta_lim = base.limits_diverging([r["delta"] for r in rows])
    state_lim = base.limits_sequential([r["model"] for r in rows] + [r["solver"] for r in rows])
    diff_lim = base.limits_diverging([r["diff"] for r in rows])
    lims = {
        "input": (*input_lim, "sequential"),
        "delta": (*delta_lim, "diverging"),
        "model": (*state_lim, "sequential"),
        "solver": (*state_lim, "sequential"),
        "diff": (*diff_lim, "diverging"),
    }

    row_label_w = 520
    img = 220
    bar_w = 13
    col_gap = 28
    row_gap = 20
    header_h = 155
    top_h = 540
    top_gap = 24
    col_label_h = 40
    margin = 34
    row_h = img + row_gap
    grid_h = len(rows) * row_h - row_gap
    col_w = img + 18 + bar_w
    width = margin * 2 + row_label_w + len(COLUMN_SPECS) * col_w + (len(COLUMN_SPECS) - 1) * col_gap
    x0 = margin + row_label_w
    top_y = header_h
    top_w = width - x0 - margin
    y_cols = header_h + top_h + top_gap
    y_grid = y_cols + col_label_h
    height = header_h + top_h + top_gap + col_label_h + grid_h + margin
    canvas = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 18), f"NS2D Final-State Comparison | {eps_text(eps_label)} | Optimizer = {method_text(method)}", font=base.FONT_TITLE, fill=(22, 27, 32))
    draw.text((margin, 54), f"Dataset index = {dataset_index}", font=base.FONT_SUBTITLE, fill=(65, 68, 74))

    top_img = plot_top_panels(cases, top_w, top_h, max_internal_label)
    canvas.paste(top_img, (x0, top_y))
    draw.rectangle((x0, top_y, x0 + top_w, top_y + top_h), outline=(70, 75, 82), width=1)

    for ci, (key, label, _) in enumerate(COLUMN_SPECS):
        x = x0 + ci * (col_w + col_gap)
        draw.text((x, y_cols + 4), label, font=base.FONT_COL, fill=(32, 37, 42))
        vmin, vmax, cmap = lims[key]
        bar = base.vertical_colorbar(grid_h, bar_w, cmap, vmin, vmax)
        bar_x = x + img + 6
        canvas.paste(bar, (bar_x, y_grid))
        draw.rectangle((bar_x, y_grid, bar_x + bar_w, y_grid + grid_h), outline=(75, 80, 85), width=1)
        draw.text((bar_x + bar_w + 3, y_grid - 2), base.fmt(vmax, 3), font=base.FONT_TINY, fill=(70, 73, 78))
        min_label = base.fmt(vmin, 3)
        bbox = draw.textbbox((0, 0), min_label, font=base.FONT_TINY)
        draw.text((bar_x + bar_w + 3, y_grid + grid_h - (bbox[3] - bbox[1]) + 1), min_label, font=base.FONT_TINY, fill=(70, 73, 78))

    for ri, row in enumerate(rows):
        y = y_grid + ri * row_h
        label_box = (margin - 4, y + 4, margin + row_label_w - 14, y + img - 4)
        if ri == 0:
            draw.rounded_rectangle(label_box, radius=6, fill=(244, 247, 250), outline=(190, 198, 206), width=1)
        elif row.get("is_max"):
            draw.rounded_rectangle(label_box, radius=6, fill=(255, 248, 247), outline=(203, 34, 47), width=3)
        else:
            draw.rounded_rectangle(label_box, radius=6, fill=(250, 250, 248), outline=(226, 228, 230), width=1)
        if ri == 1:
            draw.line((margin, y - 8, width - margin, y - 8), fill=(210, 215, 220), width=1)
        fills = [(35, 39, 44), (45, 48, 52), (184, 24, 36) if row.get("is_max") else (68, 72, 78)]
        draw_lines(
            draw,
            (margin + 8, y + 18),
            [row["label"], row["sub"], row.get("sub2", "")],
            [base.FONT_ROW, base.FONT_SMALL, base.FONT_SMALL],
            fills,
            gap=5,
        )
        for ci, (key, _, _) in enumerate(COLUMN_SPECS):
            x = x0 + ci * (col_w + col_gap)
            vmin, vmax, cmap = lims[key]
            tile = Image.fromarray(base.colorize(row[key], cmap, vmin, vmax), mode="RGB").resize((img, img), Image.Resampling.BILINEAR)
            canvas.paste(tile, (x, y))
            outline = (203, 34, 47) if row.get("is_max") else (40, 44, 49)
            draw.rectangle((x, y, x + img, y + img), outline=outline, width=3 if row.get("is_max") else 1)

    out_name = f"{eps_label}_{method}_spectrum_loss_curves_dataset{dataset_index}.png"
    out_path = OUT_DIR / out_name
    canvas.save(out_path)

    summaries = []
    for row in rows:
        s = dict(row["summary"])
        s.update({
            "eps_label": eps_label,
            "epsilon_alpha": eps_text(eps_label),
            "method": method,
            "optimizer": method_text(method),
            "dataset_index": dataset_index,
            "output_png": rel(out_path),
            "max_candidate_label": max_pretty_label,
            "max_adv_true_loss": max_adv_true_loss,
        })
        summaries.append(s)
    return RenderSummary(eps_label, method, dataset_index, len(rows) - 1, out_path, clean_loss, max_pretty_label, max_adv_true_loss, summaries)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rendered: list[RenderSummary], missing: list[dict[str, Any]], summary_csv: Path, manifest_path: Path) -> None:
    lines = [
        "# NS2D Heatmaps With Delta Spectra And Loss Curves - 2026-05-24",
        "",
        "Observed from saved `final_state_outputs.npz`, `per_step_metrics.csv`, and `per_sample_step_metrics.csv` only. No model inference, solver rollout, attack update, PyTorch, JAX, or GPU work was run.",
        "",
        "Each PNG has a 2 x 2 top line-plot block. The loss curves use the same final all-W true-loss metric for every attack target.",
        "",
        "## Outputs",
        "",
        f"- Summary CSV: [{summary_csv.name}]({report_rel(summary_csv)})",
        f"- Manifest JSON: [{manifest_path.name}]({report_rel(manifest_path)})",
        f"- PNG directory: `{OUT_DIR.relative_to(ROOT).as_posix()}/`",
        "",
        "## Rendered Figures",
        "",
        "| Epsilon / Alpha | Optimizer | attack rows | highlighted final true-loss row | highlighted true loss | PNG |",
        "|---|---|---:|---|---:|---|",
    ]
    for item in rendered:
        lines.append(f"| {eps_text(item.eps_label)} | {method_text(item.method)} | {item.n_attack_rows} | {item.max_candidate_label} | {base.fmt(item.max_adv_true_loss)} | [{item.output_png.name}]({report_rel(item.output_png)}) |")
    if missing:
        lines += ["", "## Missing Expected Canonical Rows", ""]
        for item in missing:
            lines.append(f"- {eps_text(item['eps_label'])}, {method_text(item['method'])}: missing {label_text(item['label'])}")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = base.discover_candidates()
    by_group: dict[tuple[str, str], list[Any]] = {}
    for c in candidates:
        by_group.setdefault((c.eps_label, c.method), []).append(c)

    rendered: list[RenderSummary] = []
    all_rows: list[dict[str, Any]] = []
    for key in sorted(by_group, key=lambda k: (base.eps_sort_key(k[0]), base.method_sort_key(k[1]))):
        eps_label, method = key
        group = sorted(by_group[key], key=base.candidate_sort_key)
        if not group:
            continue
        item = render_group(eps_label, method, group)
        rendered.append(item)
        all_rows.extend(item.rows)

    missing = []
    expected = ["loss1/all_w", "loss2/all_a_target_w", "loss3/all_w"]
    for eps in base.EPS_ORDER:
        for method in base.METHOD_ORDER:
            present = {c.label for c in by_group.get((eps, method), [])}
            if not present:
                continue
            for label in expected:
                if label not in present:
                    missing.append({"eps_label": eps, "method": method, "label": label})

    summary_csv = OUT_DIR / "row_loss_summary.csv"
    manifest_path = OUT_DIR / "manifest.json"
    write_csv(summary_csv, all_rows)
    manifest = {
        "created_by": "tools/plot_ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524.py",
        "created_date": "2026-05-24",
        "spectrum_xmax_normalized_radius": SPECTRUM_XMAX,
        "loss_curve_xmax": LOSS_CURVE_XMAX,
        "spectrum_xticks": [0, 0.05, 0.10, 0.15, 0.20, 0.25],
        "spectrum_and_loss_curve_layout": "selected index panel plus all-sample mean/std panel for both delta spectrum and true-loss curve",
        "output_dir": rel(OUT_DIR),
        "rendered_png_count": len(rendered),
        "summary_csv": rel(summary_csv),
        "report_md": rel(REPORT_PATH),
        "run_roots": [rel(path) for path in base.RUN_ROOTS if path.exists()],
        "missing_expected_canonical_rows": missing,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(rendered, missing, summary_csv, manifest_path)
    print(f"Rendered {len(rendered)} NS2D PNG files")
    print(f"Output dir: {rel(OUT_DIR)}")
    print(f"Report: {rel(REPORT_PATH)}")


if __name__ == "__main__":
    main()
