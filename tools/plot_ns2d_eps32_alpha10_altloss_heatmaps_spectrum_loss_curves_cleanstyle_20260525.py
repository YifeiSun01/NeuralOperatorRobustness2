#!/usr/bin/env python3
"""Original-cleanstyle NS2D eps32/alpha10 alt-loss comparison.

Derived from tools/plot_ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524.py.
It keeps the original PIL canvas layout, chart panel sizing, fonts, colorbars,
and heatmap tile sizes. It only changes the data rows to compare baseline
Loss 3/all-W against the nine alternative loss3 metrics.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
ORIG_SCRIPT = ROOT / "tools" / "plot_ns2d_heatmaps_spectrum_loss_curves_cleanstyle_20260524.py"

spec = importlib.util.spec_from_file_location("ns2d_cleanstyle_orig_20260524", ORIG_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {ORIG_SCRIPT}")
orig = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = orig
spec.loader.exec_module(orig)
base = orig.base

ALT_LOSS_SCRIPT = ROOT / "2D_NS_FNO2d_recurrent/perturbation_methods/ns2d_alternative_losses.py"
loss_spec = importlib.util.spec_from_file_location("ns2d_alternative_losses_for_plot_20260525", ALT_LOSS_SCRIPT)
if loss_spec is None or loss_spec.loader is None:
    raise RuntimeError(f"Cannot import {ALT_LOSS_SCRIPT}")
loss_impl = importlib.util.module_from_spec(loss_spec)
sys.modules[loss_spec.name] = loss_impl
loss_spec.loader.exec_module(loss_impl)

DEFAULT_BASELINE_DIR = (
    ROOT
    / "2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack"
    / "full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10"
    / "mode_wwwwwwwwww_p2_q2_20260522_074135_UTC"
    / "batch_0000_0009/loss3/steepest_add"
)
DEFAULT_ALT_ROOT = (
    ROOT
    / "2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack"
    / "eps32_alpha10_steepest_add_loss3_allw_alt5_b10_20260525_035657_UTC/eps32_alpha10"
)
DEFAULT_OUT_DIR = ROOT / "docs/ns2d_eps32_alpha10_altloss_original_cleanstyle_20260525"

SAMPLE_POSITION = 0
SPECTRUM_XMAX = orig.SPECTRUM_XMAX
LOSS_CURVE_XMAX = orig.LOSS_CURVE_XMAX
SPECTRUM_FLOOR = orig.SPECTRUM_FLOOR
EPS = orig.EPS

ALT_METRICS = (
    "dists",
    "ms_ssim",
    "scattering2d",
    "affine_dists",
    "local_warp_dists",
    "homography_dists",
    "tps_dists",
    "elastic_dists",
    "svf_dists",
)
EXPLICIT_ALIGNMENT_KEYS = {
    "affine_dists",
    "local_warp_dists",
    "homography_dists",
    "tps_dists",
    "elastic_dists",
    "svf_dists",
}
ALIGNMENT_COLUMNS = {"aligned_model", "aligned_diff", "warp_mag"}
LABELS = {
    "baseline_loss3": "Loss 3 baseline",
    "dists": "DISTS",
    "ms_ssim": "MS-SSIM",
    "scattering2d": "Scattering2D",
    "affine_dists": "Affine + DISTS",
    "local_warp_dists": "Local warp + DISTS",
    "homography_dists": "Homography + DISTS",
    "tps_dists": "TPS + DISTS",
    "elastic_dists": "Elastic + DISTS",
    "svf_dists": "SVF + DISTS",
}
FULL_LOSS_NAMES = {
    "baseline_loss3": "Loss 3 baseline: original all-W final-state W/Q norm objective",
    "dists": "DISTS: Deep Image Structure and Texture Similarity distance",
    "ms_ssim": "MS-SSIM: Multi-Scale Structural Similarity distance, 1 - MS-SSIM",
    "scattering2d": "Scattering2D: 2D wavelet scattering feature L2 distance",
    "affine_dists": "Affine + DISTS: bounded affine alignment followed by DISTS",
    "local_warp_dists": "Local warp + DISTS: bounded local deformation / dense-warp alignment followed by DISTS",
    "homography_dists": "Homography + DISTS: bounded projective corner warp followed by DISTS",
    "tps_dists": "TPS + DISTS: bounded thin-plate-spline control-grid warp followed by DISTS",
    "elastic_dists": "Elastic + DISTS: bounded smoothed elastic deformation followed by DISTS",
    "svf_dists": "SVF + DISTS: bounded stationary-velocity-field diffeomorphic-style warp followed by DISTS",
}
SHORT_LABELS = {
    "baseline_loss3": "L3 baseline",
    "dists": "DISTS",
    "ms_ssim": "MS-SSIM",
    "scattering2d": "Scat2D",
    "affine_dists": "Aff+DISTS",
    "local_warp_dists": "Warp+DISTS",
    "homography_dists": "Homog+DISTS",
    "tps_dists": "TPS+DISTS",
    "elastic_dists": "Elastic+DISTS",
    "svf_dists": "SVF+DISTS",
}
LINE_COLORS = {
    "baseline_loss3": "#c8353e",
    "dists": "#3267b1",
    "ms_ssim": "#d28b26",
    "scattering2d": "#7c5ab8",
    "affine_dists": "#4c9a57",
    "local_warp_dists": "#2b8cbe",
    "homography_dists": "#8c6d31",
    "tps_dists": "#cc6677",
    "elastic_dists": "#117733",
    "svf_dists": "#44aa99",
}
COLUMN_SPECS = [
    ("initial", "Initial Condition", "sequential"),
    ("delta", "Perturbation", "diverging"),
    ("final", "Final Condition", "sequential"),
    ("model", "Model Output", "sequential"),
    ("solver", "Solver Output", "sequential"),
    ("diff", "Model - Solver", "diverging"),
    ("aligned_model", "Aligned Model", "sequential"),
    ("aligned_diff", "Aligned Model - Solver", "diverging"),
    ("warp_mag", "Alignment Warp", "sequential"),
]

BASE_PLOT_ALIGNMENT_ARGS: dict[str, Any] = {
    "loss3_metric": "qnorm",
    "q_order": 2,
    "loss3_image_normalization": "pair_minmax_detached",
    "loss3_metric_eps": 1e-6,
    "loss3_align_objective": "l2",
    "loss3_scattering_j": 3,
    "loss3_affine_inner_steps": 8,
    "loss3_affine_lr": 0.05,
    "loss3_affine_max_shift_ratio": 0.05,
    "loss3_affine_max_angle_deg": 10.0,
    "loss3_affine_max_log_scale": 0.09531017980432493,
    "loss3_affine_reg_weight": 0.01,
    "loss3_local_grid_size": 8,
    "loss3_local_inner_steps": 8,
    "loss3_local_lr": 0.05,
    "loss3_local_max_disp_ratio": 0.03,
    "loss3_local_mag_weight": 0.01,
    "loss3_local_smooth_weight": 0.05,
    "loss3_homography_inner_steps": 8,
    "loss3_homography_lr": 0.05,
    "loss3_homography_max_corner_ratio": 0.05,
    "loss3_homography_reg_weight": 0.01,
    "loss3_tps_grid_size": 4,
    "loss3_tps_inner_steps": 8,
    "loss3_tps_lr": 0.05,
    "loss3_tps_max_disp_ratio": 0.05,
    "loss3_tps_offset_weight": 0.01,
    "loss3_tps_smooth_weight": 0.05,
    "loss3_elastic_grid_size": 16,
    "loss3_elastic_inner_steps": 8,
    "loss3_elastic_lr": 0.05,
    "loss3_elastic_max_disp_ratio": 0.05,
    "loss3_elastic_smooth_kernel": 9,
    "loss3_elastic_smooth_passes": 2,
    "loss3_elastic_mag_weight": 0.01,
    "loss3_elastic_smooth_weight": 0.03,
    "loss3_svf_grid_size": 8,
    "loss3_svf_inner_steps": 8,
    "loss3_svf_lr": 0.05,
    "loss3_svf_max_vel_ratio": 0.04,
    "loss3_svf_int_steps": 5,
    "loss3_svf_mag_weight": 0.01,
    "loss3_svf_smooth_weight": 0.05,
}

BUDGET_PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    "weak": {},
    "medium": {
        "loss3_scattering_j": 4,
        "loss3_affine_inner_steps": 12,
        "loss3_affine_max_shift_ratio": 0.08,
        "loss3_affine_max_angle_deg": 15.0,
        "loss3_affine_max_log_scale": 0.1823215567939546,
        "loss3_affine_reg_weight": 0.005,
        "loss3_local_grid_size": 10,
        "loss3_local_inner_steps": 12,
        "loss3_local_max_disp_ratio": 0.05,
        "loss3_local_mag_weight": 0.005,
        "loss3_local_smooth_weight": 0.02,
        "loss3_homography_inner_steps": 12,
        "loss3_homography_max_corner_ratio": 0.07,
        "loss3_homography_reg_weight": 0.008,
        "loss3_tps_inner_steps": 12,
        "loss3_tps_max_disp_ratio": 0.07,
        "loss3_tps_offset_weight": 0.008,
        "loss3_tps_smooth_weight": 0.03,
        "loss3_elastic_inner_steps": 12,
        "loss3_elastic_max_disp_ratio": 0.06,
        "loss3_elastic_mag_weight": 0.005,
        "loss3_elastic_smooth_weight": 0.02,
        "loss3_svf_grid_size": 10,
        "loss3_svf_inner_steps": 12,
        "loss3_svf_max_vel_ratio": 0.06,
        "loss3_svf_mag_weight": 0.005,
        "loss3_svf_smooth_weight": 0.03,
    },
    "strong": {
        "loss3_scattering_j": 4,
        "loss3_affine_inner_steps": 20,
        "loss3_affine_max_shift_ratio": 0.12,
        "loss3_affine_max_angle_deg": 25.0,
        "loss3_affine_max_log_scale": 0.30010459245033816,
        "loss3_affine_reg_weight": 0.002,
        "loss3_local_grid_size": 12,
        "loss3_local_inner_steps": 20,
        "loss3_local_max_disp_ratio": 0.08,
        "loss3_local_mag_weight": 0.002,
        "loss3_local_smooth_weight": 0.01,
        "loss3_homography_inner_steps": 20,
        "loss3_homography_max_corner_ratio": 0.10,
        "loss3_homography_reg_weight": 0.005,
        "loss3_tps_grid_size": 5,
        "loss3_tps_inner_steps": 20,
        "loss3_tps_max_disp_ratio": 0.10,
        "loss3_tps_offset_weight": 0.01,
        "loss3_tps_smooth_weight": 0.02,
        "loss3_elastic_inner_steps": 20,
        "loss3_elastic_max_disp_ratio": 0.08,
        "loss3_elastic_mag_weight": 0.002,
        "loss3_elastic_smooth_weight": 0.01,
        "loss3_svf_grid_size": 12,
        "loss3_svf_inner_steps": 20,
        "loss3_svf_max_vel_ratio": 0.08,
        "loss3_svf_int_steps": 6,
        "loss3_svf_mag_weight": 0.002,
        "loss3_svf_smooth_weight": 0.01,
    },
    "very_strong": {
        "loss3_scattering_j": 5,
        "loss3_affine_inner_steps": 30,
        "loss3_affine_max_shift_ratio": 0.20,
        "loss3_affine_max_angle_deg": 45.0,
        "loss3_affine_max_log_scale": 0.6931471805599453,
        "loss3_affine_reg_weight": 0.0005,
        "loss3_local_grid_size": 16,
        "loss3_local_inner_steps": 30,
        "loss3_local_max_disp_ratio": 0.15,
        "loss3_local_mag_weight": 0.0005,
        "loss3_local_smooth_weight": 0.003,
        "loss3_homography_inner_steps": 30,
        "loss3_homography_max_corner_ratio": 0.18,
        "loss3_homography_reg_weight": 0.001,
        "loss3_tps_grid_size": 6,
        "loss3_tps_inner_steps": 30,
        "loss3_tps_max_disp_ratio": 0.18,
        "loss3_tps_offset_weight": 0.003,
        "loss3_tps_smooth_weight": 0.006,
        "loss3_elastic_grid_size": 20,
        "loss3_elastic_inner_steps": 30,
        "loss3_elastic_max_disp_ratio": 0.15,
        "loss3_elastic_smooth_kernel": 7,
        "loss3_elastic_smooth_passes": 1,
        "loss3_elastic_mag_weight": 0.0005,
        "loss3_elastic_smooth_weight": 0.003,
        "loss3_svf_grid_size": 16,
        "loss3_svf_inner_steps": 30,
        "loss3_svf_max_vel_ratio": 0.14,
        "loss3_svf_int_steps": 7,
        "loss3_svf_mag_weight": 0.0005,
        "loss3_svf_smooth_weight": 0.003,
    },
}


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    short_label: str
    path: Path


@dataclass
class Case:
    candidate: Candidate
    data: dict[str, np.ndarray]
    per_step: list[dict[str, str]]
    per_sample_step: list[dict[str, str]]
    final_metrics: list[dict[str, str]]
    step_trace: dict[str, np.ndarray]
    dataset_index: int
    clean_loss: float
    adv_loss: float
    delta_l2: float
    delta_linf: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, fallback: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def relpath(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def discover_alt_dir(alt_root: Path, metric: str) -> Path:
    final_npzs = sorted((alt_root / metric).glob("mode_*/batch_0000_0009/loss3/steepest_add/final_state_outputs.npz"))
    if not final_npzs:
        raise SystemExit(f"Expected at least one completed method dir for {metric}, found 0 under {alt_root}")
    if len(final_npzs) > 1:
        print(f"WARNING: found {len(final_npzs)} completed runs for {metric}; using latest path {final_npzs[-1]}", file=sys.stderr)
    return final_npzs[-1].parent


def build_candidates(baseline_dir: Path, alt_root: Path) -> list[Candidate]:
    items = [Candidate("baseline_loss3", LABELS["baseline_loss3"], SHORT_LABELS["baseline_loss3"], baseline_dir / "final_state_outputs.npz")]
    for metric in ALT_METRICS:
        items.append(Candidate(metric, LABELS[metric], SHORT_LABELS[metric], discover_alt_dir(alt_root, metric) / "final_state_outputs.npz"))
    return items


def metric_row(rows: list[dict[str, str]], sample_position: int) -> dict[str, str]:
    for row in rows:
        if int(fnum(row.get("sample_position"), -1)) == sample_position:
            return row
    return rows[sample_position] if len(rows) > sample_position else {}


def load_case(candidate: Candidate, sample_position: int) -> Case:
    method_dir = candidate.path.parent
    data = load_npz(candidate.path)
    final_metrics = read_csv(method_dir / "final_state_metrics.csv")
    row = metric_row(final_metrics, sample_position)
    clean_loss = fnum(row.get("clean_true_loss"), float(data["clean_true_loss"][sample_position]))
    adv_loss = fnum(row.get("adv_true_loss"), float(data["adv_true_loss"][sample_position]))
    delta = base.finite(data["final_delta"][sample_position])
    return Case(
        candidate=candidate,
        data=data,
        per_step=read_csv(method_dir / "per_step_metrics.csv"),
        per_sample_step=read_csv(method_dir / "per_sample_step_metrics.csv"),
        final_metrics=final_metrics,
        step_trace=load_npz(method_dir / "step_sample_trace.npz"),
        dataset_index=int(np.asarray(data["dataset_indices"])[sample_position]),
        clean_loss=clean_loss,
        adv_loss=adv_loss,
        delta_l2=fnum(row.get("delta_l2"), float(np.linalg.norm(delta.ravel()))),
        delta_linf=fnum(row.get("delta_linf"), float(np.max(np.abs(delta)))),
    )


def rows_for_sample(rows: list[dict[str, str]], sample_position: int) -> list[dict[str, str]]:
    out = [row for row in rows if int(fnum(row.get("sample_position"), -1)) == sample_position]
    return sorted(out, key=lambda row: fnum(row.get("k")))


def curve_for_case(case: Case, selected: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if selected:
        rows = rows_for_sample(case.per_sample_step, SAMPLE_POSITION)
        x = np.asarray([fnum(row.get("k")) for row in rows], dtype=np.float64)
        y = np.asarray([fnum(row.get("active_loss_value"), fnum(row.get("loss3"))) for row in rows], dtype=np.float64)
        denom = y[0] if y.size and math.isfinite(float(y[0])) and abs(float(y[0])) > 1e-12 else 1.0
        return x, y / denom, None
    rows = case.per_step
    x = np.asarray([fnum(row.get("k")) for row in rows], dtype=np.float64)
    y = np.asarray([fnum(row.get("active_loss_mean"), fnum(row.get("loss3_mean"))) for row in rows], dtype=np.float64)
    std = np.asarray([fnum(row.get("loss3_std")) for row in rows], dtype=np.float64)
    denom = y[0] if y.size and math.isfinite(float(y[0])) and abs(float(y[0])) > 1e-12 else 1.0
    return x, y / denom, std / abs(denom)


def rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def wrap_text_by_pixels(draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: int) -> list[str]:
    words = str(text).split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_wrapped_row_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    title: str,
    sub: str,
    sub2: str,
    max_width: int,
    title_fill: tuple[int, int, int],
    sub_fill: tuple[int, int, int],
    sub2_fill: tuple[int, int, int],
) -> None:
    x, y = xy
    draw.text((x, y), title, font=base.FONT_ROW, fill=title_fill)
    bbox = draw.textbbox((x, y), title, font=base.FONT_ROW)
    y += bbox[3] - bbox[1] + 7
    for line in wrap_text_by_pixels(draw, sub, base.FONT_SMALL, max_width):
        draw.text((x, y), line, font=base.FONT_SMALL, fill=sub_fill)
        bbox = draw.textbbox((x, y), line, font=base.FONT_SMALL)
        y += bbox[3] - bbox[1] + 4
    if sub2:
        y += 2
    for line in wrap_text_by_pixels(draw, sub2, base.FONT_SMALL, max_width):
        draw.text((x, y), line, font=base.FONT_SMALL, fill=sub2_fill)
        bbox = draw.textbbox((x, y), line, font=base.FONT_SMALL)
        y += bbox[3] - bbox[1] + 4


def qnorm_from_trace(case: Case) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(case.step_trace["k"], dtype=np.float64)
    diff = np.asarray(case.step_trace["adv_model_minus_solver"], dtype=np.float64)
    y = np.linalg.norm(diff.reshape(diff.shape[0], -1), ord=2, axis=1)
    return k, y


def final_wq_mean(case: Case) -> float:
    vals = [fnum(row.get("adv_true_loss")) for row in case.final_metrics]
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(arr))


def clean_wq_mean(case: Case) -> float:
    vals = [fnum(row.get("clean_true_loss")) for row in case.final_metrics]
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(arr))


def draw_bar_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    cases: list[Case],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1, y1), radius=6, fill=(255, 255, 253), outline=(184, 193, 204), width=1)
    draw.text((x0 + 12, y0 + 8), title, font=base.FONT_TINY, fill=(30, 34, 40))
    plot = (x0 + 66, y0 + 34, x1 - 18, y1 - 45)
    px0, py0, px1, py1 = plot
    adv = np.asarray([final_wq_mean(case) for case in cases], dtype=np.float64)
    clean = float(np.nanmean([clean_wq_mean(case) for case in cases]))
    ymax = float(np.nanmax(np.concatenate([adv, np.asarray([clean])])) * 1.12)
    ymax = ymax if math.isfinite(ymax) and ymax > 0 else 1.0
    for frac in (0.25, 0.5, 0.75):
        gy = int(py1 - (py1 - py0) * frac)
        draw.line((px0, gy, px1, gy), fill=(226, 228, 226), width=1)
    draw.rectangle(plot, outline=(67, 72, 80), width=1)
    n = len(cases)
    slot = (px1 - px0) / max(n, 1)
    bar_w = max(12, int(slot * 0.55))
    for i, case in enumerate(cases):
        cx = int(px0 + slot * (i + 0.5))
        h = int((adv[i] / ymax) * (py1 - py0))
        color = rgb(LINE_COLORS[case.candidate.key])
        draw.rectangle((cx - bar_w // 2, py1 - h, cx + bar_w // 2, py1), fill=color, outline=(45, 49, 55))
        label = case.candidate.short_label
        bbox = draw.textbbox((0, 0), label, font=base.FONT_TINY)
        draw.text((cx - (bbox[2] - bbox[0]) // 2, py1 + 5), label, font=base.FONT_TINY, fill=(55, 60, 68))
    clean_y = int(py1 - (clean / ymax) * (py1 - py0))
    draw.line((px0, clean_y, px1, clean_y), fill=(15, 15, 15), width=2)
    draw.text((px0 + 6, max(py0 + 3, clean_y - 15)), "clean mean", font=base.FONT_TINY, fill=(15, 15, 15))
    for value, ypos in [(ymax, py0), (0.0, py1 - 10)]:
        draw.text((x0 + 9, ypos), base.fmt(value, 3), font=base.FONT_TINY, fill=(76, 82, 91))
    draw.text((px0, y1 - 24), "method", font=base.FONT_TINY, fill=(76, 82, 91))


def plot_top_panels(cases: list[Case], width: int, height: int) -> Image.Image:
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

    prepared: list[dict[str, Any]] = []
    selected_spec_lines = []
    mean_spec_lines = []
    selected_loss3_lines = []
    for case in cases:
        color = rgb(LINE_COLORS[case.candidate.key])
        width_line = 6 if case.candidate.key == "baseline_loss3" else 4
        data = case.data
        f_sel, a_sel = orig.normalized_radial_amplitude(data["final_delta"][SAMPLE_POSITION])
        f_mean, a_mean, a_std = orig.spectrum_stack(data["final_delta"])
        k_loss3, y_loss3 = qnorm_from_trace(case)
        mask_sel = f_sel <= SPECTRUM_XMAX
        mask_mean = f_mean <= SPECTRUM_XMAX
        k_loss3_mask = k_loss3 <= LOSS_CURVE_XMAX
        spec_sel_y = np.log10(np.maximum(a_sel[mask_sel], SPECTRUM_FLOOR))
        spec_mean_y = np.log10(np.maximum(a_mean[mask_mean], SPECTRUM_FLOOR))
        spec_mean_lo = np.log10(np.maximum(a_mean[mask_mean] - a_std[mask_mean], SPECTRUM_FLOOR))
        spec_mean_hi = np.log10(np.maximum(a_mean[mask_mean] + a_std[mask_mean], SPECTRUM_FLOOR))
        item = {
            "label": case.candidate.short_label,
            "color": color,
            "width": width_line,
            "f_sel": f_sel[mask_sel],
            "spec_sel_y": spec_sel_y,
            "f_mean": f_mean[mask_mean],
            "spec_mean_y": spec_mean_y,
            "spec_mean_lo": spec_mean_lo,
            "spec_mean_hi": spec_mean_hi,
            "k_loss3": k_loss3[k_loss3_mask],
            "loss3_wq": y_loss3[k_loss3_mask],
        }
        prepared.append(item)
        selected_spec_lines.append((item["f_sel"], item["spec_sel_y"]))
        mean_spec_lines.append((item["f_mean"], item["spec_mean_y"]))
        selected_loss3_lines.append((item["k_loss3"], item["loss3_wq"]))

    y_sel_spec = orig.chart_line_limits(selected_spec_lines)
    y_mean_spec = orig.chart_line_limits(mean_spec_lines)
    y_sel_spec = (max(y_sel_spec[0], math.log10(SPECTRUM_FLOOR)), y_sel_spec[1])
    y_mean_spec = (max(y_mean_spec[0], math.log10(SPECTRUM_FLOOR)), y_mean_spec[1])
    y_loss3 = orig.chart_line_limits(selected_loss3_lines)

    orig.draw_chart_panel(
        canvas, draw, boxes[0], "Single index delta spectrum (log10 amp)",
        [{"x": p["f_sel"], "y": p["spec_sel_y"], "color": p["color"], "width": p["width"], "label": p["label"]} for p in prepared],
        (0.0, SPECTRUM_XMAX), y_sel_spec, "freq |k| (0-0.25)", "", True,
    )
    orig.draw_chart_panel(
        canvas, draw, boxes[1], "All samples mean delta spectrum (log10 amp)",
        [{"x": p["f_mean"], "y": p["spec_mean_y"], "lower": p["spec_mean_lo"], "upper": p["spec_mean_hi"], "color": p["color"], "width": p["width"], "label": p["label"]} for p in prepared],
        (0.0, SPECTRUM_XMAX), y_mean_spec, "freq |k| (0-0.25)", "", True,
    )
    orig.draw_chart_panel(
        canvas, draw, boxes[2], "Single index all-W Loss 3 / W-Q norm curve",
        [{"x": p["k_loss3"], "y": p["loss3_wq"], "color": p["color"], "width": p["width"], "label": p["label"]} for p in prepared],
        (0.0, LOSS_CURVE_XMAX), y_loss3, "attack step", "", True,
    )
    draw_bar_panel(canvas, draw, boxes[3], "Final batch mean all-W Loss 3 / W-Q norm", cases)
    return canvas


ATTACK_FIELD_KEYS = {
    "initial": "x_clean",
    "delta": "final_delta",
    "final": "x_adv",
    "model": "adv_model_final",
    "solver": "adv_solver_final",
    "diff": "adv_model_minus_solver",
    "aligned_model": "recomputed_offline_alignment_of_adv_model_final",
    "aligned_diff": "recomputed_offline_alignment_model_minus_adv_solver_final",
    "warp_mag": "recomputed_offline_alignment_grid_magnitude",
}
CLEAN_FIELD_KEYS = {
    "initial": "x_clean",
    "delta": "zeros_like_x_clean",
    "final": "x_clean",
    "model": "clean_model_final",
    "solver": "clean_solver_final",
    "diff": "clean_model_minus_solver",
    "aligned_model": "clean_model_final_no_alignment",
    "aligned_diff": "clean_model_minus_solver_no_alignment",
    "warp_mag": "zeros_like_x_clean",
}


def _as_torch_image(arr: np.ndarray):
    import torch

    x = torch.from_numpy(np.asarray(arr, dtype=np.float32)).view(1, 1, *arr.shape)
    return x


def _base_grid_torch(height: int, width: int, dtype, device):
    import torch

    ys = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
    xs = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).unsqueeze(0)


def _normalize_pair_torch(model_img, solver_img):
    both = __import__("torch").cat([model_img, solver_img], dim=1)
    reduce_dims = tuple(range(1, both.ndim))
    lo = both.amin(dim=reduce_dims, keepdim=True).detach()
    hi = both.amax(dim=reduce_dims, keepdim=True).detach()
    scale = (hi - lo).clamp_min(1e-6)
    return (model_img - lo) / scale, (solver_img - lo) / scale


def _affine_theta_from_raw(raw):
    import torch

    max_shift_ratio = 0.05
    max_angle = 10.0 * math.pi / 180.0
    max_log_scale = math.log(1.1)
    tx = torch.tanh(raw[:, 0]) * max_shift_ratio * 2.0
    ty = torch.tanh(raw[:, 1]) * max_shift_ratio * 2.0
    angle = torch.tanh(raw[:, 2]) * max_angle
    log_scale = torch.tanh(raw[:, 3]) * max_log_scale
    scale = torch.exp(log_scale)
    cos_a = torch.cos(angle) * scale
    sin_a = torch.sin(angle) * scale
    theta = torch.zeros(raw.shape[0], 2, 3, device=raw.device, dtype=raw.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    reg = tx.square() + ty.square() + angle.square() + log_scale.square()
    return theta, reg


def _aligned_affine_model(model: np.ndarray, solver: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    model_raw = _as_torch_image(model)
    solver_raw = _as_torch_image(solver)
    model_norm, solver_norm = _normalize_pair_torch(model_raw, solver_raw)
    raw = torch.zeros(1, 4, dtype=model_norm.dtype, device=model_norm.device, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.05)
    for _ in range(8):
        theta, reg = _affine_theta_from_raw(raw)
        grid = F.affine_grid(theta, model_norm.shape, align_corners=False)
        warped = F.grid_sample(model_norm, grid, mode="bilinear", padding_mode="border", align_corners=False)
        loss = (warped - solver_norm).square().mean() + 0.01 * reg.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        theta, _ = _affine_theta_from_raw(raw.detach())
        grid = F.affine_grid(theta, model_raw.shape, align_corners=False)
        aligned = F.grid_sample(model_raw, grid, mode="bilinear", padding_mode="border", align_corners=False)
        base_grid = _base_grid_torch(model.shape[0], model.shape[1], grid.dtype, grid.device)
        warp_mag = torch.linalg.vector_norm(grid - base_grid, ord=2, dim=-1)
    return aligned[0, 0].cpu().numpy(), warp_mag[0].cpu().numpy()


def _aligned_local_model(model: np.ndarray, solver: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    model_raw = _as_torch_image(model)
    solver_raw = _as_torch_image(solver)
    model_norm, solver_norm = _normalize_pair_torch(model_raw, solver_raw)
    _, _, height, width = model_norm.shape
    raw = torch.zeros(1, 2, 8, 8, dtype=model_norm.dtype, device=model_norm.device, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.05)
    base_grid = _base_grid_torch(height, width, model_norm.dtype, model_norm.device)
    for _ in range(8):
        disp_low = torch.tanh(raw) * 0.03 * 2.0
        disp = F.interpolate(disp_low, size=(height, width), mode="bilinear", align_corners=False)
        grid = base_grid + disp.permute(0, 2, 3, 1)
        warped = F.grid_sample(model_norm, grid, mode="bilinear", padding_mode="border", align_corners=False)
        mag = disp.square().flatten(1).mean(dim=1)
        dx = disp[:, :, :, 1:] - disp[:, :, :, :-1]
        dy = disp[:, :, 1:, :] - disp[:, :, :-1, :]
        smooth = dx.square().flatten(1).mean(dim=1) + dy.square().flatten(1).mean(dim=1)
        loss = (warped - solver_norm).square().mean() + 0.01 * mag.mean() + 0.05 * smooth.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        disp_low = torch.tanh(raw.detach()) * 0.03 * 2.0
        disp = F.interpolate(disp_low, size=(height, width), mode="bilinear", align_corners=False)
        grid = base_grid + disp.permute(0, 2, 3, 1)
        aligned = F.grid_sample(model_raw, grid, mode="bilinear", padding_mode="border", align_corners=False)
        warp_mag = torch.linalg.vector_norm(disp.permute(0, 2, 3, 1), ord=2, dim=-1)
    return aligned[0, 0].cpu().numpy(), warp_mag[0].cpu().numpy()


def infer_budget_preset(path: Path) -> str:
    haystack = " ".join(str(part).lower() for part in path.parts)
    for preset in ("very_strong", "strong", "medium", "weak"):
        if preset in haystack:
            return preset
    return "weak"


def plot_alignment_args_for_case(case: Case) -> SimpleNamespace:
    params = dict(BASE_PLOT_ALIGNMENT_ARGS)
    preset = infer_budget_preset(case.candidate.path)
    params.update(BUDGET_PRESET_OVERRIDES[preset])
    return SimpleNamespace(**params)


def _metric_computer_for_case(case: Case):
    args = plot_alignment_args_for_case(case)
    computer = loss_impl.Loss3MetricComputer(args)
    computer.eval()
    return computer, infer_budget_preset(case.candidate.path)


def _tensor_image_to_np(x) -> np.ndarray:
    import torch

    with torch.no_grad():
        return torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)[0, 0].cpu().numpy()


def _vector_norm_np(field) -> np.ndarray:
    import torch

    with torch.no_grad():
        mag = torch.linalg.vector_norm(field.detach().permute(0, 2, 3, 1), ord=2, dim=-1)
        return torch.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)[0].cpu().numpy()


def _pixel_mesh_torch(height: int, width: int, dtype, device):
    import torch

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(xx)
    return xx, yy, torch.stack([xx, yy, ones], dim=-1).reshape(-1, 3).T


def _projective_warp_mag(matrix, height: int, width: int) -> np.ndarray:
    import torch

    with torch.no_grad():
        if matrix.shape[-2:] == (2, 3):
            bottom = torch.tensor([0.0, 0.0, 1.0], dtype=matrix.dtype, device=matrix.device).view(1, 1, 3).repeat(matrix.shape[0], 1, 1)
            mat = torch.cat([matrix, bottom], dim=1)
        else:
            mat = matrix
        xx, yy, pts = _pixel_mesh_torch(height, width, mat.dtype, mat.device)
        mapped = mat[0] @ pts
        mapped_x = (mapped[0] / mapped[2].clamp_min(1e-8)).reshape(height, width)
        mapped_y = (mapped[1] / mapped[2].clamp_min(1e-8)).reshape(height, width)
        dx = (mapped_x - xx) * (2.0 / max(1, width - 1))
        dy = (mapped_y - yy) * (2.0 / max(1, height - 1))
        mag = torch.sqrt(dx.square() + dy.square())
        return torch.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()


def _tps_control_warp_mag(points_src, points_dst, height: int, width: int) -> np.ndarray:
    import torch.nn.functional as F

    grid_size = int(round(points_src.shape[1] ** 0.5))
    offsets = (points_dst - points_src).reshape(1, grid_size, grid_size, 2).permute(0, 3, 1, 2)
    dense = F.interpolate(offsets, size=(height, width), mode="bilinear", align_corners=False)
    return _vector_norm_np(dense)


def _aligned_official_model(case: Case, model: np.ndarray, solver: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    import torch

    key = case.candidate.key
    computer, preset = _metric_computer_for_case(case)
    model_raw = _as_torch_image(model)
    solver_raw = _as_torch_image(solver)
    model_norm, solver_norm = computer._normalized_images(model_raw, solver_raw)
    _, _, height, width = model_raw.shape

    if key == "affine_dists":
        matrix, _ = computer._estimate_affine(model_norm, solver_norm)
        aligned = loss_impl._warp_affine_kornia(model_raw, matrix.to(dtype=model_raw.dtype, device=model_raw.device))
        return _tensor_image_to_np(aligned), _projective_warp_mag(matrix, height, width), f"offline_recomputed_kornia_affine_alignment_model_to_solver_budget_{preset}"

    if key == "local_warp_dists":
        disp, _ = computer._estimate_local_disp(model_norm, solver_norm)
        aligned = computer._warp_dense_monai_xy(model_raw, disp.to(dtype=model_raw.dtype, device=model_raw.device))
        return _tensor_image_to_np(aligned), _vector_norm_np(disp), f"offline_recomputed_monai_dense_warp_alignment_model_to_solver_budget_{preset}"

    if key == "homography_dists":
        matrix, _ = computer._estimate_homography(model_norm, solver_norm)
        aligned = loss_impl._warp_perspective_kornia(model_raw, matrix.to(dtype=model_raw.dtype, device=model_raw.device))
        return _tensor_image_to_np(aligned), _projective_warp_mag(matrix, height, width), f"offline_recomputed_kornia_homography_alignment_model_to_solver_budget_{preset}"

    if key == "tps_dists":
        points_src, points_dst, _ = computer._estimate_tps(model_norm, solver_norm)
        aligned = loss_impl._warp_tps_kornia(
            model_raw,
            points_src.to(dtype=model_raw.dtype, device=model_raw.device),
            points_dst.to(dtype=model_raw.dtype, device=model_raw.device),
        )
        return _tensor_image_to_np(aligned), _tps_control_warp_mag(points_src, points_dst, height, width), f"offline_recomputed_kornia_tps_alignment_model_to_solver_budget_{preset}"

    if key == "elastic_dists":
        noise, _ = computer._estimate_elastic_disp(model_norm, solver_norm)
        aligned = loss_impl._warp_elastic_kornia(
            model_raw,
            noise.to(dtype=model_raw.dtype, device=model_raw.device),
            computer.elastic_smooth_kernel,
            computer.elastic_smooth_passes,
        )
        return _tensor_image_to_np(aligned), _vector_norm_np(noise), f"offline_recomputed_kornia_elastic_alignment_model_to_solver_budget_{preset}"

    if key == "svf_dists":
        ddf, _ = computer._estimate_svf_disp(model_norm, solver_norm)
        aligned = computer._warp_dense_monai_xy(model_raw, ddf.to(dtype=model_raw.dtype, device=model_raw.device))
        return _tensor_image_to_np(aligned), _vector_norm_np(ddf), f"offline_recomputed_monai_svf_alignment_model_to_solver_budget_{preset}"

    return model.copy(), np.zeros_like(model), "no_explicit_alignment_for_this_metric"


def aligned_model_for_case(case: Case, model: np.ndarray, solver: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if case.candidate.key not in EXPLICIT_ALIGNMENT_KEYS:
        return model.copy(), np.zeros_like(model), "no_explicit_alignment_for_this_metric"
    try:
        return _aligned_official_model(case, model, solver)
    except Exception as exc:
        print(f"WARNING: alignment visualization failed for {case.candidate.key}: {exc}", file=sys.stderr)
        return model.copy(), np.zeros_like(model), f"alignment_visualization_failed_{type(exc).__name__}"


def field_bundle_for_clean(case: Case) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    x_clean = base.finite(case.data["x_clean"][SAMPLE_POSITION])
    model = base.finite(case.data["clean_model_final"][SAMPLE_POSITION])
    solver = base.finite(case.data["clean_solver_final"][SAMPLE_POSITION])
    diff = base.finite(case.data["clean_model_minus_solver"][SAMPLE_POSITION])
    fields = {
        "initial": x_clean,
        "delta": np.zeros_like(x_clean),
        "final": x_clean,
        "model": model,
        "solver": solver,
        "diff": diff,
        "aligned_model": model,
        "aligned_diff": diff,
        "warp_mag": np.zeros_like(x_clean),
    }
    return fields, dict(CLEAN_FIELD_KEYS)


def field_bundle_for_attack(case: Case) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    model = base.finite(case.data["adv_model_final"][SAMPLE_POSITION])
    solver = base.finite(case.data["adv_solver_final"][SAMPLE_POSITION])
    aligned_model, warp_mag, alignment_source = aligned_model_for_case(case, model, solver)
    fields = {
        "initial": base.finite(case.data["x_clean"][SAMPLE_POSITION]),
        "delta": base.finite(case.data["final_delta"][SAMPLE_POSITION]),
        "final": base.finite(case.data["x_adv"][SAMPLE_POSITION]),
        "model": model,
        "solver": solver,
        "diff": base.finite(case.data["adv_model_minus_solver"][SAMPLE_POSITION]),
        "aligned_model": base.finite(aligned_model),
        "aligned_diff": base.finite(aligned_model - solver),
        "warp_mag": base.finite(warp_mag),
    }
    keys = dict(ATTACK_FIELD_KEYS)
    keys["alignment_source"] = alignment_source
    return fields, keys


def add_field_sources(summary: dict[str, Any], source_keys: dict[str, str]) -> dict[str, Any]:
    for field_name, npz_key in source_keys.items():
        summary[f"{field_name}_npz_key"] = npz_key
    return summary



def build_rows(cases: list[Case]) -> list[dict[str, Any]]:
    base_case = cases[0]
    clean_fields, clean_sources = field_bundle_for_clean(base_case)
    clean_summary = add_field_sources(
        {
            "row_type": "clean",
            "method_label": "Clean baseline",
            "clean_wq_norm": base_case.clean_loss,
            "final_evasion_wq_norm": base_case.clean_loss,
            "wq_norm_increase": 0.0,
            "delta_l2": 0.0,
            "delta_linf": 0.0,
            "source_npz": relpath(base_case.candidate.path),
        },
        clean_sources,
    )
    rows = [
        {
            "kind": "clean",
            "label": "Clean baseline",
            "sub": f"Clean W/Q norm = {base.fmt(base_case.clean_loss)}",
            "sub2": "Before perturbation; clean fields come from the baseline run NPZ",
            **clean_fields,
            "source_keys": clean_sources,
            "is_baseline_attack": False,
            "summary": clean_summary,
        }
    ]
    for case in cases:
        fields, source_keys = field_bundle_for_attack(case)
        delta = fields["delta"]
        change = case.adv_loss - case.clean_loss
        active_initial = fnum(case.per_step[0].get("active_loss_mean"), fnum(case.per_step[0].get("loss3_mean"))) if case.per_step else math.nan
        active_final = fnum(case.per_step[-1].get("active_loss_mean"), fnum(case.per_step[-1].get("loss3_mean"))) if case.per_step else math.nan
        summary = add_field_sources(
            {
                "row_type": "attack",
                "method_key": case.candidate.key,
                "method_label": case.candidate.label,
                "full_loss_name": FULL_LOSS_NAMES[case.candidate.key],
                "clean_wq_norm": case.clean_loss,
                "final_evasion_wq_norm": case.adv_loss,
                "wq_norm_increase": change,
                "delta_l2": case.delta_l2,
                "delta_linf": case.delta_linf,
                "active_loss_initial_mean": active_initial,
                "active_loss_final_mean": active_final,
                "active_loss_ratio_to_k0": active_final / active_initial if active_initial else math.nan,
                "source_npz": relpath(case.candidate.path),
            },
            source_keys,
        )
        rows.append(
            {
                "kind": "attack",
                "label": case.candidate.label,
                "sub": FULL_LOSS_NAMES[case.candidate.key],
                "sub2": f"Clean W/Q {base.fmt(case.clean_loss)} -> final W/Q {base.fmt(case.adv_loss)}; delta L2={base.fmt(case.delta_l2)}, Linf={base.fmt(case.delta_linf)}",
                **fields,
                "source_keys": source_keys,
                "has_explicit_alignment": case.candidate.key in EXPLICIT_ALIGNMENT_KEYS,
                "is_baseline_attack": case.candidate.key == "baseline_loss3",
                "summary": summary,
            }
        )
    return rows


def render(cases: list[Case], out_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    rows = build_rows(cases)
    dataset_index = cases[0].dataset_index

    initial_lim = base.limits_sequential([r["initial"] for r in rows])
    delta_lim = base.limits_diverging([r["delta"] for r in rows])
    final_lim = base.limits_sequential([r["final"] for r in rows])
    explicit_rows = [r for r in rows if r.get("has_explicit_alignment")]
    aligned_state_arrays = [r["aligned_model"] for r in explicit_rows]
    aligned_diff_arrays = [r["aligned_diff"] for r in explicit_rows]
    warp_arrays = [r["warp_mag"] for r in explicit_rows]
    state_lim = base.limits_sequential([r["model"] for r in rows] + [r["solver"] for r in rows] + aligned_state_arrays)
    diff_lim = base.limits_diverging([r["diff"] for r in rows] + aligned_diff_arrays)
    warp_lim = base.limits_sequential(warp_arrays) if warp_arrays else (0.0, 1.0)
    lims = {
        "initial": (*initial_lim, "sequential"),
        "delta": (*delta_lim, "diverging"),
        "final": (*final_lim, "sequential"),
        "model": (*state_lim, "sequential"),
        "solver": (*state_lim, "sequential"),
        "diff": (*diff_lim, "diverging"),
        "aligned_model": (*state_lim, "sequential"),
        "aligned_diff": (*diff_lim, "diverging"),
        "warp_mag": (*warp_lim, "sequential"),
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
    draw.text((margin, 18), "NS2D Final-State Comparison | Epsilon = 32, Alpha = 10 | Optimizer = Steepest Add", font=base.FONT_TITLE, fill=(22, 27, 32))
    draw.text((margin, 54), f"Dataset index = {dataset_index}; baseline Loss 3/all-W plus nine alternative loss3 metrics", font=base.FONT_SUBTITLE, fill=(65, 68, 74))
    draw.text((margin, 78), "Top panels overlay all methods on common axes. Loss curve is the same all-W Loss 3 / W-Q norm for every method.", font=base.FONT_SUBTITLE, fill=(65, 68, 74))

    top_img = plot_top_panels(cases, top_w, top_h)
    canvas.paste(top_img, (x0, top_y))
    draw.rectangle((x0, top_y, x0 + top_w, top_y + top_h), outline=(70, 75, 82), width=1)

    explicit_row_indices = [idx for idx, row in enumerate(rows) if row.get("has_explicit_alignment")]
    explicit_bar_y = y_grid + explicit_row_indices[0] * row_h if explicit_row_indices else y_grid
    explicit_bar_h = (explicit_row_indices[-1] - explicit_row_indices[0] + 1) * row_h - row_gap if explicit_row_indices else grid_h

    for ci, (key, label, _) in enumerate(COLUMN_SPECS):
        x = x0 + ci * (col_w + col_gap)
        draw.text((x, y_cols + 4), label, font=base.FONT_COL, fill=(32, 37, 42))
        vmin, vmax, cmap = lims[key]
        bar_y = explicit_bar_y if key in ALIGNMENT_COLUMNS else y_grid
        bar_h_current = explicit_bar_h if key in ALIGNMENT_COLUMNS else grid_h
        bar = base.vertical_colorbar(bar_h_current, bar_w, cmap, vmin, vmax)
        bar_x = x + img + 6
        canvas.paste(bar, (bar_x, bar_y))
        draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h_current), outline=(75, 80, 85), width=1)
        draw.text((bar_x + bar_w + 3, bar_y - 2), base.fmt(vmax, 3), font=base.FONT_TINY, fill=(70, 73, 78))
        min_label = base.fmt(vmin, 3)
        bbox = draw.textbbox((0, 0), min_label, font=base.FONT_TINY)
        draw.text((bar_x + bar_w + 3, bar_y + bar_h_current - (bbox[3] - bbox[1]) + 1), min_label, font=base.FONT_TINY, fill=(70, 73, 78))

    for ri, row in enumerate(rows):
        y = y_grid + ri * row_h
        label_box = (margin - 4, y + 4, margin + row_label_w - 14, y + img - 4)
        if ri == 0:
            draw.rounded_rectangle(label_box, radius=6, fill=(244, 247, 250), outline=(190, 198, 206), width=1)
        elif row.get("is_baseline_attack"):
            draw.rounded_rectangle(label_box, radius=6, fill=(255, 248, 247), outline=(203, 34, 47), width=3)
        else:
            draw.rounded_rectangle(label_box, radius=6, fill=(250, 250, 248), outline=(226, 228, 230), width=1)
        if ri == 1:
            draw.line((margin, y - 8, width - margin, y - 8), fill=(210, 215, 220), width=1)
        fills = [(35, 39, 44), (45, 48, 52), (184, 24, 36) if row.get("is_baseline_attack") else (68, 72, 78)]
        draw_wrapped_row_label(
            draw,
            (margin + 8, y + 18),
            row["label"],
            row["sub"],
            row.get("sub2", ""),
            row_label_w - 42,
            fills[0],
            fills[1],
            fills[2],
        )
        for ci, (key, _, _) in enumerate(COLUMN_SPECS):
            x = x0 + ci * (col_w + col_gap)
            if key in ALIGNMENT_COLUMNS and not row.get("has_explicit_alignment"):
                continue
            vmin, vmax, cmap = lims[key]
            tile = Image.fromarray(base.colorize(row[key], cmap, vmin, vmax), mode="RGB").resize((img, img), Image.Resampling.BILINEAR)
            canvas.paste(tile, (x, y))
            outline = (203, 34, 47) if row.get("is_baseline_attack") else (40, 44, 49)
            draw.rectangle((x, y, x + img, y + img), outline=outline, width=3 if row.get("is_baseline_attack") else 1)

    out_path = out_dir / "eps32_alpha10_steepest_add_altloss_original_cleanstyle_dataset0.png"
    canvas.save(out_path)

    summary_rows = []
    for row in rows:
        s = dict(row["summary"])
        s.update({
            "dataset_index": dataset_index,
            "epsilon": 32,
            "alpha": 10,
            "optimizer": "steepest_add",
            "output_png": relpath(out_path),
        })
        summary_rows.append(s)
    return out_path, summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--alt-root", type=Path, default=DEFAULT_ALT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(args.baseline_dir, args.alt_root)
    cases = [load_case(candidate, SAMPLE_POSITION) for candidate in candidates]
    png, rows = render(cases, args.out_dir)
    summary_csv = args.out_dir / "row_loss_summary.csv"
    write_csv(summary_csv, rows)
    manifest = {
        "created_by": relpath(SCRIPT),
        "derived_from": relpath(ORIG_SCRIPT),
        "output_dir": relpath(args.out_dir),
        "png": relpath(png),
        "summary_csv": relpath(summary_csv),
        "layout": "Original NS2D cleanstyle PIL layout: 2x2 top overlay panels plus heatmap rows, 220px tiles, 540px top panel block.",
        "heatmap_columns": [label for _, label, _ in COLUMN_SPECS],
        "full_loss_names": FULL_LOSS_NAMES,
        "source_policy": "Each attack heatmap row is built from one run's own final_state_outputs.npz. The first six columns map to x_clean, final_delta, x_adv, adv_model_final, adv_solver_final, and adv_model_minus_solver. The alignment columns are shown only for methods with explicit warp/alignment, affine_dists, local_warp_dists, homography_dists, tps_dists, elastic_dists, and svf_dists; rows without explicit warp leave those cells blank. Alignment fields are recomputed offline from that same model-solver pair for visualization using the corresponding official Kornia/MONAI warp backend and the budget preset inferred from the run directory; the attack NPZ does not store the internal alignment fields. The clean row uses clean_model_final, clean_solver_final, and clean_model_minus_solver from the baseline NPZ, with alignment cells blank.",
        "attack_field_npz_keys": ATTACK_FIELD_KEYS,
        "clean_field_npz_keys": CLEAN_FIELD_KEYS,
        "top_panels": [
            "Single index delta spectrum, all methods overlaid",
            "All samples mean delta spectrum, all methods overlaid",
            "Single index all-W Loss 3 / W-Q norm curve, recomputed from step_sample_trace.npz",
            "Final batch mean all-W Loss 3 / W-Q norm bar chart",
        ],
        "loss_curve_policy": "The loss curve is not each method's active optimization objective. It is the same all-W Loss 3 / W-Q norm for every method, recomputed for sample_position=0 as ||adv_model_final - adv_solver_final||_2 from step_sample_trace.npz. Batch per-step all-W Loss 3 was not saved for all samples, so the batch panel uses final_state_metrics.csv final adv_true_loss means instead of a fake batch curve.",
        "runs": [{"key": c.candidate.key, "label": c.candidate.label, "source_npz": relpath(c.candidate.path)} for c in cases],
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
