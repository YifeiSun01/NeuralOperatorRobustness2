#!/usr/bin/env python3
"""Plot before/after alignment diagnostics for NS2D explicit-warp losses."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
ALT_PLOT_SCRIPT = ROOT / "tools" / "plot_ns2d_eps32_alpha10_altloss_heatmaps_spectrum_loss_curves_cleanstyle_20260525.py"

spec = importlib.util.spec_from_file_location("ns2d_altloss_plot_20260525", ALT_PLOT_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {ALT_PLOT_SCRIPT}")
alt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = alt
spec.loader.exec_module(alt)
base = alt.base

EXPLICIT_KEYS = tuple(key for key in alt.ALT_METRICS if key in alt.EXPLICIT_ALIGNMENT_KEYS)
FRAME_INDICES = (0, -1)
COLUMNS = [
    ("model", "Model Output\nBefore Warp", "sequential"),
    ("aligned", "Aligned Model\nAfter Warp", "sequential"),
    ("diff", "Aligned - Model\nPointwise Diff", "diverging"),
    ("solver", "Solver Output", "sequential"),
    ("model_solver_diff", "Model - Solver\nBefore Warp", "diverging"),
    ("aligned_solver_diff", "Aligned Model\n- Solver", "diverging"),
    ("warp_mag", "Warp Magnitude", "sequential"),
]


def finite(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def fnum(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(cases: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected = [case for case in cases if case.candidate.key in EXPLICIT_KEYS]
    for case in selected:
        model_stack = np.asarray(case.step_trace["adv_model_final"])
        solver_stack = np.asarray(case.step_trace["adv_solver_final"])
        k_values = np.asarray(case.step_trace["k"])
        for frame_index in FRAME_INDICES:
            real_index = model_stack.shape[0] + frame_index if frame_index < 0 else frame_index
            model = finite(model_stack[real_index])
            solver = finite(solver_stack[real_index])
            aligned, warp_mag, source = alt.aligned_model_for_case(case, model, solver)
            aligned = finite(aligned)
            warp_mag = finite(warp_mag)
            diff = finite(aligned - model)
            model_solver_diff = finite(model - solver)
            aligned_solver_diff = finite(aligned - solver)
            abs_diff = np.abs(diff.astype(np.float64))
            abs_model_solver = np.abs(model_solver_diff.astype(np.float64))
            abs_aligned_solver = np.abs(aligned_solver_diff.astype(np.float64))
            abs_warp = np.abs(warp_mag.astype(np.float64))
            k = int(k_values[real_index]) if k_values.ndim > 0 else int(real_index)
            summary = {
                "method_key": case.candidate.key,
                "method_label": case.candidate.label,
                "frame_index": int(real_index),
                "attack_step_k": k,
                "mean_abs_aligned_minus_model": float(abs_diff.mean()),
                "max_abs_aligned_minus_model": float(abs_diff.max()),
                "l2_aligned_minus_model": float(np.linalg.norm(diff.reshape(-1).astype(np.float64))),
                "mean_abs_model_minus_solver": float(abs_model_solver.mean()),
                "max_abs_model_minus_solver": float(abs_model_solver.max()),
                "l2_model_minus_solver": float(np.linalg.norm(model_solver_diff.reshape(-1).astype(np.float64))),
                "mean_abs_aligned_minus_solver": float(abs_aligned_solver.mean()),
                "max_abs_aligned_minus_solver": float(abs_aligned_solver.max()),
                "l2_aligned_minus_solver": float(np.linalg.norm(aligned_solver_diff.reshape(-1).astype(np.float64))),
                "l2_aligned_minus_solver_delta_vs_raw": float(
                    np.linalg.norm(aligned_solver_diff.reshape(-1).astype(np.float64))
                    - np.linalg.norm(model_solver_diff.reshape(-1).astype(np.float64))
                ),
                "nonzero_pixels_gt_1e-8": int((abs_diff > 1e-8).sum()),
                "pixels": int(abs_diff.size),
                "nonzero_fraction_gt_1e-8": float((abs_diff > 1e-8).mean()),
                "warp_mean": float(abs_warp.mean()),
                "warp_max": float(abs_warp.max()),
                "alignment_source": source,
                "source_npz": str(case.candidate.path.relative_to(ROOT)),
            }
            rows.append({
                "label": f"{case.candidate.label} | step {k}",
                "notes": [
                    f"Loss3/W-Q L2 model-solver: {base.fmt(summary['l2_model_minus_solver'], 4)}",
                    f"Loss3/W-Q L2 aligned-solver: {base.fmt(summary['l2_aligned_minus_solver'], 4)}",
                    f"aligned-model mean abs: {base.fmt(summary['mean_abs_aligned_minus_model'], 4)}",
                    f"aligned-model max abs: {base.fmt(summary['max_abs_aligned_minus_model'], 4)}",
                    f"nonzero pixels: {summary['nonzero_pixels_gt_1e-8']}/{summary['pixels']}",
                    f"warp mean/max: {base.fmt(summary['warp_mean'], 4)} / {base.fmt(summary['warp_max'], 4)}",
                ],
                "model": model,
                "aligned": aligned,
                "diff": diff,
                "solver": solver,
                "model_solver_diff": model_solver_diff,
                "aligned_solver_diff": aligned_solver_diff,
                "warp_mag": warp_mag,
                "summary": summary,
            })
            summary_rows.append(summary)
    return rows, summary_rows


def draw_multiline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: Any, fill: tuple[int, int, int], line_gap: int = 4) -> None:
    x, y = xy
    for line in str(text).split("\n"):
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += bbox[3] - bbox[1] + line_gap


def render(cases: list[Any], out_dir: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    rows, summary_rows = build_rows(cases)
    if not rows:
        raise RuntimeError("No explicit-warp rows found")

    state_lim = base.limits_sequential([r["model"] for r in rows] + [r["aligned"] for r in rows] + [r["solver"] for r in rows])
    diff_lim = base.limits_diverging(
        [r["diff"] for r in rows]
        + [r["model_solver_diff"] for r in rows]
        + [r["aligned_solver_diff"] for r in rows]
    )
    warp_lim = base.limits_sequential([r["warp_mag"] for r in rows])
    lims = {
        "model": (*state_lim, "sequential"),
        "aligned": (*state_lim, "sequential"),
        "diff": (*diff_lim, "diverging"),
        "solver": (*state_lim, "sequential"),
        "model_solver_diff": (*diff_lim, "diverging"),
        "aligned_solver_diff": (*diff_lim, "diverging"),
        "warp_mag": (*warp_lim, "sequential"),
    }

    margin = 34
    row_label_w = 560
    img = 236
    bar_w = 13
    col_gap = 28
    row_gap = 24
    header_h = 132
    col_label_h = 62
    row_h = img + row_gap
    grid_h = len(rows) * row_h - row_gap
    col_w = img + 18 + bar_w
    x0 = margin + row_label_w
    y_cols = header_h
    y_grid = y_cols + col_label_h
    width = margin * 2 + row_label_w + len(COLUMNS) * col_w + (len(COLUMNS) - 1) * col_gap
    height = header_h + col_label_h + grid_h + margin

    canvas = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "NS2D Explicit Alignment Diagnostic | Warp Before/After Difference", font=base.FONT_TITLE, fill=(22, 27, 32))
    draw.text((margin, 54), "Rows show attack step 0 and the final saved step for every explicit warp method: Affine, Local warp, Homography, TPS, Elastic, and SVF.", font=base.FONT_SUBTITLE, fill=(65, 68, 74))
    draw.text((margin, 78), "Loss labels use the same Loss3/W-Q L2 distance before and after the plotting-time alignment warp; alignment is recomputed for visualization with the matching official backend.", font=base.FONT_SUBTITLE, fill=(65, 68, 74))
    draw.text((margin, 102), "Aligned - Model shows the direct pixelwise effect of the warp; Model - Solver and Aligned - Solver show raw vs aligned errors.", font=base.FONT_SUBTITLE, fill=(65, 68, 74))

    for ci, (key, label, _) in enumerate(COLUMNS):
        x = x0 + ci * (col_w + col_gap)
        draw_multiline(draw, (x, y_cols + 4), label, base.FONT_COL, (32, 37, 42), line_gap=2)
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
        draw.rounded_rectangle(label_box, radius=6, fill=(250, 250, 248), outline=(218, 222, 226), width=1)
        draw.text((margin + 8, y + 14), row["label"], font=base.FONT_ROW, fill=(35, 39, 44))
        note_y = y + 48
        for note in row["notes"]:
            draw.text((margin + 8, note_y), note, font=base.FONT_SMALL, fill=(54, 58, 64))
            note_y += 26
        for ci, (key, _, _) in enumerate(COLUMNS):
            x = x0 + ci * (col_w + col_gap)
            vmin, vmax, cmap = lims[key]
            tile = Image.fromarray(base.colorize(row[key], cmap, vmin, vmax), mode="RGB").resize((img, img), Image.Resampling.BILINEAR)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + img, y + img), outline=(40, 44, 49), width=1)

    out_png = out_dir / "eps32_alpha10_steepest_add_alignment_before_after_diff_dataset0.png"
    out_csv = out_dir / "alignment_before_after_diff_summary.csv"
    canvas.save(out_png)
    write_csv(out_csv, summary_rows)
    return out_png, out_csv, summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=alt.DEFAULT_BASELINE_DIR)
    parser.add_argument("--alt-root", type=Path, default=alt.DEFAULT_ALT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=alt.DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = alt.build_candidates(args.baseline_dir, args.alt_root)
    cases = [alt.load_case(candidate, alt.SAMPLE_POSITION) for candidate in candidates]
    png, csv_path, rows = render(cases, args.out_dir)
    manifest = {
        "created_by": str(SCRIPT.relative_to(ROOT)),
        "uses_alignment_functions_from": str(ALT_PLOT_SCRIPT.relative_to(ROOT)),
        "png": str(png.relative_to(ROOT)),
        "summary_csv": str(csv_path.relative_to(ROOT)),
        "frame_indices": list(FRAME_INDICES),
        "explicit_methods": list(EXPLICIT_KEYS),
        "columns": [label for _, label, _ in COLUMNS],
    }
    manifest_path = args.out_dir / "alignment_before_after_diff_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
