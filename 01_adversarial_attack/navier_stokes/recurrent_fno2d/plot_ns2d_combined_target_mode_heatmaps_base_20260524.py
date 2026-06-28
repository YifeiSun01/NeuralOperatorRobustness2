#!/usr/bin/env python3
"""Render compact combined NS2D target/mode heatmaps.

One PNG is produced for each available epsilon/alpha x optimizer method. Each
PNG uses one fixed sample and one clean initial condition, then stacks all
available loss/mode attack rows for that epsilon/method.

Offline visualization only: reads final_state_outputs.npz/final_state_metrics.csv
and does not run model inference, solver rollout, PyTorch, JAX, or GPU work.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "2D_NS_FNO2d_recurrent" / "perturbation_results" / "ns2d_recurrent_core4_attack"
RUN_ROOTS = [
    DATA_ROOT / "full_adw_b10_pair_outer_baseline_first_20260522",
    DATA_ROOT / "minimal_mechanism_eps4_eps2_eps1_b10_20260523_111410_UTC",
]
OUT_DIR = ROOT / "docs" / "ns2d_combined_target_mode_heatmaps_20260524"
REPORT_PATH = ROOT / "docs" / "ns2d_combined_target_mode_heatmaps_20260524.md"
SAMPLE_POSITION = 0
METHOD_ORDER = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
EPS_ORDER = [
    "eps32_alpha10",
    "eps16_alpha5",
    "eps8_alpha2p5",
    "eps4_alpha1p25",
    "eps2_alpha0p625",
    "eps1_alpha0p3125",
]
MODE_LABELS = {
    "wwwwwwwwww": "all_w",
    "dddddddddw": "all_d_target_w",
    "aaaaaaaaaw": "all_a_target_w",
    "wwwwwddddw": "w1_5_d6_9_target_w",
    "dddddwwwww": "d1_5_w6_9_target_w",
    "aaaaaddddw": "a1_5_d6_9_target_w",
}
CANDIDATE_ORDER = [
    "loss1/all_w",
    "loss2/all_a_target_w",
    "loss3/all_w",
    "loss3/all_d_target_w",
    "loss3/all_a_target_w",
    "loss3/w1_5_d6_9_target_w",
    "loss3/d1_5_w6_9_target_w",
    "loss3/a1_5_d6_9_target_w",
]
COLUMN_SPECS = [
    ("input", "initial / x_adv", "sequential"),
    ("delta", "final_delta", "diverging"),
    ("model", "model output", "sequential"),
    ("solver", "solver output", "sequential"),
    ("diff", "model - solver", "diverging"),
]

FONT_REGULAR_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def font(path: Path, size: int):
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(FONT_BOLD_PATH, 24)
FONT_SUBTITLE = font(FONT_REGULAR_PATH, 14)
FONT_COL = font(FONT_BOLD_PATH, 13)
FONT_ROW = font(FONT_BOLD_PATH, 12)
FONT_SMALL = font(FONT_REGULAR_PATH, 11)
FONT_TINY = font(FONT_REGULAR_PATH, 9)


@dataclass(frozen=True)
class Candidate:
    eps_label: str
    method: str
    loss_type: str
    mode_spec: str
    label: str
    path: Path


@dataclass
class RenderSummary:
    eps_label: str
    method: str
    dataset_index: int
    n_attack_rows: int
    output_png: Path
    clean_loss: float
    rows: List[dict]


def fnum(value: Any, fallback: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def fmt(value: Any, digits: int = 4) -> str:
    x = fnum(value)
    if not math.isfinite(x):
        return "NA"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.2e}"
    return f"{x:.{digits}g}"


def eps_sort_key(label: str) -> tuple[int, str]:
    try:
        return (EPS_ORDER.index(label), label)
    except ValueError:
        return (999, label)


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (999, method)


def candidate_sort_key(c: Candidate) -> tuple[int, str]:
    try:
        return (CANDIDATE_ORDER.index(c.label), c.label)
    except ValueError:
        return (999, c.label)


def parse_eps(path: Path) -> Optional[str]:
    for part in path.parts:
        if re.match(r"^eps[^/]+_alpha[^/]+$", part):
            return part
    return None


def parse_mode(path: Path) -> Optional[str]:
    for part in path.parts:
        if part.startswith("mode_"):
            body = part[len("mode_"):]
            return body.split("_", 1)[0]
    return None


def classify(loss_type: str, mode_spec: str) -> Optional[str]:
    if loss_type == "loss1" and mode_spec == "wwwwwwwwww":
        return "loss1/all_w"
    if loss_type == "loss2" and mode_spec == "aaaaaaaaaw":
        return "loss2/all_a_target_w"
    if loss_type == "loss3":
        mode_label = MODE_LABELS.get(mode_spec)
        if mode_label:
            return f"loss3/{mode_label}"
    return None


def discover_candidates() -> List[Candidate]:
    candidates: List[Candidate] = []
    for run_root in RUN_ROOTS:
        if not run_root.exists():
            continue
        for path in sorted(run_root.rglob("final_state_outputs.npz")):
            eps_label = parse_eps(path)
            mode_spec = parse_mode(path)
            if eps_label is None or mode_spec is None:
                continue
            if eps_label not in EPS_ORDER:
                continue
            try:
                loss_type = path.parts[-3]
                method = path.parts[-2]
            except IndexError:
                continue
            if method not in METHOD_ORDER or loss_type not in {"loss1", "loss2", "loss3"}:
                continue
            label = classify(loss_type, mode_spec)
            if label is None:
                continue
            candidates.append(Candidate(eps_label, method, loss_type, mode_spec, label, path))
    return candidates


def finite(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def widen(vmin: float, vmax: float) -> tuple[float, float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return -1.0, 1.0
    if abs(vmax - vmin) < 1e-12:
        bump = max(abs(vmin), 1.0) * 1e-3
        return vmin - bump, vmax + bump
    return vmin, vmax


def limits_sequential(arrays: Iterable[np.ndarray]) -> tuple[float, float]:
    vals = [finite(a) for a in arrays]
    return widen(min(float(a.min()) for a in vals), max(float(a.max()) for a in vals))


def limits_diverging(arrays: Iterable[np.ndarray]) -> tuple[float, float]:
    vals = [finite(a) for a in arrays]
    lim = max(float(np.max(np.abs(a))) for a in vals)
    if not math.isfinite(lim) or lim <= 0:
        lim = 1.0
    return -lim, lim


def interp(a: tuple[int, int, int], b: tuple[int, int, int], t: np.ndarray) -> np.ndarray:
    aa = np.array(a, dtype=np.float64)
    bb = np.array(b, dtype=np.float64)
    return aa + (bb - aa) * t[..., None]


def colorize_diverging(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = finite(arr)
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    blue, white, red = (49, 112, 187), (248, 248, 248), (202, 0, 32)
    rgb = np.zeros(arr.shape + (3,), dtype=np.float64)
    lower = norm <= 0.5
    tl = np.where(lower, norm / 0.5, 0.0)
    tu = np.where(~lower, (norm - 0.5) / 0.5, 0.0)
    rgb[lower] = interp(blue, white, tl)[lower]
    rgb[~lower] = interp(white, red, tu)[~lower]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def colorize_seq(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = finite(arr)
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    stops = [(36, 35, 88), (31, 111, 163), (43, 161, 129), (253, 231, 37)]
    pos = norm * (len(stops) - 1)
    idx = np.minimum(np.floor(pos).astype(int), len(stops) - 2)
    t = pos - idx
    rgb = np.zeros(arr.shape + (3,), dtype=np.float64)
    for i in range(len(stops) - 1):
        mask = idx == i
        if np.any(mask):
            rgb[mask] = interp(stops[i], stops[i + 1], t)[mask]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def colorize(arr: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    if cmap == "diverging":
        return colorize_diverging(arr, vmin, vmax)
    return colorize_seq(arr, vmin, vmax)


def vertical_colorbar(height: int, width: int, cmap: str, vmin: float, vmax: float) -> Image.Image:
    grad = np.tile(np.linspace(vmax, vmin, height, dtype=np.float64)[:, None], (1, width))
    return Image.fromarray(colorize(grad, cmap, vmin, vmax), mode="RGB")


def read_metric_row(metrics_path: Path) -> dict[str, str]:
    if not metrics_path.exists():
        return {}
    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if int(fnum(row.get("sample_position", -1))) == SAMPLE_POSITION:
            return row
    return rows[SAMPLE_POSITION] if len(rows) > SAMPLE_POSITION else {}


def draw_text_lines(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], font_obj, fill=(45, 48, 52), gap=2) -> int:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font_obj, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font_obj)
        y += (bbox[3] - bbox[1]) + gap
    return y


def load_case(candidate: Candidate) -> dict[str, Any]:
    data = np.load(candidate.path)
    dataset_index = int(np.asarray(data["dataset_indices"])[SAMPLE_POSITION])
    metric = read_metric_row(candidate.path.with_name("final_state_metrics.csv"))
    clean_loss = fnum(metric.get("clean_true_loss"), float(data["clean_true_loss"][SAMPLE_POSITION]))
    adv_loss = fnum(metric.get("adv_true_loss"), float(data["adv_true_loss"][SAMPLE_POSITION]))
    inc = fnum(metric.get("true_loss_increase"), adv_loss - clean_loss)
    ratio = fnum(metric.get("true_loss_ratio"), adv_loss / clean_loss if clean_loss else math.nan)
    delta_l2 = fnum(metric.get("delta_l2"))
    delta_linf = fnum(metric.get("delta_linf"))
    return {
        "candidate": candidate,
        "data": data,
        "dataset_index": dataset_index,
        "clean_loss": clean_loss,
        "adv_loss": adv_loss,
        "increase": inc,
        "ratio": ratio,
        "delta_l2": delta_l2,
        "delta_linf": delta_linf,
    }


def render_group(eps_label: str, method: str, candidates: list[Candidate]) -> RenderSummary:
    cases = [load_case(c) for c in sorted(candidates, key=candidate_sort_key)]
    base = cases[0]
    base_data = base["data"]
    dataset_index = base["dataset_index"]
    x_clean = finite(base_data["x_clean"][SAMPLE_POSITION])
    clean_model = finite(base_data["clean_model_final"][SAMPLE_POSITION])
    clean_solver = finite(base_data["clean_solver_final"][SAMPLE_POSITION])
    clean_diff = finite(base_data["clean_model_minus_solver"][SAMPLE_POSITION])
    zero_delta = np.zeros_like(x_clean)
    clean_loss = base["clean_loss"]

    rows: list[dict[str, Any]] = [{
        "kind": "clean",
        "label": "clean baseline",
        "sub": f"true_loss={fmt(clean_loss)}",
        "input": x_clean,
        "delta": zero_delta,
        "model": clean_model,
        "solver": clean_solver,
        "diff": clean_diff,
        "summary": {
            "row_type": "clean",
            "candidate_label": "clean baseline",
            "clean_true_loss": clean_loss,
            "adv_true_loss": clean_loss,
            "true_loss_increase": 0.0,
            "true_loss_ratio": 1.0,
            "delta_l2": 0.0,
            "delta_linf": 0.0,
            "source_npz": str(cases[0]["candidate"].path.relative_to(ROOT)),
        },
    }]

    for case in cases:
        d = case["data"]
        c = case["candidate"]
        rows.append({
            "kind": "attack",
            "label": c.label,
            "sub": f"true_loss {fmt(case['clean_loss'])}->{fmt(case['adv_loss'])}  +{fmt(case['increase'])}  x{fmt(case['ratio'], 3)}",
            "sub2": f"||delta||2={fmt(case['delta_l2'])}, ||delta||inf={fmt(case['delta_linf'])}",
            "input": finite(d["x_adv"][SAMPLE_POSITION]),
            "delta": finite(d["final_delta"][SAMPLE_POSITION]),
            "model": finite(d["adv_model_final"][SAMPLE_POSITION]),
            "solver": finite(d["adv_solver_final"][SAMPLE_POSITION]),
            "diff": finite(d["adv_model_minus_solver"][SAMPLE_POSITION]),
            "summary": {
                "row_type": "attack",
                "candidate_label": c.label,
                "loss_type": c.loss_type,
                "mode_spec": c.mode_spec,
                "method": c.method,
                "clean_true_loss": case["clean_loss"],
                "adv_true_loss": case["adv_loss"],
                "true_loss_increase": case["increase"],
                "true_loss_ratio": case["ratio"],
                "delta_l2": case["delta_l2"],
                "delta_linf": case["delta_linf"],
                "source_npz": str(c.path.relative_to(ROOT)),
            },
        })

    # Shared scales within each figure; model and solver intentionally share a scale.
    input_lim = limits_sequential([r["input"] for r in rows])
    delta_lim = limits_diverging([r["delta"] for r in rows])
    state_lim = limits_sequential([r["model"] for r in rows] + [r["solver"] for r in rows])
    diff_lim = limits_diverging([r["diff"] for r in rows])
    lims = {
        "input": (*input_lim, "sequential"),
        "delta": (*delta_lim, "diverging"),
        "model": (*state_lim, "sequential"),
        "solver": (*state_lim, "sequential"),
        "diff": (*diff_lim, "diverging"),
    }

    row_label_w = 285
    img = 136
    bar_w = 9
    col_gap = 18
    row_gap = 14
    header_h = 112
    col_label_h = 28
    margin = 22
    row_h = img + row_gap
    nrows = len(rows)
    grid_h = nrows * row_h - row_gap
    col_w = img + 14 + bar_w
    width = margin * 2 + row_label_w + len(COLUMN_SPECS) * col_w + (len(COLUMN_SPECS) - 1) * col_gap
    height = header_h + col_label_h + grid_h + margin
    canvas = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)

    title = f"NS2D combined final-state heatmaps | {eps_label} | {method}"
    draw.text((margin, 18), title, font=FONT_TITLE, fill=(22, 27, 32))
    draw.text((margin, 50), f"sample_position={SAMPLE_POSITION}, dataset_index={dataset_index}; first row is the same clean initial condition", font=FONT_SUBTITLE, fill=(65, 68, 74))
    draw.text((margin, 70), "Rows below clean baseline show x_adv = clean initial condition + final_delta. Loss text is final all-W true_loss only; no per-state mean/std is shown.", font=FONT_SUBTITLE, fill=(65, 68, 74))

    x0 = margin + row_label_w
    y_cols = header_h
    y_grid = header_h + col_label_h
    for ci, (key, label, cmap_name) in enumerate(COLUMN_SPECS):
        x = x0 + ci * (col_w + col_gap)
        draw.text((x, y_cols + 3), label, font=FONT_COL, fill=(32, 37, 42))
        vmin, vmax, cmap = lims[key]
        # One vertical colorbar per column, shared over all rows.
        bar = vertical_colorbar(grid_h, bar_w, cmap, vmin, vmax)
        bar_x = x + img + 5
        canvas.paste(bar, (bar_x, y_grid))
        draw.rectangle((bar_x, y_grid, bar_x + bar_w, y_grid + grid_h), outline=(75, 80, 85), width=1)
        draw.text((bar_x + bar_w + 2, y_grid - 2), fmt(vmax, 3), font=FONT_TINY, fill=(70, 73, 78))
        min_label = fmt(vmin, 3)
        bbox = draw.textbbox((0, 0), min_label, font=FONT_TINY)
        draw.text((bar_x + bar_w + 2, y_grid + grid_h - (bbox[3] - bbox[1]) + 1), min_label, font=FONT_TINY, fill=(70, 73, 78))

    for ri, row in enumerate(rows):
        y = y_grid + ri * row_h
        label_x = margin
        label_lines = [row["label"], row["sub"]]
        if row.get("sub2"):
            label_lines.append(row["sub2"])
        draw_text_lines(draw, (label_x, y + 14), label_lines, FONT_ROW if ri == 0 else FONT_SMALL, fill=(35, 39, 44), gap=3)
        if ri == 0:
            draw.line((margin, y - 7, width - margin, y - 7), fill=(210, 215, 220), width=1)
        for ci, (key, _, _) in enumerate(COLUMN_SPECS):
            x = x0 + ci * (col_w + col_gap)
            vmin, vmax, cmap = lims[key]
            tile = Image.fromarray(colorize(row[key], cmap, vmin, vmax), mode="RGB").resize((img, img), Image.Resampling.BILINEAR)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + img, y + img), outline=(40, 44, 49), width=1)

    out_name = f"{eps_label}_{method}_combined_target_mode_sample{SAMPLE_POSITION}_idx{dataset_index}.png"
    out_path = OUT_DIR / out_name
    canvas.save(out_path)

    summaries = []
    for row in rows:
        s = dict(row["summary"])
        s.update({
            "eps_label": eps_label,
            "method": method,
            "sample_position": SAMPLE_POSITION,
            "dataset_index": dataset_index,
            "output_png": str(out_path.relative_to(ROOT)),
        })
        summaries.append(s)

    return RenderSummary(eps_label, method, dataset_index, len(rows) - 1, out_path, clean_loss, summaries)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rel_from_report(path: Path) -> str:
    return path.relative_to(REPORT_PATH.parent).as_posix()


def write_report(rendered: list[RenderSummary], missing: list[dict[str, Any]], summary_csv: Path, manifest_path: Path) -> None:
    lines: list[str] = []
    lines.append("# NS2D combined target/mode heatmaps (2026-05-24)")
    lines.append("")
    lines.append("Observed from saved `final_state_outputs.npz` and `final_state_metrics.csv` files only. No model inference, solver rollout, attack update, PyTorch, JAX, or GPU work was run.")
    lines.append("")
    lines.append("Each PNG is one `epsilon/alpha x optimizer` figure. The first row is the clean baseline using the same initial condition. Subsequent rows show all locally available loss/mode candidates for that epsilon/method. Columns are `initial/x_adv`, `final_delta`, `model output`, `solver output`, and `model - solver`.")
    lines.append("")
    lines.append("The loss text in each row is the final all-W `true_loss`, plus increase over the clean baseline and the ratio. Per-state mean/std statistics are intentionally not shown.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Summary CSV: [{summary_csv.name}]({rel_from_report(summary_csv)})")
    lines.append(f"- Manifest JSON: [{manifest_path.name}]({rel_from_report(manifest_path)})")
    lines.append(f"- PNG directory: `{OUT_DIR.relative_to(ROOT).as_posix()}/`")
    lines.append("")
    lines.append("## Rendered Figures")
    lines.append("")
    lines.append("| eps/alpha | method | attack rows | PNG |")
    lines.append("|---|---|---:|---|")
    for item in rendered:
        lines.append(f"| `{item.eps_label}` | `{item.method}` | {item.n_attack_rows} | [{item.output_png.name}]({rel_from_report(item.output_png)}) |")
    lines.append("")
    lines.append("## Data Coverage Notes")
    lines.append("")
    lines.append("- `eps32_alpha10` and `eps8_alpha2p5` include the extra available `loss3` A/D/W modes in addition to `loss1/all_w`, `loss2/all_a_target_w`, and `loss3/all_w`.")
    lines.append("- `eps16_alpha5`, `eps4_alpha1p25`, and `eps2_alpha0p625` only have the canonical three rows in the organized local artifacts used here.")
    lines.append("- `eps1_alpha0p3125` only has `loss1/all_w` final-state outputs locally; no matching `loss2` or `loss3` final-state outputs were found in the organized run, so those rows cannot be drawn from current local evidence.")
    if missing:
        lines.append("")
        lines.append("## Missing Expected Canonical Rows")
        lines.append("")
        for item in missing:
            lines.append(f"- `{item['eps_label']}` `{item['method']}` missing `{item['label']}`")
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = discover_candidates()
    by_group: Dict[tuple[str, str], list[Candidate]] = {}
    for c in candidates:
        by_group.setdefault((c.eps_label, c.method), []).append(c)

    rendered: list[RenderSummary] = []
    all_summary_rows: list[dict[str, Any]] = []
    for key in sorted(by_group, key=lambda k: (eps_sort_key(k[0]), method_sort_key(k[1]))):
        eps_label, method = key
        group = sorted(by_group[key], key=candidate_sort_key)
        if not group:
            continue
        item = render_group(eps_label, method, group)
        rendered.append(item)
        all_summary_rows.extend(item.rows)

    missing = []
    expected_canonical = ["loss1/all_w", "loss2/all_a_target_w", "loss3/all_w"]
    for eps in EPS_ORDER:
        for method in METHOD_ORDER:
            present = {c.label for c in by_group.get((eps, method), [])}
            if not present:
                continue
            for label in expected_canonical:
                if label not in present:
                    missing.append({"eps_label": eps, "method": method, "label": label})

    summary_csv = OUT_DIR / "row_loss_summary.csv"
    manifest_path = OUT_DIR / "manifest.json"
    write_csv(summary_csv, all_summary_rows)
    manifest = {
        "created_by": "tools/plot_ns2d_combined_target_mode_heatmaps.py",
        "created_date": "2026-05-24",
        "sample_position": SAMPLE_POSITION,
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "rendered_png_count": len(rendered),
        "summary_csv": str(summary_csv.relative_to(ROOT)),
        "report_md": str(REPORT_PATH.relative_to(ROOT)),
        "run_roots": [str(p.relative_to(ROOT)) for p in RUN_ROOTS],
        "candidate_order": CANDIDATE_ORDER,
        "column_specs": [list(c) for c in COLUMN_SPECS],
        "missing_expected_canonical_rows": missing,
        "rendered": [
            {
                "eps_label": item.eps_label,
                "method": item.method,
                "dataset_index": item.dataset_index,
                "n_attack_rows": item.n_attack_rows,
                "output_png": str(item.output_png.relative_to(ROOT)),
            }
            for item in rendered
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_report(rendered, missing, summary_csv, manifest_path)

    print(f"Rendered {len(rendered)} combined PNG files")
    print(f"Output dir: {OUT_DIR.relative_to(ROOT)}")
    print(f"Summary CSV: {summary_csv.relative_to(ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")
    if missing:
        print(f"Missing expected canonical rows: {len(missing)}")
        for item in missing:
            print(f"  - {item['eps_label']} {item['method']} {item['label']}")


if __name__ == "__main__":
    main()
