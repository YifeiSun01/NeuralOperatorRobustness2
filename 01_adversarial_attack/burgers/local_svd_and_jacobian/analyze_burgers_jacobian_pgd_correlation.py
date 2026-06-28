#!/usr/bin/env python3
"""Correlate saved Burgers error-Jacobian norms with saved PGD loss growth.

This script intentionally uses only the Python standard library plus numpy.
Matplotlib/pandas are not required in the current environment; plots are simple
SVG scatter plots written directly.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SVD_CSV = ROOT / (
    "forensics/burgers_loss3_selective_round03_loss123_final_extension_"
    "jacobian_svd_rep20_top100_20260606/"
    "round03_loss123_final_extension_jacobian_svd_summary.csv"
)
ATTACK_CSV = ROOT / (
    "forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_"
    "20step_20260607/summary_by_model_dataset.csv"
)
OUT_DIR = ROOT / "analysis_outputs/burgers_jacobian_pgd_correlation_20260608"

MODEL_MAP = {
    "baseline": ("baseline", "exact"),
    "loss1_epoch5000": ("loss1_epoch8000", "proxy_svd_epoch5000_attack_epoch8000"),
    "loss2_epoch2000": ("loss2_epoch2000", "exact"),
    "loss3_epoch1500": ("loss3_epoch1500", "exact"),
}

MODEL_COLORS = {
    "baseline": "#4c4c4c",
    "loss1_epoch8000": "#2563eb",
    "loss2_epoch2000": "#0891b2",
    "loss3_epoch1500": "#d97706",
}


def as_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def dataset_split(dataset_id: str) -> str:
    if dataset_id.startswith("train_original"):
        return "train"
    if dataset_id.startswith("test_original"):
        return "test"
    return "generalization"


def weighted_mean(rows: list[dict[str, object]], value_key: str, weight_key: str) -> float:
    total_weight = 0.0
    total_value = 0.0
    for row in rows:
        value = float(row[value_key])
        weight = float(row[weight_key])
        if math.isfinite(value) and math.isfinite(weight) and weight > 0:
            total_value += value * weight
            total_weight += weight
    return total_value / total_weight if total_weight else float("nan")


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson(x_values: list[float], y_values: list[float]) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x_values: list[float], y_values: list[float]) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    return pearson(rank_values(x).tolist(), rank_values(y).tolist())


def partial_pearson(
    x_values: list[float], y_values: list[float], control_values: list[float]
) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    z = np.asarray(control_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    if len(x) < 4 or float(np.std(z)) == 0.0:
        return float("nan")
    z_design = np.column_stack([np.ones(len(z)), z])
    bx, *_ = np.linalg.lstsq(z_design, x, rcond=None)
    by, *_ = np.linalg.lstsq(z_design, y, rcond=None)
    x_resid = x - z_design @ bx
    y_resid = y - z_design @ by
    return pearson(x_resid.tolist(), y_resid.tolist())


def corr_row(
    analysis_set: str,
    rows: list[dict[str, object]],
    x_key: str,
    y_key: str,
    note: str,
    control_key: str | None = None,
) -> dict[str, object]:
    x = [float(row[x_key]) for row in rows]
    y = [float(row[y_key]) for row in rows]
    valid = [
        row
        for row in rows
        if math.isfinite(float(row[x_key])) and math.isfinite(float(row[y_key]))
    ]
    out = {
        "analysis_set": analysis_set,
        "x": x_key,
        "y": y_key,
        "n": len(valid),
        "pearson": pearson(x, y),
        "spearman": spearman(x, y),
        "partial_control": control_key or "",
        "partial_pearson": "",
        "note": note,
    }
    if control_key:
        z = [float(row[control_key]) for row in rows]
        out["partial_pearson"] = partial_pearson(x, y, z)
    return out


def format_number(value: object, precision: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.{precision}g}"


def write_scatter_svg(
    path: Path,
    rows: list[dict[str, object]],
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    log_x: bool = False,
    log_y: bool = False,
    label_key: str | None = None,
) -> None:
    width = 980
    height = 720
    margin_left = 105
    margin_right = 35
    margin_top = 70
    margin_bottom = 95
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    points = []
    for row in rows:
        x = float(row[x_key])
        y = float(row[y_key])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if log_x:
            if x <= 0:
                continue
            x = math.log10(x)
        if log_y:
            if y <= 0:
                continue
            y = math.log10(y)
        points.append((x, y, row))

    if not points:
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    x_pad = 0.08 * (x_max - x_min)
    y_pad = 0.08 * (y_max - y_min)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_h

    def tick_values(v_min: float, v_max: float, count: int = 5) -> list[float]:
        return [v_min + i * (v_max - v_min) / (count - 1) for i in range(count)]

    pearson_r = pearson([float(r[x_key]) for r in rows], [float(r[y_key]) for r in rows])
    spearman_r = spearman([float(r[x_key]) for r in rows], [float(r[y_key]) for r in rows])

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="21" font-weight="700" fill="#222">{title}</text>',
        f'<text x="{width / 2}" y="55" text-anchor="middle" font-family="Arial" font-size="13" fill="#555">Pearson r={format_number(pearson_r, 4)}, Spearman rho={format_number(spearman_r, 4)}, n={len(points)}</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d6d3cc" stroke-width="1"/>',
    ]

    for tick in tick_values(x_min, x_max):
        x_pos = sx(tick)
        label_value = 10**tick if log_x else tick
        lines.extend(
            [
                f'<line x1="{x_pos:.2f}" x2="{x_pos:.2f}" y1="{margin_top}" y2="{margin_top + plot_h}" stroke="#eee9df" stroke-width="1"/>',
                f'<text x="{x_pos:.2f}" y="{margin_top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#555">{format_number(label_value, 3)}</text>',
            ]
        )
    for tick in tick_values(y_min, y_max):
        y_pos = sy(tick)
        label_value = 10**tick if log_y else tick
        lines.extend(
            [
                f'<line x1="{margin_left}" x2="{margin_left + plot_w}" y1="{y_pos:.2f}" y2="{y_pos:.2f}" stroke="#eee9df" stroke-width="1"/>',
                f'<text x="{margin_left - 14}" y="{y_pos + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{format_number(label_value, 3)}</text>',
            ]
        )

    lines.extend(
        [
            f'<text x="{margin_left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="15" fill="#222">{x_label}</text>',
            f'<text x="24" y="{margin_top + plot_h / 2}" text-anchor="middle" font-family="Arial" font-size="15" fill="#222" transform="rotate(-90 24 {margin_top + plot_h / 2})">{y_label}</text>',
        ]
    )

    for x, y, row in points:
        model = str(row.get("attack_model") or row.get("model") or row.get("checkpoint_label"))
        color = MODEL_COLORS.get(model, "#7c3aed")
        marker = "circle"
        if row.get("checkpoint_match") == "proxy_svd_epoch5000_attack_epoch8000":
            marker = "diamond"
        x_pos = sx(x)
        y_pos = sy(y)
        label = str(row.get(label_key, "")) if label_key else ""
        if marker == "diamond":
            r = 6
            lines.append(
                f'<polygon points="{x_pos:.2f},{y_pos-r:.2f} {x_pos+r:.2f},{y_pos:.2f} {x_pos:.2f},{y_pos+r:.2f} {x_pos-r:.2f},{y_pos:.2f}" fill="{color}" fill-opacity="0.82" stroke="#222" stroke-width="0.7"/>'
            )
        else:
            lines.append(
                f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" r="6" fill="{color}" fill-opacity="0.82" stroke="#222" stroke-width="0.7"/>'
            )
        if label:
            safe_label = (
                label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            lines.append(
                f'<text x="{x_pos + 8:.2f}" y="{y_pos - 8:.2f}" font-family="Arial" font-size="10" fill="#333">{safe_label}</text>'
            )

    legend_x = margin_left + 12
    legend_y = margin_top + 18
    for idx, (model, color) in enumerate(MODEL_COLORS.items()):
        y = legend_y + idx * 20
        lines.append(
            f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}" fill-opacity="0.82" stroke="#222" stroke-width="0.7"/>'
        )
        lines.append(
            f'<text x="{legend_x + 12}" y="{y + 4}" font-family="Arial" font-size="12" fill="#333">{model}</text>'
        )
    lines.append(
        f'<polygon points="{legend_x+220},{legend_y-5} {legend_x+226},{legend_y} {legend_x+220},{legend_y+5} {legend_x+214},{legend_y}" fill="{MODEL_COLORS["loss1_epoch8000"]}" fill-opacity="0.82" stroke="#222" stroke-width="0.7"/>'
    )
    lines.append(
        f'<text x="{legend_x + 234}" y="{legend_y + 4}" font-family="Arial" font-size="12" fill="#333">loss1 SVD epoch5000 proxy</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    svd_rows_raw = read_csv(SVD_CSV)
    attack_rows_raw = read_csv(ATTACK_CSV)

    svd_error_rows = []
    for row in svd_rows_raw:
        if row.get("jacobian_kind") != "error":
            continue
        checkpoint = row["checkpoint_label"]
        if checkpoint not in MODEL_MAP:
            continue
        attack_model, checkpoint_match = MODEL_MAP[checkpoint]
        svd_error_rows.append(
            {
                "sample_id": row["sample_id"],
                "source_split": row["source_split"],
                "dataset_id": row["dataset_id"],
                "local_index": row["local_index"],
                "svd_checkpoint_label": checkpoint,
                "attack_model": attack_model,
                "checkpoint_match": checkpoint_match,
                "spectral_norm": as_float(row["spectral_norm"]),
                "fro_norm": as_float(row["fro_norm"]),
                "effective_rank": as_float(row["effective_rank"]),
                "baseline_error_spectral_norm": as_float(
                    row["baseline_error_spectral_norm"]
                ),
                "spectral_norm_ratio_to_baseline": as_float(
                    row["error_spectral_norm_ratio_to_baseline"]
                ),
            }
        )

    attack_rows = []
    for row in attack_rows_raw:
        final_loss = as_float(row["final_loss_mean"])
        initial_loss = as_float(row["initial_loss_mean"])
        final_rms = as_float(row["final_diff_rms_mean"])
        initial_rms = as_float(row["initial_diff_rms_mean"])
        attack_rows.append(
            {
                "model": row["model"],
                "dataset_id": row["dataset_id"],
                "dataset_index": int(row["dataset_index"]),
                "split": dataset_split(row["dataset_id"]),
                "sample_count": as_float(row["sample_count"]),
                "initial_loss_mean": initial_loss,
                "final_loss_mean": final_loss,
                "attack_loss_gain": final_loss - initial_loss,
                "loss_ratio_final_over_initial": as_float(
                    row["loss_ratio_final_over_initial"]
                ),
                "initial_diff_rms_mean": initial_rms,
                "final_diff_rms_mean": final_rms,
                "attack_rms_gain": final_rms - initial_rms,
            }
        )

    attack_by_model_dataset = {
        (row["model"], row["dataset_id"]): row for row in attack_rows
    }

    generalization_join = []
    for svd in svd_error_rows:
        if svd["source_split"] != "generalization":
            continue
        attack = attack_by_model_dataset.get((svd["attack_model"], svd["dataset_id"]))
        baseline_attack = attack_by_model_dataset.get(("baseline", svd["dataset_id"]))
        if not attack or not baseline_attack:
            continue
        generalization_join.append(
            {
                **svd,
                "attack_dataset_sample_count": attack["sample_count"],
                "attack_initial_loss_mean": attack["initial_loss_mean"],
                "attack_final_loss_mean": attack["final_loss_mean"],
                "attack_loss_gain": attack["attack_loss_gain"],
                "attack_final_rms_mean": attack["final_diff_rms_mean"],
                "attack_rms_gain": attack["attack_rms_gain"],
                "attack_final_loss_ratio_to_baseline": (
                    attack["final_loss_mean"] / baseline_attack["final_loss_mean"]
                ),
                "attack_final_rms_ratio_to_baseline": (
                    attack["final_diff_rms_mean"]
                    / baseline_attack["final_diff_rms_mean"]
                ),
                "plot_label": f'{svd["attack_model"].replace("_epoch", "e")}:d{svd["dataset_id"][-2:]}',
            }
        )

    svd_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in svd_error_rows:
        svd_groups[(row["attack_model"], row["source_split"])].append(row)

    attack_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in attack_rows:
        attack_groups[(row["model"], row["split"])].append(row)

    model_split_rows = []
    for key, group in sorted(svd_groups.items()):
        attack_model, split = key
        attacks = attack_groups.get(key)
        baseline_attacks = attack_groups.get(("baseline", split))
        if not attacks or not baseline_attacks:
            continue
        baseline_final_loss = weighted_mean(
            baseline_attacks, "final_loss_mean", "sample_count"
        )
        baseline_final_rms = weighted_mean(
            baseline_attacks, "final_diff_rms_mean", "sample_count"
        )
        checkpoint_matches = sorted({str(row["checkpoint_match"]) for row in group})
        sigma_values = [float(row["spectral_norm"]) for row in group]
        sigma_ratio_values = [
            float(row["spectral_norm_ratio_to_baseline"]) for row in group
        ]
        row = {
            "attack_model": attack_model,
            "source_split": split,
            "checkpoint_match": "|".join(checkpoint_matches),
            "svd_n": len(group),
            "attack_dataset_count": len(attacks),
            "attack_sample_count": sum(float(r["sample_count"]) for r in attacks),
            "spectral_norm_mean": mean(sigma_values),
            "spectral_norm_median": median(sigma_values),
            "spectral_norm_max": max(sigma_values),
            "spectral_norm_ratio_to_baseline_mean": mean(sigma_ratio_values),
            "attack_initial_loss_mean": weighted_mean(
                attacks, "initial_loss_mean", "sample_count"
            ),
            "attack_final_loss_mean": weighted_mean(
                attacks, "final_loss_mean", "sample_count"
            ),
            "attack_loss_gain": weighted_mean(attacks, "attack_loss_gain", "sample_count"),
            "attack_final_rms_mean": weighted_mean(
                attacks, "final_diff_rms_mean", "sample_count"
            ),
            "attack_rms_gain": weighted_mean(attacks, "attack_rms_gain", "sample_count"),
        }
        row["attack_final_loss_ratio_to_baseline"] = (
            row["attack_final_loss_mean"] / baseline_final_loss
        )
        row["attack_final_rms_ratio_to_baseline"] = (
            row["attack_final_rms_mean"] / baseline_final_rms
        )
        row["plot_label"] = (
            f'{attack_model.replace("_epoch", "e")}:{split[:3]}'
        )
        model_split_rows.append(row)

    exact_model_split_rows = [
        row for row in model_split_rows if row["checkpoint_match"] == "exact"
    ]
    exact_generalization_rows = [
        row for row in generalization_join if row["checkpoint_match"] == "exact"
    ]

    correlations = []
    for name, rows in [
        ("model_split_all_with_loss1_proxy", model_split_rows),
        ("model_split_exact_checkpoints", exact_model_split_rows),
        ("generalization_dataset_all_with_loss1_proxy", generalization_join),
        ("generalization_dataset_exact_checkpoints", exact_generalization_rows),
    ]:
        if not rows:
            continue
        if name.startswith("model_split"):
            correlations.extend(
                [
                    corr_row(
                        name,
                        rows,
                        "spectral_norm_mean",
                        "attack_final_loss_mean",
                        "Raw aggregate MSE can be confounded by split/dataset difficulty.",
                        control_key="attack_initial_loss_mean",
                    ),
                    corr_row(
                        name,
                        rows,
                        "spectral_norm_mean",
                        "attack_final_rms_mean",
                        "Raw aggregate RMS can be confounded by split/dataset difficulty.",
                        control_key="attack_initial_loss_mean",
                    ),
                    corr_row(
                        name,
                        rows,
                        "spectral_norm_mean",
                        "attack_loss_gain",
                        "Attack MSE gain above clean error.",
                        control_key="attack_initial_loss_mean",
                    ),
                    corr_row(
                        name,
                        rows,
                        "spectral_norm_ratio_to_baseline_mean",
                        "attack_final_loss_ratio_to_baseline",
                        "Baseline-normalized within each split.",
                    ),
                    corr_row(
                        name,
                        rows,
                        "spectral_norm_ratio_to_baseline_mean",
                        "attack_final_rms_ratio_to_baseline",
                        "Baseline-normalized within each split.",
                    ),
                ]
            )
        else:
            correlations.extend(
                [
                    corr_row(
                        name,
                        rows,
                        "spectral_norm",
                        "attack_final_loss_mean",
                        "SVD is one representative sample; PGD loss is dataset mean.",
                        control_key="attack_initial_loss_mean",
                    ),
                    corr_row(
                        name,
                        rows,
                        "spectral_norm",
                        "attack_final_rms_mean",
                        "SVD is one representative sample; PGD RMS is dataset mean.",
                        control_key="attack_initial_loss_mean",
                    ),
                    corr_row(
                        name,
                        rows,
                        "spectral_norm",
                        "attack_loss_gain",
                        "Attack MSE gain above clean error.",
                        control_key="attack_initial_loss_mean",
                    ),
                    corr_row(
                        name,
                        rows,
                        "spectral_norm_ratio_to_baseline",
                        "attack_final_loss_ratio_to_baseline",
                        "Baseline-normalized per dataset.",
                    ),
                    corr_row(
                        name,
                        rows,
                        "spectral_norm_ratio_to_baseline",
                        "attack_final_rms_ratio_to_baseline",
                        "Baseline-normalized per dataset.",
                    ),
                ]
            )

    write_csv(
        OUT_DIR / "model_split_join.csv",
        model_split_rows,
        [
            "attack_model",
            "source_split",
            "checkpoint_match",
            "svd_n",
            "attack_dataset_count",
            "attack_sample_count",
            "spectral_norm_mean",
            "spectral_norm_median",
            "spectral_norm_max",
            "spectral_norm_ratio_to_baseline_mean",
            "attack_initial_loss_mean",
            "attack_final_loss_mean",
            "attack_loss_gain",
            "attack_final_rms_mean",
            "attack_rms_gain",
            "attack_final_loss_ratio_to_baseline",
            "attack_final_rms_ratio_to_baseline",
            "plot_label",
        ],
    )
    write_csv(
        OUT_DIR / "generalization_dataset_join.csv",
        generalization_join,
        [
            "sample_id",
            "source_split",
            "dataset_id",
            "local_index",
            "svd_checkpoint_label",
            "attack_model",
            "checkpoint_match",
            "spectral_norm",
            "fro_norm",
            "effective_rank",
            "baseline_error_spectral_norm",
            "spectral_norm_ratio_to_baseline",
            "attack_dataset_sample_count",
            "attack_initial_loss_mean",
            "attack_final_loss_mean",
            "attack_loss_gain",
            "attack_final_rms_mean",
            "attack_rms_gain",
            "attack_final_loss_ratio_to_baseline",
            "attack_final_rms_ratio_to_baseline",
            "plot_label",
        ],
    )
    write_csv(
        OUT_DIR / "correlations.csv",
        correlations,
        [
            "analysis_set",
            "x",
            "y",
            "n",
            "pearson",
            "spearman",
            "partial_control",
            "partial_pearson",
            "note",
        ],
    )

    write_scatter_svg(
        OUT_DIR / "model_split_sigma_vs_attack_mse.svg",
        model_split_rows,
        "spectral_norm_mean",
        "attack_final_loss_mean",
        "Model-split aggregate: Jacobian norm vs final PGD MSE",
        "mean error-Jacobian spectral norm",
        "weighted final attack MSE",
        log_x=True,
        log_y=True,
        label_key="plot_label",
    )
    write_scatter_svg(
        OUT_DIR / "model_split_sigma_ratio_vs_attack_mse_ratio.svg",
        model_split_rows,
        "spectral_norm_ratio_to_baseline_mean",
        "attack_final_loss_ratio_to_baseline",
        "Model-split baseline-normalized ratios",
        "mean spectral-norm ratio to baseline",
        "final attack MSE ratio to baseline",
        log_x=True,
        log_y=True,
        label_key="plot_label",
    )
    write_scatter_svg(
        OUT_DIR / "generalization_sigma_vs_attack_mse.svg",
        generalization_join,
        "spectral_norm",
        "attack_final_loss_mean",
        "Generalization datasets: representative Jacobian norm vs PGD MSE",
        "representative error-Jacobian spectral norm",
        "dataset mean final attack MSE",
        log_x=True,
        log_y=True,
        label_key="plot_label",
    )
    write_scatter_svg(
        OUT_DIR / "generalization_sigma_ratio_vs_attack_mse_ratio.svg",
        generalization_join,
        "spectral_norm_ratio_to_baseline",
        "attack_final_loss_ratio_to_baseline",
        "Generalization datasets: baseline-normalized ratios",
        "representative spectral-norm ratio to baseline",
        "dataset final attack MSE ratio to baseline",
        log_x=True,
        log_y=True,
        label_key="plot_label",
    )

    summary = {
        "source_files": {
            "svd_csv": str(SVD_CSV.relative_to(ROOT)),
            "attack_csv": str(ATTACK_CSV.relative_to(ROOT)),
        },
        "outputs": {
            "model_split_join": str((OUT_DIR / "model_split_join.csv").relative_to(ROOT)),
            "generalization_dataset_join": str(
                (OUT_DIR / "generalization_dataset_join.csv").relative_to(ROOT)
            ),
            "correlations": str((OUT_DIR / "correlations.csv").relative_to(ROOT)),
        },
        "row_counts": {
            "svd_error_rows": len(svd_error_rows),
            "attack_rows": len(attack_rows),
            "model_split_join": len(model_split_rows),
            "model_split_exact": len(exact_model_split_rows),
            "generalization_dataset_join": len(generalization_join),
            "generalization_dataset_exact": len(exact_generalization_rows),
        },
        "checkpoint_note": (
            "No local loss1 epoch8000 Jacobian/SVD artifact was found; loss1 rows "
            "use epoch5000 SVD as a proxy when included. Exact-checkpoint analyses "
            "exclude loss1."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
