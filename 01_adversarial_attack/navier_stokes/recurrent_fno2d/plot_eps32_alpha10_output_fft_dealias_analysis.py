#!/usr/bin/env python3
"""CPU-only FFT/dealiasing analysis for eps32_alpha10 saved final outputs.

Reads saved final_state_outputs.npz files. Does not import torch/JAX or rerun
model/solver. It visualizes Fourier magnitudes for final_delta, model outputs,
solver outputs, and model-solver differences, and measures power outside the
2/3-rule per-axis cutoff square.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
MODE_NAMES = {
    "wwwwwwwwww": "all_w",
    "aaaaaaaaaw": "all_a_target_w",
    "dddddddddw": "all_d_target_w",
    "wwwwwddddw": "w1_5_d6_9_target_w",
    "dddddwwwww": "d1_5_w6_9_target_w",
    "aaaaaddddw": "a1_5_d6_9_target_w",
}
BLOCK_ORDER = [
    "loss1/all_w",
    "loss2/all_a_target_w",
    "loss3/all_w",
    "loss3/all_d_target_w",
    "loss3/w1_5_d6_9_target_w",
    "loss3/d1_5_w6_9_target_w",
    "loss3/a1_5_d6_9_target_w",
]
FIELDS = [
    ("final_delta", "final delta"),
    ("clean_model_final", "clean FNO final"),
    ("clean_solver_final", "clean solver final"),
    ("clean_model_minus_solver", "clean FNO - solver"),
    ("adv_model_final", "adv FNO final"),
    ("adv_solver_final", "adv solver final"),
    ("adv_model_minus_solver", "adv FNO - solver"),
]
TWO_THIRDS_CUTOFF = 1.0 / 3.0
AXIS_BAND = 4.0 / 256.0
LOW_CUTOFF = 0.15
MID_CUTOFF = 0.35


def block_from_path(path: Path) -> tuple[str, str, str]:
    mode_part = next(part for part in path.parts if part.startswith("mode_"))
    mode_spec = mode_part.split("_p2_q2", 1)[0][len("mode_"):]
    mode = MODE_NAMES.get(mode_spec, mode_spec)
    loss = next(part for part in path.parts if part in ("loss1", "loss2", "loss3"))
    return f"{loss}/{mode}", loss, mode_spec


def ordered_blocks(blocks: dict[str, Any]) -> list[str]:
    return [b for b in BLOCK_ORDER if b in blocks] + sorted([b for b in blocks if b not in BLOCK_ORDER])


def discover(pair_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    for npz_path in sorted(pair_root.glob("mode_*/batch_0000_0009/loss*/*/final_state_outputs.npz")):
        method = npz_path.parent.name
        if method not in METHODS:
            continue
        block, loss, mode_spec = block_from_path(npz_path)
        z = np.load(npz_path)
        item: dict[str, Any] = {
            "npz": str(npz_path),
            "loss": loss,
            "mode_spec": mode_spec,
            "dataset_indices": z["dataset_indices"].astype(int),
            "clean_true_loss": z["clean_true_loss"].astype(np.float64),
            "adv_true_loss": z["adv_true_loss"].astype(np.float64),
        }
        for key, _label in FIELDS:
            if key in z.files:
                item[key] = z[key].astype(np.float32)
        blocks.setdefault(block, {})[method] = item
    return blocks


def freq_grids(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq = np.fft.fftshift(np.fft.fftfreq(n))
    fx, fy = np.meshgrid(freq, freq, indexing="xy")
    rho = np.sqrt(fx * fx + fy * fy) / np.sqrt(0.5 * 0.5 + 0.5 * 0.5)
    return fx, fy, rho


def fft_power(arr: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(arr, axes=(-2, -1)), axes=(-2, -1))
    return np.abs(f) ** 2


def log_magnitude(sample: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(sample))
    return np.log10(np.abs(f) + 1e-6)


def metrics(power: np.ndarray, fx: np.ndarray, fy: np.ndarray, rho: np.ndarray) -> dict[str, np.ndarray]:
    flat_power = power.reshape(power.shape[0], -1).astype(np.float64)
    flat_fx = fx.reshape(-1)
    flat_fy = fy.reshape(-1)
    flat_rho = rho.reshape(-1)
    dc = flat_rho < 1e-12
    non_dc = ~dc
    total = np.maximum(np.sum(flat_power[:, non_dc], axis=1), 1e-30)

    square_inside = (np.abs(flat_fx) <= TWO_THIRDS_CUTOFF) & (np.abs(flat_fy) <= TWO_THIRDS_CUTOFF) & non_dc
    square_outside = (~((np.abs(flat_fx) <= TWO_THIRDS_CUTOFF) & (np.abs(flat_fy) <= TWO_THIRDS_CUTOFF))) & non_dc
    cross_outside = square_outside & ((np.abs(flat_fx) <= AXIS_BAND) | (np.abs(flat_fy) <= AXIS_BAND))
    nonaxis_outside = square_outside & (~cross_outside)
    low = (flat_rho > 1e-12) & (flat_rho <= LOW_CUTOFF)
    mid = (flat_rho > LOW_CUTOFF) & (flat_rho <= MID_CUTOFF)
    high = flat_rho > MID_CUTOFF

    inside_power = np.sum(flat_power[:, square_inside], axis=1)
    outside_power = np.sum(flat_power[:, square_outside], axis=1)
    cross_power = np.sum(flat_power[:, cross_outside], axis=1)
    nonaxis_power = np.sum(flat_power[:, nonaxis_outside], axis=1)
    low_power = np.sum(flat_power[:, low], axis=1)
    mid_power = np.sum(flat_power[:, mid], axis=1)
    high_power = np.sum(flat_power[:, high], axis=1)
    centroid = np.sum(flat_power[:, non_dc] * flat_rho[non_dc][None, :], axis=1) / total
    return {
        "square_inside_frac": inside_power / total,
        "square_outside_frac": outside_power / total,
        "cross_outside_frac": cross_power / total,
        "nonaxis_outside_frac": nonaxis_power / total,
        "cross_share_of_outside": cross_power / np.maximum(outside_power, 1e-30),
        "low_frac": low_power / total,
        "mid_frac": mid_power / total,
        "high_frac": high_power / total,
        "spectral_centroid": centroid,
    }


def draw_cutoff(ax) -> None:
    c = TWO_THIRDS_CUTOFF
    ax.plot([-c, c, c, -c, -c], [-c, -c, c, c, -c], color="cyan", linestyle="--", linewidth=0.8, alpha=0.85)
    ax.axhline(0.0, color="white", linewidth=0.35, alpha=0.45)
    ax.axvline(0.0, color="white", linewidth=0.35, alpha=0.45)


def save_field_grid(blocks: dict[str, Any], field: str, label: str, out_dir: Path) -> Path:
    labels = ordered_blocks(blocks)
    spectra: dict[tuple[str, str], np.ndarray] = {}
    values = []
    for block in labels:
        for method in METHODS:
            item = blocks.get(block, {}).get(method)
            if not item or field not in item:
                continue
            spec = log_magnitude(item[field][0])
            spectra[(block, method)] = spec
            values.append(spec)
    vals = np.concatenate([v.ravel() for v in values]) if values else np.asarray([0.0, 1.0])
    vmin, vmax = np.percentile(vals[np.isfinite(vals)], [1, 99.7])

    fig, axes = plt.subplots(len(labels), len(METHODS), figsize=(4.2 * len(METHODS), 3.1 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axes = np.asarray([axes])
    fig.suptitle(f"eps32_alpha10 {label} FFT log magnitude | sample_position=0 | cyan box: |k| <= 1/3", fontsize=13)
    im = None
    for i, block in enumerate(labels):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            spec = spectra.get((block, method))
            if spec is None:
                ax.axis("off")
                continue
            im = ax.imshow(spec, origin="lower", cmap="magma", extent=(-0.5, 0.5, -0.5, 0.5), vmin=vmin, vmax=vmax)
            draw_cutoff(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{block}\n{method}", fontsize=8)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label=f"log10(|FFT({label})|)")
    safe = field.replace("_", "-")
    out = out_dir / f"eps32_alpha10_{safe}_fft_log_magnitude_sample0_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_metric_heatmap(rows: list[dict[str, Any]], field: str, metric: str, title: str, out_dir: Path) -> Path:
    blocks = [b for b in BLOCK_ORDER if any(r["block"] == b and r["field"] == field for r in rows)]
    mat = np.full((len(blocks), len(METHODS)), np.nan)
    for r in rows:
        if r["field"] == field and r["block"] in blocks and r["method"] in METHODS:
            mat[blocks.index(r["block"]), METHODS.index(r["method"])] = float(r[f"{metric}_mean"])
    fig, ax = plt.subplots(figsize=(7.2, max(5.5, 0.62 * len(blocks))), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=35, ha="right")
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels(blocks)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2e}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    out = out_dir / f"eps32_alpha10_{field}_{metric}_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    blocks = discover(args.pair_root)
    first_item = next(iter(next(iter(blocks.values())).values()))
    n = int(first_item["final_delta"].shape[-1])
    fx, fy, rho = freq_grids(n)

    pngs = []
    for field, label in FIELDS:
        pngs.append(str(save_field_grid(blocks, field, label, args.out_dir)))

    rows: list[dict[str, Any]] = []
    for block in ordered_blocks(blocks):
        for method in METHODS:
            item = blocks.get(block, {}).get(method)
            if not item:
                continue
            for field, label in FIELDS:
                if field not in item:
                    continue
                power = fft_power(item[field])
                m = metrics(power, fx, fy, rho)
                row: dict[str, Any] = {
                    "block": block,
                    "loss": item["loss"],
                    "mode_spec": item["mode_spec"],
                    "method": method,
                    "field": field,
                    "field_label": label,
                    "epsilon": 32.0,
                    "alpha": 10.0,
                    "sample_count": int(item[field].shape[0]),
                    "two_thirds_cutoff_per_axis": TWO_THIRDS_CUTOFF,
                    "axis_band_half_width": AXIS_BAND,
                    "source_npz": item["npz"],
                }
                for key, val in m.items():
                    row[f"{key}_mean"] = float(np.mean(val))
                    row[f"{key}_std"] = float(np.std(val))
                rows.append(row)

    csv_path = args.out_dir / "eps32_alpha10_output_fft_dealias_metrics_summary.csv"
    write_csv(csv_path, rows)

    heatmaps = []
    for field in ["final_delta", "adv_model_final", "adv_solver_final", "adv_model_minus_solver"]:
        heatmaps.append(str(save_metric_heatmap(rows, field, "square_outside_frac", f"{field}: power outside |kx|,|ky| <= 1/3", args.out_dir)))
        heatmaps.append(str(save_metric_heatmap(rows, field, "cross_outside_frac", f"{field}: outside-cutoff power in axis cross arms", args.out_dir)))

    aggregate: dict[str, dict[str, Any]] = {}
    for field, _label in FIELDS:
        for loss in ["loss1", "loss2", "loss3"]:
            group = [r for r in rows if r["field"] == field and r["loss"] == loss]
            if not group:
                continue
            key = f"{field}/{loss}"
            aggregate[key] = {
                "n_settings": len(group),
                "square_outside_frac_mean": float(np.mean([r["square_outside_frac_mean"] for r in group])),
                "cross_outside_frac_mean": float(np.mean([r["cross_outside_frac_mean"] for r in group])),
                "cross_share_of_outside_mean": float(np.mean([r["cross_share_of_outside_mean"] for r in group])),
                "spectral_centroid_mean": float(np.mean([r["spectral_centroid_mean"] for r in group])),
                "mid_frac_mean": float(np.mean([r["mid_frac_mean"] for r in group])),
                "high_frac_mean": float(np.mean([r["high_frac_mean"] for r in group])),
            }

    report = {
        "pair_root": str(args.pair_root),
        "out_dir": str(args.out_dir),
        "csv": str(csv_path),
        "pngs": pngs,
        "heatmaps": heatmaps,
        "two_thirds_cutoff_per_axis": TWO_THIRDS_CUTOFF,
        "axis_band_half_width": AXIS_BAND,
        "aggregate_by_field_loss": aggregate,
    }
    report_path = args.out_dir / "eps32_alpha10_output_fft_dealias_analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
