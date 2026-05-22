#!/usr/bin/env python3
"""CPU-only linear-y radial FFT profiles for eps32_alpha10 final_delta.

Reads saved final_state_outputs.npz files. Does not import torch/JAX or rerun
model/solver. This complements the log-y method-grouped final-delta spectrum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
METHOD_LABELS = {
    "raw_add": "raw_add / PGD",
    "raw_replace": "raw_replace",
    "steepest_add": "steepest_add / LP steepest PGD",
    "steepest_replace": "steepest_replace / GPI-style",
}
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
    "d1_5_w6_9_target_w",
]
# Keep a typo-proof explicit order below.
BLOCK_ORDER = [
    "loss1/all_w",
    "loss2/all_a_target_w",
    "loss3/all_w",
    "loss3/all_d_target_w",
    "loss3/w1_5_d6_9_target_w",
    "loss3/d1_5_w6_9_target_w",
    "loss3/a1_5_d6_9_target_w",
]
BLOCK_COLORS = {
    "loss1/all_w": "#1f77b4",
    "loss2/all_a_target_w": "#ff7f0e",
    "loss3/all_w": "#2ca02c",
    "loss3/all_d_target_w": "#d62728",
    "loss3/w1_5_d6_9_target_w": "#9467bd",
    "loss3/d1_5_w6_9_target_w": "#8c564b",
    "loss3/a1_5_d6_9_target_w": "#17becf",
}
NBINS = 96
RADIAL_AXIS_CUTOFF = np.sqrt(2.0) / 3.0
RADIAL_CORNER_CUTOFF = 2.0 / 3.0


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
        blocks.setdefault(block, {})[method] = {
            "npz": str(npz_path),
            "loss": loss,
            "mode_spec": mode_spec,
            "final_delta": z["final_delta"].astype(np.float32),
        }
    return blocks


def frequency_grid(n: int) -> np.ndarray:
    freq = np.fft.fftshift(np.fft.fftfreq(n))
    fx, fy = np.meshgrid(freq, freq, indexing="xy")
    return np.sqrt(fx * fx + fy * fy) / np.sqrt(0.5 * 0.5 + 0.5 * 0.5)


def fft_power(arr: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(arr, axes=(-2, -1)), axes=(-2, -1))
    return np.abs(f) ** 2


def radial_profile(power: np.ndarray, rho: np.ndarray, nbins: int = NBINS) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    flat_rho = rho.reshape(-1)
    profiles = []
    for sample_power in power:
        flat = sample_power.reshape(-1)
        prof = np.zeros(nbins, dtype=np.float64)
        for i in range(nbins):
            mask = (flat_rho >= bins[i]) & (flat_rho < bins[i + 1])
            prof[i] = np.mean(flat[mask]) if np.any(mask) else np.nan
        total = np.nansum(prof)
        if total > 0:
            prof = prof / total
        profiles.append(prof)
    return centers, np.asarray(profiles)


def save_linear_y(blocks: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    labels = ordered_blocks(blocks)
    first = next(item for block in blocks.values() for item in block.values())
    rho = frequency_grid(int(first["final_delta"].shape[-1]))

    summary_rows = []
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(11.5, 3.0 * len(METHODS)), constrained_layout=True)
    fig.suptitle("eps32_alpha10 final_delta radial FFT spectra grouped by optimizer | linear y", fontsize=14)
    for row, method in enumerate(METHODS):
        ax = axes[row]
        for block in labels:
            item = blocks.get(block, {}).get(method)
            if not item:
                continue
            centers, profs = radial_profile(fft_power(item["final_delta"]), rho)
            mean = np.nanmean(profs, axis=0)
            std = np.nanstd(profs, axis=0)
            color = BLOCK_COLORS.get(block, None)
            ax.plot(centers, mean, color=color, label=block, linewidth=1.8)
            ax.fill_between(centers, mean - std, mean + std, color=color, alpha=0.10, linewidth=0)
            summary_rows.append({
                "block": block,
                "method": method,
                "first_bin_rho": float(centers[0]),
                "first_bin_mean": float(mean[0]),
                "second_bin_mean": float(mean[1]),
                "third_bin_mean": float(mean[2]),
                "peak_bin": int(np.nanargmax(mean)),
                "peak_rho": float(centers[int(np.nanargmax(mean))]),
                "peak_mean": float(np.nanmax(mean)),
            })
        ax.axvline(RADIAL_AXIS_CUTOFF, color="black", linestyle=":", linewidth=1.2, alpha=0.85, label="axis cutoff rho=sqrt(2)/3" if row == 0 else None)
        ax.axvline(RADIAL_CORNER_CUTOFF, color="black", linestyle="--", linewidth=1.2, alpha=0.85, label="box-corner rho=2/3" if row == 0 else None)
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylabel("normalized radial power")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=7, ncol=2, loc="best")
    axes[-1].set_xlabel("normalized radial frequency rho")
    out = out_dir / "eps32_alpha10_final-delta_radial_fft_by_method_all_blocks_linear_y.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # Also save a low-frequency zoom to make first-bin differences unmistakable.
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(11.5, 3.0 * len(METHODS)), constrained_layout=True)
    fig.suptitle("eps32_alpha10 final_delta radial FFT spectra | linear y low-frequency zoom", fontsize=14)
    for row, method in enumerate(METHODS):
        ax = axes[row]
        for block in labels:
            item = blocks.get(block, {}).get(method)
            if not item:
                continue
            centers, profs = radial_profile(fft_power(item["final_delta"]), rho)
            mean = np.nanmean(profs, axis=0)
            std = np.nanstd(profs, axis=0)
            color = BLOCK_COLORS.get(block, None)
            ax.plot(centers, mean, color=color, label=block, linewidth=1.8)
            ax.fill_between(centers, mean - std, mean + std, color=color, alpha=0.10, linewidth=0)
        ax.set_xlim(0.0, 0.12)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylabel("normalized radial power")
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=7, ncol=2, loc="best")
    axes[-1].set_xlabel("normalized radial frequency rho")
    zoom = out_dir / "eps32_alpha10_final-delta_radial_fft_by_method_all_blocks_linear_y_lowfreq_zoom.png"
    fig.savefig(zoom, dpi=150)
    plt.close(fig)

    csv_path = out_dir / "eps32_alpha10_final_delta_radial_linear_y_first_bins.csv"
    import csv
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return out, zoom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blocks = discover(args.pair_root)
    out, zoom = save_linear_y(blocks, args.out_dir)
    report = {
        "pair_root": str(args.pair_root),
        "out_dir": str(args.out_dir),
        "pngs": [str(out), str(zoom)],
        "note": "Linear-y final_delta radial FFT profiles grouped by optimizer; includes low-frequency zoom.",
    }
    report_path = args.out_dir / "eps32_alpha10_final_delta_radial_linear_y_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
