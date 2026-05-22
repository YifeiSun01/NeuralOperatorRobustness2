#!/usr/bin/env python3
"""CPU-only method-grouped curves for eps32_alpha10 NS2D recurrent attack.

This complements the block-grouped plots. It reads saved per-step CSV/NPZ files
and creates views where each optimizer/method panel overlays all loss/mode
blocks. No torch/JAX/model/solver is imported or run.
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
SPECTRAL_FIELDS = [
    ("final_delta", "final delta"),
    ("adv_model_final", "adv FNO final"),
    ("adv_solver_final", "adv solver final"),
    ("adv_model_minus_solver", "adv FNO - solver"),
]
NBINS = 96
RADIAL_AXIS_CUTOFF = np.sqrt(2.0) / 3.0
RADIAL_CORNER_CUTOFF = 2.0 / 3.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None, "nan", "None"):
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([f(row, key) for row in rows], dtype=np.float64)


def block_from_path(path: Path) -> tuple[str, str, str]:
    mode_part = next(part for part in path.parts if part.startswith("mode_"))
    mode_spec = mode_part.split("_p2_q2", 1)[0][len("mode_"):]
    mode = MODE_NAMES.get(mode_spec, mode_spec)
    loss = next(part for part in path.parts if part in ("loss1", "loss2", "loss3"))
    return f"{loss}/{mode}", loss, mode_spec


def ordered_blocks(blocks: dict[str, Any]) -> list[str]:
    return [b for b in BLOCK_ORDER if b in blocks] + sorted([b for b in blocks if b not in BLOCK_ORDER])


def discover_loss_curves(pair_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    for csv_path in sorted(pair_root.glob("mode_*/batch_0000_0009/loss*/*/per_step_metrics.csv")):
        method = csv_path.parent.name
        if method not in METHODS:
            continue
        block, loss, mode_spec = block_from_path(csv_path)
        rows = read_rows(csv_path)
        if not rows:
            continue
        active_key = f"{loss}_mean"
        active_std_key = f"{loss}_std"
        blocks.setdefault(block, {})[method] = {
            "csv": str(csv_path),
            "loss": loss,
            "mode_spec": mode_spec,
            "method": method,
            "k": series(rows, "k"),
            "objective": series(rows, active_key),
            "objective_std": series(rows, active_std_key),
            "true": series(rows, "true_loss_mean"),
            "true_std": series(rows, "true_loss_std"),
            "boundary": series(rows, "boundary_ratio_mean"),
            "epsilon": float(series(rows, "epsilon")[-1]),
            "alpha": float(series(rows, "alpha")[-1]),
        }
    return blocks


def discover_final_outputs(pair_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    for npz_path in sorted(pair_root.glob("mode_*/batch_0000_0009/loss*/*/final_state_outputs.npz")):
        method = npz_path.parent.name
        if method not in METHODS:
            continue
        block, loss, mode_spec = block_from_path(npz_path)
        z = np.load(npz_path)
        item: dict[str, Any] = {"npz": str(npz_path), "loss": loss, "mode_spec": mode_spec, "method": method}
        for field, _label in SPECTRAL_FIELDS:
            if field in z.files:
                item[field] = z[field].astype(np.float32)
        blocks.setdefault(block, {})[method] = item
    return blocks


def shade(ax, x: np.ndarray, y: np.ndarray, std: np.ndarray, color: str, alpha: float = 0.10) -> None:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(std)
    if np.any(mask):
        ax.fill_between(x[mask], y[mask] - std[mask], y[mask] + std[mask], color=color, alpha=alpha, linewidth=0)


def save_loss_curves_by_method(blocks: dict[str, Any], out_dir: Path, *, logy: bool) -> Path:
    labels = ordered_blocks(blocks)
    fig, axes = plt.subplots(len(METHODS), 2, figsize=(16, 3.15 * len(METHODS)), constrained_layout=True)
    fig.suptitle(
        "eps32_alpha10: curves grouped by optimizer; each panel overlays all loss/mode blocks"
        + (" (log y)" if logy else " (linear y)"),
        fontsize=14,
    )
    for row, method in enumerate(METHODS):
        for block in labels:
            item = blocks.get(block, {}).get(method)
            if not item:
                continue
            color = BLOCK_COLORS.get(block, None)
            k = item["k"]
            axes[row, 0].plot(k, item["objective"], color=color, label=block, linewidth=1.8)
            shade(axes[row, 0], k, item["objective"], item["objective_std"], color or "gray")
            axes[row, 1].plot(k, item["true"], color=color, label=block, linewidth=1.8)
            shade(axes[row, 1], k, item["true"], item["true_std"], color or "gray")
        axes[row, 0].set_title(f"{METHOD_LABELS[method]}: active objective")
        axes[row, 1].set_title(f"{METHOD_LABELS[method]}: true all-W final loss")
        for ax in axes[row]:
            ax.set_xlabel("attack step k")
            ax.grid(True, alpha=0.28)
            if logy:
                ax.set_yscale("log")
            ax.legend(fontsize=7, ncol=2, loc="best")
    suffix = "logy" if logy else "linear"
    out = out_dir / f"eps32_alpha10_loss_curves_by_method_all_blocks_{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


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


def save_radial_spectrum_by_method(blocks: dict[str, Any], field: str, label: str, out_dir: Path) -> Path:
    labels = ordered_blocks(blocks)
    first = next(item for block in blocks.values() for item in block.values() if field in item)
    rho = frequency_grid(int(first[field].shape[-1]))
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(11.5, 3.0 * len(METHODS)), constrained_layout=True)
    fig.suptitle(f"eps32_alpha10 {label}: radial FFT spectra grouped by optimizer", fontsize=14)
    for row, method in enumerate(METHODS):
        ax = axes[row]
        for block in labels:
            item = blocks.get(block, {}).get(method)
            if not item or field not in item:
                continue
            power = fft_power(item[field])
            centers, profs = radial_profile(power, rho)
            mean = np.nanmean(profs, axis=0)
            std = np.nanstd(profs, axis=0)
            color = BLOCK_COLORS.get(block, None)
            ax.plot(centers, mean, color=color, label=block, linewidth=1.7)
            ax.fill_between(centers, mean - std, mean + std, color=color, alpha=0.08, linewidth=0)
        ax.axvline(RADIAL_AXIS_CUTOFF, color="black", linestyle=":", linewidth=1.2, alpha=0.85, label="axis cutoff rho=sqrt(2)/3" if row == 0 else None)
        ax.axvline(RADIAL_CORNER_CUTOFF, color="black", linestyle="--", linewidth=1.2, alpha=0.85, label="box-corner rho=2/3" if row == 0 else None)
        ax.set_yscale("log")
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylabel("normalized radial power")
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=7, ncol=2, loc="best")
    axes[-1].set_xlabel("normalized radial frequency rho")
    safe = field.replace("_", "-")
    out = out_dir / f"eps32_alpha10_{safe}_radial_fft_by_method_all_blocks.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loss_blocks = discover_loss_curves(args.pair_root)
    final_blocks = discover_final_outputs(args.pair_root)
    pngs = [
        str(save_loss_curves_by_method(loss_blocks, args.out_dir, logy=False)),
        str(save_loss_curves_by_method(loss_blocks, args.out_dir, logy=True)),
    ]
    for field, label in SPECTRAL_FIELDS:
        pngs.append(str(save_radial_spectrum_by_method(final_blocks, field, label, args.out_dir)))

    report = {
        "pair_root": str(args.pair_root),
        "out_dir": str(args.out_dir),
        "grouping": "method panels, all loss/mode blocks overlaid",
        "radial_axis_cutoff_rho": float(RADIAL_AXIS_CUTOFF),
        "radial_corner_cutoff_rho": float(RADIAL_CORNER_CUTOFF),
        "pngs": pngs,
    }
    report_path = args.out_dir / "eps32_alpha10_method_grouped_curves_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
