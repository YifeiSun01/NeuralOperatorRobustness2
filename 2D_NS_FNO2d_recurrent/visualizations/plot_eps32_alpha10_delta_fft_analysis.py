#!/usr/bin/env python3
"""CPU-only Fourier analysis for eps32_alpha10 final_delta arrays.

Reads saved final_state_outputs.npz files. Does not import torch/jax or rerun
model/solver.
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
METHOD_COLORS = {
    "raw_add": "tab:blue",
    "raw_replace": "tab:orange",
    "steepest_add": "tab:green",
    "steepest_replace": "tab:red",
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
LOW_CUTOFF = 0.15
MID_CUTOFF = 0.35
NBINS = 72


def block_from_path(path: Path) -> tuple[str, str, str]:
    mode_part = next(part for part in path.parts if part.startswith("mode_"))
    mode_spec = mode_part.split("_p2_q2", 1)[0][len("mode_"):]
    mode = MODE_NAMES.get(mode_spec, mode_spec)
    loss = next(part for part in path.parts if part in ("loss1", "loss2", "loss3"))
    return f"{loss}/{mode}", loss, mode_spec


def frequency_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq = np.fft.fftshift(np.fft.fftfreq(n))
    fx, fy = np.meshgrid(freq, freq, indexing="xy")
    radius = np.sqrt(fx * fx + fy * fy)
    rho = radius / np.sqrt(0.5 * 0.5 + 0.5 * 0.5)
    return fx, fy, rho


def fft_power(delta: np.ndarray) -> np.ndarray:
    # delta shape [B, H, W]. FFT normalization is irrelevant after normalizing fractions.
    f = np.fft.fftshift(np.fft.fft2(delta, axes=(-2, -1)), axes=(-2, -1))
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


def metrics_for_power(power: np.ndarray, rho: np.ndarray) -> dict[str, np.ndarray]:
    flat_rho = rho.reshape(-1)
    flat_power = power.reshape(power.shape[0], -1).astype(np.float64)
    dc_mask = flat_rho < 1e-12
    non_dc = ~dc_mask
    total = np.sum(flat_power[:, non_dc], axis=1)
    total = np.maximum(total, 1e-30)
    low = (flat_rho > 1e-12) & (flat_rho <= LOW_CUTOFF)
    mid = (flat_rho > LOW_CUTOFF) & (flat_rho <= MID_CUTOFF)
    high = flat_rho > MID_CUTOFF
    low_power = np.sum(flat_power[:, low], axis=1)
    mid_power = np.sum(flat_power[:, mid], axis=1)
    high_power = np.sum(flat_power[:, high], axis=1)
    centroid = np.sum(flat_power[:, non_dc] * flat_rho[non_dc][None, :], axis=1) / total
    rms_radius = np.sqrt(np.sum(flat_power[:, non_dc] * (flat_rho[non_dc] ** 2)[None, :], axis=1) / total)
    high_low_ratio = high_power / np.maximum(low_power, 1e-30)
    return {
        "low_frac": low_power / total,
        "mid_frac": mid_power / total,
        "high_frac": high_power / total,
        "spectral_centroid": centroid,
        "spectral_rms_radius": rms_radius,
        "high_low_ratio": high_low_ratio,
    }


def discover(pair_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for npz_path in sorted(pair_root.glob("mode_*/batch_0000_0009/loss*/*/final_state_outputs.npz")):
        method = npz_path.parent.name
        if method not in METHODS:
            continue
        block, loss, mode_spec = block_from_path(npz_path)
        z = np.load(npz_path)
        delta = z["final_delta"].astype(np.float32)
        out.setdefault(block, {})[method] = {
            "npz": str(npz_path),
            "loss": loss,
            "mode_spec": mode_spec,
            "dataset_indices": z["dataset_indices"].astype(int),
            "delta": delta,
            "clean_true_loss": z["clean_true_loss"].astype(np.float64),
            "adv_true_loss": z["adv_true_loss"].astype(np.float64),
        }
    return out


def ordered_blocks(blocks: dict[str, Any]) -> list[str]:
    return [b for b in BLOCK_ORDER if b in blocks] + sorted([b for b in blocks if b not in BLOCK_ORDER])


def save_fft_magnitude_grid(blocks: dict[str, Any], out_dir: Path, rho: np.ndarray) -> Path:
    labels = ordered_blocks(blocks)
    spectra = {}
    values = []
    for block in labels:
        for method in METHODS:
            item = blocks.get(block, {}).get(method)
            if not item:
                continue
            power = fft_power(item["delta"][:1])[0]
            amp = np.sqrt(power)
            log_amp = np.log10(amp + 1e-6)
            spectra[(block, method)] = log_amp
            values.append(log_amp)
    vals = np.concatenate([v.ravel() for v in values])
    vmin, vmax = np.percentile(vals[np.isfinite(vals)], [1, 99.5])

    fig, axes = plt.subplots(len(labels), len(METHODS), figsize=(4.2 * len(METHODS), 3.2 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axes = np.asarray([axes])
    fig.suptitle("eps32_alpha10 final_delta FFT log magnitude | sample_position=0", fontsize=14)
    im = None
    for i, block in enumerate(labels):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            spec = spectra.get((block, method))
            if spec is None:
                ax.axis("off")
                continue
            im = ax.imshow(spec, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{block}\n{method}", fontsize=8)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label="log10(|FFT(delta)|)")
    out = out_dir / "eps32_alpha10_final_delta_fft_log_magnitude_sample0_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_radial_profiles(blocks: dict[str, Any], out_dir: Path, rho: np.ndarray) -> tuple[Path, dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    labels = ordered_blocks(blocks)
    profiles: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 2.7 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axes = np.asarray([axes])
    fig.suptitle("eps32_alpha10 final_delta radial FFT power spectrum | batch mean +/- std", fontsize=14)
    for i, block in enumerate(labels):
        ax = axes[i]
        for method in METHODS:
            item = blocks.get(block, {}).get(method)
            if not item:
                continue
            power = fft_power(item["delta"])
            centers, profs = radial_profile(power, rho)
            mean = np.nanmean(profs, axis=0)
            std = np.nanstd(profs, axis=0)
            profiles[(block, method)] = (centers, mean, std)
            color = METHOD_COLORS.get(method)
            ax.plot(centers, mean, color=color, label=method, linewidth=1.8)
            ax.fill_between(centers, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.axvline(LOW_CUTOFF, color="gray", linestyle=":", linewidth=1)
        ax.axvline(MID_CUTOFF, color="gray", linestyle=":", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylabel("normalized radial power")
        ax.set_title(block)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=4)
    axes[-1].set_xlabel("normalized radial frequency rho")
    out = out_dir / "eps32_alpha10_final_delta_radial_fft_profiles.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, profiles


def save_band_heatmaps(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    labels = [b for b in BLOCK_ORDER if any(r["block"] == b for r in rows)]
    metrics = ["low_frac_mean", "mid_frac_mean", "high_frac_mean", "spectral_centroid_mean", "high_low_ratio_mean"]
    titles = ["Low freq energy", "Mid freq energy", "High freq energy", "Spectral centroid", "High/low ratio"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), max(6, 0.7 * len(labels))), constrained_layout=True)
    for ax, metric, title in zip(axes, metrics, titles):
        mat = np.full((len(labels), len(METHODS)), np.nan)
        for r in rows:
            if r["block"] in labels and r["method"] in METHODS:
                mat[labels.index(r["block"]), METHODS.index(r["method"])] = r[metric]
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    value = mat[i, j]
                    label = f"{value:.2f}" if metric == "high_low_ratio_mean" else f"{value:.3f}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    out = out_dir / "eps32_alpha10_final_delta_fft_band_metrics_heatmaps.png"
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
    sample = next(iter(next(iter(blocks.values())).values()))["delta"]
    n = int(sample.shape[-1])
    _, _, rho = frequency_grid(n)

    rows = []
    for block in ordered_blocks(blocks):
        for method in METHODS:
            item = blocks.get(block, {}).get(method)
            if not item:
                continue
            power = fft_power(item["delta"])
            metrics = metrics_for_power(power, rho)
            true_inc = item["adv_true_loss"] - item["clean_true_loss"]
            row = {
                "block": block,
                "method": method,
                "epsilon": 32.0,
                "alpha": 10.0,
                "sample_count": int(item["delta"].shape[0]),
                "low_cutoff_rho": LOW_CUTOFF,
                "mid_cutoff_rho": MID_CUTOFF,
                "clean_true_loss_mean": float(np.mean(item["clean_true_loss"])),
                "adv_true_loss_mean": float(np.mean(item["adv_true_loss"])),
                "true_loss_increase_mean": float(np.mean(true_inc)),
                "true_loss_increase_std": float(np.std(true_inc)),
                "delta_l2_mean": float(np.mean(np.linalg.norm(item["delta"].reshape(item["delta"].shape[0], -1), axis=1))),
                "delta_linf_mean": float(np.mean(np.max(np.abs(item["delta"].reshape(item["delta"].shape[0], -1)), axis=1))),
                "source_npz": item["npz"],
            }
            for key, arr in metrics.items():
                row[f"{key}_mean"] = float(np.mean(arr))
                row[f"{key}_std"] = float(np.std(arr))
            rows.append(row)

    fft_grid = save_fft_magnitude_grid(blocks, args.out_dir, rho)
    radial_png, profiles = save_radial_profiles(blocks, args.out_dir, rho)
    heatmap_png = save_band_heatmaps(rows, args.out_dir)

    summary_csv = args.out_dir / "eps32_alpha10_final_delta_fft_metrics_summary.csv"
    write_csv(summary_csv, rows)
    report = {
        "note": "CPU-only Fourier analysis of saved final_delta arrays; DC excluded from band fractions and spectral centroid.",
        "pair_root": str(args.pair_root),
        "out_dir": str(args.out_dir),
        "low_cutoff_rho": LOW_CUTOFF,
        "mid_cutoff_rho": MID_CUTOFF,
        "fft_log_magnitude_sample0_grid": str(fft_grid),
        "radial_profiles_png": str(radial_png),
        "band_metrics_heatmaps_png": str(heatmap_png),
        "summary_csv": str(summary_csv),
        "rows": rows,
    }
    report_path = args.out_dir / "eps32_alpha10_final_delta_fft_analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[done] {fft_grid}")
    print(f"[done] {radial_png}")
    print(f"[done] {heatmap_png}")
    print(f"[done] {summary_csv}")
    print(f"[done] {report_path}")


if __name__ == "__main__":
    main()
