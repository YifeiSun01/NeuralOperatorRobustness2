#!/usr/bin/env python3
"""Add final attack-delta FFT figures to the Darcy CFlow polished report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = PROJECT / "outputs/darcy_cflow_timematched_organized_release_20260614"
DEFAULT_FINAL = PROJECT / "outputs/darcy_cflow_final_robustness_20260615"


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    color: str


METHODS = [
    Method("baseline", "baseline", "#6b7280"),
    Method("loss1", "loss1", "#7c3aed"),
    Method("loss2", "loss2", "#2563eb"),
    Method("loss3", "loss3", "#059669"),
    Method("physics_loss", "Physics Loss", "#f59e0b"),
    Method("random_clean", "random clean", "#db2777"),
    Method("random_solver", "random solver", "#111827"),
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT))
    except ValueError:
        return str(path)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def squeeze_delta(delta: np.ndarray) -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float64)
    while arr.ndim > 2 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected delta with shape [N,H,W] or [N,H,W,1], got {delta.shape}")
    return np.nan_to_num(arr, copy=False)


def fft_power2(image: np.ndarray) -> np.ndarray:
    d = np.asarray(image, dtype=np.float64)
    d = d - np.mean(d)
    fft = np.fft.fftshift(np.fft.fft2(d))
    return np.abs(fft) ** 2


def radial_bins(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    fy = np.fft.fftshift(np.fft.fftfreq(h, d=1.0 / h))
    fx = np.fft.fftshift(np.fft.fftfreq(w, d=1.0 / w))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    bins = np.floor(radius).astype(int)
    return bins, np.arange(int(bins.max()) + 1, dtype=float)


def radial_power_from_power2(power2: np.ndarray, bins: np.ndarray) -> np.ndarray:
    radial = np.bincount(bins.ravel(), weights=power2.ravel(), minlength=int(bins.max()) + 1).astype(float)
    total = float(np.sum(radial[1:]))
    if total > 0:
        radial = radial / total
    return radial


def spectral_stats(radial: np.ndarray, freq: np.ndarray) -> dict[str, float]:
    if radial.size <= 1:
        return {"high_freq_ratio": float("nan"), "spectral_centroid": float("nan")}
    non_dc = radial[1:]
    freq_non_dc = freq[1:]
    total = float(np.sum(non_dc))
    if total <= 0:
        return {"high_freq_ratio": 0.0, "spectral_centroid": 0.0}
    norm_freq = freq_non_dc / max(float(freq_non_dc.max()), 1.0)
    high = norm_freq >= 0.5
    return {
        "high_freq_ratio": float(np.sum(non_dc[high]) / total),
        "spectral_centroid": float(np.sum(non_dc * norm_freq) / total),
    }


def method_delta_files(delta_dir: Path, method: Method, attack: pd.DataFrame, split: str) -> list[Path]:
    sub = attack[(attack["method"] == method.key) & (attack["split"] == split)].copy()
    paths = sorted({Path(p) for p in sub["delta_npz"].dropna().astype(str)})
    resolved = []
    for path in paths:
        p = PROJECT / path if not path.is_absolute() else path
        if p.exists():
            resolved.append(p)
    return resolved


def load_method_fft(method: Method, files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    dataset_rows = []
    mean_power2 = None
    total_images = 0
    bins = None
    freq = None

    for path in files:
        data = np.load(path)
        delta = squeeze_delta(data["delta"])
        dataset_id = str(data["dataset_id"].item()) if np.asarray(data["dataset_id"]).ndim == 0 else path.stem
        sample_powers = []
        for image in delta:
            p2 = fft_power2(image)
            if mean_power2 is None:
                mean_power2 = np.zeros_like(p2, dtype=np.float64)
                bins, freq = radial_bins(p2.shape)
            mean_power2 += p2
            total_images += 1
            sample_powers.append(radial_power_from_power2(p2, bins))
        matrix = np.vstack(sample_powers)
        radial = np.nanmean(matrix, axis=0)
        stats = spectral_stats(radial, freq)
        dataset_rows.append(
            {
                "method_key": method.key,
                "method": method.label,
                "dataset_id": dataset_id,
                "delta_npz": rel(path),
                "sample_count": int(delta.shape[0]),
                "fft_high_freq_ratio": stats["high_freq_ratio"],
                "fft_spectral_centroid": stats["spectral_centroid"],
                "radial_power_json": json.dumps([float(x) for x in radial[1:]], separators=(",", ":")),
            }
        )

    if mean_power2 is None or bins is None or freq is None:
        raise FileNotFoundError(f"no delta NPZ files found for {method.key}")
    mean_power2 = mean_power2 / max(total_images, 1)
    total = float(np.sum(mean_power2))
    if total > 0:
        mean_power2 = mean_power2 / total
    dataset_df = pd.DataFrame(dataset_rows)
    radial_matrix = np.vstack([np.asarray(json.loads(row), dtype=float) for row in dataset_df["radial_power_json"]])
    return mean_power2, freq[1:], radial_matrix, dataset_df


def plot_method_fft(
    method: Method,
    mean_power2: np.ndarray,
    freq_modes: np.ndarray,
    radial_matrix: np.ndarray,
    dataset_df: pd.DataFrame,
    method_dir: Path,
) -> list[Path]:
    method_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    log_power = np.log10(np.clip(mean_power2, 1e-20, None))
    finite = log_power[np.isfinite(log_power)]
    vmin, vmax = np.nanpercentile(finite, [2, 99.6]) if finite.size else (-12, 0)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    im = ax.imshow(log_power, cmap="magma", origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(f"{method.label}: final attack delta FFT log power", loc="left", fontweight="bold", pad=10)
    ax.set_xlabel("shifted Fourier x index")
    ax.set_ylabel("shifted Fourier y index")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("log10 normalized mean FFT power")
    fig.suptitle("Darcy CFlow final delta FFT heatmap", fontsize=14, fontweight="bold", y=0.99)
    out = method_dir / f"polished_final_delta_fft_log_magnitude_heatmap_{method.key}.png"
    fig.savefig(out)
    plt.close(fig)
    outputs.append(out)

    order = np.argsort(dataset_df["fft_spectral_centroid"].to_numpy(dtype=float))
    sorted_matrix = radial_matrix[order]
    sorted_ids = dataset_df.iloc[order]["dataset_id"].to_list()
    log_matrix = np.log10(np.clip(sorted_matrix, 1e-18, None))
    finite = log_matrix[np.isfinite(log_matrix)]
    vmin, vmax = np.nanpercentile(finite, [2, 99.5]) if finite.size else (-18, 0)

    fig = plt.figure(figsize=(14.0, 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.28], hspace=0.24)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_line = fig.add_subplot(gs[1, 0])
    im = ax_heat.imshow(
        log_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        extent=[float(freq_modes[0]), float(freq_modes[-1]), 0, len(sorted_ids)],
    )
    ax_heat.set_title(f"{method.label}: final delta radial FFT power by dataset", loc="left", fontweight="bold", pad=8)
    ax_heat.set_xlabel("radial Fourier mode")
    ax_heat.set_ylabel("generalization datasets")
    tick_idx = np.linspace(0, len(sorted_ids) - 1, min(10, len(sorted_ids))).astype(int)
    ax_heat.set_yticks(tick_idx + 0.5)
    ax_heat.set_yticklabels(
        [sorted_ids[i].replace("darcy_binary_loss3targeted_20260611_", "") for i in tick_idx],
        fontsize=7,
    )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.024, pad=0.018)
    cbar.set_label("log10 normalized radial FFT power")

    mean_radial = np.nanmean(sorted_matrix, axis=0)
    ax_line.plot(freq_modes, mean_radial + 1e-18, color=method.color, lw=2.1)
    ax_line.set_yscale("log")
    ax_line.set_xlabel("radial Fourier mode")
    ax_line.set_ylabel("mean power")
    ax_line.set_title("Mean final-delta radial spectrum across the 50 generalization datasets", loc="left", fontsize=10.5, fontweight="bold")
    ax_line.grid(True, alpha=0.25)
    fig.suptitle("Darcy CFlow final attack delta frequency content", fontsize=15, fontweight="bold", y=0.99)
    out = method_dir / f"polished_final_delta_fft_dataset_radial_heatmap_{method.key}.png"
    fig.savefig(out)
    plt.close(fig)
    outputs.append(out)
    return outputs


def plot_overview(method_images: dict[str, np.ndarray], method_radials: dict[str, tuple[np.ndarray, np.ndarray]], out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    logs = {k: np.log10(np.clip(v, 1e-20, None)) for k, v in method_images.items()}
    finite = np.concatenate([v[np.isfinite(v)].ravel() for v in logs.values()])
    vmin, vmax = np.nanpercentile(finite, [2, 99.6]) if finite.size else (-12, 0)

    fig, axes = plt.subplots(2, 4, figsize=(15.6, 7.8), constrained_layout=False)
    axes = axes.ravel()
    im = None
    for ax, method in zip(axes, METHODS):
        arr = logs.get(method.key)
        if arr is None:
            ax.axis("off")
            continue
        im = ax.imshow(arr, cmap="magma", origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(method.label, fontweight="bold", fontsize=11, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(METHODS) :]:
        ax.axis("off")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.022, pad=0.018)
        cbar.set_label("log10 normalized mean FFT power")
    fig.suptitle("Darcy CFlow final attack delta FFT heatmaps", fontsize=17, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.94,
        "Mean over final attack50 deltas on the 50 binary 20260611 generalization datasets.",
        ha="center",
        fontsize=10.5,
        color="#4b5563",
    )
    out = out_dir / "polished_final_delta_fft_log_magnitude_grid_all7.png"
    fig.savefig(out)
    plt.close(fig)
    outputs.append(out)

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    for method in METHODS:
        item = method_radials.get(method.key)
        if item is None:
            continue
        freq, radial = item
        ax.plot(freq, radial + 1e-18, color=method.color, lw=2.0, alpha=0.86, label=method.label)
    ax.set_yscale("log")
    ax.set_xlabel("radial Fourier mode")
    ax.set_ylabel("mean normalized power")
    ax.set_title("Final attack delta radial FFT spectra", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, frameon=False, loc="upper right")
    fig.suptitle("Darcy CFlow final delta frequency comparison", fontsize=16, fontweight="bold", y=0.99)
    out = out_dir / "polished_final_delta_radial_spectrum_all7.png"
    fig.savefig(out)
    plt.close(fig)
    outputs.append(out)
    return outputs


def update_readme(readme: Path, generated: list[Path]) -> None:
    current = readme.read_text(encoding="utf-8") if readme.exists() else "# Darcy CFlow Polished Report\n"
    marker = "\n## Final Delta FFT Addendum\n"
    section = marker + "\n".join(
        [
            "Final attack-delta FFT figures were added from the completed attack50 delta NPZ files.",
            "",
            "These are final-checkpoint FFT diagnostics, not epoch-wise attack-probe histories.",
            "",
            "Generated files:",
            *[f"- `{rel(p)}`" for p in generated],
            "",
        ]
    )
    if marker in current:
        current = current.split(marker)[0].rstrip() + "\n" + section
    else:
        current = current.rstrip() + "\n" + section
    readme.write_text(current, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--final-bundle", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--split", default="generalization")
    args = parser.parse_args()

    setup_style()
    release = args.release.resolve()
    final = args.final_bundle.resolve()
    attack_csv = final / "data" / "robustness_attack_52datasets_samples.csv"
    delta_dir = final / "data" / "robustness_deltas"
    attack = pd.read_csv(attack_csv)

    fig_root = release / "figures" / "polished_report"
    data_root = release / "data" / "polished_report" / "final_delta_fft"
    overview_dir = fig_root / "final_delta_fft"
    data_root.mkdir(parents=True, exist_ok=True)
    overview_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    all_rows = []
    method_images: dict[str, np.ndarray] = {}
    method_radials: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for method in METHODS:
        files = method_delta_files(delta_dir, method, attack, args.split)
        mean_power2, freq_modes, radial_matrix, dataset_df = load_method_fft(method, files)
        method_images[method.key] = mean_power2
        method_radials[method.key] = (freq_modes, np.nanmean(radial_matrix, axis=0))
        all_rows.append(dataset_df)

        method_dir = fig_root / method.key
        generated.extend(plot_method_fft(method, mean_power2, freq_modes, radial_matrix, dataset_df, method_dir))

    metrics = pd.concat(all_rows, ignore_index=True)
    metrics_csv = data_root / "final_delta_fft_dataset_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)

    summary = (
        metrics.groupby(["method_key", "method"], as_index=False)
        .agg(
            datasets=("dataset_id", "nunique"),
            samples=("sample_count", "sum"),
            fft_high_freq_ratio_mean=("fft_high_freq_ratio", "mean"),
            fft_high_freq_ratio_median=("fft_high_freq_ratio", "median"),
            fft_spectral_centroid_mean=("fft_spectral_centroid", "mean"),
            fft_spectral_centroid_median=("fft_spectral_centroid", "median"),
        )
        .sort_values("fft_spectral_centroid_mean")
    )
    summary_csv = data_root / "final_delta_fft_summary_by_model.csv"
    summary.to_csv(summary_csv, index=False)

    generated.extend(plot_overview(method_images, method_radials, overview_dir))

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "release": rel(release),
        "final_bundle": rel(final),
        "source_attack_csv": rel(attack_csv),
        "source_delta_dir": rel(delta_dir),
        "split": args.split,
        "figures": [rel(p) for p in generated],
        "tables": [rel(metrics_csv), rel(summary_csv)],
        "notes": [
            "These are final-checkpoint attack50 delta FFT diagnostics on the binary 20260611 root.",
            "They are not the old epoch-wise attack-probe FFT history because the organized release does not contain per-epoch attack_probe_samples for every final model.",
        ],
    }
    manifest_path = release / "manifests" / "final_delta_fft_polished_report_20260615.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_readme(fig_root / "README.md", generated + [metrics_csv, summary_csv, manifest_path])
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
