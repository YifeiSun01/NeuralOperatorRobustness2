#!/usr/bin/env python3
"""Polished SVD visualizations for the Burgers p2q2 checkpoint series."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
FORENSICS = REPO / "forensics" / "burgers_p2q2_checkpoint_series_jacobian_svd_20260601"
OUT_DIR = (
    REPO
    / "visualizations"
    / "burgers_p2q2_adv_training_20260601"
    / "polished_selected_download_burgers_p2q2_20260601"
)

MODEL_SERIES = [
    ("baseline", "Baseline", "#5d6470"),
    ("p2q2_epoch0200", "Epoch 200", "#2f6f9f"),
    ("p2q2_epoch0400", "Epoch 400", "#2f9b75"),
    ("p2q2_epoch0600", "Epoch 600", "#d9a441"),
    ("p2q2_epoch0800", "Epoch 800", "#c35b5b"),
    ("p2q2_epoch1000", "Epoch 1000", "#7b5fb3"),
]
ERROR_SERIES = [
    ("baseline_error", "Baseline error", "#5d6470"),
    ("p2q2_epoch0200_error", "Epoch 200 error", "#2f6f9f"),
    ("p2q2_epoch0400_error", "Epoch 400 error", "#2f9b75"),
    ("p2q2_epoch0600_error", "Epoch 600 error", "#d9a441"),
    ("p2q2_epoch0800_error", "Epoch 800 error", "#c35b5b"),
    ("p2q2_epoch1000_error", "Epoch 1000 error", "#7b5fb3"),
]
SOLVER_STYLE = ("solver", "Solver", "#111111")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d7dce2",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )


def svd_path(sample_dir: Path, subdir: str) -> Path | None:
    paths = list((sample_dir / subdir).glob("*_jacobian_svd.npz"))
    return paths[0] if paths else None


def load_npz_arrays(subdir: str) -> list[np.lib.npyio.NpzFile]:
    arrays = []
    for sample_dir in sorted(FORENSICS.glob("sample_[0-9][0-9][0-9]")):
        path = svd_path(sample_dir, subdir)
        if path is not None:
            arrays.append(np.load(path))
    return arrays


def singular_value_stats(subdir: str) -> tuple[np.ndarray, np.ndarray]:
    arrs = load_npz_arrays(subdir)
    values = np.stack([np.asarray(z["singular_values"], dtype=np.float64)[:100] for z in arrs])
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def vector_matrix(z: np.lib.npyio.NpzFile, side: str) -> np.ndarray:
    if side == "right":
        return np.asarray(z["right_singular_vectors"], dtype=np.float64)
    if side == "left":
        return np.asarray(z["left_singular_vectors"], dtype=np.float64).T
    raise ValueError(side)


def normalized_fft_power(vec: np.ndarray) -> np.ndarray:
    power = np.abs(np.fft.rfft(np.asarray(vec, dtype=np.float64).reshape(-1))) ** 2
    power[0] = 0.0
    total = power.sum()
    if total <= 1e-30:
        return power
    return power / total


def spectrum_stats(subdir: str, side: str, rank: int) -> tuple[np.ndarray, np.ndarray]:
    spectra = []
    for z in load_npz_arrays(subdir):
        mat = vector_matrix(z, side)
        spectra.append(normalized_fft_power(mat[rank - 1]))
    data = np.stack(spectra)
    return data.mean(axis=0), data.std(axis=0, ddof=1)


def frequency_metrics(subdir: str, side: str, ranks: range) -> dict[str, tuple[float, float]]:
    centroids = []
    high50 = []
    high75 = []
    sign_changes = []
    for z in load_npz_arrays(subdir):
        mat = vector_matrix(z, side)
        sv = np.asarray(z["singular_values"], dtype=np.float64)
        k = min(max(ranks), mat.shape[0], sv.size)
        use = [r - 1 for r in ranks if r <= k]
        weights = sv[use] ** 2
        weights = weights / weights.sum() if weights.sum() > 1e-30 else np.ones(len(use)) / len(use)
        sample_centroids = []
        sample_high50 = []
        sample_high75 = []
        sample_sign_changes = []
        for w, idx in zip(weights, use):
            spec = normalized_fft_power(mat[idx])
            modes = np.arange(spec.size, dtype=np.float64)
            centroid = float((modes * spec).sum())
            max_mode = spec.size - 1
            h50 = float(spec[int(np.ceil(0.50 * max_mode)) :].sum())
            h75 = float(spec[int(np.ceil(0.75 * max_mode)) :].sum())
            vec = mat[idx]
            sc = float(np.mean(np.diff(np.signbit(vec)) != 0))
            sample_centroids.append(w * centroid)
            sample_high50.append(w * h50)
            sample_high75.append(w * h75)
            sample_sign_changes.append(w * sc)
        centroids.append(float(np.sum(sample_centroids)))
        high50.append(float(np.sum(sample_high50)))
        high75.append(float(np.sum(sample_high75)))
        sign_changes.append(float(np.sum(sample_sign_changes)))
    def mean_std(xs: list[float]) -> tuple[float, float]:
        arr = np.asarray(xs, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=1))
    return {
        "centroid": mean_std(centroids),
        "high50": mean_std(high50),
        "high75": mean_std(high75),
        "sign_changes": mean_std(sign_changes),
    }


def plot_singular_values() -> list[Path]:
    ranks = np.arange(1, 101)
    outputs = []

    fig, ax = plt.subplots(figsize=(11.5, 7.0), constrained_layout=True)
    mean, std = singular_value_stats(SOLVER_STYLE[0])
    ax.plot(ranks, mean, color=SOLVER_STYLE[2], lw=2.4, ls="--", label=SOLVER_STYLE[1])
    for subdir, label, color in MODEL_SERIES:
        mean, std = singular_value_stats(subdir)
        lower = np.maximum(mean - std, 1e-12)
        upper = mean + std
        ax.plot(ranks, mean, color=color, lw=2.0, label=label)
        ax.fill_between(ranks, lower, upper, color=color, alpha=0.14, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("Singular value rank")
    ax.set_ylabel("Singular value")
    ax.set_title("Model Jacobian top 100 singular values across adversarial-training checkpoints")
    ax.legend(ncol=4, loc="upper right")
    outputs.append(OUT_DIR / "svd_model_jacobian_top100_singular_values_mean_std.png")
    fig.savefig(outputs[-1], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 7.0), constrained_layout=True)
    for subdir, label, color in ERROR_SERIES:
        mean, std = singular_value_stats(subdir)
        lower = np.maximum(mean - std, 1e-12)
        upper = mean + std
        ax.plot(ranks, mean, color=color, lw=2.0, label=label)
        ax.fill_between(ranks, lower, upper, color=color, alpha=0.16, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("Singular value rank")
    ax.set_ylabel("Singular value of J_model - J_solver")
    ax.set_title("Error Jacobian top 100 singular values shrink during p=2, q=2 adversarial training")
    ax.legend(ncol=3, loc="upper right")
    outputs.append(OUT_DIR / "svd_error_jacobian_top100_singular_values_mean_std.png")
    fig.savefig(outputs[-1], dpi=220)
    plt.close(fig)
    return outputs


def plot_frequency_metrics() -> Path:
    epochs = np.array([0, 200, 400, 600, 800, 1000], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), constrained_layout=True)
    panels = [
        ("right", "centroid", "Right/input singular vectors: spectral centroid", "Fourier mode"),
        ("left", "centroid", "Left/output singular vectors: spectral centroid", "Fourier mode"),
        ("right", "high50", "Right/input singular vectors: high-frequency energy", "Power share"),
        ("left", "high50", "Left/output singular vectors: high-frequency energy", "Power share"),
    ]
    for ax, (side, metric, title, ylabel) in zip(axes.flat, panels):
        for series, name, linestyle in [
            (MODEL_SERIES, "Model Jacobian top20", "-"),
            (ERROR_SERIES, "Error Jacobian top20", "--"),
        ]:
            means = []
            stds = []
            for subdir, _, _ in series:
                m, s = frequency_metrics(subdir, side, range(1, 21))[metric]
                means.append(m)
                stds.append(s)
            means = np.asarray(means)
            stds = np.asarray(stds)
            color = "#2f6f9f" if name.startswith("Model") else "#c35b5b"
            ax.plot(epochs, means, color=color, lw=2.2, ls=linestyle, label=name)
            ax.fill_between(epochs, np.maximum(means - stds, 0), means + stds, color=color, alpha=0.13)
        ax.set_title(title)
        ax.set_xlabel("Training epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
    out = OUT_DIR / "svd_singular_vector_frequency_metrics_top20_mean_std.png"
    fig.suptitle("Frequency metrics of top-20 singular vectors across p=2, q=2 adversarial-training checkpoints", y=1.02, fontsize=15)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rank_spectra(series: list[tuple[str, str, str]], side: str, prefix: str, title: str) -> Path:
    modes = np.arange(513)
    fig, axes = plt.subplots(5, 1, figsize=(12.5, 12.5), sharex=True, constrained_layout=True)
    global_min = np.inf
    global_max = 0.0
    cache: dict[tuple[str, int], np.ndarray] = {}
    for subdir, _, _ in series:
        for rank in range(1, 6):
            mean, _ = spectrum_stats(subdir, side, rank)
            cache[(subdir, rank)] = mean
            positive = mean[1:]
            global_min = min(global_min, float(positive[positive > 0].min()) if np.any(positive > 0) else global_min)
            global_max = max(global_max, float(positive.max()))
    for rank, ax in enumerate(axes, start=1):
        for subdir, label, color in series:
            mean = cache[(subdir, rank)]
            ax.plot(modes[1:], np.maximum(mean[1:], 1e-12), color=color, lw=1.55, alpha=0.88, label=label)
        ax.set_yscale("log")
        ax.set_ylim(max(global_min * 0.6, 1e-12), global_max * 1.8)
        ax.set_xlim(1, 512)
        ax.set_ylabel(f"rank {rank}")
        ax.grid(True, which="major", alpha=0.5)
        if rank == 1:
            ax.legend(ncol=3, loc="upper right")
    axes[-1].set_xlabel("Fourier mode")
    fig.suptitle(title, fontsize=15, y=1.01)
    out = OUT_DIR / f"{prefix}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def update_manifest(paths: list[Path]) -> None:
    manifest = OUT_DIR / "selected_figures_manifest.json"
    data = {}
    if manifest.exists():
        data = json.loads(manifest.read_text())
    existing = data.get("figures", [])
    names = {item.get("file") for item in existing if isinstance(item, dict)}
    for path in paths:
        if path.name not in names:
            existing.append({"file": path.name, "kind": "burgers_p2q2_svd_diagnostic"})
    data["figures"] = existing
    manifest.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs.extend(plot_singular_values())
    outputs.append(plot_frequency_metrics())
    outputs.append(
        plot_rank_spectra(
            ERROR_SERIES,
            "right",
            "svd_error_right_input_singular_vector_fft_spectra_ranks1to5",
            "Error Jacobian right/input singular-vector spectra for ranks 1 to 5",
        )
    )
    outputs.append(
        plot_rank_spectra(
            ERROR_SERIES,
            "left",
            "svd_error_left_output_singular_vector_fft_spectra_ranks1to5",
            "Error Jacobian left/output singular-vector spectra for ranks 1 to 5",
        )
    )
    outputs.append(
        plot_rank_spectra(
            MODEL_SERIES,
            "right",
            "svd_model_right_input_singular_vector_fft_spectra_ranks1to5",
            "Model Jacobian right/input singular-vector spectra for ranks 1 to 5",
        )
    )
    outputs.append(
        plot_rank_spectra(
            MODEL_SERIES,
            "left",
            "svd_model_left_output_singular_vector_fft_spectra_ranks1to5",
            "Model Jacobian left/output singular-vector spectra for ranks 1 to 5",
        )
    )
    update_manifest(outputs)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
