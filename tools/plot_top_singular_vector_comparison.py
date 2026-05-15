#!/usr/bin/env python3
"""Plot top right singular vectors and Fourier spectra across FNO/solver/DeepONet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FNO_ROOT = PROJECT_ROOT / "forensics" / "local_jacobian_frequency_20260514" / "01_explicit_jacobian_multi_index"
DEFAULT_FNO_SOLVER_ROOT = PROJECT_ROOT / "forensics" / "fno_solver_jacobian_similarity_20260514_raw_recomputed"
DEFAULT_DEEPONET_ROOT = PROJECT_ROOT / "forensics" / "deeponet_solver_jacobian_similarity_20260515"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "forensics" / "top_singular_vector_comparison_20260515"
EPS = 1e-12


def load_svd(npz_path: Path) -> dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path) as data:
        return {
            "s": np.asarray(data["singular_values"], dtype=np.float64),
            "Vh": np.asarray(data["right_singular_vectors"], dtype=np.float64),
        }


def svd_paths(args: argparse.Namespace, index: int) -> list[dict[str, Any]]:
    return [
        {
            "key": "fno_nu0p001",
            "label": "FNO nu=0.001",
            "short": "FNO",
            "path": args.fno_root / f"index_{index:03d}" / "fno" / f"fno_index{index}_jacobian_svd.npz",
            "color": "#1f77b4",
        },
        {
            "key": "solver_nu0p001",
            "label": "solver nu=0.001",
            "short": "solver 0.001",
            "path": args.fno_solver_root / f"index_{index:03d}" / "solver" / f"solver_index{index}_jacobian_svd.npz",
            "color": "#2ca02c",
        },
        {
            "key": "deeponet_nu0p01",
            "label": "DeepONet nu=0.01",
            "short": "DeepONet",
            "path": args.deeponet_root / f"index_{index:03d}" / "deeponet" / f"deeponet_index{index}_jacobian_svd.npz",
            "color": "#d62728",
        },
        {
            "key": "solver_nu0p01",
            "label": "solver nu=0.01",
            "short": "solver 0.01",
            "path": args.deeponet_root / f"index_{index:03d}" / "solver" / f"solver_index{index}_jacobian_svd.npz",
            "color": "#9467bd",
        },
    ]


def align_for_display(v: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    out = np.asarray(v, dtype=np.float64).copy()
    if reference is not None:
        dot = float(np.dot(out, reference))
        if abs(dot) > 1e-8:
            if dot < 0.0:
                out *= -1.0
        else:
            idx = int(np.argmax(np.abs(out)))
            if out[idx] < 0.0:
                out *= -1.0
    else:
        idx = int(np.argmax(np.abs(out)))
        if out[idx] < 0.0:
            out *= -1.0
    return out


def normalize_shape(v: np.ndarray) -> np.ndarray:
    return v / (float(np.max(np.abs(v))) + EPS)


def zero_crossings(v: np.ndarray) -> int:
    x = np.asarray(v, dtype=np.float64)
    nz = x[np.abs(x) > 1e-14]
    if nz.size <= 1:
        return 0
    return int(np.sum(np.signbit(nz[:-1]) != np.signbit(nz[1:])))


def high_freq_energy(v: np.ndarray, cutoff: int) -> float:
    coeff = np.fft.rfft(v)
    energy = np.abs(coeff) ** 2
    return float(energy[cutoff:].sum() / (energy.sum() + EPS))


def spectrum(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coeff = np.fft.rfft(v)
    energy = np.abs(coeff) ** 2
    energy = energy / (energy.sum() + EPS)
    k = np.arange(len(energy))
    return k, energy


def plot_index(index: int, entries: list[dict[str, Any]], out_dir: Path, top_k: int, domain: float) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loaded: list[dict[str, Any]] = []
    for entry in entries:
        svd = load_svd(entry["path"])
        item = dict(entry)
        item.update(svd)
        loaded.append(item)

    n = loaded[0]["Vh"].shape[1]
    x = np.linspace(0.0, domain, n, endpoint=False)
    top_k = min(top_k, *(len(item["s"]) for item in loaded))
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any]] = []
    fig, axes = plt.subplots(top_k, len(loaded), figsize=(4.5 * len(loaded), 2.5 * top_k), sharex=True, sharey=True)
    if top_k == 1:
        axes = np.asarray([axes])
    for r in range(top_k):
        ref = align_for_display(loaded[0]["Vh"][r], None)
        for c, item in enumerate(loaded):
            raw = item["Vh"][r]
            aligned = align_for_display(raw, ref)
            y = normalize_shape(aligned)
            ax = axes[r, c]
            ax.plot(x, y, lw=1.2, color=item["color"])
            ax.axhline(0.0, lw=0.6, color="0.75")
            ax.set_ylim(-1.08, 1.08)
            if c == 0:
                ax.set_ylabel(f"rank {r + 1}\nshape")
            if r == 0:
                ax.set_title(item["label"])
            if r == top_k - 1:
                ax.set_xlabel("x")
            zc = zero_crossings(aligned)
            hi128 = high_freq_energy(aligned, 128)
            ax.text(
                0.02,
                0.92,
                f"sigma={item['s'][r]:.3g}\nhi128={hi128:.2g}\nzc={zc}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "0.82", "alpha": 0.88},
            )
            metrics.append(
                {
                    "sample_index": index,
                    "rank": r + 1,
                    "operator": item["key"],
                    "label": item["label"],
                    "singular_value": float(item["s"][r]),
                    "right_hi128": hi128,
                    "right_zero_crossings": zc,
                }
            )
    fig.suptitle(f"Top-{top_k} right singular vectors, sample {index}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    line_path = out_dir / f"index_{index:03d}_top{top_k}_right_singular_vectors_lines.png"
    fig.savefig(line_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(top_k, len(loaded), figsize=(4.5 * len(loaded), 2.5 * top_k), sharex=True, sharey=True)
    if top_k == 1:
        axes = np.asarray([axes])
    for r in range(top_k):
        ref = align_for_display(loaded[0]["Vh"][r], None)
        for c, item in enumerate(loaded):
            aligned = align_for_display(item["Vh"][r], ref)
            k, energy = spectrum(aligned)
            ax = axes[r, c]
            ax.semilogy(k[1:], energy[1:] + EPS, lw=1.1, color=item["color"])
            ax.axvline(128, lw=0.7, color="0.4", ls="--")
            ax.set_xlim(1, n // 2)
            ax.set_ylim(1e-16, 1.0)
            if c == 0:
                ax.set_ylabel(f"rank {r + 1}\nenergy frac")
            if r == 0:
                ax.set_title(item["label"])
            if r == top_k - 1:
                ax.set_xlabel("Fourier mode k")
            ax.grid(alpha=0.25)
    fig.suptitle(f"Fourier energy of top-{top_k} right singular vectors, sample {index}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fft_path = out_dir / f"index_{index:03d}_top{top_k}_right_singular_vectors_fft.png"
    fig.savefig(fft_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(top_k, 1, figsize=(12.0, 2.5 * top_k), sharex=True)
    if top_k == 1:
        axes = np.asarray([axes])
    for r in range(top_k):
        ref = align_for_display(loaded[0]["Vh"][r], None)
        ax = axes[r]
        for item in loaded:
            aligned = align_for_display(item["Vh"][r], ref)
            ax.plot(x, normalize_shape(aligned), lw=1.1, color=item["color"], label=f"{item['short']} sigma={item['s'][r]:.3g}")
        ax.axhline(0.0, lw=0.6, color="0.75")
        ax.set_ylim(-1.08, 1.08)
        ax.set_ylabel(f"rank {r + 1}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("x")
    fig.suptitle(f"Overlay: top-{top_k} right singular vector shapes, sample {index}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    overlay_path = out_dir / f"index_{index:03d}_top{top_k}_right_singular_vectors_overlay.png"
    fig.savefig(overlay_path, dpi=180)
    plt.close(fig)

    return {"line_plot": str(line_path), "fft_plot": str(fft_path), "overlay_plot": str(overlay_path), "metrics": metrics}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 7, 40, 47, 115])
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--domain", type=float, default=2.0)
    parser.add_argument("--fno-root", type=Path, default=DEFAULT_FNO_ROOT)
    parser.add_argument("--fno-solver-root", type=Path, default=DEFAULT_FNO_SOLVER_ROOT)
    parser.add_argument("--deeponet-root", type=Path, default=DEFAULT_DEEPONET_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "sample_indices": args.sample_indices,
        "top_k": args.top_k,
        "operator_columns": [
            "FNO nu=0.001",
            "solver nu=0.001",
            "DeepONet nu=0.01",
            "solver nu=0.01",
        ],
        "note": "Right singular vectors are sign-aligned for display and scaled by max absolute value; singular values are shown in subplot labels.",
        "samples": [],
    }
    for index in args.sample_indices:
        print(f"[plot] index={index}", flush=True)
        out_dir = args.out_root / f"index_{index:03d}"
        result = plot_index(index, svd_paths(args, index), out_dir, args.top_k, args.domain)
        all_metrics.extend(result["metrics"])
        manifest["samples"].append({k: v for k, v in result.items() if k != "metrics"})
    write_csv(args.out_root / "top_right_singular_vector_plot_metrics.csv", all_metrics)
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# Top Right Singular Vector Comparison",
        "",
        "This directory compares the top right singular vectors of:",
        "",
        "- FNO `nu=0.001`",
        "- solver `nu=0.001`",
        "- DeepONet `nu=0.01`",
        "- solver `nu=0.01`",
        "",
        "Each sample has:",
        "",
        "- `*_lines.png`: top-4 right singular vectors as line plots; titles include singular value, `hi128`, and zero crossings.",
        "- `*_fft.png`: Fourier energy spectra of the same vectors, with `k=128` marked.",
        "- `*_overlay.png`: shape-normalized overlay for direct visual comparison.",
        "",
        "Display note: singular vectors are sign-ambiguous, so signs are aligned for plotting. Shapes are scaled by max absolute value; singular-value magnitudes are shown separately in the labels.",
    ]
    (args.out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
