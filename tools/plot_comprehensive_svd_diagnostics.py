#!/usr/bin/env python3
"""Comprehensive local-Jacobian SVD shape and spectrum diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FNO_RAW_ROOT = PROJECT_ROOT / "forensics" / "fno_solver_jacobian_similarity_20260514_raw_recomputed"
DEFAULT_DEEPONET_ROOT = PROJECT_ROOT / "forensics" / "deeponet_solver_jacobian_similarity_20260515"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "forensics" / "comprehensive_svd_diagnostics_20260515"
EPS = 1e-12


@dataclass(frozen=True)
class OperatorSpec:
    key: str
    family: str
    role: str
    label: str
    short_label: str
    color: str
    root: Path
    subdir: str
    filename_prefix: str


def operator_specs(args: argparse.Namespace) -> list[OperatorSpec]:
    return [
        OperatorSpec("fno_model", "FNO nu=0.001", "model", "FNO model nu=0.001", "FNO model", "#1f77b4", args.fno_raw_root, "fno", "fno"),
        OperatorSpec("fno_solver", "FNO nu=0.001", "solver", "solver nu=0.001", "solver 0.001", "#2ca02c", args.fno_raw_root, "solver", "solver"),
        OperatorSpec("fno_error", "FNO nu=0.001", "error", "FNO error J_m-J_s", "FNO error", "#ff7f0e", args.fno_raw_root, "error", "error"),
        OperatorSpec("deeponet_model", "DeepONet nu=0.01", "model", "DeepONet model nu=0.01", "DeepONet model", "#d62728", args.deeponet_root, "deeponet", "deeponet"),
        OperatorSpec("deeponet_solver", "DeepONet nu=0.01", "solver", "solver nu=0.01", "solver 0.01", "#9467bd", args.deeponet_root, "solver", "solver"),
        OperatorSpec("deeponet_error", "DeepONet nu=0.01", "error", "DeepONet error J_m-J_s", "DeepONet error", "#8c564b", args.deeponet_root, "error", "error"),
    ]


def npz_path(spec: OperatorSpec, index: int) -> Path:
    return spec.root / f"index_{index:03d}" / spec.subdir / f"{spec.filename_prefix}_index{index}_jacobian_svd.npz"


def load_svd(spec: OperatorSpec, index: int) -> dict[str, Any]:
    path = npz_path(spec, index)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {
            "spec": spec,
            "path": path,
            "s": np.asarray(data["singular_values"], dtype=np.float64),
            "Vh": np.asarray(data["right_singular_vectors"], dtype=np.float64),
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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
    return np.arange(len(energy)), energy


def align_vector(v: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    out = np.asarray(v, dtype=np.float64).copy()
    if reference is not None:
        dot = float(np.dot(out, reference))
        if abs(dot) > 1e-12:
            if dot < 0.0:
                out *= -1.0
            return out
    idx = int(np.argmax(np.abs(out)))
    if out[idx] < 0.0:
        out *= -1.0
    return out


def shape_scale(v: np.ndarray) -> np.ndarray:
    return v / (float(np.max(np.abs(v))) + EPS)


def roughness(v: np.ndarray) -> tuple[float, float]:
    return float(np.linalg.norm(np.diff(v))), float(np.linalg.norm(np.diff(v, n=2)))


def vector_metric_rows(index: int, loaded: dict[str, dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, item in loaded.items():
        spec: OperatorSpec = item["spec"]
        s = item["s"]
        Vh = item["Vh"]
        for rank in range(top_k):
            v = Vh[rank]
            r1, r2 = roughness(v)
            rows.append(
                {
                    "sample_index": index,
                    "family": spec.family,
                    "operator": spec.key,
                    "role": spec.role,
                    "rank": rank + 1,
                    "singular_value": float(s[rank]),
                    "right_hi16": high_freq_energy(v, 16),
                    "right_hi32": high_freq_energy(v, 32),
                    "right_hi64": high_freq_energy(v, 64),
                    "right_hi128": high_freq_energy(v, 128),
                    "right_hi256": high_freq_energy(v, 256),
                    "right_zero_crossings": zero_crossings(v),
                    "right_rough1": r1,
                    "right_rough2": r2,
                }
            )
    return rows


def pair_specs() -> list[tuple[str, str, str, str]]:
    return [
        ("FNO nu=0.001", "fno_model", "fno_solver", "model_vs_solver"),
        ("FNO nu=0.001", "fno_model", "fno_error", "model_vs_error"),
        ("FNO nu=0.001", "fno_solver", "fno_error", "solver_vs_error"),
        ("DeepONet nu=0.01", "deeponet_model", "deeponet_solver", "model_vs_solver"),
        ("DeepONet nu=0.01", "deeponet_model", "deeponet_error", "model_vs_error"),
        ("DeepONet nu=0.01", "deeponet_solver", "deeponet_error", "solver_vs_error"),
    ]


def overlap_matrix(a: np.ndarray, b: np.ndarray, top_k: int) -> np.ndarray:
    return np.abs(a[:top_k] @ b[:top_k].T)


def principal_angle_stats(a: np.ndarray, b: np.ndarray, k: int) -> dict[str, float]:
    _, s, _ = np.linalg.svd(a[:k] @ b[:k].T, full_matrices=False)
    s = np.clip(s, 0.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return {
        "min_cos": float(np.min(s)),
        "mean_cos": float(np.mean(s)),
        "max_angle_deg": float(np.max(angles)),
        "mean_angle_deg": float(np.mean(angles)),
    }


def angle_rows(index: int, loaded: dict[str, dict[str, Any]], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairwise: list[dict[str, Any]] = []
    principal: list[dict[str, Any]] = []
    for family, a_key, b_key, pair in pair_specs():
        A = loaded[a_key]["Vh"]
        B = loaded[b_key]["Vh"]
        M = overlap_matrix(A, B, top_k)
        for i in range(top_k):
            for j in range(top_k):
                val = float(M[i, j])
                pairwise.append(
                    {
                        "sample_index": index,
                        "family": family,
                        "pair": pair,
                        "rank_a": i + 1,
                        "rank_b": j + 1,
                        "abs_cos": val,
                        "angle_deg": float(np.degrees(np.arccos(np.clip(val, 0.0, 1.0)))),
                    }
                )
        for k in range(1, top_k + 1):
            stats = principal_angle_stats(A, B, k)
            principal.append({"sample_index": index, "family": family, "pair": pair, "k": k, **stats})
    return pairwise, principal


def orthogonality_rows(index: int, loaded: dict[str, dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    I = np.eye(top_k)
    for key, item in loaded.items():
        spec: OperatorSpec = item["spec"]
        V = item["Vh"][:top_k]
        G = V @ V.T
        E = G - I
        off = E.copy()
        np.fill_diagonal(off, 0.0)
        rows.append(
            {
                "sample_index": index,
                "family": spec.family,
                "operator": spec.key,
                "role": spec.role,
                "top_k": top_k,
                "max_offdiag_abs": float(np.max(np.abs(off))),
                "fro_offdiag": float(np.linalg.norm(off)),
                "max_diag_error": float(np.max(np.abs(np.diag(G) - 1.0))),
            }
        )
    return rows


def load_index(args: argparse.Namespace, index: int) -> dict[str, dict[str, Any]]:
    return {spec.key: load_svd(spec, index) for spec in operator_specs(args)}


def plot_sixrow_vectors(index: int, loaded: dict[str, dict[str, Any]], out_dir: Path, top_k: int, domain: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [spec.key for spec in operator_specs(argparse.Namespace(fno_raw_root=DEFAULT_FNO_RAW_ROOT, deeponet_root=DEFAULT_DEEPONET_ROOT))]
    n = loaded[keys[0]]["Vh"].shape[1]
    x = np.linspace(0.0, domain, n, endpoint=False)
    fig, axes = plt.subplots(len(keys), top_k, figsize=(3.0 * top_k, 1.75 * len(keys)), sharex=True, sharey=True)
    for row, key in enumerate(keys):
        item = loaded[key]
        spec: OperatorSpec = item["spec"]
        for col in range(top_k):
            ax = axes[row, col]
            v = align_vector(item["Vh"][col])
            y = shape_scale(v)
            ax.plot(x, y, lw=0.95, color=spec.color)
            ax.axhline(0.0, lw=0.45, color="0.78")
            if row == 0:
                ax.set_title(f"rank {col + 1}", fontsize=9)
            if col == 0:
                ax.set_ylabel(spec.short_label, fontsize=8)
            if row == len(keys) - 1:
                ax.set_xlabel("x", fontsize=8)
            hi128 = high_freq_energy(v, 128)
            zc = zero_crossings(v)
            ax.text(
                0.02,
                0.94,
                f"sigma={item['s'][col]:.3g}\nhi128={hi128:.2g}\nzc={zc}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=6.2,
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "0.85", "alpha": 0.86},
            )
            ax.tick_params(labelsize=7, length=2)
            ax.set_ylim(-1.08, 1.08)
    fig.suptitle(f"Top-{top_k} right singular vector shapes, sample {index}", y=0.997, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_dir / f"index_{index:03d}_sixrow_top{top_k}_right_vectors.png", dpi=190)
    plt.close(fig)


def plot_sixrow_fft(index: int, loaded: dict[str, dict[str, Any]], out_dir: Path, top_k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [spec.key for spec in operator_specs(argparse.Namespace(fno_raw_root=DEFAULT_FNO_RAW_ROOT, deeponet_root=DEFAULT_DEEPONET_ROOT))]
    fig, axes = plt.subplots(len(keys), top_k, figsize=(3.0 * top_k, 1.75 * len(keys)), sharex=True, sharey=True)
    for row, key in enumerate(keys):
        item = loaded[key]
        spec: OperatorSpec = item["spec"]
        for col in range(top_k):
            ax = axes[row, col]
            k, e = spectrum(item["Vh"][col])
            ax.semilogy(k[1:], e[1:] + EPS, lw=0.9, color=spec.color)
            ax.axvline(128, lw=0.6, color="0.35", ls="--")
            ax.grid(alpha=0.2)
            ax.set_xlim(1, len(e) - 1)
            ax.set_ylim(1e-16, 1.0)
            if row == 0:
                ax.set_title(f"rank {col + 1}", fontsize=9)
            if col == 0:
                ax.set_ylabel(spec.short_label, fontsize=8)
            if row == len(keys) - 1:
                ax.set_xlabel("Fourier k", fontsize=8)
            ax.tick_params(labelsize=7, length=2)
    fig.suptitle(f"Fourier energy spectra of top-{top_k} right singular vectors, sample {index}", y=0.997, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_dir / f"index_{index:03d}_sixrow_top{top_k}_fft_energy.png", dpi=190)
    plt.close(fig)


def plot_overlap_heatmaps(index: int, loaded: dict[str, dict[str, Any]], out_dir: Path, top_k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), constrained_layout=True)
    pair_data = [p for p in pair_specs()]
    for idx, (family, a_key, b_key, pair) in enumerate(pair_data):
        ax = axes[idx // 3, idx % 3]
        M = overlap_matrix(loaded[a_key]["Vh"], loaded[b_key]["Vh"], top_k)
        im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
        diag_mean = float(np.mean(np.diag(M)))
        ax.set_title(f"{family}\n{pair}, diag mean={diag_mean:.3f}", fontsize=10)
        ax.set_xlabel("rank b")
        ax.set_ylabel("rank a")
        ax.set_xticks(np.arange(top_k))
        ax.set_yticks(np.arange(top_k))
        ax.set_xticklabels(np.arange(1, top_k + 1), fontsize=7)
        ax.set_yticklabels(np.arange(1, top_k + 1), fontsize=7)
        for i in range(top_k):
            for j in range(top_k):
                if top_k <= 8:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.7, color="white" if M[i, j] < 0.45 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Right singular vector absolute cosine heatmaps, sample {index}", fontsize=13)
    fig.savefig(out_dir / f"index_{index:03d}_pairwise_right_vector_cosine_heatmaps.png", dpi=190)
    plt.close(fig)


def plot_principal_angles(index: int, principal_rows: list[dict[str, Any]], out_dir: Path, top_k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair_styles = {
        "model_vs_solver": ("#1f77b4", "model vs solver"),
        "model_vs_error": ("#d62728", "model vs error"),
        "solver_vs_error": ("#2ca02c", "solver vs error"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, family in zip(axes, ["FNO nu=0.001", "DeepONet nu=0.01"]):
        for pair, (color, label) in pair_styles.items():
            rows = [r for r in principal_rows if r["family"] == family and r["pair"] == pair]
            rows.sort(key=lambda r: r["k"])
            ax.plot([r["k"] for r in rows], [r["mean_angle_deg"] for r in rows], marker="o", lw=1.4, color=color, label=label)
        ax.set_title(family)
        ax.set_xlabel("top-k subspace")
        ax.set_xticks(range(1, top_k + 1))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("mean principal angle (deg); lower means more similar")
    fig.suptitle(f"Top-k right-subspace principal angles, sample {index}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"index_{index:03d}_principal_angle_curves.png", dpi=190)
    plt.close(fig)


def plot_singular_spectra(index: int, loaded: dict[str, dict[str, Any]], out_dir: Path, max_rank: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    groups = [
        ("FNO nu=0.001", ["fno_model", "fno_solver", "fno_error"]),
        ("DeepONet nu=0.01", ["deeponet_model", "deeponet_solver", "deeponet_error"]),
    ]
    for ax, (title, keys) in zip(axes, groups):
        for key in keys:
            item = loaded[key]
            spec: OperatorSpec = item["spec"]
            s = item["s"][:max_rank]
            ax.semilogy(np.arange(1, len(s) + 1), s + EPS, lw=1.3, color=spec.color, label=f"{spec.role}, sigma1={s[0]:.3g}")
        ax.set_title(title)
        ax.set_xlabel("rank")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("singular value")
    fig.suptitle(f"Singular value spectra, sample {index}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"index_{index:03d}_singular_value_spectra.png", dpi=190)
    plt.close(fig)


def plot_orthogonality(index: int, loaded: dict[str, dict[str, Any]], out_dir: Path, top_k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [spec.key for spec in operator_specs(argparse.Namespace(fno_raw_root=DEFAULT_FNO_RAW_ROOT, deeponet_root=DEFAULT_DEEPONET_ROOT))]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), constrained_layout=True)
    for idx, key in enumerate(keys):
        ax = axes[idx // 3, idx % 3]
        item = loaded[key]
        spec: OperatorSpec = item["spec"]
        V = item["Vh"][:top_k]
        E = np.abs(V @ V.T - np.eye(top_k))
        vmax = max(1e-14, float(np.max(E)))
        im = ax.imshow(E, vmin=0.0, vmax=vmax, cmap="magma", origin="lower")
        off = E.copy()
        np.fill_diagonal(off, 0.0)
        ax.set_title(f"{spec.short_label}\nmax offdiag={np.max(off):.1e}", fontsize=10)
        ax.set_xlabel("rank")
        ax.set_ylabel("rank")
        ax.set_xticks(np.arange(top_k))
        ax.set_yticks(np.arange(top_k))
        ax.set_xticklabels(np.arange(1, top_k + 1), fontsize=7)
        ax.set_yticklabels(np.arange(1, top_k + 1), fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Within-operator right singular vector orthogonality error |VV^T-I|, sample {index}", fontsize=13)
    fig.savefig(out_dir / f"index_{index:03d}_orthogonality_error_heatmaps.png", dpi=190)
    plt.close(fig)


def per_index_plots(args: argparse.Namespace, index: int, loaded: dict[str, dict[str, Any]], rows: dict[str, list[dict[str, Any]]]) -> None:
    out_dir = args.out_root / f"index_{index:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_sixrow_vectors(index, loaded, out_dir, args.top_k, args.domain)
    plot_sixrow_fft(index, loaded, out_dir, args.top_k)
    plot_overlap_heatmaps(index, loaded, out_dir, args.top_k)
    plot_principal_angles(index, rows["principal"], out_dir, args.top_k)
    plot_singular_spectra(index, loaded, out_dir, args.spectrum_max_rank)
    plot_orthogonality(index, loaded, out_dir, args.top_k)


def aggregate_mean(rows: list[dict[str, Any]], keys: list[str], value_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        group_key = tuple(row[k] for k in keys)
        groups.setdefault(group_key, []).append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items(), key=lambda kv: kv[0]):
        row = {k: v for k, v in zip(keys, group_key)}
        for field in value_fields:
            vals = np.asarray([float(r[field]) for r in group_rows], dtype=np.float64)
            row[f"{field}_mean"] = float(np.mean(vals))
            row[f"{field}_std"] = float(np.std(vals, ddof=0))
            row[f"{field}_min"] = float(np.min(vals))
            row[f"{field}_max"] = float(np.max(vals))
        out.append(row)
    return out


def plot_aggregate_fft(args: argparse.Namespace, all_loaded: dict[int, dict[str, dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = operator_specs(args)
    fig, axes = plt.subplots(len(specs), args.top_k, figsize=(3.0 * args.top_k, 1.75 * len(specs)), sharex=True, sharey=True)
    for row, spec in enumerate(specs):
        for rank in range(args.top_k):
            spectra = []
            sigmas = []
            hi128s = []
            zcs = []
            for loaded in all_loaded.values():
                item = loaded[spec.key]
                k, e = spectrum(item["Vh"][rank])
                spectra.append(e)
                sigmas.append(float(item["s"][rank]))
                hi128s.append(high_freq_energy(item["Vh"][rank], 128))
                zcs.append(zero_crossings(item["Vh"][rank]))
            E = np.stack(spectra, axis=0)
            mean = E.mean(axis=0)
            std = E.std(axis=0)
            ax = axes[row, rank]
            ax.semilogy(k[1:], mean[1:] + EPS, lw=1.0, color=spec.color)
            lower = np.maximum(mean - std, EPS)
            upper = np.minimum(mean + std, 1.0)
            ax.fill_between(k[1:], lower[1:], upper[1:], color=spec.color, alpha=0.18, lw=0)
            ax.axvline(128, lw=0.6, color="0.35", ls="--")
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(f"rank {rank + 1}", fontsize=9)
            if rank == 0:
                ax.set_ylabel(spec.short_label, fontsize=8)
            if row == len(specs) - 1:
                ax.set_xlabel("Fourier k", fontsize=8)
            ax.text(
                0.03,
                0.94,
                f"mean sigma={np.mean(sigmas):.3g}\nmean hi128={np.mean(hi128s):.2g}\nmean zc={np.mean(zcs):.1f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=5.9,
                bbox={"boxstyle": "round,pad=0.13", "fc": "white", "ec": "0.86", "alpha": 0.86},
            )
            ax.tick_params(labelsize=7, length=2)
    axes[0, 0].set_ylim(1e-16, 1.0)
    axes[0, 0].set_xlim(1, len(k) - 1)
    fig.suptitle(f"Mean Fourier energy spectra across {len(all_loaded)} samples, top-{args.top_k} right singular vectors", y=0.997, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.out_root / f"aggregate_mean_top{args.top_k}_fft_energy_grid.png", dpi=190)
    plt.close(fig)


def plot_aggregate_shapes(args: argparse.Namespace, all_loaded: dict[int, dict[str, dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = operator_specs(args)
    n = next(iter(all_loaded.values()))[specs[0].key]["Vh"].shape[1]
    x = np.linspace(0.0, args.domain, n, endpoint=False)
    fig, axes = plt.subplots(len(specs), args.top_k, figsize=(3.0 * args.top_k, 1.75 * len(specs)), sharex=True, sharey=True)
    for row, spec in enumerate(specs):
        for rank in range(args.top_k):
            ref = None
            shapes = []
            for index in sorted(all_loaded):
                v = all_loaded[index][spec.key]["Vh"][rank]
                aligned = align_vector(v, ref)
                if ref is None:
                    ref = aligned
                shapes.append(shape_scale(aligned))
            S = np.stack(shapes, axis=0)
            mean = S.mean(axis=0)
            std = S.std(axis=0)
            ax = axes[row, rank]
            ax.plot(x, mean, lw=1.0, color=spec.color)
            ax.fill_between(x, np.maximum(mean - std, -1.05), np.minimum(mean + std, 1.05), color=spec.color, alpha=0.18, lw=0)
            ax.axhline(0.0, lw=0.45, color="0.78")
            ax.set_ylim(-1.08, 1.08)
            if row == 0:
                ax.set_title(f"rank {rank + 1}", fontsize=9)
            if rank == 0:
                ax.set_ylabel(spec.short_label, fontsize=8)
            if row == len(specs) - 1:
                ax.set_xlabel("x", fontsize=8)
            ax.tick_params(labelsize=7, length=2)
    fig.suptitle("Mean display-aligned singular-vector shapes across samples (diagnostic; phase/sign sensitive)", y=0.997, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.out_root / f"aggregate_mean_top{args.top_k}_shape_grid.png", dpi=190)
    plt.close(fig)


def plot_aggregate_overlap(args: argparse.Namespace, all_loaded: dict[int, dict[str, dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), constrained_layout=True)
    for idx, (family, a_key, b_key, pair) in enumerate(pair_specs()):
        mats = [overlap_matrix(loaded[a_key]["Vh"], loaded[b_key]["Vh"], args.top_k) for loaded in all_loaded.values()]
        M = np.mean(np.stack(mats, axis=0), axis=0)
        ax = axes[idx // 3, idx % 3]
        im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
        diag_mean = float(np.mean(np.diag(M)))
        ax.set_title(f"{family}\n{pair}, mean diag={diag_mean:.3f}", fontsize=10)
        ax.set_xlabel("rank b")
        ax.set_ylabel("rank a")
        ax.set_xticks(np.arange(args.top_k))
        ax.set_yticks(np.arange(args.top_k))
        ax.set_xticklabels(np.arange(1, args.top_k + 1), fontsize=7)
        ax.set_yticklabels(np.arange(1, args.top_k + 1), fontsize=7)
        for i in range(args.top_k):
            for j in range(args.top_k):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.7, color="white" if M[i, j] < 0.45 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Mean right singular vector absolute cosine heatmaps across {len(all_loaded)} samples", fontsize=13)
    fig.savefig(args.out_root / f"aggregate_mean_pairwise_right_vector_cosine_heatmaps_top{args.top_k}.png", dpi=190)
    plt.close(fig)


def plot_aggregate_principal(args: argparse.Namespace, principal_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair_styles = {
        "model_vs_solver": ("#1f77b4", "model vs solver"),
        "model_vs_error": ("#d62728", "model vs error"),
        "solver_vs_error": ("#2ca02c", "solver vs error"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, family in zip(axes, ["FNO nu=0.001", "DeepONet nu=0.01"]):
        for pair, (color, label) in pair_styles.items():
            means = []
            stds = []
            ks = list(range(1, args.top_k + 1))
            for k in ks:
                vals = [float(r["mean_angle_deg"]) for r in principal_rows if r["family"] == family and r["pair"] == pair and r["k"] == k]
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=0)))
            means_arr = np.asarray(means)
            stds_arr = np.asarray(stds)
            ax.plot(ks, means_arr, marker="o", lw=1.4, color=color, label=label)
            ax.fill_between(ks, means_arr - stds_arr, means_arr + stds_arr, color=color, alpha=0.15, lw=0)
        ax.set_title(family)
        ax.set_xlabel("top-k subspace")
        ax.set_xticks(range(1, args.top_k + 1))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("mean principal angle (deg); lower means more similar")
    fig.suptitle("Mean top-k right-subspace principal angles across samples", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out_root / f"aggregate_principal_angle_curves_top{args.top_k}.png", dpi=190)
    plt.close(fig)


def plot_aggregate_singular_spectra(args: argparse.Namespace, all_loaded: dict[int, dict[str, dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    groups = [
        ("FNO nu=0.001", ["fno_model", "fno_solver", "fno_error"]),
        ("DeepONet nu=0.01", ["deeponet_model", "deeponet_solver", "deeponet_error"]),
    ]
    for ax, (title, keys) in zip(axes, groups):
        for key in keys:
            spec: OperatorSpec = next(item["spec"] for item in all_loaded[next(iter(all_loaded))].values() if item["spec"].key == key)
            arr = np.stack([loaded[key]["s"][: args.spectrum_max_rank] for loaded in all_loaded.values()], axis=0)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            x = np.arange(1, len(mean) + 1)
            ax.semilogy(x, mean + EPS, lw=1.3, color=spec.color, label=f"{spec.role}, mean sigma1={mean[0]:.3g}")
            ax.fill_between(x, np.maximum(mean - std, EPS), mean + std, color=spec.color, alpha=0.15, lw=0)
        ax.set_title(title)
        ax.set_xlabel("rank")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("singular value")
    fig.suptitle("Mean singular value spectra across samples", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out_root / f"aggregate_singular_value_spectra_rank{args.spectrum_max_rank}.png", dpi=190)
    plt.close(fig)


def plot_aggregate_orthogonality(args: argparse.Namespace, all_loaded: dict[int, dict[str, dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = operator_specs(args)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), constrained_layout=True)
    for idx, spec in enumerate(specs):
        mats = []
        for loaded in all_loaded.values():
            V = loaded[spec.key]["Vh"][: args.top_k]
            mats.append(np.abs(V @ V.T - np.eye(args.top_k)))
        E = np.mean(np.stack(mats, axis=0), axis=0)
        ax = axes[idx // 3, idx % 3]
        vmax = max(1e-14, float(np.max(E)))
        im = ax.imshow(E, vmin=0.0, vmax=vmax, cmap="magma", origin="lower")
        off = E.copy()
        np.fill_diagonal(off, 0.0)
        ax.set_title(f"{spec.short_label}\nmean max offdiag approx {np.max(off):.1e}", fontsize=10)
        ax.set_xlabel("rank")
        ax.set_ylabel("rank")
        ax.set_xticks(np.arange(args.top_k))
        ax.set_yticks(np.arange(args.top_k))
        ax.set_xticklabels(np.arange(1, args.top_k + 1), fontsize=7)
        ax.set_yticklabels(np.arange(1, args.top_k + 1), fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Mean orthogonality error |VV^T-I| across samples, top-{args.top_k}", fontsize=13)
    fig.savefig(args.out_root / f"aggregate_orthogonality_error_heatmaps_top{args.top_k}.png", dpi=190)
    plt.close(fig)


def write_summary(args: argparse.Namespace, aggregate_metrics: list[dict[str, Any]], aggregate_principal: list[dict[str, Any]], aggregate_orth: list[dict[str, Any]]) -> None:
    def find_metric(operator: str, rank: int, field: str) -> float:
        for row in aggregate_metrics:
            if row["operator"] == operator and row["rank"] == rank:
                return float(row[f"{field}_mean"])
        return math.nan

    def find_principal(family: str, pair: str, k: int, field: str) -> float:
        for row in aggregate_principal:
            if row["family"] == family and row["pair"] == pair and row["k"] == k:
                return float(row[f"{field}_mean"])
        return math.nan

    lines = [
        "# Comprehensive SVD Diagnostics",
        "",
        "This directory contains dense line/heatmap diagnostics for the local Jacobian SVDs.",
        "",
        "Rows in the main per-index figures are:",
        "",
        "1. FNO model `nu=0.001`",
        "2. solver `nu=0.001`",
        "3. FNO error `J_m - J_s`",
        "4. DeepONet model `nu=0.01`",
        "5. solver `nu=0.01`",
        "6. DeepONet error `J_m - J_s`",
        "",
        "Columns are right singular vector ranks. Right singular vectors are input perturbation directions.",
        "",
        "## Key Aggregate Readout",
        "",
        "| quantity | FNO side | DeepONet side |",
        "|---|---:|---:|",
        f"| model top-1 sigma | {find_metric('fno_model', 1, 'singular_value'):.4g} | {find_metric('deeponet_model', 1, 'singular_value'):.4g} |",
        f"| solver top-1 sigma | {find_metric('fno_solver', 1, 'singular_value'):.4g} | {find_metric('deeponet_solver', 1, 'singular_value'):.4g} |",
        f"| error top-1 sigma | {find_metric('fno_error', 1, 'singular_value'):.4g} | {find_metric('deeponet_error', 1, 'singular_value'):.4g} |",
        f"| model top-1 hi128 | {find_metric('fno_model', 1, 'right_hi128'):.4g} | {find_metric('deeponet_model', 1, 'right_hi128'):.4g} |",
        f"| error top-1 hi128 | {find_metric('fno_error', 1, 'right_hi128'):.4g} | {find_metric('deeponet_error', 1, 'right_hi128'):.4g} |",
        f"| model top-1 zero crossings | {find_metric('fno_model', 1, 'right_zero_crossings'):.2f} | {find_metric('deeponet_model', 1, 'right_zero_crossings'):.2f} |",
        f"| error top-1 zero crossings | {find_metric('fno_error', 1, 'right_zero_crossings'):.2f} | {find_metric('deeponet_error', 1, 'right_zero_crossings'):.2f} |",
        f"| model-vs-solver k=1 mean principal angle | {find_principal('FNO nu=0.001', 'model_vs_solver', 1, 'mean_angle_deg'):.2f} deg | {find_principal('DeepONet nu=0.01', 'model_vs_solver', 1, 'mean_angle_deg'):.2f} deg |",
        f"| model-vs-solver k=8 mean principal angle | {find_principal('FNO nu=0.001', 'model_vs_solver', args.top_k, 'mean_angle_deg'):.2f} deg | {find_principal('DeepONet nu=0.01', 'model_vs_solver', args.top_k, 'mean_angle_deg'):.2f} deg |",
        f"| model-vs-error k=1 mean principal angle | {find_principal('FNO nu=0.001', 'model_vs_error', 1, 'mean_angle_deg'):.2f} deg | {find_principal('DeepONet nu=0.01', 'model_vs_error', 1, 'mean_angle_deg'):.2f} deg |",
        "",
        "## Generated Plot Families",
        "",
        "Per index under `index_*/`:",
        "",
        "- `*_sixrow_top8_right_vectors.png`: six-row line grid of top-8 right singular vectors with `sigma`, `hi128`, and zero crossings.",
        "- `*_sixrow_top8_fft_energy.png`: Fourier energy spectra for the same vectors.",
        "- `*_pairwise_right_vector_cosine_heatmaps.png`: model/solver/error pairwise absolute-cosine heatmaps.",
        "- `*_principal_angle_curves.png`: top-k subspace principal-angle curves.",
        "- `*_singular_value_spectra.png`: singular value spectra with spectral norms in labels.",
        "- `*_orthogonality_error_heatmaps.png`: within-operator orthogonality sanity-check heatmaps.",
        "",
        "Aggregate across five samples:",
        "",
        "- `aggregate_mean_top8_fft_energy_grid.png`: mean Fourier spectrum after Fourier transform, with shaded sample variability.",
        "- `aggregate_mean_top8_shape_grid.png`: display-aligned mean vector shapes; useful visually but less invariant than spectra/subspaces.",
        "- `aggregate_mean_pairwise_right_vector_cosine_heatmaps_top8.png`: mean cross-operator vector cosine heatmaps.",
        "- `aggregate_principal_angle_curves_top8.png`: mean top-k principal angles with variability.",
        "- `aggregate_singular_value_spectra_rank64.png`: mean singular spectra.",
        "- `aggregate_orthogonality_error_heatmaps_top8.png`: mean numerical orthogonality errors.",
        "",
        "## Orthogonality Interpretation",
        "",
        "Within one SVD, the right singular vectors are orthonormal by construction. Therefore, the angle between different ranks inside the same operator is expected to be 90 degrees, up to numerical error. This is mostly a sanity check, not the main scientific comparison.",
        "",
        "The meaningful comparison is cross-operator: whether the model top-k right singular subspace aligns with the solver top-k subspace, and whether the error subspace aligns with the model. The plots show FNO model/solver subspaces are close, while DeepONet model/error subspaces are close and DeepONet model/solver subspaces are far apart.",
    ]
    (args.out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 7, 40, 47, 115])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--spectrum-max-rank", type=int, default=64)
    parser.add_argument("--domain", type=float, default=2.0)
    parser.add_argument("--fno-raw-root", type=Path, default=DEFAULT_FNO_RAW_ROOT)
    parser.add_argument("--deeponet-root", type=Path, default=DEFAULT_DEEPONET_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    all_loaded: dict[int, dict[str, dict[str, Any]]] = {}
    all_metrics: list[dict[str, Any]] = []
    all_pairwise: list[dict[str, Any]] = []
    all_principal: list[dict[str, Any]] = []
    all_orth: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "sample_indices": args.sample_indices,
        "top_k": args.top_k,
        "spectrum_max_rank": args.spectrum_max_rank,
        "fno_raw_root": str(args.fno_raw_root),
        "deeponet_root": str(args.deeponet_root),
        "out_root": str(args.out_root),
        "operators": [spec.__dict__ | {"root": str(spec.root)} for spec in operator_specs(args)],
    }
    for index in args.sample_indices:
        print(f"[index] {index}", flush=True)
        loaded = load_index(args, index)
        all_loaded[index] = loaded
        metrics = vector_metric_rows(index, loaded, args.top_k)
        pairwise, principal = angle_rows(index, loaded, args.top_k)
        orth = orthogonality_rows(index, loaded, args.top_k)
        all_metrics.extend(metrics)
        all_pairwise.extend(pairwise)
        all_principal.extend(principal)
        all_orth.extend(orth)
        per_index_plots(args, index, loaded, {"principal": principal}, )

    aggregate_metrics = aggregate_mean(
        all_metrics,
        ["family", "operator", "role", "rank"],
        ["singular_value", "right_hi16", "right_hi32", "right_hi64", "right_hi128", "right_hi256", "right_zero_crossings", "right_rough1", "right_rough2"],
    )
    aggregate_principal = aggregate_mean(
        all_principal,
        ["family", "pair", "k"],
        ["min_cos", "mean_cos", "max_angle_deg", "mean_angle_deg"],
    )
    aggregate_orth = aggregate_mean(
        all_orth,
        ["family", "operator", "role", "top_k"],
        ["max_offdiag_abs", "fro_offdiag", "max_diag_error"],
    )

    write_csv(args.out_root / "singular_vector_metrics.csv", all_metrics)
    write_csv(args.out_root / "pairwise_right_vector_cosines.csv", all_pairwise)
    write_csv(args.out_root / "principal_angles.csv", all_principal)
    write_csv(args.out_root / "orthogonality_summary.csv", all_orth)
    write_csv(args.out_root / "aggregate_singular_vector_metrics.csv", aggregate_metrics)
    write_csv(args.out_root / "aggregate_principal_angles.csv", aggregate_principal)
    write_csv(args.out_root / "aggregate_orthogonality_summary.csv", aggregate_orth)

    print("[aggregate-plots]", flush=True)
    plot_aggregate_fft(args, all_loaded)
    plot_aggregate_shapes(args, all_loaded)
    plot_aggregate_overlap(args, all_loaded)
    plot_aggregate_principal(args, all_principal)
    plot_aggregate_singular_spectra(args, all_loaded)
    plot_aggregate_orthogonality(args, all_loaded)

    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_summary(args, aggregate_metrics, aggregate_principal, aggregate_orth)
    print(f"[done] wrote {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
