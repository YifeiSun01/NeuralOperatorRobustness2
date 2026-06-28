#!/usr/bin/env python3
"""Plot NS2D JVP/VJP mechanism outputs for the final Loss3 pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def plot_alignment(run_frames: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    out: list[Path] = []
    frames = []
    for run, df in run_frames.items():
        if df.empty:
            continue
        temp = df.copy()
        temp["run"] = run
        frames.append(temp)
    if not frames:
        return out
    df = pd.concat(frames, ignore_index=True)
    for col in ("k", "cos_top_with_saved_direction", "cos_top_with_grad", "cos_top_with_final_delta", "top_sigma"):
        if col in df:
            df[col] = to_num(df[col])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=False)
    metrics = [
        ("cos_top_with_saved_direction", "cos(top singular, saved direction)"),
        ("cos_top_with_grad", "cos(top singular, gradient)"),
        ("cos_top_with_final_delta", "cos(top singular, final delta)"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        if metric not in df:
            ax.set_axis_off()
            continue
        for (run, method), group in df.groupby(["run", "method"], dropna=False):
            mean = group.groupby("k", dropna=False)[metric].mean().reset_index().sort_values("k")
            ax.plot(mean["k"], mean[metric], marker="o", linewidth=1.8, label=f"{run}:{method}")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_title(title)
        ax.set_xlabel("attack step k")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("mean cosine")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("NS2D JVP/VJP: Top Singular Direction Alignment", y=0.98)
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    path = out_dir / "ns2d_jvp_vjp_top_direction_alignment.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    out.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for (run, method), group in df.groupby(["run", "method"], dropna=False):
        mean = group.groupby("k", dropna=False)["top_sigma"].mean().reset_index().sort_values("k")
        ax.plot(mean["k"], mean["top_sigma"], marker="o", linewidth=1.8, label=f"{run}:{method}")
    ax.set_title("NS2D JVP/VJP: Estimated Top Singular Value Along Path")
    ax.set_xlabel("attack step k")
    ax.set_ylabel("mean top singular value estimate")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out_dir / "ns2d_jvp_vjp_top_sigma_path.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    out.append(path)
    return out


def plot_surrogate(run_frames: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    out: list[Path] = []
    frames = []
    for run, df in run_frames.items():
        if df.empty:
            continue
        temp = df.copy()
        temp["run"] = run
        frames.append(temp)
    if not frames:
        return out
    df = pd.concat(frames, ignore_index=True)
    for col in ("radius_fraction", "relative_abs_error", "residual_relative_error", "true_gain", "predicted_gain"):
        if col in df:
            df[col] = to_num(df[col])
    focus = df[df["direction"].isin(["top_singular", "saved_direction", "grad"])].copy()
    if focus.empty:
        return out

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=False)
    metrics = [
        ("relative_abs_error", "Loss gain relative absolute error"),
        ("residual_relative_error", "Residual prediction relative error"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for (run, method, direction), group in focus.groupby(["run", "method", "direction"], dropna=False):
            mean = group.groupby("radius_fraction", dropna=False)[metric].mean().reset_index().sort_values("radius_fraction")
            label = f"{run}:{method}:{direction}"
            ax.plot(mean["radius_fraction"], mean[metric], marker="o", linewidth=1.5, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("radius fraction of epsilon")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("mean error")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.suptitle("NS2D JVP/VJP: Local Residual-Linear Surrogate Accuracy", y=0.98)
    fig.tight_layout(rect=(0, 0.18, 1, 0.92))
    path = out_dir / "ns2d_jvp_vjp_surrogate_error_by_radius.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    out.append(path)
    return out


def plot_candidates(run_frames: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    out: list[Path] = []
    frames = []
    for run, df in run_frames.items():
        if df.empty:
            continue
        temp = df.copy()
        temp["run"] = run
        frames.append(temp)
    if not frames:
        return out
    df = pd.concat(frames, ignore_index=True)
    if "true_gain_mean" in df:
        df["true_gain_mean"] = to_num(df["true_gain_mean"])
    focus_candidates = [
        "add_top_singular",
        "replace_top_singular",
        "add_saved_direction",
        "replace_saved_direction",
        "add_grad",
        "replace_grad",
    ]
    df = df[df["candidate"].isin(focus_candidates)].copy()
    if df.empty:
        return out
    df["label"] = df["run"].astype(str) + ":" + df["method"].astype(str) + ":" + df["candidate"].astype(str)
    df = df.sort_values(["run", "method", "candidate"])

    fig, ax = plt.subplots(figsize=(max(12, 0.45 * len(df)), 5.2))
    colors = ["#54a24b" if str(c).startswith("add_") else "#b279a2" for c in df["candidate"]]
    ax.bar(range(len(df)), df["true_gain_mean"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("mean exact true gain")
    ax.set_title("NS2D JVP/VJP: Add vs Replace Candidate True Gain")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "ns2d_jvp_vjp_candidate_true_gain.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    out.append(path)
    return out


def write_report(out_dir: Path, figures: list[Path], run_dirs: dict[str, Path]) -> None:
    lines = [
        "# NS2D JVP/VJP Mechanism Figures - 2026-06-23",
        "",
        "These figures visualize the matrix-free NS2D Jacobian probes used by the final Loss3 mechanism pipeline.",
        "",
        "## Source Runs",
        "",
    ]
    for label, directory in run_dirs.items():
        lines.append(f"- {label}: `{rel(directory)}`")
    lines.extend(["", "## Figures", ""])
    for fig in figures:
        lines.append(f"- `{rel(fig)}`")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation/figures/ns2d_jvp_vjp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_root = args.full_root if args.full_root.is_absolute() else PROJECT_ROOT / args.full_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {
        "N3": full_root / "ns2d_jvp_vjp_spectrum_N3",
        "N2_topup": full_root / "ns2d_jvp_vjp_spectrum_N2_topup",
    }
    spectrum = {label: read_csv(directory / "spectrum_path.csv") for label, directory in run_dirs.items()}
    surrogate = {label: read_csv(directory / "quadratic_surrogate_rows.csv") for label, directory in run_dirs.items()}
    candidates = {label: read_csv(directory / "candidate_comparison_aggregate.csv") for label, directory in run_dirs.items()}
    figures: list[Path] = []
    figures.extend(plot_alignment(spectrum, out_dir))
    figures.extend(plot_surrogate(surrogate, out_dir))
    figures.extend(plot_candidates(candidates, out_dir))
    write_report(out_dir, figures, run_dirs)
    manifest = {
        "status": "completed",
        "out_dir": str(out_dir),
        "figures": [str(p) for p in figures],
        "num_figures": len(figures),
        "run_dirs": {label: str(path) for label, path in run_dirs.items()},
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
