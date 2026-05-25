#!/usr/bin/env python3
"""Clean-style NS2D comparison for baseline loss3 and alternative loss3 metrics.

Reads saved CSV/NPZ artifacts only. It does not import torch or jax and does
not rerun the model, solver, or attack.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DIR = (
    PROJECT_ROOT
    / "2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack"
    / "full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10"
    / "mode_wwwwwwwwww_p2_q2_20260522_074135_UTC"
    / "batch_0000_0009/loss3/steepest_add"
)
DEFAULT_ALT_ROOT = (
    PROJECT_ROOT
    / "2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack"
    / "eps32_alpha10_steepest_add_loss3_allw_alt5_b10_20260525_035657_UTC/eps32_alpha10"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs/ns2d_eps32_alpha10_altloss_cleanstyle_comparison_20260525"

ALT_METRICS = ("dists", "ms_ssim", "scattering2d", "affine_dists", "local_warp_dists")
METRIC_LABELS = {
    "baseline_loss3": "Loss 3 baseline",
    "dists": "DISTS",
    "ms_ssim": "MS-SSIM",
    "scattering2d": "Scattering2D",
    "affine_dists": "Affine + DISTS",
    "local_warp_dists": "Local warp + DISTS",
}
COLORS = {
    "baseline_loss3": "#2ca02c",
    "dists": "#1f77b4",
    "ms_ssim": "#ff7f0e",
    "scattering2d": "#9467bd",
    "affine_dists": "#8c564b",
    "local_warp_dists": "#d62728",
}
NBINS = 96
RADIAL_AXIS_CUTOFF = np.sqrt(2.0) / 3.0
RADIAL_CORNER_CUTOFF = 2.0 / 3.0


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    method_dir: Path
    color: str


@dataclass
class RunData:
    spec: RunSpec
    rows: list[dict[str, str]]
    sample_rows: list[dict[str, str]]
    final_npz: dict[str, np.ndarray]
    final_metrics: list[dict[str, str]]


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def discover_alt_method_dir(alt_root: Path, metric: str) -> Path:
    matches = sorted((alt_root / metric).glob("mode_*/batch_0000_0009/loss3/steepest_add"))
    if len(matches) != 1:
        raise SystemExit(f"Expected one method dir for {metric} under {alt_root}, found {len(matches)}")
    return matches[0]


def build_specs(baseline_dir: Path, alt_root: Path) -> list[RunSpec]:
    specs = [
        RunSpec(
            key="baseline_loss3",
            label=METRIC_LABELS["baseline_loss3"],
            method_dir=baseline_dir,
            color=COLORS["baseline_loss3"],
        )
    ]
    for metric in ALT_METRICS:
        specs.append(
            RunSpec(
                key=metric,
                label=METRIC_LABELS[metric],
                method_dir=discover_alt_method_dir(alt_root, metric),
                color=COLORS[metric],
            )
        )
    return specs


def load_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {key: z[key] for key in z.files}


def load_run(spec: RunSpec) -> RunData:
    per_step = spec.method_dir / "per_step_metrics.csv"
    per_sample = spec.method_dir / "per_sample_step_metrics.csv"
    final_npz = spec.method_dir / "final_state_outputs.npz"
    final_metrics = spec.method_dir / "final_state_metrics.csv"
    missing = [p for p in (per_step, per_sample, final_npz, final_metrics) if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required files for {spec.key}: {missing}")
    return RunData(
        spec=spec,
        rows=read_rows(per_step),
        sample_rows=read_rows(per_sample),
        final_npz=load_npz(final_npz),
        final_metrics=read_rows(final_metrics),
    )


def finite_concat(arrays: list[np.ndarray]) -> np.ndarray:
    vals = []
    for arr in arrays:
        x = np.asarray(arr, dtype=np.float64).ravel()
        vals.append(x[np.isfinite(x)])
    if not any(v.size for v in vals):
        return np.asarray([0.0, 1.0])
    return np.concatenate([v for v in vals if v.size])


def sym_limits(arrays: list[np.ndarray], percentile: float = 99.0) -> tuple[float, float]:
    vals = finite_concat(arrays)
    vmax = float(np.percentile(np.abs(vals), percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(np.abs(vals))) if vals.size else 1.0
    vmax = max(vmax, 1e-12)
    return -vmax, vmax


def value_limits(arrays: list[np.ndarray], percentile: float = 99.0) -> tuple[float, float]:
    vals = finite_concat(arrays)
    lo = float(np.percentile(vals, 100.0 - percentile))
    hi = float(np.percentile(vals, percentile))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def frequency_grid(n: int) -> np.ndarray:
    freq = np.fft.fftshift(np.fft.fftfreq(n))
    fx, fy = np.meshgrid(freq, freq, indexing="xy")
    return np.sqrt(fx * fx + fy * fy) / np.sqrt(0.5 * 0.5 + 0.5 * 0.5)


def radial_profile(field: np.ndarray, nbins: int = NBINS) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(field, dtype=np.float64)
    power = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
    rho = frequency_grid(int(arr.shape[-1]))
    bins = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    prof = np.full(nbins, np.nan, dtype=np.float64)
    flat_rho = rho.ravel()
    flat_power = power.ravel()
    for i in range(nbins):
        mask = (flat_rho >= bins[i]) & (flat_rho < bins[i + 1])
        if np.any(mask):
            prof[i] = float(np.mean(flat_power[mask]))
    total = float(np.nansum(prof))
    if total > 0:
        prof = prof / total
    return centers, prof


def sample_rows_for_position(rows: list[dict[str, str]], sample_position: int) -> list[dict[str, str]]:
    out = [row for row in rows if int(float(row.get("sample_position", -1))) == sample_position]
    return sorted(out, key=lambda row: f(row, "k"))


def final_metric_mean(run: RunData, key: str) -> float:
    vals = [f(row, key) for row in run.final_metrics]
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(arr))


def final_metric_sample(run: RunData, key: str, sample_position: int) -> float:
    for row in run.final_metrics:
        if int(float(row.get("sample_position", -1))) == sample_position:
            return f(row, key)
    return float("nan")


def metric_summary_rows(runs: list[RunData]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        active = series(run.rows, "loss3_mean")
        boundary = series(run.rows, "boundary_ratio_mean")
        delta_l2 = series(run.rows, "delta_l2_mean")
        time_min = series(run.rows, "seconds_since_method_start") / 60.0
        rows.append(
            {
                "key": run.spec.key,
                "label": run.spec.label,
                "dataset_indices": " ".join(str(int(x)) for x in run.final_npz["dataset_indices"]),
                "active_metric_initial_mean": float(active[0]),
                "active_metric_final_mean": float(active[-1]),
                "active_metric_ratio": float(active[-1] / active[0]) if active[0] != 0 else float("nan"),
                "original_qnorm_clean_mean": final_metric_mean(run, "clean_true_loss"),
                "original_qnorm_adv_mean": final_metric_mean(run, "adv_true_loss"),
                "original_qnorm_increase_mean": final_metric_mean(run, "true_loss_increase"),
                "boundary_ratio_final_mean": float(boundary[-1]),
                "delta_l2_final_mean": float(delta_l2[-1]),
                "runtime_min": float(time_min[-1]),
                "source_dir": str(run.spec.method_dir),
            }
        )
    return rows


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 240,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.8,
        }
    )


def shade(ax, x: np.ndarray, y: np.ndarray, std: np.ndarray, color: str, alpha: float = 0.10) -> None:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(std)
    if np.any(mask):
        ax.fill_between(x[mask], y[mask] - std[mask], y[mask] + std[mask], color=color, alpha=alpha, linewidth=0)


def save_mean_curve_summary(runs: list[RunData], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)
    fig.suptitle("NS2D eps32 alpha10, steepest_add, loss3/all_w: baseline vs alternative metrics", fontsize=13)

    for run in runs:
        key = run.spec.key
        color = run.spec.color
        k = series(run.rows, "k")
        active = series(run.rows, "loss3_mean")
        active_std = series(run.rows, "loss3_std")
        denom = active[0] if np.isfinite(active[0]) and abs(active[0]) > 1e-12 else 1.0
        axes[0, 0].plot(k, active / denom, color=color, label=run.spec.label)
        shade(axes[0, 0], k, active / denom, active_std / abs(denom), color, alpha=0.08)
        axes[1, 0].plot(k, series(run.rows, "delta_l2_mean"), color=color, label=run.spec.label)
        shade(axes[1, 0], k, series(run.rows, "delta_l2_mean"), series(run.rows, "delta_l2_std"), color)
        axes[1, 1].plot(k, series(run.rows, "boundary_ratio_mean"), color=color, label=run.spec.label)

    labels = [run.spec.label for run in runs]
    x = np.arange(len(runs))
    adv = [final_metric_mean(run, "adv_true_loss") for run in runs]
    clean = [final_metric_mean(run, "clean_true_loss") for run in runs]
    axes[0, 1].bar(x, adv, color=[run.spec.color for run in runs], alpha=0.88)
    axes[0, 1].plot(x, clean, color="black", marker="o", linewidth=1.2, label="clean qnorm")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=28, ha="right")
    axes[0, 1].set_title("Final original all-W qnorm, batch mean")
    axes[0, 1].set_ylabel("qnorm")
    axes[0, 1].grid(True, axis="y", alpha=0.25)
    axes[0, 1].legend(loc="best")

    axes[0, 0].set_title("Active metric, normalized to k=0")
    axes[0, 0].set_ylabel("metric / metric(k=0)")
    axes[0, 0].set_xlabel("attack step k")
    axes[0, 0].grid(True, alpha=0.28)
    axes[0, 0].legend(ncol=2, loc="best")

    axes[1, 0].set_title("Delta L2 norm, batch mean")
    axes[1, 0].set_xlabel("attack step k")
    axes[1, 0].set_ylabel("||delta||_2")
    axes[1, 0].grid(True, alpha=0.28)
    axes[1, 0].axhline(32.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    axes[1, 1].set_title("Boundary ratio, batch mean")
    axes[1, 1].set_xlabel("attack step k")
    axes[1, 1].set_ylabel("||delta||_2 / epsilon")
    axes[1, 1].set_ylim(-0.02, 1.08)
    axes[1, 1].grid(True, alpha=0.28)
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    out = out_dir / "ns2d_eps32_alpha10_altloss_mean_curves_cleanstyle.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_sample0_comparison(runs: list[RunData], out_dir: Path, sample_position: int) -> Path:
    base = runs[0]
    sample_index = int(base.final_npz["dataset_indices"][sample_position])
    x_clean = base.final_npz["x_clean"][sample_position]
    clean_error = base.final_npz["clean_model_minus_solver"][sample_position]

    field_arrays = [x_clean] + [run.final_npz["x_adv"][sample_position] for run in runs]
    delta_arrays = [run.final_npz["final_delta"][sample_position] for run in runs]
    error_arrays = [clean_error] + [run.final_npz["adv_model_minus_solver"][sample_position] for run in runs]
    field_vlim = value_limits(field_arrays, percentile=99.0)
    delta_vlim = sym_limits(delta_arrays, percentile=99.0)
    error_vlim = sym_limits(error_arrays, percentile=99.0)

    row_items: list[tuple[str, str, dict[str, Any] | RunData]] = [
        ("Initial condition", "initial", {"field": x_clean, "error": clean_error})
    ]
    row_items.extend((run.spec.label, "attack", run) for run in runs)

    nrows = len(row_items)
    fig, axes = plt.subplots(
        nrows,
        4,
        figsize=(15.2, 2.05 * nrows),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.2, 1.55]},
        constrained_layout=True,
    )
    fig.suptitle(
        f"NS2D eps32 alpha10 steepest_add comparison, dataset {sample_index}: initial, baseline loss3, alternative loss3 metrics",
        fontsize=13,
    )

    field_im = delta_im = error_im = None
    for row, (label, kind, item) in enumerate(row_items):
        ax_field, ax_delta, ax_spec, ax_curve = axes[row]
        if kind == "initial":
            field = item["field"]  # type: ignore[index]
            error = item["error"]  # type: ignore[index]
            field_im = ax_field.imshow(field, origin="lower", cmap="viridis", vmin=field_vlim[0], vmax=field_vlim[1])
            ax_field.set_title(f"{label}\nx_clean")
            ax_delta.axis("off")
            error_im = ax_delta.imshow(error, origin="lower", cmap="coolwarm", vmin=error_vlim[0], vmax=error_vlim[1])
            ax_delta.set_title("Clean FNO - solver")
            centers, prof = radial_profile(field)
            ax_spec.plot(centers, prof, color="black", linewidth=1.7)
            ax_spec.set_title("Initial condition spectrum")
            ax_curve.axis("off")
            clean_q = final_metric_sample(base, "clean_true_loss", sample_position)
            ax_curve.text(0.02, 0.78, f"dataset index: {sample_index}", transform=ax_curve.transAxes)
            ax_curve.text(0.02, 0.58, f"clean qnorm: {clean_q:.4g}", transform=ax_curve.transAxes)
        else:
            run = item  # type: ignore[assignment]
            assert isinstance(run, RunData)
            field = run.final_npz["x_adv"][sample_position]
            delta = run.final_npz["final_delta"][sample_position]
            error = run.final_npz["adv_model_minus_solver"][sample_position]
            color = run.spec.color

            field_im = ax_field.imshow(field, origin="lower", cmap="viridis", vmin=field_vlim[0], vmax=field_vlim[1])
            ax_field.set_title(f"{label}\nx + final delta")
            delta_im = ax_delta.imshow(delta, origin="lower", cmap="coolwarm", vmin=delta_vlim[0], vmax=delta_vlim[1])
            l2 = float(np.linalg.norm(delta.reshape(-1)))
            linf = float(np.max(np.abs(delta)))
            final_q = final_metric_sample(run, "adv_true_loss", sample_position)
            clean_q = final_metric_sample(run, "clean_true_loss", sample_position)
            ax_delta.set_title(f"Final delta\nL2={l2:.2f}, Linf={linf:.3f}")

            centers, prof = radial_profile(delta)
            ax_spec.plot(centers, prof, color=color, linewidth=1.8)
            ax_spec.set_title("Final delta spectrum")

            srows = sample_rows_for_position(run.sample_rows, sample_position)
            k = series(srows, "k")
            active = series(srows, "loss3")
            denom = active[0] if np.isfinite(active[0]) and abs(active[0]) > 1e-12 else 1.0
            ax_curve.plot(k, active / denom, color=color, label="active metric / k0")
            ax_curve.set_title(f"sample active metric; final qnorm={final_q:.4g}")
            ax_curve.set_ylabel("normalized metric")
            ax_curve.text(
                0.02,
                0.80,
                f"qnorm increase: {final_q - clean_q:+.4g}",
                transform=ax_curve.transAxes,
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
            )

            error_im = ax_field.figure.axes[row * 4 + 1].images[0] if False else error_im

        for ax in (ax_field, ax_delta):
            ax.set_xticks([])
            ax.set_yticks([])
        ax_spec.axvline(RADIAL_AXIS_CUTOFF, color="black", linestyle=":", linewidth=0.9, alpha=0.75)
        ax_spec.axvline(RADIAL_CORNER_CUTOFF, color="black", linestyle="--", linewidth=0.9, alpha=0.75)
        ax_spec.set_yscale("log")
        ax_spec.set_xlabel("normalized radial frequency")
        ax_spec.set_ylabel("power")
        ax_spec.grid(True, alpha=0.25)
        if kind == "attack":
            ax_curve.set_xlabel("attack step k")
            ax_curve.grid(True, alpha=0.25)

    # Replace the second column for attack rows with model-solver error overlays
    # on a twin inset-like axis would be too dense; keep delta in column 2 and
    # include error in a separate compact grid for clarity.
    if field_im is not None:
        fig.colorbar(field_im, ax=axes[:, 0].ravel().tolist(), fraction=0.022, pad=0.01, label="state")
    if delta_im is not None:
        fig.colorbar(delta_im, ax=axes[1:, 1].ravel().tolist(), fraction=0.022, pad=0.01, label="delta")

    out = out_dir / "ns2d_eps32_alpha10_altloss_sample0_cleanstyle_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_error_grid(runs: list[RunData], out_dir: Path, sample_position: int) -> Path:
    base = runs[0]
    sample_index = int(base.final_npz["dataset_indices"][sample_position])
    arrays = [base.final_npz["clean_model_minus_solver"][sample_position]]
    arrays += [run.final_npz["adv_model_minus_solver"][sample_position] for run in runs]
    vlim = sym_limits(arrays, percentile=99.0)
    labels = ["Initial clean error"] + [run.spec.label for run in runs]
    fig, axes = plt.subplots(1, len(labels), figsize=(2.35 * len(labels), 2.45), constrained_layout=True)
    fig.suptitle(f"NS2D dataset {sample_index}: FNO - solver final-state error", fontsize=12)
    im = None
    for ax, label, arr in zip(axes, labels, arrays):
        im = ax.imshow(arr, origin="lower", cmap="coolwarm", vmin=vlim[0], vmax=vlim[1])
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
    out = out_dir / "ns2d_eps32_alpha10_altloss_sample0_model_solver_error_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--alt-root", type=Path, default=DEFAULT_ALT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-position", type=int, default=0)
    args = parser.parse_args()

    setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args.baseline_dir, args.alt_root)
    runs = [load_run(spec) for spec in specs]

    pngs = [
        str(save_mean_curve_summary(runs, args.out_dir)),
        str(save_sample0_comparison(runs, args.out_dir, args.sample_position)),
        str(save_error_grid(runs, args.out_dir, args.sample_position)),
    ]
    summary = metric_summary_rows(runs)
    summary_csv = args.out_dir / "ns2d_eps32_alpha10_altloss_summary.csv"
    write_csv(summary_csv, summary)

    report = {
        "note": "CPU-only clean-style plots from saved artifacts; no model/solver/attack rerun.",
        "baseline_dir": str(args.baseline_dir),
        "alt_root": str(args.alt_root),
        "out_dir": str(args.out_dir),
        "sample_position": args.sample_position,
        "runs": [{"key": run.spec.key, "label": run.spec.label, "source": str(run.spec.method_dir)} for run in runs],
        "pngs": pngs,
        "summary_csv": str(summary_csv),
        "radial_axis_cutoff_rho": float(RADIAL_AXIS_CUTOFF),
        "radial_corner_cutoff_rho": float(RADIAL_CORNER_CUTOFF),
    }
    report_path = args.out_dir / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
