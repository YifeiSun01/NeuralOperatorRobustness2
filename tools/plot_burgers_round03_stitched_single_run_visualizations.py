#!/usr/bin/env python3
"""Plot one Burgers round03 run after stitching base and later training.

This is for user-facing "single run" visualizations: loss1 should appear as a
single 0..3000 training curve, and loss2/loss3 as single 0..1000 curves, even
though the later epochs were produced by separate resume run directories.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIABLE_SCRIPT = PROJECT_ROOT / "tools/plot_burgers_training_run_visualizations_variable_epoch.py"


def load_variable_module():
    spec = importlib.util.spec_from_file_location("variable_epoch_burgers_plots", VARIABLE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {VARIABLE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", required=True, type=Path)
    parser.add_argument("--extension-run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--suffix", default="round03_long")
    parser.add_argument("--max-lines-per-panel", type=int, default=5)
    parser.add_argument("--skip-fft", action="store_true")
    return parser.parse_args()


def burgers_dir(run_dir: Path) -> Path:
    return run_dir / "burgers"


def read_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = burgers_dir(run_dir) / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def stitched_eval(base_run: Path, extension_run: Path) -> pd.DataFrame:
    base = read_csv(base_run, "eval_metrics.csv")
    ext = read_csv(extension_run, "eval_metrics.csv")
    if "phase" in ext.columns:
        ext = ext[ext["phase"] != "resume_checkpoint_before_adversarial_training"].copy()
    df = pd.concat([base, ext], ignore_index=True)
    return (
        df.sort_values(["epoch", "dataset_id", "phase"])
        .drop_duplicates(["epoch", "dataset_id"], keep="first")
        .sort_values(["manual_tier", "manual_rank", "dataset_id", "epoch"])
        .reset_index(drop=True)
    )


def stitched_epoch_summary(base_run: Path, extension_run: Path) -> pd.DataFrame:
    base = read_csv(base_run, "attack_epoch_summary.csv")
    ext = read_csv(extension_run, "attack_epoch_summary.csv")
    df = pd.concat([base, ext], ignore_index=True)
    return df.sort_values(["epoch"]).drop_duplicates(["epoch"], keep="first").reset_index(drop=True)


def stitched_bucket_summary(base_run: Path, extension_run: Path) -> pd.DataFrame:
    base = read_csv(base_run, "attack_epsilon_bucket_summary.csv")
    ext = read_csv(extension_run, "attack_epsilon_bucket_summary.csv")
    df = pd.concat([base, ext], ignore_index=True)
    keys = ["epoch", "bucket_index"]
    return df.sort_values(keys).drop_duplicates(keys, keep="first").reset_index(drop=True)


def stitched_probe_summary(base_run: Path, extension_run: Path) -> pd.DataFrame:
    base = read_csv(base_run, "attack_probe_samples.csv")
    ext = read_csv(extension_run, "attack_probe_samples.csv")
    df = pd.concat([base, ext], ignore_index=True)
    keys = ["epoch", "probe_rank"]
    return df.sort_values(keys).drop_duplicates(keys, keep="first").reset_index(drop=True)


def resolve_npz(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def compute_delta_fft_matrix_from_probe(plotmod, probe_df: pd.DataFrame, epochs: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    freq_modes = None
    for epoch in sorted(int(e) for e in epochs):
        d = probe_df[probe_df["epoch"] == epoch]
        if d.empty:
            continue
        npz_path = resolve_npz(str(d["npz_path"].iloc[0]))
        data = dict(np.load(npz_path, allow_pickle=True))
        powers = []
        for sample_idx in range(data["delta"].shape[0]):
            freq, power = plotmod.fft_power(data["delta"][sample_idx])
            if freq_modes is None:
                freq_modes = freq[1:]
            powers.append(power[1:])
        rows.append(np.nanmean(np.asarray(powers, dtype=float), axis=0))
    if freq_modes is None or not rows:
        raise FileNotFoundError("no attack probe npz rows available for stitched FFT plot")
    return np.asarray(sorted(epochs), dtype=int), np.asarray(freq_modes, dtype=float), np.asarray(rows, dtype=float)


def selected_fft_epochs(max_epoch: int, available_epochs: np.ndarray) -> list[int]:
    if max_epoch >= 3000:
        requested = [50, 200, 400, 800, 1200, 1800, 2400, 3000]
    elif max_epoch >= 1000:
        requested = [50, 100, 200, 400, 600, 800, 1000]
    else:
        requested = [int(round(max_epoch * f)) for f in [0.05, 0.20, 0.40, 0.60, 0.80, 1.00]]
    out: list[int] = []
    for epoch in requested:
        chosen = int(available_epochs[np.argmin(np.abs(available_epochs - epoch))])
        if chosen not in out:
            out.append(chosen)
    return out


def plot_delta_fft_from_probe(plotmod, probe_df: pd.DataFrame, polished_dir: Path, suffix: str) -> Path:
    raw_epochs = np.asarray(sorted(int(e) for e in probe_df["epoch"].dropna().unique()), dtype=int)
    epochs, freq_modes, raw_matrix = compute_delta_fft_matrix_from_probe(plotmod, probe_df, list(raw_epochs))
    epoch_smooth = plotmod.rolling_epoch_matrix(raw_matrix, 25)
    log_power = np.log10(np.clip(raw_matrix, 1e-18, None))
    finite = log_power[np.isfinite(log_power)]
    vmin, vmax = np.nanpercentile(finite, [2, 99.5])

    max_epoch = int(epochs[-1])
    selected = selected_fft_epochs(max_epoch, epochs)
    fig = plt.figure(figsize=(18, 12.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.28], hspace=0.24)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_line = fig.add_subplot(gs[1, 0])

    im = ax_heat.imshow(
        log_power,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[float(freq_modes[0]), float(freq_modes[-1]), float(epochs[0]), float(epochs[-1])],
        cmap="magma",
        vmin=float(vmin),
        vmax=float(vmax),
    )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.024, pad=0.018)
    cbar.set_label("log10 normalized FFT power")
    ax_heat.set_title("Raw delta FFT power heatmap, no moving average", loc="left", fontweight="bold")
    ax_heat.set_xlabel("Fourier mode")
    ax_heat.set_ylabel("training epoch")
    ax_heat.set_xlim(1, min(512, float(freq_modes[-1])))
    ax_heat.set_yticks(plotmod.major_ticks(max_epoch))

    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(selected)))
    for epoch, color in zip(selected, colors):
        idx = int(np.argmin(np.abs(epochs - epoch)))
        ax_line.plot(freq_modes, epoch_smooth[idx] + 1e-18, color=color, lw=1.55, alpha=0.84, label=f"epoch {epochs[idx]}")
    ax_line.set_yscale("log")
    ax_line.set_xlim(1, min(512, float(freq_modes[-1])))
    ax_line.set_title("Selected spectra from 25-epoch moving-average matrix; top heatmap remains raw", loc="left", fontweight="bold", fontsize=10.5)
    ax_line.set_xlabel("Fourier mode")
    ax_line.set_ylabel("normalized power")
    ax_line.legend(ncol=min(8, len(selected)), loc="upper right", fontsize=7.6)

    fig.suptitle("Delta frequency content during adversarial training", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return plotmod.savefig(
        fig,
        polished_dir / plotmod.name_with_suffix("polished_checkpoint_style_delta_fft_raw_heatmap_25epoch_smoothed_lines.png", suffix),
    )


def main() -> None:
    args = parse_args()
    plotmod = load_variable_module()
    plotmod.setup_matplotlib()

    out_dir = args.out_dir
    polished_dir = out_dir / "polished_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    polished_dir.mkdir(parents=True, exist_ok=True)

    eval_df = stitched_eval(args.base_run_dir, args.extension_run_dir)
    attack_df = stitched_epoch_summary(args.base_run_dir, args.extension_run_dir)
    bucket_df = stitched_bucket_summary(args.base_run_dir, args.extension_run_dir)
    probe_df = stitched_probe_summary(args.base_run_dir, args.extension_run_dir)

    outputs: list[str] = []
    outputs.append(str(plotmod.plot_attack_loss(attack_df, bucket_df, polished_dir, args.suffix)))
    for metric in ["rmse", "relative_l2"]:
        outputs.append(str(plotmod.plot_corrected_loss(eval_df, metric, polished_dir, args.suffix)))
        outputs.append(str(plotmod.plot_checkpoint_style_loss(eval_df, metric, polished_dir, args.suffix)))
        for smooth_window, short in [(None, "raw"), (25, "ma25")]:
            out = out_dir / plotmod.name_with_suffix(
                f"{metric}_grouped_shared_y_distinct_datasets_max5_high_transparency_{short}.png",
                args.suffix,
            )
            index = plotmod.plot_high_transparency_max5(eval_df, metric, out, args.max_lines_per_panel, smooth_window)
            csv_path = out.with_suffix(".csv")
            index.to_csv(csv_path, index=False)
            outputs.extend([str(out), str(csv_path)])

    if not args.skip_fft:
        outputs.append(str(plot_delta_fft_from_probe(plotmod, probe_df, polished_dir, args.suffix)))

    manifest = {
        "base_run_dir": str(args.base_run_dir),
        "extension_run_dir": str(args.extension_run_dir),
        "out_dir": str(out_dir),
        "max_eval_epoch": int(eval_df["epoch"].max()),
        "max_attack_epoch": int(attack_df["epoch"].max()),
        "suffix": args.suffix,
        "outputs": outputs,
    }
    manifest_path = polished_dir / plotmod.name_with_suffix("variable_epoch_visualization_manifest.json", args.suffix)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n".join(outputs + [str(manifest_path)]), flush=True)


if __name__ == "__main__":
    main()
