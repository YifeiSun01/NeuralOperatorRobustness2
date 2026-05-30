#!/usr/bin/env python3
"""Diagnose why some 2D NS generalization datasets have lower loss.

The script compares train/test/generalization datasets for the recurrent
FNO2d model, records absolute and relative errors, measures simple target
range/energy/complexity statistics, and saves example panels for low-loss and
high-loss cases.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NS_ROOT = PROJECT_ROOT / "2D_NS_FNO2d_recurrent"
if str(NS_ROOT) not in sys.path:
    sys.path.insert(0, str(NS_ROOT))

from models.FNO2d import FNO2d, RecurrentPredictor  # noqa: E402


DEFAULT_CHECKPOINT = (
    NS_ROOT
    / "saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC"
    / "checkpoints/best.pt"
)
DEFAULT_TRAIN = (
    NS_ROOT
    / "datasets/exponax_datasets/t20/real_initial_laxmap_single/train"
    / "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_ntimepoints21_all_frames.pt"
)
DEFAULT_TEST = (
    NS_ROOT
    / "datasets/exponax_datasets/t20/real_initial_laxmap_single/test"
    / "dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt"
)
DEFAULT_GEN_DIR = PROJECT_ROOT / "generalization_datasets/ns2d"
DEFAULT_OUT = PROJECT_ROOT / "analysis_outputs/ns2d_generalization_loss_diagnostics"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def load_y(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "y" not in data:
        raise ValueError(f"{path} must be a dict containing key 'y'")
    y = data["y"]
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    if y.ndim != 4:
        raise ValueError(f"{path} expected y shape [N,H,W,T], got {tuple(y.shape)}")
    return y.float().contiguous(), dict(data.get("metadata", {}))


def parse_name(path: Path, group: str) -> dict[str, Any]:
    name = path.stem
    out: dict[str, Any] = {"dataset": name, "group": group, "tier": ""}
    if name.startswith("ns_near_"):
        out["tier"] = "near"
    elif name.startswith("ns_mid_"):
        out["tier"] = "mid"
    elif name.startswith("ns_far_"):
        out["tier"] = "far"
    elif group in {"train", "test"}:
        out["tier"] = group

    for key in ("alpha", "tau", "scale", "shift"):
        m = re.search(rf"{key}(m?\d+(?:p\d+)?)", name)
        if m:
            raw = m.group(1).replace("p", ".")
            if raw.startswith("m"):
                raw = "-" + raw[1:]
            try:
                out[key] = float(raw)
            except ValueError:
                out[key] = raw
    return out


def load_model(checkpoint: Path, modes: int, width: int, t_in: int, t_out: int, device: torch.device) -> RecurrentPredictor:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = FNO2d(modes1=modes, modes2=modes, width=width, in_channels=t_in)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return RecurrentPredictor(model, T_out=t_out, step=1).to(device).eval()


def spectral_high_fraction(frame: torch.Tensor, cutoff_fraction: float = 0.25) -> float:
    arr = frame.detach().float()
    ft = torch.fft.rfft2(arr)
    power = (ft.real.square() + ft.imag.square())
    h, w2 = power.shape
    ky = torch.fft.fftfreq(h, device=power.device).abs().reshape(h, 1)
    kx = torch.fft.rfftfreq((w2 - 1) * 2, device=power.device).abs().reshape(1, w2)
    radius = torch.sqrt(kx.square() + ky.square())
    mask = radius >= cutoff_fraction * radius.max().clamp_min(1e-12)
    total = power.sum().clamp_min(1e-12)
    return float(power[mask].sum().div(total).cpu())


def sample_stats(x: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    err = pred - target
    target_flat = target.reshape(-1)
    pred_flat = pred.reshape(-1)
    err_flat = err.reshape(-1)
    target_norm = torch.linalg.norm(target_flat).clamp_min(1e-12)
    x0 = x[..., 0]
    y_last = target[..., -1]
    return {
        "rmse": float(torch.sqrt(torch.mean(err_flat.square())).cpu()),
        "mae": float(torch.mean(err_flat.abs()).cpu()),
        "relative_l2": float((torch.linalg.norm(err_flat) / target_norm).cpu()),
        "target_rms": float(torch.sqrt(torch.mean(target_flat.square())).cpu()),
        "target_abs_mean": float(target_flat.abs().mean().cpu()),
        "target_std": float(target_flat.std(unbiased=False).cpu()),
        "target_range": float((target_flat.max() - target_flat.min()).cpu()),
        "pred_range": float((pred_flat.max() - pred_flat.min()).cpu()),
        "err_range": float((err_flat.max() - err_flat.min()).cpu()),
        "initial_range": float((x0.max() - x0.min()).cpu()),
        "initial_std": float(x0.std(unbiased=False).cpu()),
        "initial_highfreq_frac": spectral_high_fraction(x0),
        "target_last_highfreq_frac": spectral_high_fraction(y_last),
    }


def evaluate_dataset(
    rp: RecurrentPredictor,
    path: Path,
    group: str,
    device: torch.device,
    *,
    t_in: int,
    t_out: int,
    target_size: int,
    max_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y, metadata = load_y(path)
    n, h, w, t = y.shape
    if t_in + t_out > t:
        raise ValueError(f"{path} has T={t}, need {t_in+t_out}")
    sx = h // target_size
    sy = w // target_size
    count = min(n, max_samples) if max_samples > 0 else n
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for i in range(count):
            sample = y[i, ::sx, ::sy, :]
            x = sample[..., :t_in].unsqueeze(0).to(device)
            target = sample[..., t_in : t_in + t_out].unsqueeze(0).to(device)
            pred = rp(x)
            stats = sample_stats(x.squeeze(0).cpu(), target.squeeze(0).cpu(), pred.squeeze(0).cpu())
            rows.append(
                {
                    **parse_name(path, group),
                    "path": str(path),
                    "sample_index": i,
                    "metadata_family": metadata.get("family", ""),
                    "metadata_description": metadata.get("description", ""),
                    **stats,
                }
            )

    df = pd.DataFrame(rows)
    summary = {
        **parse_name(path, group),
        "path": str(path),
        "n_evaluated": int(count),
        "metadata_family": metadata.get("family", ""),
        "metadata_description": metadata.get("description", ""),
    }
    for col in [
        "rmse",
        "mae",
        "relative_l2",
        "target_rms",
        "target_abs_mean",
        "target_std",
        "target_range",
        "initial_range",
        "initial_std",
        "initial_highfreq_frac",
        "target_last_highfreq_frac",
    ]:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=0))
    return summary, rows


def add_group_baselines(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    refs = {}
    for group in ("train", "test"):
        row = out[out["group"] == group]
        if not row.empty:
            refs[group] = {
                "rmse": float(row.iloc[0]["rmse_mean"]),
                "relative_l2": float(row.iloc[0]["relative_l2_mean"]),
            }
    for ref_name, vals in refs.items():
        out[f"rmse_vs_{ref_name}"] = out["rmse_mean"] / vals["rmse"]
        out[f"relative_l2_vs_{ref_name}"] = out["relative_l2_mean"] / vals["relative_l2"]
        out[f"rmse_delta_{ref_name}"] = out["rmse_mean"] - vals["rmse"]
        out[f"relative_l2_delta_{ref_name}"] = out["relative_l2_mean"] - vals["relative_l2"]
    return out


def scatter_plots(summary: pd.DataFrame, outdir: Path) -> None:
    plot_df = summary.copy()
    color_map = {"train": "#2ca02c", "test": "#ff7f0e", "near": "#4c78a8", "mid": "#b279a2", "far": "#e45756"}
    colors = [color_map.get(str(t), "#777777") for t in plot_df["tier"]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    panels = [
        ("target_rms_mean", "rmse_mean", "Target RMS", "Absolute RMSE"),
        ("target_rms_mean", "relative_l2_mean", "Target RMS", "Relative L2"),
        ("initial_highfreq_frac_mean", "rmse_mean", "Initial high-frequency fraction", "Absolute RMSE"),
        ("target_range_mean", "rmse_mean", "Target range", "Absolute RMSE"),
    ]
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.ravel(), panels):
        ax.scatter(plot_df[xcol], plot_df[ycol], c=colors, s=42, alpha=0.85, edgecolor="white", linewidth=0.4)
        for _, row in plot_df.iterrows():
            if row["group"] in {"train", "test"}:
                ax.annotate(row["group"], (row[xcol], row[ycol]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=8)
        for k, v in color_map.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5)
    fig.savefig(outdir / "dataset_loss_vs_scale_complexity.png", dpi=180)
    plt.close(fig)

    ordered = summary.sort_values("rmse_mean")
    fig, ax = plt.subplots(figsize=(14, max(7, len(ordered) * 0.16)))
    ax.barh(np.arange(len(ordered)), ordered["rmse_mean"], color=[color_map.get(str(t), "#777777") for t in ordered["tier"]])
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(ordered["dataset"], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(float(summary.loc[summary["group"] == "test", "rmse_mean"].iloc[0]), color="#ff7f0e", lw=1.5, label="test")
    ax.axvline(float(summary.loc[summary["group"] == "train", "rmse_mean"].iloc[0]), color="#2ca02c", lw=1.5, label="train")
    ax.set_xlabel("Absolute RMSE, lower is better")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "dataset_rmse_ranked.png", dpi=180)
    plt.close(fig)


def plot_example_panel(
    rp: RecurrentPredictor,
    row: pd.Series,
    outpath: Path,
    device: torch.device,
    *,
    t_in: int,
    t_out: int,
    target_size: int,
) -> None:
    y, _ = load_y(Path(row["path"]))
    _, h, w, _ = y.shape
    sx = h // target_size
    sy = w // target_size
    sample = y[int(row["sample_index"]), ::sx, ::sy, :]
    x = sample[..., :t_in].unsqueeze(0).to(device)
    target = sample[..., t_in : t_in + t_out].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = rp(x)
    x_cpu = x.squeeze(0).cpu()
    target_cpu = target.squeeze(0).cpu()
    pred_cpu = pred.squeeze(0).cpu()
    err_cpu = pred_cpu - target_cpu

    fields = [
        ("initial t0", x_cpu[..., 0], "coolwarm"),
        ("ground truth final", target_cpu[..., -1], "coolwarm"),
        ("model output last", pred_cpu[..., -1], "coolwarm"),
        ("model - target", err_cpu[..., -1], "RdBu_r"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    title = (
        f"{row['dataset']} | sample {int(row['sample_index'])} | "
        f"RMSE={row['rmse']:.4g}, relL2={row['relative_l2']:.4g}, "
        f"target_rms={row['target_rms']:.4g}, range={row['target_range']:.4g}"
    )
    fig.suptitle(title, fontsize=10)
    for ax, (name, arr, cmap) in zip(axes, fields):
        vals = arr.numpy()
        if "model - target" in name:
            vmax = float(np.percentile(np.abs(vals), 99))
            vmin = -vmax
        else:
            vmax = float(np.percentile(np.abs(vals), 99))
            vmin = -vmax
        im = ax.imshow(vals, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def write_summary_note(summary: pd.DataFrame, sample_df: pd.DataFrame, outdir: Path) -> None:
    gen = summary[summary["group"] == "generalization"].copy()
    test_rmse = float(summary.loc[summary["group"] == "test", "rmse_mean"].iloc[0])
    test_rel = float(summary.loc[summary["group"] == "test", "relative_l2_mean"].iloc[0])
    train_rmse = float(summary.loc[summary["group"] == "train", "rmse_mean"].iloc[0])
    train_rel = float(summary.loc[summary["group"] == "train", "relative_l2_mean"].iloc[0])
    lower_abs = int((gen["rmse_mean"] < test_rmse).sum())
    lower_rel = int((gen["relative_l2_mean"] < test_rel).sum())
    corr_cols = ["target_rms_mean", "target_range_mean", "initial_highfreq_frac_mean", "target_last_highfreq_frac_mean"]
    corrs = {
        col: {
            "rmse_corr": float(gen[[col, "rmse_mean"]].corr().iloc[0, 1]),
            "relative_l2_corr": float(gen[[col, "relative_l2_mean"]].corr().iloc[0, 1]),
        }
        for col in corr_cols
        if gen[col].std(ddof=0) > 0
    }
    low = gen.sort_values("rmse_mean").head(8)[["dataset", "tier", "rmse_mean", "relative_l2_mean", "target_rms_mean", "target_range_mean", "initial_highfreq_frac_mean"]]
    high = gen.sort_values("rmse_mean", ascending=False).head(8)[["dataset", "tier", "rmse_mean", "relative_l2_mean", "target_rms_mean", "target_range_mean", "initial_highfreq_frac_mean"]]

    def simple_markdown_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.6g}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# NS2D generalization loss diagnostics",
        "",
        f"- train: RMSE={train_rmse:.6g}, relative_L2={train_rel:.6g}",
        f"- test: RMSE={test_rmse:.6g}, relative_L2={test_rel:.6g}",
        f"- generalization datasets below test absolute RMSE: {lower_abs}/{len(gen)}",
        f"- generalization datasets below test relative L2: {lower_rel}/{len(gen)}",
        "",
        "## Main interpretation",
        "",
        "Absolute loss can become smaller when the target trajectory has smaller amplitude/range or lower high-frequency content. "
        "That does not always mean the model generalized better; relative L2 divides by the target norm and is the better first check. "
        "If both absolute RMSE and relative L2 drop, the case is probably genuinely easier. If only RMSE drops while relative L2 is flat or worse, "
        "the low loss is mostly a scale/range artifact.",
        "",
        "## Correlations on generalization datasets",
        "",
        "```json",
        json.dumps(to_jsonable(corrs), indent=2, sort_keys=True),
        "```",
        "",
        "## Lowest absolute RMSE generalization datasets",
        "",
        simple_markdown_table(low),
        "",
        "## Highest absolute RMSE generalization datasets",
        "",
        simple_markdown_table(high),
        "",
        "## Files",
        "",
        "- `dataset_summary.csv`: per-dataset metrics and train/test ratios",
        "- `sample_metrics.csv`: per-sample metrics",
        "- `dataset_loss_vs_scale_complexity.png`: scale/complexity diagnostics",
        "- `dataset_rmse_ranked.png`: ranked dataset loss",
        "- `examples/`: low/high loss sample panels",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--examples-per-side", type=int, default=6)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--width", type=int, default=60)
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--t-out", type=int, default=10)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "examples").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[load] checkpoint={args.checkpoint}")
    print(f"[device] {device}")
    rp = load_model(args.checkpoint, args.modes, args.width, args.t_in, args.t_out, device)

    datasets = [("train", args.train), ("test", args.test)]
    datasets += [("generalization", p) for p in sorted(args.gen_dir.glob("*.pt"))]

    summaries: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for idx, (group, path) in enumerate(datasets, start=1):
        print(f"[{idx}/{len(datasets)}] {group}: {path.name}")
        summary, rows = evaluate_dataset(
            rp,
            path,
            group,
            device,
            t_in=args.t_in,
            t_out=args.t_out,
            target_size=args.target_size,
            max_samples=args.max_samples,
        )
        summaries.append(summary)
        sample_rows.extend(rows)

    summary_df = add_group_baselines(pd.DataFrame(summaries))
    sample_df = pd.DataFrame(sample_rows)
    summary_df.to_csv(args.outdir / "dataset_summary.csv", index=False)
    sample_df.to_csv(args.outdir / "sample_metrics.csv", index=False)
    (args.outdir / "run_config.json").write_text(json.dumps(to_jsonable(vars(args)), indent=2), encoding="utf-8")

    scatter_plots(summary_df, args.outdir)

    gen_samples = sample_df[sample_df["group"] == "generalization"].copy()
    chosen = pd.concat(
        [
            gen_samples.sort_values("rmse").head(args.examples_per_side).assign(example_bucket="low_rmse"),
            gen_samples.sort_values("rmse", ascending=False).head(args.examples_per_side).assign(example_bucket="high_rmse"),
        ],
        ignore_index=True,
    )
    chosen.to_csv(args.outdir / "chosen_examples.csv", index=False)
    for i, row in chosen.iterrows():
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{i:02d}_{row['example_bucket']}_{row['dataset']}_sample{int(row['sample_index'])}")
        plot_example_panel(
            rp,
            row,
            args.outdir / "examples" / f"{slug}.png",
            device,
            t_in=args.t_in,
            t_out=args.t_out,
            target_size=args.target_size,
        )

    write_summary_note(summary_df, sample_df, args.outdir)
    print(f"[done] wrote {args.outdir}")


if __name__ == "__main__":
    main()
