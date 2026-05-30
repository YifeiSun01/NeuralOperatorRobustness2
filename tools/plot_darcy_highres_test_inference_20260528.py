#!/usr/bin/env python3
"""Visualize Darcy FNO high-resolution test inference.

Loads a trained Darcy FNO checkpoint, runs it on held-out test samples at a
chosen resolution, and plots coefficient A, model output, solver ground truth,
and model-solver difference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DARCY_ROOT = ROOT / "2D_Darcy_FNO2d"
if str(DARCY_ROOT) not in sys.path:
    sys.path.insert(0, str(DARCY_ROOT))

from models.FNO2d import FNO2d


def parse_indices(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    denom = np.linalg.norm(target.reshape(-1))
    if denom == 0:
        return float("nan")
    return float(np.linalg.norm((pred - target).reshape(-1)) / denom)


def make_plot(
    *,
    out_path: Path,
    sample_index: int,
    a: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    diff: np.ndarray,
    metrics: dict[str, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    u_min = float(min(np.min(pred), np.min(true)))
    u_max = float(max(np.max(pred), np.max(true)))
    d_abs = float(np.percentile(np.abs(diff), 99.5))
    if not np.isfinite(d_abs) or d_abs == 0:
        d_abs = float(np.max(np.abs(diff)) + 1e-12)

    panels = [
        ("Coefficient A", a, "viridis", None, None),
        ("Model U", pred, "viridis", u_min, u_max),
        ("Solver U", true, "viridis", u_min, u_max),
        ("Model - Solver", diff, "coolwarm", -d_abs, d_abs),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    for ax, (title, arr, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        (
            f"Darcy test sample {sample_index} at 421x421 | "
            f"relL2={metrics['rel_l2']:.4e}, MAE={metrics['mae']:.4e}, "
            f"max|diff|={metrics['max_abs']:.4e}"
        ),
        fontsize=13,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DARCY_ROOT / "saved_models" / "2D" / "darcy_N1500_nx85_m64_w60_e500_20260528" / "best.pt",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DARCY_ROOT
        / "datasets"
        / "grf_darcy_20260528_N1500"
        / "test"
        / "dim2d_darcy_nx421_N300_solver=jaxcg_solve421_alpha2_tau3_binary3-12_f1_seed45_test.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis_outputs" / "darcy_85model_on_421_test_20260528",
    )
    parser.add_argument("--indices", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.checkpoint = args.checkpoint.resolve()
    args.dataset = args.dataset.resolve()
    args.out_dir = args.out_dir.resolve()
    indices = parse_indices(args.indices)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = FNO2d(
        modes1=int(config.get("modes", 64)),
        modes2=int(config.get("modes", 64)),
        width=int(config.get("width", 60)),
        num_layers=int(config.get("num_layers", 4)),
        in_channels=1,
        out_channels=1,
        padding=int(config.get("padding", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    x = data["x"].float()
    y = data["y"].float()
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"expected x/y as (N,S,S), got {tuple(x.shape)} and {tuple(y.shape)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, float | int | str]] = []
    pred_records: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for sample_index in indices:
            if sample_index < 0 or sample_index >= x.shape[0]:
                raise IndexError(f"sample index {sample_index} outside test set size {x.shape[0]}")
            xx = x[sample_index : sample_index + 1].unsqueeze(-1).to(device, non_blocking=True)
            pred_t = model(xx)[0, ..., 0]
            if device.type == "cuda":
                torch.cuda.synchronize()

            a_np = tensor_to_numpy(x[sample_index])
            true_np = tensor_to_numpy(y[sample_index])
            pred_np = tensor_to_numpy(pred_t)
            diff_np = pred_np - true_np
            row = {
                "sample_index": sample_index,
                "rel_l2": rel_l2(pred_np, true_np),
                "mse": float(np.mean(diff_np**2)),
                "mae": float(np.mean(np.abs(diff_np))),
                "max_abs": float(np.max(np.abs(diff_np))),
                "pred_min": float(np.min(pred_np)),
                "pred_max": float(np.max(pred_np)),
                "true_min": float(np.min(true_np)),
                "true_max": float(np.max(true_np)),
            }
            metrics_rows.append(row)
            pred_records[f"sample_{sample_index}_a"] = a_np
            pred_records[f"sample_{sample_index}_pred"] = pred_np
            pred_records[f"sample_{sample_index}_true"] = true_np
            pred_records[f"sample_{sample_index}_diff"] = diff_np
            make_plot(
                out_path=args.out_dir / f"darcy_421_test_sample_{sample_index:03d}.png",
                sample_index=sample_index,
                a=a_np,
                pred=pred_np,
                true=true_np,
                diff=diff_np,
                metrics=row,
            )

    metrics_path = args.out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    np.savez_compressed(args.out_dir / "sample_predictions.npz", **pred_records)
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "out_dir": str(args.out_dir),
        "indices": indices,
        "mean_rel_l2": float(np.mean([float(r["rel_l2"]) for r in metrics_rows])),
        "mean_mae": float(np.mean([float(r["mae"]) for r in metrics_rows])),
        "max_abs_over_samples": float(max(float(r["max_abs"]) for r in metrics_rows)),
        "metrics_csv": str(metrics_path),
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
