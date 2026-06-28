#!/usr/bin/env python3
"""Visualize recurrent FNO test predictions against ground truth frames.

For each selected test sample this script writes:

* one 4-panel GIF per sample:
  initial condition, solver / ground-truth frame, model prediction frame,
  and prediction-minus-solver frame.

The model forward pass is GPU-only by default. Plotting happens after tensors
are copied back to CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NS_ROOT = PROJECT_ROOT / "2D_NS_FNO2d_recurrent"
if str(NS_ROOT) not in sys.path:
    sys.path.insert(0, str(NS_ROOT))

from models.FNO2d import FNO2d, RecurrentPredictor


DEFAULT_TEST_PATH = (
    NS_ROOT
    / "datasets"
    / "exponax_datasets"
    / "t20"
    / "real_initial_laxmap_single"
    / "test"
    / "dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_ntimepoints21_all_frames.pt"
)
DEFAULT_OUT_DIR = NS_ROOT / "visualizations" / "fno2d_recurrent_test_predictions"


def parse_indices(text: str, n_total: int, num_samples: int, seed: int) -> list[int]:
    if text:
        out = [int(part.strip()) for part in text.split(",") if part.strip()]
    else:
        rng = random.Random(seed)
        out = sorted(rng.sample(range(n_total), k=min(num_samples, n_total)))
    for idx in out:
        if idx < 0 or idx >= n_total:
            raise ValueError(f"Sample index {idx} out of range [0, {n_total}).")
    return out


def load_checkpoint_state(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        state = payload.get("model_state_dict") or payload.get("state_dict") or payload.get("model") or payload
    else:
        state = payload
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def load_model(args: argparse.Namespace, device):
    model = FNO2d(
        modes1=args.modes1,
        modes2=args.modes2,
        width=args.width,
        num_layers=args.num_layers,
        in_channels=args.t_in,
    ).to(device)
    model.load_state_dict(load_checkpoint_state(args.checkpoint), strict=True)
    recurrent = RecurrentPredictor(model, T_out=args.t_out, step=args.step).to(device)
    recurrent.eval()
    for param in recurrent.parameters():
        param.requires_grad_(False)
    return recurrent


def robust_range(arr: np.ndarray, *, symmetric: bool = False) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(finite), 99.5))
        vmax = max(vmax, 1e-12)
        return -vmax, vmax
    vmin, vmax = np.nanpercentile(finite, [0.5, 99.5])
    if math.isclose(float(vmin), float(vmax)):
        pad = max(abs(float(vmin)) * 0.01, 1e-6)
        return float(vmin - pad), float(vmax + pad)
    return float(vmin), float(vmax)


def add_colorbar(fig, image, ax):
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)


def save_four_panel_gif(
    *,
    out_path: Path,
    sample_index: int,
    initial: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    diff: np.ndarray,
) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ic_vmin, ic_vmax = robust_range(initial)
    truth_vmin, truth_vmax = robust_range(truth)
    pred_vmin, pred_vmax = robust_range(pred)
    diff_vmin, diff_vmax = robust_range(diff, symmetric=True)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7), constrained_layout=True)
    fig.suptitle(f"FNO2d recurrent test sample {sample_index}", fontsize=12)
    images = [
        axes[0].imshow(initial, cmap="viridis", vmin=ic_vmin, vmax=ic_vmax, origin="lower"),
        axes[1].imshow(truth[0], cmap="viridis", vmin=truth_vmin, vmax=truth_vmax, origin="lower"),
        axes[2].imshow(pred[0], cmap="viridis", vmin=pred_vmin, vmax=pred_vmax, origin="lower"),
        axes[3].imshow(diff[0], cmap="coolwarm", vmin=diff_vmin, vmax=diff_vmax, origin="lower"),
    ]
    titles = [
        axes[0].set_title("initial condition"),
        axes[1].set_title("solver truth t=11"),
        axes[2].set_title("model output t=11"),
        axes[3].set_title("model - solver t=11"),
    ]
    for ax, image in zip(axes, images):
        ax.set_xticks([])
        ax.set_yticks([])
        add_colorbar(fig, image, ax)

    def update(frame_idx):
        images[1].set_data(truth[frame_idx])
        images[2].set_data(pred[frame_idx])
        images[3].set_data(diff[frame_idx])
        titles[1].set_text(f"solver truth t={frame_idx + 11}")
        titles[2].set_text(f"model output t={frame_idx + 11}")
        titles[3].set_text(f"model - solver t={frame_idx + 11}")
        return (*images, *titles)

    anim = FuncAnimation(fig, update, frames=range(diff.shape[0]), interval=350, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=3))
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--indices", default="", help="Comma-separated test indices. If empty, randomly sample.")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--modes1", type=int, default=64)
    parser.add_argument("--modes2", type=int, default=64)
    parser.add_argument("--width", type=int, default=60)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--t-out", type=int, default=10)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required for official visualization inference; refusing CPU fallback.")
    device = torch.device(args.device if torch.cuda.is_available() or args.allow_cpu else "cpu")

    data = torch.load(args.test_path, map_location="cpu", weights_only=False)
    if "y" not in data:
        raise ValueError("Expected test dataset to contain key 'y' with all frames.")
    y = data["y"].float()
    if y.ndim != 4 or y.shape[-1] < args.t_in + args.t_out:
        raise ValueError(f"Expected y shape [N,H,W,{args.t_in + args.t_out}+], got {tuple(y.shape)}")
    indices = parse_indices(args.indices, int(y.shape[0]), args.num_samples, args.seed)

    recurrent = load_model(args, device)
    x = y[indices, ..., : args.t_in].to(device)
    truth = y[indices, ..., args.t_in : args.t_in + args.t_out].cpu().numpy()
    initial = y[indices, ..., 0].cpu().numpy()

    with torch.inference_mode():
        pred = recurrent(x).detach().cpu().numpy()

    diff = pred - truth
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pos, sample_index in enumerate(indices):
        panel_gif = args.out_dir / f"sample_{sample_index:03d}_initial_solver_model_diff.gif"
        save_four_panel_gif(
            out_path=panel_gif,
            sample_index=sample_index,
            initial=initial[pos],
            truth=np.moveaxis(truth[pos], -1, 0),
            pred=np.moveaxis(pred[pos], -1, 0),
            diff=np.moveaxis(diff[pos], -1, 0),
        )
        rows.append(
            {
                "sample_index": int(sample_index),
                "panel_gif": str(panel_gif),
                "mse": float(np.mean(diff[pos] ** 2)),
                "relative_l2": float(np.linalg.norm(diff[pos].reshape(-1)) / (np.linalg.norm(truth[pos].reshape(-1)) + 1e-12)),
            }
        )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "test_path": str(args.test_path),
                "indices": [int(i) for i in indices],
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "indices": indices, "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
