#!/usr/bin/env python3
"""Plot DeepONet predictions against Burgers ground truth for train/test samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np
import torch

import deepxde as dde

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_burgers_deeponet_deepxde import DEFAULT_TEST, DEFAULT_TRAIN, format_float_tag, load_split, make_grid, periodic_features_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path("deeponet_training_runs/burgers_nu0p001_deeponet1d_500"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test_path", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--domain", type=float, default=2.0)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--train_samples", type=int, default=3)
    parser.add_argument("--test_samples", type=int, default=3)
    parser.add_argument("--branch_widths", default=None)
    parser.add_argument("--trunk_widths", default=None)
    parser.add_argument("--activation", default=None)
    parser.add_argument("--kernel_initializer", default=None)
    return parser.parse_args()


def parse_widths(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def finite_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "training_logs" / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_model(args: argparse.Namespace, config: dict[str, Any]) -> dde.Model:
    branch_layers = config.get("branch_layers")
    trunk_layers = config.get("trunk_layers")
    periodic_trunk = bool(config.get("periodic_trunk", False))
    if branch_layers is None:
        branch_widths = args.branch_widths or "256,256,256"
        branch_layers = [args.nx, *parse_widths(branch_widths)]
    if trunk_layers is None:
        trunk_widths = args.trunk_widths or "128,128,256"
        trunk_layers = [4 if periodic_trunk else 1, *parse_widths(trunk_widths)]
    activation = args.activation or config.get("activation", "gelu")
    kernel_initializer = args.kernel_initializer or config.get("kernel_initializer", "Glorot normal")

    trunk = make_grid(args.nx, args.domain)
    dummy_branch = np.zeros((1, args.nx), dtype=np.float32)
    dummy_y = np.zeros((1, args.nx), dtype=np.float32)
    data = dde.data.TripleCartesianProd(
        X_train=(dummy_branch, trunk),
        y_train=dummy_y,
        X_test=(dummy_branch, trunk),
        y_test=dummy_y,
    )
    net = dde.nn.DeepONetCartesianProd(branch_layers, trunk_layers, activation, kernel_initializer)
    if periodic_trunk:
        net.apply_feature_transform(lambda x: periodic_features_torch(x, args.domain))
    stats_path = args.run_dir / "training_logs" / "output_transform_stats.npz"
    if bool(config.get("output_transform", False)):
        if not stats_path.exists():
            raise FileNotFoundError(
                f"DeepONet config requires output_transform=True, but missing {stats_path}. "
                "This file stores per-grid y_mean/y_std and is required for physical-scale predictions."
            )
        stats = np.load(stats_path)
        if "y_mean" not in stats or "y_std" not in stats:
            raise KeyError(f"{stats_path} must contain 'y_mean' and 'y_std'. Found keys: {list(stats.files)}")
        y_mean = stats["y_mean"].astype(np.float32)
        y_std = stats["y_std"].astype(np.float32)
        expected_shape = (1, int(args.nx))
        if y_mean.shape != expected_shape or y_std.shape != expected_shape:
            raise ValueError(
                f"{stats_path} has invalid output-transform shapes: "
                f"y_mean={y_mean.shape}, y_std={y_std.shape}, expected {expected_shape}."
            )
        y_mean_t = torch.as_tensor(y_mean)
        y_std_t = torch.as_tensor(y_std)

        def output_transform(_inputs: tuple[torch.Tensor, torch.Tensor], outputs: torch.Tensor) -> torch.Tensor:
            return outputs * y_std_t.to(outputs.device) + y_mean_t.to(outputs.device)

        net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    model.compile("adam", lr=float(config.get("lr", 1e-3)), loss="MSE")
    return model


def predict_samples(
    model: dde.Model,
    split_name: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    trunk: np.ndarray,
    count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for idx in range(min(count, len(inputs))):
        pred = np.asarray(model.predict((inputs[idx : idx + 1], trunk)), dtype=np.float64)[0]
        target = targets[idx].astype(np.float64)
        inp = inputs[idx].astype(np.float64)
        err = pred - target
        mse = float(np.mean(err**2))
        rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(target), 1e-12))
        row = {
            "split": split_name,
            "sample_index": idx,
            "mse": finite_float(mse),
            "rel_l2": finite_float(rel_l2),
            "max_abs_error": finite_float(np.max(np.abs(err))),
        }
        rows.append(row)
        samples.append({"row": row, "input": inp, "target": target, "pred": pred, "error": err})
    return rows, samples


def plot_combined(path: Path, trunk: np.ndarray, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xgrid = trunk.reshape(-1)
    nrows = len(samples)
    fig, axes = plt.subplots(nrows, 3, figsize=(16, max(3.0 * nrows, 5.0)), sharex=True)
    if nrows == 1:
        axes = np.asarray([axes])

    for row_idx, sample in enumerate(samples):
        meta = sample["row"]
        title_prefix = f"{meta['split']} sample {meta['sample_index']}"

        ax = axes[row_idx, 0]
        ax.plot(xgrid, sample["input"], color="0.25", linewidth=1.1)
        ax.set_title(f"{title_prefix}: input")
        ax.set_ylabel("u")
        ax.grid(True, alpha=0.25)

        ax = axes[row_idx, 1]
        ax.plot(xgrid, sample["target"], color="#1f77b4", linewidth=1.4, label="ground truth")
        ax.plot(xgrid, sample["pred"], color="#d62728", linewidth=1.2, linestyle="--", label="DeepONet")
        ax.fill_between(xgrid, sample["target"], sample["pred"], color="#d62728", alpha=0.12, linewidth=0)
        ax.set_title(f"output overlay | MSE={meta['mse']:.3e}, rel L2={meta['rel_l2']:.3e}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[row_idx, 2]
        ax.plot(xgrid, sample["error"], color="#7b3294", linewidth=1.1)
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_title(f"prediction - ground truth | max={meta['max_abs_error']:.3e}")
        ax.grid(True, alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    fig.suptitle("Burgers DeepONet predictions on training and testing samples", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.run_dir)
    nu = args.nu if args.nu is not None else float(config.get("nu", 0.001))
    checkpoint = args.checkpoint or args.run_dir / "checkpoints" / f"deeponet_burgers_nu{format_float_tag(nu)}.pt"
    outdir = args.outdir or args.run_dir / "plots" / "train_test_predictions"
    trunk = make_grid(args.nx, args.domain)

    model = build_model(args, config)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.net.load_state_dict(ckpt["model_state_dict"])

    train_x, train_y, _ = load_split(args.train_path)
    test_x, test_y, _ = load_split(args.test_path)
    train_rows, train_samples = predict_samples(model, "train", train_x, train_y, trunk, args.train_samples)
    test_rows, test_samples = predict_samples(model, "test", test_x, test_y, trunk, args.test_samples)
    rows = train_rows + test_rows
    samples = train_samples + test_samples

    write_csv(outdir / "prediction_metrics.csv", rows)
    plot_combined(outdir / "deeponet_train_test_prediction_overlay.png", trunk, samples)
    print(f"[done] wrote {outdir / 'deeponet_train_test_prediction_overlay.png'}")
    print(f"[done] wrote {outdir / 'prediction_metrics.csv'}")


if __name__ == "__main__":
    main()
