#!/usr/bin/env python3
"""Train a DeepXDE DeepONet on the Burgers nu=0.001 dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch

import deepxde as dde

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "1D_Burgers"
    / "datasets"
    / "1D"
    / "Burgers"
    / "batched_exponax_splits"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
)
DEFAULT_TRAIN = DEFAULT_DATA_DIR / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt"
DEFAULT_TEST = DEFAULT_DATA_DIR / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt"


def format_float_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def finite_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_split(path: Path, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "x" not in data or "y" not in data:
        raise ValueError(f"{path} must be a dict containing x and y tensors.")
    x = data["x"].detach().cpu().float().numpy()
    y = data["y"].detach().cpu().float().numpy()
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if y.ndim == 3 and y.shape[-1] == 1:
        y = y[..., 0]
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError(f"Expected x/y with matching shape [N,nx], got {x.shape} and {y.shape}.")
    if max_samples is not None:
        x = x[:max_samples]
        y = y[:max_samples]
    return x.astype(np.float32), y.astype(np.float32), data.get("metadata", {})


def make_grid(nx: int, domain: float) -> np.ndarray:
    return np.linspace(0.0, float(domain), int(nx), endpoint=False, dtype=np.float32)[:, None]


def periodic_features_torch(x: torch.Tensor, domain: float) -> torch.Tensor:
    theta = 2.0 * math.pi * x[:, 0:1] / float(domain)
    return torch.cat(
        [
            torch.cos(theta),
            torch.sin(theta),
            torch.cos(2.0 * theta),
            torch.sin(2.0 * theta),
        ],
        dim=1,
    )


def evaluate_mse(model: dde.Model, branch_x: np.ndarray, trunk_x: np.ndarray, y_true: np.ndarray, batch_size: int) -> float:
    sq_sum = 0.0
    count = 0
    for start in range(0, len(branch_x), batch_size):
        stop = min(start + batch_size, len(branch_x))
        pred = model.predict((branch_x[start:stop], trunk_x))
        pred = np.asarray(pred, dtype=np.float64)
        target = y_true[start:stop].astype(np.float64)
        sq_sum += float(np.sum((pred - target) ** 2))
        count += int(np.prod(target.shape))
    return sq_sum / max(count, 1)


def rel_l2_mean(model: dde.Model, branch_x: np.ndarray, trunk_x: np.ndarray, y_true: np.ndarray, batch_size: int) -> float:
    values: list[float] = []
    for start in range(0, len(branch_x), batch_size):
        stop = min(start + batch_size, len(branch_x))
        pred = np.asarray(model.predict((branch_x[start:stop], trunk_x)), dtype=np.float64)
        target = y_true[start:stop].astype(np.float64)
        num = np.linalg.norm((pred - target).reshape(stop - start, -1), axis=1)
        den = np.linalg.norm(target.reshape(stop - start, -1), axis=1)
        values.extend((num / np.maximum(den, 1e-12)).tolist())
    return float(np.mean(values)) if values else float("nan")


def plot_loss_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.asarray([row["step"] for row in rows], dtype=float)
    train = np.asarray([row.get("train_loss_0", np.nan) for row in rows], dtype=float)
    test = np.asarray([row.get("test_loss_0", np.nan) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train, label="train MSE")
    ax.plot(steps, test, label="test MSE")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("MSE")
    ax.set_title("Burgers DeepONet Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_inference_samples(
    outdir: Path,
    model: dde.Model,
    branch_x: np.ndarray,
    trunk_x: np.ndarray,
    y_true: np.ndarray,
    num_samples: int,
) -> list[dict[str, Any]]:
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    xgrid = trunk_x.reshape(-1)
    count = min(num_samples, len(branch_x))
    for idx in range(count):
        pred = np.asarray(model.predict((branch_x[idx : idx + 1], trunk_x)), dtype=np.float64)[0]
        target = y_true[idx].astype(np.float64)
        inp = branch_x[idx].astype(np.float64)
        error = pred - target
        mse = float(np.mean(error**2))
        rel_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(target), 1e-12))
        rows.append({"sample_index": idx, "mse": mse, "rel_l2": rel_l2})

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(xgrid, inp, color="0.25", linewidth=1.1, label="input u0")
        axes[0].plot(xgrid, target, color="#1f77b4", linewidth=1.4, label="true solver output")
        axes[0].plot(xgrid, pred, color="#d62728", linewidth=1.2, linestyle="--", label="DeepONet prediction")
        axes[0].set_ylabel("u")
        axes[0].set_title(f"DeepONet test sample {idx} | MSE={mse:.3e}, rel L2={rel_l2:.3e}")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=9)

        axes[1].plot(xgrid, error, color="#7b3294", linewidth=1.1, label="prediction - true")
        axes[1].axhline(0.0, color="0.3", linewidth=0.8)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("error")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(outdir / f"test_sample_{idx}_prediction.png", dpi=180)
        plt.close(fig)
    return rows


def parse_widths(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test_path", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "deeponet_training_runs" / "burgers_nu0p001_deeponet1d_500")
    parser.add_argument("--nu", type=float, default=0.001)
    parser.add_argument("--domain", type=float, default=2.0)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=None, help="Override epochs * ceil(n_train / batch_size).")
    parser.add_argument("--batch_size", type=int, default=64, help="Branch batch size; use 0 for full-batch training.")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Recorded in config; DeepXDE PyTorch compile uses Adam without decoupled weight decay.")
    parser.add_argument("--branch_widths", default="256,256,256")
    parser.add_argument("--trunk_widths", default="128,128,256")
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--kernel_initializer", default="Glorot normal")
    parser.add_argument("--periodic_trunk", action="store_true", help="Use Lu-group Burgers periodic trunk Fourier features.")
    parser.add_argument("--output_transform", action="store_true", help="Use y mean/std output transform as in the Burgers DeepONet example.")
    parser.add_argument("--decay_inverse_time", action="store_true", help="Use DeepXDE inverse-time learning-rate decay.")
    parser.add_argument(
        "--lu_burgers_reference",
        action="store_true",
        help="Use the public Burgers DeepONet-style defaults: tanh, 128-width nets, periodic trunk, output transform, lr decay.",
    )
    parser.add_argument("--display_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--compare_samples", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lu_burgers_reference:
        args.activation = "tanh"
        args.branch_widths = "128,128,128,128"
        args.trunk_widths = "128,128,128"
        args.periodic_trunk = True
        args.output_transform = True
        args.decay_inverse_time = True

    start_time = time.perf_counter()
    train_batch_size = None if args.batch_size <= 0 else args.batch_size
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    (args.output_dir / "training_logs").mkdir(exist_ok=True)

    dde.config.set_random_seed(args.seed)
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    train_x, train_y, train_meta = load_split(args.train_path, args.max_train_samples)
    test_x, test_y, test_meta = load_split(args.test_path, args.max_test_samples)
    if train_x.shape[1] != args.nx or test_x.shape[1] != args.nx:
        raise ValueError(f"nx mismatch: args.nx={args.nx}, train={train_x.shape}, test={test_x.shape}")

    trunk = make_grid(args.nx, args.domain)
    data = dde.data.TripleCartesianProd(
        X_train=(train_x, trunk),
        y_train=train_y,
        X_test=(test_x, trunk),
        y_test=test_y,
    )

    branch_layers = [args.nx, *parse_widths(args.branch_widths)]
    trunk_input_dim = 4 if args.periodic_trunk else 1
    trunk_layers = [trunk_input_dim, *parse_widths(args.trunk_widths)]
    if branch_layers[-1] != trunk_layers[-1]:
        raise ValueError(f"DeepONet branch/trunk last widths must match, got {branch_layers[-1]} and {trunk_layers[-1]}.")

    net = dde.nn.DeepONetCartesianProd(
        branch_layers,
        trunk_layers,
        args.activation,
        args.kernel_initializer,
    )
    if args.periodic_trunk:
        net.apply_feature_transform(lambda x: periodic_features_torch(x, args.domain))
    y_mean = np.mean(train_y, axis=0, keepdims=True).astype(np.float32)
    y_std = np.std(train_y, axis=0, keepdims=True).astype(np.float32)
    y_std = np.maximum(y_std, 1e-6).astype(np.float32)
    if args.output_transform:
        y_mean_t = torch.as_tensor(y_mean)
        y_std_t = torch.as_tensor(y_std)

        def output_transform(_inputs: tuple[torch.Tensor, torch.Tensor], outputs: torch.Tensor) -> torch.Tensor:
            return outputs * y_std_t.to(outputs.device) + y_mean_t.to(outputs.device)

        net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    decay = None
    if args.decay_inverse_time:
        steps_per_epoch = 1 if train_batch_size is None else math.ceil(len(train_x) / train_batch_size)
        decay_steps = args.iterations if args.iterations is not None else int(args.epochs * steps_per_epoch)
        decay = ("inverse time", max(decay_steps // 5, 1), 0.5)
    model.compile(
        "adam",
        lr=args.lr,
        decay=decay,
        loss="MSE",
        metrics=["mean l2 relative error"],
    )

    iterations = args.iterations
    if iterations is None:
        steps_per_epoch = 1 if train_batch_size is None else math.ceil(len(train_x) / train_batch_size)
        iterations = int(args.epochs * steps_per_epoch)

    config = vars(args) | {
        "iterations": iterations,
        "train_batch_size_resolved": train_batch_size,
        "branch_layers": branch_layers,
        "trunk_layers": trunk_layers,
        "train_shape": train_x.shape,
        "test_shape": test_x.shape,
        "trunk_shape": trunk.shape,
        "trunk_input_dim": trunk_input_dim,
        "periodic_trunk": args.periodic_trunk,
        "output_transform": args.output_transform,
        "decay_inverse_time": args.decay_inverse_time,
        "decay": decay,
        "y_mean_shape": y_mean.shape,
        "y_std_shape": y_std.shape,
        "deepxde_version": dde.__version__,
        "torch_version": torch.__version__,
    }
    save_json(args.output_dir / "training_logs" / "config.json", config)
    save_json(
        args.output_dir / "training_logs" / "dataset_info.json",
        {
            "train_path": args.train_path,
            "test_path": args.test_path,
            "train_x_shape": train_x.shape,
            "train_y_shape": train_y.shape,
            "test_x_shape": test_x.shape,
            "test_y_shape": test_y.shape,
            "train_metadata": train_meta,
            "test_metadata": test_meta,
        },
    )
    np.savez(args.output_dir / "training_logs" / "output_transform_stats.npz", y_mean=y_mean, y_std=y_std)

    loss_history, train_state = model.train(
        iterations=iterations,
        batch_size=train_batch_size,
        display_every=args.display_every,
    )

    loss_rows = []
    for step, loss_train, loss_test in zip(loss_history.steps, loss_history.loss_train, loss_history.loss_test):
        row: dict[str, Any] = {"step": int(step)}
        for i, value in enumerate(np.atleast_1d(loss_train)):
            row[f"train_loss_{i}"] = float(value)
        for i, value in enumerate(np.atleast_1d(loss_test)):
            row[f"test_loss_{i}"] = float(value)
        loss_rows.append(row)
    write_csv(args.output_dir / "training_logs" / "losses.csv", loss_rows)
    plot_loss_history(args.output_dir / "plots" / "training_loss.png", loss_rows)

    train_mse = evaluate_mse(model, train_x, trunk, train_y, args.eval_batch_size)
    test_mse = evaluate_mse(model, test_x, trunk, test_y, args.eval_batch_size)
    train_rel_l2 = rel_l2_mean(model, train_x, trunk, train_y, args.eval_batch_size)
    test_rel_l2 = rel_l2_mean(model, test_x, trunk, test_y, args.eval_batch_size)
    sample_rows = plot_inference_samples(args.output_dir / "plots" / "test_samples", model, test_x, trunk, test_y, args.compare_samples)
    write_csv(args.output_dir / "training_logs" / "inference_sample_metrics.csv", sample_rows)

    checkpoint_path = args.output_dir / "checkpoints" / f"deeponet_burgers_nu{format_float_tag(args.nu)}.pt"
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "config": finite_json(config),
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_rel_l2": train_rel_l2,
            "test_rel_l2": test_rel_l2,
        },
        checkpoint_path,
    )
    summary = {
        "output_dir": args.output_dir,
        "checkpoint_path": checkpoint_path,
        "iterations": iterations,
        "epochs_equivalent": iterations / max((1 if train_batch_size is None else math.ceil(len(train_x) / train_batch_size)), 1),
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_rel_l2_mean": train_rel_l2,
        "test_rel_l2_mean": test_rel_l2,
        "inference_sample_metrics": sample_rows,
        "loss_plot": args.output_dir / "plots" / "training_loss.png",
        "sample_plot_dir": args.output_dir / "plots" / "test_samples",
        "total_seconds": time.perf_counter() - start_time,
        "best_step": int(getattr(train_state, "best_step", -1)),
        "best_loss_train": finite_json(getattr(train_state, "best_loss_train", None)),
        "best_loss_test": finite_json(getattr(train_state, "best_loss_test", None)),
    }
    save_json(args.output_dir / "training_logs" / "summary.json", summary)
    print(f"[done] saved DeepONet checkpoint to {checkpoint_path}")
    print(f"[done] train_mse={train_mse:.8e} test_mse={test_mse:.8e}")
    print(f"[done] train_rel_l2={train_rel_l2:.8e} test_rel_l2={test_rel_l2:.8e}")


if __name__ == "__main__":
    main()
