#!/usr/bin/env python3
"""Train a static FNO2d on Darcy coefficient-to-solution pairs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
DARCY_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
if str(DARCY_ROOT) not in sys.path:
    sys.path.insert(0, str(DARCY_ROOT))

from models.FNO2d import FNO2d


class RelativeLpLoss:
    def __init__(self, p: float = 2.0, eps: float = 1e-12):
        self.p = p
        self.eps = eps

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        diff = torch.linalg.vector_norm(pred_flat - target_flat, ord=self.p, dim=1)
        denom = torch.linalg.vector_norm(target_flat, ord=self.p, dim=1).clamp_min(self.eps)
        return torch.mean(diff / denom)


def count_params(model: torch.nn.Module) -> int:
    total = 0
    for param in model.parameters():
        multiplier = 2 if param.is_complex() else 1
        total += multiplier * math.prod(param.shape)
    return int(total)


def load_xy(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "x" not in data or "y" not in data:
        raise KeyError(f"{path} must contain x and y tensors")
    x = data["x"].float()
    y = data["y"].float()
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"expected x/y shape (N,S,S), got {tuple(x.shape)} and {tuple(y.shape)}")
    return x.unsqueeze(-1).contiguous(), y.unsqueeze(-1).contiguous(), data.get("metadata", {})


def make_loaders(args: argparse.Namespace):
    x_train, y_train, train_meta = load_xy(args.train_path)
    if args.ntrain is not None:
        x_train = x_train[: args.ntrain]
        y_train = y_train[: args.ntrain]

    if args.test_path is not None:
        x_test, y_test, test_meta = load_xy(args.test_path)
        if args.ntest is not None:
            x_test = x_test[: args.ntest]
            y_test = y_test[: args.ntest]
    else:
        if args.ntest is None:
            args.ntest = min(100, max(1, x_train.shape[0] // 10))
        if args.ntest >= x_train.shape[0]:
            raise ValueError("--ntest must be smaller than the train file sample count when --test-path is omitted")
        x_test = x_train[-args.ntest :].clone()
        y_test = y_train[-args.ntest :].clone()
        x_train = x_train[: -args.ntest]
        y_train = y_train[: -args.ntest]
        test_meta = {"split_from_train_path": str(args.train_path), "ntest": args.ntest}

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, train_meta, test_meta, x_train.shape[1]


def evaluate(model, loader, loss_fn, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xx, yy in loader:
            xx = xx.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)
            pred = model(xx)
            loss = loss_fn(pred, yy)
            batch_n = xx.shape[0]
            total += float(loss.item()) * batch_n
            count += batch_n
    return total / max(1, count)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--ntrain", type=int, default=None)
    parser.add_argument("--ntest", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-dir", type=Path, default=DARCY_ROOT / "saved_models" / "2D")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.train_path = args.train_path.resolve()
    args.test_path = args.test_path.resolve() if args.test_path is not None else None
    args.save_dir = args.save_dir.resolve()

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Darcy training; pass --no-require-cuda to override.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, train_meta, test_meta, resolution = make_loaders(args)
    model = FNO2d(
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        num_layers=args.num_layers,
        in_channels=1,
        out_channels=1,
        padding=args.padding,
    ).to(device)
    model_parameters = count_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * len(train_loader)),
    )
    loss_fn = RelativeLpLoss()

    run_name = args.run_name or (
        f"darcy_nx{resolution}_modes{args.modes}_width{args.width}_"
        f"epochs{args.epochs}_{time.strftime('%Y%m%d_%H%M%S_UTC', time.gmtime())}"
    )
    run_dir = args.save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    config = {
        "project_root": str(PROJECT_ROOT),
        "darcy_root": str(DARCY_ROOT),
        "train_path": str(args.train_path),
        "test_path": str(args.test_path) if args.test_path else None,
        "resolution": resolution,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "modes": args.modes,
        "width": args.width,
        "num_layers": args.num_layers,
        "padding": args.padding,
        "model_parameters": model_parameters,
        "train_metadata": train_meta,
        "test_metadata": test_meta,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(json.dumps(config, indent=2))
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("epoch,seconds,train_rel_l2,test_rel_l2,lr,cuda_max_memory_mib\n")
        best_test = float("inf")
        for epoch in tqdm(range(args.epochs), desc="Training Darcy FNO2d", unit="epoch"):
            model.train()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            train_total = 0.0
            train_count = 0
            for xx, yy in train_loader:
                xx = xx.to(device, non_blocking=True)
                yy = yy.to(device, non_blocking=True)
                pred = model(xx)
                loss = loss_fn(pred, yy)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_n = xx.shape[0]
                train_total += float(loss.item()) * batch_n
                train_count += batch_n
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            train_loss = train_total / max(1, train_count)
            test_loss = evaluate(model, test_loader, loss_fn, device)
            max_mem = (
                torch.cuda.max_memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else 0.0
            )
            lr = optimizer.param_groups[0]["lr"]
            log_file.write(f"{epoch},{seconds:.6f},{train_loss:.8e},{test_loss:.8e},{lr:.8e},{max_mem:.2f}\n")
            log_file.flush()

            if test_loss < best_test:
                best_test = test_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "test_rel_l2": test_loss,
                    },
                    run_dir / "best.pt",
                )
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "test_rel_l2": test_loss,
                    },
                    run_dir / f"epoch{epoch + 1:04d}.pt",
                )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": args.epochs - 1,
            "test_rel_l2": evaluate(model, test_loader, loss_fn, device),
        },
        run_dir / "final.pt",
    )
    print(f"[saved] {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
