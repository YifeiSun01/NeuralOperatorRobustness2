#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train FNO1d on original dataset mixed with a chosen attacked dataset(s).

Three modes:
  - replace50: replace 50% of original training samples with attacked
  - add50:     append attacked samples equal to 50% of original training size
  - add100:    append attacked samples equal to 100% of original training size

Folder layout (assumed):
  - This file:        1D_Burgers/training_models_expanded/train_FNO1d_mixed.py
  - Save models/logs: 1D_Burgers/saved_models_expanded/...

Original dataset example:
  pt_orig = ".../1D_Burgers/datasets/1D/Burgers/pos/dim1d_nx1024_..._nu0.0005_t1.0_seed45.pt"

Attacked datasets root example:
  attacks_root = ".../1D_Burgers/datasets/1D/Burgers/expanded_pos/t1"

Under attacks_root we look for subdirs like:
  nu0.0005_pgd/norm2_eps50_alpha0.1_steps20/N=1500/dataset.pt

You can pick one subdir, or run ALL allowed subdirs listed below.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO1d import FNO1d
from utilities3 import LpLoss, count_params

# ------------------------ Allowed attacked subdirs ------------------------
# 只在这些子目录中循环（相对 nu0.0005_pgd/）
ALLOWED_ATTACK_SUBDIRS = [
    "norm2_eps20_alpha0.1_steps20/N=1500",
    "norm2_eps20_alpha0.5_steps20/N=1500",
    "norm2_eps20_alpha1_steps20/N=1500",
    "norm2_eps50_alpha0.01_steps20/N=1500",
    "norm2_eps50_alpha0.05_steps20/N=1500",
    "norm2_eps50_alpha0.1_steps20/N=1500",
    "norm2_eps50_alpha0.5_steps20/N=1500",
    "norm2_eps50_alpha1_steps20/N=1500",
]

# ------------------------ Helpers ------------------------

def torch_load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def ensure_2d_xy(x, y):
    """
    Make sure x,y are shape (N, X).
    - x: (N, X) or (N, X, 1) -> squeeze to (N, X)
    - y: (N, X) or (N, X, T) -> take final frame to (N, X) if 3D
    """
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError(f"x must be (N,X) or (N,X,1), got {tuple(x.shape)}")

    if y.ndim == 3:
        # assume (N, X, T), take final frame
        y = y[..., -1]
    if y.ndim != 2:
        raise ValueError(f"y must be (N,X) or (N,X,T), got {tuple(y.shape)}")

    return x, y

def make_mix(original_xy, attacked_xy, mode, ntrain):
    """
    Build mixed training set according to mode.
    original_xy: (x0, y0) from original dataset (shape (N, X))
    attacked_xy: (xa, ya) from attacked dataset (shape (N, X))
    mode in {"replace50", "add50", "add100"}
    returns: (x_train, y_train)
    """
    x0, y0 = original_xy
    xa, ya = attacked_xy

    if x0.shape[1] != xa.shape[1] or y0.shape[1] != ya.shape[1]:
        raise ValueError(f"X-size mismatch: original ({x0.shape[1]}) vs attacked ({xa.shape[1]})")

    # Base original train split
    x_train = x0[:ntrain].clone()
    y_train = y0[:ntrain].clone()

    if mode == "replace50":
        k = ntrain // 2
        if xa.shape[0] < k:
            raise ValueError(f"Attacked dataset too small for replace50: need {k}, have {xa.shape[0]}")
        x_train[:k] = xa[:k]
        y_train[:k] = ya[:k]
        return x_train, y_train

    elif mode == "add50":
        k = ntrain // 2
        if xa.shape[0] < k:
            raise ValueError(f"Attacked dataset too small for add50: need {k}, have {xa.shape[0]}")
        x_train = torch.cat([x_train, xa[:k]], dim=0)
        y_train = torch.cat([y_train, ya[:k]], dim=0)
        return x_train, y_train

    elif mode == "add100":
        k = ntrain
        if xa.shape[0] < k:
            raise ValueError(f"Attacked dataset too small for add100: need {k}, have {xa.shape[0]}")
        x_train = torch.cat([x_train, xa[:k]], dim=0)
        y_train = torch.cat([y_train, ya[:k]], dim=0)
        return x_train, y_train

    else:
        raise ValueError(f"Unknown mode {mode}")

def find_attack_pt(attacks_root: Path, subdir_rel: str) -> Path:
    """
    attacks_root/nu0.0005_pgd/{subdir_rel}/dataset.pt  (or .../N=1500/dataset.pt)
    """
    nu_dir = attacks_root / "nu0.0005_pgd"
    cand_dir = nu_dir / subdir_rel
    # typical filename
    pt = cand_dir / "dataset.pt"
    if pt.is_file():
        return pt
    # fallback: search within cand_dir
    found = list(cand_dir.rglob("dataset.pt"))
    if found:
        return found[0]
    raise FileNotFoundError(f"dataset.pt not found under {cand_dir}")

def save_run_artifacts(save_dir: Path, model, config: dict, log_lines):
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = save_dir / "model.pth"
    torch.save(model.state_dict(), model_path)

    # Save config JSON
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Save log
    with open(save_dir / "train.log", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    return model_path

def build_save_dir(root_1d_burgers: Path, modes: int, width: int, epochs: int,
                   mode_name: str, attack_tag: str):
    """
    Return directory:
      1D_Burgers/saved_models_expanded/1D/modes{modes}_width{width}_epochs{epochs}/unnormalized/{mode_name}/{attack_tag}
    """
    return (root_1d_burgers / f"saved_models_expanded/1D/modes{modes}_width{width}_epochs{epochs}"
            / "unnormalized" / mode_name / attack_tag)

def parse_attack_tag_from_subdir(subdir_rel: str) -> str:
    # e.g. "norm2_eps50_alpha0.1_steps20/N=1500" -> "norm2_eps50_alpha0.1_steps20"
    p = Path(subdir_rel)
    return p.parent.name if p.name.startswith("N=") else p.name

# ------------------------ Training ------------------------

def train_one_mix(pt_orig: Path,
                  pt_attack: Path,
                  save_dir: Path,
                  ntrain=1000,
                  ntest=100,
                  batch_size=20,
                  learning_rate=1e-3,
                  epochs=500,
                  modes=16,
                  width=64,
                  device: str = "cuda"):
    """
    Train one run with a prepared (x_train, y_train) & original test-set.
    The mixing mode has been applied before calling this (we re-apply here to keep code self-contained).
    """

    log = []
    t0 = time.time()
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    log.append(f"device = {dev}")

    # ---- Load datasets
    data_o = torch_load_pt(pt_orig)
    xo_raw, yo_raw = data_o["x"], data_o["y"]
    xo, yo = ensure_2d_xy(xo_raw, yo_raw)
    N_total, s = xo.shape

    data_a = torch_load_pt(pt_attack)
    xa_raw, ya_raw = data_a["x"], data_a["y"]
    xa, ya = ensure_2d_xy(xa_raw, ya_raw)

    if xa.shape[1] != s or ya.shape[1] != s:
        raise ValueError(f"attacked X size {xa.shape[1]} != original X size {s}")

    # We will not mix here; the caller should pass already-mixed tensors if desired.
    # But to keep CLI simple, we return original tensors to be mixed by caller.

    # Build test split from original
    x_test = xo[-ntest:, :].clone()
    y_test = yo[-ntest:, :].clone()

    # reshape for FNO input
    def to_loader(x2d, y2d, bs, shuffle):
        x_in = x2d.reshape(x2d.shape[0], s, 1).to(dev)  # (N, s, 1)
        y_in = y2d.to(dev)
        ds = TensorDataset(x_in, y_in)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=True)

    # build model
    model = FNO1d(modes, width).to(dev)
    n_params = count_params(model)
    log.append(f"modes={modes}, width={width}, epochs={epochs}, lr={learning_rate}, batch={batch_size}")
    log.append(f"model params = {n_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    iterations = epochs * (ntrain // batch_size)  # cosine steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    myloss = LpLoss(size_average=False)

    # Return pieces needed by outer caller to re-use when mixing per-run
    meta = {
        "dev": dev,
        "s": s,
        "xo": xo, "yo": yo,
        "xa": xa, "ya": ya,
        "x_test": x_test, "y_test": y_test,
        "model": model, "optimizer": optimizer,
        "scheduler": scheduler, "myloss": myloss,
        "batch_size": batch_size, "epochs": epochs,
        "iterations": iterations, "n_params": int(n_params),
        "log": log,
    }
    return meta

def run_train_loop(meta, x_train_2d, y_train_2d, save_dir: Path, run_config: dict):
    dev = meta["dev"]
    s = meta["s"]
    model = meta["model"]
    optimizer = meta["optimizer"]
    scheduler = meta["scheduler"]
    myloss = meta["myloss"]
    batch_size = meta["batch_size"]
    epochs = meta["epochs"]
    iterations = meta["iterations"]
    x_test, y_test = meta["x_test"].to(dev), meta["y_test"].to(dev)
    log = meta["log"]

    train_loader = DataLoader(
        TensorDataset(x_train_2d.reshape(x_train_2d.shape[0], s, 1).to(dev),
                      y_train_2d.to(dev)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader  = DataLoader(
        TensorDataset(x_test.reshape(x_test.shape[0], s, 1), y_test),
        batch_size=batch_size, shuffle=False, drop_last=True
    )

    # training
    for ep in range(epochs):
        model.train()
        t1 = time.time()
        train_mse = 0.0
        train_l2  = 0.0
        nb = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            mse = F.mse_loss(out.view(xb.shape[0], -1), yb.view(xb.shape[0], -1), reduction="mean")
            l2  = myloss(out.view(xb.shape[0], -1), yb.view(xb.shape[0], -1))
            l2.backward()
            optimizer.step()
            scheduler.step()

            train_mse += float(mse.item())
            train_l2  += float(l2.item())
            nb += 1

        # evaluation
        model.eval()
        test_l2 = 0.0
        nb2 = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                test_l2 += float(myloss(out.view(xb.shape[0], -1), yb.view(xb.shape[0], -1)).item())
                nb2 += 1

        train_mse /= max(1, nb)
        train_l2  /= y_train_2d.shape[0]  # LpLoss sum normalized by N
        test_l2   /= y_test.shape[0]

        elapsed = time.time() - t1
        log.append(f"epoch:{ep:04d}  time:{elapsed:.3f}s  train_mse:{train_mse:.6e}  "
                   f"train_l2:{train_l2:.6e}  test_l2:{test_l2:.6e}")

    # save artifacts
    config_to_save = {**run_config, "n_params": meta["n_params"]}
    model_path = save_run_artifacts(save_dir, model, config_to_save, log)
    print(f"[saved] {model_path}")

# ------------------------ CLI & Orchestration ------------------------

def main():
    parser = argparse.ArgumentParser(description="Train FNO1d with attacked/original mixes.")
    parser.add_argument("--pt_original", type=str, required=True,
                        help="Path to original .pt dataset (pos/...)")
    parser.add_argument("--attacks_root", type=str, required=True,
                        help="Path to attacked datasets root (expanded_pos/t1)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["replace50", "add50", "add100", "all"],
                        help="Mixing mode; 'all' runs all three")
    parser.add_argument("--attack_subdir", type=str, default="ALL",
                        help="One subdir under nu0.0005_pgd to use; 'ALL' loops ALLOWED_ATTACK_SUBDIRS")
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # resolve paths relative to 1D_Burgers/
    script_dir = Path(__file__).resolve().parent
    root_1d_burgers = script_dir.parent  # 1D_Burgers/
    save_root = root_1d_burgers / "saved_models_expanded"

    pt_orig = Path(args.pt_original).resolve()
    attacks_root = Path(args.attacks_root).resolve()
    nu_dir = attacks_root / "nu0.0005_pgd"
    if not pt_orig.is_file():
        raise FileNotFoundError(f"Original dataset not found: {pt_orig}")
    if not nu_dir.is_dir():
        raise FileNotFoundError(f"Attacked root missing 'nu0.0005_pgd': {nu_dir}")

    # which attacked subdirs
    subdirs = ALLOWED_ATTACK_SUBDIRS if args.attack_subdir == "ALL" else [args.attack_subdir]
    # expand to Path form under nu0.0005_pgd
    attack_pts = []
    for sd in subdirs:
        try:
            pt = find_attack_pt(attacks_root, sd)
            attack_pts.append((sd, pt))
        except Exception as e:
            print(f"[skip] {sd}: {e}")

    if not attack_pts:
        raise RuntimeError("No attacked datasets found to train on.")

    # which modes
    modes_to_run = ["replace50", "add50", "add100"] if args.mode == "all" else [args.mode]

    # pre-load base objects once (model etc. created per run anyway)
    # but we reuse loading helpers
    # Prepare a base meta with datasets and initialized model; we will re-init for each run
    # (We want a fresh model per run)
    for subdir_rel, pt_attack in attack_pts:
        attack_tag = parse_attack_tag_from_subdir(subdir_rel)
        print(f"\n=== Attacked: {attack_tag} ===")
        # Load once to get xo,yo,xa,ya, etc., and create a *fresh* model for this attacked set
        meta = train_one_mix(
            pt_orig, pt_attack,
            save_dir=Path("."),  # not used here
            ntrain=args.ntrain, ntest=args.ntest,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            modes=args.modes, width=args.width,
            device=args.device,
        )

        xo, yo = meta["xo"], meta["yo"]
        xa, ya = meta["xa"], meta["ya"]

        for mode_name in modes_to_run:
            # Build mixed train set for this mode
            x_tr, y_tr = make_mix((xo, yo), (xa, ya), mode_name, ntrain=args.ntrain)

            # New model per run (re-init everything)
            meta_run = train_one_mix(
                pt_orig, pt_attack,
                save_dir=Path("."),  # not used
                ntrain=args.ntrain, ntest=args.ntest,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                epochs=args.epochs,
                modes=args.modes, width=args.width,
                device=args.device,
            )
            # Reuse created objects and overwrite with our mixed train tensors
            save_dir = build_save_dir(
                root_1d_burgers, args.modes, args.width, args.epochs,
                mode_name=mode_name, attack_tag=attack_tag
            )
            run_cfg = {
                "timestamp": datetime.now().isoformat(),
                "pt_original": str(pt_orig),
                "pt_attack": str(pt_attack),
                "attack_tag": attack_tag,
                "mode": mode_name,
                "ntrain": args.ntrain,
                "ntest": args.ntest,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "modes": args.modes,
                "width": args.width,
                "device": args.device,
            }

            # plug meta_run into loop to actually train & save
            run_train_loop(meta_run, x_tr, y_tr, save_dir, run_cfg)

    print("\nAll trainings finished.\n")

if __name__ == "__main__":
    main()
