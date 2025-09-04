# -*- coding: utf-8 -*-
import argparse
import json
import random
from pathlib import Path
from timeit import default_timer

import torch
from tqdm import tqdm

from utilities3 import *                 # LpLoss, count_params, etc.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO2d import FNO2d, RecurrentPredictor

# ===========================
# Global config / constants
# ===========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 固定基准训练/测试规模
BASE_NTRAIN = 1000
BASE_NTEST  = 100

# 模型/数据几何
MODES = 96
WIDTH = 80
S = 256
T_IN = 10
T_OUT = 10
STEP = 1

# ===========================
# Utilities
# ===========================
def set_all_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_y_tensor(pt_path: Path) -> torch.Tensor:
    d = torch.load(pt_path, weights_only=False)
    y = d["y"]
    return y.cpu() if y.is_cuda else y

def sample_from_single_file(pt_path: Path, k: int, seed: int) -> dict:
    """
    从单一 expanded 文件里抽取 k 条样本。
    - 若 k <= N：无放回随机采样
    - 若 k > N：有放回采样，保证恰好取到 k 条
    返回:
      {
        "selected": Tensor[k, H, W, T],
        "manifest": {...}
      }
    """
    y = load_y_tensor(pt_path)  # [N, H, W, T]
    N = y.shape[0]
    g = torch.Generator(device='cpu').manual_seed(seed)
    if k <= 0:
        idx = torch.empty((0,), dtype=torch.long)
        sel = torch.empty((0, *y.shape[1:]), dtype=y.dtype)
        with_repl = False
    elif k <= N:
        idx = torch.randperm(N, generator=g)[:k]
        sel = y[idx]
        with_repl = False
    else:
        idx = torch.randint(low=0, high=N, size=(k,), generator=g)
        sel = y[idx]
        with_repl = True
    return {
        "selected": sel,
        "manifest": {
            "file": str(pt_path),
            "k_requested": int(k),
            "N_in_file": int(N),
            "indices": idx.tolist(),
            "with_replacement": with_repl
        }
    }

def combine_and_shuffle(y_list, seed: int):
    y = torch.cat(y_list, dim=0)
    g = torch.Generator(device='cpu').manual_seed(seed)
    perm = torch.randperm(y.shape[0], generator=g)
    return y[perm]

def split_idx(total_n: int, ntrain: int, ntest: int, seed: int):
    if ntrain + ntest > total_n:
        raise ValueError(f"Split {ntrain}+{ntest}>{total_n}")
    g = torch.Generator(device='cpu').manual_seed(seed)
    perm = torch.randperm(total_n, generator=g)
    return perm[:ntrain], perm[ntrain:ntrain+ntest]

def make_loaders_from_y(y_all: torch.Tensor, ntrain: int, ntest: int, s: int,
                        T_in: int, T_out: int, batch_size: int, seed: int):
    """
    y_all: [N, H, W, T_total]
    """
    H = y_all.shape[1]
    sub = H // s
    if sub <= 0:
        raise ValueError(f"Invalid sub-sampling: H={H}, s={s}")

    idx_tr, idx_te = split_idx(y_all.shape[0], ntrain, ntest, seed)

    y_tr = y_all[idx_tr][:, ::sub, ::sub, :]
    y_te = y_all[idx_te][:, ::sub, ::sub, :]

    x_tr, t_tr = y_tr[..., :T_in], y_tr[..., T_in:T_in+T_out]
    x_te, t_te = y_te[..., :T_in], y_te[..., T_in:T_in+T_out]

    x_tr = x_tr.reshape(ntrain, s, s, T_in)
    x_te = x_te.reshape(ntest,  s, s, T_in)

    train_ds = torch.utils.data.TensorDataset(x_tr, t_tr)
    test_ds  = torch.utils.data.TensorDataset(x_te, t_te)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, sub

def build_model():
    fno = FNO2d(MODES, MODES, WIDTH).to(DEVICE)
    rec = RecurrentPredictor(fno, T_out=T_OUT, step=STEP).to(DEVICE)
    return fno, rec

def train_once(train_loader, test_loader, iterations: int, epochs: int, run_log_file: Path, run_tag: str):
    fno, rec = build_model()
    params = count_params(fno)
    optimizer = torch.optim.Adam(fno.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations if iterations > 0 else 1)
    myloss = LpLoss(size_average=False)

    with open(run_log_file, "a") as lf:
        lf.write(f"model parameters: {params}\n")

    for ep in tqdm(range(epochs), desc=f"Training FNO2d [{run_tag}]"):
        fno.train()
        t1 = default_timer()
        train_l2 = 0.0
        for xx, yy in train_loader:
            xx, yy = xx.to(DEVICE), yy.to(DEVICE)
            pred = rec(xx)
            bs = pred.size(0)
            loss = myloss(pred.view(bs, -1), yy.view(bs, -1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_l2 += float(loss.item())

        test_l2 = 0.0
        with torch.no_grad():
            fno.eval()
            for xx, yy in test_loader:
                xx, yy = xx.to(DEVICE), yy.to(DEVICE)
                pred = rec(xx)
                bs = pred.size(0)
                loss = myloss(pred.view(bs, -1), yy.view(bs, -1))
                test_l2 += float(loss.item())

        t2 = default_timer()
        with open(run_log_file, "a") as lf:
            lf.write(f"epoch:{ep}, time:{t2-t1:.6f}, train l2:{train_l2:.8f}, test l2:{test_l2:.8f}\n")

    return fno

def print_config(title: str, cfg: dict):
    print("\n" + "="*16 + f" {title} " + "="*16)
    for k, v in cfg.items():
        print(f"{k:<26}: {v}")
    print("="*54 + "\n")

# ===========================
# Main
# ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded_file", type=str, required=True, help="单个新数据 .pt 文件路径")
    ap.add_argument("--percent", type=float, required=True, help="如 0.25 表示 25%")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_all_seeds(args.seed)

    # 以此脚本所在路径向上两级，定位到 2D_NS_FNO2d_recurrent/
    current_file_path = Path(__file__).resolve().parent.parent

    # 原始数据路径
    dataset_name = "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pt"
    orig_path = current_file_path / "datasets" / "exponax_datasets" / "t20" / dataset_name
    if not orig_path.exists():
        raise FileNotFoundError(f"Original dataset not found: {orig_path}")

    # 单个扩展文件（由 shell 传入）
    exp_path = Path(args.expanded_file).resolve()
    if not exp_path.exists():
        raise FileNotFoundError(exp_path)

    expanded_stem = exp_path.stem
    expanded_name = exp_path.name
    orig_name = orig_path.name
    pct_display = f"{args.percent:.2%}"   # 如 "25.00%"

    # 保存根目录：.../saved_models_expanded/2D/<expanded_stem>/<XXpct>/{add|replace}/...
    save_root = current_file_path / "saved_models_expanded" / "2D"
    pct_tag = f"{int(round(args.percent*100))}pct"
    base_dir = save_root / expanded_stem / pct_tag

    # 预先构造 ADD / REPLACE 目录与目标 ckpt 路径
    add_dir = base_dir / "add" / f"modes{MODES}_width{WIDTH}_epochs{args.epochs}_Tin{T_IN}_T{T_OUT}"
    rep_dir = base_dir / "replace" / f"modes{MODES}__width{WIDTH}_epochs{args.epochs}_Tin{T_IN}_T{T_OUT}"
    add_ckpt = add_dir / f"NS_2d_FNO_model_trainedby_{orig_path.stem}.pth"
    rep_ckpt = rep_dir / f"NS_2d_FNO_model_trainedby_{orig_path.stem}.pth"

    need_add = not add_ckpt.exists()
    need_rep = not rep_ckpt.exists()

    # 若都已存在，直接退出
    if not need_add and not need_rep:
        print(f"[SKIP] Both targets exist. ADD: {add_ckpt} | REPLACE: {rep_ckpt}. Nothing to do.")
        return

    # 需要训练任意一路时再加载原始数据，以免不必要的 IO
    data = torch.load(orig_path, weights_only=False)
    y_orig = data["y"].cpu()
    N0 = y_orig.shape[0]

    # ===========================
    # 1) ADD（仅在缺失时训练）
    # ===========================
    if need_add:
        add_dir.mkdir(parents=True, exist_ok=True)
        k_add = int(round(args.percent * N0))
        pick_add = sample_from_single_file(exp_path, k_add, seed=args.seed + 11)
        y_aug = combine_and_shuffle([y_orig, pick_add["selected"]], seed=args.seed + 21)

        ntrain_aug = int(round(BASE_NTRAIN * (1.0 + args.percent)))
        ntest_aug  = int(round(BASE_NTEST  * (1.0 + args.percent)))
        iterations_aug = args.epochs * max(1, (ntrain_aug // args.batch_size))

        train_loader, test_loader, sub = make_loaders_from_y(
            y_aug, ntrain_aug, ntest_aug, S, T_IN, T_OUT, args.batch_size, seed=args.seed + 31
        )

        log_add = add_dir / f"NS_2d_FNO_log_trainedby_{orig_path.stem}.txt"
        cfg_add = {
            "mode": "ADD (添加)",
            "expanded_dataset_file": expanded_name,
            "percent_added": pct_display,
            "samples_added_vs_total": f"{k_add} / {N0}",
            "original_dataset_file": orig_name,
            "total_samples_after_merge": int(y_aug.shape[0]),
            "ntrain": ntrain_aug,
            "ntest": ntest_aug,
            "modes": MODES,
            "width": WIDTH,
            "s": S,
            "T_in": T_IN,
            "T_out": T_OUT,
            "step": STEP,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "iterations": iterations_aug,
            "subsample_factor": sub,
            "device": str(DEVICE),
            "seed": args.seed,
        }
        print_config("ADD RUN CONFIG", cfg_add)

        with open(log_add, "w") as lf:
            lf.write("NS 2d FNO training log (ADD)\n\n")
            json.dump({"config": cfg_add}, lf, indent=2)
            lf.write("\n\n")
        with open(add_dir / "manifest_add.json", "w") as mf:
            json.dump(pick_add["manifest"], mf, indent=2)

        model_add = train_once(train_loader, test_loader, iterations_aug, args.epochs, log_add, "ADD")
        torch.save(model_add.state_dict(), add_ckpt)
        print(f"[SAVED][ADD] {add_ckpt}")
    else:
        print(f"[SKIP][ADD] Found existing model: {add_ckpt} — skip training.")

    # ===========================
    # 2) REPLACE（仅在缺失时训练）
    # ===========================
    if need_rep:
        rep_dir.mkdir(parents=True, exist_ok=True)
        k_rep = int(round(args.percent * N0))
        pick_rep = sample_from_single_file(exp_path, k_rep, seed=args.seed + 41)

        y_replaced = y_orig.clone()
        if k_rep > 0:
            g = torch.Generator(device='cpu').manual_seed(args.seed + 51)
            idx_orig = torch.randperm(N0, generator=g)[:k_rep]
            y_replaced[idx_orig] = pick_rep["selected"]

        y_replaced = combine_and_shuffle([y_replaced], seed=args.seed + 61)

        ntrain_rep, ntest_rep = BASE_NTRAIN, BASE_NTEST
        iterations_rep = args.epochs * max(1, (ntrain_rep // args.batch_size))

        tr_r, te_r, sub_r = make_loaders_from_y(
            y_replaced, ntrain_rep, ntest_rep, S, T_IN, T_OUT, args.batch_size, seed=args.seed + 71
        )

        log_rep = rep_dir / f"NS_2d_FNO_log_trainedby_{orig_path.stem}.txt"
        cfg_rep = {
            "mode": "REPLACE (替换)",
            "expanded_dataset_file": expanded_name,
            "percent_replaced": pct_display,
            "samples_replaced_vs_total": f"{k_rep} / {N0}",
            "original_dataset_file": orig_name,
            "total_samples_after_replace": int(y_replaced.shape[0]),
            "ntrain": ntrain_rep,
            "ntest": ntest_rep,
            "modes": MODES,
            "width": WIDTH,
            "s": S,
            "T_in": T_IN,
            "T_out": T_OUT,
            "step": STEP,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "iterations": iterations_rep,
            "subsample_factor": sub_r,
            "device": str(DEVICE),
            "seed": args.seed,
        }
        print_config("REPLACE RUN CONFIG", cfg_rep)

        with open(log_rep, "w") as lf:
            lf.write("NS 2d FNO training log (REPLACE)\n\n")
            json.dump({"config": cfg_rep}, lf, indent=2)
            lf.write("\n\n")
        with open(rep_dir / "manifest_replace.json", "w") as mf:
            json.dump(pick_rep["manifest"], mf, indent=2)

        model_rep = train_once(tr_r, te_r, iterations_rep, args.epochs, log_rep, "REPLACE")
        torch.save(model_rep.state_dict(), rep_ckpt)
        print(f"[SAVED][REPLACE] {rep_ckpt}")
    else:
        print(f"[SKIP][REPLACE] Found existing model: {rep_ckpt} — skip training.")

if __name__ == "__main__":
    main()


