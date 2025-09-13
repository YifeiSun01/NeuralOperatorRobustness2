# -*- coding: utf-8 -*-
import argparse
import json
import math
import random
import gc
from pathlib import Path
from timeit import default_timer

import torch
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
from tqdm import tqdm

from utilities3 import *  # LpLoss, count_params, etc.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO2d import FNO2d, RecurrentPredictor

# ===========================
# Global config / constants
# ===========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型/数据几何
MODES = 96
WIDTH = 80
S = 256
T_IN = 10
T_OUT = 10
STEP = 1

# 构建一个固定的小测试集规模（跨多文件按比例抽样）
DEFAULT_NTEST = 100

# ===========================
# Utilities
# ===========================
def set_all_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_pt_files(root: Path):
    return sorted([p for p in root.rglob("*.pt") if p.is_file()])

@torch.no_grad()
def load_y_tensor_meta(pt_path: Path):
    """
    仅获取形状/句柄，并尽量用 mmap 映射，避免把整个 6GB 直接读入 RAM。
    """
    d = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)
    y = d["y"]
    return y  # CPU tensor（可能由 OS 按需分页加载）

def subsample_spatial(y: torch.Tensor, s: int):
    """
    y: [N, H, W, T_total] --> 下采样到 [N, s, s, T_total]
    """
    H = y.shape[1]
    sub = H // s
    if sub <= 0:
        raise ValueError(f"Invalid sub-sampling: H={H}, s={s}")
    return y[:, ::sub, ::sub, :], sub

def print_config(title: str, cfg: dict):
    print("\n" + "="*16 + f" {title} " + "="*16)
    for k, v in cfg.items():
        print(f"{k:<28}: {v}")
    print("="*60 + "\n")

# ===========================
# IterableDataset（流式多文件）
# ===========================
class MultiPTStream(IterableDataset):
    """
    - 递归找到的多份 .pt 数据，每次只打开一个文件，遍历完释放；
    - 每个文件内部打乱样本顺序；文件顺序也按 epoch+seed 打乱；
    - 支持 verbose：切换文件时打印“当前正在运行哪个数据集”。
    """
    def __init__(self, files, s, T_in, T_out, seed=1234, shuffle_files=True, shuffle_within=True, verbose=True):
        super().__init__()
        self.files = list(files)
        self.s = s
        self.T_in = T_in
        self.T_out = T_out
        self.seed = int(seed)
        self.shuffle_files = shuffle_files
        self.shuffle_within = shuffle_within
        self.verbose = verbose
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __iter__(self):
        # 基于 epoch+seed 打乱文件顺序
        files_order = list(self.files)
        rng = random.Random(self.seed * 10007 + self._epoch)
        if self.shuffle_files:
            rng.shuffle(files_order)

        total_files = len(files_order)

        for idx_file, f in enumerate(files_order, 1):
            # 1) 打开单个文件（mmap）
            y = load_y_tensor_meta(Path(f))  # CPU tensor, OS 按需分页
            N, H, W, Ttot = map(int, y.shape)

            if self.verbose:
                tqdm.write(f"[Epoch {self._epoch:04d}] ▶ START file {idx_file}/{total_files}: {f} | N={N} H={H} W={W} T={Ttot}")

            # 2) 下采样到 (N, s, s, T)
            y_s, _ = subsample_spatial(y, self.s)
            N_sub = y_s.shape[0]

            # 3) 索引顺序（文件内打乱）
            if self.shuffle_within:
                g = torch.Generator(device='cpu').manual_seed(self.seed ^ (hash(f) & 0xFFFFFFFF) ^ self._epoch)
                indices = torch.randperm(N_sub, generator=g)
            else:
                indices = torch.arange(N_sub)

            # 4) 逐样本产出，交给 DataLoader 组 batch
            for i in indices.tolist():
                xi = y_s[i, :, :, :self.T_in]                      # [s, s, T_in]
                yi = y_s[i, :, :, self.T_in:self.T_in+self.T_out]  # [s, s, T_out]
                yield xi, yi

            # 5) 释放当前文件张量，避免常驻内存
            del y_s, y
            gc.collect()

            if self.verbose:
                tqdm.write(f"[Epoch {self._epoch:04d}] ■ DONE  file {idx_file}/{total_files}: {f}")

# ===========================
# Test set 构建（跨文件小规模抽样）
# ===========================
@torch.no_grad()
def build_small_test_loader(files, ntest, s, T_in, T_out, seed=2025, batch_size=20):
    if ntest <= 0:
        return None

    # 先收集每个文件的 N
    infos = []
    totalN = 0
    for f in files:
        y = load_y_tensor_meta(Path(f))
        N = int(y.shape[0])
        infos.append({"file": str(f), "N": N})
        totalN += N

    # 按比例分配每个文件应抽的样本数（至少 1；但不超过该文件 N）
    picks = []
    remain = ntest
    for i, it in enumerate(infos):
        if i == len(infos) - 1:
            ki = remain
        else:
            ki = max(1, round(ntest * it["N"] / max(1, totalN)))
            ki = min(ki, it["N"])
            remain -= ki

        if ki <= 0:
            continue

        y = load_y_tensor_meta(Path(it["file"]))
        y_s, _ = subsample_spatial(y, s)
        N = y_s.shape[0]
        g = torch.Generator(device='cpu').manual_seed(seed ^ (hash(it["file"]) & 0xFFFFFFFF))
        idx = torch.randperm(N, generator=g)[:ki]
        xs = y_s[idx, :, :, :T_in]
        ys = y_s[idx, :, :, T_in:T_in+T_out]
        picks.append((xs, ys))
        del y_s, y
        gc.collect()

    if not picks:
        return None

    x_all = torch.cat([p[0] for p in picks], dim=0)
    y_all = torch.cat([p[1] for p in picks], dim=0)

    ds = TensorDataset(x_all, y_all)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return dl

# ===========================
# Model
# ===========================
def build_model():
    fno = FNO2d(MODES, MODES, WIDTH).to(DEVICE)
    rec = RecurrentPredictor(fno, T_out=T_OUT, step=STEP).to(DEVICE)
    return fno, rec

# ===========================
# Train loop
# ===========================
def train_all_merged(train_loader, test_loader, epochs, run_log_file: Path,
                     ckpt_dir: Path, ckpt_every: int, run_tag: str, steps_per_epoch: int):

    fno, rec = build_model()
    params = count_params(fno)
    optimizer = torch.optim.Adam(fno.parameters(), lr=1e-3, weight_decay=1e-4)

    # 方便余弦调度：用「估计的 steps_per_epoch」× epochs
    total_iters = max(1, steps_per_epoch * max(1, epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)
    myloss = LpLoss(size_average=False)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(run_log_file, "a") as lf:
        lf.write(f"model parameters: {params}\n")
        lf.flush()

    for ep in range(epochs):
        # 若底层 dataset 支持 epoch（MultiPTStream 支持），通知一下
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(ep)

        fno.train()
        t1 = default_timer()
        train_l2 = 0.0

        for xx, yy in train_loader:
            xx, yy = xx.to(DEVICE, non_blocking=True), yy.to(DEVICE, non_blocking=True)
            pred = rec(xx)
            bs = pred.size(0)
            loss = myloss(pred.view(bs, -1), yy.view(bs, -1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_l2 += float(loss.item())

        # 验证
        fno.eval()
        test_l2 = 0.0
        if test_loader is not None:
            with torch.no_grad():
                for xx, yy in test_loader:
                    xx, yy = xx.to(DEVICE, non_blocking=True), yy.to(DEVICE, non_blocking=True)
                    pred = rec(xx)
                    bs = pred.size(0)
                    loss = myloss(pred.view(bs, -1), yy.view(bs, -1))
                    test_l2 += float(loss.item())
        else:
            test_l2 = float('nan')

        # 统计与打印
        t2 = default_timer()
        line = (
            f"epoch:{ep+1}/{epochs}, time:{t2-t1:.6f}s, "
            f"train l2:{train_l2:.8f}, test l2:{'N/A' if math.isnan(test_l2) else f'{test_l2:.8f}'}"
        )
        print(line)
        with open(run_log_file, "a") as lf:
            lf.write(line + "\n")
            lf.flush()

        # 每 ckpt_every 个 epoch 保存一次
        if (ep + 1) % ckpt_every == 0 or ep == epochs - 1:
            ckpt_path = ckpt_dir / f"epoch_{ep+1:04d}.pth"
            torch.save(fno.state_dict(), ckpt_path)
            torch.save(fno.state_dict(), ckpt_dir / "latest.pth")

    return fno

# ===========================
# Main
# ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded_dir", type=str, required=True, help="包含多个 .pt 的目录（递归检索）")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--ntest", type=int, default=DEFAULT_NTEST)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=0)  # 大数据建议先用 0，避免多进程多份拷贝
    ap.add_argument("--verbose_file_switch", action="store_true", default=True,
                    help="切换到新的 .pt 文件时打印开始/结束信息")
    args = ap.parse_args()

    set_all_seeds(args.seed)

    # 以此脚本所在路径向上两级，定位到 2D_NS_FNO2d_recurrent/
    current_file_path = Path(__file__).resolve().parent.parent

    # 原始 train 数据（用于记录到日志；真正训练只看 expanded_dir）
    dataset_name = "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pt"
    orig_path = current_file_path / "datasets" / "exponax_datasets" / "t20" / dataset_name
    if not orig_path.exists():
        print(f"[WARN] Original dataset not found: {orig_path} (仅用于记录，不影响训练)")

    # 递归找到全部 .pt
    exp_root = Path(args.expanded_dir).resolve()
    files = list_pt_files(exp_root)
    if not files:
        raise FileNotFoundError(f"No .pt files found under: {exp_root}")

    # === 提前准备保存目录与日志文件（后面要写日志/清单） ===
    save_root = current_file_path / "saved_models_expanded" / "2D_ALLMERGE"
    base_dir = save_root / f"merged_{exp_root.name}" / f"modes{MODES}_width{WIDTH}_epochs{args.epochs}_Tin{T_IN}_T{T_OUT}_S{S}"
    base_dir.mkdir(parents=True, exist_ok=True)
    log_file = base_dir / "train_log.txt"
    ckpt_dir = base_dir / "checkpoints"

    # 收集信息并估算总样本数与 steps/epoch
    file_infos = []
    totalN = 0
    for f in files:
        y = load_y_tensor_meta(f)
        N, H, W, Ttot = map(int, y.shape)
        file_infos.append({"file": str(f), "N": N, "H": H, "W": W, "T_total": Ttot})
        totalN += N
        del y
        gc.collect()

    # 全量清单：打印 + 记录到日志
    print("\n[FOUND DATASETS — full list]")
    for i, info in enumerate(file_infos, 1):
        print(f"[{i:04d}] {info['file']} | N={info['N']} H={info['H']} W={info['W']} T_total={info['T_total']}")

    with open(log_file, "a") as lf:
        lf.write("\n[FOUND DATASETS — full list]\n")
        for i, info in enumerate(file_infos, 1):
            lf.write(f"[{i:04d}] {info['file']} | N={info['N']} H={info['H']} W={info['W']} T_total={info['T_total']}\n")

    # 训练数据（流式）
    train_ds = MultiPTStream(
        files=files, s=S, T_in=T_IN, T_out=T_OUT, seed=args.seed,
        shuffle_files=True, shuffle_within=True, verbose=args.verbose_file_switch
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False
    )

    # 估算每个 epoch 的 step 数（遍历所有文件所有样本 / batch_size）
    steps_per_epoch = math.ceil(totalN / max(1, args.batch_size))

    # 测试集（小而固定）
    test_loader = build_small_test_loader(
        files=files, ntest=args.ntest, s=S, T_in=T_IN, T_out=T_OUT,
        seed=args.seed + 999, batch_size=args.batch_size
    )

    # 打印/记录配置与数据清单
    cfg = {
        "mode": "ALL-MERGED-STREAM",
        "expanded_root": str(exp_root),
        "num_pt_files": len(files),
        "total_samples_N": int(totalN),
        "modes": MODES, "width": WIDTH, "s": S,
        "T_in": T_IN, "T_out": T_OUT, "step": STEP,
        "batch_size": args.batch_size, "epochs": args.epochs,
        "ckpt_every": args.ckpt_every, "device": str(DEVICE),
        "seed": args.seed, "num_workers": args.num_workers,
        "verbose_file_switch": bool(args.verbose_file_switch),
    }
    print_config("ALL-MERGED RUN CONFIG", cfg)
    with open(base_dir / "found_datasets.json", "w") as f:
        json.dump({"config": cfg, "files": file_infos}, f, indent=2)

    # 训练
    model = train_all_merged(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        run_log_file=log_file,
        ckpt_dir=ckpt_dir,
        ckpt_every=args.ckpt_every,
        run_tag="ALLMERGE",
        steps_per_epoch=steps_per_epoch,
    )

    # 最终保存（收尾）
    torch.save(model.state_dict(), base_dir / f"final.pth")
    print(f"[SAVED] final model -> {base_dir/'final.pth'}")

if __name__ == "__main__":
    main()




