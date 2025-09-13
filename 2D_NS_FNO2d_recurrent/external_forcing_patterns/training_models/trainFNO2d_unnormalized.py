#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
from pathlib import Path
from timeit import default_timer
from tqdm import tqdm

import torch
import torch.nn.functional as F

from utilities3 import *           # 提供 LpLoss, count_params 等
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.FNO2d import FNO2d, RecurrentPredictor

################################################################
#  configurations（和你原来保持一致）
################################################################
ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 0.001

modes = 64
width = 60
s = 256
T_in = 10
T = 10
step = 1
epochs_list = [500]              # 如需多组 epoch，可在这里扩展

print("Training FNO2d (multi-pattern loop)")
print(f"{'Variable':<15} {'Value':<10}")
print("-" * 25)
print(f"{'ntrain':<15} {ntrain:<10}")
print(f"{'ntest':<15} {ntest:<10}")
print(f"{'batch_size':<15} {batch_size:<10}")
print(f"{'learning_rate':<15} {learning_rate:<10}")
print(f"{'modes':<15} {modes:<10}")
print(f"{'width':<15} {width:<10}")
print(f"{'s':<15} {s:<10}")
print(f"{'T_in':<15} {T_in:<10}")
print(f"{'T':<15} {T:<10}")
print(f"{'step':<15} {step:<10}")

# ============== 路径配置（使用你给的绝对路径） ==============
BASE_DIR = Path("/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/external_forcing_patterns")
DATASETS_DIR = BASE_DIR / "datasets" / "all_patterns"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"

# ============== 6 个 pattern 的 train .pt 文件名映射（按你提供的精确文件名） ==============
PATTERN_TO_TRAIN_FILE = {
    # "isoCircles": "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_forcingPatternisoCircles.pt",
    # "petals":     "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_forcingPatternpetals.pt",
    # "ringsCos":   "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_forcingPatternringsCos.pt",
    "ringsL1":    "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_forcingPatternringsL1.pt",
    # "ringsLinf":  "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_forcingPatternringsLinf.pt",
    # 注意：你给的 sBands 文件名里是 forcingPatternsBands（多了个 s），这里按原样写
    # "sBands":     "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_forcingPatternsBands.pt",
}

# ============== 设备 & 损失 ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myloss = LpLoss(size_average=False)

def load_xy_train_val(train_pt_path: Path, ntrain: int, nval: int, s_out: int):
    """从 train .pt 中切分出 train / val（与之前一致：[:ntrain] / [-nval:]）。
       下采样比率 sub 依据数据分辨率自动计算。
    """
    data = torch.load(train_pt_path, weights_only=False)

    # 计算下采样因子：优先用 data['x']，否则回退到 data['y']
    if "x" in data:
        nx = data["x"].shape[1]
    else:
        nx = data["y"].shape[1]
    sub = nx // s_out

    # 组装切片（假设 data['y'] 形状为 [N, H, W, T_total]）
    x_train = data['y'][:ntrain, ::sub, ::sub, :T_in]
    y_train = data['y'][:ntrain, ::sub, ::sub, T_in:T_in+T]

    x_val   = data['y'][-nval:, ::sub, ::sub, :T_in]
    y_val   = data['y'][-nval:, ::sub, ::sub, T_in:T_in+T]

    # 显式 reshape，保持 (N, s, s, T_in)
    x_train = x_train.reshape(ntrain, s_out, s_out, T_in)
    x_val   = x_val.reshape(nval,   s_out, s_out, T_in)

    return (x_train, y_train), (x_val, y_val), sub

def make_loaders(x_train, y_train, x_val, y_val, batch_size: int):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val, y_val),
        batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader

def train_one_pattern(pattern: str, train_file: str):
    pattern_dir = DATASETS_DIR / pattern
    train_pt = pattern_dir / train_file
    assert train_pt.exists(), f"[{pattern}] 训练文件不存在：{train_pt}"

    print("\n" + "="*80)
    print(f"Pattern: {pattern}")
    print(f"Train PT: {train_pt}")

    # 数据切分（与之前一致）
    (x_train, y_train), (x_val, y_val), sub = load_xy_train_val(train_pt, ntrain, ntest, s)
    print(f"{'sub':<15} {sub:<10}")
    print(f"[{pattern}] x_train: {tuple(x_train.shape)}, y_train: {tuple(y_train.shape)}")
    print(f"[{pattern}] x_val  : {tuple(x_val.shape)}, y_val  : {tuple(y_val.shape)}")

    # DataLoader
    train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val, batch_size)
    steps_per_epoch = len(train_loader)

    for epochs in epochs_list:
        iterations = epochs * steps_per_epoch
        print(f"{'iterations':<15} {iterations:<10} (epochs={epochs} * steps/epoch={steps_per_epoch})")

        # ============== 模型 & 优化器 & 调度器 ==============
        base_model = FNO2d(modes, modes, width).to(device)
        recurrent_model = RecurrentPredictor(base_model, T_out=T, step=step).to(device)
        model_parameters = count_params(base_model)
        print(f"[{pattern}] params: {model_parameters}")

        optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # CosineAnnealingLR 的 T_max 取“总迭代步数”（按 batch 计），符合官方定义
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        # ============== 日志 / 保存路径 ==============
        save_root = SAVED_MODELS_DIR / pattern / f"modes{modes}_width{width}_epochs{epochs}_Tin{T_in}_T{T}"
        save_root.mkdir(parents=True, exist_ok=True)

        file_tag = f"NS_2d_FNO_trained_on_{train_pt.stem}"
        log_path = save_root / f"{file_tag}.log.txt"
        with open(log_path, "w") as log_f:
            log_f.write("NS 2d FNO training log\n\n")
            log_f.write(f"pattern: {pattern}\n")
            log_f.write(f"training dataset: {train_pt.name}\n")
            log_f.write(f"modes: {modes}\nwidth: {width}\n")
            log_f.write(f"model parameters: {model_parameters}\n")
            log_f.write(f"ntrain={ntrain}, nval={ntest}, batch_size={batch_size}\n")
            log_f.write(f"T_in={T_in}, T_out={T}, step={step}\n")
            log_f.write(f"iterations={iterations}, steps_per_epoch={steps_per_epoch}\n\n")

            # ============== 训练 ==============
            pbar_epochs = tqdm(range(epochs), desc=f"[{pattern}] Training FNO2d Recurrent", ncols=100)
            for ep in pbar_epochs:
                base_model.train()
                t1 = default_timer()

                train_l2 = 0.0
                for xx, yy in train_loader:
                    xx = xx.to(device)   # (B, s, s, T_in)
                    yy = yy.to(device)   # (B, s, s, T)

                    pred = recurrent_model(xx)  # (B, s, s, T)
                    B = pred.size(0)            # 动态 batch 大小，避免最后一个 batch reshape 出错
                    loss = myloss(pred.view(B, -1), yy.view(B, -1))

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    train_l2 += loss.item()

                # 验证
                val_l2 = 0.0
                base_model.eval()
                with torch.no_grad():
                    for xx, yy in val_loader:
                        xx = xx.to(device)
                        yy = yy.to(device)
                        pred = recurrent_model(xx)
                        B = pred.size(0)
                        loss = myloss(pred.view(B, -1), yy.view(B, -1))
                        val_l2 += loss.item()

                t2 = default_timer()
                elapsed = t2 - t1

                log_f.write(f"epoch:{ep}, time:{elapsed:.6f}, train l2:{train_l2:.8f}, val l2:{val_l2:.8f}\n")
                pbar_epochs.set_postfix(train_l2=f"{train_l2:.3e}", val_l2=f"{val_l2:.3e}")

        # 保存权重（state_dict）
        model_path = save_root / f"{file_tag}.pth"
        torch.save(base_model.state_dict(), model_path)
        print(f"[{pattern}] Model saved to: {model_path}")

def main():
    # 依次训练六个 pattern
    for pattern, train_file in PATTERN_TO_TRAIN_FILE.items():
        train_one_pattern(pattern, train_file)

if __name__ == "__main__":
    main()
