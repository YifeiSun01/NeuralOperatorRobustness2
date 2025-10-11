# -*- coding: utf-8 -*-
"""
Adversarial-Continued Training with global epoch numbering.

- 自动识别预训练权重的起始 epoch (base_epoch)：
  1) 优先读取 checkpoint/state 中的 'epoch'
  2) 若无，则从文件名匹配 'epoch####'
  3) 均无则认为是 0（全新或未知轮次）
- 训练期间所有保存的 .pt/.pth 文件名使用 global_epoch = base_epoch + ep
- 保存的 .pt/.pth 内容字典里也写入 "epoch": global_epoch
- 兼容 DataParallel 保存的 state_dict（键名带 'module.'）

参考：
- PyTorch saving/loading state_dict & checkpoints
- CosineAnnealingLR: T_max/last_epoch 的语义（如需严格续训，可设置 last_epoch）
"""

from __future__ import annotations
import os
import re
import math
import argparse
from pathlib import Path
from timeit import default_timer

import torch
from tqdm import tqdm

# ====== 你的工程内模块 ======
from PGD_attack_batch import attack_and_rollout_seq, PGDAdamConfig
from FNO2d import FNO2d, RecurrentPredictor
from utilities3 import *
from solver import DifferentiablePDESolver

# ====== JAX 缓存（PDE 求解器用到 JAX 时提升 warmup 后复用效率）======
import jax
jax.config.update("jax_compilation_cache_dir", "/blue/shiboli.fsu/yifeisun.umich/.jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# 非本地文件系统时，JAX 文档建议安装 etils: pip install etils

# =========================
# 配置区（按需调整）
# =========================
modes = 96
width = 80
epochs = 500
T_in = 10
T = 10
step = 1

ntrain = 1000
ntest = 100

batch_size = 2
learning_rate = 0.001

current_file_path = Path(__file__).resolve()

# ====== 设备与预训练模型路径（按需调整）======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 示例1：从“原始、全新模型”开始（没有 epoch 信息）
pretrained_model_name = "NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames"
pretrained_path = current_file_path.parent / f"original_model/{pretrained_model_name}.pth"

# 示例2：从中途 epoch 快照续训（文件名里含 epoch####）
# pretrained_model_name = "NS_2d_FNO_model_epoch0010"
# pretrained_path = current_file_path.parent / f"saved_models/modes96_width80_epochs2000_Tin10_T10_batch2_2/{pretrained_model_name}.pth"

dataset_name = "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames"
dataset_path = current_file_path.parent / f"dataset/{dataset_name}.pt"

# ====== PDE solver ======
pde_solver = DifferentiablePDESolver(nu=1e-5, device=device)

# ====== 攻击配置（与你之前缩放一致）======
size = 256
attack_cfg = PGDAdamConfig(
    epsilon = 0.0080 * (size * size),
    alpha   = 100.0,
    num_steps = 1,
    norm = 2,
    mode_spec = "wwwwwwwwww",
    beta1=0.9, beta2=0.999, adam_eps=1e-8, amsgrad=False, use_sign_for_linf=True,
    # ---- new: enable uniform scalar shift instead of PGD ----
    use_uniform_shift = True,
    uniform_shift_range = (-3.0, 3.0)
)

# =========================
# 工具函数：epoch 识别与 state_dict 适配
# =========================
def infer_base_epoch(pretrained_path: Path, loaded_state) -> int:
    """
    解析预训练模型的起始 epoch (base_epoch)：
    1) 若 checkpoint/state 中存在 'epoch' 字段，优先使用
    2) 否则，从文件名匹配 'epoch####'
    3) 都没有则 0
    """
    # 1) checkpoint/state 中的 'epoch'
    try:
        if isinstance(loaded_state, dict) and 'epoch' in loaded_state:
            val = loaded_state['epoch']
            # 兼容 float/int
            return int(val)
        # 若是标准 checkpoint 结构
        if isinstance(loaded_state, dict) and 'state_dict' in loaded_state and 'epoch' in loaded_state:
            return int(loaded_state['epoch'])
    except Exception:
        pass

    # 2) 文件名中匹配 epoch####（如 NS_2d_FNO_model_epoch0030.pth）
    m = re.search(r'epoch(\d+)', str(pretrained_path))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    # 3) 默认
    return 0

def extract_state_dict_maybe(state):
    """
    - 若 state 是“裸”的 state_dict，直接返回
    - 若 state 是 checkpoint（含 'state_dict'），返回其中的 state_dict
    """
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        return state['state_dict']
    return state

def try_load_state_dict(model: torch.nn.Module, state_dict: dict):
    """
    尝试多种方式装载（处理 DataParallel 的 'module.' 前缀差异）：
    1) 直接 load
    2) 剥离 'module.' 前缀再 load
    3) 给 keys 添加 'module.' 前缀再 load
    若仍失败则抛出原始异常
    """
    try:
        model.load_state_dict(state_dict)
        return
    except Exception as e_direct:
        # 尝试去掉 'module.' 前缀
        try:
            stripped = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    stripped[k[len('module.'):]] = v
                else:
                    stripped[k] = v
            model.load_state_dict(stripped)
            return
        except Exception as e_strip:
            # 尝试添加 'module.' 前缀
            try:
                added = {}
                for k, v in state_dict.items():
                    if not k.startswith('module.'):
                        added['module.' + k] = v
                    else:
                        added[k] = v
                model.load_state_dict(added)
                return
            except Exception as e_add:
                # 三种方式均失败：抛出最初的错误
                raise e_direct

def describe_pretrained_status(base_epoch: int) -> str:
    if base_epoch > 0:
        return f"已训练到 epoch={base_epoch}，本次将从该全局轮次继续。"
    else:
        return "未检测到已训练轮次（可能是全新模型或未知），本次按全新开始（base_epoch=0）。"

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# =========================
# 打印参数/路径/设备/攻击配置
# =========================
print("=" * 40 + " 参数配置 " + "=" * 40)
print(f"modes={modes}, width={width}, epochs={epochs}")
print(f"T_in={T_in}, T={T}, step={step}")
print(f"ntrain={ntrain}, ntest={ntest}")
print(f"batch_size={batch_size}, learning_rate={learning_rate}")
print()

print("=" * 40 + " 路径配置 " + "=" * 40)
print(f"current_file_path={current_file_path}")
print(f"pretrained_model_name={pretrained_model_name}")
print(f"pretrained_path={pretrained_path}")
print(f"dataset_name={dataset_name}")
print(f"dataset_path={dataset_path}")
print()

print("=" * 40 + " 设备与PDE solver " + "=" * 40)
print(f"device={device}")
print(f"pde_solver={pde_solver}")
print()

print("=" * 40 + " 攻击配置 " + "=" * 40)
print(attack_cfg)
print()

# =========================
# 构建模型 & 加载预训练
# =========================
model = FNO2d(modes, modes, width, in_channels=T_in).to(device)
recurrent_model = RecurrentPredictor(model, T_out=T, step=step).to(device)

# assert pretrained_path.exists(), f"预训练权重不存在: {pretrained_path}"
state_any = torch.load(pretrained_path, map_location=device, weights_only=False)

# 推断 base_epoch
base_epoch = infer_base_epoch(pretrained_path, state_any)

# 取出可能的 state_dict 并尝试多路加载（兼容 DataParallel）
state_dict = extract_state_dict_maybe(state_any)
try_load_state_dict(model, state_dict)

# ==== 这里打印本次到底加载了什么 + 从哪里开始 epoch ====
print("="*40 + " 预训练加载状态 " + "="*40)
print(f"Loaded pretrained weights from: {pretrained_path}")
print(f"模型起始（base_epoch）= {base_epoch}")
print("判定：", describe_pretrained_status(base_epoch))
print("（训练期间所有保存将使用 global_epoch = base_epoch + ep 命名）")
print()

model_parameters = count_params(model)
print("params:", model_parameters)

optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# 注：严格意义的“续训 scheduler”可设置 last_epoch；此处不改动你的原逻辑
iterations = epochs * (ntrain // batch_size)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
myloss     = LpLoss(size_average=False)

# ====== 路径 & 日志 ======
snap_dir = current_file_path.parent / f"saved_models/modes{modes}_width{width}_epochs{epochs}_Tin{T_in}_T{T}_batch{batch_size}_2"
snap_dir.mkdir(parents=True, exist_ok=True)
log_save_path = snap_dir / f"{pretrained_model_name}.txt"
log_file = open(log_save_path, "a", buffering=1)
log_file.write("\n\n=== Adversarial-Continued Training (attack first frame + PDE labels) ===\n")
log_file.write(f"[resume_from] {pretrained_path}\n")
log_file.write(f"[base_epoch] {base_epoch}\n")

# ====== 数据 ======
data = torch.load(dataset_path, map_location=device, weights_only=False)

sub = data["x"].shape[1] // size
x_train = data['y'][:ntrain,::sub,::sub,:T_in].contiguous().to(device)
y_train = data['y'][:ntrain,::sub,::sub,T_in:T+T_in].contiguous().to(device)
x_test  = data['y'][-ntest:,::sub,::sub,:T_in].contiguous().to(device)
y_test  = data['y'][-ntest:,::sub,::sub,T_in:T+T_in].contiguous().to(device)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader  = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=batch_size, shuffle=False, drop_last=False
)

# ====== 训练循环（每个 batch：只攻第0帧 + PDE 生成 Tin/Tout）======
first_train_batch_cache = None  # 缓存“干净 batch”用于日志/快照（会用新 API 生成 adv 与标签）

# ---- 预热（无PGD、无backward）----
with torch.no_grad():
    # 1) 触发 JAX jit + cuFFT 计划（形状与训练一致）
    dummy = torch.zeros((batch_size, size, size), device=device, dtype=torch.float32)
    _ = pde_solver.rollout_seconds(dummy, T_seconds=T_in + T - 1)

    # 2) 触发一次 FNO+Recurrent 的卷积/前向内核加载
    dummy_xx = torch.zeros((batch_size, size, size, T_in), device=device, dtype=torch.float32)
    _ = recurrent_model(dummy_xx)

for ep in tqdm(range(epochs), desc="Adversarial Continued Training"):
    global_epoch = base_epoch + ep  # 全局 epoch（续训友好）

    model.train()
    epoch_t0 = default_timer()

    perturb_time = 0.0
    train_time   = 0.0
    eval_time    = 0.0
    train_l2     = 0.0

    # 缓存第一个 batch（干净，用于周期性日志/快照）
    if first_train_batch_cache is None:
        with torch.no_grad():
            xx0, yy0 = next(iter(train_loader))
            first_train_batch_cache = (xx0.to(device), yy0.to(device))

    # ---- train pass ----
    num_batches = len(train_loader)
    for batch_idx, (xx, yy) in enumerate(train_loader, start=1):
        xx = xx.to(device)  # (B,s,s,T_in)
        # yy (loader 给的) 不再用作训练标签，我们用 PDE rollout 的 yy

        # 1) 对抗 + PDE 标签
        _sync()
        t0 = default_timer()
        adv_xx, yy_pde = attack_and_rollout_seq(xx, recurrent_model, pde_solver, Tout=T, cfg=attack_cfg)
        _sync()
        per_batch_perturb = default_timer() - t0
        perturb_time += per_batch_perturb

        # 2) 训练步
        _sync()
        t1 = default_timer()
        optimizer.zero_grad(set_to_none=True)
        pred = recurrent_model(adv_xx)                     # (B,s,s,T)
        loss = myloss(pred.reshape(pred.shape[0], -1), yy_pde.reshape(yy_pde.shape[0], -1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        _sync()
        per_batch_train = default_timer() - t1
        train_time += per_batch_train
        train_l2 += float(loss.item())

        msg_batch = (f"[ep_local {ep:04d} | ep_global {global_epoch:04d} | "
                     f"batch {batch_idx:04d}/{num_batches:04d}] "
                     f"perturb_time:{per_batch_perturb:.3f}s | train_time:{per_batch_train:.3f}s "
                     f"| loss:{float(loss.item()):.6f}")
        print(msg_batch)
        log_file.write(msg_batch + "\n")

    # ---- eval pass (clean set; 用干净 xx / 干净标签 y_test 评估基准泛化) ----
    model.eval()
    t2 = default_timer()
    test_l2 = 0.0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            pred = recurrent_model(xx)
            loss = myloss(pred.reshape(pred.shape[0], -1), yy.reshape(yy.shape[0], -1))
            test_l2 += float(loss.item())
    eval_time += (default_timer() - t2)

    total_time = default_timer() - epoch_t0

    if (ep % 10) == 0:
        xx0, _ = first_train_batch_cache

        # 1) 生成对抗序列（PGD 里要反传到 x_adv）
        with torch.enable_grad():
            adv_xx0, yy_adv0 = attack_and_rollout_seq(
                xx0, recurrent_model, pde_solver, Tout=T, cfg=attack_cfg
            )

        # 2) 纯推理与日志/快照
        with torch.no_grad():
            pred_clean = recurrent_model(xx0)
            clean_loss = myloss(pred_clean.reshape(pred_clean.shape[0], -1),
                                y_train[:xx0.shape[0]].reshape(xx0.shape[0], -1))

            pred_adv0 = recurrent_model(adv_xx0)
            adv_loss0 = myloss(pred_adv0.reshape(pred_adv0.shape[0], -1),
                               yy_adv0.reshape(yy_adv0.shape[0], -1))

            log_file.write(
                f"[ep_local {ep:04d} | ep_global {global_epoch:04d}] "
                f"first-batch clean_l2={float(clean_loss):.6f} "
                f"| adv_l2={float(adv_loss0):.6f}\n"
            )

            # —— 用 global_epoch 命名 —— 
            torch.save(
                {
                    "epoch": global_epoch,  # 存入全局 epoch
                    "x_adv": adv_xx0.detach().cpu(),
                    "y": yy_adv0.detach().cpu()
                },
                snap_dir / f"adv_first_batch_epoch{global_epoch:04d}.pt"
            )

    # —— 周期性存模型 —— 
    if (ep % 10) == 0 and ep > 0:
        model_snap = snap_dir / f"NS_2d_FNO_model_epoch{global_epoch:04d}.pth"
        torch.save({"epoch": global_epoch, "state_dict": model.state_dict()}, model_snap)

    msg = (f"epoch(local):{ep:04d} | epoch(global):{global_epoch:04d} | "
           f"perturb_time:{perturb_time:.3f}s | train_time:{train_time:.3f}s | "
           f"eval_time:{eval_time:.3f}s | total_time:{total_time:.3f}s | "
           f"train_l2:{train_l2:.6f} | test_l2:{test_l2:.6f}")
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

# —— 训练结束：保存最终模型（可选带上最终全局 epoch）—— 
final_global_epoch = base_epoch + (epochs - 1)
final_path = snap_dir / f"NS_2d_FNO_model_trainedby_{Path(dataset_name).stem}_final_epoch{final_global_epoch:04d}.pth"
torch.save({"epoch": final_global_epoch, "state_dict": model.state_dict()}, final_path)
log_file.write(f"\n[Done] saved final (global_epoch={final_global_epoch}) to {final_path}\n")
log_file.close()
print(f"Model saved to {final_path}")



