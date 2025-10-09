#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continual Train <-> JAX/Exponax Attack Loop for FNO2d (PyTorch + JAX)

- 产物全部写到脚本同目录的 outputs/ 下（模型、对抗数据、日志、指标、快照）
- 可断点续跑：自动从 snapshot_round*.json 推断起始轮
- 训练：clean + 历史 expanded 按 mix_ratio 混入
- 评测：train/test 写入 metrics.csv（去重）
- 攻击（PGD+Exponax）：
    * 步数恒定（--steps），默认 10
    * alpha/epsilon 可提供列表，逐轮取索引 (rd-1)
    * 每轮只攻击 attack_ratio 比例
    * 下一轮攻击的初始池 = 原始 train 的 x + 上一轮对抗 x（避免反复 IO）
    * 详细计时、进度与清理显存

要求工程内已有：
- utilities3.py: LpLoss, count_params
- models/FNO2d.py: FNO2d, RecurrentPredictor
"""

import os
import sys
import re
import csv
import json
import math
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ---- 内存策略（PyTorch + JAX）----
# 使用 cudaMallocAsync 能显著缓解碎片（PyTorch 1.13+/2.x）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
# JAX（可选）
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from tqdm import tqdm

# ------------------ 项目内模块 ------------------
HERE = Path(__file__).resolve().parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities3 import LpLoss, count_params                         # noqa: E402
from models.FNO2d import FNO2d, RecurrentPredictor                  # noqa: E402

# ------------------ JAX / Exponax ------------------
import jax
import jax.numpy as jnp
import jaxlib
import exponax as ex

import shutil, subprocess

def _bytes_to_mib(x: int) -> float:
    return float(x) / (1024**2)

def gpu_mem_string(device: torch.device | str = "cuda") -> str:
    if device == "cpu" or not torch.cuda.is_available():
        return "CPU mode (no CUDA)"
    dev = torch.device(device)
    dev_id = dev.index if dev.index is not None else 0
    free_b, total_b = torch.cuda.mem_get_info(dev_id)
    alloc = torch.cuda.memory_allocated(dev_id)
    reserv = torch.cuda.memory_reserved(dev_id)
    return (f"GPU{dev_id} mem | free { _bytes_to_mib(free_b):.1f} MiB / total { _bytes_to_mib(total_b):.1f} MiB | "
            f"allocated { _bytes_to_mib(alloc):.1f} MiB | reserved { _bytes_to_mib(reserv):.1f} MiB")

def print_cuda_mem(tag: str = "", device: torch.device | str = "cuda"):
    print(f"[Mem]{' ' + tag if tag else ''} {gpu_mem_string(device)}", flush=True)

def try_nvsmi_once() -> None:
    try:
        if shutil.which("nvidia-smi"):
            out = subprocess.check_output(
                ["nvidia-smi","--query-gpu=name,memory.total,memory.used,memory.free","--format=csv,noheader,nounits"]
            )
            print("[nvidia-smi] name,total(MiB),used(MiB),free(MiB)")
            print(out.decode("utf-8").strip(), flush=True)
    except Exception:
        pass

import re
from typing import Optional

def _extract_round_from_name(name: str) -> Optional[int]:
    """
    从模型文件名中提取轮次：匹配 FNO2d_r<NUM>_m...*.pth
    返回 int 或 None
    """
    m = re.search(r"FNO2d_r(\d+)_", name)
    return int(m.group(1)) if m else None

def latest_model_round(models_dir: Path) -> int:
    """
    在 models_dir 下寻找所有 FNO2d_r*_m*.pth，返回最大 r。若无则 0。
    """
    if not models_dir.exists():
        return 0
    max_r = 0
    for p in models_dir.glob("FNO2d_r*_m*.pth"):
        rid = _extract_round_from_name(p.name)
        if rid is not None:
            max_r = max(max_r, rid)
    return max_r

def latest_any_round(out_root: Path) -> int:
    """
    综合“模型文件最大 r”和“snapshot_round*.json 最大 r”，取更大者。
    优先模型（更可靠），snapshot 仅作兜底。
    """
    models_dir = out_root / "models"
    r_models = latest_model_round(models_dir)
    # snapshot 兜底
    snaps = sorted(out_root.glob("snapshot_round*.json"))
    r_snaps = 0
    if snaps:
        try:
            r_snaps = max(int(p.stem.split("round")[-1]) for p in snaps)
        except Exception:
            r_snaps = 0
    return max(r_models, r_snaps)

def find_latest_ckpt_path_for_round(models_dir: Path, r: int) -> Optional[Path]:
    """
    给定轮次 r，在 models/ 下找该轮的模型文件（宽/模数等超参若变动，用通配符兜住）。
    例如：FNO2d_r6_m96x96_w80_Tin10_T10.pth
    """
    cands = list(models_dir.glob(f"FNO2d_r{r}_m*x*_w*_Tin*_T*.pth")) + \
            list(models_dir.glob(f"FNO2d_r{r}_m*.pth"))
    if not cands:
        return None
    # 若有多个，按修改时间取最新
    return max(cands, key=lambda p: p.stat().st_mtime)


# ==== 显存诊断与清理工具 ====
from datetime import datetime
import gc, gzip



def _fmt_bytes(n: int | float) -> str:
    try:
        n = float(n)
    except Exception:
        return str(n)
    for unit in ["B","KiB","MiB","GiB","TiB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"

def log_torch_mem(tag: str, device: torch.device | str = "cuda",
                  summary_to: Path | None = None,
                  snapshot_to: Path | None = None,
                  list_tensors_to: Path | None = None):
    """
    打印 +（可选）把 PyTorch 显存摘要/快照/存活 CUDA 张量清单写到文件。
    - summary_to: 写入 torch.cuda.memory_summary
    - snapshot_to: 写入 torch.cuda.memory_snapshot（gzip 压缩的 JSON）
    - list_tensors_to: 写入当前 Python 进程里还活着的 CUDA 张量（前 200 个，按大小排序）
    """
    if device == "cpu" or not torch.cuda.is_available():
        print(f"[MemTorch]{' ' + tag if tag else ''} CPU only", flush=True); return
    dev = torch.device(device)
    torch.cuda.synchronize(dev)
    free_b, total_b = torch.cuda.mem_get_info(dev)
    cur = torch.cuda.memory_allocated(dev)
    res = torch.cuda.memory_reserved(dev)
    max_alloc = torch.cuda.max_memory_allocated(dev)
    max_res   = torch.cuda.max_memory_reserved(dev)
    print(f"[MemTorch]{' ' + tag if tag else ''} | "
          f"free/total={_fmt_bytes(free_b)}/{_fmt_bytes(total_b)} | "
          f"alloc={_fmt_bytes(cur)} res={_fmt_bytes(res)} | "
          f"max_alloc={_fmt_bytes(max_alloc)} max_res={_fmt_bytes(max_res)}", flush=True)

    if summary_to:
        try:
            summary_to.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_to, "w", encoding="utf-8") as f:
                f.write(torch.cuda.memory_summary(device=dev, abbreviated=False))
        except Exception as e:
            print(f"[MemTorch] write memory_summary failed: {e}")

    if snapshot_to:
        try:
            snapshot_to.parent.mkdir(parents=True, exist_ok=True)
            snap = torch.cuda.memory_snapshot()
            with gzip.open(snapshot_to, "wt", encoding="utf-8") as f:
                import json
                json.dump(snap, f, default=str)
        except Exception as e:
            print(f"[MemTorch] write memory_snapshot failed: {e}")

    if list_tensors_to:
        try:
            seen = []
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda:
                        numel = obj.numel()
                        bytes_ = numel * obj.element_size()
                        seen.append((bytes_, tuple(obj.shape), str(obj.dtype)))
                except Exception:
                    pass
            seen.sort(key=lambda x: x[0], reverse=True)
            list_tensors_to.parent.mkdir(parents=True, exist_ok=True)
            with open(list_tensors_to, "w", encoding="utf-8") as f:
                f.write(f"# live CUDA tensors at {datetime.now().isoformat()}\n")
                total = 0
                for i, (b, shp, dt) in enumerate(seen[:200]):
                    total += b
                    f.write(f"{i:4d} | {_fmt_bytes(b):>10} | {shp} | {dt}\n")
                f.write(f"\nTotal(top200) = {_fmt_bytes(total)}\n")
        except Exception as e:
            print(f"[MemTorch] write live tensor list failed: {e}")

def jax_mem_cleanup():
    """清 JAX 编译/执行缓存，释放 XLA 常驻内存。"""
    try:
        if hasattr(JaxPDEWrapper, "_vjp_cache"):
            JaxPDEWrapper._vjp_cache.clear()
    except Exception:
        pass
    try:
        jax.clear_caches()
    except Exception:
        pass
    try:
        for d in jax.devices():
            try:
                jax.block_until_ready(jnp.array(0).device_buffer)
            except Exception:
                pass
    except Exception:
        pass

def hard_cuda_gc(tag: str = ""):
    """强制 Python/GPU 清理顺序：先 gc，再清 CUDA allocator，再收 IPC 残留，再打印。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    print_cuda_mem(f"after hard_cuda_gc {tag}")

# ------------------ 工具函数 ------------------
def set_all_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def torch_load_compat(path: Path, map_location=None):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)

def ensure_batched_nhwt(seq: torch.Tensor) -> torch.Tensor:
    if seq.dim() == 3:   # (H,W,T) -> (1,H,W,T)
        return seq.unsqueeze(0)
    if seq.dim() == 4:   # (N,H,W,T)
        return seq
    raise ValueError(f"Expected (H,W,T) or (N,H,W,T), got {tuple(seq.shape)}")

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def mape(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.where(np.abs(a) < eps, eps, np.abs(a))
    return float(np.mean(np.abs((b - a) / denom)) * 100.0)

def parse_dataset_name(stem: str) -> Dict[str, str]:
    info: Dict[str, str] = {}
    pats = {
        "nx":           r"_nx(\d+)",
        "N":            r"_N(\d+)",
        "nu":           r"_nu([-\d\.]+)",
        "t":            r"_t([-\d\.]+)",
        "ntimepoints":  r"_ntimepoints(\d+)",
        "length_scale": r"_length_scale([-\d\.]+)",
        "variance":     r"_variance([-\d\.]+)",
        "alpha":        r"_alpha([-\d\.]+)",
        "tau":          r"_tau([-\d\.]+)",
        "period":       r"_period([-\d\.]+)",
        "exp_factor":   r"_exp_factor([-\d\.]+)",
        "vmax":         r"_vmax([-\d\.]+)",
        "vmin":         r"_vmin([-\d\.]+)",
    }
    for k, pat in pats.items():
        m = re.search(pat, stem)
        if m: info[k] = m.group(1)
    m = re.search(r"solver=([^_]+)", stem)
    if m: info["solver"] = m.group(1)
    return info

def make_loaders_from_y(y_all: torch.Tensor, ntrain: int, ntest: int, s: int,
                        T_in: int, T_out: int, batch_size: int, seed: int):
    """
    y_all: [N, H, W, T_total]
    取前 T_in 作为输入，后 T_out 作为监督。
    """
    H = y_all.shape[1]
    sub = H // s
    if sub <= 0:
        raise ValueError(f"Invalid sub-sampling: H={H}, s={s}")

    g = torch.Generator(device='cpu').manual_seed(seed)
    perm = torch.randperm(y_all.shape[0], generator=g)
    idx_tr = perm[:ntrain]
    idx_te = perm[ntrain:ntrain+ntest]

    y_tr = y_all[idx_tr][:, ::sub, ::sub, :]
    y_te = y_all[idx_te][:, ::sub, ::sub, :]

    x_tr, t_tr = y_tr[..., :T_in], y_tr[..., T_in:T_in+T_out]
    x_te, t_te = y_te[..., :T_in], y_te[..., T_in:T_in+T_out]

    x_tr = x_tr.reshape(ntrain, s, s, T_in)
    x_te = x_te.reshape(ntest,  s, s, T_in)

    train_ds = TensorDataset(x_tr, t_tr)
    test_ds  = TensorDataset(x_te, t_te)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True
    )

    return train_loader, test_loader, sub

def build_model(m1: int, m2: int, width: int, T_out: int, step: int, device: torch.device):
    fno = FNO2d(m1, m2, width).to(device)
    rec = RecurrentPredictor(fno, T_out=T_out, step=step).to(device)
    return fno, rec

def train_epochs(fno, rec, train_loader, test_loader, epochs: int, lr: float,
                 iterations: int, log_file: Path, device: torch.device):
    myloss = LpLoss(size_average=False)
    opt = torch.optim.Adam(fno.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iterations))
    with open(log_file, "a") as lf:
        lf.write(f"model parameters: {count_params(fno)}\n")

    print_cuda_mem("before training", device)

    for ep in tqdm(range(epochs), desc="Training"):
        fno.train()
        t1 = time.perf_counter()
        train_l2 = 0.0
        for xx, yy in train_loader:
            xx, yy = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True)
            pred = rec(xx)
            bs = pred.size(0)
            loss = myloss(pred.view(bs, -1), yy.view(bs, -1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            train_l2 += float(loss.item())
        test_l2 = 0.0
        with torch.no_grad():
            fno.eval()
            for xx, yy in test_loader:
                xx, yy = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True)
                pred = rec(xx)
                bs = pred.size(0)
                loss = myloss(pred.view(bs, -1), yy.view(bs, -1))
                test_l2 += float(loss.item())
        t2 = time.perf_counter()
        with open(log_file, "a") as lf:
            lf.write(f"epoch:{ep}, time:{t2-t1:.6f}, train l2:{train_l2:.8f}, test l2:{test_l2:.8f}\n")
        print_cuda_mem(f"after epoch {ep}", device)

def eval_dataset_file(pt_path: Path, rec: RecurrentPredictor, device: torch.device,
                      t_in: int, t_out: int) -> Tuple[int, float, float, float, float, float, float]:
    data_obj = torch_load_compat(pt_path, map_location="cpu")
    seq = None
    for key in ("y", "x"):
        if key in data_obj and torch.is_tensor(data_obj[key]):
            seq = data_obj[key]; break
    if seq is None:
        for v in data_obj.values():
            if torch.is_tensor(v) and v.dim() >= 3: seq = v; break
    if seq is None:
        raise ValueError(f"No sequence tensor found in {pt_path}")
    seq = ensure_batched_nhwt(seq)
    N, H, W, T_total = seq.shape
    if t_in + t_out > T_total:
        raise RuntimeError(f"T_in({t_in})+T_out({t_out}) > T_total({T_total}) for {pt_path}")

    rm, ma, mp = [], [], []
    with torch.no_grad():
        for idx in range(N):
            sample = seq[idx]  # (H,W,T)
            current_input  = sample[..., :t_in].unsqueeze(0).to(device)
            ground_truth   = sample[..., t_in:t_in+t_out].unsqueeze(0)
            pred = rec(current_input)
            gt_last = ground_truth[0, ..., -1].cpu().numpy()
            pr_last = pred[0, ..., -1].cpu().numpy()
            rm.append(rmse(gt_last, pr_last))
            ma.append(mae(gt_last, pr_last))
            mp.append(mape(gt_last, pr_last))
    return N, float(np.mean(rm)), float(np.std(rm)), float(np.mean(ma)), float(np.std(ma)), float(np.mean(mp)), float(np.std(mp))

def iter_pt_files(root: Path):
    for p in sorted(root.rglob("*.pt")):
        if p.is_file():
            yield p

def eval_many_pt_files(dir_path: Path, rec_eval: RecurrentPredictor, device: torch.device,
                       t_in: int, t_out: int, csv_path: Path, header: List[str],
                       round_id: int, model_name: str, model_path: Path):
    count = 0
    for p in iter_pt_files(dir_path):
        try:
            N_eval, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(p, rec_eval, device, t_in, t_out)
            row = [round_id, "generalizability", p.stem, str(p), N_eval,
                   rmean, rstd, amean, astd, mmean, mstd, model_name, str(model_path)]
            if not already_logged(csv_path, round_id, "generalizability", p, model_name):
                write_metrics_row(csv_path, header, row)
            count += 1
            if (count % 20) == 0:
                print(f"[Round {round_id}] Eval generalizability progress: {count}", flush=True)
                print_cuda_mem(f"eval generalizability #{count}", device)
        except Exception as e:
            print(f"[Round {round_id}] Eval error on {p.name}: {e}")
    print(f"[Round {round_id}] Generalizability eval done. Files processed: {count}")

# ------------------ JAX/Exponax 攻击 ------------------
def spectral_upsample(field: torch.Tensor, target_size=256, device=torch.device("cuda")):
    *batch_dims, H, W = field.shape
    assert target_size >= H and target_size >= W
    field = field.to(device)
    freq = torch.fft.fft2(field, norm='ortho')
    freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))
    pad_H = (target_size - H) // 2
    pad_W = (target_size - W) // 2
    new_freq_shifted = torch.zeros(*batch_dims, target_size, target_size,
                                   dtype=freq_shifted.dtype, device=freq_shifted.device)
    new_freq_shifted[..., pad_H:pad_H+H, pad_W:pad_W+W] = freq_shifted
    new_freq = torch.fft.ifftshift(new_freq_shifted, dim=(-2, -1))
    upsampled = torch.fft.ifft2(new_freq, norm='ortho')
    return (target_size / H) * upsampled.real

class JaxPDEWrapper(torch.autograd.Function):
    _vjp_cache = {}
    @staticmethod
    def forward(ctx, a_torch, g):
        if not a_torch.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
        a_jax = jax.dlpack.from_dlpack(a_dlpack)
        g_output_jax = g(a_jax)
        out_dlpack = jax.dlpack.to_dlpack(g_output_jax)
        g_output = torch.utils.dlpack.from_dlpack(out_dlpack)
        ctx.save_for_backward(a_torch)
        ctx.g = g
        return g_output
    @staticmethod
    def backward(ctx, grad_output):
        a_torch, = ctx.saved_tensors
        g = ctx.g
        cache_key = id(g)
        if cache_key not in JaxPDEWrapper._vjp_cache:
            def jax_vjp(a_jax, grad_jax):
                _, vjp_fn = jax.vjp(g, a_jax)
                return vjp_fn(grad_jax)
            JaxPDEWrapper._vjp_cache[cache_key] = jax.jit(jax_vjp)
        jitted_vjp = JaxPDEWrapper._vjp_cache[cache_key]
        grad_dlpack = torch.utils.dlpack.to_dlpack(grad_output.contiguous())
        grad_jax = jax.dlpack.from_dlpack(grad_dlpack)
        a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
        a_jax = jax.dlpack.from_dlpack(a_dlpack)
        grad_input_jax, = jitted_vjp(a_jax, grad_jax)
        grad_input_dlpack = jax.dlpack.to_dlpack(grad_input_jax)
        return torch.utils.dlpack.from_dlpack(grad_input_dlpack), None

def generate_sequence_every_second(u0, nu, T_seconds, fixed_step=0.005):
    num_steps = int(T_seconds / fixed_step)
    u0 = jnp.rot90(jnp.flipud(u0), 3)
    full_ic = jnp.expand_dims(jnp.array(u0, dtype=jnp.float32), axis=0)
    stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
        2, 1, u0.shape[0], fixed_step, diffusivity=nu, order=4
    )
    rollout_stepper = ex.rollout(stepper, num_steps, include_init=True)
    result = rollout_stepper(full_ic)
    idxs = jnp.array([int(s / fixed_step) for s in range(T_seconds + 1)], dtype=jnp.int32)
    picked = result[idxs, 0, ...]
    picked = jnp.swapaxes(picked, -1, -2)
    return picked

class DifferentiablePDESolver:
    def __init__(self, nu, device="cuda"):
        self.nu = nu
        self.device = device
    def rollout_seconds(self, x0, T_seconds):
        def solver_func(a):
            return generate_sequence_every_second(a, self.nu, T_seconds)
        return JaxPDEWrapper.apply(x0.contiguous(), solver_func)

class PDEAttackSystem:
    def __init__(self, recurrent_model, nu=1e-5, device="cuda", mode_spec="wwwwwwwwww"):
        assert len(mode_spec) == 10 and set(mode_spec).issubset(set("adw"))
        self.recurrent_model = recurrent_model
        self.nu = nu
        self.device = device
        self.mode_spec = mode_spec
        for p in self.recurrent_model.parameters():
            p.requires_grad = False
        self.recurrent_model.eval()
        self.pde_solver = DifferentiablePDESolver(nu, device)

    def forward(self, x0):
        need_solver_seconds = set()
        for t in range(1, 10):
            if self.mode_spec[t-1] in ("w", "d"):
                need_solver_seconds.add(t)
        ch19 = self.mode_spec[9]
        if ch19 in ("w", "d"):
            need_solver_seconds.add(19)
        seq_0_to_T = None
        if len(need_solver_seconds) > 0:
            T_required = max(need_solver_seconds)
            seq_0_to_T = self.pde_solver.rollout_seconds(x0, T_required)
        frames = [x0]
        for t in range(1, 10):
            ch = self.mode_spec[t-1]
            assert seq_0_to_T is not None and (seq_0_to_T.shape[0]-1) >= t
            f = seq_0_to_T[t]
            if ch == 'd':
                f = f.detach()
            frames.append(f)
        input_sequence = torch.stack(frames, dim=0)
        input_batch = input_sequence.permute(1, 2, 0).unsqueeze(0)
        predictions = self.recurrent_model(input_batch)   # (1,H,W,10)
        pred_19 = predictions[0, ..., -1]
        if ch19 == 'a':
            raise NotImplementedError("approx mode disabled")
        else:
            assert seq_0_to_T is not None and (seq_0_to_T.shape[0]-1) >= 19
            true_19 = seq_0_to_T[19]
            if ch19 == 'd':
                true_19 = true_19.detach()
        return pred_19, true_19

    def pgd_attack_adam(self, initial_x0, epsilon=0.1, alpha=0.01, num_steps=10,
                        norm="inf", beta1=0.9, beta2=0.999, adam_eps=1e-8,
                        use_sign_for_linf=True, amsgrad=False):
        x_adv = initial_x0.clone().detach().to(self.device).requires_grad_(True)
        original_x0 = initial_x0.clone().detach().to(self.device)
        m = torch.zeros_like(x_adv); v = torch.zeros_like(x_adv)
        vhat_max = torch.zeros_like(x_adv) if amsgrad else None
        t_adam = 0
        for _ in range(num_steps):
            if x_adv.grad is not None: x_adv.grad.zero_()
            pred_19, true_19 = self.forward(x_adv)
            loss = F.mse_loss(pred_19, true_19, reduction="sum")
            loss.backward()
            with torch.no_grad():
                t_adam += 1
                g = x_adv.grad
                m = beta1 * m + (1.0 - beta1) * g
                v = beta2 * v + (1.0 - beta2) * (g * g)
                m_hat = m / (1.0 - (beta1 ** t_adam))
                v_hat = v / (1.0 - (beta2 ** t_adam))
                denom = (torch.maximum(vhat_max, v_hat) if amsgrad else v_hat).sqrt() + adam_eps
                d_t = m_hat / denom
                if norm in ('inf', 'Linf', '∞'):
                    direction = torch.sign(d_t) if use_sign_for_linf else d_t
                    x_adv.data = x_adv + alpha * direction
                    delta = x_adv - original_x0
                    x_adv.data = original_x0 + torch.clamp(delta, -epsilon, epsilon)
                elif norm in (2, '2', 'L2'):
                    direction = d_t / (torch.norm(d_t, p=2) + 1e-12)
                    x_adv.data = x_adv + alpha * direction
                    current_delta = x_adv - original_x0
                    delta_norm = torch.norm(current_delta, p=2)
                    if delta_norm > epsilon:
                        x_adv.data = original_x0 + current_delta * (epsilon / (delta_norm + 1e-12))
                x_adv.grad.zero_()
        with torch.no_grad():
            seq_0_to_20 = self.pde_solver.rollout_seconds(x_adv, 20)  # [21,H,W]
            frames = [x_adv] + [seq_0_to_20[t] for t in range(1, 10)]
            input_sequence = torch.stack(frames, dim=0)
            input_batch = input_sequence.permute(1, 2, 0).unsqueeze(0)
            predictions = self.recurrent_model(input_batch)
            pred_19_final = predictions[0, ..., -1]
            true_19_final = seq_0_to_20[19]
            final_true_loss = F.mse_loss(pred_19_final, true_19_final, reduction="sum")
            y_all_frames = seq_0_to_20.permute(1, 2, 0).contiguous()
        return x_adv.detach(), y_all_frames.detach(), float(final_true_loss.item())

# ------------------ CSV/快照/续跑 ------------------
def write_metrics_row(csv_path: Path, header: List[str], row: List[Any]):
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)

def already_logged(csv_path: Path, round_id: int, dataset_group: str, dataset_path: Path, model_name: str) -> bool:
    if not csv_path.exists():
        return False
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row.get("round") == str(round_id) and
                row.get("dataset_group") == dataset_group and
                row.get("dataset_path") == str(dataset_path) and
                row.get("model_name") == model_name
            ):
                return True
    return False

def latest_completed_round(root: Path) -> int:
    snaps = sorted(root.glob("snapshot_round*.json"))
    if not snaps:
        return 0
    return max(int(p.stem.split("round")[-1]) for p in snaps)

# ------------------ 主流程 ------------------
def main():
    ap = argparse.ArgumentParser("Continual Train-Attack Loop (PyTorch + JAX/Exponax)")
    # 数据路径
    ap.add_argument("--train_pt", type=str, required=True, help="原始 TRAIN .pt （含 y 全序列）")
    ap.add_argument("--test_pt",  type=str, required=True, help="原始 TEST  .pt")

    # 模型与训练
    ap.add_argument("--modes1", type=int, default=64)
    ap.add_argument("--modes2", type=int, default=64)
    ap.add_argument("--width",  type=int, default=60)
    ap.add_argument("--Tin",    type=int, default=10)
    ap.add_argument("--Tout",   type=int, default=10)
    ap.add_argument("--step",   type=int, default=1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--ntrain", type=int, default=1000)
    ap.add_argument("--ntest",  type=int, default=100)

    # 攻击/JAX PDE
    ap.add_argument("--nu", type=float, default=1e-5)
    ap.add_argument("--attack_split", type=str, choices=["train","test"], default="train")
    ap.add_argument("--mode_spec", type=str, default="wwwwwwwwww")

    # 步数保持常数
    ap.add_argument("--steps", type=int, default=10, help="PGD steps (常数)")

    # alpha / epsilon 列表
    ap.add_argument("--attack_alpha_list", type=str, default="",
                    help="例如 '2.5,5,10,20,50'")
    ap.add_argument("--attack_epsilon_list", type=str, default="",
                    help="例如 '8,15,35,75,105'")

    # 仅攻一部分
    ap.add_argument("--attack_ratio", type=float, default=1.0, help="每轮只攻击初始池的比例，例如 0.25")

    # “翻倍”池：把上一轮对抗 x 并入下一轮攻击的初始池
    ap.add_argument("--reuse_prev_adv", action="store_true", default=True,
                    help="默认开启。把上一轮对抗 x 并入下一轮攻击初始池。")

    # 持续学习
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--mix_ratio", type=float, default=0.25, help="训练时 expanded 占 clean 的比例（ADD）")
    ap.add_argument("--device", type=str, default="auto", help="cuda|cpu|auto")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_root", type=str, default=None, help="输出根目录（默认 ./outputs 与脚本同级）")
    
    ap.add_argument("--generalizability_dir", type=str,
                default=str((HERE / "../datasets/exponax_datasets/t20/generalizability").resolve()),
                help="Directory with .pt files for generalizability metrics")
    ap.add_argument("--eval_generalizability", action="store_true", default=True,
                    help="Evaluate and append metrics for all .pt files under generalizability_dir")
    
    ap.add_argument("--debug_mem", action="store_true", default=False,
                help="启用显存诊断输出与快照")
    ap.add_argument("--dump_mem_snapshot_each_round", action="store_true", default=False,
                    help="每轮关键阶段写 memory_snapshot（gzip JSON）")
    ap.add_argument("--hard_reset_cuda_after_attack", action="store_true", default=False,
                    help="每轮攻击后重建模型，必要时用以完全释放碎片（会更慢）")

    args = ap.parse_args()

    # 设备 & 随机种
    set_all_seeds(args.seed)
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu"))
    print(f"[Device] {device}")

    # 产物目录 —— 固定在脚本目录下
    DEFAULT_OUT = (HERE / "outputs").resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else DEFAULT_OUT
    out_root.mkdir(parents=True, exist_ok=True)
    models_dir  = out_root / "models"
    logs_dir    = out_root / "logs"
    metrics_csv = out_root / "metrics.csv"
    expanded_root = out_root / "expanded"    # 对抗数据只写这里
    for d in [models_dir, logs_dir, expanded_root]:
        d.mkdir(parents=True, exist_ok=True)

    # === 新增：工具函数（仅在 main 内部使用）===
    import re
    from typing import Optional

    def _extract_round_from_name(name: str) -> Optional[int]:
        m = re.search(r"FNO2d_r(\d+)_", name)
        return int(m.group(1)) if m else None

    def latest_model_round(models_dir: Path) -> int:
        if not models_dir.exists():
            return 0
        max_r = 0
        for p in models_dir.glob("FNO2d_r*_m*.pth"):
            rid = _extract_round_from_name(p.name)
            if rid is not None:
                max_r = max(max_r, rid)
        return max_r

    def latest_any_round(out_root: Path) -> int:
        # 优先模型；snapshot 兜底
        r_models = latest_model_round(out_root / "models")
        snaps = sorted(out_root.glob("snapshot_round*.json"))
        r_snaps = 0
        if snaps:
            try:
                r_snaps = max(int(p.stem.split("round")[-1]) for p in snaps)
            except Exception:
                r_snaps = 0
        return max(r_models, r_snaps)

    def find_latest_ckpt_path_for_round(models_dir: Path, r: int) -> Optional[Path]:
        cands = list(models_dir.glob(f"FNO2d_r{r}_m*x*_w*_Tin*_T*.pth")) + \
                list(models_dir.glob(f"FNO2d_r{r}_m*.pth"))
        if not cands:
            return None
        return max(cands, key=lambda p: p.stat().st_mtime)

    # === 修改1：续跑范围（优先以“现有模型的最新 r”为准；无模型再回退 snapshot）===
    latest_r = latest_any_round(out_root)
    start_round = latest_r + 1
    end_round   = start_round + args.rounds - 1
    print(f"[Resume] detected latest_r={latest_r}  -> start_round={start_round}, end_round={end_round} (out_root={out_root})")

    # 读取原始数据
    train_pt = Path(args.train_pt).resolve()
    test_pt  = Path(args.test_pt).resolve()
    if not train_pt.exists() or not test_pt.exists():
        raise FileNotFoundError("train_pt or test_pt not found")
    data_tr = torch_load_compat(train_pt, map_location="cpu")
    data_te = torch_load_compat(test_pt,  map_location="cpu")
    y_tr_all = data_tr["y"].cpu()
    y_te_all = data_te["y"].cpu()
    N0 = int(y_tr_all.shape[0])

    # CSV 头
    header = [
        "round","dataset_group","dataset_name","dataset_path","n",
        "rmse_mean","rmse_std","mae_mean","mae_std","mape_mean","mape_std",
        "model_name","model_path"
    ]

    # alpha/epsilon 列表（逐轮索引）
    def parse_num_list(s: str):
        if not s: return []
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    alpha_list = parse_num_list(args.attack_alpha_list)   # 直接使用为数值
    eps_list   = parse_num_list(args.attack_epsilon_list) # 直接使用为数值

    # 历史 expanded（用于训练混入 & 作为下轮攻击池）
    expanded_files: List[Path] = []

    # ====== 轮次循环 ======
    for rd in range(start_round, end_round + 1):
        if args.debug_mem:
            log_torch_mem(f"r{rd:03d}_loop_entry", device)

        print("\n" + "="*24 + f" ROUND {rd}/{end_round} " + "="*24)

        # 计划保存为 r{rd} 的目标 ckpt（即若已有 rK，则当前 rd=K+1）
        ckpt_name = f"FNO2d_r{rd}_m{args.modes1}x{args.modes2}_w{args.width}_Tin{args.Tin}_T{args.Tout}.pth"
        ckpt_path = models_dir / ckpt_name
        log_path  = logs_dir / f"train_r{rd}.txt"

        # ---------- 构建训练数据（clean + expanded 混入） ----------
        tr_loader, te_loader, sub = make_loaders_from_y(
            y_tr_all, args.ntrain, args.ntest, s=256,
            T_in=args.Tin, T_out=args.Tout,
            batch_size=args.batch_size, seed=args.seed + rd*10
        )

        mix_ratio = max(0.0, float(args.mix_ratio))
        clean_n   = len(tr_loader.dataset)
        exp_pick  = 0
        exp_pool  = 0
        exp_sources = []
        counts_by_round = {}

        if expanded_files and mix_ratio > 0:
            exp_tensors = []
            meta = []
            cursor = 0
            for p in expanded_files:
                obj = torch_load_compat(p, map_location="cpu")
                if "y" in obj and obj["y"].ndim == 4:
                    y = obj["y"]                     # (Nexp, H, W, 21)
                    exp_tensors.append(y)
                    exp_sources.append(str(p))
                    exp_pool += y.shape[0]
                    m = re.search(r"round(\d+)", p.parent.name)
                    rid = int(m.group(1)) if m else -1
                    meta.append({"round": rid, "start": cursor, "end": cursor + y.shape[0]})
                    cursor += y.shape[0]

            if exp_tensors:
                y_exp_all = torch.cat(exp_tensors, dim=0)
                need = int(round(args.ntrain * mix_ratio))
                g = torch.Generator(device='cpu').manual_seed(args.seed + rd*100)
                idx = torch.randint(low=0, high=y_exp_all.shape[0], size=(need,), generator=g)

                counts_by_round = {}
                for j in idx.tolist():
                    for m in meta:
                        if m["start"] <= j < m["end"]:
                            counts_by_round[m["round"]] = counts_by_round.get(m["round"], 0) + 1
                            break

                y_pick = y_exp_all[idx]
                x_exp = y_pick[..., :args.Tin]
                t_exp = y_pick[..., args.Tin:args.Tin+args.Tout]

                if x_exp.shape[1] != 256 or x_exp.shape[2] != 256:
                    sub_exp = x_exp.shape[1] // 256
                    x_exp = x_exp[:, ::sub_exp, ::sub_exp, :]
                    t_exp = t_exp[:, ::sub_exp, ::sub_exp, :]

                ds_exp = TensorDataset(x_exp, t_exp)
                tr_loader = DataLoader(
                    ConcatDataset([tr_loader.dataset, ds_exp]),
                    batch_size=args.batch_size, shuffle=True, drop_last=True,
                    pin_memory=True,
                    generator=torch.Generator().manual_seed(args.seed + rd*123)
                )
                exp_pick = need

        # ===== 训练样本构成日志 =====
        total_train_n = len(tr_loader.dataset)
        breakdown_str = ", ".join(
            f"r{int(k)}:{v}" for k, v in sorted(counts_by_round.items())
        ) if counts_by_round else "none"

        msg = (
            f"[Round {rd}] Train composition -> total: {total_train_n} "
            f"(clean: {clean_n}, expanded: {exp_pick}; pool={exp_pool})\n"
            f"[Round {rd}] Expanded-by-round: {breakdown_str}\n"
            f"[Round {rd}] Expanded sources used as pool: {len(exp_sources)} file(s)\n"
            + "\n".join(f"    - {s}" for s in exp_sources)
        )
        print(msg, flush=True)
        with open(log_path, "a") as lf:
            lf.write(msg + "\n")

        # === 修改2：模型加载策略（自动 rK -> 训练保存为 rK+1）===
        fno, rec = build_model(args.modes1, args.modes2, args.width, args.Tout, args.step, device)

        if ckpt_path.exists():
            # 该轮 r{rd} 已经训练过：直接加载并跳过训练
            print(f"[Round {rd}] CKPT exists: {ckpt_path} -> 跳过训练，直接加载")
            sd = torch.load(ckpt_path, map_location=device)
            fno.load_state_dict(sd)
        else:
            # 自动寻找 models/ 下现存的最新 rK，作为热启动
            latest_r_models = latest_model_round(models_dir)
            if latest_r_models > 0:
                prev_path = find_latest_ckpt_path_for_round(models_dir, latest_r_models)
                if prev_path is not None:
                    print(f"[Round {rd}] Warm-start from latest model r{latest_r_models}: {prev_path}")
                    fno.load_state_dict(torch.load(prev_path, map_location=device))
                else:
                    print(f"[Round {rd}] No previous ckpt matched in models/, training from scratch")
            else:
                print(f"[Round {rd}] No existing model found in models/, training from scratch")

            iters = args.epochs * max(1, (len(tr_loader.dataset) // args.batch_size))
            with open(log_path, "w") as lf:
                lf.write("NS 2d FNO continual training log\n")
                lf.write(json.dumps({
                    "round": rd, "params": {
                        "modes1": args.modes1, "modes2": args.modes2, "width": args.width,
                        "Tin": args.Tin, "Tout": args.Tout, "step": args.step,
                        "epochs": args.epochs, "batch_size": args.batch_size,
                        "lr": args.lr, "subsample_factor": sub, "device": str(device)
                    }}, indent=2) + "\n\n")
            train_epochs(fno, rec, tr_loader, te_loader, args.epochs, args.lr, iters, log_path, device)
            torch.save(fno.state_dict(), ckpt_path)
            print(f"[Round {rd}] Saved CKPT -> {ckpt_path}")

        # ---------- 评测（train/test） ----------
        fno.eval()
        rec_eval = RecurrentPredictor(fno, T_out=args.Tout, step=args.step).to(device)
        for (dg, p) in [("train", train_pt), ("test", test_pt)]:
            try:
                N_eval, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(p, rec_eval, device, args.Tin, args.Tout)
                row = [rd, dg, p.stem, str(p), N_eval, rmean, rstd, amean, astd, mmean, mstd, ckpt_name, str(ckpt_path)]
                if not already_logged(metrics_csv, rd, dg, p, ckpt_name):
                    write_metrics_row(metrics_csv, header, row)
                print(f"[Round {rd}] Eval {dg}: RMSEμ={rmean:.4g}, MAEμ={amean:.4g}, MAPEμ={mmean:.2f}%")
            except Exception as e:
                print(f"[Round {rd}] Eval error on {p.name}: {e}")
        
        # ---------- 评测（generalizability 全量 .pt） ----------
        if args.eval_generalizability:
            gen_dir = Path(args.generalizability_dir).resolve()
            if gen_dir.exists():
                print(f"[Round {rd}] Eval generalizability under: {gen_dir}")
                print_cuda_mem("before generalizability eval", device)
                eval_many_pt_files(gen_dir, rec_eval, device,
                                   args.Tin, args.Tout,
                                   metrics_csv, header,
                                   rd, ckpt_name, ckpt_path)
                print_cuda_mem("after generalizability eval", device)
            else:
                print(f"[Round {rd}] Skipped generalizability: dir not found -> {gen_dir}")

        import gc; gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print_cuda_mem("before attack (after eval cleanup)", device)

        # ---------- 攻击（PGD + Exponax） ----------
        split_tag = args.attack_split

        # 选择 alpha/epsilon
        size = 256
        eps  = eps_list[min(rd-1, len(eps_list)-1)] if eps_list else 0.0002 * (size**2) * (1.15 ** (rd-1))
        alpha= alpha_list[min(rd-1, len(alpha_list)-1)] if alpha_list else 0.01 * 100.0 * (1.10 ** (rd-1))
        steps= int(args.steps)

        # 初始池：严格在 CPU 上构造
        data_raw = torch_load_compat(train_pt if split_tag=="train" else test_pt, map_location="cpu")
        X0_base  = data_raw["x"].detach().cpu().float()  # (N, H0, W0) on CPU

        # 并入上一轮对抗 x（CPU）
        if args.reuse_prev_adv and len(expanded_files) > 0:
            prev_obj = torch_load_compat(expanded_files[-1], map_location="cpu")
            if "x" in prev_obj and prev_obj["x"].ndim == 3:
                X0_prev = prev_obj["x"].detach().cpu().float()   # (M,256,256) on CPU
                X0_cpu  = torch.cat([X0_base, X0_prev], dim=0)
                print(f"[Round {rd}] Attack pool: base {X0_base.shape[0]} + prev_adv {X0_prev.shape[0]} -> {X0_cpu.shape[0]}")
            else:
                X0_cpu = X0_base
        else:
            X0_cpu = X0_base

        # 只抽取比例 M（仍在 CPU）
        M = int(math.ceil(X0_cpu.shape[0] * max(0.0, min(1.0, args.attack_ratio))))
        g = torch.Generator(device='cpu').manual_seed(args.seed + rd*999)
        perm = torch.randperm(X0_cpu.shape[0], generator=g)[:M]
        X0_cpu = X0_cpu[perm]
        print(f"[Round {rd}] Will attack {M} samples (ratio={args.attack_ratio}) | eps={eps}, alpha={alpha}, steps={steps}")

        # 输出文件
        exp_dir = expanded_root / f"round{rd:03d}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_name = (f"adv_{split_tag}_mspec={args.mode_spec}_norm2_alpha{alpha:.6g}_epsilon{eps:.6g}_steps{steps}.pt")
        exp_path = exp_dir / exp_name

        if exp_path.exists():
            print(f"[Round {rd}] Expanded exists -> {exp_path} (skip attack)")
        else:
            rec_attack = RecurrentPredictor(fno, T_out=args.Tout, step=args.step).to(device)
            attack_system = PDEAttackSystem(recurrent_model=rec_attack, nu=args.nu, device=device, mode_spec=args.mode_spec)
            H = W = 256
            x_out    = torch.empty((M, H, W),     dtype=torch.float32)  # CPU
            y_out    = torch.empty((M, H, W, 21), dtype=torch.float32)  # CPU
            loss_out = torch.empty((M,),          dtype=torch.float32)  # CPU
            per_csv = exp_dir / f"attack_round{rd:03d}_timing.csv"

            with open(per_csv, "w", newline="") as fcsv:
                w = csv.writer(fcsv); w.writerow(["idx","elapsed_sec","avg_sec","loss"])
                start_all = time.perf_counter()
                bar = tqdm(range(M), desc=f"[Round {rd}] Attack {split_tag}", unit="sample")
                for i in bar:
                    t0 = time.perf_counter()

                    x0_cpu = X0_cpu[i]
                    x0     = spectral_upsample(x0_cpu, 256, device=device)

                    x_final, y_all_frames, final_true_loss = attack_system.pgd_attack_adam(
                        initial_x0=x0, epsilon=eps, alpha=alpha, num_steps=steps,
                        norm="2", beta1=0.9, beta2=0.999, adam_eps=1e-8, use_sign_for_linf=True, amsgrad=False
                    )

                    x_out[i]    = x_final.detach().float().cpu()
                    y_out[i]    = y_all_frames.detach().float().cpu()
                    loss_out[i] = float(final_true_loss)

                    dt  = time.perf_counter() - t0
                    avg = (time.perf_counter() - start_all) / (i+1)
                    w.writerow([i, f"{dt:.4f}", f"{avg:.4f}", f"{final_true_loss:.6g}"])
                    if (i+1) % 5 == 0:
                        bar.set_postfix({"avg_s": f"{avg:.2f}", "last": f"{dt:.2f}", "lossμ": f"{loss_out[:i+1].mean().item():.4g}"})
                        print_cuda_mem(f"attack post idx={i}", device)

                    torch.cuda.empty_cache()

            torch.save({"x": x_out, "y": y_out, "loss": loss_out}, exp_path)
            with open(exp_dir / "manifest.json", "w") as mf:
                json.dump({
                    "round": rd, "split": split_tag,
                    "alpha": alpha, "epsilon": eps, "steps": steps,
                    "attack_ratio": args.attack_ratio,
                    "pool": {"base_train_x": int(X0_base.shape[0]),
                             "prev_adv_x": int(X0_cpu.shape[0] - min(X0_base.shape[0], X0_cpu.shape[0])) if args.reuse_prev_adv else 0,
                             "attacked": int(M)}
                }, mf, indent=2)
            print(f"[Round {rd}] Saved expanded -> {exp_path}")

        # 让最新 expanded 参与下轮训练混入 & 作为下一轮 prev_adv
        if exp_path.exists():
            expanded_files.append(exp_path)

        # 阶段内存清理
        del data_raw, X0_base
        if 'X0_prev' in locals(): del X0_prev
        del X0_cpu
        if 'rec_attack' in locals(): del rec_attack
        if 'attack_system' in locals(): del attack_system
        import gc; gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # ---------- 快照 ----------
        snap = {
            "round": rd,
            "ckpt": str(ckpt_path),
            "expanded_added": str(exp_path),
            "mix_ratio_next_round": args.mix_ratio,
            "attack": {
                "split": split_tag, "nu": args.nu, "mode_spec": args.mode_spec,
                "epsilon": eps, "alpha": alpha, "steps": steps, "norm": "2",
                "attack_ratio": args.attack_ratio
            },
            "train_params": {
                "epochs": args.epochs, "lr": args.lr, "batch": args.batch_size,
                "ntrain": args.ntrain, "ntest": args.ntest
            },
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(out_root / f"snapshot_round{rd}.json", "w") as f:
            json.dump(snap, f, indent=2)

        print(f"[Round {rd}] Done.")

    print("\nAll rounds finished.")
    print(f"Models   -> {models_dir}")
    print(f"Expanded -> {expanded_root}")
    print(f"Metrics  -> {metrics_csv}")
    print(f"Logs     -> {logs_dir}")


if __name__ == "__main__":
    try_nvsmi_once()
    jax.config.update("jax_enable_x64", False)
    main()
