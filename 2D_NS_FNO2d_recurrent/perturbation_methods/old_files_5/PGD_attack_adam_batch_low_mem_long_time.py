#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-parity PGD attack script (v1 aligned to v2 semantics)

Goal: Make the first (batch) script behave IDENTICALLY to the second (single-sample)
script in every respect — numerical path, tensor transforms, solver/model wiring,
logging, saving — with the ONLY difference that:
  * the forward pass accepts (B,H,W) and runs PDE + FNO on the whole batch at once
  * the loss and gradients are computed jointly over the whole batch (batch-sum),
    i.e., one backward() over the aggregated loss.

Key choices copied from v2:
- JAX<->PyTorch bridge uses explicit to_dlpack/from_dlpack pairs.
- generate_sequence_every_second() reproduces v2 logic (flip+rot90, checkpoint, sparse
  save of integral seconds) and returns frames in the same orientation; for batch,
  we vmap it so final shape is (T+1, B, H, W) to match v1 call site.
- forward() uses the exact same mode_spec logic (a/d/w) and assembles (B,H,W,10)
  for the recurrent model; pred_19 and true_19 are chosen the same way as v2.
- pgd_attack_adam() computes surrogate_loss = MSE(pred_19, true_19, reduction="sum"),
  i.e., a single scalar over the entire batch; its L2 direction normalizes by the
  global L2-norm of the whole batch tensor, which equals the v2 behavior when B=1
  and yields a batch-wide direction for B>1.
- Recorder is kept identical to v2 (single pkl per run), only the stored arrays now
  contain a batch dimension when B>1.

This file is self-contained and ready to run in place of the first script.
"""

from datetime import datetime
import os
import sys
import time
import gc
import warnings
import subprocess
from pathlib import Path
from pprint import pprint

import torch
import torch.nn.functional as F

import jax
import jax.numpy as jnp
import jaxlib

from jax.extend import backend as jax_backend

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.FNO2d import FNO2d, RecurrentPredictor
import numpy as np
import pickle
import csv

# ====================== 环境与诊断（与 v2 对齐） ======================
DEBUG_GRAD = os.environ.get("DEBUG_GRAD", "0") == "1"

def print_versions():
    print("\n" + "="*40 + " 环境诊断 " + "="*40)
    print("\n[PyTorch]")
    print(f"Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        try:
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        except Exception:
            print("cuDNN Version: <unavailable>")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

    print("\n[JAX]")
    print(f"JAX Version: {jax.__version__}")
    print(f"JAXlib Version: {jaxlib.__version__}")
    print(f"JAX Backend: {jax_backend.get_backend().platform}")
    try:
        from jax.lib import xla_bridge
        print(f"JAX Devices: {xla_bridge.devices()}")
    except Exception as e:
        print(f"JAX Devices Error: {str(e)}")

    print("\n[CUDA/cuDNN System Info]")
    try:
        nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
        print(nvcc_out.split("\n")[0])
    except Exception:
        print("nvcc not found")
    print("="*40 + " 诊断结束 " + "="*40 + "\n")

print_versions()
jax.config.update("jax_enable_x64", False)

def _fmt_bytes(n: int) -> str:
    s = float(n)
    for unit in ["B","KB","MB","GB","TB"]:
        if abs(s) < 1024:
            return f"{s:.2f} {unit}"
        s /= 1024
    return f"{s:.2f} PB"

# ====================== 频域上采样（与 v2 对齐） ======================
def spectral_upsample(field: torch.Tensor, target_size: int = 256):
    *batch_dims, H, W = field.shape
    assert target_size >= H and target_size >= W, "目标尺寸必须大于或等于输入尺寸"
    field = field.to(device)  # 与 v2 一致，使用全局 device
    freq = torch.fft.fft2(field, norm='ortho')
    freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))
    pad_H = (target_size - H) // 2
    pad_W = (target_size - W) // 2
    new_freq_shifted = torch.zeros(
        *batch_dims, target_size, target_size,
        dtype=freq_shifted.dtype, device=freq_shifted.device
    )
    new_freq_shifted[..., pad_H:pad_H+H, pad_W:pad_W+W] = freq_shifted
    new_freq = torch.fft.ifftshift(new_freq_shifted, dim=(-2, -1))
    upsampled = torch.fft.ifft2(new_freq, norm='ortho')
    return (target_size / H) * upsampled.real

# ====================== JAX<->PyTorch 桥（与 v2 对齐，显式 DLPack） ======================
class JaxPDEWrapper(torch.autograd.Function):
    _vjp_cache = {}

    @staticmethod
    def forward(ctx, a_torch: torch.Tensor, g):
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
    def backward(ctx, grad_output: torch.Tensor):
        (a_torch,) = ctx.saved_tensors
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

# ====================== PDE 求解（函数保持与 v2 完全一致） ======================
def generate_sequence_every_second(u0: jnp.ndarray, nu: float, T_seconds: int, fixed_step: float = 0.005):
    """
    稀疏保存：只保留整秒帧。与 v2 完全一致的单样本版本：
      输入  u0: (H,W)
      返回 (T_seconds+1, H, W) —— 注意本函数末尾做了 swapaxes 与 v2 对齐。
    """
    # 坐标系对齐（与 v2）
    u0_proc = jnp.rot90(jnp.flip(u0, axis=-2), 3, axes=(-2, -1))  # (H, W)
    H, W = u0_proc.shape
    assert H == W, "目前只支持正方域网格"

    # 这里直接引用你的 Exponax 步进器（保持与 v2 一致）
    import exponax as ex
    stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
        2, 1, H, fixed_step, diffusivity=nu, order=4
    )

    steps_per_sec = int(round(1.0 / fixed_step))

    def step_once(u):
        return stepper(u[None, ...])[0]

    step_once = jax.checkpoint(step_once)

    def micro(carry, _):
        u = step_once(carry)
        return u, None

    def run_one_second(carry, _):
        u, _ = jax.lax.scan(micro, carry, None, length=steps_per_sec)
        return u, u

    _, seconds = jax.lax.scan(run_one_second, u0_proc, None, length=T_seconds)
    seq = jnp.concatenate([u0_proc[None, ...], seconds], axis=0)  # (T+1, H, W)

    # 与 v2 对齐：交换最后两维
    seq = jnp.swapaxes(seq, -1, -2)
    return seq  # (T+1, W, H)

# 批处理版本：vmap 单样本函数，得到 (T+1, B, H, W)
def _batched_generate_sequence(u0_batched: jnp.ndarray, nu: float, T_seconds: int, fixed_step: float):
    # u0_batched: (B,H,W)
    fn = lambda a: generate_sequence_every_second(a, nu, T_seconds, fixed_step)
    # 让批维作为 in_axes=0，产出 out_axes=1 以得到 (T+1,B,H,W)
    return jax.vmap(fn, in_axes=0, out_axes=1)(u0_batched)

class DifferentiablePDESolver:
    def __init__(self, nu: float, device="cuda", *, fixed_step: float = 0.005):
        self.nu = nu
        self.device = device
        self.fixed_step = fixed_step
        self.perf_records = {}

    def rollout_seconds(self, x0: torch.Tensor, T_seconds: int):
        # x0: (H,W) or (B,H,W)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        assert isinstance(x0, torch.Tensor) and x0.is_cuda

        if x0.ndim == 2:
            g = jax.jit(lambda a: generate_sequence_every_second(a, self.nu, T_seconds, self.fixed_step))
        elif x0.ndim == 3:
            g = jax.jit(lambda A: _batched_generate_sequence(A, self.nu, T_seconds, self.fixed_step))
        else:
            raise ValueError(f"x0 ndim must be 2 or 3, got {x0.ndim}")

        seq_tensor = JaxPDEWrapper.apply(x0.contiguous(), g)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        dm = (torch.cuda.memory_allocated() - m0) if torch.cuda.is_available() else 0
        self.perf_records[f"rollout_0_{T_seconds}"] = (dt, dm)
        return seq_tensor  # (T+1,H,W) or (T+1,B,H,W)

# ====================== 近似检索器（与 v2 对齐，但支持批） ======================
class ApproximatePDESolutionFinder:
    def __init__(self, device="cuda"):
        self.device = device
        self.perf_records = {}
        self.dataset_path = Path(__file__).parent.parent / "datasets" / "exponax_datasets" / "t20" / "dictionary" / \
            "dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt"
        self.dataset = torch.load(self.dataset_path, weights_only=False, map_location=device)
        self.x_dataset = self.dataset["x"].to(self.device)          # (N,256,256)
        self.y_dataset = self.dataset["y"].to(self.device)          # (N,256,256,21)

    @staticmethod
    def _mse_match_batch(x_dataset: torch.Tensor, queries: torch.Tensor):
        # x_dataset: (N,H,W), queries: (B,H,W)
        # 返回 (B,) best indices
        diffs = x_dataset.unsqueeze(0) - queries.unsqueeze(1)   # (B,N,H,W)
        mse = torch.mean(diffs**2, dim=(2,3))                   # (B,N)
        return torch.argmin(mse, dim=1)                         # (B,)

    def find(self, x0: torch.Tensor):
        # x0: (B,H,W) or (H,W)
        if x0.ndim == 2:
            x0 = x0.unsqueeze(0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        idx = self._mse_match_batch(self.x_dataset, x0)        # (B,)
        y = self.y_dataset[idx]                                # (B,256,256,21)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        dm = (torch.cuda.memory_allocated() - m0) if torch.cuda.is_available() else 0
        self.perf_records["all 0-10 and 19 secs"] = (dt, dm)

        input_seq = torch.cat([x0.unsqueeze(-1), y[..., 1:10]], dim=-1)  # (B,H,W,10)
        true19 = y[..., 19]                                              # (B,H,W)
        return input_seq, true19

    def get_perf_records(self):
        return self.perf_records

# ====================== 工具函数：单样本 MSE（保存时用） ======================
### CHANGED: 新增 — 用于 per-sample 现算 loss / surrogate_loss
def _per_sample_mse(a: np.ndarray, b: np.ndarray, reduction: str = "sum") -> float:
    """
    以 numpy 计算单个样本像素级 MSE。
    reduction = "sum"  -> SSE（像素求和）
    reduction = "mean" -> MSE（像素平均）
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    se = (a - b) ** 2
    return float(se.sum() if reduction == "sum" else se.mean())

# ====================== 记录器（保持与 v2 一致的接口与字段） ======================
class AttackRecorder:
    def __init__(self):
        self.steps = []
        self.metadata = {}

    def record_step(self, *,
                    step,
                    x0,
                    output,
                    truth=None,
                    surrogate_truth=None,
                    loss=None,
                    surrogate_loss=None,
                    gradient=None,
                    numerical_gradient=None,
                    reach_boundary=False,
                    scaled_gradient=None):
        rec = {
            "step": step,
            "x0": x0.detach().cpu().numpy().copy(),
            "output": output.detach().cpu().numpy().copy(),
            "reach_boundary": reach_boundary,
        }
        if truth is not None:
            rec["truth"] = truth.detach().cpu().numpy().copy()
        if surrogate_truth is not None:
            rec["surrogate_truth"] = surrogate_truth.detach().cpu().numpy().copy()
        if loss is not None:
            rec["loss"] = float(loss.item() if torch.is_tensor(loss) else loss)
        if surrogate_loss is not None:
            rec["surrogate_loss"] = float(surrogate_loss.item() if torch.is_tensor(surrogate_loss) else surrogate_loss)
        if gradient is not None:
            rec["gradient"] = gradient.detach().cpu().numpy().copy()
        if scaled_gradient is not None:
            rec["scaled_gradient"] = scaled_gradient.detach().cpu().numpy().copy()
        if numerical_gradient is not None:
            rec["numerical_gradient"] = numerical_gradient.detach().cpu().numpy().copy()
        self.steps.append(rec)

    def save(self, filename: str, metadata=None):
        if metadata:
            self.metadata.update(metadata)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump({"steps": self.steps, "metadata": self.metadata}, f)

    ### CHANGED: 仅此函数逻辑更新 —— 保存 per-sample PKL 时现算每个样本自己的 loss
    def save_per_sample_pkls(self, out_dir: Path, global_ids, *, loss_reduction: str = "sum"):
        """
        将批记录拆成 per-sample PKL。
        与原先不同点：
          - 不再把 batch 的 scalar loss 原封不动写入每个样本，
          - 而是用该样本的 output 与 truth/surrogate_truth 现算每个样本自己的 loss。
            * loss: 优先与 truth 比较；无 truth 时与 surrogate_truth 比较
            * surrogate_loss: 仅当 surrogate_truth 存在时计算
          - reduction 可选 "sum"/"mean"，默认 "sum"（对齐反传时 batch-sum 的语义）
        """
        os.makedirs(out_dir, exist_ok=True)
        if len(self.steps) == 0:
            return
        any_step = self.steps[0]
        arr = any_step["x0"]
        assert isinstance(arr, np.ndarray) and arr.ndim >= 3, "expect batch-arrays recorded"
        B = arr.shape[0]
        assert len(global_ids) == B, "global_ids length must match batch size"

        for b in range(B):
            rec_b = {"steps": [], "metadata": dict(self.metadata)}
            rec_b["metadata"]["global_id"] = int(global_ids[b])
            # 可选：记录 per-sample 保存的 reduction 方式
            rec_b["metadata"]["per_sample_loss_reduction"] = str(loss_reduction)

            for step_rec in self.steps:
                step_b = {"step": step_rec["step"], "reach_boundary": step_rec.get("reach_boundary", False)}

                # —— 逐键切片保存（含 x0/output/truth/surrogate_truth/gradient/...）
                for key in [
                    "x0", "output", "truth", "surrogate_truth",
                    "gradient", "scaled_gradient", "numerical_gradient"
                ]:
                    if key in step_rec:
                        arrk = step_rec[key]
                        # 仅当该字段是批数组且首维对齐 B 时才切片
                        if isinstance(arrk, np.ndarray) and arrk.shape[0] == B:
                            step_b[key] = arrk[b].copy()

                # —— 现算每个样本自己的 loss / surrogate_loss（不再拷贝 batch scalar）
                out_b  = step_b.get("output", None)
                tru_b  = step_b.get("truth", None)
                surr_b = step_b.get("surrogate_truth", None)

                # loss: 优先与 truth 比较，若无 truth 则与 surrogate_truth 比较
                if out_b is not None and tru_b is not None:
                    step_b["loss"] = _per_sample_mse(out_b, tru_b, reduction=loss_reduction)
                elif out_b is not None and surr_b is not None:
                    step_b["loss"] = _per_sample_mse(out_b, surr_b, reduction=loss_reduction)

                # surrogate_loss: 仅当 surrogate_truth 存在时计算
                if out_b is not None and surr_b is not None:
                    step_b["surrogate_loss"] = _per_sample_mse(out_b, surr_b, reduction=loss_reduction)

                rec_b["steps"].append(step_b)

            # 写 per-sample 文件
            with open(out_dir / f"sample_{int(global_ids[b]):06d}.pkl", "wb") as f:
                pickle.dump(rec_b, f)

    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        rec = AttackRecorder()
        rec.steps = data["steps"]
        rec.metadata = data["metadata"]
        return rec

# ====================== 主系统（与 v2 语义对齐；仅扩展为批） ======================
class PDEAttackSystem:
    def __init__(self, recurrent_model: torch.nn.Module, nu=1e-5, device="cuda", mode_spec="wwwwwwwwww"):
        self.recurrent_model = recurrent_model.to(device)
        self.nu = nu
        self.device = device
        self.mode_spec = mode_spec

        assert isinstance(self.mode_spec, str) and len(self.mode_spec) == 10
        assert set(self.mode_spec).issubset(set("adw"))

        for p in self.recurrent_model.parameters():
            p.requires_grad = False
        self.recurrent_model.eval()

        self.pde_solver = DifferentiablePDESolver(nu, device)
        self.pde_sol_finder = ApproximatePDESolutionFinder(device)

        self.model_perf = (0.0, 0)
        self.pde_perf = (0.0, 0)
        self.pde_records_0_to_9 = {}
        self.pde_records_19 = {}

    @staticmethod
    def _format_memory(size):
        return _fmt_bytes(size)

    def forward(self, x0: torch.Tensor):
        # x0: (B,H,W) or (H,W) → 内部统一 (B,H,W)
        if x0.ndim == 2:
            x0 = x0.unsqueeze(0)
        B, H, W = x0.shape

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_t0 = time.perf_counter()
        model_m0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # 需要 PDE 的秒数
        need_solver_seconds = set()
        for t in range(1, 10):
            if self.mode_spec[t-1] in ("w", "d"):
                need_solver_seconds.add(t)
        ch19 = self.mode_spec[9]
        if ch19 in ("w", "d"):
            need_solver_seconds.add(19)

        # 近似路径（若存在 'a'）
        input_sequence_approx, true19_approx = None, None
        if 'a' in self.mode_spec:
            input_sequence_approx, true19_approx = self.pde_sol_finder.find(x0)

        # 一次性 PDE rollout 到所需最大秒数
        seq_0_to_T = None
        pde_duration = 0.0
        pde_mem = 0
        if len(need_solver_seconds) > 0:
            T_required = max(need_solver_seconds)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pde_t0 = time.perf_counter()
            pde_m0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            seq_0_to_T = self.pde_solver.rollout_seconds(x0, T_required)  # (T+1,B,H,W)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pde_duration = time.perf_counter() - pde_t0
            pde_mem = (torch.cuda.memory_allocated() - pde_m0) if torch.cuda.is_available() else 0

        # 组装 0..9 帧 → (B,H,W,10)
        frames = [x0]
        for t in range(1, 10):
            ch = self.mode_spec[t-1]
            if ch == 'a':
                assert input_sequence_approx is not None
                f = input_sequence_approx[..., t]
            else:
                assert seq_0_to_T is not None and (seq_0_to_T.shape[0]-1) >= t
                f = seq_0_to_T[t]
                if ch == 'd':
                    f = f.detach()
            frames.append(f)
        input_batch = torch.stack(frames, dim=-1)  # (B,H,W,10)

        # FNO 预测 10..19，取 19
        predictions = self.recurrent_model(input_batch)   # (B,H,W,10)
        pred_19 = predictions[..., -1]                    # (B,H,W)

        # true_19（第10位）
        if ch19 == 'a':
            assert true19_approx is not None
            true_19 = true19_approx
        else:
            assert seq_0_to_T is not None and (seq_0_to_T.shape[0]-1) >= 19
            true_19 = seq_0_to_T[19]
            if ch19 == 'd':
                true_19 = true_19.detach()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_dt = time.perf_counter() - model_t0
        model_dm = (torch.cuda.memory_allocated() - model_m0) if torch.cuda.is_available() else 0

        self.model_perf = (model_dt, model_dm)
        self.pde_perf = (pde_duration, pde_mem)
        self.pde_records_0_to_9 = {}  # 统一使用聚合统计（与 v2 的 print 格式一致）
        self.pde_records_19 = {19: (pde_duration, pde_mem)} if 19 in need_solver_seconds else {}
        return pred_19, true_19

    def print_detailed_perf(self):
        print("\n=== 详细性能报告 ===")
        model_duration, model_mem = self.model_perf
        print(f"[模型部分] 总时间: {model_duration:.4f}s, 显存: {self._format_memory(model_mem)}")
        print("\n[19秒 PDE求解]")
        for t, (dur, mem) in self.pde_records_19.items():
            print(f"  t={t}s - 时间: {dur:.6f}s | 显存: {self._format_memory(mem)}")
        pde_duration, pde_mem = self.pde_perf
        print(f"\n[PDE求解部分] 总时间: {pde_duration:.4f}s, 显存: {self._format_memory(pde_mem)}")
        total = model_duration + pde_duration
        print(f"\n[总计] 时间: {total:.4f}s")

    def pgd_attack_adam(self,
                        initial_x0: torch.Tensor,
                        *,
                        epsilon=0.1,
                        alpha=0.01,
                        num_steps=10,
                        norm="inf",
                        idx_tag="idx",
                        beta1=0.9,
                        beta2=0.999,
                        adam_eps=1e-8,
                        use_sign_for_linf=True,
                        amsgrad=False,
                        global_ids=None):
        recorder = AttackRecorder()
        x_adv = initial_x0.clone().detach().to(self.device)
        if x_adv.ndim == 2:
            x_adv = x_adv.unsqueeze(0)
        x_adv.requires_grad_(True)
        original_x0 = x_adv.clone().detach()

        # Adam state
        m = torch.zeros_like(x_adv)
        v = torch.zeros_like(x_adv)
        vhat_max = torch.zeros_like(x_adv) if amsgrad else None
        t_adam = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for step in range(num_steps):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            step_t0 = time.perf_counter()
            step_m0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            # ---- Forward ----
            pred_19, true_19 = self.forward(x_adv)

            model_duration, model_mem = self.model_perf
            pde_duration, pde_mem = self.pde_perf
            print(f"\n[Step {step}] ============ 详细性能分析 ============")
            print(f"[模型部分] 总时间: {model_duration:.6f}s | 显存: {self._format_memory(model_mem)}")
            print(f"[PDE 部分] 总时间: {pde_duration:.6f}s | 显存: {self._format_memory(pde_mem)}")

            # ---- Loss（与 v2 一致：对 forward.true_19 做 surrogate 反传）----
            surrogate_loss = F.mse_loss(pred_19, true_19, reduction="sum")
            print("surrogate_loss (vs forward.true_19): ", float(surrogate_loss.item()))

            if DEBUG_GRAD:
                g_tmp = torch.autograd.grad(surrogate_loss, x_adv, retain_graph=True, allow_unused=True)[0]
                print("||grad(surrogate_loss, x_adv)|| =", None if g_tmp is None else float(g_tmp.norm().item()))

            # 若末位 a：仅用于记录（no-grad 跑真实 PDE@t19）
            real_true_19 = None
            real_loss = None
            is_a19 = (self.mode_spec[9] == 'a')
            if is_a19:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    real_true_19 = self.pde_solver.rollout_seconds(x_adv, 19)[19]
                    real_loss = F.mse_loss(pred_19, real_true_19, reduction="sum")
                    print("real_loss (compare-to-PDE @t19):", float(real_loss.item()))

            # ---- Backward ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            bwd_t0 = time.perf_counter()
            bwd_m0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            surrogate_loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[BWD ] Time: {time.perf_counter()-bwd_t0:.6f}s | Mem: {self._format_memory((torch.cuda.memory_allocated()-bwd_m0) if torch.cuda.is_available() else 0)}")

            # ---- PGD + Adam 更新（方向与 v2 一致；对整个 batch 张量做 L2 归一）----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            upd_t0 = time.perf_counter()
            upd_m0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            with torch.no_grad():
                t_adam += 1
                g = x_adv.grad
                m = beta1 * m + (1.0 - beta1) * g
                v = beta2 * v + (1.0 - beta2) * (g * g)
                m_hat = m / (1.0 - (beta1 ** t_adam))
                v_hat = v / (1.0 - (beta2 ** t_adam))
                if amsgrad:
                    vhat_max = torch.maximum(vhat_max, v_hat)
                    denom = vhat_max.sqrt() + adam_eps
                else:
                    denom = v_hat.sqrt() + adam_eps
                d_t = m_hat / denom

                delta = x_adv - original_x0
                if norm in ('inf', 'Linf', '∞'):
                    direction = torch.sign(d_t) if use_sign_for_linf else d_t
                    x_adv.data += alpha * direction
                    current_delta = x_adv - original_x0
                    # 与 v2 保持：边界标记只作布尔参考，不统计百分比
                    at_upper = (current_delta >= epsilon - 1e-6)
                    at_lower = (current_delta <= -epsilon + 1e-6)
                    reach_boundary = bool((at_upper | at_lower).any().item())
                    x_adv.data = original_x0 + torch.clamp(current_delta, -epsilon, epsilon)
                elif norm in (2, '2', 'L2'):
                    # **按样本独立归一/投影**：用每个 record 自己的 L2 范数
                    B = x_adv.shape[0]
                    flat = d_t.view(B, -1)
                    flat_norm = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True) + 1e-12
                    direction = (flat / flat_norm).view_as(d_t)
                    x_adv.data += alpha * direction
                    current_delta = x_adv - original_x0
                    flat_delta = current_delta.view(B, -1)
                    delta_norm = torch.linalg.vector_norm(flat_delta, ord=2, dim=1, keepdim=True)
                    scale = torch.clamp(epsilon / (delta_norm + 1e-12), max=1.0)
                    x_adv.data = (original_x0 + (flat_delta * scale).view_as(current_delta))
                    reach_boundary = bool((delta_norm > (epsilon - 1e-6)).any().item())
                else:
                    raise ValueError(f"Unsupported norm: {norm}")

                # 记录（与 v2 相同字段；数组含批维）
                if is_a19:
                    recorder.record_step(
                        step=step,
                        x0=x_adv.clone().detach(),
                        output=pred_19.clone().detach(),
                        truth=real_true_19.clone().detach() if real_true_19 is not None else None,
                        surrogate_truth=true_19.clone().detach(),
                        loss=real_loss if real_loss is not None else float('nan'),
                        surrogate_loss=surrogate_loss,
                        gradient=x_adv.grad.clone().detach(),
                        numerical_gradient=None,
                        reach_boundary=reach_boundary,
                        scaled_gradient=d_t.clone().detach(),
                    )
                else:
                    recorder.record_step(
                        step=step,
                        x0=x_adv.clone().detach(),
                        output=pred_19.clone().detach(),
                        truth=true_19.clone().detach(),
                        surrogate_truth=None,
                        loss=surrogate_loss,
                        surrogate_loss=None,
                        gradient=x_adv.grad.clone().detach(),
                        numerical_gradient=None,
                        reach_boundary=reach_boundary,
                        scaled_gradient=d_t.clone().detach(),
                    )
                x_adv.grad.zero_()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[UPDT] Time: {time.perf_counter()-upd_t0:.6f}s | Mem: {self._format_memory((torch.cuda.memory_allocated()-upd_m0) if torch.cuda.is_available() else 0)}")

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            else:
                peak = 0
            step_dt = time.perf_counter() - step_t0
            step_dm = (torch.cuda.memory_allocated() - step_m0) if torch.cuda.is_available() else 0
            print(f"[STEP] Total: {step_dt:.6f}s | ΔMem: {self._format_memory(step_dm)} | Peak: {self._format_memory(peak)}")

        # 保存（沿用 v2 的单 pkl 方案；idx_tag 允许批范围）
        mode_tag = getattr(self, "mode_spec", "undefined")
        run_dir = Path(f"perturbation_results_batch{BATCH_SIZE}") / (
            f"modespec={mode_tag}_norm{norm}_alpha{alpha}_epsilon{epsilon}_steps{num_steps}_{idx_tag}"
        )
        os.makedirs(run_dir, exist_ok=True)
        meta = {
            "mode_spec": mode_tag,
            "norm": norm,
            "epsilon": epsilon,
            "alpha": alpha,
            "num_steps": num_steps,
            "index": idx_tag,
            "adam_beta1": beta1,
            "adam_beta2": beta2,
            "adam_eps": adam_eps,
            "use_sign_for_linf": use_sign_for_linf,
            "amsgrad": amsgrad,
            "global_ids": list(map(int, global_ids)) if global_ids is not None else None,
        }
        recorder.save(str(run_dir / "batch.pkl"), metadata=meta)
        if global_ids is not None:
            # 显式指定 per-sample loss 的 reduction（与反传的 batch-sum 对齐）
            ### CHANGED: 这里传 loss_reduction="sum"
            recorder.save_per_sample_pkls(run_dir / "per_sample", global_ids, loss_reduction="sum")
        return x_adv

# ====================== 入口（批量切片，但其余流程对齐 v2） ======================
if __name__ == "__main__":
    T = 10
    step = 1
    train_test = "test"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_test = torch.load(
        f'{script_dir}/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_{train_test}_all_frames.pt',
        weights_only=False, map_location=device
    )

    model_instance = FNO2d(modes1=64, modes2=64, width=60, in_channels=10)
    state_dict = torch.load(
        f'{script_dir}/saved_models/2D/modes64_width60_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth',
        map_location=device
    )
    model_instance.load_state_dict(state_dict)
    model_instance.eval()

    size = 256
    ratio = 100

    DEFAULT_ADAM = {
        "beta1": 0.9,
        "beta2": 0.999,
        "adam_eps": 1e-8,
        "use_sign_for_linf": True,
        "amsgrad": False,
    }

    input_list = [
        {"norm": 2, "epsilon": 0.0002*size**2, "alpha": 0.01*ratio,  "num_steps": 100},
        {"norm": 2, "epsilon": 0.0006*size**2, "alpha": 0.01*ratio,  "num_steps": 100},
        {"norm": 2, "epsilon": 0.0012*size**2, "alpha": 0.01*ratio,  "num_steps": 100},
    ]
    print("input_list:", input_list)

    MODE_SPEC = os.environ.get("MODE_SPEC", "wwwwwwwwww")
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "7"))

    X_all = data_test["x"]  # (N,256,256)
    N_total = min(X_all.shape[0], 20)

    for cfg in input_list:
        recurrent_model = RecurrentPredictor(model_instance, T_out=T, step=step).to(device)
        attack_system = PDEAttackSystem(
            recurrent_model=recurrent_model,
            nu=1e-5,
            device=device,
            mode_spec=MODE_SPEC,
        )

        norm      = cfg.get("norm", "inf")
        epsilon   = cfg.get("epsilon", 0.1)
        alpha     = cfg.get("alpha", 0.01)
        num_steps = cfg.get("num_steps", 10)

        beta1            = cfg.get("beta1", DEFAULT_ADAM["beta1"])
        beta2            = cfg.get("beta2", DEFAULT_ADAM["beta2"])
        adam_eps         = cfg.get("adam_eps", DEFAULT_ADAM["adam_eps"])
        use_sign_for_linf= cfg.get("use_sign_for_linf", DEFAULT_ADAM["use_sign_for_linf"])
        amsgrad          = cfg.get("amsgrad", DEFAULT_ADAM["amsgrad"])

        print("="*80)
        print(
            "norm: {norm}, epsilon: {epsilon}, alpha: {alpha}, num_steps: {num_steps}, "
            "mode_spec: {mode_spec}, b1: {b1}, b2: {b2}, adam_eps: {ae}, sign_Linf: {sgn}, amsgrad: {ams}".format(
                norm=norm, epsilon=epsilon, alpha=alpha, num_steps=num_steps,
                mode_spec=MODE_SPEC, b1=beta1, b2=beta2, ae=adam_eps, sgn=use_sign_for_linf, ams=amsgrad)
        )

        # 批量切片：保持“同时输入/输出经 solver 与 FNO”的唯一差异
        for start in range(0, N_total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_total)
            x_batch = X_all[start:end].to(device)
            print("x_batch.shape:", tuple(x_batch.shape), "batch start:", start, "end:", end)
            x_batch = spectral_upsample(x_batch, 256).to(device)
            idx_tag = f"idx{start}-{end-1}"

            global_ids = list(range(start, end))
            _ = attack_system.pgd_attack_adam(
                initial_x0=x_batch,
                epsilon=epsilon,
                alpha=alpha,
                num_steps=num_steps,
                norm=norm,
                idx_tag=idx_tag,
                beta1=beta1,
                beta2=beta2,
                adam_eps=adam_eps,
                use_sign_for_linf=use_sign_for_linf,
                amsgrad=amsgrad,
                global_ids=global_ids,
            )

            # 清理显存
            del x_batch
            torch.cuda.empty_cache()
            gc.collect()

        # 本组结束清理
        del recurrent_model, attack_system
        torch.cuda.empty_cache()
        gc.collect()

    print("PGD 批量攻击完成！")








