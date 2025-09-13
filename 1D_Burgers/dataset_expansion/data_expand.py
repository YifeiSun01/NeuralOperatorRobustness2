#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PGD attacks on an entire 1D Burgers dataset (nu=0.0005), final-time only.

- 只做 with-solver：loss = || G(a) - g(a, t_final) ||^2
- PDE 只取最终时刻 t_final
- 输出文件结构与原始数据完全一致：keys = ['x','y','t_final']
  其中 x: (N, X) 为扰动后的输入，y: (N, X) 为 PDE 最后一帧
- 支持多组 --inputs="norm,epsilon,steps,alpha"（可重复）
- tqdm 进度条 + 每组组合耗时 + 总耗时
"""

import os, sys, time, subprocess
from pathlib import Path
from datetime import datetime

# --- 让 JAX 用 CUDA，避免 TPU 提示 ---
os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")

from absl import app, flags
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
from tqdm import tqdm

# 项目内导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO1d import FNO1d
from solvers.burgers1d_solvers import (
    ExponaxBurgersSolver1D,
    SciPyBurgersSolver1D,
    SciPySpectralBurgersSolver1D,
    PhiFlowBurgersSolver1D,
)

# JAX 用 float32（你的模型是 float32）
jax.config.update("jax_enable_x64", False)

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_pt", None, "路径：源数据 .pt（包含 'x' 与 'y'）")
flags.DEFINE_string("model_pth", None, "路径：FNO1d 权重 .pth")
flags.DEFINE_string("solver", "exponax", "exponax|scipy|scipy_spectral|phiflow")
flags.DEFINE_float("nu", 0.0005, "Burgers 粘性系数 nu")
flags.DEFINE_float("t_final", 1.0, "终止时间")
flags.DEFINE_float("dt", 0.001, "内部 PDE 步长")
flags.DEFINE_string("out_root", "./datasets/1D/Burgers/expanded/t1", "输出根目录")
# 可重复：--inputs="inf,0.1,100,0.001" --inputs="2,20,100,0.1"
flags.DEFINE_multi_string("inputs", None, "重复参数：norm,epsilon,steps,alpha  如 '2,10,100,0.01'")

def get_gpu_info():
    try:
        part = os.environ.get("SLURM_JOB_PARTITION", "N/A")
        node = os.environ.get("SLURMD_NODENAME", "N/A")
        gvis = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")
        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
                encoding="utf-8"
            ).strip()
        except FileNotFoundError:
            smi = "nvidia-smi not found"
        print("========== GPU ==========")
        print(f"Partition: {part}")
        print(f"Node: {node}")
        print(f"CUDA_VISIBLE_DEVICES: {gvis}")
        print("GPU Details:\n" + smi)
        print("=========================")
    except Exception as e:
        print(f"GPU info error: {e}")

class JaxPDEWrapper(torch.autograd.Function):
    """把 JAX 的 g(a) 包成 Torch 可反传算子（通过 vjp）"""
    _vjp_cache = {}

    @staticmethod
    def forward(ctx, a_torch, g_callable):
        if not a_torch.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        a_dl = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
        a_jax = jax.dlpack.from_dlpack(a_dl)
        out_jax = g_callable(a_jax)          # 这里直接返回最终帧 (X,)
        out_dl = jax.dlpack.to_dlpack(out_jax)
        out_t = torch.utils.dlpack.from_dlpack(out_dl)
        ctx.save_for_backward(a_torch)
        ctx.g = g_callable
        return out_t

    @staticmethod
    def backward(ctx, grad_output):
        (a_torch,) = ctx.saved_tensors
        g = ctx.g
        key = id(g)
        if key not in JaxPDEWrapper._vjp_cache:
            def vjp_fn(a_jax, grad_jax):
                _, vjp = jax.vjp(g, a_jax)
                return vjp(grad_jax)
            JaxPDEWrapper._vjp_cache[key] = jax.jit(vjp_fn)
        vjp_jit = JaxPDEWrapper._vjp_cache[key]
        grad_dl = torch.utils.dlpack.to_dlpack(grad_output.contiguous())
        grad_jax = jax.dlpack.from_dlpack(grad_dl)
        a_dl = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
        a_jax = jax.dlpack.from_dlpack(a_dl)
        grad_in_jax, = vjp_jit(a_jax, grad_jax)
        grad_in_dl = jax.dlpack.to_dlpack(grad_in_jax)
        return torch.utils.dlpack.from_dlpack(grad_in_dl), None

class PDEAttackSystem1D:
    """只做 with-solver、最终时刻的 PGD"""
    def __init__(self, model_G: torch.nn.Module, g_callable, device: torch.device):
        self.G = model_G.eval()
        for p in self.G.parameters():
            p.requires_grad = False
        self.g = g_callable
        self.device = device

    def build_model_input(self, a: torch.Tensor) -> torch.Tensor:
        # FNO1d 期望 (B, X, 1)
        if a.dim() == 1:
            a = a.unsqueeze(0).unsqueeze(-1)  # (1, X, 1)
        elif a.dim() == 2:
            a = a.unsqueeze(-1)               # (B, X, 1)
        return a

    def forward_G(self, a: torch.Tensor) -> torch.Tensor:
        return self.G(self.build_model_input(a))  # -> (1, X, 1)

    def forward_true_final(self, a: torch.Tensor) -> torch.Tensor:
        # 返回最终帧 (X,)
        return JaxPDEWrapper.apply(a, self.g)

    def pgd(self, a0: torch.Tensor, epsilon: float, alpha: float, steps: int, norm: str = "2"):
        a0 = a0.detach().to(self.device)          # 原始输入
        delta = torch.zeros_like(a0, requires_grad=True)
        for _ in range(steps):
            if delta.grad is not None:
                delta.grad.zero_()
            a = a0 + delta

            pred = self.forward_G(a)              # (1, X, 1)
            true_final = self.forward_true_final(a)   # (X,)

            if pred.dim() == 3:
                pred = pred[0]
            pred = pred.squeeze(-1)               # (X,)

            loss = F.mse_loss(pred, true_final, reduction="sum")
            loss.backward()

            with torch.no_grad():
                g = delta.grad
                if str(norm).lower() in ("inf", "linf", "∞"):
                    delta += alpha * torch.sign(g)
                    delta.clamp_(-epsilon, epsilon)
                else:  # L2
                    g_norm = torch.norm(g, p=2) + 1e-12
                    delta += alpha * g / g_norm
                    d_norm = torch.norm(delta, p=2)
                    if d_norm > epsilon:
                        delta.mul_(epsilon / d_norm)
        return (a0 + delta).detach()

    @torch.no_grad()
    def rollout_final(self, a: torch.Tensor) -> torch.Tensor:
        return self.forward_true_final(a)         # (X,)

def parse_inputs_list(inputs_flag_values):
    """解析重复 --inputs 'norm,epsilon,steps,alpha'"""
    if not inputs_flag_values:
        return []
    combos = []
    for s in inputs_flag_values:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) != 4:
            raise ValueError(f"--inputs expects 4 comma-separated values, got: {s}")
        norm_s, eps_s, steps_s, alpha_s = parts
        norm_val = norm_s.lower()
        if norm_val in ("linf", "inf", "∞"):
            norm_val = "inf"
        elif norm_val in ("2", "2.0"):
            norm_val = "2"
        else:
            try:
                if int(float(norm_s)) == 2:
                    norm_val = "2"
            except Exception:
                pass
        epsilon = float(eps_s)
        steps = int(float(steps_s))
        alpha = float(alpha_s)
        combos.append((norm_val, epsilon, steps, alpha))
    return combos

def fmt_float(v: float) -> str:
    return f"{v:.6g}"

def main(_):
    assert FLAGS.dataset_pt and FLAGS.model_pth, "--dataset_pt 与 --model_pth 必填"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    get_gpu_info()

    # 读入原始数据（需要 weights_only=False）
    ds_path = Path(FLAGS.dataset_pt).resolve()
    data = torch.load(ds_path, map_location=device, weights_only=False)
    assert "x" in data and "y" in data, "Dataset must have keys 'x' and 'y'"
    x_src = data["x"].to(device)   # (N, X) 或 (N, X, 1)
    y_src = data["y"].to(device)   # 仅用于尺寸与 N,X；不会参与损失
    N = x_src.shape[0]
    X = x_src.shape[1] if x_src.dim() >= 2 else x_src.shape[-1]

    # 模型
    model = FNO1d(modes=16, width=64)
    model.load_state_dict(torch.load(FLAGS.model_pth, map_location=device, weights_only=True))
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # 选择求解器
    if FLAGS.solver == "exponax":
        solver = ExponaxBurgersSolver1D(X, nu=FLAGS.nu, bc="periodic", xlim=(0, 1))
    elif FLAGS.solver == "scipy":
        solver = SciPyBurgersSolver1D(X, nu=FLAGS.nu, bc="periodic")
    elif FLAGS.solver == "scipy_spectral":
        solver = SciPySpectralBurgersSolver1D(X, nu=FLAGS.nu, bc="periodic")
    elif FLAGS.solver == "phiflow":
        solver = PhiFlowBurgersSolver1D(X, nu=FLAGS.nu, bc="periodic")
    else:
        raise ValueError("Unknown solver")

    # 只取端点 [0, t_final]，并且在 JIT 内部用 [1][1] 取最终帧，避免 ndim/dict 分歧
    t_span = (0.0, float(FLAGS.t_final))

    @jax.jit
    def g_callable(u0: jnp.ndarray):
        # solve(...) -> (times, states)，取最终帧
        return solver.solve(
            u0.astype(jnp.float32),
            t_final=float(FLAGS.t_final),
            t_eval=t_span,
            step=float(FLAGS.dt),
        )[1][1]  # (X,)

    system = PDEAttackSystem1D(model, g_callable, device)

    # 解析输入组合
    combos = parse_inputs_list(FLAGS.inputs)
    if not combos:
        raise ValueError("请至少提供一组 --inputs='norm,epsilon,steps,alpha'")

    total_combos = len(combos)
    overall_start = time.perf_counter()
    datasets_done = 0
    samples_done_total = 0
    samples_total_all = N * total_combos

    print("=" * 80)
    print(f"Dataset N={N}, X={X}; nu={FLAGS.nu}, solver={FLAGS.solver} (final-only)")
    sys.stdout.flush()

    # 外层：参数组合进度
    for (norm, epsilon, steps, alpha) in tqdm(combos, desc="Param combos", position=0, dynamic_ncols=True):
        combo_start = time.perf_counter()

        # 分配输出
        x_out = torch.empty_like(x_src[..., 0] if x_src.dim() == 3 else x_src)  # (N, X)
        y_out = torch.empty((N, X), dtype=torch.float32, device=device)         # (N, X)

        # 内层：样本进度
        last_loss = float('nan')
        with tqdm(total=N,
                  desc=f"Samples (norm={norm}, eps={epsilon}, alpha={alpha}, steps={steps})",
                  position=1,
                  leave=False,
                  dynamic_ncols=True) as pbar:
            for i in range(N):
                a0 = x_src[i]
                if a0.dim() == 2:  # (X, C) with C==1
                    a0 = a0[..., 0]
                a0 = a0.contiguous()

                a_adv = system.pgd(a0, epsilon=epsilon, alpha=alpha, steps=steps, norm=norm)

                with torch.no_grad():
                    y_final = system.rollout_final(a_adv)  # (X,)
                    pred = system.forward_G(a_adv)
                    if pred.dim() == 3:
                        pred = pred[0]
                    pred = pred.squeeze(-1)               # (X,)
                    loss = F.mse_loss(pred, y_final, reduction="sum")
                    last_loss = float(loss.item())

                x_out[i] = a_adv.detach()
                y_out[i] = y_final.detach().float()

                pbar.set_postfix({"loss": f"{last_loss:.3e}", "done": f"{i+1}/{N}"})
                pbar.update(1)
                samples_done_total += 1
                torch.cuda.empty_cache()

        # —— 保存：和原始数据完全一致的键与形状 ——
        out = {
            "x": x_out.detach().float().cpu(),                                 # (N, X)
            "y": y_out.detach().float().cpu(),                                 # (N, X) 最终帧
            "t_final": float(data.get("t_final", float(FLAGS.t_final))),       # 保留原 t_final 或用 FLAGS.t_final
        }

        save_root = (
            Path(FLAGS.out_root)
            / f"nu{fmt_float(float(FLAGS.nu))}_pgd"
            / f"norm{str(norm).lower()}_eps{fmt_float(float(epsilon))}_alpha{fmt_float(float(alpha))}_steps{int(steps)}"
            / f"N={N}"
        )
        save_root.mkdir(parents=True, exist_ok=True)
        out_path = save_root / "dataset.pt"
        torch.save(out, out_path)

        combo_time = time.perf_counter() - combo_start
        datasets_done += 1
        print(f"\n✅ Saved: {out_path}")
        print(f"⏱️  Combo time: {combo_time:.2f}s | Done {datasets_done}/{total_combos} combos | "
              f"Samples: {samples_done_total}/{samples_total_all}")
        sys.stdout.flush()

    total_time = time.perf_counter() - overall_start
    print("\n" + "=" * 80)
    print(f"🎉 All done. Total combos: {total_combos}, Total samples: {samples_total_all}")
    print(f"🧭 Total wall time: {total_time:.2f} seconds")
    print("=" * 80)
    sys.stdout.flush()

if __name__ == "__main__":
    app.run(main)


