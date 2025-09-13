
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
        # a_torch = torch.ascontiguous_tensor(a_torch, memory_format=torch.contiguous_format)
        # a_dlpack = torch.utils.dlpack.to_dlpack(a_torch)
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
        
        print("shape:", x0.shape, "stride:", x0.stride(), "is_contig:", x0.is_contiguous())

        seq_tensor = JaxPDEWrapper.apply(x0.contiguous(), g)
        # x0 = torch.ascontiguous_tensor(x0, memory_format=torch.contiguous_format)
        # seq_tensor = JaxPDEWrapper.apply(x0, g)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        dm = (torch.cuda.memory_allocated() - m0) if torch.cuda.is_available() else 0
        self.perf_records[f"rollout_0_{T_seconds}"] = (dt, dm)
        return seq_tensor  # (T+1,H,W) or (T+1,B,H,W)