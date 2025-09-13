#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch PGD attack for 2D NS (external forcing patterns) with explicit CLI:

Required:
  --dataset_pt=...pt          # 输入数据集路径
  --model_pth=...pth          # FNO2d 权重路径
  --forcing_pattern=ringsCos  # solver 外力图案字符串
  --save_to=...               # 输出 .pt 文件路径，或目录（目录时自动命名）

Only an outer tqdm progress bar is shown (pattern/batch/samples/now/ETA).
"""

import os, sys, time, glob, gc, warnings, subprocess
from pathlib import Path
from datetime import datetime, timedelta

os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")  # JAX on CUDA if available

from absl import app, flags
FLAGS = flags.FLAGS

# ---- 必填接口 ----
flags.DEFINE_string('dataset_pt', None, 'Path to input dataset .pt (must contain keys x,y).')
flags.DEFINE_string('model_pth',  None, 'Path to FNO2d weights .pth.')
flags.DEFINE_enum(  'forcing_pattern', None,
                   ['ringsCos','sBands','isoCircles','petals','ringsL1','ringsLinf'],
                   'Forcing pattern string used by the PDE solver.')
flags.DEFINE_string('save_to',   None, 'Output .pt filepath OR a directory; dir -> auto filename.')

# ---- 其它可调 ----
flags.DEFINE_integer('batch_size', 8, 'Batch size for PGD.')
flags.DEFINE_string('mode_spec', 'wwwwwwwwww', '10-char mode_spec using a/d/w.')
flags.DEFINE_enum('norm', '2', ['2','inf'], 'PGD norm (L2 or Linf).')
flags.DEFINE_float('eps_mult', 0.0002, 'epsilon multiplier: epsilon = eps_mult * size^2')
flags.DEFINE_float('alpha_mult', 0.01,  'alpha multiplier: alpha = alpha_mult * ratio (ratio=100)')
flags.DEFINE_integer('num_steps', 100,   'PGD steps')

# PDE/rollout
flags.DEFINE_float('nu', 1e-5, 'Viscosity.')
flags.DEFINE_integer('T_seconds', 20, 'Total seconds to rollout (20 -> 21 frames).')
flags.DEFINE_float('fixed_step', 0.005, 'Internal PDE step; integral-seconds saved.')

import torch
import torch.nn.functional as F
from tqdm import tqdm
import jax, jax.numpy as jnp, jaxlib

# 项目内导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.FNO2d import FNO2d, RecurrentPredictor

# ---------------- DLPack 桥（JAX<->PyTorch） ----------------
# 参考官方 API：jax.dlpack.{from_dlpack,to_dlpack}, torch.{from_dlpack}/torch.utils.dlpack.to_dlpack
# https://docs.jax.dev/en/latest/jax.dlpack.html ; https://docs.pytorch.org/docs/stable/generated/torch.from_dlpack.html
class JaxPDEWrapper(torch.autograd.Function):
    _vjp_cache = {}

    @staticmethod
    def forward(ctx, a_torch: torch.Tensor, g_callable):
        if not a_torch.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        a_dl = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
        a_jax = jax.dlpack.from_dlpack(a_dl)            # shares memory if possible  :contentReference[oaicite:1]{index=1}
        out_jax = g_callable(a_jax)
        out_dl  = jax.dlpack.to_dlpack(out_jax)
        out_t   = torch.utils.dlpack.from_dlpack(out_dl) # shares memory with input dlpack  :contentReference[oaicite:2]{index=2}
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

# ---------------- PDE（Patterns stepper，按 forcing_pattern） ----------------
def generate_sequence_every_second_patterns(u0: jnp.ndarray,
                                            nu: float,
                                            T_seconds: int,
                                            fixed_step: float,
                                            forcing_pattern: str):
    # 方向对齐：flip+rot90（与你的 v2 一致）
    u0p = jnp.rot90(jnp.flip(u0, axis=-2), 3, axes=(-2, -1))  # (H,W)
    H, W = u0p.shape
    assert H == W, "Only square grids supported."
    import exponax as ex
    stepper = ex.stepper._navier_stokes.NavierStokesVorticityPatterns(
        2, 1, H, fixed_step, diffusivity=nu, order=4,
        num_circle_points=16, dealiasing_fraction=2/3,
        forcing_pattern=forcing_pattern
    )
    steps_per_sec = int(round(1.0 / fixed_step))

    def step_once(u): return stepper(u[None, ...])[0]
    step_once = jax.checkpoint(step_once)

    def micro(carry, _):
        u = step_once(carry); return u, None

    def run_one_second(carry, _):
        u, _ = jax.lax.scan(micro, carry, None, length=steps_per_sec)
        return u, u

    _, seconds = jax.lax.scan(run_one_second, u0p, None, length=T_seconds)
    seq = jnp.concatenate([u0p[None, ...], seconds], axis=0)   # (T+1, H, W)
    seq = jnp.swapaxes(seq, -1, -2)                            # (T+1, W, H)
    return seq

def batched_patterns_rollout(u0_batched: jnp.ndarray,
                             nu: float, T_seconds: int, fixed_step: float, forcing_pattern: str):
    fn = lambda a: generate_sequence_every_second_patterns(a, nu, T_seconds, fixed_step, forcing_pattern)
    return jax.vmap(fn, in_axes=0, out_axes=1)(u0_batched)     # (T+1,B,H,W)

class DifferentiablePDESolverPatterns:
    def __init__(self, nu: float, forcing_pattern: str, device="cuda", fixed_step: float = 0.005):
        self.nu = nu
        self.device = device
        self.fixed_step = fixed_step
        self.forcing_pattern = forcing_pattern
        self.perf_records = {}

    def rollout_seconds(self, x0: torch.Tensor, T_seconds: int):
        assert isinstance(x0, torch.Tensor) and x0.is_cuda
        if x0.ndim == 2:
            g = jax.jit(lambda a: generate_sequence_every_second_patterns(
                a, self.nu, T_seconds, self.fixed_step, self.forcing_pattern))
        elif x0.ndim == 3:
            g = jax.jit(lambda A: batched_patterns_rollout(
                A, self.nu, T_seconds, self.fixed_step, self.forcing_pattern))
        else:
            raise ValueError(f"x0 ndim must be 2 or 3, got {x0.ndim}")
        return JaxPDEWrapper.apply(x0.contiguous(), g)  # (T+1,H,W) or (T+1,B,H,W)

# ---------------- 攻击系统（只外层 tqdm） ----------------
class PDEAttackSystem:
    def __init__(self, recurrent_model: torch.nn.Module, nu: float, forcing_pattern: str,
                 device="cuda", mode_spec="wwwwwwwwww", fixed_step=0.005):
        self.recurrent_model = recurrent_model.to(device).eval()
        for p in self.recurrent_model.parameters(): p.requires_grad = False
        self.nu = nu
        self.device = device
        self.mode_spec = mode_spec
        assert isinstance(mode_spec, str) and len(mode_spec) == 10 and set(mode_spec) <= set("adw")
        self.pde_solver = DifferentiablePDESolverPatterns(nu, forcing_pattern, device, fixed_step)

    def forward(self, x0: torch.Tensor, T_seconds: int):
        if x0.ndim == 2: x0 = x0.unsqueeze(0)  # (B,H,W)
        need_solver_seconds = {t for t in range(1,10) if self.mode_spec[t-1] in ("w","d")}
        if self.mode_spec[9] in ("w","d"): need_solver_seconds.add(19)
        assert len(need_solver_seconds) > 0
        T_required = max(need_solver_seconds)

        seq = self.pde_solver.rollout_seconds(x0, T_required)   # (T+1,B,H,W)
        frames = [x0]
        for t in range(1,10):
            f = seq[t]
            if self.mode_spec[t-1] == 'd': f = f.detach()
            frames.append(f)
        input_batch = torch.stack(frames, dim=-1)               # (B,H,W,10)

        preds  = self.recurrent_model(input_batch)              # (B,H,W,10)
        pred_19 = preds[..., -1]
        true_19 = seq[19]
        if self.mode_spec[9] == 'd': true_19 = true_19.detach()
        return pred_19, true_19

    def pgd_attack_adam(self, initial_x0: torch.Tensor, *,
                        epsilon: float, alpha: float, num_steps: int, norm: str, T_seconds: int,
                        beta1=0.9, beta2=0.999, adam_eps=1e-8, use_sign_for_linf=True, amsgrad=False):
        x_adv = initial_x0.clone().detach().to(self.device)
        if x_adv.ndim == 2: x_adv = x_adv.unsqueeze(0)
        x_adv.requires_grad_(True)
        original_x0 = x_adv.clone().detach()

        m = torch.zeros_like(x_adv); v = torch.zeros_like(x_adv)
        vhat_max = torch.zeros_like(x_adv) if amsgrad else None
        t_adam = 0

        for _ in range(num_steps):
            if x_adv.grad is not None: x_adv.grad.zero_()
            pred_19, true_19 = self.forward(x_adv, T_seconds)
            loss = F.mse_loss(pred_19, true_19, reduction="sum")  # batch-sum
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

                if norm == 'inf':
                    direction = torch.sign(d_t) if use_sign_for_linf else d_t
                    x_adv.data += alpha * direction
                    delta = x_adv - original_x0
                    x_adv.data = original_x0 + torch.clamp(delta, -epsilon, epsilon)
                else:  # L2 (per-sample projection)
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
            x_adv.grad.zero_()

        # final rollout for all frames 0..T_seconds
        with torch.no_grad():
            seq = self.pde_solver.rollout_seconds(x_adv, T_seconds)   # (T+1,B,H,W)
            y_all = seq.permute(1, 2, 3, 0).contiguous()               # (B,H,W,T+1)
        return x_adv.detach(), y_all.detach()

# ---------------- 仅外层 tqdm 的批处理主流程 ----------------
def format_now_eta(start_ts, done, total):
    now = datetime.now()
    if done <= 0:
        return now.strftime("%Y-%m-%d %H:%M:%S"), "estimating..."
    elapsed = (now - start_ts).total_seconds()
    rem_sec = elapsed / max(done, 1) * (total - done)
    eta = now + timedelta(seconds=rem_sec)
    return now.strftime("%Y-%m-%d %H:%M:%S"), eta.strftime("%Y-%m-%d %H:%M:%S")

def main(_):
    assert FLAGS.dataset_pt and FLAGS.model_pth and FLAGS.forcing_pattern and FLAGS.save_to, \
        "--dataset_pt/--model_pth/--forcing_pattern/--save_to are required."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size = 256
    ratio = 100.0
    epsilon = float(FLAGS.eps_mult) * (size ** 2)
    alpha   = float(FLAGS.alpha_mult) * ratio
    norm    = str(FLAGS.norm)
    steps   = int(FLAGS.num_steps)
    Tsec    = int(FLAGS.T_seconds)
    B       = int(FLAGS.batch_size)

    # Load dataset & model
    ds_path = Path(FLAGS.dataset_pt).resolve()
    data = torch.load(ds_path, map_location=device, weights_only=False)
    X = data['x'].to(device)   # (N,256,256)
    N, H, W = X.shape

    base = FNO2d(modes1=64, modes2=64, width=60, in_channels=10).to(device).eval()
    state_dict = torch.load(FLAGS.model_pth, map_location=device)
    base.load_state_dict(state_dict)
    model = RecurrentPredictor(base, T_out=10, step=1).to(device).eval()
    for p in model.parameters(): p.requires_grad = False

    system = PDEAttackSystem(
        recurrent_model=model,
        nu=float(FLAGS.nu),
        forcing_pattern=str(FLAGS.forcing_pattern),
        device=device,
        mode_spec=str(FLAGS.mode_spec),
        fixed_step=float(FLAGS.fixed_step),
    )

    # Preallocate outputs (same structure as input)
    x_out = torch.empty((N, H, W),           dtype=torch.float32, device='cpu')
    y_out = torch.empty((N, H, W, Tsec+1),   dtype=torch.float32, device='cpu')

    # Single outer tqdm
    pbar = tqdm(
        total=N,
        desc=f"[{FLAGS.forcing_pattern}] B={B}",
        dynamic_ncols=True,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    start_ts = datetime.now()
    total_batches = (N + B - 1) // B

    for b_idx, start in enumerate(range(0, N, B), 1):
        end = min(start + B, N)
        x_batch = X[start:end]
        x_adv, y_all = system.pgd_attack_adam(
            initial_x0=x_batch,
            epsilon=epsilon, alpha=alpha, num_steps=steps, norm=norm, T_seconds=Tsec,
            beta1=0.9, beta2=0.999, adam_eps=1e-8, use_sign_for_linf=True, amsgrad=False
        )
        x_out[start:end] = x_adv.detach().float().cpu()
        y_out[start:end] = y_all.detach().float().cpu()

        pbar.update(end - start)
        now_s, eta_s = format_now_eta(start_ts, pbar.n, pbar.total)
        pbar.set_postfix(
            pattern=FLAGS.forcing_pattern,
            batch=f"{b_idx}/{total_batches}",
            samples=f"{pbar.n}/{N}",
            now=now_s,
            ETA=eta_s
        )
        del x_batch, x_adv, y_all
        torch.cuda.empty_cache(); gc.collect()

    pbar.close()

    # ---- Save ----
    save_to = Path(FLAGS.save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True) if save_to.suffix == ".pt" else save_to.mkdir(parents=True, exist_ok=True)
    def fmt(v): return f"{v:.6g}"

    if save_to.suffix == ".pt":
        out_path = save_to
    else:
        # 目录：自动命名
        base_name = (
            f"dim2d_nx256_N{N}_solver=exponax_nu0.000_t{Tsec:.1f}_"
            f"forcingPattern{FLAGS.forcing_pattern}"
            f"_modespec={FLAGS.mode_spec}"
            f"_norm{norm}_alpha{fmt(alpha)}_epsilon{fmt(epsilon)}_steps{steps}"
            f"_batch{B}.pt"
        )
        out_path = save_to / base_name

    torch.save({'x': x_out, 'y': y_out}, out_path)
    print(f"\n✅ Saved expanded dataset (same keys/shapes as input): {out_path}")

if __name__ == "__main__":
    app.run(main)


