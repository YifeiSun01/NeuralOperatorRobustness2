#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from tqdm import tqdm
from datetime import datetime
import time
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO2d import FNO2d, RecurrentPredictor
import torch
from pathlib import Path
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import exponax as ex
import torch.cuda as cuda
import warnings
import gc
import torch.nn.functional as F
import jaxlib
from pprint import pprint
import subprocess

# ============ 环境打印 ============
DEBUG_GRAD = os.environ.get("DEBUG_GRAD", "0") == "1"

def print_versions():
    print("\n" + "="*40 + " 环境诊断 " + "="*40)
    print("\n[PyTorch]")
    print(f"Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    print("\n[JAX]")
    print(f"JAX Version: {jax.__version__}")
    print(f"JAXlib Version: {jaxlib.__version__}")
    print(f"JAX Backend: {jax.lib.xla_bridge.get_backend().platform}")
    try:
        from jax.lib import xla_bridge
        print(f"JAX Devices: {xla_bridge.devices()}")
    except Exception as e:
        print(f"JAX Devices Error: {str(e)}")

    print("\n[CUDA/cuDNN System Info]")
    try:
        nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
        print(nvcc_out.split("\n")[0])
    except:
        print("nvcc not found")
    print("="*40 + " 诊断结束 " + "="*40 + "\n")

print_versions()
jax.config.update("jax_enable_x64", False)

# ============ 工具函数 ============
def spectral_upsample(field, target_size=256):
    """
    支持 [..., H, W] 的批输入，返回相同前缀批维，最后两维为 target_size x target_size。
    """
    *batch_dims, H, W = field.shape
    assert target_size >= H and target_size >= W, "目标尺寸必须大于或等于输入尺寸"
    field = field  # 由调用处放到正确 device
    freq = torch.fft.fft2(field, norm='ortho')
    freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))
    pad_H = (target_size - H) // 2
    pad_W = (target_size - W) // 2
    new_freq_shifted = torch.zeros(
        *batch_dims, target_size, target_size,
        dtype=freq_shifted.dtype, device=freq_shifted.device
    )
    start_H = pad_H
    start_W = pad_W
    new_freq_shifted[..., start_H:start_H+H, start_W:start_W+W] = freq_shifted
    new_freq = torch.fft.ifftshift(new_freq_shifted, dim=(-2, -1))
    upsampled = torch.fft.ifft2(new_freq, norm='ortho')
    return (target_size / H) * upsampled.real

# ---- JAX<->PyTorch zero-copy via DLPack ----
class JaxPDEWrapper(torch.autograd.Function):
    _vjp_cache = {}
    @staticmethod
    def forward(ctx, a_torch, g):
        if not a_torch.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        # to JAX
        a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
        a_jax = jax.dlpack.from_dlpack(a_dlpack)
        g_output_jax = g(a_jax)
        # back to torch
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

# ---------- 支持批维度 (B,H,W) ----------
def generate_sequence_every_second(u0, nu, T_seconds, fixed_step=0.005):
    """
    u0: jnp.ndarray, 形状 (H,W) 或 (B,H,W)
    返回: jnp.ndarray, 形状 (T_seconds+1, B, H, W)
    """
    # 规范化为 (B,H,W)
    if u0.ndim == 2:
        u0 = u0[None, ...]               # (1,H,W)
    B, H, W = u0.shape

    # 旋转/翻转在最后两维上进行
    u0_proc = jnp.rot90(jnp.flip(u0, axis=-2), 3, axes=(-2, -1))   # (B,H,W)

    # Exponax stepper：第三个参数是空间网格大小（此处用 H）
    stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
        2, 1, H, fixed_step, diffusivity=nu, order=4
    )
    num_steps = int(T_seconds / fixed_step)
    rollout_stepper = ex.rollout(stepper, num_steps, include_init=True)
    result = rollout_stepper(u0_proc)               # 形状约为 (T_steps+1, B, H, W)

    # 每整秒抽样
    idxs = jnp.array([int(s / fixed_step) for s in range(T_seconds + 1)], dtype=jnp.int32)
    picked = result[idxs, ...]                      # (T_seconds+1, B, H, W)
    return picked

class DifferentiablePDESolver:
    def __init__(self, nu, device="cuda"):
        self.nu = nu
        self.device = device
        self.perf_records = {}
    def rollout_seconds(self, x0, T_seconds):
        """
        x0: torch.Tensor, (B,H,W) or (H,W)
        return: torch.Tensor, (T_seconds+1, B, H, W)
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        start_mem = torch.cuda.memory_allocated()
        assert isinstance(x0, torch.Tensor) and x0.is_cuda
        def solver_func(a_jax):
            return generate_sequence_every_second(a_jax, self.nu, T_seconds)
        seq_tensor = JaxPDEWrapper.apply(x0.contiguous(), solver_func)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - start_time
        end_mem = torch.cuda.memory_allocated()
        mem_usage = end_mem - start_mem
        self.perf_records[f"rollout_0_{T_seconds}"] = (duration, mem_usage)
        return seq_tensor
    def get_perf_records(self):
        records = self.perf_records
        self.perf_records = {}
        return records

# ---------- 近似检索器（支持批） ----------
class ApproximatePDESolutionFinder:
    def __init__(self, device="cuda"):
        self.device = device
        self.perf_records = {}
        self.dataset_path = Path(__file__).parent.parent / "datasets" / "exponax_datasets" / "t20" / \
            "dictionary" / "dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt"
        self.dataset = torch.load(self.dataset_path, weights_only=False, map_location=device)
        self.x_dataset = self.dataset["x"].to(self.device)           # (N,256,256)
        self.y_dataset = self.dataset["y"].to(self.device)           # (N,256,256,21)

    @staticmethod
    def _mse_match_batch(x_dataset, queries):
        """
        x_dataset: (N,H,W)
        queries:   (B,H,W)
        return: best_idx per batch, (B,)
        """
        # (B,N,H,W)
        diffs = x_dataset.unsqueeze(0) - queries.unsqueeze(1)
        mse = torch.mean(diffs**2, dim=(2,3))        # (B,N)
        best_idx = torch.argmin(mse, dim=1)          # (B,)
        return best_idx

    def find(self, x0):
        """
        x0: torch.Tensor, (B,H,W)
        return:
          input_seq:    (B,H,W,10)   # x0 + t=1..9
          true19_batch: (B,H,W)      # t=19
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        start_mem  = torch.cuda.memory_allocated()

        assert isinstance(x0, torch.Tensor) and x0.is_cuda
        if x0.ndim == 2:    # 单样本也处理为批
            x0 = x0[None, ...]

        idx = self._mse_match_batch(self.x_dataset, x0)         # (B,)
        y = self.y_dataset[idx]                                 # (B,256,256,21)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - start_time
        end_mem  = torch.cuda.memory_allocated()
        mem_usage = end_mem - start_mem
        self.perf_records["all 0-10 and 19 secs"] = (duration, mem_usage)

        input_seq = torch.cat([x0.unsqueeze(-1), y[..., 1:10]], dim=-1)   # (B,H,W,10)
        true19_approx = y[..., 19]                                        # (B,H,W)
        return input_seq, true19_approx

    def get_perf_records(self):
        return self.perf_records

# ---------- 记录器（批也支持） ----------
class AttackRecorder:
    def __init__(self):
        self.steps = []
        self.metadata = {}
    def record_step(
        self, step, x0, output,
        truth=None, surrogate_truth=None,
        loss=None, surrogate_loss=None,
        gradient=None, numerical_gradient=None,
        reach_boundary=False, scaled_gradient=None,
    ):
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
            rec["loss"] = float(loss.item()) if torch.is_tensor(loss) else float(loss)
        if surrogate_loss is not None:
            rec["surrogate_loss"] = float(surrogate_loss.item()) if torch.is_tensor(surrogate_loss) else float(surrogate_loss)
        if gradient is not None:
            rec["gradient"] = gradient.detach().cpu().numpy().copy()
        if scaled_gradient is not None:
            rec["scaled_gradient"] = scaled_gradient.detach().cpu().numpy().copy()
        if numerical_gradient is not None:
            rec["numerical_gradient"] = numerical_gradient.detach().cpu().numpy().copy()
        self.steps.append(rec)
    def save(self, filename, metadata=None):
        if metadata:
            self.metadata.update(metadata)
        with open(filename, 'wb') as f:
            pickle.dump({"steps": self.steps, "metadata": self.metadata}, f)
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        recorder = AttackRecorder()
        recorder.steps = data["steps"]
        recorder.metadata = data["metadata"]
        return recorder

# ============ 主系统（支持 a/d/w + 批） ============
class PDEAttackSystem:
    def __init__(self, recurrent_model, nu=1e-5, device="cuda", mode_spec="wwwwwwwwww",
                 loss_reduction="sum"):
        """
        mode_spec: 长度10，仅含 'a' 'd' 'w'
        loss_reduction: 'sum' 或 'mean'
        """
        self.recurrent_model = recurrent_model.to(device)
        self.nu = nu
        self.device = device
        self.mode_spec = mode_spec
        self.loss_reduction = loss_reduction

        assert isinstance(self.mode_spec, str) and len(self.mode_spec) == 10
        assert set(self.mode_spec).issubset(set("adw"))

        for p in self.recurrent_model.parameters():
            p.requires_grad = False
        self.recurrent_model.eval()

        self.pde_solver = DifferentiablePDESolver(nu, device)
        self.pde_sol_finder = ApproximatePDESolutionFinder(device)

    def _format_memory(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if abs(size) < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} GB"

    def forward(self, x0):
        """
        x0: (B,H,W) 或 (H,W)
        return: pred_19 (B,H,W), true_19 (B,H,W)
        """
        if x0.ndim == 2:
            x0 = x0[None, ...]
        B, H, W = x0.shape

        # 需要 PDE 的秒数
        need_solver_seconds = set()
        for t in range(1, 10):
            if self.mode_spec[t-1] in ("w", "d"):
                need_solver_seconds.add(t)
        ch19 = self.mode_spec[9]
        if ch19 in ("w", "d"):
            need_solver_seconds.add(19)

        # 近似序列（若包含 'a'）
        input_sequence_approx, true19_approx = None, None
        self.pde_records_0_to_9_and_19 = {}
        if 'a' in self.mode_spec:
            input_sequence_approx, true19_approx = self.pde_sol_finder.find(x0)

        # 求解器（一次回放到最大需要秒）
        seq_0_to_T = None
        if len(need_solver_seconds) > 0:
            T_required = max(need_solver_seconds)
            seq_0_to_T = self.pde_solver.rollout_seconds(x0, T_required)  # (T+1,B,H,W)

        # 组装 0..9 输入，得到 (B,H,W,10)
        frames = [x0]  # t=0
        for t in range(1, 10):
            ch = self.mode_spec[t-1]
            if ch == 'a':
                f = input_sequence_approx[..., t]                 # (B,H,W)
            else:
                f = seq_0_to_T[t]                                 # (B,H,W)
                if ch == 'd':
                    f = f.detach()
            frames.append(f)
        input_batch = torch.stack(frames, dim=-1)                 # (B,H,W,10)

        # 预测 10..19，取 19
        predictions = self.recurrent_model(input_batch)           # (B,H,W,10)
        pred_19     = predictions[..., -1]                        # (B,H,W)

        # 组装真值 19
        if ch19 == 'a':
            true_19 = true19_approx
        else:
            true_19 = seq_0_to_T[19]
            if ch19 == 'd':
                true_19 = true_19.detach()
        return pred_19, true_19

    def pgd_attack_adam(
        self,
        initial_x0,              # (B,H,W) or (H,W)
        epsilon=0.1,
        alpha=0.01,
        num_steps=10,
        norm="inf",
        idx_tag="batch",
        beta1=0.9,
        beta2=0.999,
        adam_eps=1e-8,
        use_sign_for_linf=True,
        amsgrad=False,
    ):
        """
        对一个 batch 的初始条件做 PGD + Adam 攻击；一次性更新整批输入。
        """
        recorder = AttackRecorder()
        x_adv = initial_x0.clone().detach().to(self.device)
        if x_adv.ndim == 2:
            x_adv = x_adv[None, ...]
        x_adv.requires_grad_(True)
        B = x_adv.shape[0]
        original_x0 = x_adv.detach().clone()

        # Adam 状态
        m = torch.zeros_like(x_adv)
        v = torch.zeros_like(x_adv)
        vhat_max = torch.zeros_like(x_adv) if amsgrad else None
        t_adam = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for step in range(num_steps):
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            # ----- Forward -----
            pred_19, true_19 = self.forward(x_adv)

            # ----- Loss（建议默认 sum；如切到 mean，请同步调整 alpha）-----
            reduction = self.loss_reduction
            surrogate_loss = F.mse_loss(pred_19, true_19, reduction=reduction)

            if DEBUG_GRAD:
                g_tmp = torch.autograd.grad(surrogate_loss, x_adv, retain_graph=True, allow_unused=True)[0]
                print(f"[debug] loss={float(surrogate_loss.item()):.6e} "
                      f"||grad||={None if g_tmp is None else float(g_tmp.norm().item()):.3e}")

            # ----- Backward -----
            surrogate_loss.backward()

            # ----- PGD + Adam 更新 -----
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
                    delta = x_adv - original_x0
                    x_adv.data = original_x0 + torch.clamp(delta, -epsilon, epsilon)
                    reach_boundary = float(((delta >= epsilon - 1e-6) | (delta <= -epsilon + 1e-6)).float().mean().item() * 100.0)
                elif norm in (2, '2', 'L2'):
                    # 逐样本 L2 约束
                    direction = d_t
                    # 展平到 (B, -1) 做范数
                    flat = direction.view(B, -1)
                    flat_norm = flat.norm(p=2, dim=1, keepdim=True) + 1e-12
                    direction = (flat / flat_norm).view_as(direction)
                    x_adv.data += alpha * direction
                    current_delta = x_adv - original_x0
                    flat_delta = current_delta.view(B, -1)
                    delta_norm = flat_delta.norm(p=2, dim=1, keepdim=True)  # (B,1)
                    scale = torch.clamp(epsilon / (delta_norm + 1e-12), max=1.0)  # (B,1)
                    x_adv.data = (original_x0 + (flat_delta * scale).view_as(current_delta))
                    reach_boundary = float(((delta_norm > epsilon - 1e-6).float().mean().item()) * 100.0)
                else:
                    raise ValueError(f"Unsupported norm: {norm}")

                recorder.record_step(
                    step=step,
                    x0=x_adv.clone().detach(),
                    output=pred_19.clone().detach(),
                    truth=true_19.clone().detach(),
                    loss=surrogate_loss,
                    gradient=x_adv.grad.clone().detach(),
                    scaled_gradient=d_t.clone().detach(),
                    reach_boundary=reach_boundary,
                )
                x_adv.grad.zero_()

            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated()
                print(f"[step {step}] loss={float(surrogate_loss.item()):.6e} "
                      f"| peak_mem={peak_mem/1024**2:.1f}MB")
                torch.cuda.reset_peak_memory_stats()

        # ---- 保存 ----
        folder_name = "perturbation_results_batch"
        os.makedirs(folder_name, exist_ok=True)
        mode_tag = getattr(self, "mode_spec", "undefined")
        save_name = (
            f"{folder_name}/"
            f"pgd_adam_modespec={mode_tag}"
            f"_norm{norm}_alpha{alpha}_epsilon{epsilon}_steps{num_steps}"
            f"_{idx_tag}.pkl"
        )
        recorder.save(
            save_name,
            metadata={
                "mode_spec": mode_tag,
                "norm": norm,
                "epsilon": epsilon,
                "alpha": alpha,
                "num_steps": num_steps,
                "idx_tag": idx_tag,
                "loss_reduction": self.loss_reduction,
                "B": x_adv.shape[0],
            },
        )
        return x_adv

# ============ 入口 ============
if __name__ == "__main__":
    T = 10
    step = 1
    train_test = "test"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取数据
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_test = torch.load(
        f'{script_dir}/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_{train_test}_all_frames.pt',
        weights_only=False, map_location=device
    )  # dict: x:(N,256,256), y:(N,256,256,20/21)...

    # FNO 模型
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

    # 攻击配置（可多组）
    input_list = [
        {"norm": 2, "epsilon": 0.0002*size**2, "alpha": 0.01*ratio,  "num_steps": 100},
        {"norm": 2, "epsilon": 0.0006*size**2, "alpha": 0.01*ratio,  "num_steps": 100},
        {"norm": 2, "epsilon": 0.0012*size**2, "alpha": 0.01*ratio,  "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0002*size**2, "alpha": 0.025*ratio, "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0006*size**2, "alpha": 0.025*ratio, "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0012*size**2, "alpha": 0.025*ratio, "num_steps": 100},
    ]
    print("input_list:", input_list)

    MODE_SPEC = os.environ.get("MODE_SPEC", "wwwwwwwwww")
    LOSS_REDUCTION = os.environ.get("LOSS_REDUCTION", "sum")   # "sum" or "mean"
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))        # 你可以手动设定 B

    # ---- 连续切片分批：[:B], [B:2B], [2B:3B], ... ----
    X_all = data_test["x"]    # (N,256,256)
    N_total = X_all.shape[0]

    for cfg in input_list:
        recurrent_model = RecurrentPredictor(model_instance, T_out=T, step=step).to(device)
        attack_system = PDEAttackSystem(
            recurrent_model=recurrent_model,
            nu=1e-5,
            device=device,
            mode_spec=MODE_SPEC,
            loss_reduction=LOSS_REDUCTION,
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
            "mode_spec: {mode_spec}, loss_reduction: {lr}, "
            "b1: {b1}, b2: {b2}, adam_eps: {ae}, sign_Linf: {sgn}, amsgrad: {ams}".format(
                norm=norm, epsilon=epsilon, alpha=alpha, num_steps=num_steps,
                mode_spec=MODE_SPEC, lr=LOSS_REDUCTION,
                b1=beta1, b2=beta2, ae=adam_eps, sgn=use_sign_for_linf, ams=amsgrad)
        )

        for start in range(0, N_total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_total)
            # 取连续切片（无手动索引）
            x_batch = X_all[start:end].to(device)                 # (B,256,256)
            x_batch = spectral_upsample(x_batch, 256).to(device)  # 若已 256，可跳过
            idx_tag = f"idx{start}-{end-1}"

            adversarial_x0 = attack_system.pgd_attack_adam(
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
            )

            # 释放显存
            del adversarial_x0, x_batch
            torch.cuda.empty_cache()
            gc.collect()

        # 清理本组模型
        del recurrent_model, attack_system
        torch.cuda.empty_cache()
        gc.collect()

    print("PGD 批量攻击完成！") 





