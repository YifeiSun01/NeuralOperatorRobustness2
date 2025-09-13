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
from torchviz import make_dot
import warnings
from torch.autograd import gradcheck
import gc

import torch.nn.functional as F
from scipy.spatial.distance import cosine
from torch.autograd.functional import jacobian

import jaxlib
from pprint import pprint
import subprocess

# 可选梯度自检（设环境变量 DEBUG_GRAD=1 开启）
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

    print("\n[GPU Details]")
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"Device {i}: {props.name}")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Total Memory: {props.total_memory/1024**3:.2f} GB")
                print(f"  Multiprocessors: {props.multi_processor_count}")
                print(f"  Max Threads per Block: {props.max_threads_per_block}")
    except Exception as e:
        print(f"GPU Details Error: {str(e)}")

    print("\n[Compatibility Warnings]")
    if torch.cuda.is_available():
        if not torch.version.cuda:
            warnings.warn("PyTorch was not built with CUDA support!")
        try:
            cudnn_ver = torch.backends.cudnn.version()
            compile_ver = torch._C._cudnn_version()
            if cudnn_ver != compile_ver:
                warnings.warn(f"cuDNN version mismatch! Runtime: {cudnn_ver}, Compiled: {compile_ver}")
        except:
            warnings.warn("Could not check cuDNN version compatibility")
    print("="*40 + " 诊断结束 " + "="*40 + "\n")

print_versions()

jax.config.update("jax_enable_x64", False)

def spectral_upsample(field, target_size=256):
    *batch_dims, H, W = field.shape
    assert target_size >= H and target_size >= W, "目标尺寸必须大于输入尺寸"
    # 依赖全局 device（在 __main__ 里定义），调用时已就绪
    field = field.to(device)
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
    """
    稀疏保存：只保留 0..T 的整数秒帧，形状 (T_seconds+1, H, W)。
    内部仍走 1/dt 微步，但不保留中间帧；反向时可通过 checkpoint 重算微步以省显存。
    """
    # 与原逻辑一致的旋转/翻转（保持坐标系约定）
    u0_proc = jnp.rot90(jnp.flip(u0, axis=-2), 3, axes=(-2, -1))   # (H, W)
    H, W = u0_proc.shape
    assert H == W, "目前只支持正方域网格"

    # Navier-Stokes vorticity stepper（与你原来的相同）
    stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
        2, 1, H, fixed_step, diffusivity=nu, order=4
    )

    steps_per_sec = int(round(1.0 / fixed_step))
    # 单个微步：stepper 期望 (1,H,W)，我们返回去掉 batch 的末状态
    def step_once(u):
        return stepper(u[None, ...])[0]

    # 前向不变；反向(grad/vjp)时重算微步中间量以省显存
    step_once = jax.checkpoint(step_once)

    # 用一个“内层 scan”推进 1 秒的微步，只取该秒末状态
    def micro(carry, _):
        u = step_once(carry)
        return u, None

    def run_one_second(carry, _):
        u, _ = jax.lax.scan(micro, carry, None, length=steps_per_sec)
        return u, u  # carry=末状态, ys=末状态（作为该秒的帧）

    # 外层 scan 推进 T_seconds 次，每次产出该秒末状态；最后再把 t=0 初值拼回去
    _, seconds = jax.lax.scan(run_one_second, u0_proc, None, length=T_seconds)
    seq = jnp.concatenate([u0_proc[None, ...], seconds], axis=0)  # (T+1, H, W)

    # 与你后续使用保持一致：交换最后两维
    seq = jnp.swapaxes(seq, -1, -2)
    return seq


class DifferentiablePDESolver:
    def __init__(self, nu, device="cuda", *, fixed_step=0.005):
        self.nu = nu
        self.device = device
        self.perf_records = {}
        self.fixed_step = fixed_step  # 新增：固定 dt（建议显式传入）

    def rollout_seconds(self, x0, T_seconds):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        start_mem = torch.cuda.memory_allocated()

        assert isinstance(x0, torch.Tensor) and x0.is_cuda

        # 闭包 + jit：T_seconds / fixed_step / nu 被视作静态常量，减少 Python 开销
        solver_func = jax.jit(
            lambda a: generate_sequence_every_second(
                a, self.nu, T_seconds, fixed_step=self.fixed_step
            )
        )

        seq_tensor = JaxPDEWrapper.apply(x0.contiguous(), solver_func)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - start_time
        end_mem = torch.cuda.memory_allocated()
        mem_usage = end_mem - start_mem
        self.perf_records[f"rollout_0_{T_seconds}"] = (duration, mem_usage)
        return seq_tensor


# ====== 近似检索器：仅当 'a' 出现时使用 ======
class ApproximatePDESolutionFinder:
    def __init__(self, device="cuda"):
        self.device = device
        self.perf_records = {}
        self.dataset_path = Path(__file__).parent.parent / "datasets" / "exponax_datasets" / "t20" / "dictionary" / "dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt"
        self.dataset = torch.load(self.dataset_path, weights_only=False, map_location=device)
        self.x_dataset = self.dataset["x"].to(self.device)                 # (N,256,256)
        self.y_dataset = self.dataset["y"].to(self.device)                 # (N,256,256,21)

    @staticmethod
    def _mse_match(x_dataset, query):
        # 注意：N=2000 时内存 ~ 2000*256*256*4B ≈ 0.5GB，可接受但偏重
        diffs = x_dataset - query.unsqueeze(0)       # (N,H,W)
        mse = torch.mean(diffs**2, dim=(1,2))        # (N,)
        best_idx = torch.argmin(mse).item()
        return best_idx

    def find(self, x0):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        start_mem  = torch.cuda.memory_allocated()
        assert isinstance(x0, torch.Tensor) and x0.is_cuda
        idx = self._mse_match(self.x_dataset, x0)
        y = self.y_dataset[idx]                      # (256,256,21)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - start_time
        end_mem  = torch.cuda.memory_allocated()
        mem_usage = end_mem - start_mem
        self.perf_records["all 0-10 and 19 secs"] = (duration, mem_usage)
        # 返回 (H,W,10): x0 + y[...,1:10] 以及 (H,W) 的第19秒
        input_seq = torch.cat([x0.unsqueeze(-1), y[..., 1:10]], dim=-1)
        true19_approx = y[..., 19]
        return input_seq, true19_approx

    def get_perf_records(self):
        # 若想每次清空，可改成与 PDE solver 一样清空
        return self.perf_records

# ====== Recorder（支持真实/近似双通道；字段可为 None）======
class AttackRecorder:
    """记录PGD攻击过程中的关键数据（支持真实/近似双真值、双损失；字段可为 None）"""
    def __init__(self):
        self.steps = []
        self.metadata = {}

    def record_step(
        self,
        step,
        x0,
        output,
        truth=None,                # 真实 solver@t19 真值（仅当 mode_spec[9]=='a' 时额外记录）
        surrogate_truth=None,      # 近似 finder@t19 真值（当 mode_spec[9]=='a' 时存在）
        loss=None,                 # 与 truth 的损失
        surrogate_loss=None,       # 与 surrogate_truth（或 forward 返回的 true_19）的损失
        gradient=None,
        numerical_gradient=None,
        reach_boundary=False,
        scaled_gradient=None,
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
            rec["loss"] = float(loss.item())
        if surrogate_loss is not None:
            rec["surrogate_loss"] = float(surrogate_loss.item())
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

# ============ 主系统（支持 a/d/w ）============
class PDEAttackSystem:
    def __init__(self, recurrent_model, nu=1e-5, device="cuda", mode_spec="wwwwwwwwww"):
        """
        mode_spec: 长度=10的字符串，只含 'a' 'd' 'w'：
           索引 0..8 -> frame1..9
           索引 9    -> frame19
        """
        self.recurrent_model = recurrent_model
        self.nu = nu
        self.device = device
        self.mode_spec = mode_spec

        # 校验 & 打印逐位状态
        assert isinstance(self.mode_spec, str), "mode_spec 必须是字符串"
        assert len(self.mode_spec) == 10, "mode_spec 长度必须为 10（对应帧1..9与帧19）"
        valid = set("adw")
        assert set(self.mode_spec).issubset(valid), "mode_spec 只能包含字符 'a','d','w'"
        self._print_mode_spec(self.mode_spec)

        for p in self.recurrent_model.parameters():
            p.requires_grad = False
        self.recurrent_model.eval()

        self.pde_solver = DifferentiablePDESolver(nu, device)
        self.pde_sol_finder = ApproximatePDESolutionFinder(device)

    @staticmethod
    def _print_mode_spec(mode_spec: str):
        mapping = { 'a': 'approx', 'd': 'detached', 'w': 'with_solver' }
        s = []
        for i, ch in enumerate(mode_spec[:9], start=1):
            s.append(f"[t={i:02d}] {mapping[ch]}")
        s.append(f"[t=19] {mapping[mode_spec[9]]}")
        print("mode_spec =", mode_spec)
        print("Per-frame modes: " + " | ".join(s))

    def _log_performance(self, step, operation, start_time, start_mem):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - start_time
        current_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        mem_usage = current_mem - start_mem
        print(f"[Step {step}] {operation} - "
              f"Time: {duration:.4f}s, "
              f"Mem: {self._format_memory(mem_usage)}")
        return duration, mem_usage

    def _format_memory(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if abs(size) < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} GB"

    def forward(self, x0):
        """
        只按需调用 PDE：根据 mode_spec 中需要 solver 的秒数（'w' 或 'd'）决定最大回放秒数；
        'a' 则调用 finder 近似（0..9 & 19）。
        返回 pred_19, true_19（true_19 由第10位模式决定：a=近似，否则=solver；d 会 detach）。
        """
        # ===== 计时 =====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_start_time = time.perf_counter()
        model_start_mem  = torch.cuda.memory_allocated()

        # ===== 解析需要 solver 的秒数 =====
        need_solver_seconds = set()
        for t in range(1, 10):
            if self.mode_spec[t-1] in ("w", "d"):
                need_solver_seconds.add(t)
        ch19 = self.mode_spec[9]
        if ch19 in ("w", "d"):
            need_solver_seconds.add(19)

        # ===== 若包含 'a'：一次 finder 拿 0..9 与 19 的近似 =====
        input_sequence_approx, true19_approx = None, None
        self.pde_records_0_to_9_and_19 = {}
        if 'a' in self.mode_spec:
            input_sequence_approx, true19_approx = self.pde_sol_finder.find(x0)
            self.pde_records_0_to_9_and_19 = self.pde_sol_finder.get_perf_records()

        # ===== 按需调用 solver（一次 rollout 到所需最大秒数）=====
        seq_0_to_T = None
        pde_duration = 0.0
        pde_mem = 0
        if len(need_solver_seconds) > 0:
            T_required = max(need_solver_seconds)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pde_start_time = time.perf_counter()
            pde_start_mem  = torch.cuda.memory_allocated()
            seq_0_to_T = self.pde_solver.rollout_seconds(x0, T_required)  # [T_required+1,H,W]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pde_duration = time.perf_counter() - pde_start_time
            pde_mem      = torch.cuda.memory_allocated() - pde_start_mem

        # ===== 组装模型输入 0..9 → (1,H,W,10) =====
        frames = [x0]  # frame0 固定为 x0
        for t in range(1, 10):     # frame1..9
            ch = self.mode_spec[t-1]
            if ch == 'a':
                assert input_sequence_approx is not None, "包含 'a' 但未得到近似序列"
                f = input_sequence_approx[..., t]
            else:
                assert seq_0_to_T is not None and (seq_0_to_T.shape[0]-1) >= t, \
                    f"需要 solver 的 t={t} 帧，但未回放到该秒或未调用 solver"
                f = seq_0_to_T[t]
                if ch == 'd':
                    f = f.detach()
            frames.append(f)
        input_sequence = torch.stack(frames, dim=0)                   # [10,H,W]
        input_batch    = input_sequence.permute(1, 2, 0).unsqueeze(0) # (1,H,W,10)

        # ===== 模型预测 10..19，取 19 =====
        predictions = self.recurrent_model(input_batch)               # (1,H,W,10)
        pred_19     = predictions[0, ..., -1]

        # ===== 组装 true_19（第10位）=====
        if ch19 == 'a':
            assert true19_approx is not None, "mode_spec[9]=='a' 但未得到 true19_approx"
            true_19 = true19_approx
        else:
            assert seq_0_to_T is not None and (seq_0_to_T.shape[0]-1) >= 19, \
                "需要 t=19 的 solver 帧，但未回放到 19s 或未调用 solver"
            true_19 = seq_0_to_T[19]
            if ch19 == 'd':
                true_19 = true_19.detach()

        # ===== 计时收尾 =====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_duration = time.perf_counter() - model_start_time
        model_mem      = torch.cuda.memory_allocated() - model_start_mem

        self.model_perf = (model_duration, model_mem)
        self.pde_perf   = (pde_duration, pde_mem)
        self.pde_records_0_to_9 = {}
        self.pde_records_19     = {19: (pde_duration, pde_mem)} if 19 in need_solver_seconds else {}
        return pred_19, true_19

    def print_detailed_perf(self):
        print("\n=== 详细性能报告 ===")
        model_duration, model_mem = self.model_perf
        print(f"[模型部分] 总时间: {model_duration:.4f}s, 显存: {self._format_memory(model_mem)}")
        print("\n[0-9秒 PDE求解]")
        for time_step, (duration, mem) in self.pde_records_0_to_9.items():
            print(f"  t={time_step}s - 时间: {duration:.6f}s, 显存: {self._format_memory(mem)}")
        print("\n[19秒 PDE求解]")
        for time_step, (duration, mem) in self.pde_records_19.items():
            print(f"  t={time_step}s - 时间: {duration:.6f}s, 显存: {self._format_memory(mem)}")
        pde_duration, pde_mem = self.pde_perf
        print(f"\n[PDE求解部分] 总时间: {pde_duration:.4f}s, 显存: {self._format_memory(pde_mem)}")
        total_duration = model_duration + pde_duration
        total_mem = model_mem + pde_mem
        print(f"\n[总计] 时间: {total_duration:.4f}s, 显存: {self._format_memory(total_mem)}")

    def pgd_attack_adam(
        self,
        initial_x0,
        epsilon=0.1,
        alpha=0.01,
        num_steps=10,
        norm="inf",
        idx=0,
        detach="with_solver",  # 兼容保留，但不再用于逻辑/文件名
        beta1=0.9,
        beta2=0.999,
        adam_eps=1e-8,
        use_sign_for_linf=True,
        amsgrad=False,
    ):
        recorder = AttackRecorder()
        x_adv = initial_x0.clone().detach().to(self.device).requires_grad_(True)
        original_x0 = initial_x0.clone().detach().to(self.device)

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
            step_start_time = time.perf_counter()
            step_start_mem = torch.cuda.memory_allocated()

            # ---- Forward ----
            pred_19, true_19 = self.forward(x_adv)

            # 性能数据打印
            model_duration, model_mem = self.model_perf
            pde_duration, pde_mem = self.pde_perf
            pde_records_0_to_9 = self.pde_records_0_to_9
            pde_record_19 = self.pde_records_19

            print(f"\n[Step {step}] ============ 详细性能分析 ============")
            print(f"[模型部分] 总时间: {model_duration:.6f}s | 显存: {self._format_memory(model_mem)}")
            print("\n[0-9秒 PDE求解详细性能]")
            for t in sorted(pde_records_0_to_9.keys()):
                duration, mem = pde_records_0_to_9[t]
                print(f"  t={t}s - 时间: {duration:.6f}s | 显存: {self._format_memory(mem)}")
            print("\n[19秒 PDE求解]")
            for t in sorted(pde_record_19.keys()):
                duration, mem = pde_record_19[t]
                print(f"  t={t}s - 时间: {duration:.6f}s | 显存: {self._format_memory(mem)}")
            print(f"\n[PDE求解部分] 总时间: {pde_duration:.6f}s | 显存: {self._format_memory(pde_mem)}")
            total_duration = model_duration + pde_duration
            total_mem = model_mem + pde_mem
            print(f"\n[本步骤总计] 时间: {total_duration:.6f}s | 显存: {self._format_memory(total_mem)}")

            # ---- Loss（仅对 surrogate 反传；若末位 a，再额外算 real，仅用于记录）----
            # ⚠️ 不要对 true_19 再做 .detach() —— 可导性已由 forward 决定
            surrogate_loss = F.mse_loss(pred_19, true_19, reduction="sum")
            print("surrogate_loss (vs forward.true_19): ", float(surrogate_loss.item()))

            # 可选梯度自检
            if DEBUG_GRAD:
                print("true_19.requires_grad =", true_19.requires_grad,
                      " (mode_spec[9] =", self.mode_spec[9], ")")
                g_tmp = torch.autograd.grad(surrogate_loss, x_adv, retain_graph=True, allow_unused=True)[0]
                print("||grad(surrogate_loss, x_adv)|| =",
                      None if g_tmp is None else float(g_tmp.norm().item()))

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

            # ---- Backward on surrogate ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_start_time = time.perf_counter()
            backward_start_mem = torch.cuda.memory_allocated()
            surrogate_loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._log_performance(step, "Backward Propagation", backward_start_time, backward_start_mem)

            # ---- PGD + Adam 更新 ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            update_start_time = time.perf_counter()
            update_start_mem = torch.cuda.memory_allocated()

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
                    at_upper_boundary = (delta >= epsilon - 1e-6).float()
                    at_lower_boundary = (delta <= -epsilon + 1e-6).float()
                    reach_boundary = (at_upper_boundary + at_lower_boundary).mean().item() * 100.0
                    x_adv.data = original_x0 + torch.clamp(delta, -epsilon, epsilon)
                elif norm in (2, '2', 'L2'):
                    direction = d_t / (torch.norm(d_t, p=2) + 1e-12)
                    x_adv.data += alpha * direction
                    current_delta = x_adv - original_x0
                    delta_norm = torch.norm(current_delta, p=2)
                    if delta_norm > epsilon:
                        x_adv.data = original_x0 + current_delta * (epsilon / (delta_norm + 1e-12))
                        reach_boundary = True
                    else:
                        reach_boundary = False
                else:
                    reach_boundary = False

                # ---------- 记录 ----------
                numerical_grad = locals().get("numerical_grad", None)
                if is_a19:
                    recorder.record_step(
                        step=step,
                        x0=x_adv.clone().detach(),
                        output=pred_19.clone().detach(),
                        truth=real_true_19.clone().detach() if real_true_19 is not None else None,
                        surrogate_truth=true_19.clone().detach(),
                        loss=real_loss if real_loss is not None else torch.tensor(float('nan')),
                        surrogate_loss=surrogate_loss,
                        gradient=x_adv.grad.clone().detach(),
                        numerical_gradient=(numerical_grad.clone().detach()
                                            if numerical_grad is not None else None),
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
                        numerical_gradient=(numerical_grad.clone().detach()
                                            if numerical_grad is not None else None),
                        reach_boundary=reach_boundary,
                        scaled_gradient=d_t.clone().detach(),
                    )
                x_adv.grad.zero_()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._log_performance(step, "PGD Update", update_start_time, update_start_mem)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_step_duration = time.perf_counter() - step_start_time
            total_step_mem = torch.cuda.memory_allocated()
            print(f"\n[Step {step}] 完整步骤总时间: {total_step_duration:.6f}s | "
                  f"总显存: {self._format_memory(total_step_mem - step_start_mem)}")

            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated()
                print(f"[Step {step}] 峰值显存: {self._format_memory(peak_mem)}")
                torch.cuda.reset_peak_memory_stats()

        # ---- 保存 ----
        folder_name = "perturbation_results_1solver_1sample"
        os.makedirs(folder_name, exist_ok=True)
        mode_tag = getattr(self, "mode_spec", "undefined")
        save_name = (
            f"{folder_name}/"
            f"pgd_attack_records_modespec={mode_tag}"
            f"_norm{norm}_alpha{alpha}_epsilon{epsilon}_steps{num_steps}"
            f"_idx{idx}|adam"
            f"_b1{beta1}_b2{beta2}_ae{adam_eps}"
            f"_sgn{int(use_sign_for_linf)}_ams{int(amsgrad)}.pkl"
        )
        recorder.save(
            save_name,
            metadata={
                "mode_spec": mode_tag,
                "norm": norm,
                "epsilon": epsilon,
                "alpha": alpha,
                "num_steps": num_steps,
                "index": idx,
                "adam_beta1": beta1,
                "adam_beta2": beta2,
                "adam_eps": adam_eps,
                "use_sign_for_linf": use_sign_for_linf,
                "amsgrad": amsgrad,
            },
        )
        return x_adv

if __name__ == "__main__":
    T = 10
    step = 1
    train_test = "test"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据和模型
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_test = torch.load(f'{script_dir}/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_{train_test}_all_frames.pt',
                           weights_only=False, map_location=device)

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
        # {"norm": 2, "epsilon": 0.0002*size**2, "alpha": 0.025*ratio, "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0006*size**2, "alpha": 0.025*ratio, "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0012*size**2, "alpha": 0.025*ratio, "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0002*size**2, "alpha": 0.05*ratio,  "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0006*size**2, "alpha": 0.05*ratio,  "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0012*size**2, "alpha": 0.05*ratio,  "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0002*size**2, "alpha": 0.1*ratio,   "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0006*size**2, "alpha": 0.1*ratio,   "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0012*size**2, "alpha": 0.1*ratio,   "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0002*size**2, "alpha": 0.5*ratio,   "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0006*size**2, "alpha": 0.5*ratio,   "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0012*size**2, "alpha": 0.5*ratio,   "num_steps": 100},
    ]
    print("input_list:", input_list)

    # 从环境变量读取 10 位 a/d/w 控制串
    MODE_SPEC = os.environ.get("MODE_SPEC", "wwwwwwwwww")

    for cfg in input_list:
        for idx in [0,1,2,3,4,5,6,7,8,9]:
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
                "index: {idx}, method: adam, mode_spec: {mode_spec}, "
                "b1: {b1}, b2: {b2}, adam_eps: {ae}, sign_Linf: {sgn}, amsgrad: {ams}".format(
                    norm=norm, epsilon=epsilon, alpha=alpha, num_steps=num_steps, idx=idx,
                    mode_spec=MODE_SPEC, b1=beta1, b2=beta2, ae=adam_eps,
                    sgn=use_sign_for_linf, ams=amsgrad)
            )

            initial_x0 = spectral_upsample(data_test["x"][idx], 256).to(device)
            adversarial_x0 = attack_system.pgd_attack_adam(
                initial_x0=initial_x0,
                epsilon=epsilon,
                alpha=alpha,
                num_steps=num_steps,
                norm=norm,
                idx=idx,
                beta1=beta1,
                beta2=beta2,
                adam_eps=adam_eps,
                use_sign_for_linf=use_sign_for_linf,
                amsgrad=amsgrad,
            )

            print("清理前显存占用:", torch.cuda.memory_allocated()/1024**2, "MB")
            del recurrent_model, attack_system, initial_x0, adversarial_x0
            torch.cuda.empty_cache()
            gc.collect()
            print("PGD攻击完成！显存已清理")
            print("当前显存占用:", torch.cuda.memory_allocated()/1024**2, "MB")





