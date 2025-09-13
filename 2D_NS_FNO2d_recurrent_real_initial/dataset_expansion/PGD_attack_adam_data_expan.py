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

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_string('mode_spec', 'wwwwwwwwww', '10-char mode_spec using a/d/w.')
flags.DEFINE_float('eps_mult', 0.0002, 'epsilon multiplier: epsilon = eps_mult * size^2')
flags.DEFINE_float('alpha_mult', 0.01, 'alpha multiplier: alpha = alpha_mult * ratio')
flags.DEFINE_integer('norm', 2, 'norm type: 2 for L2, or use string "inf" for Linf')
flags.DEFINE_integer('num_steps', 100, 'PGD steps')
flags.DEFINE_string('train_test', 'train', 'which split to load: train or test')

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
        self.perf_records = {}
    def rollout_seconds(self, x0, T_seconds):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        start_mem = torch.cuda.memory_allocated()
        assert isinstance(x0, torch.Tensor) and x0.is_cuda
        def solver_func(a):
            return generate_sequence_every_second(a, self.nu, T_seconds)
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
        beta1=0.9,
        beta2=0.999,
        adam_eps=1e-8,
        use_sign_for_linf=True,
        amsgrad=False,
    ):
        # 不再使用 AttackRecorder，不保存每步中间结果
        x_adv = initial_x0.clone().detach().to(self.device).requires_grad_(True)
        original_x0 = initial_x0.clone().detach().to(self.device)

        # Adam state
        m = torch.zeros_like(x_adv)
        v = torch.zeros_like(x_adv)
        vhat_max = torch.zeros_like(x_adv) if amsgrad else None
        t_adam = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # PGD 迭代
        for step in range(num_steps):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            # ---- Forward：你当前 forward 在 mode_spec='w'*10 时，true_19 来自 solver ----
            pred_19, true_19 = self.forward(x_adv)

            # 用 true loss 反传（全 'w'，没有 surrogate）
            loss = F.mse_loss(pred_19, true_19, reduction="sum")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            loss.backward()

            # ---- PGD + Adam 更新（与你原逻辑一致）----
            with torch.no_grad():
                t_adam += 1
                g = x_adv.grad
                m = beta1 * m + (1.0 - beta1) * g
                v = beta2 * v + (1.0 - beta2) * (g * g)
                m_hat = m / (1.0 - (beta1 ** t_adam))
                v_hat = v / (1.0 - (beta2 ** t_adam))
                denom = (torch.maximum(vhat_max, v_hat) if amsgrad else v_hat).sqrt() + adam_eps
                d_t = m_hat / denom

                delta = x_adv - original_x0
                if norm in ('inf', 'Linf', '∞'):
                    direction = torch.sign(d_t) if use_sign_for_linf else d_t
                    x_adv.data += alpha * direction
                    delta = x_adv - original_x0
                    x_adv.data = original_x0 + torch.clamp(delta, -epsilon, epsilon)
                elif norm in (2, '2', 'L2'):
                    direction = d_t / (torch.norm(d_t, p=2) + 1e-12)
                    x_adv.data += alpha * direction
                    current_delta = x_adv - original_x0
                    delta_norm = torch.norm(current_delta, p=2)
                    if delta_norm > epsilon:
                        x_adv.data = original_x0 + current_delta * (epsilon / (delta_norm + 1e-12))
                # else: 其它范数不处理

                x_adv.grad.zero_()

        # ====== 攻击结束：仅此时计算并返回最终结果 ======
        # 用最终 x_adv 作为初值，solver rollout 0..20s（得到 21 帧）
        with torch.no_grad():
            seq_0_to_20 = self.pde_solver.rollout_seconds(x_adv, 20)  # [21, H, W] on device
            # 组装 FNO 的输入（t=0 用 x_adv 自身，其余 1..9 用 solver 帧）
            frames = [x_adv] + [seq_0_to_20[t] for t in range(1, 10)]
            input_sequence = torch.stack(frames, dim=0)                   # [10, H, W]
            input_batch = input_sequence.permute(1, 2, 0).unsqueeze(0)    # (1,H,W,10)
            predictions = self.recurrent_model(input_batch)               # (1,H,W,10)
            pred_19_final = predictions[0, ..., -1]                       # (H, W)
            true_19_final = seq_0_to_20[19]                               # (H, W)
            final_true_loss = F.mse_loss(pred_19_final, true_19_final, reduction="sum")

        # y 的存储格式与原数据一致：(H, W, 21)
        y_all_frames = seq_0_to_20.permute(1, 2, 0).contiguous()

        # 只返回最终三件事：x（H,W）、y（H,W,21）、loss（标量float）
        return x_adv.detach(), y_all_frames.detach(), float(final_true_loss.item())

def main(argv):
    global device  # spectral_upsample 用到了这个全局
    T = 10
    step = 1
    train_test = FLAGS.train_test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据和模型
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_raw = torch.load(
        f'{script_dir}/datasets/exponax_datasets/t20/dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_{train_test}_all_frames.pt',
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

    # 本次任务的单组参数（来自 CLI）
    mode_spec = FLAGS.mode_spec
    eps_mult = float(FLAGS.eps_mult)
    alpha_mult = float(FLAGS.alpha_mult)
    norm = FLAGS.norm
    num_steps = int(FLAGS.num_steps)

    # 计算实际数值
    epsilon = eps_mult * (size ** 2)
    alpha = alpha_mult * ratio

    print("=" * 80)
    print(f"[RUN] mode_spec={mode_spec} | eps_mult={eps_mult} -> epsilon={epsilon} | "
          f"alpha_mult={alpha_mult} -> alpha={alpha} | norm={norm} | steps={num_steps} | split={train_test}")

    # 数据集大小
    # N = int(data_raw["x"].shape[0])
    N = 500
    H = 256
    W = 256

    # 预分配输出
    x_out = torch.empty((N, H, W), dtype=torch.float32)
    y_out = torch.empty((N, H, W, 21), dtype=torch.float32)
    loss_out = torch.empty((N,), dtype=torch.float32)

    # 复用同一个模型与系统
    recurrent_model = RecurrentPredictor(model_instance, T_out=T, step=step).to(device)
    attack_system = PDEAttackSystem(
        recurrent_model=recurrent_model,
        nu=1e-5,
        device=device,
        mode_spec=mode_spec,
    )

    DEFAULT_ADAM = {
        "beta1": 0.9,
        "beta2": 0.999,
        "adam_eps": 1e-8,
        "use_sign_for_linf": True,
        "amsgrad": False,
    }
    
    for idx in tqdm(range(N), total=N, desc="PGD attack", unit="sample"):
        initial_x0 = spectral_upsample(data_raw["x"][idx], 256).to(device)

        x_final, y_all_frames, final_true_loss = attack_system.pgd_attack_adam(
            initial_x0=initial_x0,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            norm=norm,
            beta1=DEFAULT_ADAM["beta1"],
            beta2=DEFAULT_ADAM["beta2"],
            adam_eps=DEFAULT_ADAM["adam_eps"],
            use_sign_for_linf=DEFAULT_ADAM["use_sign_for_linf"],
            amsgrad=DEFAULT_ADAM["amsgrad"],
        )

        x_out[idx] = x_final.detach().float().cpu()
        y_out[idx] = y_all_frames.detach().float().cpu()
        loss_out[idx] = float(final_true_loss)

        # 释放显存
        del initial_x0, x_final, y_all_frames
        torch.cuda.empty_cache()

    # 保存（与原数据结构保持一致：x/y，加上 loss；文件名包含参数）
    save_dir = Path(script_dir) / "datasets" / "expanded_exponax_datasets" / "t20" / f"N={N}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 文件名里用实际 alpha/epsilon；避免小数点混乱，格式化为紧凑科学计数
    def fmt(v): return f"{v:.6g}"
    out_path = save_dir / (
        f"dim2d_nx256_N{N}_solver=exponax_nu0.000_t20.0_"
        f"{train_test}_all_frames"
        f"_modespec={mode_spec}"
        f"_norm{norm}_alpha{fmt(alpha)}_epsilon{fmt(epsilon)}_steps{num_steps}.pt"
    )

    torch.save({"x": x_out, "y": y_out, "loss": loss_out}, out_path)
    print(f"✅ Saved perturbed dataset: {out_path}")

if __name__ == "__main__":
    app.run(main)




