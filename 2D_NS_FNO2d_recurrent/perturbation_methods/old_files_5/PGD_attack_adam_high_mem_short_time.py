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

def print_versions():
    print("\n" + "="*40 + " 环境诊断 " + "="*40)
    
    # PyTorch 信息
    print("\n[PyTorch]")
    print(f"Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    
    # JAX 信息
    print("\n[JAX]")
    print(f"JAX Version: {jax.__version__}")
    print(f"JAXlib Version: {jaxlib.__version__}")
    print(f"JAX Backend: {jax.lib.xla_bridge.get_backend().platform}")
    try:
        from jax.lib import xla_bridge
        print(f"JAX Devices: {xla_bridge.devices()}")
    except Exception as e:
        print(f"JAX Devices Error: {str(e)}")
    
    # CUDA/cuDNN 系统信息
    print("\n[CUDA/cuDNN System Info]")
    try:
        nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
        print(nvcc_out.split("\n")[0])
    except:
        print("nvcc not found")
    
    # GPU 详细参数
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
    
    # 版本兼容性检查
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

# 执行诊断
print_versions()


jax.config.update("jax_enable_x64", False)

def spectral_upsample(field, target_size=256):
    *batch_dims, H, W = field.shape
    assert target_size >= H and target_size >= W, "目标尺寸必须大于输入尺寸"
    
    # 确保输入在GPU上
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

        # PyTorch → JAX
        a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
        a_jax = jax.dlpack.from_dlpack(a_dlpack)

        # 运行预编译函数
        g_output_jax = g(a_jax)
        out_dlpack = jax.dlpack.to_dlpack(g_output_jax)
        g_output = torch.utils.dlpack.from_dlpack(out_dlpack)

        # 保存用于 backward 的 context
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

        # PyTorch → JAX
        grad_dlpack = torch.utils.dlpack.to_dlpack(grad_output.contiguous())
        grad_jax = jax.dlpack.from_dlpack(grad_dlpack)

        a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
        a_jax = jax.dlpack.from_dlpack(a_dlpack)

        # 调用 VJP
        grad_input_jax, = jitted_vjp(a_jax, grad_jax)

        # JAX → PyTorch
        grad_input_dlpack = jax.dlpack.to_dlpack(grad_input_jax)
        return torch.utils.dlpack.from_dlpack(grad_input_dlpack), None

def generate_sequence_every_second(u0, nu, T_seconds, fixed_step=0.005):
    """
    Given initial state u0, return frames at t=0,1,...,T_seconds (inclusive),
    computed in one rollout. Output shape: [T_seconds+1, H, W]
    """
    num_steps = int(T_seconds / fixed_step)

    # 与单帧版本保持一致的预处理
    u0 = jnp.rot90(jnp.flipud(u0), 3)

    full_ic = jnp.expand_dims(jnp.array(u0, dtype=jnp.float32), axis=0)

    stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
        2, 1, u0.shape[0], fixed_step,
        diffusivity=nu, order=4
    )
    # include_init=True 这样第 0 帧就是初始状态
    rollout_stepper = ex.rollout(stepper, num_steps, include_init=True)  # [num_steps+1, 1, H, W]
    result = rollout_stepper(full_ic)  # JAX array

    # 选择每秒的索引：0, 1s, 2s, ..., T_seconds
    idxs = jnp.array([int(s / fixed_step) for s in range(T_seconds + 1)], dtype=jnp.int32)
    picked = result[idxs, 0, ...]              # [T_seconds+1, H, W]
    picked = jnp.swapaxes(picked, -1, -2)      # 等价于每帧的 .T（与你单帧版本保持一致）

    return picked

class DifferentiablePDESolver:
    def __init__(self, nu, device="cuda"):
        self.nu = nu
        self.device = device
        # 添加性能记录字典
        self.perf_records = {}
    
    def rollout_seconds(self, x0, T_seconds):
        """
        一次性返回 0..T_seconds 的每秒一帧，形状 [T_seconds+1, H, W] 的 torch.Tensor
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        start_mem = torch.cuda.memory_allocated()

        assert isinstance(x0, torch.Tensor) and x0.is_cuda

        def solver_func(a):
            return generate_sequence_every_second(a, self.nu, T_seconds)

        seq_tensor = JaxPDEWrapper.apply(x0.contiguous(), solver_func)  # [T_seconds+1, H, W]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - start_time
        end_mem = torch.cuda.memory_allocated()
        mem_usage = end_mem - start_mem

        # 记录：这次是 0..T_seconds 的整体回放
        self.perf_records[f"rollout_0_{T_seconds}"] = (duration, mem_usage)
        return seq_tensor
    
    def get_perf_records(self):
        """获取并清除性能记录"""
        records = self.perf_records
        self.perf_records = {}
        return records

class AttackRecorder:
    """记录PGD攻击过程中的关键数据"""
    def __init__(self):
        self.steps = []
        self.metadata = {}
    
    def record_step(self, step, x0, output, truth, loss, gradient, reach_boundary=False, scaled_gradient=None):
        """记录每一步攻击的数据"""
        record = {
            "step": step,
            "x0": x0.detach().cpu().numpy().copy(),
            "output": output.detach().cpu().numpy().copy(),
            "truth": truth.detach().cpu().numpy().copy(),
            "loss": float(loss.item()),
            "gradient": gradient.detach().cpu().numpy().copy(),
            "reach_boundary": reach_boundary,
        }

        if scaled_gradient is not None:
            record["scaled_gradient"] = scaled_gradient.detach().cpu().numpy().copy()
        
        self.steps.append(record)
    
    def save(self, filename, metadata=None):
        """保存记录到文件"""
        if metadata:
            self.metadata.update(metadata)
            
        with open(filename, 'wb') as f:
            pickle.dump({
                "steps": self.steps,
                "metadata": self.metadata
            }, f)
    
    @staticmethod
    def load(filename):
        """从文件加载记录"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        recorder = AttackRecorder()
        recorder.steps = data["steps"]
        recorder.metadata = data["metadata"]
        return recorder

class PDEAttackSystem:
    def __init__(self, recurrent_model, nu=1e-5, device="cuda", detach="with_solver"):
        self.recurrent_model = recurrent_model
        self.nu = nu
        self.device = device
        self.detach = detach
        self.t_detached = []
        print(self.detach)
        if self.detach == "with_solver":
            pass
        elif self.detach == "detached5to9":
            self.t_detached += [5,6,7,8,9]
        elif self.detach == "detached1to9":
            self.t_detached += [1,2,3,4,5,6,7,8,9]
        elif self.detach == "detached1to8":
            self.t_detached += [1,2,3,4,5,6,7,8]
        elif self.detach == "detached1to7":
            self.t_detached += [1,2,3,4,5,6,7]
        elif self.detach == "detached1to6":
            self.t_detached += [1,2,3,4,5,6]
        elif self.detach == "detached1to5":
            self.t_detached += [1,2,3,4,5]
        elif self.detach == "detached13579":
            self.t_detached += [1,3,5,7,9]
        elif self.detach == "detached2468":
            self.t_detached += [2,4,6,8]
        else:
            pass
        print(self.t_detached)

        for param in self.recurrent_model.parameters():
            param.requires_grad = False
        self.recurrent_model.eval()

        self.pde_solver = DifferentiablePDESolver(nu, device)

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
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} GB"

    def forward(self, x0):
        # 记录模型部分开始（保持你的计时逻辑）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_start_time = time.perf_counter()
        model_start_mem = torch.cuda.memory_allocated()

        # 一次性求解 0..19 秒
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pde_start_time = time.perf_counter()
        pde_start_mem = torch.cuda.memory_allocated()

        seq_0_to_19 = self.pde_solver.rollout_seconds(x0, 19)  # [20, H, W]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pde_duration = time.perf_counter() - pde_start_time
        pde_mem = torch.cuda.memory_allocated() - pde_start_mem

        # 输入模型用 0..9 秒
        input_sequence = seq_0_to_19[:10]  # [10, H, W]
        # 套用与 generate_input_sequence 相同的 detach 规则（保持一致性）
        frames = []
        for t in range(10):
            f = input_sequence[t]
            if t in self.t_detached:
                f = f.detach()
            frames.append(f)
        input_sequence = torch.stack(frames, dim=0)  # [10, H, W]

        # 组织成模型需要的形状 (1, H, W, 10)
        input_batch = input_sequence.permute(1, 2, 0).unsqueeze(0)

        # 模型预测 10..19 秒
        predictions = self.recurrent_model(input_batch)
        pred_19 = predictions[0, ..., -1]  # 第 19 秒的预测

        # 直接从同一次 PDE 回放里拿第 19 秒真值
        true_19 = seq_0_to_19[19]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_duration = time.perf_counter() - model_start_time
        model_mem = torch.cuda.memory_allocated() - model_start_mem

        # 存储性能数据（名字沿用你的原接口）
        self.model_perf = (model_duration, model_mem)
        self.pde_perf = (pde_duration, pde_mem)

        # 这两项不再逐秒细分，保留空/占位即可，避免其他打印报错
        self.pde_records_0_to_9 = {}
        self.pde_records_19 = {19: (pde_duration, pde_mem)}

        return pred_19, true_19

    def print_detailed_perf(self):
        """打印详细的性能数据"""
        print("\n=== 详细性能报告 ===")
        
        # 打印模型性能
        model_duration, model_mem = self.model_perf
        print(f"[模型部分] 总时间: {model_duration:.4f}s, 显存: {self._format_memory(model_mem)}")
        
        # 打印0-9秒每个时间步的PDE求解性能
        print("\n[0-9秒 PDE求解]")
        for time_step, (duration, mem) in self.pde_records_0_to_9.items():
            print(f"  t={time_step}s - 时间: {duration:.6f}s, 显存: {self._format_memory(mem)}")
        
        # 打印19秒PDE求解性能
        print("\n[19秒 PDE求解]")
        for time_step, (duration, mem) in self.pde_records_19.items():
            print(f"  t={time_step}s - 时间: {duration:.6f}s, 显存: {self._format_memory(mem)}")
        
        # 打印19秒PDE求解的总体性能
        pde_duration, pde_mem = self.pde_perf
        print(f"\n[PDE求解部分] 总时间: {pde_duration:.4f}s, 显存: {self._format_memory(pde_mem)}")
        
        # 计算并打印总性能
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
        detach="with_solver",
        # ---- Adam hyperparams ----
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

            # 进行前向传播
            pred_19, true_19 = self.forward(x_adv)
            
            # 获取性能数据
            model_duration, model_mem = self.model_perf
            pde_duration, pde_mem = self.pde_perf
            pde_records_0_to_9 = self.pde_records_0_to_9  # 新增：获取0-9秒每个时间步的记录
            pde_record_19 = self.pde_records_19           # 新增：获取19秒的记录
            
            # 打印详细性能数据 (修改后的打印格式)
            print(f"\n[Step {step}] ============ 详细性能分析 ============")
            
            # 1. 打印模型部分性能
            print(f"[模型部分] 总时间: {model_duration:.6f}s | 显存: {self._format_memory(model_mem)}")
            
            # 2. 打印0-9秒PDE求解的详细性能
            print("\n[0-9秒 PDE求解详细性能]")
            for t in sorted(pde_records_0_to_9.keys()):
                duration, mem = pde_records_0_to_9[t]
                print(f"  t={t}s - 时间: {duration:.6f}s | 显存: {self._format_memory(mem)}")
            
            # 3. 打印19秒PDE求解性能
            print("\n[19秒 PDE求解]")
            for t in sorted(pde_record_19.keys()):
                duration, mem = pde_record_19[t]
                print(f"  t={t}s - 时间: {duration:.6f}s | 显存: {self._format_memory(mem)}")
            
            # 4. 打印PDE求解总体性能
            print(f"\n[PDE求解部分] 总时间: {pde_duration:.6f}s | 显存: {self._format_memory(pde_mem)}")
            
            # 5. 打印本步骤总性能
            total_duration = model_duration + pde_duration
            total_mem = model_mem + pde_mem
            print(f"\n[本步骤总计] 时间: {total_duration:.6f}s | 显存: {self._format_memory(total_mem)}")

            # 原有损失计算和反向传播保持不变
            loss = F.mse_loss(pred_19, true_19, reduction="sum")
            print("orginal_loss: ", loss)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_start_time = time.perf_counter()
            backward_start_mem = torch.cuda.memory_allocated()
            loss.backward()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._log_performance(step, "Backward Propagation", backward_start_time, backward_start_mem)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            update_start_time = time.perf_counter()
            update_start_mem = torch.cuda.memory_allocated()

            with torch.no_grad():
                # ---------- Adam 缩放 ----------
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

                # ---------- PGD 步进 + 投影 ----------
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
                recorder.record_step(
                    step=step,
                    x0=x_adv.clone().detach(),
                    output=pred_19.clone().detach(),
                    truth=true_19.clone().detach(),
                    loss=loss,
                    gradient=x_adv.grad.clone().detach(),
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

        folder_name = "perturbation_results_1solver"
        os.makedirs(folder_name, exist_ok=True)

        # 文件名里加入 adam 超参
        save_name = (
            f"{folder_name}/"
            f"pgd_attack_records_norm{norm}_alpha{alpha}_epsilon{epsilon}_steps{num_steps}"
            f"_idx{idx}_{detach}|adam"
            f"_b1{beta1}_b2{beta2}_ae{adam_eps}"
            f"_sgn{int(use_sign_for_linf)}_ams{int(amsgrad)}.pkl"
        )

        recorder.save(
            save_name,
            metadata={
                "norm": norm,
                "epsilon": epsilon,
                "alpha": alpha,
                "num_steps": num_steps,
                "index": idx,
                "detached_mode": detach,
                "adam_beta1": beta1,
                "adam_beta2": beta2,
                "adam_eps": adam_eps,
                "use_sign_for_linf": use_sign_for_linf,
                "amsgrad": amsgrad,
            },
        )

        return x_adv

    def _format_memory(self, size):
        """内存格式化辅助函数"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} GB"

if __name__ == "__main__":
    T = 10
    step = 1
    train_test = "test"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据和模型（保持不变）
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
    
    # 主循环
    size = 256
    ratio = 100

    DEFAULT_ADAM = {
        "beta1": 0.9,
        "beta2": 0.999,
        "adam_eps": 1e-8,
        "use_sign_for_linf": True,  # 仅对 L_inf 有效；更稳的话设 True
        "amsgrad": False,
    }

    input_list = [
        # {"norm": 2, "epsilon": 0.0002*size**2, "alpha": 0.01*ratio,  "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0006*size**2, "alpha": 0.01*ratio,  "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0012*size**2,"alpha": 0.01*ratio,  "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0002*size**2, "alpha": 0.025*ratio, "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0006*size**2, "alpha": 0.025*ratio, "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0012*size**2,"alpha": 0.025*ratio, "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0002*size**2, "alpha": 0.05*ratio,  "num_steps": 100},
        {"norm": 2, "epsilon": 0.0006*size**2, "alpha": 0.05*ratio,  "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0012*size**2,"alpha": 0.05*ratio,  "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0002*size**2, "alpha": 0.1*ratio,   "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0006*size**2, "alpha": 0.1*ratio,   "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0012*size**2,"alpha": 0.1*ratio,   "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0002*size**2, "alpha": 0.5*ratio,   "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0006*size**2, "alpha": 0.5*ratio,   "num_steps": 100},
        # {"norm": 2, "epsilon": 0.0012*size**2,"alpha": 0.5*ratio,   "num_steps": 100},
    ]
    print("input_list:", input_list)
    for cfg in input_list:
        for idx in [1]:
            for detach in [
                # "with_solver",
                # "detached5to9",
                # "detached1to9",
                "detached1to5",
                # "detached1to8",
                # "detached1to7",
                # "detached1to6",
                # "detached13579",
                # "detached2468",
            ]:
                # 每次循环创建新的模型实例（确保无残留）
                recurrent_model = RecurrentPredictor(model_instance, T_out=T, step=step).to(device)
                attack_system = PDEAttackSystem(
                    recurrent_model=recurrent_model,
                    nu=1e-5,
                    device=device,
                    detach=detach,
                )

                # 基础参数（若字典里缺失，提供默认值）
                norm      = cfg.get("norm", "inf")
                epsilon   = cfg.get("epsilon", 0.1)
                alpha     = cfg.get("alpha", 0.01)
                num_steps = cfg.get("num_steps", 10)

                # Adam 参数：用 dict 的值覆盖默认
                beta1            = cfg.get("beta1", DEFAULT_ADAM["beta1"])
                beta2            = cfg.get("beta2", DEFAULT_ADAM["beta2"])
                adam_eps         = cfg.get("adam_eps", DEFAULT_ADAM["adam_eps"])
                use_sign_for_linf= cfg.get("use_sign_for_linf", DEFAULT_ADAM["use_sign_for_linf"])
                amsgrad          = cfg.get("amsgrad", DEFAULT_ADAM["amsgrad"])

                print("="*80)
                print(
                    "norm: {norm}, epsilon: {epsilon}, alpha: {alpha}, num_steps: {num_steps}, "
                    "index: {idx}, method: adam, detach: {detach}, "
                    "b1: {b1}, b2: {b2}, adam_eps: {ae}, sign_Linf: {sgn}, amsgrad: {ams}".format(
                        norm=norm, epsilon=epsilon, alpha=alpha, num_steps=num_steps, idx=idx,
                        detach=detach, b1=beta1, b2=beta2, ae=adam_eps,
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
                    detach=detach,
                    beta1=beta1,
                    beta2=beta2,
                    adam_eps=adam_eps,
                    use_sign_for_linf=use_sign_for_linf,
                    amsgrad=amsgrad,
                )

                # 强制清理
                print("清理前显存占用:", torch.cuda.memory_allocated()/1024**2, "MB")
                del recurrent_model, attack_system, initial_x0, adversarial_x0
                torch.cuda.empty_cache()
                gc.collect()
                print("PGD攻击完成！显存已清理")
                print("当前显存占用:", torch.cuda.memory_allocated()/1024**2, "MB")




