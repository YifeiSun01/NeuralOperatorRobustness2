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
            
def generate_single_timestep_solution(u0, nu, target_time_k):
    fixed_step = 0.005  
    num_steps = int(target_time_k / fixed_step)
    
    # 预处理（保持原有旋转逻辑）
    u0 = jnp.rot90(jnp.flipud(u0), 3)

    # JAX 输入准备
    full_ic = jnp.expand_dims(jnp.array(u0, dtype=jnp.float32), axis=0)

    # 设置 Navier-Stokes vorticity 模型
    stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
        2, 1, u0.shape[0], fixed_step, 
        diffusivity=nu, order=4
    )
    rollout_stepper = ex.rollout(stepper, num_steps, include_init=False)

    # print("full_ic shape:", full_ic.shape)
    # print("full_ic dtype:", full_ic.dtype)

    # 获取目标时刻结果
    result = rollout_stepper(full_ic)
    # print("rollout_stepper output shape:", result.shape)
    final_state = result[-1, 0, ...].T

    return final_state

class DifferentiablePDESolver:
    def __init__(self, nu, device="cuda"):
        self.nu = nu
        self.device = device
        # 添加性能记录字典
        self.perf_records = {}
        
    def solve(self, x0, target_time):
        """
        可微分PDE求解器
        :param x0: 初始状态 (torch.Tensor)
        :param target_time: 目标时间
        :return: 目标时间的解 (torch.Tensor)
        """
        # 记录PDE求解开始
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        start_mem = torch.cuda.memory_allocated()
        
        assert isinstance(x0, torch.Tensor), "输入必须是torch.Tensor"
        assert x0.is_cuda, "输入必须在GPU上"
        
        def solver_func(a):
            return generate_single_timestep_solution(a, self.nu, target_time)

        # 计算目标时间的真实解
        solution_tensor = JaxPDEWrapper.apply(x0.contiguous(), solver_func)
        
        # 记录PDE求解结束
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - start_time
        end_mem = torch.cuda.memory_allocated()
        mem_usage = end_mem - start_mem
        
        # 记录性能数据
        self.perf_records[target_time] = (duration, mem_usage)
        
        return solution_tensor
    
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
    
    def record_step(self, step, x0, output, truth, loss, gradient, numerical_gradient=None, reach_boundary=False):
        """记录每一步攻击的数据"""
        record = {
            "step": step,
            "x0": x0.detach().cpu().numpy().copy(),
            "output": output.detach().cpu().numpy().copy(),
            "truth": truth.detach().cpu().numpy().copy(),
            "loss": float(loss.item()),
            "gradient": gradient.detach().cpu().numpy().copy(),
            "reach_boundary": reach_boundary
        }
        
        if numerical_gradient is not None:
            record["numerical_gradient"] = numerical_gradient.detach().cpu().numpy().copy()
        
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
    def __init__(self, recurrent_model, nu=1e-5, device="cuda"):
        self.recurrent_model = recurrent_model
        self.nu = nu
        self.device = device

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

    def generate_input_sequence(self, x0):
        time_points = list(range(10))
        x_list = [x0]
        # 记录0秒性能（初始状态不计算）
        self.pde_solver.perf_records[0] = (0.0, 0)
        
        for t in time_points[1:]:
            # 记录每个时间步的求解性能
            x_t = self.pde_solver.solve(x0, t)
            if t >= 1 and t <= 9:
                x_t = x_t.detach()
            x_list.append(x_t)
        
        return torch.stack(x_list, dim=0)

    def forward(self, x0):
        # 记录模型部分开始
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_start_time = time.perf_counter()
        model_start_mem = torch.cuda.memory_allocated()
        
        # 生成输入序列 (0-9秒) - 包含多次PDE求解
        input_sequence = self.generate_input_sequence(x0)
        
        # 获取0-9秒的PDE求解性能记录
        pde_records_0_to_9 = self.pde_solver.get_perf_records()
        
        # 增加batch维度 (1, H, W, 10)
        input_batch = input_sequence.permute(1, 2, 0).unsqueeze(0)
        
        # 通过循环模型预测10-19秒 - 多次模型调用
        predictions = self.recurrent_model(input_batch)
        
        # 提取预测的最后一帧 (第19秒)
        pred_19 = predictions[0, ..., -1]
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_duration = time.perf_counter() - model_start_time
        model_mem = torch.cuda.memory_allocated() - model_start_mem
        
        # 记录PDE求解部分开始 (第19秒)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pde_start_time = time.perf_counter()
        pde_start_mem = torch.cuda.memory_allocated()
        
        # 计算真实解 (第19秒)
        true_19 = self.pde_solver.solve(x0, 19.0)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pde_duration = time.perf_counter() - pde_start_time
        pde_mem = torch.cuda.memory_allocated() - pde_start_mem
        
        # 获取19秒的求解性能记录
        pde_records_19 = self.pde_solver.get_perf_records()
        
        # 存储性能数据
        self.model_perf = (model_duration, model_mem)
        self.pde_perf = (pde_duration, pde_mem)
        self.pde_records_0_to_9 = pde_records_0_to_9
        self.pde_records_19 = pde_records_19
        
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

    def pgd_attack(self, initial_x0, epsilon=0.1, alpha=0.01, num_steps=10, norm="inf",
               enable_numerical_check=False, delta=1e-6, need_full_resolution=False, idx=0):
        recorder = AttackRecorder()
        x_adv = initial_x0.clone().detach().to(self.device).requires_grad_(True)
        print("x_adv.shape: ", x_adv.shape)
        original_x0 = initial_x0.clone().detach().to(self.device)

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
            # make_dot(loss, params={"x_adv": x_adv}).render(f"perturbation_results/fno_grad_graph_alpha{alpha}_epsilon{epsilon}_steps{num_steps}_step{step}", format="pdf")
            loss.backward()

            """
            def forward_fn(x):
                pred_19, true_19 = self.forward(x)
                return F.mse_loss(pred_19, true_19, reduction="sum")
            
            # 检查梯度
            x_test = x_adv.detach().clone().requires_grad_(True)

            # 定义不同的容差组合
            tolerance_combinations = [
                {"atol": 1e-3, "rtol": 1e-2, "name": "宽松"},
                {"atol": 1e-4, "rtol": 1e-3, "name": "默认"},
                {"atol": 1e-5, "rtol": 1e-4, "name": "严格"},
                {"atol": 1e-6, "rtol": 1e-5, "name": "超严格"},
            ]

            print("\n梯度检查结果比较:")
            for tol in tolerance_combinations:
                test_result = gradcheck(
                    forward_fn,
                    x_test,
                    eps=delta,
                    atol=tol["atol"],
                    rtol=tol["rtol"],
                    raise_exception=False
                )
                print(f"{tol['name']}标准 (atol={tol['atol']:.0e}, rtol={tol['rtol']:.0e}): {'✅通过' if test_result else '❌失败'}")
            """    

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._log_performance(step, "Backward Propagation", backward_start_time, backward_start_mem)
            
            print("x_adv.requires_grad: ", x_adv.requires_grad)
            print("pred_19.requires_grad, true_19.requires_grad: ", pred_19.requires_grad, true_19.requires_grad)

            print("Autograd grad max: ", x_adv.grad.max())
            print("Autograd grad min: ", x_adv.grad.min())
            print("Autograd grad mean: ", x_adv.grad.mean())

            if enable_numerical_check:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                num_grad_start_time = time.perf_counter()
                num_grad_start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                h, w = x_adv.shape[-2:]
                stride = 32
                sampled_h, sampled_w = (h + stride - 1) // stride, (w + stride - 1) // stride
                numerical_grad_sampled = torch.zeros(sampled_h, sampled_w, device=x_adv.device)
                
                delta = 1e-4
                total_points = sampled_h * sampled_w
                computed_points = 0
                
                print(f"\n开始数值梯度计算，共需计算 {total_points} 个点（步长 {stride}）")
                
                # with torch.no_grad(), torch.cuda.amp.autocast():
                with torch.no_grad():
                    start_time = time.time()  # 记录整体开始时间
                    for i_idx, i in enumerate(range(0, h, stride)):
                        for j_idx, j in enumerate(range(0, w, stride)):
                            point_start_time = time.time()  # 单点开始时间
                            computed_points = i_idx * sampled_w + j_idx + 1
                            
                            # --- 原有计算逻辑 ---
                            mask = torch.zeros_like(x_adv)
                            mask[i,j] = delta
                            pred_plus, true_plus = self.forward(x_adv + mask)
                            print("pred plus diff: ",F.mse_loss(pred_plus, pred_19, reduction="sum"),", true plus diff: ",F.mse_loss(true_plus, true_19, reduction="sum"))
                            loss_plus = F.mse_loss(pred_plus, true_plus, reduction="sum")
                            print("loss_plus: ", loss_plus)
                            pred_minus, true_minus = self.forward(x_adv - mask)
                            print("pred minus diff: ",F.mse_loss(pred_minus, pred_19, reduction="sum"),", true minus diff: ",F.mse_loss(true_minus, true_19, reduction="sum"))
                            loss_minus = F.mse_loss(pred_minus, true_minus, reduction="sum")
                            print("loss_minus: ", loss_minus)
                            numerical_grad_sampled[i_idx, j_idx] = (loss_plus - loss_minus) / (2 * delta)
                            print("gradient at this entry: ", numerical_grad_sampled[i_idx, j_idx])
                            # -------------------
                            
                            # 计算时间统计
                            point_time = time.time() - point_start_time  # 当前点耗时
                            total_time = time.time() - start_time       # 累计总耗时
                            avg_time = total_time / computed_points    # 平均每点耗时
                            remaining_time = avg_time * (total_points - computed_points)  # 预计剩余时间
                            
                            # 进度打印（带时间信息）
                            print(
                                f"进度: ({i},{j}) [{computed_points}/{total_points}] | "
                                f"当前点耗时: {point_time:.2f}s | "
                                f"累计: {total_time//60:.0f}m{total_time%60:.0f}s | "
                                f"剩余预估: {remaining_time//60:.0f}m{remaining_time%60:.0f}s", 
                                end='\r'
                            )
                            
                            if computed_points % 10 == 0 and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                
                # 最终进度显示
                print(f"\n数值梯度计算完成！共计算 {computed_points} 个点")
                
                if need_full_resolution:
                    numerical_grad = F.interpolate(
                        numerical_grad_sampled.unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode='bilinear'
                    ).squeeze()
                else:
                    numerical_grad = numerical_grad_sampled
                
                self._log_performance(step, "Numerical Gradient", num_grad_start_time, num_grad_start_mem)

            # 记录步骤数据 (更新后的record_step)
            

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            update_start_time = time.perf_counter()
            update_start_mem = torch.cuda.memory_allocated()

            with torch.no_grad():
                if norm == 'inf':
                    # L∞攻击（保持原始2D操作）
                    perturbation = alpha * x_adv.grad.sign()
                    x_adv.data += perturbation
                    delta_2 = x_adv - original_x0
                    at_upper_boundary = (delta_2 >= epsilon - 1e-6).float() 
                    at_lower_boundary = (delta_2 <= -epsilon + 1e-6).float()
                    reach_boundary = (at_upper_boundary + at_lower_boundary).mean().item() * 100
                    delta = torch.clamp(delta_2, -epsilon, epsilon)
                    x_adv.data = original_x0 + delta

                elif norm == 2:
                    reach_boundary = False
                    # L2攻击（完全2D实现）
                    grad = x_adv.grad
                    
                    # 计算全局L2范数（单张2D图像）
                    grad_norm = torch.norm(grad, p=2)  # 标量值
                    normalized_grad = grad / (grad_norm + 1e-10)
                    
                    # 应用扰动
                    x_adv.data += alpha * normalized_grad
                    
                    # 计算当前扰动范数
                    current_delta = x_adv - original_x0
                    delta_norm = torch.norm(current_delta, p=2)
                    
                    # 投影到L2球
                    if delta_norm > epsilon:
                        x_adv.data = original_x0 + current_delta * (epsilon / delta_norm)
                        reach_boundary = True

                recorder.record_step(
                    step=step,
                    x0=x_adv.clone().detach(),
                    output=pred_19.clone().detach(),
                    truth=true_19.clone().detach(),
                    loss=loss,
                    gradient=x_adv.grad.clone().detach(),
                    numerical_gradient=numerical_grad.clone().detach() if enable_numerical_check else None,
                    reach_boundary=reach_boundary
                )
                
                # 清零梯度
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
        
        os.makedirs("perturbation_results", exist_ok=True)
        recorder.save(f"perturbation_results/pgd_attack_records_norm{norm}_alpha{alpha}_epsilon{epsilon}_steps{num_steps}_idx{idx}_detached1to9.pkl", metadata={"norm":norm, "epsilon":epsilon, "alpha":alpha, "num_steps":num_steps, "index":idx})
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
    """
    input_list = [
        # (2, 0.1*size**2, 0.01*ratio, 100),
        # (2, 0.1*size**2, 0.05*ratio, 100),
        # (2, 0.5*size**2, 0.1*ratio, 100),
        # (2, 0.5*size**2, 0.05*ratio, 100),
        # (2, 0.5*size**2, 0.01*ratio, 100),
        # (2, 1*size**2, 0.5*ratio, 100),
        # (2, 1*size**2, 0.1*ratio, 100),
        # (2, 1*size**2, 1*ratio, 100),
        # (2, 0.1*size**2, 0.1*ratio, 100),
        # (2, 0.01*size**2, 0.1*ratio, 100),
        # (2, 0.001*size**2, 0.1*ratio, 100),

        (2, 0.0001*size**2, 0.01*ratio, 100),
        (2, 0.0001*size**2, 0.02*ratio, 100),
        (2, 0.0001*size**2, 0.05*ratio, 100),

        # ("inf", 0.1, 0.05, 100),
        # ("inf", 0.1, 0.01, 100),
        # ("inf", 0.5, 0.05, 100),
        # ("inf", 0.5, 0.01, 100),
        # ("inf", 1, 0.01, 100),
        # ("inf", 1, 0.05, 100),
        # ("inf", 1, 0.1, 100),
        ]
    """

    input_list = [
        (2, 0.0002*size**2, 0.01*ratio, 100),
        # (2, 0.0004*size**2, 0.02*ratio, 100),
        (2, 0.0006*size**2, 0.05*ratio, 100),
    ]
    print("input_list: ", input_list)
    for input in input_list:
            # for idx in [1,2,3]:
            for idx in [1]:
                # 每次循环创建新的模型实例（确保无残留）
                recurrent_model = RecurrentPredictor(model_instance, T_out=T, step=step).to(device)
                attack_system = PDEAttackSystem(
                    recurrent_model=recurrent_model,
                    nu=1e-5,
                    device=device
                )
                
                norm, epsilon, alpha, num_steps = input
                print("="*80)
                print(f"norm: {norm}, epsilon: {epsilon}, alpha: {alpha}, num_steps: {num_steps}, index: {idx}")
                
                initial_x0 = spectral_upsample(data_test["x"][idx], 256).to(device)
                adversarial_x0 = attack_system.pgd_attack(
                    initial_x0=initial_x0,
                    epsilon=epsilon,
                    alpha=alpha,
                    num_steps=num_steps,
                    norm=norm,
                    idx=idx,
                )
                
                # 强制清理
                print("清理前显存占用:", torch.cuda.memory_allocated()/1024**2, "MB")
                del recurrent_model, attack_system, initial_x0, adversarial_x0
                torch.cuda.empty_cache()
                gc.collect()
                
                print("PGD攻击完成！显存已清理")
                print("当前显存占用:", torch.cuda.memory_allocated()/1024**2, "MB")

