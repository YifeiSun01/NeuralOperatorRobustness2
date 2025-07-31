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

import torch.nn.functional as F  
from scipy.spatial.distance import cosine
from torch.autograd.functional import jacobian

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
        # if a_torch.dtype != torch.float64:
        #     raise ValueError("Input must be torch.float64 for gradcheck compatibility")

        try:
            # PyTorch → JAX
            a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
            a_jax = jax.dlpack.from_dlpack(a_dlpack)

            # 运行预编译函数
            g_output_jax = g(a_jax)
            out_dlpack = jax.dlpack.to_dlpack(g_output_jax)
            g_output = torch.utils.dlpack.from_dlpack(out_dlpack)

            ctx.save_for_backward(a_torch)
            ctx.g = g
            return g_output

        except Exception as e:
            import warnings
            warnings.warn(f"GPU transfer failed, fallback to CPU: {str(e)}")
            a_np = a_torch.detach().cpu().numpy()
            g_output_np = g(jnp.array(a_np))
            return torch.as_tensor(np.asarray(g_output_np),
                                   device=a_torch.device,
                                   dtype=a_torch.dtype)

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

        try:
            grad_dlpack = torch.utils.dlpack.to_dlpack(grad_output.contiguous())
            grad_jax = jax.dlpack.from_dlpack(grad_dlpack)

            a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
            a_jax = jax.dlpack.from_dlpack(a_dlpack)

            grad_input_jax, = jitted_vjp(a_jax, grad_jax)
            grad_input_dlpack = jax.dlpack.to_dlpack(grad_input_jax)
            return torch.utils.dlpack.from_dlpack(grad_input_dlpack), None

        except Exception as e:
            import warnings
            warnings.warn(f"Backward GPU failed: {str(e)}")
            grad_np = grad_output.detach().cpu().numpy()
            a_np = a_torch.detach().cpu().numpy()
            _, vjp_fn = jax.vjp(g, jnp.array(a_np))
            grad_input_jax, = vjp_fn(jnp.array(grad_np))
            return torch.as_tensor(np.asarray(grad_input_jax),
                                   device=a_torch.device,
                                   dtype=a_torch.dtype), None

def generate_single_timestep_solution(grf, nu, target_time_k):
    """
    优化版本：只计算并返回target_time_k时刻的解
    参数:
        grf: 输入场 (JAX数组/numpy数组/PyTorch张量)
        nu: 粘度系数
        target_time_k: 目标时间（秒）
    返回:
        目标时刻的解（numpy数组）
    """
    fixed_step = 0.005  
    num_steps = int(target_time_k / fixed_step)

    # 统一输入格式处理
    if isinstance(grf, torch.Tensor):
        u0 = grf.cpu().numpy()
    elif isinstance(grf, (jnp.ndarray, np.ndarray)):
        u0 = np.array(grf)
    else:
        raise TypeError(f"不支持的输入类型: {type(grf)}")
    
    # 预处理（保持原有旋转逻辑）
    u0 = np.rot90(np.flipud(u0), 3)

    # JAX 输入准备
    full_ic = jnp.expand_dims(jnp.array(u0, dtype=jnp.float32), axis=0)

    # 设置 Navier-Stokes vorticity 模型
    stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
        2, 1, grf.shape[0], fixed_step, 
        diffusivity=nu, order=4
    )
    rollout_stepper = ex.rollout(stepper, num_steps, include_init=False)

    print("full_ic shape:", full_ic.shape)
    print("full_ic dtype:", full_ic.dtype)

    # 获取目标时刻结果
    result = rollout_stepper(full_ic)
    print("rollout_stepper output shape:", result.shape)
    final_state = result[-1, 0, ...].T.block_until_ready()
    final_state = np.array(final_state)

    # 数值稳定性检查
    if not np.isfinite(final_state).all():
        raise ValueError(f"数值不稳定！在 {target_time_k} 秒发现非法值（NaN/Inf）")

    return final_state

class DifferentiablePDESolver:
    def __init__(self, nu, device="cuda"):
        self.nu = nu
        self.device = device
        self.cached_solutions = {}
        
    def solve(self, x0, target_time):
        """
        可微分PDE求解器
        :param x0: 初始状态 (torch.Tensor)
        :param target_time: 目标时间
        :return: 目标时间的解 (torch.Tensor)
        """
        # 检查缓存
        cache_key = (tuple(x0.flatten().tolist()), target_time)
        if cache_key in self.cached_solutions:
            return self.cached_solutions[cache_key]
        
        # 转换输入格式
        if isinstance(x0, torch.Tensor):
            x0_np = x0.detach().cpu().numpy()
        else:
            x0_np = np.array(x0)
        
        # 计算目标时间的真实解
        solution = generate_single_timestep_solution(
            x0_np, self.nu, target_time
        )
        
        # 转换为PyTorch张量
        solution_tensor = torch.tensor(
            solution, dtype=torch.float32, device=self.device
        )
        
        # 缓存结果
        self.cached_solutions[cache_key] = solution_tensor
        return solution_tensor
    
    def solve_sequence(self, x0, time_points):
        """
        计算时间序列的解
        :param x0: 初始状态
        :param time_points: 时间点列表
        :return: 解的序列 (torch.Tensor, shape=[len(time_points), H, W])
        """
        solutions = [x0]
        for t in time_points[1:]:
            solutions.append(self.solve(x0, t))
        return torch.stack(solutions, dim=0)

class AttackRecorder:
    """记录PGD攻击过程中的关键数据"""
    def __init__(self):
        self.steps = []
        self.metadata = {}
    
    def record_step(self, step, x0, output, truth, loss, gradient):
        """记录每一步攻击的数据"""
        # 转换为numpy数组以节省空间
        self.steps.append({
            "step": step,
            "x0": x0.detach().cpu().numpy().copy(),
            "output": output.detach().cpu().numpy().copy(),
            "truth": truth.detach().cpu().numpy().copy(),
            "loss": float(loss.item()),
            "gradient": gradient.detach().cpu().numpy().copy()
        })
    
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
        
        # 冻结模型参数
        for param in self.recurrent_model.parameters():
            param.requires_grad = False
        self.recurrent_model.eval()
        
        # 创建PDE求解器
        self.pde_solver = DifferentiablePDESolver(nu, device)

    def _log_performance(self, step, operation, start_time, start_mem):
        """记录性能指标并打印日志"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保CUDA操作完成
        
        duration = time.perf_counter() - start_time  # 使用perf_counter
        current_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        mem_usage = current_mem - start_mem
        
        def format_memory(size):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f"{size:.2f} {unit}"
                size /= 1024
            return f"{size:.2f} GB"
        
        print(f"[Step {step}] {operation} - "
            f"Time: {duration:.4f}s, "
            f"Mem: {format_memory(mem_usage)}")
        return duration, mem_usage
    
    def generate_input_sequence(self, x0):
        """生成0-9秒的输入序列（简化版，直接使用PDE求解器）"""
        time_points = list(range(10))  # 0-9秒
        x_list = [x0]  # 初始状态
        
        for t in time_points[1:]:
            x_t = self.pde_solver.solve(x0, t)  # 自动使用JaxPDEWrapper计算梯度
            x_list.append(x_t)
        
        return torch.stack(x_list, dim=0)  # (T=10, H, W)

    def forward(self, x0):
        """完整的前向传播流程"""
        # 1. 深度学习模型前向传播
        model_start_time = time.perf_counter()  # 改为perf_counter
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_start_mem = torch.cuda.memory_allocated()
        
        # 生成输入序列 (0-9秒)
        input_sequence = self.generate_input_sequence(x0)
        
        # 增加batch维度 (1, H, W, 10)
        input_batch = input_sequence.permute(1, 2, 0).unsqueeze(0)
        
        # 通过循环模型预测10-19秒
        predictions = self.recurrent_model(input_batch)
        
        # 提取预测的最后一帧 (第19秒)
        pred_19 = predictions[0, ..., -1]
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_duration = time.perf_counter() - model_start_time
        model_mem = torch.cuda.memory_allocated() - model_start_mem
        
        # 2. PDE求解器前向传播
        pde_start_time = time.perf_counter()  # 改为perf_counter
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pde_start_mem = torch.cuda.memory_allocated()
        
        # 计算真实解 (第19秒)
        true_19 = self.pde_solver.solve(x0, 19.0)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pde_duration = time.perf_counter() - pde_start_time
        pde_mem = torch.cuda.memory_allocated() - pde_start_mem
        
        # 存储性能数据（现在存储的是开始时间而非持续时间）
        self.model_perf = (model_start_time, model_start_mem)
        self.pde_perf = (pde_start_time, pde_start_mem)
        
        return pred_19, true_19
    
    def pgd_attack(self, initial_x0, epsilon=0.1, alpha=0.01, num_steps=10):
        """执行PGD攻击（保留完整监控和记录功能）"""
        recorder = AttackRecorder()
        x_adv = initial_x0.clone().detach().to(self.device).requires_grad_(True)
        original_x0 = initial_x0.clone().detach().to(self.device)
        
        if cuda.is_available():
            cuda.reset_peak_memory_stats()
        
        for step in range(num_steps):
            step_start_time = time.perf_counter()  # 改为perf_counter
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start_mem = torch.cuda.memory_allocated()
            
            # 完整前向传播
            pred_19, true_19 = self.forward(x_adv)
            
            # 获取性能数据（现在获取的是开始时间戳）
            model_start_time, model_start_mem = self.model_perf
            pde_start_time, pde_start_mem = self.pde_perf
            
            # 记录性能（传递原始时间戳）
            self._log_performance(step, "Deep Model Forward", model_start_time, model_start_mem)
            self._log_performance(step, "PDE Solver Forward", pde_start_time, pde_start_mem)
            
            # 3. 计算损失
            loss_start_time = time.time()
            loss = F.mse_loss(pred_19, true_19)
            loss_duration = time.time() - loss_start_time
            
            # 4. 反向传播
            backward_start_time = time.time()
            backward_start_mem = cuda.memory_allocated() if cuda.is_available() else 0
            loss.backward()
            backward_duration = time.time() - backward_start_time
            backward_mem = (cuda.memory_allocated() if cuda.is_available() else 0) - backward_start_mem
            
            # 记录反向传播性能
            self._log_performance(step, "Backward Propagation", backward_duration, backward_mem)
            
            # 记录总步骤性能
            step_duration = time.time() - step_start_time
            step_mem = (cuda.memory_allocated() if cuda.is_available() else 0) - step_start_mem
            self._log_performance(step, "Total Step", step_duration, step_mem)
            
            # 记录当前状态
            recorder.record_step(
                step=step,
                x0=x_adv.clone().detach(),
                output=pred_19.clone().detach(),
                truth=true_19.clone().detach(),
                loss=loss,
                gradient=x_adv.grad.clone().detach()
            )
            
            # 5. PGD更新
            update_start_time = time.time()
            update_start_mem = cuda.memory_allocated() if cuda.is_available() else 0
            with torch.no_grad():
                perturbation = alpha * x_adv.grad.sign()
                x_adv.data += perturbation
                delta = torch.clamp(x_adv - original_x0, -epsilon, epsilon)
                x_adv.data = original_x0 + delta
                x_adv.grad.zero_()
            update_duration, update_mem = self._log_performance(
                step, "PGD Update", update_start_time, update_start_mem
            )
            
            # 记录整体性能
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_duration = time.perf_counter() - step_start_time
            total_mem = torch.cuda.memory_allocated()
            print(f"[Step {step}] Total Step - "
                f"Time: {total_duration:.4f}s, "
                f"Mem: {self._format_memory(total_mem - step_start_mem)}")
            
            if cuda.is_available():
                peak_mem = cuda.max_memory_allocated()
                print(f"[Step {step}] Peak Memory: {self._format_memory(peak_mem)}")
                cuda.reset_peak_memory_stats()
        
        recorder.save("pgd_attack_records.pkl")
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
    # train_test = "train"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load the data file
    data_test = torch.load(f'{script_dir}/datasets/exponax_datasets/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_{train_test}_all_frames.pt', 
                           weights_only=False, map_location=torch.device(device))
    model_instance = FNO2d(
        modes1=64, 
        modes2=64, 
        width=60
    )
    state_dict = torch.load(
        f'{script_dir}/saved_models/2D/modes64_width60_epochs500/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth',
        map_location=torch.device(device)
    )
    model_instance.load_state_dict(state_dict)
    model_instance.eval() 
    recurrent_model = RecurrentPredictor(model_instance, T_out=T, step=step).to(device)

    # 2. 准备初始状态（第0帧）
    # 这里使用随机数据作为示例，您应该替换为真实数据
    idx = 0
    initial_x0 = spectral_upsample(data_test["x"][idx], 256)  # 形状(H, W)

    print(f"Model device: {next(recurrent_model.parameters()).device}")
    print(f"Input device: {initial_x0.device}")

    # 3. 创建攻击系统
    attack_system = PDEAttackSystem(
        recurrent_model=recurrent_model,  # 替换为您的预训练模型
        nu=1e-5,                          # 粘度系数
        device=device                     # 计算设备
    )

    # 4. 执行PGD攻击
    epsilon = 0.1   # 扰动边界
    alpha = 0.01     # 攻击步长
    num_steps = 10   # 攻击迭代次数

    adversarial_x0 = attack_system.pgd_attack(
        initial_x0=initial_x0,
        epsilon=epsilon,
        alpha=alpha,
        num_steps=num_steps
    )

    # 5. 保存结果
    torch.save(adversarial_x0, "adversarial_input.pt")
    print("PGD攻击完成！对抗样本已保存")

    # 6. 分析结果
    # 加载攻击记录
    with open("pgd_attack_records.pkl", 'rb') as f:
        attack_records = pickle.load(f)

    # 打印最终损失
    final_loss = attack_records["steps"][-1]["loss"]
    print(f"最终损失值: {final_loss:.6f}")

    # 计算并打印扰动大小
    perturbation = torch.norm(adversarial_x0 - initial_x0).item()
    print(f"最终扰动大小: {perturbation:.6f}")

    """
    # 可视化攻击过程（可选）
    import matplotlib.pyplot as plt

    # 提取损失值
    losses = [step["loss"] for step in attack_records["steps"]]

    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o')
    plt.title("PGD Attack Progress")
    plt.xlabel("Attack Step")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("attack_progress.png")
    plt.close()

    print("攻击过程可视化已保存为 attack_progress.png")
    """
