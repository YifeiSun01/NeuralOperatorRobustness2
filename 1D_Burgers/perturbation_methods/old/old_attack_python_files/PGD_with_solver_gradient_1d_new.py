import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO1d import FNO1d
from GRFs.generateGRFs import GRFGenerator
import importlib
# from solvers import burgers1d_solvers
# importlib.reload(burgers1d_solvers)
from solvers.burgers1d_solvers import *
import torch
from tqdm import tqdm
from pathlib import Path
import pickle
import jax
from torch2jax import t2j
import jax.numpy as jnp
from torch.func import jacrev

import torch
import jax
import jax.numpy as jnp
import numpy as np
import gc
from torch.cuda.amp import autocast, GradScaler


class JaxPDEWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_torch, g):
        # PyTorch -> JAX（无需 detach）
        a_jax = jnp.array(a_torch.cpu().numpy())
        g_output_jax = g(a_jax)
        ctx.save_for_backward(a_torch)
        ctx.g = g
        # JAX -> PyTorch（高效转换）
        return torch.as_tensor(np.asarray(g_output_jax), device=a_torch.device, dtype=a_torch.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a_torch, = ctx.saved_tensors
        g = ctx.g
        # PyTorch -> JAX
        a_jax = jnp.array(a_torch.cpu().numpy())
        # 计算 VJP
        _, vjp_fun = jax.vjp(g, a_jax)
        # 确保上游梯度是 NumPy 数组（兼容多维）
        grad_out_np = np.asarray(grad_output.cpu().numpy())
        grad_input_jax, = vjp_fun(grad_out_np)
        # JAX -> PyTorch
        grad_input = torch.as_tensor(np.asarray(grad_input_jax), device=a_torch.device, dtype=a_torch.dtype)
        return grad_input, None  # 对 g 的梯度为 None


    """
class JaxPDEWrapper(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, a_torch, g):
        # 转换 PyTorch 张量到 JAX 数组
        a_jax = jnp.array(a_torch.detach().cpu().numpy())
        
        # 执行 JAX PDE 求解
        g_output_jax = g(a_jax)  # 你的 JAX PDE 求解函数
        
        # 保存输入用于反向传播
        ctx.save_for_backward(a_torch)
        ctx.g = g
        
        # 转换回 PyTorch 张量
        return torch.from_numpy(np.array(g_output_jax)).to(a_torch.device).to(a_torch.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        a_torch, = ctx.saved_tensors
        g = ctx.g

        # 1. PyTorch->NumPy->JAX
        a_jax = jnp.array(a_torch.detach().cpu().numpy())
        # 2. 前向计算一次，获取 vjp 函数
        y_jax, vjp_fun = jax.vjp(g, a_jax)
        # 3. 将上游梯度转为 NumPy
        grad_out_np = np.array(grad_output.detach().cpu().numpy())
        # 4. 直接计算 VJP
        grad_input_jax, = vjp_fun(grad_out_np)
        # 5. 转回 PyTorch
        grad_input = torch.from_numpy(np.array(grad_input_jax)) \
                        .to(a_torch.device).to(a_torch.dtype)
        return grad_input, None
    """




    """
    @staticmethod
    def backward(ctx, grad_output):
        a_torch, = ctx.saved_tensors
        g = ctx.g

        # 1. PyTorch->NumPy->JAX
        a_np = a_torch.detach().cpu().numpy()
        a_jax = jnp.array(a_np)
        print(f"a_np:      shape={a_np.shape},      type={type(a_np)}")
        print(f"a_jax:     shape={a_jax.shape},     type={type(a_jax)}")

        # 2. 前向计算一次，获取 VJP 函数
        y_jax, vjp_fun = jax.vjp(g, a_jax)
        print(f"y_jax:     shape={y_jax.shape},     type={type(y_jax)}")
        print(f"vjp_fun:   type={type(vjp_fun)}")  # vjp_fun 是 callable，不含 shape

        # 3. 将上游梯度转为 NumPy
        grad_out_np = np.array(grad_output.detach().cpu().numpy())
        print(f"grad_out_np:       shape={grad_out_np.shape},       type={type(grad_out_np)}")

        # 4. 直接计算 VJP
        grad_input_jax, = vjp_fun(grad_out_np)
        print(f"grad_input_jax:    shape={grad_input_jax.shape},    type={type(grad_input_jax)}")

        # 5. 转回 PyTorch
        grad_input_np = np.array(grad_input_jax)
        grad_input = torch.from_numpy(grad_input_np) \
                        .to(a_torch.device).to(a_torch.dtype)
        print(f"grad_input_np:     shape={grad_input_np.shape},     type={type(grad_input_np)}")
        print(f"grad_input:        shape={grad_input.shape},        type={type(grad_input)}")

        return grad_input, None
    """


    """
    @staticmethod
    def backward(ctx, grad_output):
        a_torch, = ctx.saved_tensors
        g = ctx.g

        # 1. PyTorch->NumPy->JAX
        a_jax = jnp.array(a_torch.detach().cpu().numpy())
        print("a_jax.shape: ", a_jax.shape)
        # 2. 前向计算一次，获取 vjp 函数
        y_jax, vjp_fun = jax.vjp(g, a_jax)
        print("a_jax.shape: ", a_jax.shape)
        # 3. 将上游梯度转为 NumPy
        grad_out_np = np.array(grad_output.detach().cpu().numpy())
        print("grad_out_np.shape: ", grad_out_np.shape)
        # 4. 直接计算 VJP
        grad_input_jax, = vjp_fun(grad_out_np)
        print("grad_input_jax.shape: ", grad_input_jax.shape)
        # 5. 转回 PyTorch
        grad_input = torch.from_numpy(np.array(grad_input_jax)) \
                        .to(a_torch.device).to(a_torch.dtype)
        print("grad_input.shape: ", grad_input.shape)
        return grad_input, None
    """
    

    

    """
    @staticmethod
    def backward(ctx, grad_output):
        a_torch, = ctx.saved_tensors
        g = ctx.g

        # 1. 转为 JAX 数组
        a_jax = jnp.array(a_torch.detach().cpu().numpy())
        print("a_jax.shape:", a_jax.shape)

        # 2. 定义单样本函数
        def g_jax_single(a_single):
            return g(a_single[None])[0]

        # 3. 批量雅可比
        jac_fn = jax.vmap(jax.jacrev(g_jax_single))
        jac_jax = jac_fn(a_jax)
        print("jac_jax.shape (JAX):", jac_jax.shape)

        # 4. 转回 PyTorch
        jac_torch = torch.from_numpy(np.array(jac_jax)) \
                        .to(a_torch.device).to(a_torch.dtype)
        print("jac_torch.shape (Torch):", jac_torch.shape)

        # grad_output 形状检查
        print("grad_output.shape (before):", grad_output.shape)
        if grad_output.dim() == 1:
            grad_output = grad_output.unsqueeze(0)
        print("grad_output.shape (after unsqueeze if any):", grad_output.shape)

        # 5. 链式法则合并
        grad_input = torch.einsum('bi,bij->bj', grad_output, jac_torch)
        print("grad_input.shape:", grad_input.shape)

        return grad_input, None
    """

    """   
    @staticmethod
    def backward(ctx, grad_output):
        # 获取前向传播保存的输入
        a_torch, = ctx.saved_tensors
        g = ctx.g
        
        # 定义 JAX 梯度函数
        def g_jax(a_jax):
            return g(a_jax)  # 你的 JAX PDE 求解函数
        
        # 计算 JAX 梯度
        jac_fn = jax.jacrev(g_jax)
        jac_jax = jac_fn(jnp.array(a_torch.cpu().numpy()))
        
        # 转换梯度回 PyTorch
        jac_torch = torch.from_numpy(np.array(jac_jax)).to(a_torch.device).to(a_torch.dtype)
        
        # 计算最终梯度 (链式法则)
        grad_input = torch.tensordot(grad_output, jac_torch, dims=grad_output.ndim)
        
        return grad_input, None  # 第二个 None 对应 pde_params 的梯度
    """


    
    """ 
    @staticmethod
    def backward(ctx, grad_output):
        a_torch, = ctx.saved_tensors
        g = ctx.g
        
        # 转换梯度到 JAX
        grad_output_jax = jnp.array(grad_output.detach().cpu().numpy())
        
        # 定义 VJP 函数
        def g_vjp(a_jax):
            # 创建 JAX 可追踪的输入
            a_jax_tracked = jax.numpy.array(a_jax)
            # 前向计算并获取 VJP 函数
            _, vjp_fn = jax.vjp(g, a_jax_tracked)
            return vjp_fn(grad_output_jax)[0]  # 返回梯度
        
        # 计算梯度
        grad_input_jax = g_vjp(jnp.array(a_torch.cpu().numpy()))
        
        # 转换回 PyTorch
        return torch.from_numpy(np.array(grad_input_jax)).to(a_torch.device), None
    """



"""
class JaxPDEWrapper(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, a_torch, g):
        # 分离计算图并转换为JAX数组
        a_jax = jnp.array(a_torch.detach().cpu().numpy())
        
        # 执行JAX PDE求解
        g_output_jax = g(a_jax)
        
        # 保存输入和函数用于反向传播
        ctx.save_for_backward(a_torch)
        ctx.g = g
        
        # 转换结果回PyTorch张量
        return torch.from_numpy(np.array(g_output_jax)).to(a_torch.device).to(a_torch.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        a_torch, = ctx.saved_tensors
        g = ctx.g
        
        # 将输入和梯度转换为JAX数组
        a_jax = jnp.array(a_torch.detach().cpu().numpy())
        grad_output_jax = jnp.array(grad_output.detach().cpu().numpy())
        
        # 定义JAX前向函数
        def g_jax(a):
            return g(a)
        
        # 使用vjp计算向量-雅可比乘积
        _, vjp_fn = jax.vjp(g_jax, a_jax)
        grad_input_jax = vjp_fn(grad_output_jax)[0]  # 获取梯度
        
        # 转换梯度回PyTorch张量
        grad_input_torch = torch.from_numpy(np.array(grad_input_jax))
        grad_input_torch = grad_input_torch.to(a_torch.device).to(a_torch.dtype)
        
        return grad_input_torch, None
"""





"""
scaler = GradScaler("cuda")

def pgd_attack(a, G, g, epsilon, alpha, num_steps, norm='inf'):
    delta = torch.zeros_like(a, requires_grad=True)
    
    for step in range(num_steps):
        with autocast("cuda"):
            # 生成扰动输入
            a_perturbed = a + delta
            
            # 前向传播 (完整计算图)
            # 阶段 1: PyTorch 模型 G
            G_output = G(a_perturbed)
            
            # 阶段 2: JAX PDE 求解器 (通过包装器)
            g_output = JaxPDEWrapper.apply(a_perturbed, g) 
            
            # 计算损失
            loss = torch.norm(G_output - g_output, p=2)**2
            
        # 反向传播自动计算完整梯度
        loss.backward()
        
        # 更新扰动 (PGD 步骤)
        with torch.no_grad():
            if norm == 'inf':
                delta.data = delta.data + alpha * delta.grad.sign()
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            elif norm == '2':
                delta.data = delta.data + alpha * delta.grad / torch.norm(delta.grad, p=2)
                l2_norm = torch.norm(delta.data, p=2)
                radius = (len(delta))**(1/2) * epsilon
                if l2_norm > radius:
                    delta.data = delta.data * (radius / l2_norm)
        # 关键清理步骤
        del G_output, g_output, loss
        torch.cuda.empty_cache()
        gc.collect()

        # 清零梯度
        delta.grad.zero_()
    
    return a + delta, delta
"""



"""
def pgd_attack(a, G, g, epsilon, alpha, num_steps, norm='inf'):
    delta = torch.zeros_like(a, requires_grad=True)
    
    for step in range(num_steps):
        # 生成扰动输入
        a_perturbed = a + delta
        
        # 前向传播 (完整计算图)
        # 阶段 1: PyTorch 模型 G
        G_output = G(a_perturbed)
        
        # 阶段 2: JAX PDE 求解器 (通过包装器)
        g_output = JaxPDEWrapper.apply(a_perturbed, g) 
        
        # 计算损失
        loss = torch.norm(G_output - g_output, p=2)**2
            
        # 反向传播自动计算完整梯度
        loss.backward()
        
        # 更新扰动 (PGD 步骤)
        with torch.no_grad():
            if norm == 'inf':
                delta.data = delta.data + alpha * torch.sign(delta.grad)
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            elif norm == '2':
                delta.data = delta.data + 10000 * alpha * delta.grad / torch.norm(delta.grad, p=2)
                l2_norm = torch.norm(delta.data, p=2)
                radius = 10 * (len(delta))**(1/2) * epsilon
                if l2_norm > radius:
                    delta.data = delta.data * (radius / l2_norm)

        delta.grad.zero_()
    
    return a + delta, delta
"""


def pgd_attack(a, G, g, epsilon, alpha, num_steps, norm='inf'):
    """
    Parameters:
        norm: 'inf' for L∞ norm, 2 for L2 norm
    """
    delta = torch.zeros_like(a, requires_grad=True)
    
    for step in range(num_steps):
        a_perturbed = a + delta
        G_output = G(a_perturbed)
        g_output = JaxPDEWrapper.apply(a_perturbed, g) 
        loss = torch.norm(G_output - g_output, p=2)**2
        loss.backward()
        if norm == 'inf':
            # L∞ projection: clip each dimension independently
            delta.data = delta.data + alpha * torch.sign(delta.grad)
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        elif norm == 2:
            # L2 projection: scale if norm exceeds epsilon
            delta.data = delta.data + 1000 * alpha * delta.grad / torch.norm(delta.grad, p=2)
            l2_norm = torch.norm(delta.data, p=2)
            radius = 10 * (len(delta))**(1/2) * epsilon
            if l2_norm > radius:
                delta.data = delta.data * (radius / l2_norm)
        delta.grad.zero_()
    return a + delta, delta


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grf_params = {
        'dim': 1,
        'nx': 1024,
        'kernel': 'gaussian',
        'kernel_params': {'correlation_length': 0.03},
        'bc': 'periodic',
        'seed': 3,
        'zero_mean':False
    }

    burgers_params = {
        'nu': 0.005,
        'simulation_time': 1.0,
        "step": 0.001
    }

    dataset_params = {
        "num_record": 1
    }

    params = {
        "GRF": grf_params,
        "dataset": dataset_params,
        "burgers": burgers_params,
    }

    shape = (grf_params["nx"],)

    result_list = []
    s = 256
    sub = grf_params["nx"] // s

    model_path = f"/home/yifeisun/adversarial_robustness_FNO/saved_models/1D/modes16_width64_epochs500/burgers_1d_FNO_model_trainedby_dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu{burgers_params['nu']}_t1.0_seed45.pth"
    solver_name = re.findall(r'solver=([a-zA-Z0-9]+)', model_path)[0]
    # print(solver_name)
    model = FNO1d(modes=16, width=64) 
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    # norm = "inf"
    # norm = 2
    for norm in [2,"inf"]:
        for i in tqdm(range(dataset_params['num_record']), desc="adversarial input samples"):
            # try:
                a = GRFGenerator.generate_grf(
                                shape=shape,
                                kernel=grf_params['kernel'],
                                kernel_params=grf_params['kernel_params'],
                                bc=grf_params['bc'],
                                seed=grf_params['seed'] + i,  # Ensure different GRFs for each sample
                                zero_mean=grf_params['zero_mean']
                            )
                
                a = a[::sub]
                # num_steps_list = [2,5,10,20,50,100,500,1000,2000]
                num_steps_list = [2,5,10,20]
                epsilon_list = [0.01,0.05,0.1,0.5]
                # epsilon_list = [0.001,0.003,0.005,0.007,0.01,0.03,0.05,0.07]
                # epsilon_list = [0.001,0.003,0.005,0.007,0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7]
                for num_steps in num_steps_list:
                    # epsilon = 0.1
                    for epsilon in epsilon_list:
                        alpha = epsilon/num_steps
                        
                        a_torch = torch.from_numpy(a).float().unsqueeze(0).unsqueeze(-1).to(device)

                        if solver_name == "exponax":
                            solver = ExponaxBurgersSolver1D(s, nu=burgers_params['nu'], bc=grf_params['bc'])
                        elif solver_name == "scipy":
                            solver = SciPyBurgersSolver1D(s, nu=burgers_params['nu'], bc=grf_params['bc'])
                        elif solver_name == "scipy_spectral":
                            solver = SciPySpectralBurgersSolver1D(s, nu=burgers_params['nu'], bc=grf_params['bc'])
                        elif solver_name == "phiflow":
                            solver = PhiFlowBurgersSolver1D(s, nu=burgers_params['nu'], bc=grf_params['bc'])
                        else:
                            raise ValueError("Specified solver not inplemented")

                        t_span = (0, burgers_params['simulation_time'])
                        PDE_func = lambda u0: solver.solve(u0, t_final=burgers_params['simulation_time'], t_eval=t_span, step=burgers_params["step"])[1][1]
                        a_delta_sum, perturbed_input = pgd_attack(a_torch, model, PDE_func, epsilon, alpha, num_steps, norm=norm)
                        a_delta_sum = a_delta_sum.cpu().detach().numpy()
                        output_of_input = np.squeeze(model(a_torch.clone()).cpu().detach().numpy())
                        output_of_perturbed_input = np.squeeze(model(a_torch.clone()+perturbed_input).cpu().detach().numpy())

                        # print(a_delta_sum.copy().shape)
                        # print(a.copy().shape) 
                        solution_a_delta_sum = np.squeeze(solver.solve(np.squeeze(a_delta_sum.copy()), burgers_params['simulation_time'], t_span, burgers_params["step"])[t_span[1]][1])
                        # print(solution_a_delta_sum.shape)
                        solution_a = np.squeeze(solver.solve(a.copy(), burgers_params['simulation_time'], t_span, burgers_params["step"])[t_span[1]][1])
                        # print(solution_a.shape)
                        
                        result_dict = {
                        "index":i,
                        "epsilon":epsilon,
                        "num_steps":num_steps,
                        "alpha":alpha,
                        "a":a,
                        "delta":np.squeeze(perturbed_input.cpu().detach().numpy()),
                        "a+delta":np.squeeze(a_delta_sum),
                        "g(a)":solution_a,
                        "G(a)":output_of_input,
                        "g(a+delta)":solution_a_delta_sum,
                        "G(a+delta)":output_of_perturbed_input,
                        "original deviation RMSE": (np.mean((output_of_input-solution_a)**2))**(1/2),
                        "perturbed deviation RMSE": (np.mean((output_of_perturbed_input-solution_a_delta_sum)**2))**(1/2)}

                        result_list.append(result_dict)
                        print("index: ", i, "num_steps: ", num_steps, "epsilon: ", epsilon)
            # except:
            #     pass

        param_names = [
                "pgdwithgrad",
                f"L{norm}",
                f"dim={grf_params['dim']}d",
                f"solver={solver_name}",
                f"kernel={grf_params['kernel']}",
                *[f"{k}={v}" for k, v in grf_params['kernel_params'].items()],
                f"bc={grf_params['bc']}",
                f"nu={burgers_params['nu']}",
                f"t={burgers_params['simulation_time']}",
                f"seed={grf_params['seed']}",
                f"N={dataset_params['num_record']}"
            ]
        filename = "_".join(param_names) + ".pkl"
        current_file_path = Path(__file__).resolve().parent.parent
        folder = '1D' if grf_params['dim'] == 1 else '2D'
        save_path = current_file_path / "perturbation_results" / folder / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(save_path)
        with open(save_path, "wb") as f:
            pickle.dump(result_list, f)

