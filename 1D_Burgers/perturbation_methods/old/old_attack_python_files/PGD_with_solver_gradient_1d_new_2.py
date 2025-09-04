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
import time

# 启用 JAX GPU 支持（需安装 jax[cuda]）
jax.config.update('jax_platform_name', 'cuda')

"""
class JaxPDEWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_torch, g):
        # 直接将 PyTorch CUDA 张量转为 JAX GPU 数组（无需 CPU 中转）
        a_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(a_torch))
        g_output_jax = g(a_jax)
        ctx.save_for_backward(a_torch)
        ctx.g = g
        # JAX GPU 数组 -> PyTorch CUDA 张量
        return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(g_output_jax))

    @staticmethod
    def backward(ctx, grad_output):
        a_torch, = ctx.saved_tensors
        g = ctx.g
        # PyTorch CUDA -> JAX GPU
        grad_out_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(grad_output))
        a_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(a_torch))
        # 计算 VJP
        _, vjp_fun = jax.vjp(g, a_jax)
        grad_input_jax, = vjp_fun(grad_out_jax)
        # JAX GPU -> PyTorch CUDA
        return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(grad_input_jax)), None
"""



"""
class JaxPDEWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_torch, g):
        # 检查输入设备一致性
        if not a_torch.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        
        try:
            # PyTorch -> JAX (GPU直连)
            a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
            a_jax = jax.dlpack.from_dlpack(a_dlpack)
            
            # 执行JAX计算（确保g已JIT编译）
            g_output_jax = g(a_jax)
            
            # JAX -> PyTorch (GPU直连)
            out_dlpack = jax.dlpack.to_dlpack(g_output_jax)
            g_output = torch.utils.dlpack.from_dlpack(out_dlpack)
            
            # 保存必要信息用于反向传播
            ctx.save_for_backward(a_torch)
            ctx.g = g
            
            return g_output
            
        except Exception as e:
            # 回退到CPU路径（带警告）
            import warnings
            warnings.warn(f"DLPack GPU transfer failed, falling back to CPU: {str(e)}")
            a_jax = jnp.array(a_torch.detach().cpu().numpy())
            g_output_jax = g(a_jax)
            return torch.as_tensor(np.asarray(g_output_jax), 
                   device=a_torch.device, 
                   dtype=a_torch.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a_torch, = ctx.saved_tensors
        g = ctx.g
        
        try:
            # PyTorch梯度 -> JAX (GPU直连)
            grad_dlpack = torch.utils.dlpack.to_dlpack(grad_output.contiguous())
            grad_jax = jax.dlpack.from_dlpack(grad_dlpack)
            
            # 输入数据 -> JAX (复用forward的转换)
            a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
            a_jax = jax.dlpack.from_dlpack(a_dlpack)
            
            # 计算VJP
            _, vjp_fn = jax.vjp(g, a_jax)
            grad_input_jax, = vjp_fn(grad_jax)
            
            # JAX梯度 -> PyTorch (GPU直连)
            grad_input_dlpack = jax.dlpack.to_dlpack(grad_input_jax)
            return torch.utils.dlpack.from_dlpack(grad_input_dlpack), None
            
        except Exception as e:
            warnings.warn(f"Backward DLPack failed: {str(e)}")
            # 回退到CPU路径
            grad_np = grad_output.detach().cpu().numpy()
            a_np = a_torch.detach().cpu().numpy()
            _, vjp_fn = jax.vjp(g, jnp.array(a_np))
            grad_input_jax, = vjp_fn(jnp.array(grad_np))
            return torch.as_tensor(np.asarray(grad_input_jax),
                   device=a_torch.device,
                   dtype=a_torch.dtype), None
"""

jax.config.update("jax_enable_x64", True)

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



def pgd_attack(a, G, g, epsilon, alpha, num_steps, norm='inf'):
    """
    Parameters:
        norm: 'inf' for L∞ norm, 2 for L2 norm
    """
    delta = torch.zeros_like(a, requires_grad=True)
    
    for step in range(num_steps):
        torch.cuda.synchronize()  # 确保CUDA操作同步
        # t1 = time.time()
        
        a_perturbed = a + delta
        # t2 = time.time()
        
        G_output = G(a_perturbed)
        # t3 = time.time()
        
        g_output = JaxPDEWrapper.apply(a_perturbed, g)
        # t4 = time.time()
        
        loss = torch.norm(G_output - g_output, p=2)**2
        # t5 = time.time()
        
        loss.backward()
        # t6 = time.time()
        
        # print(f"Step {step}: Perturb={t2-t1:.3f}s, G={t3-t2:.3f}s, g={t4-t3:.3f}s, Loss={t5-t4:.3f}s, Backward={t6-t5:.3f}s")
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

