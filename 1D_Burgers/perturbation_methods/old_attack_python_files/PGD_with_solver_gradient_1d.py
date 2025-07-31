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

torch.cuda.empty_cache()

def pgd_attack(a, G, g, epsilon, alpha, num_steps, norm='inf'):
    delta = torch.zeros_like(a, requires_grad=True)
    gradients = []
    
    # JAX转换辅助函数
    def t2j(torch_tensor):
        """PyTorch Tensor -> JAX数组"""
        return jnp.array(torch_tensor.detach().cpu().numpy())
    
    def j2t(jax_array):
        """JAX数组 -> PyTorch Tensor"""
        return torch.from_numpy(np.array(jax_array)).to(a.device).to(a.dtype)
    
    for step in range(num_steps):
        # 计算扰动输入
        a_perturbed = a + delta
        
        # 计算G的输出 (PyTorch)
        G_output = G(a_perturbed).squeeze()
        
        # 计算g的输出 (JAX)
        a_perturbed_jax = t2j(a_perturbed)
        g_output = g(a_perturbed_jax)
        g_output_torch = j2t(g_output)
        
        # 确保输出维度匹配
        assert G_output.shape == g_output_torch.shape, \
            f"模型输出形状不匹配: G {G_output.shape} vs g {g_output_torch.shape}"
        
        # 计算squared loss
        residual = G_output - g_output_torch
        loss = torch.norm(residual, p=2)**2
        
        # 计算梯度 (PyTorch部分)
        loss.backward(retain_graph=True)
        grad_delta_pytorch = delta.grad.clone()
        
        # 计算JAX模型的Jacobian贡献
        def g_jax_wrapper(x_jax):
            """包装JAX模型，确保输出形状正确"""
            return g(x_jax).reshape(G_output.shape)
        
        # 计算JAX模型的Jacobian
        jac_g_fn = jax.jacrev(g_jax_wrapper)
        jac_g = jac_g_fn(a_perturbed_jax)
        
        # 将JAX Jacobian转换为PyTorch Tensor
        jac_g_torch = j2t(jac_g)
        
        # 计算完整的梯度 (按照理论公式)
        residual_flat = residual.reshape(-1)
        jac_g_flat = jac_g_torch.reshape(len(residual_flat), -1)
        
        # 计算PyTorch模型的Jacobian
        def G_pytorch_wrapper(x):
            return G(x).reshape(G_output.shape)
        
        jac_G = jacrev(G_pytorch_wrapper)(a_perturbed)
        jac_G_flat = jac_G.reshape(len(residual_flat), -1)
        
        # 完整梯度计算
        complete_grad = 2 * (jac_G_flat - jac_g_flat).T @ residual_flat
        complete_grad = complete_grad.reshape(a.shape)
        
        # 记录梯度
        gradients.append(complete_grad.detach().cpu().numpy())
        
        # 更新delta
        with torch.no_grad():
            if norm == 'inf':
                delta.data = delta.data + alpha * complete_grad.sign()
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            elif norm == '2':
                delta.data = delta.data + alpha * complete_grad / torch.norm(complete_grad, p=2)
                l2_norm = torch.norm(delta.data, p=2)
                radius = (len(delta))**(1/2) * epsilon
                if l2_norm > radius:
                    delta.data = delta.data * (radius / l2_norm)
        
        # 清零梯度
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
        'nu': 0.0005,
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
    s = 1024
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
                # epsilon_list = [0.001,0.003,0.005,0.007,0.01,0.03,0.05,0.07]
                # epsilon_list = [0.001,0.003,0.005,0.007,0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7]
                epsilon_list = [0.01,0.05,0.1,0.5]
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

