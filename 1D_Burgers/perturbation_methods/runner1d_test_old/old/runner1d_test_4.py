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
from models.FNO1d import FNO1d
from GRFs.generateGRFs import GRFGenerator
from solvers.burgers1d_solvers import *
import torch
from tqdm import tqdm
from pathlib import Path
import pickle
import numpy as np
import jax

from PGD_without_solver_gradient_1d import find_closest_ground_truth
from PGD_with_solver_gradient_1d_new_2 import JaxPDEWrapper

from torch.autograd import gradcheck, gradgradcheck

def compare_gradient_attack(a, G, g, x_dict, y_dict, epsilon, alpha, num_steps, norm='inf', 
                          fd_delta=1e-8, enable_fd=False, use_double=True):
    """
    比较三种梯度计算方法的差异：
    1. 反向传播带solver vs 不带solver
    2. 反向传播带solver vs gradcheck带solver 
    3. 反向传播不带solver vs gradcheck不带solver 
    
    Parameters:
        use_double (bool): If True, uses double precision (float64) for all computations.
                           If False, uses single precision (float32).
    """
    # Note: In PyTorch, double and float64 are indeed the same thing
    # torch.double == torch.float64
    # torch.float == torch.float32
    
    # Convert all inputs to double precision if requested
    original_dtypes = {name: param.dtype for name, param in G.named_parameters()}
    
    if use_double:
        dtype = torch.float64
        # 使用模型的to_dtype方法转换整个模型
        G = G.to_dtype(torch.float64)
    else:
        dtype = torch.float32
        G = G.to_dtype(torch.float32)

    # 转换输入数据
    a = a.to(dtype)
    g = g.to(dtype) if isinstance(g, torch.Tensor) else g
    
    # 转换字典数据
    def convert_data(data):
        if isinstance(data, torch.Tensor):
            return data.to(dtype)
        elif isinstance(data, dict):
            return {k: convert_data(v) for k, v in data.items()}
        return data
        
    x_dict = convert_data(x_dict)
    y_dict = convert_data(y_dict)
    
    grad_records = []
    delta_with = torch.zeros_like(a, requires_grad=True, dtype=dtype)
    delta_without = torch.zeros_like(a, requires_grad=True, dtype=dtype)

    def wrap_with_solver(input):
        return torch.norm(G(input) - JaxPDEWrapper.apply(input, g), p=2)**2

    def wrap_without_solver(input):
        return torch.norm(G(input) - find_closest_ground_truth(input, x_dict, y_dict), p=2)**2

    total_start_time = time.time()
    
    for step in range(num_steps):
        step_start_time = time.time()
        step_times = {}
        
        # ========== Backpropagation calculations ==========
        # With solver gradient
        torch.cuda.synchronize()
        t0 = time.time()
        
        a_with = a + delta_with
        t1 = time.time()
        
        loss_with = wrap_with_solver(a_with)
        t2 = time.time()
        
        loss_with.backward()
        torch.cuda.synchronize()
        t3 = time.time()
        
        grad_with = delta_with.grad.clone().detach()
        delta_with.grad.zero_()
        t4 = time.time()
        
        step_times.update({
            'with_forward': t2-t1,
            'with_backward': t3-t2,
            'with_grad_extract': t4-t3
        })
        
        # Without solver gradient
        torch.cuda.synchronize()
        t0 = time.time()
        
        a_without = a + delta_without
        t1 = time.time()
        
        loss_without = wrap_without_solver(a_without)
        t2 = time.time()
        
        loss_without.backward()
        torch.cuda.synchronize()
        t3 = time.time()
        
        grad_without = delta_without.grad.clone().detach()
        delta_without.grad.zero_()
        t4 = time.time()
        
        step_times.update({
            'without_forward': t2-t1,
            'without_backward': t3-t2,
            'without_grad_extract': t4-t3
        })
        
        # ========== PyTorch gradcheck ==========
        gc_start = time.time()
        # With solver gradcheck
        a_with_gc = a_with.detach().clone().requires_grad_(True)
        gradcheck_with = gradcheck(wrap_with_solver, (a_with_gc,), eps=fd_delta, atol=1e-2, rtol=1e-2)
        gc_with_time = time.time() - gc_start
        
        gc_start = time.time()
        # Without solver gradcheck
        a_without_gc = a_without.detach().clone().requires_grad_(True)
        gradcheck_without = gradcheck(wrap_without_solver, (a_without_gc,), eps=fd_delta, atol=1e-2, rtol=1e-2)
        gc_without_time = time.time() - gc_start
        
        step_times.update({
            'gc_with_total': gc_with_time,
            'gc_without_total': gc_without_time,
            'gradcheck_with_passed': gradcheck_with,
            'gradcheck_without_passed': gradcheck_without
        })

        print(step_times)
        
        # ========== Update perturbations ==========
        def update(delta, grad):
            if norm == 'inf':
                delta.data.add_(alpha * torch.sign(grad))
                delta.data.clamp_(-epsilon, epsilon)
            elif norm == 2:
                delta.data.add_(1000 * alpha * grad / (torch.norm(grad, p=2) + 1e-15))
                l2_norm = torch.norm(delta.data, p=2)
                radius = 10 * (len(delta))**(1/2) * epsilon
                if l2_norm > radius:
                    delta.data = delta.data * (radius / l2_norm)
        
        update(delta_with, grad_with)
        update(delta_without, grad_without)
    
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    return grad_records

def precompile_components(model, g, example_input):
    """
    预编译所有关键组件（无返回值版）
    参数:
        model: PyTorch模型
        g: JAX求解器函数
        example_input: 真实输入样本（用于确定shape/dtype）
    """
    # 1. 生成虚拟输入（基于真实输入的meta信息）
    dummy_input = torch.rand_like(example_input)
    
    # 2. 预编译JAX求解器（forward + backward）
    print("Compiling JAX solver...")
    with torch.no_grad():
        _ = JaxPDEWrapper.apply(dummy_input, g)  # Forward编译
        grad_output = torch.rand_like(dummy_input)
        _ = JaxPDEWrapper.backward(None, grad_output)  # Backward编译
    
    # 3. 预编译PyTorch模型（3次前向确保CUDA内核稳定）
    print("Compiling PyTorch model...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    
    # 4. 清空缓存（可选）
    torch.cuda.empty_cache()


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])

def main(argv):
    # print("=== Entered main() ===", flush=True)
    config = FLAGS.config
    workdir = FLAGS.workdir
    # print("=== Flags parsed ===")

    model_path = config.model_path
    workdir = config.workdir
    attack_method = config.attack_method
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M") 
    device = config.device
    nu = float(re.findall(r'nu([a-zA-Z0-9|.]+)', model_path)[0])
    num_steps_list = config.num_steps_list
    epsilon_list = config.epsilon_list
    num_samples = config.num_samples

    def convert_to_cpu_serializable(data):
        if isinstance(data, dict):
            return {k: convert_to_cpu_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_to_cpu_serializable(v) for v in data]
        elif isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        else:
            return data
    

    print("===============Experiment===============")
    
    print("\n\n\n")
    print("model_path: ",model_path)
    print("workdir: ",workdir)
    print("device: ",device)
    print("nu: ",nu)
    print("jax.devices()", jax.devices())
    print("\n\n\n")

    print("===============Running===============")
    print("\n\n\n")

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
        'nu': nu,
        'simulation_time': 1.0,
        "step": 0.001
    }

    dataset_params = {
        "num_record": 10
    }

    params = {
        "GRF": grf_params,
        "dataset": dataset_params,
        "burgers": burgers_params,
    }

    shape = (grf_params["nx"],)

    s = 1024
    sub = grf_params["nx"] // s

    solver_name = re.findall(r'solver=([a-zA-Z0-9]+)', model_path)[0]

    model = FNO1d(modes=16, width=64) 
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    current_file_path = Path(__file__).resolve().parent.parent
    # gradient_folder = current_file_path / "gradient_comparison_results" / ('1D' if grf_params['dim'] == 1 else '2D')
    gradient_folder = current_file_path / "gradient_comparison_results"
    gradient_folder.mkdir(parents=True, exist_ok=True)

    input_path = config.dict_input_path
    output_path = config.dict_output_path
    print("input_path: ",input_path)
    print("output_path: ",output_path)
    x_dict = torch.load(input_path)[:,::sub].to(device)
    y_dict = torch.load(output_path)[:,::sub].to(device)

    norm_list = [2, "inf"]
    
    gradient_records = {}
    for norm in norm_list:
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
                for num_steps in num_steps_list:
                    # epsilon = 0.1
                    for epsilon in epsilon_list:
                        start_time = time.time()

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

                        PDE_func = jax.jit(lambda u0: solver.solve(u0, t_final=burgers_params['simulation_time'], t_eval=t_span, step=burgers_params["step"])[1][1])
                        records = compare_gradient_attack(
                            a_torch, model, PDE_func, x_dict, y_dict,
                            epsilon=epsilon,
                            alpha=epsilon/num_steps,
                            num_steps=num_steps,
                            norm=norm,
                            enable_fd=True,
                            fd_delta=1e-8
                        )
                        gradient_records[(f"norm_{norm}",f"index_{i}",f"numsteps_{num_steps}",f"epsilon_{epsilon}")] = records
                        print(" \n     index:", i, ",    num_steps:", num_steps, ",    epsilon:", epsilon, ",    run time:", end_time-start_time)
            # except:
            #     pass

        gradient_filename = f"gradient_comparison_{solver_name}_nu{nu}_eps{epsilon}_steps{num_steps}_norm{norm}.pkl"
        with open(gradient_folder / gradient_filename, "wb") as f:
            pickle.dump(gradient_records, f)



if __name__ == "__main__":
    app.run(main)