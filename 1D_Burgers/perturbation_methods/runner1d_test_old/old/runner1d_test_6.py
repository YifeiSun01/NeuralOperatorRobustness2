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

import torch.nn.functional as F  # 添加这行导入
from scipy.spatial.distance import cosine

def compare_gradient_attack(a, G, g, x_dict, y_dict, epsilon, alpha, num_steps, norm='inf', 
                          fd_delta=1e-8, enable_fd=False, use_double=True):
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

    # 定义梯度比较指标计算函数
    def compute_gradient_metrics(grad1, grad2, name1="Grad1", name2="Grad2"):
        grad1_flat = grad1.flatten()
        grad2_flat = grad2.flatten()
        
        diff = grad1_flat - grad2_flat
        abs_diff = torch.abs(diff)
        rel_diff = abs_diff / (torch.abs(grad2_flat) + 1e-10)
        
        metrics = {
            'rmse': torch.sqrt(torch.mean(diff**2)).item(),
            'cosine_similarity': F.cosine_similarity(grad1_flat.unsqueeze(0), grad2_flat.unsqueeze(0)).item(),
            'max_abs_diff': torch.max(abs_diff).item(),
            'mean_abs_diff': torch.mean(abs_diff).item(),
            'max_rel_diff': torch.max(rel_diff).item(),
            'median_rel_diff': torch.median(rel_diff).item(),
            'grad1_norm': torch.norm(grad1_flat).item(),
            'grad2_norm': torch.norm(grad2_flat).item(),
            'norm_ratio': (torch.norm(grad1_flat)/torch.norm(grad2_flat)).item(),
            'grad1_max': torch.max(grad1_flat).item(),
            'grad1_min': torch.min(grad1_flat).item(),
            'grad2_max': torch.max(grad2_flat).item(),
            'grad2_min': torch.min(grad2_flat).item()
        }
        
        # 打印详细的梯度对比
        print(f"\n{'='*50}")
        print(f"Detailed Gradient Comparison: {name1} vs {name2}")
        print(f"{'Metric':<25} | {name1:<15} | {name2:<15} | Difference")
        print(f"{'-'*25}+{'-'*17}+{'-'*17}+{'-'*15}")
        print(f"{'Norm (L2)':<25} | {metrics['grad1_norm']:<15.6e} | {metrics['grad2_norm']:<15.6e} | ratio: {metrics['norm_ratio']:.4f}")
        print(f"{'Max value':<25} | {metrics['grad1_max']:<15.6e} | {metrics['grad2_max']:<15.6e} | diff: {metrics['grad1_max']-metrics['grad2_max']:.4e}")
        print(f"{'Min value':<25} | {metrics['grad1_min']:<15.6e} | {metrics['grad2_min']:<15.6e} | diff: {metrics['grad1_min']-metrics['grad2_min']:.4e}")
        print(f"{'RMSE':<25} | {'-':<15} | {'-':<15} | {metrics['rmse']:.6e}")
        print(f"{'Max abs diff':<25} | {'-':<15} | {'-':<15} | {metrics['max_abs_diff']:.6e}")
        print(f"{'Mean abs diff':<25} | {'-':<15} | {'-':<15} | {metrics['mean_abs_diff']:.6e}")
        print(f"{'Max rel diff':<25} | {'-':<15} | {'-':<15} | {metrics['max_rel_diff']:.6e}")
        print(f"{'Median rel diff':<25} | {'-':<15} | {'-':<15} | {metrics['median_rel_diff']:.6e}")
        print(f"{'Cosine similarity':<25} | {'-':<15} | {'-':<15} | {metrics['cosine_similarity']:.6f}")
        print(f"{'='*50}")
        
        return metrics

    # 增强版的gradcheck打印函数
    def print_gradcheck_results(gc_result, name):
        print(f"\n{'#'*50}")
        print(f"## Detailed GradCheck Results for {name}")
        print(f"{'#'*50}")
        
        print(f"\nOverall GradCheck Passed: {gc_result['passed']}")
        
        for i, metric in enumerate(gc_result['metrics']):
            print(f"\nParameter {i} Comparison (Analytical vs Numerical):")
            print(f"  - RMSE: {metric['rmse']:.6e}")
            print(f"  - Cosine Similarity: {metric['cosine_similarity']:.6f}")
            print(f"  - Max Absolute Difference: {metric['max_abs_diff']:.6e}")
            print(f"  - Mean Absolute Difference: {metric['mean_abs_diff']:.6e}")
            print(f"  - Max Relative Difference: {metric['max_rel_diff']:.6e}")
            print(f"  - Norm Ratio (Analytical/Numerical): {metric['norm_ratio']:.6f}")
            
            # 打印前5个梯度值的对比
            a_grad = gc_result['analytical_grads'][i].flatten()[:5]
            n_grad = gc_result['numerical_grads'][i].flatten()[:5]
            print("\n  First 5 elements comparison:")
            print(f"  {'Index':<8} | {'Analytical':<15} | {'Numerical':<15} | {'Diff':<15} | {'Rel.Diff':<15}")
            for idx, (a, n) in enumerate(zip(a_grad, n_grad)):
                diff = a - n
                rel_diff = abs(diff)/(abs(n)+1e-10)
                print(f"  {idx:<8} | {a:<15.6e} | {n:<15.6e} | {diff:<15.6e} | {rel_diff:<15.6e}")

    # 定义详细gradcheck函数
    def detailed_gradcheck(func, inputs, eps=1e-6):
        """
        使用reverse-mode自动微分的梯度检查
        """
        inputs = [i.detach().clone().requires_grad_(True) for i in inputs]
        output = func(*inputs)
        grads = torch.autograd.grad(output, inputs, create_graph=True)
        
        numerical_grads = []
        for i, inp in enumerate(inputs):
            def f(inp_copy):
                tmp_inputs = [x if j != i else inp_copy for j, x in enumerate(inputs)]
                return func(*tmp_inputs)
            
            # 强制使用reverse-mode
            numerical_grad = torch.autograd.functional.jacobian(
                f, 
                inp, 
                create_graph=False,
                strategy='reverse-mode'  # 关键修改
            )
            numerical_grads.append(numerical_grad)
        
        # 计算指标
        metrics = []
        for a_grad, n_grad in zip(grads, numerical_grads):
            metrics.append(compute_gradient_metrics(a_grad, n_grad))
        
        # 检查是否通过标准gradcheck
        passed = torch.autograd.gradcheck(
            func, 
            inputs, 
            eps=eps,
            atol=1e-5, 
            rtol=1e-3, 
            raise_exception=False
        )
        
        return {
            'passed': passed,
            'metrics': metrics,
            'analytical_grads': grads,
            'numerical_grads': numerical_grads
        }

    # 初始化记录
    grad_records = []
    delta_with = torch.zeros_like(a, requires_grad=True, dtype=dtype)
    delta_without = torch.zeros_like(a, requires_grad=True, dtype=dtype)

    # 定义损失函数
    def wrap_with_solver(input):
        return torch.norm(G(input) - JaxPDEWrapper.apply(input, g), p=2)**2

    def wrap_without_solver(input):
        return torch.norm(G(input) - find_closest_ground_truth(input, x_dict, y_dict), p=2)**2

    # 主循环
    for step in range(num_steps):
        step_metrics = {}
        
        # 1. 带solver的反向传播
        a_with = a + delta_with
        loss_with = wrap_with_solver(a_with)
        loss_with.backward()
        grad_with = delta_with.grad.clone().detach()
        delta_with.grad.zero_()
        
        # 2. 不带solver的反向传播
        a_without = a + delta_without
        loss_without = wrap_without_solver(a_without)
        loss_without.backward()
        grad_without = delta_without.grad.clone().detach()
        delta_without.grad.zero_()
        
        # 3. 计算两种梯度差异
        grad_diff_metrics = compute_gradient_metrics(grad_with, grad_without)
        step_metrics['grad_diff'] = grad_diff_metrics
        
        # 4. 详细gradcheck
        # 带solver的检查
        gc_with = detailed_gradcheck(wrap_with_solver, [a_with.detach().clone()], eps=fd_delta)
        # 不带solver的检查
        gc_without = detailed_gradcheck(wrap_without_solver, [a_without.detach().clone()], eps=fd_delta)
        
        step_metrics.update({
            'gradcheck_with': gc_with,
            'gradcheck_without': gc_without,
            'gradient_norms': {
                'with_solver': torch.norm(grad_with).item(),
                'without_solver': torch.norm(grad_without).item()
            }
        })
        
        # 记录步骤指标
        grad_records.append(step_metrics)
        
        # 更新扰动
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
        
        print(f"\n{'='*80}")
        print(f"Step {step} Gradient Comparison: With Solver vs Without Solver")
        print(f"{'='*80}")
        grad_diff_metrics = compute_gradient_metrics(
            grad_with, grad_without, 
            "With Solver", "Without Solver"
        )
        
        # 打印gradcheck的详细结果
        print_gradcheck_results(gc_with, "With Solver")
        print_gradcheck_results(gc_without, "Without Solver")
        
        # 打印梯度范数信息
        print(f"\nGradient Norms:")
        print(f"  - With solver: {torch.norm(grad_with).item():.6e}")
        print(f"  - Without solver: {torch.norm(grad_without).item():.6e}")
        print(f"  - Norm ratio (with/without): {torch.norm(grad_with).item()/torch.norm(grad_without).item():.6f}")

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
                        print(" \n     index:", i, ",    num_steps:", num_steps, ",    epsilon:", epsilon)
            # except:
            #     pass

        gradient_filename = f"gradient_comparison_{solver_name}_nu{nu}_eps{epsilon}_steps{num_steps}_norm{norm}.pkl"
        with open(gradient_folder / gradient_filename, "wb") as f:
            pickle.dump(gradient_records, f)



if __name__ == "__main__":
    app.run(main)