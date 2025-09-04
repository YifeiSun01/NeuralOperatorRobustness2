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

def compare_gradient_attack(a, G, g, x_dict, y_dict, epsilon, alpha, num_steps, norm='inf', 
                          fd_delta=1e-3, enable_fd=False):
    """
    比较三种梯度计算方法的差异：
    1. 反向传播带solver vs 不带solver
    2. 反向传播带solver vs 有限差分带solver (仅当enable_fd=True)
    3. 反向传播不带solver vs 有限差分不带solver (仅当enable_fd=True)
    
    参数:
        fd_delta: 有限差分步长(推荐1e-5到1e-7)
        enable_fd: 是否启用有限差分计算
    """
    
    grad_records = []
    delta_with = torch.zeros_like(a, requires_grad=True)
    delta_without = torch.zeros_like(a, requires_grad=True)

    def compute_grad_similarity(grad1, grad2):
        """计算两个梯度之间的相似性指标"""
        start_time = time.time()
        # 确保张量在CPU上并转换为numpy数组
        grad1_np = grad1.cpu().flatten().numpy() if torch.is_tensor(grad1) else grad1.flatten()
        grad2_np = grad2.cpu().flatten().numpy() if torch.is_tensor(grad2) else grad2.flatten()
        
        rmse = np.sqrt(np.mean((grad1_np - grad2_np)**2))
        cos_sim = np.dot(grad1_np, grad2_np) / (np.linalg.norm(grad1_np)*np.linalg.norm(grad2_np)+1e-18)
        angle = np.arccos(np.clip(cos_sim, -1, 1)) * 180/np.pi
        elapsed = time.time() - start_time
        print("compute gradient similarity time: ",elapsed)
        return {'rmse': rmse, 'cos_sim': cos_sim, 'angle': angle, 'compute_time': elapsed}

    def finite_difference(a_perturbed, use_solver, batch_size=1):
        """分批次计算有限差分，显存友好版"""
        total_start = time.time()
        grad = torch.zeros_like(a_perturbed)
        h = fd_delta
        numel = a_perturbed.numel()
        
        batch_times = []
        compute_times = []
        
        for i in range(0, numel, batch_size):
            batch_start = time.time()
            
            indices = torch.arange(i, min(i+batch_size, numel))
            deltas = torch.zeros(len(indices), *a_perturbed.shape, device=a_perturbed.device)
            deltas.view(len(indices), -1)[:, indices] = h
            
            with torch.no_grad():
                a_plus = a_perturbed + deltas  # [batch_size, ...]
                a_minus = a_perturbed - deltas
            
            # 计算损失
            compute_start = time.time()
            if use_solver:
                out_plus = torch.stack([torch.norm(G(ap) - JaxPDEWrapper.apply(ap, g), p=2)**2 for ap in a_plus])
                out_minus = torch.stack([torch.norm(G(am) - JaxPDEWrapper.apply(am, g), p=2)**2 for am in a_minus])
            else:
                out_plus = torch.stack([torch.norm(G(ap) - find_closest_ground_truth(ap, x_dict, y_dict), p=2)**2 for ap in a_plus])
                out_minus = torch.stack([torch.norm(G(am) - find_closest_ground_truth(am, x_dict, y_dict), p=2)**2 for am in a_minus])
            
            # 计算梯度
            grad.view(-1)[indices] = (out_plus - out_minus) / (2*h)
            
            print(f"     get one entry of gradient time of index {i}: ",time.time() - batch_start)
            compute_times.append(time.time() - compute_start)
            batch_times.append(time.time() - batch_start)
            
            # 释放显存
            del deltas, a_plus, a_minus, out_plus, out_minus
            torch.cuda.empty_cache()
        
        total_time = time.time() - total_start
        return {
            'grad': grad,
            'total_time': total_time,
            'avg_batch_time': np.mean(batch_times),
            'avg_compute_time': np.mean(compute_times),
            'num_batches': len(batch_times)
        }

    total_start_time = time.time()
    
    for step in range(num_steps):
        step_start_time = time.time()
        step_times = {}
        
        # ========== 反向传播计算 ==========
        # 带solver梯度
        torch.cuda.synchronize()
        t0 = time.time()
        
        a_with = a + delta_with
        t1 = time.time()
        
        G_with = G(a_with)
        t2 = time.time()
        
        g_with = JaxPDEWrapper.apply(a_with, g)
        t3 = time.time()
        
        loss_with = torch.norm(G_with - g_with, p=2)**2
        t4 = time.time()
        
        loss_with.backward()
        torch.cuda.synchronize()
        t5 = time.time()
        
        grad_with = delta_with.grad.clone().detach()
        delta_with.grad.zero_()
        t6 = time.time()
        
        step_times.update({
            'with_perturb': t1-t0,
            'with_G': t2-t1,
            'with_g': t3-t2,
            'with_loss': t4-t3,
            'with_backward': t5-t4,
            'with_grad_extract': t6-t5
        })
        
        # 不带solver梯度
        torch.cuda.synchronize()
        t0 = time.time()
        
        a_without = a + delta_without
        t1 = time.time()
        
        y_closest = find_closest_ground_truth(a_without, x_dict, y_dict)
        t2 = time.time()
        
        G_without = G(a_without)
        t3 = time.time()
        
        loss_without = torch.norm(G_without - y_closest, p=2)**2
        t4 = time.time()
        
        loss_without.backward()
        torch.cuda.synchronize()
        t5 = time.time()
        
        grad_without = delta_without.grad.clone().detach()
        delta_without.grad.zero_()
        t6 = time.time()
        
        step_times.update({
            'without_perturb': t1-t0,
            'without_find_closest': t2-t1,
            'without_G': t3-t2,
            'without_loss': t4-t3,
            'without_backward': t5-t4,
            'without_grad_extract': t6-t5
        })
        
        # ========== 有限差分计算 ==========
        if enable_fd:
            fd_start = time.time()
            fd_with_result = finite_difference(a_with, use_solver=True)
            grad_with_fd = fd_with_result['grad']
            fd_with_time = time.time() - fd_start
            
            fd_start = time.time()
            fd_without_result = finite_difference(a_without, use_solver=False)
            grad_without_fd = fd_without_result['grad']
            fd_without_time = time.time() - fd_start
            
            step_times.update({
                'fd_with_total': fd_with_time,
                'fd_without_total': fd_without_time,
                'fd_with_details': fd_with_result,
                'fd_without_details': fd_without_result
            })
        else:
            grad_with_fd, grad_without_fd = None, None
        
        # ========== 记录比较结果 ==========
        record = {
            'step': step,
            'step_time': time.time() - step_start_time,
            'timings': step_times
        }
        
        # 1. 反向传播带solver vs 不带solver (总是计算)
        bp_compare = compute_grad_similarity(grad_with, grad_without)
        record.update({f'bp_vs_nosolver_{k}': v for k,v in bp_compare.items()})
        
        if enable_fd:
            # 2. 反向传播带solver vs 有限差分带solver
            with_compare = compute_grad_similarity(grad_with, grad_with_fd)
            record.update({f'with_vs_fd_{k}': v for k,v in with_compare.items()})
            
            # 3. 反向传播不带solver vs 有限差分不带solver
            without_compare = compute_grad_similarity(grad_without, grad_without_fd)
            record.update({f'without_vs_fd_{k}': v for k,v in without_compare.items()})
            
            # 保存原始梯度值
            record.update({
                'grad_with': grad_with.cpu().numpy(),
                'grad_without': grad_without.cpu().numpy(),
                'grad_with_fd': grad_with_fd.cpu().numpy(),
                'grad_without_fd': grad_without_fd.cpu().numpy()
            })
        else:
            # 当enable_fd=False时，只保存反向传播的梯度
            record.update({
                'grad_with': grad_with.cpu().numpy(),
                'grad_without': grad_without.cpu().numpy()
            })
        
        grad_records.append(record)
        
        # ========== 打印信息 ==========
        log_msg = (
            f"Step {step+1}/{num_steps} (Total: {time.time()-total_start_time:.2f}s, Step: {time.time()-step_start_time:.2f}s):\n"
            "┌───────────────────────────────────────────────────────┐\n"
            f"│ 1. BP(w/solver) vs BP(w/o solver):                  │\n"
            f"│    RMSE = {bp_compare['rmse']:.10e}                  │\n"
            f"│    Cosine = {bp_compare['cos_sim']:.10f}             │\n"
            f"│    Angle = {bp_compare['angle']:.10f}°               │\n"
            "├───────────────────────────────────────────────────────┤\n"
            f"│ Timing (w/solver):                                  │\n"
            f"│   Perturb: {step_times['with_perturb']:.3f}s          │\n"
            f"│   G: {step_times['with_G']:.3f}s, g: {step_times['with_g']:.3f}s │\n"
            f"│   Loss: {step_times['with_loss']:.3f}s                │\n"
            f"│   Backward: {step_times['with_backward']:.3f}s         │\n"
            "├───────────────────────────────────────────────────────┤\n"
            f"│ Timing (w/o solver):                                │\n"
            f"│   Perturb: {step_times['without_perturb']:.3f}s        │\n"
            f"│   Find closest: {step_times['without_find_closest']:.3f}s │\n"
            f"│   G: {step_times['without_G']:.3f}s                   │\n"
            f"│   Loss: {step_times['without_loss']:.3f}s              │\n"
            f"│   Backward: {step_times['without_backward']:.3f}s       │\n"
        )

        if enable_fd:
            log_msg += (
                "├───────────────────────────────────────────────────────┤\n"
                f"│ 2. BP(w/solver) vs FD(w/solver):                │\n"
                f"│    RMSE = {with_compare['rmse']:.10e}            │\n"
                f"│    Cosine = {with_compare['cos_sim']:.10f}       │\n"
                f"│    Angle = {with_compare['angle']:.10f}°         │\n"
                "├───────────────────────────────────────────────────────┤\n"
                f"│ 3. BP(w/o solver) vs FD(w/o solver):            │\n"
                f"│    RMSE = {without_compare['rmse']:.10e}         │\n"
                f"│    Cosine = {without_compare['cos_sim']:.10f}    │\n"
                f"│    Angle = {without_compare['angle']:.10f}°      │\n"
                "├───────────────────────────────────────────────────────┤\n"
                f"│ FD Timing:                                      │\n"
                f"│   With solver: {step_times['fd_with_total']:.3f}s   │\n"
                f"│   Without solver: {step_times['fd_without_total']:.3f}s │\n"
                f"│   Avg batch time: {step_times['fd_with_details']['avg_batch_time']:.3f}s │\n"
                f"│   Avg compute time: {step_times['fd_with_details']['avg_compute_time']:.3f}s │\n"
            )

        log_msg += "└───────────────────────────────────────────────────────┘"
        print(log_msg)
        
        # ========== 更新扰动 ==========
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