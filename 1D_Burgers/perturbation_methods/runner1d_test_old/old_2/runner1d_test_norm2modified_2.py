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

import torch.nn.functional as F  
from scipy.spatial.distance import cosine

def compare_gradient_attack(a, G, g, x_dict, y_dict, epsilon, alpha, num_steps, norm='inf', 
                          fd_delta=1e-8, enable_fd=False, use_double=True):
    original_dtypes = {name: param.dtype for name, param in G.named_parameters()}

    if use_double:
        dtype = torch.float64
        G = G.to_dtype(torch.float64)
    else:
        dtype = torch.float32
        G = G.to_dtype(torch.float32)

    a = a.to(dtype)
    g = g.to(dtype) if isinstance(g, torch.Tensor) else g
    
    def convert_data(data):
        if isinstance(data, torch.Tensor):
            return data.to(dtype)
        elif isinstance(data, dict):
            return {k: convert_data(v) for k, v in data.items()}
        return data
        
    x_dict = convert_data(x_dict)
    y_dict = convert_data(y_dict)

    delta_with = torch.zeros_like(a, requires_grad=True, dtype=dtype)
    delta_without = torch.zeros_like(a, requires_grad=True, dtype=dtype)
    grad_records = {} 
    grad_records["step_update"] = {}

    def wrap_with_solver(input):
        # Initialize timing dict
        timing = {}
        
        # Time G computation
        start = time.time()
        G_output = G(input)
        timing['G_time'] = time.time() - start
        
        # Time JaxPDEWrapper (g) computation
        start = time.time()
        g_output = JaxPDEWrapper.apply(input, g)
        timing['g_time'] = time.time() - start
        
        # Time norm computation
        start = time.time()
        loss = torch.norm(G_output - g_output, p=2)**2
        timing['norm_time'] = time.time() - start
        
        return loss, timing

    def wrap_without_solver(input):
        # Initialize timing dict
        timing = {}
        
        # Time G computation
        start = time.time()
        G_output = G(input)
        timing['G_time'] = time.time() - start
        
        # Time JaxPDEWrapper (g) computation
        start = time.time()
        g_output = JaxPDEWrapper.apply(input, g)
        timing['find_time'] = time.time() - start
        
        # Time norm computation
        start = time.time()
        loss = torch.norm(G_output - g_output.detach(), p=2)**2
        timing['norm_time'] = time.time() - start
        
        return loss, timing

    def compute_metrics_less(vec1, vec2, name1="vec1", name2="vec2"):
        vec1_flat = vec1.clone().detach().flatten()
        vec2_flat = vec2.clone().detach().flatten()
        
        diff = vec1_flat - vec2_flat
        rmse = torch.sqrt(torch.mean(diff**2))
        
        metrics = {
            'vec1_length': len(vec1_flat),
            'vec2_length': len(vec1_flat),
            'vec1_norm': torch.norm(vec1_flat).item(),
            'vec2_norm': torch.norm(vec2_flat).item(),
            'norm_ratio': (torch.norm(vec1_flat)/torch.norm(vec2_flat)).item(),
            'rmse': rmse.item(),
            'cosine_similarity': F.cosine_similarity(vec1_flat.unsqueeze(0), vec2_flat.unsqueeze(0)).item(),
            'angle_degrees': torch.rad2deg(torch.acos(torch.clamp(
                torch.tensor(F.cosine_similarity(vec1_flat.unsqueeze(0), vec2_flat.unsqueeze(0))), 
                -1.0, 1.0))).item(),
        }
        
        return metrics

    def print_combined_comparison_table(grad_diff_metrics, step):
        print(f"\n{'=' * 120}")
        print(f"COMBINED COMPARISON ANALYSIS (Step {step})".center(120))
        print(f"{'=' * 120}")
        
        # Main header
        print(f"{'Comparison':<50} | {'L2 Norm Ratio':<15} | {'RMSE':<15} | "
            f"{'Cosine Similarity':<18} | {'Angle (Degrees)':<15}")
        print(f"{'-' * 30}+{'-' * 17}+{'-' * 17}+{'-' * 20}+{'-' * 15}")
        
        # Gradients comparison row
        angle_str = f"{grad_diff_metrics['angle_degrees']:.5f}°"
        print(f"{'1. Analytical Gradients with vs without solver gradient':<50} | "
            f"{grad_diff_metrics['norm_ratio']:<15.5f} | "
            f"{grad_diff_metrics['rmse']:<15.6e} | "
            f"{grad_diff_metrics['cosine_similarity']:<18.6f} | "
            f"{angle_str:<15}")
        
        print(f"{'=' * 120}\n")
    
    def compute_loss_metrics(a, delta, G, g, use_solver=True):
        if use_solver:
            return {
                'combined_loss': wrap_with_solver(a + delta)[0].item()
            }
        else:
            return {
                'combined_loss': wrap_without_solver(a + delta)[0].item()
            }

    def compute_all_loss_metrics(a, delta_with, delta_without, G, g):
        return {
            'with_solver': {
                'delta_with': compute_loss_metrics(a, delta_with, G, g, use_solver=True),
                'delta_without_solver': compute_loss_metrics(a, delta_without, G, g, use_solver=True)
            }
        }

    def print_complete_comparison(initial, final, final_step, timing_metrics=None):
        """Compare initial (step=0) and final step losses with percentage increase and timing metrics"""
        # Calculate absolute increases
        with_solver_increase = final['with_solver']['delta_with']['combined_loss'] - initial['with_solver']['delta_with']['combined_loss']
        without_solver_increase = final['with_solver']['delta_without_solver']['combined_loss'] - initial['with_solver']['delta_without_solver']['combined_loss']
        
        # Calculate percentage increases (with protection against division by zero)
        def calc_pct_increase(final, initial):
            return ((final - initial) / (initial + 1e-10)) * 100
        
        with_solver_pct = calc_pct_increase(final['with_solver']['delta_with']['combined_loss'], 
                                        initial['with_solver']['delta_with']['combined_loss'])
        without_solver_pct = calc_pct_increase(final['with_solver']['delta_without_solver']['combined_loss'], 
                                            initial['with_solver']['delta_without_solver']['combined_loss'])

        # Calculate average timing metrics if available
        avg_timing_metrics = None
        if timing_metrics and isinstance(timing_metrics, list) and len(timing_metrics) > 1:
            num_steps = len(timing_metrics)
            avg_timing_metrics = {
                'with_solver_forward_avg': sum(m["with_solver"]['with_solver_forward_time'] for m in timing_metrics[1:]) / (num_steps - 1),
                'with_solver_backward_avg': sum(m["with_solver"]['with_solver_backward_time'] for m in timing_metrics[1:]) / (num_steps - 1),
                'with_solver_G_avg': sum(m["with_solver"]['with_solver_G_time'] for m in timing_metrics[1:]) / (num_steps - 1),
                'with_solver_g_avg': sum(m["with_solver"]['with_solver_g_time'] for m in timing_metrics[1:]) / (num_steps - 1),
                'without_solver_forward_avg': sum(m["without_solver"]['without_solver_forward_time'] for m in timing_metrics[1:]) / (num_steps - 1),
                'without_solver_backward_avg': sum(m["without_solver"]['without_solver_backward_time'] for m in timing_metrics[1:]) / (num_steps - 1),
                'without_solver_G_avg': sum(m["without_solver"]['without_solver_G_time'] for m in timing_metrics[1:]) / (num_steps - 1),
                'without_solver_find_avg': sum(m["without_solver"]['without_solver_find_time'] for m in timing_metrics[1:]) / (num_steps - 1),
            }

        # Determine table width based on whether timing metrics are available
        if timing_metrics:
            table_width = 130
            header = (f"{'Case':<30} | {'Initial Loss':<15} | {'Final Loss':<15} | {'Abs Increase':<15} | {'% Increase':<12} | "
                    f"{'Fwd Time':<10} | {'Bwd Time':<10} | {'G Time':<10} | {'g/find Time':<15}")
            separator = f"{'-'*30}+{'-'*17}+{'-'*17}+{'-'*17}+{'-'*14}+{'-'*12}+{'-'*12}+{'-'*12}+{'-'*17}"
        else:
            table_width = 92
            header = f"{'Case':<30} | {'Initial Loss':<15} | {'Final Loss':<15} | {'Abs Increase':<15} | {'% Increase':<12}"
            separator = f"{'-'*30}+{'-'*17}+{'-'*17}+{'-'*17}+{'-'*14}"

        # Print header
        print(f"\n{f'COMPREHENSIVE ANALYSIS (Initial Step 0 vs Final Step {final_step})':^{table_width}}")
        print(f"{'-'*table_width}")
        print(header)
        print(separator)
        
        # Row for with solver (using g)
        if timing_metrics:
            if avg_timing_metrics:  # Use average metrics if available
                print(f"{'perturbation with solver':<30} | "
                    f"{initial['with_solver']['delta_with']['combined_loss']:<15.4e} | "
                    f"{final['with_solver']['delta_with']['combined_loss']:<15.4e} | "
                    f"{with_solver_increase:<15.4e} | "
                    f"{with_solver_pct:<12.2f}% | "
                    f"{avg_timing_metrics['with_solver_forward_avg']:<10.4f} | "
                    f"{avg_timing_metrics['with_solver_backward_avg']:<10.4f} | "
                    f"{avg_timing_metrics['with_solver_G_avg']:<10.4f} | "
                    f"{avg_timing_metrics['with_solver_g_avg']:<15.4f}")
            else:  # Single timing metric case
                print(f"{'perturbation with solver':<30} | "
                    f"{initial['with_solver']['delta_with']['combined_loss']:<15.4e} | "
                    f"{final['with_solver']['delta_with']['combined_loss']:<15.4e} | "
                    f"{with_solver_increase:<15.4e} | "
                    f"{with_solver_pct:<12.2f}% | "
                    f"{timing_metrics[0]['with_solver']['with_solver_forward_time']:<10.4f} | "
                    f"{timing_metrics[0]['with_solver']['with_solver_backward_time']:<10.4f} | "
                    f"{timing_metrics[0]['with_solver']['with_solver_G_time']:<10.4f} | "
                    f"{timing_metrics[0]['with_solver']['with_solver_g_time']:<15.4f}")
        else:
            print(f"{'perturbation with solver':<30} | "
                f"{initial['with_solver']['delta_with']['combined_loss']:<15.4e} | "
                f"{final['with_solver']['delta_with']['combined_loss']:<15.4e} | "
                f"{with_solver_increase:<15.4e} | "
                f"{with_solver_pct:<12.2f}%")
        
        # Row for without solver (using find)
        if timing_metrics:
            if avg_timing_metrics:  # Use average metrics if available
                print(f"{'perturbation without solver':<30} | "
                    f"{initial['with_solver']['delta_without_solver']['combined_loss']:<15.4e} | "
                    f"{final['with_solver']['delta_without_solver']['combined_loss']:<15.4e} | "
                    f"{without_solver_increase:<15.4e} | "
                    f"{without_solver_pct:<12.2f}% | "
                    f"{avg_timing_metrics['without_solver_forward_avg']:<10.4f} | "
                    f"{avg_timing_metrics['without_solver_backward_avg']:<10.4f} | "
                    f"{avg_timing_metrics['without_solver_G_avg']:<10.4f} | "
                    f"{avg_timing_metrics['without_solver_find_avg']:<15.4f}")
            else:  # Single timing metric case
                print(f"{'perturbation without solver':<30} | "
                    f"{initial['with_solver']['delta_without_solver']['combined_loss']:<15.4e} | "
                    f"{final['with_solver']['delta_without_solver']['combined_loss']:<15.4e} | "
                    f"{without_solver_increase:<15.4e} | "
                    f"{without_solver_pct:<12.2f}% | "
                    f"{timing_metrics[0]['without_solver']['without_solver_forward_time']:<10.4f} | "
                    f"{timing_metrics[0]['without_solver']['without_solver_backward_time']:<10.4f} | "
                    f"{timing_metrics[0]['without_solver']['without_solver_G_time']:<10.4f} | "
                    f"{timing_metrics[0]['without_solver']['without_solver_find_time']:<15.4f}")
        else:
            print(f"{'perturbation without solver':<30} | "
                f"{initial['with_solver']['delta_without_solver']['combined_loss']:<15.4e} | "
                f"{final['with_solver']['delta_without_solver']['combined_loss']:<15.4e} | "
                f"{without_solver_increase:<15.4e} | "
                f"{without_solver_pct:<12.2f}%")

    initial_losses = None
    all_timing_metrics = []

    old_a = a.clone()

    for step in range(num_steps):
        step_metrics = {}
        timing_metrics = {"with_solver":{},"without_solver":{}}
        
        # 1. Compute gradient with solver (with timing)
        a_with = a + delta_with
        start_forward = time.time()
        loss_with, forward_timing_with = wrap_with_solver(a_with)
        timing_metrics["with_solver"]['with_solver_forward_time'] = time.time() - start_forward
        timing_metrics["with_solver"].update({f'with_solver_{k}':v for k,v in forward_timing_with.items()})
        
        start_backward = time.time()
        loss_with.backward()
        timing_metrics["with_solver"]['with_solver_backward_time'] = time.time() - start_backward
        
        grad_with = delta_with.grad.clone().detach()
        delta_with.grad.zero_()
        
        # 2. Compute gradient without solver (with timing)
        a_without = a + delta_without
        start_forward = time.time()
        loss_without, forward_timing_without = wrap_without_solver(a_without)
        timing_metrics["without_solver"]['without_solver_forward_time'] = time.time() - start_forward
        timing_metrics["without_solver"].update({f'without_solver_{k}':v for k,v in forward_timing_without.items()})
        
        start_backward = time.time()
        loss_without.backward()
        timing_metrics["without_solver"]['without_solver_backward_time'] = time.time() - start_backward
        
        grad_without = delta_without.grad.clone().detach()
        delta_without.grad.zero_()
        
        # 3. Compare solver vs non-solver gradients
        grad_diff_metrics = compute_metrics_less(grad_with, grad_without, "Gradient With Solver", "Gradient Without Solver")
        
        # Update perturbations
        def update(delta, grad):
            if norm == 'inf':
                delta.data.add_(alpha * torch.sign(grad))
                delta.data.clamp_(-epsilon, epsilon)
            elif norm == 2:
                delta.data.add_(1000 * alpha * grad / (torch.norm(grad, p=2) + 1e-15))
                radius = 10 * (len(delta))**(1/2) * epsilon
                l2_norm = torch.norm(delta.data, p=2)
                if l2_norm > radius:
                    delta.data = delta.data * (radius / l2_norm)
        
        # 1. Compute ALL losses BEFORE update
        current_before = compute_all_loss_metrics(a, delta_with, delta_without, G, g)
        
        # Store initial losses if first step
        if step == 0:
            initial_losses = current_before
        
        # 2. Update perturbations
        update(delta_with, grad_with)
        update(delta_without, grad_without)
        
        # 3. Compute ALL losses AFTER update
        current_after = compute_all_loss_metrics(a, delta_with, delta_without, G, g)
        
        # Store final losses if last step
        if step == num_steps - 1:
            final_losses = current_after
            
            grad_records["final_values"] = {}

            grad_records["final_values"]["a_values"] = {}
            grad_records["final_values"]["a_values"]['a'] = old_a.squeeze().clone().detach()
            grad_records["final_values"]["a_values"]['a_with'] = a_with.squeeze().clone().detach()
            grad_records["final_values"]["a_values"]['a_without'] = a_without.squeeze().clone().detach()

            grad_records["final_values"]["G_values"] = {}
            grad_records["final_values"]["G_values"]['G(a)'] = G(old_a).squeeze().clone().detach()
            grad_records["final_values"]["G_values"]['G(a_with)'] = G(a_with).squeeze().clone().detach()
            grad_records["final_values"]["G_values"]['G(a_without)'] = G(a_without).squeeze().clone().detach()

            grad_records["final_values"]["g_values"] = {}
            grad_records["final_values"]["g_values"]['g(a)'] = JaxPDEWrapper.apply(old_a, g).squeeze().clone().detach()
            grad_records["final_values"]["g_values"]['g(a_with)'] = JaxPDEWrapper.apply(a_with, g).squeeze().clone().detach()
            grad_records["final_values"]["g_values"]['g(a_without)'] = JaxPDEWrapper.apply(a_with, g).squeeze().clone().detach()

        # Store metrics
        step_metrics_add = {
            'grad_metrics': grad_diff_metrics,
            'loss_metrics': {
                'before': {
                    'with_solver': current_before['with_solver']['delta_with'],
                    'with_solver_on_without': current_before['with_solver']['delta_without_solver'],
                },
                'after': {
                    'with_solver': current_after['with_solver']['delta_with'],
                    'with_solver_on_without': current_after['with_solver']['delta_without_solver'],
                }
            },
            "timing_metrics": timing_metrics
        }
        step_metrics.update(step_metrics_add)
        grad_records["step_update"][f"step_{step}"] = step_metrics
        all_timing_metrics.append(timing_metrics)

        if step == num_steps - 1 and initial_losses is not None:
            print(f"\n{'#'*80}")
            print(f"## FINAL COMPARISON: INITIAL (Step 0) vs FINAL (Step {step}) RESULTS")
            print(f"{'#'*80}")
            
            print_combined_comparison_table(grad_diff_metrics, step)
            print_complete_comparison(initial_losses, final_losses, step, all_timing_metrics)

    return grad_records

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])

def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir

    model_path = config.model_path
    workdir = config.workdir
    attack_method = config.attack_method
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M") 
    device = config.device
    nu = float(re.findall(r'nu([a-zA-Z0-9|.]+)', model_path)[0])
    num_steps_list = config.num_steps_list
    epsilon_list = config.epsilon_list
    num_samples = config.num_samples
    num_records = config.num_records

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
        'seed': 3003,
        'zero_mean':False
    }

    burgers_params = {
        'nu': nu,
        'simulation_time': 1.0,
        "step": 0.001
    }

    dataset_params = {
        "num_record": num_records
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
    gradient_folder = current_file_path / "perturbation_methods" / "gradient_comparison_results"
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
                        if norm == 2:
                            gradient_records[(f"norm_{norm}",f"index_{i}",f"numsteps_{num_steps}",f"epsilon_{10 * (len(delta))**(1/2) * epsilon:.4f}",f"alpha_{1000*epsilon/num_steps:.5f}")] = convert_to_cpu_serializable(records)
                            print(" \n     index:", i, ",    norm:", norm, ",    num_steps:", num_steps, ",    epsilon:", 10 * (len(delta))**(1/2) * epsilon, ",    alpha:", 1000*epsilon/num_steps)
                        elif norm == "inf":
                            gradient_records[(f"norm_{norm}",f"index_{i}",f"numsteps_{num_steps}",f"epsilon_{epsilon}",f"alpha_{epsilon/num_steps:.5f}")] = convert_to_cpu_serializable(records)
                            print(" \n     index:", i, ",    norm:", norm, ",    num_steps:", num_steps, ",    epsilon:", epsilon, ",    alpha:", epsilon/num_steps)
            # except:
            #     pass

    gradient_filename = f"gradient_comparison_norm2modified_{solver_name}_nu{nu}.pkl"
    with open(gradient_folder / gradient_filename, "wb") as f:
        pickle.dump(gradient_records, f)



if __name__ == "__main__":
    app.run(main)