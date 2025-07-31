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

    def wrap_with_solver(input):
        return torch.norm(G(input) - JaxPDEWrapper.apply(input, g), p=2)**2

    def wrap_without_solver(input):
        return torch.norm(G(input) - find_closest_ground_truth(input, x_dict, y_dict), p=2)**2

    def compute_metrics_less(vec1, vec2, name1="vec1", name2="vec2"):
        vec1_flat = vec1.flatten()
        vec2_flat = vec2.flatten()
        
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

    def print_comparison_table(metrics, title, name1, name2):
        print(f"\n{'='*100}")
        print(f"{title.center(100)}")
        print(f"{'='*100}")
        
        print(f"{'Metric':<30} | {name1:<20} | {name2:<20} | {'Difference/Analysis':<25}")
        print(f"{'-'*30}+{'-'*22}+{'-'*22}+{'-'*25}")
        print(f"{'Vector Length':<30} | {metrics['vec1_length']:<20} | {metrics['vec2_length']:<20} |")
        print(f"{'L2 Norm':<30} | {metrics['vec1_norm']:<20.6e} | {metrics['vec2_norm']:<20.6e} | ratio: {metrics['norm_ratio']:.2f}")
        print(f"{'RMSE':<30} | {'-':<20} | {'-':<20} | {metrics['rmse']:.6e}")
        print(f"{'Cosine Similarity':<30} | {'-':<20} | {'-':<20} | {metrics['cosine_similarity']:.6f}")
        print(f"{'Angle (Degrees)':<30} | {'-':<20} | {'-':<20} | {metrics['angle_degrees']:.2f}°")
        
        print(f"{'='*100}\n")

    
    def compute_loss_metrics(a, delta, G, g, x_dict, y_dict, use_solver=True):
        if use_solver:
            return {
                'pde_loss': torch.norm(JaxPDEWrapper.apply(a + delta, g)).item(),
                'fno_loss': torch.norm(G(a + delta)).item(),
                'combined_loss': wrap_with_solver(a + delta).item()
            }
        else:
            return {
                'ground_truth_loss': torch.norm(find_closest_ground_truth(a + delta, x_dict, y_dict)).item(),
                'fno_loss': torch.norm(G(a + delta)).item(),
                'combined_loss': wrap_without_solver(a + delta).item()
            }

    def compute_all_loss_metrics(a, delta_with, delta_without, G, g, x_dict, y_dict):
        return {
            'with_solver': {
                'delta_with': compute_loss_metrics(a, delta_with, G, g, x_dict, y_dict, use_solver=True),
                'delta_without_solver': compute_loss_metrics(a, delta_without, G, g, x_dict, y_dict, use_solver=True),
                'delta_without_nosolver': compute_loss_metrics(a, delta_without, G, g, x_dict, y_dict, use_solver=False)
            }
        }

    def print_complete_comparison(before, after, step):
        with_solver_increase = after['with_solver']['delta_with']['combined_loss'] - before['with_solver']['delta_with']['combined_loss']
        without_solver_increase = after['with_solver']['delta_without_solver']['combined_loss'] - before['with_solver']['delta_without_solver']['combined_loss']
        without_nosolver_increase = after['with_solver']['delta_without_nosolver']['combined_loss'] - before['with_solver']['delta_without_nosolver']['combined_loss']
        
        print(f"\n{f'PERTURBATION LOSS ANALYSIS (Step {step})':^100}")
        print(f"{'-'*100}")
        
        print(f"{'ABSOLUTE LOSS VALUES':^100}")
        print(f"{'-'*100}")
        print(f"{'Case':<30} | {'Before':<20} | {'After':<20} | {'Increase':<20}")
        print(f"{'-'*30}+{'-'*22}+{'-'*22}+{'-'*20}")

        print(f"{'Delta_with (solver)':<30} | {before['with_solver']['delta_with']['combined_loss']:<20.6e} | "
            f"{after['with_solver']['delta_with']['combined_loss']:<20.6e} | {with_solver_increase:<20.6e}")
        print(f"{'Delta_without (solver)':<30} | {before['with_solver']['delta_without_solver']['combined_loss']:<20.6e} | "
            f"{after['with_solver']['delta_without_solver']['combined_loss']:<20.6e} | {without_solver_increase:<20.6e}")
        print(f"{'Delta_without (non-solver)':<30} | {before['with_solver']['delta_without_nosolver']['combined_loss']:<20.6e} | "
            f"{after['with_solver']['delta_without_nosolver']['combined_loss']:<20.6e} | {without_nosolver_increase:<20.6e}")

    for step in range(num_steps):
        step_metrics = {}
        
        # 1. Compute gradient with solver
        a_with = a + delta_with
        loss_with = wrap_with_solver(a_with)
        loss_with.backward()
        grad_with = delta_with.grad.clone().detach()
        delta_with.grad.zero_()
        
        # 2. Compute gradient without solver
        a_without = a + delta_without
        loss_without = wrap_without_solver(a_without)
        loss_without.backward()
        grad_without = delta_without.grad.clone().detach()
        delta_without.grad.zero_()
        
        # 3. Compare solver vs non-solver gradients
        grad_diff_metrics = compute_metrics_less(grad_with, grad_without, "Gradient With Solver", "Gradient Without Solver")
        
        # Store current perturbations
        step_metrics['a_with'] = a_with.clone().detach()
        step_metrics['a_without'] = a_without.clone().detach()

        # Update perturbations
        def update(delta, grad):
            if norm == 'inf':
                delta.data.add_(alpha * torch.sign(grad))
                delta.data.clamp_(-epsilon, epsilon)
            elif norm == 2:
                delta.data.add_(alpha * grad / (torch.norm(grad, p=2) + 1e-15))
                l2_norm = torch.norm(delta.data, p=2)
                if l2_norm > epsilon:
                    delta.data = delta.data * (epsilon / l2_norm)
        
        # 1. Compute ALL losses BEFORE update
        before = compute_all_loss_metrics(a, delta_with, delta_without, G, g, x_dict, y_dict)
        
        # 2. Update perturbations
        update(delta_with, grad_with)
        update(delta_without, grad_without)
        
        # 3. Compute ALL losses AFTER update
        after = compute_all_loss_metrics(a, delta_with, delta_without, G, g, x_dict, y_dict)
        
        # 4. Compute delta metrics
        delta_metrics = compute_metrics_less(delta_with, delta_without, "Delta With Solver", "Delta Without Solver")
        a_metrics = compute_metrics_less(a_with, a_without, "a (perturbation) With Solver", "a (perturbation) Without Solver")
        
        # Store metrics
        step_metrics_add = {
            'grad_metrics': grad_diff_metrics,
            'delta_metrics': delta_metrics,
            'a_metrics': a_metrics,
            'loss_metrics': {
                'before': {
                    'with_solver': before['with_solver']['delta_with'],
                    'with_solver_on_without': before['with_solver']['delta_without_solver'],
                    'without_solver': before['with_solver']['delta_without_nosolver']
                },
                'after': {
                    'with_solver': after['with_solver']['delta_with'],
                    'with_solver_on_without': after['with_solver']['delta_without_solver'],
                    'without_solver': after['with_solver']['delta_without_nosolver']
                }
            }
        }
        step_metrics.update(step_metrics_add)
        grad_records[f"step_{step}"] = step_metrics

        # Only print for final step
        if step == 0 or step == num_steps - 1:
            print(f"\n{'#'*80}")
            print(f"## FINAL STEP {step} GRADIENT COMPARISON RESULTS")
            print(f"{'#'*80}")
            
            print_comparison_table(grad_diff_metrics, 
                                f"COMPARISON 1: Analytical Gradients With Solver vs Without Solver (Step {step})",
                                "With Solver", "Without Solver")
            
            print_comparison_table(delta_metrics, 
                                f"COMPARISON 2: DELTA COMPARISON AFTER UPDATE With Solver vs Without Solver (Step {step})",
                                "With Solver", "Without Solver")
            
            print_comparison_table(a_metrics, 
                                f"COMPARISON 3: A (PERTURBATION) COMPARISON AFTER UPDATE With Solver vs Without Solver (Step {step})",
                                "With Solver", "Without Solver")
            
            print_complete_comparison(before, after, step)

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
                        print(" \n     index:", i, ",    norm:", norm, ",    num_steps:", num_steps, ",    epsilon:", epsilon)
            # except:
            #     pass

        gradient_filename = f"gradient_comparison_{solver_name}_nu{nu}_eps{epsilon}_steps{num_steps}_norm{norm}.pkl"
        with open(gradient_folder / gradient_filename, "wb") as f:
            pickle.dump(gradient_records, f)



if __name__ == "__main__":
    app.run(main)