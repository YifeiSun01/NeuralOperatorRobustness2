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

import torch.nn.functional as F  
from scipy.spatial.distance import cosine

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

def find_closest_ground_truth(a_perturbed, x_dict, y_dict):
    # Ensure a_perturbed has the correct shape for broadcasting
    # If a_perturbed is [256], reshape it to [1, 256]
    if len(a_perturbed.shape) == 1:
        a_perturbed = a_perturbed.unsqueeze(0)  # Shape: [1, 256]
    
    # Compute distances using broadcasting
    a_perturbed = a_perturbed.squeeze(-1)
    # print(x_dict.shape,a_perturbed.shape)
    distances = torch.norm(x_dict - a_perturbed, p=2, dim=1)  # Shape: [20000]
    
    # Find the closest index
    closest_idx = torch.argmin(distances)
    
    # Return the corresponding y_dict value
    return y_dict[closest_idx]



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
    delta_approximate = torch.zeros_like(a, requires_grad=True, dtype=dtype)
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

    def wrap_approximate(input):
        # Initialize timing dict
        timing = {}
        
        # Time G computation
        start = time.time()
        G_output = G(input)
        timing['G_time'] = time.time() - start
        
        # Time find_closest_ground_truth computation
        start = time.time()
        closest_output = find_closest_ground_truth(input, x_dict, y_dict)
        timing['find_time'] = time.time() - start
        
        # Time norm computation
        start = time.time()
        loss = torch.norm(G_output - closest_output, p=2)**2
        timing['norm_time'] = time.time() - start
        
        return loss, timing

    def compute_metrics_less(vec1, vec2, name1="vec1", name2="vec2"):
        vec1_flat = vec1.clone().detach().flatten()
        vec2_flat = vec2.clone().detach().flatten()
        name1 = "_".join(name1.lower().split(" "))
        name2 = "_".join(name2.lower().split(" "))
        
        diff = vec1_flat - vec2_flat
        rmse = torch.sqrt(torch.mean(diff**2))
        ame = torch.mean(torch.abs(diff)).item()  # Absolute Mean Error
        pme = torch.mean(torch.abs(diff)/torch.abs(vec1_flat)).item() * 100  # Percentage Mean Error
        
        # Calculate sign agreement
        sign1 = torch.sign(vec1_flat)
        sign2 = torch.sign(vec2_flat)
        sign_agreement = torch.mean((sign1 == sign2).float()).item() * 100  # percentage
        
        metrics = {
            f'{name1}_length': len(vec1_flat),
            f'{name2}_length': len(vec1_flat),
            f'{name1}_norm': torch.norm(vec1_flat).item(),
            f'{name2}_norm': torch.norm(vec2_flat).item(),
            'norm_ratio': (torch.norm(vec1_flat)/torch.norm(vec2_flat)).item(),
            'rmse': rmse.item(),
            'ame': ame,  # Absolute Mean Error
            'pme': pme,  # Percentage Mean Error (%)
            'cosine_similarity': F.cosine_similarity(vec1_flat.unsqueeze(0), vec2_flat.unsqueeze(0)).item(),
            'angle_degrees': torch.rad2deg(torch.acos(torch.clamp(
                F.cosine_similarity(vec1_flat.unsqueeze(0), vec2_flat.unsqueeze(0)), 
                -1.0, 1.0))).item(),
            'sign_agreement_percentage': sign_agreement,
        }
        
        return metrics

    def compute_metrics_less_2(vec1, vec2, name1="vec1", name2="vec2"):
        vec1_flat = vec1.clone().detach().flatten()
        vec2_flat = vec2.clone().detach().flatten()
        name1 = "_".join(name1.lower().split(" "))
        name2 = "_".join(name2.lower().split(" "))
        
        # Calculate sign agreement
        sign1 = torch.sign(vec1_flat)
        sign2 = torch.sign(vec2_flat)
        sign_agreement = torch.mean((sign1 == sign2).float()).item() * 100  # percentage
        
        metrics = {
            f'{name1}_length': len(vec1_flat),
            f'{name2}_length': len(vec1_flat),
            f'{name1}_norm': torch.norm(vec1_flat).item(),
            f'{name2}_norm': torch.norm(vec2_flat).item(),
            'cosine_similarity': F.cosine_similarity(vec1_flat.unsqueeze(0), vec2_flat.unsqueeze(0)).item(),
            'angle_degrees': torch.rad2deg(torch.acos(torch.clamp(
                F.cosine_similarity(vec1_flat.unsqueeze(0), vec2_flat.unsqueeze(0)), 
                -1.0, 1.0))).item(),
            'sign_agreement_percentage': sign_agreement,
        }
        
        return metrics

    def print_comparison_table(grad_diff_metrics_1, grad_diff_metrics_2, grad_diff_metrics_3, step, text="GRADIENT"):
        print(f"\n{'=' * 160}")
        print(f"{text} COMPARISON ANALYSIS (Step {step})".center(160))
        print(f"{'=' * 160}")
        
        # Updated header with new metrics
        print(f"{'Comparison':<50} | {'L2 Norm Ratio':<15} | {'RMSE':<15} | {'AME':<15} | {'PME (%)':<15} | "
            f"{'Cosine Similarity':<18} | {'Angle (Degrees)':<15} | {'Sign Agree %':<15}")
        print(f"{'-' * 50}+{'-' * 17}+{'-' * 17}+{'-' * 17}+{'-' * 17}+{'-' * 20}+{'-' * 17}+{'-' * 17}")
        
        # Row 1
        angle_str = f"{grad_diff_metrics_1['angle_degrees']:.5f}°"
        print(f"{'1. with PDE solver vs PDE detached':<50} | "
            f"{grad_diff_metrics_1['norm_ratio']:<15.5f} | "
            f"{grad_diff_metrics_1['rmse']:<15.6e} | "
            f"{grad_diff_metrics_1['ame']:<15.6e} | "
            f"{grad_diff_metrics_1['pme']:<15.2f} | "
            f"{grad_diff_metrics_1['cosine_similarity']:<18.6f} | "
            f"{angle_str:<15} | "
            f"{grad_diff_metrics_1['sign_agreement_percentage']:<15.2f}")

        # Row 2
        angle_str = f"{grad_diff_metrics_2['angle_degrees']:.5f}°"
        print(f"{'2. with PDE solver vs PDE approximated':<50} | "
            f"{grad_diff_metrics_2['norm_ratio']:<15.5f} | "
            f"{grad_diff_metrics_2['rmse']:<15.6e} | "
            f"{grad_diff_metrics_2['ame']:<15.6e} | "
            f"{grad_diff_metrics_2['pme']:<15.2f} | "
            f"{grad_diff_metrics_2['cosine_similarity']:<18.6f} | "
            f"{angle_str:<15} | "
            f"{grad_diff_metrics_2['sign_agreement_percentage']:<15.2f}")

        # Row 3
        angle_str = f"{grad_diff_metrics_3['angle_degrees']:.5f}°"
        print(f"{'3. PDE detached vs PDE approximated':<50} | "
            f"{grad_diff_metrics_3['norm_ratio']:<15.5f} | "
            f"{grad_diff_metrics_3['rmse']:<15.6e} | "
            f"{grad_diff_metrics_3['ame']:<15.6e} | "
            f"{grad_diff_metrics_3['pme']:<15.2f} | "
            f"{grad_diff_metrics_3['cosine_similarity']:<18.6f} | "
            f"{angle_str:<15} | "
            f"{grad_diff_metrics_3['sign_agreement_percentage']:<15.2f}")
        
        print(f"{'=' * 160}\n")

    def compute_all_loss_metrics(a, delta_with, delta_without, delta_approximate, G, g):
        return {
        'delta_with': wrap_with_solver(a + delta_with)[0].item(),
        'delta_without_solver': wrap_with_solver(a + delta_without)[0].item(),
        'delta_approximate': wrap_with_solver(a + delta_approximate)[0].item(),
        }

    def print_complete_comparison(initial, final, final_step, timing_metrics=None):
        """Compare initial (step=0) and final step losses with percentage increase and timing metrics"""
        # Calculate absolute increases
        with_solver_increase = final['delta_with'] - initial['delta_with']
        without_solver_increase = final['delta_without_solver'] - initial['delta_without_solver']
        approximate_increase = final['delta_approximate'] - initial['delta_approximate']
        
        # Calculate percentage increases (with protection against division by zero)
        def calc_pct_increase(final, initial):
            return ((final - initial) / (initial + 1e-10)) * 100
        
        with_solver_pct = calc_pct_increase(final['delta_with'], 
                                        initial['delta_with'])
        without_solver_pct = calc_pct_increase(final['delta_without_solver'], 
                                            initial['delta_without_solver'])
        approximate_pct = calc_pct_increase(final['delta_approximate'], 
                                            initial['delta_approximate'])

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
                'approximate_forward_avg': sum(m["approximate"]['approximate_forward_time'] for m in timing_metrics[1:]) / (num_steps - 1),
                'approximate_backward_avg': sum(m["approximate"]['approximate_backward_time'] for m in timing_metrics[1:]) / (num_steps - 1),
                'approximate_G_avg': sum(m["approximate"]['approximate_G_time'] for m in timing_metrics[1:]) / (num_steps - 1),
                'approximate_find_avg': sum(m["approximate"]['approximate_find_time'] for m in timing_metrics[1:]) / (num_steps - 1),
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
        
        if timing_metrics:
            if avg_timing_metrics:  # Use average metrics if available
                print(f"{'perturbation with solver':<30} | "
                    f"{initial['delta_with']:<15.4e} | "
                    f"{final['delta_with']:<15.4e} | "
                    f"{with_solver_increase:<15.4e} | "
                    f"{with_solver_pct:<12.2f}% | "
                    f"{avg_timing_metrics['with_solver_forward_avg']:<10.4f} | "
                    f"{avg_timing_metrics['with_solver_backward_avg']:<10.4f} | "
                    f"{avg_timing_metrics['with_solver_G_avg']:<10.4f} | "
                    f"{avg_timing_metrics['with_solver_g_avg']:<15.4f}")
            else:  # Single timing metric case
                print(f"{'perturbation with solver':<30} | "
                    f"{initial['delta_with']:<15.4e} | "
                    f"{final['delta_with']:<15.4e} | "
                    f"{with_solver_increase:<15.4e} | "
                    f"{with_solver_pct:<12.2f}% | "
                    f"{timing_metrics[0]['with_solver']['with_solver_forward_time']:<10.4f} | "
                    f"{timing_metrics[0]['with_solver']['with_solver_backward_time']:<10.4f} | "
                    f"{timing_metrics[0]['with_solver']['with_solver_G_time']:<10.4f} | "
                    f"{timing_metrics[0]['with_solver']['with_solver_g_time']:<15.4f}")
        else:
            print(f"{'perturbation with solver':<30} | "
                f"{initial['delta_with']:<15.4e} | "
                f"{final['delta_with']:<15.4e} | "
                f"{with_solver_increase:<15.4e} | "
                f"{with_solver_pct:<12.2f}%")
        
        if timing_metrics:
            if avg_timing_metrics:  # Use average metrics if available
                print(f"{'perturbation PDE detached':<30} | "
                    f"{initial['delta_without_solver']:<15.4e} | "
                    f"{final['delta_without_solver']:<15.4e} | "
                    f"{without_solver_increase:<15.4e} | "
                    f"{without_solver_pct:<12.2f}% | "
                    f"{avg_timing_metrics['without_solver_forward_avg']:<10.4f} | "
                    f"{avg_timing_metrics['without_solver_backward_avg']:<10.4f} | "
                    f"{avg_timing_metrics['without_solver_G_avg']:<10.4f} | "
                    f"{avg_timing_metrics['without_solver_find_avg']:<15.4f}")
            else:  # Single timing metric case
                print(f"{'perturbation PDE detached':<30} | "
                    f"{initial['delta_without_solver']:<15.4e} | "
                    f"{final['delta_without_solver']:<15.4e} | "
                    f"{without_solver_increase:<15.4e} | "
                    f"{without_solver_pct:<12.2f}% | "
                    f"{timing_metrics[0]['without_solver']['without_solver_forward_time']:<10.4f} | "
                    f"{timing_metrics[0]['without_solver']['without_solver_backward_time']:<10.4f} | "
                    f"{timing_metrics[0]['without_solver']['without_solver_G_time']:<10.4f} | "
                    f"{timing_metrics[0]['without_solver']['without_solver_find_time']:<15.4f}")
        else:
            print(f"{'perturbation PDE detached':<30} | "
                f"{initial['delta_without_solver']:<15.4e} | "
                f"{final['delta_without_solver']:<15.4e} | "
                f"{without_solver_increase:<15.4e} | "
                f"{without_solver_pct:<12.2f}%")

        if timing_metrics:
            if avg_timing_metrics:  # Use average metrics if available
                print(f"{'perturbation PDE approximate':<30} | "
                    f"{initial['delta_approximate']:<15.4e} | "
                    f"{final['delta_approximate']:<15.4e} | "
                    f"{approximate_increase:<15.4e} | "
                    f"{approximate_pct:<12.2f}% | "
                    f"{avg_timing_metrics['approximate_forward_avg']:<10.4f} | "
                    f"{avg_timing_metrics['approximate_backward_avg']:<10.4f} | "
                    f"{avg_timing_metrics['approximate_G_avg']:<10.4f} | "
                    f"{avg_timing_metrics['approximate_find_avg']:<15.4f}")
            else:  # Single timing metric case
                print(f"{'perturbation approximate':<30} | "
                    f"{initial['delta_approximate']:<15.4e} | "
                    f"{final['delta_approximate']:<15.4e} | "
                    f"{approximate_increase:<15.4e} | "
                    f"{approximate_pct:<12.2f}% | "
                    f"{timing_metrics[0]['approximate']['approximate_forward_time']:<10.4f} | "
                    f"{timing_metrics[0]['approximate']['approximate_backward_time']:<10.4f} | "
                    f"{timing_metrics[0]['approximate']['approximate_G_time']:<10.4f} | "
                    f"{timing_metrics[0]['approximate']['approximate_find_time']:<15.4f}")
        else:
            print(f"{'perturbation PDE approximate':<30} | "
                f"{initial['delta_approximate']:<15.4e} | "
                f"{final['delta_approximate']:<15.4e} | "
                f"{approximate_increase:<15.4e} | "
                f"{approximate_pct:<12.2f}%")

    initial_losses = None
    all_timing_metrics = []

    old_a = a.clone()

    current_after = compute_all_loss_metrics(a, delta_with, delta_without, delta_approximate, G, g)
    step_metrics = {}
    step_metrics_add = {
            'loss_metrics': {
                    'with_solver': current_after['delta_with'],
                    'without_solver': current_after['delta_without_solver'],
                    'approximate': current_after['delta_approximate'],
            }
        }
    step_metrics.update(step_metrics_add)
    grad_records["step_update"][f"step_0"] = step_metrics
    
    def zero_all_gradients():
        for param in G.parameters():
            if param.grad is not None:
                param.grad.detach_() 
                param.grad.zero_()

        if hasattr(JaxPDEWrapper, '_vjp_cache'):
            JaxPDEWrapper._vjp_cache.clear() 

    for step in range(num_steps):
        step_metrics = {}
        timing_metrics = {"with_solver":{},"without_solver":{},"approximate":{}}
        
        # 1. Compute gradient with solver (with timing)
        a_with = a + delta_with
        zero_all_gradients()
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
        zero_all_gradients()
        start_forward = time.time()
        loss_without, forward_timing_without = wrap_without_solver(a_without)
        timing_metrics["without_solver"]['without_solver_forward_time'] = time.time() - start_forward
        timing_metrics["without_solver"].update({f'without_solver_{k}':v for k,v in forward_timing_without.items()})
        
        start_backward = time.time()
        loss_without.backward()
        timing_metrics["without_solver"]['without_solver_backward_time'] = time.time() - start_backward
        
        grad_without = delta_without.grad.clone().detach()
        delta_without.grad.zero_()

        # 3. Compute gradient approximate (with timing)
        a_approximate = a + delta_approximate
        zero_all_gradients()
        start_forward = time.time()
        loss_approximate, forward_timing_approximate = wrap_approximate(a_approximate)
        timing_metrics["approximate"]['approximate_forward_time'] = time.time() - start_forward
        timing_metrics["approximate"].update({f'approximate_{k}':v for k,v in forward_timing_approximate.items()})
        
        start_backward = time.time()
        loss_approximate.backward()
        timing_metrics["approximate"]['approximate_backward_time'] = time.time() - start_backward
        
        grad_approximate = delta_approximate.grad.clone().detach()
        delta_approximate.grad.zero_()
        
        # 3. Compare solver vs non-solver gradients
        grad_diff_metrics_1 = compute_metrics_less(grad_with, grad_without, "Gradient With Solver", "Gradient Detached")
        grad_diff_metrics_2 = compute_metrics_less(grad_with, grad_approximate, "Gradient With Solver", "Gradient Approximated")
        grad_diff_metrics_3 = compute_metrics_less(grad_without, grad_approximate, "Gradient Detached", "Gradient Approximated")
        
        # Update perturbations
        def update(delta, grad):
            if norm == 'inf':
                # Perform the update
                delta.data.add_(alpha * torch.sign(grad))

                # Before clamping, count how many entries will exceed bounds
                pre_clip = delta.data.clone()
                clipped = pre_clip.clamp(-epsilon, epsilon)

                # Calculate percentage of entries that were clipped (i.e., hit boundary)
                num_total = delta.numel()
                num_clipped = (pre_clip != clipped).sum().item()
                clipped_percentage = num_clipped / num_total

                # Apply the clipping
                delta.data.copy_(clipped)

                return clipped_percentage  # float between 0 and 1

            elif norm == 2:
                # Perform the update
                delta.data.add_(alpha * grad / (torch.norm(grad, p=2) + 1e-15))

                # Compute the current L2 norm
                l2_norm = torch.norm(delta.data, p=2)
                radius = epsilon

                # Check whether it exceeds the radius
                exceeded = l2_norm > radius

                if exceeded:
                    delta.data = delta.data * (radius / l2_norm)

                return exceeded  # Boolean
        
        # 1. Compute ALL losses BEFORE update
        current_before = compute_all_loss_metrics(a, delta_with, delta_without, delta_approximate, G, g)
        
        # Store initial losses if first step
        if step == 0:
            initial_losses = current_before
        
        # 2. Update perturbations
        with_boundary = update(delta_with, grad_with)
        without_boundary = update(delta_without, grad_without)
        approximate_boundary = update(delta_approximate, grad_approximate)

        delta_diff_metrics_1 = compute_metrics_less(delta_with, delta_without, "Delta With Solver", "Delta Detached")
        delta_diff_metrics_2 = compute_metrics_less(delta_with, delta_approximate, "Delta With Solver", "Delta Approximated")
        delta_diff_metrics_3 = compute_metrics_less(delta_without, delta_approximate, "Delta Detached", "Delta Approximated")

        grad_delta_diff_metrics_1 = compute_metrics_less_2(grad_with, delta_with, "Gradient With Solver", "Delta With Solver")
        grad_delta_diff_metrics_2 = compute_metrics_less_2(grad_without, delta_without, "Gradient Detached", "Delta Detached")
        grad_delta_diff_metrics_3 = compute_metrics_less_2(grad_approximate, delta_approximate, "Gradient Approximated", "Delta Approximated")
        
        # 3. Compute ALL losses AFTER update
        current_after = compute_all_loss_metrics(a, delta_with, delta_without, delta_approximate, G, g)

        grad_records["step_update"][f"step_{step+1}"] = {}

        grad_records["step_update"][f"step_{step+1}"]["values"] = {}

        grad_records["step_update"][f"step_{step+1}"]["values"]["a_values"] = {}
        grad_records["step_update"][f"step_{step+1}"]["values"]["a_values"]['a'] = old_a.squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["a_values"]['a_with'] = a_with.squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["a_values"]['a_without'] = a_without.squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["a_values"]['a_approximate'] = a_approximate.squeeze().clone().detach()

        grad_records["step_update"][f"step_{step+1}"]["values"]["G_values"] = {}
        grad_records["step_update"][f"step_{step+1}"]["values"]["G_values"]['G(a)'] = G(old_a).squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["G_values"]['G(a_with)'] = G(a_with).squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["G_values"]['G(a_without)'] = G(a_without).squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["G_values"]['G(a_approximate)'] = G(a_approximate).squeeze().clone().detach()

        grad_records["step_update"][f"step_{step+1}"]["values"]["g_values"] = {}
        grad_records["step_update"][f"step_{step+1}"]["values"]["g_values"]['g(a)'] = JaxPDEWrapper.apply(old_a, g).squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["g_values"]['g(a_with)'] = JaxPDEWrapper.apply(a_with, g).squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["g_values"]['g(a_without)'] = JaxPDEWrapper.apply(a_without, g).squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["g_values"]['g(a_approximate)'] = JaxPDEWrapper.apply(a_approximate, g).squeeze().clone().detach()
        
        # Store final losses if last step
        if step == num_steps - 1:
            final_losses = current_after
            
            grad_records["final_values"] = {}

            grad_records["final_values"]["a_values"] = {}
            grad_records["final_values"]["a_values"]['a'] = old_a.squeeze().clone().detach()
            grad_records["final_values"]["a_values"]['a_with'] = a_with.squeeze().clone().detach()
            grad_records["final_values"]["a_values"]['a_without'] = a_without.squeeze().clone().detach()
            grad_records["final_values"]["a_values"]['a_approximate'] = a_approximate.squeeze().clone().detach()

            grad_records["final_values"]["G_values"] = {}
            grad_records["final_values"]["G_values"]['G(a)'] = G(old_a).squeeze().clone().detach()
            grad_records["final_values"]["G_values"]['G(a_with)'] = G(a_with).squeeze().clone().detach()
            grad_records["final_values"]["G_values"]['G(a_without)'] = G(a_without).squeeze().clone().detach()
            grad_records["final_values"]["G_values"]['G(a_approximate)'] = G(a_approximate).squeeze().clone().detach()

            grad_records["final_values"]["g_values"] = {}
            grad_records["final_values"]["g_values"]['g(a)'] = JaxPDEWrapper.apply(old_a, g).squeeze().clone().detach()
            grad_records["final_values"]["g_values"]['g(a_with)'] = JaxPDEWrapper.apply(a_with, g).squeeze().clone().detach()
            grad_records["final_values"]["g_values"]['g(a_without)'] = JaxPDEWrapper.apply(a_without, g).squeeze().clone().detach()
            grad_records["final_values"]["g_values"]['g(a_approximate)'] = JaxPDEWrapper.apply(a_approximate, g).squeeze().clone().detach()

        # Store metrics
        step_metrics_add = {
            "reaches_boundary": {
                "with_solver": with_boundary,
                "detached": without_boundary,
                "approximated": approximate_boundary,
            },
            'grad_metrics': {
                "withsolver_vs_detached": grad_diff_metrics_1,
                "withsolver_vs_approximated": grad_diff_metrics_2,
                "detached_vs_approximated": grad_diff_metrics_3,
            },
            'delta_metrics': {
                "withsolver_vs_detached": delta_diff_metrics_1,
                "withsolver_vs_approximated": delta_diff_metrics_2,
                "detached_vs_approximated": delta_diff_metrics_3,
            },
            'grad_delta_metrics': {
                "withsolver": grad_delta_diff_metrics_1,
                "detached": grad_delta_diff_metrics_2,
                "approximated": grad_delta_diff_metrics_3,
            },
            'loss_metrics': {
                    'with_solver': current_after['delta_with'],
                    'without_solver': current_after['delta_without_solver'],
                    'approximate': current_after['delta_approximate'],
            },
            "timing_metrics": timing_metrics
        }
        step_metrics.update(step_metrics_add)
        grad_records["step_update"][f"step_{step+1}"].update(step_metrics)
        all_timing_metrics.append(timing_metrics)

        print_comparison_table(grad_diff_metrics_1, grad_diff_metrics_2, grad_diff_metrics_3, step, text="GRADIENT")
        print_comparison_table(delta_diff_metrics_1, delta_diff_metrics_2, delta_diff_metrics_3, step, text="DELTA")

        if step == num_steps - 1 and initial_losses is not None:
            print(f"\n{'#'*80}")
            print(f"## FINAL COMPARISON: INITIAL (Step 0) vs FINAL (Step {step}) RESULTS")
            print(f"{'#'*80}")
            
            # print_comparison_table(grad_diff_metrics_1, grad_diff_metrics_2, grad_diff_metrics_3, step, text="GRADIENT")
            # print_comparison_table(delta_diff_metrics_1, delta_diff_metrics_2, delta_diff_metrics_3, step, text="DELTA")
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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M") 
    device = config.device
    nu = float(re.findall(r'nu([a-zA-Z0-9|.]+)', model_path)[0])
    num_records = config.num_records
    inputs = config.inputs

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

    gradient_records = {}

    for input_tuple in inputs:
        norm, epsilon, num_steps, alpha = input_tuple
        for i in tqdm(range(dataset_params['num_record']), desc="adversarial input samples"):
            start_time = time.time()
            a = GRFGenerator.generate_grf(
                            shape=shape,
                            kernel=grf_params['kernel'],
                            kernel_params=grf_params['kernel_params'],
                            bc=grf_params['bc'],
                            seed=grf_params['seed'] + i,  # Ensure different GRFs for each sample
                            zero_mean=grf_params['zero_mean']
                        )
            
            a = a[::sub]
            
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
                alpha=alpha,
                num_steps=num_steps,
                norm=norm,
                enable_fd=True,
                fd_delta=1e-8
            )
            gradient_records[(f"norm_{norm}",f"index_{i}",f"numsteps_{num_steps}",f"epsilon_{epsilon}",f"alpha_{alpha:.5f}")] = convert_to_cpu_serializable(records)
            total_time = time.time() - start_time
            print(" \n     index:", i, ",    norm:", norm, ",    num_steps:", num_steps, ",    epsilon:", epsilon, ",    alpha:", alpha, ",    total time:", total_time)

    gradient_filename = f"gradient_comparison_{solver_name}_nu{nu}_nsamples{num_records}.pkl"
    with open(gradient_folder / gradient_filename, "wb") as f:
        pickle.dump(gradient_records, f)

if __name__ == "__main__":
    app.run(main)