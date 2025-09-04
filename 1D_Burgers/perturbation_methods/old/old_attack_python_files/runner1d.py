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

from PGD_without_solver_gradient_1d import pgd_attack as pgd_without_solver_grad_attack
from PGD_with_solver_gradient_1d_new import pgd_attack as pgd_with_solver_grad_attack
from random_1d import random_attack

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
    print("attack_method: ",attack_method)
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

    if attack_method == "PGD without solver gradient" or attack_method == "random":
        input_path = config.dict_input_path
        output_path = config.dict_output_path
        print("input_path: ",input_path)
        print("output_path: ",output_path)
        x_dict = torch.load(input_path)[:,::sub].to(device)
        y_dict = torch.load(output_path)[:,::sub].to(device)

    solver_name = re.findall(r'solver=([a-zA-Z0-9]+)', model_path)[0]

    model = FNO1d(modes=16, width=64) 
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    if attack_method != "random":
        norm_list = [2, "inf"]
    else:
        norm_list = ["none"]  # Dummy value to run once without norm

    for norm in norm_list:
        result_list = []
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

                        if attack_method == "PGD without solver gradient":
                            a_delta_sum, perturbed_input = pgd_without_solver_grad_attack(a_torch, model, x_dict, y_dict, epsilon, alpha, num_steps, norm=norm)
                        elif attack_method == "PGD with solver gradient":
                            PDE_func = lambda u0: solver.solve(u0, t_final=burgers_params['simulation_time'], t_eval=t_span, step=burgers_params["step"])[1][1]
                            a_delta_sum, perturbed_input = pgd_with_solver_grad_attack(a_torch, model, PDE_func, epsilon, alpha, num_steps, norm=norm)
                        elif attack_method == "random":
                            a_delta_sum, perturbed_input, max_rmse = random_attack(a_torch, model, x_dict, y_dict, epsilon, num_samples)
                        a_delta_sum = a_delta_sum.cpu().detach().numpy()
                        output_of_input = np.squeeze(model(a_torch.clone()).cpu().detach().numpy())
                        output_of_perturbed_input = np.squeeze(model(a_torch.clone()+perturbed_input).cpu().detach().numpy())
                        
                        solution_a_delta_sum = np.squeeze(solver.solve(np.squeeze(a_delta_sum.copy()), burgers_params['simulation_time'], t_span, burgers_params["step"])[t_span[1]][1])
                        solution_a = np.squeeze(solver.solve(a.copy(), burgers_params['simulation_time'], t_span, burgers_params["step"])[t_span[1]][1])
                        
                        if attack_method == "PGD without solver gradient":
                           result_dict = {
                                        "index":i,
                                        "epsilon":epsilon,
                                        "num_steps":num_steps,
                                        "norm":norm,
                                        "alpha":alpha,
                                        "dict_len":y_dict.shape[0],
                                        "a":a,
                                        "delta":np.squeeze(perturbed_input.cpu().detach().numpy()),
                                        "a+delta":np.squeeze(a_delta_sum),
                                        "g(a)":solution_a,
                                        "G(a)":output_of_input,
                                        "g(a+delta)":solution_a_delta_sum,
                                        "G(a+delta)":output_of_perturbed_input,
                                        "original deviation RMSE": (np.mean((output_of_input-solution_a)**2))**(1/2),
                                        "perturbed deviation RMSE": (np.mean((output_of_perturbed_input-solution_a_delta_sum)**2))**(1/2)
                                    }
                        elif attack_method == "PGD with solver gradient":
                                    result_dict = {
                                        "index":i,
                                        "epsilon":epsilon,
                                        "num_steps":num_steps,
                                        "norm":norm,
                                        "alpha":alpha,
                                        "a":a,
                                        "delta":np.squeeze(perturbed_input.cpu().detach().numpy()),
                                        "a+delta":np.squeeze(a_delta_sum),
                                        "g(a)":solution_a,
                                        "G(a)":output_of_input,
                                        "g(a+delta)":solution_a_delta_sum,
                                        "G(a+delta)":output_of_perturbed_input,
                                        "original deviation RMSE": (np.mean((output_of_input-solution_a)**2))**(1/2),
                                        "perturbed deviation RMSE": (np.mean((output_of_perturbed_input-solution_a_delta_sum)**2))**(1/2)
                                    }
                        elif attack_method == "random":
                                    result_dict = {
                                        "index":i,
                                        "epsilon":epsilon,
                                        "num_samples":num_samples,
                                        "norm":norm,
                                        "max_rmse":max_rmse,
                                        "dict_len":y_dict.shape[0],
                                        "a":a,
                                        "delta":np.squeeze(perturbed_input.cpu().detach().numpy()),
                                        "a+delta":np.squeeze(a_delta_sum),
                                        "g(a)":solution_a,
                                        "G(a)":output_of_input,
                                        "g(a+delta)":solution_a_delta_sum,
                                        "G(a+delta)":output_of_perturbed_input,
                                        "original deviation RMSE": (np.mean((output_of_input-solution_a)**2))**(1/2),
                                        "perturbed deviation RMSE": (np.mean((output_of_perturbed_input-solution_a_delta_sum)**2))**(1/2)
                                    }

                        result_list.append(result_dict)
                        result_list = convert_to_cpu_serializable(result_list)
                        end_time = time.time()
                        print(" \n     index:", i, ",    num_steps:", num_steps, ",    epsilon:", epsilon, ",    run time:", end_time-start_time)
            # except:
            #     pass

        param_names = [
            "pgdwithgrad" if attack_method == "PGD with solver gradient" 
            else "pgdnograd" if attack_method == "PGD without solver gradient" 
            else "rdm" if attack_method == "random" else "none",  # 默认情况（如果不是前两种）
            f"L{norm}" if attack_method != "random" else "none",
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



if __name__ == "__main__":
    app.run(main)