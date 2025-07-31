import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO1d import FNO1d
from GRFs.generateGRFs import GRFGenerator
from solvers.burgers1d_solvers import *
import torch
from tqdm import tqdm
from pathlib import Path
import pickle
import time

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

def pgd_attack(a, G, x_dict, y_dict, epsilon, alpha, num_steps, norm='inf'):
    """
    Parameters:
        norm: 'inf' for L∞ norm, 2 for L2 norm
    """
    delta = torch.zeros_like(a, requires_grad=True)
    
    for step in range(num_steps):
        # t1 = time.time()
        a_perturbed = a + delta
        # t2 = time.time()
        y_closest = find_closest_ground_truth(a_perturbed, x_dict, y_dict)
        # t3 = time.time()
        G_output = G(a_perturbed)
        # t4 = time.time()
        loss = torch.norm(G_output - y_closest, p=2)**2
        # t5 = time.time()
        loss.backward()
        # t6 = time.time()
        # print(f"Step {step}: Perturb={t2-t1:.3f}s, find_closest={t3-t2:.3f}s, G={t4-t3:.3f}s, Loss={t5-t4:.3f}s, Backward={t6-t5:.3f}s")
        
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

    input_path = f"/home/yifeisun/adversarial_robustness_FNO/perturbation_methods/x_y_dict/1D/Burgers/inputs_dim1d_nx1024_N20000_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu{burgers_params['nu']}_t1.0_seed45.pt"
    output_path = f"/home/yifeisun/adversarial_robustness_FNO/perturbation_methods/x_y_dict/1D/Burgers/outputs_dim1d_nx1024_N20000_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu{burgers_params['nu']}_t1.0_seed45.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_dict = torch.load(input_path)[:,::sub].to(device)
    y_dict = torch.load(output_path)[:,::sub].to(device)
    # print({"x_dict_shape":x_dict.shape,"y_dict_shape":y_dict.shape})

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
    # for norm in [2,"inf"]:
    for norm in [2]:
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
                num_steps_list = [2,5,10,20,50,100,500,1000,2000]
                # epsilon_list = [0.001,0.003,0.005,0.007,0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7]
                epsilon_list = [0.01,0.05,0.1,0.5]
                # num_steps_list = [2,5,10,20]
                # epsilon_list = [0.001,0.003,0.005,0.007,0.01,0.03,0.05,0.07]
                for num_steps in num_steps_list:
                    # epsilon = 0.1
                    for epsilon in epsilon_list:
                        alpha = epsilon/num_steps
                        
                        a_torch = torch.from_numpy(a).float().unsqueeze(0).unsqueeze(-1).to(device)
                        a_delta_sum, perturbed_input = pgd_attack(a_torch, model, x_dict, y_dict, epsilon, alpha, num_steps, norm=norm)
                        a_delta_sum = a_delta_sum.cpu().detach().numpy()
                        output_of_input = np.squeeze(model(a_torch.clone()).cpu().detach().numpy())
                        output_of_perturbed_input = np.squeeze(model(a_torch.clone()+perturbed_input).cpu().detach().numpy())
                        
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
                        "dict_len":y_dict.shape[0],
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
                "pgdnograd",
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




