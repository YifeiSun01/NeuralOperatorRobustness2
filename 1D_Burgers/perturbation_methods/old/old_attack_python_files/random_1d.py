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

def random_attack(a, G, x_dict, y_dict, epsilon, num_samples=10000):
    max_rmse = 0
    worst_delta = torch.zeros_like(a)
    worst_a = a.clone()
    
    for _ in range(num_samples):
        # Generate random perturbation in [-epsilon, epsilon]
        delta = torch.empty_like(a).uniform_(-epsilon, epsilon)
        
        # Perturb the input
        a_perturbed = a + delta
        
        # Find closest ground truth output
        y_closest = find_closest_ground_truth(a_perturbed, x_dict, y_dict)
        
        # Compute FNO output
        with torch.no_grad():
            G_output = G(a_perturbed)
        
        # Calculate RMSE
        rmse = torch.sqrt(torch.mean((G_output - y_closest)**2))
        
        # Update worst case if found
        if rmse > max_rmse:
            max_rmse = rmse
            worst_delta = delta.clone()
            worst_a = a_perturbed.clone()

        if _ %1000 == 1:
            print(_)
    
    return worst_a, worst_delta, max_rmse

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
        "num_record": 10
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

    input_path = "/home/yifeisun/adversarial_robustness_FNO/perturbation_methods/x_y_dict/1D/Burgers/inputs_dim1d_nx1024_N20000_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.005_t1.0_seed45.pt"
    output_path = "/home/yifeisun/adversarial_robustness_FNO/perturbation_methods/x_y_dict/1D/Burgers/outputs_dim1d_nx1024_N20000_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.005_t1.0_seed45.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_dict = torch.load(input_path)[...,::sub].to(device)
    y_dict = torch.load(output_path)[...,::sub].to(device)

    model_path = "/home/yifeisun/adversarial_robustness_FNO/saved_models/1D/modes16_width64_epochs500/burgers_1d_FNO_model_trainedby_dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.005_t1.0_seed45.pth"
    solver_name = re.findall(r'solver=([a-zA-Z0-9]+)', model_path)[0]
    # print(solver_name)
    model = FNO1d(modes=16, width=64) 
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    for i in tqdm(range(dataset_params['num_record']), desc="adversarial input samples"):
        try:
            a = GRFGenerator.generate_grf(
                            shape=shape,
                            kernel=grf_params['kernel'],
                            kernel_params=grf_params['kernel_params'],
                            bc=grf_params['bc'],
                            seed=grf_params['seed'] + i,  # Ensure different GRFs for each sample
                            zero_mean=grf_params['zero_mean']
                        )
            
            a = a[::sub]
            num_samples_list = [1000,5000,10000,20000]
            epsilon_list = [0.001,0.003,0.005,0.007,0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7]
            # epsilon = 0.1
            for epsilon in epsilon_list:
                for num_samples in num_samples_list:
                
                    a_torch = torch.from_numpy(a).float().unsqueeze(0).unsqueeze(-1).to(device)
                    a_delta_sum, perturbed_input, max_rmse = random_attack(a_torch, model, x_dict, y_dict, epsilon, num_samples)
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
                    "num_samples":num_samples,
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
                    "perturbed deviation RMSE": (np.mean((output_of_perturbed_input-solution_a_delta_sum)**2))**(1/2)}

                    result_list.append(result_dict)
                    print(epsilon, i)
        except:
            pass

    param_names = [
            "rdm",
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

