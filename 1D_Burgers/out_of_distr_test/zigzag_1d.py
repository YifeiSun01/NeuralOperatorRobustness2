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
import numpy as np

def generate_zigzag(shape, num_peaks=5, amplitude=1.0, seed=None, zero_mean=False):
    """
    Generate a piecewise linear zigzag function
    shape: tuple - dimensions of the output array
    num_peaks: int - number of peaks/valleys
    amplitude: float - maximum amplitude of the zigzag
    seed: int - optional random seed for reproducibility
    zero_mean: bool - whether to subtract the mean to make zero-centered
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.linspace(0, 1, shape[0])
    # Random peak positions
    peaks = np.sort(np.random.uniform(0, 1, num_peaks))
    # Create piecewise linear function
    y = np.zeros_like(x)
    current_val = np.random.uniform(-amplitude, amplitude)
    
    for i in range(len(peaks)+1):
        if i == 0:
            segment = x <= peaks[0] if len(peaks) > 0 else x <= 1
            next_val = np.random.uniform(-amplitude, amplitude)
        elif i == len(peaks):
            segment = x > peaks[-1]
            next_val = np.random.uniform(-amplitude, amplitude)
        else:
            segment = (x > peaks[i-1]) & (x <= peaks[i])
            next_val = np.random.uniform(-amplitude, amplitude)
        
        y[segment] = np.linspace(current_val, next_val, np.sum(segment))
        current_val = next_val
    
    # Normalize to [0,1] range first (optional, can remove if not needed)
    y = y * 2 * amplitude - amplitude  # Scale to [-amplitude, amplitude]
    y = (y - y.min()) / (y.max() - y.min())
    # y = (y - y.max()) / (y.max() - y.min())

    if zero_mean:
        y -= y.mean()
    
    return y

if __name__ == "__main__":
    if len(sys.argv) > 1:
        zero_mean = sys.argv[1] == "True"
        print(f"Received from shell script: zero_mean={zero_mean}")
    else:
        print("No argument passed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for num_peaks in [10,30,50,100]:
        zigzag_params = {
            'initial_condition': 'zigzag',  # Changed from GRF
            'dim': 1,
            'nx': 1024,
            'num_peaks': num_peaks,  # Number of peaks in zigzag
            'amplitude': 1.0,  # Maximum amplitude
            'bc': 'periodic',
            'seed': 3,
            'zero_mean': zero_mean
        }

        burgers_params = {
            'nu': 0.0005,
            'simulation_time': 1.0,
            "step": 0.001
        }

        dataset_params = {
            "num_record": 5
        }

        params = {
            "initial_condition": zigzag_params,  # Changed from GRF
            "dataset": dataset_params,
            "burgers": burgers_params,
        }

        shape = (zigzag_params["nx"],)

        result_list = []
        s = 1024
        sub = zigzag_params["nx"] // s

        current_file_path = Path(__file__).resolve().parent.parent
        model_path = f"{current_file_path}/saved_models/1D/modes16_width64_epochs500/zero/burgers_1d_FNO_model_trainedby_dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu{burgers_params['nu']}_t1.0_seed45.pth"
        solver_name = re.findall(r'solver=([a-zA-Z0-9]+)', model_path)[0]
        # print(solver_name)
        model = FNO1d(modes=16, width=64) 
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model = model.to(device)
        for param in model.parameters():
            param.requires_grad = False

        for i in tqdm(range(dataset_params['num_record']), desc="adversarial input samples"):
            a = generate_zigzag(
                shape=shape,
                num_peaks=zigzag_params['num_peaks'],
                amplitude=zigzag_params['amplitude'],
                seed=zigzag_params['seed'] + i,
                zero_mean=zigzag_params['zero_mean']
            )
            a = a + 0.1
            a_sub = a[::sub]
            a_torch = torch.from_numpy(a_sub).float().unsqueeze(0).unsqueeze(-1).to(device)
            output_of_input = np.squeeze(model(a_torch.clone()).cpu().detach().numpy())
            
            if solver_name == "exponax":
                solver = ExponaxBurgersSolver1D(s, nu=burgers_params['nu'], bc=zigzag_params['bc'])
            elif solver_name == "scipy":
                solver = SciPyBurgersSolver1D(s, nu=burgers_params['nu'], bc=zigzag_params['bc'])
            elif solver_name == "scipy_spectral":
                solver = SciPySpectralBurgersSolver1D(s, nu=burgers_params['nu'], bc=zigzag_params['bc'])
            elif solver_name == "phiflow":
                solver = PhiFlowBurgersSolver1D(s, nu=burgers_params['nu'], bc=zigzag_params['bc'])
            else:
                raise ValueError("Specified solver not inplemented")
            
            t_span = (0, burgers_params['simulation_time'])
            solution_a = np.squeeze(solver.solve(a.copy(), burgers_params['simulation_time'], t_span, burgers_params["step"])[t_span[1]][1])
            # print(solution_a.shape)
            
            result_dict = {
            "index":i,
            "a":a,
            "g(a)":solution_a,
            "G(a)":output_of_input,
            }

            result_list.append(result_dict)
            print("index: ", i)

        param_names = [
            "zigzag",  # Changed from gaussiandiffdistr
            f"peaks{zigzag_params['num_peaks']}",  # New parameter
            f"amp{zigzag_params['amplitude']}",  # New parameter
            f"dim={zigzag_params['dim']}d",
            f"solver={solver_name}",
            f"bc={zigzag_params['bc']}",
            f"nu={burgers_params['nu']}",
            f"t={burgers_params['simulation_time']}",
            f"seed={zigzag_params['seed']}",
            f"N={dataset_params['num_record']}",
            f"mean={zigzag_params['zero_mean']}"
        ]
        filename = "_".join(param_names) + ".pkl"
        current_file_path = Path(__file__).resolve().parent.parent
        folder = '1D' if zigzag_params['dim'] == 1 else '2D'
        save_path = current_file_path / "out_of_distr_test" / folder / f"result_mean={zigzag_params['zero_mean']}" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print("saved path:  ", save_path)
        with open(save_path, "wb") as f:
            pickle.dump(result_list, f)




