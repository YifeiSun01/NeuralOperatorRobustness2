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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        zero_mean = sys.argv[1] == "True"
        print(f"Received from shell script: zero_mean={zero_mean}")
    else:
        print("No argument passed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cl in [0.005, 0.03, 0.2]:
        for nu in [0.5, 1.5, 2.5]:
            grf_params = {
                'dim': 1,
                'nx': 1024,
                'kernel': 'matern',
                'kernel_params': {'correlation_length': cl,"nu":nu},
                'bc': 'periodic',
                'seed': 3,
                'zero_mean':zero_mean
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
                "GRF": grf_params,
                "dataset": dataset_params,
                "burgers": burgers_params,
            }

            shape = (grf_params["nx"],)

            result_list = []
            s = 1024
            sub = grf_params["nx"] // s
            
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
                    # try:
                        a = GRFGenerator.generate_grf(
                                        shape=shape,
                                        kernel=grf_params['kernel'],
                                        kernel_params=grf_params['kernel_params'],
                                        bc=grf_params['bc'],
                                        seed=grf_params['seed'] + i,  # Ensure different GRFs for each sample
                                        zero_mean=grf_params['zero_mean']
                                    )
                        a = a + 0.1
                        a_sub = a[::sub]
                                
                        a_torch = torch.from_numpy(a_sub).float().unsqueeze(0).unsqueeze(-1).to(device)
                        output_of_input = np.squeeze(model(a_torch.clone()).cpu().detach().numpy())
                        
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
                    # except:
                    #     pass

            param_names = [
                    "materndiffdistr",
                    f"cl{grf_params['kernel_params']['correlation_length']}",
                    f"nu{grf_params['kernel_params']['nu']}",
                    f"dim={grf_params['dim']}d",
                    f"solver={solver_name}",
                    f"kernel={grf_params['kernel']}",
                    *[f"{k}={v}" for k, v in grf_params['kernel_params'].items()],
                    f"bc={grf_params['bc']}",
                    f"nu={burgers_params['nu']}",
                    f"t={burgers_params['simulation_time']}",
                    f"seed={grf_params['seed']}",
                    f"N={dataset_params['num_record']}",
                    f"mean={grf_params['zero_mean']}"
                ]
            filename = "_".join(param_names) + ".pkl"
            current_file_path = Path(__file__).resolve().parent.parent
            folder = '1D' if grf_params['dim'] == 1 else '2D'
            save_path = current_file_path / "out_of_distr_test" / folder / f"result_mean={grf_params['zero_mean']}" / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print("saved path:  ", save_path)
            with open(save_path, "wb") as f:
                pickle.dump(result_list, f)