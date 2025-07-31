from pathlib import Path
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GRFs.generateGRFs import GRFGenerator
from solvers.burgers1d_solvers import *
import torch
import numpy as np

def generate_burgers_dataset(params, save_dir="datasets/1D/Burgers", solver_name="exponax"):
    grf_params = params["GRF"]
    dataset_params = params["dataset"]
    burgers_params = params["burgers"]
    np.random.seed(grf_params['seed'])
    
    num_record = dataset_params['num_record'] 
    nu = burgers_params['nu']
    t_final = burgers_params["simulation_time"]
    step = burgers_params["step"]
    t_eval = np.linspace(0, t_final, int((t_final/step)/10)+1)
    nx = grf_params['nx']
    dim = grf_params['dim']
    
    if dim == 1:
        shape = (grf_params['nx'],)
    else:
        raise ValueError("dim must be 1 ")
    
    solver_name = solver_name.lower()
    if solver_name == "scipy":
        solver = SciPyBurgersSolver1D(nx, nu=nu, bc=grf_params['bc'])
    elif solver_name == "scipy_spectral":
        solver = SciPySpectralBurgersSolver1D(nx, nu=nu, bc=grf_params['bc'])
    elif solver_name == "exponax":
        solver = ExponaxBurgersSolver1D(nx, nu=nu, bc=grf_params['bc'], xlim=(0,1))
    elif solver_name == "phiflow":
        solver = PhiFlowBurgersSolver1D(nx, nu=nu, bc=grf_params['bc'])
    else:
        raise ValueError("Specified solver not inplemented")
    
    input_tensors = []
    output_tensors = []
    
    for i in tqdm(range(num_record), desc="Generating samples"):
        # try:
            grf = GRFGenerator.generate_grf(
                shape=shape,
                kernel=grf_params['kernel'],
                kernel_params=grf_params['kernel_params'],
                bc=grf_params['bc'],
                seed=grf_params['seed'] + i,  # Ensure different GRFs for each sample
                zero_mean=grf_params['zero_mean']
            )

            # print("GRF generated sucessfully!")
            # print(f"GRF shape: {grf.shape}, type: {type(grf)}")
            # print("t_span: ", t_span)
            # print(grf.copy().shape)
            solution = solver.solve(grf.copy(), t_final, t_eval, step)
            final_solution = solution[t_eval[-1]][1]

            # print("Burgers solution generated sucessfully!")
            
            # Convert to PyTorch tensors and store
            input_tensors.append(torch.tensor(grf.copy(), dtype=torch.float32))
            if isinstance(final_solution, np.ndarray):
                output_tensors.append(torch.tensor(final_solution, dtype=torch.float32))
            else:
                output_tensors.append(torch.from_numpy(np.array(final_solution)).float())

        # except Exception as e:
        #     print(f"Error processing sample {i}: {e}")
    
    input_tensor = torch.stack(input_tensors)  # Shape: (N, spatial_dim1, spatial_dim2, ...)
    output_tensor = torch.stack(output_tensors)  # Shape: (N, spatial_dim1, spatial_dim2, ...)
    print({"input tensor shape":input_tensor.shape,"output tensor shape":output_tensor.shape})
    
    param_names = [
        f"dim{dim}d",
        f"nx{nx}",
        f"N{num_record}",
        f"solver={solver_name}",
        f"kernel={grf_params['kernel']}",
        *[f"{k}{v}" for k, v in grf_params['kernel_params'].items()],
        f"bc{grf_params['bc']}",
        f"nu{nu}",
        f"t{t_final}",
        f"seed{grf_params['seed']}",
    ]
    filename = "_".join(param_names) + ".pt"
    
    current_file_path = Path(__file__).resolve().parent.parent
    # folder1 = 'init_all_pos' if grf_params['zero_mean'] == False else 'init_neg_pos'
    # folder2 = '1d' if dim == 1 else '2d'
    # save_path = current_file_path / save_dir / folder1 / folder2 / filename
    save_path = current_file_path / save_dir / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(save_path)
    torch.save({'x': input_tensor, 'y': output_tensor, "t_final": t_eval[-1]}, save_path)
    print("x shape: ",input_tensor.shape,"y shape",output_tensor.shape)

