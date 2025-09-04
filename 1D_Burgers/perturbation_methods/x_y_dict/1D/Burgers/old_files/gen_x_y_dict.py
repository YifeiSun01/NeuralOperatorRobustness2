from pathlib import Path
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from GRFs.generateGRFs import GRFGenerator
from solvers.burgers1d_solvers import *
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的是 {'GPU (CUDA)' if device.type == 'cuda' else 'CPU'} 进行计算")
print(f"设备名称: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

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
        solver = ExponaxBurgersSolver1D(nx, nu=nu, bc=grf_params['bc'])
    elif solver_name == "phiflow":
        solver = PhiFlowBurgersSolver1D(nx, nu=nu, bc=grf_params['bc'])
    else:
        raise ValueError("Specified solver not inplemented")
    
    input_tensors = []
    output_tensors = []
    
    for i in tqdm(range(num_record), desc="Generating samples"):
        try:
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

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
    
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

    current_file_path = Path(__file__).resolve().parent
    folder = '1D' if dim == 1 else '2D'
    save_path = current_file_path / folder / save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    filename = "_".join(param_names) + ".pt"
    input_filename = f"inputs_{filename}"
    output_filename = f"outputs_{filename}"
    
    torch.save(input_tensor, save_path / input_filename)
    torch.save(output_tensor, save_path / output_filename)
    print(f"Saved input tensor to {save_path / input_filename}")
    print(f"Saved output tensor to {save_path / output_filename}")

grf_params = {
    'dim': 1,
    'nx': 1024,
    'kernel': 'gaussian',
    'kernel_params': {'correlation_length': 0.03},
    'bc': 'periodic',
    'seed': 1000045,
    'zero_mean':False
}

burger_params = {
    'nu': 0.01,
    'simulation_time': 1.0,
    "step": 0.001
}

dataset_params = {
    "num_record": 20000
}

params = {
    "GRF": grf_params,
    "dataset": dataset_params,
    "burgers": burger_params,
}

for solver_name in ["exponax"]: 
    print("solver: ", solver_name)
    print(params)
    generate_burgers_dataset(params, save_dir="Burgers", solver_name=solver_name)