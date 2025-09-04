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
    t_eval = np.linspace(0, t_final, int((t_final/step)/10) + 1)
    nx = grf_params['nx']
    dim = grf_params['dim']

    if dim == 1:
        sample_shape = (nx,)
    else:
        raise ValueError("dim must be 1 ")

    # 设备：尽量用 GPU；没有就回退 CPU（保存格式不变）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的是 {'GPU (CUDA)' if device.type == 'cuda' else 'CPU'} 进行计算")
    if device.type == 'cuda':
        print(f"设备名称: {torch.cuda.get_device_name(0)}")

    # 选择求解器（求解器内部是否用 GPU 取决于它自己的实现/配置）
    sname = solver_name.lower()
    if sname == "scipy":
        solver = SciPyBurgersSolver1D(nx, nu=nu, bc=grf_params['bc'])
    elif sname == "scipy_spectral":
        solver = SciPySpectralBurgersSolver1D(nx, nu=nu, bc=grf_params['bc'])
    elif sname == "exponax":
        # solver = ExponaxBurgersSolver1D(nx, nu=nu, bc=grf_params['bc'], xlim=(0,1))
        solver = ExponaxBurgersSolver1D(nx, nu=nu, bc=grf_params['bc'])
    elif sname == "phiflow":
        solver = PhiFlowBurgersSolver1D(nx, nu=nu, bc=grf_params['bc'])
    else:
        raise ValueError("Specified solver not inplemented")

    # 预分配 GPU 缓冲（减少反复分配/拷贝）
    x_buf = torch.empty((num_record, *sample_shape), dtype=torch.float32, device=device)
    y_buf = torch.empty_like(x_buf)

    for i in tqdm(range(num_record), desc="Generating samples"):
        # 生成 GRF（NumPy）
        grf = GRFGenerator.generate_grf(
            shape=sample_shape,
            kernel=grf_params['kernel'],
            kernel_params=grf_params['kernel_params'],
            bc=grf_params['bc'],
            seed=grf_params['seed'] + i,
            zero_mean=grf_params['zero_mean'],
        )

        # 求解（返回通常是 NumPy；是否GPU由求解器决定）
        solution = solver.solve(grf.copy(), t_final, t_eval, step)
        final_solution = solution[t_eval[-1]][1]

        # 转为 GPU Tensor 并写入预分配缓冲
        x_tensor = torch.as_tensor(grf, dtype=torch.float32, device=device)
        if isinstance(final_solution, np.ndarray):
            y_tensor = torch.as_tensor(final_solution, dtype=torch.float32, device=device)
        else:
            y_tensor = torch.as_tensor(np.array(final_solution), dtype=torch.float32, device=device)

        # 拷贝到缓冲（避免重新分配）
        x_buf[i].copy_(x_tensor)
        y_buf[i].copy_(y_tensor)

    # 为保持“保存形式不变”，在 CPU 上保存
    input_tensor = x_buf.detach().cpu()    # Shape: (N, nx)
    output_tensor = y_buf.detach().cpu()   # Shape: (N, nx)
    print({"input tensor shape": input_tensor.shape, "output tensor shape": output_tensor.shape})

    # 文件名与保存路径保持不变
    param_names = [
        f"dim{dim}d",
        f"nx{nx}",
        f"N{num_record}",
        f"solver={sname}",
        f"kernel={grf_params['kernel']}",
        *[f"{k}{v}" for k, v in grf_params['kernel_params'].items()],
        f"bc{grf_params['bc']}",
        f"nu{nu}",
        f"t{t_final}",
        f"seed{grf_params['seed']}",
    ]
    filename = "_".join(param_names) + ".pt"

    current_file_path = Path(__file__).resolve().parent.parent
    save_path = current_file_path / save_dir / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(save_path)

    torch.save({'x': input_tensor, 'y': output_tensor, "t_final": t_eval[-1]}, save_path)
    print("x shape: ", input_tensor.shape, "y shape", output_tensor.shape)