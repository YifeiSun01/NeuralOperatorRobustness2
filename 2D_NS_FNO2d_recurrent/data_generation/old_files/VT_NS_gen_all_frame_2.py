import numpy as np
import jax
import jax.numpy as jnp
import exponax as ex
import torch
import os
from tqdm import tqdm

def safe_getattr(obj, attr, default="N/A"):
    return getattr(obj, attr, default)

print("PyTorch version:", torch.__version__)
print("CUDA version used by PyTorch:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")
    for i in range(num_devices):
        print(f"\n--- Device {i} ---")
        print("Device name:", torch.cuda.get_device_name(i))
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  MultiProcessor count: {safe_getattr(props, 'multi_processor_count')}")
        print(f"  Memory Bus Width: {safe_getattr(props, 'memory_bus_width', 'N/A')} bits")
        print(f"  Memory Clock Rate: {safe_getattr(props, 'memory_clock_rate', 0) / 1e3:.0f} MHz")
        print(f"  GPU Clock Rate: {safe_getattr(props, 'clock_rate', 0) / 1e3:.0f} MHz")
        print(f"  Shared memory per block: {safe_getattr(props, 'shared_memory_per_block', 0) / 1024:.1f} KB")
        print(f"  Registers per block: {safe_getattr(props, 'regs_per_block', 'N/A')}")
        
        if hasattr(props, "max_threads_per_block"):
            print(f"  Max Threads per block: {props.max_threads_per_block}")
        if hasattr(props, "max_threads_dim"):
            print(f"  Max Threads dim: {props.max_threads_dim}")
        if hasattr(props, "max_grid_size"):
            print(f"  Max Grid size: {props.max_grid_size}")

    current_device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(current_device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(current_device) / (1024 ** 3)
    print(f"\n--- Memory usage on Device {current_device} ---")
    print(f"Allocated memory: {allocated:.2f} GB")
    print(f"Reserved memory:  {reserved:.2f} GB")
else:
    print("No CUDA device available.")



jax.config.update("jax_enable_x64", True)

# Set default device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ExponaxVTSolver2D:
    def __init__(self, nx, ny, Lx, Ly, nu, forcing=None, bc='periodic'):
        self.nx = nx
        self.Lx = Lx
        self.x = torch.linspace(-Lx/2, Lx/2, nx, device=device)
        self.dx = 2/nx
        self.nu = nu
        self.bc = bc

    def get_closest_solutions(self, sol_dict, t_eval):
        closest_solutions = {}
        sorted_keys = sorted(sol_dict.keys())

        for t in t_eval:
            closest_key = min(sorted_keys, key=lambda k: abs(k - t))
            closest_solutions[t] = (closest_key, sol_dict[closest_key])
        
        return closest_solutions

    def solve(self, u0, t_final, t_eval, step):
        full_ic = jnp.expand_dims(jnp.array(u0, dtype=jnp.float32), axis=0)  # use float32 to save memory

        stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
            2, self.Lx, self.nx, step,
            diffusivity=self.nu, order=4,
            num_circle_points=128,
            dealiasing_fraction=0.5
        )

        # Make set of t_eval for fast lookup
        t_eval_set = set(t_eval)
        sol_dict = {}
        
        t = 0.0
        u = full_ic
        i = 0

        while t <= t_final + 1e-8:
            if round(t, 8) in t_eval_set:  # Only save if needed
                u_cpu = np.array(u[0, ...].T.block_until_ready())  # move to host early
                sol_dict[round(t, 8)] = u_cpu

            u = stepper(u)
            t += step
            i += 1

        return self.get_closest_solutions(sol_dict, t_eval)

def generate_vorticity_transport_data(time_list, grf, nu, ntimepoints):
    # 1. 初始化 solver
    exponaxsolver = ExponaxVTSolver2D(
        nx=grf.shape[0], ny=grf.shape[1], 
        Lx=1, Ly=1, nu=nu
    )

    # 2. 转换输入为 numpy
    if hasattr(grf, '__jax_array__') or isinstance(grf, jnp.ndarray):
        grf_numpy = np.array(grf)
    elif isinstance(grf, np.ndarray):
        grf_numpy = grf
    elif isinstance(grf, torch.Tensor):
        grf_numpy = grf.cpu().numpy()
    else:
        raise TypeError(f"Unsupported input type: {type(grf)}")

    # 3. 对齐方向（Zongyi代码习惯）
    grf_numpy = np.rot90(np.flipud(grf_numpy), 3)

    # 4. 只保存 selected_times（稀疏时间点）
    selected_indices = np.linspace(0, len(time_list) - 1, ntimepoints, dtype=int)
    selected_times = time_list[selected_indices]

    # 5. 解 NS 方程，自动按需保留时间点，节省显存
    exponax_solutions = exponaxsolver.solve(
        grf_numpy,
        t_final=float(time_list[-1]),
        t_eval=[0.0] + list(selected_times),
        step=0.0001
    )

    # 6. x 是初始时刻
    x_numpy = exponax_solutions[0.0][1]
    x = torch.from_numpy(x_numpy).float().to(device)

    # 7. y 是 selected_times 的解
    y_numpy = np.stack([exponax_solutions[float(t)][1] for t in selected_times])
    y = torch.from_numpy(y_numpy).float().to(device)

    return x, y, selected_times

def spectral_upsample(field, target_size=256):
    *batch_dims, H, W = field.shape
    assert target_size >= H and target_size >= W, "目标尺寸必须大于输入尺寸"
    freq = torch.fft.fft2(field, norm='ortho')
    freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))
    pad_H = (target_size - H) // 2
    pad_W = (target_size - W) // 2
    new_freq_shifted = torch.zeros(
        *batch_dims, target_size, target_size, 
        dtype=freq_shifted.dtype, device=freq_shifted.device
    )
    start_H = pad_H
    start_W = pad_W
    new_freq_shifted[..., start_H:start_H+H, start_W:start_W+W] = freq_shifted
    new_freq = torch.fft.ifftshift(new_freq_shifted, dim=(-2, -1))
    upsampled = torch.fft.ifft2(new_freq, norm='ortho')
    return (target_size / H) * upsampled.real

class DatasetGenerator:
    def __init__(self, base_seed=0):
        self.global_seed_counter = base_seed
        self.dataset_counter = 0
    
    def _generate_filename(self, params):
        return (f"dim{params['dim']}d_nx{params['nx']}_N{params['nsamples']}_"
                f"solver=exponax_"
                f"nu{params['nu_ns']:.3f}_"
                f"t{params['tfinal']:.1f}_"
                "test_"
                f"all_frames.pt")
    
    def generate_dataset(self, nu_ns, tfinal, nsamples, save_dir=None):
        dataset_x = []
        dataset_y = []
        start_seed = self.global_seed_counter
        zongyi_dataset = torch.load('../2D_NS_old/2D_NS_Zongyi_Li/recurrent/datasets/2D/NS/NS_data_zongyi_test_all_frame.pt', weights_only=False)

        # Add tqdm progress bar for sample generation
        for idx in tqdm(range(nsamples), desc="Generating samples", unit="sample"):
            
            # Generate on GPU
            grf = zongyi_dataset["x"][idx]
            size = 256
            grf = spectral_upsample(grf, size)
            
            time_list = np.linspace(0, tfinal, 5001)
            ntimepoints = tfinal + 1
            x, y, selected_times = generate_vorticity_transport_data(
                time_list,
                grf,
                nu_ns,
                ntimepoints
            )
            
            dataset_x.append(x)
            dataset_y.append(y)
            self.global_seed_counter += 1
        
        # Stack on GPU with progress indication
        with tqdm(total=2, desc="Stacking tensors") as pbar:
            tensor_x = torch.stack(dataset_x).to(device)
            pbar.update(1)
            tensor_y = torch.stack(dataset_y).to(device)
            pbar.update(1)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_params = {
                'dim': 2,
                'nx': grf.shape[0],
                'nsamples': nsamples,
                'nu_ns': nu_ns,
                'tfinal': tfinal,
            }
            
            filename = self._generate_filename(file_params)
            save_path = os.path.join(save_dir, filename)
            
            # Add progress bar for saving
            with tqdm(total=1, desc="Saving dataset") as pbar:
                torch.save({
                    'x': tensor_x.cpu(),
                    'y': tensor_y.permute(0,2,3,1).cpu(),
                    'metadata': {
                        'nu_ns': nu_ns,
                        'tfinal': tfinal,
                        'nsamples': nsamples,
                        "times": selected_times
                    }
                }, save_path)
                pbar.update(1)
            
            print(f"\nDataset saved to {save_path}")
            return save_path
        else:
            return tensor_x, tensor_y

if __name__ == "__main__":
    # Usage Example
    generator = DatasetGenerator(base_seed=0)

    print("Generating first dataset:")
    path1 = generator.generate_dataset(
        nu_ns=1e-5,
        tfinal=20,
        nsamples=5,
        save_dir="./datasets"
    )




