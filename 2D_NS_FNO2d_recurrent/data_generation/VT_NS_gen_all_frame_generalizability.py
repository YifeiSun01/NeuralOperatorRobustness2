import numpy as np
import jax
import jax.numpy as jnp
import exponax as ex
import torch
import os
from tqdm import tqdm
import jaxlib
from jax.lib import xla_bridge
from torch.utils import dlpack as torch_dlpack
import jax.dlpack as jax_dlpack
import math

def print_memory_stats():
    try:
        mem_stats = xla_bridge.get_backend().memory_stats()
        print(f"\n--- JAX Memory Usage ---")
        print(f"Used: {mem_stats['bytes_in_use']/1e9:.1f} GB")
        print(f"Total: {mem_stats['bytes_limit']/1e9:.1f} GB")
        print(f"Peak: {mem_stats['peak_bytes_in_use']/1e9:.1f} GB")
    except Exception as e:
        print(f"\nMemory stats unavailable: {str(e)}")

# 在初始化完成后调用一次
# print_memory_stats()

def safe_getattr(obj, attr, default="N/A"):
    return getattr(obj, attr, default)

print("\n--- PyTorch Info ---")
print("PyTorch version:", torch.__version__)
print("CUDA version used by PyTorch:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())

print("\n--- JAX Info ---")
print("JAX version:", jax.__version__)
print("JAXlib version:", jaxlib.__version__)
print("JAX default backend:", jax.default_backend())
print("JAX devices:", jax.devices())

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")
    for i in range(num_devices):
        print(f"\n--- Device {i} ---")
        print("Device name:", torch.cuda.get_device_name(i))
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / (1024**3):.2f} GiB")
        print(f"  Total memory: {props.total_memory / (1000**3):.2f} GB")
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



# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_prng_impl", "unsafe_rbg")
jax.config.update("jax_debug_nans", False) 

# Set default device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def torch_to_jax(tensor):
    """将 PyTorch 张量安全转换为 JAX 数组"""
    return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor.contiguous()))

def jax_to_torch(jax_array):
    """将JAX数组转换为PyTorch张量而不复制数据"""
    return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(jax_array))

class PeriodicGRF:
    def __init__(self, dim, size, kernel="matern", device="cpu", **kernel_params):
        """
        dim: 1 or 2
        size: grid size per dim
        kernel: kernel name string
        kernel_params: kernel-specific params
        """
        self.dim = dim
        self.size = size
        self.device = device
        self.kernel = kernel.lower()
        self.params = kernel_params

        freqs_1d = torch.fft.fftfreq(size, d=1.0/size).to(device)

        if dim == 1:
            self.k = freqs_1d
        elif dim == 2:
            kx, ky = torch.meshgrid(freqs_1d, freqs_1d, indexing='ij')
            self.k = torch.sqrt(kx**2 + ky**2)
        else:
            raise NotImplementedError("Only dim=1 or 2 supported.")

        self.sqrt_spec_density = self._compute_sqrt_spectral_density()

    def _compute_sqrt_spectral_density(self):
        k = self.k
        p = self.params
        kernel = self.kernel

        if kernel == "matern":
            # Matern: (4pi^2 k^2 + tau^2)^(-alpha/2)
            alpha = p.get("alpha", 2.5)
            tau = p.get("tau", 7.0)
            sigma = p.get("sigma", tau ** (0.5*(2*alpha - self.dim)))
            spec = (4 * (math.pi ** 2) * k ** 2 + tau ** 2) ** (-alpha / 2)
            spec[0] = 0.0
            spec = (self.size**self.dim) * math.sqrt(2.0) *  sigma * spec

        elif kernel == "rbf" or kernel == "squared_exponential":
            length_scale = p.get("length_scale", 0.1)
            variance = p.get("variance", 1.0)
            spec = variance * torch.exp(-2 * (math.pi ** 2) * (length_scale ** 2) * k ** 2)
            spec[0] = 0.0

        elif kernel == "exponential":
            length_scale = p.get("length_scale", 0.1)
            variance = p.get("variance", 1.0)
            spec = variance / (1 + (2 * math.pi * length_scale * k) ** 2)
            spec[0] = 0.0

        elif kernel == "rq" or kernel == "rational_quadratic":
            length_scale = p.get("length_scale", 0.1)
            alpha = p.get("alpha", 1.0)
            variance = p.get("variance", 1.0)
            spec = variance * (1 + (k ** 2) / (2 * alpha * length_scale ** 2)) ** (-alpha)
            spec[0] = 0.0

        elif kernel == "periodic":
            length_scale = p.get("length_scale", 0.1)
            period = p.get("period", 1.0)*self.size

            # 生成坐标网格，从 -size//2 到 +size//2（保证中心点在0）
            coords = torch.arange(-(self.size//2), self.size//2, device=self.device, dtype=torch.float32)
            xx, yy = torch.meshgrid(coords, coords, indexing='ij')

            # 计算每个点到原点的距离（欧氏距离）
            dist = torch.sqrt(xx**2 + yy**2)

            # 计算Periodic ACF矩阵
            K_time = torch.exp(-2.0 * (torch.sin(math.pi * dist / period) ** 2) / (length_scale ** 2))

            # 计算功率谱密度PSD（多维FFT）
            spec = torch.fft.fftn(K_time)

            # PSD是复数，取实部或模长作为频谱（通常实部即可）
            spec = torch.real(spec)

        elif kernel == "linear":
            # 线性核非平稳，频谱不存在简单闭式，暂不支持
            raise NotImplementedError("Linear kernel not supported in spectral form.")

        elif kernel == "spectral_mixture":
            # 频谱为多个高斯峰叠加
            # params: weights, means, variances — 都是列表或tensor
            weights = torch.tensor(p.get("weights", [1.0]), device=self.device)
            means = torch.tensor(p.get("means", [0.0]), device=self.device)
            variances = torch.tensor(p.get("variances", [0.1]), device=self.device)
            spec = torch.zeros_like(k)
            for w, m, v in zip(weights, means, variances):
                spec += w * torch.exp(-0.5 * ((k - m) ** 2) / v)
            spec[0] = 0.0

        elif kernel == "cauchy":
            # Cauchy kernel ~ (1 + (k/l)^2)^(-alpha)
            length_scale = p.get("length_scale", 0.1)
            alpha = p.get("alpha", 1.0)
            variance = p.get("variance", 1.0)
            spec = variance * (1 + (k / length_scale) ** 2) ** (-alpha)
            spec[0] = 0.0

        elif kernel == "powered_exponential":
            # 广义指数核，spec ~ exp(-(k*l)^gamma), gamma in (0,2]
            length_scale = p.get("length_scale", 0.1)
            gamma = p.get("gamma", 1.5)
            variance = p.get("variance", 1.0)
            spec = variance * torch.exp(-(k * length_scale) ** gamma)
            spec[0] = 0.0

        elif kernel == "wave":
            # 波动核示意，谱为双峰对应振荡频率
            freq0 = p.get("freq", 5.0)
            variance = p.get("variance", 1.0)
            spec = variance * (torch.exp(-0.5 * ((k - freq0) ** 2) / 0.1) +
                               torch.exp(-0.5 * ((k + freq0) ** 2) / 0.1))
            spec[0] = 0.0

        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # return torch.sqrt(spec)
        return spec

    def sample(self, N):
        shape = (N,) + self.k.shape
        coeff = torch.randn(shape, dtype=torch.cfloat, device=self.device)
        coeff = coeff * self.sqrt_spec_density
        if self.dim == 1:
            field = torch.fft.ifft(coeff, dim=-1).real
        else:
            field = torch.fft.ifftn(coeff, dim=(-2, -1)).real
        return field

def scale_to_range(tensor, min_val=-0.4, max_val=0.4):
    t_min = tensor.amin(dim=(1,2), keepdim=True)
    t_max = tensor.amax(dim=(1,2), keepdim=True)
    # Avoid division by zero
    scaled = (tensor - t_min) / (t_max - t_min + 1e-8)  
    scaled = scaled * (max_val - min_val) + min_val
    return scaled

class LogGaussianRF:
    def __init__(self, dim, size, kernel="matern", device="cpu", exp_factor=1.0, **kernel_params):
        """
        dim: 维度
        size: 每维的网格点数
        kernel: 核的类型
        kernel_params: 传递给 PeriodicGRF 的核参数
        exp_factor: 指数放大因子
        """
        self.dim = dim
        self.size = size
        self.kernel = kernel
        self.device = device
        self.exp_factor = exp_factor
        self.kernel_params = kernel_params

        # 内部创建 PeriodicGRF 实例
        self.grf = PeriodicGRF(
            dim=dim,
            size=size,
            kernel=kernel,
            device=device,
            **kernel_params
        )

    def sample(self, N):
        """
        返回 log GRF 样本, shape: (N, size, size)
        """
        sample_grf = self.grf.sample(N)  # shape: (N, size, size)

        # 缩放到 0~1
        scaled = scale_to_range(sample_grf)

        # 乘指数因子后取 exp
        sample_log_grf = torch.exp(self.exp_factor * scaled)

        # 减去每个 sample 的均值（广播维度）
        mean_per_sample = sample_log_grf.mean(dim=(-2, -1), keepdim=True)
        sample_log_grf = sample_log_grf - mean_per_sample

        return sample_log_grf

class NegativeLogGaussianRF:
    def __init__(self, dim, size, kernel="matern", device="cpu", exp_factor=1.0, **kernel_params):
        """
        dim, size, kernel, kernel_params, exp_factor 与 LogGaussianRF 一样。
        """
        self.log_grf = LogGaussianRF(
            dim=dim,
            size=size,
            kernel=kernel,
            device=device,
            exp_factor=exp_factor,
            **kernel_params
        )

    def sample(self, N):
        """
        返回负 log GRF 样本，shape: (N, size, size)
        """
        log_samples = self.log_grf.sample(N)
        return -log_samples

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
        # 如果输入是PyTorch张量，直接转换为JAX数组
        if isinstance(u0, torch.Tensor):
            u0_jax = torch_to_jax(u0)
        else:
            u0_jax = jnp.array(u0, dtype=jnp.float32)
            
        full_ic = jnp.expand_dims(u0_jax, axis=0)
        
        # 初始化求解器
        slower_NS_stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
            2, self.Lx, self.nx, step, 
            diffusivity=self.nu, order=4,
            num_circle_points=16, 
            dealiasing_fraction=2/3
        )
        
        longer_rollout_NS_stepper = ex.rollout(
            slower_NS_stepper, int(t_final/step), include_init=True
        )

        # 运行模拟
        longer_trajectory = longer_rollout_NS_stepper(full_ic)
        
        # 直接转换结果到PyTorch张量，避免通过numpy
        sol_dict = {}
        for i in range(longer_trajectory.shape[0]):
            # 直接从JAX转换到PyTorch，保持在GPU上
            frame = longer_trajectory[i, 0, ...].T
            frame.block_until_ready()  # 确保计算完成
            sol_dict[i*step] = jax_to_torch(frame)
        
        sol_dict = self.get_closest_solutions(sol_dict, t_eval)
        
        return sol_dict

def check_solution_stability(solution_tensor):
    """
    检查解的数值稳定性
    返回True如果解是稳定的（没有NaN/Inf/过大值），否则返回False
    """
    if solution_tensor is None:
        return False
    
    # 检查NaN和Inf
    if torch.isnan(solution_tensor).any() or torch.isinf(solution_tensor).any():
        return False

    """
    # 检查值范围是否合理（这里假设涡度值在-100到100之间是合理的）
    max_val = torch.max(torch.abs(solution_tensor))
    if max_val > 1000:  # 值过大可能表示数值不稳定
        return False
    
    # 检查梯度变化是否过大（可能导致爆炸）
    grad_x, grad_y = torch.gradient(solution_tensor, dim=(0, 1))
    max_grad = max(torch.max(torch.abs(grad_x)), torch.max(torch.abs(grad_y)))
    if max_grad > 100:  # 梯度过大可能表示数值不稳定
        return False
    """

    return True

def generate_vorticity_transport_data(time_list, grf, nu, ntimepoints):
    # 确保输入在GPU上
    if not isinstance(grf, torch.Tensor):
        grf = torch.tensor(grf, device=device)
    else:
        grf = grf.to(device)
    
    # 使用PyTorch操作替代numpy操作
    grf = torch.flip(grf, [0])  # 替代np.flipud
    grf = torch.rot90(grf, k=3, dims=[0, 1])  # 替代np.rot90(..., 3)
    
    # 初始化求解器
    exponaxsolver = ExponaxVTSolver2D(
        nx=grf.shape[0], ny=grf.shape[1], 
        Lx=1, Ly=1, nu=nu
    )
    
    # 尝试不同的时间步长
    step_options = [0.01, 0.005, 0.001, 0.0005, 0.0001]  # 从大到小尝试
    solution_attempts = 0
    valid_solution = False
    
    for step_size in step_options:
        solution_attempts += 1
        try:
            # 获取解
            exponax_solutions = exponaxsolver.solve(
                grf,
                t_final=time_list[-1],
                t_eval=[0] + time_list,
                step=step_size
            )
            
            # 提取所有解用于稳定性检查
            all_solutions = torch.stack([sol[1] for sol in exponax_solutions.values()])
            
            # 检查解的稳定性
            if check_solution_stability(all_solutions):
                valid_solution = True
                print(f"Step size {step_size} produced stable solution.")
                break
            else:
                print(f"Step size {step_size} produced unstable solution. Trying smaller step.")
        except Exception as e:
            print(f"Error with step size {step_size}: {str(e)}. Trying smaller step.")
    
    if not valid_solution:
        raise RuntimeError(f"Failed to find stable solution after {solution_attempts} attempts with step sizes: {step_options}")
    
    selected_indices = np.linspace(0, len(time_list)-1, ntimepoints, dtype=int)
    selected_times = time_list[selected_indices]
    
    # 结果已经在GPU上
    x = exponax_solutions[0][1]
    y = torch.stack([exponax_solutions[selected_time][1] for selected_time in selected_times])
    
    return x, y, selected_times

def spectral_upsample(field, target_size=256):
    *batch_dims, H, W = field.shape
    assert target_size >= H and target_size >= W, "目标尺寸必须大于输入尺寸"
    
    # 确保输入在GPU上
    field = field.to(device)
    
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
    def __init__(self, base_seed=0, train_test="train"):
        self.global_seed_counter = base_seed
        self.dataset_counter = 0
        self.train_test = train_test
        self.grfs = []  # Store GRF configurations here
    
    def add_grf_config(self, grf_type="GRF", kernel="rbf", vmax=0.4, vmin=-0.4, **kernel_params):
        """Add a GRF configuration to be generated.
        
        Args:
            grf_type: "GRF" (PeriodicGRF), "LogGRF", or "NegLogGRF"
            kernel: kernel type ("rbf", "matern", "periodic", etc.)
            kernel_params: dictionary of kernel-specific parameters
        """
        config = {
            'grf_type': grf_type,
            'kernel': kernel,
            'params': kernel_params,
            "vmax": vmax,
            "vmin": vmin,
        }
        self.grfs.append(config)
    
    def _generate_initial_conditions(self, N, config_idx=0):
        """Generate N initial conditions using the specified GRF configuration."""
        if not self.grfs:
            raise ValueError("No GRF configurations added. Call add_grf_config() first.")
        
        config = self.grfs[config_idx]
        grf_type = config['grf_type']
        kernel = config['kernel']
        params = config['params']
        vmax = config['vmax']
        vmin = config['vmin']
        
        # Generate the base GRF
        size = 256
        if grf_type == "GRF":
            grf = PeriodicGRF(dim=2, size=size, kernel=kernel, device=device, **params)
            samples = grf.sample(N)
            scaled_samples = scale_to_range(samples, min_val=vmin, max_val=vmax)
        elif grf_type == "LogGRF":
            exp_factor = params.pop('exp_factor', 1.0)
            grf = LogGaussianRF(dim=2, size=size, kernel=kernel, device=device, exp_factor=exp_factor, **params)
            samples = grf.sample(N)
            scaled_samples = scale_to_range(samples, min_val=vmin, max_val=vmax)
        elif grf_type == "NegLogGRF":
            exp_factor = params.pop('exp_factor', 1.0)
            grf = NegativeLogGaussianRF(dim=2, size=size, kernel=kernel, device=device, exp_factor=exp_factor, **params)
            samples = grf.sample(N)
            scaled_samples = scale_to_range(samples, min_val=vmin, max_val=vmax)
        else:
            raise ValueError(f"Unknown GRF type: {grf_type}")

        return scaled_samples
    
    def _generate_filename(self, params, grf_config, batch_idx=None):
        """Generate filename with all parameters included."""
        fname = (f"dim{params['dim']}d_nx{params['nx']}_N{params['nsamples']}_"
                 f"solver=exponax_"
                 f"nu{params['nu_ns']:.3f}_"
                 f"t{params['tfinal']:.1f}_"
                 f"{self.train_test}_"
                 f"ntimepoints{self.ntimepoints}_")
        
        # Add GRF parameters to filename
        fname += f"{grf_config['grf_type']}_{grf_config['kernel']}_"
        for k, v in grf_config['params'].items():
            if isinstance(v, float):
                fname += f"{k}{v:.3f}_"
            else:
                fname += f"{k}{v}_"

        fname += f"vmax{grf_config['vmax']}_vmin{grf_config['vmin']}_"
        
        if batch_idx is not None:
            fname += f"batch{batch_idx}_"
        fname += "all_frames.pt"
        return fname
    
    def generate_dataset(self, nu_ns, tfinal, nsamples, ntimepoints, 
                        batch_size=10, save_dir="./datasets", grf_config_idx=0):
        """Generate dataset with specified parameters."""
        self.ntimepoints = ntimepoints
        total_batches = (nsamples + batch_size - 1) // batch_size
        grf_config = self.grfs[grf_config_idx]

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, nsamples)
            num_samples_this_batch = end_idx - start_idx

            file_params = {
                'dim': 2,
                'nx': 256,
                'nsamples': num_samples_this_batch,
                'nu_ns': nu_ns,
                'tfinal': tfinal,
                "ntimepoints": ntimepoints
            }

            filename = self._generate_filename(file_params, grf_config, batch_idx=batch_idx)
            save_path = os.path.join(save_dir, filename)

            if os.path.exists(save_path):
                print(f"Batch {batch_idx} already exists, skipping...")
                continue

            print(f"\nStarting batch {batch_idx + 1}/{total_batches}...")
            dataset_x, dataset_y = [], []
            failed_indices = []
            selected_times = None

            # Generate initial conditions for this batch
            initial_conditions = self._generate_initial_conditions(num_samples_this_batch, grf_config_idx)

            for i in tqdm(range(num_samples_this_batch), desc=f"Generating batch {batch_idx}", unit="sample"):
                try:
                    grf = initial_conditions[i]
                    time_list = np.linspace(0, tfinal, 5001)

                    x, y, selected_times = generate_vorticity_transport_data(time_list, grf, nu_ns, ntimepoints)

                    dataset_x.append(x)
                    dataset_y.append(y)

                    if i % 10 == 0:
                        jax.clear_caches()

                except Exception as e:
                    print(f"[Warning] Sample {i} failed: {str(e)}")
                    failed_indices.append(i)
                    continue

            if not dataset_x:
                print(f"[Error] All samples in batch {batch_idx} failed. Skipping save.")
                continue

            tensor_x = torch.stack(dataset_x)
            tensor_y = torch.stack(dataset_y)

            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'x': tensor_x.cpu(),
                'y': tensor_y.permute(0, 2, 3, 1).cpu(),
                'metadata': {
                    'nu_ns': nu_ns,
                    'tfinal': tfinal,
                    'nsamples': len(dataset_x),
                    'batch_idx': batch_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'times': selected_times,
                    'grf_config': grf_config  # Include all GRF parameters in metadata
                }
            }, save_path)
            print(f"[Saved] Batch {batch_idx} saved to {save_path}")

            if failed_indices:
                fail_log_path = os.path.join(save_dir, f"failed_indices_batch{batch_idx}.txt")
                with open(fail_log_path, "w") as f:
                    for idx in failed_indices:
                        f.write(f"{idx}\n")
                print(f"[Info] Failed samples in batch {batch_idx} logged at {fail_log_path}")

if __name__ == "__main__":
    # Common parameters
    train_test_samples = [("test", 10)]
    tfinal = 20
    ntimepoints = tfinal + 1
    
    # Define the vmin/vmax ranges to explore
    scaling_ranges = [
        (-0.4, 0.4),  # Default range
        (-1, 0),  # Wider range
        (0, 1),  # Narrower range
        (-1, 1),  # Asymmetric range
        (1, 1.5),
        (-1.5, -1),
    ]

    save_directory = "./datasets/exponax_datasets/t20/generalizability"

    # RBF Kernel configurations
    print("===========RBF kernel GRF=============")
    for train_test, nsamples in train_test_samples:
        for length_scale in [0.02, 0.05, 0.1, 0.2]:
            for variance in [0.1, 0.5, 1.0, 2.0, 5]:
                for exp_factor in [2.5]:
                    for vmin, vmax in scaling_ranges:
                        print(f"\n\n\nGenerating {train_test} set with RBF kernel: length_scale={length_scale}, variance={variance}, exp_factor={exp_factor}, vmin={vmin}, vmax={vmax}")
                        
                        generator = DatasetGenerator(base_seed=0, train_test=train_test)
                        
                        # Add standard GRF
                        generator.add_grf_config(
                            grf_type="GRF",
                            kernel="rbf",
                            vmin=vmin,
                            vmax=vmax,
                            length_scale=length_scale,
                            variance=variance
                        )
                        
                        # Add LogGRF
                        generator.add_grf_config(
                            grf_type="LogGRF",
                            kernel="rbf",
                            vmin=vmin,
                            vmax=vmax,
                            length_scale=length_scale,
                            variance=variance,
                            exp_factor=exp_factor
                        )
                        
                        # Add NegLogGRF
                        generator.add_grf_config(
                            grf_type="NegLogGRF",
                            kernel="rbf",
                            vmin=vmin,
                            vmax=vmax,
                            length_scale=length_scale,
                            variance=variance,
                            exp_factor=exp_factor
                        )
                        
                        # Generate datasets for each configuration
                        for i in range(len(generator.grfs)):
                            generator.generate_dataset(
                                nu_ns=1e-5,
                                tfinal=tfinal,
                                nsamples=nsamples,
                                batch_size=nsamples,
                                ntimepoints=ntimepoints,
                                save_dir=save_directory,
                                grf_config_idx=i
                            )

    # Matern Kernel configurations
    print("===========Matern kernel GRF=============")
    for train_test, nsamples in train_test_samples:
        for alpha in [1.5, 2, 2.5, 3.5]:
            for tau in [1, 3, 7, 11, 15]:
                for exp_factor in [2.5]:
                    for vmin, vmax in scaling_ranges:
                        print(f"\n\n\nGenerating {train_test} set with Matern kernel: alpha={alpha}, tau={tau}, exp_factor={exp_factor}, vmin={vmin}, vmax={vmax}")
                        
                        generator = DatasetGenerator(base_seed=0, train_test=train_test)
                        
                        # Add standard GRF
                        generator.add_grf_config(
                            grf_type="GRF",
                            kernel="matern",
                            vmin=vmin,
                            vmax=vmax,
                            alpha=alpha,
                            tau=tau
                        )
                        
                        # Add LogGRF
                        generator.add_grf_config(
                            grf_type="LogGRF",
                            kernel="matern",
                            vmin=vmin,
                            vmax=vmax,
                            alpha=alpha,
                            tau=tau,
                            exp_factor=exp_factor
                        )
                        
                        # Add NegLogGRF
                        generator.add_grf_config(
                            grf_type="NegLogGRF",
                            kernel="matern",
                            vmin=vmin,
                            vmax=vmax,
                            alpha=alpha,
                            tau=tau,
                            exp_factor=exp_factor
                        )
                        
                        # Generate datasets
                        for i in range(len(generator.grfs)):
                            generator.generate_dataset(
                                nu_ns=1e-5,
                                tfinal=tfinal,
                                nsamples=nsamples,
                                batch_size=nsamples,
                                ntimepoints=ntimepoints,
                                save_dir=save_directory,
                                grf_config_idx=i
                            )

    # RQ Kernel configurations
    print("===========RQ kernel GRF=============")
    for train_test, nsamples in train_test_samples:
        for length_scale in [0.05, 0.1, 0.2]:
            for variance in [0.7, 1.0, 1.7]:
                for alpha in [0.75, 1.0, 1.5, 1.7]:
                    for exp_factor in [2.5]:
                        for vmin, vmax in scaling_ranges:
                            print(f"\n\n\nGenerating {train_test} set with RQ kernel: length_scale={length_scale}, variance={variance}, alpha={alpha}, exp_factor={exp_factor}, vmin={vmin}, vmax={vmax}")
                            
                            generator = DatasetGenerator(base_seed=0, train_test=train_test)
                            
                            # Add standard GRF
                            generator.add_grf_config(
                                grf_type="GRF",
                                kernel="rq",
                                vmin=vmin,
                                vmax=vmax,
                                length_scale=length_scale,
                                variance=variance,
                                alpha=alpha
                            )
                            
                            # Add LogGRF
                            generator.add_grf_config(
                                grf_type="LogGRF",
                                kernel="rq",
                                vmin=vmin,
                                vmax=vmax,
                                length_scale=length_scale,
                                variance=variance,
                                alpha=alpha,
                                exp_factor=exp_factor
                            )
                            
                            # Add NegLogGRF
                            generator.add_grf_config(
                                grf_type="NegLogGRF",
                                kernel="rq",
                                vmin=vmin,
                                vmax=vmax,
                                length_scale=length_scale,
                                variance=variance,
                                alpha=alpha,
                                exp_factor=exp_factor
                            )
                            
                            # Generate datasets
                            for i in range(len(generator.grfs)):
                                generator.generate_dataset(
                                    nu_ns=1e-5,
                                    tfinal=tfinal,
                                    nsamples=nsamples,
                                    batch_size=nsamples,
                                    ntimepoints=ntimepoints,
                                    save_dir=save_directory,
                                    grf_config_idx=i
                                )

    # Periodic Kernel configurations
    print("===========Periodic kernel GRF=============")
    for train_test, nsamples in train_test_samples:
        for length_scale in [40]:
            for period in [0.05, 0.1, 0.2, 0.5]:
                for exp_factor in [2.5]:
                    for vmin, vmax in scaling_ranges:
                        print(f"\n\n\nGenerating {train_test} set with Periodic kernel: length_scale={length_scale}, period={period}, exp_factor={exp_factor}, vmin={vmin}, vmax={vmax}")
                        
                        generator = DatasetGenerator(base_seed=0, train_test=train_test)
                        
                        # Add standard GRF
                        generator.add_grf_config(
                            grf_type="GRF",
                            kernel="periodic",
                            vmin=vmin,
                            vmax=vmax,
                            length_scale=length_scale,
                            period=period
                        )
                        
                        # Add LogGRF
                        generator.add_grf_config(
                            grf_type="LogGRF",
                            kernel="periodic",
                            vmin=vmin,
                            vmax=vmax,
                            length_scale=length_scale,
                            period=period,
                            exp_factor=exp_factor
                        )
                        
                        # Add NegLogGRF
                        generator.add_grf_config(
                            grf_type="NegLogGRF",
                            kernel="periodic",
                            vmin=vmin,
                            vmax=vmax,
                            length_scale=length_scale,
                            period=period,
                            exp_factor=exp_factor
                        )
                        
                        # Generate datasets
                        for i in range(len(generator.grfs)):
                            generator.generate_dataset(
                                nu_ns=1e-5,
                                tfinal=tfinal,
                                nsamples=nsamples,
                                batch_size=nsamples,
                                ntimepoints=ntimepoints,
                                save_dir=save_directory,
                                grf_config_idx=i
                            )



