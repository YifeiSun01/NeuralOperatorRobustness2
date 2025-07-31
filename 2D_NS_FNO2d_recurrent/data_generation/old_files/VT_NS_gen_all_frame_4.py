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
    step_options = [0.01, 0.005, 0.001]  # 从大到小尝试
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
    def __init__(self, base_seed=0):
        self.global_seed_counter = base_seed
        self.dataset_counter = 0
        # 预加载数据集到GPU
        self.zongyi_dataset = self._load_dataset_to_gpu()
    
    def _load_dataset_to_gpu(self):
        """加载数据集并预处理到GPU"""
        # 注意：实际路径需要根据环境调整
        dataset_path = '../2D_NS_old/2D_NS_Zongyi_Li/recurrent/datasets/2D/NS/NS_data_zongyi_test_all_frame.pt'
        # dataset_path = '../2D_NS_old/2D_NS_Zongyi_Li/recurrent/datasets/2D/NS/NS_data_zongyi_train_all_frame.pt'
        
        print(f"Loading dataset from {dataset_path}...")
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)
        
        # 预上采样所有数据到GPU
        print("Pre-upsampling dataset on GPU...")
        upsampled_data = []
        for i in tqdm(range(len(dataset['x'])), desc="Upsampling", unit="sample"):
            upsampled_data.append(spectral_upsample(dataset['x'][i], 256))
        
        # 替换原始数据
        dataset['x'] = torch.stack(upsampled_data)
        return dataset
    
    def _generate_filename(self, params):
        return (f"dim{params['dim']}d_nx{params['nx']}_N{params['nsamples']}_"
                f"solver=exponax_"
                f"nu{params['nu_ns']:.3f}_"
                f"t{params['tfinal']:.1f}_"
                "test_"
                # "train_"
                f"all_frames.pt")
    
    def generate_dataset(self, nu_ns, tfinal, nsamples, save_dir=None):
        dataset_x = []
        dataset_y = []
        
        # 使用预加载到GPU的数据集
        for idx in tqdm(range(nsamples), desc="Generating samples", unit="sample"):
            # 直接从GPU获取预上采样的数据
            grf = self.zongyi_dataset["x"][idx]
            
            time_list = np.linspace(0, tfinal, 5001)
            ntimepoints = tfinal + 1
            
            # 生成数据（全部在GPU上）
            x, y, selected_times = generate_vorticity_transport_data(
                time_list,
                grf,
                nu_ns,
                ntimepoints
            )
            
            dataset_x.append(x)
            dataset_y.append(y)
            
            # 每10个样本清理一次JAX缓存
            if idx % 10 == 0:
                jax.clear_caches()
        
        # 直接在GPU上堆叠
        tensor_x = torch.stack(dataset_x)
        tensor_y = torch.stack(dataset_y)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_params = {
                'dim': 2,
                'nx': 256,  # 上采样后固定为256
                'nsamples': nsamples,
                'nu_ns': nu_ns,
                'tfinal': tfinal,
            }
            
            filename = self._generate_filename(file_params)
            save_path = os.path.join(save_dir, filename)
            
            # 保存时移动到CPU以减少内存占用
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
            
            print(f"\nDataset saved to {save_path}")
            return save_path
        else:
            return tensor_x, tensor_y

if __name__ == "__main__":
    # 使用示例
    generator = DatasetGenerator(base_seed=0)

    print("Generating dataset:")
    path1 = generator.generate_dataset(
        nu_ns=1e-5,
        tfinal=20,
        # nsamples=1150,
        nsamples=50,
        save_dir="./datasets"
    )




