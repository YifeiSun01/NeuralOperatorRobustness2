import numpy as np
from scipy.ndimage import gaussian_filter
import jax
import jax.numpy as jnp
import exponax as ex
import torch
import os
from tqdm import tqdm
import time
from jax.lib import xla_bridge

# 设置环境变量减少内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Set default device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 显存监控函数
def print_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        print(f"{prefix} GPU: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free={free:.2f}GB")
    else:
        import psutil
        mem = psutil.virtual_memory()
        print(f"{prefix} RAM: Used={mem.used/1024**3:.2f}GB, Free={mem.free/1024**3:.2f}GB")

# JAX内存清理函数
def clear_jax_mem():
    try:
        backend = xla_bridge.get_backend()
        if hasattr(backend, 'defragment'):
            backend.defragment()
    except Exception as e:
        print(f"JAX memory cleanup failed: {e}")

class GRFGenerator:
    @staticmethod
    def generate_grf(shape, kernel='gaussian', kernel_params=None, bc='periodic', seed=None, zero_mean=False):
        if seed is not None:
            key = jax.random.PRNGKey(seed)
        else:
            key = jax.random.PRNGKey(0)
        
        # Generate random field on GPU using JAX
        with jax.default_device(jax.devices('gpu')[0] if jax.devices('gpu') else jax.default_device(jax.devices('cpu')[0])):
            white_noise = jax.random.normal(key, shape=shape)
            
            if kernel == 'gaussian':
                cl = kernel_params.get('correlation_length', 1.0)*shape[0]
                if bc == 'periodic':
                    mode = 'wrap'
                else:
                    mode = 'nearest'
                grf = gaussian_filter(white_noise, sigma=cl, mode=mode)
            elif kernel == 'matern':
                if bc != 'periodic':
                    raise NotImplementedError("Matern kernel currently only supports periodic BCs")
                cl = kernel_params.get('correlation_length', 1.0)*shape[0]
                nu = kernel_params.get('nu', 1.5)
                if len(shape) == 1:
                    grf = GRFGenerator._matern_1d(shape, cl, nu, seed)
                elif len(shape) == 2:
                    grf = GRFGenerator._matern_2d(shape, cl, nu, seed)
                else:
                    raise ValueError("Only 1D or 2D GRFs supported")
            else:
                raise ValueError(f"Unknown kernel: {kernel}")
            
            grf = (grf - grf.min()) / (grf.max() - grf.min())
            if zero_mean:
                grf -= grf.mean()
            
            return grf
    
    @staticmethod
    def _matern_1d(shape, cl, nu, seed):
        key = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(0)
        with jax.default_device(jax.devices('gpu')[0] if jax.devices('gpu') else jax.default_device(jax.devices('cpu')[0])):
            k = jnp.fft.fftfreq(shape[0])
            scaling = (2 * jnp.pi) ** 2 / (cl ** 2)
            k_sq = (2 * jnp.pi * k) ** 2
            sk = (scaling + k_sq) ** (-nu - 0.5)
            noise = (jax.random.normal(key, shape=shape) + 1j * jax.random.normal(key+1, shape=shape)) * jnp.sqrt(sk)
            grf = jnp.fft.ifft(noise).real
            return grf
    
    @staticmethod
    def _matern_2d(shape, cl, nu, seed):
        key = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(0)
        with jax.default_device(jax.devices('gpu')[0] if jax.devices('gpu') else jax.default_device(jax.devices('cpu')[0])):
            nx, ny = shape
            kx = jnp.fft.fftfreq(nx)
            ky = jnp.fft.fftfreq(ny)
            kxx, kyy = jnp.meshgrid(kx, ky, indexing='ij')
            k_sq = (2 * jnp.pi * kxx) ** 2 + (2 * jnp.pi * kyy) ** 2
            scaling = (2 * jnp.pi) ** 2 / (cl ** 2)
            sk = (scaling + k_sq) ** (-nu - 1)
            noise = (jax.random.normal(key, shape=shape) + 1j * jax.random.normal(key+1, shape=shape)) * jnp.sqrt(sk)
            grf = jnp.fft.ifft2(noise).real
            return grf

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
        # Move initial condition to GPU
        full_ic = jnp.expand_dims(jnp.array(u0, dtype=jnp.float32), axis=0)
        
        # Initialize solver on GPU
        slower_NS_stepper = ex.stepper.NavierStokesVorticity(
            2, self.Lx, self.nx, step, 
            diffusivity=self.nu
        )
        
        longer_rollout_NS_stepper = ex.rollout(
            slower_NS_stepper, int(t_final/step), include_init=True
        )

        # Run simulation on GPU
        longer_trajectory = longer_rollout_NS_stepper(full_ic)
        
        # Convert results back to numpy for compatibility
        sol_dict = {i*step: np.array(longer_trajectory[i, 0, ...].T.block_until_ready())
            for i in range(longer_trajectory.shape[0])}
        sol_dict = self.get_closest_solutions(sol_dict, t_eval)
        
        return sol_dict

def generate_vorticity_transport_data(time_list, grf_params, grf, nu, ntimepoints):
    # Initialize solver
    exponaxsolver = ExponaxVTSolver2D(
        nx=grf_params["nx"], ny=grf_params["nx"], 
        Lx=128, Ly=128, nu=nu
    )
    
    # Convert input to numpy (handling both JAX and numpy arrays)
    if hasattr(grf, '__jax_array__') or isinstance(grf, jnp.ndarray):  # JAX array
        grf_numpy = np.array(grf)  # Convert JAX to numpy
    elif isinstance(grf, np.ndarray):  # Already numpy
        grf_numpy = grf
    elif isinstance(grf, torch.Tensor):  # PyTorch tensor
        grf_numpy = grf.cpu().numpy()
    else:
        raise TypeError(f"Unsupported input type: {type(grf)}")
    
    # Get solutions
    exponax_solutions = exponaxsolver.solve(
        grf_numpy,  # Use converted numpy array
        t_final=time_list[-1],
        t_eval=[0] + time_list,
        step=0.01
    )

    selected_indices = np.linspace(0, len(time_list)-1, ntimepoints, dtype=int)
    selected_times = time_list[selected_indices]
    
    # 使用半精度节省显存
    x = torch.from_numpy(exponax_solutions[0][1]).half().to(device)  # 半精度
    y_numpy = np.array([exponax_solutions[selected_time][1] for selected_time in selected_times])
    y = torch.from_numpy(y_numpy).half().to(device)  # 半精度
    
    return x, y, selected_times

class DatasetGenerator:
    def __init__(self, base_seed=0):
        self.global_seed_counter = base_seed
        self.dataset_counter = 0
        self.last_memory_check = time.time()
    
    def _generate_filename(self, params):
        return (f"dim{params['dim']}d_nx{params['nx']}_N{params['nsamples']}_"
                f"solver=exponax_kernel={params['kernel']}_"
                f"correlation_length{params['cl']:.2f}_"
                f"bc{params['bc']}_nu{params['nu_ns']:.3f}_"
                f"t{params['tfinal']:.1f}_"
                f"seed{params['seed']}_all_frames.pt")
    
    def generate_dataset(self, cl, nu_grf, nu_ns, tfinal, nsamples, save_dir=None, chunk_size=20):
        start_seed = self.global_seed_counter
        temp_dir = None
        
        # 如果保存目录存在，创建临时目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            temp_dir = os.path.join(save_dir, f"temp_{start_seed}")
            os.makedirs(temp_dir, exist_ok=True)
        
        # 分块处理
        chunk_count = (nsamples + chunk_size - 1) // chunk_size
        
        for chunk_idx in tqdm(range(chunk_count), desc="Processing chunks"):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, nsamples)
            chunk_samples = chunk_end - chunk_start
            
            chunk_x, chunk_y = [], []
            selected_times = None
            
            # 处理当前chunk
            for sample_idx in range(chunk_samples):
                # 每10个样本检查一次内存
                if time.time() - self.last_memory_check > 300:  # 每分钟检查一次
                    print_gpu_memory(f"Chunk {chunk_idx} sample {sample_idx}")
                    self.last_memory_check = time.time()
                
                grf_params = {
                    'dim': 2,
                    'nx': 256,
                    'kernel': 'matern',
                    'kernel_params': {'correlation_length': cl, "nu": nu_grf},
                    'bc': 'periodic',
                    'seed': self.global_seed_counter,
                    'zero_mean': False
                }
                
                # Generate on GPU
                grf = GRFGenerator.generate_grf(
                    shape=(grf_params["nx"], grf_params["nx"]),
                    kernel=grf_params['kernel'],
                    kernel_params=grf_params['kernel_params'],
                    bc=grf_params['bc'],
                    seed=int(grf_params['seed']),
                    zero_mean=grf_params['zero_mean']
                )
                
                time_list = np.linspace(0, tfinal, 5001)
                ntimepoints = 41
                x, y, selected_times = generate_vorticity_transport_data(
                    time_list,
                    grf_params,
                    np.array(grf),
                    nu_ns,
                    ntimepoints
                )
                
                # 立即转移到CPU并释放GPU显存
                chunk_x.append(x.cpu())
                chunk_y.append(y.cpu())
                self.global_seed_counter += 1
                
                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                clear_jax_mem()
            
            # 堆叠当前chunk
            current_x = torch.stack(chunk_x)
            current_y = torch.stack(chunk_y)
            
            if save_dir and temp_dir:
                # 保存临时chunk文件
                chunk_file = os.path.join(temp_dir, f"chunk_{chunk_start}_{chunk_end}.pt")
                torch.save({
                    'x': current_x,
                    'y': current_y,
                    'metadata': {
                        'times': selected_times,
                        'seed_range': (chunk_start + start_seed, chunk_end - 1 + start_seed)
                    }
                }, chunk_file)
            
            # 立即释放内存
            del chunk_x, chunk_y, current_x, current_y
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 合并所有chunk（仅在保存模式时）
        if save_dir and temp_dir:
            # 收集所有临时文件
            chunk_files = sorted([
                f for f in os.listdir(temp_dir) 
                if f.startswith("chunk_") and f.endswith(".pt")
            ])
            
            # 初始化最终数据
            final_data = {
                'u0': [],
                'u': [],
                'metadata': {
                    'cl': cl,
                    'nu_grf': nu_grf,
                    'nu_ns': nu_ns,
                    'tfinal': tfinal,
                    'nsamples': nsamples,
                    'seed_range': (start_seed, self.global_seed_counter - 1),
                    'times': selected_times
                }
            }
            
            # 逐步加载并合并
            for f in tqdm(chunk_files, desc="Merging chunks"):
                chunk_path = os.path.join(temp_dir, f)
                chunk = torch.load(chunk_path)
                final_data['u0'].append(chunk['x'])
                final_data['u'].append(chunk['y'])
                os.remove(chunk_path)  # 删除临时文件
            
            # 使用concat避免内存峰值
            final_data['u0'] = torch.cat(final_data['u0'])
            final_data['u'] = torch.cat(final_data['u'])
            
            # 保存最终文件
            file_params = {
                'dim': 2,
                'nx': 256,
                'nsamples': nsamples,
                'kernel': 'matern',
                'cl': cl,
                'bc': 'periodic',
                'nu_ns': nu_ns,
                'tfinal': tfinal,
                'seed': start_seed
            }
            filename = self._generate_filename(file_params)
            save_path = os.path.join(save_dir, filename)
            
            # 添加进度条保存
            with tqdm(total=1, desc="Saving final dataset") as pbar:
                torch.save(final_data, save_path)
                pbar.update(1)
            
            # 清理临时目录
            os.rmdir(temp_dir)
            print(f"\nDataset saved to {save_path}")
            return save_path
        
        else:
            # 非保存模式返回最后一个chunk（不适合大数据集）
            return current_x.to(device), current_y.to(device)

if __name__ == "__main__":
    # 打印初始显存状态
    print_gpu_memory("Initial memory state")
    
    # Usage Example
    generator = DatasetGenerator(base_seed=0)

    print("Generating first dataset:")
    path1 = generator.generate_dataset(
        cl=0.5,
        nu_grf=1.5,
        nu_ns=0.01,
        tfinal=40,
        nsamples=100,
        save_dir="./datasets",
        chunk_size=10  # 根据GPU显存调整
    )

    print("Generating second dataset:")
    path2 = generator.generate_dataset(
        cl=0.5,
        nu_grf=1.5,
        nu_ns=0.01,
        tfinal=40,
        nsamples=100,
        save_dir="./datasets",
        chunk_size=10  # 根据GPU显存调整
    )
    
    # 打印最终显存状态
    print_gpu_memory("Final memory state")