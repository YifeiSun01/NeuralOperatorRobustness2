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
from pathlib import Path

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
    def __init__(self, nx, ny, Lx, Ly, nu, forcing=None, bc='periodic', forcing_pattern="ringsCos"):
        self.nx = nx
        self.Lx = Lx
        self.x = torch.linspace(-Lx/2, Lx/2, nx, device=device)
        self.dx = 2/nx
        self.nu = nu
        self.bc = bc
        self.forcing_pattern = forcing_pattern

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
        slower_NS_stepper = ex.stepper._navier_stokes.NavierStokesVorticityPatterns(
            2, self.Lx, self.nx, step, 
            diffusivity=self.nu, order=4,
            num_circle_points=16, 
            dealiasing_fraction=2/3,
            forcing_pattern=self.forcing_pattern
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

def generate_vorticity_transport_data(time_list, grf, nu, ntimepoints, forcing_pattern):
    # 确保输入在GPU上
    if not isinstance(grf, torch.Tensor):
        grf = torch.tensor(grf, device=device)
    else:
        grf = grf.to(device)

    # PyTorch 版本的翻转/旋转
    grf = torch.flip(grf, [0])
    grf = torch.rot90(grf, k=3, dims=[0, 1])

    # 初始化求解器
    exponaxsolver = ExponaxVTSolver2D(
        nx=grf.shape[0], ny=grf.shape[1],
        Lx=1, Ly=1, nu=nu, forcing_pattern=forcing_pattern
    )

    # 尝试不同的时间步长（不捕获异常；一旦报错会直接打印完整堆栈）
    step_options = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    # step_options = [0.001, 0.0005, 0.0001]
    valid_solution = False
    chosen_solutions = None

    for step_size in step_options:
        exponax_solutions = exponaxsolver.solve(
            grf,
            t_final=time_list[-1],
            t_eval=[0] + time_list,
            step=step_size
        )
        # 提取所有解用于稳定性检查
        all_solutions = torch.stack([sol[1] for sol in exponax_solutions.values()])

        if check_solution_stability(all_solutions):
            valid_solution = True
            chosen_solutions = exponax_solutions
            print(f"Step size {step_size} produced stable solution.")
            break
        else:
            print(f"Step size {step_size} produced unstable solution. Trying smaller step.")

    if not valid_solution:
        raise RuntimeError(
            f"No stable solution found. Tried step sizes: {step_options}"
        )

    selected_indices = np.linspace(0, len(time_list)-1, ntimepoints, dtype=int)
    selected_times = time_list[selected_indices]

    # 结果已经在GPU上
    x = chosen_solutions[0][1]
    y = torch.stack([chosen_solutions[selected_time][1] for selected_time in selected_times])

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
        self.zongyi_dataset = self._load_dataset_to_gpu()

    def _load_dataset_to_gpu(self):
        dataset_path = f'/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_old/2D_NS_Zongyi_Li/recurrent/datasets/2D/NS/NS_data_zongyi_{self.train_test}_all_frame_real_initial.pt'
        print(f"Loading dataset from {dataset_path}...")
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)
        print("Pre-upsampling dataset on GPU...")
        upsampled_data = [
            spectral_upsample(dataset['x'][i], 256)
            for i in tqdm(range(len(dataset['x'])), desc="Upsampling", unit="sample")
        ]
        dataset['x'] = torch.stack(upsampled_data)
        return dataset

    def _generate_filename(self, params, batch_idx=None):
        fname = (f"dim{params['dim']}d_nx{params['nx']}_N{params['nsamples']}_"
                 f"solver=exponax_"
                 f"nu{params['nu_ns']:.3f}_"
                 f"t{params['tfinal']:.1f}_"
                 f"{self.train_test}_"
                 # f"ntimepoints{self.ntimepoints}_"
                 f"forcingPattern{params['forcing_pattern']}")
        if batch_idx is not None:
            # fname += f"batch{batch_idx}_"
            pass
        fname += ".pt"
        return fname

    def generate_dataset(self, nu_ns, tfinal, nsamples, ntimepoints, batch_size=200, save_dir="./datasets", forcing_pattern="ringsCos"):
        self.ntimepoints = ntimepoints
        total_batches = (nsamples + batch_size - 1) // batch_size

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
                "ntimepoints": ntimepoints,
                "forcing_pattern": forcing_pattern
            }

            filename = self._generate_filename(file_params, batch_idx=batch_idx)
            save_path = os.path.join(save_dir, filename)

            if os.path.exists(save_path):
                print(f"Batch {batch_idx} already exists, skipping...")
                continue

            print(f"\nStarting batch {batch_idx + 1}/{total_batches}...")
            dataset_x, dataset_y = [], []
            failed_indices = []
            selected_times = None

            for i in tqdm(range(start_idx, end_idx), desc=f"Generating batch {batch_idx}", unit="sample"):
                grf = self.zongyi_dataset["x"][i]
                time_list = np.linspace(0, tfinal, 5001)

                x, y, selected_times = generate_vorticity_transport_data(
                    time_list, grf, nu_ns, ntimepoints, forcing_pattern=forcing_pattern
                )

                dataset_x.append(x)
                dataset_y.append(y)

                if i % 10 == 0:
                    jax.clear_caches()

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
                    'times': selected_times
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
    # 使用示例
    for train_test,nsamples in [("test",50),("train",1150)]:
    # for train_test,nsamples in [("test",10)]:
        generator = DatasetGenerator(base_seed=0, train_test=train_test)

        print("Generating datasets for all patterns...")
        tfinal = 20
        nsamples = nsamples  # 保持你原值

        # 六个外力图案（与你上面定义的名字一致）
        
        # patterns = ["ringsCos", "sBands", "isoCircles", "petals", "ringsL1", "ringsLinf"]
        patterns = ["none"]

        # 基础保存目录：<repo_root>/datasets
        base_dir = Path(__file__).parent.parent / "datasets"
        base_dir.mkdir(parents=True, exist_ok=True)  # 确保存在

        # 保存每个 pattern 的输出路径，便于后续使用
        generated_paths = {}

        for i, pat in enumerate(patterns, start=1):
            out_dir = base_dir / pat
            out_dir.mkdir(parents=True, exist_ok=True)  # 为该图案建子目录

            print(f"[{i}/{len(patterns)}] pattern = {pat}")
            path_i = generator.generate_dataset(
                nu_ns=1e-5,
                tfinal=tfinal,
                nsamples=nsamples,
                batch_size=nsamples,
                ntimepoints=tfinal + 1,
                save_dir=out_dir,   # 每种图案单独目录
                forcing_pattern=pat         # 关键：把当前图案名传进去
            )
            generated_paths[pat] = path_i

        print("Done. Summary:")
        for pat, p in generated_paths.items():
            print(f"  {pat}: {p}")




