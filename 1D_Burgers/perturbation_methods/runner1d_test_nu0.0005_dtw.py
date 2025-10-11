from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from tqdm import tqdm
from datetime import datetime
import time
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO1d import FNO1d
from GRFs.generateGRFs import GRFGenerator
from solvers.burgers1d_solvers import *
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from torch.autograd.functional import jacobian
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from pathlib import Path
import subprocess

jax.config.update("jax_enable_x64", True)

# --- 只用 tslearn 实现 DTW / SoftDTW ---
from tslearn.metrics import dtw, SoftDTWLossPyTorch  # 经典 DTW + 可微 SoftDTW

def get_gpu_info():
    try:
        partition = os.environ.get("SLURM_JOB_PARTITION", "N/A")
        node_name = os.environ.get("SLURMD_NODENAME", "N/A")
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")
        try:
            nvidia_smi_output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
                encoding="utf-8"
            ).strip()
        except FileNotFoundError:
            nvidia_smi_output = "nvidia-smi 未找到，可能此节点没有 NVIDIA GPU"
        print("========== 当前作业 GPU 信息 ==========")
        print(f"分区: {partition}")
        print(f"节点: {node_name}")
        print(f"GPU ID (CUDA_VISIBLE_DEVICES): {gpu_ids}")
        print("\nGPU 详细信息:")
        print(nvidia_smi_output)
        print("=====================================")
    except Exception as e:
        print(f"获取 GPU 信息时出错: {e}")

class JaxPDEWrapper(torch.autograd.Function):
    _vjp_cache = {}

    @staticmethod
    def forward(ctx, a_torch, g):
        if not a_torch.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        try:
            a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
            a_jax = jax.dlpack.from_dlpack(a_dlpack)
            g_output_jax = g(a_jax)
            out_dlpack = jax.dlpack.to_dlpack(g_output_jax)
            g_output = torch.utils.dlpack.from_dlpack(out_dlpack)
            ctx.save_for_backward(a_torch)
            ctx.g = g
            return g_output
        except Exception as e:
            import warnings
            warnings.warn(f"GPU transfer failed, fallback to CPU: {str(e)}")
            a_np = a_torch.detach().cpu().numpy()
            g_output_np = g(jnp.array(a_np))
            return torch.as_tensor(np.asarray(g_output_np),
                                   device=a_torch.device,
                                   dtype=a_torch.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a_torch, = ctx.saved_tensors
        g = ctx.g
        cache_key = id(g)
        if cache_key not in JaxPDEWrapper._vjp_cache:
            def jax_vjp(a_jax, grad_jax):
                _, vjp_fn = jax.vjp(g, a_jax)
                return vjp_fn(grad_jax)
            JaxPDEWrapper._vjp_cache[cache_key] = jax.jit(jax_vjp)
        jitted_vpj = JaxPDEWrapper._vjp_cache[cache_key]
        try:
            grad_dlpack = torch.utils.dlpack.to_dlpack(grad_output.contiguous())
            grad_jax = jax.dlpack.from_dlpack(grad_dlpack)
            a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
            a_jax = jax.dlpack.from_dlpack(a_dlpack)
            grad_input_jax, = jitted_vpj(a_jax, grad_jax)
            grad_input_dlpack = jax.dlpack.to_dlpack(grad_input_jax)
            return torch.utils.dlpack.from_dlpack(grad_input_dlpack), None
        except Exception as e:
            import warnings
            warnings.warn(f"Backward GPU failed: {str(e)}")
            grad_np = grad_output.detach().cpu().numpy()
            a_np = a_torch.detach().cpu().numpy()
            _, vjp_fn = jax.vjp(g, jnp.array(a_np))
            grad_input_jax, = vjp_fn(jnp.array(grad_np))
            return torch.as_tensor(np.asarray(grad_input_jax),
                                   device=a_torch.device,
                                   dtype=a_torch.dtype), None

def find_closest_ground_truth(a_perturbed, x_dict, y_dict):
    if len(a_perturbed.shape) == 1:
        a_perturbed = a_perturbed.unsqueeze(0)
    a_perturbed = a_perturbed.squeeze(-1)
    distances = torch.norm(x_dict - a_perturbed, p=2, dim=1)
    closest_idx = torch.argmin(distances)
    return y_dict[closest_idx]

def compare_gradient_attack(a, G, g, x_dict, y_dict, epsilon, alpha, num_steps,
                            norm='inf', fd_delta=1e-8, enable_fd=False, use_double=True):
    original_dtypes = {name: param.dtype for name, param in G.named_parameters()}

    if use_double:
        dtype = torch.float64
        G = G.to(dtype)
    else:
        dtype = torch.float32
        G = G.to(dtype)

    a = a.to(dtype)
    if isinstance(g, torch.Tensor):
        g = g.to(dtype)

    def convert_data(data):
        if isinstance(data, torch.Tensor):
            return data.to(dtype)
        elif isinstance(data, dict):
            return {k: convert_data(v) for k, v in data.items()}
        return data
    x_dict = convert_data(x_dict)
    y_dict = convert_data(y_dict)

    # SoftDTW（可微）损失对象；与数据放到同一 device
    sdtw_loss_fn = SoftDTWLossPyTorch(gamma=1.0, normalize=False).to(a.device)

    delta_with = torch.zeros_like(a, requires_grad=True, dtype=dtype)
    delta_without = torch.zeros_like(a, requires_grad=True, dtype=dtype)
    delta_approximate = torch.zeros_like(a, requires_grad=True, dtype=dtype)
    grad_records = {}
    grad_records["step_update"] = {}

    def wrap_with_solver(input_tensor):
        G_output = G(input_tensor)
        g_output = JaxPDEWrapper.apply(input_tensor, g)
        mse_loss = F.mse_loss(G_output, g_output)
        return mse_loss, (G_output, g_output)

    def wrap_without_solver(input_tensor):
        G_output = G(input_tensor)
        g_output = JaxPDEWrapper.apply(input_tensor, g)
        mse_loss = F.mse_loss(G_output, g_output.detach())
        return mse_loss, (G_output, g_output.detach())

    def wrap_approximate(input_tensor):
        G_output = G(input_tensor)
        closest = find_closest_ground_truth(input_tensor, x_dict, y_dict)
        mse_loss = F.mse_loss(G_output, closest)
        return mse_loss, (G_output, closest)

    def compute_all_loss_metrics(a, delta_with, delta_without, delta_approximate):
        loss_dict = {}
        modes = {
            "with_solver": a + delta_with,
            "without_solver": a + delta_without,
            "approximate": a + delta_approximate,
        }
        for mode_name, a_mode in modes.items():
            # 计算 MSE + 取得配对序列
            if mode_name == "with_solver":
                mse_loss, (G_out, g_out) = wrap_with_solver(a_mode)
            elif mode_name == "without_solver":
                mse_loss, (G_out, g_out) = wrap_without_solver(a_mode)
            else:
                mse_loss, (G_out, g_out) = wrap_approximate(a_mode)

            # 压到 1D（确保是时间序列）
            G_flat = G_out.squeeze()
            g_flat = g_out.squeeze()

            # --- 经典 DTW（tslearn）：用 numpy 计算，记录为标量 ---
            dtw_loss = float(dtw(G_flat.detach().cpu().numpy(),
                                 g_flat.detach().cpu().numpy()))

            # --- Soft-DTW（可微，tslearn）：(B, T, D) ---
            G_seq = G_flat.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            g_seq = g_flat.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            # 确保 device/dtype 一致
            G_seq = G_seq.to(a.device, dtype=dtype)
            g_seq = g_seq.to(a.device, dtype=dtype)
            softdtw_value = sdtw_loss_fn(G_seq, g_seq)  # shape [1]
            softdtw_loss = softdtw_value.squeeze()

            # 记录 3×3 指标
            loss_dict[f"mse_{mode_name}"] = mse_loss.item()
            loss_dict[f"dtw_{mode_name}"] = dtw_loss
            loss_dict[f"softdtw_{mode_name}"] = softdtw_loss.item()

        return loss_dict

    # step 0
    initial_losses = compute_all_loss_metrics(a, delta_with, delta_without, delta_approximate)
    grad_records["step_update"]["step_0"] = {"loss_metrics": initial_losses}

    for step in range(num_steps):
        # with_solver
        a_with = a + delta_with
        loss_with, _ = wrap_with_solver(a_with)
        loss_with.backward()
        grad_with = delta_with.grad.clone().detach()
        delta_with.grad.zero_()

        # without_solver
        a_wo = a + delta_without
        loss_wo, _ = wrap_without_solver(a_wo)
        loss_wo.backward()
        grad_wo = delta_without.grad.clone().detach()
        delta_without.grad.zero_()

        # approximate
        a_ap = a + delta_approximate
        loss_ap, _ = wrap_approximate(a_ap)
        loss_ap.backward()
        grad_ap = delta_approximate.grad.clone().detach()
        delta_approximate.grad.zero_()

        def update(delta, grad):
            if norm == 'inf':
                delta.data.add_(alpha * torch.sign(grad))
                delta.data.copy_(delta.data.clamp(-epsilon, epsilon))
            elif norm == '2':
                delta.data.add_(alpha * grad / (torch.norm(grad, p=2) + 1e-15))
                norm2 = torch.norm(delta.data, p=2)
                if norm2 > epsilon:
                    delta.data.mul_(epsilon / norm2)

        update(delta_with, grad_with)
        update(delta_without, grad_wo)
        update(delta_approximate, grad_ap)

        # 记录本步 9 个指标
        cur_losses = compute_all_loss_metrics(a, delta_with, delta_without, delta_approximate)
        grad_records["step_update"][f"step_{step+1}"] = {"loss_metrics": cur_losses}

    return grad_records

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])

def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    get_gpu_info()

    model_path = config.model_path
    workdir = config.workdir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M")
    device = config.device
    nu = float(re.findall(r'nu([a-zA-Z0-9|.]+)', model_path)[0])
    num_records = config.num_records
    inputs = config.inputs

    def convert_to_cpu_serializable(data):
        if isinstance(data, dict):
            return {k: convert_to_cpu_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_to_cpu_serializable(v) for v in data]
        elif isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        else:
            return data

    print("===============Experiment===============")
    print("model_path: ", model_path)
    print("workdir: ", workdir)
    print("nu: ", nu)
    print("jax.devices()", jax.devices())

    grf_params = {
        'dim': 1,
        'nx': 1024,
        'kernel': 'gaussian',
        'kernel_params': {'correlation_length': 0.03},
        'bc': 'periodic',
        'seed': 3003,
        'zero_mean': False
    }
    burgers_params = {
        'nu': nu,
        'simulation_time': 1.0,
        "step": 0.0001
    }
    dataset_params = {
        "num_record": num_records
    }
    params = {
        "GRF": grf_params,
        "dataset": dataset_params,
        "burgers": burgers_params,
    }
    shape = (grf_params["nx"],)
    s = 1024
    sub = grf_params["nx"] // s
    solver_name = re.findall(r'solver=([a-zA-Z0-9]+)', model_path)[0]

    model = FNO1d(modes=16, width=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    current_file_path = Path(__file__).resolve().parent.parent
    N = re.search(r'(?:^|_)N(\d+)(?:_|$)', config.dict_input_path).group(1)
    gradient_folder = current_file_path / "perturbation_methods" / "nu=0.0005_PGD_results" / f"approxN{N}"
    gradient_folder.mkdir(parents=True, exist_ok=True)

    input_path = config.dict_input_path
    output_path = config.dict_output_path
    print("input_path: ", input_path)
    print("output_path: ", output_path)
    x_dict = torch.load(input_path)[:, ::sub].to(device)
    y_dict = torch.load(output_path)[:, ::sub].to(device)

    gradient_records = {}

    for input_tuple in inputs:
        norm, epsilon, num_steps, alpha = input_tuple
        for i in tqdm(range(dataset_params['num_record']), desc="adversarial input samples"):
            start_time = time.time()
            a = GRFGenerator.generate_grf(
                shape=shape,
                kernel=grf_params['kernel'],
                kernel_params=grf_params['kernel_params'],
                bc=grf_params['bc'],
                seed=grf_params['seed'] + i,
                zero_mean=grf_params['zero_mean']
            )
            a = a[::sub]
            a_torch = torch.from_numpy(a).float().unsqueeze(0).unsqueeze(-1).to(device)

            if solver_name == "exponax":
                solver = ExponaxBurgersSolver1D(s, nu=burgers_params['nu'], bc=grf_params['bc'], xlim=(0,1))
            elif solver_name == "scipy":
                solver = SciPyBurgersSolver1D(s, nu=burgers_params['nu'], bc=grf_params['bc'])
            elif solver_name == "scipy_spectral":
                solver = SciPySpectralBurgersSolver1D(s, nu=burgers_params['nu'], bc=grf_params['bc'])
            elif solver_name == "phiflow":
                solver = PhiFlowBurgersSolver1D(s, nu=burgers_params['nu'], bc=grf_params['bc'])
            else:
                raise ValueError("Specified solver not implemented")

            t_span = (0, burgers_params['simulation_time'])
            PDE_func = jax.jit(lambda u0: solver.solve(
                u0, t_final=burgers_params['simulation_time'],
                t_eval=t_span, step=burgers_params["step"]
            )[1][1])

            records = compare_gradient_attack(
                a_torch, model, PDE_func, x_dict, y_dict,
                epsilon=epsilon, alpha=alpha,
                num_steps=num_steps, norm=norm,
                enable_fd=True, fd_delta=1e-8
            )
            gradient_records[(f"norm_{norm}", f"index_{i}", f"numsteps_{num_steps}",
                               f"epsilon_{epsilon}", f"alpha_{alpha:.5f}")] = convert_to_cpu_serializable(records)
            total_time = time.time() - start_time
            print("\n index:", i, ", norm:", norm, ", num_steps:", num_steps,
                  ", epsilon:", epsilon, ", alpha:", alpha, ", total time:", total_time)

    # 保存文件：文件名带 dtw 后缀；每步已记录 3×3 共 9 个指标（mse/dtw/softdtw × with/detached/approx）
    gradient_filename = f"gradient_test_{solver_name}_nu{nu}_nsamples{num_records}_dtw.pkl"
    with open(gradient_folder / gradient_filename, "wb") as f:
        pickle.dump(gradient_records, f)
    print(f"Pickle file saved to {gradient_folder}/{gradient_filename}")

if __name__ == "__main__":
    app.run(main)
