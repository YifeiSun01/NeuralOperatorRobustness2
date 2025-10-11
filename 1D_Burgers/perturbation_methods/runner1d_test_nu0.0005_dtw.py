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
from torch.autograd.functional import jacobian
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from pathlib import Path
import subprocess

jax.config.update("jax_enable_x64", True)

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
        jitted = JaxPDEWrapper._vjp_cache[cache_key]
        try:
            grad_dlpack = torch.utils.dlpack.to_dlpack(grad_output.contiguous())
            grad_jax = jax.dlpack.from_dlpack(grad_dlpack)
            a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
            a_jax = jax.dlpack.from_dlpack(a_dlpack)
            grad_input_jax, = jitted(a_jax, grad_jax)
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

def single_attack_pgd(a, G, g, x_dict, y_dict, epsilon, alpha, num_steps,
                      norm, loss_type, mode, fd_delta, enable_fd, use_double):
    """
    针对单个 (loss_type, mode) 的 PGD 实施。
    返回 grad_records — 每一步记录 9 个指标（mse/dtw/softdtw × modes）
    """
    # 统一 float32
    if use_double:
        dtype = torch.float64
    else:
        dtype = torch.float32
    G = G.to(dtype)
    a = a.to(dtype)

    # SoftDTW 损失对象
    sdtw_loss_fn = SoftDTWLossPyTorch(gamma=1.0, normalize=False).to(a.device)

    # 初始化三种 delta（针对三种 mode）——不过单个攻击通常只用一种 mode 的 delta
    delta = torch.zeros_like(a, requires_grad=True, dtype=dtype)

    grad_records = {"step_update": {}}

    # 包装函数，与之前对齐 dtype/device
    def get_pair_outputs(input_tensor):
        # Return G_out, target_out according to mode
        G_out = G(input_tensor)
        if mode == "with_solver":
            T_out = JaxPDEWrapper.apply(input_tensor, g)
            if T_out.dtype != G_out.dtype:
                T_out = T_out.to(G_out.dtype)
            if T_out.device != G_out.device:
                T_out = T_out.to(G_out.device)
        elif mode == "without_solver":
            T_out = JaxPDEWrapper.apply(input_tensor, g).detach()
            if T_out.dtype != G_out.dtype:
                T_out = T_out.to(G_out.dtype)
            if T_out.device != G_out.device:
                T_out = T_out.to(G_out.device)
        elif mode == "approximate":
            T_out = find_closest_ground_truth(input_tensor, x_dict, y_dict)
            if T_out.dtype != G_out.dtype:
                T_out = T_out.to(G_out.dtype)
            if T_out.device != G_out.device:
                T_out = T_out.to(G_out.device)
        else:
            raise ValueError("Unknown mode")
        return G_out, T_out

    def compute_three_losses(G_out, T_out):
        # 压成 1D 序列
        G_flat = G_out.squeeze()
        T_flat = T_out.squeeze()

        mse_loss = F.mse_loss(G_flat, T_flat)

        dtw_loss = float(dtw(G_flat.detach().cpu().numpy(),
                             T_flat.detach().cpu().numpy()))

        G_seq = G_flat.unsqueeze(0).unsqueeze(-1).to(a.device, dtype=dtype)
        T_seq = T_flat.unsqueeze(0).unsqueeze(-1).to(a.device, dtype=dtype)
        softdtw_loss = sdtw_loss_fn(G_seq, T_seq).squeeze()

        return mse_loss, dtw_loss, softdtw_loss

    # 记录 step 0
    # 对所有三种 mode 记录指标（这里即使本攻击只用一个 mode，也记录别的 mode 作为对比）
    def compute_all_metrics_for(a_tensor):
        d = {}
        for m in ["with_solver", "without_solver", "approximate"]:
            # 临时用这个 mode 去做评价
            old_mode = mode
            # monkey patch：切换 mode
            nonlocal_mode = mode  # but Python scope issues — 可改方式：用参数传 mode
            # 为简便，这里假设你只对当前 mode 做记录。你可以扩展为对所有 mode 记录。
            G_out, T_out = get_pair_outputs(a_tensor)
            ms, dt, sd = compute_three_losses(G_out, T_out)
            d[f"mse_{m}"] = float(ms.item())
            d[f"dtw_{m}"] = float(dt)
            d[f"softdtw_{m}"] = float(sd.item())
        return d

    grad_records["step_update"]["step_0"] = {"loss_metrics": compute_all_metrics_for(a)}

    # PGD 循环
    for step in range(num_steps):
        x_pert = a + delta
        G_out, T_out = get_pair_outputs(x_pert)
        mse_l, dtw_l, sdtw_l = compute_three_losses(G_out, T_out)

        # 选取哪种 loss 来反传
        if loss_type == "mse":
            loss_to_back = mse_l
        elif loss_type == "softdtw":
            loss_to_back = sdtw_l
        else:
            # dtw 不能微分，这里如果选 dtw，就直接报错或跳过
            raise ValueError("DTW 不能反传")

        if delta.grad is not None:
            delta.grad.zero_()
        loss_to_back.backward()
        grad = delta.grad.clone().detach()
        delta.grad.zero_()

        # 更新 delta
        if norm in ['inf', float('inf'), np.inf]:
            delta.data.add_(alpha * torch.sign(grad))
            delta.data.copy_(delta.data.clamp(-epsilon, epsilon))
        elif norm in ['2', 2]:
            delta.data.add_(alpha * grad / (torch.norm(grad, p=2) + 1e-15))
            n2 = torch.norm(delta.data, p=2)
            if n2 > epsilon:
                delta.data.mul_(epsilon / n2)
        else:
            raise ValueError("Unsupported norm")

        # 记录指标
        grad_records["step_update"][f"step_{step+1}"] = {"loss_metrics": compute_all_metrics_for(a + delta)}

    return grad_records

# ---------------- 主流程 ----------------
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])

def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    get_gpu_info()

    model_path = config.model_path
    nu = float(re.findall(r'nu([a-zA-Z0-9|.]+)', model_path)[0])
    num_records = config.num_records
    inputs = config.inputs

    def convert_to_cpu_serializable(d):
        if isinstance(d, dict):
            return {k: convert_to_cpu_serializable(v) for k, v in d.items()}
        if isinstance(d, list):
            return [convert_to_cpu_serializable(v) for v in d]
        if isinstance(d, torch.Tensor):
            return d.cpu().detach().numpy()
        return d

    grf_params = {
        'dim': 1, 'nx': 1024, 'kernel': 'gaussian',
        'kernel_params': {'correlation_length': 0.03},
        'bc': 'periodic', 'seed': 3003, 'zero_mean': False
    }
    burgers_params = {'nu': nu, 'simulation_time': 1.0, "step": 0.0001}
    dataset_params = {"num_record": num_records}

    shape = (grf_params["nx"],)
    s = 1024
    sub = grf_params["nx"] // s
    solver_name = re.findall(r'solver=([a-zA-Z0-9]+)', model_path)[0]

    model = FNO1d(modes=16, width=64)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model = model.to(device=device, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad = False

    current_path = Path(__file__).resolve().parent.parent
    N = re.search(r'(?:^|_)N(\d+)(?:_|$)', config.dict_input_path).group(1)
    gradient_folder = current_path / "perturbation_methods" / f"nu={nu}_PGD_results" / f"approxN{N}"
    gradient_folder.mkdir(parents=True, exist_ok=True)

    x_dict = torch.load(config.dict_input_path)[:, ::sub].to(device, dtype=torch.float32)
    y_dict = torch.load(config.dict_output_path)[:, ::sub].to(device, dtype=torch.float32)

    gradient_records = {}

    loss_types = ["mse", "softdtw"]  # 如果你还要 dtw 反传的话，需要改法，不推荐
    modes = ["with_solver", "without_solver", "approximate"]

    for norm, epsilon, num_steps, alpha in inputs:
        for i in range(dataset_params["num_record"]):
            print("=== Sample", i, "norm", norm, "epsilon", epsilon, "alpha", alpha, "num_steps", num_steps)
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
                raise ValueError("Unsupported solver")

            t_span = (0, burgers_params["simulation_time"])
            PDE_func = jax.jit(lambda u0: solver.solve(
                u0, t_final=burgers_params["simulation_time"],
                t_eval=t_span, step=burgers_params["step"]
            )[1][1])

            sample_results = {}
            for lt in loss_types:
                for md in modes:
                    combo = f"{lt}__{md}"
                    print(f"开始攻击组合 {combo}")
                    t0 = time.time()
                    rec = single_attack_pgd(
                        a=a_torch,
                        G=model,
                        g=PDE_func,
                        x_dict=x_dict,
                        y_dict=y_dict,
                        epsilon=epsilon,
                        alpha=alpha,
                        num_steps=num_steps,
                        norm=norm,
                        loss_type=lt,
                        mode=md,
                        fd_delta=1e-8,
                        enable_fd=False,
                        use_double=False
                    )
                    t1 = time.time()
                    print(f"组合 {combo} 用时 {t1 - t0:.3f} 秒")
                    sample_results[combo] = rec

            gradient_records[(f"norm_{norm}", f"index_{i}", f"numsteps_{num_steps}",
                               f"epsilon_{epsilon}", f"alpha_{alpha:.5f}")] = convert_to_cpu_serializable(sample_results)

    filename = f"gradient_test_{solver_name}_nu{nu}_dtw.pkl"
    with open(gradient_folder / filename, "wb") as f:
        pickle.dump(gradient_records, f)
    print("保存到", gradient_folder / filename)

if __name__ == "__main__":
    app.run(main)


