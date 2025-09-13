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
from tqdm import tqdm
from pathlib import Path
import pickle
import numpy as np
import jax
import jax.numpy as jnp  # === NEW: jnp 在回退路径里会用到

import torch.nn.functional as F
from scipy.spatial.distance import cosine
from torch.autograd.functional import jacobian

jax.config.update("jax_enable_x64", True)
import subprocess


def get_gpu_info():
    try:
        # 获取当前分区（Slurm 环境变量）
        partition = os.environ.get("SLURM_JOB_PARTITION", "N/A")

        # 获取当前节点名
        node_name = os.environ.get("SLURMD_NODENAME", "N/A")

        # 获取当前使用的 GPU ID（如果用 --gres）
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")

        # 调用 nvidia-smi 获取详细信息
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
        # if a_torch.dtype != torch.float64:
        #     raise ValueError("Input must be torch.float64 for gradcheck compatibility")

        try:
            # PyTorch → JAX
            a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
            a_jax = jax.dlpack.from_dlpack(a_dlpack)

            # 运行预编译函数
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

        jitted_vjp = JaxPDEWrapper._vjp_cache[cache_key]

        try:
            grad_dlpack = torch.utils.dlpack.to_dlpack(grad_output.contiguous())
            grad_jax = jax.dlpack.from_dlpack(grad_dlpack)

            a_dlpack = torch.utils.dlpack.to_dlpack(a_torch.contiguous())
            a_jax = jax.dlpack.from_dlpack(a_dlpack)

            grad_input_jax, = jitted_vjp(a_jax, grad_jax)
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


# ====================== 自适应 dt 选择器（JIT 缓存 + 稳定性检查） ======================
class AdaptiveJittedPDE:
    """
    为给定 solver 构造一组 dt 候选；每次根据当前输入选择“稳定”的 dt，并返回
    对应的已 jitted 纯函数 g_fixed(u0)->y。稳定性优先用 checkify.float_checks，
    不可用时回退到 isfinite 检查。
    仅为少数候选 dt 编译一次并缓存，避免频繁重编译。
    """
    def __init__(self, solver, *, t_final: float, base_step: float,
                 shrink_seq=(1.0, 0.5, 0.1, 0.01), use_checkify=True):
        self.solver = solver
        self.t_final = float(t_final)
        self.base_step = float(base_step)
        self.candidates = [float(self.base_step * s) for s in shrink_seq]
        self._cache = {}  # dt -> (jitted_fun, checked_fun or None)
        self._use_checkify = use_checkify
        self._checkify = None
        if use_checkify:
            try:
                from jax.experimental import checkify
                self._checkify = checkify
            except Exception:
                self._use_checkify = False

    def _compile_for_dt(self, dt: float):
        if dt not in self._cache:
            t_final = self.t_final

            # 与你原代码一致的输出索引：solve(...)[1][1]
            def _run(u0):
                return self.solver.solve(u0, t_final=t_final,
                                         t_eval=(0.0, t_final), step=dt)[1][1]
            jitted = jax.jit(_run)
            checked = None
            if self._use_checkify:
                checked = self._checkify.checkify(jitted, errors=self._checkify.float_checks)
            self._cache[dt] = (jitted, checked)
        return self._cache[dt]

    def select(self, torch_input: torch.Tensor):
        """
        基于当前输入，挑选一个“稳定”的 jitted 函数返回。
        注意：这里会在 JAX 侧执行一次以检测稳定性；真正用于反传的是返回的 jitted fun。
        """
        a_dl = torch.utils.dlpack.to_dlpack(torch_input.contiguous())
        a_jax = jax.dlpack.from_dlpack(a_dl)

        last_exc = None
        for dt in self.candidates:
            jitted, checked = self._compile_for_dt(dt)
            try:
                if checked is not None:
                    err, y = checked(a_jax)
                    if not err.get():  # 无 NaN/Inf
                        return jitted
                else:
                    y = jitted(a_jax)
                    if bool(jnp.all(jnp.isfinite(y))):
                        return jitted
            except Exception as e:
                last_exc = e
                continue

        # 所有候选都失败：回退到最小 dt 的 jitted（上层可决定是否继续/报错）
        if last_exc is not None:
            print(f"[AdaptiveJittedPDE] All candidates failed; last error: {last_exc!r}")
        return self._compile_for_dt(self.candidates[-1])[0]


def find_closest_ground_truth(a_perturbed, x_dict, y_dict):
    # Ensure a_perturbed has the correct shape for broadcasting
    # If a_perturbed is [256], reshape it to [1, 256]
    if len(a_perturbed.shape) == 1:
        a_perturbed = a_perturbed.unsqueeze(0)  # Shape: [1, 256]

    # Compute distances using broadcasting
    a_perturbed = a_perturbed.squeeze(-1)
    distances = torch.norm(x_dict - a_perturbed, p=2, dim=1)  # Shape: [20000]

    # Find the closest index
    closest_idx = torch.argmin(distances)

    # Return the corresponding y_dict value
    return y_dict[closest_idx]


def compare_gradient_attack(a, G, g_adaptive, x_dict, y_dict, epsilon, alpha, num_steps, norm='inf',
                            fd_delta=1e-8, enable_fd=False, use_double=True):
    original_dtypes = {name: param.dtype for name, param in G.named_parameters()}

    if use_double:
        dtype = torch.float64
        G = G.to_dtype(torch.float64)
    else:
        dtype = torch.float32
        G = G.to_dtype(torch.float32)

    a = a.to(dtype)
    # g_adaptive 是选择器对象，不需要 .to(dtype)

    def convert_data(data):
        if isinstance(data, torch.Tensor):
            return data.to(dtype)
        elif isinstance(data, dict):
            return {k: convert_data(v) for k, v in data.items()}
        return data

    x_dict = convert_data(x_dict)
    y_dict = convert_data(y_dict)

    delta_with = torch.zeros_like(a, requires_grad=True, dtype=dtype)
    delta_without = torch.zeros_like(a, requires_grad=True, dtype=dtype)
    delta_approximate = torch.zeros_like(a, requires_grad=True, dtype=dtype)
    grad_records = {}
    grad_records["step_update"] = {}

    def wrap_with_solver(input):
        # Initialize timing dict
        timing = {}

        # Time G computation
        start = time.time()
        G_output = G(input)
        timing['G_time'] = time.time() - start

        # Time JaxPDEWrapper (g) computation
        # === 自适应选择稳定 dt 的 jitted 函数
        g_fixed = g_adaptive.select(input)
        start = time.time()
        g_output = JaxPDEWrapper.apply(input, g_fixed)
        timing['g_time'] = time.time() - start

        # Time norm computation
        start = time.time()
        loss = torch.norm(G_output - g_output, p=2)**2
        timing['norm_time'] = time.time() - start

        return loss, timing

    def wrap_without_solver(input):
        # Initialize timing dict
        timing = {}

        # Time G computation
        start = time.time()
        G_output = G(input)
        timing['G_time'] = time.time() - start

        # Time JaxPDEWrapper (g) computation
        g_fixed = g_adaptive.select(input)
        start = time.time()
        g_output = JaxPDEWrapper.apply(input, g_fixed)
        timing['find_time'] = time.time() - start

        # Time norm computation
        start = time.time()
        loss = torch.norm(G_output - g_output.detach(), p=2)**2
        timing['norm_time'] = time.time() - start

        return loss, timing

    def wrap_approximate(input):
        # Initialize timing dict
        timing = {}

        # Time G computation
        start = time.time()
        G_output = G(input)
        timing['G_time'] = time.time() - start

        # Time find_closest_ground_truth computation
        start = time.time()
        closest_output = find_closest_ground_truth(input, x_dict, y_dict)
        timing['find_time'] = time.time() - start

        # Time norm computation
        start = time.time()
        loss = torch.norm(G_output - closest_output, p=2)**2
        timing['norm_time'] = time.time() - start

        return loss, timing

    def compute_all_loss_metrics(a, delta_with, delta_without, delta_approximate, G, g_adaptive):
        return {
            'delta_with': wrap_with_solver(a + delta_with)[0].item(),
            'delta_without_solver': wrap_with_solver(a + delta_without)[0].item(),
            'delta_approximate': wrap_with_solver(a + delta_approximate)[0].item(),
            'delta_approximate_surrogate': wrap_approximate(a + delta_approximate)[0].item(),
        }

    all_timing_metrics = []

    old_a = a.clone()

    current_after = compute_all_loss_metrics(a, delta_with, delta_without, delta_approximate, G, g_adaptive)
    step_metrics = {}
    step_metrics_add = {
        'loss_metrics': {
            'with_solver': current_after['delta_with'],
            'without_solver': current_after['delta_without_solver'],
            'approximate': current_after['delta_approximate'],
            'approximate_surrogate': current_after['delta_approximate_surrogate'],
        }
    }
    step_metrics.update(step_metrics_add)
    grad_records["step_update"][f"step_0"] = step_metrics

    for step in range(num_steps):
        step_metrics = {}
        timing_metrics = {"with_solver": {}, "without_solver": {}, "approximate": {}}

        # 1. Compute gradient with solver (with timing)
        a_with = a + delta_with
        start_forward = time.time()
        loss_with, forward_timing_with = wrap_with_solver(a_with)
        timing_metrics["with_solver"]['with_solver_forward_time'] = time.time() - start_forward
        timing_metrics["with_solver"].update({f'with_solver_{k}': v for k, v in forward_timing_with.items()})

        start_backward = time.time()
        loss_with.backward()
        timing_metrics["with_solver"]['with_solver_backward_time'] = time.time() - start_backward

        grad_with = delta_with.grad.clone().detach()
        delta_with.grad.zero_()

        # 2. Compute gradient without solver (with timing)
        a_without = a + delta_without
        start_forward = time.time()
        loss_without, forward_timing_without = wrap_without_solver(a_without)
        timing_metrics["without_solver"]['without_solver_forward_time'] = time.time() - start_forward
        timing_metrics["without_solver"].update({f'without_solver_{k}': v for k, v in forward_timing_without.items()})

        start_backward = time.time()
        loss_without.backward()
        timing_metrics["without_solver"]['without_solver_backward_time'] = time.time() - start_backward

        grad_without = delta_without.grad.clone().detach()
        delta_without.grad.zero_()

        # 3. Compute gradient approximate (with timing)
        a_approximate = a + delta_approximate
        start_forward = time.time()
        loss_approximate, forward_timing_approximate = wrap_approximate(a_approximate)
        timing_metrics["approximate"]['approximate_forward_time'] = time.time() - start_forward
        timing_metrics["approximate"].update({f'approximate_{k}': v for k, v in forward_timing_approximate.items()})

        start_backward = time.time()
        loss_approximate.backward()
        timing_metrics["approximate"]['approximate_backward_time'] = time.time() - start_backward

        grad_approximate = delta_approximate.grad.clone().detach()
        delta_approximate.grad.zero_()

        # Update perturbations
        def update(delta, grad):
            if norm == 'inf':
                delta.data.add_(alpha * torch.sign(grad))
                pre_clip = delta.data.clone()
                clipped = pre_clip.clamp(-epsilon, epsilon)
                num_total = delta.numel()
                num_clipped = (pre_clip != clipped).sum().item()
                clipped_percentage = num_clipped / num_total
                delta.data.copy_(clipped)
                return clipped_percentage  # float between 0 and 1
            elif norm == 2:
                delta.data.add_(alpha * grad / (torch.norm(grad, p=2) + 1e-15))
                l2_norm = torch.norm(delta.data, p=2)
                radius = epsilon
                exceeded = l2_norm > radius
                if exceeded:
                    delta.data = delta.data * (radius / l2_norm)
                return exceeded  # Boolean

        # 2. Update perturbations
        with_boundary = update(delta_with, grad_with)
        without_boundary = update(delta_without, grad_without)
        approximate_boundary = update(delta_approximate, grad_approximate)

        # 3. Compute ALL losses AFTER update
        current_after = compute_all_loss_metrics(a, delta_with, delta_without, delta_approximate, G, g_adaptive)

        grad_records["step_update"][f"step_{step+1}"] = {}
        grad_records["step_update"][f"step_{step+1}"]["values"] = {}

        grad_records["step_update"][f"step_{step+1}"]["values"]["a_values"] = {}
        grad_records["step_update"][f"step_{step+1}"]["values"]["a_values"]['a'] = old_a.squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["a_values"]['a_with'] = a_with.squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["a_values"]['a_without'] = a_without.squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["a_values"]['a_approximate'] = a_approximate.squeeze().clone().detach()

        grad_records["step_update"][f"step_{step+1}"]["values"]["G_values"] = {}
        grad_records["step_update"][f"step_{step+1}"]["values"]["G_values"]['G(a)'] = G(old_a).squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["G_values"]['G(a_with)'] = G(a_with).squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["G_values"]['G(a_without)'] = G(a_without).squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["G_values"]['G(a_approximate)'] = G(a_approximate).squeeze().clone().detach()

        grad_records["step_update"][f"step_{step+1}"]["values"]["g_values"] = {}
        grad_records["step_update"][f"step_{step+1}"]["values"]["g_values"]['g(a)'] = \
            JaxPDEWrapper.apply(old_a, g_adaptive.select(old_a)).squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["g_values"]['g(a_with)'] = \
            JaxPDEWrapper.apply(a_with, g_adaptive.select(a_with)).squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["g_values"]['g(a_without)'] = \
            JaxPDEWrapper.apply(a_without, g_adaptive.select(a_without)).squeeze().clone().detach()
        grad_records["step_update"][f"step_{step+1}"]["values"]["g_values"]['g(a_approximate)'] = \
            JaxPDEWrapper.apply(a_approximate, g_adaptive.select(a_approximate)).squeeze().clone().detach()

        # Store final losses if last step
        if step == num_steps - 1:
            final_losses = current_after

            grad_records["final_values"] = {}

            grad_records["final_values"]["a_values"] = {}
            grad_records["final_values"]["a_values"]['a'] = old_a.squeeze().clone().detach()
            grad_records["final_values"]["a_values"]['a_with'] = a_with.squeeze().clone().detach()
            grad_records["final_values"]["a_values"]['a_without'] = a_without.squeeze().clone().detach()
            grad_records["final_values"]["a_values"]['a_approximate'] = a_approximate.squeeze().clone().detach()

            grad_records["final_values"]["G_values"] = {}
            grad_records["final_values"]["G_values"]['G(a)'] = G(old_a).squeeze().clone().detach()
            grad_records["final_values"]["G_values"]['G(a_with)'] = G(a_with).squeeze().clone().detach()
            grad_records["final_values"]["G_values"]['G(a_without)'] = G(a_without).squeeze().clone().detach()
            grad_records["final_values"]["G_values"]['G(a_approximate)'] = G(a_approximate).squeeze().clone().detach()

            grad_records["final_values"]["g_values"] = {}
            grad_records["final_values"]["g_values"]['g(a)'] = \
                JaxPDEWrapper.apply(old_a, g_adaptive.select(old_a)).squeeze().clone().detach()
            grad_records["final_values"]["g_values"]['g(a_with)'] = \
                JaxPDEWrapper.apply(a_with, g_adaptive.select(a_with)).squeeze().clone().detach()
            grad_records["final_values"]["g_values"]['g(a_without)'] = \
                JaxPDEWrapper.apply(a_without, g_adaptive.select(a_without)).squeeze().clone().detach()
            grad_records["final_values"]["g_values"]['g(a_approximate)'] = \
                JaxPDEWrapper.apply(a_approximate, g_adaptive.select(a_approximate)).squeeze().clone().detach()

        # Store metrics
        step_metrics_add = {
            "reaches_boundary": {
                "with_solver": with_boundary,
                "detached": without_boundary,
                "approximated": approximate_boundary,
            },
            'loss_metrics': {
                'with_solver': current_after['delta_with'],
                'without_solver': current_after['delta_without_solver'],
                'approximate': current_after['delta_approximate'],
                'approximate_surrogate': current_after['delta_approximate_surrogate'],
            },
            "timing_metrics": timing_metrics
        }
        step_metrics.update(step_metrics_add)
        grad_records["step_update"][f"step_{step+1}"].update(step_metrics)
        all_timing_metrics.append(timing_metrics)

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

    print("\n\n\n")
    print("model_path: ", model_path)
    print("workdir: ", workdir)
    print("device: ", device)
    print("nu: ", nu)
    print("jax.devices()", jax.devices())
    print("\n\n\n")

    print("===============Running===============")
    print("\n\n\n")

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
        "step": 0.001
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
    print("model_path: ", model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    current_file_path = Path(__file__).resolve().parent.parent
    N = re.search(r'(?:^|_)N(\d+)(?:_|$)', config.dict_input_path).group(1)
    gradient_folder = current_file_path / "perturbation_methods" / "nu=0.0005_PGD_results_adaptive" / f"approxN{N}"
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
                seed=grf_params['seed'] + i,  # Ensure different GRFs for each sample
                zero_mean=grf_params['zero_mean']
            )

            a = a[::sub]

            a_torch = torch.from_numpy(a).float().unsqueeze(0).unsqueeze(-1).to(device)

            if solver_name == "exponax":
                solver = ExponaxBurgersSolver1D(s, nu=burgers_params['nu'], bc=grf_params['bc'], xlim=(0, 1))
            elif solver_name == "scipy":
                solver = SciPyBurgersSolver1D(s, nu=burgers_params['nu'], bc=grf_params['bc'])
            elif solver_name == "scipy_spectral":
                solver = SciPySpectralBurgersSolver1D(s, nu=burgers_params['nu'], bc=grf_params['bc'])
            elif solver_name == "phiflow":
                solver = PhiFlowBurgersSolver1D(s, nu=burgers_params['nu'], bc=grf_params['bc'])
            else:
                raise ValueError("Specified solver not inplemented")

            t_span = (0, burgers_params['simulation_time'])

            # === 自适应 dt 选择器：按候选比例从大到小尝试，自动避开 NaN/Inf
            g_adaptive = AdaptiveJittedPDE(
                solver,
                t_final=burgers_params['simulation_time'],
                base_step=burgers_params["step"],
                shrink_seq=(1.0, 0.5, 0.1, 0.01),  # 需要更激进可改为 (1.0, 0.1, 0.01, 0.001)
                use_checkify=True
            )

            records = compare_gradient_attack(
                a_torch, model, g_adaptive, x_dict, y_dict,
                epsilon=epsilon,
                alpha=alpha,
                num_steps=num_steps,
                norm=norm,
                enable_fd=True,
                fd_delta=1e-8
            )
            gradient_records[(f"norm_{norm}", f"index_{i}", f"numsteps_{num_steps}", f"epsilon_{epsilon}", f"alpha_{alpha:.5f}")] = convert_to_cpu_serializable(records)
            total_time = time.time() - start_time
            print(" \n     index:", i, ",    norm:", norm, ",    num_steps:", num_steps, ",    epsilon:", epsilon, ",    alpha:", alpha, ",    total time:", total_time)

    gradient_filename = f"gradient_test_{solver_name}_nu{nu}_nsamples{num_records}.pkl"
    with open(gradient_folder / gradient_filename, "wb") as f:
        pickle.dump(gradient_records, f)
    print(f"Pickle file saved to {gradient_folder}/{gradient_filename}")


if __name__ == "__main__":
    app.run(main)
