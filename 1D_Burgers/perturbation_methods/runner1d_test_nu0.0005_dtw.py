from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from tqdm import tqdm
import time
import datetime as _dt
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO1d import FNO1d
from GRFs.generateGRFs import GRFGenerator
from solvers.burgers1d_solvers import *
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from pathlib import Path
import subprocess

jax.config.update("jax_enable_x64", True)

# 仅保留可微 SoftDTW
from tslearn.metrics import SoftDTWLossPyTorch  # 仅可微 SoftDTW


# ----------------- 打印与计时工具 -----------------
class Tee:
    """把 stdout/stderr 同步写到文件与控制台"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

class Timer:
    """简单计时工具，支持分阶段记录"""
    def __init__(self):
        self._t0 = time.perf_counter()
        self._last = self._t0
        self.stamps = []
    def lap(self, label):
        now = time.perf_counter()
        self.stamps.append((label, now - self._last))
        self._last = now
    def total(self):
        return time.perf_counter() - self._t0
    def to_string(self, prefix="  "):
        parts = [f"{prefix}{lbl:<20s}: {dt*1000:.2f} ms" for lbl, dt in self.stamps]
        parts.append(f"{prefix}{'TOTAL':<20s}: {self.total()*1000:.2f} ms")
        return "\n".join(parts)


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
            # 修复：缓存正确的 jax_vjp
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
                      norm, loss_type, mode, fd_delta, enable_fd, use_double,
                      verbose_log_file=None):
    dtype = torch.float64 if use_double else torch.float32
    G = G.to(dtype)
    a = a.to(dtype)

    sdtw_loss_fn = SoftDTWLossPyTorch(gamma=1.0, normalize=False).to(a.device)

    delta = torch.zeros_like(a, requires_grad=True, dtype=dtype)

    records = {"step": {}}

    # helper functions
    def get_pair_outputs(input_tensor, timer=None):
        """
        返回三元组 (G_out, T_used_for_loss, T_record_fullsolver)
        - G_out: 模型的输出（tensor）
        - T_used_for_loss: 根据 mode 决定，用于反传/攻击的目标（可能 detached 或 approximate）
        - T_record_fullsolver: 始终用完整 solver 计算（不 detach），用于记录并保存到日志
        """
        if timer: timer.lap("pre-G-forward")
        G_out = G(input_tensor)
        if timer: timer.lap("G-forward")

        # 完整 solver ground truth
        T_full = JaxPDEWrapper.apply(input_tensor, g)
        if timer: timer.lap("T_full-solver")

        # 根据 mode 决定 loss 用目标
        if mode == "with_solver":
            T_used = T_full
        elif mode == "without_solver":
            T_used = T_full.detach()
        elif mode == "approximate":
            T_used = find_closest_ground_truth(input_tensor, x_dict, y_dict)
        else:
            raise ValueError("Unknown mode")

        # 对齐 dtype 与 device
        if T_used.dtype != G_out.dtype:
            T_used = T_used.to(G_out.dtype)
        if T_used.device != G_out.device:
            T_used = T_used.to(G_out.device)

        if T_full.dtype != G_out.dtype:
            T_full = T_full.to(G_out.dtype)
        if T_full.device != G_out.device:
            T_full = T_full.to(G_out.device)

        return G_out, T_used, T_full

    def compute_losses(G_out, T_out, timer=None):
        G_flat = G_out.squeeze()
        T_flat = T_out.squeeze()
        mse_loss = F.mse_loss(G_flat, T_flat)
        if timer: timer.lap("loss-MSE")

        G_seq = G_flat.unsqueeze(0).unsqueeze(-1).to(a.device, dtype=dtype)
        T_seq = T_flat.unsqueeze(0).unsqueeze(-1).to(a.device, dtype=dtype)
        softdtw_loss = sdtw_loss_fn(G_seq, T_seq).squeeze()
        if timer: timer.lap("loss-SoftDTW")

        return mse_loss, softdtw_loss

    # ---------- step_0 记录 ----------
    timer0 = Timer()
    inp0 = a.clone().detach().cpu().numpy()
    G0, T0_used, T0_full = get_pair_outputs(a, timer=timer0)
    mse0, sdtw0 = compute_losses(G0, T0_used, timer=timer0)
    timer0.lap("compute-losses")

    G0_np = G0.clone().detach().cpu().numpy()
    T0_np = T0_full.clone().detach().cpu().numpy()
    records["step"]["step_0"] = {
        "input": inp0,
        "G_out": G0_np,
        "T_out": T0_np,
        "loss": {"mse": float(mse0.item()), "softdtw": float(sdtw0.item())}
    }
    if verbose_log_file:
        print("[step_0 timings]\n" + timer0.to_string("    "), file=verbose_log_file)

    # ---------- PGD 迭代 ----------
    step_iter = tqdm(range(1, num_steps + 1), desc="PGD steps", leave=False)
    for step in step_iter:
        st = Timer()

        # 前向：G 与 T_full / T_used
        x_pert = a + delta
        st.lap("build-x_pert")
        G_out, T_used, T_full = get_pair_outputs(x_pert, timer=st)

        # 计算损失（对 T_used）
        mse_l, sdtw_l = compute_losses(G_out, T_used, timer=st)

        # 选择反传目标
        loss_to_back = mse_l if loss_type == "mse" else sdtw_l

        # 反传
        if delta.grad is not None:
            delta.grad.zero_()
        st.lap("pre-backward")
        loss_to_back.backward()
        st.lap("backward")
        grad = delta.grad.clone().detach()
        delta.grad.zero_()
        st.lap("grab-grad")

        # δ 更新
        if norm in ['inf', float('inf'), np.inf]:
            delta.data.add_(alpha * torch.sign(grad))
            delta.data.copy_(delta.data.clamp(-epsilon, epsilon))
        elif norm in ['2', 2]:
            denom = (torch.norm(grad, p=2) + 1e-15)
            delta.data.add_(alpha * grad / denom)
            n2 = torch.norm(delta.data, p=2)
            if n2 > epsilon:
                delta.data.mul_(epsilon / n2)
        else:
            raise ValueError(f"Unsupported norm: {norm}")
        st.lap("delta-update")

        # 统计信息
        with torch.no_grad():
            delta_inf = float(torch.norm(delta, p=float('inf')).item())
            delta_2 = float(torch.norm(delta, p=2).item())
            grad_2 = float(torch.norm(grad, p=2).item())
            gmin = float(torch.min(G_out).item())
            gmax = float(torch.max(G_out).item())

        # 控制台精简摘要
        step_iter.set_postfix(
            loss=loss_type,
            mse=f"{mse_l.item():.3e}",
            sdtw=f"{sdtw_l.item():.3e}",
            d2=f"{delta_2:.3e}",
            dinf=f"{delta_inf:.3e}"
        )

        # 详细计时写日志
        if verbose_log_file:
            print(f"[step_{step:03d}] "
                  f"loss_type={loss_type}  mse={mse_l.item():.6e}  sdtw={sdtw_l.item():.6e}  "
                  f"||grad||2={grad_2:.6e}  ||delta||2={delta_2:.6e}  ||delta||inf={delta_inf:.6e}  "
                  f"G[min,max]=[{gmin:.4e},{gmax:.4e}]",
                  file=verbose_log_file)
            print(st.to_string("    "), file=verbose_log_file)

        # 记录（T_out 一律为完整 solver 的输出）
        inp = (a + delta).clone().detach().cpu().numpy()
        G_np = G_out.clone().detach().cpu().numpy()
        T_np = T_full.clone().detach().cpu().numpy()
        records["step"][f"step_{step}"] = {
            "input": inp,
            "G_out": G_np,
            "T_out": T_np,
            "loss": {"mse": float(mse_l.item()), "softdtw": float(sdtw_l.item())}
        }

    return records


# ---------------- 主流程 ----------------
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])


def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir

    # ---- 准备日志（同时输出到终端与文件）----
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = workdir if workdir else "."
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"runlog_{ts}.txt")
    logfile = open(log_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.stdout, logfile)
    sys.stderr = Tee(sys.stderr, logfile)
    print(f"[LOG] 日志文件: {log_path}")

    # ---- 环境信息 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    get_gpu_info()

    # ---- 解析配置 ----
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

    # ---- 模型加载 ----
    print("[LOAD] Loading model:", model_path)
    t_load = Timer()
    model = FNO1d(modes=16, width=64)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model = model.to(device=device, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad = False
    print(t_load.to_string("    "))

    # ---- 输出目录 ----
    current_path = Path(__file__).resolve().parent.parent
    N = re.search(r'(?:^|_)N(\d+)(?:_|$)', config.dict_input_path).group(1)
    gradient_folder = current_path / "perturbation_methods" / f"nu={nu}_PGD_results" / f"approxN{N}"
    gradient_folder.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] Save folder: {gradient_folder}")

    # ---- 载入字典 ----
    print("[LOAD] Loading dict tensors ...")
    t_dict = Timer()
    x_dict = torch.load(config.dict_input_path)[:, ::sub].to(device, dtype=torch.float32)
    y_dict = torch.load(config.dict_output_path)[:, ::sub].to(device, dtype=torch.float32)
    print(t_dict.to_string("    "))

    gradient_records = {}

    # 仅保留这两种优化目标
    loss_types = ["mse", "softdtw"]
    modes = ["with_solver", "without_solver", "approximate"]

    # ---- 外层循环加进度条 ----
    cfg_bar = tqdm(inputs, desc="Configs", leave=True)
    for norm, epsilon, num_steps, alpha in cfg_bar:
        cfg_bar.set_postfix(norm=str(norm), eps=epsilon, steps=num_steps, alpha=f"{alpha:.3e}")

        smp_bar = tqdm(range(dataset_params["num_record"]), desc="Samples", leave=True)
        for i in smp_bar:
            smp_bar.set_postfix(sample=i)

            print(f"\n=== Sample {i} | norm={norm} epsilon={epsilon} alpha={alpha} num_steps={num_steps} ===")
            t_sample = Timer()

            # 生成输入
            t_gen = Timer()
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
            print("[GEN]\n" + t_gen.to_string("    "))

            # 构造 solver
            t_solver = Timer()
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
                t_eval=t_span, step=burgers_params['step']
            )[1][1])
            print("[SOLVER-BUILD]\n" + t_solver.to_string("    "))

            sample_results = {}
            for lt in tqdm(loss_types, desc="Loss types", leave=False):
                for md in tqdm(modes, desc="Modes", leave=False):
                    combo = f"{lt}__{md}"
                    print(f"\n[ATTACK] combo={combo}")
                    t_attack = Timer()

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
                        use_double=False,
                        verbose_log_file=logfile
                    )
                    print("[ATTACK DONE]\n" + t_attack.to_string("    "))
                    sample_results[combo] = rec

            # 样本级别累计时间
            print("[SAMPLE TOTAL]\n" + t_sample.to_string("    "))

            key = (f"norm_{norm}", f"index_{i}", f"numsteps_{num_steps}",
                   f"epsilon_{epsilon}", f"alpha_{alpha:.5f}")
            gradient_records[key] = convert_to_cpu_serializable(sample_results)

    filename = f"gradient_test_{solver_name}_nu{nu}_dtw.pkl"
    out_path = gradient_folder / filename
    print("[SAVE] Writing pickle:", out_path)
    t_save = Timer()
    with open(out_path, "wb") as f:
        pickle.dump(gradient_records, f)
    print(t_save.to_string("    "))
    print("保存到", out_path)
    print(f"[LOG] 详细日志保存在: {log_path}")


if __name__ == "__main__":
    app.run(main)



