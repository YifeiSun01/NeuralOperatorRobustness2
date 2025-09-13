#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative Train→Attack→Replace for 1D Burgers (final-time only, with-solver PGD)

- 每轮流程：
  1) 用当前数据集训练 FNO1d（默认 500 epochs）
  2) 选取 attack_ratio 比例的样本做 PGD（loss=||G(a)-g(a,t_final)||^2，仅最终帧）
  3) 用攻击后的样本替换这部分老数据，形成“下一轮训练数据集”
  4) 保存本轮模型和数据集，继续下一轮（用本轮的模型与数据集作为起点）
- 数据集键约定：{'x': (N,X)[或(N,X,1)], 'y': (N,X), 't_final': float}
- 攻击参数通过 --inputs="norm,epsilon,steps,alpha" 指定（可给 1 组或 K 组）
- 例子：
  python iterative_train_attack_replace.py \\
    --dataset_pt ./datasets/1D/Burgers/pos/dim1d_nx1024_...pt \\
    --rounds 3 --attack_ratio 0.3 \\
    --inputs "2,10,100,0.01" \\
    --solver exponax --nu 0.0005 --t_final 1.0 --dt 0.001

依赖：
- models/FNO1d.py  提供 FNO1d
- utilities3.py    提供 LpLoss, count_params
- solvers/burgers1d_solvers.py  提供 ExponaxBurgersSolver1D 等（.solve 返回 (times, states)）
"""

import os, sys, time, math, random, json, subprocess
from pathlib import Path
from datetime import datetime

# 让 JAX 用 CUDA
os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")

from absl import app, flags
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import jax
import jax.numpy as jnp

# 项目内导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO1d import FNO1d
from utilities3 import LpLoss, count_params
from solvers.burgers1d_solvers import (
    ExponaxBurgersSolver1D,
    SciPyBurgersSolver1D,
    SciPySpectralBurgersSolver1D,
    PhiFlowBurgersSolver1D,
)

# JAX 用 float32
jax.config.update("jax_enable_x64", False)

# === 固定输出到脚本同级 results 目录 ===
CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"

FLAGS = flags.FLAGS
# 主要数据与输出
flags.DEFINE_string("dataset_pt", None, "初始训练数据（.pt，含 'x','y'）")
# 下两项 flag 为兼容保留，但本脚本会统一写到 results/ 下（可忽略）
flags.DEFINE_string("out_models_root", "./saved_models/1D/iterative_attack", "(unused) models root")
flags.DEFINE_string("out_datasets_root", "./datasets/1D/Burgers/expanded/iterative", "(unused) datasets root")

# 训练超参
flags.DEFINE_integer("rounds", 3, "总轮数 K")
flags.DEFINE_integer("epochs", 500, "每轮训练 epochs")
flags.DEFINE_integer("batch_size", 20, "batch size")
flags.DEFINE_float("lr", 1e-3, "初始学习率")
flags.DEFINE_float("weight_decay", 1e-4, "权重衰减")
flags.DEFINE_integer("modes", 16, "FNO1d modes")
flags.DEFINE_integer("width", 64, "FNO1d width")

# 训练/测试划分（沿用你原来的习惯：前 ntrain 训练，后 ntest 测试）
flags.DEFINE_integer("ntrain", 1000, "训练样本数（若数据不足则取可用上限）")
flags.DEFINE_integer("ntest", 100, "测试样本数（若数据不足则取可用上限）")

# 攻击与替换
flags.DEFINE_float("attack_ratio", 0.3, "每轮攻击并替换的比例（0~1）")
# 可重复：--inputs="inf,0.1,100,0.001"（若只给 1 组则每轮共用；若给 K 组则各轮分别使用）
flags.DEFINE_multi_string("inputs", None, "PGD 参数：norm,epsilon,steps,alpha，如 '2,10,100,0.01'")

# PDE/求解器
flags.DEFINE_string("solver", "exponax", "exponax|scipy|scipy_spectral|phiflow")
flags.DEFINE_float("nu", 0.0005, "Burgers 粘性系数")
flags.DEFINE_float("t_final", 1.0, "终止时间")
flags.DEFINE_float("dt", 0.001, "求解步长")

# 其它
flags.DEFINE_integer("seed", 2025, "随机种子")
flags.DEFINE_bool("use_cuda", True, "是否使用 CUDA（若可用）")


# ----------------------------- 工具函数 -----------------------------
def get_gpu_info():
    try:
        part = os.environ.get("SLURM_JOB_PARTITION", "N/A")
        node = os.environ.get("SLURMD_NODENAME", "N/A")
        gvis = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")
        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
                encoding="utf-8"
            ).strip()
        except Exception:
            smi = "nvidia-smi not found or not available"
        print("========== GPU ==========")
        print(f"Partition: {part}")
        print(f"Node: {node}")
        print(f"CUDA_VISIBLE_DEVICES: {gvis}")
        print("GPU Details:\n" + smi)
        print("=========================")
    except Exception as e:
        print(f"GPU info error: {e}")


def parse_inputs_list(inputs_flag_values):
    """解析 --inputs 'norm,epsilon,steps,alpha'（可给 1 组或 K 组）"""
    if not inputs_flag_values:
        raise ValueError("请至少提供一组 --inputs='norm,epsilon,steps,alpha'")
    combos = []
    for s in inputs_flag_values:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError(f"--inputs 需 4 个逗号分隔值，得到：{s}")
        norm_s, eps_s, steps_s, alpha_s = parts
        norm_val = norm_s.lower()
        if norm_val in ("linf", "inf", "∞"):
            norm_val = "inf"
        elif norm_val in ("2", "2.0"):
            norm_val = "2"
        else:
            try:
                if int(float(norm_s)) == 2:
                    norm_val = "2"
            except Exception:
                pass
        epsilon = float(eps_s)
        steps = int(float(steps_s))
        alpha = float(alpha_s)
        combos.append((norm_val, epsilon, steps, alpha))
    return combos


def fmt_float(v: float) -> str:
    return f"{v:.6g}"


def ensure_x_y_shapes(x, y):
    """
    x: (N,X) or (N,X,1) → 返回 (N,X) float32
    y: (N,X)            → 返回 (N,X) float32
    """
    if x.dim() == 3:
        assert x.shape[-1] == 1, f"x shape {tuple(x.shape)} 最后一维应为 1"
        x = x[..., 0]
    assert y.dim() == 2, f"期望 y 为 (N,X)，得到 {tuple(y.shape)}"
    return x.contiguous().float(), y.contiguous().float()


def make_dataloaders(x, y, ntrain, ntest, batch_size):
    """
    x,y: (N,X) float32
    - 训练：前 ntrain
    - 测试：后 ntest
    输入给 FNO1d 前会 unsqueeze(-1) → (B,X,1)
    """
    N = x.shape[0]
    ntrain = min(ntrain, N)
    ntest = min(ntest, max(0, N - ntrain))
    x_train = x[:ntrain]
    y_train = y[:ntrain]
    x_test = x[N - ntest:] if ntest > 0 else x[:0]
    y_test = y[N - ntest:] if ntest > 0 else y[:0]

    x_train_in = x_train.unsqueeze(-1)
    x_test_in = x_test.unsqueeze(-1)

    tr_loader = DataLoader(TensorDataset(x_train_in, y_train), batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(TensorDataset(x_test_in, y_test), batch_size=batch_size, shuffle=False)
    return tr_loader, te_loader, ntrain, ntest


# --------------------------- JAX-PDE 适配（修复版） ---------------------------
class JaxPDEWrapper(torch.autograd.Function):
    """把 JAX 的 g(a) 包成 Torch 可反传算子（通过 vjp）。a:(X,) → g(a):(X,)"""

    _vjp_cache = {}

    @staticmethod
    def _torch_to_jax(a_torch: torch.Tensor):
        # 关键修复：dlpack 交接时使用 detach().contiguous()
        dl = torch.utils.dlpack.to_dlpack(a_torch.detach().contiguous())
        return jax.dlpack.from_dlpack(dl)

    @staticmethod
    def _jax_to_torch(a_jax):
        dl = jax.dlpack.to_dlpack(a_jax)
        return torch.utils.dlpack.from_dlpack(dl)

    @staticmethod
    def forward(ctx, a_torch, g_callable):
        if not a_torch.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        # 保存原始（可带梯度）的 a_torch 以便 backward 构造 vjp
        ctx.save_for_backward(a_torch)
        ctx.g = g_callable

        a_jax = JaxPDEWrapper._torch_to_jax(a_torch)
        out_jax = g_callable(a_jax)  # (X,)
        out_t = JaxPDEWrapper._jax_to_torch(out_jax)
        return out_t

    @staticmethod
    def backward(ctx, grad_output):
        (a_torch,) = ctx.saved_tensors
        g = ctx.g

        key = id(g)
        if key not in JaxPDEWrapper._vjp_cache:
            def vjp_fn(a_jax, grad_jax):
                _, vjp = jax.vjp(g, a_jax)
                return vjp(grad_jax)
            JaxPDEWrapper._vjp_cache[key] = jax.jit(vjp_fn)
        vjp_jit = JaxPDEWrapper._vjp_cache[key]

        grad_jax = JaxPDEWrapper._torch_to_jax(grad_output)
        a_jax = JaxPDEWrapper._torch_to_jax(a_torch)

        (grad_in_jax,) = vjp_jit(a_jax, grad_jax)
        grad_in_torch = JaxPDEWrapper._jax_to_torch(grad_in_jax)
        return grad_in_torch, None


class PDEAttackSystem1D:
    """只做 with-solver、最终时刻 PGD；攻击时临时冻结模型参数，结束后可恢复"""

    def __init__(self, model_G: torch.nn.Module, g_callable, device: torch.device):
        self.G = model_G
        # 记录原始 requires_grad 标志并临时冻结
        self._flags = [p.requires_grad for p in self.G.parameters()]
        for p in self.G.parameters():
            p.requires_grad_(False)
        self.G.eval()

        self.g = g_callable
        self.device = device

    def restore(self):
        """恢复模型参数的 requires_grad 到攻击前状态"""
        for p, f in zip(self.G.parameters(), self._flags):
            p.requires_grad_(f)

    def _build_in(self, a: torch.Tensor) -> torch.Tensor:
        # FNO1d 期望 (B,X,1)
        if a.dim() == 1:
            a = a.unsqueeze(0).unsqueeze(-1)
        elif a.dim() == 2:
            a = a.unsqueeze(-1)
        return a

    def forward_G(self, a: torch.Tensor) -> torch.Tensor:
        return self.G(self._build_in(a))  # → (1,X,1)

    def forward_true_final(self, a: torch.Tensor) -> torch.Tensor:
        return JaxPDEWrapper.apply(a, self.g)  # (X,)

    def pgd(self, a0: torch.Tensor, epsilon: float, alpha: float, steps: int, norm: str = "2"):
        a0 = a0.detach().to(self.device)
        delta = torch.zeros_like(a0, requires_grad=True)
        for _ in range(steps):
            if delta.grad is not None:
                delta.grad.zero_()
            a = a0 + delta

            pred = self.forward_G(a)                 # (1,X,1)
            true_final = self.forward_true_final(a)  # (X,)

            if pred.dim() == 3:
                pred = pred[0]
            pred = pred.squeeze(-1)                  # (X,)

            loss = F.mse_loss(pred, true_final, reduction="sum")
            loss.backward()

            with torch.no_grad():
                g = delta.grad
                if str(norm).lower() in ("inf", "linf", "∞"):
                    delta += alpha * torch.sign(g)
                    delta.clamp_(-epsilon, epsilon)
                else:
                    g_norm = torch.norm(g, p=2) + 1e-12
                    delta += alpha * g / g_norm
                    d_norm = torch.norm(delta, p=2)
                    if d_norm > epsilon:
                        delta.mul_(epsilon / d_norm)
        return (a0 + delta).detach()

    @torch.no_grad()
    def rollout_final(self, a: torch.Tensor) -> torch.Tensor:
        return self.forward_true_final(a)  # (X,)


def build_solver(solver_name: str, X: int, nu: float, t_final: float, dt: float):
    if solver_name == "exponax":
        solver = ExponaxBurgersSolver1D(X, nu=nu, bc="periodic", xlim=(0, 1))
    elif solver_name == "scipy":
        solver = SciPyBurgersSolver1D(X, nu=nu, bc="periodic")
    elif solver_name == "scipy_spectral":
        solver = SciPySpectralBurgersSolver1D(X, nu=nu, bc="periodic")
    elif solver_name == "phiflow":
        solver = PhiFlowBurgersSolver1D(X, nu=nu, bc="periodic")
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    t_span = (0.0, float(t_final))

    @jax.jit
    def g_callable(u0: jnp.ndarray):
        return solver.solve(
            u0.astype(jnp.float32),
            t_final=float(t_final),
            t_eval=t_span,
            step=float(dt),
        )[1][1].astype(jnp.float32)  # 明确输出 float32

    return g_callable


# --------------------------- 训练 / 评估 ---------------------------
def train_one_round(model, tr_loader, te_loader, epochs, lr, weight_decay, device, round_idx, log_to: Path):
    # 保险：每轮开始前确保可训练
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # CosineAnnealingLR 以迭代步数为 T_max
    total_iters = epochs * max(1, len(tr_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)

    myloss = LpLoss(size_average=False)
    log_to.parent.mkdir(parents=True, exist_ok=True)
    with open(log_to, "w", encoding="utf-8") as f:
        f.write(f"Round {round_idx} training log\n")
        f.flush()

        for ep in tqdm(range(epochs), desc=f"[Round {round_idx}] Train FNO1d", dynamic_ncols=True):
            model.train()
            train_mse = 0.0
            train_l2 = 0.0
            for xb, yb in tr_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad(set_to_none=True)
                out = model(xb)  # (B,X,1)

                # 和你原脚本一致的损失
                mse = F.mse_loss(out.view(out.size(0), -1), yb.view(yb.size(0), -1), reduction="mean")
                l2 = myloss(out.view(out.size(0), -1), yb.view(yb.size(0), -1))
                l2.backward()
                optimizer.step()
                scheduler.step()

                train_mse += mse.item()
                train_l2 += l2.item()

            # eval
            model.eval()
            test_l2 = 0.0
            with torch.no_grad():
                for xb, yb in te_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    out = model(xb)
                    test_l2 += myloss(out.view(out.size(0), -1), yb.view(yb.size(0), -1)).item()

            ntrain = len(tr_loader.dataset)
            ntest = len(te_loader.dataset)
            train_mse /= max(1, len(tr_loader))
            train_l2 /= max(1, ntrain)
            test_l2 /= max(1, ntest)

            f.write(f"epoch:{ep}, train mse:{train_mse:.8f}, train l2:{train_l2:.8f}, test l2:{test_l2:.8f}\n")
            if ep % 10 == 0:
                f.flush()

    return model


# --------------------------- 攻击 / 替换 ---------------------------
def attack_replace_subset(x, y, system: PDEAttackSystem1D, idx_attack, norm, epsilon, steps, alpha, device):
    """
    x,y: (N,X) on device
    idx_attack: list/array of indices to attack
    用 g(a_adv) 作为新 y（最终帧）；x 替换为 a_adv
    """
    x_new = x.clone()
    y_new = y.clone()
    last_loss = float("nan")

    with tqdm(total=len(idx_attack),
              desc=f"Attack {len(idx_attack)} samples (norm={norm}, eps={epsilon}, alpha={alpha}, steps={steps})",
              dynamic_ncols=True) as pbar:
        for i in idx_attack:
            # --- PGD 需要梯度，不能在 no_grad 里 ---
            a0 = x[i].contiguous()
            a_adv = system.pgd(a0, epsilon=epsilon, alpha=alpha, steps=steps, norm=norm)

            # --- PGD 完成后，下面无需梯度 ---
            with torch.no_grad():
                y_final = system.rollout_final(a_adv)   # (X,)

                # 仅记录一下 loss（非必须）
                pred = system.forward_G(a_adv)
                if pred.dim() == 3:
                    pred = pred[0]
                pred = pred.squeeze(-1)
                loss = F.mse_loss(pred, y_final, reduction="sum")
                last_loss = float(loss.item())

                x_new[i] = a_adv.detach()
                y_new[i] = y_final.detach().float()

            pbar.set_postfix({"loss": f"{last_loss:.3e}"})
            pbar.update(1)
            torch.cuda.empty_cache()

    return x_new, y_new


# --------------------------- 主流程 ---------------------------
def main(_):
    assert FLAGS.dataset_pt, "--dataset_pt 必填"
    # 随机种子
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed_all(FLAGS.seed)

    device = torch.device("cuda" if (FLAGS.use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    get_gpu_info()

    # 加载初始数据
    ds_path = Path(FLAGS.dataset_pt).resolve()
    data0 = torch.load(ds_path, map_location="cpu", weights_only=False)
    assert "x" in data0 and "y" in data0, "dataset 必须包含 'x' 与 'y'"
    t_final_val = float(data0.get("t_final", float(FLAGS.t_final)))

    x_cpu, y_cpu = ensure_x_y_shapes(data0["x"], data0["y"])
    N, X = x_cpu.shape
    print(f"Loaded dataset: N={N}, X={X}, t_final={t_final_val}")

    # 准备求解器 g(a)（固定 X/nu/t_final/dt）
    g_callable = build_solver(FLAGS.solver, X, float(FLAGS.nu), t_final_val, float(FLAGS.dt))

    # 解析攻击参数
    combos = parse_inputs_list(FLAGS.inputs)
    if len(combos) == 1:
        combos = combos * FLAGS.rounds  # 所有轮共用
    elif len(combos) != FLAGS.rounds:
        raise ValueError(f"--inputs 提供 {len(combos)} 组，但 rounds={FLAGS.rounds}，需 1 组或 K 组对应")

    # === 统一输出到脚本同级 results/ 目录 ===
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "datasets").mkdir(parents=True, exist_ok=True)

    model_root = RESULTS_DIR / "models" / f"m{FLAGS.modes}_w{FLAGS.width}_e{FLAGS.epochs}_{stamp}"
    data_root  = RESULTS_DIR / "datasets" / f"nu{fmt_float(FLAGS.nu)}_{FLAGS.solver}_m{FLAGS.modes}_w{FLAGS.width}_e{FLAGS.epochs}_{stamp}"
    model_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    # 保存本次运行的配置
    with open(model_root / "run_config.json", "w", encoding="utf-8") as fcfg:
        json.dump({
            "dataset_pt": str(ds_path),
            "rounds": FLAGS.rounds,
            "epochs": FLAGS.epochs,
            "batch_size": FLAGS.batch_size,
            "lr": FLAGS.lr,
            "weight_decay": FLAGS.weight_decay,
            "modes": FLAGS.modes,
            "width": FLAGS.width,
            "ntrain": FLAGS.ntrain,
            "ntest": FLAGS.ntest,
            "attack_ratio": FLAGS.attack_ratio,
            "inputs": combos,
            "solver": FLAGS.solver,
            "nu": FLAGS.nu,
            "t_final": t_final_val,
            "dt": FLAGS.dt,
            "seed": FLAGS.seed,
        }, fcfg, indent=2)

    # 初始化模型（同一对象跨轮迭代继续训练）
    model = FNO1d(FLAGS.modes, FLAGS.width).to(device)
    print(f"Model params: {count_params(model)}")

    # 本地拷贝到 device
    x_dev = x_cpu.to(device)
    y_dev = y_cpu.to(device)

    # 训练/测试划分
    tr_loader, te_loader, ntrain_eff, ntest_eff = make_dataloaders(
        x_dev, y_dev, FLAGS.ntrain, FLAGS.ntest, FLAGS.batch_size
    )

    # ===== 开始迭代 K 轮 =====
    overall_start = time.perf_counter()

    for rd in range(1, FLAGS.rounds + 1):
        print("\n" + "=" * 80)
        print(f"▶ Round {rd}/{FLAGS.rounds}: train → attack({FLAGS.attack_ratio*100:.1f}%) → replace → save")

        # 1) 训练本轮（显式解冻+train，避免上一轮攻击冻结残留）
        model = train_one_round(
            model=model,
            tr_loader=tr_loader,
            te_loader=te_loader,
            epochs=FLAGS.epochs,
            lr=FLAGS.lr,
            weight_decay=FLAGS.weight_decay,
            device=device,
            round_idx=rd,
            log_to=(model_root / f"round{rd:02d}_train_log.txt"),
        )

        # 保存模型
        round_model_dir = model_root / f"round{rd:02d}"
        round_model_dir.mkdir(parents=True, exist_ok=True)
        model_path = round_model_dir / f"burgers1d_FNO_round{rd:02d}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"✅ [Round {rd}] Model saved: {model_path}")

        # 2) 攻击并替换 attack_ratio 比例样本（攻击期间临时冻结，结束后恢复）
        (norm, epsilon, steps, alpha) = combos[rd - 1]
        system = PDEAttackSystem1D(model, g_callable, device)

        k = max(1, int(round(float(FLAGS.attack_ratio) * N)))
        idx_all = list(range(N))
        random.shuffle(idx_all)
        idx_attack = sorted(idx_all[:k])

        x_dev, y_dev = attack_replace_subset(
            x_dev, y_dev, system, idx_attack, norm, float(epsilon), int(steps), float(alpha), device
        )

        # 恢复模型参数可训练状态，供下一轮继续训练
        system.restore()
        del system

        # 3) 保存“本轮生成的新数据集”（全量 N 条，已替换攻击子集）
        round_data_dir = data_root / f"round{rd:02d}" / f"norm{str(norm).lower()}_eps{fmt_float(epsilon)}_alpha{fmt_float(alpha)}_steps{int(steps)}" / f"N={N}"
        round_data_dir.mkdir(parents=True, exist_ok=True)
        out_pt = round_data_dir / "dataset.pt"

        meta = {
            "round": rd,
            "attack_ratio": float(FLAGS.attack_ratio),
            "indices_attacked": idx_attack,
            "norm": norm,
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "steps": int(steps),
            "nu": float(FLAGS.nu),
            "solver": FLAGS.solver,
            "t_final": t_final_val,
            "dt": float(FLAGS.dt),
            "source_dataset": str(ds_path) if rd == 1 else "previous_round_dataset",
        }

        torch.save(
            {
                "x": x_dev.detach().float().cpu(),    # (N,X)
                "y": y_dev.detach().float().cpu(),    # (N,X) 仅最终帧
                "t_final": float(t_final_val),
                "meta": meta,
            },
            out_pt,
        )
        print(f"✅ [Round {rd}] Dataset saved: {out_pt}")

        # 4) 用“新数据”更新 dataloader，进入下一轮
        tr_loader, te_loader, _, _ = make_dataloaders(
            x_dev, y_dev, FLAGS.ntrain, FLAGS.ntest, FLAGS.batch_size
        )

    total_t = time.perf_counter() - overall_start
    print("\n" + "=" * 80)
    print(f"🎉 Completed {FLAGS.rounds} rounds. Total wall time: {total_t/60:.2f} min")
    print(f"Models root:   {model_root}")
    print(f"Datasets root: {data_root}")
    print("=" * 80)


if __name__ == "__main__":
    app.run(main)
