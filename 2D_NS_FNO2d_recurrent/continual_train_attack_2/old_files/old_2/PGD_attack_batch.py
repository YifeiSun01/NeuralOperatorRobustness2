# Minimal PGD(+Adam) attack core with a sequence API:
#   - attack_batch:            (B,H,W)                 -> (B,H,W)
#   - attack_and_rollout_seq:  (B,H,W,Tin) + Tout,solver -> ( (B,H,W,Tin), (B,H,W,Tout) )
#
# No recorder, no file I/O, no prints.

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from typing import List, Tuple

# --------------------------- small utils ---------------------------

def _ensure_batched_seq(seq: torch.Tensor) -> torch.Tensor:
    """Make sure seq is (T+1, B, H, W). Accepts (T+1,H,W) and unsqueezes B=1."""
    if seq.ndim == 3:
        seq = seq.unsqueeze(1)
    return seq

def _toggle_requires_grad(module: torch.nn.Module, flag: bool) -> List[bool]:
    """Set requires_grad for all params; return original flags for restoration."""
    orig = []
    for p in module.parameters():
        orig.append(p.requires_grad)
        p.requires_grad_(flag)
    return orig

def _restore_requires_grad(module: torch.nn.Module, flags: List[bool]) -> None:
    for p, f in zip(module.parameters(), flags):
        p.requires_grad_(f)

def _assemble_input_frames_from_pde(
    x0: torch.Tensor,  # (B,H,W)
    pde_solver,        # rollout_seconds(x0, T) -> (T+1,B,H,W) or (T+1,H,W)
    mode_spec: str = "wwwwwwwwww",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (B,H,W,10) for recurrent model; return also true_19 (B,H,W).
    Only 'w' / 'd' are supported in this minimal core.
    """
    assert isinstance(mode_spec, str) and len(mode_spec) == 10
    need = {t for t in range(1, 10) if mode_spec[t-1] in ("w", "d")}
    if mode_spec[9] in ("w", "d"):
        need.add(19)
    T_required = max(need) if need else 0

    seq = pde_solver.rollout_seconds(x0, T_required)         # (T+1,B,H,W) | (T+1,H,W)
    seq = _ensure_batched_seq(seq)                           # (T+1,B,H,W)

    frames = [x0]  # t=0
    for t in range(1, 10):
        f = seq[t]                                          # (B,H,W)
        if mode_spec[t-1] == 'd':
            f = f.detach()
        frames.append(f)
    input_batch = torch.stack(frames, dim=-1)               # (B,H,W,10)

    assert mode_spec[9] in ('w', 'd'), "mode_spec[9] must be 'w' or 'd'."
    true_19 = seq[19]
    if mode_spec[9] == 'd':
        true_19 = true_19.detach()
    return input_batch, true_19  # (B,H,W,10), (B,H,W)

# --------------------------- config ---------------------------

@dataclass
class PGDAdamConfig:
    epsilon: float = 0.1
    alpha: float = 0.01
    num_steps: int = 10
    norm: str | int = "inf"   # 'inf' or 2
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    amsgrad: bool = False
    use_sign_for_linf: bool = True
    mode_spec: str = "wwwwwwwwww"   # minimal core uses only 'w'/'d'

@torch.no_grad()
def _project_l2_(x_adv, x0, eps):
    """In-place L2 projection per sample. Shapes: (B,H,W)."""
    B = x_adv.shape[0]
    delta = (x_adv - x0).view(B, -1)
    norm = torch.linalg.vector_norm(delta, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = (eps / norm).clamp(max=1.0)
    x_adv.copy_(x0 + (delta * scale).view_as(x_adv))
    return x_adv

# --------------------------- core: (B,H,W) -> (B,H,W) ---------------------------

def attack_batch(
    x_batch: torch.Tensor,             # (B,H,W), float32, on correct device
    recurrent_model: torch.nn.Module,  # RecurrentPredictor(FNO2d,...), eval recommended
    pde_solver,                        # has rollout_seconds(x0, T_seconds)
    cfg: PGDAdamConfig = PGDAdamConfig(),
) -> torch.Tensor:
    """
    PGD(+Adam) on a batch and return the perturbed batch (B,H,W).
    Loss = sum over batch & pixels: MSE(pred_19, true_19, reduction='sum').
    L2 branch: per-sample normalization & projection; Linf branch: sign by default.
    """
    assert x_batch.ndim == 3, "x_batch must be (B,H,W)"
    x0   = x_batch.detach()
    x_adv = x0.clone().requires_grad_(True)
    # x_adv = torch.ascontiguous_tensor(x0, memory_format=torch.contiguous_format).clone().requires_grad_(True)

    # ---- freeze model params during PGD to avoid accumulating grads on weights ----
    saved_flags = _toggle_requires_grad(recurrent_model, False)
    recurrent_model.eval()

    # Adam state
    m = torch.zeros_like(x_adv)
    v = torch.zeros_like(x_adv)
    vhat_max = torch.zeros_like(x_adv) if cfg.amsgrad else None

    try:
        for _ in range(cfg.num_steps):
            # Forward -> loss (params frozen; gradients flow to x_adv)
            input_batch, true_19 = _assemble_input_frames_from_pde(x_adv, pde_solver, mode_spec=cfg.mode_spec)
            pred = recurrent_model(input_batch)      # (B,H,W,10)
            pred_19 = pred[..., -1]                  # (B,H,W)
            loss = F.mse_loss(pred_19, true_19, reduction="sum")

            # Backward on x_adv
            x_adv.grad = None
            loss.backward()

            # Adam step on x_adv
            with torch.no_grad():
                g = x_adv.grad
                m.mul_(cfg.beta1).add_(g, alpha=(1.0 - cfg.beta1))
                v.mul_(cfg.beta2).addcmul_(g, g, value=(1.0 - cfg.beta2))
                if cfg.amsgrad:
                    torch.maximum(vhat_max, v, out=vhat_max)
                    denom = vhat_max.sqrt().add_(cfg.adam_eps)
                else:
                    denom = v.sqrt().add_(cfg.adam_eps)
                d_t = m / denom

                if cfg.norm in ('inf', 'Linf', '∞'):
                    direction = d_t.sign() if cfg.use_sign_for_linf else d_t
                    x_adv.add_(direction, alpha=cfg.alpha)
                    # Linf projection
                    x_adv.copy_(x0 + (x_adv - x0).clamp(min=-cfg.epsilon, max=cfg.epsilon))
                elif cfg.norm in (2, '2'):
                    # per-sample normalized direction + projection
                    B = x_adv.shape[0]
                    flat = d_t.view(B, -1)
                    norm = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
                    direction = (flat / norm).view_as(d_t)
                    x_adv.add_(direction, alpha=cfg.alpha)
                    _project_l2_(x_adv, x0, cfg.epsilon)
                else:
                    raise ValueError(f"Unsupported norm: {cfg.norm}")
    finally:
        # restore requires_grad flags
        _restore_requires_grad(recurrent_model, saved_flags)

    return x_adv.detach()

# -------------------- new: (B,H,W,Tin) -> (B,H,W,Tin),(B,H,W,Tout) --------------------

def attack_and_rollout_seq(
    x_seq: torch.Tensor,               # (B,H,W,Tin), float32, on correct device
    recurrent_model: torch.nn.Module,  # RecurrentPredictor(FNO2d,...)
    pde_solver,                        # rollout_seconds(x0, T_seconds)
    Tout: int,                         # number of target steps after Tin
    cfg: PGDAdamConfig = PGDAdamConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    只攻击第0帧：取 x_seq[...,0] -> attack_batch -> x0_adv
    然后用 x0_adv 作为 PDE 初值，一次性 rollout 出 Tin+Tout 帧。
    切分返回：
        adv_xx = (B,H,W,Tin)  = seq_all[0:Tin].permute(1,2,3,0)
        yy     = (B,H,W,Tout) = seq_all[Tin:Tin+Tout].permute(1,2,3,0)
    """
    assert x_seq.ndim == 4, "x_seq must be (B,H,W,Tin)"
    B, H, W, Tin = x_seq.shape
    assert Tin >= 1 and Tout >= 1, "Tin/Tout must be positive."

    # 1) 只攻击第 0 帧
    x0 = x_seq[..., 0].contiguous()            # (B,H,W)
    # x0 = torch.ascontiguous_tensor(x_seq[..., 0], memory_format=torch.contiguous_format)
    x0_adv = attack_batch(x0, recurrent_model, pde_solver, cfg=cfg)  # (B,H,W)

    # 2) PDE rollout 产生 Tin+Tout 帧（含 t=0）
    T_required = Tin + Tout - 1
    seq = pde_solver.rollout_seconds(x0_adv, T_required)     # (T+1,B,H,W)|(T+1,H,W)
    seq = _ensure_batched_seq(seq)                           # (T+1,B,H,W)

    # 3) 切分 (xx, yy)
    seq_xx  = seq[0:Tin]                       # (Tin,B,H,W)
    seq_yy  = seq[Tin:Tin+Tout]                # (Tout,B,H,W)

    adv_xx = seq_xx.permute(1, 2, 3, 0).contiguous()  # -> (B,H,W,Tin)
    yy     = seq_yy.permute(1, 2, 3, 0).contiguous()  # -> (B,H,W,Tout)
    return adv_xx, yy


