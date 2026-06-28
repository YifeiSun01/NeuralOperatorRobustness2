#!/usr/bin/env python3
"""Probe NS2D loss3 adversarial-training batch memory on one CUDA GPU.

This runs the same expensive core used by tools/adversarial_training.py for
NS2D solver-label loss3:

1. perturb the initial vorticity frame x0;
2. roll out the differentiable Navier-Stokes solver from x0_adv;
3. backpropagate MSE(FNO(solver_frames[:10]), solver_frames[10:20]) to x0_adv;
4. regenerate attacked solver pairs without gradient;
5. run a real optimizer update through the recurrent FNO.

The probe can use the retained NS2D checkpoint when it is present. If the
checkpoint is missing, it can instantiate the same architecture with random
weights for memory-capacity testing only; the JSON result records that source.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NS_ROOT = PROJECT_ROOT / "2D_NS_FNO2d_recurrent"
DEFAULT_CKPT = (
    NS_ROOT
    / "saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/best.pt"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.adversarial_training import finite_mse, ns2d_solver_pair_from_initial  # noqa: E402
from tools.evaluate_generalization_models import checkpoint_state, load_module, load_ns2d_model  # noqa: E402


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S_UTC", time.gmtime())


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def gpu_preflight(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("NS2D loss3 batch probe is GPU-only; pass --device cuda and fix CUDA if unavailable.")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false; refusing CPU fallback.")

    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    preflight: dict[str, Any] = {
        "nvidia_smi": run_text(["nvidia-smi"]),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_device_name": torch.cuda.get_device_name(device),
        "torch_compute_capability": f"sm_{props.major}{props.minor}",
        "torch_cuda_arch_list": getattr(torch.cuda, "get_arch_list", lambda: [])(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    try:
        import jax

        preflight["jax_version"] = jax.__version__
        preflight["jax_default_backend"] = jax.default_backend()
        preflight["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as exc:
        preflight["jax_import_error"] = repr(exc)
    if preflight["torch_compute_capability"] == "sm_70":
        arch_list = list(preflight.get("torch_cuda_arch_list") or [])
        if arch_list and not any("sm_70" in str(item) for item in arch_list):
            raise RuntimeError(f"PyTorch CUDA arch list does not include sm_70: {arch_list}")
    return preflight


def instantiate_ns2d_model(device: torch.device):
    if str(NS_ROOT) not in sys.path:
        sys.path.insert(0, str(NS_ROOT))
    mod = load_module("ns_fno2d_probe_20260608", NS_ROOT / "models" / "FNO2d.py")
    fno = mod.FNO2d(modes1=64, modes2=64, width=60, num_layers=4, in_channels=10).to(device)
    return mod.RecurrentPredictor(fno, T_out=10, step=1).to(device)


def load_probe_model(device: torch.device, checkpoint: Path, allow_random: bool) -> tuple[torch.nn.Module, dict[str, Any]]:
    if checkpoint.exists() and checkpoint.resolve() == DEFAULT_CKPT.resolve():
        model = load_ns2d_model(device)
        return model, {"model_source": "checkpoint", "checkpoint": str(checkpoint)}
    if checkpoint.exists():
        model = instantiate_ns2d_model(device)
        model.load_state_dict(checkpoint_state(checkpoint), strict=True)
        model.eval()
        return model, {"model_source": "checkpoint", "checkpoint": str(checkpoint)}
    if not allow_random:
        raise FileNotFoundError(f"NS2D checkpoint missing: {checkpoint}")
    model = instantiate_ns2d_model(device)
    model.eval()
    return model, {
        "model_source": "random_same_architecture_memory_probe",
        "checkpoint": str(checkpoint),
        "checkpoint_missing": True,
        "note": "Random weights are valid only for memory-capacity probing, not for scientific metrics.",
    }


def make_synthetic_x0(batch_size: int, grid_size: int, device: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    x = torch.randn((batch_size, grid_size, grid_size), generator=gen, device=device, dtype=torch.float32)
    for _ in range(2):
        x = 0.5 * x + 0.125 * (
            torch.roll(x, 1, -1)
            + torch.roll(x, -1, -1)
            + torch.roll(x, 1, -2)
            + torch.roll(x, -1, -2)
        )
    x = x / x.flatten(1).std(dim=1).clamp_min(1e-6).view(-1, 1, 1)
    return x.contiguous()


def per_sample_range(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    return flat.max(dim=1).values - flat.min(dim=1).values


def expand_per_sample(values: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return values.view(values.shape[0], *([1] * (like.ndim - 1)))


def safe_memory_stats(device: torch.device) -> dict[str, float]:
    try:
        return {
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        }
    except Exception:
        return {"peak_cuda_allocated_gib": float("nan"), "peak_cuda_reserved_gib": float("nan")}


def run_loss3_probe(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    device = torch.device(args.device)
    started = now_stamp()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight = gpu_preflight(device)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))

    model, model_info = load_probe_model(device, args.checkpoint.resolve(), bool(args.allow_random_model_for_memory_probe))
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    cfg = {
        "label_mode": "solver",
        "ns2d_solver_remat": str(args.solver_remat),
        "ns2d_solver_remat_chunk_steps": int(args.solver_remat_chunk_steps),
    }
    batch_size = int(args.batch_size)
    x0 = make_synthetic_x0(batch_size, int(args.grid_size), device, int(args.seed))
    eps = per_sample_range(x0) * float(args.epsilon_fraction)
    alpha = eps * float(args.alpha_ratio)
    eps_view = expand_per_sample(eps, x0)
    alpha_view = expand_per_sample(alpha, x0)
    delta = torch.zeros_like(x0)
    x0_adv = (x0 + delta).detach()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    attack_losses: list[float] = []
    train_losses: list[float] = []
    status = "passed"
    exit_code = 0
    error_message = ""
    error_trace = ""

    try:
        for _step in range(max(1, int(args.attack_steps))):
            x0_adv = x0_adv.detach().requires_grad_(True)
            x_seq_adv, y_seq_adv = ns2d_solver_pair_from_initial(x0_adv, cfg, for_attack=True)
            pred_adv = model(x_seq_adv)
            loss = finite_mse(pred_adv, y_seq_adv)
            attack_losses.append(float(loss.detach().cpu()))
            grad = torch.autograd.grad(loss, x0_adv, only_inputs=True)[0]
            grad = torch.nan_to_num(grad)
            if str(args.attack_method) == "fast_replace_linf":
                delta = eps_view * grad.sign()
            elif str(args.attack_method) == "fast_add_linf":
                delta = torch.clamp((x0_adv.detach() - x0) + alpha_view * grad.sign(), -eps_view, eps_view)
            else:
                raise ValueError(f"unsupported attack_method={args.attack_method!r}")
            x0_adv = (x0 + delta).detach()
            del x_seq_adv, y_seq_adv, pred_adv, loss, grad

        with torch.no_grad():
            x_train, y_train = ns2d_solver_pair_from_initial(x0_adv, cfg, for_attack=False)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        optimizer_batch = max(1, int(args.optimizer_batch_size))
        for start_idx in range(0, batch_size, optimizer_batch):
            xb = x_train[start_idx : start_idx + optimizer_batch]
            yb = y_train[start_idx : start_idx + optimizer_batch]
            pred = model(xb)
            train_loss = finite_mse(pred, yb)
            scaled = train_loss * (int(xb.shape[0]) / batch_size)
            scaled.backward()
            train_losses.append(float(train_loss.detach().cpu()))
        optimizer.step()
        model.eval()
    except RuntimeError as exc:
        error_message = str(exc)
        error_trace = traceback.format_exc()
        oom_markers = (
            "out of memory",
            "CUBLAS_STATUS_ALLOC_FAILED",
            "CUDA error: an illegal memory access",
            "CUDA error: unknown error",
        )
        if any(marker.lower() in error_message.lower() for marker in oom_markers):
            status = "failed_oom"
            exit_code = 42
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            status = "failed_runtime_error"
            exit_code = 1
    except Exception as exc:
        error_message = str(exc)
        error_trace = traceback.format_exc()
        status = "failed_exception"
        exit_code = 1

    elapsed = time.perf_counter() - start_time
    if status == "passed":
        delta_final = x0_adv.detach() - x0
        boundary_ratio = float((delta_final.abs() / eps_view.clamp_min(1e-12)).mean().cpu())
        delta_linf = float(delta_final.abs().flatten(1).max(dim=1).values.mean().cpu())
        delta_l2 = float(torch.sqrt(delta_final.pow(2).flatten(1).mean(dim=1)).mean().cpu())
    else:
        boundary_ratio = float("nan")
        delta_linf = float("nan")
        delta_l2 = float("nan")

    result = {
        "created_utc": now_stamp(),
        "started_utc": started,
        "status": status,
        "exit_code": exit_code,
        "batch_size": batch_size,
        "optimizer_batch_size": int(args.optimizer_batch_size),
        "attack_steps": int(args.attack_steps),
        "attack_method": str(args.attack_method),
        "epsilon_fraction": float(args.epsilon_fraction),
        "alpha_ratio": float(args.alpha_ratio),
        "solver_remat": str(args.solver_remat),
        "solver_remat_chunk_steps": int(args.solver_remat_chunk_steps),
        "grid_size": int(args.grid_size),
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
        "attack_loss_first": attack_losses[0] if attack_losses else float("nan"),
        "attack_loss_last": attack_losses[-1] if attack_losses else float("nan"),
        "train_loss_mean": float(np.mean(train_losses)) if train_losses else float("nan"),
        "boundary_ratio_mean": boundary_ratio,
        "delta_linf_mean": delta_linf,
        "delta_l2_rms_mean": delta_l2,
        "gpu_preflight": preflight,
        **safe_memory_stats(device),
        **model_info,
    }
    if status != "passed":
        result["error_message"] = error_message
        result["traceback"] = error_trace

    json_path = out_dir / f"batch_{batch_size:03d}.json"
    json_path.write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")
    csv_path = out_dir / "batch_probe_results.csv"
    csv_fields = [
        "created_utc",
        "status",
        "batch_size",
        "optimizer_batch_size",
        "attack_steps",
        "attack_method",
        "epsilon_fraction",
        "alpha_ratio",
        "solver_remat",
        "solver_remat_chunk_steps",
        "elapsed_seconds",
        "peak_cuda_allocated_gib",
        "peak_cuda_reserved_gib",
        "attack_loss_first",
        "attack_loss_last",
        "train_loss_mean",
        "boundary_ratio_mean",
        "delta_linf_mean",
        "delta_l2_rms_mean",
        "model_source",
        "checkpoint_missing",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: jsonable(result.get(key, "")) for key in csv_fields})
    return exit_code, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--optimizer-batch-size", type=int, default=1)
    parser.add_argument("--attack-steps", type=int, default=10)
    parser.add_argument("--attack-method", choices=["fast_add_linf", "fast_replace_linf"], default="fast_add_linf")
    parser.add_argument("--epsilon-fraction", type=float, default=0.035)
    parser.add_argument("--alpha-ratio", type=float, default=0.2)
    parser.add_argument("--solver-remat", choices=["none", "micro", "step", "chunk", "second"], default="chunk")
    parser.add_argument("--solver-remat-chunk-steps", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "forensics/ns2d_loss3_adversarial_training_batch_probe_20260608")
    parser.add_argument(
        "--allow-random-model-for-memory-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the same NS2D architecture with random weights if the retained checkpoint is missing. This is for memory probing only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code, result = run_loss3_probe(args)
    print(json.dumps(jsonable({k: result.get(k) for k in (
        "status",
        "batch_size",
        "elapsed_seconds",
        "peak_cuda_allocated_gib",
        "peak_cuda_reserved_gib",
        "model_source",
        "checkpoint_missing",
    )}), indent=2), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
