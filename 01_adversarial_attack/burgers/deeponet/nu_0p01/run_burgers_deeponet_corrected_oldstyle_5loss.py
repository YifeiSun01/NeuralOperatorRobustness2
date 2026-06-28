#!/usr/bin/env python3
"""DeepONet version of the corrected old-style Burgers 5-loss PGD runner.

This intentionally reuses the FNO runner's attack loop, loss definitions,
dictionary handling, summaries, and pickle layout.  The only behavioral change
is that f(.) is loaded from a trained DeepONet checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_burgers_corrected_oldstyle_5loss as base  # noqa: E402
from plot_burgers_deeponet_predictions import build_model, load_config  # noqa: E402
from train_burgers_deeponet_deepxde import format_float_tag, make_grid  # noqa: E402


def default_deeponet_run_dir(nu: float) -> Path:
    if abs(float(nu) - 0.01) < 1e-12:
        return PROJECT_ROOT / "deeponet_training_runs" / "burgers_nu0p01_deeponet_lu_ref_50k"
    if abs(float(nu) - 0.001) < 1e-12:
        return PROJECT_ROOT / "deeponet_training_runs" / "burgers_nu0p001_deeponet_lu_ref_50k"
    raise ValueError(f"No default DeepONet run directory registered for nu={nu}; pass --model_path.")


def default_deeponet_model_path(nu: float) -> Path:
    run_dir = default_deeponet_run_dir(nu)
    return run_dir / "checkpoints" / f"deeponet_burgers_nu{format_float_tag(nu)}.pt"


def output_transform_stats_path_from_model(model_path: Path) -> Path:
    return model_path.parents[1] / "training_logs" / "output_transform_stats.npz"


def validate_output_transform_stats(model_path: Path, config: dict, nx: int) -> Path | None:
    if not bool(config.get("output_transform", False)):
        return None
    stats_path = output_transform_stats_path_from_model(model_path)
    if not stats_path.exists():
        raise FileNotFoundError(
            f"DeepONet config requires output_transform=True, but missing {stats_path}. "
            "Do not run DeepONet attacks without this file; loss/gradients would be on the wrong scale."
        )
    stats = np.load(stats_path)
    if "y_mean" not in stats or "y_std" not in stats:
        raise KeyError(f"{stats_path} must contain 'y_mean' and 'y_std'. Found keys: {list(stats.files)}")
    expected_shape = (1, int(nx))
    y_mean_shape = tuple(stats["y_mean"].shape)
    y_std_shape = tuple(stats["y_std"].shape)
    if y_mean_shape != expected_shape or y_std_shape != expected_shape:
        raise ValueError(
            f"{stats_path} has invalid output-transform shapes: "
            f"y_mean={y_mean_shape}, y_std={y_std_shape}, expected {expected_shape}."
        )
    return stats_path


class DeepONetBurgersWrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, nx: int, domain: float, device: torch.device):
        super().__init__()
        self.net = net
        trunk = torch.as_tensor(make_grid(nx, domain), dtype=torch.float32, device=device)
        self.register_buffer("trunk", trunk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected DeepONet Burgers input [B,nx,1], got {tuple(x.shape)}")
        branch = x[..., 0]
        out = self.net((branch, self.trunk))
        if out.ndim == 2:
            out = out[..., None]
        return out


def load_deeponet_model(path: Path, device: torch.device):
    config = load_config(path.parents[1])
    nu = float(config.get("nu", 0.01))
    args = SimpleNamespace(
        run_dir=path.parents[1],
        branch_widths=None,
        trunk_widths=None,
        activation=None,
        kernel_initializer=None,
        nx=int(config.get("nx", 1024)),
        domain=float(config.get("domain", 2.0)),
    )
    stats_path = validate_output_transform_stats(path, config, args.nx)
    dde_model = build_model(args, config)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    dde_model.net.load_state_dict(ckpt["model_state_dict"])
    dde_model.net.to(device)
    dde_model.net.eval()
    for param in dde_model.net.parameters():
        param.requires_grad_(False)
    wrapper = DeepONetBurgersWrapper(dde_model.net, nx=args.nx, domain=args.domain, device=device).to(device)
    wrapper.eval()
    wrapper.deeponet_nu = nu
    wrapper.deeponet_config = config
    wrapper.deeponet_output_transform_stats_path = str(stats_path) if stats_path is not None else None
    return wrapper


def patched_default_model_path(nu: float) -> Path:
    return default_deeponet_model_path(nu)


def patched_load_model(path: Path, device: torch.device):
    return load_deeponet_model(path, device)


def patched_validate_core_paths(model_path: Path, test_path: Path, args: argparse.Namespace) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    base.validate_path_nu("test set", test_path, args.nu)
    config = load_config(model_path.parents[1])
    model_nu = float(config.get("nu", args.nu))
    base.assert_close_value("DeepONet config nu", model_nu, args.nu)
    stats_path = validate_output_transform_stats(model_path, config, args.nx)
    return {
        "model_path_nu": model_nu,
        "test_path_nu": base.parse_nu_from_path(test_path),
        "solver_nu": args.nu,
        "solver_domain": args.domain,
        "solver_nx": args.nx,
        "solver_t_final": args.t_final,
        "solver_dt": args.dt,
        "model_type": "DeepONet",
        "output_transform_stats_path": str(stats_path) if stats_path is not None else None,
    }


def patched_default_tag(args: argparse.Namespace) -> str:
    norm = "linf" if args.norm in {"inf", "linf", "infinity"} else f"l{args.norm}"
    return (
        f"deeponet_nu{base.fmt_float_for_path(args.nu)}_domain{base.fmt_float_for_path(args.domain)}_"
        f"{norm}_eps{base.fmt_float_for_path(args.epsilon)}_alpha{base.fmt_float_for_path(args.alpha)}_"
        f"steps{args.steps}_index{args.index}_rs"
    )


def main() -> None:
    base.default_model_path = patched_default_model_path
    base.load_model = patched_load_model
    base.validate_core_paths = patched_validate_core_paths
    base.default_tag = patched_default_tag
    base.main()


if __name__ == "__main__":
    main()
