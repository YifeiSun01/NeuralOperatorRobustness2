#!/usr/bin/env python3
"""Repair unstable samples in the NS2D recurrent attack dictionary.

The script loads an existing dictionary, finds samples with non-finite values or
large amplitudes, reruns only those initial conditions with smaller Exponax time
steps, and writes a repaired dictionary file.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import jax
import numpy as np
import torch
from tqdm import tqdm

from generate_ns_dictionary_batched import (
    DEFAULT_OUTPUT_DIR,
    jax_to_torch,
    json_ready,
    make_ns_rollout_fn,
    parse_float_list,
    stability_mask,
    torch_to_jax,
)

THIS_FILE = Path(__file__).resolve()
NS_ROOT = THIS_FILE.parents[1]
DEFAULT_INPUT = (
    NS_ROOT
    / "datasets"
    / "exponax_datasets"
    / "t20"
    / "dictionary"
    / "dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt"
)


def default_repaired_path(input_path: Path, max_abs_threshold: float | None, max_rms_threshold: float | None) -> Path:
    parts = []
    if max_abs_threshold is not None:
        parts.append(f"maxabs{max_abs_threshold:g}")
    if max_rms_threshold is not None:
        parts.append(f"maxrms{max_rms_threshold:g}")
    suffix = "_repaired_" + "_".join(parts or ["unstable"])
    return input_path.with_name(input_path.stem + suffix + input_path.suffix)


def saved_x_to_rollout_input(x_saved: torch.Tensor) -> torch.Tensor:
    # Invert the orientation transform in make_ns_rollout_fn.one():
    # saved first frame = swapaxes(rot90(flip(u0, axis=-2), 3), -1, -2).
    return torch.flip(torch.rot90(torch.swapaxes(x_saved, -1, -2), 1, dims=(-2, -1)), dims=(-2,)).contiguous()


def scan_unstable_indices(
    y: torch.Tensor,
    *,
    chunk_size: int,
    max_abs_threshold: float | None,
    max_rms_threshold: float | None,
) -> tuple[list[int], dict[str, Any]]:
    n = int(y.shape[0])
    finite = torch.empty(n, dtype=torch.bool)
    max_abs = torch.empty(n, dtype=torch.float64)
    rms = torch.empty(n, dtype=torch.float64)
    unstable = torch.empty(n, dtype=torch.bool)
    for start in tqdm(range(0, n, chunk_size), desc="scan existing", unit="chunk"):
        end = min(start + chunk_size, n)
        yc = y[start:end].float()
        flat = yc.reshape(end - start, -1)
        finite[start:end] = torch.isfinite(flat).all(dim=1).cpu()
        safe = torch.nan_to_num(flat, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        max_abs[start:end] = safe.abs().amax(dim=1).double().cpu()
        rms[start:end] = torch.sqrt(torch.nan_to_num(flat.square(), nan=float("inf"), posinf=float("inf"), neginf=float("inf")).mean(dim=1)).double().cpu()
        mask = ~finite[start:end]
        if max_abs_threshold is not None:
            mask |= max_abs[start:end] > max_abs_threshold
        if max_rms_threshold is not None:
            mask |= rms[start:end] > max_rms_threshold
        unstable[start:end] = mask
    indices = torch.nonzero(unstable, as_tuple=False).flatten().tolist()
    summary = {
        "finite_count": int(finite.sum().item()),
        "nonfinite_indices": torch.nonzero(~finite, as_tuple=False).flatten().tolist(),
        "unstable_indices": indices,
        "unstable_count": len(indices),
        "thresholds": {
            "max_abs_threshold": max_abs_threshold,
            "max_rms_threshold": max_rms_threshold,
        },
        "max_abs_selected": {str(i): float(max_abs[i].item()) for i in indices},
        "rms_selected": {str(i): float(rms[i].item()) for i in indices},
        "max_abs_global_max": float(max_abs.max().item()),
        "rms_global_max": float(rms.max().item()),
    }
    return [int(i) for i in indices], summary


def rerun_indices(args: argparse.Namespace, x: torch.Tensor, target_indices: list[int]) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    if not target_indices:
        return {}, {"accepted": [], "failed": [], "step_records": []}
    device = torch.device("cuda")
    accepted: dict[int, torch.Tensor] = {}
    remaining = list(target_indices)
    step_records = []
    rollout_cache: dict[tuple[float, int], object] = {}

    for step in args.repair_step_options:
        if not remaining:
            break
        step_t0 = time.perf_counter()
        next_remaining: list[int] = []
        accepted_this_step = []
        for start in tqdm(range(0, len(remaining), args.solver_batch_size), desc=f"rerun dt={step:g}", unit="subbatch"):
            sub_indices = remaining[start : start + args.solver_batch_size]
            x_saved = x.index_select(0, torch.tensor(sub_indices, dtype=torch.long)).to(device=device, dtype=torch.float32)
            u0 = saved_x_to_rollout_input(x_saved)
            sub_n = len(sub_indices)
            cache_key = (float(step), sub_n)
            if cache_key not in rollout_cache:
                rollout_cache[cache_key] = make_ns_rollout_fn(
                    nx=args.nx,
                    nu=args.nu,
                    t_final=args.t_final,
                    fixed_step=float(step),
                    domain_extent=args.domain_extent,
                    solver_mode=args.solver_mode,
                )
            y_sub = rollout_cache[cache_key](torch_to_jax(u0))
            y_sub.block_until_ready()
            y_try = jax_to_torch(y_sub).contiguous()
            stable = stability_mask(
                y_try,
                max_abs_threshold=args.max_abs_threshold,
                max_rms_threshold=args.max_rms_threshold,
            )
            for local_pos, global_idx in enumerate(sub_indices):
                if bool(stable[local_pos].item()):
                    accepted[global_idx] = y_try[local_pos].detach().cpu().contiguous()
                    accepted_this_step.append(global_idx)
                else:
                    next_remaining.append(global_idx)
            del x_saved, u0, y_sub, y_try, stable
            torch.cuda.empty_cache()
        step_records.append(
            {
                "step": float(step),
                "attempted": len(remaining),
                "accepted": len(accepted_this_step),
                "remaining": len(next_remaining),
                "accepted_indices": accepted_this_step,
                "seconds": time.perf_counter() - step_t0,
            }
        )
        remaining = next_remaining

    return accepted, {"accepted": sorted(accepted), "failed": remaining, "step_records": step_records}


def generate(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available; refusing to repair with CPU solver")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend is {jax.default_backend()!r}, expected 'gpu'")
    if args.max_abs_threshold is None and args.max_rms_threshold is None:
        raise ValueError("At least one of --max-abs-threshold or --max-rms-threshold must be set")

    t0 = time.perf_counter()
    input_path = args.input.resolve()
    if args.in_place:
        output_path = input_path
    else:
        output_path = args.output.resolve() if args.output else default_repaired_path(input_path, args.max_abs_threshold, args.max_rms_threshold)
    report_path = args.report.resolve() if args.report else input_path.with_name(input_path.stem + "_inplace_repair_report.json" if args.in_place else output_path.stem + "_repair_report.json")
    if output_path.exists() and output_path != input_path and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")

    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    x = payload["x"].contiguous()
    y = payload["y"].contiguous()
    metadata = dict(payload.get("metadata", {}))

    target_indices, scan_summary = scan_unstable_indices(
        y,
        chunk_size=args.scan_chunk_size,
        max_abs_threshold=args.max_abs_threshold,
        max_rms_threshold=args.max_rms_threshold,
    )
    if args.indices:
        requested = [int(item.strip()) for item in args.indices.split(",") if item.strip()]
        target_indices = sorted(set(target_indices).union(requested))
        scan_summary["manual_indices_added"] = requested
    print(json.dumps(json_ready({"target_count": len(target_indices), "target_indices": target_indices}), indent=2))

    accepted, rerun_summary = rerun_indices(args, x, target_indices)
    for idx, y_new in accepted.items():
        y[idx].copy_(y_new)
        x[idx].copy_(y_new[..., 0])

    if rerun_summary["failed"] and args.fail_on_unstable:
        raise RuntimeError(f"Still unstable after all repair steps: {rerun_summary['failed']}")

    repair_metadata = {
        "source_input": str(input_path),
        "repair_script": str(THIS_FILE),
        "repair_thresholds": {
            "max_abs_threshold": args.max_abs_threshold,
            "max_rms_threshold": args.max_rms_threshold,
        },
        "repair_step_options": args.repair_step_options,
        "solver_batch_size": args.solver_batch_size,
        "solver_mode": args.solver_mode,
        "repaired_indices": rerun_summary["accepted"],
        "unrepaired_indices": rerun_summary["failed"],
    }
    metadata.setdefault("repair_history", [])
    metadata["repair_history"].append(repair_metadata)
    metadata["nsamples"] = int(y.shape[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_path = output_path
    if output_path == input_path:
        save_path = input_path.with_name(input_path.name + ".tmp_repair")
    torch.save({"x": x, "y": y, "metadata": metadata}, save_path)
    if save_path != output_path:
        save_path.replace(output_path)
    report = {
        "input": input_path,
        "output": output_path,
        "in_place": bool(args.in_place),
        "report": report_path,
        "total_seconds": time.perf_counter() - t0,
        "scan_summary": scan_summary,
        "rerun_summary": rerun_summary,
        "metadata_added": repair_metadata,
        "output_size_bytes": output_path.stat().st_size,
        "gpu": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_device": torch.cuda.get_device_name(0),
            "torch_capability": torch.cuda.get_device_capability(0),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
        },
    }
    report_path.write_text(json.dumps(json_ready(report), indent=2), encoding="utf-8")
    print(f"[saved] {output_path}")
    print(f"[report] {report_path}")
    print(json.dumps(json_ready({
        "repaired_count": len(rerun_summary["accepted"]),
        "failed_count": len(rerun_summary["failed"]),
        "total_seconds": report["total_seconds"],
    }), indent=2))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--in-place", action="store_true", help="Replace unstable samples in the input .pt path itself via a temporary file and atomic rename.")
    parser.add_argument("--indices", default=None, help="Optional comma-separated indices to force rerun in addition to threshold hits.")
    parser.add_argument("--max-abs-threshold", type=float, default=10.0)
    parser.add_argument("--max-rms-threshold", type=float, default=None)
    parser.add_argument("--repair-step-options", type=parse_float_list, default=parse_float_list("0.0005,0.0001"))
    parser.add_argument("--solver-batch-size", type=int, default=50)
    parser.add_argument("--solver-mode", choices=["vmap", "lax-map"], default="vmap")
    parser.add_argument("--scan-chunk-size", type=int, default=20)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nu", type=float, default=1e-5)
    parser.add_argument("--t-final", type=int, default=20)
    parser.add_argument("--domain-extent", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-unstable", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
