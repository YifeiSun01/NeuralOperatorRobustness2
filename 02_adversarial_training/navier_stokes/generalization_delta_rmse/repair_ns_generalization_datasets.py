#!/usr/bin/env python3
"""Repair unstable NS2D generalization ground-truth samples.

A generated NS sample is treated as exploded if any rollout value is non-finite
or leaves the configured amplitude range.  Unstable samples are rerun from the
saved initial condition with halved Exponax solver steps until they become
stable, then only those samples are replaced in the dataset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import jax
import torch
from tqdm import tqdm

from generate_generalization_datasets import (
    PROJECT_ROOT,
    jax_to_torch,
    json_ready,
    make_ns_rollout_fn,
    ns_stability_stats,
    ns_step_schedule,
    torch_to_jax,
)

THIS_FILE = Path(__file__).resolve()
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "generalization_datasets" / "ns2d"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "analysis_outputs" / "ns2d_generalization_repair_reports"


def saved_x_to_rollout_input(x_saved: torch.Tensor) -> torch.Tensor:
    # Invert make_ns_rollout_fn.one(): saved x is the first saved frame after
    # rot90/flip orientation plus final axis swap.
    return torch.flip(torch.rot90(torch.swapaxes(x_saved, -1, -2), 1, dims=(-2, -1)), dims=(-2,)).contiguous()


def scan_unstable_indices(
    y: torch.Tensor,
    *,
    chunk_size: int,
    max_abs_threshold: float,
    include_initial_frame: bool,
) -> tuple[list[int], dict[str, Any], torch.Tensor]:
    n = int(y.shape[0])
    y_check = y if include_initial_frame else y[..., 1:]
    finite = torch.empty(n, dtype=torch.bool)
    max_abs = torch.empty(n, dtype=torch.float64)
    stable = torch.empty(n, dtype=torch.bool)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        stable_chunk, max_abs_chunk, finite_chunk = ns_stability_stats(y_check[start:end], max_abs_threshold)
        stable[start:end] = stable_chunk
        max_abs[start:end] = max_abs_chunk.double()
        finite[start:end] = finite_chunk
    unstable = ~stable
    indices = torch.nonzero(unstable, as_tuple=False).flatten().tolist()
    summary = {
        "sample_count": n,
        "stable_count": int(stable.sum().item()),
        "unstable_count": len(indices),
        "unstable_indices": [int(i) for i in indices],
        "nonfinite_indices": torch.nonzero(~finite, as_tuple=False).flatten().tolist(),
        "max_abs_threshold": max_abs_threshold,
        "include_initial_frame": bool(include_initial_frame),
        "checked_frames": "all_frames" if include_initial_frame else "solver_rollout_excluding_initial_frame",
        "max_abs_global_max": float(max_abs.max().item()) if n else 0.0,
        "max_abs_selected": {str(i): float(max_abs[i].item()) for i in indices},
    }
    return [int(i) for i in indices], summary, max_abs


def default_output_path(input_path: Path, output_root: Path | None, max_abs_threshold: float) -> Path:
    if output_root is None:
        return input_path.with_name(f"{input_path.stem}_repaired_maxabs{max_abs_threshold:g}{input_path.suffix}")
    return output_root / input_path.name


def ensure_sample_metadata(metadata: dict[str, Any], n: int, dataset_id: str) -> list[dict[str, Any]]:
    existing = metadata.get("sample_metadata")
    if isinstance(existing, list) and len(existing) == n:
        return [dict(item) if isinstance(item, dict) else {"sample_index": i, "dataset_id": dataset_id} for i, item in enumerate(existing)]
    return [{"sample_index": i, "dataset_id": dataset_id} for i in range(n)]


def rerun_unstable_samples(
    args: argparse.Namespace,
    x: torch.Tensor,
    y: torch.Tensor,
    target_indices: list[int],
    *,
    initial_fixed_step: float,
    sample_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    if not target_indices:
        return {"accepted": [], "failed": [], "step_records": []}

    device = torch.device("cuda")
    full_step_options = ns_step_schedule(initial_fixed_step, args.max_step_halvings)
    repair_step_options = full_step_options[1:] if not args.include_initial_step else full_step_options
    if not repair_step_options:
        raise ValueError("No repair steps available. Increase --max-step-halvings or use --include-initial-step.")

    remaining = list(target_indices)
    accepted: list[int] = []
    step_records = []
    rollout_cache: dict[tuple[int, float], Any] = {}

    for halving, step in enumerate(repair_step_options, start=(0 if args.include_initial_step else 1)):
        if not remaining:
            break
        step_t0 = time.perf_counter()
        next_remaining: list[int] = []
        accepted_this_step: list[int] = []
        for start in tqdm(range(0, len(remaining), args.solver_batch_size), desc=f"rerun dt={step:g}", unit="subbatch"):
            sub_indices = remaining[start : start + args.solver_batch_size]
            idx_cpu = torch.tensor(sub_indices, dtype=torch.long)
            x_saved = x.index_select(0, idx_cpu).to(device=device, dtype=torch.float32)
            u0 = saved_x_to_rollout_input(x_saved)
            sub_n = len(sub_indices)
            cache_key = (sub_n, float(step))
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
            y_check = y_try if args.include_initial_frame else y_try[..., 1:]
            stable, max_abs, finite = ns_stability_stats(y_check, args.max_abs_threshold)
            for local_pos, global_idx in enumerate(sub_indices):
                sample_metadata[global_idx].update(
                    {
                        "repair_last_fixed_step": float(step),
                        "repair_last_step_halvings": int(halving),
                        "repair_last_max_abs": float(max_abs[local_pos].item()),
                        "repair_last_finite": bool(finite[local_pos].item()),
                    }
                )
                if bool(stable[local_pos].item()):
                    y_new = y_try[local_pos].detach().cpu().contiguous()
                    y[global_idx].copy_(y_new)
                    x[global_idx].copy_(y_new[..., 0])
                    sample_metadata[global_idx].update(
                        {
                            "repaired": True,
                            "repair_fixed_step": float(step),
                            "repair_step_halvings": int(halving),
                            "repair_max_abs": float(max_abs[local_pos].item()),
                            "repair_finite": bool(finite[local_pos].item()),
                        }
                    )
                    accepted.append(global_idx)
                    accepted_this_step.append(global_idx)
                else:
                    next_remaining.append(global_idx)
            del idx_cpu, x_saved, u0, y_sub, y_try, stable, max_abs, finite
            torch.cuda.empty_cache()
        step_records.append(
            {
                "fixed_step": float(step),
                "step_halvings": int(halving),
                "attempted": int(len(remaining)),
                "accepted": int(len(accepted_this_step)),
                "remaining": int(len(next_remaining)),
                "accepted_indices": accepted_this_step,
                "seconds": time.perf_counter() - step_t0,
            }
        )
        remaining = next_remaining

    return {"accepted": sorted(accepted), "failed": sorted(remaining), "step_records": step_records}


def repair_one_file(args: argparse.Namespace, input_path: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    x = payload["x"].contiguous()
    y = payload["y"].contiguous()
    metadata = dict(payload.get("metadata", {}))
    dataset_id = str(metadata.get("dataset_id", input_path.stem))

    target_indices, before_scan, _ = scan_unstable_indices(
        y,
        chunk_size=args.scan_chunk_size,
        max_abs_threshold=args.max_abs_threshold,
        include_initial_frame=args.include_initial_frame,
    )
    if args.indices:
        manual = [int(item.strip()) for item in args.indices.split(",") if item.strip()]
        target_indices = sorted(set(target_indices).union(manual))
        before_scan["manual_indices_added"] = manual

    initial_fixed_step = float(args.initial_fixed_step or metadata.get("fixed_step_initial") or metadata.get("fixed_step") or 0.01)
    output_path = input_path if args.in_place else default_output_path(input_path, args.output_root, args.max_abs_threshold)
    if output_path.exists() and output_path != input_path and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")

    sample_metadata = ensure_sample_metadata(metadata, int(y.shape[0]), dataset_id)
    if args.dry_run or not target_indices:
        rerun_summary = {"accepted": [], "failed": target_indices if args.dry_run else [], "step_records": []}
        after_scan = before_scan
    else:
        rerun_summary = rerun_unstable_samples(
            args,
            x,
            y,
            target_indices,
            initial_fixed_step=initial_fixed_step,
            sample_metadata=sample_metadata,
        )
        _, after_scan, _ = scan_unstable_indices(
            y,
            chunk_size=args.scan_chunk_size,
            max_abs_threshold=args.max_abs_threshold,
            include_initial_frame=args.include_initial_frame,
        )

    if rerun_summary["failed"] and args.fail_on_unrepaired and not args.dry_run:
        raise RuntimeError(f"Still unstable after repair steps for {input_path}: {rerun_summary['failed']}")

    repair_metadata = {
        "repair_script": str(THIS_FILE),
        "source_input": str(input_path),
        "thresholds": {"max_abs_threshold": args.max_abs_threshold},
        "checked_frames": "all_frames" if args.include_initial_frame else "solver_rollout_excluding_initial_frame",
        "initial_fixed_step": initial_fixed_step,
        "include_initial_step": bool(args.include_initial_step),
        "max_step_halvings": args.max_step_halvings,
        "repair_step_options": ns_step_schedule(initial_fixed_step, args.max_step_halvings)[0 if args.include_initial_step else 1 :],
        "solver_batch_size": args.solver_batch_size,
        "solver_mode": args.solver_mode,
        "target_indices": target_indices,
        "repaired_indices": rerun_summary["accepted"],
        "unrepaired_indices": rerun_summary["failed"],
    }
    if not args.dry_run and target_indices:
        metadata.setdefault("repair_history", [])
        metadata["repair_history"].append(repair_metadata)
        metadata["sample_metadata"] = sample_metadata
        metadata["ns_ground_truth_stability_checked"] = True
        metadata["ns_max_abs_threshold"] = args.max_abs_threshold
        metadata["ns_stability_check_frames"] = "all_frames" if args.include_initial_frame else "solver_rollout_excluding_initial_frame"
        metadata["adaptive_step_halving_repair"] = True
        metadata["failed_stability_indices"] = rerun_summary["failed"]
        metadata["nsamples"] = int(y.shape[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_path = output_path
        if output_path == input_path:
            save_path = input_path.with_name(input_path.name + ".tmp_repair")
        torch.save({"x": x, "y": y, "metadata": metadata}, save_path)
        if save_path != output_path:
            save_path.replace(output_path)

    summary = {
        "input": input_path,
        "output": output_path,
        "dry_run": bool(args.dry_run),
        "in_place": bool(args.in_place),
        "before_scan": before_scan,
        "after_scan": after_scan,
        "rerun_summary": rerun_summary,
        "metadata_added": repair_metadata,
        "seconds": time.perf_counter() - t0,
    }
    return summary


def discover_files(args: argparse.Namespace) -> list[Path]:
    if args.files:
        files = [Path(item).resolve() for item in args.files]
    else:
        files = sorted(args.input_root.glob(args.glob))
    if args.limit_files is not None:
        files = files[: args.limit_files]
    return files


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--glob", default="*.pt")
    parser.add_argument("--files", nargs="*", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--indices", default=None, help="Comma-separated sample indices to rerun in every selected file, in addition to threshold hits.")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--max-abs-threshold", type=float, default=5.0)
    parser.add_argument("--max-step-halvings", type=int, default=6)
    parser.add_argument("--include-initial-frame", action="store_true", help="Also require the saved t=0 input frame to satisfy the amplitude threshold. By default only solver-generated t>0 frames are checked.")
    parser.add_argument("--include-initial-step", action="store_true")
    parser.add_argument("--initial-fixed-step", type=float, default=None)
    parser.add_argument("--solver-batch-size", type=int, default=5)
    parser.add_argument("--solver-mode", choices=["vmap", "lax-map"], default="vmap")
    parser.add_argument("--scan-chunk-size", type=int, default=10)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nu", type=float, default=1e-5)
    parser.add_argument("--t-final", type=int, default=20)
    parser.add_argument("--domain-extent", type=float, default=1.0)
    parser.add_argument("--fail-on-unrepaired", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.dry_run:
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is not available; refusing to repair NS ground truth on CPU")
        if jax.default_backend() != "gpu":
            raise RuntimeError(f"JAX backend is {jax.default_backend()!r}, expected 'gpu'")

    files = discover_files(args)
    if not files:
        raise FileNotFoundError(f"No files matched under {args.input_root} with glob {args.glob!r}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for input_path in tqdm(files, desc="NS files", unit="file"):
        summary = repair_one_file(args, input_path)
        summaries.append(summary)
        report_path = args.report_dir / f"{input_path.stem}_repair_report.json"
        report_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
        before = summary["before_scan"]["unstable_count"]
        after = summary["after_scan"]["unstable_count"]
        print(json.dumps(json_ready({"file": input_path.name, "unstable_before": before, "unstable_after": after, "repaired": len(summary["rerun_summary"]["accepted"])}), indent=2))

    combined = {
        "files": len(summaries),
        "unstable_before_total": int(sum(s["before_scan"]["unstable_count"] for s in summaries)),
        "unstable_after_total": int(sum(s["after_scan"]["unstable_count"] for s in summaries)),
        "repaired_total": int(sum(len(s["rerun_summary"]["accepted"]) for s in summaries)),
        "failed_total": int(sum(len(s["rerun_summary"]["failed"]) for s in summaries)),
        "summaries": summaries,
    }
    combined_path = args.report_dir / "combined_repair_report.json"
    combined_path.write_text(json.dumps(json_ready(combined), indent=2), encoding="utf-8")
    print(json.dumps(json_ready({k: combined[k] for k in ["files", "unstable_before_total", "unstable_after_total", "repaired_total", "failed_total"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
