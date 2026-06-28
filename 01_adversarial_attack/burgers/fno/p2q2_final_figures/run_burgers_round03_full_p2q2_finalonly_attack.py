#!/usr/bin/env python3
"""Run final-only Burgers round03 p2q2 attacks over train/test/generalization datasets.

This script records compact outputs only: initial loss, final loss, final delta,
and per-dataset mean FFT power of the final delta. It intentionally does not
save intermediate attack trajectories.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO / "tools" / "plot_burgers_p2q2_baseline_vs_epoch1000_attack_gif.py"
TRAIN_TEST_ROOT = (
    REPO
    / "1D_Burgers/datasets/1D/Burgers/batched_exponax_splits"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
)
TRAIN_PATH = TRAIN_TEST_ROOT / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt"
TEST_PATH = TRAIN_TEST_ROOT / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt"
GEN_ROOT = REPO / "generalization_datasets_rmse_1p5_3x_all_ns50/burgers"

MODELS = {
    "baseline": REPO / "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt",
    "loss1_epoch8000": REPO / "adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt",
    "loss2_epoch2000": REPO / "adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt",
    "loss3_epoch1500": REPO / "adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt",
}
MODEL_ORDER = ["baseline", "loss1_epoch8000", "loss2_epoch2000", "loss3_epoch1500"]


@dataclass
class SampleRecord:
    global_sample_id: int
    split: str
    dataset_id: str
    source_path: str
    source_index: int
    dataset_sample_offset: int


def load_base_module():
    spec = importlib.util.spec_from_file_location("burgers_p2q2_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_x(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "x" not in data:
        raise ValueError(f"expected dict with x tensor in {path}")
    x = data["x"].float()
    if x.ndim != 2 or x.shape[1] != 1024:
        raise ValueError(f"expected x shape [N,1024], got {tuple(x.shape)} from {path}")
    return x


def build_samples(train_count: int, gen_root: Path) -> tuple[torch.Tensor, list[SampleRecord], list[str]]:
    xs: list[torch.Tensor] = []
    manifest: list[SampleRecord] = []
    dataset_ids: list[str] = []

    def add_dataset(split: str, dataset_id: str, path: Path, x: torch.Tensor, indices: Iterable[int]) -> None:
        dataset_ids.append(dataset_id)
        for offset, idx in enumerate(indices):
            manifest.append(
                SampleRecord(
                    global_sample_id=len(manifest),
                    split=split,
                    dataset_id=dataset_id,
                    source_path=str(path),
                    source_index=int(idx),
                    dataset_sample_offset=offset,
                )
            )
        xs.append(x[list(indices)])

    train_x = load_x(TRAIN_PATH)
    train_indices = list(range(min(train_count, train_x.shape[0])))
    add_dataset("train", "train_original_gaussian_corr0p03_first50", TRAIN_PATH, train_x, train_indices)

    test_x = load_x(TEST_PATH)
    add_dataset("test", "test_original_gaussian_corr0p03", TEST_PATH, test_x, range(test_x.shape[0]))

    for path in sorted(gen_root.glob("*.pt")):
        x = load_x(path)
        add_dataset("generalization", path.stem, path, x, range(x.shape[0]))

    x_all = torch.cat(xs, dim=0).contiguous()
    if len(manifest) != x_all.shape[0]:
        raise RuntimeError("manifest/sample count mismatch")
    return x_all, manifest, dataset_ids


def rms_l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).pow(2).mean(dim=1).sqrt()


def run_initial_record(base_mod, model, x_clean: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        solver = base_mod.burgers_solver_target(x_clean)
        pred = model(x_clean)
        loss = (pred - solver).pow(2).mean(dim=(1, 2))
        diff_rms = rms_l2_norm(pred - solver)
    torch.cuda.synchronize()
    return loss.detach().cpu().numpy().astype(np.float32), diff_rms.detach().cpu().numpy().astype(np.float32)


def attack_batch(base_mod, model, x_clean: torch.Tensor, steps: int, epsilon_rms: float, alpha_rms: float, initial_delta: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    delta = torch.zeros_like(x_clean) if initial_delta is None else initial_delta.detach().clone().to(device=x_clean.device, dtype=x_clean.dtype)
    if delta.shape != x_clean.shape:
        raise ValueError(f"initial_delta shape {tuple(delta.shape)} does not match x_clean shape {tuple(x_clean.shape)}")
    for _step in range(1, steps + 1):
        delta_var = delta.detach().clone().requires_grad_(True)
        x_adv = x_clean + delta_var
        solver = base_mod.burgers_solver_target(x_adv)
        pred = model(x_adv)
        loss = (pred - solver).pow(2).mean(dim=(1, 2)).mean()
        grad = torch.autograd.grad(loss, delta_var, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            delta = delta_var + alpha_rms * base_mod.normalize_rms_l2(grad)
            delta = base_mod.project_rms_l2(delta, epsilon_rms)
    with torch.no_grad():
        x_final = x_clean + delta
        solver = base_mod.burgers_solver_target(x_final)
        pred = model(x_final)
        final_loss = (pred - solver).pow(2).mean(dim=(1, 2))
        final_diff_rms = rms_l2_norm(pred - solver)
        delta_rms = rms_l2_norm(delta)
    torch.cuda.synchronize()
    return (
        final_loss.detach().cpu().numpy().astype(np.float32),
        final_diff_rms.detach().cpu().numpy().astype(np.float32),
        delta.detach().cpu().numpy()[..., 0].astype(np.float32),
        delta_rms.detach().cpu().numpy().astype(np.float32),
    )


def fft_dataset_summaries(final_delta: np.ndarray, manifest: list[SampleRecord], dataset_ids: list[str]) -> tuple[np.ndarray, list[dict[str, float | str | int]]]:
    n = final_delta.shape[1]
    fft = np.fft.rfft(final_delta, axis=1)
    power = (np.abs(fft) ** 2 / float(n)).astype(np.float64)
    modes = np.arange(power.shape[1], dtype=np.float64)
    mean_power = np.zeros((len(dataset_ids), power.shape[1]), dtype=np.float32)
    rows: list[dict[str, float | str | int]] = []
    ds_to_indices: dict[str, list[int]] = {ds: [] for ds in dataset_ids}
    for i, rec in enumerate(manifest):
        ds_to_indices[rec.dataset_id].append(i)
    for dsi, ds in enumerate(dataset_ids):
        idx = np.asarray(ds_to_indices[ds], dtype=np.int64)
        p = power[idx].mean(axis=0)
        mean_power[dsi] = p.astype(np.float32)
        nonzero = p[1:]
        denom = float(nonzero.sum())
        if denom <= 0:
            low = mid = high = centroid = float("nan")
        else:
            low = float(p[1:33].sum() / denom)
            mid = float(p[33:129].sum() / denom)
            high = float(p[129:].sum() / denom)
            centroid = float((modes[1:] * nonzero).sum() / denom)
        rows.append(
            {
                "dataset_id": ds,
                "dataset_index": dsi,
                "sample_count": int(idx.size),
                "fft_total_nonzero_power": denom,
                "fft_low_1_32_frac": low,
                "fft_mid_33_128_frac": mid,
                "fft_high_129_512_frac": high,
                "fft_spectral_centroid": centroid,
            }
        )
    return mean_power, rows


def load_resume_arrays(resume_root: Path, model_name: str, n: int) -> dict[str, np.ndarray]:
    model_dir = resume_root / model_name
    delta_path = model_dir / "final_delta_by_sample.npz"
    loss_path = model_dir / "losses_and_delta_rms_by_sample.npz"
    if not delta_path.exists():
        raise FileNotFoundError(f"missing resume delta for {model_name}: {delta_path}")
    if not loss_path.exists():
        raise FileNotFoundError(f"missing resume losses for {model_name}: {loss_path}")
    delta = np.load(delta_path)["final_delta"].astype(np.float32)
    losses = np.load(loss_path)
    if delta.shape != (n, 1024):
        raise ValueError(f"resume delta for {model_name} has shape {delta.shape}, expected {(n, 1024)}")
    out = {
        "resume_delta": delta,
        "original_initial_loss": losses["initial_loss"].astype(np.float32),
        "original_initial_diff_rms": losses["initial_diff_rms"].astype(np.float32),
        "resume_start_loss": losses["final_loss"].astype(np.float32),
        "resume_start_diff_rms": losses["final_diff_rms"].astype(np.float32),
        "resume_start_delta_rms": losses["final_delta_rms"].astype(np.float32),
    }
    for key, arr in out.items():
        if arr.shape[0] != n:
            raise ValueError(f"resume array {key} for {model_name} has first dim {arr.shape[0]}, expected {n}")
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def gpu_preflight() -> dict[str, object]:
    info: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    info.update(
        {
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "arch_list": torch.cuda.get_arch_list(),
        }
    )
    x = torch.ones((128, 128), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    info["cuda_matmul_0_0"] = float(y[0, 0].item())
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--train-count", type=int, default=50)
    parser.add_argument("--gen-root", type=Path, default=GEN_ROOT)
    parser.add_argument("--run-name", default="burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607")
    parser.add_argument("--out-root", type=Path, default=REPO / "forensics")
    parser.add_argument("--models", nargs="*", default=MODEL_ORDER, choices=MODEL_ORDER)
    parser.add_argument("--max-samples", type=int, default=None, help="debug only: truncate samples after manifest construction")
    parser.add_argument("--resume-from-root", type=Path, default=None, help="existing final-only run root containing per-model final_delta_by_sample.npz")
    parser.add_argument("--resume-start-steps", type=int, default=None, help="completed attack steps in --resume-from-root; inferred from its config.json when omitted")
    parser.add_argument("--target-total-steps", type=int, default=None, help="total desired attack steps after resume, e.g. 40 or 60; additional steps are target minus resume-start")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("steps must be nonnegative")
    out_dir = args.out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "delta_by_model").mkdir(exist_ok=True)
    progress_path = out_dir / "progress.jsonl"
    base_mod = load_base_module()
    gpu = gpu_preflight()
    device = torch.device("cuda")
    gen_root = args.gen_root
    if not gen_root.is_absolute():
        gen_root = (REPO / gen_root).resolve()
    if not gen_root.exists():
        raise FileNotFoundError(f"generalization root does not exist: {gen_root}")
    if not any(gen_root.glob("*.pt")):
        raise FileNotFoundError(f"generalization root has no .pt files: {gen_root}")
    x_all, manifest, dataset_ids = build_samples(args.train_count, gen_root)
    if args.max_samples is not None:
        x_all = x_all[: args.max_samples].contiguous()
        manifest = manifest[: args.max_samples]
        dataset_ids = list(dict.fromkeys(rec.dataset_id for rec in manifest))
    n = int(x_all.shape[0])

    resume_root = args.resume_from_root
    resume_start_steps = 0
    additional_steps = args.steps
    target_total_steps = args.steps
    if resume_root is not None:
        resume_root = resume_root.resolve()
        if not resume_root.exists():
            raise FileNotFoundError(resume_root)
        resume_config_path = resume_root / "config.json"
        resume_config = json.loads(resume_config_path.read_text()) if resume_config_path.exists() else {}
        resume_start_steps = args.resume_start_steps if args.resume_start_steps is not None else int(resume_config.get("target_total_steps", resume_config.get("steps", 0)))
        if args.target_total_steps is not None:
            target_total_steps = args.target_total_steps
            additional_steps = target_total_steps - resume_start_steps
        else:
            additional_steps = args.steps
            target_total_steps = resume_start_steps + additional_steps
        if additional_steps < 0:
            raise ValueError(f"target_total_steps {target_total_steps} is less than resume_start_steps {resume_start_steps}")
        if args.max_samples is not None:
            raise ValueError("--max-samples is not supported with --resume-from-root because resume arrays are full-run arrays")
    config = {
        "run_name": args.run_name,
        "created_unix_time": time.time(),
        "steps": additional_steps,
        "resume_from_root": str(resume_root) if resume_root is not None else None,
        "resume_start_steps": resume_start_steps,
        "target_total_steps": target_total_steps,
        "batch_size": args.batch_size,
        "epsilon_rms": float(base_mod.EPSILON_RMS),
        "alpha_rms": float(base_mod.ALPHA_RMS),
        "train_count": args.train_count,
        "gen_root": str(gen_root),
        "model_order": args.models,
        "models": {name: str(MODELS[name]) for name in args.models},
        "sample_count": n,
        "dataset_count": len(dataset_ids),
        "dataset_ids": dataset_ids,
        "gpu_preflight": gpu,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps([asdict(r) for r in manifest], indent=2) + "\n")
    print(json.dumps({"event": "start", "out_dir": str(out_dir), "sample_count": n, "dataset_count": len(dataset_ids), "additional_steps": additional_steps, "resume_start_steps": resume_start_steps, "target_total_steps": target_total_steps, "batch_size": args.batch_size, "resume_from_root": str(resume_root) if resume_root is not None else None}, indent=2), flush=True)

    all_summary_rows: list[dict[str, object]] = []
    model_timing: dict[str, object] = {}
    for model_name in args.models:
        model_start = time.time()
        ckpt = MODELS[model_name]
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        print(json.dumps({"event": "load_model", "model": model_name, "checkpoint": str(ckpt)}), flush=True)
        model = base_mod.load_model(ckpt, device)
        resume_arrays = load_resume_arrays(resume_root, model_name, n) if resume_root is not None else None
        initial_loss = np.empty((n,), dtype=np.float32)
        initial_diff_rms = np.empty((n,), dtype=np.float32)
        resume_start_loss = np.full((n,), np.nan, dtype=np.float32)
        resume_start_diff_rms = np.full((n,), np.nan, dtype=np.float32)
        resume_start_delta_rms = np.full((n,), np.nan, dtype=np.float32)
        final_loss = np.empty((n,), dtype=np.float32)
        final_diff_rms = np.empty((n,), dtype=np.float32)
        final_delta_rms = np.empty((n,), dtype=np.float32)
        final_delta = np.empty((n, 1024), dtype=np.float32)
        if resume_arrays is not None:
            initial_loss[:] = resume_arrays["original_initial_loss"]
            initial_diff_rms[:] = resume_arrays["original_initial_diff_rms"]
            resume_start_loss[:] = resume_arrays["resume_start_loss"]
            resume_start_diff_rms[:] = resume_arrays["resume_start_diff_rms"]
            resume_start_delta_rms[:] = resume_arrays["resume_start_delta_rms"]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            batch_t0 = time.time()
            x_batch = x_all[start:end].to(device=device, dtype=torch.float32).unsqueeze(-1)
            if resume_arrays is None:
                init_l, init_r = run_initial_record(base_mod, model, x_batch)
                initial_loss[start:end] = init_l
                initial_diff_rms[start:end] = init_r
                initial_delta = None
            else:
                initial_delta = torch.from_numpy(resume_arrays["resume_delta"][start:end]).to(device=device, dtype=torch.float32).unsqueeze(-1)
            fin_l, fin_r, delta_np, delta_r = attack_batch(base_mod, model, x_batch, additional_steps, float(base_mod.EPSILON_RMS), float(base_mod.ALPHA_RMS), initial_delta=initial_delta)
            final_loss[start:end] = fin_l
            final_diff_rms[start:end] = fin_r
            final_delta[start:end] = delta_np
            final_delta_rms[start:end] = delta_r
            batch_sec = time.time() - batch_t0
            progress = {
                "event": "batch_done",
                "model": model_name,
                "start": start,
                "end": end,
                "samples": end - start,
                "resume_start_steps": resume_start_steps,
                "additional_steps": additional_steps,
                "target_total_steps": target_total_steps,
                "batch_seconds": batch_sec,
                "initial_loss_mean": float(initial_loss[start:end].mean()),
                "resume_start_loss_mean": float(resume_start_loss[start:end].mean()) if resume_arrays is not None else None,
                "resume_start_delta_rms_mean": float(resume_start_delta_rms[start:end].mean()) if resume_arrays is not None else None,
                "final_loss_mean": float(fin_l.mean()),
                "final_delta_rms_mean": float(delta_r.mean()),
                "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
                "unix_time": time.time(),
            }
            with progress_path.open("a") as f:
                f.write(json.dumps(progress) + "\n")
            print(json.dumps(progress), flush=True)
            del x_batch
        mean_power, fft_rows = fft_dataset_summaries(final_delta, manifest, dataset_ids)
        model_dir = out_dir / model_name
        model_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            model_dir / "losses_and_delta_rms_by_sample.npz",
            initial_loss=initial_loss,
            initial_diff_rms=initial_diff_rms,
            final_loss=final_loss,
            final_diff_rms=final_diff_rms,
            final_delta_rms=final_delta_rms,
            resume_start_loss=resume_start_loss,
            resume_start_diff_rms=resume_start_diff_rms,
            resume_start_delta_rms=resume_start_delta_rms,
            resume_start_steps=np.asarray(resume_start_steps, dtype=np.int32),
            additional_steps=np.asarray(additional_steps, dtype=np.int32),
            target_total_steps=np.asarray(target_total_steps, dtype=np.int32),
        )
        np.savez_compressed(model_dir / "final_delta_by_sample.npz", final_delta=final_delta)
        np.savez_compressed(model_dir / "fft_power_mean_by_dataset.npz", fft_power_mean=mean_power, dataset_ids=np.asarray(dataset_ids, dtype=object))
        write_csv(
            model_dir / "fft_summary_by_dataset.csv",
            fft_rows,
            ["dataset_id", "dataset_index", "sample_count", "fft_total_nonzero_power", "fft_low_1_32_frac", "fft_mid_33_128_frac", "fft_high_129_512_frac", "fft_spectral_centroid"],
        )
        ds_indices: dict[str, list[int]] = {ds: [] for ds in dataset_ids}
        for i, rec in enumerate(manifest):
            ds_indices[rec.dataset_id].append(i)
        for row in fft_rows:
            ds = str(row["dataset_id"])
            idx = np.asarray(ds_indices[ds], dtype=np.int64)
            init_mean = float(initial_loss[idx].mean())
            final_mean = float(final_loss[idx].mean())
            all_summary_rows.append(
                {
                    "model": model_name,
                    "dataset_id": ds,
                    "dataset_index": row["dataset_index"],
                    "sample_count": row["sample_count"],
                    "initial_loss_mean": init_mean,
                    "final_loss_mean": final_mean,
                    "loss_ratio_final_over_initial": final_mean / max(init_mean, 1e-30),
                    "initial_diff_rms_mean": float(initial_diff_rms[idx].mean()),
                    "resume_start_loss_mean": float(np.nanmean(resume_start_loss[idx])) if resume_root is not None else float("nan"),
                    "resume_start_delta_rms_mean": float(np.nanmean(resume_start_delta_rms[idx])) if resume_root is not None else float("nan"),
                    "final_diff_rms_mean": float(final_diff_rms[idx].mean()),
                    "final_delta_rms_mean": float(final_delta_rms[idx].mean()),
                    "fft_low_1_32_frac": row["fft_low_1_32_frac"],
                    "fft_mid_33_128_frac": row["fft_mid_33_128_frac"],
                    "fft_high_129_512_frac": row["fft_high_129_512_frac"],
                    "fft_spectral_centroid": row["fft_spectral_centroid"],
                }
            )
        model_sec = time.time() - model_start
        model_timing[model_name] = {
            "seconds": model_sec,
            "resume_start_steps": resume_start_steps,
            "additional_steps": additional_steps,
            "target_total_steps": target_total_steps,
            "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
            "initial_loss_mean": float(initial_loss.mean()),
            "final_loss_mean": float(final_loss.mean()),
            "final_delta_rms_mean": float(final_delta_rms.mean()),
        }
        print(json.dumps({"event": "model_done", "model": model_name, **model_timing[model_name]}, indent=2), flush=True)
        del model
        torch.cuda.empty_cache()
    write_csv(
        out_dir / "summary_by_model_dataset.csv",
        all_summary_rows,
        [
            "model", "dataset_id", "dataset_index", "sample_count", "initial_loss_mean", "final_loss_mean", "loss_ratio_final_over_initial",
            "initial_diff_rms_mean", "resume_start_loss_mean", "resume_start_delta_rms_mean", "final_diff_rms_mean", "final_delta_rms_mean", "fft_low_1_32_frac", "fft_mid_33_128_frac", "fft_high_129_512_frac", "fft_spectral_centroid",
        ],
    )
    final_summary = {"config": config, "model_timing": model_timing, "finished_unix_time": time.time()}
    (out_dir / "summary.json").write_text(json.dumps(final_summary, indent=2) + "\n")
    print(json.dumps({"event": "done", "out_dir": str(out_dir), "model_timing": model_timing}, indent=2), flush=True)


if __name__ == "__main__":
    main()
