#!/usr/bin/env python3
"""Evaluate Burgers random-field final models with the loss/attack/SVD suite.

This script is intentionally narrow: it uses the GitHub working-tree code, but
expects checkpoint/data artifacts to already exist under the standard project
paths.  It evaluates the two 2026-06-12 random-field models:

- random_clean_y: random delta, original clean target fixed
- random_solver_y: random delta, solver target recomputed at x + delta

Outputs are written incrementally so a long run can be monitored or resumed.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.analyze_fno_solver_jacobian_similarity import compute_solver_jacobian  # noqa: E402
from tools.analyze_local_jacobian_fno_deeponet import compute_explicit_jacobian  # noqa: E402
from tools.compare_burgers_adversarial_jacobian_svd import (  # noqa: E402
    aggregate_rows,
    load_burgers_model_from_checkpoint,
    load_x_sample,
    save_configured_svd,
    singular_rows,
    sv_summary,
)
from tools.evaluate_generalization_models import checkpoint_state, load_module, torch_load  # noqa: E402
from tools.run_burgers_round03_full_p2q2_finalonly_attack import (  # noqa: E402
    SampleRecord,
    attack_batch,
    fft_dataset_summaries,
    load_base_module,
    load_x,
    run_initial_record,
)


DEFAULT_TRAIN_TEST_ROOT = (
    REPO
    / "1D_Burgers/datasets/1D/Burgers/batched_exponax_splits"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
)
DEFAULT_TRAIN_PATH = DEFAULT_TRAIN_TEST_ROOT / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt"
DEFAULT_TEST_PATH = DEFAULT_TRAIN_TEST_ROOT / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt"
DEFAULT_GEN_ROOT = REPO / "generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers"
DEFAULT_SVD_MANIFEST = REPO / "forensics/burgers_wideparam_loss3targeted_full1024_svd_attack25_reuse3_20260611/sample_manifest.csv"
DEFAULT_OUT_ROOT = REPO / "forensics/burgers_random_field_final_models_full_suite_20260613"

MODEL_SPECS = {
    "random_clean_y": REPO / "adversarial_training_runs/burgers_wideparam_random_field_clean_y_2000ep_20260612/burgers/checkpoints/burgers_epoch2000_step002000.pt",
    "random_solver_y": REPO / "adversarial_training_runs/burgers_wideparam_random_field_solver_y_2000ep_20260612/burgers/checkpoints/burgers_epoch2000_step002000.pt",
}
MODEL_ORDER = ["random_clean_y", "random_solver_y"]


@dataclass
class DatasetInfo:
    split: str
    dataset_id: str
    path: Path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(payload)) + "\n")


def gpu_preflight() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    x = torch.ones((128, 128), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "arch_list": torch.cuda.get_arch_list(),
        "cuda_matmul_0_0": float(y[0, 0].item()),
    }


def load_model(path: Path, device: torch.device):
    mod = load_module("burgers_fno1d_random_field_eval", REPO / "1D_Burgers/models/FNO1d.py")
    model = mod.FNO1d(modes=16, width=64, num_layers=4, dtype=torch.float32).to(device)
    model.load_state_dict(checkpoint_state(path), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def sample_metrics(pred: torch.Tensor, y: torch.Tensor) -> dict[str, np.ndarray]:
    diff = pred - y
    flat = diff.flatten(1)
    yflat = y.flatten(1)
    mse = flat.square().mean(dim=1)
    rmse = torch.sqrt(mse)
    rel = torch.sqrt(flat.square().sum(dim=1) / yflat.square().sum(dim=1).clamp_min(1e-20))
    mae = flat.abs().mean(dim=1)
    return {
        "mse": mse.detach().cpu().numpy().astype(np.float64),
        "rmse": rmse.detach().cpu().numpy().astype(np.float64),
        "relative_l2": rel.detach().cpu().numpy().astype(np.float64),
        "mae": mae.detach().cpu().numpy().astype(np.float64),
    }


def dataset_infos(train_path: Path, test_path: Path, gen_root: Path) -> list[DatasetInfo]:
    infos = [
        DatasetInfo("train", "train_original_gaussian_corr0p03", train_path),
        DatasetInfo("test", "test_original_gaussian_corr0p03", test_path),
    ]
    for path in sorted(gen_root.glob("*.pt")):
        infos.append(DatasetInfo("generalization", path.stem, path))
    return infos


def evaluate_clean(args: argparse.Namespace, model_paths: dict[str, Path], out_dir: Path, device: torch.device) -> None:
    clean_dir = out_dir / "clean_loss"
    clean_dir.mkdir(parents=True, exist_ok=True)
    progress_path = clean_dir / "progress.jsonl"
    infos = dataset_infos(args.train_path, args.test_path, args.gen_root)
    models = {name: load_model(path, device) for name, path in model_paths.items()}
    dataset_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for dsi, info in enumerate(infos):
        data = torch_load(info.path)
        x = data["x"].float().unsqueeze(-1)
        y = data["y"].float().unsqueeze(-1)
        if args.clean_max_samples > 0:
            x = x[: args.clean_max_samples]
            y = y[: args.clean_max_samples]
        arrays = {name: {metric: [] for metric in ["mse", "rmse", "relative_l2", "mae"]} for name in model_paths}
        dt0 = time.time()
        with torch.no_grad():
            for start in range(0, int(x.shape[0]), args.clean_batch_size):
                xb = x[start : start + args.clean_batch_size].to(device)
                yb = y[start : start + args.clean_batch_size].to(device)
                for name, model in models.items():
                    metrics = sample_metrics(model(xb), yb)
                    for key, arr in metrics.items():
                        arrays[name][key].append(arr)
        row: dict[str, Any] = {
            "split": info.split,
            "dataset_id": info.dataset_id,
            "path": str(info.path.relative_to(REPO)),
            "num_samples": int(x.shape[0]),
        }
        for name in model_paths:
            for key in ["mse", "rmse", "relative_l2", "mae"]:
                arr = np.concatenate(arrays[name][key])
                row[f"{name}_{key}_mean"] = float(arr.mean())
                row[f"{name}_{key}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            for i in range(int(x.shape[0])):
                sample_rows.append({
                    "split": info.split,
                    "dataset_id": info.dataset_id,
                    "sample_index": i,
                    **{f"{name}_{metric}": float(np.concatenate(arrays[name][metric])[i]) for name in model_paths for metric in ["mse", "rmse", "relative_l2", "mae"]},
                })
        dataset_rows.append(row)
        event = {"event": "clean_dataset_done", "index": dsi + 1, "total": len(infos), "dataset_id": info.dataset_id, "seconds": time.time() - dt0}
        print(json.dumps(event), flush=True)
        append_jsonl(progress_path, event)
    write_csv(clean_dir / "per_dataset_clean_metrics.csv", dataset_rows)
    write_csv(clean_dir / "per_sample_clean_metrics.csv", sample_rows)
    summary: dict[str, Any] = {"elapsed_seconds": time.time() - t0, "datasets": len(dataset_rows), "models": list(model_paths)}
    for split in ["train", "test", "generalization"]:
        subset = [r for r in dataset_rows if r["split"] == split]
        if not subset:
            continue
        summary[split] = {}
        for name in model_paths:
            summary[split][name] = {metric: float(np.mean([r[f"{name}_{metric}_mean"] for r in subset])) for metric in ["mse", "rmse", "relative_l2", "mae"]}
    write_json(clean_dir / "summary.json", summary)
    del models
    torch.cuda.empty_cache()


def build_attack_samples(train_path: Path, test_path: Path, gen_root: Path, train_count: int, max_samples: int | None) -> tuple[torch.Tensor, list[SampleRecord], list[str]]:
    xs: list[torch.Tensor] = []
    manifest: list[SampleRecord] = []
    dataset_ids: list[str] = []

    def add_dataset(split: str, dataset_id: str, path: Path, x: torch.Tensor, indices: list[int]) -> None:
        dataset_ids.append(dataset_id)
        for offset, idx in enumerate(indices):
            manifest.append(SampleRecord(len(manifest), split, dataset_id, str(path), int(idx), offset))
        xs.append(x[indices])

    train_x = load_x(train_path)
    add_dataset("train", "train_original_gaussian_corr0p03_first50", train_path, train_x, list(range(min(train_count, int(train_x.shape[0])))))
    test_x = load_x(test_path)
    add_dataset("test", "test_original_gaussian_corr0p03", test_path, test_x, list(range(int(test_x.shape[0]))))
    for path in sorted(gen_root.glob("*.pt")):
        x = load_x(path)
        add_dataset("generalization", path.stem, path, x, list(range(int(x.shape[0]))))
    x_all = torch.cat(xs, dim=0).contiguous()
    if max_samples is not None and max_samples > 0:
        x_all = x_all[:max_samples].contiguous()
        manifest = manifest[:max_samples]
        dataset_ids = list(dict.fromkeys(rec.dataset_id for rec in manifest))
    return x_all, manifest, dataset_ids


def evaluate_attack(args: argparse.Namespace, model_paths: dict[str, Path], out_dir: Path, device: torch.device) -> None:
    attack_dir = out_dir / "p2q2_attack"
    attack_dir.mkdir(parents=True, exist_ok=True)
    progress_path = attack_dir / "progress.jsonl"
    base_mod = load_base_module()
    x_all, manifest, dataset_ids = build_attack_samples(args.train_path, args.test_path, args.gen_root, args.attack_train_count, args.attack_max_samples)
    write_json(attack_dir / "manifest.json", [asdict(r) for r in manifest])
    write_json(
        attack_dir / "config.json",
        {
            "steps": args.attack_steps,
            "batch_size": args.attack_batch_size,
            "epsilon_rms": float(base_mod.EPSILON_RMS),
            "alpha_rms": float(base_mod.ALPHA_RMS),
            "sample_count": int(x_all.shape[0]),
            "dataset_count": len(dataset_ids),
            "models": model_paths,
        },
    )
    rows: list[dict[str, Any]] = []
    for model_name, ckpt in model_paths.items():
        mt0 = time.time()
        model = base_mod.load_model(ckpt, device)
        n = int(x_all.shape[0])
        initial_loss = np.empty((n,), dtype=np.float32)
        initial_diff_rms = np.empty((n,), dtype=np.float32)
        final_loss = np.empty((n,), dtype=np.float32)
        final_diff_rms = np.empty((n,), dtype=np.float32)
        final_delta_rms = np.empty((n,), dtype=np.float32)
        final_delta = np.empty((n, 1024), dtype=np.float32)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, n, args.attack_batch_size):
            end = min(start + args.attack_batch_size, n)
            bt0 = time.time()
            x_batch = x_all[start:end].to(device=device, dtype=torch.float32).unsqueeze(-1)
            init_l, init_r = run_initial_record(base_mod, model, x_batch)
            fin_l, fin_r, delta_np, delta_r = attack_batch(base_mod, model, x_batch, args.attack_steps, float(base_mod.EPSILON_RMS), float(base_mod.ALPHA_RMS))
            initial_loss[start:end] = init_l
            initial_diff_rms[start:end] = init_r
            final_loss[start:end] = fin_l
            final_diff_rms[start:end] = fin_r
            final_delta[start:end] = delta_np
            final_delta_rms[start:end] = delta_r
            event = {
                "event": "attack_batch_done",
                "model": model_name,
                "start": start,
                "end": end,
                "seconds": time.time() - bt0,
                "initial_loss_mean": float(init_l.mean()),
                "final_loss_mean": float(fin_l.mean()),
                "attack_increase_mean": float((fin_l - init_l).mean()),
                "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
            }
            print(json.dumps(event), flush=True)
            append_jsonl(progress_path, event)
            del x_batch
        model_dir = attack_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model_dir / "losses_and_delta_rms_by_sample.npz", initial_loss=initial_loss, initial_diff_rms=initial_diff_rms, final_loss=final_loss, final_diff_rms=final_diff_rms, final_delta_rms=final_delta_rms)
        np.savez_compressed(model_dir / "final_delta_by_sample.npz", final_delta=final_delta)
        mean_power, fft_rows = fft_dataset_summaries(final_delta, manifest, dataset_ids)
        np.savez_compressed(model_dir / "fft_power_mean_by_dataset.npz", fft_power_mean=mean_power, dataset_ids=np.asarray(dataset_ids, dtype=object))
        write_csv(model_dir / "fft_summary_by_dataset.csv", fft_rows)
        ds_indices: dict[str, list[int]] = {ds: [] for ds in dataset_ids}
        for i, rec in enumerate(manifest):
            ds_indices[rec.dataset_id].append(i)
        for row in fft_rows:
            ds = str(row["dataset_id"])
            idx = np.asarray(ds_indices[ds], dtype=np.int64)
            rows.append({
                "model": model_name,
                "split": next((rec.split for rec in manifest if rec.dataset_id == ds), ""),
                "dataset_id": ds,
                "sample_count": int(idx.size),
                "initial_loss_mean": float(initial_loss[idx].mean()),
                "final_loss_mean": float(final_loss[idx].mean()),
                "attack_increase_mean": float((final_loss[idx] - initial_loss[idx]).mean()),
                "initial_diff_rms_mean": float(initial_diff_rms[idx].mean()),
                "final_diff_rms_mean": float(final_diff_rms[idx].mean()),
                "final_delta_rms_mean": float(final_delta_rms[idx].mean()),
                **row,
            })
        event = {"event": "attack_model_done", "model": model_name, "seconds": time.time() - mt0, "final_loss_mean": float(final_loss.mean())}
        print(json.dumps(event), flush=True)
        append_jsonl(progress_path, event)
        del model
        torch.cuda.empty_cache()
    write_csv(attack_dir / "summary_by_model_dataset.csv", rows)


def read_svd_manifest(path: Path, max_samples: int) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_samples > 0:
        rows = rows[:max_samples]
    out = []
    for i, row in enumerate(rows):
        row = dict(row)
        row["sample_id"] = int(row.get("sample_id", i))
        row["local_index"] = int(float(row["local_index"]))
        row["dataset_path"] = str(Path(row["dataset_path"]))
        out.append(row)
    return out


def find_attack_delta_for_sample(attack_dir: Path, model_name: str, sample: dict[str, Any]) -> np.ndarray | None:
    manifest_path = attack_dir / "manifest.json"
    delta_path = attack_dir / model_name / "final_delta_by_sample.npz"
    if not manifest_path.exists() or not delta_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_path = str(Path(sample["dataset_path"]).resolve())
    target_idx = int(sample["local_index"])
    for i, rec in enumerate(manifest):
        if str(Path(rec["source_path"]).resolve()) == target_path and int(rec["source_index"]) == target_idx:
            with np.load(delta_path) as z:
                return np.asarray(z["final_delta"][i], dtype=np.float64).reshape(-1)
    return None


def evaluate_svd(args: argparse.Namespace, model_paths: dict[str, Path], out_dir: Path, device: torch.device) -> None:
    svd_dir = out_dir / "jacobian_svd"
    svd_dir.mkdir(parents=True, exist_ok=True)
    progress_path = svd_dir / "progress.jsonl"
    samples = read_svd_manifest(args.svd_manifest, args.svd_max_samples)
    write_csv(svd_dir / "sample_manifest.csv", samples)
    solver_args = SimpleNamespace(
        burgers_nu=args.burgers_nu,
        burgers_t_final=args.burgers_t_final,
        burgers_dt=args.burgers_dt,
        burgers_domain=args.burgers_domain,
        burgers_jax_solver_dtype=args.burgers_jax_solver_dtype,
    )
    svd_args = SimpleNamespace(svd_method=args.svd_method, top_k=args.svd_top_k, svd_solver=args.svd_solver)
    models = {name: load_burgers_model_from_checkpoint(path, device) for name, path in model_paths.items()}
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    jb_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = int(sample["sample_id"])
        sample_dir = svd_dir / f"sample_{sample_id:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_json(sample_dir / "sample.json", sample)
        x = load_x_sample(Path(sample["dataset_path"]), int(sample["local_index"]))
        st0 = time.time()
        print(json.dumps({"event": "svd_sample_start", "sample_id": sample_id, "dataset_id": sample["dataset_id"]}), flush=True)
        J_solver = compute_solver_jacobian(x, solver_args, device, progress_prefix=f"solver_sample{sample_id}")
        s_solver = save_configured_svd("solver", J_solver, sample_dir, sample_id, svd_args)["s"]
        summary_rows.append(sv_summary("solver", s_solver, sample=sample, model_name="solver"))
        top_rows.extend(singular_rows(sample, "solver", "solver", s_solver, int(args.svd_top_k)))
        for model_name, model in models.items():
            mt0 = time.time()
            J_model = compute_explicit_jacobian(model, x, device, progress_prefix=f"{model_name}_sample{sample_id}")
            model_svd = save_configured_svd(model_name, J_model, sample_dir, sample_id, svd_args)
            s_model = model_svd["s"]
            summary_rows.append(sv_summary(f"{model_name}_model", s_model, sample=sample, model_name=model_name))
            top_rows.extend(singular_rows(sample, model_name, "model", s_model, int(args.svd_top_k)))
            J_error = J_model.astype(np.float64) - J_solver.astype(np.float64)
            error_svd = save_configured_svd(f"{model_name}_error", J_error, sample_dir, sample_id, svd_args)
            s_error = error_svd["s"]
            summary_rows.append(sv_summary(f"{model_name}_error", s_error, sample=sample, model_name=model_name))
            top_rows.extend(singular_rows(sample, model_name, "error", s_error, int(args.svd_top_k)))
            delta = find_attack_delta_for_sample(out_dir / "p2q2_attack", model_name, sample)
            if delta is not None:
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm > 0:
                    jb = J_error @ delta
                    right = np.asarray(error_svd["Vh"], dtype=np.float64)
                    v1 = right[0] / max(np.linalg.norm(right[0]), 1e-30)
                    jb_rows.append({
                        "sample_id": sample_id,
                        "source_split": sample.get("source_split", ""),
                        "dataset_id": sample["dataset_id"],
                        "local_index": int(sample["local_index"]),
                        "model": model_name,
                        "delta_l2": delta_norm,
                        "j_error_delta_l2": float(np.linalg.norm(jb)),
                        "j_error_delta_rms": float(np.sqrt(np.mean(jb * jb))),
                        "delta_cosine_top_error_right_singular_vector": float(np.dot(delta, v1) / max(delta_norm, 1e-30)),
                        "top_error_singular_value": float(s_error[0]),
                    })
            runtime_rows.append({"sample_id": sample_id, "component": model_name, "seconds": time.time() - mt0})
            torch.cuda.empty_cache()
        event = {"event": "svd_sample_done", "sample_id": sample_id, "seconds": time.time() - st0}
        print(json.dumps(event), flush=True)
        append_jsonl(progress_path, event)
        write_csv(svd_dir / "jacobian_svd_summary.partial.csv", summary_rows)
        write_csv(svd_dir / "top_singular_values_long.partial.csv", top_rows)
        write_csv(svd_dir / "j_error_times_attack_delta.partial.csv", jb_rows)
    write_csv(svd_dir / "jacobian_svd_summary.csv", summary_rows)
    write_csv(svd_dir / "top_singular_values_long.csv", top_rows)
    write_csv(svd_dir / "aggregate_jacobian_svd_summary.csv", aggregate_rows(summary_rows))
    write_csv(svd_dir / "j_error_times_attack_delta.csv", jb_rows)
    write_csv(svd_dir / "runtime.csv", runtime_rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    ap.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    ap.add_argument("--gen-root", type=Path, default=DEFAULT_GEN_ROOT)
    ap.add_argument("--svd-manifest", type=Path, default=DEFAULT_SVD_MANIFEST)
    ap.add_argument("--stage", action="append", choices=["clean", "attack", "svd"], help="May be passed multiple times. Default: all stages.")
    ap.add_argument("--clean-batch-size", type=int, default=256)
    ap.add_argument("--clean-max-samples", type=int, default=0)
    ap.add_argument("--attack-steps", type=int, default=20)
    ap.add_argument("--attack-batch-size", type=int, default=500)
    ap.add_argument("--attack-train-count", type=int, default=50)
    ap.add_argument("--attack-max-samples", type=int, default=0)
    ap.add_argument("--svd-max-samples", type=int, default=25)
    ap.add_argument("--svd-method", choices=["full", "topk"], default="topk")
    ap.add_argument("--svd-top-k", type=int, default=20)
    ap.add_argument("--svd-solver", choices=["propack", "arpack", "lobpcg"], default="propack")
    ap.add_argument("--burgers-nu", type=float, default=0.001)
    ap.add_argument("--burgers-t-final", type=float, default=1.0)
    ap.add_argument("--burgers-dt", type=float, default=0.001)
    ap.add_argument("--burgers-domain", type=float, default=2.0)
    ap.add_argument("--burgers-jax-solver-dtype", choices=["float32", "float64"], default="float64")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_root = args.out_root.resolve()
    args.train_path = args.train_path.resolve()
    args.test_path = args.test_path.resolve()
    args.gen_root = args.gen_root.resolve()
    args.svd_manifest = args.svd_manifest.resolve()
    args.attack_max_samples = None if int(args.attack_max_samples) <= 0 else int(args.attack_max_samples)
    stages = args.stage or ["clean", "attack", "svd"]
    args.out_root.mkdir(parents=True, exist_ok=True)
    preflight = gpu_preflight()
    write_json(args.out_root / "gpu_preflight.json", preflight)
    model_paths = {name: path.resolve() for name, path in MODEL_SPECS.items()}
    for label, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")
    for path in [args.train_path, args.test_path, args.gen_root, args.svd_manifest]:
        if not path.exists():
            raise FileNotFoundError(path)
    write_json(args.out_root / "config.json", {"args": vars(args), "stages": stages, "models": model_paths, "gpu_preflight": preflight})
    device = torch.device("cuda")
    for stage in stages:
        event = {"event": "stage_start", "stage": stage, "unix_time": time.time()}
        print(json.dumps(event), flush=True)
        append_jsonl(args.out_root / "progress.jsonl", event)
        if stage == "clean":
            evaluate_clean(args, model_paths, args.out_root, device)
        elif stage == "attack":
            evaluate_attack(args, model_paths, args.out_root, device)
        elif stage == "svd":
            evaluate_svd(args, model_paths, args.out_root, device)
        event = {"event": "stage_done", "stage": stage, "unix_time": time.time()}
        print(json.dumps(event), flush=True)
        append_jsonl(args.out_root / "progress.jsonl", event)
    write_json(args.out_root / "done.json", {"status": "complete", "finished_unix_time": time.time(), "stages": stages})
    print(json.dumps({"event": "done", "out_root": str(args.out_root), "stages": stages}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
