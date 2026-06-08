#!/usr/bin/env python3
"""Attack the 20 Burgers SVD sample points and correlate with error-Jacobian norms."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
FULL_ATTACK_SCRIPT = REPO / "tools" / "run_burgers_round03_full_p2q2_finalonly_attack.py"
DEFAULT_SVD_ROOT = REPO / "forensics" / "burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606"
DEFAULT_SVD_MANIFEST = DEFAULT_SVD_ROOT / "round03_loss123_final_extension_sample_manifest.csv"
DEFAULT_SVD_SUMMARY = DEFAULT_SVD_ROOT / "round03_loss123_final_extension_jacobian_svd_summary.csv"

MODELS = {
    "baseline": REPO / "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt",
    "loss1_epoch5000": REPO / "adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue3000to5000_20260606/burgers/checkpoints/burgers_epoch5000_step015000.pt",
    "loss2_epoch2000": REPO / "adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt",
    "loss3_epoch1500": REPO / "adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt",
}
MODEL_ORDER = ["baseline", "loss1_epoch5000", "loss2_epoch2000", "loss3_epoch1500"]


def load_full_attack_module():
    spec = importlib.util.spec_from_file_location("burgers_full_attack", FULL_ATTACK_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(FULL_ATTACK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_x_dataset(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "x" not in data:
        raise ValueError(f"expected dict with x in {path}")
    x = data["x"].float()
    if x.ndim != 2 or x.shape[1] != 1024:
        raise ValueError(f"expected [N,1024] x tensor in {path}, got {tuple(x.shape)}")
    return x


def load_svd_manifest_samples(manifest_path: Path) -> tuple[torch.Tensor, list[dict[str, object]]]:
    rows = read_csv(manifest_path)
    cache: dict[Path, torch.Tensor] = {}
    xs: list[torch.Tensor] = []
    manifest: list[dict[str, object]] = []
    for row in rows:
        path = Path(row["dataset_path"])
        if path not in cache:
            cache[path] = load_x_dataset(path)
        idx = int(float(row["local_index"]))
        x = cache[path][idx].float()
        if x.ndim != 1 or x.numel() != 1024:
            raise ValueError(f"bad sample shape {tuple(x.shape)} for {path} index {idx}")
        xs.append(x)
        manifest.append(
            {
                "sample_id": int(row["sample_id"]),
                "source_split": row["source_split"],
                "dataset_id": row["dataset_id"],
                "dataset_path": str(path),
                "local_index": idx,
                "manual_rank": float(row["manual_rank"]) if row.get("manual_rank", "") != "" else math.nan,
            }
        )
    return torch.stack(xs, dim=0).contiguous(), manifest


def run_nvidia_smi(out_dir: Path) -> str:
    proc = subprocess.run(["nvidia-smi"], check=True, text=True, capture_output=True)
    (out_dir / "nvidia_smi.txt").write_text(proc.stdout)
    return proc.stdout


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


def pearson(x: list[float], y: list[float]) -> float:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 2 or float(xx.std()) == 0.0 or float(yy.std()) == 0.0:
        return math.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty(arr.size, dtype=np.float64)
    i = 0
    while i < arr.size:
        j = i + 1
        while j < arr.size and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 2:
        return math.nan
    return pearson(rankdata(xx.tolist()).tolist(), rankdata(yy.tolist()).tolist())


def load_error_svd_rows(svd_summary: Path) -> dict[tuple[int, str], dict[str, object]]:
    out: dict[tuple[int, str], dict[str, object]] = {}
    for row in read_csv(svd_summary):
        if row.get("jacobian_kind") != "error":
            continue
        label = row["checkpoint_label"]
        if label not in MODELS:
            continue
        sample_id = int(row["sample_id"])
        out[(sample_id, label)] = {
            "sample_id": sample_id,
            "source_split": row["source_split"],
            "dataset_id": row["dataset_id"],
            "local_index": int(float(row["local_index"])),
            "model": label,
            "error_spectral_norm": float(row["error_spectral_norm"] or row["spectral_norm"]),
            "fro_norm": float(row["fro_norm"]),
            "effective_rank": float(row["effective_rank"]),
            "sv_01": float(row["sv_01"]),
            "sv_02": float(row["sv_02"]),
            "sv_05": float(row["sv_05"]),
            "sv_10": float(row["sv_10"]),
            "sv_20": float(row["sv_20"]),
        }
    return out


def add_corr(rows: list[dict[str, object]], scope: str, summary: list[dict[str, object]]) -> None:
    for x_key in ["error_spectral_norm", "fro_norm", "effective_rank", "sv_01", "sv_02", "sv_05", "sv_10", "sv_20"]:
        for y_key in ["final_loss", "attack_increase", "final_diff_rms", "attack_increase_rms"]:
            xs = [float(r[x_key]) for r in rows]
            ys = [float(r[y_key]) for r in rows]
            summary.append(
                {
                    "scope": scope,
                    "n": len(rows),
                    "x": x_key,
                    "y": y_key,
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                }
            )


def compute_joined_correlations(out_dir: Path, manifest: list[dict[str, object]], svd_summary: Path) -> None:
    svd_by_key = load_error_svd_rows(svd_summary)
    joined: list[dict[str, object]] = []
    manifest_by_id = {int(row["sample_id"]): row for row in manifest}
    for model_name in MODEL_ORDER:
        loss_npz = np.load(out_dir / model_name / "losses_and_delta_rms_by_sample.npz")
        for i in range(len(manifest)):
            key = (int(manifest[i]["sample_id"]), model_name)
            if key not in svd_by_key:
                continue
            initial = float(loss_npz["initial_loss"][i])
            final = float(loss_npz["final_loss"][i])
            initial_rms = float(loss_npz["initial_diff_rms"][i])
            final_rms = float(loss_npz["final_diff_rms"][i])
            rec = dict(svd_by_key[key])
            rec.update(
                {
                    "global_row": i,
                    "dataset_path": manifest_by_id[int(manifest[i]["sample_id"])]["dataset_path"],
                    "initial_loss": initial,
                    "final_loss": final,
                    "attack_increase": final - initial,
                    "initial_diff_rms": initial_rms,
                    "final_diff_rms": final_rms,
                    "attack_increase_rms": final_rms - initial_rms,
                    "final_delta_rms": float(loss_npz["final_delta_rms"][i]),
                }
            )
            joined.append(rec)
    joined_fields = [
        "sample_id",
        "global_row",
        "source_split",
        "dataset_id",
        "local_index",
        "model",
        "error_spectral_norm",
        "fro_norm",
        "effective_rank",
        "sv_01",
        "sv_02",
        "sv_05",
        "sv_10",
        "sv_20",
        "initial_loss",
        "final_loss",
        "attack_increase",
        "initial_diff_rms",
        "final_diff_rms",
        "attack_increase_rms",
        "final_delta_rms",
        "dataset_path",
    ]
    write_csv(out_dir / "error_svd_attack_joined_rows.csv", joined, joined_fields)

    summary: list[dict[str, object]] = []
    add_corr(joined, "all_20_samples_all4_models", summary)
    add_corr([r for r in joined if r["model"] != "baseline"], "all_20_samples_trained3", summary)
    for split in ["train", "test", "generalization"]:
        add_corr([r for r in joined if r["source_split"] == split], f"{split}_all4_models", summary)
        add_corr([r for r in joined if r["source_split"] == split and r["model"] != "baseline"], f"{split}_trained3", summary)
    for model_name in MODEL_ORDER:
        add_corr([r for r in joined if r["model"] == model_name], f"model_{model_name}", summary)
    write_csv(out_dir / "error_svd_attack_correlation_summary.csv", summary, ["scope", "n", "x", "y", "pearson", "spearman"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svd-manifest", type=Path, default=DEFAULT_SVD_MANIFEST)
    parser.add_argument("--svd-summary", type=Path, default=DEFAULT_SVD_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=REPO / "forensics" / "burgers_svd20_p2q2_attack_correlation_20260608")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--models", nargs="*", default=MODEL_ORDER, choices=MODEL_ORDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_nvidia_smi(args.out_dir)
    gpu = gpu_preflight()
    full_attack = load_full_attack_module()
    base_mod = full_attack.load_base_module()
    x_all, manifest = load_svd_manifest_samples(args.svd_manifest)
    n = int(x_all.shape[0])
    if n != 20:
        print(json.dumps({"event": "warning", "message": f"expected 20 SVD samples, got {n}"}), flush=True)
    config = {
        "created_unix_time": time.time(),
        "svd_manifest": str(args.svd_manifest),
        "svd_summary": str(args.svd_summary),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "epsilon_rms": float(base_mod.EPSILON_RMS),
        "alpha_rms": float(base_mod.ALPHA_RMS),
        "models": {name: str(MODELS[name]) for name in args.models},
        "model_order": args.models,
        "sample_count": n,
        "gpu_preflight": gpu,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "start", "out_dir": str(args.out_dir), "samples": n, "models": args.models}, indent=2), flush=True)

    device = torch.device("cuda")
    x_all = x_all.contiguous()
    progress_path = args.out_dir / "progress.jsonl"
    summary_rows: list[dict[str, object]] = []
    timing: dict[str, object] = {}
    for model_name in args.models:
        ckpt = MODELS[model_name]
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        t0 = time.time()
        model = base_mod.load_model(ckpt, device)
        initial_loss = np.empty((n,), dtype=np.float32)
        initial_diff_rms = np.empty((n,), dtype=np.float32)
        final_loss = np.empty((n,), dtype=np.float32)
        final_diff_rms = np.empty((n,), dtype=np.float32)
        final_delta_rms = np.empty((n,), dtype=np.float32)
        final_delta = np.empty((n, 1024), dtype=np.float32)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            bt = time.time()
            x_batch = x_all[start:end].to(device=device, dtype=torch.float32).unsqueeze(-1)
            init_l, init_r = full_attack.run_initial_record(base_mod, model, x_batch)
            fin_l, fin_r, delta_np, delta_r = full_attack.attack_batch(
                base_mod,
                model,
                x_batch,
                args.steps,
                float(base_mod.EPSILON_RMS),
                float(base_mod.ALPHA_RMS),
            )
            initial_loss[start:end] = init_l
            initial_diff_rms[start:end] = init_r
            final_loss[start:end] = fin_l
            final_diff_rms[start:end] = fin_r
            final_delta[start:end] = delta_np
            final_delta_rms[start:end] = delta_r
            progress = {
                "event": "batch_done",
                "model": model_name,
                "start": start,
                "end": end,
                "batch_seconds": time.time() - bt,
                "initial_loss_mean": float(init_l.mean()),
                "final_loss_mean": float(fin_l.mean()),
                "attack_increase_mean": float((fin_l - init_l).mean()),
                "final_delta_rms_mean": float(delta_r.mean()),
                "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
            }
            with progress_path.open("a") as f:
                f.write(json.dumps(progress) + "\n")
            print(json.dumps(progress), flush=True)
            del x_batch
        model_dir = args.out_dir / model_name
        model_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            model_dir / "losses_and_delta_rms_by_sample.npz",
            initial_loss=initial_loss,
            initial_diff_rms=initial_diff_rms,
            final_loss=final_loss,
            final_diff_rms=final_diff_rms,
            final_delta_rms=final_delta_rms,
        )
        np.savez_compressed(model_dir / "final_delta_by_sample.npz", final_delta=final_delta)
        for i, rec in enumerate(manifest):
            summary_rows.append(
                {
                    "model": model_name,
                    "sample_id": rec["sample_id"],
                    "source_split": rec["source_split"],
                    "dataset_id": rec["dataset_id"],
                    "local_index": rec["local_index"],
                    "initial_loss": float(initial_loss[i]),
                    "final_loss": float(final_loss[i]),
                    "attack_increase": float(final_loss[i] - initial_loss[i]),
                    "initial_diff_rms": float(initial_diff_rms[i]),
                    "final_diff_rms": float(final_diff_rms[i]),
                    "attack_increase_rms": float(final_diff_rms[i] - initial_diff_rms[i]),
                    "final_delta_rms": float(final_delta_rms[i]),
                }
            )
        timing[model_name] = {
            "seconds": time.time() - t0,
            "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
            "initial_loss_mean": float(initial_loss.mean()),
            "final_loss_mean": float(final_loss.mean()),
            "attack_increase_mean": float((final_loss - initial_loss).mean()),
        }
        print(json.dumps({"event": "model_done", "model": model_name, **timing[model_name]}, indent=2), flush=True)
        del model
        torch.cuda.empty_cache()

    write_csv(
        args.out_dir / "attack_loss_by_model_sample.csv",
        summary_rows,
        [
            "model",
            "sample_id",
            "source_split",
            "dataset_id",
            "local_index",
            "initial_loss",
            "final_loss",
            "attack_increase",
            "initial_diff_rms",
            "final_diff_rms",
            "attack_increase_rms",
            "final_delta_rms",
        ],
    )
    compute_joined_correlations(args.out_dir, manifest, args.svd_summary)
    summary = {"config": config, "model_timing": timing, "finished_unix_time": time.time()}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "done", "out_dir": str(args.out_dir), "model_timing": timing}, indent=2), flush=True)


if __name__ == "__main__":
    main()
