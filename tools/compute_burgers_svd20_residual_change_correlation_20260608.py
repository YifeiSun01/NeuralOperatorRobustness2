#!/usr/bin/env python3
"""Compute residual-change correlations for existing Burgers SVD20 attack deltas.

This reuses saved final_delta_by_sample.npz files from prior SVD20 P2Q2 attack
runs. It does not rerun PGD and does not recompute Jacobian/SVD. For each saved
final delta, it computes

    delta_e = e(x + delta) - e(x)

where e(z) = model(z) - burgers_solver(z), then correlates residual-change
metrics with the existing error-Jacobian spectral-norm summary rows.
"""

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
SVD_ATTACK_SCRIPT = REPO / "tools" / "run_burgers_svd20_p2q2_attack_correlation_20260608.py"
FULL_ATTACK_SCRIPT = REPO / "tools" / "run_burgers_round03_full_p2q2_finalonly_attack.py"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    if "sm_70" not in info["arch_list"]:
        raise RuntimeError(f"PyTorch arch list does not include sm_70: {info['arch_list']}")
    x = torch.ones((128, 128), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    info["cuda_matmul_0_0"] = float(y[0, 0].item())
    return info


def run_nvidia_smi(out_path: Path) -> None:
    proc = subprocess.run(["nvidia-smi"], check=True, text=True, capture_output=True)
    out_path.write_text(proc.stdout)


def load_config(attack_root: Path) -> dict[str, object]:
    config_path = attack_root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return json.loads(config_path.read_text())


def resolve_root_config(attack_root: Path) -> tuple[Path, Path, dict[str, Path], list[str]]:
    cfg = load_config(attack_root)
    svd_manifest = Path(str(cfg["svd_manifest"]))
    svd_summary = Path(str(cfg["svd_summary"]))
    models_raw = cfg["models"]
    if not isinstance(models_raw, dict):
        raise ValueError(f"config models is not a dict in {attack_root}")
    model_paths = {str(k): Path(str(v)) for k, v in models_raw.items()}
    order_raw = cfg.get("model_order", list(model_paths.keys()))
    model_order = [str(x) for x in order_raw]
    return svd_manifest, svd_summary, model_paths, model_order


def load_error_svd_rows(svd_summary: Path, model_labels: set[str]) -> dict[tuple[int, str], dict[str, object]]:
    out: dict[tuple[int, str], dict[str, object]] = {}
    for row in read_csv(svd_summary):
        if row.get("jacobian_kind") != "error":
            continue
        raw_label = row["checkpoint_label"]
        label = raw_label
        if label not in model_labels and label.endswith("_error"):
            candidate = label[: -len("_error")]
            if candidate in model_labels:
                label = candidate
        if label not in model_labels:
            continue
        sample_id = int(row["sample_id"])
        out[(sample_id, label)] = {
            "sample_id": sample_id,
            "source_split": row["source_split"],
            "dataset_id": row["dataset_id"],
            "local_index": int(float(row["local_index"])),
            "model": label,
            "svd_checkpoint_label": raw_label,
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


def rms_l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).pow(2).mean(dim=1).sqrt()


def compute_model_residual_metrics(base_mod, model, x_clean_cpu: torch.Tensor, final_delta: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    n = x_clean_cpu.shape[0]
    arrays = {
        "initial_loss_mse": np.empty((n,), dtype=np.float64),
        "final_loss_mse": np.empty((n,), dtype=np.float64),
        "endpoint_growth_mse": np.empty((n,), dtype=np.float64),
        "clean_residual_rms": np.empty((n,), dtype=np.float64),
        "final_residual_rms": np.empty((n,), dtype=np.float64),
        "residual_change_rms": np.empty((n,), dtype=np.float64),
        "residual_change_mse": np.empty((n,), dtype=np.float64),
        "cross_term_mse": np.empty((n,), dtype=np.float64),
        "endpoint_decomp_error": np.empty((n,), dtype=np.float64),
        "residual_change_clean_cosine": np.empty((n,), dtype=np.float64),
    }
    device = torch.device("cuda")
    delta_cpu = torch.from_numpy(final_delta.astype(np.float32, copy=False))
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = x_clean_cpu[start:end].to(device=device, dtype=torch.float32).unsqueeze(-1)
            d = delta_cpu[start:end].to(device=device, dtype=torch.float32).unsqueeze(-1)
            x_adv = x + d
            solver0 = base_mod.burgers_solver_target(x)
            pred0 = model(x)
            e0 = pred0 - solver0
            solver1 = base_mod.burgers_solver_target(x_adv)
            pred1 = model(x_adv)
            e1 = pred1 - solver1
            de = e1 - e0
            init_loss = e0.pow(2).mean(dim=(1, 2))
            final_loss = e1.pow(2).mean(dim=(1, 2))
            de_mse = de.pow(2).mean(dim=(1, 2))
            cross = 2.0 * (e0 * de).mean(dim=(1, 2))
            e0_rms = rms_l2_norm(e0)
            e1_rms = rms_l2_norm(e1)
            de_rms = rms_l2_norm(de)
            cosine = (e0 * de).sum(dim=(1, 2)) / (e0.reshape(e0.shape[0], -1).norm(dim=1) * de.reshape(de.shape[0], -1).norm(dim=1) + 1e-12)
            sl = slice(start, end)
            arrays["initial_loss_mse"][sl] = init_loss.detach().cpu().numpy().astype(np.float64)
            arrays["final_loss_mse"][sl] = final_loss.detach().cpu().numpy().astype(np.float64)
            arrays["endpoint_growth_mse"][sl] = (final_loss - init_loss).detach().cpu().numpy().astype(np.float64)
            arrays["clean_residual_rms"][sl] = e0_rms.detach().cpu().numpy().astype(np.float64)
            arrays["final_residual_rms"][sl] = e1_rms.detach().cpu().numpy().astype(np.float64)
            arrays["residual_change_rms"][sl] = de_rms.detach().cpu().numpy().astype(np.float64)
            arrays["residual_change_mse"][sl] = de_mse.detach().cpu().numpy().astype(np.float64)
            arrays["cross_term_mse"][sl] = cross.detach().cpu().numpy().astype(np.float64)
            arrays["endpoint_decomp_error"][sl] = ((final_loss - init_loss) - (cross + de_mse)).abs().detach().cpu().numpy().astype(np.float64)
            arrays["residual_change_clean_cosine"][sl] = cosine.detach().cpu().numpy().astype(np.float64)
            del x, d, x_adv, solver0, pred0, e0, solver1, pred1, e1, de
    torch.cuda.synchronize()
    return arrays


def add_corr(rows: list[dict[str, object]], scope: str, root_name: str, summary: list[dict[str, object]]) -> None:
    if len(rows) < 2:
        return
    x_keys = ["error_spectral_norm", "fro_norm", "effective_rank", "sv_01", "sv_02", "sv_05", "sv_10", "sv_20"]
    y_keys = [
        "endpoint_growth_mse",
        "residual_change_mse",
        "residual_change_rms",
        "cross_term_mse",
        "residual_change_clean_cosine",
        "final_loss_mse",
        "initial_loss_mse",
    ]
    for x_key in x_keys:
        for y_key in y_keys:
            xs = [float(r[x_key]) for r in rows]
            ys = [float(r[y_key]) for r in rows]
            summary.append(
                {
                    "attack_root": root_name,
                    "scope": scope,
                    "n": len(rows),
                    "x": x_key,
                    "y": y_key,
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                }
            )


def process_attack_root(attack_root: Path, batch_size: int, gpu: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    svd_attack_mod = import_module(SVD_ATTACK_SCRIPT, "burgers_svd_attack_mod")
    full_attack_mod = import_module(FULL_ATTACK_SCRIPT, "burgers_full_attack_mod")
    base_mod = full_attack_mod.load_base_module()

    svd_manifest, svd_summary, model_paths, model_order = resolve_root_config(attack_root)
    x_all, manifest = svd_attack_mod.load_svd_manifest_samples(svd_manifest)
    n = int(x_all.shape[0])
    svd_by_key = load_error_svd_rows(svd_summary, set(model_order))
    joined: list[dict[str, object]] = []
    timing: dict[str, object] = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for model_name in model_order:
        ckpt = model_paths[model_name]
        delta_path = attack_root / model_name / "final_delta_by_sample.npz"
        if not delta_path.exists():
            raise FileNotFoundError(delta_path)
        final_delta = np.load(delta_path)["final_delta"].astype(np.float32)
        if final_delta.shape != (n, 1024):
            raise ValueError(f"{delta_path} has shape {final_delta.shape}, expected {(n, 1024)}")
        t0 = time.time()
        model = base_mod.load_model(ckpt, torch.device("cuda"))
        metrics = compute_model_residual_metrics(base_mod, model, x_all, final_delta, batch_size)
        timing[model_name] = {
            "seconds": time.time() - t0,
            "endpoint_growth_mse_mean": float(np.mean(metrics["endpoint_growth_mse"])),
            "residual_change_mse_mean": float(np.mean(metrics["residual_change_mse"])),
            "cross_term_mse_mean": float(np.mean(metrics["cross_term_mse"])),
            "max_abs_decomp_error": float(np.max(metrics["endpoint_decomp_error"])),
        }
        for i, rec in enumerate(manifest):
            key = (int(rec["sample_id"]), model_name)
            if key not in svd_by_key:
                continue
            out = dict(svd_by_key[key])
            out.update(
                {
                    "global_row": i,
                    "dataset_path": rec["dataset_path"],
                    "attack_root": attack_root.name,
                    "initial_loss_mse": float(metrics["initial_loss_mse"][i]),
                    "final_loss_mse": float(metrics["final_loss_mse"][i]),
                    "endpoint_growth_mse": float(metrics["endpoint_growth_mse"][i]),
                    "clean_residual_rms": float(metrics["clean_residual_rms"][i]),
                    "final_residual_rms": float(metrics["final_residual_rms"][i]),
                    "residual_change_rms": float(metrics["residual_change_rms"][i]),
                    "residual_change_mse": float(metrics["residual_change_mse"][i]),
                    "cross_term_mse": float(metrics["cross_term_mse"][i]),
                    "endpoint_decomp_error": float(metrics["endpoint_decomp_error"][i]),
                    "residual_change_clean_cosine": float(metrics["residual_change_clean_cosine"][i]),
                }
            )
            joined.append(out)
        del model
        torch.cuda.empty_cache()

    joined_fields = [
        "attack_root",
        "sample_id",
        "global_row",
        "source_split",
        "dataset_id",
        "local_index",
        "model",
        "svd_checkpoint_label",
        "error_spectral_norm",
        "fro_norm",
        "effective_rank",
        "sv_01",
        "sv_02",
        "sv_05",
        "sv_10",
        "sv_20",
        "initial_loss_mse",
        "final_loss_mse",
        "endpoint_growth_mse",
        "clean_residual_rms",
        "final_residual_rms",
        "residual_change_rms",
        "residual_change_mse",
        "cross_term_mse",
        "endpoint_decomp_error",
        "residual_change_clean_cosine",
        "dataset_path",
    ]
    write_csv(attack_root / "error_svd_residual_change_joined_rows.csv", joined, joined_fields)

    summary: list[dict[str, object]] = []
    trained = [r for r in joined if r["model"] != "baseline"]
    add_corr(joined, "all_samples_all_models", attack_root.name, summary)
    add_corr(trained, "all_samples_trained", attack_root.name, summary)
    for split in ["train", "test", "generalization"]:
        add_corr([r for r in joined if r["source_split"] == split], f"{split}_all_models", attack_root.name, summary)
        add_corr([r for r in trained if r["source_split"] == split], f"{split}_trained", attack_root.name, summary)
    for model_name in model_order:
        add_corr([r for r in joined if r["model"] == model_name], f"model_{model_name}", attack_root.name, summary)
    write_csv(attack_root / "error_svd_residual_change_correlation_summary.csv", summary, ["attack_root", "scope", "n", "x", "y", "pearson", "spearman"])
    run_info = {
        "created_unix_time": time.time(),
        "attack_root": str(attack_root),
        "svd_manifest": str(svd_manifest),
        "svd_summary": str(svd_summary),
        "batch_size": batch_size,
        "model_order": model_order,
        "gpu_preflight": gpu,
        "timing": timing,
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
    }
    (attack_root / "residual_change_metrics_config.json").write_text(json.dumps(run_info, indent=2) + "\n")
    return joined, summary


def extract_key_rows(all_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in all_summary:
        if row["scope"] not in {"all_samples_all_models", "all_samples_trained", "generalization_all_models", "generalization_trained"}:
            continue
        if row["x"] != "error_spectral_norm":
            continue
        if row["y"] not in {"endpoint_growth_mse", "residual_change_mse", "residual_change_rms", "cross_term_mse"}:
            continue
        out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-root", type=Path, action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--summary-csv", type=Path, default=REPO / "forensics" / "burgers_existing_svd_residual_change_correlation_key_summary_20260608.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    gpu = gpu_preflight()
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    run_nvidia_smi(args.summary_csv.parent / "burgers_existing_svd_residual_change_nvidia_smi_20260608.txt")
    all_summary: list[dict[str, object]] = []
    all_joined: list[dict[str, object]] = []
    for root in args.attack_root:
        root = root.resolve()
        print(json.dumps({"event": "root_start", "attack_root": str(root)}), flush=True)
        joined, summary = process_attack_root(root, args.batch_size, gpu)
        all_joined.extend(joined)
        all_summary.extend(summary)
        print(json.dumps({"event": "root_done", "attack_root": str(root), "joined_rows": len(joined), "summary_rows": len(summary)}), flush=True)
    key_rows = extract_key_rows(all_summary)
    write_csv(args.summary_csv, key_rows, ["attack_root", "scope", "n", "x", "y", "pearson", "spearman"])
    write_csv(args.summary_csv.with_name(args.summary_csv.stem + "_all_summary.csv"), all_summary, ["attack_root", "scope", "n", "x", "y", "pearson", "spearman"])
    print(json.dumps({"event": "done", "roots": len(args.attack_root), "key_summary": str(args.summary_csv), "key_rows": len(key_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
