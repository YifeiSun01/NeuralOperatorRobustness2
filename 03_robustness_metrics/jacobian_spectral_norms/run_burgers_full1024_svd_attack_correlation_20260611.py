#!/usr/bin/env python3
"""Run a 3-sample full-1024 Burgers SVD vs 15-step attack correlation probe.

This intentionally uses full 1024 x 1024 dense Jacobians and full SVD. It does
not use coarse/block-projection SVD proxies.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analyze_fno_solver_jacobian_similarity import compute_solver_jacobian  # noqa: E402
from tools.analyze_local_jacobian_fno_deeponet import compute_explicit_jacobian  # noqa: E402
from tools.plot_burgers_p2q2_baseline_vs_epoch1000_attack_gif import (  # noqa: E402
    burgers_solver_target,
    load_model,
    normalize_rms_l2,
    project_rms_l2,
    rms_l2_norm,
)

BASELINE_CKPT = PROJECT_ROOT / "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt"
LOSS1_CKPT = PROJECT_ROOT / "adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt"
LOSS2_CKPT = PROJECT_ROOT / "adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt"
LOSS3_CKPT = PROJECT_ROOT / "adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt"
GEN_ROOT = PROJECT_ROOT / "generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers"
AUDIT_CSV = PROJECT_ROOT / "forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_20260611/wide_parameter_dataset_audit.csv"
OUT_ROOT = PROJECT_ROOT / "forensics/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611"
REPORT_MD = PROJECT_ROOT / "docs/burgers_wideparam_loss3targeted_full1024_svd_attack3_20260611.md"

MODEL_SPECS = [
    {"key": "baseline", "label": "baseline", "epoch": 0, "checkpoint": BASELINE_CKPT},
    {"key": "loss1", "label": "loss1_epoch8000", "epoch": 8000, "checkpoint": LOSS1_CKPT},
    {"key": "loss2", "label": "loss2_epoch2000", "epoch": 2000, "checkpoint": LOSS2_CKPT},
    {"key": "loss3", "label": "loss3_epoch1500", "epoch": 1500, "checkpoint": LOSS3_CKPT},
]
EPS = 1e-12


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


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
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_audit() -> dict[str, dict[str, str]]:
    if not AUDIT_CSV.exists():
        return {}
    with AUDIT_CSV.open(newline="", encoding="utf-8") as f:
        return {row["dataset_id"]: row for row in csv.DictReader(f)}


def torch_load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def dataset_id_for_path(path: Path) -> str:
    try:
        data = torch_load(path)
        meta = data.get("metadata", {})
        if isinstance(meta, dict) and meta.get("dataset_id"):
            return str(meta["dataset_id"])
    except Exception:
        pass
    return path.stem


def dataset_size(path: Path) -> int:
    data = torch_load(path)
    return int(data["x"].shape[0])


def load_x(path: Path, index: int) -> np.ndarray:
    data = torch_load(path)
    x = data["x"][index].float().reshape(-1).numpy().astype(np.float32)
    if x.shape[0] != 1024:
        raise ValueError(f"expected 1024 values, got {x.shape} from {path}")
    return x


def select_random_generalization_samples(gen_root: Path, n_samples: int, seed: int) -> list[dict[str, Any]]:
    paths = sorted(gen_root.glob("burgers_widevis_l3target_d*.pt"))
    if len(paths) < 1:
        raise FileNotFoundError(f"no selected generalization datasets under {gen_root}")
    audit = read_audit()
    rng = np.random.default_rng(seed)
    seen: set[tuple[str, int]] = set()
    rows: list[dict[str, Any]] = []
    while len(rows) < n_samples:
        path = paths[int(rng.integers(0, len(paths)))]
        n = dataset_size(path)
        idx = int(rng.integers(0, n))
        key = (str(path), idx)
        if key in seen:
            continue
        seen.add(key)
        dataset_id = dataset_id_for_path(path)
        audit_row = audit.get(dataset_id, {})
        rows.append(
            {
                "sample_id": len(rows),
                "source_split": "generalization",
                "dataset_id": dataset_id,
                "dataset_path": str(path),
                "local_index": idx,
                "family": audit_row.get("family", ""),
                "display_label": audit_row.get("display_label", dataset_id),
                "target_min": audit_row.get("target_min", ""),
                "target_max": audit_row.get("target_max", ""),
                "spectral_centroid_mean": audit_row.get("spectral_centroid_mean", ""),
                "total_variation_mean": audit_row.get("total_variation_mean", ""),
            }
        )
    return rows


def load_models(device: torch.device) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    for spec in MODEL_SPECS:
        ckpt = Path(spec["checkpoint"])
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        print(f"[model] loading {spec['label']} from {ckpt}", flush=True)
        models[str(spec["key"])] = load_model(ckpt, device)
    return models


def run_attack_for_model(
    model_key: str,
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    *,
    attack_steps: int,
    epsilon_rms: float,
    alpha_rms: float,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    delta = torch.zeros_like(x_clean)
    step_records: list[dict[str, Any]] = []
    loss_history: list[np.ndarray] = []
    delta_history: list[np.ndarray] = []
    final_delta_np: np.ndarray | None = None

    def record(step: int, delta_now: torch.Tensor) -> None:
        nonlocal final_delta_np
        with torch.no_grad():
            x_adv = x_clean + delta_now
            solver = burgers_solver_target(x_adv)
            pred = model(x_adv)
            per_sample_loss = (pred - solver).pow(2).mean(dim=(1, 2))
            delta_rms = rms_l2_norm(delta_now)
        loss_np = per_sample_loss.detach().cpu().numpy().astype(np.float64)
        rms_np = delta_rms.detach().cpu().numpy().astype(np.float64)
        loss_history.append(loss_np)
        delta_history.append(rms_np)
        final_delta_np = delta_now.detach().cpu().numpy()[..., 0].astype(np.float32)
        for sample_i in range(x_clean.shape[0]):
            step_records.append(
                {
                    "model_key": model_key,
                    "sample_id": int(sample_i),
                    "attack_step": int(step),
                    "loss_mse": float(loss_np[sample_i]),
                    "delta_rms": float(rms_np[sample_i]),
                    "boundary_ratio": float(rms_np[sample_i] / max(epsilon_rms, EPS)),
                }
            )

    record(0, delta)
    started = time.perf_counter()
    print(f"[attack] {model_key}: {attack_steps} steps epsilon_rms={epsilon_rms} alpha_rms={alpha_rms}", flush=True)
    for step in range(1, attack_steps + 1):
        delta_var = delta.detach().clone().requires_grad_(True)
        x_adv = x_clean + delta_var
        solver = burgers_solver_target(x_adv)
        pred = model(x_adv)
        per_sample_loss = (pred - solver).pow(2).mean(dim=(1, 2))
        loss = per_sample_loss.mean()
        grad = torch.autograd.grad(loss, delta_var, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            delta = delta_var + alpha_rms * normalize_rms_l2(grad)
            delta = project_rms_l2(delta, epsilon_rms)
        record(step, delta)
        if step in {1, 5, 10, attack_steps}:
            print(
                f"[attack] {model_key} step {step:03d}/{attack_steps}: "
                f"mean_loss={float(loss_history[-1].mean()):.4e}, mean_delta_rms={float(delta_history[-1].mean()):.4f}",
                flush=True,
            )
    attack_seconds = time.perf_counter() - started
    np.savez_compressed(
        out_dir / f"{model_key}_attack_trace.npz",
        loss=np.stack(loss_history, axis=0),
        delta_rms=np.stack(delta_history, axis=0),
        final_delta=final_delta_np,
        steps=np.arange(attack_steps + 1, dtype=np.int32),
    )
    for row in step_records:
        row["attack_wall_sec_model_batch"] = attack_seconds
    return step_records, final_delta_np if final_delta_np is not None else np.zeros((x_clean.shape[0], 1024), dtype=np.float32)


def vector_abs_cos(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= EPS:
        return math.nan
    return float(abs(np.dot(aa, bb)) / denom)


def subspace_mean_cos(vh_a: np.ndarray, vh_b: np.ndarray, k: int) -> float:
    k = min(k, vh_a.shape[0], vh_b.shape[0])
    if k <= 0:
        return math.nan
    qa, _ = np.linalg.qr(vh_a[:k].T)
    qb, _ = np.linalg.qr(vh_b[:k].T)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0)))


def svd_summary(s: np.ndarray) -> dict[str, float]:
    energy = s * s
    p = energy / (float(energy.sum()) + EPS)
    return {
        "spectral_norm": float(s[0]),
        "fro_norm": float(np.linalg.norm(s)),
        "effective_rank": float(np.exp(-np.sum(p * np.log(p + EPS)))),
        "top8_energy": float(energy[:8].sum() / (float(energy.sum()) + EPS)),
        "top20_energy": float(energy[:20].sum() / (float(energy.sum()) + EPS)),
    }


def save_full_svd(name: str, J: np.ndarray, sample_dir: Path, sample_id: int, *, reuse_existing: bool) -> dict[str, Any]:
    model_dir = sample_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    npz_path = model_dir / f"{name}_index{sample_id}_jacobian_svd.npz"
    if reuse_existing and npz_path.exists():
        started = time.perf_counter()
        z = np.load(npz_path, allow_pickle=False)
        out = {
            "J": z["jacobian"].astype(np.float32),
            "s": z["singular_values"].astype(np.float64),
            "U": z["left_singular_vectors"].astype(np.float32),
            "Vh": z["right_singular_vectors"].astype(np.float32),
            "npz_path": npz_path,
            "svd_seconds": 0.0,
            "source": "reused",
            "load_seconds": time.perf_counter() - started,
        }
        return out
    A = np.asarray(J, dtype=np.float64)
    started = time.perf_counter()
    print(f"[full-svd] {name} sample={sample_id} matrix={A.shape}", flush=True)
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    svd_seconds = time.perf_counter() - started
    np.savez_compressed(
        npz_path,
        jacobian=A.astype(np.float32),
        singular_values=s.astype(np.float64),
        left_singular_vectors=U.astype(np.float32),
        right_singular_vectors=Vh.astype(np.float32),
        svd_method=np.array("full_np_linalg_svd"),
        full_matrix_shape=np.asarray(A.shape, dtype=np.int32),
    )
    save_json(
        model_dir / f"{name}_index{sample_id}_summary.json",
        {"name": name, "sample_id": sample_id, "npz_path": relpath(npz_path), "svd_seconds": svd_seconds, **svd_summary(s)},
    )
    return {"J": A.astype(np.float32), "s": s, "U": U.astype(np.float32), "Vh": Vh.astype(np.float32), "npz_path": npz_path, "svd_seconds": svd_seconds, "source": "computed"}


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if len(xx) < 2 or float(np.std(xx)) == 0.0 or float(np.std(yy)) == 0.0:
        return math.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def spearman(x: list[float], y: list[float]) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if len(xx) < 2:
        return math.nan
    return pearson(rank_values(xx).tolist(), rank_values(yy).tolist())


def correlation_rows(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = [
        ("error_spectral_norm", "attack_loss_growth_abs"),
        ("error_spectral_norm", "attack_loss_growth_ratio"),
        ("error_spectral_norm", "attack_final_mse"),
        ("error_spectral_norm", "attack_log_growth"),
        ("model_solver_top1_right_abs_cos", "attack_loss_growth_abs"),
        ("error_top_right_final_delta_abs_cos", "attack_loss_growth_abs"),
    ]
    groups: list[tuple[str, list[dict[str, Any]]]] = [("all_model_sample_pairs", joined)]
    for sample_id in sorted({int(r["sample_id"]) for r in joined}):
        groups.append((f"sample_{sample_id:03d}_across_models", [r for r in joined if int(r["sample_id"]) == sample_id]))
    for model_key in sorted({str(r["model_key"]) for r in joined}):
        groups.append((f"model_{model_key}_across_samples", [r for r in joined if str(r["model_key"]) == model_key]))
    out: list[dict[str, Any]] = []
    for group_name, rows in groups:
        for x_key, y_key in pairs:
            valid = [r for r in rows if math.isfinite(float(r.get(x_key, math.nan))) and math.isfinite(float(r.get(y_key, math.nan)))]
            out.append(
                {
                    "analysis_set": group_name,
                    "x": x_key,
                    "y": y_key,
                    "n": len(valid),
                    "pearson": pearson([float(r.get(x_key, math.nan)) for r in rows], [float(r.get(y_key, math.nan)) for r in rows]),
                    "spearman": spearman([float(r.get(x_key, math.nan)) for r in rows], [float(r.get(y_key, math.nan)) for r in rows]),
                }
            )
    return out


def write_report(path: Path, config: dict[str, Any], samples: list[dict[str, Any]], joined: list[dict[str, Any]], corr: list[dict[str, Any]], runtime: dict[str, Any]) -> None:
    all_corr = [r for r in corr if r["analysis_set"] == "all_model_sample_pairs"]
    lines = [
        "# Burgers Full-1024 SVD vs 15-Step Attack Probe - 2026-06-11",
        "",
        "This is the initial 3-generalization-sample probe. It uses full `1024 x 1024` dense Jacobians and full SVD, not block projection or any coarse proxy.",
        "",
        "## Output Files",
        "",
        f"- output root: `{relpath(Path(config['out_root']))}`",
        f"- sample manifest: `{relpath(Path(config['out_root']) / 'sample_manifest.csv')}`",
        f"- attack step table: `{relpath(Path(config['out_root']) / 'attack_step_metrics.csv')}`",
        f"- joined SVD/attack table: `{relpath(Path(config['out_root']) / 'svd_attack_joined_metrics.csv')}`",
        f"- correlation table: `{relpath(Path(config['out_root']) / 'svd_attack_correlations.csv')}`",
        f"- runtime JSON: `{relpath(Path(config['out_root']) / 'runtime_summary.json')}`",
        "- per-sample full SVD NPZ files: `sample_*/{solver,baseline_model,baseline_error,...}/*_jacobian_svd.npz`",
        "",
        "## Runtime",
        "",
        f"- total wall seconds: `{runtime.get('total_wall_sec', math.nan):.3f}`",
        f"- attack wall seconds total: `{runtime.get('attack_wall_sec_total', math.nan):.3f}`",
        f"- jacobian+SVD wall seconds total: `{runtime.get('jacobian_svd_wall_sec_total', math.nan):.3f}`",
        "",
        "## Samples",
        "",
        "| sample_id | dataset_id | index | family | display_label |",
        "|---:|---|---:|---|---|",
    ]
    for s in samples:
        lines.append(f"| {s['sample_id']} | `{s['dataset_id']}` | {s['local_index']} | `{s.get('family','')}` | {s.get('display_label','')} |")
    lines.extend([
        "",
        "## Joined Metrics",
        "",
        "| sample | model | err sigma1 | attack init MSE | attack final MSE | growth abs | growth ratio | delta RMS | err-v1 vs final delta cos |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in joined:
        lines.append(
            f"| {r['sample_id']} | {r['model_key']} | {r['error_spectral_norm']:.6g} | "
            f"{r['attack_initial_mse']:.6g} | {r['attack_final_mse']:.6g} | "
            f"{r['attack_loss_growth_abs']:.6g} | {r['attack_loss_growth_ratio']:.6g} | "
            f"{r['attack_final_delta_rms']:.6g} | {r['error_top_right_final_delta_abs_cos']:.6g} |"
        )
    lines.extend([
        "",
        "## Correlations Across 12 Model-Sample Pairs",
        "",
        "| x | y | n | Pearson | Spearman |",
        "|---|---|---:|---:|---:|",
    ])
    for r in all_corr:
        lines.append(f"| `{r['x']}` | `{r['y']}` | {r['n']} | {float(r['pearson']):.6g} | {float(r['spearman']):.6g} |")
    lines.extend([
        "",
        "## Config",
        "",
        "```json",
        json.dumps(jsonable(config), indent=2),
        "```",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--generalization-root", type=Path, default=GEN_ROOT)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--attack-steps", type=int, default=15)
    parser.add_argument("--epsilon-rms", type=float, default=0.12)
    parser.add_argument("--alpha-rms", type=float, default=0.012)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--burgers-nu", type=float, default=1e-3)
    parser.add_argument("--burgers-t-final", type=float, default=1.0)
    parser.add_argument("--burgers-dt", type=float, default=0.001)
    parser.add_argument("--burgers-domain", type=float, default=2.0)
    parser.add_argument("--burgers-jax-solver-dtype", default="float64")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overall_start = time.perf_counter()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA is not available; full 1024 Jacobian/SVD probe will be very slow", flush=True)

    samples = select_random_generalization_samples(args.generalization_root.resolve(), int(args.num_samples), int(args.seed))
    write_csv(out_root / "sample_manifest.csv", samples)
    save_json(out_root / "sample_manifest.json", samples)

    x_np = np.stack([load_x(Path(s["dataset_path"]), int(s["local_index"])) for s in samples], axis=0)
    x_clean = torch.as_tensor(x_np, dtype=torch.float32, device=device).unsqueeze(-1)
    models = load_models(device)

    attack_start = time.perf_counter()
    attack_rows: list[dict[str, Any]] = []
    final_deltas: dict[str, np.ndarray] = {}
    for spec in MODEL_SPECS:
        model_key = str(spec["key"])
        rows, delta_np = run_attack_for_model(
            model_key,
            models[model_key],
            x_clean,
            attack_steps=int(args.attack_steps),
            epsilon_rms=float(args.epsilon_rms),
            alpha_rms=float(args.alpha_rms),
            out_dir=out_root / "attack_traces",
        )
        attack_rows.extend(rows)
        final_deltas[model_key] = delta_np
    attack_wall = time.perf_counter() - attack_start
    write_csv(out_root / "attack_step_metrics.csv", attack_rows)

    attack_by_pair: dict[tuple[int, str], dict[str, float]] = {}
    for sample in samples:
        sid = int(sample["sample_id"])
        for spec in MODEL_SPECS:
            model_key = str(spec["key"])
            rows = [r for r in attack_rows if int(r["sample_id"]) == sid and str(r["model_key"]) == model_key]
            rows.sort(key=lambda r: int(r["attack_step"]))
            init = float(rows[0]["loss_mse"])
            final = float(rows[-1]["loss_mse"])
            attack_by_pair[(sid, model_key)] = {
                "attack_initial_mse": init,
                "attack_final_mse": final,
                "attack_loss_growth_abs": final - init,
                "attack_loss_growth_ratio": final / (init + EPS),
                "attack_log_growth": math.log((final + EPS) / (init + EPS)),
                "attack_final_delta_rms": float(rows[-1]["delta_rms"]),
                "attack_final_boundary_ratio": float(rows[-1]["boundary_ratio"]),
            }

    jac_start = time.perf_counter()
    solver_args = SimpleNamespace(
        burgers_nu=float(args.burgers_nu),
        burgers_t_final=float(args.burgers_t_final),
        burgers_dt=float(args.burgers_dt),
        burgers_domain=float(args.burgers_domain),
        burgers_jax_solver_dtype=str(args.burgers_jax_solver_dtype),
    )
    svd_rows: list[dict[str, Any]] = []
    joined_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for sample, x in zip(samples, x_np):
        sid = int(sample["sample_id"])
        sample_dir = out_root / f"sample_{sid:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_json(sample_dir / "sample.json", sample)
        print(f"[sample] sid={sid} dataset={sample['dataset_id']} index={sample['local_index']}", flush=True)

        solver_npz = sample_dir / "solver" / f"solver_index{sid}_jacobian_svd.npz"
        if args.reuse_existing and solver_npz.exists():
            solver_svd = save_full_svd("solver", np.zeros((1024, 1024), dtype=np.float32), sample_dir, sid, reuse_existing=True)
            solver_j = solver_svd["J"]
            solver_jac_sec = 0.0
            solver_jac_source = "reused"
        else:
            started = time.perf_counter()
            solver_j = compute_solver_jacobian(x, solver_args, device, progress_prefix=f"sample{sid}_solver")
            solver_jac_sec = time.perf_counter() - started
            solver_jac_source = "computed"
            solver_svd = save_full_svd("solver", solver_j, sample_dir, sid, reuse_existing=False)
        runtime_rows.append({"sample_id": sid, "model_key": "solver", "component": "solver_jacobian", "seconds": solver_jac_sec, "source": solver_jac_source})
        runtime_rows.append({"sample_id": sid, "model_key": "solver", "component": "solver_full_svd", "seconds": float(solver_svd["svd_seconds"]), "source": solver_svd["source"]})
        row = {**sample, "model_key": "solver", "checkpoint_label": "solver", "jacobian_kind": "solver", **svd_summary(solver_svd["s"]), "npz_path": relpath(Path(solver_svd["npz_path"]))}
        svd_rows.append(row)

        for spec in MODEL_SPECS:
            model_key = str(spec["key"])
            label = str(spec["label"])
            model_name = f"{model_key}_model"
            model_npz = sample_dir / model_name / f"{model_name}_index{sid}_jacobian_svd.npz"
            if args.reuse_existing and model_npz.exists():
                model_svd = save_full_svd(model_name, np.zeros((1024, 1024), dtype=np.float32), sample_dir, sid, reuse_existing=True)
                model_j = model_svd["J"]
                model_jac_sec = 0.0
                model_jac_source = "reused"
            else:
                started = time.perf_counter()
                model_j = compute_explicit_jacobian(models[model_key], x, device, progress_prefix=f"{model_key}_sample{sid}")
                model_jac_sec = time.perf_counter() - started
                model_jac_source = "computed"
                model_svd = save_full_svd(model_name, model_j, sample_dir, sid, reuse_existing=False)
            runtime_rows.append({"sample_id": sid, "model_key": model_key, "component": "model_jacobian", "seconds": model_jac_sec, "source": model_jac_source})
            runtime_rows.append({"sample_id": sid, "model_key": model_key, "component": "model_full_svd", "seconds": float(model_svd["svd_seconds"]), "source": model_svd["source"]})
            model_summary = {**sample, "model_key": model_key, "checkpoint_label": label, "jacobian_kind": "model", **svd_summary(model_svd["s"]), "npz_path": relpath(Path(model_svd["npz_path"]))}
            model_summary["model_solver_top1_right_abs_cos"] = vector_abs_cos(model_svd["Vh"][0], solver_svd["Vh"][0])
            model_summary["model_solver_top1_left_abs_cos"] = vector_abs_cos(model_svd["U"][:, 0], solver_svd["U"][:, 0])
            model_summary["model_solver_top10_right_subspace_mean_cos"] = subspace_mean_cos(model_svd["Vh"], solver_svd["Vh"], 10)
            svd_rows.append(model_summary)

            error_name = f"{model_key}_error"
            error_npz = sample_dir / error_name / f"{error_name}_index{sid}_jacobian_svd.npz"
            if args.reuse_existing and error_npz.exists():
                error_svd = save_full_svd(error_name, np.zeros((1024, 1024), dtype=np.float32), sample_dir, sid, reuse_existing=True)
                error_svd_sec = 0.0
                error_source = "reused"
            else:
                started = time.perf_counter()
                error_j = model_j.astype(np.float64) - solver_j.astype(np.float64)
                error_svd = save_full_svd(error_name, error_j, sample_dir, sid, reuse_existing=False)
                error_svd_sec = time.perf_counter() - started
                error_source = "computed"
                del error_j
            runtime_rows.append({"sample_id": sid, "model_key": model_key, "component": "error_full_svd", "seconds": error_svd_sec, "source": error_source})
            error_summary = {**sample, "model_key": model_key, "checkpoint_label": label, "jacobian_kind": "error", **svd_summary(error_svd["s"]), "npz_path": relpath(Path(error_svd["npz_path"]))}
            svd_rows.append(error_summary)

            final_delta = final_deltas[model_key][sid]
            attack_metrics = attack_by_pair[(sid, model_key)]
            joined = {
                **sample,
                "model_key": model_key,
                "checkpoint_label": label,
                "checkpoint_epoch": int(spec["epoch"]),
                "checkpoint_path": relpath(Path(spec["checkpoint"])),
                "error_spectral_norm": float(error_svd["s"][0]),
                "model_spectral_norm": float(model_svd["s"][0]),
                "solver_spectral_norm": float(solver_svd["s"][0]),
                "model_solver_top1_right_abs_cos": model_summary["model_solver_top1_right_abs_cos"],
                "model_solver_top1_left_abs_cos": model_summary["model_solver_top1_left_abs_cos"],
                "model_solver_top10_right_subspace_mean_cos": model_summary["model_solver_top10_right_subspace_mean_cos"],
                "error_top_right_final_delta_abs_cos": vector_abs_cos(error_svd["Vh"][0], final_delta),
                "error_top_left_model_solver_diff_abs_cos": math.nan,
                **attack_metrics,
                "solver_svd_npz": relpath(Path(solver_svd["npz_path"])),
                "model_svd_npz": relpath(Path(model_svd["npz_path"])),
                "error_svd_npz": relpath(Path(error_svd["npz_path"])),
            }
            joined_rows.append(joined)
            del model_j
            if device.type == "cuda":
                torch.cuda.empty_cache()
        write_csv(out_root / "svd_summary.partial.csv", svd_rows)
        write_csv(out_root / "svd_attack_joined_metrics.partial.csv", joined_rows)
        write_csv(out_root / "runtime_components.partial.csv", runtime_rows)

    jac_wall = time.perf_counter() - jac_start
    corr = correlation_rows(joined_rows)
    total_wall = time.perf_counter() - overall_start
    runtime_summary = {
        "total_wall_sec": total_wall,
        "attack_wall_sec_total": attack_wall,
        "jacobian_svd_wall_sec_total": jac_wall,
        "num_samples": len(samples),
        "attack_steps": int(args.attack_steps),
        "full_jacobian_shape": [1024, 1024],
        "uses_block_projection": False,
        "uses_topk_svd": False,
    }

    write_csv(out_root / "svd_summary.csv", svd_rows)
    write_csv(out_root / "svd_attack_joined_metrics.csv", joined_rows)
    write_csv(out_root / "svd_attack_correlations.csv", corr)
    write_csv(out_root / "runtime_components.csv", runtime_rows)
    save_json(out_root / "runtime_summary.json", runtime_summary)
    config = {
        "out_root": out_root,
        "report_md": args.report_md,
        "generalization_root": args.generalization_root,
        "num_samples": int(args.num_samples),
        "seed": int(args.seed),
        "attack_steps": int(args.attack_steps),
        "epsilon_rms": float(args.epsilon_rms),
        "alpha_rms": float(args.alpha_rms),
        "device": str(device),
        "reuse_existing": bool(args.reuse_existing),
        "full_jacobian_shape": [1024, 1024],
        "svd_method": "full_np_linalg_svd",
        "model_specs": MODEL_SPECS,
        "solver": {
            "burgers_nu": float(args.burgers_nu),
            "burgers_t_final": float(args.burgers_t_final),
            "burgers_dt": float(args.burgers_dt),
            "burgers_domain": float(args.burgers_domain),
            "burgers_jax_solver_dtype": str(args.burgers_jax_solver_dtype),
        },
    }
    save_json(out_root / "config.json", config)
    write_report(args.report_md.resolve(), config, samples, joined_rows, corr, runtime_summary)
    print(json.dumps(jsonable({"status": "ok", "out_root": out_root, "report_md": args.report_md, **runtime_summary}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
