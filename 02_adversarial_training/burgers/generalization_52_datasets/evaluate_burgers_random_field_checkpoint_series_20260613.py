#!/usr/bin/env python3
"""Evaluate a Burgers random-field checkpoint series.

The script accepts arbitrary ``label=checkpoint.pt`` model specs and runs the
same three post-training diagnostics used for the random-field 2000 epoch
models:

1. clean loss on train/test/50 generated datasets,
2. p=2/q=2 self-attack loss increase and adversarial delta,
3. local Jacobian/SVD diagnostics on a fixed sample manifest.

It then writes a postprocess table that keeps the algebraic quantities separate:
``J_error @ delta`` is the response to the adversarial delta, while
``J_error.T @ clean_error`` is the bias-gradient / A^T b quantity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.evaluate_burgers_random_field_final_models_20260613 import (  # noqa: E402
    DEFAULT_GEN_ROOT,
    DEFAULT_SVD_MANIFEST,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    append_jsonl,
    evaluate_attack,
    evaluate_clean,
    evaluate_svd,
    gpu_preflight,
    load_model,
    write_csv,
    write_json,
)
from tools.evaluate_generalization_models import tensor_xy, torch_load  # noqa: E402


DEFAULT_OUT_ROOT = REPO / "forensics/burgers_random_field_checkpoint_series_4000_6000_8000_20260613"
EPS = 1e-12


def parse_model_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model spec must be label=/path/to/checkpoint.pt")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("empty model label")
    bad = [ch for ch in label if not (ch.isalnum() or ch in {"_", "-", "."})]
    if bad:
        raise argparse.ArgumentTypeError(f"model label {label!r} contains unsupported characters")
    return label, Path(raw_path).expanduser()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def finite_stats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def safe_corr(x: list[float], y: list[float], method: str) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return math.nan
    a = a[mask]
    b = b[mask]
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return math.nan
    if method == "pearson":
        return float(pearsonr(a, b).statistic)
    if method == "spearman":
        return float(spearmanr(a, b).statistic)
    raise ValueError(method)


def safe_cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return math.nan
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= EPS:
        return math.nan
    return float(np.dot(av, bv) / denom)


def angle_deg_from_abs_cos(value: float) -> float:
    if not math.isfinite(value):
        return math.nan
    return float(math.degrees(math.acos(max(-1.0, min(1.0, abs(value))))))


def subspace_metrics(a_cols: np.ndarray, b_cols: np.ndarray, k: int) -> dict[str, float]:
    kk = min(int(k), int(a_cols.shape[1]), int(b_cols.shape[1]))
    if kk <= 0:
        return {
            "mean_principal_cos": math.nan,
            "min_principal_cos": math.nan,
            "rms_principal_cos": math.nan,
            "max_angle_deg": math.nan,
            "mean_angle_deg": math.nan,
        }
    qa, _ = np.linalg.qr(np.asarray(a_cols[:, :kk], dtype=np.float64))
    qb, _ = np.linalg.qr(np.asarray(b_cols[:, :kk], dtype=np.float64))
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    s = np.clip(np.abs(s), 0.0, 1.0)
    return {
        "mean_principal_cos": float(np.mean(s)),
        "min_principal_cos": float(np.min(s)),
        "rms_principal_cos": float(np.sqrt(np.mean(s * s))),
        "max_angle_deg": float(math.degrees(math.acos(float(np.min(s))))),
        "mean_angle_deg": float(np.mean(np.degrees(np.arccos(s)))),
    }


def svd_npz_path(svd_dir: Path, sample_id: int, name: str) -> Path:
    return svd_dir / f"sample_{sample_id:03d}" / name / f"{name}_index{sample_id}_jacobian_svd.npz"


def load_svd_npz(svd_dir: Path, sample_id: int, name: str) -> dict[str, np.ndarray]:
    path = svd_npz_path(svd_dir, sample_id, name)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def attack_index_map(attack_dir: Path) -> dict[tuple[str, int], int]:
    manifest_path = attack_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], int] = {}
    for i, rec in enumerate(manifest):
        out[(str(Path(rec["source_path"]).resolve()), int(rec["source_index"]))] = i
    return out


def load_attack_arrays(attack_dir: Path, model_order: list[str]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for model in model_order:
        model_dir = attack_dir / model
        losses_path = model_dir / "losses_and_delta_rms_by_sample.npz"
        delta_path = model_dir / "final_delta_by_sample.npz"
        if not losses_path.exists() or not delta_path.exists():
            continue
        with np.load(losses_path) as losses, np.load(delta_path) as delta:
            out[model] = {
                "initial_loss": np.asarray(losses["initial_loss"], dtype=np.float64),
                "final_loss": np.asarray(losses["final_loss"], dtype=np.float64),
                "final_delta_rms": np.asarray(losses["final_delta_rms"], dtype=np.float64),
                "final_delta": np.asarray(delta["final_delta"], dtype=np.float64),
            }
    return out


def load_xy_sample(path: Path, local_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch_load(path)
    x, y = tensor_xy(data, "burgers")
    return x[int(local_index)].float(), y[int(local_index)].float()


def model_error_vector(model, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        pred = model(x.to(device).reshape(1, -1, 1))
        err = pred.reshape(-1).detach().cpu() - y.reshape(-1).detach().cpu()
    return err.numpy().astype(np.float64)


def summarize_clean_loss(out_root: Path, model_order: list[str]) -> list[dict[str, Any]]:
    path = out_root / "clean_loss" / "per_dataset_clean_metrics.csv"
    if not path.exists():
        return []
    rows = read_csv(path)
    out: list[dict[str, Any]] = []
    for split in ("train", "test", "generalization"):
        subset = [row for row in rows if row.get("split") == split]
        if not subset:
            continue
        for model in model_order:
            item: dict[str, Any] = {"split": split, "model": model, "dataset_count": len(subset)}
            for metric in ("mse", "rmse", "relative_l2", "mae"):
                vals = [finite_float(row.get(f"{model}_{metric}_mean")) for row in subset]
                for stat, value in finite_stats(vals).items():
                    item[f"{metric}_{stat}"] = value
            out.append(item)
    return out


def postprocess(args: argparse.Namespace, model_paths: dict[str, Path], device: torch.device) -> None:
    out_root = args.out_root.resolve()
    post_dir = out_root / "postprocess"
    post_dir.mkdir(parents=True, exist_ok=True)
    model_order = list(model_paths)
    svd_dir = out_root / "jacobian_svd"
    attack_dir = out_root / "p2q2_attack"
    sample_rows = read_csv(svd_dir / "sample_manifest.csv")
    attack_lookup = attack_index_map(attack_dir)
    attack_arrays = load_attack_arrays(attack_dir, model_order)
    models = {name: load_model(path, device) for name, path in model_paths.items()}

    rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for raw_sample in sample_rows:
        sample_id = int(float(raw_sample["sample_id"]))
        dataset_path = Path(raw_sample["dataset_path"]).resolve()
        local_index = int(float(raw_sample["local_index"]))
        x, y = load_xy_sample(dataset_path, local_index)
        solver_svd = load_svd_npz(svd_dir, sample_id, "solver")
        solver_v_cols = np.asarray(solver_svd["right_singular_vectors"], dtype=np.float64).T
        solver_u_cols = np.asarray(solver_svd["left_singular_vectors"], dtype=np.float64)
        attack_i = attack_lookup.get((str(dataset_path), local_index))
        for model_name in model_order:
            t0 = time.time()
            model_svd = load_svd_npz(svd_dir, sample_id, model_name)
            error_svd = load_svd_npz(svd_dir, sample_id, f"{model_name}_error")
            j_error = np.asarray(error_svd["jacobian"], dtype=np.float64)
            s_error = np.asarray(error_svd["singular_values"], dtype=np.float64)
            model_v_cols = np.asarray(model_svd["right_singular_vectors"], dtype=np.float64).T
            model_u_cols = np.asarray(model_svd["left_singular_vectors"], dtype=np.float64)
            error_v_cols = np.asarray(error_svd["right_singular_vectors"], dtype=np.float64).T
            clean_error = model_error_vector(models[model_name], x, y, device)
            clean_error_norm = float(np.linalg.norm(clean_error))
            bias_gradient = j_error.T @ clean_error
            bias_gradient_norm = float(np.linalg.norm(bias_gradient))
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "source_split": raw_sample.get("source_split", ""),
                "dataset_id": raw_sample.get("dataset_id", ""),
                "dataset_path": str(dataset_path),
                "local_index": local_index,
                "model": model_name,
                "checkpoint_path": str(model_paths[model_name]),
                "clean_residual_l2": clean_error_norm,
                "clean_residual_rmse": float(np.sqrt(np.mean(clean_error * clean_error))),
                "clean_residual_mse": float(np.mean(clean_error * clean_error)),
                "error_spectral_norm": float(s_error[0]) if s_error.size else math.nan,
                "error_fro_norm_topk_svd_reported": float(np.linalg.norm(s_error)),
                "error_fro_norm_full_matrix": float(np.linalg.norm(j_error, ord="fro")),
                "svd_method": str(np.asarray(error_svd["svd_method"]).item()),
                "svd_top_k": int(np.asarray(error_svd["top_k"]).item()) if "top_k" in error_svd else int(s_error.size),
                "bias_gradient_norm": bias_gradient_norm,
                "bias_gradient_rms": float(np.sqrt(np.mean(bias_gradient * bias_gradient))),
            }
            energy = s_error * s_error
            if energy.size and float(energy.sum()) > 0:
                p = energy / float(energy.sum())
                row["error_effective_rank_topk_only"] = float(np.exp(-np.sum(p * np.log(p + EPS))))
                row["error_top8_energy_topk_basis"] = float(energy[:8].sum() / float(energy.sum()))
                row["error_top20_energy_topk_basis"] = float(energy[:20].sum() / float(energy.sum()))
            for rank in range(min(20, s_error.size)):
                row[f"error_sv_{rank + 1:02d}"] = float(s_error[rank])

            for k in (1, 5, 10, 20):
                right = subspace_metrics(model_v_cols, solver_v_cols, k)
                left = subspace_metrics(model_u_cols, solver_u_cols, k)
                prefix_r = f"model_solver_top{k}_right"
                prefix_l = f"model_solver_top{k}_left"
                for name, value in right.items():
                    row[f"{prefix_r}_{name}"] = value
                for name, value in left.items():
                    row[f"{prefix_l}_{name}"] = value

            top_error_right = error_v_cols[:, 0] if error_v_cols.size else None
            row["clean_error_cos_bias_gradient"] = safe_cosine(clean_error, bias_gradient)
            row["clean_error_abs_cos_bias_gradient"] = abs(row["clean_error_cos_bias_gradient"]) if math.isfinite(row["clean_error_cos_bias_gradient"]) else math.nan
            row["top_error_right_cos_bias_gradient"] = safe_cosine(top_error_right, bias_gradient)
            row["top_error_right_abs_cos_bias_gradient"] = abs(row["top_error_right_cos_bias_gradient"]) if math.isfinite(row["top_error_right_cos_bias_gradient"]) else math.nan
            row["top_error_right_bias_gradient_angle_deg"] = angle_deg_from_abs_cos(row["top_error_right_cos_bias_gradient"])

            arrays = attack_arrays.get(model_name)
            if arrays is not None and attack_i is not None and attack_i < int(arrays["initial_loss"].shape[0]):
                delta = np.asarray(arrays["final_delta"][attack_i], dtype=np.float64).reshape(-1)
                j_error_delta = j_error @ delta
                delta_norm = float(np.linalg.norm(delta))
                row.update(
                    {
                        "attack_manifest_index": int(attack_i),
                        "attack_initial_loss": float(arrays["initial_loss"][attack_i]),
                        "attack_final_loss": float(arrays["final_loss"][attack_i]),
                        "attack_loss_increase": float(arrays["final_loss"][attack_i] - arrays["initial_loss"][attack_i]),
                        "attack_delta_l2": delta_norm,
                        "attack_delta_rms": float(arrays["final_delta_rms"][attack_i]),
                        "j_error_delta_l2": float(np.linalg.norm(j_error_delta)),
                        "j_error_delta_rms": float(np.sqrt(np.mean(j_error_delta * j_error_delta))),
                        "attack_delta_cos_top_error_right": safe_cosine(delta, top_error_right),
                        "attack_delta_cos_bias_gradient": safe_cosine(delta, bias_gradient),
                    }
                )
                row["attack_delta_abs_cos_top_error_right"] = abs(row["attack_delta_cos_top_error_right"]) if math.isfinite(row["attack_delta_cos_top_error_right"]) else math.nan
                row["attack_delta_abs_cos_bias_gradient"] = abs(row["attack_delta_cos_bias_gradient"]) if math.isfinite(row["attack_delta_cos_bias_gradient"]) else math.nan
                row["attack_delta_top_error_angle_deg"] = angle_deg_from_abs_cos(row["attack_delta_cos_top_error_right"])
                row["attack_delta_bias_gradient_angle_deg"] = angle_deg_from_abs_cos(row["attack_delta_cos_bias_gradient"])
            rows.append(row)
            runtime_rows.append({"sample_id": sample_id, "model": model_name, "postprocess_seconds": time.time() - t0})
    write_csv(post_dir / "svd_attack_bias_gradient_by_sample.csv", rows)
    write_csv(post_dir / "postprocess_runtime.csv", runtime_rows)

    clean_summary = summarize_clean_loss(out_root, model_order)
    write_csv(post_dir / "clean_loss_summary_by_split_model.csv", clean_summary)

    model_summary: list[dict[str, Any]] = []
    for model_name in model_order:
        subset = [row for row in rows if row["model"] == model_name]
        item: dict[str, Any] = {"model": model_name, "checkpoint_path": str(model_paths[model_name]), "svd_sample_count": len(subset)}
        for metric in (
            "attack_loss_increase",
            "attack_final_loss",
            "attack_delta_l2",
            "attack_delta_rms",
            "error_spectral_norm",
            "error_fro_norm_full_matrix",
            "error_fro_norm_topk_svd_reported",
            "error_effective_rank_topk_only",
            "bias_gradient_norm",
            "j_error_delta_l2",
            "model_solver_top20_right_mean_principal_cos",
            "model_solver_top20_left_mean_principal_cos",
            "attack_delta_abs_cos_top_error_right",
            "attack_delta_abs_cos_bias_gradient",
            "top_error_right_abs_cos_bias_gradient",
        ):
            stats = finite_stats([finite_float(row.get(metric)) for row in subset])
            for stat, value in stats.items():
                item[f"{metric}_{stat}"] = value
        for clean_row in clean_summary:
            if clean_row.get("model") == model_name and clean_row.get("split") == "generalization":
                item["generalization_rmse_mean"] = clean_row.get("rmse_mean")
                item["generalization_relative_l2_mean"] = clean_row.get("relative_l2_mean")
                item["generalization_mse_mean"] = clean_row.get("mse_mean")
        model_summary.append(item)
    write_csv(post_dir / "model_level_summary.csv", model_summary)

    corr_metrics = [
        "error_spectral_norm",
        "error_fro_norm_full_matrix",
        "error_fro_norm_topk_svd_reported",
        "bias_gradient_norm",
        "j_error_delta_l2",
        "model_solver_top20_right_mean_principal_cos",
        "model_solver_top20_left_mean_principal_cos",
        "attack_delta_l2",
        "attack_delta_abs_cos_top_error_right",
        "attack_delta_abs_cos_bias_gradient",
        "top_error_right_abs_cos_bias_gradient",
        "clean_residual_mse",
    ]
    corr_rows: list[dict[str, Any]] = []
    yvals = [finite_float(row.get("attack_loss_increase")) for row in rows]
    for metric in corr_metrics:
        xvals = [finite_float(row.get(metric)) for row in rows]
        corr_rows.append(
            {
                "scope": "all_models_all_svd_samples",
                "metric": metric,
                "n": int(np.isfinite(np.asarray(xvals)) .sum()),
                "pearson_with_attack_loss_increase": safe_corr(xvals, yvals, "pearson"),
                "spearman_with_attack_loss_increase": safe_corr(xvals, yvals, "spearman"),
            }
        )
    write_csv(post_dir / "metric_correlations_with_attack_loss_increase.csv", corr_rows)

    rank_rows: list[dict[str, Any]] = []
    for sample_id in sorted({int(row["sample_id"]) for row in rows}):
        subset = [row for row in rows if int(row["sample_id"]) == sample_id]
        y = [finite_float(row.get("attack_loss_increase")) for row in subset]
        for metric in corr_metrics:
            x = [finite_float(row.get(metric)) for row in subset]
            rank_rows.append(
                {
                    "sample_id": sample_id,
                    "metric": metric,
                    "model_count": len(subset),
                    "spearman_across_models_with_attack_loss_increase": safe_corr(x, y, "spearman"),
                }
            )
    write_csv(post_dir / "per_sample_model_rank_similarity.csv", rank_rows)

    pair_rows: list[dict[str, Any]] = []
    for i, a in enumerate(corr_metrics):
        for b in corr_metrics[i + 1 :]:
            avals = [finite_float(row.get(a)) for row in rows]
            bvals = [finite_float(row.get(b)) for row in rows]
            pair_rows.append(
                {
                    "metric_a": a,
                    "metric_b": b,
                    "pearson": safe_corr(avals, bvals, "pearson"),
                    "spearman": safe_corr(avals, bvals, "spearman"),
                }
            )
    write_csv(post_dir / "metric_pairwise_correlations.csv", pair_rows)
    write_markdown_summary(post_dir, model_summary, corr_rows, rank_rows, clean_summary, model_order)
    del models
    torch.cuda.empty_cache()


def fmt(value: Any) -> str:
    value = finite_float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def write_markdown_summary(
    post_dir: Path,
    model_summary: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    clean_summary: list[dict[str, Any]],
    model_order: list[str],
) -> None:
    lines = [
        "# Burgers Random-Field Checkpoint Series Summary",
        "",
        "This is an automatically generated summary. It keeps `J_error @ delta` and `J_error^T clean_error` as separate quantities.",
        "",
        "## Clean Generalization",
        "",
        "| model | gen RMSE | gen relative L2 | gen MSE |",
        "|---|---:|---:|---:|",
    ]
    clean_by_model = {row["model"]: row for row in clean_summary if row.get("split") == "generalization"}
    for model in model_order:
        row = clean_by_model.get(model, {})
        lines.append(f"| {model} | {fmt(row.get('rmse_mean'))} | {fmt(row.get('relative_l2_mean'))} | {fmt(row.get('mse_mean'))} |")

    lines.extend(
        [
            "",
            "## Robustness And Jacobian Means",
            "",
            "| model | attack increase | error spectral | true error Fro | bias `||J_error^T e||` | `||J_error delta||` | top20 right solver cos | top20 left solver cos |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in model_summary:
        lines.append(
            "| {model} | {attack} | {spec} | {fro} | {bias} | {jdelta} | {right} | {left} |".format(
                model=row["model"],
                attack=fmt(row.get("attack_loss_increase_mean")),
                spec=fmt(row.get("error_spectral_norm_mean")),
                fro=fmt(row.get("error_fro_norm_full_matrix_mean")),
                bias=fmt(row.get("bias_gradient_norm_mean")),
                jdelta=fmt(row.get("j_error_delta_l2_mean")),
                right=fmt(row.get("model_solver_top20_right_mean_principal_cos_mean")),
                left=fmt(row.get("model_solver_top20_left_mean_principal_cos_mean")),
            )
        )

    rank_summary: list[dict[str, Any]] = []
    for metric in sorted({row["metric"] for row in rank_rows}):
        vals = [finite_float(row["spearman_across_models_with_attack_loss_increase"]) for row in rank_rows if row["metric"] == metric]
        stats = finite_stats(vals)
        rank_summary.append({"metric": metric, **stats})
    rank_summary.sort(key=lambda r: (-(r["mean"] if math.isfinite(r["mean"]) else -999), r["metric"]))

    lines.extend(
        [
            "",
            "## Correlation With Attack Loss Increase",
            "",
            "| metric | Pearson | Spearman |",
            "|---|---:|---:|",
        ]
    )
    for row in sorted(corr_rows, key=lambda r: (-(abs(finite_float(r.get("spearman_with_attack_loss_increase"))) if math.isfinite(finite_float(r.get("spearman_with_attack_loss_increase"))) else -1), r["metric"])):
        lines.append(f"| {row['metric']} | {fmt(row.get('pearson_with_attack_loss_increase'))} | {fmt(row.get('spearman_with_attack_loss_increase'))} |")

    lines.extend(
        [
            "",
            "## Per-Sample Across-Model Rank Similarity",
            "",
            "| metric | mean Spearman | median Spearman |",
            "|---|---:|---:|",
        ]
    )
    for row in rank_summary:
        lines.append(f"| {row['metric']} | {fmt(row.get('mean'))} | {fmt(row.get('median'))} |")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `svd_attack_bias_gradient_by_sample.csv`: per model/sample mechanism table.",
            "- `model_level_summary.csv`: model-level means and spreads.",
            "- `metric_correlations_with_attack_loss_increase.csv`: scalar correlation against self-attack increase.",
            "- `per_sample_model_rank_similarity.csv`: sample-wise model ranking agreement.",
            "- `metric_pairwise_correlations.csv`: pairwise similarity among mechanism metrics.",
        ]
    )
    (post_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--model-spec", action="append", type=parse_model_spec, required=True, help="label=/path/to/checkpoint.pt; repeat for every model")
    ap.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    ap.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    ap.add_argument("--gen-root", type=Path, default=DEFAULT_GEN_ROOT)
    ap.add_argument("--svd-manifest", type=Path, default=DEFAULT_SVD_MANIFEST)
    ap.add_argument("--stage", action="append", choices=["clean", "attack", "svd", "postprocess"], help="May be passed multiple times. Default: all stages.")
    ap.add_argument("--clean-batch-size", type=int, default=256)
    ap.add_argument("--clean-combined", action=argparse.BooleanOptionalAction, default=True, help="Evaluate all same-shaped clean datasets as one combined tensor, then split metrics back by dataset.")
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
    stages = args.stage or ["clean", "attack", "svd", "postprocess"]
    args.out_root.mkdir(parents=True, exist_ok=True)

    model_paths: dict[str, Path] = {}
    for label, path in args.model_spec:
        if label in model_paths:
            raise ValueError(f"duplicate model label: {label}")
        model_paths[label] = path.resolve()
    for label, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")
    for path in [args.train_path, args.test_path, args.gen_root, args.svd_manifest]:
        if not path.exists():
            raise FileNotFoundError(path)

    preflight = gpu_preflight()
    write_json(args.out_root / "gpu_preflight.json", preflight)
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
        elif stage == "postprocess":
            postprocess(args, model_paths, device)
        event = {"event": "stage_done", "stage": stage, "unix_time": time.time()}
        print(json.dumps(event), flush=True)
        append_jsonl(args.out_root / "progress.jsonl", event)

    write_json(args.out_root / "done.json", {"status": "complete", "finished_unix_time": time.time(), "stages": stages, "models": list(model_paths)})
    print(json.dumps({"event": "done", "out_root": str(args.out_root), "stages": stages}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
