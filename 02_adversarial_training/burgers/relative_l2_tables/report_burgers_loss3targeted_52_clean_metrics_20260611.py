#!/usr/bin/env python3
"""Create a detailed 52-dataset clean-metric report for Burgers loss3-targeted data."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analyze_burgers_wideparam_visible_clean_loss_final_models_20260611 import (  # noqa: E402
    MODEL_ORDER,
    SHORT,
    eval_dataset,
    mean_std,
    paired,
    write_csv,
    write_json,
)
from tools.evaluate_burgers_neutral_generalization_final_models_20260608 import (  # noqa: E402
    MODEL_SPECS,
    load_model_for_spec,
)
from tools.generate_burgers_semantic_loss3_favored_generalization_20260611 import params_key  # noqa: E402

SPLIT_DIR = (
    PROJECT_ROOT
    / "1D_Burgers/datasets/1D/Burgers/batched_exponax_splits"
    / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45"
)
TRAIN_PATH = SPLIT_DIR / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_train.pt"
TEST_PATH = SPLIT_DIR / "dim1d_nx1024_N1500_solver=exponax_batched_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.001_t1.0_seed45_test.pt"
GEN_METRICS = PROJECT_ROOT / "forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611/per_dataset_clean_metrics.csv"
GEN_SAMPLE_METRICS = PROJECT_ROOT / "forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_clean_loss_final_models_20260611/per_sample_clean_metrics.csv"
GEN_AUDIT = PROJECT_ROOT / "forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_20260611/wide_parameter_dataset_audit.csv"
GEN_MANIFEST = PROJECT_ROOT / "generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/manifest.json"
OUT_ROOT = PROJECT_ROOT / "forensics/burgers_semantic_wideparam_visible_loss3targeted_round00_52dataset_clean_metrics_20260611"
DOC_PATH = PROJECT_ROOT / "docs/burgers_loss3targeted_52dataset_clean_metrics_20260611.md"


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def spectrum_stats(x: torch.Tensor) -> dict[str, float]:
    flat = x.float().reshape(x.shape[0], -1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    fft = torch.fft.rfft(centered, dim=1).abs().pow(2)
    total = fft.sum(dim=1).clamp_min(1e-20)
    freqs = torch.arange(fft.shape[1], device=fft.device).float()
    centroid = (fft * freqs[None, :]).sum(dim=1) / total
    low = fft[:, 1:8].sum(dim=1) / total
    mid = fft[:, 8:64].sum(dim=1) / total
    high = fft[:, 64:].sum(dim=1) / total
    diff = flat[:, 1:] - flat[:, :-1]
    return {
        "x_min": float(flat.min()),
        "x_max": float(flat.max()),
        "x_mean": float(flat.mean()),
        "x_std": float(flat.std(unbiased=False)),
        "spectral_centroid_mean": float(centroid.mean()),
        "spectral_low_frac_mean": float(low.mean()),
        "spectral_mid_frac_mean": float(mid.mean()),
        "spectral_high_frac_mean": float(high.mean()),
        "total_variation_mean": float(diff.abs().mean()),
    }


def flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "base_family",
        "kernel",
        "correlation_length",
        "matern_nu",
        "nu",
        "spectral_alpha",
        "k0",
        "frequencies",
        "decay",
        "frequency",
        "duty",
        "target_min",
        "target_max",
        "transform",
        "transform_variant",
        "base_config",
    ]
    out = {}
    for key in keys:
        value = params.get(key, "")
        if isinstance(value, (list, tuple, dict)):
            value = json.dumps(value, sort_keys=True)
        out[f"param_{key}"] = value
    out["params_json"] = json.dumps(params, sort_keys=True)
    out["params_key"] = params_key(params) if params else ""
    return out


def enrich_generalization_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_items = load_json(GEN_MANIFEST)
    manifest = {item["dataset_id"]: item for item in manifest_items}
    audit = {row["dataset_id"]: row for row in read_rows(GEN_AUDIT)}
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        dataset_id = row["dataset_id"]
        m = manifest.get(dataset_id, {})
        params = dict(m.get("params", {}))
        a = audit.get(dataset_id, {})
        enriched = {
            "dataset_order": idx + 2,
            "dataset_group": "generalization",
            "dataset_id": dataset_id,
            "display_label": row.get("display_label", params.get("display_label", dataset_id)),
            "description": m.get("description", ""),
            "path": row.get("path", m.get("path", "")),
            "n_samples": row.get("n_samples", m.get("n", "")),
            "family": params.get("base_family", row.get("family", "")),
            "source_type": m.get("source_type", "loss3targeted_generalization"),
            "solver": "exponax Burgers",
            "nx": 1024,
            "nu": 0.001,
            "t_final": 1.0,
            "split_seed": "",
            "split_indices_count": "",
        }
        for key in [
            "x_min",
            "x_max",
            "x_mean",
            "x_std",
            "spectral_centroid_mean",
            "spectral_low_frac_mean",
            "spectral_mid_frac_mean",
            "spectral_high_frac_mean",
            "total_variation_mean",
        ]:
            enriched[key] = a.get(key, "")
        enriched.update(flatten_params(params))
        for key, value in row.items():
            if key not in {"dataset_id", "display_label", "path", "n_samples", "family"}:
                enriched[key] = value
        out.append(enriched)
    return out


def evaluate_train_test(batch_size: int = 128) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    specs = [s for s in MODEL_SPECS if s["model"] in MODEL_ORDER]
    models = {s["model"]: load_model_for_spec(s, device) for s in specs}
    for model in models.values():
        model.eval()

    dataset_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    split_paths = [("train", TRAIN_PATH, 1), ("test", TEST_PATH, 2)]
    for split_name, path, order in split_paths:
        metrics, _ = eval_dataset(path, models, batch_size, device)
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        meta = dict(data.get("metadata", {}))
        stats = spectrum_stats(data["x"].float())
        params = {
            "base_family": "train_test_gaussian",
            "kernel": meta.get("kernel", "gaussian"),
            "correlation_length": meta.get("correlation_length", 0.03),
            "bc": meta.get("bc", "periodic"),
            "seed": meta.get("seed", 45),
            "zero_mean": meta.get("zero_mean", False),
            "target_min": float(data["x"].min()),
            "target_max": float(data["x"].max()),
            "solver": meta.get("solver", "exponax_batched"),
            "nu": meta.get("nu", 0.001),
            "t_final": meta.get("t_final", 1.0),
            "split": split_name,
        }
        dataset_id = f"burgers_original_{split_name}_gaussian_corr0p03_seed45"
        display = f"Original Burgers {split_name}; Gaussian GRF corr=0.03; range observed [{stats['x_min']:.4g},{stats['x_max']:.4g}]"
        drow: dict[str, Any] = {
            "dataset_order": order,
            "dataset_group": split_name,
            "dataset_id": dataset_id,
            "display_label": display,
            "description": "Original train/test split generated with Gaussian GRF kernel, correlation_length=0.03, periodic BC, nu=0.001, t_final=1.0.",
            "path": str(path),
            "n_samples": int(data["x"].shape[0]),
            "family": "train_test_gaussian",
            "source_type": "original_train_test_split",
            "solver": meta.get("solver", "exponax_batched"),
            "nx": meta.get("nx", 1024),
            "nu": meta.get("nu", 0.001),
            "t_final": meta.get("t_final", 1.0),
            "split_seed": meta.get("split_seed", ""),
            "split_indices_count": len(meta.get("split_indices", [])),
            **stats,
            **flatten_params(params),
        }
        for model_name in MODEL_ORDER:
            short = SHORT[model_name]
            for metric in ["rmse", "mse", "relative_l2"]:
                stats_dict = mean_std(metrics[model_name][metric])
                for k, v in stats_dict.items():
                    drow[f"{short}_{metric}_{k}"] = v
        for model_name in ["loss1_epoch8000", "loss2_epoch2000", "loss3_epoch1500"]:
            short = SHORT[model_name]
            for metric in ["rmse", "mse", "relative_l2"]:
                ps = paired(metrics[model_name][metric], metrics["baseline"][metric])
                for k, v in ps.items():
                    drow[f"{metric}_{short}_vs_baseline_{k}"] = v
        drow["rmse_loss3_best_among_adv"] = bool(
            drow["loss3_rmse_mean"] < drow["loss1_rmse_mean"] and drow["loss3_rmse_mean"] < drow["loss2_rmse_mean"]
        )
        drow["rmse_loss3_best_all4"] = bool(
            drow["loss3_rmse_mean"] < drow["baseline_rmse_mean"]
            and drow["loss3_rmse_mean"] < drow["loss1_rmse_mean"]
            and drow["loss3_rmse_mean"] < drow["loss2_rmse_mean"]
        )
        dataset_rows.append(drow)
        n = len(metrics["baseline"]["rmse"])
        for si in range(n):
            srow: dict[str, Any] = {
                "dataset_group": split_name,
                "dataset_id": dataset_id,
                "family": "train_test_gaussian",
                "path": str(path),
                "sample_index": si,
            }
            for model_name in MODEL_ORDER:
                short = SHORT[model_name]
                for metric in ["rmse", "mse", "relative_l2"]:
                    srow[f"{short}_{metric}"] = float(metrics[model_name][metric][si])
            sample_rows.append(srow)
    return dataset_rows, sample_rows


def combine_sample_rows(gen_rows: list[dict[str, Any]], split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = {item["dataset_id"]: item for item in load_json(GEN_MANIFEST)}
    out = list(split_rows)
    for row in gen_rows:
        item = manifest.get(row["dataset_id"], {})
        params = item.get("params", {})
        out.append({
            "dataset_group": "generalization",
            "dataset_id": row["dataset_id"],
            "family": params.get("base_family", row.get("family", "")),
            "path": item.get("path", ""),
            **row,
        })
    return out


def markdown_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        "dataset_group",
        "dataset_id",
        "family",
        "n_samples",
        "baseline_rmse_mean",
        "loss1_rmse_mean",
        "loss2_rmse_mean",
        "loss3_rmse_mean",
        "rmse_loss1_vs_baseline_reduction_pct",
        "rmse_loss2_vs_baseline_reduction_pct",
        "rmse_loss3_vs_baseline_reduction_pct",
        "baseline_relative_l2_mean",
        "loss1_relative_l2_mean",
        "loss2_relative_l2_mean",
        "loss3_relative_l2_mean",
        "relative_l2_loss1_vs_baseline_reduction_pct",
        "relative_l2_loss2_vs_baseline_reduction_pct",
        "relative_l2_loss3_vs_baseline_reduction_pct",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            else:
                try:
                    fv = float(value)
                    if col not in {"n_samples"}:
                        value = f"{fv:.6g}"
                except (TypeError, ValueError):
                    pass
            vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(rows: list[dict[str, Any]], out_root: Path) -> None:
    train = next(r for r in rows if r["dataset_group"] == "train")
    test = next(r for r in rows if r["dataset_group"] == "test")
    gen = [r for r in rows if r["dataset_group"] == "generalization"]
    loss3_gen_wins = sum(1 for r in gen if str(r.get("rmse_loss3_best_all4")).lower() == "true" or r.get("rmse_loss3_best_all4") is True)
    report = [
        "# Burgers Loss3-Targeted 52-Dataset Clean Metrics - 2026-06-11",
        "",
        "This report records clean inference metrics for 52 datasets: the original train split, the original test split, and 50 loss3-targeted generalization datasets.",
        "",
        "No adversarial attack is included in these numbers.",
        "",
        "## Output Files",
        "",
        f"- 52-row dataset table: `{out_root / 'per_dataset_52_clean_metrics.csv'}`",
        f"- sample-level clean metrics: `{out_root / 'per_sample_52_clean_metrics.csv'}`",
        f"- compact summary JSON: `{out_root / 'summary_52_clean_metrics.json'}`",
        "",
        "## Dataset Sources",
        "",
        f"- Train split: `{TRAIN_PATH}`",
        f"- Test split: `{TEST_PATH}`",
        "- Generalization root: `generalization_datasets_burgers_semantic_wideparam_visible_loss3targeted_20260611/round_00/burgers`",
        "- Generalization policy: 50 unique non-attack semantic datasets, selected from wide visible parameters while enforcing loss3 clean-inference advantage.",
        "",
        "## Train/Test Metrics",
        "",
        markdown_table([train, test]),
        "",
        "## Generalization Metrics",
        "",
        f"Loss3 is clean-RMSE best on {loss3_gen_wins}/50 generalization datasets.",
        "",
        markdown_table(gen),
        "",
        "## Parameter And Characteristic Columns",
        "",
        "The CSV includes the dataset path, display label, description, family, solver, `nx`, `nu`, `t_final`, observed `x` statistics, spectral centroid/low/mid/high fractions, total variation, flattened parameter columns, and full JSON parameters.",
        "",
        "Key flattened parameter columns include `param_base_family`, `param_kernel`, `param_correlation_length`, `param_matern_nu`, `param_spectral_alpha`, `param_k0`, `param_frequencies`, `param_decay`, `param_frequency`, `param_duty`, `param_target_min`, and `param_target_max`.",
    ]
    DOC_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    train_test_dataset_rows, train_test_sample_rows = evaluate_train_test(batch_size=128)
    gen_dataset_rows = enrich_generalization_rows(read_rows(GEN_METRICS))
    all_dataset_rows = train_test_dataset_rows + gen_dataset_rows
    all_dataset_rows.sort(key=lambda r: int(r["dataset_order"]))
    gen_sample_rows = read_rows(GEN_SAMPLE_METRICS)
    all_sample_rows = combine_sample_rows(gen_sample_rows, train_test_sample_rows)
    write_csv(OUT_ROOT / "per_dataset_52_clean_metrics.csv", all_dataset_rows)
    write_csv(OUT_ROOT / "per_sample_52_clean_metrics.csv", all_sample_rows)
    summary = {
        "n_datasets": len(all_dataset_rows),
        "n_sample_rows": len(all_sample_rows),
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH),
        "generalization_metrics_source": str(GEN_METRICS),
        "dataset_table": str(OUT_ROOT / "per_dataset_52_clean_metrics.csv"),
        "sample_table": str(OUT_ROOT / "per_sample_52_clean_metrics.csv"),
        "report": str(DOC_PATH),
        "loss3_best_all4_count": int(sum(bool(r.get("rmse_loss3_best_all4")) for r in all_dataset_rows)),
        "loss3_best_all4_generalization_count": int(sum(bool(r.get("rmse_loss3_best_all4")) for r in gen_dataset_rows)),
        "family_counts": {},
    }
    for row in all_dataset_rows:
        fam = str(row.get("family", ""))
        summary["family_counts"][fam] = summary["family_counts"].get(fam, 0) + 1
    write_json(OUT_ROOT / "summary_52_clean_metrics.json", summary)
    write_report(all_dataset_rows, OUT_ROOT)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
