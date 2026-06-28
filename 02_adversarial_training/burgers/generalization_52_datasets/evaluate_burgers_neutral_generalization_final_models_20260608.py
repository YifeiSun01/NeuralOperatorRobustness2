#!/usr/bin/env python3
"""Evaluate Burgers final self-training models on neutral semantic generalization data.

This script intentionally does not use the round03 loss3-selective adversarial
stress root. It evaluates baseline/loss1/loss2/loss3 on train/test plus the
neutral semantic Burgers root with kernel/transform/range dataset IDs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_generalization_models import (  # noqa: E402
    DatasetSpec,
    build_specs,
    checkpoint_state,
    evaluate_dataset,
    load_burgers_model,
    torch_load,
)

DEFAULT_OUT_ROOT = PROJECT_ROOT / "forensics/burgers_neutral_generalization_final_models_20260608"
DEFAULT_REPORT = PROJECT_ROOT / "docs/burgers_neutral_generalization_final_models_20260608.md"
NEUTRAL_ROOT = PROJECT_ROOT / "generalization_datasets_rmse_1p5_3x_all_ns50"

MODEL_SPECS = [
    {
        "model": "baseline",
        "epoch": 0,
        "checkpoint": PROJECT_ROOT / "1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt",
        "note": "retained trained baseline loaded by evaluate_generalization_models.py",
    },
    {
        "model": "loss1_epoch8000",
        "epoch": 8000,
        "checkpoint": PROJECT_ROOT / "adversarial_training_runs/burgers_loss3_selective_round03_loss1_continue5000to8000_20260607/burgers/checkpoints/burgers_epoch8000_step024000.pt",
        "note": "final loss1 continuation checkpoint requested for neutral evaluation",
    },
    {
        "model": "loss2_epoch2000",
        "epoch": 2000,
        "checkpoint": PROJECT_ROOT / "adversarial_training_runs/burgers_loss3_selective_round03_loss2_continue1000to2000_20260606/burgers/checkpoints/burgers_epoch2000_step006000.pt",
        "note": "final loss2 continuation checkpoint requested for neutral evaluation",
    },
    {
        "model": "loss3_epoch1500",
        "epoch": 1500,
        "checkpoint": PROJECT_ROOT / "adversarial_training_runs/burgers_loss3_selective_round03_loss3_continue1000to1500_20260606/burgers/checkpoints/burgers_epoch1500_step004500.pt",
        "note": "final loss3 continuation checkpoint requested for neutral evaluation",
    },
]

METRICS = ["rmse", "mae", "relative_l2", "accuracy_score"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def gpu_preflight() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this official evaluation; torch.cuda.is_available() is false")
    info.update(
        {
            "torch_device_name": torch.cuda.get_device_name(0),
            "torch_device_capability": torch.cuda.get_device_capability(0),
            "torch_arch_list": torch.cuda.get_arch_list(),
            "torch_cuda_sanity_sum": float(torch.ones(1024, device="cuda").sum().detach().cpu()),
        }
    )
    try:
        import jax  # type: ignore

        info.update(
            {
                "jax_version": jax.__version__,
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(d) for d in jax.devices()],
            }
        )
    except Exception as exc:  # noqa: BLE001
        info["jax_error"] = f"{type(exc).__name__}: {exc}"
    try:
        info["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception as exc:  # noqa: BLE001
        info["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return info


def load_model_for_spec(model_spec: dict[str, Any], device: torch.device):
    model = load_burgers_model(device)
    if model_spec["model"] != "baseline":
        state = checkpoint_state(Path(model_spec["checkpoint"]))
        model.load_state_dict(state, strict=True)
        model.eval()
    return model


def assert_neutral_burgers_specs(specs: list[DatasetSpec]) -> dict[str, Any]:
    gen = [s for s in specs if s.split == "generalization"]
    if len(gen) != 50:
        raise RuntimeError(f"expected 50 neutral Burgers generalization datasets, got {len(gen)}")
    bad_ids = [s.dataset_id for s in gen if "loss3_selective" in s.dataset_id or "pool" in s.dataset_id]
    if bad_ids:
        raise RuntimeError(f"neutral root contains attack-like dataset ids: {bad_ids[:5]}")
    provenance_rows = []
    attack_like_rows = []
    for spec in gen:
        data = torch_load(spec.path)
        meta = data.get("metadata", {}) if isinstance(data, dict) else {}
        params = meta.get("params", {}) if isinstance(meta, dict) else {}
        forbidden = [k for k in ("attack_objective", "attack_steps", "epsilon_fraction") if k in params]
        if forbidden:
            attack_like_rows.append({"dataset_id": spec.dataset_id, "forbidden_fields": ",".join(forbidden)})
        provenance_rows.append(
            {
                "dataset_id": spec.dataset_id,
                "split": spec.split,
                "manual_tier": spec.manual_tier,
                "source": spec.source,
                "path": rel(spec.path),
                "metadata_family": meta.get("family", ""),
                "metadata_similarity_tier": meta.get("similarity_tier", ""),
                "params": json.dumps(params, sort_keys=True),
                "num_samples": int(data.get("x").shape[0]) if isinstance(data, dict) and "x" in data else "",
            }
        )
    if attack_like_rows:
        raise RuntimeError(f"neutral root contains attack-like params: {attack_like_rows[:5]}")
    counts: dict[str, int] = defaultdict(int)
    for row in provenance_rows:
        counts[str(row["metadata_similarity_tier"] or row["manual_tier"])] += 1
    return {"provenance_rows": provenance_rows, "tier_counts": dict(counts)}


def split_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["split"]))].append(row)
    for (model, split), items in sorted(groups.items()):
        summary: dict[str, Any] = {
            "model": model,
            "split": split,
            "dataset_count": len(items),
            "total_samples_evaluated": sum(int(r["num_samples_evaluated"]) for r in items),
        }
        for metric in METRICS:
            vals = np.array([float(r[metric]) for r in items if math.isfinite(float(r[metric]))], dtype=float)
            summary[f"{metric}_dataset_mean"] = float(vals.mean()) if vals.size else float("nan")
            summary[f"{metric}_dataset_min"] = float(vals.min()) if vals.size else float("nan")
            summary[f"{metric}_dataset_max"] = float(vals.max()) if vals.size else float("nan")
        out.append(summary)
    return out


def relative_to_baseline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["dataset_id"], r["split"]): r for r in rows if r["model"] == "baseline"}
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["model"] == "baseline":
            continue
        base = by_key[(row["dataset_id"], row["split"])]
        out_row = {
            "model": row["model"],
            "split": row["split"],
            "dataset_id": row["dataset_id"],
            "manual_tier": row.get("manual_tier", ""),
            "path": row.get("path", ""),
        }
        for metric in METRICS:
            b = float(base[metric])
            v = float(row[metric])
            out_row[f"baseline_{metric}"] = b
            out_row[f"model_{metric}"] = v
            out_row[f"delta_{metric}"] = v - b
            out_row[f"pct_change_{metric}"] = 100.0 * (v / b - 1.0) if b != 0 and math.isfinite(b) else float("nan")
        out.append(out_row)
    return out


def relative_split_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_by_split = {r["split"]: r for r in summary_rows if r["model"] == "baseline"}
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        if row["model"] == "baseline":
            continue
        base = base_by_split[row["split"]]
        out_row = {
            "model": row["model"],
            "split": row["split"],
            "dataset_count": row["dataset_count"],
            "total_samples_evaluated": row["total_samples_evaluated"],
        }
        for metric in METRICS:
            key = f"{metric}_dataset_mean"
            b = float(base[key])
            v = float(row[key])
            out_row[f"baseline_{key}"] = b
            out_row[f"model_{key}"] = v
            out_row[f"delta_{key}"] = v - b
            out_row[f"pct_change_{key}"] = 100.0 * (v / b - 1.0) if b != 0 and math.isfinite(b) else float("nan")
        out.append(out_row)
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[str], *, max_rows: int | None = None) -> list[str]:
    show = rows if max_rows is None else rows[:max_rows]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in show:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_report(report: Path, out_root: Path, provenance: dict[str, Any], summary_rows: list[dict[str, Any]], rel_summary: list[dict[str, Any]], elapsed_sec: float) -> None:
    lines: list[str] = []
    lines.extend(
        [
            "# Burgers neutral semantic generalization final-model evaluation - 2026-06-08",
            "",
            "Status: completed GPU inference evaluation on train/test plus neutral semantic Burgers generalization datasets.",
            "",
            "Observed source data:",
            f"- Neutral generalization root: `{rel(NEUTRAL_ROOT / 'burgers')}`.",
            "- This root contains exactly 50 Burgers `.pt` files with semantic IDs such as `burgers_far_sawtooth_add_scale0p3_shift0` and `burgers_mid_matern_corr0p75_nu2p5`.",
            "- Provenance check found no `attack_objective`, `attack_steps`, or `epsilon_fraction` fields in the neutral Burgers dataset metadata.",
            f"- Tier counts from metadata: `{json.dumps(provenance['tier_counts'], sort_keys=True)}`.",
            "- Caveat: this root is non-attack semantic generalization data. Some `target_*` datasets are target-band semantic candidates, so the root is neutral/non-attack but not claimed to be a random sample from all possible distributions.",
            "",
            "Models evaluated:",
        ]
    )
    for spec in MODEL_SPECS:
        lines.append(f"- `{spec['model']}`: epoch `{spec['epoch']}`, checkpoint `{rel(Path(spec['checkpoint']))}`")
    lines.extend(
        [
            "",
            "Output files:",
            f"- Per-dataset metrics: `{rel(out_root / 'per_dataset_metrics.csv')}`",
            f"- Split summary: `{rel(out_root / 'split_summary.csv')}`",
            f"- Relative-to-baseline by dataset: `{rel(out_root / 'relative_to_baseline_by_dataset.csv')}`",
            f"- Relative-to-baseline split summary: `{rel(out_root / 'relative_to_baseline_split_summary.csv')}`",
            f"- Dataset provenance audit: `{rel(out_root / 'neutral_dataset_provenance.csv')}`",
            f"- GPU preflight: `{rel(out_root / 'gpu_preflight.json')}`",
            f"- Manifest: `{rel(out_root / 'manifest.json')}`",
            "",
            f"Evaluation wall time: `{elapsed_sec:.2f}` seconds.",
            "",
            "Split Mean Raw Metrics:",
        ]
    )
    preferred = ["model", "split", "dataset_count", "rmse_dataset_mean", "relative_l2_dataset_mean", "mae_dataset_mean"]
    lines.extend(markdown_table(summary_rows, preferred))
    lines.extend(["", "Percent Change vs Baseline by Split (negative means lower loss/error):"])
    pct_cols = [
        "model",
        "split",
        "pct_change_rmse_dataset_mean",
        "pct_change_relative_l2_dataset_mean",
        "pct_change_mae_dataset_mean",
    ]
    lines.extend(markdown_table(rel_summary, pct_cols))
    lines.extend(
        [
            "",
            "Inference:",
            "- This evaluation is the appropriate neutral semantic check for train/test/generalization loss changes; it does not use the round03 attack-generated stress root.",
            "- Use the relative-to-baseline CSVs for per-dataset increase/decrease percentages and the split table above for aggregate train/test/generalization changes.",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-max-samples", type=int, default=0, help="0 means all samples")
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    report = args.report.resolve()

    preflight = gpu_preflight()
    write_json(out_root / "gpu_preflight.json", preflight)
    device = torch.device("cuda")

    for spec in MODEL_SPECS:
        if not Path(spec["checkpoint"]).exists():
            raise FileNotFoundError(spec["checkpoint"])

    specs = [s for s in build_specs(NEUTRAL_ROOT) if s.task == "burgers"]
    provenance = assert_neutral_burgers_specs(specs)
    write_csv(out_root / "neutral_dataset_provenance.csv", provenance["provenance_rows"])

    max_samples = None if int(args.eval_max_samples) <= 0 else int(args.eval_max_samples)
    all_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for model_spec in MODEL_SPECS:
        model = load_model_for_spec(model_spec, device)
        for spec in specs:
            result = evaluate_dataset(model, spec, device, int(args.eval_batch_size), max_samples)
            row = {
                "model": model_spec["model"],
                "model_epoch": model_spec["epoch"],
                "checkpoint_path": rel(Path(model_spec["checkpoint"])),
                **result,
            }
            all_rows.append(row)
        del model
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - t0

    summary_rows = split_summaries(all_rows)
    rel_rows = relative_to_baseline(all_rows)
    rel_summary = relative_split_summary(summary_rows)

    write_csv(out_root / "per_dataset_metrics.csv", all_rows)
    write_csv(out_root / "split_summary.csv", summary_rows)
    write_csv(out_root / "relative_to_baseline_by_dataset.csv", rel_rows)
    write_csv(out_root / "relative_to_baseline_split_summary.csv", rel_summary)

    manifest = {
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "neutral_root": NEUTRAL_ROOT / "burgers",
        "dataset_count": len([s for s in specs if s.split == "generalization"]),
        "total_eval_specs": len(specs),
        "eval_batch_size": int(args.eval_batch_size),
        "eval_max_samples": max_samples,
        "models": MODEL_SPECS,
        "outputs": {
            "per_dataset_metrics": out_root / "per_dataset_metrics.csv",
            "split_summary": out_root / "split_summary.csv",
            "relative_to_baseline_by_dataset": out_root / "relative_to_baseline_by_dataset.csv",
            "relative_to_baseline_split_summary": out_root / "relative_to_baseline_split_summary.csv",
            "neutral_dataset_provenance": out_root / "neutral_dataset_provenance.csv",
            "gpu_preflight": out_root / "gpu_preflight.json",
            "report": report,
        },
        "elapsed_sec": elapsed,
        "provenance_tier_counts": provenance["tier_counts"],
    }
    write_json(out_root / "manifest.json", manifest)
    write_report(report, out_root, provenance, summary_rows, rel_summary, elapsed)
    print(json.dumps(json_ready(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
