#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "adversarial_training_runs" / "darcy_cflow_wallclock_raw52x5_20260608"
DOC = ROOT / "docs" / "darcy_cflow_wallclock_raw52x5_20260608.md"
LEDGER = ROOT / "EXPERIMENT_LEDGER.md"

RUNS = {
    "loss1": ROOT / "adversarial_training_runs" / "darcy_lossdrop50_loss1_single_gpu_time_matched_loss3wall_20260608" / "darcy",
    "loss2": ROOT / "adversarial_training_runs" / "darcy_lossdrop50_loss2_single_gpu_time_matched_loss3wall_20260608" / "darcy",
    "loss3": ROOT / "adversarial_training_runs" / "darcy_lossdrop50_loss3_500ep_fromscreen_20260607" / "darcy",
    "physics": ROOT / "adversarial_training_runs" / "darcy_lossdrop50_physics_time_matched_loss3wall_20260608" / "darcy",
}
MODEL_ORDER = ["baseline", "loss1", "loss2", "loss3", "physics"]
SPLIT_ORDER = {"train": 0, "test": 1, "generalization": 2}
METRICS = ["rmse", "mae", "relative_l2", "accuracy_score", "invalid_value_fraction"]


def read_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def read_eval_rows(run_dir: Path) -> List[dict]:
    with (run_dir / "eval_metrics.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def dataset_sort_key(row: dict) -> Tuple[int, float, str]:
    try:
        manual_rank = float(row.get("manual_rank") or 0.0)
    except ValueError:
        manual_rank = 0.0
    return (SPLIT_ORDER.get(row["split"], 99), manual_rank, row["dataset_id"])


def row_key(row: dict) -> Tuple[str, str]:
    return (row["split"], row["dataset_id"])


def select_epoch_rows(rows: Iterable[dict], epoch: int) -> List[dict]:
    return [r for r in rows if int(r["epoch"]) == epoch]


def numeric(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def build_long_row(model: str, row: dict, dataset_index: int, run_name: str, run_dir: str, objective: str, epoch: int, elapsed_seconds: float, elapsed_minutes: float, final_checkpoint: str, wallclock_role: str) -> dict:
    return {
        "dataset_index": dataset_index,
        "model": model,
        "objective": objective,
        "wallclock_role": wallclock_role,
        "run_name": run_name,
        "run_dir": run_dir,
        "model_epoch": epoch,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_minutes,
        "final_checkpoint": final_checkpoint,
        "split": row["split"],
        "dataset_id": row["dataset_id"],
        "source": row["source"],
        "manual_tier": row["manual_tier"],
        "manual_rank": row["manual_rank"],
        "path": row["path"],
        "num_samples_evaluated": row["num_samples_evaluated"],
        "rmse": row["rmse"],
        "mae": row["mae"],
        "relative_l2": row["relative_l2"],
        "accuracy_score": row["accuracy_score"],
        "finite_value_count": row["finite_value_count"],
        "invalid_value_count": row["invalid_value_count"],
        "invalid_value_fraction": row["invalid_value_fraction"],
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise SystemExit(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals)


def build_wide_rows(long_rows: List[dict]) -> List[dict]:
    by_dataset: Dict[Tuple[int, str, str], dict] = {}
    for r in long_rows:
        key = (int(r["dataset_index"]), r["split"], r["dataset_id"])
        out = by_dataset.setdefault(key, {
            "dataset_index": r["dataset_index"],
            "split": r["split"],
            "dataset_id": r["dataset_id"],
            "source": r["source"],
            "manual_tier": r["manual_tier"],
            "manual_rank": r["manual_rank"],
            "path": r["path"],
            "num_samples_evaluated": r["num_samples_evaluated"],
        })
        model = r["model"]
        out[f"{model}_epoch"] = r["model_epoch"]
        out[f"{model}_elapsed_minutes"] = r["elapsed_minutes"]
        for metric in METRICS:
            out[f"{model}_{metric}"] = r[metric]
    rows = [by_dataset[k] for k in sorted(by_dataset.keys())]
    for out in rows:
        baseline_rel = numeric(out["baseline_relative_l2"])
        baseline_rmse = numeric(out["baseline_rmse"])
        for model in ["loss1", "loss2", "loss3", "physics"]:
            out[f"{model}_relative_l2_drop_pct_vs_baseline"] = (baseline_rel - numeric(out[f"{model}_relative_l2"])) / baseline_rel * 100.0
            out[f"{model}_rmse_drop_pct_vs_baseline"] = (baseline_rmse - numeric(out[f"{model}_rmse"])) / baseline_rmse * 100.0
    return rows


def build_split_summary(long_rows: List[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    baseline_by_dataset = {(r["split"], r["dataset_id"]): r for r in long_rows if r["model"] == "baseline"}
    for r in long_rows:
        groups.setdefault((r["model"], r["split"]), []).append(r)
    out = []
    for model in MODEL_ORDER:
        for split in ["train", "test", "generalization"]:
            rows = groups.get((model, split), [])
            if not rows:
                continue
            baseline_rows = [baseline_by_dataset[(r["split"], r["dataset_id"])] for r in rows]
            rmse_mean = mean(numeric(r["rmse"]) for r in rows)
            rel_mean = mean(numeric(r["relative_l2"]) for r in rows)
            acc_mean = mean(numeric(r["accuracy_score"]) for r in rows)
            base_rmse_mean = mean(numeric(r["rmse"]) for r in baseline_rows)
            base_rel_mean = mean(numeric(r["relative_l2"]) for r in baseline_rows)
            out.append({
                "model": model,
                "split": split,
                "dataset_count": len(rows),
                "model_epoch": rows[0]["model_epoch"],
                "elapsed_minutes": rows[0]["elapsed_minutes"],
                "rmse_mean": rmse_mean,
                "relative_l2_mean": rel_mean,
                "accuracy_score_mean": acc_mean,
                "rmse_drop_pct_vs_baseline": (base_rmse_mean - rmse_mean) / base_rmse_mean * 100.0 if model != "baseline" else 0.0,
                "relative_l2_drop_pct_vs_baseline": (base_rel_mean - rel_mean) / base_rel_mean * 100.0 if model != "baseline" else 0.0,
            })
    return out


def write_doc(path: Path, metadata: dict, split_rows: List[dict]) -> None:
    def row_for(model: str, split: str) -> dict:
        return next(r for r in split_rows if r["model"] == model and r["split"] == split)
    lines = [
        "# Darcy/C-flow raw 52x5 wall-clock table - 2026-06-08",
        "",
        "Status: complete. This is the corrected raw-data view: 52 datasets (`train`, `test`, and 50 separate `generalization` datasets) times 5 models (`baseline`, `loss1`, `loss2`, `loss3`, `physics`).",
        "",
        "Comparison axis: same wall-clock budget for the four self-training models. Epochs differ by objective and are not forced to match.",
        "",
        "## Outputs",
        "",
        f"- Long raw table, 260 rows: `{metadata['outputs']['long_csv']}`",
        f"- Wide raw table, 52 rows: `{metadata['outputs']['wide_csv']}`",
        f"- Split summary: `{metadata['outputs']['split_summary_csv']}`",
        "- Manifest: `adversarial_training_runs/darcy_cflow_wallclock_raw52x5_20260608/manifest.json`",
        "",
        "## Same Wall-Clock Generalization Summary",
        "",
        "| model | elapsed min | final epoch | gen rel-L2 mean | gen RMSE mean | gen rel-L2 drop vs baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ["loss1", "loss2", "loss3", "physics"]:
        r = row_for(model, "generalization")
        lines.append(f"| {model} | {float(r['elapsed_minutes']):.2f} | {r['model_epoch']} | {float(r['relative_l2_mean']):.6g} | {float(r['rmse_mean']):.9g} | {float(r['relative_l2_drop_pct_vs_baseline']):.2f}% |")
    lines += [
        "",
        "## Notes",
        "",
        "- Baseline is the common epoch-0 row, stored once per dataset rather than duplicated per objective.",
        "- The 50 generalization datasets are separate rows in both raw tables; they are not averaged away before export.",
        "- Split summaries are derived from the raw table and should be treated as summaries, not the original data.",
        "- Observed ranking by same wall-clock generalization relative-L2: loss3, physics, loss2, loss1.",
    ]
    path.write_text("\n".join(lines) + "\n")


def update_ledger(metadata: dict) -> None:
    heading = "## 2026-06-08 - Darcy/C-flow corrected raw 52x5 wall-clock table"
    lines = [
        heading,
        "",
        "Status: generated the corrected raw Darcy/C-flow table requested by the user. The table separates all 50 generalization datasets and includes train/test, giving 52 datasets times 5 models = 260 raw rows.",
        "",
        "Observed evidence:",
        f"- Long raw CSV: `{metadata['outputs']['long_csv']}` with `{metadata['long_rows']}` rows.",
        f"- Wide raw CSV: `{metadata['outputs']['wide_csv']}` with 52 dataset rows.",
        f"- Split summary CSV: `{metadata['outputs']['split_summary_csv']}`.",
        f"- Dedicated doc: `{metadata['outputs']['doc']}`.",
        f"- Four self-training models use same-wall-clock final checkpoints, not same epoch: loss1 epoch `{metadata['self_training_final_epochs']['loss1']}`, loss2 epoch `{metadata['self_training_final_epochs']['loss2']}`, loss3 epoch `{metadata['self_training_final_epochs']['loss3']}`, physics epoch `{metadata['self_training_final_epochs']['physics']}`.",
        "",
        "Inference:",
        "- The raw data product fixes the previous aggregation problem: the 50 generalization datasets are no longer collapsed before inspection. Epoch count is explicitly treated as an output of fixed wall-clock budget, not the matching variable.",
        "",
        "Remaining work:",
        "- Regenerate downstream plots directly from `darcy_cflow_raw52x5_wallclock_long.csv` or `darcy_cflow_raw52x5_wallclock_wide.csv` if per-dataset visualization is needed.",
        "",
    ]
    text = LEDGER.read_text()
    if heading not in text:
        LEDGER.write_text("\n".join(lines) + "\n" + text)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {name: read_summary(path) for name, path in RUNS.items()}
    eval_rows = {name: read_eval_rows(path) for name, path in RUNS.items()}
    baseline_rows_by_run = {name: {row_key(r): r for r in select_epoch_rows(rows, 0)} for name, rows in eval_rows.items()}
    baseline_keys = set(next(iter(baseline_rows_by_run.values())).keys())
    for name, rows_by_key in baseline_rows_by_run.items():
        if set(rows_by_key.keys()) != baseline_keys:
            raise SystemExit(f"baseline dataset keys mismatch for {name}")
    baseline_source_name = "loss1"
    baseline_rows = sorted(baseline_rows_by_run[baseline_source_name].values(), key=dataset_sort_key)
    if len(baseline_rows) != 52:
        raise SystemExit(f"expected 52 baseline datasets, got {len(baseline_rows)}")
    long_rows: List[dict] = []
    dataset_meta = {}
    for dataset_index, base in enumerate(baseline_rows, start=1):
        key = row_key(base)
        dataset_meta[key] = {"dataset_index": dataset_index}
        long_rows.append(build_long_row("baseline", base, dataset_index, "baseline_from_common_epoch0", str(RUNS[baseline_source_name].parent.relative_to(ROOT)), "baseline", 0, 0.0, 0.0, "baseline_initial_checkpoint", "baseline_epoch0"))
    sorted_keys = [row_key(r) for r in baseline_rows]
    for model in ["loss1", "loss2", "loss3", "physics"]:
        run_dir = RUNS[model]
        summary = summaries[model]
        epoch = int(summary["epochs"])
        rows_by_key = {row_key(r): r for r in select_epoch_rows(eval_rows[model], epoch)}
        if set(rows_by_key.keys()) != baseline_keys:
            raise SystemExit(f"final dataset keys mismatch for {model}")
        for key in sorted_keys:
            r = rows_by_key[key]
            long_rows.append(build_long_row(model, r, dataset_meta[key]["dataset_index"], run_dir.parent.name, str(run_dir.parent.relative_to(ROOT)), model, epoch, float(summary.get("elapsed_seconds", 0.0)), float(summary.get("elapsed_minutes", 0.0)), summary.get("final_checkpoint", ""), "final_same_wallclock_budget"))
    if len(long_rows) != 260:
        raise SystemExit(f"expected 260 long rows, got {len(long_rows)}")
    long_csv = OUT_DIR / "darcy_cflow_raw52x5_wallclock_long.csv"
    wide_csv = OUT_DIR / "darcy_cflow_raw52x5_wallclock_wide.csv"
    split_csv = OUT_DIR / "darcy_cflow_raw52x5_wallclock_split_summary.csv"
    write_csv(long_csv, long_rows)
    wide_rows = build_wide_rows(long_rows)
    if len(wide_rows) != 52:
        raise SystemExit(f"expected 52 wide rows, got {len(wide_rows)}")
    write_csv(wide_csv, wide_rows)
    split_rows = build_split_summary(long_rows)
    write_csv(split_csv, split_rows)
    metadata = {
        "status": "complete",
        "comparison_axis": "same wall-clock final checkpoint for self-training models; baseline epoch 0",
        "dataset_count": 52,
        "model_count": 5,
        "long_rows": len(long_rows),
        "models": MODEL_ORDER,
        "self_training_final_epochs": {name: int(summaries[name]["epochs"]) for name in RUNS},
        "self_training_elapsed_minutes": {name: float(summaries[name]["elapsed_minutes"]) for name in RUNS},
        "source_run_dirs": {name: str(path.relative_to(ROOT)) for name, path in RUNS.items()},
        "outputs": {"long_csv": str(long_csv.relative_to(ROOT)), "wide_csv": str(wide_csv.relative_to(ROOT)), "split_summary_csv": str(split_csv.relative_to(ROOT)), "doc": str(DOC.relative_to(ROOT))},
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    write_doc(DOC, metadata, split_rows)
    update_ledger(metadata)


if __name__ == "__main__":
    main()
