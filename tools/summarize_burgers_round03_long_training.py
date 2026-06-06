#!/usr/bin/env python3
"""Summarize Burgers round03 long loss1/loss2/loss3 training runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    "loss1": PROJECT_ROOT / "adversarial_training_runs/burgers_loss3_selective_round03_loss1_1000ep_long_20260605",
    "loss2": PROJECT_ROOT / "adversarial_training_runs/burgers_loss3_selective_round03_loss2_500ep_long_20260605",
    "loss3": PROJECT_ROOT / "adversarial_training_runs/burgers_loss3_selective_round03_loss3_500ep_long_20260605",
}
DEFAULT_OUT_DIR = PROJECT_ROOT / "forensics/burgers_loss3_selective_round03_long_training_comparison_20260605"
DEFAULT_DOC = PROJECT_ROOT / "docs/burgers_loss3_selective_round03_long_training_report_20260605.md"
SPLITS = ["train", "test", "generalization", "ALL"]
SELECTED_EPOCHS = [0, 10, 50, 100, 200, 300, 400, 500]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        text = row.get(key, "")
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def as_int(row: dict[str, str], key: str, default: int = -1) -> int:
    try:
        text = row.get(key, "")
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.fmean(vals) if vals else float("nan")


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(val):
        return ""
    return f"{val:.{digits}g}"


def markdown_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]], limit: int | None = None) -> list[str]:
    use_rows = rows if limit is None else rows[:limit]
    out = ["| " + " | ".join(label for label, _ in cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in use_rows:
        out.append("| " + " | ".join(fmt(row.get(key, "")) for _, key in cols) + " |")
    return out


@dataclass
class RunData:
    loss: str
    run_root: Path
    burgers_dir: Path
    summary: dict[str, Any]
    split_rows: list[dict[str, str]]
    metric_rows: list[dict[str, str]]
    checkpoint_rows: list[dict[str, str]]
    attack_rows: list[dict[str, str]]
    train_rows: list[dict[str, str]]
    eval_pass_rows: list[dict[str, str]]
    elapsed_by_epoch: dict[int, float]

    def split_metric(self, epoch: int, split: str) -> dict[str, str] | None:
        matches = [r for r in self.split_rows if as_int(r, "epoch") == epoch and r.get("split") == split]
        return matches[-1] if matches else None

    def generated_metric_rows(self, epoch: int) -> list[dict[str, str]]:
        return [r for r in self.metric_rows if as_int(r, "epoch") == epoch and r.get("split") == "generalization"]

    def final_checkpoint(self) -> dict[str, str] | None:
        finals = [r for r in self.checkpoint_rows if r.get("checkpoint_reason") == "final"]
        return finals[-1] if finals else (self.checkpoint_rows[-1] if self.checkpoint_rows else None)

    def nearest_checkpoint(self, target_seconds: float) -> dict[str, str] | None:
        rows = [r for r in self.checkpoint_rows if math.isfinite(as_float(r, "wall_elapsed_seconds"))]
        if not rows:
            return None
        return min(rows, key=lambda r: abs(as_float(r, "wall_elapsed_seconds") - target_seconds))

    def max_wall_seconds(self) -> float:
        vals = [as_float(r, "wall_elapsed_seconds") for r in self.checkpoint_rows]
        vals = [v for v in vals if math.isfinite(v)]
        return max(vals) if vals else max(self.elapsed_by_epoch.values(), default=0.0)


def build_elapsed_by_epoch(train_rows: list[dict[str, str]], eval_rows: list[dict[str, str]], checkpoint_rows: list[dict[str, str]]) -> dict[int, float]:
    # train_start_wall in adversarial_training.py starts after baseline evaluation.
    train_by_epoch: dict[int, float] = {}
    for row in train_rows:
        epoch = as_int(row, "epoch")
        if epoch <= 0:
            continue
        train_by_epoch[epoch] = train_by_epoch.get(epoch, 0.0) + as_float(row, "step_wall_sec", 0.0)
    eval_by_epoch: dict[int, float] = {}
    for row in eval_rows:
        epoch = as_int(row, "epoch")
        if epoch <= 0:
            continue
        eval_by_epoch[epoch] = as_float(row, "eval_wall_sec", 0.0)
    elapsed: dict[int, float] = {0: 0.0}
    total = 0.0
    for epoch in sorted(set(train_by_epoch) | set(eval_by_epoch)):
        total += train_by_epoch.get(epoch, 0.0) + eval_by_epoch.get(epoch, 0.0)
        elapsed[epoch] = total
    # Prefer exact wall times from checkpoints for checkpointed epochs.
    for row in checkpoint_rows:
        epoch = as_int(row, "epoch")
        wall = as_float(row, "wall_elapsed_seconds")
        if epoch >= 0 and math.isfinite(wall):
            elapsed[epoch] = wall
    return elapsed


def load_run(loss: str, run_root: Path) -> RunData:
    burgers_dir = run_root / "burgers"
    with (run_root / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    split_rows = read_csv(burgers_dir / "eval_split_summary.csv")
    metric_rows = read_csv(burgers_dir / "eval_metrics.csv")
    checkpoint_rows = read_csv(burgers_dir / "checkpoints.csv")
    attack_rows = read_csv(burgers_dir / "attack_epoch_summary.csv")
    train_rows = read_csv(burgers_dir / "train_steps.csv")
    eval_pass_rows = read_csv(burgers_dir / "evaluation_passes.csv")
    elapsed_by_epoch = build_elapsed_by_epoch(train_rows, eval_pass_rows, checkpoint_rows)
    return RunData(loss, run_root, burgers_dir, summary, split_rows, metric_rows, checkpoint_rows, attack_rows, train_rows, eval_pass_rows, elapsed_by_epoch)


def split_metric_row(run: RunData, comparison: str, epoch: int, split: str) -> dict[str, Any] | None:
    row = run.split_metric(epoch, split)
    if row is None:
        return None
    return {
        "comparison": comparison,
        "loss": run.loss,
        "epoch": epoch,
        "wall_seconds": run.elapsed_by_epoch.get(epoch, float("nan")),
        "wall_hours": run.elapsed_by_epoch.get(epoch, float("nan")) / 3600.0,
        "split": split,
        "dataset_count": as_int(row, "dataset_count"),
        "total_samples_evaluated": as_int(row, "total_samples_evaluated"),
        "rmse": as_float(row, "rmse_dataset_mean"),
        "mae": as_float(row, "mae_dataset_mean"),
        "relative_l2": as_float(row, "relative_l2_dataset_mean"),
        "accuracy_score": as_float(row, "accuracy_score_dataset_mean"),
    }


def build_epoch_aligned(runs: dict[str, RunData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in SELECTED_EPOCHS:
        for run in runs.values():
            if epoch == 0 or epoch in run.elapsed_by_epoch:
                for split in SPLITS:
                    out = split_metric_row(run, "same_epoch", epoch, split)
                    if out is not None:
                        rows.append(out)
    for loss, run in runs.items():
        final = run.final_checkpoint()
        if final:
            epoch = as_int(final, "epoch")
            for split in SPLITS:
                out = split_metric_row(run, "run_final", epoch, split)
                if out is not None:
                    rows.append(out)
    return rows


def checkpoint_selection_rows(runs: dict[str, RunData]) -> list[dict[str, Any]]:
    targets: list[tuple[str, float, str]] = []
    for loss in ["loss1", "loss2", "loss3"]:
        final = runs[loss].final_checkpoint()
        if final:
            targets.append((f"same_wall_as_{loss}_final", as_float(final, "wall_elapsed_seconds"), loss))
    explicit_targets = sorted({as_float(r, "checkpoint_wall_target_seconds") for run in runs.values() for r in run.checkpoint_rows if r.get("checkpoint_reason") == "wall_clock" and math.isfinite(as_float(r, "checkpoint_wall_target_seconds"))})
    for target in explicit_targets:
        targets.append((f"wall_target_{int(round(target)):07d}s", target, "configured_wall_checkpoint"))
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for label, target, source in targets:
        if not math.isfinite(target):
            continue
        for loss, run in runs.items():
            ckpt = run.nearest_checkpoint(target)
            if ckpt is None:
                continue
            key = (label, loss)
            if key in seen:
                continue
            seen.add(key)
            wall = as_float(ckpt, "wall_elapsed_seconds")
            epoch = as_int(ckpt, "epoch")
            rows.append({
                "selection": label,
                "target_source": source,
                "target_wall_seconds": target,
                "target_wall_hours": target / 3600.0,
                "loss": loss,
                "selected_epoch": epoch,
                "selected_global_step": as_int(ckpt, "global_step"),
                "selected_wall_seconds": wall,
                "selected_wall_hours": wall / 3600.0,
                "wall_delta_seconds": wall - target,
                "target_reached_by_run": run.max_wall_seconds() >= target,
                "checkpoint_reason": ckpt.get("checkpoint_reason", ""),
                "checkpoint_wall_target_seconds": ckpt.get("checkpoint_wall_target_seconds", ""),
                "checkpoint_path": ckpt.get("checkpoint_path", ""),
            })
    return rows


def build_wall_aligned(runs: dict[str, RunData], selections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ratios: list[dict[str, Any]] = []
    for sel in selections:
        epoch = int(sel["selected_epoch"])
        loss = str(sel["loss"])
        run = runs[loss]
        for split in SPLITS:
            out = split_metric_row(run, str(sel["selection"]), epoch, split)
            if out is None:
                continue
            out.update({
                "target_wall_seconds": sel["target_wall_seconds"],
                "target_wall_hours": sel["target_wall_hours"],
                "selected_checkpoint_wall_seconds": sel["selected_wall_seconds"],
                "selected_checkpoint_wall_hours": sel["selected_wall_hours"],
                "wall_delta_seconds": sel["wall_delta_seconds"],
                "target_reached_by_run": sel["target_reached_by_run"],
                "checkpoint_path": sel["checkpoint_path"],
            })
            rows.append(out)
    by_key = {(r["comparison"], r["split"], r["loss"]): r for r in rows}
    for comparison in sorted({r["comparison"] for r in rows}):
        for split in SPLITS:
            l3 = by_key.get((comparison, split, "loss3"))
            if l3 is None:
                continue
            for base_loss in ["loss1", "loss2"]:
                base = by_key.get((comparison, split, base_loss))
                if base is None:
                    continue
                ratios.append({
                    "comparison": comparison,
                    "split": split,
                    "baseline_loss": base_loss,
                    "baseline_epoch": base["epoch"],
                    "loss3_epoch": l3["epoch"],
                    "baseline_target_reached": base.get("target_reached_by_run", True),
                    "loss3_target_reached": l3.get("target_reached_by_run", True),
                    "baseline_rmse": base["rmse"],
                    "loss3_rmse": l3["rmse"],
                    "loss3_rmse_over_baseline": l3["rmse"] / base["rmse"] if base["rmse"] else float("nan"),
                    "baseline_relative_l2": base["relative_l2"],
                    "loss3_relative_l2": l3["relative_l2"],
                    "loss3_relative_l2_over_baseline": l3["relative_l2"] / base["relative_l2"] if base["relative_l2"] else float("nan"),
                })
    return rows, ratios


def per_dataset_advantage(runs: dict[str, RunData], comparison: str, epochs: dict[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_dataset: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    gen_by_loss: dict[str, dict[str, dict[str, str]]] = {}
    for loss, epoch in epochs.items():
        gen_by_loss[loss] = {r["dataset_id"]: r for r in runs[loss].generated_metric_rows(epoch)}
    common_ids = sorted(set.intersection(*(set(v) for v in gen_by_loss.values()))) if gen_by_loss else []
    for dataset_id in common_ids:
        row: dict[str, Any] = {"comparison": comparison, "dataset_id": dataset_id}
        for loss, epoch in epochs.items():
            m = gen_by_loss[loss][dataset_id]
            row[f"{loss}_epoch"] = epoch
            row[f"{loss}_rmse"] = as_float(m, "rmse")
            row[f"{loss}_relative_l2"] = as_float(m, "relative_l2")
        for base_loss in ["loss1", "loss2"]:
            if base_loss in epochs and "loss3" in epochs:
                row[f"loss3_win_vs_{base_loss}_rmse"] = row["loss3_rmse"] < row[f"{base_loss}_rmse"]
                row[f"loss3_rmse_advantage_pct_vs_{base_loss}"] = 100.0 * (row[f"{base_loss}_rmse"] - row["loss3_rmse"]) / row[f"{base_loss}_rmse"]
        per_dataset.append(row)
    for base_loss in ["loss1", "loss2"]:
        flag = f"loss3_win_vs_{base_loss}_rmse"
        vals = [float(r.get(f"loss3_rmse_advantage_pct_vs_{base_loss}", float("nan"))) for r in per_dataset]
        wins = sum(1 for r in per_dataset if r.get(flag) is True)
        summaries.append({
            "comparison": comparison,
            "baseline_loss": base_loss,
            "datasets_compared": len(per_dataset),
            "loss3_rmse_wins": wins,
            "loss3_rmse_win_fraction": wins / len(per_dataset) if per_dataset else float("nan"),
            "loss3_rmse_advantage_pct_mean": finite_mean(vals),
            "loss3_rmse_advantage_pct_min": min([v for v in vals if math.isfinite(v)], default=float("nan")),
            "loss3_rmse_advantage_pct_max": max([v for v in vals if math.isfinite(v)], default=float("nan")),
        })
    return per_dataset, summaries


def build_per_dataset_outputs(runs: dict[str, RunData]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for epoch in [10, 50, 100, 200, 300, 400, 500]:
        if all(epoch in run.elapsed_by_epoch for run in runs.values()):
            r, s = per_dataset_advantage(runs, f"same_epoch_{epoch}", {"loss1": epoch, "loss2": epoch, "loss3": epoch})
            rows.extend(r)
            summaries.extend(s)
    finals = {}
    for loss, run in runs.items():
        ckpt = run.final_checkpoint()
        if ckpt:
            finals[loss] = as_int(ckpt, "epoch")
    if set(finals) == set(runs):
        r, s = per_dataset_advantage(runs, "run_final", finals)
        rows.extend(r)
        summaries.extend(s)
    return rows, summaries


def best_epoch_rows(runs: dict[str, RunData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for loss, run in runs.items():
        for split in SPLITS:
            candidates = []
            for row in run.split_rows:
                epoch = as_int(row, "epoch")
                if row.get("split") != split or epoch < 0:
                    continue
                candidates.append((as_float(row, "rmse_dataset_mean"), epoch, row))
            if not candidates:
                continue
            rmse, epoch, row = min(candidates, key=lambda x: x[0])
            rows.append({
                "loss": loss,
                "split": split,
                "best_epoch_by_rmse": epoch,
                "best_wall_hours": run.elapsed_by_epoch.get(epoch, float("nan")) / 3600.0,
                "best_rmse": rmse,
                "best_relative_l2": as_float(row, "relative_l2_dataset_mean"),
            })
    return rows


def attack_delta_rows(runs: dict[str, RunData]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    wanted = set(SELECTED_EPOCHS + [1000])
    for loss, run in runs.items():
        all_l2 = []
        all_linf = []
        for row in run.attack_rows:
            epoch = as_int(row, "epoch")
            l2 = as_float(row, "delta_l2_rms_mean")
            linf = as_float(row, "delta_linf_mean")
            all_l2.append(l2)
            all_linf.append(linf)
            if epoch in wanted:
                selected.append({
                    "loss": loss,
                    "epoch": epoch,
                    "delta_l2_rms_mean": l2,
                    "delta_linf_mean": linf,
                    "boundary_ratio_mean": as_float(row, "boundary_ratio_mean"),
                    "epsilon_mean": as_float(row, "epsilon_mean"),
                    "attack_loss_gain_mean": as_float(row, "attack_loss_gain_mean"),
                    "train_loss_used_for_optimizer_updates_mean": as_float(row, "train_loss_used_for_optimizer_updates_mean"),
                    "attack_wall_sec_total": as_float(row, "attack_wall_sec_total"),
                    "attack_samples_per_sec": as_float(row, "attack_samples_per_sec"),
                })
        final = run.attack_rows[-1] if run.attack_rows else {}
        summary.append({
            "loss": loss,
            "epochs": len(run.attack_rows),
            "delta_l2_rms_mean_over_epochs": finite_mean(all_l2),
            "delta_linf_mean_over_epochs": finite_mean(all_linf),
            "final_epoch": as_int(final, "epoch", -1),
            "final_delta_l2_rms_mean": as_float(final, "delta_l2_rms_mean"),
            "final_delta_linf_mean": as_float(final, "delta_linf_mean"),
            "final_boundary_ratio_mean": as_float(final, "boundary_ratio_mean"),
        })
    return selected, summary


def write_report(args: argparse.Namespace, runs: dict[str, RunData], outputs: dict[str, list[dict[str, Any]]]) -> None:
    lines: list[str] = []
    lines.extend([
        "# Burgers Round03 Long Training Report",
        "",
        "Date: 2026-06-05 UTC.",
        "",
        "Observed from local run directories:",
    ])
    for loss, run in runs.items():
        final = run.final_checkpoint()
        lines.append(f"- `{loss}`: `{run.run_root.relative_to(PROJECT_ROOT)}` final epoch {as_int(final or {}, 'epoch')} wall {fmt(as_float(final or {}, 'wall_elapsed_seconds') / 3600.0)} h")
    lines.extend([
        "",
        "Round03 generalization root: `generalization_datasets_burgers_loss3_selective_search/round_03`.",
        "The long runs use full train/test/generated50 evaluation every epoch and checkpoint every 100 epochs plus configured wall-clock checkpoints.",
        "",
        "## Same-Epoch Metrics",
        "",
    ])
    same_epoch_key = {10, 50, 100, 200, 300, 400, 500}
    compact = [r for r in outputs["epoch_aligned"] if r["comparison"] == "same_epoch" and r["epoch"] in same_epoch_key and r["split"] in {"test", "generalization"}]
    lines.extend(markdown_table(compact, [("epoch", "epoch"), ("loss", "loss"), ("split", "split"), ("wall h", "wall_hours"), ("RMSE", "rmse"), ("rel L2", "relative_l2")], limit=80))
    lines.extend(["", "## Final Metrics", ""])
    finals = [r for r in outputs["epoch_aligned"] if r["comparison"] == "run_final" and r["split"] in {"train", "test", "generalization"}]
    lines.extend(markdown_table(finals, [("loss", "loss"), ("epoch", "epoch"), ("split", "split"), ("wall h", "wall_hours"), ("RMSE", "rmse"), ("rel L2", "relative_l2")]))
    lines.extend(["", "## Wall-Clock Pair Ratios", ""])
    ratio_focus = [r for r in outputs["wall_ratios"] if r["comparison"] in {"same_wall_as_loss1_final", "same_wall_as_loss2_final"} and r["split"] in {"test", "generalization", "ALL"}]
    lines.extend(markdown_table(ratio_focus, [("comparison", "comparison"), ("split", "split"), ("base", "baseline_loss"), ("base ep", "baseline_epoch"), ("loss3 ep", "loss3_epoch"), ("loss3/base RMSE", "loss3_rmse_over_baseline"), ("loss3/base relL2", "loss3_relative_l2_over_baseline")]))
    lines.extend(["", "## Generated Dataset Win Counts", ""])
    lines.extend(markdown_table(outputs["per_dataset_summary"], [("comparison", "comparison"), ("base", "baseline_loss"), ("n", "datasets_compared"), ("loss3 wins", "loss3_rmse_wins"), ("win frac", "loss3_rmse_win_fraction"), ("adv pct mean", "loss3_rmse_advantage_pct_mean"), ("adv pct min", "loss3_rmse_advantage_pct_min")], limit=40))
    lines.extend(["", "## Attack Delta Summary", ""])
    lines.extend(markdown_table(outputs["attack_delta_summary"], [("loss", "loss"), ("epochs", "epochs"), ("mean l2 rms", "delta_l2_rms_mean_over_epochs"), ("mean linf", "delta_linf_mean_over_epochs"), ("final epoch", "final_epoch"), ("final l2 rms", "final_delta_l2_rms_mean"), ("final linf", "final_delta_linf_mean")]))
    lines.extend([
        "",
        "## Output CSVs",
        "",
        f"- `{args.out_dir.relative_to(PROJECT_ROOT) / 'epoch_aligned_split_metrics.csv'}`",
        f"- `{args.out_dir.relative_to(PROJECT_ROOT) / 'wall_clock_checkpoint_selection.csv'}`",
        f"- `{args.out_dir.relative_to(PROJECT_ROOT) / 'wall_clock_aligned_split_metrics.csv'}`",
        f"- `{args.out_dir.relative_to(PROJECT_ROOT) / 'wall_clock_pairwise_ratios.csv'}`",
        f"- `{args.out_dir.relative_to(PROJECT_ROOT) / 'per_dataset_generated_advantage.csv'}`",
        f"- `{args.out_dir.relative_to(PROJECT_ROOT) / 'per_dataset_generated_advantage_summary.csv'}`",
        f"- `{args.out_dir.relative_to(PROJECT_ROOT) / 'best_epoch_by_split.csv'}`",
        f"- `{args.out_dir.relative_to(PROJECT_ROOT) / 'attack_delta_selected_epochs.csv'}`",
        f"- `{args.out_dir.relative_to(PROJECT_ROOT) / 'attack_delta_summary_by_loss.csv'}`",
        "",
        "Inference from these tables should be made after the Jacobian/SVD diagnostics are added, because prediction and local linearization can diverge.",
    ])
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss1-run", type=Path, default=DEFAULT_RUNS["loss1"])
    parser.add_argument("--loss2-run", type=Path, default=DEFAULT_RUNS["loss2"])
    parser.add_argument("--loss3-run", type=Path, default=DEFAULT_RUNS["loss3"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_DOC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        "loss1": load_run("loss1", args.loss1_run.resolve()),
        "loss2": load_run("loss2", args.loss2_run.resolve()),
        "loss3": load_run("loss3", args.loss3_run.resolve()),
    }
    epoch_aligned = build_epoch_aligned(runs)
    selections = checkpoint_selection_rows(runs)
    wall_aligned, wall_ratios = build_wall_aligned(runs, selections)
    per_dataset_rows, per_dataset_summary = build_per_dataset_outputs(runs)
    best_rows = best_epoch_rows(runs)
    attack_selected, attack_summary = attack_delta_rows(runs)

    write_csv(args.out_dir / "epoch_aligned_split_metrics.csv", epoch_aligned)
    write_csv(args.out_dir / "wall_clock_checkpoint_selection.csv", selections)
    write_csv(args.out_dir / "wall_clock_aligned_split_metrics.csv", wall_aligned)
    write_csv(args.out_dir / "wall_clock_pairwise_ratios.csv", wall_ratios)
    write_csv(args.out_dir / "per_dataset_generated_advantage.csv", per_dataset_rows)
    write_csv(args.out_dir / "per_dataset_generated_advantage_summary.csv", per_dataset_summary)
    write_csv(args.out_dir / "best_epoch_by_split.csv", best_rows)
    write_csv(args.out_dir / "attack_delta_selected_epochs.csv", attack_selected)
    write_csv(args.out_dir / "attack_delta_summary_by_loss.csv", attack_summary)

    outputs = {
        "epoch_aligned": epoch_aligned,
        "selections": selections,
        "wall_aligned": wall_aligned,
        "wall_ratios": wall_ratios,
        "per_dataset": per_dataset_rows,
        "per_dataset_summary": per_dataset_summary,
        "best": best_rows,
        "attack_delta_selected": attack_selected,
        "attack_delta_summary": attack_summary,
    }
    write_report(args, runs, outputs)
    print(json.dumps({"out_dir": str(args.out_dir), "report_path": str(args.report_path), "rows": {k: len(v) for k, v in outputs.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
