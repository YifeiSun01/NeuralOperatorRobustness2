#!/usr/bin/env python3
"""Build wide ratio tables for Burgers FNO/DeepONet 5-loss attack runs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = (
    "loss1",
    "loss2_fixed",
    "loss2_dict_N200",
    "loss2_dict_N2000",
    "loss2_dict_N20000",
    "loss3_stopgrad",
    "loss3",
)

EXCLUDE_PARTS = (
    "batch_validation",
    "smoke",
    "comparison_reports",
)


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.6g}"
    return str(value)


def model_from_path(path: Path) -> str:
    text = str(path).lower()
    if "deeponet" in text:
        return "DeepONet"
    return "FNO"


def should_skip(path: Path) -> bool:
    text = str(path).lower()
    return any(part in text for part in EXCLUDE_PARTS)


def read_summary(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row.get("method", "")
            if method:
                rows[method] = dict(row)
    return rows


def index_from_tag(tag: str, fallback: str | None) -> int | None:
    match = re.search(r"_index(\d+)(?:_|$)", tag)
    if match:
        return int(match.group(1))
    value = parse_float(fallback)
    return int(value) if math.isfinite(value) else None


def best_method(ratios: dict[str, float]) -> tuple[str, float]:
    finite = [(method, value) for method, value in ratios.items() if math.isfinite(value)]
    if not finite:
        return "", float("nan")
    return max(finite, key=lambda item: item[1])


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def variance(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = mean(values)
    return sum((value - mu) ** 2 for value in values) / len(values)


def collect_rows(result_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(result_root.rglob("true_loss_summary.csv")):
        if should_skip(path):
            continue
        by_method = read_summary(path)
        if not by_method:
            continue
        methods_present = [method for method in METHODS if method in by_method]
        if not methods_present:
            continue
        first = by_method[methods_present[0]]
        tag = first.get("tag") or path.parent.name
        ratios = {method: parse_float(by_method.get(method, {}).get("true_loss_ratio")) for method in METHODS}
        finals = {method: parse_float(by_method.get(method, {}).get("final_true_loss")) for method in METHODS}
        increases = {method: parse_float(by_method.get(method, {}).get("true_loss_increase")) for method in METHODS}
        best, best_ratio = best_method(ratios)
        row: dict[str, Any] = {
            "model": model_from_path(path),
            "result_root": str(path.parent.parent),
            "tag": tag,
            "index": index_from_tag(tag, first.get("initial_condition_index")),
            "nu": parse_float(first.get("nu")),
            "domain": parse_float(first.get("domain")),
            "norm": first.get("norm", ""),
            "epsilon": parse_float(first.get("epsilon")),
            "alpha": parse_float(first.get("alpha")),
            "steps": int(parse_float(first.get("steps"))) if math.isfinite(parse_float(first.get("steps"))) else "",
        }
        for method in METHODS:
            row[f"{method}_ratio"] = ratios[method]
            row[f"{method}_increase"] = increases[method]
        for method in METHODS:
            row[f"{method}_final_true_loss"] = finals[method]
        row["best_method"] = best
        row["best_ratio"] = best_ratio
        row["source_summary_csv"] = str(path)
        out.append(row)
    return out


def aggregate_by_parameter(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["model"],
            row["result_root"],
            row["nu"],
            row["domain"],
            row["norm"],
            row["epsilon"],
            row["alpha"],
            row["steps"],
        )
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        model, root, nu, domain, norm, epsilon, alpha, steps = key
        summary: dict[str, Any] = {
            "model": model,
            "result_root": root,
            "nu": nu,
            "domain": domain,
            "norm": norm,
            "epsilon": epsilon,
            "alpha": alpha,
            "steps": steps,
            "n_runs": len(group_rows),
            "indices": " ".join(str(row["index"]) for row in sorted(group_rows, key=lambda r: r["index"] if r["index"] is not None else -1)),
        }
        mean_ratios: dict[str, float] = {}
        win_counts = {method: 0 for method in METHODS}
        for row in group_rows:
            if row["best_method"] in win_counts:
                win_counts[row["best_method"]] += 1
        for method in METHODS:
            values = [row[f"{method}_ratio"] for row in group_rows if math.isfinite(row[f"{method}_ratio"])]
            ratio_var = variance(values)
            mean_ratios[method] = mean(values)
            summary[f"{method}_ratio_mean"] = mean_ratios[method]
            summary[f"{method}_ratio_variance"] = ratio_var
            summary[f"{method}_ratio_std"] = math.sqrt(ratio_var) if math.isfinite(ratio_var) else float("nan")
            increases = [row[f"{method}_increase"] for row in group_rows if math.isfinite(row[f"{method}_increase"])]
            increase_var = variance(increases)
            summary[f"{method}_increase_mean"] = mean(increases)
            summary[f"{method}_increase_variance"] = increase_var
            summary[f"{method}_increase_std"] = math.sqrt(increase_var) if math.isfinite(increase_var) else float("nan")
            finals = [row[f"{method}_final_true_loss"] for row in group_rows if math.isfinite(row[f"{method}_final_true_loss"])]
            final_var = variance(finals)
            summary[f"{method}_final_true_loss_mean"] = mean(finals)
            summary[f"{method}_final_true_loss_variance"] = final_var
            summary[f"{method}_final_true_loss_std"] = math.sqrt(final_var) if math.isfinite(final_var) else float("nan")
            summary[f"{method}_wins"] = win_counts[method]
        best, best_ratio = best_method(mean_ratios)
        summary["best_mean_method"] = best
        summary["best_mean_ratio"] = best_ratio
        mean_increases = {
            method: summary[f"{method}_increase_mean"]
            for method in METHODS
        }
        best_increase_method, best_increase = best_method(mean_increases)
        summary["best_increase_mean_method"] = best_increase_method
        summary["best_increase_mean"] = best_increase
        mean_final_losses = {
            method: summary[f"{method}_final_true_loss_mean"]
            for method in METHODS
        }
        best_final_method, best_final = best_method(mean_final_losses)
        summary["best_final_true_loss_mean_method"] = best_final_method
        summary["best_final_true_loss_mean"] = best_final
        out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], *, aggregate: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if aggregate:
        fields = [
            "model",
            "nu",
            "norm",
            "epsilon",
            "alpha",
            "n_runs",
            "indices",
            *[f"{method}_ratio_mean" for method in METHODS],
            "best_mean_method",
            "best_mean_ratio",
            "best_increase_mean_method",
            "best_increase_mean",
            "best_final_true_loss_mean_method",
            "best_final_true_loss_mean",
        ]
    else:
        fields = [
            "model",
            "nu",
            "norm",
            "epsilon",
            "alpha",
            "index",
            *[f"{method}_ratio" for method in METHODS],
            "best_method",
            "best_ratio",
        ]
    lines = [
        "# Burgers Attack Ratio Summary",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", type=Path, default=Path("results"))
    parser.add_argument("--outdir", type=Path, default=Path("results/burgers_attack_ratio_summary"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.result_root)
    rows.sort(key=lambda row: (row["model"], row["nu"], row["norm"], row["epsilon"], row["alpha"], row["index"] if row["index"] is not None else -1, row["tag"]))
    aggregate_rows = aggregate_by_parameter(rows)
    write_csv(args.outdir / "burgers_attack_ratio_runs.csv", rows)
    write_csv(args.outdir / "burgers_attack_ratio_by_parameter.csv", aggregate_rows)
    write_markdown(args.outdir / "burgers_attack_ratio_runs.md", rows, aggregate=False)
    write_markdown(args.outdir / "burgers_attack_ratio_by_parameter.md", aggregate_rows, aggregate=True)
    print(f"[done] runs: {len(rows)}")
    print(f"[done] parameter groups: {len(aggregate_rows)}")
    print(f"[done] saved {args.outdir / 'burgers_attack_ratio_runs.csv'}")
    print(f"[done] saved {args.outdir / 'burgers_attack_ratio_by_parameter.csv'}")


if __name__ == "__main__":
    main()
