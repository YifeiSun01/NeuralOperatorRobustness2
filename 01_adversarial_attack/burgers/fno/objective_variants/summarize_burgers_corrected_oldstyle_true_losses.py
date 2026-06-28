#!/usr/bin/env python3
"""Aggregate true-loss summaries across multiple initial-condition indices."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUP_FIELDS = ("nu", "domain", "norm", "epsilon", "alpha", "steps", "method")
VALUE_FIELDS = (
    "initial_true_loss",
    "final_true_loss",
    "true_loss_increase",
    "true_loss_ratio",
)


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = [parse_float(row.get(field)) for row in rows]
    return [value for value in values if math.isfinite(value)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def variance(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = mean(values)
    return sum((value - mu) ** 2 for value in values) / len(values)


def read_rows(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("*/true_loss_summary.csv"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        tag = path.parent.name
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row.setdefault("tag", tag)
                if not row["tag"]:
                    row["tag"] = tag
                row["source_summary_csv"] = str(path)
                rows.append(row)
    return rows


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


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], result_root: Path) -> None:
    lines = [
        "# Aggregated True Loss Summary",
        "",
        f"- result_root: `{result_root}`",
        "- grouping: `nu, domain, norm, epsilon, alpha, steps, method`",
        "- variance uses population variance over available initial-condition indices.",
        "",
        "| nu | norm | epsilon | alpha | method | n | index list | final mean | final std | increase mean | ratio mean | ratio std |",
        "|---:|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {nu} | {norm} | {epsilon} | {alpha} | {method} | {n} | {indices} | {final_mean} | {final_std} | {inc_mean} | {ratio_mean} | {ratio_std} |".format(
                nu=fmt(row["nu"]),
                norm=row["norm"],
                epsilon=fmt(row["epsilon"]),
                alpha=fmt(row["alpha"]),
                method=row["method"],
                n=row["num_initial_conditions"],
                indices=row["initial_condition_indices"],
                final_mean=fmt(row["final_true_loss_mean"]),
                final_std=fmt(row["final_true_loss_std"]),
                inc_mean=fmt(row["true_loss_increase_mean"]),
                ratio_mean=fmt(row["true_loss_ratio_mean"]),
                ratio_std=fmt(row["true_loss_ratio_std"]),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        base = dict(zip(GROUP_FIELDS, key, strict=True))
        indices = sorted({int(float(row["initial_condition_index"])) for row in group_rows if row.get("initial_condition_index", "") != ""})
        summary: dict[str, Any] = {
            **base,
            "num_initial_conditions": len(indices),
            "num_rows": len(group_rows),
            "initial_condition_indices": " ".join(str(index) for index in indices),
        }
        for field in VALUE_FIELDS:
            values = finite_values(group_rows, field)
            var = variance(values)
            summary[f"{field}_mean"] = mean(values)
            summary[f"{field}_variance"] = var
            summary[f"{field}_std"] = math.sqrt(var) if math.isfinite(var) else float("nan")
            summary[f"{field}_count"] = len(values)
        out.append(summary)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_root",
        type=Path,
        default=Path("results/burgers_corrected_oldstyle_5loss_clean_rerun"),
        help="Root containing one subdirectory per run, each with true_loss_summary.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for aggregate CSV/Markdown. Defaults to result_root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir or args.result_root
    rows = read_rows(args.result_root)
    aggregate_rows = aggregate(rows)
    write_csv(outdir / "true_loss_summary_all_indices.csv", rows)
    write_csv(outdir / "true_loss_summary_aggregate.csv", aggregate_rows)
    write_markdown(outdir / "true_loss_summary_aggregate.md", aggregate_rows, args.result_root)
    print(f"[done] read {len(rows)} method rows from {args.result_root}")
    print(f"[done] aggregate rows: {len(aggregate_rows)}")
    print(f"[done] saved {outdir / 'true_loss_summary_aggregate.csv'}")
    print(f"[done] saved {outdir / 'true_loss_summary_aggregate.md'}")


if __name__ == "__main__":
    main()
