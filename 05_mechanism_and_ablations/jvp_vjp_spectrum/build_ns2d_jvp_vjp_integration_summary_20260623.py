#!/usr/bin/env python3
"""Summarize NS2D JVP/VJP mechanism probes into the Loss3 evidence record."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def group_mean(rows: list[dict[str, Any]], keys: tuple[str, ...], values: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = {k: v for k, v in zip(keys, key)}
        row["n"] = len(items)
        for value in values:
            row[f"{value}_mean"] = mean([fnum(item.get(value)) for item in items])
        out.append(row)
    return out


def collect_run(run_label: str, run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        notes.append(f"- `{rel(run_dir)}` is not complete yet: missing manifest.")
        return rows, notes
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows.append(
        {
            "run": run_label,
            "metric_group": "manifest",
            "metric": "status",
            "value": manifest.get("status", "unknown"),
            "source": rel(manifest_path),
        }
    )
    for key, value in (manifest.get("row_counts") or {}).items():
        rows.append(
            {
                "run": run_label,
                "metric_group": "manifest",
                "metric": f"row_count_{key}",
                "value": value,
                "source": rel(manifest_path),
            }
        )
    rows.append(
        {
            "run": run_label,
            "metric_group": "manifest",
            "metric": "jvp_mode_used",
            "value": json.dumps(manifest.get("jvp_mode_used", {}), sort_keys=True),
            "source": rel(manifest_path),
        }
    )

    spectrum = read_csv(run_dir / "spectrum_path.csv")
    for row in group_mean(
        spectrum,
        ("method",),
        ("top_sigma", "cos_top_with_grad", "cos_top_with_saved_direction", "left_cos_top_u_with_residual"),
    ):
        for key, value in row.items():
            if key in {"method", "n"}:
                continue
            rows.append(
                {
                    "run": run_label,
                    "metric_group": "spectrum_by_method",
                    "method": row["method"],
                    "n": row["n"],
                    "metric": key,
                    "value": value,
                    "source": rel(run_dir / "spectrum_path.csv"),
                }
            )

    surrogate = read_csv(run_dir / "quadratic_surrogate_aggregate.csv")
    for item in surrogate:
        direction = item.get("direction", "")
        radius = item.get("radius_fraction", "")
        if direction not in {"top_singular", "saved_direction", "grad"}:
            continue
        if radius not in {"0.1", "0.3", "1.0"}:
            continue
        for metric in ("true_gain_mean", "predicted_gain_mean", "relative_abs_error_mean", "residual_relative_error_mean"):
            rows.append(
                {
                    "run": run_label,
                    "metric_group": "surrogate",
                    "method": item.get("method", ""),
                    "direction": direction,
                    "radius_fraction": radius,
                    "metric": metric,
                    "value": item.get(metric, ""),
                    "source": rel(run_dir / "quadratic_surrogate_aggregate.csv"),
                }
            )

    candidates = read_csv(run_dir / "candidate_comparison_aggregate.csv")
    for item in candidates:
        candidate = item.get("candidate", "")
        if candidate not in {"add_top_singular", "replace_top_singular", "add_saved_direction", "replace_saved_direction", "add_grad", "replace_grad"}:
            continue
        for metric in ("true_gain_mean", "predicted_gain_mean", "relative_abs_error_mean", "cos_step_with_grad_mean", "cos_step_with_top_mean"):
            rows.append(
                {
                    "run": run_label,
                    "metric_group": "candidate",
                    "method": item.get("method", ""),
                    "candidate": candidate,
                    "metric": metric,
                    "value": item.get(metric, ""),
                    "source": rel(run_dir / "candidate_comparison_aggregate.csv"),
                }
            )
    return rows, notes


def write_doc(path: Path, metrics_path: Path, rows: list[dict[str, Any]], notes: list[str], run_dirs: dict[str, Path]) -> None:
    complete = [label for label, directory in run_dirs.items() if (directory / "manifest.json").exists()]
    lines = [
        "# NS2D JVP/VJP Integration Summary - 2026-06-23",
        "",
        "This file connects the new NS2D matrix-free Jacobian probes to the existing Loss3 mechanism story.",
        "",
        "## Status",
        "",
        f"- Completed probe groups: {', '.join(complete) if complete else 'none yet'}",
        f"- Key metrics CSV: `{rel(metrics_path)}`",
    ]
    if notes:
        lines.extend(["", "## Pending / Notes", ""])
        lines.extend(notes)
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Replacement has a power-iteration-like explanation only if the saved replacement direction aligns with the local top singular direction and the local residual-linear surrogate remains accurate at large radius.",
            "- Add is favored when the top direction is unstable, the full-budget surrogate error is large, or replace candidates have lower true gain than add candidates despite comparable local predicted gain.",
            "- If `jvp_mode_used` is finite-difference-heavy, report the result as a matrix-free finite-difference JVP plus exact VJP probe, not as a pure forward-mode autograd JVP.",
            "",
            "## Files To Read After Completion",
            "",
        ]
    )
    for label, directory in run_dirs.items():
        lines.append(f"- {label}: `{rel(directory)}`")
    lines.extend(
        [
            "",
            "The final cross-system explanation should combine these metrics with the existing Burgers landscape/ridge evidence, Darcy flip-set evidence, and NS2D exact first-order/linearity/boundary-arc evidence.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation")
    parser.add_argument("--out-csv", type=Path, default=PROJECT_ROOT / "analysis_outputs/mechanism_20260622/full_mechanism_validation/cross_system_summary/ns2d_jvp_vjp_key_metrics.csv")
    parser.add_argument("--doc", type=Path, default=PROJECT_ROOT / "docs/loss3_ns2d_jvp_vjp_integration_summary_20260623.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_root = args.full_root if args.full_root.is_absolute() else PROJECT_ROOT / args.full_root
    run_dirs = {
        "N3": full_root / "ns2d_jvp_vjp_spectrum_N3",
        "N2_topup": full_root / "ns2d_jvp_vjp_spectrum_N2_topup",
    }
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for label, directory in run_dirs.items():
        part_rows, part_notes = collect_run(label, directory)
        rows.extend(part_rows)
        notes.extend(part_notes)
    out_csv = args.out_csv if args.out_csv.is_absolute() else PROJECT_ROOT / args.out_csv
    doc = args.doc if args.doc.is_absolute() else PROJECT_ROOT / args.doc
    write_csv(out_csv, rows)
    write_doc(doc, out_csv, rows, notes, run_dirs)
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(out_csv), "doc": str(doc)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
