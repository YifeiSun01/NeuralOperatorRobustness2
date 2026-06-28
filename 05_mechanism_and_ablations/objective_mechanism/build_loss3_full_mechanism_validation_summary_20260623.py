#!/usr/bin/env python3
"""Build a compact cross-system mechanism summary for Loss3 replace/add."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "analysis_outputs/mechanism_20260622"
FULL = ROOT / "full_mechanism_validation"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fnum(x: Any) -> float:
    try:
        out = float(x)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_completed(path: Path) -> bool:
    payload = read_json(path)
    return payload.get("status") == "completed"


def prefer_completed(primary: Path, fallback: Path) -> Path:
    if manifest_completed(primary / "manifest.json"):
        return primary
    return fallback


def pick(rows: list[dict[str, str]], **where: str) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(k)) == str(v) for k, v in where.items()):
            return row
    return {}


def fmt(x: Any, digits: int = 4) -> str:
    value = fnum(x)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


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


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    burgers_first_dir = prefer_completed(
        FULL / "burgers_first_order_prediction_N20",
        ROOT / "replace_add_validation/diagnostics/burgers_first_order_prediction",
    )
    burgers_landscape_dir = prefer_completed(
        FULL / "burgers_landscape_ridge_probe_N20",
        ROOT / "replace_add_validation/diagnostics/burgers_landscape_ridge_probe",
    )
    burgers_is_n20 = burgers_first_dir.name.endswith("_N20") and burgers_landscape_dir.name.endswith("_N20")
    burgers_first = read_csv(burgers_first_dir / "first_order_prediction_by_candidate.csv")
    burgers_arc = read_csv(burgers_landscape_dir / "tables/boundary_arc_aggregate.csv")
    b_add = pick(burgers_first, candidate="add_from_saved_direction")
    b_rep = pick(burgers_first, candidate="replace_from_saved_direction")
    b_arc = [r for r in burgers_arc if r.get("pair") == "steepest_replace__steepest_add"]
    b_arc_vals = [fnum(r.get("loss3_q_mean")) for r in b_arc]
    rows.append(
        {
            "system": "Burgers 1D",
            "evidence_n": "first-order N=20; landscape N=20; formal curves N=100" if burgers_is_n20 else "first-order N=5; landscape N=5; formal curves N=100",
            "replace_speed_observation": "fast boundary/high early loss",
            "first_order_add_rel_error_mean": fnum(b_add.get("relative_abs_error_mean")),
            "first_order_replace_rel_error_mean": fnum(b_rep.get("relative_abs_error_mean")),
            "ridge_min_steepest_replace_to_add": min(b_arc_vals) if b_arc_vals else float("nan"),
            "mechanism_read": "replacement works mainly because boundary/high-loss region is forgiving; full-budget first-order prediction is not uniformly accurate",
            "source_first_order": rel(burgers_first_dir),
            "source_landscape": rel(burgers_landscape_dir),
        }
    )

    darcy_speed = read_csv(FULL / "darcy_flipset_mechanism/speed_aggregate.csv")
    darcy_overlap = read_csv(FULL / "darcy_flipset_mechanism/final_pairwise_flip_overlap_aggregate.csv")
    darcy_near = read_csv(FULL / "darcy_flipset_mechanism/near_optimal_method_sets_aggregate.csv")
    d_add = pick(darcy_speed, method="raw_add")
    d_rep = pick(darcy_speed, method="raw_replace")
    d_pair = pick(darcy_overlap, pair="raw_add__raw_replace")
    d_near = darcy_near[0] if darcy_near else {}
    rows.append(
        {
            "system": "Darcy Flow",
            "evidence_n": "formal/flip-set N=20",
            "replace_speed_observation": f"k90 replace {fmt(d_rep.get('k90_mean'))} vs add {fmt(d_add.get('k90_mean'))}",
            "final_replace_loss_mean": fnum(d_rep.get("final_loss3_mean")),
            "final_add_loss_mean": fnum(d_add.get("final_loss3_mean")),
            "final_add_replace_jaccard": fnum(d_pair.get("jaccard_mean")),
            "near95_method_count_mean": fnum(d_near.get("near95_count_among_methods_mean")),
            "mechanism_read": "replacement-like flip selection is much faster and ends higher on average; add and replace often choose different final flip sets",
        }
    )

    ns_first = read_csv(FULL / "ns2d_exact_mechanism/first_order_by_candidate.csv")
    ns_line = read_csv(FULL / "ns2d_exact_mechanism/linearity_aggregate.csv")
    ns_arc = read_csv(FULL / "ns2d_exact_mechanism/boundary_arc_aggregate.csv")
    ns_direction = read_csv(ROOT / "replace_add_validation/diagnostics/ns2d_direction_stability/step_sample_direction_summary.csv")
    n_add = pick(ns_first, candidate="add_from_saved_direction")
    n_rep = pick(ns_first, candidate="replace_from_saved_direction")
    line_rep = [fnum(r.get("scalar_linearity_c_phi_mean")) for r in ns_line if r.get("method") == "steepest_replace"]
    arc_vals = [fnum(r.get("loss3_mean")) for r in ns_arc if r.get("pair") == "steepest_replace__steepest_add"]
    rep_post = [fnum(r.get("post_boundary_true_loss_gain")) for r in ns_direction if r.get("method") in {"raw_replace", "steepest_replace"}]
    add_post = [fnum(r.get("post_boundary_true_loss_gain")) for r in ns_direction if r.get("method") == "steepest_add"]
    rows.append(
        {
            "system": "NS2D",
            "evidence_n": "direction N=3; exact probe N=3",
            "replace_speed_observation": "replacement reaches boundary immediately but loses after boundary in trace",
            "first_order_add_rel_error_mean": fnum(n_add.get("relative_abs_error_mean")),
            "first_order_replace_rel_error_mean": fnum(n_rep.get("relative_abs_error_mean")),
            "linearity_c_phi_steepest_replace_mean": sum(line_rep) / len(line_rep) if line_rep else float("nan"),
            "arc_min_steepest_replace_to_add": min(arc_vals) if arc_vals else float("nan"),
            "replace_post_boundary_gain_mean": sum(rep_post) / len(rep_post) if rep_post else float("nan"),
            "steepest_add_post_boundary_gain_mean": sum(add_post) / len(add_post) if add_post else float("nan"),
            "mechanism_read": "if exact probe confirms poorer full-budget prediction/linearity or narrow arcs, this supports path-dependent add advantage",
        }
    )

    ns_topup_dir = FULL / "ns2d_exact_mechanism_N2_topup"
    if manifest_completed(ns_topup_dir / "manifest.json"):
        ns_topup_first = read_csv(ns_topup_dir / "first_order_by_candidate.csv")
        ns_topup_line = read_csv(ns_topup_dir / "linearity_aggregate.csv")
        ns_topup_arc = read_csv(ns_topup_dir / "boundary_arc_aggregate.csv")
        ns_topup_direction = read_csv(FULL / "ns2d_direction_stability_N2_topup/step_sample_direction_summary.csv")
        nt_add = pick(ns_topup_first, candidate="add_from_saved_direction")
        nt_rep = pick(ns_topup_first, candidate="replace_from_saved_direction")
        nt_line_rep = [fnum(r.get("scalar_linearity_c_phi_mean")) for r in ns_topup_line if r.get("method") == "steepest_replace"]
        nt_arc_vals = [fnum(r.get("loss3_mean")) for r in ns_topup_arc if r.get("pair") == "steepest_replace__steepest_add"]
        nt_rep_post = [fnum(r.get("post_boundary_true_loss_gain")) for r in ns_topup_direction if r.get("method") in {"raw_replace", "steepest_replace"}]
        nt_add_post = [fnum(r.get("post_boundary_true_loss_gain")) for r in ns_topup_direction if r.get("method") == "steepest_add"]
        rows.append(
            {
                "system": "NS2D top-up",
                "evidence_n": "direction N=2; exact probe N=2; dataset indices 3,4",
                "replace_speed_observation": "top-up repeat of exact NS mechanism probes on held-out additional indices",
                "first_order_add_rel_error_mean": fnum(nt_add.get("relative_abs_error_mean")),
                "first_order_replace_rel_error_mean": fnum(nt_rep.get("relative_abs_error_mean")),
                "linearity_c_phi_steepest_replace_mean": sum(nt_line_rep) / len(nt_line_rep) if nt_line_rep else float("nan"),
                "arc_min_steepest_replace_to_add": min(nt_arc_vals) if nt_arc_vals else float("nan"),
                "replace_post_boundary_gain_mean": sum(nt_rep_post) / len(nt_rep_post) if nt_rep_post else float("nan"),
                "steepest_add_post_boundary_gain_mean": sum(nt_add_post) / len(nt_add_post) if nt_add_post else float("nan"),
                "mechanism_read": "additional NS samples check whether the N=3 exact-mechanism pattern repeats",
                "source_exact": rel(ns_topup_dir),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=FULL / "cross_system_summary")
    parser.add_argument("--doc", type=Path, default=PROJECT_ROOT / "docs/loss3_full_mechanism_validation_summary_20260623.md")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    doc = args.doc if args.doc.is_absolute() else PROJECT_ROOT / args.doc
    rows = build_rows()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "mechanism_summary_rows.csv", rows)

    lines = [
        "# Loss3 Full Mechanism Validation Summary - 2026-06-23",
        "",
        "This note summarizes the current replace/add mechanism evidence across Burgers, Darcy Flow, and NS2D.",
        "",
        "## Artifacts",
        "",
        f"- Burgers validation: `{rel(ROOT / 'replace_add_validation')}`",
        f"- Burgers extended validation when present: `{rel(FULL / 'burgers_first_order_prediction_N20')}`, `{rel(FULL / 'burgers_landscape_ridge_probe_N20')}`",
        f"- Darcy flip-set validation: `{rel(FULL / 'darcy_flipset_mechanism')}`",
        f"- NS2D exact validation: `{rel(FULL / 'ns2d_exact_mechanism')}`",
        f"- NS2D top-up exact validation when present: `{rel(FULL / 'ns2d_exact_mechanism_N2_topup')}`",
        f"- Summary CSV: `{rel(out_dir / 'mechanism_summary_rows.csv')}`",
        "",
        "## Summary Table",
        "",
        "| System | Evidence | Replace/Add Observation | Mechanism Read |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['evidence_n']} | {row['replace_speed_observation']} | {row['mechanism_read']} |"
        )
    lines.extend(
        [
            "",
            "## Compact Numeric Rows",
            "",
        ]
    )
    for row in rows:
        lines.append(f"### {row['system']}")
        for key, value in row.items():
            if key == "system":
                continue
            lines.append(f"- `{key}`: {fmt(value) if isinstance(value, (int, float)) else value}")
        lines.append("")
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "completed", "doc": str(doc), "csv": str(out_dir / "mechanism_summary_rows.csv")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
