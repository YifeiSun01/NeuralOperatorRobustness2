#!/usr/bin/env python3
"""Re-score endpoint cap volume by same-sample Lmax.

The first boundary-volume probe measured endpoint caps relative to each endpoint's
own loss. That can make a low-loss NS replace endpoint look locally broad. This
post-process instead asks whether cap samples are close to the best observed loss
for the same system/sample:

    ratio_to_lmax = L(delta_cap) / Lmax(sample)

where Lmax is the best optimizer endpoint already recorded in
global_boundary_volume.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


TAUS = (0.90, 0.95, 0.99)


def _float_key(value: str) -> str:
    return f"{float(value):.12g}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_probability(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    ratios: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["system"]),
            str(row["endpoint_method"]),
            _float_key(str(row["theta"])),
            _float_key(str(row["tau"])),
        )
        groups[key].append(float(row["is_high_lmax"]))
        ratios[key].append(float(row["ratio_to_lmax"]))

    out = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], float(k[2]), float(k[3]))):
        system, endpoint_method, theta, tau = key
        vals = groups[key]
        rs = ratios[key]
        out.append(
            {
                "system": system,
                "endpoint_method": endpoint_method,
                "theta": float(theta),
                "tau": float(tau),
                "n": len(vals),
                "p_high_lmax": mean(vals),
                "ratio_to_lmax_mean": mean(rs),
                "ratio_to_lmax_min": min(rs),
                "ratio_to_lmax_max": max(rs),
            }
        )
    return out


def aggregate_width(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        key = (str(row["system"]), str(row["endpoint_method"]), _float_key(str(row["tau"])))
        groups[key].append(row)

    out = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], float(k[2]))):
        system, endpoint_method, tau = key
        pts = sorted(groups[key], key=lambda r: float(r["theta"]))
        auc = 0.0
        last_theta_over_half = None
        for row in pts:
            if float(row["p_high_lmax"]) >= 0.5:
                last_theta_over_half = float(row["theta"])
        for a, b in zip(pts, pts[1:]):
            x0 = float(a["theta"])
            x1 = float(b["theta"])
            y0 = float(a["p_high_lmax"])
            y1 = float(b["p_high_lmax"])
            auc += 0.5 * (y0 + y1) * (x1 - x0)
        out.append(
            {
                "system": system,
                "endpoint_method": endpoint_method,
                "tau": float(tau),
                "theta50_lmax": "" if last_theta_over_half is None else last_theta_over_half,
                "auc_lmax": auc,
                "theta_values": ";".join(_float_key(str(r["theta"])) for r in pts),
                "p_values": ";".join(_float_key(str(r["p_high_lmax"])) for r in pts),
            }
        )
    return out


def summarize_for_markdown(summary_rows: list[dict[str, object]], width_rows: list[dict[str, object]]) -> str:
    lines = []
    lines.append("# Loss3 boundary cap volume, normalized by same-sample Lmax")
    lines.append("")
    lines.append("This post-process reuses the completed boundary-volume probe and changes the cap criterion from")
    lines.append("")
    lines.append(r"\\[L(\\delta_{cap}) / L(\\delta_{endpoint})\\]")
    lines.append("")
    lines.append("to the stricter same-sample criterion")
    lines.append("")
    lines.append(r"\\[L(\\delta_{cap}) / L_{max}\\]")
    lines.append("")
    lines.append("where `Lmax` is the best observed optimizer endpoint for that same system/sample.")
    lines.append("")
    lines.append("## Key p0.95 table")
    lines.append("")
    lines.append("| system | endpoint | theta | p(loss >= 0.95 Lmax) | mean loss/Lmax | n |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        if abs(float(row["tau"]) - 0.95) > 1e-9:
            continue
        lines.append(
            "| {system} | {endpoint_method} | {theta:.3g} | {p_high_lmax:.3f} | {ratio_to_lmax_mean:.3f} | {n} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("## Width summary")
    lines.append("")
    lines.append("| system | endpoint | tau | theta50 | AUC |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in width_rows:
        lines.append(
            "| {system} | {endpoint_method} | {tau:.2f} | {theta50_lmax} | {auc_lmax:.4f} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If Burgers replace has nonzero/high p0.95 near its endpoint while NS replace is zero, that supports the hypothesis that Burgers replace lands in an optimizer-relevant high-loss region but NS replace does not.")
    lines.append("- This is stricter than endpoint-relative cap volume, because NS replace is judged against the sample's best observed loss, not against its own lower endpoint.")
    lines.append("- The global random boundary result remains a separate statement: high-loss points are rare under completely random boundary sampling for both systems.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "analysis_outputs/mechanism_20260622/full_mechanism_validation/"
            "boundary_volume_probe_20260623"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    tables_dir = output_dir / "tables"

    global_rows = read_rows(input_dir / "tables" / "global_boundary_volume.csv")
    cap_rows = read_rows(input_dir / "tables" / "endpoint_cap_volume.csv")

    lmax_by_sample: dict[tuple[str, str, str], float] = {}
    for row in global_rows:
        key = (row["system"], row["dataset_index"], row["sample_position"])
        lmax = float(row["lmax"])
        prev = lmax_by_sample.get(key)
        if prev is not None and not math.isclose(prev, lmax, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Conflicting Lmax for {key}: {prev} vs {lmax}")
        lmax_by_sample[key] = lmax

    rescored = []
    for row in cap_rows:
        key = (row["system"], row["dataset_index"], row["sample_position"])
        if key not in lmax_by_sample:
            raise KeyError(f"Missing Lmax for cap row key {key}")
        loss = float(row["loss"])
        lmax = lmax_by_sample[key]
        tau = float(row["tau"])
        ratio = loss / lmax if lmax != 0 else float("nan")
        rescored.append(
            {
                **row,
                "lmax": lmax,
                "ratio_to_lmax": ratio,
                "is_high_lmax": 1.0 if ratio >= tau else 0.0,
            }
        )

    fieldnames = list(cap_rows[0].keys()) + ["lmax", "ratio_to_lmax", "is_high_lmax"]
    write_rows(tables_dir / "endpoint_cap_volume_lmax.csv", fieldnames, rescored)

    summary = aggregate_probability(rescored)
    summary_fields = [
        "system",
        "endpoint_method",
        "theta",
        "tau",
        "n",
        "p_high_lmax",
        "ratio_to_lmax_mean",
        "ratio_to_lmax_min",
        "ratio_to_lmax_max",
    ]
    write_rows(tables_dir / "endpoint_cap_volume_lmax_summary.csv", summary_fields, summary)

    width = aggregate_width(summary)
    width_fields = [
        "system",
        "endpoint_method",
        "tau",
        "theta50_lmax",
        "auc_lmax",
        "theta_values",
        "p_values",
    ]
    write_rows(tables_dir / "endpoint_cap_width_lmax_summary.csv", width_fields, width)

    doc = summarize_for_markdown(summary, width)
    (output_dir / "boundary_cap_lmax_interpretation.md").write_text(doc, encoding="utf-8")

    manifest = {
        "status": "completed",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "rows": {
            "global_boundary_volume": len(global_rows),
            "endpoint_cap_volume": len(cap_rows),
            "endpoint_cap_volume_lmax": len(rescored),
            "endpoint_cap_volume_lmax_summary": len(summary),
            "endpoint_cap_width_lmax_summary": len(width),
        },
        "outputs": [
            str(tables_dir / "endpoint_cap_volume_lmax.csv"),
            str(tables_dir / "endpoint_cap_volume_lmax_summary.csv"),
            str(tables_dir / "endpoint_cap_width_lmax_summary.csv"),
            str(output_dir / "boundary_cap_lmax_interpretation.md"),
        ],
    }
    (output_dir / "boundary_cap_lmax_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
