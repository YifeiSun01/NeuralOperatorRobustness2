#!/usr/bin/env python3
"""Draw a dependency-free SVG for Lmax-normalized endpoint cap volume."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "analysis_outputs/mechanism_20260622/full_mechanism_validation/"
            "boundary_volume_probe_20260623/tables/endpoint_cap_volume_lmax_summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis_outputs/mechanism_20260622/full_mechanism_validation/"
            "boundary_volume_probe_20260623/figures/endpoint_cap_lmax_p095_by_theta.svg"
        ),
    )
    args = parser.parse_args()

    rows = read_rows(args.summary)
    width = 920
    height = 560
    left = 90
    right = 260
    top = 55
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = sorted({float(r["theta"]) for r in rows})
    xmax = max(xs)

    colors = {
        ("burgers1d", "steepest_add"): "#2563eb",
        ("burgers1d", "steepest_replace"): "#16a34a",
        ("ns2d", "steepest_add"): "#dc2626",
        ("ns2d", "steepest_replace"): "#9333ea",
    }
    labels = {
        ("burgers1d", "steepest_add"): "Burgers steepest_add",
        ("burgers1d", "steepest_replace"): "Burgers steepest_replace",
        ("ns2d", "steepest_add"): "NS2D steepest_add",
        ("ns2d", "steepest_replace"): "NS2D steepest_replace",
    }

    def sx(theta: float) -> float:
        return left + (theta / xmax) * plot_w

    def sy(prob: float) -> float:
        return top + (1.0 - prob) * plot_h

    series: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if abs(float(row["tau"]) - 0.95) > 1e-9:
            continue
        key = (row["system"], row["endpoint_method"])
        series[key].append((float(row["theta"]), float(row["p_high_lmax"])))

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append('<text x="90" y="32" font-family="Arial, sans-serif" font-size="22" font-weight="700">Endpoint cap volume normalized by same-sample Lmax</text>')
    svg.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>')

    for y in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = sy(y)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(f'<text x="{left - 12}" y="{yy + 5:.2f}" font-family="Arial, sans-serif" font-size="13" text-anchor="end" fill="#374151">{y:.2f}</text>')
    for x in xs:
        xx = sx(x)
        svg.append(f'<line x1="{xx:.2f}" y1="{top + plot_h}" x2="{xx:.2f}" y2="{top + plot_h + 6}" stroke="#111827" stroke-width="1"/>')
        svg.append(f'<text x="{xx:.2f}" y="{top + plot_h + 25}" font-family="Arial, sans-serif" font-size="13" text-anchor="middle" fill="#374151">{x:g}</text>')

    for key in sorted(series):
        pts = sorted(series[key])
        color = colors.get(key, "#111827")
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in pts)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:
            svg.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="{color}"/>')

    svg.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 22}" font-family="Arial, sans-serif" font-size="15" text-anchor="middle" fill="#111827">cap angular radius theta</text>')
    svg.append(f'<text x="22" y="{top + plot_h / 2:.2f}" font-family="Arial, sans-serif" font-size="15" text-anchor="middle" fill="#111827" transform="rotate(-90 22 {top + plot_h / 2:.2f})">p(loss >= 0.95 Lmax)</text>')

    lx = left + plot_w + 34
    ly = top + 20
    svg.append(f'<text x="{lx}" y="{ly}" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">Legend</text>')
    for i, key in enumerate(sorted(series)):
        y = ly + 30 + 28 * i
        color = colors.get(key, "#111827")
        svg.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 34}" y2="{y}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        svg.append(f'<circle cx="{lx + 17}" cy="{y}" r="4" fill="{color}"/>')
        svg.append(f'<text x="{lx + 46}" y="{y + 5}" font-family="Arial, sans-serif" font-size="14" fill="#111827">{labels.get(key, key[0] + " " + key[1])}</text>')

    svg.append('</svg>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
