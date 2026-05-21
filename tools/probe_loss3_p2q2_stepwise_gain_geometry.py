#!/usr/bin/env python3
"""Stepwise p2q2 geometry-vs-next-loss-gain probe for Loss 3.

This is post-processing only. It reads existing p2q2 300-step
per_sample_step_metrics.csv files and asks, at each sample and step k:

- how large is the tangent projection of grad L relative to delta?
- how large is the radial/normal component?
- how tangent/radial is the proposed update direction?
- how large is the local-linear predicted gain of the actual proposal?
- does any of that correlate with L_{k+1} - L_k?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_sweep_p2q2_300steps_20260520"
DEFAULT_OUT = PROJECT_ROOT / "forensics" / "loss3_p2q2_stepwise_gain_geometry_probe_20260520"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "loss3_p2q2_stepwise_gain_geometry_probe_20260520.md"
CORE4 = ("raw_add", "raw_replace", "steepest_add", "steepest_replace")
ADDITIVE = {"raw_add", "steepest_add"}
REPLACEMENT = {"raw_replace", "steepest_replace"}
METRICS = (
    "grad_tangent_ratio",
    "grad_radial_signed_ratio",
    "grad_radial_abs_ratio",
    "grad_tangent_norm",
    "grad_radial_signed_norm",
    "grad_radial_abs_norm",
    "direction_tangent_ratio",
    "direction_radial_signed_ratio",
    "direction_radial_abs_ratio",
    "direction_l2",
    "linear_gain_unprojected",
    "linear_gain_projected",
    "actual_delta_angle_next",
    "actual_step_l2_over_eps_next",
)


def fnum(x: Any) -> float:
    if x in {None, "", "nan", "NaN"}:
        return math.nan
    try:
        return float(x)
    except Exception:
        return math.nan


def finite(x: float) -> bool:
    return math.isfinite(float(x))


def clamp_cos(c: float) -> float:
    return max(-1.0, min(1.0, c))


def tangent_ratio(cosine: float) -> float:
    if not finite(cosine):
        return math.nan
    c = clamp_cos(cosine)
    return math.sqrt(max(0.0, 1.0 - c * c))


@dataclass
class OnlineStats:
    n: int = 0
    sx: float = 0.0
    sx2: float = 0.0

    def add(self, x: float) -> None:
        if not finite(x):
            return
        self.n += 1
        self.sx += float(x)
        self.sx2 += float(x) * float(x)

    @property
    def mean(self) -> float:
        return self.sx / self.n if self.n else math.nan

    @property
    def std(self) -> float:
        if not self.n:
            return math.nan
        var = max(0.0, self.sx2 / self.n - self.mean * self.mean)
        return math.sqrt(var)


@dataclass
class OnlineCorr:
    n: int = 0
    sx: float = 0.0
    sy: float = 0.0
    sx2: float = 0.0
    sy2: float = 0.0
    sxy: float = 0.0

    def add(self, x: float, y: float) -> None:
        if not (finite(x) and finite(y)):
            return
        self.n += 1
        self.sx += x
        self.sy += y
        self.sx2 += x * x
        self.sy2 += y * y
        self.sxy += x * y

    @property
    def r(self) -> float:
        if self.n < 3:
            return math.nan
        num = self.n * self.sxy - self.sx * self.sy
        vx = self.n * self.sx2 - self.sx * self.sx
        vy = self.n * self.sy2 - self.sy * self.sy
        den = math.sqrt(max(0.0, vx) * max(0.0, vy))
        return num / den if den > 1e-30 else math.nan


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_roots(sweep: Path) -> list[Path]:
    return sorted([p.parent for p in sweep.glob("*/per_sample_step_metrics.csv")], key=lambda p: p.name)


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int], list[dict[str, str]]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = row.get("method", "")
        if method not in CORE4:
            continue
        sample = int(float(row.get("sample_position", 0)))
        groups[(method, sample)].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: int(float(r.get("k", 0))))
    return groups


def scope_for(boundary_ratio: float) -> list[str]:
    scopes = ["all"]
    if finite(boundary_ratio):
        if boundary_ratio >= 0.99:
            scopes.append("post_boundary_099")
        elif boundary_ratio < 0.99:
            scopes.append("pre_boundary_099")
        if boundary_ratio >= 0.95:
            scopes.append("near_or_post_boundary_095")
    return scopes


def step_metrics(row: dict[str, str], next_row: dict[str, str], method: str) -> dict[str, float]:
    epsilon = fnum(row.get("epsilon"))
    alpha = fnum(row.get("alpha"))
    delta_l2 = fnum(row.get("delta_l2"))
    grad_l2 = fnum(row.get("grad_l2"))
    direction_l2 = fnum(row.get("direction_l2"))
    projection_shrink = fnum(row.get("projection_shrink_factor"))
    cos_delta_grad = fnum(row.get("cos_delta_grad"))
    cos_delta_direction = fnum(row.get("cos_delta_direction"))
    cos_grad_direction = fnum(row.get("cos_grad_direction"))
    loss = fnum(row.get("loss3_q"))
    next_loss = fnum(next_row.get("loss3_q"))
    next_gain = next_loss - loss if finite(loss) and finite(next_loss) else math.nan

    grad_tan = tangent_ratio(cos_delta_grad)
    dir_tan = tangent_ratio(cos_delta_direction)
    dot_g_delta = delta_l2 * grad_l2 * cos_delta_grad if all(finite(v) for v in [delta_l2, grad_l2, cos_delta_grad]) else math.nan
    dot_g_dir = grad_l2 * direction_l2 * cos_grad_direction if all(finite(v) for v in [grad_l2, direction_l2, cos_grad_direction]) else math.nan

    linear_unproj = math.nan
    linear_proj = math.nan
    if method in ADDITIVE:
        if finite(alpha) and finite(dot_g_dir):
            linear_unproj = alpha * dot_g_dir
        if all(finite(v) for v in [projection_shrink, alpha, dot_g_dir, dot_g_delta]):
            # p=2 projection is scalar shrink of delta + alpha d.
            linear_proj = (projection_shrink - 1.0) * dot_g_delta + projection_shrink * alpha * dot_g_dir
    elif method in REPLACEMENT:
        if finite(epsilon) and finite(dot_g_dir) and finite(dot_g_delta):
            linear_unproj = epsilon * dot_g_dir - dot_g_delta
        if all(finite(v) for v in [projection_shrink, epsilon, dot_g_dir, dot_g_delta]):
            linear_proj = projection_shrink * epsilon * dot_g_dir - dot_g_delta

    return {
        "epsilon": epsilon,
        "alpha": alpha,
        "k": fnum(row.get("k")),
        "boundary_ratio": fnum(row.get("boundary_ratio")),
        "next_loss_gain": next_gain,
        "grad_tangent_ratio": grad_tan,
        "grad_radial_signed_ratio": cos_delta_grad,
        "grad_radial_abs_ratio": abs(cos_delta_grad) if finite(cos_delta_grad) else math.nan,
        "grad_tangent_norm": grad_l2 * grad_tan if finite(grad_l2) and finite(grad_tan) else math.nan,
        "grad_radial_signed_norm": grad_l2 * cos_delta_grad if finite(grad_l2) and finite(cos_delta_grad) else math.nan,
        "grad_radial_abs_norm": grad_l2 * abs(cos_delta_grad) if finite(grad_l2) and finite(cos_delta_grad) else math.nan,
        "direction_tangent_ratio": dir_tan,
        "direction_radial_signed_ratio": cos_delta_direction,
        "direction_radial_abs_ratio": abs(cos_delta_direction) if finite(cos_delta_direction) else math.nan,
        "direction_l2": direction_l2,
        "linear_gain_unprojected": linear_unproj,
        "linear_gain_projected": linear_proj,
        "actual_delta_angle_next": fnum(next_row.get("delta_prev_angle_degrees")),
        "actual_step_l2_over_eps_next": fnum(next_row.get("delta_step_l2_over_epsilon")),
    }


def analyze(sweep: Path, out_dir: Path, sample_limit_per_method_scope: int, seed: int) -> dict[str, Any]:
    roots = load_roots(sweep)
    step_aggs: dict[tuple[str, float, float, str, int], dict[str, OnlineStats]] = defaultdict(lambda: defaultdict(OnlineStats))
    corrs: dict[tuple[str, str, str], OnlineCorr] = defaultdict(OnlineCorr)
    setting_corrs: dict[tuple[str, float, float, str, str, str], OnlineCorr] = defaultdict(OnlineCorr)
    scatter: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    rng = random.Random(seed)
    total_pairs = 0

    for root in roots:
        rows = read_csv(root / "per_sample_step_metrics.csv")
        groups = group_rows(rows)
        for (method, _sample), group in groups.items():
            for i in range(len(group) - 1):
                m = step_metrics(group[i], group[i + 1], method)
                k = int(m["k"]) if finite(m["k"]) else -1
                eps = m["epsilon"]
                alpha = m["alpha"]
                total_pairs += 1
                key = (root.name, eps, alpha, method, k)
                for name in ("next_loss_gain", "boundary_ratio", *METRICS):
                    step_aggs[key][name].add(m.get(name, math.nan))
                for scope in scope_for(m["boundary_ratio"]):
                    for metric in METRICS:
                        x = m.get(metric, math.nan)
                        y = m.get("next_loss_gain", math.nan)
                        corrs[(method, scope, metric)].add(x, y)
                        setting_corrs[(root.name, eps, alpha, method, scope, metric)].add(x, y)
                        # Keep bounded scatter samples for key plots only.
                        if scope == "post_boundary_099" and metric in {"grad_tangent_ratio", "grad_radial_signed_ratio", "direction_tangent_ratio", "linear_gain_projected"}:
                            skey = (method, scope, metric)
                            if finite(x) and finite(y):
                                buf = scatter[skey]
                                if len(buf) < sample_limit_per_method_scope:
                                    buf.append((x, y))
                                else:
                                    j = rng.randint(0, total_pairs - 1)
                                    if j < sample_limit_per_method_scope:
                                        buf[j] = (x, y)

    step_rows: list[dict[str, Any]] = []
    for (root_name, eps, alpha, method, k), stats in sorted(step_aggs.items(), key=lambda item: (item[0][0], item[0][3], item[0][4])):
        row: dict[str, Any] = {"setting_root": root_name, "epsilon": eps, "alpha": alpha, "method": method, "k": k}
        for name, st in stats.items():
            row[f"{name}_mean"] = st.mean
            row[f"{name}_std"] = st.std
            row[f"{name}_count"] = st.n
        step_rows.append(row)

    corr_rows: list[dict[str, Any]] = []
    for (method, scope, metric), c in sorted(corrs.items()):
        corr_rows.append({"method": method, "scope": scope, "metric": metric, "pearson_r_with_next_loss_gain": c.r, "pair_count": c.n})

    setting_corr_rows: list[dict[str, Any]] = []
    for (root_name, eps, alpha, method, scope, metric), c in sorted(setting_corrs.items()):
        setting_corr_rows.append({"setting_root": root_name, "epsilon": eps, "alpha": alpha, "method": method, "scope": scope, "metric": metric, "pearson_r_with_next_loss_gain": c.r, "pair_count": c.n})

    tables = out_dir / "tables"
    write_csv(tables / "stepwise_geometry_mean_by_setting_method_step.csv", step_rows)
    write_csv(tables / "stepwise_geometry_correlations_by_method_scope.csv", corr_rows)
    write_csv(tables / "stepwise_geometry_correlations_by_setting_method_scope.csv", setting_corr_rows)

    scatter_dir = out_dir / "scatter_samples"
    scatter_rows = []
    for (method, scope, metric), points in scatter.items():
        for x, y in points:
            scatter_rows.append({"method": method, "scope": scope, "metric": metric, "x": x, "next_loss_gain": y})
    write_csv(scatter_dir / "post_boundary_scatter_sample.csv", scatter_rows)

    write_figures(out_dir, scatter, corr_rows)
    return {
        "setting_root_count": len(roots),
        "sample_step_pair_count": total_pairs,
        "step_row_count": len(step_rows),
        "correlation_row_count": len(corr_rows),
        "setting_correlation_row_count": len(setting_corr_rows),
        "scatter_sample_row_count": len(scatter_rows),
    }


def write_figures(out_dir: Path, scatter: dict[tuple[str, str, str], list[tuple[float, float]]], corr_rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("grad_tangent_ratio", "direction_tangent_ratio", "linear_gain_projected"):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=False)
        axes = axes.ravel()
        for ax, method in zip(axes, CORE4):
            pts = scatter.get((method, "post_boundary_099", metric), [])
            if pts:
                x = np.asarray([p[0] for p in pts], dtype=float)
                y = np.asarray([p[1] for p in pts], dtype=float)
                ax.hexbin(x, y, gridsize=45, mincnt=1, cmap="viridis")
            ax.set_title(method)
            ax.set_xlabel(metric)
            ax.set_ylabel("next loss gain")
            ax.grid(True, alpha=0.2)
        fig.suptitle(f"Post-boundary stepwise {metric} vs next loss gain")
        fig.tight_layout()
        fig.savefig(fig_dir / f"post_boundary_{metric}_vs_next_loss_gain.png", dpi=170)
        plt.close(fig)

    # Compact bar plot of correlations for the most relevant post-boundary metrics.
    wanted = ["grad_tangent_ratio", "grad_radial_signed_ratio", "direction_tangent_ratio", "linear_gain_projected", "actual_delta_angle_next"]
    rows = [r for r in corr_rows if r["scope"] == "post_boundary_099" and r["metric"] in wanted]
    if rows:
        methods = list(CORE4)
        x = np.arange(len(wanted))
        width = 0.18
        fig, ax = plt.subplots(figsize=(11, 4.8))
        for i, method in enumerate(methods):
            vals = []
            for metric in wanted:
                hit = next((r for r in rows if r["method"] == method and r["metric"] == metric), None)
                vals.append(float(hit["pearson_r_with_next_loss_gain"]) if hit and finite(fnum(hit["pearson_r_with_next_loss_gain"])) else np.nan)
            ax.bar(x + (i - 1.5) * width, vals, width, label=method)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(wanted, rotation=25, ha="right")
        ax.set_ylabel("Pearson r with next-step loss gain")
        ax.set_title("Post-boundary stepwise geometry correlations")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "post_boundary_stepwise_correlation_bars.png", dpi=170)
        plt.close(fig)


def md_table(rows: list[dict[str, Any]], fields: list[str], max_rows: int | None = None) -> str:
    show = rows[:max_rows] if max_rows is not None else rows
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in show:
        vals = []
        for field in fields:
            value = row.get(field, "")
            fv = fnum(value)
            vals.append(f"{fv:.4g}" if finite(fv) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_doc(doc: Path, out_dir: Path, manifest: dict[str, Any]) -> None:
    corr = read_csv(out_dir / "tables" / "stepwise_geometry_correlations_by_method_scope.csv")
    post = [r for r in corr if r["scope"] == "post_boundary_099" and r["metric"] in {"grad_tangent_ratio", "grad_radial_signed_ratio", "direction_tangent_ratio", "linear_gain_projected", "actual_delta_angle_next"}]
    post.sort(key=lambda r: (r["method"], r["metric"]))
    lines = [
        "# Loss 3 p2q2 Stepwise Geometry vs Next-Step Gain Probe - 2026-05-20",
        "",
        "Status: generated from existing p2q2 300-step per-sample metrics; no optimizer/model experiment was rerun.",
        "",
        "## What This Tests",
        "",
        "For each sample/method/step k, this probe compares geometry at step k with the immediate next loss increment:",
        "",
        "```text",
        "next_loss_gain = loss3_q[k+1] - loss3_q[k]",
        "```",
        "",
        "The goal is to test the user's finer question: does a larger tangent or radial component at a specific step predict a faster loss increase at the next step?",
        "",
        "## Tables and Figures",
        "",
        f"- Stepwise means: `{rel(out_dir / 'tables' / 'stepwise_geometry_mean_by_setting_method_step.csv')}`",
        f"- Method/scope correlations: `{rel(out_dir / 'tables' / 'stepwise_geometry_correlations_by_method_scope.csv')}`",
        f"- Setting-level correlations: `{rel(out_dir / 'tables' / 'stepwise_geometry_correlations_by_setting_method_scope.csv')}`",
        f"- Scatter sample: `{rel(out_dir / 'scatter_samples' / 'post_boundary_scatter_sample.csv')}`",
        f"- Figures: `{rel(out_dir / 'figures')}`",
        "",
        "## Post-Boundary Correlations With Next-Step Loss Gain",
        "",
        md_table(post, ["method", "metric", "pearson_r_with_next_loss_gain", "pair_count"]),
        "",
        "## Interpretation Guide",
        "",
        "- `grad_tangent_ratio`: normalized size of `(I - u u^T) grad L`. It measures available tangent gradient, not whether the chosen update uses it correctly.",
        "- `grad_radial_signed_ratio`: radial/normal alignment of the gradient with delta. Near 1 means mostly radial outward; near 0 means mostly tangent.",
        "- `direction_tangent_ratio`: how tangent the method's proposed direction is relative to current delta.",
        "- `linear_gain_projected`: local-linear prediction of the actual projected proposal's immediate gain, using recorded cosines and projection shrink. This is the most direct one-step predictor among these metrics.",
        "- `actual_delta_angle_next`: the actual angle between delta[k] and delta[k+1].",
        "",
        "Important caveat: a large tangent gradient is opportunity. It does not by itself guarantee the method's next proposal moves in the useful tangent direction. The one-step linear gain is the better test of whether the chosen step is aligned with the loss.",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--sample-limit-per-method-scope", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep = args.sweep_root if args.sweep_root.is_absolute() else PROJECT_ROOT / args.sweep_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    doc = args.doc if args.doc.is_absolute() else PROJECT_ROOT / args.doc
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze(sweep, out_dir, args.sample_limit_per_method_scope, args.seed)
    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_root": str(sweep),
        "out_dir": str(out_dir),
        "doc": str(doc),
        **summary,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_doc(doc, out_dir, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
