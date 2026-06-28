#!/usr/bin/env python3
"""Analyze final-delta similarity for the loss3 alpha/epsilon core-four sweep.

This is post-processing only. It reads each method's final_deltas.npz under
completed setting roots and does not run the model, solver, or optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_sweep_20260519"
DEFAULT_OUT = PROJECT_ROOT / "forensics" / "loss3_alpha_epsilon_core4_delta_similarity_20260520"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "loss3_alpha_epsilon_core4_delta_similarity_20260520.md"
CORE4 = ("raw_add", "raw_replace", "steepest_add", "steepest_replace")
METHOD_LABELS = {
    "raw_add": "raw add",
    "raw_replace": "raw replace",
    "steepest_add": "steepest add",
    "steepest_replace": "steepest replace",
}


def fnum(value) -> float:
    if value in {None, "", "nan", "NaN"}:
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def finite_stats(values) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "median": math.nan,
            "finite_count": 0,
            "nonfinite_count": int(arr.size),
        }
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "median": float(np.median(finite)),
        "finite_count": int(finite.size),
        "nonfinite_count": int(arr.size - finite.size),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def setting_label(epsilon: float, alpha: float) -> str:
    return f"e{epsilon:g}/a{alpha:g}"


def setting_slug(epsilon: float, alpha: float) -> str:
    def tag(value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    return f"eps{tag(epsilon)}_alpha{tag(alpha)}"


def manifest_completed(root: Path) -> bool:
    path = root / "manifest.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except Exception:
        return False


def config_value(root: Path, key: str, fallback=None):
    for name in ("config.json", "manifest.json"):
        path = root / name
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if key in payload:
                    return payload[key]
            except Exception:
                pass
    return fallback


def root_setting(root: Path) -> tuple[float, float]:
    return float(config_value(root, "epsilon")), float(config_value(root, "alpha"))


def root_pq(root: Path) -> tuple[str, str]:
    p_order = config_value(root, "p_order", config_value(root, "p", "unknown"))
    q_order = config_value(root, "q_order", config_value(root, "q", "unknown"))
    return str(p_order), str(q_order)


def load_completed_roots(sweep_root: Path) -> list[Path]:
    manifest_path = sweep_root / "sweep_manifest.json"
    roots: list[Path] = []
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in payload.get("completed", []):
                root = Path(item.get("out_root", ""))
                if not root.is_absolute():
                    root = PROJECT_ROOT / root
                if manifest_completed(root):
                    roots.append(root)
        except Exception:
            pass
    if not roots:
        roots = [p.parent for p in sweep_root.glob("*/manifest.json") if manifest_completed(p.parent)]
    return sorted(dict.fromkeys(roots), key=lambda r: (*root_setting(r), root_pq(r), r.name))


def load_method_delta(root: Path, method: str) -> dict[int, np.ndarray]:
    path = root / method / "final_deltas.npz"
    if not path.exists():
        return {}
    z = np.load(path)
    dataset_index = z["dataset_index"].astype(int)
    final_delta = z["final_delta"].astype(np.float64).reshape(len(dataset_index), -1)
    return {int(idx): final_delta[pos] for pos, idx in enumerate(dataset_index)}


def cosine_rows(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1e-30 or norm_b <= 1e-30:
        cosine = math.nan
        unit_l2 = math.nan
    else:
        cosine = float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))
        unit_l2 = float(np.linalg.norm(a / norm_a - b / norm_b))

    diff = float(np.linalg.norm(a - b))
    scale = 0.5 * (norm_a + norm_b)
    rel_l2 = diff / scale if scale > 1e-30 else math.nan

    ac = a - np.mean(a)
    bc = b - np.mean(b)
    norm_ac = float(np.linalg.norm(ac))
    norm_bc = float(np.linalg.norm(bc))
    centered = float(np.clip(np.dot(ac, bc) / (norm_ac * norm_bc), -1.0, 1.0)) if norm_ac > 1e-30 and norm_bc > 1e-30 else math.nan

    sa = np.abs(np.fft.rfft(a))
    sb = np.abs(np.fft.rfft(b))
    norm_sa = float(np.linalg.norm(sa))
    norm_sb = float(np.linalg.norm(sb))
    spectral = float(np.clip(np.dot(sa, sb) / (norm_sa * norm_sb), -1.0, 1.0)) if norm_sa > 1e-30 and norm_sb > 1e-30 else math.nan

    nonzero = (np.abs(a) > 1e-12) | (np.abs(b) > 1e-12)
    sign_agree = float(np.mean(np.sign(a[nonzero]) == np.sign(b[nonzero]))) if np.any(nonzero) else math.nan
    return cosine, abs(cosine) if math.isfinite(cosine) else math.nan, centered, spectral, rel_l2, unit_l2, sign_agree


def analyze_root(root: Path) -> tuple[list[dict], list[dict]]:
    epsilon, alpha = root_setting(root)
    p_order, q_order = root_pq(root)
    by_method = {method: load_method_delta(root, method) for method in CORE4}
    sample_ids = sorted(set.intersection(*(set(v) for v in by_method.values() if v))) if all(by_method.values()) else []
    per_sample: list[dict] = []
    summary: list[dict] = []
    for method_a, method_b in combinations(CORE4, 2):
        values = defaultdict(list)
        for sample_id in sample_ids:
            a = by_method[method_a][sample_id]
            b = by_method[method_b][sample_id]
            cosine, abs_cosine, centered, spectral, rel_l2, unit_l2, sign_agree = cosine_rows(a, b)
            norm_a = float(np.linalg.norm(a))
            norm_b = float(np.linalg.norm(b))
            l2_diff = float(np.linalg.norm(a - b))
            row = {
                "source_root": str(root),
                "epsilon": epsilon,
                "alpha": alpha,
                "p_order": p_order,
                "q_order": q_order,
                "dataset_index": sample_id,
                "method_a": method_a,
                "method_b": method_b,
                "cosine": cosine,
                "abs_cosine": abs_cosine,
                "centered_cosine": centered,
                "spectral_magnitude_cosine": spectral,
                "relative_l2_distance": rel_l2,
                "unit_direction_l2_distance": unit_l2,
                "sign_agreement": sign_agree,
                "norm_a": norm_a,
                "norm_b": norm_b,
                "l2_distance": l2_diff,
            }
            per_sample.append(row)
            for key in (
                "cosine",
                "abs_cosine",
                "centered_cosine",
                "spectral_magnitude_cosine",
                "relative_l2_distance",
                "unit_direction_l2_distance",
                "sign_agreement",
                "norm_a",
                "norm_b",
                "l2_distance",
            ):
                values[key].append(row[key])
        out = {
            "source_root": str(root),
            "epsilon": epsilon,
            "alpha": alpha,
            "setting": setting_label(epsilon, alpha),
            "p_order": p_order,
            "q_order": q_order,
            "method_a": method_a,
            "method_b": method_b,
            "sample_count": len(sample_ids),
        }
        for key, vals in values.items():
            stats = finite_stats(vals)
            out[f"{key}_mean"] = stats["mean"]
            out[f"{key}_std"] = stats["std"]
            out[f"{key}_min"] = stats["min"]
            out[f"{key}_max"] = stats["max"]
            out[f"{key}_median"] = stats["median"]
        summary.append(out)
    return per_sample, summary


def method_pair_label(row_or_pair) -> str:
    if isinstance(row_or_pair, tuple):
        a, b = row_or_pair
    else:
        a, b = row_or_pair["method_a"], row_or_pair["method_b"]
    return f"{METHOD_LABELS.get(a, a)} vs {METHOD_LABELS.get(b, b)}"


def plot_pair_setting_heatmap(rows: list[dict], metric: str, title: str, out_path: Path, cmap: str, vmin=None, vmax=None, fmt=".2f") -> None:
    settings = sorted({(float(r["epsilon"]), float(r["alpha"])) for r in rows})
    pairs = list(combinations(CORE4, 2))
    labels_x = [setting_label(eps, alpha) for eps, alpha in settings]
    labels_y = [method_pair_label(pair) for pair in pairs]
    matrix = np.full((len(pairs), len(settings)), np.nan)
    by_key = {(r["method_a"], r["method_b"], float(r["epsilon"]), float(r["alpha"])): r for r in rows}
    for i, pair in enumerate(pairs):
        for j, (eps, alpha) in enumerate(settings):
            row = by_key.get((pair[0], pair[1], eps, alpha))
            if row is not None:
                matrix[i, j] = fnum(row.get(metric))
    fig, ax = plt.subplots(figsize=(max(14, 1.25 * len(settings) + 4), 6.2))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=42, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=9)
    ax.set_title(title, fontsize=14, pad=16)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(value):
                color = "black" if (vmin is not None and vmax is not None and value > (vmin + vmax) / 2) else "white"
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=7.5, color=color)
    fig.subplots_adjust(left=0.22, right=0.88, top=0.82, bottom=0.27)
    cax = fig.add_axes([0.91, 0.27, 0.018, 0.55])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(metric, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_rollup_matrix(rows: list[dict], metric: str, title: str, out_path: Path, cmap: str, vmin=None, vmax=None, fmt=".2f") -> None:
    matrix = np.full((len(CORE4), len(CORE4)), np.nan)
    for i in range(len(CORE4)):
        matrix[i, i] = 1.0 if "cosine" in metric or "agreement" in metric else 0.0
    by_key = {(r["method_a"], r["method_b"]): r for r in rows}
    for i, a in enumerate(CORE4):
        for j, b in enumerate(CORE4):
            if i == j:
                continue
            key = (a, b) if (a, b) in by_key else (b, a)
            row = by_key.get(key)
            if row is not None:
                matrix[i, j] = fnum(row.get(metric))
    labels = [METHOD_LABELS[m] for m in CORE4]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=13, pad=14)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(value):
                color = "black" if (vmin is not None and vmax is not None and value > (vmin + vmax) / 2) else "white"
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, shrink=0.78, label=metric)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def rollup(summary_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in summary_rows:
        groups[(row["method_a"], row["method_b"])].append(row)
    out: list[dict] = []
    for (method_a, method_b), group in sorted(groups.items()):
        row = {"method_a": method_a, "method_b": method_b, "setting_count": len(group)}
        for metric in (
            "cosine_mean",
            "abs_cosine_mean",
            "centered_cosine_mean",
            "spectral_magnitude_cosine_mean",
            "relative_l2_distance_mean",
            "unit_direction_l2_distance_mean",
            "sign_agreement_mean",
            "l2_distance_mean",
        ):
            stats = finite_stats([g[metric] for g in group])
            row[f"across_settings_{metric}"] = stats["mean"]
            row[f"across_settings_{metric}_std"] = stats["std"]
            row[f"across_settings_{metric}_min"] = stats["min"]
            row[f"across_settings_{metric}_max"] = stats["max"]
        out.append(row)
    return out


def write_doc(doc_path: Path, out_dir: Path, roots: list[Path], rollup_rows: list[dict], summary_csv: Path, per_sample_csv: Path, rollup_csv: Path, figures: dict[str, Path]) -> None:
    top_cos = sorted(rollup_rows, key=lambda r: -fnum(r["across_settings_cosine_mean"]))
    low_rel = sorted(rollup_rows, key=lambda r: fnum(r["across_settings_relative_l2_distance_mean"]))
    lines = [
        "# Loss3 Alpha/Epsilon Core-Four Final Delta Similarity - 2026-05-20",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "Observed from completed `p=2,q=2` alpha/epsilon sweep artifacts only. No optimizer experiment was rerun for this analysis.",
        "",
        "Similarity is computed per sample after aligning methods by `dataset_index`, then summarized over the batch and across settings.",
        "",
        "Metrics:",
        "",
        "- `cosine`: signed direction similarity of final delta.",
        "- `centered_cosine`: cosine after subtracting each delta's spatial mean; closer to shape similarity.",
        "- `spectral_magnitude_cosine`: cosine between Fourier magnitude spectra; closer to frequency-content similarity.",
        "- `relative_l2_distance`: actual final perturbation distance normalized by average norm.",
        "- `sign_agreement`: fraction of grid points with matching sign.",
        "",
        "## Key Outputs",
        "",
        f"- Pairwise summary by setting: `{rel(summary_csv)}`",
        f"- Per-sample pairwise table: `{rel(per_sample_csv)}`",
        f"- Across-setting rollup: `{rel(rollup_csv)}`",
    ]
    for label, path in figures.items():
        lines.append(f"- {label}: `{rel(path)}`")
    lines.extend(["", "## Rollup Highlights", ""])
    lines.append("Highest mean cosine pairs across settings:")
    lines.append("")
    lines.append("| pair | mean cosine | mean centered cosine | mean spectral cosine | mean relative L2 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in top_cos[:6]:
        lines.append(
            f"| {method_pair_label(row)} | {fnum(row['across_settings_cosine_mean']):.4f} | "
            f"{fnum(row['across_settings_centered_cosine_mean']):.4f} | "
            f"{fnum(row['across_settings_spectral_magnitude_cosine_mean']):.4f} | "
            f"{fnum(row['across_settings_relative_l2_distance_mean']):.4f} |"
        )
    lines.extend(["", "Smallest relative L2 pairs across settings:", ""])
    lines.append("| pair | mean relative L2 | mean cosine | mean sign agreement |")
    lines.append("|---|---:|---:|---:|")
    for row in low_rel[:6]:
        lines.append(
            f"| {method_pair_label(row)} | {fnum(row['across_settings_relative_l2_distance_mean']):.4f} | "
            f"{fnum(row['across_settings_cosine_mean']):.4f} | "
            f"{fnum(row['across_settings_sign_agreement_mean']):.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Observed evidence should be read with the existing smoothness and high-frequency heatmaps. High delta similarity means the methods find broadly similar final perturbation shapes; it does not by itself decide which optimizer reaches that shape faster or with better post-boundary loss growth.",
        "",
        "Inference: if `steepest_replace` / GPI has high similarity to the other final deltas while also showing faster boundary arrival, stronger post-boundary loss gain, and low high-frequency/peakiness metrics, then it is a stronger practical optimizer for this fixed-budget `p=2,q=2` loss3 setting.",
        "",
        "## Inputs",
        "",
    ])
    for root in roots:
        eps, alpha = root_setting(root)
        p_order, q_order = root_pq(root)
        lines.append(f"- `{rel(root)}` ({setting_label(eps, alpha)}, p={p_order}, q={q_order})")
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--p-filter", default="2")
    parser.add_argument("--q-filter", default="2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root if args.sweep_root.is_absolute() else PROJECT_ROOT / args.sweep_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    doc_path = args.doc if args.doc.is_absolute() else PROJECT_ROOT / args.doc
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = load_completed_roots(sweep_root)
    roots = [root for root in roots if root_pq(root) == (str(args.p_filter), str(args.q_filter))]
    if not roots:
        raise SystemExit("No completed roots matched the requested P/Q filters.")

    all_per_sample: list[dict] = []
    all_summary: list[dict] = []
    for root in roots:
        per_sample, summary = analyze_root(root)
        all_per_sample.extend(per_sample)
        all_summary.extend(summary)

    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    per_sample_csv = tables_dir / "final_delta_pairwise_similarity_per_sample.csv"
    summary_csv = tables_dir / "final_delta_pairwise_similarity_summary.csv"
    rollup_csv = tables_dir / "final_delta_pairwise_similarity_rollup.csv"
    rollup_rows = rollup(all_summary)
    write_csv(per_sample_csv, all_per_sample)
    write_csv(summary_csv, all_summary)
    write_csv(rollup_csv, rollup_rows)

    figures = {
        "Mean cosine by setting heatmap": figures_dir / "pairwise_mean_cosine_by_setting.png",
        "Mean centered-cosine by setting heatmap": figures_dir / "pairwise_mean_centered_cosine_by_setting.png",
        "Mean spectral-cosine by setting heatmap": figures_dir / "pairwise_mean_spectral_cosine_by_setting.png",
        "Mean relative-L2 by setting heatmap": figures_dir / "pairwise_mean_relative_l2_by_setting.png",
        "Across-setting mean cosine matrix": figures_dir / "rollup_mean_cosine_matrix.png",
        "Across-setting mean relative-L2 matrix": figures_dir / "rollup_mean_relative_l2_matrix.png",
    }
    plot_pair_setting_heatmap(all_summary, "cosine_mean", "Final delta mean cosine by setting", figures["Mean cosine by setting heatmap"], "coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plot_pair_setting_heatmap(all_summary, "centered_cosine_mean", "Final delta centered-cosine by setting", figures["Mean centered-cosine by setting heatmap"], "coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plot_pair_setting_heatmap(all_summary, "spectral_magnitude_cosine_mean", "Final delta spectral-magnitude cosine by setting", figures["Mean spectral-cosine by setting heatmap"], "viridis", vmin=0, vmax=1, fmt=".2f")
    max_rel = max([fnum(r["relative_l2_distance_mean"]) for r in all_summary if math.isfinite(fnum(r["relative_l2_distance_mean"]))] or [1.0])
    plot_pair_setting_heatmap(all_summary, "relative_l2_distance_mean", "Final delta relative L2 distance by setting", figures["Mean relative-L2 by setting heatmap"], "magma_r", vmin=0, vmax=max_rel, fmt=".2f")
    plot_rollup_matrix(rollup_rows, "across_settings_cosine_mean", "Across-setting mean final-delta cosine", figures["Across-setting mean cosine matrix"], "coolwarm", vmin=-1, vmax=1, fmt=".2f")
    max_roll_rel = max([fnum(r["across_settings_relative_l2_distance_mean"]) for r in rollup_rows if math.isfinite(fnum(r["across_settings_relative_l2_distance_mean"]))] or [1.0])
    plot_rollup_matrix(rollup_rows, "across_settings_relative_l2_distance_mean", "Across-setting mean final-delta relative L2", figures["Across-setting mean relative-L2 matrix"], "magma_r", vmin=0, vmax=max_roll_rel, fmt=".2f")

    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_root": str(sweep_root),
        "out_dir": str(out_dir),
        "p_filter": str(args.p_filter),
        "q_filter": str(args.q_filter),
        "completed_setting_count": len(roots),
        "tables": {
            "per_sample": str(per_sample_csv),
            "summary": str(summary_csv),
            "rollup": str(rollup_csv),
        },
        "figures": {label: str(path) for label, path in figures.items()},
        "doc": str(doc_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_doc(doc_path, out_dir, roots, rollup_rows, summary_csv, per_sample_csv, rollup_csv, figures)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
