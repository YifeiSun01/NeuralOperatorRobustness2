#!/usr/bin/env python3
"""Plot final perturbation comparisons across attack methods."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOSSES = ("loss1", "loss2", "loss3")
VARIANTS = ("original", "increment_ratio", "regularized")
METHODS = ("pgd", "lp_steepest_pgd", "generalized_power")
INITIAL_MODES = ("random", "zero")
METHOD_LABELS = {
    "pgd": "projected gradient descent",
    "lp_steepest_pgd": "LP steepest PGD",
    "generalized_power": "generalized power iteration",
}
METHOD_SHORT_LABELS = {
    "pgd": "PGD",
    "lp_steepest_pgd": "LP",
    "generalized_power": "GPI",
}
METHOD_COLORS = {
    "pgd": "#2563eb",
    "lp_steepest_pgd": "#dc2626",
    "generalized_power": "#059669",
}
METHOD_PAIRS = (
    ("pgd", "lp_steepest_pgd"),
    ("pgd", "generalized_power"),
    ("lp_steepest_pgd", "generalized_power"),
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
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


def finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def finite_mean_std(values: Any) -> tuple[float | None, float | None, int, int]:
    arr = np.asarray(values, dtype=np.float64)
    total_count = int(arr.size)
    finite = np.isfinite(arr)
    finite_count = int(np.count_nonzero(finite))
    if finite_count == 0:
        return None, None, finite_count, total_count
    finite_values = arr[finite]
    return (
        float(np.mean(finite_values)),
        float(np.std(finite_values, ddof=0)),
        finite_count,
        total_count,
    )


def finite_mean_std_axis0(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    count = np.count_nonzero(finite, axis=0)
    cleaned = np.where(finite, arr, 0.0)
    mean = np.divide(cleaned.sum(axis=0), count, out=np.full(arr.shape[1:], np.nan), where=count > 0)
    centered = np.where(finite, arr - mean, 0.0)
    var = np.divide((centered * centered).sum(axis=0), count, out=np.full(arr.shape[1:], np.nan), where=count > 0)
    return mean, np.sqrt(var), count


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def format_pm(mean: float | None, std: float | None, *, suffix: str = "") -> str:
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.4g}{suffix}"
    return f"{mean:.4g}+/-{std:.3g}{suffix}"


def load_config(root: Path) -> dict[str, Any]:
    config_path = root / "config.json"
    if config_path.exists():
        return read_json(config_path)
    return {}


def resolve_path(value: Any) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    return PROJECT_ROOT / path


def load_burgers_batch(config: dict[str, Any]) -> np.ndarray:
    import torch

    path = resolve_path(config["burgers_test_path"])
    start = int(config.get("start_index", 0))
    batch_size = int(config.get("batch_size", 100))
    data = torch.load(path, map_location="cpu", weights_only=False)
    x = data["x"][start : start + batch_size].float().cpu().numpy()
    return x[..., None].astype(np.float32)


def infer_model_name(config: dict[str, Any]) -> str:
    if config.get("model_label"):
        return str(config["model_label"])
    if config.get("model_kind") == "deeponet":
        return "DeepONet/default net"
    checkpoint = str(config.get("burgers_torch_checkpoint", "")).lower()
    if "deeponet" in checkpoint:
        return "DeepONet"
    if "fno" in checkpoint:
        return "FNO"
    return "model"


def format_config_summary(config: dict[str, Any]) -> str:
    if not config:
        return "parameters: config.json not found"
    start = int(config.get("start_index", 0))
    batch = int(config.get("batch_size", 100))
    end = start + batch - 1
    p = config.get("p_order", config.get("p", "?"))
    q = config.get("q_order", config.get("q", "?"))
    return (
        f"model={infer_model_name(config)}, nu={config.get('burgers_nu', '?')}, "
        f"batch={batch}, index={start}-{end}, steps={config.get('steps', '?')}, "
        f"p={p}, q={q}, epsilon={config.get('epsilon', '?')}, alpha={config.get('alpha', '?')}"
    )


def run_initial_delta(run_dir: Path, fallback: str) -> str:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        return str(summary.get("initial_delta", fallback))
    name = run_dir.name
    if name.endswith("_init_random"):
        return "random"
    if name.endswith("_init_zero"):
        return "zero"
    return fallback


def find_run_dir(root: Path, optimized_loss: str, variant: str, method: str, initial_delta: str) -> Path | None:
    candidates = [
        root / f"{optimized_loss}_{variant}_{method}_init_{initial_delta}",
        root / f"{optimized_loss}_{variant}_{method}",
    ]
    for run_dir in candidates:
        if not (run_dir / "final_delta.npz").exists():
            continue
        fallback = initial_delta if optimized_loss == "loss1" else "zero"
        if run_initial_delta(run_dir, fallback) == initial_delta:
            return run_dir
    return None


def load_delta_group(
    root: Path,
    optimized_loss: str,
    variant: str,
    initial_delta: str,
) -> dict[str, dict[str, Any]]:
    group: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        run_dir = find_run_dir(root, optimized_loss, variant, method, initial_delta)
        if run_dir is None:
            continue
        data = np.load(run_dir / "final_delta.npz")
        summary = read_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
        group[method] = {
            "run_dir": run_dir,
            "final_delta": data["final_delta"].astype(np.float64),
            "dataset_index": data["dataset_index"].astype(np.int64),
            "final_delta_pnorm": data["final_delta_pnorm"].astype(np.float64),
            "summary": summary,
        }
    return group


def available_initial_modes(root: Path, loss: str) -> list[str]:
    if loss != "loss1":
        return ["zero"]
    modes: set[str] = set()
    for variant in VARIANTS:
        for method in METHODS:
            for mode in INITIAL_MODES:
                run_dir = find_run_dir(root, loss, variant, method, mode)
                if run_dir is not None:
                    modes.add(mode)
    return [mode for mode in INITIAL_MODES if mode in modes]


def vector_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    finite = np.isfinite(a_flat).all(axis=1) & np.isfinite(b_flat).all(axis=1)
    dot = np.sum(a_flat * b_flat, axis=1)
    norm_a = np.linalg.norm(a_flat, axis=1)
    norm_b = np.linalg.norm(b_flat, axis=1)
    valid = finite & np.isfinite(dot) & np.isfinite(norm_a) & np.isfinite(norm_b) & (norm_a > 0) & (norm_b > 0)
    cos = np.full(a_flat.shape[0], np.nan, dtype=np.float64)
    cos[valid] = dot[valid] / (norm_a[valid] * norm_b[valid])
    finite_cos = np.isfinite(cos)
    cos[finite_cos] = np.clip(cos[finite_cos], -1.0, 1.0)
    return cos


def cosine_rows(
    *,
    optimized_loss: str,
    variant: str,
    initial_delta: str,
    group: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    title_parts: list[str] = []
    dataset_index = next(iter(group.values()))["dataset_index"]
    for method_a, method_b in METHOD_PAIRS:
        if method_a not in group or method_b not in group:
            continue
        cos = vector_cosine(group[method_a]["final_delta"], group[method_b]["final_delta"])
        angles = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        cos_mean, cos_std, finite_count, total_count = finite_mean_std(cos)
        angle_mean, angle_std, _, _ = finite_mean_std(angles)
        pair_name = f"{METHOD_SHORT_LABELS[method_a]}-{METHOD_SHORT_LABELS[method_b]}"
        title_parts.append(
            f"{pair_name}: cos={format_pm(cos_mean, cos_std)}, angle={format_pm(angle_mean, angle_std, suffix=' deg')}, n={finite_count}/{total_count}"
        )
        summary_rows.append(
            {
                "optimized_loss": optimized_loss,
                "objective_variant": variant,
                "initial_delta": initial_delta,
                "method_a": method_a,
                "method_b": method_b,
                "pair": pair_name,
                "cosine_mean": cos_mean,
                "cosine_std": cos_std,
                "cosine_variance": None if cos_std is None else cos_std * cos_std,
                "angle_degrees_mean": angle_mean,
                "angle_degrees_std": angle_std,
                "angle_degrees_variance": None if angle_std is None else angle_std * angle_std,
                "finite_count": finite_count,
                "nonfinite_count": total_count - finite_count,
            }
        )
        for i, value in enumerate(cos):
            angle = float(angles[i]) if np.isfinite(angles[i]) else None
            sample_rows.append(
                {
                    "optimized_loss": optimized_loss,
                    "objective_variant": variant,
                    "initial_delta": initial_delta,
                    "sample_position": i,
                    "dataset_index": int(dataset_index[i]),
                    "method_a": method_a,
                    "method_b": method_b,
                    "pair": pair_name,
                    "cosine_similarity": float(value) if np.isfinite(value) else None,
                    "angle_degrees": angle,
                }
            )
    return sample_rows, summary_rows, "\n".join(title_parts)


def sample_label(method: str, group_item: dict[str, Any], sample_position: int) -> str:
    norms = group_item["final_delta_pnorm"]
    norm = finite_float(norms[sample_position]) if sample_position < len(norms) else None
    return (
        f"{METHOD_SHORT_LABELS[method]} "
        + r"$\|\delta\|_p$="
        + f"{format_metric(norm)}"
    )


def plot_loss_group(
    *,
    output_dir: Path,
    x0: np.ndarray,
    x_grid: np.ndarray,
    optimized_loss: str,
    initial_delta: str,
    variant_groups: dict[str, dict[str, dict[str, Any]]],
    cosine_titles: dict[str, str],
    sample_position: int,
    config_summary: str,
    dpi: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    init_suffix = f"_init_{initial_delta}" if optimized_loss == "loss1" else ""
    base_name = f"{optimized_loss}{init_suffix}_final_delta_comparison_index{sample_position}"
    first_group = next(iter(variant_groups.values()))
    first_item = next(iter(first_group.values()))
    dataset_index = int(first_item["dataset_index"][sample_position])

    if sample_position < 0 or sample_position >= x0.shape[0]:
        raise IndexError(f"sample_position={sample_position} outside batch size {x0.shape[0]}")

    x0_line = x0[sample_position, :, 0].astype(np.float64)
    fig, axes = plt.subplots(len(VARIANTS), 2, figsize=(18.0, 17.0), sharex=True)
    fig.suptitle(
        f"{optimized_loss.upper()} Final Delta Comparison | dataset index {dataset_index} | initial delta: {initial_delta}\n"
        f"Each curve is one sample only; cosine and angle stats are computed per sample first, then mean/std over the batch.\n"
        f"{config_summary}",
        fontsize=14,
        fontweight="bold",
    )

    for row_idx, variant in enumerate(VARIANTS):
        group = variant_groups.get(variant)
        delta_ax = axes[row_idx, 0]
        adv_ax = axes[row_idx, 1]
        if not group:
            for ax in (delta_ax, adv_ax):
                ax.set_axis_off()
            continue

        for method in METHODS:
            if method not in group:
                continue
            delta = group[method]["final_delta"]
            if sample_position >= delta.shape[0]:
                raise IndexError(f"sample_position={sample_position} outside {method} batch size {delta.shape[0]}")
            delta_line = delta[sample_position, :, 0].astype(np.float64)
            mask = np.isfinite(delta_line)
            label = sample_label(method, group[method], sample_position)
            delta_ax.plot(
                x_grid,
                np.ma.masked_where(~mask, delta_line),
                color=METHOD_COLORS[method],
                linewidth=2.1,
                label=label,
            )
            adv_line = x0_line + delta_line
            adv_mask = np.isfinite(adv_line)
            adv_ax.plot(
                x_grid,
                np.ma.masked_where(~adv_mask, adv_line),
                color=METHOD_COLORS[method],
                linewidth=2.1,
                label=label,
            )

        delta_ax.axhline(0.0, color="#52525b", linewidth=0.9, alpha=0.8)
        adv_ax.plot(x_grid, x0_line, color="#18181b", linewidth=1.5, linestyle="--", label="initial condition")
        metric_text = cosine_titles.get(variant, "")
        delta_ax.set_title(f"{variant}: final delta\n{metric_text}", fontsize=10.5, loc="left", pad=8)
        adv_ax.set_title(f"{variant}: initial + final delta\n{metric_text}", fontsize=10.5, loc="left", pad=8)
        delta_ax.set_ylabel("delta", fontsize=11)
        adv_ax.set_ylabel("x + delta", fontsize=11)
        for ax in (delta_ax, adv_ax):
            ax.grid(True, color="#d4d4d8", linewidth=0.8, alpha=0.75)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="best", fontsize=8.5, frameon=True)

    axes[-1, 0].set_xlabel("x", fontsize=12)
    axes[-1, 1].set_xlabel("x", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out_path = output_dir / f"{base_name}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/three_loss_batch100_loss_only"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--losses", nargs="+", choices=LOSSES, default=list(LOSSES))
    parser.add_argument("--sample-position", type=int, default=0)
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    config = load_config(args.root)
    output_dir = args.output_dir or args.root / "figures" / "final_delta_comparison" / "png"
    if output_dir.exists() and not args.no_clean:
        shutil.rmtree(output_dir)
    if not list(args.root.rglob("final_delta.npz")):
        print(f"[warn] no final_delta.npz files found under {args.root}; rerun attacks with the updated script")
        return

    x0 = load_burgers_batch(config)
    nx = int(x0.shape[1])
    domain = float(config.get("burgers_domain", 2.0))
    x_grid = np.linspace(0.0, domain, nx, endpoint=False)
    config_summary = format_config_summary(config)

    all_sample_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []
    for optimized_loss in args.losses:
        for initial_delta in available_initial_modes(args.root, optimized_loss):
            variant_groups: dict[str, dict[str, dict[str, Any]]] = {}
            cosine_titles: dict[str, str] = {}
            for variant in VARIANTS:
                group = load_delta_group(args.root, optimized_loss, variant, initial_delta)
                if len(group) < 2:
                    continue
                sample_rows, summary_rows, cosine_title = cosine_rows(
                    optimized_loss=optimized_loss,
                    variant=variant,
                    initial_delta=initial_delta,
                    group=group,
                )
                all_sample_rows.extend(sample_rows)
                all_summary_rows.extend(summary_rows)
                if len(group) == len(METHODS):
                    variant_groups[variant] = group
                    cosine_titles[variant] = cosine_title
            if variant_groups:
                plot_path = plot_loss_group(
                    output_dir=output_dir,
                    x0=x0,
                    x_grid=x_grid,
                    optimized_loss=optimized_loss,
                    initial_delta=initial_delta,
                    variant_groups=variant_groups,
                    cosine_titles=cosine_titles,
                    sample_position=args.sample_position,
                    config_summary=config_summary,
                    dpi=args.dpi,
                )
                plot_rows.append(
                    {
                        "optimized_loss": optimized_loss,
                        "initial_delta": initial_delta,
                        "sample_position": args.sample_position,
                        "final_delta_comparison_plot": str(plot_path),
                    }
                )
                print(f"[combined] {plot_path}")

    summary_dir = args.root / "final_delta_comparisons"
    write_csv(summary_dir / "cosine_similarity_summary.csv", all_summary_rows)
    write_csv(summary_dir / "cosine_similarity_per_sample.csv", all_sample_rows)
    write_json(summary_dir / "plot_manifest.json", plot_rows)
    print(f"[summary] {summary_dir / 'cosine_similarity_summary.csv'}")


if __name__ == "__main__":
    main()
