#!/usr/bin/env python3
"""Plot optimizer-grouped loss curves for Burgers and NS2D.

The target loss is held fixed inside each panel; the four optimizer variants are
overlaid. This is an offline plotting script and does not run model/solver code.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GITCLEAN = Path("/workspace/NeuralOperatorRobustness2_gitclean")
OUT_DIR = GITCLEAN / "docs" / "optimizer_grouped_loss_curves_20260524"
REPORT = GITCLEAN / "docs" / "optimizer_grouped_loss_curves_20260524.md"
BUNDLE = GITCLEAN / "docs" / "ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524"

METHODS = ["raw_add", "raw_replace", "steepest_add", "steepest_replace"]
METHOD_LABELS = {
    "raw_add": "Raw Add",
    "raw_replace": "Raw Replace",
    "steepest_add": "Steepest Add",
    "steepest_replace": "Steepest Replace",
}
METHOD_COLORS = {
    "raw_add": "#2f6fb0",
    "raw_replace": "#d95f02",
    "steepest_add": "#1b9e77",
    "steepest_replace": "#c51b29",
}
LOSS_LABELS = {
    "loss1": "Loss 1 / all W",
    "loss2": "Loss 2 / all A -> W",
    "loss3": "Loss 3 / all W",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def farray(items: list[dict[str, str]], key: str) -> np.ndarray:
    out = []
    for row in items:
        try:
            out.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            out.append(math.nan)
    return np.asarray(out, dtype=np.float64)


def roughness(y: np.ndarray) -> float:
    y = y[np.isfinite(y)]
    if y.size < 3:
        return math.nan
    dy = np.diff(y)
    span = max(float(np.nanmax(y) - np.nanmin(y)), 1e-12)
    return float(np.nanmean(np.abs(dy)) / span)


def negative_step_fraction(y: np.ndarray) -> float:
    y = y[np.isfinite(y)]
    if y.size < 2:
        return math.nan
    dy = np.diff(y)
    return float(np.mean(dy < 0))


def solid_line_ylim(curves: list[dict[str, np.ndarray | float | str]]) -> tuple[float, float]:
    vals = []
    for curve in curves:
        y = curve.get("y")
        if not isinstance(y, np.ndarray):
            continue
        y = y[np.isfinite(y)]
        if y.size:
            vals.append(y)
    if not vals:
        return 0.0, 1.0
    all_vals = np.concatenate(vals)
    lo = float(np.min(all_vals))
    hi = float(np.max(all_vals))
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0) * 0.1
    return lo - 0.08 * span, hi + 0.12 * span


def load_curve(path: Path, y_key: str, std_key: str | None = None, x_max: float = 100.0) -> dict[str, np.ndarray | float | str]:
    data = rows(path)
    x = farray(data, "k")
    y = farray(data, y_key)
    std = farray(data, std_key) if std_key else np.full_like(y, math.nan)
    keep = np.isfinite(x) & np.isfinite(y) & (x <= x_max)
    x = x[keep]
    y = y[keep]
    std = std[keep] if std.shape == keep.shape else np.full_like(y, math.nan)
    return {
        "x": x,
        "y": y,
        "std": std,
        "roughness": roughness(y),
        "negative_step_fraction": negative_step_fraction(y),
        "final": float(y[-1]) if y.size else math.nan,
        "source": str(path),
    }


def burgers_paths() -> dict[str, dict[str, tuple[Path, str, str]]]:
    base12 = GITCLEAN / "forensics/loss1_loss2_core4_p2q2_baseline_marked_angles_20260521/fno_nu0p001_eps4_alpha0p4_batch100_steps100_p2_q2"
    base3 = GITCLEAN / "forensics/loss3_alpha_epsilon_core4_baseline_giftrace_20260520/fno_nu0p001_eps4_alpha0p4_batch100_steps300_p2_q2"
    out: dict[str, dict[str, tuple[Path, str, str]]] = {"loss1": {}, "loss2": {}, "loss3": {}}
    for method in METHODS:
        out["loss1"][method] = (base12 / "loss1_original" / method / "per_step_metrics.csv", "optimized_loss_mean", "optimized_loss_std")
        out["loss2"][method] = (base12 / "loss2_original" / method / "per_step_metrics.csv", "optimized_loss_mean", "optimized_loss_std")
        out["loss3"][method] = (base3 / method / "per_step_metrics.csv", "loss3_q_mean", "loss3_q_std")
    return out


def ns_paths() -> dict[str, dict[str, dict[str, tuple[Path, str, str]]]]:
    root = GITCLEAN / "2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack"
    cases = {
        "eps32_alpha10": {
            "label": "Epsilon = 32, Alpha = 10",
            "loss1_base": root / "full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_055315_UTC/batch_0000_0009/loss1",
            "loss2_base": root / "full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_aaaaaaaaaw_p2_q2_20260522_070858_UTC/batch_0000_0009/loss2",
            "loss3_base": root / "full_adw_b10_pair_outer_baseline_first_20260522/eps32_alpha10/mode_wwwwwwwwww_p2_q2_20260522_074135_UTC/batch_0000_0009/loss3",
        },
        "eps1_alpha0p3125": {
            "label": "Epsilon = 1, Alpha = 0.3125",
            "loss1_base": root / "minimal_mechanism_eps4_eps2_eps1_b10_20260523_111410_UTC/eps1_alpha0p3125/mode_wwwwwwwwww_p2_q2_20260523_185512_UTC/batch_0000_0009/loss1",
            "loss2_base": root / "eps1_missing_loss2_loss3_b10_20260524_024533_UTC/eps1_alpha0p3125/mode_aaaaaaaaaw_p2_q2_20260524_024538_UTC/batch_0000_0009/loss2",
            "loss3_base": root / "eps1_missing_loss2_loss3_b10_20260524_024533_UTC/eps1_alpha0p3125/mode_wwwwwwwwww_p2_q2_20260524_031339_UTC/batch_0000_0009/loss3",
        },
    }
    out: dict[str, dict[str, dict[str, tuple[Path, str, str]]]] = {}
    for case, spec in cases.items():
        out[case] = {"_label": spec["label"]}  # type: ignore[dict-item]
        for loss in ["loss1", "loss2", "loss3"]:
            out[case][loss] = {}
            base = spec[f"{loss}_base"]
            for method in METHODS:
                out[case][loss][method] = (base / method / "per_step_metrics.csv", "active_loss_mean", f"{loss}_std")
    return out


def draw_burgers(curves: dict[str, dict[str, dict[str, np.ndarray | float | str]]]) -> Path:
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 180,
        "savefig.dpi": 240,
    })
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 11.2), sharex=True)
    fig.suptitle("Burgers optimizer comparison with target loss fixed\nEpsilon = 4, Alpha = 0.4, p = 2, q = 2", fontsize=17, weight="bold")
    for ax, loss in zip(axes, ["loss1", "loss2", "loss3"]):
        solid_curves = []
        for method in METHODS:
            c = curves[loss][method]
            solid_curves.append(c)
            x = c["x"]
            y = c["y"]
            std = c["std"]
            lw = 3.2 if method == "steepest_replace" else 2.4
            ax.plot(x, y, color=METHOD_COLORS[method], lw=lw, label=METHOD_LABELS[method])
            if isinstance(std, np.ndarray) and np.isfinite(std).any():
                ax.fill_between(x, y - std, y + std, color=METHOD_COLORS[method], alpha=0.10, linewidth=0)
        ax.set_title(LOSS_LABELS[loss])
        ax.set_ylabel("optimized target loss")
        ax.grid(True, color="#d8d8d2", linewidth=0.7, alpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, 100)
        ax.set_ylim(*solid_line_ylim(solid_curves))
    axes[-1].set_xlabel("attack step")
    axes[0].legend(ncol=4, loc="upper left", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "burgers_optimizer_grouped_target_loss_curves_eps4_alpha0p4.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def draw_ns(curves: dict[str, dict[str, dict[str, dict[str, np.ndarray | float | str]]]]) -> Path:
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 180,
        "savefig.dpi": 240,
    })
    case_order = ["eps32_alpha10", "eps1_alpha0p3125"]
    fig, axes = plt.subplots(3, 2, figsize=(15.5, 12.2), sharex=True)
    fig.suptitle("NS2D optimizer comparison with target loss fixed\nactive target loss; p = 2, q = 2", fontsize=17, weight="bold")
    labels = {"eps32_alpha10": "Epsilon = 32, Alpha = 10", "eps1_alpha0p3125": "Epsilon = 1, Alpha = 0.3125"}
    for ci, case in enumerate(case_order):
        for ri, loss in enumerate(["loss1", "loss2", "loss3"]):
            ax = axes[ri, ci]
            solid_curves = []
            for method in METHODS:
                c = curves[case][loss][method]
                solid_curves.append(c)
                x = c["x"]
                y = c["y"]
                std = c["std"]
                lw = 3.2 if method == "steepest_replace" else 2.4
                ax.plot(x, y, color=METHOD_COLORS[method], lw=lw, label=METHOD_LABELS[method])
                if isinstance(std, np.ndarray) and np.isfinite(std).any():
                    ax.fill_between(x, y - std, y + std, color=METHOD_COLORS[method], alpha=0.10, linewidth=0)
            if ri == 0:
                ax.set_title(labels[case])
            ax.text(0.02, 0.92, LOSS_LABELS[loss], transform=ax.transAxes, va="top", ha="left", fontsize=11, weight="bold")
            ax.set_ylabel("active target loss")
            ax.grid(True, color="#d8d8d2", linewidth=0.7, alpha=0.85)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(0, 100)
            ax.set_ylim(*solid_line_ylim(solid_curves))
    for ax in axes[-1, :]:
        ax.set_xlabel("attack step")
    axes[0, 0].legend(ncol=4, loc="upper left", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "ns2d_optimizer_grouped_target_loss_curves_eps32_vs_eps1.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def load_burgers() -> dict[str, dict[str, dict[str, np.ndarray | float | str]]]:
    data = {}
    for loss, by_method in burgers_paths().items():
        data[loss] = {}
        for method, (path, y_key, std_key) in by_method.items():
            data[loss][method] = load_curve(path, y_key, std_key, x_max=100)
    return data


def load_ns() -> dict[str, dict[str, dict[str, dict[str, np.ndarray | float | str]]]]:
    data = {}
    for case, by_loss in ns_paths().items():
        data[case] = {}
        for loss in ["loss1", "loss2", "loss3"]:
            data[case][loss] = {}
            for method, (path, y_key, std_key) in by_loss[loss].items():  # type: ignore[index]
                data[case][loss][method] = load_curve(path, y_key, std_key, x_max=100)
    return data


def rel(path: Path) -> str:
    return path.relative_to(GITCLEAN).as_posix()


def write_report(burgers: dict, ns: dict, burgers_png: Path, ns_png: Path) -> None:
    lines = [
        "# Optimizer-Grouped Loss Curves - 2026-05-24",
        "",
        "Observed from saved `per_step_metrics.csv` files only. No model inference, solver rollout, or attack update was run.",
        "",
        "Each panel fixes the attack target loss and overlays the four optimizer methods.",
        "",
        "## Figures",
        "",
        f"- Burgers: [{burgers_png.name}]({rel(burgers_png)})",
        f"- NS2D: [{ns_png.name}]({rel(ns_png)})",
        "",
        "## Interpretation",
        "",
        "- For `p = 2`, the steepest direction is an L2-normalized gradient direction. The direction is not fundamentally different from PGD; the main difference is normalization and how the update is combined with the existing perturbation.",
        "- `replace` discards the previous perturbation direction and uses the current proposal after projection. When the gradient direction rotates between steps, this can move the perturbation to a very different point on the epsilon-ball boundary, producing visible oscillation.",
        "- `add` keeps memory of the previous perturbation through `delta + alpha * direction`, so projection tends to smooth the trajectory.",
        "- At very small epsilon, the problem is closer to a local linear regime, so add/replace/raw/steepest can look much more similar.",
        "",
        "## Roughness Summary",
        "",
        "Roughness is mean absolute one-step loss change divided by the curve range; it is a simple diagnostic for jaggedness, not a new objective.",
        "",
        "### Burgers",
        "",
        "| target | method | final loss | roughness | negative-step fraction | source |",
        "|---|---|---:|---:|---:|---|",
    ]
    for loss in ["loss1", "loss2", "loss3"]:
        for method in METHODS:
            c = burgers[loss][method]
            lines.append(f"| {LOSS_LABELS[loss]} | {METHOD_LABELS[method]} | {c['final']:.6g} | {c['roughness']:.4g} | {c['negative_step_fraction']:.3g} | `{rel(Path(c['source']))}` |")
    lines += ["", "### NS2D", "", "| case | target | method | final loss | roughness | negative-step fraction | source |", "|---|---|---|---:|---:|---:|---|"]
    case_labels = {"eps32_alpha10": "Epsilon = 32, Alpha = 10", "eps1_alpha0p3125": "Epsilon = 1, Alpha = 0.3125"}
    for case in ["eps32_alpha10", "eps1_alpha0p3125"]:
        for loss in ["loss1", "loss2", "loss3"]:
            for method in METHODS:
                c = ns[case][loss][method]
                lines.append(f"| {case_labels[case]} | {LOSS_LABELS[loss]} | {METHOD_LABELS[method]} | {c['final']:.6g} | {c['roughness']:.4g} | {c['negative_step_fraction']:.3g} | `{rel(Path(c['source']))}` |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_to_bundle(paths: list[Path]) -> None:
    ns_dir = BUNDLE / "NS2D" / "optimizer_grouped_loss_curves"
    burgers_dir = BUNDLE / "Burgers" / "optimizer_grouped_loss_curves"
    ns_dir.mkdir(parents=True, exist_ok=True)
    burgers_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.name.startswith("ns2d"):
            target = ns_dir / path.name
        elif path.name.startswith("burgers"):
            target = burgers_dir / path.name
        else:
            continue
        target.write_bytes(path.read_bytes())
    if REPORT.exists():
        (BUNDLE / REPORT.name).write_text(REPORT.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    burgers = load_burgers()
    ns = load_ns()
    burgers_png = draw_burgers(burgers)
    ns_png = draw_ns(ns)
    write_report(burgers, ns, burgers_png, ns_png)
    copy_to_bundle([burgers_png, ns_png])
    print(burgers_png)
    print(ns_png)
    print(REPORT)


if __name__ == "__main__":
    main()
