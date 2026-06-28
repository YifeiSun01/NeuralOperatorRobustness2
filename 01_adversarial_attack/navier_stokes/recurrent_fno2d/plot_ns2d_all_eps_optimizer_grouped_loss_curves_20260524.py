#!/usr/bin/env python3
"""Plot NS2D optimizer-grouped target-loss curves for every available epsilon.

Each output PNG fixes one epsilon/alpha setting. Rows are the available target
loss/mode candidates for that setting; each row overlays the optimizer methods.
This is offline-only and reads saved per_step_metrics.csv files.
"""

from __future__ import annotations

import csv
import json
import math
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/workspace/NeuralOperatorRobustness2_gitclean")
DATA_ROOT = ROOT / "2D_NS_FNO2d_recurrent/perturbation_results/ns2d_recurrent_core4_attack"
OUT_DIR = ROOT / "docs/optimizer_grouped_loss_curves_20260524/ns2d_all_eps"
REPORT = ROOT / "docs/ns2d_all_eps_optimizer_grouped_loss_curves_20260524.md"
BUNDLE = ROOT / "docs/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524"
BUNDLE_TAR = ROOT / "docs/ns_burgers_spectrum_loss_curve_cleanstyle_bundle_20260524.tar.gz"

RUN_ROOTS = [
    DATA_ROOT / "full_adw_b10_pair_outer_baseline_first_20260522",
    DATA_ROOT / "minimal_mechanism_eps4_eps2_eps1_b10_20260523_111410_UTC",
    DATA_ROOT / "eps1_missing_loss2_loss3_b10_20260524_024533_UTC",
    DATA_ROOT / "eps1_missing_loss2_loss3_b10_20260524_023355_UTC",
]
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
EPS_ORDER = [
    "eps160_alpha50",
    "eps32_alpha10",
    "eps16_alpha5",
    "eps8_alpha2p5",
    "eps4_alpha1p25",
    "eps2_alpha0p625",
    "eps1_alpha0p3125",
]
EPS_PRETTY = {
    "eps160_alpha50": "Epsilon = 160, Alpha = 50",
    "eps32_alpha10": "Epsilon = 32, Alpha = 10",
    "eps16_alpha5": "Epsilon = 16, Alpha = 5",
    "eps8_alpha2p5": "Epsilon = 8, Alpha = 2.5",
    "eps4_alpha1p25": "Epsilon = 4, Alpha = 1.25",
    "eps2_alpha0p625": "Epsilon = 2, Alpha = 0.625",
    "eps1_alpha0p3125": "Epsilon = 1, Alpha = 0.3125",
}
MODE_LABELS = {
    "wwwwwwwwww": "all_w",
    "dddddddddw": "all_d_target_w",
    "aaaaaaaaaw": "all_a_target_w",
    "wwwwwddddw": "w1_5_d6_9_target_w",
    "dddddwwwww": "d1_5_w6_9_target_w",
    "aaaaaddddw": "a1_5_d6_9_target_w",
}
TARGET_ORDER = [
    "loss1/all_w",
    "loss2/all_a_target_w",
    "loss3/all_w",
    "loss3/all_d_target_w",
    "loss3/all_a_target_w",
    "loss3/w1_5_d6_9_target_w",
    "loss3/d1_5_w6_9_target_w",
    "loss3/a1_5_d6_9_target_w",
]
TARGET_LABELS = {
    "loss1/all_w": "Loss 1 / all W",
    "loss2/all_a_target_w": "Loss 2 / all A -> W",
    "loss3/all_w": "Loss 3 / all W",
    "loss3/all_d_target_w": "Loss 3 / all D -> W",
    "loss3/all_a_target_w": "Loss 3 / all A -> W",
    "loss3/w1_5_d6_9_target_w": "Loss 3 / W steps 1-5, D steps 6-9 -> W",
    "loss3/d1_5_w6_9_target_w": "Loss 3 / D steps 1-5, W steps 6-9 -> W",
    "loss3/a1_5_d6_9_target_w": "Loss 3 / A steps 1-5, D steps 6-9 -> W",
}


@dataclass(frozen=True)
class Candidate:
    eps: str
    target: str
    method: str
    final_outputs: Path
    metrics: Path


def parse_eps(path: Path) -> str | None:
    for part in path.parts:
        if re.match(r"^eps[^/]+_alpha[^/]+$", part):
            return part
    return None


def parse_mode(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("mode_"):
            return part[len("mode_") :].split("_", 1)[0]
    return None


def classify(loss_type: str, mode_spec: str) -> str | None:
    if loss_type == "loss1" and mode_spec == "wwwwwwwwww":
        return "loss1/all_w"
    if loss_type == "loss2" and mode_spec == "aaaaaaaaaw":
        return "loss2/all_a_target_w"
    if loss_type == "loss3":
        mode = MODE_LABELS.get(mode_spec)
        if mode:
            return f"loss3/{mode}"
    return None


def discover() -> dict[str, dict[str, dict[str, Candidate]]]:
    out: dict[str, dict[str, dict[str, Candidate]]] = {}
    for root in RUN_ROOTS:
        if not root.exists():
            continue
        for final_outputs in root.rglob("final_state_outputs.npz"):
            eps = parse_eps(final_outputs)
            mode_spec = parse_mode(final_outputs)
            if eps is None or mode_spec is None:
                continue
            try:
                loss_type = final_outputs.parts[-3]
                method = final_outputs.parts[-2]
            except IndexError:
                continue
            if method not in METHODS:
                continue
            target = classify(loss_type, mode_spec)
            if target is None:
                continue
            metrics = final_outputs.with_name("per_step_metrics.csv")
            if not metrics.exists():
                continue
            prev = out.setdefault(eps, {}).setdefault(target, {}).get(method)
            cand = Candidate(eps, target, method, final_outputs, metrics)
            if prev is None or final_outputs.stat().st_mtime > prev.final_outputs.stat().st_mtime:
                out[eps][target][method] = cand
    return out


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def farr(rows: list[dict[str, str]], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            vals.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            vals.append(math.nan)
    return np.asarray(vals, dtype=np.float64)


def target_std_key(target: str) -> str:
    if target.startswith("loss1/"):
        return "loss1_std"
    if target.startswith("loss2/"):
        return "loss2_std"
    return "loss3_std"


def target_mean_key(target: str) -> str:
    if target.startswith("loss1/"):
        return "loss1_mean"
    if target.startswith("loss2/"):
        return "loss2_mean"
    return "loss3_mean"


def solid_line_ylim(curves: list[dict[str, np.ndarray | float]]) -> tuple[float, float]:
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


def load_curve(cand: Candidate) -> dict[str, np.ndarray | float]:
    data = read_rows(cand.metrics)
    x = farr(data, "k")
    y = farr(data, "active_loss_mean")
    if not np.isfinite(y).any():
        y = farr(data, target_mean_key(cand.target))
    std = farr(data, target_std_key(cand.target))
    keep = np.isfinite(x) & np.isfinite(y) & (x <= 100)
    x = x[keep]
    y = y[keep]
    std = std[keep] if std.shape == keep.shape else np.full_like(y, math.nan)
    return {
        "x": x,
        "y": y,
        "std": std,
        "final": float(y[-1]) if y.size else math.nan,
        "negative_step_fraction": float(np.mean(np.diff(y) < 0)) if y.size > 1 else math.nan,
        "source": str(cand.metrics),
    }


def eps_sort_key(eps: str) -> tuple[int, str]:
    try:
        return (EPS_ORDER.index(eps), eps)
    except ValueError:
        return (999, eps)


def target_sort_key(target: str) -> tuple[int, str]:
    try:
        return (TARGET_ORDER.index(target), target)
    except ValueError:
        return (999, target)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def draw_eps(eps: str, group: dict[str, dict[str, Candidate]]) -> tuple[Path, list[dict[str, str | float]]]:
    targets = sorted(group, key=target_sort_key)
    rows_out: list[dict[str, str | float]] = []
    nrows = len(targets)
    fig_h = max(4.0, 2.45 * nrows + 1.2)
    fig, axes = plt.subplots(nrows, 1, figsize=(13.5, fig_h), sharex=True)
    if nrows == 1:
        axes = [axes]
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "figure.dpi": 180,
        "savefig.dpi": 240,
    })
    fig.suptitle(f"NS2D optimizer comparison with target loss fixed\n{EPS_PRETTY.get(eps, eps)}", fontsize=16.5, weight="bold")
    for ax, target in zip(axes, targets):
        solid_curves = []
        for method in METHODS:
            cand = group[target].get(method)
            if cand is None:
                continue
            curve = load_curve(cand)
            solid_curves.append(curve)
            x = curve["x"]
            y = curve["y"]
            std = curve["std"]
            lw = 3.2 if method == "steepest_replace" else 2.35
            ax.plot(x, y, color=METHOD_COLORS[method], lw=lw, label=METHOD_LABELS[method])
            if isinstance(std, np.ndarray) and np.isfinite(std).any():
                ax.fill_between(x, y - std, y + std, color=METHOD_COLORS[method], alpha=0.09, linewidth=0)
            rows_out.append({
                "eps": eps,
                "target": target,
                "method": method,
                "final": float(curve["final"]),
                "negative_step_fraction": float(curve["negative_step_fraction"]),
                "source": rel(cand.metrics),
            })
        ax.text(0.012, 0.90, TARGET_LABELS.get(target, target), transform=ax.transAxes, va="top", ha="left", fontsize=11, weight="bold")
        ax.set_ylabel("active target loss")
        ax.set_xlim(0, 100)
        ax.set_ylim(*solid_line_ylim(solid_curves))
        ax.grid(True, color="#d8d8d2", linewidth=0.7, alpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("attack step")
    axes[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.16), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT_DIR / f"{eps}_optimizer_grouped_target_loss_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, rows_out


def update_bundle(pngs: list[Path]) -> None:
    if not BUNDLE.exists():
        return
    dst = BUNDLE / "NS2D" / "optimizer_grouped_loss_curves" / "all_eps"
    dst.mkdir(parents=True, exist_ok=True)
    for png in pngs:
        (dst / png.name).write_bytes(png.read_bytes())
    if REPORT.exists():
        (BUNDLE / REPORT.name).write_text(REPORT.read_text(encoding="utf-8"), encoding="utf-8")
    readme = BUNDLE / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# NS2D And Burgers Cleanstyle Figures\n"
    note = "- `NS2D/optimizer_grouped_loss_curves/all_eps/`: one optimizer comparison figure per available NS2D epsilon/alpha setting.\n"
    if "NS2D/optimizer_grouped_loss_curves/all_eps/" not in text:
        text += "\n" + note
    readme.write_text(text, encoding="utf-8")
    if BUNDLE_TAR.exists():
        BUNDLE_TAR.unlink()
    with tarfile.open(BUNDLE_TAR, "w:gz") as tar:
        tar.add(BUNDLE, arcname=BUNDLE.name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    discovered = discover()
    pngs: list[Path] = []
    all_rows: list[dict[str, str | float]] = []
    for eps in sorted(discovered, key=eps_sort_key):
        png, rows = draw_eps(eps, discovered[eps])
        pngs.append(png)
        all_rows.extend(rows)
    lines = [
        "# NS2D All-Epsilon Optimizer-Grouped Loss Curves - 2026-05-24",
        "",
        "Observed from saved `final_state_outputs.npz` and `per_step_metrics.csv` files only. No model inference, solver rollout, attack update, PyTorch, JAX, or GPU work was run.",
        "",
        "Each PNG fixes one epsilon/alpha setting. Within each PNG, rows are the locally available target loss/mode candidates, and each row overlays the optimizer methods.",
        "",
        "## Figures",
        "",
        "| Epsilon / Alpha | targets | PNG |",
        "|---|---:|---|",
    ]
    for eps in sorted(discovered, key=eps_sort_key):
        png = OUT_DIR / f"{eps}_optimizer_grouped_target_loss_curves.png"
        lines.append(f"| {EPS_PRETTY.get(eps, eps)} | {len(discovered[eps])} | [{png.name}]({rel(png)}) |")
    lines += [
        "",
        "## Notes",
        "",
        "- `eps32_alpha10` and `eps8_alpha2p5` include the extra local `loss3` A/D/W mode variants.",
        "- `eps16_alpha5`, `eps4_alpha1p25`, `eps2_alpha0p625`, and `eps1_alpha0p3125` include the canonical loss targets currently visible in the organized local artifacts.",
        "- `eps160_alpha50` now includes the locally visible completed `loss1/all_w`, `loss2/all_a_target_w`, and `loss3/all_a_target_w` targets.",
        "",
        "## Curve Index",
        "",
        "| eps | target | method | final active loss | negative-step fraction | source |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in all_rows:
        lines.append(f"| {row['eps']} | {TARGET_LABELS.get(str(row['target']), str(row['target']))} | {METHOD_LABELS.get(str(row['method']), str(row['method']))} | {float(row['final']):.6g} | {float(row['negative_step_fraction']):.3g} | `{row['source']}` |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_bundle(pngs)
    print(f"png_count={len(pngs)}")
    for png in pngs:
        print(png)
    print(REPORT)
    if BUNDLE_TAR.exists():
        print(BUNDLE_TAR)


if __name__ == "__main__":
    main()
