# -*- coding: utf-8 -*-
import os
import re
import pickle
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless 环境安全绘图
import matplotlib.pyplot as plt

# ---------------------------
# 1) 扫描 pickle_files 目录
# ---------------------------
def list_all_pickles(base_dir: Path, levels: int | None = None) -> list[dict]:
    """
    Recursively list all .pkl files under base_dir and capture folder names
    between base_dir ('pickle_files') and each file.

    Added:
      - Prints each file, its absolute path, its relative path, and the ancestor
        folders between base_dir and the file.
      - 'levels' lets you show only the last N ancestor folders (None = all).

    Returns a list of dicts (unchanged schema for downstream use):
        {
            "abs_path": Path,
            "rel_parts": [folder, subfolder, ...],  # always ALL ancestors (for later grouping)
            "rel_path": Path,                       # path relative to base_dir
            "stem": str,                            # filename without suffix
        }
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"'pickle_files' not found: {base_dir}")

    results = []
    files = sorted(p for p in base_dir.rglob("*.pkl") if p.is_file())
    print(f"[SCAN] Found {len(files)} .pkl under {base_dir}")

    for p in files:
        rel_path = p.relative_to(base_dir)          # e.g. sub1/sub2/foo.pkl
        rel_parts_all = list(rel_path.parts[:-1])   # all folders between base and file

        # what to print as “ancestors”: last N or all
        if levels is None:
            shown_parts = rel_parts_all
            tag = "all"
        else:
            n = max(0, int(levels))
            shown_parts = rel_parts_all[-n:] if n > 0 else []
            tag = str(n)

        ancestors_str = "/".join(shown_parts) if shown_parts else "(root)"

        print(
            f"  - file        : {rel_path.name}\n"
            f"    abs path    : {p.resolve()}\n"
            f"    rel path    : {rel_path}\n"
            f"    ancestors[{tag}]: {ancestors_str}\n"
        )

        results.append({
            "abs_path": p.resolve(),
            "rel_parts": rel_parts_all,   # keep ALL for later folder-based grouping
            "rel_path": rel_path,
            "stem": p.stem,
        })

    return results


# ---------------------------
# 2) 解析 tuple key
# ---------------------------
_KEY_PATTERNS = {
    "norm": re.compile(r"^norm_([A-Za-z0-9\.]+)$"),
    "index": re.compile(r"^index_(\d+)$"),
    "numsteps": re.compile(r"^numsteps_(\d+)$"),
    "epsilon": re.compile(r"^epsilon_([0-9eE\.\+\-]+)$"),
    "alpha": re.compile(r"^alpha_([0-9eE\.\+\-]+)$"),
}

def parse_tuple_key(tuple_key):
    """
    tuple_key looks like: ("norm_inf", "index_3", "numsteps_100", "epsilon_0.05", "alpha_1.0")
    Returns a dict with parsed values (strings kept for norm; numbers for others).
    """
    info = {"norm": None, "index": None, "numsteps": None, "epsilon": None, "alpha": None}
    for item in tuple_key:
        for name, pat in _KEY_PATTERNS.items():
            m = pat.match(item)
            if m:
                val = m.group(1)
                if name in ("index", "numsteps"):
                    info[name] = int(val)
                elif name in ("epsilon", "alpha"):
                    try:
                        info[name] = float(val)
                    except Exception:
                        info[name] = val
                else:
                    info[name] = val
                break
    return info


# ---------------------------
# 3) 从 grad_records 提取四条 loss 曲线
# ---------------------------
LOSS_KEYS = ("with_solver", "without_solver", "approximate", "approximate_surrogate")

def extract_loss_series(grad_records):
    """
    grad_records: dict with "step_update" -> { "step_0": {...}, "step_1": {...}, ... }
    Each step has "loss_metrics" dict containing:
        - with_solver
        - without_solver
        - approximate
        - approximate_surrogate
    Returns dict: {loss_key: [v0, v1, ...]}, and step_indices [0..N]
    """
    if "step_update" not in grad_records or not isinstance(grad_records["step_update"], dict):
        return {}, []

    steps = []
    for k in grad_records["step_update"].keys():
        m = re.match(r"^step_(\d+)$", k)
        if m:
            steps.append(int(m.group(1)))
    steps = sorted(steps)

    series = {k: [] for k in LOSS_KEYS}
    for s in steps:
        step_key = f"step_{s}"
        step_dict = grad_records["step_update"].get(step_key, {})
        loss_metrics = step_dict.get("loss_metrics", {})
        for lk in LOSS_KEYS:
            v = loss_metrics.get(lk, None)
            # Robust float conversion
            try:
                v = float(v) if v is not None else None
            except Exception:
                v = None
            series[lk].append(v)
    return series, steps


# ---------------------------
# 4) 绘图：两子图、每图三条线
# ---------------------------
def plot_two_subplots(series, steps, figure_title, out_path: Path):
    """
    Left: with_solver / without_solver / approximate
    Right: with_solver / without_solver / approximate_surrogate
    Robust to NaNs: NaN segments are skipped; finite parts are drawn.
    Labels include first/last finite values; subplot titles show first/last step values and NaN count.
    """
    if not steps:
        print("[Skip] Empty steps ->", out_path)
        return

    import numpy as np
    import math

    steps_np = np.asarray(steps, dtype=float)

    def to_nan_array(vals):
        arr = []
        for v in vals:
            try:
                x = float(v) if v is not None else np.nan
            except Exception:
                x = np.nan
            if not math.isfinite(x):
                x = np.nan
            arr.append(x)
        return np.asarray(arr, dtype=float)

    # build arrays with NaNs where invalid
    arr_with     = to_nan_array(series.get("with_solver", []))
    arr_without  = to_nan_array(series.get("without_solver", []))
    arr_approx   = to_nan_array(series.get("approximate", []))
    arr_surr     = to_nan_array(series.get("approximate_surrogate", []))

    # y-lims based on finite values only
    all_finite = np.concatenate([
        arr[np.isfinite(arr)] for arr in [arr_with, arr_without, arr_approx, arr_surr]
        if arr.size > 0
    ]) if steps_np.size else np.array([])
    ymin, ymax = (np.min(all_finite), np.max(all_finite)) if all_finite.size else (None, None)

    def label_first_last(arr, base_label):
        finite_idx = np.where(np.isfinite(arr))[0]
        if finite_idx.size == 0:
            return base_label
        first = arr[finite_idx[0]]
        last  = arr[finite_idx[-1]]
        return f"{base_label} (first={first:.3f}, last={last:.3f})"

    def plot_line(ax, y_arr, base_label, **kwargs):
        # Matplotlib will break the line at NaNs automatically
        if np.isfinite(y_arr).any():
            ax.plot(steps_np, y_arr,
                    label=label_first_last(y_arr, base_label),
                    linewidth=2, **kwargs)

    # helpers for titles: show first/last *step* values (may be NaN) and NaN count
    def fmt_any(v):
        if v is None:
            return "NA"
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "NaN" if math.isnan(v) else ("+Inf" if v > 0 else "-Inf")
        return f"{v:.3f}"

    approx_first = arr_approx[0] if arr_approx.size else None
    approx_last  = arr_approx[-1] if arr_approx.size else None
    approx_nans  = int(np.isnan(arr_approx).sum()) if arr_approx.size else 0

    surr_first = arr_surr[0] if arr_surr.size else None
    surr_last  = arr_surr[-1] if arr_surr.size else None
    surr_nans  = int(np.isnan(arr_surr).sum()) if arr_surr.size else 0

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    fig.suptitle(figure_title, y=0.98)

    def common(ax):
        plot_line(ax, arr_with,    "with solver")
        plot_line(ax, arr_without, "detached", linestyle="--")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True)
        if (ymin is not None) and (ymax is not None) and (ymin < ymax):
            pad = 0.05 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

    # Left: approximate (true)
    ax_left = axes[0]
    common(ax_left)
    plot_line(ax_left, arr_approx, "approximated (true loss)", linestyle="-.")
    ax_left.set_title(
        f"Loss curves (approximated TRUE) | approximated first={fmt_any(approx_first)}, "
        f"last={fmt_any(approx_last)}, NaNs={approx_nans}"
    )
    ax_left.legend(loc="best")

    # Right: approximate_surrogate
    ax_right = axes[1]
    common(ax_right)
    plot_line(ax_right, arr_surr, "approximated (surrogate loss)", linestyle="-.")
    ax_right.set_title(
        f"Loss curves (approximated SURROGATE) | approximated first={fmt_any(surr_first)}, "
        f"last={fmt_any(surr_last)}, NaNs={surr_nans}"
    )
    ax_right.legend(loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# 5) 主流程：扫描->逐文件->逐 key 绘图
# ---------------------------
def main():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir / "pickle_files"
    if not base_dir.exists():
        raise FileNotFoundError(f"'pickle_files' not found next to this script: {base_dir}")

    print(f"[Scan] Base folder: {base_dir}")
    files = list_all_pickles(base_dir)
    print(f"[Found] {len(files)} pickle files")

    out_root = script_dir / "plots"

    for item in files:
        abs_path: Path = item["abs_path"]
        rel_parts = item["rel_parts"]
        stem = item["stem"]

        print(f"\n[Load] {abs_path}")
        try:
            with open(abs_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[Warn] Failed to load: {abs_path}  -> {e}")
            continue

        if not isinstance(data, dict):
            print(f"[Warn] Not a dict at top level -> {abs_path}")
            continue

        # Expecting: { tuple_key: grad_records, ... }
        for tuple_key, grad_records in data.items():
            # Normalize tuple_key to tuple of strings
            if isinstance(tuple_key, (list, tuple)):
                tk = tuple(tuple_key)
            else:
                # If the pickle was saved differently, try to coerce
                tk = (str(tuple_key),)

            # Parse key info for titles / filenames
            info = parse_tuple_key(tk)
            norm = info["norm"]
            idx = info["index"]
            numsteps = info["numsteps"]
            epsilon = info["epsilon"]
            alpha = info["alpha"]

            # Extract series
            series, steps = extract_loss_series(grad_records)
            if not steps:
                print(f"[Skip] No step_update/loss_metrics for key={tk}")
                continue

            # Build figure title (English)
            rel_folder = "/".join(rel_parts) if rel_parts else "."
            fig_title = (
                f"Loss Curves — file: {stem}  |  under: {rel_folder}\n"
                f"norm: {norm} | index: {idx} | steps: {numsteps} | "
                f"epsilon: {epsilon} | alpha: {alpha}"
            )

            # Output path mirrors subfolders under plots/
            out_dir = out_root.joinpath(*rel_parts, stem)
            safe_norm = str(norm).replace("/", "_")
            out_name = (
                f"{stem}__norm={safe_norm}_index={idx}_steps={numsteps}_"
                f"epsilon={epsilon}_alpha={alpha}.png"
            )
            out_path = out_dir / out_name

            plot_two_subplots(series, steps, fig_title, out_path)
            print(f"  -> Saved: {out_path}")

    print("\n[Done] All figures saved under:", out_root)


if __name__ == "__main__":
    main()