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

import numpy as np
from tqdm import tqdm
import matplotlib.animation as animation

def _to_np_1d(x):
    """Robustly convert torch/np/list to 1D numpy array of float."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x)
    # squeeze to 1D if possible
    if x.ndim > 1:
        x = np.squeeze(x)
    return x.astype(float)

def create_adversarial_evolution_gif(grad_records: dict, tuple_key, out_path: Path):
    """
    Create a 7-axes GIF for one config key inside a pickle file:
      1) Perturbed inputs vs original (4 lines: a, a_with, a_without, a_approximate)
      2) Loss progression (with / detached / approximate TRUE)
      3) Original:      G(a) vs g(a)
      4) With solver:   G(a_with) vs g(a_with)
      5) Detached:      G(a_without) vs g(a_without)
      6) Approximate:   G(a_approximate) vs g(a_approximate)
      7) Loss progression (with / detached / approximate SURROGATE)

    GIF saved to out_path (".gif"); parent dirs are created if needed.
    """
    if "step_update" not in grad_records:
        print(f"[Skip] No 'step_update' for key={tuple_key}")
        return

    step_data = grad_records["step_update"]

    # steps in order; we skip 'step_0' for the perturbed input frames but keep it for loss curves
    step_names = sorted([k for k in step_data.keys() if k.startswith("step_")],
                        key=lambda s: int(s.split("_")[1]))
    if not step_names:
        print(f"[Skip] Empty steps for key={tuple_key}")
        return

    # valid steps for frames: step_1 ... step_N
    valid_steps = [s for s in step_names if s != "step_0"]
    if not valid_steps:
        print(f"[Skip] No step_1.. for key={tuple_key}")
        return

    # ---------- Pre-collect time series ----------
    # 1) inputs (per step) - these exist under each step's ["values"]["a_values"]
    a_keys = ["a", "a_with", "a_without", "a_approximate"]

    all_values = {k: [] for k in a_keys}
    # also collect G() and g() curves for 4 cases
    G_keys = ["G(a)", "G(a_with)", "G(a_without)", "G(a_approximate)"]
    g_keys = ["g(a)", "g(a_with)", "g(a_without)", "g(a_approximate)"]
    all_G_values = {k: [] for k in G_keys}
    all_g_values = {k: [] for k in g_keys}

    for s in valid_steps:
        vals = step_data[s].get("values", {})
        a_vals = vals.get("a_values", {})
        G_vals = vals.get("G_values", {})
        g_vals = vals.get("g_values", {})

        for k in a_keys:
            if k in a_vals:
                all_values[k].append(_to_np_1d(a_vals[k]))
        for k in G_keys:
            if k in G_vals:
                all_G_values[k].append(_to_np_1d(G_vals[k]))
        for k in g_keys:
            if k in g_vals:
                all_g_values[k].append(_to_np_1d(g_vals[k]))

    # convert lists -> np arrays
    for k in all_values:
        if len(all_values[k]) == 0:
            print(f"[Warn] Missing {k} sequence for key={tuple_key}")
            return
        all_values[k] = np.vstack(all_values[k])  # shape: [T, X]
    for d in (all_G_values, all_g_values):
        for k in d:
            if len(d[k]) == 0:
                print(f"[Warn] Missing {k} sequence for key={tuple_key}")
                return
            d[k] = np.vstack(d[k])                # shape: [T, X]

    # 2) Loss sequences (include step_0)
    #    Each step dict has "loss_metrics": { with_solver, without_solver, approximate, approximate_surrogate }
    loss_with, loss_detached, loss_true, loss_surr = [], [], [], []
    for s in step_names:
        lm = step_data[s].get("loss_metrics", {})
        loss_with.append(lm.get("with_solver", np.nan))
        loss_detached.append(lm.get("without_solver", np.nan))
        loss_true.append(lm.get("approximate", np.nan))
        loss_surr.append(lm.get("approximate_surrogate", np.nan))
    loss_with = np.asarray(loss_with, dtype=float)
    loss_detached = np.asarray(loss_detached, dtype=float)
    loss_true = np.asarray(loss_true, dtype=float)
    loss_surr = np.asarray(loss_surr, dtype=float)

    # ---------- Axes limits ----------
    # y-lims for inputs
    v_all = np.hstack([all_values[k].ravel() for k in a_keys])
    v_min, v_max = np.nanmin(v_all), np.nanmax(v_all)
    pad = 0.1 * (v_max - v_min + 1e-12)
    adv_ylim = (v_min - pad, v_max + pad)

    # y-lims for G/g
    gg_all = np.hstack([d.ravel() for d in list(all_G_values.values()) + list(all_g_values.values())])
    gg_min, gg_max = np.nanmin(gg_all), np.nanmax(gg_all)
    pad = 0.1 * (gg_max - gg_min + 1e-12)
    gg_ylim = (gg_min - pad, gg_max + pad)

    # y-lims for loss (use all 4)
    ll_all = np.hstack([loss_with, loss_detached, loss_true, loss_surr])
    lmin, lmax = np.nanmin(ll_all), np.nanmax(ll_all)
    if not np.isfinite(lmin) or not np.isfinite(lmax):
        lmin, lmax = 0.0, 1.0
    elif lmin == lmax:
        lmin -= 0.1
        lmax += 0.1
    else:
        pad = 0.1 * (lmax - lmin)
        lmin -= pad
        lmax += pad
    # if non-negative, clamp bottom at 0
    if np.nanmin(ll_all) >= 0:
        lmin = max(0.0, lmin)

    # ---------- Figure & axes (7 subplots) ----------
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 4, wspace=0.35, hspace=0.45)

    # Layout:
    # row 0: [ax1: inputs] [ax3: Gg original] [ax4: Gg with] [ax5: Gg detached]
    # row 1: [ax2: loss TRUE] [ax6: Gg approximate] [ax7: loss SURROGATE] [ax8: (unused)]
    ax1 = fig.add_subplot(gs[0, 0])  # inputs
    ax3 = fig.add_subplot(gs[0, 1])  # G/g original
    ax4 = fig.add_subplot(gs[0, 2])  # G/g with solver
    ax5 = fig.add_subplot(gs[0, 3])  # G/g detached
    ax2 = fig.add_subplot(gs[1, 0])  # loss TRUE
    ax6 = fig.add_subplot(gs[1, 1])  # G/g approximate
    ax7 = fig.add_subplot(gs[1, 2])  # loss SURROGATE
    ax8 = fig.add_subplot(gs[1, 3])  # unused
    ax8.axis("off")

    T, X = all_values["a"].shape
    x_grid = np.linspace(0.0, 1.0, X)

    # ---- ax1 inputs ----
    ln_inp = {
        "a":              ax1.plot(x_grid, all_values["a"][0],              lw=1.5, alpha=0.6, label="Original (a)")[0],
        "a_with":         ax1.plot(x_grid, all_values["a_with"][0],         lw=1.8, alpha=0.9, label="With solver")[0],
        "a_without":      ax1.plot(x_grid, all_values["a_without"][0],      lw=1.8, alpha=0.9, label="Detached")[0],
        "a_approximate":  ax1.plot(x_grid, all_values["a_approximate"][0],  lw=1.8, alpha=0.9, label="Approximate")[0],
    }
    ax1.set_ylim(*adv_ylim)
    ax1.set_title("Perturbed inputs vs Original")
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Value")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # ---- loss lines (TRUE) on ax2 ----
    ln_loss_true = {
        "with":     ax2.plot([0], [loss_with[0]],     lw=2.0, marker="o", ms=2, label="with solver")[0],
        "detached": ax2.plot([0], [loss_detached[0]], lw=2.0, marker="o", ms=2, label="detached")[0],
        "approx":   ax2.plot([0], [loss_true[0]],     lw=2.0, marker="o", ms=2, label="approximate (true)")[0],
    }
    ax2.set_xlim(0, len(step_names)-1)
    ax2.set_ylim(lmin, lmax)
    ax2.set_title("Loss progression (TRUE)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)
    ax2.legend()

    # ---- G/g helper ----
    def _init_Gg(ax, title):
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(*gg_ylim)
        ax.set_xlabel("Position")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        G_line = ax.plot([], [], lw=2.0, label="DeepONet")[0]
        g_line = ax.plot([], [], lw=2.0, label="PDE Solver")[0]
        ttl = ax.set_title(title)
        ax.legend()
        return {"G": G_line, "g": g_line, "title": ttl}

    gg_axes = {
        "orig": _init_Gg(ax3, "Original: DeepONet vs PDE Solver"),
        "with": _init_Gg(ax4, "With solver: DeepONet vs PDE Solver"),
        "det":  _init_Gg(ax5, "Detached: DeepONet vs PDE Solver"),
        "app":  _init_Gg(ax6, "Approximate: DeepONet vs PDE Solver"),
    }

    # ---- loss lines (SURROGATE) on ax7 ----
    ln_loss_surr = {
        "with":     ax7.plot([0], [loss_with[0]],     lw=2.0, marker="o", ms=2, label="with solver")[0],
        "detached": ax7.plot([0], [loss_detached[0]], lw=2.0, marker="o", ms=2, label="detached")[0],
        "approx_s": ax7.plot([0], [loss_surr[0]],     lw=2.0, marker="o", ms=2, label="approximate (surrogate)")[0],
    }
    ax7.set_xlim(0, len(step_names)-1)
    ax7.set_ylim(lmin, lmax)
    ax7.set_title("Loss progression (SURROGATE)")
    ax7.set_xlabel("Step")
    ax7.set_ylabel("Loss")
    ax7.grid(alpha=0.3)
    ax7.legend()

    # ---- update ----
    def _metrics(G, g):
        d = G - g
        rmse = np.sqrt(np.mean(d**2))
        mae  = np.mean(np.abs(d))
        return rmse, mae

    def update(frame: int):
        # frame indexes valid_steps (0..T-1). For loss we include step_0 -> use +1 length.
        t = frame

        # inputs
        ln_inp["a"].set_ydata(all_values["a"][t])
        ln_inp["a_with"].set_ydata(all_values["a_with"][t])
        ln_inp["a_without"].set_ydata(all_values["a_without"][t])
        ln_inp["a_approximate"].set_ydata(all_values["a_approximate"][t])

        # loss TRUE
        xs = np.arange(0, t+2)  # include step_0 .. step_(t+1)
        ln_loss_true["with"].set_data(xs,     loss_with[:t+2])
        ln_loss_true["detached"].set_data(xs, loss_detached[:t+2])
        ln_loss_true["approx"].set_data(xs,   loss_true[:t+2])

        # set G/g curves for four panels
        gg_map = {
            "orig": ("G(a)",            "g(a)"),
            "with": ("G(a_with)",       "g(a_with)"),
            "det":  ("G(a_without)",    "g(a_without)"),
            "app":  ("G(a_approximate)","g(a_approximate)"),
        }
        for key, (Gk, gk) in gg_map.items():
            Gv = all_G_values[Gk][t]
            gv = all_g_values[gk][t]
            gg_axes[key]["G"].set_data(x_grid, Gv)
            gg_axes[key]["g"].set_data(x_grid, gv)
            rmse, mae = _metrics(Gv, gv)
            gg_axes[key]["title"].set_text(
                gg_axes[key]["title"].get_text().split("\n")[0] + f"\nRMSE={rmse:.4f}, MAE={mae:.4f}"
            )

        # loss SURR
        ln_loss_surr["with"].set_data(xs,     loss_with[:t+2])
        ln_loss_surr["detached"].set_data(xs, loss_detached[:t+2])
        ln_loss_surr["approx_s"].set_data(xs, loss_surr[:t+2])

        # super title with boundary flags
        step_name = valid_steps[t]
        rb = step_data[step_name].get("reaches_boundary", {})
        btxt = " | ".join([f"{k.replace('_',' ')}: {v}" for k, v in rb.items()]) if rb else ""
        fig.suptitle(f"Config: {tuple_key} | {step_name} | Reaches boundary: {btxt}", y=0.99, fontsize=12)

        return (
            list(ln_inp.values())
            + list(ln_loss_true.values())
            + [gg_axes[k]["G"] for k in gg_axes]
            + [gg_axes[k]["g"] for k in gg_axes]
            + [gg_axes[k]["title"] for k in gg_axes]
            + list(ln_loss_surr.values())
            + [fig._suptitle]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # filename
    if isinstance(tuple_key, (list, tuple)):
        fname = "gif__" + "_".join(str(x) for x in tuple_key) + ".gif"
    else:
        fname = f"gif__{tuple_key}.gif"
    gif_path = out_path.parent / fname

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, T), interval=300, blit=False)
    with tqdm(total=T, desc=f"Saving {gif_path.name}") as pbar:
        ani.save(gif_path, writer=animation.PillowWriter(fps=3),
                 progress_callback=lambda i, n: pbar.update(1), dpi=100)
    plt.close(fig)
    print(f"✅ Saved GIF: {gif_path}")

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

    out_root = script_dir / "gifs"

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

            dummy_png_path = out_dir / "placeholder.png"  # 仅传目录给函数使用
            create_adversarial_evolution_gif(grad_records, tk, dummy_png_path)
            print(f"  -> Saved: {out_path}")

    print("\n[Done] All figures saved under:", out_root)


if __name__ == "__main__":
    main()