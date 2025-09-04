# -*- coding: utf-8 -*-
"""
Plot ΔRMSE (model - baseline) with grouped bars by dataset groups.

- All paths are resolved RELATIVE TO THIS SCRIPT'S DIRECTORY.
- Progress is printed at every major step (loading CSV, pivoting, subtracting baseline,
  family filtering, per-model plotting, file saves, etc.)
- Headless-friendly (matplotlib Agg backend).

Usage examples (run in the same folder as this script):
  python plot_rmse_minus.py
  python plot_rmse_minus.py --csv data/combined_eval.csv
  python plot_rmse_minus.py --save-root plots/models_rmse_minus
  python plot_rmse_minus.py --csv combined_eval__baseline_NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_n__set_saved_models_expanded.csv

"""

from __future__ import annotations
import os
import re
import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---- headless backend (must be set before importing pyplot) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap

# ===================== CLI & PATHS =====================
def make_paths():
    script_dir = Path(__file__).resolve().parent
    return script_dir

def parse_args(script_dir: Path):
    p = argparse.ArgumentParser(description="Plot ΔRMSE (model - baseline) from evaluation CSV (paths relative to script).")
    p.add_argument(
        "--csv",
        type=str,
        default="combined_eval__baseline_NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_n__set_saved_models_expanded.csv",
        help="Relative path to input CSV (default: a long combined_eval_*.csv)."
    )
    p.add_argument(
        "--save-root",
        type=str,
        default="plots/models_rmse_minus",
        help="Relative path to output root directory for images and CSV (default: plots/models_rmse_minus)."
    )
    p.add_argument(
        "--pivot-csv-name",
        type=str,
        default="pivot_rmse_minus_baseline_from_pivot.csv",
        help="File name for the pivot-minus CSV (saved under save-root)."
    )
    return p.parse_args()

# ===================== CONFIGS =====================
SENTINEL = '__NA__'

DATASET_COLS = [
    'type','solver','nx','N','nu','t','ntimepoints','length_scale','variance','alpha',
    'tau','period','exp_factor','vmax','vmin','dataset_group','dataset_name',
    'dataset_path','n'
]
MODEL_COLS = [
    'model_group','model_name','model_path','model_source_dir','modes1','modes2',
    'width','Tin','Tout','step','epochs','seed','strategy','pct_tag','percent',
    'expanded_stem','expanded_file_name','expanded_file_path','orig_train_dataset',
    'with_replacement','k_requested','N_in_file'
]

GROUP_ORDER = ["generalizability", "test", "train", "expanded", "other"]
GROUP_BASE_COLOR = {
    "test":      "#ff7f0e",  # orange
    "train":     "#2ca02c",  # green
    "expanded":  "#9467bd",  # purple
    "other":     "#7f7f7f",  # gray
}

KERNEL_ORDER = ["gaussian", "rbf", "rq", "matern", "periodic", "other"]
KERNEL_HATCH = {
    "gaussian": "-",
    "rbf": ".",
    "rq": "\\",
    "matern": "/",
    "periodic": "x",
    "other": None,
}

# Figure/layout params
FIG_DPI = 160
MAX_FIG_W = 80
MIN_FIG_W = 12
PX_PER_BAR = 0.05

# ===================== UTILITIES =====================
def log(msg: str):
    print(msg, flush=True)

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", str(s))
    return s[:230]

def sanitize_title(s: str) -> str:
    return sanitize_filename(s)

def str_or_none(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)

def as_float_or_nan(x):
    try:
        f = float(x)
        return f
    except Exception:
        return np.nan

def canonical_range_tuple(lo, hi):
    lo = as_float_or_nan(lo)
    hi = as_float_or_nan(hi)
    if np.isnan(lo) or np.isnan(hi):
        return None
    return (lo, hi) if lo <= hi else (hi, lo)

def range_label_from_tuple(rt):
    if rt is None:
        return "range=unknown"
    lo, hi = rt
    return f"range=({lo:g},{hi:g})"

# ===================== PIVOT BUILDING =====================
def _levels_with_variation(df, cols):
    present = [c for c in cols if c in df.columns]
    return [c for c in present if df[c].dropna().nunique() >= 2]

def _fill_keys_with_sentinel(df, keys):
    out = df.copy()
    for c in keys:
        out[c] = out[c].astype('object').where(out[c].notna(), SENTINEL)
    return out

def build_pivot_rmse_raw(df, value='rmse_mean', aggfunc='mean'):
    ds_levels = _levels_with_variation(df, DATASET_COLS)
    ds_levels = [c for c in ds_levels if c != 'dataset_path']
    model_levels = _levels_with_variation(df, MODEL_COLS)
    model_levels = [c for c in model_levels
                    if c not in {'model_path','model_source_dir',
                                 'expanded_stem','expanded_file_name','expanded_file_path'}]
    key_cols = list(set(ds_levels + model_levels))
    df2 = _fill_keys_with_sentinel(df, key_cols)

    if not ds_levels:
        ds_levels = ['__ALL_DATASETS__']; df2['__ALL_DATASETS__'] = '__ALL_DATASETS__'
    if not model_levels:
        model_levels = ['__ALL_MODELS__']; df2['__ALL_MODELS__'] = '__ALL_MODELS__'

    piv = pd.pivot_table(
        df2, index=ds_levels, columns=model_levels, values=value,
        aggfunc=aggfunc, observed=False
    ).sort_index().sort_index(axis=1)

    if isinstance(piv.index, pd.MultiIndex): piv.index.names = ds_levels
    else: piv.index.name = ds_levels[0]
    if isinstance(piv.columns, pd.MultiIndex): piv.columns.names = model_levels
    else: piv.columns.name = model_levels[0]
    return piv

def subtract_baseline_per_row(pivot: pd.DataFrame, reduce='mean'):
    if 'model_group' not in (pivot.columns.names or []):
        raise ValueError("列层级里没有 'model_group'，无法识别 baseline。")
    base_block = pivot.xs('baseline', axis=1, level='model_group', drop_level=False)
    if reduce == 'mean':
        base_series = base_block.mean(axis=1, skipna=True)
    elif reduce == 'first':
        base_series = base_block.apply(lambda r: r.dropna().iloc[0] if r.notna().any() else pd.NA, axis=1)
    else:
        raise ValueError("reduce 仅支持 'mean' 或 'first'")
    pivot_delta = pivot.subtract(base_series, axis=0)
    return pivot_delta, base_series

def ensure_rmse_minus(pivot: pd.DataFrame):
    if isinstance(pivot.columns, pd.MultiIndex) and "model_group" in pivot.columns.names:
        if "baseline" in pivot.columns.get_level_values("model_group"):
            base_block = pivot.xs("baseline", axis=1, level="model_group", drop_level=False)
            max_abs = float(base_block.abs().max().max())
            if max_abs > 1e-9:
                pivot_minus, _ = subtract_baseline_per_row(pivot, reduce="mean")
                return pivot_minus
    return pivot

# ===================== PARSERS =====================
def parse_kernel_from_text(txt: str):
    txt = (txt or "").lower()
    for kw, canon in [
        ("gaussian", "gaussian"), ("rbf", "rbf"), ("se", "gaussian"),
        ("sqexp", "gaussian"), ("squared_exponential", "gaussian"),
        ("exp_quadratic", "gaussian"), ("expquadratic", "gaussian"),
        ("rq", "rq"), ("rational_quadratic", "rq"), ("rationalquadratic", "rq"),
        ("matern", "matern"), ("periodic", "periodic"),
    ]:
        if kw in txt:
            return canon
    m = re.search(r"(?:negloggrf|loggrf|grf)[\-_]?([a-z0-9]+)", txt)
    if m:
        return parse_kernel_from_text(m.group(1))
    return "other"

def extract_kernel_from_row(idx_names, row_tuple):
    if "type" in idx_names:
        v = row_tuple[idx_names.index("type")]
        k = parse_kernel_from_text(str_or_none(v))
        if k != "other":
            return k
    if "dataset_name" in idx_names:
        v = row_tuple[idx_names.index("dataset_name")]
        k = parse_kernel_from_text(str_or_none(v))
        if k != "other":
            return k
    return "other"

def extract_range_from_row(idx_names, row_tuple):
    have_vmin = "vmin" in idx_names
    have_vmax = "vmax" in idx_names
    if have_vmin and have_vmax:
        rt = canonical_range_tuple(row_tuple[idx_names.index("vmin")],
                                   row_tuple[idx_names.index("vmax")])
        if rt is not None:
            return rt
    if "dataset_name" in idx_names:
        dn = str_or_none(row_tuple[idx_names.index("dataset_name")]) or ""
        dn_low = dn.lower()
        m1 = re.search(r"vmax([\-0-9\.]+).*?vmin([\-0-9\.]+)", dn_low)
        m2 = re.search(r"vmin([\-0-9\.]+).*?vmax([\-0-9\.]+)", dn_low)
        m = m1 or m2
        if m:
            rt = canonical_range_tuple(m.group(1), m.group(2))
            if rt is not None:
                return rt
    return None

def infer_group_from_row(idx_names, row_tuple):
    if "dataset_group" in idx_names:
        v = str_or_none(row_tuple[idx_names.index("dataset_group")])
        if v:
            return v
    cand = ""
    if "dataset_name" in idx_names:
        cand = (str_or_none(row_tuple[idx_names.index("dataset_name")]) or "").lower()
    if "generalizability" in cand:
        return "generalizability"
    if "expanded" in cand:
        return "expanded"
    if "_train_" in cand or re.search(r"(?:^|_)train(?:_|$)", cand):
        return "train"
    if "_test_" in cand or re.search(r"(?:^|_)test(?:_|$)", cand):
        return "test"
    return "other"

# ===================== COLOR / ORDER =====================
def build_generalizability_range_palette(pivot: pd.DataFrame, cmap_name="coolwarm"):
    if isinstance(pivot.index, pd.MultiIndex):
        idx_names = pivot.index.names
        rows = list(pivot.index)
    else:
        idx_names = [pivot.index.name or "index"]
        rows = [(v,) for v in pivot.index.values]

    ranges = []
    for r in rows:
        if infer_group_from_row(idx_names, r) != "generalizability":
            continue
        rt = extract_range_from_row(idx_names, r)
        ranges.append(rt)

    uniq_tuples = sorted(set(ranges), key=lambda rt: (math.inf, math.inf) if rt is None else (rt[0], rt[1]))
    if not uniq_tuples:
        return {}

    labels = [range_label_from_tuple(rt) for rt in uniq_tuples]
    cmap   = get_cmap(cmap_name)
    lo, hi = 0.15, 0.85
    n      = len(labels)
    vals   = [0.5] if n == 1 else np.linspace(lo, hi, n)
    return {lab: cmap(v) for lab, v in zip(labels, vals)}

def compute_row_order_and_layout(pivot: pd.DataFrame, range_palette: dict):
    if isinstance(pivot.index, pd.MultiIndex):
        idx_names = pivot.index.names
        rows = list(pivot.index)
    else:
        idx_names = [pivot.index.name or "index"]
        rows = [(v,) for v in pivot.index.values]

    meta = []
    for i, r in enumerate(rows):
        g  = infer_group_from_row(idx_names, r)
        k  = extract_kernel_from_row(idx_names, r) if g == "generalizability" else "other"
        rt = extract_range_from_row(idx_names, r)   if g == "generalizability" else None
        rl = range_label_from_tuple(rt)             if g == "generalizability" else None
        meta.append({"i": i, "group": g, "kernel": k, "range_tuple": rt, "range_label": rl})

    def range_key(rt):
        if rt is None:
            return (math.inf, math.inf)
        lo, hi = rt
        return (lo, hi)

    order_buckets = {g: [] for g in GROUP_ORDER}
    for m in meta:
        order_buckets.setdefault(m["group"], [])
        order_buckets[m["group"]].append(m)

    row_order = []
    tick_pos, tick_lab, boundaries = [], [], []
    gen_sub_ticks, gen_sub_labels, gen_sub_boundaries = [], [], []
    gen_segments = []
    start_running = 0

    for gi, g in enumerate(GROUP_ORDER):
        items = order_buckets.get(g, [])
        if not items:
            continue

        if g == "generalizability":
            items = sorted(
                items,
                key=lambda m: (
                    KERNEL_ORDER.index(m["kernel"]) if m["kernel"] in KERNEL_ORDER else len(KERNEL_ORDER),
                    range_key(m["range_tuple"]),
                    m["i"],
                ),
            )
            if items:
                cur_key = None
                seg_start_local = None
                for j, m in enumerate(items):
                    key = (m["kernel"], m["range_label"])
                    if cur_key is None:
                        cur_key = key
                        seg_start_local = j
                    elif key != cur_key:
                        start_index = start_running + seg_start_local
                        end_index   = start_running + j - 1
                        mid = (start_index + end_index) / 2.0
                        rg = cur_key[1] or "range=unknown"
                        rg_short = re.sub(r"^range=", "", rg)
                        gen_sub_ticks.append(mid)
                        gen_sub_labels.append(f"{cur_key[0]}\n{rg_short}")
                        gen_sub_boundaries.append(start_running + j - 0.5)
                        gen_segments.append({
                            "start": start_index,
                            "end": end_index,
                            "kernel": cur_key[0],
                            "range_label": cur_key[1],
                        })
                        cur_key = key
                        seg_start_local = j
                # tail
                start_index = start_running + seg_start_local
                end_index   = start_running + len(items) - 1
                mid = (start_index + end_index) / 2.0
                rg = cur_key[1] or "range=unknown"
                rg_short = re.sub(r"^range=", "", rg)
                gen_sub_ticks.append(mid)
                gen_sub_labels.append(f"{cur_key[0]}\n{rg_short}")
                gen_segments.append({
                    "start": start_index,
                    "end": end_index,
                    "kernel": cur_key[0],
                    "range_label": cur_key[1],
                })
        else:
            items = sorted(items, key=lambda m: m["i"])

        cnt = len(items)
        mid = start_running + (cnt - 1) / 2.0
        tick_pos.append(mid)
        tick_lab.append(g)

        row_order.extend([m["i"] for m in items])
        start_running += cnt

        if gi < len(GROUP_ORDER) - 1 and start_running > 0:
            boundaries.append(start_running - 0.5)

    meta_by_new = [meta[i] for i in row_order]
    sub_xtick_info = {
        "ticks": gen_sub_ticks,
        "labels": gen_sub_labels,
        "boundaries": gen_sub_boundaries,
        "segments": gen_segments,
    }
    return row_order, None, tick_pos, tick_lab, boundaries, meta_by_new, sub_xtick_info

# ===================== SUMMARY & PLOTTING =====================
def summarize_good_bad(series: pd.Series, meta_by_new: list):
    overall = {"better": 0, "worse": 0}
    by_kernel = {}
    by_range  = {}
    for (v, m) in zip(series.to_numpy(), meta_by_new):
        if m["group"] != "generalizability":
            continue
        if pd.isna(v):
            continue
        tag = "better" if (v < 0) else "worse"
        overall[tag] += 1
        k = m["kernel"]
        r = m["range_label"] or "range=unknown"
        by_kernel.setdefault(k, {"better": 0, "worse": 0})
        by_range.setdefault(r, {"better": 0, "worse": 0})
        by_kernel[k][tag] += 1
        by_range[r][tag]  += 1
    return overall, by_kernel, by_range

def format_title_with_stats(base_title: str, overall, by_kernel, by_range):
    part_overall = f"Genl better={overall['better']} | worse={overall['worse']}"
    ks = [k for k in KERNEL_ORDER if k in by_kernel] + [k for k in by_kernel.keys() if k not in KERNEL_ORDER]
    part_kernel = "; ".join([f"{k}:{by_kernel[k]['better']}/{by_kernel[k]['worse']}" for k in ks]) or "no-genl"
    rs = sorted(by_range.keys())
    part_range  = "; ".join([f"{r}:{by_range[r]['better']}/{by_range[r]['worse']}" for r in rs]) or "no-genl"
    return f"{base_title}\n{part_overall} | by kernel [{part_kernel}] | by range [{part_range}]"

def plot_all_models_from_pivot(pivot_minus, save_dir=None):
    log(f"[plot] preparing color palette and row layout ...")
    range_palette = build_generalizability_range_palette(pivot_minus, cmap_name="coolwarm")
    row_order, _, tick_pos, tick_lab, boundaries, meta_by_new, sub_xtick_info = compute_row_order_and_layout(
        pivot_minus, range_palette
    )
    pivot_ord = pivot_minus.iloc[row_order, :]

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        log(f"[plot] output directory: {save_dir}")

    # iterate non-baseline model columns
    if isinstance(pivot_ord.columns, pd.MultiIndex):
        col_mask = np.ones(len(pivot_ord.columns), dtype=bool)
        if "model_group" in pivot_ord.columns.names:
            col_mask &= (pivot_ord.columns.get_level_values("model_group") != "baseline")
        cols = [c for c, m in zip(pivot_ord.columns, col_mask) if m]
    else:
        cols = [c for c in pivot_ord.columns if str(c).lower() != "baseline"]

    log(f"[plot] models to plot: {len(cols)}")

    def _format_ratio_str(num, den):
        if den <= 0:
            return "N/A"
        return f"{(num/den)*100:.0f}%"

    for ci, c in enumerate(cols, 1):
        series = pivot_ord[c]
        # title
        if isinstance(pivot_ord.columns, pd.MultiIndex):
            parts = [f"{name}={val}" for name, val in zip(pivot_ord.columns.names, c if isinstance(c, tuple) else (c,))]
            col_title = " | ".join(parts)
        else:
            col_title = f"model={c}"

        log(f"[plot {ci}/{len(cols)}] drawing: {col_title}")

        fig_w = max(20, int(len(series) * 0.06))
        fig, ax = plt.subplots(figsize=(fig_w, 10))

        x = np.arange(len(series))
        y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

        # colors & hatches
        colors, hatches = [], []
        for m in meta_by_new:
            if m["group"] == "generalizability":
                label = m["range_label"] or "range=unknown"
                colors.append(range_palette.get(label, get_cmap("coolwarm")(0.5)))
                hatches.append(KERNEL_HATCH.get(m["kernel"], None))
            else:
                colors.append(GROUP_BASE_COLOR.get(m["group"], GROUP_BASE_COLOR["other"]))
                hatches.append(None)

        bars = ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.0, align="center")
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)

        # lock x range to bars
        ax.set_xlim(-0.5, len(series) - 0.5)
        ax.margins(x=0)

        # title with stats
        overall, by_kernel, by_range = summarize_good_bad(series, meta_by_new)
        base_title = f"{col_title} — ΔRMSE (model - baseline)"
        title_en = format_title_with_stats(base_title, overall, by_kernel, by_range)
        ax.set_title(title_en, fontsize=14)

        # axes
        ax.set_ylabel("ΔRMSE", fontsize=12)
        ax.axhline(0.0, color="black", lw=1.0, alpha=0.8)
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.6)

        # bottom ticks: four major groups
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, fontsize=12, rotation=90, ha="center", va="top")
        for b in boundaries:
            ax.axvline(b, color="black", lw=1.0, ls=":", alpha=0.5)

        # top ticks: generalizability segments with Better/Worse
        segs = (sub_xtick_info or {}).get("segments", [])
        trail = 0
        for i in range(len(meta_by_new) - 1, -1, -1):
            if meta_by_new[i]["group"] == "generalizability":
                break
            trail += 1

        if segs:
            top_ticks, top_labels = [], []
            for seg in segs:
                s, e = int(seg["start"]), int(seg["end"])
                block  = y[s:e+1]
                valid  = ~np.isnan(block)
                den    = int(valid.sum())
                better = int((block[valid] < 0).sum())
                worse  = int((block[valid] > 0).sum())
                rb = _format_ratio_str(better, den)
                rw = _format_ratio_str(worse , den)
                rg = seg.get("range_label") or "range=unknown"
                rg_short = re.sub(r"^range=", "", rg)
                lbl = f"{seg['kernel']}\n{rg_short}\nBetter: {better}/{den} ({rb})\nWorse: {worse}/{den} ({rw})"
                top_labels.append(lbl)
                top_ticks.append((s + e) / 2.0)

            ghost_ticks = []
            if trail > 0:
                start_trail = len(series) - trail
                ghost_ticks = list(range(start_trail, len(series)))

            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())
            ax_top.set_xticks(top_ticks + ghost_ticks)
            ax_top.set_xticklabels(top_labels + [""] * len(ghost_ticks), rotation=90, fontsize=8)
            ax_top.tick_params(axis='x', pad=2, length=3)
            for sp in ["top", "bottom", "left", "right"]:
                ax_top.spines[sp].set_visible(False)

            inner_boundaries = [segs[i]["start"] - 0.5 for i in range(1, len(segs))]
            for vb in inner_boundaries:
                ax.axvline(vb, color="gray", lw=0.8, ls="--", alpha=0.6)

        # symmetric y-limits
        finite = pd.Series(y).dropna().to_numpy()
        ymax = float(np.nanmax(np.abs(finite))) * 1.15 if finite.size else 1.0
        ax.set_ylim(-ymax if ymax > 0 else -1.0, ymax if ymax > 0 else 1.0)

        # legend
        legend_elems = []
        range_labels_in_plot = [m["range_label"] for m in meta_by_new if m["group"] == "generalizability"]
        for r in sorted({rl for rl in range_labels_in_plot if rl}):
            p = Patch(facecolor=range_palette.get(r, get_cmap("coolwarm")(0.5)))
            p.set_label(r)
            legend_elems.append(p)
        for k in sorted({m["kernel"] for m in meta_by_new if m["group"] == "generalizability"},
                        key=lambda k: (KERNEL_ORDER.index(k) if k in KERNEL_ORDER else len(KERNEL_ORDER))):
            h = KERNEL_HATCH.get(k)
            if h:
                p = Patch(facecolor="white", edgecolor="black", hatch=h, linewidth=0.0)
                p.set_label(f"kernel={k}")
                legend_elems.append(p)
        for g in ["test", "train", "expanded"]:
            if any(m["group"] == g for m in meta_by_new):
                legend_elems.append(Patch(facecolor=GROUP_BASE_COLOR[g], edgecolor="none", label=g))
        if legend_elems:
            ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Legend")

        plt.tight_layout()
        if save_dir:
            out_path = Path(save_dir) / f"{sanitize_title(col_title)}.png"
            fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig)
            log(f"   -> saved: {out_path}")
        else:
            plt.close(fig)

def model_col_iter(pivot_minus: pd.DataFrame):
    if isinstance(pivot_minus.columns, pd.MultiIndex):
        col_mask = np.ones(len(pivot_minus.columns), dtype=bool)
        if "model_group" in pivot_minus.columns.names:
            col_mask &= (pivot_minus.columns.get_level_values("model_group") != "baseline")
        cols = [c for c, m in zip(pivot_minus.columns, col_mask) if m]
        for c in cols:
            s = pivot_minus[c]
            parts = []
            for name, val in zip(pivot_minus.columns.names, c if isinstance(c, tuple) else (c,)):
                parts.append(f"{name}={val}")
            title = " | ".join(parts)
            yield c, title, s
    else:
        for c in pivot_minus.columns:
            if str(c).lower() == "baseline":
                continue
            yield c, f"model={c}", pivot_minus[c]

# ===================== FAMILY FILTER =====================
def filter_pivot_by_family_including_others(pivot: pd.DataFrame, family: str) -> pd.DataFrame:
    if not isinstance(pivot.index, pd.MultiIndex) or ("type" not in pivot.index.names):
        raise ValueError("pivot must have a MultiIndex with a 'type' level")

    names = pivot.index.names
    type_lv = pivot.index.get_level_values("type").astype(str).str.lower()

    fam = str(family).lower()
    if fam == "grf":
        mask_family = (
            type_lv.str.startswith("grf_")
            & ~type_lv.str.startswith("loggrf_")
            & ~type_lv.str.startswith("negloggrf_")
        )
    elif fam == "loggrf":
        mask_family = type_lv.str.startswith("loggrf_")
    elif fam == "negloggrf":
        mask_family = type_lv.str.startswith("negloggrf_")
    else:
        raise ValueError("family must be one of: 'GRF', 'LogGRF', 'NegLogGRF'")

    if "dataset_group" in names:
        dg = pivot.index.get_level_values("dataset_group").astype(str).str.lower()
        is_gen = (dg == "generalizability")
        is_others = dg.isin(["test", "train", "expanded"])
    else:
        if "dataset_name" in names:
            dn = pivot.index.get_level_values("dataset_name").astype(str).str.lower()
        else:
            dn = pd.Index([""] * len(pivot))
        is_gen = dn.str.contains("generalizability")
        is_others = (
            dn.str.contains(r"(?:^|_)test(?:_|$)")
            | dn.str.contains(r"(?:^|_)train(?:_|$)")
            | dn.str.contains("expanded")
        )

    mask = (is_gen & mask_family) | is_others
    return pivot.loc[mask]

def run_family_plots(pivot_minus: pd.DataFrame, save_root: Path):
    fam_dirs = {
        "GRF":       save_root / "GRF",
        "LogGRF":    save_root / "LogGRF",
        "NegLogGRF": save_root / "NegLogGRF",
    }
    for fam, out_dir in fam_dirs.items():
        sub = filter_pivot_by_family_including_others(pivot_minus, fam)
        if sub.empty:
            log(f"[skip] {fam}: no rows after filtering.")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"[run] {fam}: {len(sub)} rows → {out_dir}")
        plot_all_models_from_pivot(sub, save_dir=str(out_dir))

# ===================== MAIN =====================
def main():
    script_dir = make_paths()
    args = parse_args(script_dir)

    csv_path = (script_dir / args.csv).resolve()
    save_root = (script_dir / args.save_root).resolve()
    pivot_csv_out = (save_root / args.pivot_csv_name).resolve()

    log(f"[paths] script_dir = {script_dir}")
    log(f"[paths] input CSV = {csv_path}")
    log(f"[paths] save_root  = {save_root}")
    save_root.mkdir(parents=True, exist_ok=True)

    # Load CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    log("[step] reading CSV ...")
    df = pd.read_csv(csv_path)
    log(f"[info] rows={len(df):,}, columns={len(df.columns)}")

    # Build pivot of raw RMSE
    log("[step] building pivot table (rmse_mean) ...")
    pivot_rmse = build_pivot_rmse_raw(df, value='rmse_mean', aggfunc='mean')
    log(f"[info] pivot shape = {pivot_rmse.shape}")

    # Subtract baseline per row
    log("[step] subtracting baseline per row ...")
    pivot_minus, baseline_series = subtract_baseline_per_row(pivot_rmse, reduce='mean')
    log("[check] baseline columns after subtraction -> max|value| = "
        f"{float(pivot_minus.xs('baseline', axis=1, level='model_group', drop_level=False).abs().max().max()) if 'baseline' in pivot_minus.columns.get_level_values('model_group') else 'N/A'}")

    # Save pivot_minus CSV
    log("[step] writing pivot-minus CSV ...")
    pivot_minus.to_csv(pivot_csv_out)
    log(f"[ok] saved pivot-minus CSV to: {pivot_csv_out}")

    # Run family plots (always keep test/train/expanded)
    log("[step] plotting by family (GRF/LogGRF/NegLogGRF) ...")
    run_family_plots(pivot_minus, save_root=save_root)

    log("[done] all tasks completed.")

if __name__ == "__main__":
    main()
