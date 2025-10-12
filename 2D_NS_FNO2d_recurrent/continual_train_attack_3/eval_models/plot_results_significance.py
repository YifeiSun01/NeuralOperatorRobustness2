# -*- coding: utf-8 -*-
"""
Plot signed −log10(p) from Welch two-sample t-tests (model vs baseline),
and save wide CSVs whose model-columns MultiIndex includes modes/width/epochs.

What changed vs ΔRMSE version:
- Build a FULL pivot that includes rmse_mean, rmse_std, and n.
- For each dataset row and each model column, compute Welch t against the
  baseline (model_group='baseline'): t, df, two-sided p, then
  y = sign(t)*(-log10(p)).
- Plot y with two symmetric dashed reference lines at ±log10(1/0.05)≈±1.301
  and ±log10(1/0.10)=±1.000 to indicate p=0.05 and p=0.10.
- Keep figure layout / grouping / coloring logic unchanged.

Formulas (Welch t & df; two-sided p) follow standard definitions.
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

# Try SciPy for exact Student's t CDF; otherwise fall back to normal approx
try:
    from scipy.stats import t as student_t
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    from math import erf, sqrt
    def _norm_sf(x):
        # survival function 1 - Phi(x)
        return 0.5 * (1.0 - erf(x / sqrt(2.0)))

# ===================== CLI & PATHS =====================
def make_paths():
    script_dir = Path(__file__).resolve().parent
    return script_dir

def parse_args(script_dir: Path):
    p = argparse.ArgumentParser(
        description="Plot signed −log10(p) (Welch two-sample t-test, model vs baseline) from evaluation CSV (paths relative to script)."
    )
    p.add_argument(
        "--csv",
        type=str,
        default="combined_eval__baseline_NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_n__set_saved_models_expanded.csv",
        help="Relative path to input CSV."
    )
    p.add_argument(
        "--save-root",
        type=str,
        default="plots/models_signed_log10p",
        help="Relative path to output root directory for images and CSV."
    )
    p.add_argument(
        "--pivot-signedlogp-csv-name",
        type=str,
        default="pivot_signed_log10p.csv",
        help="File name for the signed −log10(p) pivot CSV (saved under save-root)."
    )
    p.add_argument(
        "--full-pivot-csv-name",
        type=str,
        default="pivot_full_metrics.csv",
        help="File name for the full-metrics pivot CSV (saved under save-root)."
    )
    p.add_argument(
        "--alpha-lines",
        type=float,
        nargs="*",
        default=[0.05, 0.10],
        help="Alpha levels to render as symmetric dashed lines (default: 0.05 0.10)."
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

# 强制出现在列 MultiIndex 的模型属性（即便只有一个取值也会纳入）
FORCED_MODEL_LEVELS = ["model_group", "model_name", "modes", "width", "epochs"]

# Figure/layout params
FIG_DPI = 160
MAX_FIG_W = 80
MIN_FIG_W = 12
PX_PER_BAR = 0.05

# ---- title wrapping config ----
TITLE_CHARS_PER_LINE = 80
TITLE_MAX_LINES      = 3

# ===================== UTILITIES =====================
def log(msg: str):
    print(msg, flush=True)

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", str(s))
    return s[:230]

def sanitize_title(s: str) -> str:
    return sanitize_filename(s)

import textwrap
def wrap_and_truncate_title(s: str, width: int = TITLE_CHARS_PER_LINE, max_lines: int = TITLE_MAX_LINES) -> str:
    if not s:
        return s
    s = re.sub(r"\s+", " ", str(s)).strip()
    lines = textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .;,:") + " …"
    return "\n".join(lines)

def str_or_none(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)

def as_float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# ===================== MODEL HYPERPARAM PARSER =====================
import re as _re
_re_hp = _re.compile(
    r"(?:^|/|_)"
    r"(?:modes(?P<modes>\d+))?"
    r"(?:_width(?P<width>\d+))?"
    r"(?:_epochs(?P<epochs>\d+))?"
    r"(?:_Tin(?P<Tin>\d+))?"
    r"(?:_(?:T|Tout)(?P<Tout>\d+))?"
    r"(?:_step(?P<step>\d+))?",
    flags=_re.IGNORECASE
)

def _first_str(*vals):
    for v in vals:
        s = str_or_none(v)
        if s: return s
    return None

def parse_hp_from_text(txt: str) -> dict:
    out = {"modes": None, "width": None, "epochs": None, "Tin": None, "Tout": None, "step": None}
    if not txt: return out
    m = _re_hp.search(txt)
    if not m: return out
    gd = m.groupdict()
    for k in out:
        v = gd.get(k)
        out[k] = int(v) if v is not None and v.isdigit() else None
    return out

def ensure_model_hparams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    need_cols = ["modes1","modes2","width","epochs","Tin","Tout","step"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    str_cols = [c for c in ["model_source_dir","model_path","model_name"] if c in df.columns]
    if not str_cols:
        str_cols = [c for c in df.columns if df[c].dtype == "object"]

    for i in range(len(df)):
        row = df.iloc[i]
        have = {k: not pd.isna(row.get(k, np.nan)) for k in ["modes1","modes2","width","epochs","Tin","Tout","step"]}
        if all(have.values()):
            continue
        txt = _first_str(*[row.get(c, None) for c in str_cols])
        hp = parse_hp_from_text(txt or "")
        for k in ["width","epochs","Tin","Tout","step"]:
            if not have[k] and hp.get(k) is not None:
                df.at[i, k] = hp[k]
        if (not have["modes1"] or not have["modes2"]) and (hp.get("modes") is not None):
            df.at[i, "modes1"] = hp["modes"]
            df.at[i, "modes2"] = hp["modes"]

    def _mk_modes_str(r):
        m1 = int(r["modes1"]) if not pd.isna(r["modes1"]) else None
        m2 = int(r["modes2"]) if not pd.isna(r["modes2"]) else None
        if m1 is None and m2 is None:
            return SENTINEL
        if m1 is None: m1 = m2
        if m2 is None: m2 = m1
        return f"{m1}x{m2}"
    df["modes"] = df.apply(_mk_modes_str, axis=1)

    for k in ["width","epochs","Tin","Tout","step"]:
        try:
            df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")
        except Exception:
            pass
    return df

# ===================== PIVOT BUILDING =====================
def _levels_with_variation(df, cols):
    present = [c for c in cols if c in df.columns]
    return [c for c in present if df[c].dropna().nunique() >= 2]

def _fill_keys_with_sentinel(df, keys):
    out = df.copy()
    for c in keys:
        out[c] = out[c].astype('object').where(out[c].notna(), SENTINEL)
    return out

def _force_append_levels(cur_levels, want_levels, df_cols):
    out = list(cur_levels)
    for lv in want_levels:
        if lv in df_cols and lv not in out:
            out.append(lv)
    return out

def build_pivot_multi_metrics(
    df: pd.DataFrame,
    values=('rmse_mean','rmse_std','mae_mean','mae_std','mape_mean','mape_std','n'),
    aggfunc='mean'
) -> pd.DataFrame:
    ds_levels = _levels_with_variation(df, DATASET_COLS)
    ds_levels = [c for c in ds_levels if c not in ('dataset_path', 'n')]
    ds_levels = _force_append_levels(ds_levels, ["type"], df.columns)

    model_levels = _levels_with_variation(df, MODEL_COLS)
    model_levels = _force_append_levels(model_levels, FORCED_MODEL_LEVELS, df.columns)

    key_cols = sorted(set(ds_levels + model_levels))
    df2 = _fill_keys_with_sentinel(df, key_cols)

    if not ds_levels:
        ds_levels = ['__ALL_DATASETS__']; df2['__ALL_DATASETS__'] = '__ALL_DATASETS__'
    if not model_levels:
        model_levels = ['__ALL_MODELS__']; df2['__ALL_MODELS__'] = '__ALL_MODELS__'

    values_present = [v for v in values if v in df.columns]
    if not values_present:
        raise ValueError(f"No requested metric columns found in CSV. Looked for: {values}")

    piv = pd.pivot_table(
        df2,
        index=ds_levels,
        columns=model_levels,
        values=values_present,
        aggfunc=aggfunc,
        observed=False
    ).sort_index().sort_index(axis=1)

    want_metrics = {'rmse_mean','rmse_std','mae_mean','mae_std','mape_mean','mape_std','n'}
    col_names = list(piv.columns.names or [])
    metric_idx = None
    for i, lv in enumerate(piv.columns.levels):
        try: vals = set(map(str, lv))
        except Exception: vals = set(str(v) for v in lv)
        if want_metrics.issubset(vals) or len(want_metrics & vals) >= 3:
            metric_idx = i; break
    if metric_idx is None:
        for i, name in enumerate(col_names):
            try:
                _ = piv.xs('rmse_mean', axis=1, level=name, drop_level=False)
                metric_idx = i; break
            except Exception:
                pass
    if metric_idx is not None:
        col_names[metric_idx] = "metric"
        piv.columns.set_names(col_names, inplace=True)

    if isinstance(piv.index, pd.MultiIndex):
        piv.index.names = ds_levels
    else:
        piv.index.name = ds_levels[0]
    return piv

# ===================== Welch t → p → signed −log10(p) =====================
def _extract_metric_block(pivot_full: pd.DataFrame, metric_name: str):
    col_names = list(pivot_full.columns.names or [])
    metric_level = None
    if "metric" in col_names:
        metric_level = "metric"
    else:
        for name in col_names:
            try:
                _ = pivot_full.xs(metric_name, axis=1, level=name, drop_level=False)
                metric_level = name; break
            except Exception:
                continue
    if metric_level is None:
        raise ValueError("Cannot find the metrics column level in pivot_full.")
    block = pivot_full.xs(metric_name, axis=1, level=metric_level, drop_level=False)
    return block, metric_level

def _ensure_baseline(block: pd.DataFrame, level='model_group') -> pd.DataFrame:
    try:
        return block.xs('baseline', axis=1, level=level, drop_level=False)
    except KeyError as e:
        raise KeyError("No model_group='baseline' found in columns.") from e

def _make_like_rmseblock(block_like: pd.DataFrame, data: np.ndarray, metric_level: str, metric_name_out: str) -> pd.DataFrame:
    # Build a DataFrame with same index/columns; replace the 'metric' level value to new metric_name_out
    out = pd.DataFrame(data, index=block_like.index, columns=block_like.columns)
    # If metric level exists, rename its values to metric_name_out (doesn't change shape)
    if metric_level in (block_like.columns.names or []):
        # Rebuild a MultiIndex with the metric level replaced by metric_name_out
        cols = []
        names = list(block_like.columns.names)
        mi_iter = block_like.columns
        for key in mi_iter:
            if isinstance(key, tuple):
                key = list(key)
                key[names.index(metric_level)] = metric_name_out
                cols.append(tuple(key))
            else:
                cols.append(metric_name_out)
        out.columns = pd.MultiIndex.from_tuples(cols, names=names) if isinstance(block_like.columns, pd.MultiIndex) else pd.Index(cols, name=metric_level)
    return out

def compute_signed_log10p_from_pivot(pivot_full: pd.DataFrame, mean_metric='rmse_mean', std_metric='rmse_std', n_metric='n'):
    # blocks
    rmse_mean_blk, metric_level = _extract_metric_block(pivot_full, mean_metric)
    rmse_std_blk, _             = _extract_metric_block(pivot_full, std_metric)
    n_blk, _                    = _extract_metric_block(pivot_full, n_metric)

    # baseline parts
    base_mean = _ensure_baseline(rmse_mean_blk)  # keep metric level
    base_std  = _ensure_baseline(rmse_std_blk)
    base_n    = _ensure_baseline(n_blk)

    # broadcast for all models
    mu_m = rmse_mean_blk.to_numpy(dtype=float)
    sd_m = rmse_std_blk.to_numpy(dtype=float)
    n_m  = n_blk.to_numpy(dtype=float)

    mu_b = base_mean.to_numpy(dtype=float)
    sd_b = base_std.to_numpy(dtype=float)
    n_b  = base_n.to_numpy(dtype=float)

    # Compute Welch t
    with np.errstate(divide='ignore', invalid='ignore'):
        se2 = (sd_m**2 / n_m) + (sd_b**2 / n_b)
        diff = (mu_m - mu_b)
        tval = diff / np.sqrt(se2)

        # Welch–Satterthwaite df
        num = se2**2
        den = ((sd_m**2 / n_m)**2) / (np.maximum(n_m - 1.0, 1.0)) + ((sd_b**2 / n_b)**2) / (np.maximum(n_b - 1.0, 1.0))
        df   = num / den

    # Two-sided p
    abs_t = np.abs(tval)
    if SCIPY_AVAILABLE:
        p = 2.0 * student_t.sf(abs_t, df)
    else:
        # Normal approximation fallback if SciPy unavailable
        p = 2.0 * _norm_sf(abs_t)

    # Numerical safety
    p = np.clip(p, 1e-300, 1.0)
    signed_logp = np.sign(tval) * (-np.log10(p))

    # Build DataFrames with same shape/columns
    t_df   = _make_like_rmseblock(rmse_mean_blk, tval, metric_level, "welch_t")
    df_df  = _make_like_rmseblock(rmse_mean_blk, df,   metric_level, "welch_df")
    p_df   = _make_like_rmseblock(rmse_mean_blk, p,    metric_level, "p_two_sided")
    s_df   = _make_like_rmseblock(rmse_mean_blk, signed_logp, metric_level, "signed_neglog10_p")

    return s_df, p_df, t_df, df_df

# ===================== (existing) KERNEL / ORDER / PLOTTING (mostly unchanged) =====================
KERNEL_ORDER = ["gaussian", "rbf", "rq", "matern", "periodic", "other"]
KERNEL_HATCH = {"gaussian": "-", "rbf": ".", "rq": "\\", "matern": "/", "periodic": "x", "other": None}
GROUP_ORDER  = ["generalizability", "test", "train", "expanded", "other"]
GROUP_BASE_COLOR = {"test":"#ff7f0e","train":"#2ca02c","expanded":"#9467bd","other":"#7f7f7f"}

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

def canonical_range_tuple(lo, hi):
    lo = as_float_or_nan(lo); hi = as_float_or_nan(hi)
    if np.isnan(lo) or np.isnan(hi):
        return None
    return (lo, hi) if lo <= hi else (hi, lo)

def range_label_from_tuple(rt):
    if rt is None: return "range=unknown"
    lo, hi = rt
    return f"range=({lo:g},{hi:g})"

def extract_range_from_row(idx_names, row_tuple):
    have_vmin = "vmin" in idx_names
    have_vmax = "vmax" in idx_names
    if have_vmin and have_vmax:
        return canonical_range_tuple(row_tuple[idx_names.index("vmin")], row_tuple[idx_names.index("vmax")])
    if "dataset_name" in idx_names:
        dn = str_or_none(row_tuple[idx_names.index("dataset_name")]) or ""
        dn_low = (dn or "").lower()
        m1 = re.search(r"vmax([\-0-9\.]+).*?vmin([\-0-9\.]+)", dn_low)
        m2 = re.search(r"vmin([\-0-9\.]+).*?vmax([\-0-9\.]+)", dn_low)
        m = m1 or m2
        if m:
            return canonical_range_tuple(m.group(1), m.group(2))
    return None

def infer_group_from_row(idx_names, row_tuple):
    if "dataset_group" in idx_names:
        v = str_or_none(row_tuple[idx_names.index("dataset_group")])
        if v: return v
    cand = ""
    if "dataset_name" in idx_names:
        cand = (str_or_none(row_tuple[idx_names.index("dataset_name")]) or "").lower()
    if "generalizability" in cand: return "generalizability"
    if "expanded" in cand: return "expanded"
    if "_train_" in cand or re.search(r"(?:^|_)train(?:_|$)", cand): return "train"
    if "_test_"  in cand or re.search(r"(?:^|_)test(?:_|$)",  cand): return "test"
    return "other"

def build_generalizability_range_palette(pivot: pd.DataFrame, cmap_name="coolwarm"):
    if isinstance(pivot.index, pd.MultiIndex):
        idx_names = pivot.index.names; rows = list(pivot.index)
    else:
        idx_names = [pivot.index.name or "index"]; rows = [(v,) for v in pivot.index.values]
    ranges = []
    for r in rows:
        if infer_group_from_row(idx_names, r) != "generalizability": continue
        rt = extract_range_from_row(idx_names, r)
        ranges.append(rt)
    uniq_tuples = sorted(set(ranges), key=lambda rt: (math.inf, math.inf) if rt is None else (rt[0], rt[1]))
    if not uniq_tuples: return {}
    labels = [range_label_from_tuple(rt) for rt in uniq_tuples]
    cmap   = get_cmap(cmap_name)
    lo, hi = 0.15, 0.85
    n      = len(labels)
    vals   = [0.5] if n == 1 else np.linspace(lo, hi, n)
    return {lab: cmap(v) for lab, v in zip(labels, vals)}

def compute_row_order_and_layout(pivot: pd.DataFrame, range_palette: dict):
    if isinstance(pivot.index, pd.MultiIndex):
        idx_names = pivot.index.names; rows = list(pivot.index)
    else:
        idx_names = [pivot.index.name or "index"]; rows = [(v,) for v in pivot.index.values]

    meta = []
    for i, r in enumerate(rows):
        g  = infer_group_from_row(idx_names, r)
        k  = extract_kernel_from_row(idx_names, r) if g == "generalizability" else "other"
        rt = extract_range_from_row(idx_names, r)   if g == "generalizability" else None
        rl = range_label_from_tuple(rt)             if g == "generalizability" else None
        meta.append({"i": i, "group": g, "kernel": k, "range_tuple": rt, "range_label": rl})

    def range_key(rt):
        if rt is None: return (math.inf, math.inf)
        lo, hi = rt; return (lo, hi)

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
        if not items: continue

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
                        cur_key = key; seg_start_local = j
                    elif key != cur_key:
                        start_index = start_running + seg_start_local
                        end_index   = start_running + j - 1
                        mid = (start_index + end_index) / 2.0
                        rg = cur_key[1] or "range=unknown"
                        rg_short = re.sub(r"^range=", "", rg)
                        gen_sub_ticks.append(mid)
                        gen_sub_labels.append(f"{cur_key[0]}\n{rg_short}")
                        gen_sub_boundaries.append(start_running + j - 0.5)
                        gen_segments.append({"start": start_index,"end": end_index,"kernel": cur_key[0],"range_label": cur_key[1]})
                        cur_key = key; seg_start_local = j
                start_index = start_running + seg_start_local
                end_index   = start_running + len(items) - 1
                mid = (start_index + end_index) / 2.0
                rg = cur_key[1] or "range=unknown"
                rg_short = re.sub(r"^range=", "", rg)
                gen_sub_ticks.append(mid)
                gen_sub_labels.append(f"{cur_key[0]}\n{rg_short}")
                gen_segments.append({"start": start_index,"end": end_index,"kernel": cur_key[0],"range_label": cur_key[1]})
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
    sub_xtick_info = {"ticks": gen_sub_ticks, "labels": gen_sub_labels, "boundaries": gen_sub_boundaries, "segments": gen_segments}
    return row_order, None, tick_pos, tick_lab, boundaries, meta_by_new, sub_xtick_info

# ===== legend helpers & column helpers =====
def _get_level_value_from_col(col_key, col_names, level_name, default="NA"):
    try:
        if isinstance(col_key, tuple) and level_name in col_names:
            idx = col_names.index(level_name); v = col_key[idx]
        else:
            v = default
    except Exception:
        v = default
    v = str(v)
    if v is None or v.strip() == "" or v == str(pd.NA) or v.lower() == "nan" or v == SENTINEL:
        v = default
    return v

def _mwe_subdir_from_col(col_key, col_names):
    modes  = _get_level_value_from_col(col_key, col_names, "modes",  default="NA")
    width  = _get_level_value_from_col(col_key, col_names, "width",  default="NA")
    epochs = _get_level_value_from_col(col_key, col_names, "epochs", default="NA")
    subdir = f"modes{modes}_width{width}_epochs{epochs}"
    return sanitize_filename(subdir)

# ===================== PLOTTING: signed −log10(p) =====================
def plot_all_models_from_pivot(pivot_signed_logp, save_dir=None, alpha_lines=(0.05, 0.10)):
    log(f"[plot] preparing color palette and row layout ...")
    range_palette = build_generalizability_range_palette(pivot_signed_logp, cmap_name="coolwarm")
    row_order, _, tick_pos, tick_lab, boundaries, meta_by_new, sub_xtick_info = compute_row_order_and_layout(
        pivot_signed_logp, range_palette
    )
    pivot_ord = pivot_signed_logp.iloc[row_order, :]

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        log(f"[plot] output directory: {save_dir}")

    # only non-baseline columns
    if isinstance(pivot_ord.columns, pd.MultiIndex):
        col_mask = np.ones(len(pivot_ord.columns), dtype=bool)
        if "model_group" in pivot_ord.columns.names:
            col_mask &= (pivot_ord.columns.get_level_values("model_group") != "baseline")
        cols = [c for c, m in zip(pivot_ord.columns, col_mask) if m]
    else:
        cols = [c for c in pivot_ord.columns if str(c).lower() != "baseline"]

    log(f"[plot] models to plot: {len(cols)}")
    col_level_names = list(pivot_ord.columns.names or [])

    def _png_name_from_model_name(name: str) -> str:
        base = str(name or "").strip()
        base = re.sub(r"\.(pth|pt|ckpt|bin|h5|onnx)$", "", base, flags=re.IGNORECASE)
        base = sanitize_title(base)
        return base + ".png"

    thr_vals = [math.log10(1.0/a) for a in alpha_lines if a > 0]
    thr_vals = sorted(set([round(v, 12) for v in thr_vals]))  # unique & stable

    for ci, c in enumerate(cols, 1):
        series = pivot_ord[c]
        model_name = _get_level_value_from_col(c, col_level_names, "model_name", default="model")
        if model_name in (SENTINEL, "NA", "", None):
            model_path = _get_level_value_from_col(c, col_level_names, "model_path", default="model")
            model_name = os.path.basename(model_path) if model_path not in (SENTINEL, "NA") else str(c)

        log(f"[plot {ci}/{len(cols)}] drawing: model_name={model_name}")

        fig_w = max(20, int(len(series) * 0.06))
        fig, ax = plt.subplots(figsize=(fig_w, 13))

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

        ax.set_xlim(-0.5, len(series) - 0.5)
        ax.margins(x=0)

        # title
        ax.set_title(wrap_and_truncate_title(str(model_name)), fontsize=14, pad=24)

        ax.set_ylabel("signed −log10(p)  (t>0 worse, t<0 better)", fontsize=12)
        ax.axhline(0.0, color="black", lw=1.0, alpha=0.8)
        # symmetric dashed reference lines for each alpha
        for v in thr_vals:
            ax.axhline(+v, color="gray", lw=1.2, ls="--", alpha=0.8)
            ax.axhline(-v, color="gray", lw=1.2, ls="--", alpha=0.8)
            # annotate on the right side
            ax.text(len(series)-0.4, v,  f" p={10**(-v):.2g}", va="bottom", ha="right", fontsize=10, color="gray")
            ax.text(len(series)-0.4, -v, f"-p={10**(-v):.2g}", va="top",    ha="right", fontsize=10, color="gray")

        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.set_yscale('symlog', linthresh=2.0, linscale=2.0)
        plt.subplots_adjust(top=0.78)

        # legend
        legend_elems = []
        raw_range_labels = [m["range_label"] for m in meta_by_new if m["group"] == "generalizability"]
        for rl in sorted({rl for rl in raw_range_labels if rl}):
            display_label = f"generalizability\n{rl}"
            color = range_palette.get(rl, get_cmap("coolwarm")(0.5))
            legend_elems.append(Patch(facecolor=color, edgecolor="none", label=display_label))

        for k in sorted({m["kernel"] for m in meta_by_new if m["group"] == "generalizability"},
                        key=lambda k: (KERNEL_ORDER.index(k) if k in KERNEL_ORDER else len(KERNEL_ORDER))):
            h = KERNEL_HATCH.get(k)
            if h:
                legend_elems.append(Patch(facecolor="white", edgecolor="black", hatch=h, linewidth=0.0,
                                          label=f"kernel={k}"))

        for g in ["test", "train", "expanded"]:
            if any(m["group"] == g for m in meta_by_new):
                legend_elems.append(Patch(facecolor=GROUP_BASE_COLOR[g], edgecolor="none", label=g))

        if legend_elems:
            ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Legend")

        # save
        if save_dir:
            mwe_subdir = _mwe_subdir_from_col(c, col_level_names)
            out_dir_final = Path(save_dir) / mwe_subdir
            out_dir_final.mkdir(parents=True, exist_ok=True)

            fname = _png_name_from_model_name(model_name)
            out_path = out_dir_final / fname
            fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig)
            log(f"   -> saved: {out_path}")
        else:
            plt.close(fig)

def model_col_iter(pivot_signed: pd.DataFrame):
    if isinstance(pivot_signed.columns, pd.MultiIndex):
        col_mask = np.ones(len(pivot_signed.columns), dtype=bool)
        if "model_group" in pivot_signed.columns.names:
            col_mask &= (pivot_signed.columns.get_level_values("model_group") != "baseline")
        cols = [c for c, m in zip(pivot_signed.columns, col_mask) if m]
        for c in cols:
            s = pivot_signed[c]
            parts = []
            for name, val in zip(pivot_signed.columns.names, c if isinstance(c, tuple) else (c,)):
                parts.append(f"{name}={val}")
            title = " | ".join(parts)
            yield c, title, s
    else:
        for c in pivot_signed.columns:
            if str(c).lower() == "baseline":
                continue
            yield c, f"model={c}", pivot_signed[c]

# ===================== FAMILY FILTER (unchanged) =====================
def filter_pivot_by_family_including_others(pivot: pd.DataFrame, family: str) -> pd.DataFrame:
    if not isinstance(pivot.index, pd.MultiIndex) or ("type" not in pivot.index.names):
        raise ValueError("pivot must have a MultiIndex with a 'type' level")

    names = pivot.index.names
    type_lv = pivot.index.get_level_values("type").astype(str).str.lower()

    fam = str(family).lower()
    if fam == "grf":
        mask_family = (type_lv.str.startswith("grf_") & ~type_lv.str.startswith("loggrf_") & ~type_lv.str.startswith("negloggrf_"))
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
        is_others = (dn.str.contains(r"(?:^|_)test(?:_|$)") | dn.str.contains(r"(?:^|_)train(?:_|$)") | dn.str.contains("expanded"))

    mask = (is_gen & mask_family) | is_others
    return pivot.loc[mask]

def run_family_plots(pivot_signed: pd.DataFrame, save_root: Path, alpha_lines=(0.05,0.10)):
    fam_dirs = {
        "GRF":       save_root / "GRF",
        "LogGRF":    save_root / "LogGRF",
        "NegLogGRF": save_root / "NegLogGRF",
    }
    for fam, out_dir in fam_dirs.items():
        sub = filter_pivot_by_family_including_others(pivot_signed, fam)
        if sub.empty:
            log(f"[skip] {fam}: no rows after filtering.")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"[run] {fam}: {len(sub)} rows → {out_dir}")
        plot_all_models_from_pivot(sub, save_dir=str(out_dir), alpha_lines=alpha_lines)

# ===================== MAIN =====================
def main():
    script_dir = make_paths()
    args = parse_args(script_dir)

    csv_path = (script_dir / args.csv).resolve()
    save_root = (script_dir / args.save_root).resolve()
    pivot_signed_csv_out = (save_root / args.pivot_signedlogp_csv_name).resolve()
    full_pivot_csv_out = (save_root / args.full_pivot_csv_name).resolve()

    log(f"[paths] script_dir = {script_dir}")
    log(f"[paths] input CSV = {csv_path}")
    log(f"[paths] save_root  = {save_root}")
    save_root.mkdir(parents=True, exist_ok=True)

    # Load CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    log("[step] reading CSV ...")
    df = pd.read_csv(csv_path, low_memory=False)
    log(f"[info] rows={len(df):,}, columns={len(df.columns)}")

    # Ensure model hyperparams columns exist / are parsed
    log("[step] inferring model hyperparams (modes/width/epochs[/Tin/Tout/step]) if missing ...")
    df = ensure_model_hparams(df)

    # Build full pivot with multiple metrics (must include rmse_mean, rmse_std, n)
    log("[step] building FULL pivot table (rmse_mean/std + n + others) ...")
    pivot_full = build_pivot_multi_metrics(
        df,
        values=('rmse_mean','rmse_std','mae_mean','mae_std','mape_mean','mape_std','n'),
        aggfunc='mean'
    )
    log(f"[info] pivot_full shape = {pivot_full.shape}; columns levels = {pivot_full.columns.names}")
    log("[step] writing FULL pivot CSV ...")
    pivot_full.to_csv(full_pivot_csv_out)
    log(f"[ok] saved FULL pivot to: {full_pivot_csv_out}")

    # Welch t → p → signed −log10(p)
    log("[step] computing Welch t / df / two-sided p / signed −log10(p) against baseline ...")
    signed_df, p_df, t_df, df_df = compute_signed_log10p_from_pivot(
        pivot_full, mean_metric='rmse_mean', std_metric='rmse_std', n_metric='n'
    )

    # Quick sanity: baseline columns should be around 0 for signed −log10(p)
    try:
        max_abs_base = float(signed_df.xs('baseline', axis=1, level='model_group', drop_level=False).abs().max().max())
    except Exception:
        max_abs_base = float("nan")
    log(f"[check] baseline signed −log10(p) -> max|value| = {max_abs_base}")

    # Save outputs
    log("[step] writing pivot CSVs ...")
    signed_df.to_csv(pivot_signed_csv_out)
    p_df.to_csv(save_root / "pivot_p_value.csv")
    t_df.to_csv(save_root / "pivot_welch_t.csv")
    df_df.to_csv(save_root / "pivot_welch_df.csv")
    log(f"[ok] saved signed −log10(p) to: {pivot_signed_csv_out}")

    # Plot family views from signed −log10(p)
    log("[step] plotting by family (GRF/LogGRF/NegLogGRF) ...")
    run_family_plots(signed_df, save_root=save_root, alpha_lines=tuple(args.alpha_lines))

    log("[done] all tasks completed.")

if __name__ == "__main__":
    main()

