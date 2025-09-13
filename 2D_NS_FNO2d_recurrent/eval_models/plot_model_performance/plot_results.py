#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot ΔRMSE (model - baseline) with grouped bars by dataset groups,
and save wide CSVs whose model-columns MultiIndex includes modes/width/epochs.
Also, split output directories by alpha/epsilon for attack settings and
include strategy(add/replace) + percentage (25/50/75/100) everywhere.

- 从 model_path / model_source_dir / model_name 解析：
    * 模型端攻击参数（mdl_alpha / mdl_epsilon）
    * 策略与比例（strategy=add|replace / pct_tag=25pct|... / percent=0~1）
- 保存结构：
    {save_root}/modes{modes}_width{width}_epochs{epochs}/alpha{a}_epsilon{e}/
        {strategy}_{pct}/model_group_..._strategy_..._pct_..._k_requested_..._alpha{a}_epsilon{e}.png
  若未检测到攻击参数（如 baseline），则第二层目录为 clean/

Usage:
  python plot_rmse_minus.py \
    --csv path/to/combined_eval.csv \
    --save-root plots/models_rmse_minus \
    --pivot-csv-name pivot_rmse_minus_baseline_from_pivot.csv \
    --full-pivot-csv-name pivot_full_metrics.csv
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
        help="File name for the ΔRMSE pivot CSV (saved under save-root)."
    )
    p.add_argument(
        "--full-pivot-csv-name",
        type=str,
        default="pivot_full_metrics.csv",
        help="File name for the full-metrics pivot CSV (saved under save-root)."
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
    'with_replacement','k_requested','N_in_file',
    # 新增：模型端攻击参数（从路径里解析）
    'mdl_alpha','mdl_epsilon'
]

# —— 强制出现在列 MultiIndex 的模型属性（即便只有一个取值也会纳入） —— #
FORCED_MODEL_LEVELS = [
    "model_group", "model_name", "modes", "width", "epochs",
    # 确保命名/标题可用
    "strategy", "pct_tag", "percent", "k_requested",
    # 也把 α/ε 强制入列，便于检查/切片（即使后面我们也做了回退解析）
    "mdl_alpha", "mdl_epsilon"
]

# Figure/layout params
FIG_DPI = 160

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

def split_center_for_title(s: str, window: int = 20) -> str:
    if not s or len(s) < 2:
        return s
    mid = len(s) // 2
    seps_priority = [('/\\',), ('_-',)]
    def nearest_break(chars: str):
        best_pos, best_dist = None, None
        lo = max(1, mid - window)
        hi = min(len(s) - 1, mid + window)
        for i in range(lo, hi + 1):
            if s[i] in chars:
                d = abs(i - mid)
                if best_dist is None or d < best_dist:
                    best_pos, best_dist = i, d
        return best_pos
    for (chars,) in seps_priority:
        pos = nearest_break(chars)
        if pos is not None:
            return s[:pos] + "\n" + s[pos+1:]
    return s[:mid] + "\n" + s[mid:]

def str_or_none(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)

def as_float_or_nan(x):
    try:
        f = float(x); return f
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

# ===================== 模型超参解析 =====================
_re_hp = re.compile(
    r"(?:^|/|_)"
    r"(?:modes(?P<modes>\d+))?"
    r"(?:_width(?P<width>\d+))?"
    r"(?:_epochs(?P<epochs>\d+))?"
    r"(?:_Tin(?P<Tin>\d+))?"
    r"(?:_(?:T|Tout)(?P<Tout>\d+))?"
    r"(?:_step(?P<step>\d+))?",
    flags=re.IGNORECASE
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
        out[k] = int(v) if v is not None and str(v).isdigit() else None
    return out

# ===================== 解析 α/ε & 策略/比例 =====================
_re_alpha = re.compile(r"(?:^|[\/_\-])alpha([\-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
_re_eps   = re.compile(r"(?:^|[\/_\-])epsilon([\-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
_re_pct   = re.compile(r"(\d{2,3})pct", re.IGNORECASE)
_re_strat = re.compile(r"(?:^|/)(add|replace)(?:/|$)", re.IGNORECASE)

def parse_attack_from_text(txt: str) -> dict:
    s = (txt or "")
    out = {"mdl_alpha": None, "mdl_epsilon": None}
    m1 = _re_alpha.search(s)
    m2 = _re_eps.search(s)
    if m1:
        try: out["mdl_alpha"] = float(m1.group(1))
        except: out["mdl_alpha"] = m1.group(1)
    if m2:
        try: out["mdl_epsilon"] = float(m2.group(1))
        except: out["mdl_epsilon"] = m2.group(1)
    return out

def parse_strategy_pct_from_text(txt: str) -> dict:
    s = (txt or "")
    out = {"strategy": None, "pct_tag": None, "percent": None}
    m_pct = _re_pct.search(s)
    m_st  = _re_strat.search(s)
    if m_pct:
        tag = f"{m_pct.group(1)}pct"
        out["pct_tag"] = tag
        try:
            out["percent"] = float(m_pct.group(1)) / 100.0
        except:
            out["percent"] = None
    if m_st:
        out["strategy"] = m_st.group(1).lower()
    return out

def ensure_model_hparams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns: modes1,modes2,width,epochs,Tin,Tout,step, modes(string 'm1xm2'),
    并从 model_source_dir/model_path/model_name 中解析：
      mdl_alpha/mdl_epsilon、strategy/pct_tag/percent（缺失时补齐）。
    """
    df = df.copy()
    need_cols = ["modes1","modes2","width","epochs","Tin","Tout","step",
                 "mdl_alpha","mdl_epsilon","strategy","pct_tag","percent"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    str_cols = [c for c in ["model_source_dir","model_path","model_name"] if c in df.columns]
    if not str_cols:
        str_cols = [c for c in df.columns if df[c].dtype == "object"]

    for i in range(len(df)):
        row = df.iloc[i]
        have = {
            "modes1": not pd.isna(row.get("modes1", np.nan)),
            "modes2": not pd.isna(row.get("modes2", np.nan)),
            "width":  not pd.isna(row.get("width",  np.nan)),
            "epochs": not pd.isna(row.get("epochs", np.nan)),
            "Tin":    not pd.isna(row.get("Tin",    np.nan)),
            "Tout":   not pd.isna(row.get("Tout",   np.nan)),
            "step":   not pd.isna(row.get("step",   np.nan)),
            "mdl_alpha":    not pd.isna(row.get("mdl_alpha",    np.nan)),
            "mdl_epsilon":  not pd.isna(row.get("mdl_epsilon",  np.nan)),
            "strategy":     str_or_none(row.get("strategy", None)) not in (None,"",str(pd.NA)),
            "pct_tag":      str_or_none(row.get("pct_tag", None)) not in (None,"",str(pd.NA)),
            "percent":      not pd.isna(row.get("percent", np.nan)),
        }
        if all(have.values()):
            continue

        txt = _first_str(*[row.get(c, None) for c in str_cols])

        # 解析宽高/epochs等
        hp = parse_hp_from_text(txt or "")
        for k in ["width","epochs","Tin","Tout","step"]:
            if not have[k] and hp.get(k) is not None:
                df.at[i, k] = hp[k]
        if (not have["modes1"] or not have["modes2"]) and (hp.get("modes") is not None):
            df.at[i, "modes1"] = hp["modes"]
            df.at[i, "modes2"] = hp["modes"]

        # 解析 α/ε
        atk = parse_attack_from_text(txt or "")
        for k in ["mdl_alpha","mdl_epsilon"]:
            if not have[k] and (atk.get(k) is not None):
                df.at[i, k] = atk[k]

        # 解析 strategy / pct
        sp = parse_strategy_pct_from_text(txt or "")
        if not have["strategy"] and sp.get("strategy") is not None:
            df.at[i, "strategy"] = sp["strategy"]
        if not have["pct_tag"] and sp.get("pct_tag") is not None:
            df.at[i, "pct_tag"] = sp["pct_tag"]
        if not have["percent"] and sp.get("percent") is not None:
            df.at[i, "percent"] = sp["percent"]

    # Build 'modes' string like "64x64"
    def _mk_modes_str(r):
        m1 = int(r["modes1"]) if not pd.isna(r["modes1"]) else None
        m2 = int(r["modes2"]) if not pd.isna(r["modes2"]) else None
        if m1 is None and m2 is None:
            return SENTINEL
        if m1 is None: m1 = m2
        if m2 is None: m2 = m1
        return f"{m1}x{m2}"
    df["modes"] = df.apply(_mk_modes_str, axis=1)

    # cast numeric
    for k in ["width","epochs","Tin","Tout","step"]:
        try:
            df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")
        except Exception:
            pass
    for k in ["mdl_alpha","mdl_epsilon","percent"]:
        try:
            df[k] = pd.to_numeric(df[k], errors="coerce")
        except Exception:
            pass

    return df

# ===================== PIVOT BUILDING =====================
def _levels_with_variation(df, cols):
    present = [c for c in cols if c in df.columns]
    return [c for c in present if df[c].dropna().nunique() >= 2]

def _fill_keys_with_sentinel(df: pd.DataFrame, keys):
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
    values=('rmse_mean','rmse_std','mae_mean','mae_std','mape_mean','mape_std'),
    aggfunc='mean'
) -> pd.DataFrame:
    """
    从明细 df 构建多指标（rmse/mae/mape 的 mean/std）透视表。
    列层级里会包含：可变的模型维度（含 mdl_alpha/mdl_epsilon/strategy/pct_tag/percent 若有变化），
    并强制加入 FORCED_MODEL_LEVELS。
    """
    ds_levels = _levels_with_variation(df, DATASET_COLS)
    ds_levels = [c for c in ds_levels if c != 'dataset_path']  # 避免超宽列进入索引
    model_levels = _levels_with_variation(df, MODEL_COLS)
    model_levels = _force_append_levels(model_levels, FORCED_MODEL_LEVELS, df.columns)

    key_cols = sorted(set(ds_levels + model_levels))
    df2 = _fill_keys_with_sentinel(df, key_cols)

    if not ds_levels:
        ds_levels = ['__ALL_DATASETS__']; df2['__ALL_DATASETS__'] = '__ALL_DATASETS__'
    if not model_levels:
        model_levels = ['__ALL_MODELS__']; df2['__ALL_MODELS__'] = '__ALL_MODELS__'

    piv = pd.pivot_table(
        df2,
        index=ds_levels,
        columns=model_levels,
        values=list(values),
        aggfunc=aggfunc,
        observed=False
    ).sort_index().sort_index(axis=1)

    # —— 识别 metrics 层级并命名为 "metric" —— #
    want_metrics = {'rmse_mean','rmse_std','mae_mean','mae_std','mape_mean','mape_std'}
    col_names = list(piv.columns.names or [])
    metric_idx = None
    for i, lv in enumerate(piv.columns.levels):
        try:
            vals = set(map(str, lv))
        except Exception:
            vals = set(str(v) for v in lv)
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

def subtract_baseline_rmse(pivot_full: pd.DataFrame, metric: str = 'rmse_mean'):
    """
    从多指标透视表 pivot_full 中抽取指定 metric（默认 rmse_mean），
    并按行减去 baseline（model_group=baseline）的平均值。
    """
    col_names = list(pivot_full.columns.names or [])
    if "model_group" not in col_names:
        raise ValueError("列层级里没有 'model_group'，无法识别 baseline。")

    metric_level = None
    if "metric" in col_names:
        metric_level = "metric"
    else:
        for name in col_names:
            try:
                _ = pivot_full.xs(metric, axis=1, level=name, drop_level=False)
                metric_level = name; break
            except Exception:
                continue
    if metric_level is None:
        raise ValueError("找不到 metrics 所在的列层级；请检查 pivot.columns.names 与 metrics 列是否存在。")

    try:
        rmse_block = pivot_full.xs(metric, axis=1, level=metric_level, drop_level=False)
    except KeyError as e:
        raise KeyError(f"在 metrics 层级 '{metric_level}' 下找不到指标 '{metric}'。") from e

    try:
        base_block = rmse_block.xs('baseline', axis=1, level='model_group', drop_level=False)
    except KeyError as e:
        raise KeyError("列中找不到 model_group='baseline' 的列，请确认 baseline 是否在输入数据中。") from e

    base_series = base_block.mean(axis=1, skipna=True)
    pivot_delta = rmse_block.subtract(base_series, axis=0)
    return pivot_delta, base_series

# ===================== KERNEL / ORDER / PLOTTING =====================
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

# ===== 列层级值 & 目录名工具 =====
def _get_level_value_from_col(col_key, col_names, level_name, default="NA"):
    try:
        if isinstance(col_key, tuple) and level_name in col_names:
            idx = col_names.index(level_name)
            v = col_key[idx]
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

def _alpha_eps_values_from_col(col_key, col_names):
    """优先从列层级取 mdl_alpha/mdl_epsilon；缺失则回退到 model_path 里再解析；都没有 → ('clean', None, None)"""
    def _val(level):
        v = _get_level_value_from_col(col_key, col_names, level, default="NA")
        return None if v in ("NA", str(pd.NA), "nan", "None", "") else v

    a = _val("mdl_alpha")
    e = _val("mdl_epsilon")

    if (a is None or e is None) and ("model_path" in (col_names or [])):
        raw_path = _get_level_value_from_col(col_key, col_names, "model_path", default="")
        atk = parse_attack_from_text(raw_path)
        if a is None and (atk.get("mdl_alpha") is not None): a = atk["mdl_alpha"]
        if e is None and (atk.get("mdl_epsilon") is not None): e = atk["mdl_epsilon"]

    if (a is None) or (e is None):
        return "clean", None, None
    return f"alpha{a}_epsilon{e}", a, e

def _alpha_eps_subdir_from_col(col_key, col_names):
    subdir, _, _ = _alpha_eps_values_from_col(col_key, col_names)
    return sanitize_filename(subdir)

def _strategy_pct_from_col(col_key, col_names, default_strategy="NA", default_pct="NA"):
    """从列层级优先取 strategy/pct_tag/percent；没有则回退从 model_path 解析。返回 (subdir_str, strategy, pct_display)"""
    strategy = _get_level_value_from_col(col_key, col_names, "strategy", default_strategy)
    pct_tag  = _get_level_value_from_col(col_key, col_names, "pct_tag",  "NA")
    percent  = _get_level_value_from_col(col_key, col_names, "percent",  "NA")

    if pct_tag != "NA":
        pct_display = pct_tag
    else:
        try:
            pct_display = f"{int(round(float(percent) * 100))}pct" if percent not in ("NA", "", "nan") else "NA"
        except Exception:
            pct_display = "NA"

    # 如果仍为 NA，尝试从路径解析
    if (strategy == "NA" or pct_display == "NA") and ("model_path" in (col_names or [])):
        raw_path = _get_level_value_from_col(col_key, col_names, "model_path", default="")
        sp = parse_strategy_pct_from_text(raw_path)
        if strategy == "NA" and sp.get("strategy"):
            strategy = sp["strategy"]
        if pct_display == "NA" and sp.get("pct_tag"):
            pct_display = sp["pct_tag"]

    return sanitize_filename(f"{strategy}_{pct_display}"), strategy, pct_display

# ===================== 绘图（含 α/ε + strategy/percentage） =====================
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

    col_level_names = list(pivot_ord.columns.names or [])

    for ci, c in enumerate(cols, 1):
        series = pivot_ord[c]

        # 父目录：modes / width / epochs
        mwe_subdir = _mwe_subdir_from_col(c, col_level_names)
        # 第二层目录：alpha / epsilon 或 clean
        alpha_eps_subdir, alpha_val, eps_val = _alpha_eps_values_from_col(c, col_level_names)
        # 第三层目录：strategy / pct
        strategy_pct_subdir, strategy, pct_display = _strategy_pct_from_col(c, col_level_names)

        # 组图
        log(f"[plot {ci}/{len(cols)}] drawing: {mwe_subdir}/{alpha_eps_subdir}/{strategy_pct_subdir}")
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

        # 标题（带 modes/width/epochs + alpha/epsilon + strategy/pct）
        title_str = f"{mwe_subdir}/{alpha_eps_subdir}/{strategy_pct_subdir} | ΔRMSE (model - baseline)"
        ax.set_title(wrap_and_truncate_title(title_str), fontsize=14, pad=24)

        ax.set_ylabel("ΔRMSE", fontsize=12)
        ax.axhline(0.0, color="black", lw=1.0, alpha=0.8)
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.6)
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

        # 保存
        if save_dir:
            out_dir_final = Path(save_dir) / mwe_subdir / alpha_eps_subdir / strategy_pct_subdir
            out_dir_final.mkdir(parents=True, exist_ok=True)

            # 文件名核心：含 group / strategy / pct / kreq / alpha / epsilon
            group    = _get_level_value_from_col(c, col_level_names, "model_group", "NA")
            kreq     = _get_level_value_from_col(c, col_level_names, "k_requested", "NA")

            fname_core = (
                f"model_group_{group}"
                f"_strategy_{strategy}"
                f"_pct_{pct_display}"
                f"_k_requested_{kreq}"
            )
            if alpha_val is None or eps_val is None:
                fname_core += "_alpha_clean"
            else:
                fname_core += f"_alpha{alpha_val}_epsilon{eps_val}"

            fname = sanitize_title(fname_core) + ".png"
            out_path = out_dir_final / fname
            fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig)
            log(f"   -> saved: {out_path}")
        else:
            plt.close(fig)

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

    # Ensure model hyperparams & parsed fields
    log("[step] inferring model hyperparams (modes/width/epochs[/Tin/Tout/step] + mdl_alpha/mdl_epsilon + strategy/pct) ...")
    df = ensure_model_hparams(df)

    # Build full pivot with multiple metrics
    log("[step] building FULL pivot table (rmse/mae/mape mean&std) ...")
    pivot_full = build_pivot_multi_metrics(
        df,
        values=('rmse_mean','rmse_std','mae_mean','mae_std','mape_mean','mape_std'),
        aggfunc='mean'
    )
    log(f"[info] pivot_full shape = {pivot_full.shape}; columns levels = {pivot_full.columns.names}")
    log("[step] writing FULL pivot CSV ...")
    pivot_full.to_csv(full_pivot_csv_out)
    log(f"[ok] saved FULL pivot to: {full_pivot_csv_out}")

    # Subtract baseline on rmse_mean only
    log("[step] subtracting baseline per row on rmse_mean ...")
    pivot_minus, base_series = subtract_baseline_rmse(pivot_full, metric='rmse_mean')

    # Quick sanity: baseline columns after subtraction should be ~0
    try:
        max_abs = float(pivot_minus.xs('baseline', axis=1, level='model_group', drop_level=False).abs().max().max())
    except Exception:
        max_abs = float("nan")
    log(f"[check] baseline columns after subtraction -> max|value| = {max_abs}")

    # Save ΔRMSE pivot
    log("[step] writing ΔRMSE pivot CSV ...")
    pivot_minus.to_csv(pivot_csv_out)
    log(f"[ok] saved ΔRMSE pivot to: {pivot_csv_out}")

    # Plot family views from ΔRMSE
    log("[step] plotting by family (GRF/LogGRF/NegLogGRF) ...")
    run_family_plots(pivot_minus, save_root=save_root)

    log("[done] all tasks completed.")

if __name__ == "__main__":
    main()
