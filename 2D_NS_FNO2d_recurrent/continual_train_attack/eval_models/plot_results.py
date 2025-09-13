# -*- coding: utf-8 -*-
"""
Plot ΔRMSE (model - baseline) with grouped bars by dataset groups,
and save wide CSVs whose model-columns MultiIndex includes modes/width/epochs.

Changes:
- Parse model hyperparams (modes/width/epochs[/Tin/Tout/step]) from strings like
  "modes64_width60_epochs500_Tin10_T10" in model_source_dir/model_path/model_name.
- Force MultiIndex column levels to include ["model_group","model_name","modes","width","epochs"]
  (even if not varying) for BOTH:
    1) a full pivot with multiple metrics (rmse_mean/std, mae_mean/std, mape_mean/std)
    2) a ΔRMSE (model - baseline) pivot (on rmse_mean only)
- Keep plotting on the ΔRMSE pivot unchanged.

Usage examples:
  python plot_rmse_minus.py
  python plot_rmse_minus.py --csv data/combined_eval.csv
  python plot_rmse_minus.py --save-root plots/models_rmse_minus
  python plot_rmse_minus.py --full-pivot-csv-name pivot_full_metrics.csv
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
TITLE_CHARS_PER_LINE = 80   # 每行最多字符数（可按需要改小/改大）
TITLE_MAX_LINES      = 3    # 最多显示几行；超过会在最后一行尾部加省略号

# ===================== UTILITIES =====================
def log(msg: str):
    print(msg, flush=True)

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", str(s))
    return s[:230]

def sanitize_title(s: str) -> str:
    return sanitize_filename(s)

import textwrap

def wrap_and_truncate_title(s: str,
                            width: int = TITLE_CHARS_PER_LINE,
                            max_lines: int = TITLE_MAX_LINES) -> str:
    """
    将长标题规整空白、按宽度自动换行，并限制行数；超出部分用省略号。
    """
    if not s:
        return s
    # 规整空白，避免 wrap 时产生奇怪的断行
    s = re.sub(r"\s+", " ", str(s)).strip()
    # 自动换行（不强行拆长词）
    lines = textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .;,:") + " …"
    return "\n".join(lines)

def split_center_for_title(s: str, window: int = 20) -> str:
    """
    将长字符串在接近中点的“合适位置”断成两行：
    优先断在 '/' 或 '\\'，其次 '_'、'-'，最后退化为严格居中。
    window: 在中点左右可接受的搜索窗口（字符数）。
    """
    if not s or len(s) < 2:
        return s
    mid = len(s) // 2

    # 候选分隔符按优先级
    seps_priority = [
        ('/\\',),          # 最高优先：路径分隔
        ('_-',),           # 次优先：常见命名分隔符
    ]

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

    # 依优先级寻找断点
    for (chars,) in seps_priority:
        pos = nearest_break(chars)
        if pos is not None:
            return s[:pos] + "\n" + s[pos+1:]  # 去掉分隔符本身，更干净

    # 若没找到，就精确一半处断开（不删字符）
    return s[:mid] + "\n" + s[mid:]


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

# ===================== MODEL HYPERPARAM PARSER =====================
_re_hp = re.compile(
    r"(?:^|/|_)"                                  # boundary-ish
    r"(?:modes(?P<modes>\d+))?"                   # modes64  -> modes
    r"(?:_width(?P<width>\d+))?"                  # _width60 -> width
    r"(?:_epochs(?P<epochs>\d+))?"                # _epochs500 -> epochs
    r"(?:_Tin(?P<Tin>\d+))?"                      # _Tin10
    r"(?:_(?:T|Tout)(?P<Tout>\d+))?"              # _T10 or _Tout10
    r"(?:_step(?P<step>\d+))?",                   # _step1
    flags=re.IGNORECASE
)

def _first_str(*vals):
    for v in vals:
        s = str_or_none(v)
        if s: return s
    return None

def parse_hp_from_text(txt: str) -> dict:
    """Parse modes/width/epochs[/Tin/Tout/step] from a single string."""
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
    """
    Ensure columns: modes1, modes2, width, epochs, Tin, Tout, step, and a string 'modes' = 'modes1xmodes2'.
    If missing (or NaN), try to parse from model_source_dir/model_path/model_name/any stringy column.
    If only one 'modes' is available (1D), set modes1=modes2=modes.
    """
    df = df.copy()
    need_cols = ["modes1","modes2","width","epochs","Tin","Tout","step"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    str_cols = [c for c in ["model_source_dir","model_path","model_name"] if c in df.columns]
    # fallback: search other object columns if above not found
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
        }
        if all(have.values()):
            continue

        txt = _first_str(*[row.get(c, None) for c in str_cols])
        hp = parse_hp_from_text(txt or "")
        # fill width/epochs/Tin/Tout/step if missing
        for k in ["width","epochs","Tin","Tout","step"]:
            if not have[k] and hp.get(k) is not None:
                df.at[i, k] = hp[k]
        # modes: prefer existing modes1/2; else infer from hp["modes"]
        if (not have["modes1"] or not have["modes2"]) and (hp.get("modes") is not None):
            df.at[i, "modes1"] = hp["modes"]
            df.at[i, "modes2"] = hp["modes"]

    # Build 'modes' string level like "64x64"
    def _mk_modes_str(r):
        m1 = int(r["modes1"]) if not pd.isna(r["modes1"]) else None
        m2 = int(r["modes2"]) if not pd.isna(r["modes2"]) else None
        if m1 is None and m2 is None:
            return SENTINEL
        if m1 is None: m1 = m2
        if m2 is None: m2 = m1
        return f"{m1}x{m2}"
    df["modes"] = df.apply(_mk_modes_str, axis=1)

    # cast width/epochs to Int64 (nullable) so sorting/stability is better later
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
    """Append any levels in want_levels that exist in df but are absent in cur_levels (even if not varying)."""
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
    关键增强点：
      - 仅使用当前 df 中“真实存在”的 metrics 列，避免 KeyError。
      - 强制把数据集层级中的 'type' 纳入 index（如果该列存在），
        以兼容后续 family 过滤函数（即便该列只有一个取值）。
      - 自动识别并命名 metrics 所在列层级为 "metric"（保持你原来的稳健逻辑）。
      - 同时强制把 ["model_group","model_name","modes","width","epochs"] 纳入列层级。
    依赖外部工具函数/常量：_levels_with_variation, _force_append_levels, _fill_keys_with_sentinel,
                             DATASET_COLS, MODEL_COLS, FORCED_MODEL_LEVELS
    """
    # 选择会出现在 index/columns 的数据/模型维度
    ds_levels = _levels_with_variation(df, DATASET_COLS)
    ds_levels = [c for c in ds_levels if c != 'dataset_path']  # 避免超宽列进入索引

    # —— 新增：强制把 'type' 纳入索引层级（若该列存在） —— #
    forced_dataset_levels = ["type"]
    ds_levels = _force_append_levels(ds_levels, forced_dataset_levels, df.columns)

    model_levels = _levels_with_variation(df, MODEL_COLS)
    model_levels = _force_append_levels(model_levels, FORCED_MODEL_LEVELS, df.columns)

    # 参与透视的键列
    key_cols = sorted(set(ds_levels + model_levels))
    df2 = _fill_keys_with_sentinel(df, key_cols)

    # 保证至少各有 1 个层级
    if not ds_levels:
        ds_levels = ['__ALL_DATASETS__']; df2['__ALL_DATASETS__'] = '__ALL_DATASETS__'
    if not model_levels:
        model_levels = ['__ALL_MODELS__']; df2['__ALL_MODELS__'] = '__ALL_MODELS__'

    # —— 新增：仅保留 df 中真实存在的 metrics 列，避免 pivot_table 的 KeyError —— #
    values_present = [v for v in values if v in df.columns]
    if not values_present:
        raise ValueError(
            f"No requested metric columns found in CSV. Looked for: {values}"
        )

    # 透视
    piv = pd.pivot_table(
        df2,
        index=ds_levels,
        columns=model_levels,
        values=values_present,      # 使用过滤后的指标列
        aggfunc=aggfunc,
        observed=False
    ).sort_index().sort_index(axis=1)

    # —— 关键增强：稳健识别并命名 metrics 列层级为 "metric" —— #
    want_metrics = {'rmse_mean','rmse_std','mae_mean','mae_std','mape_mean','mape_std'}
    col_names = list(piv.columns.names or [])
    metric_idx = None

    # 方案 A：看该层取值集合（交集≥3 认为是 metrics 层；注意有时只会有 1~2 个）
    for i, lv in enumerate(piv.columns.levels):
        try:
            vals = set(map(str, lv))
        except Exception:
            vals = set(str(v) for v in lv)
        if want_metrics.issubset(vals) or len(want_metrics & vals) >= 3:
            metric_idx = i
            break

    # 方案 B：逐层尝试 xs('rmse_mean') 来探测
    if metric_idx is None:
        for i, name in enumerate(col_names):
            try:
                _ = piv.xs('rmse_mean', axis=1, level=name, drop_level=False)
                metric_idx = i
                break
            except Exception:
                pass

    # 找到了就命名该层为 "metric"
    if metric_idx is not None:
        col_names[metric_idx] = "metric"
        piv.columns.set_names(col_names, inplace=True)

    # 设定索引层级名字
    if isinstance(piv.index, pd.MultiIndex):
        piv.index.names = ds_levels
    else:
        piv.index.name = ds_levels[0]

    return piv


def subtract_baseline_rmse(pivot_full: pd.DataFrame, metric: str = 'rmse_mean'):
    """
    从多指标透视表 pivot_full 中抽取指定 metric（默认 rmse_mean），
    并按行减去 baseline（model_group=baseline）的平均值，返回 (ΔRMSE pivot, baseline_series)。

    增强点：
      - 不再假设列里一定存在名为 "metric" 的层级；若缺失，会逐层尝试用 xs 探测 metrics 所在层级。
      - 保持返回形状与原列结构一致（保留列 MultiIndex）。
    """
    col_names = list(pivot_full.columns.names or [])
    if "model_group" not in col_names:
        raise ValueError("列层级里没有 'model_group'，无法识别 baseline。")

    # 定位 metrics 层级
    metric_level = None
    if "metric" in col_names:
        metric_level = "metric"
    else:
        # 逐层尝试对 'metric' 指定值做切片，能切出来的那层就是 metrics 层
        for name in col_names:
            try:
                _ = pivot_full.xs(metric, axis=1, level=name, drop_level=False)
                metric_level = name
                break
            except Exception:
                continue

    if metric_level is None:
        raise ValueError("找不到 metrics 所在的列层级；请检查 pivot.columns.names 与 metrics 列是否存在。")

    # 取出该 metric 的块（保留 metrics 层以维持形状稳定）
    try:
        rmse_block = pivot_full.xs(metric, axis=1, level=metric_level, drop_level=False)
    except KeyError as e:
        raise KeyError(f"在 metrics 层级 '{metric_level}' 下找不到指标 '{metric}'。") from e

    # baseline 切片
    try:
        base_block = rmse_block.xs('baseline', axis=1, level='model_group', drop_level=False)
    except KeyError as e:
        raise KeyError("列中找不到 model_group='baseline' 的列，请确认 baseline 是否在输入数据中。") from e

    # baseline 按行平均
    base_series = base_block.mean(axis=1, skipna=True)

    # 逐行相减
    pivot_delta = rmse_block.subtract(base_series, axis=0)

    return pivot_delta, base_series

def ensure_rmse_minus(pivot: pd.DataFrame):
    # legacy safety (kept for compatibility, but unused in the new path)
    if isinstance(pivot.columns, pd.MultiIndex) and "model_group" in pivot.columns.names:
        if "baseline" in pivot.columns.get_level_values("model_group"):
            base_block = pivot.xs("baseline", axis=1, level="model_group", drop_level=False)
            max_abs = float(base_block.abs().max().max())
            if max_abs > 1e-9:
                pivot_minus, _ = subtract_baseline_rmse(pivot, metric="rmse_mean")
                return pivot_minus
    return pivot

# ===================== (existing) KERNEL / ORDER / PLOTTING (unchanged) =====================
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

# ===== 新增：从列MultiIndex里提取指定层级的值，并构建子目录名 =====
def _get_level_value_from_col(col_key, col_names, level_name, default="NA"):
    """
    col_key: 可能是元组（MultiIndex的一个键），也可能是标量
    col_names: pivot.columns.names（列表）
    """
    try:
        if isinstance(col_key, tuple) and level_name in col_names:
            idx = col_names.index(level_name)
            v = col_key[idx]
        else:
            v = default
    except Exception:
        v = default
    # 统一成字符串，处理缺失/哨兵
    v = str(v)
    if v is None or v.strip() == "" or v == str(pd.NA) or v.lower() == "nan" or v == SENTINEL:
        v = default
    return v

def _mwe_subdir_from_col(col_key, col_names):
    """
    由当前列键（包含 MultiIndex 各层级取值）生成末级目录名：
      modes{val}_width{val}_epochs{val}
    """
    modes  = _get_level_value_from_col(col_key, col_names, "modes",  default="NA")
    width  = _get_level_value_from_col(col_key, col_names, "width",  default="NA")
    epochs = _get_level_value_from_col(col_key, col_names, "epochs", default="NA")
    subdir = f"modes{modes}_width{width}_epochs{epochs}"
    return sanitize_filename(subdir)

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

    # 仅绘制非 baseline 列
    if isinstance(pivot_ord.columns, pd.MultiIndex):
        col_mask = np.ones(len(pivot_ord.columns), dtype=bool)
        if "model_group" in pivot_ord.columns.names:
            col_mask &= (pivot_ord.columns.get_level_values("model_group") != "baseline")
        cols = [c for c, m in zip(pivot_ord.columns, col_mask) if m]
    else:
        cols = [c for c in pivot_ord.columns if str(c).lower() != "baseline"]

    log(f"[plot] models to plot: {len(cols)}")

    # —— 列层级名字缓存 —— #
    col_level_names = list(pivot_ord.columns.names or [])

    # —— 小工具：由 model_name 生成 png 文件名（去掉常见权重扩展名） —— #
    def _png_name_from_model_name(name: str) -> str:
        base = str(name or "").strip()
        base = re.sub(r"\.(pth|pt|ckpt|bin|h5|onnx)$", "", base, flags=re.IGNORECASE)
        base = sanitize_title(base)
        return base + ".png"

    for ci, c in enumerate(cols, 1):
        series = pivot_ord[c]

        # =============== 取该列的 model_name（标题 & 文件名都用它） =============== #
        model_name = _get_level_value_from_col(c, col_level_names, "model_name", default="model")
        if model_name in (SENTINEL, "NA", "", None):
            # 兜底：用 model_path 的最后一段
            model_path = _get_level_value_from_col(c, col_level_names, "model_path", default="model")
            model_name = os.path.basename(model_path) if model_path not in (SENTINEL, "NA") else str(c)

        log(f"[plot {ci}/{len(cols)}] drawing: model_name={model_name}")

        fig_w = max(20, int(len(series) * 0.06))
        fig, ax = plt.subplots(figsize=(fig_w, 13))

        x = np.arange(len(series))
        y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

        # 颜色与纹理
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

        # ======== 标题：只用模型文件名（自动换行/截断） ========
        title_str = str(model_name)
        ax.set_title(wrap_and_truncate_title(title_str), fontsize=14, pad=24)

        ax.set_ylabel("ΔRMSE", fontsize=12)
        ax.axhline(0.0, color="black", lw=1.0, alpha=0.8)
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.6)
        plt.subplots_adjust(top=0.78)

        # 图例
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

        # ======== 保存：目录仍按 modes/width/epochs；文件名 = model_name（转 .png） ========
        if save_dir:
            # 父目录仍用 M/W/E 子目录（与你原先一致）
            mwe_subdir = _mwe_subdir_from_col(c, col_level_names)  # e.g., modes96x96_width80_epochs2000
            out_dir_final = Path(save_dir) / mwe_subdir
            out_dir_final.mkdir(parents=True, exist_ok=True)

            fname = _png_name_from_model_name(model_name)
            out_path = out_dir_final / fname
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

    # Subtract baseline on rmse_mean only, keep column MultiIndex (with modes/width/epochs + metric)
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
