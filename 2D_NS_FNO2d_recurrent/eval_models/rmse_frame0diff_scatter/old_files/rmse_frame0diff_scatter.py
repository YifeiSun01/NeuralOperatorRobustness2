#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delta-RMSE vs Similarity Scatter (per-model linkage)

- 自动识别并解析“pivot 宽表”（顶部 meta 行：model_group/strategy/pct_tag/percent/k_requested；
  其后一行字段名包含 dataset_group,dataset_name,n,...，再往后每列是一个模型）的 CSV。
- 若输入本就是 ΔRMSE（文件名含 minus_baseline 或列名是 delta_rmse/rmse_frame0diff/frame0diff），跳过 baseline 合并；
  否则从 perf 内部的 baseline 行聚合得到 ΔRMSE。
- 逐“模型”处理：按每个模型的 expanded 训练集，从 sim.csv 里挑相应一侧（target_name/gen_name 等），
  再和另一侧的数据集对齐，绘制 ΔRMSE vs 相似度散点。

新增参数：
  --force-pivot  强制按 pivot 宽表解析
  --skip-pivot   强制按普通 CSV 解析
  --rmse-col COL 显式指定 perf 里的指标列（若自动识别失败，可用此项，如 val_25）
"""

import argparse, re, hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

# ------------------------- 全局候选 -------------------------
SIM_CANDIDATES = [
    'hist_js_bits','hist_tv','hist_emd','hist_emd_norm','hist_overlap_coeff',
    'hist_range_overlap_ratio','hist_kl_pq_nat','hist_kl_qp_nat',
    'global_mean_absdiff','global_std_absdiff','global_q50_absdiff',
    'mean_map_pearson_r','mean_map_norm_mae','std_map_pearson_r','std_map_norm_mae'
]
GROUP_BASE_COLOR = {"train":"#1f77b4", "test":"#ff7f0e", "expanded":"#2ca02c", "generalizability":"#9467bd"}

# ------------------------- pivot 识别与解析 -------------------------
def _is_pivot_like_csv(path: str) -> bool:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            head = [next(f) for _ in range(8)]
        h = "\n".join(head).lower()
        return ('model_group' in h) and ('strategy' in h) and ('pct_tag' in h) and ('dataset_name' in h)
    except Exception:
        return False

def _pivot_to_long_dataframe(pivot_csv_path: str) -> pd.DataFrame:
    """解析这类结构：
       第 0..k-1 行：meta 行（model_group/strategy/pct_tag/percent/k_requested/……）
       第 k 行：字段名（含 dataset_group, dataset_name, n, ……）
       第 k+1..  ：数据行；从 dataset_name 之后的列，每列就是一个“模型的数值列”。
    """
    df0 = pd.read_csv(pivot_csv_path, header=None, dtype=str, keep_default_na=False)
    ncols = df0.shape[1]

    # 找“字段名行”（包含 dataset_name / name）
    header_row = None
    for i in range(min(15, len(df0))):
        row = [c.strip() for c in df0.iloc[i].tolist()]
        if any(x.lower() == 'dataset_name' for x in row) or any(x.lower() == 'name' for x in row):
            header_row = i
            break
    if header_row is None:
        raise SystemExit("pivot 解析失败：前 15 行内未找到字段名行（需含 dataset_name/name）")

    # 字段名：为空的补 col{j}
    raw_header = [df0.iloc[header_row, j].strip() for j in range(ncols)]
    header = [ (c if c else f"col{j}") for j,c in enumerate(raw_header) ]

    # 关键列位置
    def _find(name: str):
        name = name.lower()
        for j,c in enumerate(header):
            if c.lower() == name:
                return j
        return None
    idx_dsname = _find('dataset_name') or _find('name')
    if idx_dsname is None:
        raise SystemExit("pivot 解析失败：字段名行不含 dataset_name/name")
    idx_first_value = idx_dsname + 1
    # 紧随其后若是 n/count 之类的统计列，跳过
    if idx_first_value < ncols and header[idx_first_value].lower() in ('n','count'):
        idx_first_value += 1

    # id_vars = dataset_name 前（含 dataset_group/参数等） + 可能的 n 列
    id_vars = header[:idx_first_value]

    # meta 行（0..header_row-1），用于给每个“值列”贴上 model_group/strategy/pct_tag/percent/k_requested
    meta_rows = [[df0.iloc[r, j].strip() if j < ncols else '' for j in range(ncols)]
                 for r in range(header_row)]

    # 数据体
    dat = df0.iloc[header_row+1:].reset_index(drop=True)
    # 给数据体设置列名（与 header 同长）
    dat.columns = header

    rows = []
    # 从第一个值列开始，每一列 melt 成一段
    for j in range(idx_first_value, ncols):
        colname = header[j]
        tmp = dat[id_vars].copy()
        # 贴上 meta
        def _meta_at(row_idx, default=''):
            return meta_rows[row_idx][j] if row_idx < len(meta_rows) else default
        tmp['model_group'] = _meta_at(0, '')
        tmp['strategy']    = _meta_at(1, '')
        tmp['pct_tag']     = _meta_at(2, '')
        tmp['percent']     = _meta_at(3, '')
        tmp['k_requested'] = _meta_at(4, '')
        # 数值列——尽力转为 float
        tmp['rmse_mean']   = pd.to_numeric(dat.iloc[:, j], errors='coerce')
        # 把列名也保留一下，方便定位
        tmp['__value_col__'] = colname
        rows.append(tmp)

    # 拼接并清理
    if not rows:
        raise SystemExit("pivot 解析失败：未发现值列（dataset_name 之后的列）")
    long = pd.concat(rows, ignore_index=True)
    # 列名统一：group / dataset_name
    if 'dataset_group' in long.columns and 'group' not in long.columns:
        long = long.rename(columns={'dataset_group': 'group'})
    if 'name' in long.columns and 'dataset_name' not in long.columns:
        long = long.rename(columns={'name': 'dataset_name'})

    keep = [c for c in [
        'group','dataset_name','rmse_mean','model_group','strategy','pct_tag','percent',
        'k_requested','__value_col__',
        'type','N','length_scale','variance','alpha','tau','period','vmax','vmin'
    ] if c in long.columns]
    long = long[keep]
    long = long[long['rmse_mean'].notna()]
    if long.empty:
        raise SystemExit("pivot 解析失败：所有值列均为空或无法转换为数值。")
    return long

# ------------------------- 常用小函数 -------------------------
def _col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # 带上可视化信息，便于排错
    cols = list(df.columns)
    raise KeyError(f"未找到这些列之一: {candidates}\n实际列名示例(前30): {cols[:30]}")

def _parse_hparams_from_text(text: str):
    if not isinstance(text, str):
        return {}
    d = {}
    m = re.search(r"modes(\d+)", text);  w = re.search(r"width(\d+)", text);  e = re.search(r"epochs(\d+)", text)
    Tin = re.search(r"Tin(\d+)", text);  Tout = re.search(r"(?:Tout|T)(\d+)", text);  step = re.search(r"step(?:s)?(\d+)", text)
    if m: d["modes"] = int(m.group(1))
    if w: d["width"] = int(w.group(1))
    if e: d["epochs"] = int(e.group(1))
    if Tin: d["Tin"] = int(Tin.group(1))
    if Tout: d["Tout"] = int(Tout.group(1))
    if step: d["step"] = int(step.group(1))
    return d

def _ensure_hparams(df: pd.DataFrame):
    for key in ["modes","width","epochs","Tin","Tout","step"]:
        if key not in df.columns:
            df[key] = np.nan
    for col in ["model_name","model_path","model_dir","path"]:
        if col in df.columns:
            parsed = df[col].astype(str).apply(_parse_hparams_from_text)
            for k in ["modes","width","epochs","Tin","Tout","step"]:
                df[k] = df[k].fillna(parsed.apply(lambda d: d.get(k, np.nan)))
    for k in ["modes","width","epochs","Tin","Tout","step"]:
        if df[k].notna().all():
            df[k] = df[k].astype(int)
    return df

def _infer_expanded_tag_from_perf_rows(df_model: pd.DataFrame, group_col: str) -> str | None:
    if group_col in df_model.columns and "dataset_name" in df_model.columns:
        cand = df_model.loc[df_model[group_col]=="expanded", "dataset_name"].dropna().astype(str).unique().tolist()
        if cand:
            return cand[0]
    for c in ("expanded","expanded_name","expanded_dataset","gen_name","generator","train_name","target_name"):
        if c in df_model.columns:
            vals = df_model[c].dropna().astype(str)
            for v in vals:
                if "train_all_frames" in v or "modespec=" in v or v.startswith("dim"):
                    return v
    return None

def _derive_range_label(vmin, vmax):
    try:
        if pd.isna(vmin) or pd.isna(vmax):
            return None
        return f"[{float(vmin):g},{float(vmax):g}]"
    except Exception:
        return None

def _kernel_marker(kernel: str):
    if not isinstance(kernel, str):
        return 'o'
    s = kernel.lower()
    if "matern" in s: return 'o'
    if "periodic" in s: return 's'
    if "rbf" in s: return '^'
    if "loggrf" in s: return 'D'
    if "negloggrf" in s: return 'X'
    if "grf" in s: return 'o'
    return 'o'

def _now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _normalize_pct_arg(pct_arg):
    if pct_arg is None or str(pct_arg).lower() == "auto":
        return None
    s = str(pct_arg).strip().lower()
    if s.endswith("pct"):
        try:
            return float(s.replace("pct", "")) / 100.0
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        raise ValueError(f"无法解析 --pct={pct_arg}")

def _pick_x_metric(args_x, df_sim_sub):
    if args_x and str(args_x).lower() != 'auto':
        if args_x not in df_sim_sub.columns:
            raise KeyError(f"相似度表缺少列 {args_x}")
        return args_x
    for c in SIM_CANDIDATES:
        if c in df_sim_sub.columns:
            return c
    num_cols = [c for c in df_sim_sub.columns if pd.api.types.is_numeric_dtype(df_sim_sub[c])]
    if not num_cols:
        raise RuntimeError("sim 子表没有可用的数值相似度列")
    return num_cols[0]

def _model_id_columns(df: pd.DataFrame):
    pref = ["model_name","model_path"]
    meta = ["model_group","strategy","pct_tag","percent","modes","width","epochs","Tin","Tout","step","__value_col__"]
    cols = [c for c in pref+meta if c in df.columns]
    return cols

def _mk_model_id_tuple(row: pd.Series, id_cols: list[str]):
    return tuple((c, row.get(c, None)) for c in id_cols)

def _mk_model_tag(row: pd.Series):
    bits = []
    for c in ["model_name","strategy","pct_tag","percent","modes","width","epochs","__value_col__"]:
        if c in row and pd.notna(row[c]):
            bits.append(f"{c}={row[c]}")
    return ", ".join(bits) if bits else "model"

def _hash_short(s: str, n=8):
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:n]

# ------------------------- 主流程 -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf-csv", required=True)
    ap.add_argument("--sim-csv",  required=True)
    ap.add_argument("--expanded", default=None)
    ap.add_argument("--expanded-like", default=None)
    ap.add_argument("--modes", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--agg", choices=["mean","median"], default="mean")
    ap.add_argument("--outdir", default="plots/scatter")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--strategy", choices=["add","replace","auto"], default="auto")
    ap.add_argument("--pct", default="auto")
    ap.add_argument("--x-metric", default="auto")
    # 新增兜底选项
    ap.add_argument("--force-pivot", action="store_true", help="强制按 pivot 宽表解析 perf")
    ap.add_argument("--skip-pivot",  action="store_true", help="强制按普通 CSV 解析 perf")
    ap.add_argument("--rmse-col",    default=None, help="显式指定 perf 指标列名（若自动识别失败）")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # === 读 CSV（pivot -> long） ===
    try:
        if args.force_pivot:
            print("[提示] 按 --force-pivot 解析 pivot 宽表…")
            raw_perf = _pivot_to_long_dataframe(args.perf_csv)
        elif args.skip_pivot:
            print("[提示] 按 --skip-pivot 直接读取普通 CSV…")
            raw_perf = pd.read_csv(args.perf_csv)
        else:
            if _is_pivot_like_csv(args.perf_csv):
                print("[提示] 检测到 perf 是 pivot 宽表，自动转换为长表…")
                raw_perf = _pivot_to_long_dataframe(args.perf_csv)
            else:
                raw_perf = pd.read_csv(args.perf_csv)
    except Exception as e:
        print(f"[警告] pivot 自动转换失败/不可用：{e}\n改为直接按常规 CSV 读取。")
        raw_perf = pd.read_csv(args.perf_csv)

    df_sim  = pd.read_csv(args.sim_csv)

    # 确保名称列
    if "dataset_name" not in raw_perf.columns and "name" in raw_perf.columns:
        raw_perf = raw_perf.rename(columns={"name":"dataset_name"})

    # group 列
    group_col = "group" if "group" in raw_perf.columns else ("dataset_group" if "dataset_group" in raw_perf.columns else None)
    if not group_col:
        raise KeyError(f"perf CSV 中未找到 'group' 或 'dataset_group' 列。现有列: {list(raw_perf.columns)[:30]}")

    # perf 指标列（允许手动指定）
    if args.rmse_col and args.rmse_col in raw_perf.columns:
        rmse_col = args.rmse_col
    else:
        rmse_col = _col(raw_perf, [
            "rmse_mean","rmse","RMSE","rmse_val","val_rmse","rmse_mean_all",
            "rmse_frame0diff","frame0diff","delta_rmse"
        ])

    # 输入是否已是 ΔRMSE
    already_delta = (
        'minus_baseline' in str(args.perf_csv).lower() or
        str(rmse_col).lower() in ['delta_rmse','rmse_frame0diff','frame0diff']
    )

    # ===== baseline：若需要 =====
    if not already_delta:
        base_mask = (raw_perf.get("model_group") == "baseline") | (raw_perf.get("strategy") == "__NA__") | (raw_perf.get("pct_tag") == "__NA__")
        df_base = raw_perf[base_mask].copy()
        if df_base.empty:
            raise RuntimeError("未找到 baseline 行；若本来就是 ΔRMSE，请在文件名里包含 minus_baseline 或把列命名为 delta_rmse。")
        base_group_col = "group" if "group" in df_base.columns else ("dataset_group" if "dataset_group" in df_base.columns else None)
        base_cols = [c for c in [base_group_col, "dataset_name"] if c in df_base.columns]
        if "dataset_name" not in base_cols:
            raise KeyError("baseline 表未包含 dataset_name 列")
        gbase = df_base.groupby(base_cols, as_index=False)[rmse_col].agg(args.agg)
    else:
        gbase = None

    # ===== 过滤（对模型端；baseline 不受影响）=====
    df_perf = raw_perf.copy()
    df_perf = _ensure_hparams(df_perf)

    if args.strategy != "auto" and "strategy" in df_perf.columns:
        df_perf = df_perf[df_perf["strategy"] == args.strategy]

    pct_val = _normalize_pct_arg(args.pct)
    if pct_val is not None:
        if "percent" in df_perf.columns:
            df_perf = df_perf[df_perf["percent"].astype(float) == float(pct_val)]
        elif "pct_tag" in df_perf.columns:
            tag = {0.25:"25pct",0.5:"50pct",0.75:"75pct",1.0:"100pct"}.get(float(pct_val))
            if tag is None:
                raise ValueError(f"percent={pct_val} 仅支持 0.25/0.5/0.75/1.0")
            df_perf = df_perf[df_perf["pct_tag"] == tag]

    for k in ["modes","width","epochs"]:
        v = getattr(args, k)
        if v is not None and k in df_perf.columns:
            df_perf = df_perf[df_perf[k] == v]

    # ===== 按“模型”分组 =====
    id_cols = _model_id_columns(df_perf)
    if not id_cols:
        df_perf["__model_id__"] = "__single__"
    else:
        df_perf["__model_id__"] = df_perf.apply(lambda r: _mk_model_id_tuple(r, id_cols), axis=1)

    cm = get_cmap("coolwarm")
    all_rows = []; plotted = 0

    for model_id, df_model in df_perf.groupby("__model_id__"):
        exp_tag = _infer_expanded_tag_from_perf_rows(df_model, group_col)
        if not exp_tag:
            print(f"[跳过] 无法从该模型的 perf 行推断 expanded 训练集：{model_id}")
            continue
        if args.expanded and exp_tag != args.expanded:
            continue
        if args.expanded_like and (args.expanded_like not in exp_tag):
            continue

        mcols = [c for c in [group_col, "dataset_name", rmse_col, "type","vmin","vmax","kernel","model_name","strategy","pct_tag","percent","modes","width","epochs","__value_col__"] if c in df_model.columns]
        df_mperf = df_model[mcols].copy()

        if not already_delta:
            merged = pd.merge(df_mperf, gbase, on=[group_col,"dataset_name"], suffixes=("","_base"), how="left")
            merged["delta_rmse"] = merged[rmse_col] - merged[f"{rmse_col}_base"]
        else:
            merged = df_mperf.copy()
            merged["delta_rmse"] = merged[rmse_col]

        # 在 sim 中，以 expanded 为 target 或 gen 侧筛选
        def _select_sim_by_side(expanded_col, other_col):
            if expanded_col not in df_sim.columns or other_col not in df_sim.columns:
                return pd.DataFrame()
            sub = df_sim[df_sim[expanded_col].astype(str) == exp_tag]
            if sub.empty:
                sub = df_sim[df_sim[expanded_col].astype(str).str.contains(re.escape(exp_tag), na=False)]
            return sub.copy()

        df_sim_exp = _select_sim_by_side("target_name", "gen_name")
        if df_sim_exp.empty:
            df_sim_exp = _select_sim_by_side("gen_name", "target_name")
            exp_col, other_col = ("gen_name", "target_name") if not df_sim_exp.empty else (None, None)
        else:
            exp_col, other_col = ("target_name", "gen_name")

        if df_sim_exp.empty or other_col is None:
            for cand in ["expanded_name","generator","source_name","source_dataset","train_name","target","dataset_name","name"]:
                if cand in df_sim.columns:
                    tmp = df_sim[df_sim[cand].astype(str).str.contains(re.escape(exp_tag), na=False)].copy()
                    if not tmp.empty:
                        df_sim_exp = tmp
                        exp_col = cand
                        other_col = "gen_name" if "gen_name" in df_sim_exp.columns else ("dataset_name" if "dataset_name" in df_sim_exp.columns else None)
                        break
        if df_sim_exp.empty or other_col is None:
            print(f"[跳过] sim 中找不到与 expanded={exp_tag} 对应的行（模型 {model_id}）")
            continue

        # range_label / kernel
        if "gen_vmin" in df_sim_exp.columns or "vmin" in df_sim_exp.columns:
            df_sim_exp["range_label"] = df_sim_exp.apply(
                lambda r: _derive_range_label(r.get("gen_vmin", r.get("vmin")), r.get("gen_vmax", r.get("vmax"))), axis=1
            )
        else:
            df_sim_exp["range_label"] = None
        if "type" in df_sim_exp.columns and "kernel" not in df_sim_exp.columns:
            df_sim_exp["kernel"] = df_sim_exp["type"]

        x = _pick_x_metric(args.x_metric, df_sim_exp)

        join_left = merged.rename(columns={"dataset_name":"_name"})
        join_right = df_sim_exp.rename(columns={other_col:"_name"})
        sim_group_col = None
        for c in ["gen_type","group","dataset_group"]:
            if c in join_right.columns:
                sim_group_col = c; break
        on_cols = ["_name"]
        if sim_group_col and group_col in join_left.columns:
            join_right = join_right.rename(columns={sim_group_col: group_col})
            on_cols = ["_name", group_col]

        df_join = pd.merge(join_left, join_right, on=on_cols, how="left", suffixes=("","_sim"))
        df_join = df_join.rename(columns={"_name":"dataset_name"})
        df_plot = df_join.dropna(subset=[x, "delta_rmse"]).copy()
        if df_plot.empty:
            print(f"[提示] 模型 {model_id} 对齐后没有可画的点（可能名称或分组不一致）")
            continue

        # 颜色：generalizability 按 range_label 渐变
        gen_mask = (df_plot[group_col] == "generalizability") if group_col in df_plot.columns else pd.Series(False, index=df_plot.index)
        uniq_ranges = sorted([r for r in df_plot.loc[gen_mask, "range_label"].dropna().astype(str).unique().tolist()])
        if uniq_ranges:
            xs = np.linspace(0.1, 0.9, num=len(uniq_ranges))
            range_palette = {rl: get_cmap("coolwarm")(x_) for rl, x_ in zip(uniq_ranges, xs)}
        else:
            range_palette = {}

        def point_color(r):
            g = r.get(group_col, None)
            if g == "generalizability":
                rl = r.get("range_label", None)
                return range_palette.get(rl, get_cmap("coolwarm")(0.5))
            return GROUP_BASE_COLOR.get(g, (0.3, 0.3, 0.3))

        def point_marker(r):
            k = r.get("kernel", r.get("type", None))
            return _kernel_marker(str(k) if k is not None else "")

        # 绘图
        fig, ax = plt.subplots(figsize=(8.8, 6.4))
        for _, r in df_plot.iterrows():
            ax.scatter(r[x], r["delta_rmse"], marker=point_marker(r), s=55,
                       edgecolor="none", facecolor=point_color(r), alpha=0.9)
        ax.axhline(0.0, linewidth=1.0, linestyle="--")
        ax.set_xlabel(x); ax.set_ylabel("ΔRMSE (model - baseline)")

        # 图例
        legend_elems = []
        for rl in [rl for rl in uniq_ranges if rl]:
            legend_elems.append(Patch(facecolor=range_palette.get(rl, get_cmap("coolwarm")(0.5)), edgecolor="none",
                                      label=f"generalizability\n{rl}"))
        uniq_kernels = sorted(set(str(k).lower() for k in df_plot.get("kernel", pd.Series(dtype=object)).dropna().unique().tolist()))
        KERNEL_ORDER = ["grf_matern","grf_periodic","grf_rbf","loggrf","negloggrf"]
        uniq_kernels = sorted(uniq_kernels, key=lambda k: (KERNEL_ORDER.index(k) if k in KERNEL_ORDER else len(KERNEL_ORDER)))
        added_shapes = set()
        for k in uniq_kernels:
            mk = _kernel_marker(k)
            if mk in added_shapes: continue
            legend_elems.append(Line2D([0],[0], marker=mk, linestyle="None", label=f"kernel~{k}",
                                       markerfacecolor="white", markeredgecolor="black", markersize=7))
            added_shapes.add(mk)
        for g in ["test","train","expanded","generalizability"]:
            if group_col in df_plot.columns and (df_plot[group_col] == g).any():
                legend_elems.append(Patch(facecolor=GROUP_BASE_COLOR.get(g, (0.3,0.3,0.3)), edgecolor="none", label=g))
        if legend_elems:
            ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Legend")

        # 标题 & 输出
        tag_row = df_model.iloc[0]
        model_tag = _mk_model_tag(tag_row)
        title_bits = [f"model: {model_tag}", f"expanded={exp_tag}"]
        if args.strategy != "auto": title_bits.append(f"strategy={args.strategy}")
        if pct_val is not None:     title_bits.append(f"pct={pct_val}")
        for k in ["modes","width","epochs"]:
            v = getattr(args, k)
            if v is not None: title_bits.append(f"{k}={v}")
        ax.set_title("ΔRMSE vs " + x + "\n" + " | ".join(title_bits))

        safe_model = _hash_short(str(model_id))
        safe_exp = re.sub(r"[^A-Za-z0-9._-]+", "_", str(exp_tag))[:100]
        tag = _now_tag()
        pct_str = "auto" if pct_val is None else str(pct_val)
        strat_str = args.strategy
        hparam_str = (f"__m{args.modes}_w{args.width}_e{args.epochs}" if (args.modes and args.width and args.epochs) else "")
        png_path = Path(args.outdir) / f"deltaRMSE_vs_{x}__model-{safe_model}__exp-{safe_exp}__{strat_str}-{pct_str}{hparam_str}__{tag}.png"
        fig.tight_layout(); fig.savefig(png_path, dpi=args.dpi); plt.close(fig)

        dump_cols = ["dataset_name", group_col, "delta_rmse", x, "range_label", "kernel", "type", "vmin", "vmax"]
        dump_cols = [c for c in dump_cols if c in df_plot.columns]
        csv_out = Path(args.outdir) / f"scatter_data__model-{safe_model}__exp-{safe_exp}__{strat_str}-{pct_str}__{x}__{tag}.csv"
        df_plot[dump_cols].to_csv(csv_out, index=False, encoding="utf-8")

        print(f"[OK] 模型 {model_tag}（exp={exp_tag}）: {png_path}")
        print(f"[OK] 数据: {csv_out}")
        plotted += 1
        all_rows.append(df_plot.assign(expanded=exp_tag, x_metric=x, __model_hash__=safe_model))

    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        csv_all = Path(args.outdir) / f"scatter_all_models__{_now_tag()}.csv"
        df_all.to_csv(csv_all, index=False, encoding="utf-8")
        print(f"[OK] 汇总数据: {csv_all}")
    else:
        print("[完成] 没有生成任何散点（可能过滤条件过严或列名对不上）。")

if __name__ == "__main__":
    main()




