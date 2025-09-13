#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
discover_and_run_scatter.py  (robust)

改动要点：
- strategy / pct 均可缺省：若 perf 没这两列 → 统一用 "auto"
- expanded 集合：优先 perf∩sim；若 perf 推不出 expanded → 退化为 sim 的 target_name 全部
- 过滤逻辑容错：仅当相应列存在且值不是 "auto" 时才做过滤
- 输出目录避免对 "auto" 用 :g（改为字符串）
- hparams 缺失时 combos=[(None,None,None)]
"""
import argparse, re, sys, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

ALLOWED_X = [
    'hist_js_bits', 'hist_tv', 'hist_emd', 'hist_emd_norm', 'hist_overlap_coeff',
    'hist_range_overlap_ratio', 'hist_kl_pq_nat', 'hist_kl_qp_nat',
    'global_mean_absdiff','global_std_absdiff','global_q50_absdiff',
    'mean_map_pearson_r','mean_map_norm_mae','std_map_pearson_r','std_map_norm_mae'
]

def _parse_list_floats(s):
    return [float(x) for x in str(s).split(",")] if s else None

def _parse_list_ints(s):
    return [int(x) for x in str(s).split(",")] if s else None

def _col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _parse_hparams_from_text(text: str):
    if not isinstance(text, str):
        return {}
    d = {}
    m = re.search(r"modes(\d+)", text)
    w = re.search(r"width(\d+)", text)
    e = re.search(r"epochs(\d+)", text)
    if m: d["modes"] = int(m.group(1))
    if w: d["width"] = int(w.group(1))
    if e: d["epochs"] = int(e.group(1))
    return d

def _ensure_hparams(df: pd.DataFrame):
    for k in ["modes","width","epochs"]:
        if k not in df.columns:
            df[k] = np.nan
    for col in ["model_name","model_path","model_dir","path"]:
        if col in df.columns:
            parsed = df[col].astype(str).apply(_parse_hparams_from_text)
            for k in ["modes","width","epochs"]:
                df[k] = df[k].fillna(parsed.apply(lambda d: d.get(k, np.nan)))
    # 保持为可空整数
    for k in ["modes","width","epochs"]:
        try:
            df[k] = df[k].astype("Int64")
        except Exception:
            pass
    return df

def _infer_expanded_tag_from_row(row):
    # 常见直给列
    for k in ["expanded_name","expanded_dataset","target_name"]:
        if k in row and isinstance(row[k], str) and row[k]:
            return row[k]
    # 路径/名称兜底
    for k in ["model_path","model_name"]:
        if k in row and isinstance(row[k], str):
            s = row[k]
            m = re.search(r"/N=\d+/(.*?)(?:/|$)", s)
            if m: return m.group(1)
            m2 = re.search(r"(dim2d_nx\d+_[^/]*train_all_frames[^/]*?)(?:/|\.pt|$)", s)
            if m2: return m2.group(1)
    return None

def _safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:120]

def _pct_dir_name(p):
    # 目录名安全转换
    return str(p) if isinstance(p, str) else f"{p:g}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf-csv", required=True)
    ap.add_argument("--sim-csv",  required=True)
    ap.add_argument("--plot-script", required=True, help="绘图脚本路径")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--x-metrics", default="all", help="'all' 或 逗号分隔的列名")
    ap.add_argument("--strategies", default="auto", help="'auto' 或 逗号分隔（如 add,replace）")
    ap.add_argument("--pct-list", default=None, help="逗号分隔（如 0.25,0.5,0.75,1.0）；默认自动从 perf 提取，若无列则用 'auto'")
    ap.add_argument("--expanded-like", default=None)
    ap.add_argument("--modes-list", default=None)
    ap.add_argument("--width-list", default=None)
    ap.add_argument("--epochs-list", default=None)
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--run", action="store_true", help="实际执行；不加则只打印命令")
    args = ap.parse_args()

    perf = pd.read_csv(args.perf_csv)
    sim  = pd.read_csv(args.sim_csv)

    # ---- expanded 候选 ----
    perf["__expanded_tag__"] = perf.apply(_infer_expanded_tag_from_row, axis=1)
    perf_exp_tags = {t for t in perf["__expanded_tag__"].dropna().astype(str).unique().tolist() if t}
    sim_target_col = _col(sim, ["target_name","expanded_name","source_name","target"])
    if not sim_target_col:
        raise SystemExit("sim.csv 未找到 target_name/expanded_name/source_name/target 任一列")
    sim_exp_tags = {t for t in sim[sim_target_col].astype(str).unique().tolist() if t}

    if perf_exp_tags:
        # 交集（考虑包含关系）
        expanded_final = set()
        for t in perf_exp_tags:
            if t in sim_exp_tags or any((t in s) or (s in t) for s in sim_exp_tags):
                expanded_final.add(t)
    else:
        # perf 推不出 expanded：退化为 sim 全部 target
        expanded_final = set(sim_exp_tags)

    if args.expanded_like:
        expanded_final = {t for t in expanded_final if args.expanded_like in t}

    if not expanded_final:
        raise SystemExit("未找到可用的 expanded（可用 --expanded-like 缩小范围，或检查 sim 的 target_name）")

    # ---- strategy / pct（允许缺省 → auto）----
    strat_col = "strategy" if "strategy" in perf.columns else None
    if args.strategies != "auto":
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    else:
        if strat_col:
            strategies = sorted([s for s in perf[strat_col].dropna().astype(str).unique().tolist() if s != "__NA__"])
            if not strategies:
                strategies = ["auto"]
        else:
            strategies = ["auto"]

    if args.pct_list:
        pct_final = _parse_list_floats(args.pct_list)
    else:
        if "percent" in perf.columns:
            pct_final = sorted([float(x) for x in perf["percent"].dropna().unique().tolist()])
            if not pct_final:
                pct_final = ["auto"]
        elif "pct_tag" in perf.columns:
            mp = {"25pct":0.25,"50pct":0.5,"75pct":0.75,"100pct":1.0}
            vals = [mp.get(x, None) for x in perf["pct_tag"].dropna().astype(str).unique().tolist()]
            pct_final = sorted(list({v for v in vals if v is not None})) or ["auto"]
        else:
            pct_final = ["auto"]

    # ---- 模型超参（可缺省）----
    perf = _ensure_hparams(perf)
    if args.modes_list and args.width_list and args.epochs_list:
        modes_list  = _parse_list_ints(args.modes_list)
        width_list  = _parse_list_ints(args.width_list)
        epochs_list = _parse_list_ints(args.epochs_list)
        combos = [(m,w,e) for m in modes_list for w in width_list for e in epochs_list]
    else:
        tmp = perf.dropna(subset=["modes","width","epochs"])
        combos = sorted({(int(m),int(w),int(e)) for m,w,e in tmp[["modes","width","epochs"]].itertuples(index=False, name=None)})
        if not combos:
            combos = [(None, None, None)]

    # ---- x-metrics ----
    if args.x_metrics == "all":
        x_metrics = [c for c in ALLOWED_X if c in sim.columns]
    else:
        x_metrics = [x.strip() for x in args.x_metrics.split(",") if x.strip() in sim.columns]
    if not x_metrics:
        raise SystemExit("没有可用的 x-metric（请检查 sim.csv 是否包含这些列）")

    outdir = Path(args.outdir)

    # ---- 生成命令 ----
    cmds = []
    for exp in sorted(expanded_final):
        for strat in strategies:
            for pct in pct_final:
                # 只在相应列存在且不是 auto 时才做筛
                sub = perf.copy()
                if "__expanded_tag__" in sub.columns:
                    sub = sub[sub["__expanded_tag__"].astype(str).str.contains(re.escape(exp))]
                if strat_col and strat != "auto":
                    sub = sub[sub[strat_col].astype(str) == strat]
                if "percent" in sub.columns and pct != "auto":
                    sub = sub[sub["percent"].astype(float) == float(pct)]
                elif "pct_tag" in sub.columns and pct != "auto":
                    mp = {0.25:"25pct",0.5:"50pct",0.75:"75pct",1.0:"100pct"}
                    sub = sub[sub["pct_tag"].astype(str) == mp.get(float(pct), "__INVALID__")]

                if sub.empty and perf_exp_tags:
                    # 如果 perf 能推 expanded，但这个组合在 perf 中完全不存在，则跳过；
                    # 若 perf 无法推 expanded（我们用 sim 托底），则不基于 perf 是否为空来跳过。
                    continue

                for (m,w,e) in combos:
                    if all(k in sub.columns for k in ["modes","width","epochs"]) and all(v is not None for v in (m,w,e)) and not sub.empty:
                        sub2 = sub[(sub["modes"]==m) & (sub["width"]==w) & (sub["epochs"]==e)]
                        if sub2.empty:
                            continue
                    for xm in x_metrics:
                        safe_exp = _safe(exp); safe_x = _safe(xm)
                        pct_dir = _pct_dir_name(pct)
                        hwetag  = f"m{m}_w{w}_e{e}" if (m is not None and w is not None and e is not None) else "any_hparams"
                        sub_out = outdir / f"exp-{safe_exp}" / f"{strat}-{pct_dir}" / hwetag / f"x={safe_x}"
                        sub_out.mkdir(parents=True, exist_ok=True)

                        cmd = [args.python_exe, str(Path(args.plot_script).resolve()),
                               "--perf-csv", str(Path(args.perf_csv).resolve()),
                               "--sim-csv",  str(Path(args.sim_csv).resolve()),
                               "--expanded", exp,
                               "--x-metric", xm,
                               "--outdir", str(sub_out)]
                        # 仍然传 strategy/pct，若你的绘图脚本不支持，可删掉两行
                        cmd += ["--strategy", strat, "--pct", str(pct)]
                        if m is not None and w is not None and e is not None:
                            cmd += ["--modes", str(m), "--width", str(w), "--epochs", str(e)]
                        cmds.append(cmd)

    if not cmds:
        print("没有发现可运行的组合。")
        return

    print(f"[发现组合] 共 {len(cmds)} 条命令：")
    for i, c in enumerate(cmds, 1):
        print(f"{i:4d}: " + " ".join([f'"{x}"' if " " in x else x for x in c]))

    if args.run:
        print("\n[执行] 开始逐条运行...")
        for i, c in enumerate(cmds, 1):
            print(f"\n--- [{i}/{len(cmds)}] ---")
            try:
                subprocess.run(c, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[错误] 命令失败：{e}")
        print("\n[完成]")
    else:
        print("\n[提示] 这是 dry-run，仅打印命令。加上 --run 才会执行。")

if __name__ == "__main__":
    main()
