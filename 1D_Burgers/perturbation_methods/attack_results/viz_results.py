# -*- coding: utf-8 -*-
import os
import re
import pickle
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# 1) 扫描 pickle_files 目录
# ---------------------------
def list_all_pickles(base_dir: Path, levels: int | None = None) -> list[dict]:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"'pickle_files' not found: {base_dir}")

    results = []
    files = sorted(p for p in base_dir.rglob("*.pkl") if p.is_file())
    print(f"[SCAN] Found {len(files)} .pkl under {base_dir}")

    for p in files:
        rel_path = p.relative_to(base_dir)
        rel_parts_all = list(rel_path.parts[:-1])

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
            "rel_parts": rel_parts_all,
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
# 3) 提取四条 loss 曲线
# ---------------------------
LOSS_KEYS = ("with_solver", "without_solver", "approximate", "approximate_surrogate")

def extract_loss_series(grad_records):
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
            try:
                v = float(v) if v is not None else None
            except Exception:
                v = None
            series[lk].append(v)
    return series, steps

# ---------------------------
# 4A) 单文件：两子图/NaN-robust
# ---------------------------
def plot_two_subplots(series, steps, figure_title, out_path: Path):
    if not steps:
        print("[Skip] Empty steps ->", out_path)
        return

    import numpy as np, math
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

    arr_with    = to_nan_array(series.get("with_solver", []))
    arr_without = to_nan_array(series.get("without_solver", []))
    arr_approx  = to_nan_array(series.get("approximate", []))
    arr_surr    = to_nan_array(series.get("approximate_surrogate", []))

    all_finite = np.concatenate([
        arr[np.isfinite(arr)] for arr in [arr_with, arr_without, arr_approx, arr_surr] if arr.size > 0
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
        if np.isfinite(y_arr).any():
            ax.plot(steps_np, y_arr,
                    label=label_first_last(y_arr, base_label),
                    linewidth=2, **kwargs)

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

    ax_left = axes[0]
    common(ax_left)
    plot_line(ax_left, arr_approx, "approximated (true loss)", linestyle="-.")
    ax_left.set_title(
        f"Loss curves (approximated TRUE) | approximated first={fmt_any(approx_first)}, "
        f"last={fmt_any(approx_last)}, NaNs={approx_nans}"
    )
    ax_left.legend(loc="best")

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
# 4B) 组合：跨多个 approxN 画一张图
# ---------------------------
def plot_combined_across_approxN(
    steps, base_with, base_without, approx_map, surr_map,
    figure_title, out_path: Path
):
    """
    steps: list[int]
    base_with/base_without: list[float] for the shared curves (from any file)
    approx_map: {approxN: list[float]} for 'approximate'
    surr_map:   {approxN: list[float]} for 'approximate_surrogate'
    """
    import numpy as np, math
    from matplotlib.colors import Normalize

    if not steps:
        print("[Skip] Empty steps ->", out_path)
        return

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

    arr_with    = to_nan_array(base_with)
    arr_without = to_nan_array(base_without)

    approxNs = sorted(approx_map.keys())
    arr_approx_list = [(N, to_nan_array(approx_map[N])) for N in approxNs]
    arr_surr_list   = [(N, to_nan_array(surr_map[N]))   for N in approxNs]

    # y-lims from all finite values
    all_arrays = [arr_with, arr_without] + [a for _, a in arr_approx_list] + [a for _, a in arr_surr_list]
    all_finite = np.concatenate([a[np.isfinite(a)] for a in all_arrays if a.size > 0]) if steps_np.size else np.array([])
    ymin, ymax = (np.min(all_finite), np.max(all_finite)) if all_finite.size else (None, None)

    # colormap for approxN curves
    cmap = plt.cm.viridis
    vmin, vmax = (min(approxNs), max(approxNs)) if approxNs else (0.0, 1.0)

    def label_first_last(arr, base_label):
        finite_idx = np.where(np.isfinite(arr))[0]
        if finite_idx.size == 0:
            return base_label
        first = arr[finite_idx[0]]
        last  = arr[finite_idx[-1]]
        return f"{base_label} (first={first:.3f}, last={last:.3f})"

    def plot_base(ax):
        if np.isfinite(arr_with).any():
            ax.plot(steps_np, arr_with, label=label_first_last(arr_with, "with solver"),
                    linewidth=2)
        if np.isfinite(arr_without).any():
            ax.plot(steps_np, arr_without, label=label_first_last(arr_without, "detached"),
                    linewidth=2, linestyle="--")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True)
        if (ymin is not None) and (ymax is not None) and (ymin < ymax):
            pad = 0.05 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

    def fmt_any(v):
        if v is None:
            return "NA"
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "NaN" if math.isnan(v) else ("+Inf" if v > 0 else "-Inf")
        return f"{v:.3f}"

    def title_suffix(arr_list, label):
        # 聚合每条 approxN 的首尾/NaN 概况用于子图标题补充
        parts = []
        for N, arr in arr_list:
            finite_idx = np.where(np.isfinite(arr))[0]
            first = arr[finite_idx[0]] if finite_idx.size else None
            last  = arr[finite_idx[-1]] if finite_idx.size else None
            nans  = int(np.isnan(arr).sum())
            parts.append(f"approxN={N}({fmt_any(first)},{fmt_any(last)},NaNs={nans})\n")
        return f"{label} | " + "; ".join(parts) if parts else label

    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
    fig.suptitle(figure_title, y=0.98)

    cmap = plt.cm.get_cmap("viridis")  # 或沿用你的cmap
    Ns_ranked = sorted({N for N, _ in arr_approx_list} | {N for N, _ in arr_surr_list})
    positions = np.linspace(0.0, 1.0, len(Ns_ranked)) if len(Ns_ranked) > 1 else [0.5]
    color_map = {N: cmap(pos) for N, pos in zip(Ns_ranked, positions)}

    # Left: true
    axL = axes[0]
    plot_base(axL)
    for N, arr in arr_approx_list:
        if np.isfinite(arr).any():
            axL.plot(
                steps_np, arr,
                label=label_first_last(arr, f"approxN={N} (true)"),
                linewidth=2, linestyle="-.",
                color=color_map[N],   # <-- 按排名取色
            )
    axL.set_title(title_suffix(arr_approx_list, "Loss curves (TRUE approximate)"))
    axL.legend(loc="best")

    # Right: surrogate
    axR = axes[1]
    plot_base(axR)
    for N, arr in arr_surr_list:
        if np.isfinite(arr).any():
            axR.plot(
                steps_np, arr,
                label=label_first_last(arr, f"approxN={N} (surrogate)"),
                linewidth=2, linestyle="-.",
                color=color_map[N],   # <-- 按排名取色
            )
    axR.set_title(title_suffix(arr_surr_list, "Loss curves (SURROGATE approximate)"))
    axR.legend(loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# 5) 主流程：按组（同 stem & 顶层文件夹）合并 approxN
# ---------------------------
def main():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir / "pickle_files"
    if not base_dir.exists():
        raise FileNotFoundError(f"'pickle_files' not found next to this script: {base_dir}")

    print(f"[Scan] Base folder: {base_dir}")
    files = list_all_pickles(base_dir)
    print(f"[Found] {len(files)} pickle files")

    # 识别 approxN 值
    def parse_approxN(rel_parts):
        for part in rel_parts:
            m = re.match(r"^approxN(\d+)$", part)
            if m:
                return int(m.group(1))
        return None

    # 分组：key = (top_folder, stem) 例如 ('nu=0.01_PGD_results', 'gradient_test_exponax_nu0.01_nsamples1')
    from collections import defaultdict
    groups = defaultdict(list)
    for item in files:
        rel_parts = item["rel_parts"]
        top = rel_parts[0] if rel_parts else "_root"
        groups[(top, item["stem"])].append({
            **item,
            "approxN": parse_approxN(rel_parts)
        })

    out_root = script_dir / "plots"

    # 遍历每个组
    for (top, stem), file_list in groups.items():
        # 把所有文件加载进来
        loaded = []
        for it in file_list:
            try:
                with open(it["abs_path"], "rb") as f:
                    data = pickle.load(f)
                loaded.append((it, data))
            except Exception as e:
                print(f"[Warn] Failed to load: {it['abs_path']} -> {e}")

        if not loaded:
            continue

        # 如果只有一个 approxN，沿用单文件绘图（保存目录仍按原相对路径）
        if len(loaded) == 1:
            it, data = loaded[0]
            rel_parts = it["rel_parts"]
            for tuple_key, grad_records in data.items():
                tk = tuple(tuple_key) if isinstance(tuple_key, (list, tuple)) else (str(tuple_key),)
                info = parse_tuple_key(tk)
                series, steps = extract_loss_series(grad_records)
                if not steps:
                    print(f"[Skip] No step_update/loss_metrics for key={tk}")
                    continue
                rel_folder = "/".join(rel_parts) if rel_parts else "."
                fig_title = (
                    f"Loss Curves — file: {stem}  |  under: {rel_folder}\n"
                    f"norm: {info['norm']} | index: {info['index']} | steps: {info['numsteps']} | "
                    f"epsilon: {info['epsilon']} | alpha: {info['alpha']}"
                )
                out_dir = out_root.joinpath(*rel_parts, stem)
                safe_norm = str(info["norm"]).replace("/", "_")
                out_name = (
                    f"{stem}__norm={safe_norm}_index={info['index']}_steps={info['numsteps']}_"
                    f"epsilon={info['epsilon']}_alpha={info['alpha']}.png"
                )
                out_path = out_dir / out_name
                plot_two_subplots(series, steps, fig_title, out_path)
                print(f"  -> Saved: {out_path}")
            continue

        # 多个 approxN：需要合成
        approxNs_in_group = sorted({it["approxN"] for it, _ in loaded if it["approxN"] is not None})
        print(f"[Combine] {top}/{stem}  approxNs={approxNs_in_group}")

        # 收集该组内所有 tuple_key
        all_keys = set()
        for _, data in loaded:
            if isinstance(data, dict):
                all_keys.update(data.keys())

        # 逐个 tuple_key 合成
        for tuple_key in all_keys:
            # 该 key 在不同 approxN 下的曲线
            approx_map = {}  # N -> approximate list
            surr_map   = {}  # N -> approximate_surrogate list
            base_with, base_without, base_steps = None, None, None

            # 先拿到最小的步数长度，避免长度差异
            lengths = []
            for _, data in loaded:
                if tuple_key in data:
                    series, steps = extract_loss_series(data[tuple_key])
                    lengths.append(len(steps))
            if not lengths:
                continue
            min_len = min(lengths)

            # 收集数据
            for it, data in loaded:
                if tuple_key not in data:
                    continue
                series, steps = extract_loss_series(data[tuple_key])
                # 截到相同长度（如果不一致）
                steps = steps[:min_len]
                if base_steps is None:
                    base_steps = steps

                # 基准两条线（取第一次出现的）
                if base_with is None:
                    base_with = series["with_solver"][:min_len]
                if base_without is None:
                    base_without = series["without_solver"][:min_len]

                N = it["approxN"]
                if N is not None:
                    approx_map[N] = series["approximate"][:min_len]
                    surr_map[N]   = series["approximate_surrogate"][:min_len]

            if not approx_map and not surr_map:
                # 没有任何 approxN 数据，跳过
                continue

            # 标题/输出名（注意：合成图保存在不含 approxN 层的“combined”目录）
            tk = tuple(tuple_key) if isinstance(tuple_key, (list, tuple)) else (str(tuple_key),)
            info = parse_tuple_key(tk)
            rel_folder = top  # 去掉 approxN 层
            fig_title = (
                f"Combined Loss Curves across approxN — file: {stem}  |  under: {rel_folder}\n"
                f"norm: {info['norm']} | index: {info['index']} | steps: {info['numsteps']} | "
                f"epsilon: {info['epsilon']} | alpha: {info['alpha']}\n"
                f"approxN: {','.join(map(str, sorted(approx_map.keys())))}"
            )

            # 保存到：plots/<top>/<stem>/_combined/
            out_dir = out_root.joinpath(top, stem, "_combined")
            safe_norm = str(info["norm"]).replace("/", "_")
            approx_tag = ",".join(map(str, sorted(approx_map.keys()))) if approx_map else "NA"
            out_name = (
                f"{stem}__norm={safe_norm}_index={info['index']}_steps={info['numsteps']}_"
                f"epsilon={info['epsilon']}_alpha={info['alpha']}__approxNs={approx_tag}.png"
            )
            out_path = out_dir / out_name

            plot_combined_across_approxN(
                steps=base_steps,
                base_with=base_with or [],
                base_without=base_without or [],
                approx_map=approx_map,
                surr_map=surr_map,
                figure_title=fig_title,
                out_path=out_path
            )
            print(f"  -> Combined Saved: {out_path}")

    print("\n[Done] All figures saved under:", out_root)

if __name__ == "__main__":
    main()