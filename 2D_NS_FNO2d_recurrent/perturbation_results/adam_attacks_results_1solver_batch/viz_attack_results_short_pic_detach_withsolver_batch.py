#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------- 基本配置 ----------------
# 只画“单个 sample”的 loss 曲线；一律用 output & truth 现算
REDUCTION   = "mean"   # 'mean' 或 'sum'：单样本像素级 MSE 的平均 / 求和
FOLDER_HINT = "perturbation_results_batch5_integer_frames"  # 仅用于输出目录命名
DPI         = 120

# 用于文件/目录命名与坐标轴标签
METRIC_TAG   = f"mse_{REDUCTION}"              # 保存文件名用
METRIC_LABEL = f"Per-sample MSE ({REDUCTION})" # y 轴标签

# ---------------- 名称解析（兜底） ----------------
_RUN_DIR_RE = re.compile(
    r"^modespec=(?P<modespec>[adw]{10})_"
    r"norm(?P<norm>[^_]+)_alpha(?P<alpha>[^_]+)_"
    r"(?:eps|epsilon)(?P<epsilon>[^_]+)_steps(?P<steps>\d+)_idx(?P<idx_tag>.+)$"
)
_SAMPLE_STEM_RE = re.compile(r"^sample_(?P<gid>\d{6})$")
_SAMPLE_NAME_RE = re.compile(r"^sample_(?P<gid>\d{6})\.pkl$")

def _parse_run_dir_name(name: str):
    m = _RUN_DIR_RE.match(name)
    return m.groupdict() if m else None

def _parse_sample_gid(path: Path):
    m = _SAMPLE_STEM_RE.match(path.stem) or _SAMPLE_NAME_RE.match(path.name)
    return int(m.group("gid")) if m else None

# ---------------- 自动探测根目录 ----------------
def autodetect_scan_roots(candidates):
    """
    在候选目录里找包含 per_sample/sample_*.pkl 的树，返回按文件数降序排序。
    """
    seen, hits = set(), []
    for c in candidates:
        if not c:
            continue
        p = Path(c).resolve()
        if p in seen or not p.exists():
            continue
        seen.add(p)
        per_samples = list(p.rglob("per_sample/sample_*.pkl"))
        if per_samples:
            hits.append({"root": p, "count": len(per_samples)})
    hits.sort(key=lambda x: x["count"], reverse=True)
    return hits

# ---------------- 每个 sample：现算单步 MSE 并取序列 ----------------
def _finite_only(seq):
    arr = np.asarray(seq, dtype=float)
    mask = np.isfinite(arr)
    return arr[mask].tolist()

def load_single_sample_loss_series(sample_pkl: Path, reduction: str = "mean"):
    """
    只返回【该 sample】每一步的损失序列。
    一律用 output 和 truth/surrogate_truth 现算：
      - reduction='mean' -> 像素平均 MSE
      - reduction='sum'  -> 像素求和 SSE
    忽略 'loss'/'surrogate_loss' 字段（那是 batch 标量）。
    """
    with open(sample_pkl, "rb") as f:
        payload = pickle.load(f)

    steps = payload.get("steps")
    if not (isinstance(steps, list) and steps and isinstance(steps[0], dict)):
        return None

    ys = []
    for rec in steps:
        out = rec.get("output")
        tru = rec.get("truth") if rec.get("truth") is not None else rec.get("surrogate_truth")
        if out is None or tru is None:
            continue
        se = (np.asarray(out, dtype=float) - np.asarray(tru, dtype=float)) ** 2
        val = float(se.mean()) if reduction == "mean" else float(se.sum())
        if np.isfinite(val):
            ys.append(val)

    return _finite_only(ys) if ys else None

# ---------------- 标题元数据（用于图标题，非必要） ----------------
def extract_meta(sample_pkl: Path):
    info = {
        "modespec": None, "norm": None, "alpha": None, "epsilon": None,
        "steps": None, "gid": _parse_sample_gid(sample_pkl)
    }
    try:
        with open(sample_pkl, "rb") as f:
            payload = pickle.load(f)
        # 修正：这里必须判断 dict 类型，而不是字符串 "dict"
        md = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    except Exception:
        md = {}
    if isinstance(md, dict):
        info["modespec"] = md.get("mode_spec") or md.get("modespec")
        info["norm"]     = md.get("norm")
        info["alpha"]    = md.get("alpha")
        info["epsilon"]  = md.get("epsilon") or md.get("eps")
        info["steps"]    = md.get("num_steps") or md.get("steps")

    # 兜底：从 run_dir 名称解析
    if any(info[k] is None for k in ["modespec", "norm", "alpha", "epsilon", "steps"]):
        run_dir = sample_pkl.parents[1].name
        parsed = _parse_run_dir_name(run_dir)
        if parsed:
            for k in ["modespec", "norm", "alpha", "epsilon", "steps"]:
                info[k] = info[k] or parsed.get(k)
    return info

# ---------------- 画单个 sample 的曲线 ----------------
def plot_single_sample_loss(sample_pkl: Path, out_root: Path, reduction: str = "mean", dpi: int = 120):
    ys = load_single_sample_loss_series(sample_pkl, reduction=reduction)
    if not ys:
        print(f"⚠️ 空序列，跳过: {sample_pkl}")
        return None

    meta = extract_meta(sample_pkl)
    xs = list(range(len(ys)))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(xs, ys, marker="o", markersize=3, linewidth=2, alpha=0.9)
    ax.set_xlabel("Step")
    ax.set_ylabel(METRIC_LABEL)

    title = f"Sample {meta.get('gid')}"
    ms = meta.get("modespec") or "?"
    eps = meta.get("epsilon")
    stp = meta.get("steps")
    title += f"  | modespec={ms}, eps={eps}, steps={stp}"
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()

    # ====== 关键修改：把 run_dir 名字放进输出路径，避免覆盖，并包含 alpha/epsilon 等 ======
    run_dir_name = sample_pkl.parents[1].name  # e.g. modespec=..._norm..._alpha..._epsilon..._steps..._idx...
    out_dir = out_root.parent.parent / "results" / FOLDER_HINT / "per_sample_loss" / METRIC_TAG / run_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_pkl.stem}_{METRIC_TAG}.png"

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_path}")
    return out_path

# ---------------- 主入口 ----------------
if __name__ == "__main__":
    # 1) 候选根目录（优先环境变量）
    env_root = os.getenv("RESULTS_ROOT", "").strip()
    candidates = []
    if env_root:
        candidates.append(Path(env_root))

    # 2) 常见候选（脚本同目录的 pickle_files/<folder>）
    here = Path(__file__).parent
    candidates += [here / "pickle_files" / FOLDER_HINT]

    hits = autodetect_scan_roots(candidates)
    if not hits:
        print("⚠️ 没找到 per_sample/sample_*.pkl")
        print("可以设置环境变量，例如：")
        print("  RESULTS_ROOT=/blue/.../perturbation_results_batch4_integer_frames  python plot_per_sample_loss.py")
        raise SystemExit(1)

    # 按你的要求：不要改这个索引
    best = hits[0]
    scan_root = best["root"]
    print(f"🔎 使用根目录: {scan_root}  (检测到 per-sample PKL 数: {best['count']})")

    # 3) 遍历全部 per-sample PKL 并逐个画图
    per_samples = sorted(scan_root.rglob("per_sample/sample_*.pkl"))
    ok, fail = 0, 0
    for sp in per_samples:
        try:
            plot_single_sample_loss(sp, out_root=scan_root, reduction=REDUCTION, dpi=DPI)
            ok += 1
        except Exception as e:
            print(f"❌ 失败: {sp}  -> {e}")
            fail += 1

    print(f"\n完成：成功 {ok} 个，失败 {fail} 个。")



