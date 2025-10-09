#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---- 工具：统一生成 2x3 子图网格（5 个子图 + 1 个隐藏位） ----
def make_axes_2x3(figsize=(12, 8)):
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    # 固定隐藏最后一个轴位（只显示前 5 个）
    axes[-1].axis("off")
    return fig, axes

# ---- 工具：生成语义标签 ----
def kept_fourier_labels(K: int):
    # 约定：k=0 -> none；其余为 "1", "1,2", "1,2,3", "1,2,3,4"
    labels = ["none"]
    for k in range(1, K):
        labels.append(",".join(str(i) for i in range(1, k+1)))
    return labels

def load_variant_tensor(file_path: Path):
    """
    加载单个 .pt 文件，返回 numpy 数组 (N, H, W, K)
    """
    t = torch.load(file_path, map_location="cpu")
    if isinstance(t, torch.Tensor):
        arr = t.numpy()
    else:
        raise TypeError(f"Expect a torch.Tensor saved at {file_path}, got {type(t)}")
    return arr

def plot_first10_samples(arr: np.ndarray, out_dir: Path, name_prefix=""):
    """
    对前 10 条样本画热图。
    统一版式：一张图 5 个子图（2x3 布局，隐藏最后一个位点）。
    子图标题：kept fourier layers: none / 1 / 1,2 / 1,2,3 / 1,2,3,4
    """
    N, H, W, K = arr.shape
    if K < 5:
        raise ValueError(f"Expect at least 5 variants on last dim, got K={K}.")
    labels = kept_fourier_labels(5)  # 只画前5个
    n_plot = min(10, N)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_plot):
        fig, axes = make_axes_2x3(figsize=(12, 8))
        fig.suptitle(f"{name_prefix} | sample {i}")

        # 只用前 5 个子图位
        for k in range(5):
            ax = axes[k]
            sns.heatmap(arr[i, :, :, k], ax=ax, cmap="viridis")
            ax.set_title(f"kept fourier layers: {labels[k]}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        savep = out_dir / f"{name_prefix}__sample_{i}.png"
        fig.savefig(savep, dpi=150)
        plt.close(fig)

def plot_stat_maps(arr: np.ndarray, out_dir: Path, name_prefix=""):
    """
    对 N 条样本在空间每个格点、每个变体，计算：
      mean, median, 25%, 75%, min, max, std
    并以统一版式作图：一张图 5 个子图（2x3，隐藏最后一个格）。
    """
    N, H, W, K = arr.shape
    if K < 5:
        raise ValueError(f"Expect at least 5 variants on last dim, got K={K}.")
    labels = kept_fourier_labels(5)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 统计量（沿样本维 N 计算）
    mean   = arr.mean(axis=0)                 # (H,W,K)
    median = np.median(arr, axis=0)           # (H,W,K)
    pct25  = np.percentile(arr, 25, axis=0)   # (H,W,K)
    pct75  = np.percentile(arr, 75, axis=0)   # (H,W,K)
    mn     = arr.min(axis=0)                  # (H,W,K)
    mx     = arr.max(axis=0)                  # (H,W,K)
    std    = arr.std(axis=0)                  # (H,W,K)

    stats = {
        "mean":   mean,
        "median": median,
        "p25":    pct25,
        "p75":    pct75,
        "min":    mn,
        "max":    mx,
        "std":    std
    }

    for stat_name, mat in stats.items():
        fig, axes = make_axes_2x3(figsize=(12, 8))
        fig.suptitle(f"{name_prefix} | {stat_name}")

        # 只用前 5 个子图位
        for k in range(5):
            ax = axes[k]
            sns.heatmap(mat[:, :, k], ax=ax, cmap="viridis")
            ax.set_title(f"kept fourier layers: {labels[k]}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        savep = out_dir / f"{name_prefix}__{stat_name}.png"
        fig.savefig(savep, dpi=150)
        plt.close(fig)

def main():
    """
    输入（与上个推理脚本一致）：
      pred_root = /.../model_explainability/other_forcing_patterns/infer_results
        └── <pattern>/
            └── <model_tag>__<pattern>__variants_keep0to{MAX_KEEP}__pred11.pt

    输出：
      plots_first10/<pattern>/<stem>/*.png
      plots_stats/<pattern>/<stem>/*.png
    """
    pred_root = Path("/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/model_explainability/other_forcing_patterns/infer_results")
    out_first10_root = Path("/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/model_explainability/other_forcing_patterns/plots/plots_first10")
    out_stats_root   = Path("/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/model_explainability/other_forcing_patterns/plots/plots_stats")

    # 递归找到所有 .pt（每个 pattern 一个或多个文件）
    files = sorted(pred_root.rglob("*.pt"))
    print(f"[SCAN] Found {len(files)} files under {pred_root}")
    if not files:
        print("[WARN] No .pt files found. Please check pred_root path.")
        return

    for f in files:
        # pattern = 上一级目录名；stem = 文件名去后缀
        pattern = f.parent.name
        stem = f.stem  # e.g., modes64_width60_epochs500_Tin10_T10__isoCircles__variants_keep0to4__pred11
        name_prefix = f"{pattern}__{stem}"

        print(f"[LOAD] {f}")
        arr = load_variant_tensor(f)  # (N,H,W,K)
        if arr.ndim != 4:
            print(f"[SKIP] Expect 4-D tensor, got {arr.shape}")
            continue

        print(f"[PLOT] first10 -> {out_first10_root / pattern / stem}")
        plot_first10_samples(arr, out_first10_root / pattern / stem, name_prefix=name_prefix)

        print(f"[PLOT] stats   -> {out_stats_root / pattern / stem}")
        plot_stat_maps(arr, out_stats_root / pattern / stem, name_prefix=name_prefix)

if __name__ == "__main__":
    main()


