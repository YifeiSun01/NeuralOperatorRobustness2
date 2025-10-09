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
    加载单个 .pt 文件，返回 shape (1150, H, W, 5) 的 numpy 数组
    """
    t = torch.load(file_path, map_location="cpu")
    arr = t.numpy()
    return arr

def plot_first10_samples(arr: np.ndarray, out_dir: Path, name_prefix=""):
    """
    对前 10 条样本画热图。
    统一版式：一张图 5 个子图（2x3 布局，隐藏最后一个位点）。
    子图标题：kept fourier layers: none / 1 / 1,2 / 1,2,3 / 1,2,3,4
    """
    N, H, W, K = arr.shape
    if K < 5:
        raise ValueError(f"Expect 5 variants on last dim, got K={K}.")
    labels = kept_fourier_labels(K)
    n_plot = min(10, N)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_plot):
        fig, axes = make_axes_2x3(figsize=(12, 8))
        fig.suptitle(f"{name_prefix} sample {i}")

        # 只用前 5 个子图位
        for k in range(5):
            ax = axes[k]
            sns.heatmap(arr[i, :, :, k], ax=ax, cmap="viridis")
            ax.set_title(f"kept fourier layers: {labels[k]}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        savep = out_dir / f"{name_prefix}_sample_{i}.png"
        fig.savefig(savep, dpi=150)
        plt.close(fig)

def plot_stat_maps(arr: np.ndarray, out_dir: Path, name_prefix=""):
    """
    对 1150 条样本在空间每个格点、每个变体，计算：
      mean, median, 25%, 75%, min, max, std
    并以统一版式作图：一张图 5 个子图（2x3，隐藏最后一个格）。
    """
    N, H, W, K = arr.shape
    if K < 5:
        raise ValueError(f"Expect 5 variants on last dim, got K={K}.")
    labels = kept_fourier_labels(K)
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
        fig.suptitle(f"{name_prefix} {stat_name}")

        # 只用前 5 个子图位
        for k in range(5):
            ax = axes[k]
            sns.heatmap(mat[:, :, k], ax=ax, cmap="viridis")
            ax.set_title(f"kept fourier layers: {labels[k]}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        savep = out_dir / f"{name_prefix}_{stat_name}.png"
        fig.savefig(savep, dpi=150)
        plt.close(fig)

def main():
    # 输入目录：存放形如 (1150, 256, 256, 5) 的 .pt 结果文件
    pred_dir = Path("/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/model_explainability/infer_results")
    # 输出目录
    out_first10 = Path("/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/model_explainability/plots/plots_first10")
    out_stats   = Path("/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/model_explainability/plots/plots_stats")

    files = sorted(pred_dir.glob("*.pt"))
    print("Found files:", files)

    for f in files:
        arr = load_variant_tensor(f)  # (N,H,W,5)
        name = f.stem
        print("Plotting first10 for", name)
        plot_first10_samples(arr, out_first10 / name, name_prefix=name)
        print("Plotting stats for", name)
        plot_stat_maps(arr, out_stats / name, name_prefix=name)

if __name__ == "__main__":
    main()


