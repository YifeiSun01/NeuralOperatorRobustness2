import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from PIL import Image
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import re

def group_pkl_files(folder_path):
    """
    输入一个文件夹路径，返回按参数分组的 .pkl 文件字典。
    
    Args:
        folder_path (str or Path): 包含 .pkl 文件的文件夹
    
    Returns:
        dict: { (alpha=..., epsilon=..., steps=..., idx=...): { attack_key: abs_path, ... }, ... }
    """
    folder = Path(folder_path)
    pkl_files = list(folder.glob("*.pkl"))

    grouped = defaultdict(dict)

    for path in pkl_files:
        fname = path.stem  # 自动去掉.pkl后缀
        
        # 更精确的正则表达式，严格匹配所有部分
        pattern = r"""
            ^pgd_attack_records_
            norm(?P<norm>\d+)_                # norm 后跟数字
            alpha(?P<alpha>[\d\.]+)_          # alpha 后跟浮点数
            epsilon(?P<epsilon>[\d\.]+)_      # epsilon 后跟浮点数
            steps(?P<steps>\d+)_              # steps 后跟数字
            idx(?P<idx>\d+)_                  # idx 后跟数字
            (?P<attack_type>                  # attack_type 的备选项
                detached\d+(to\d+)?(_\d+)?    # detachedX(toY)?(_Z)? 的形式
                |constant\d+(to\d+)?(_\d+)?   # constantX(toY)?(_Z)? 的形式
                |approximated
                |withsolver
            )
            $
        """
        
        match = re.match(pattern, fname, re.VERBOSE)
        if not match:
            print(f"Warning: 无法解析文件名: {fname}")
            continue
        
        params = match.groupdict()
        attack_type = params['attack_type']
        
        # 不再修改attack_type，保留原始值
        # detached1to9、detached19、detached1to9_19 将作为三种不同攻击类型
        
        param_key = (
            f"alpha={params['alpha']}",
            f"epsilon={params['epsilon']}",
            f"steps={params['steps']}",
            f"idx={params['idx']}"
        )
        
        grouped[param_key][attack_type] = str(path.resolve())

    return grouped

def visualize_attack_comparison(
    param_key,
    files_dict,
    fps=2,
    dpi=100,
    approximated_prefix="surrogate_",
    panel="diff",                 # ← 新增：选择要画的面板类型
    cmap="viridis"                # 颜色表可改：'viridis'/'coolwarm' 等
):
    """
    panel 可选：
        'diff'         : output - truth（默认）
        'grad'         : 原始梯度 gradient
        'scaled_grad'  : Adam 缩放后梯度 scaled_gradient
        'x0'           : 当前对抗输入 x0
        'output'       : 预测
        'truth'        : 真值
    """

    # 读取文件
    file_map = files_dict[param_key]
    script_dir = Path(".").resolve()
    temp_dir = script_dir / "temp_frames"
    temp_dir.mkdir(exist_ok=True)

    data_per_file = {}
    print("加载文件...")
    for attack_name, fpath in file_map.items():
        with open(fpath, "rb") as f:
            payload = pickle.load(f)
        records = payload["steps"]
        data_per_file[attack_name] = records

    # 计算全局色标（不同面板不同规则）
    def to_np(x):
        if isinstance(x, np.ndarray):
            return x
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # 遍历所有记录，得到 vmin/vmax
    vmax = 0.0
    vmin = 0.0
    has_any = False

    def update_range(arr, mode):
        nonlocal vmax, vmin, has_any
        arr = to_np(arr)
        if mode in ("diff", "grad", "scaled_grad"):
            amax = float(np.max(np.abs(arr)))
            if amax > vmax:
                vmax = amax
            has_any = True
        else:
            amax = float(np.max(arr))
            amin = float(np.min(arr))
            if not has_any:
                vmax, vmin = amax, amin
            else:
                vmax = max(vmax, amax)
                vmin = min(vmin, amin)
            has_any = True

    for attack_name, records in data_per_file.items():
        for rec in records:
            if panel == "diff":
                truth = rec[f"{approximated_prefix}truth"] if attack_name == "approximated" else rec["truth"]
                update_range(to_np(rec["output"]) - to_np(truth), "diff")
            elif panel == "grad":
                if "gradient" in rec:
                    update_range(rec["gradient"], "grad")
            elif panel == "scaled_grad":
                if "scaled_gradient" in rec:
                    update_range(rec["scaled_gradient"], "scaled_grad")
            elif panel == "x0":
                update_range(rec["x0"], "x0")
            elif panel == "output":
                update_range(rec["output"], "output")
            elif panel == "truth":
                truth = rec[f"{approximated_prefix}truth"] if attack_name == "approximated" else rec["truth"]
                update_range(truth, "truth")
            else:
                raise ValueError(f"未知面板类型 panel={panel}")

    # 对称色标（仅对“差分/梯度类”）
    use_symmetric = panel in ("diff", "grad", "scaled_grad")
    if use_symmetric:
        vmin_plot, vmax_plot = -vmax, vmax
    else:
        vmin_plot, vmax_plot = vmin, vmax

    # 一些标题片段
    panel_title_map = {
        "diff": "Prediction − Truth",
        "grad": "Attack Gradient",
        "scaled_grad": "Scaled Gradient (Adam)",
        "x0": "Adversarial Input (x0)",
        "output": "Prediction",
        "truth": "Ground Truth",
    }
    panel_title = panel_title_map[panel]

    # 准备绘制
    num_steps = len(next(iter(data_per_file.values())))
    attack_names = sorted(data_per_file.keys())
    n_attacks = len(attack_names)

    # 布局：3×5，其中最后一格画 loss
    n_rows, n_cols = 3, 5
    total_axes = n_rows * n_cols
    max_attack_axes = total_axes - 1  # 留 1 个轴给 loss
    colors = plt.cm.get_cmap('tab20')
    color_map = {name: colors(i % 20) for i, name in enumerate(attack_names)}

    frame_files = []
    for step in range(num_steps):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(30, 15))
        axs = axs.flatten()

        # 逐攻击画所选面板
        for i, attack_name in enumerate(attack_names[:max_attack_axes]):
            ax = axs[i]
            rec = data_per_file[attack_name][step]

            # 取当前面板数据
            if panel == "diff":
                output = to_np(rec["output"])
                truth = to_np(rec[f"{approximated_prefix}truth"] if attack_name == "approximated" else rec["truth"])
                img = output - truth
                subtitle_extra = f" | mean|diff|={np.mean(np.abs(img)):.4f}"
            elif panel == "grad":
                if "gradient" not in rec:
                    ax.axis("off"); continue
                img = to_np(rec["gradient"])
                subtitle_extra = f" | mean|g|={np.mean(np.abs(img)):.4f}"
            elif panel == "scaled_grad":
                if "scaled_gradient" not in rec:
                    ax.axis("off"); continue
                img = to_np(rec["scaled_gradient"])
                subtitle_extra = f" | mean|ĝ|={np.mean(np.abs(img)):.4f}"
            elif panel == "x0":
                img = to_np(rec["x0"])
                subtitle_extra = ""
            elif panel == "output":
                img = to_np(rec["output"])
                subtitle_extra = ""
            elif panel == "truth":
                img = to_np(rec[f"{approximated_prefix}truth"] if attack_name == "approximated" else rec["truth"])
                subtitle_extra = ""
            else:
                ax.axis("off"); continue

            # reach_boundary（若存在）
            reach_boundary = rec.get("reach_boundary", None)
            rb_str = f" | reach_boundary={reach_boundary}" if reach_boundary is not None else ""

            # 画图
            if use_symmetric:
                norm = Normalize(vmin=vmin_plot, vmax=vmax_plot)
            else:
                norm = Normalize(vmin=vmin_plot, vmax=vmax_plot)

            im = ax.imshow(img, cmap=cmap, norm=norm)
            ax.set_title(f"{attack_name} | {panel_title}\nStep {step}{subtitle_extra}{rb_str}")
            plt.colorbar(im, ax=ax)

        # 画 loss 曲线（最后一个轴）
        ax_loss = axs[max_attack_axes]
        for attack_name in attack_names:
            recs = data_per_file[attack_name][:step+1]
            if attack_name == "approximated":
                ls = [r[f"{approximated_prefix}loss"] for r in recs]
            else:
                ls = [r["loss"] for r in recs]
            ax_loss.plot(range(step+1), ls, label=attack_name,
                         color=color_map[attack_name], marker='o', markersize=3, alpha=0.35)
        # loss 面板标题：按当前 step 排序展示
        loss_pairs = []
        for name in attack_names:
            rec = data_per_file[name][step]
            loss_val = rec[f"{approximated_prefix}loss"] if name == "approximated" else rec["loss"]
            loss_pairs.append((name, float(loss_val)))
        loss_pairs.sort(key=lambda x: x[1], reverse=True)
        loss_strs = [f"{n}: {int(round(v))}" for n, v in loss_pairs]
        half = len(loss_strs) // 2
        loss_info = " | ".join(loss_strs[:half]) + "\n" + " | ".join(loss_strs[half:])

        # 全程固定 y 轴范围（用全局 min/max）
        # 重新扫描时可缓存到上面，但这里简单做一次
        all_losses = []
        for name, recs in data_per_file.items():
            if name == "approximated":
                all_losses += [r[f"{approximated_prefix}loss"] for r in recs]
            else:
                all_losses += [r["loss"] for r in recs]
        min_loss = float(np.min(all_losses))
        max_loss = float(np.max(all_losses))

        ax_loss.set_title(f"Loss Progression (0 → {step})\n{loss_info}")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_xlim(-0.5, num_steps - 0.5)
        ax_loss.set_ylim(min_loss * 0.9, max_loss * 1.1)
        ax_loss.legend()
        ax_loss.grid(True)

        # 其余多余轴关掉
        for j in range(max_attack_axes+1, total_axes):
            axs[j].axis('off')

        plt.tight_layout()
        frame_path = temp_dir / f"frame_{step:04d}.png"
        plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
        frame_files.append(frame_path)
        plt.close()

    # 合成 GIF
    images = [Image.open(f) for f in frame_files]
    out_dir = Path(__file__).parent / "gifs"
    out_dir.mkdir(exist_ok=True)
    # 文件名包含 panel
    if approximated_prefix == "surrogate_":
        output_path = out_dir / f"attack_comparison_{'_'.join(param_key)}_{panel}_surrogate.gif"
    elif approximated_prefix == "":
        output_path = out_dir / f"attack_comparison_{'_'.join(param_key)}_{panel}_true.gif"
    else:
        output_path = out_dir / f"attack_comparison_{'_'.join(param_key)}_{panel}.gif"

    images[0].save(output_path, save_all=True, append_images=images[1:], duration=1000//fps, loop=0)

    # 清理
    for f in frame_files:
        os.remove(f)
    temp_dir.rmdir()
    print(f"✅ GIF 保存到: {output_path}")

if __name__ == "__main__":
    folder_path = Path(__file__).parent / "pickle_files"
    grouped = group_pkl_files(folder_path) 
    print(grouped)
    for key_tuple in grouped:
        # 画“原始梯度”
        visualize_attack_comparison(key_tuple, grouped, fps=5, dpi=100, approximated_prefix = "surrogate_", panel="grad", cmap="viridis")

        # 画“当前对抗输入 x0”
        visualize_attack_comparison(key_tuple, grouped, fps=5, dpi=100, approximated_prefix = "surrogate_", panel="x0", cmap="viridis")

