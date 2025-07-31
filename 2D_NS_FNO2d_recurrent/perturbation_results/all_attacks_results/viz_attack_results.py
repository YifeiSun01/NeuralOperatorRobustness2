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

def visualize_attack_comparison(param_key, files_dict, fps=2, dpi=100, approximated_prefix = "surrogate_"):
    
    file_map = files_dict[param_key]
    script_dir = Path(".").resolve()
    temp_dir = script_dir / "temp_frames"
    temp_dir.mkdir(exist_ok=True)

    data_per_file = {}
    max_diff = 0.0
    max_loss = 0.0
    min_loss = float("inf")

    print(f"加载文件...")
    for attack_name, fpath in file_map.items():
        with open(fpath, "rb") as f:
            records = pickle.load(f)["steps"]
            data_per_file[attack_name] = records
            for rec in records:
                if attack_name == "approximated":
                    truth = rec[f'{approximated_prefix}truth']
                else:
                    truth = rec['truth']
                diff = rec['output'] - truth
                max_diff = max(max_diff, np.max(np.abs(diff)))
            if attack_name == "approximated":
                losses = [r[f'{approximated_prefix}loss'] for r in records]
            else:
                losses = [r['loss'] for r in records]
            max_loss = max(max_loss, max(losses))
            min_loss = min(min_loss, min(losses))

    num_steps = len(next(iter(data_per_file.values())))

    frame_files = []
    attack_names = sorted(data_per_file.keys())
    n_attacks = len(attack_names)

    for step in range(num_steps):
        fig, axs = plt.subplots(3, 5, figsize=(30, 15))
        axs = axs.flatten()

        # 前6个图画diff heatmap，最多画6个攻击方法，少于6时空着
        for i in range(11):
            ax = axs[i]
            if i < n_attacks:
                attack_name = attack_names[i]
                rec = data_per_file[attack_name][step]
                output = rec['output']
                if attack_name == "approximated":
                    truth = rec[f'{approximated_prefix}truth']
                else:
                    truth = rec['truth']
                reach_boundary = rec['reach_boundary']
                diff = output - truth
                mean_abs_diff = np.mean(np.abs(diff))

                im = ax.imshow(diff, cmap='coolwarm', norm=Normalize(vmin=-max_diff, vmax=max_diff))
                ax.set_title(f"{attack_name}\nStep {step} mean_abs_diff={mean_abs_diff:.4f}\nreach_boundary={reach_boundary}")
                plt.colorbar(im, ax=ax)
            else:
                ax.axis('off')  # 空白图隐藏坐标轴

        colors = plt.cm.get_cmap('tab10')  # 取10种不同颜色，或者换成别的调色板
        color_map = {attack_name: colors(i % 10) for i, attack_name in enumerate(attack_names)}

        ax_loss = axs[11]
        for attack_name in attack_names:
            if attack_name == "approximated":
                losses = [r[f'{approximated_prefix}loss'] for r in data_per_file[attack_name][:step+1]]
            else:
                losses = [r['loss'] for r in data_per_file[attack_name][:step+1]]
            ax_loss.plot(range(step+1), losses, label=attack_name, color=color_map[attack_name], marker='o', markersize=3, alpha=0.3)

        loss_pairs = []
        for name in attack_names:
            if name == "approximated":
                # For "approximated" attack, use 'surrogate_loss'
                loss = data_per_file[name][step][f'{approximated_prefix}loss']
            else:
                # For other attacks, use regular 'loss'
                loss = data_per_file[name][step]['loss']
            loss_pairs.append((name, loss))
        loss_pairs.sort(key=lambda x: x[1], reverse=True)  # 按 loss 降序排序
        loss_strs = [f"{name}: {int(round(loss_val))}" for name, loss_val in loss_pairs]

        # 分成两行，中间加换行符
        half = len(loss_strs) // 2
        loss_info = " | ".join(loss_strs[:half]) + "\n" + " | ".join(loss_strs[half:])

        ax_loss.set_title(f"Loss Progression (step 0 to {step})\n{loss_info}")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_xlim(-0.5, num_steps - 0.5)
        ax_loss.set_ylim(min_loss * 0.9, max_loss * 1.1)
        ax_loss.legend()
        ax_loss.grid(True)

        # 剩余两个图隐藏
        for j in [11,]:
            axs[j].axis('off')

        plt.tight_layout()
        frame_path = temp_dir / f"frame_{step:04d}.png"
        plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
        frame_files.append(frame_path)
        plt.close()

    # 合成 GIF
    images = [Image.open(f) for f in frame_files]
    if approximated_prefix == "surrogate_":
        output_path = Path(__file__).parent / "gifs" / f"attack_comparison_{'_'.join(param_key)}_surrogate.gif"
    elif approximated_prefix == "":
        output_path = Path(__file__).parent / "gifs" / f"attack_comparison_{'_'.join(param_key)}_true.gif"
    else:
        pass
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=1000//fps, loop=0)

    # 清理临时文件
    for f in frame_files:
        os.remove(f)
    temp_dir.rmdir()
    print(f"✅ GIF 保存到: {output_path}")

if __name__ == "__main__":
    folder_path = Path(__file__).parent / "pickle_files"
    grouped = group_pkl_files(folder_path) 
    print(grouped)
    for key_tuple in grouped:
        visualize_attack_comparison(key_tuple, grouped, fps=5, dpi=100, approximated_prefix = "surrogate_")
        visualize_attack_comparison(key_tuple, grouped, fps=5, dpi=100, approximated_prefix = "")

