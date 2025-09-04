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

def visualize_attack_comparison(param_key, files_dict, dpi=100, approximated_prefix="surrogate_"):
    """
    比较不同攻击方法的最终结果（只生成最后一帧的静态图）
    参数:
        param_key: 参数组合的键
        files_dict: 包含文件路径的字典
        dpi: 图像分辨率
        approximated_prefix: 近似攻击使用的前缀
    """
    file_map = files_dict[param_key]
    script_dir = Path(".").resolve()
    
    data_per_file = {}
    max_loss = 0.0
    min_loss = float("inf")

    print(f"加载文件...")
    for attack_name, fpath in file_map.items():
        with open(fpath, "rb") as f:
            records = pickle.load(f)["steps"]
            data_per_file[attack_name] = records
            if attack_name == "approximated":
                losses = [r[f'{approximated_prefix}loss'] for r in records]
            else:
                losses = [r['loss'] for r in records]
            max_loss = max(max_loss, max(losses))
            min_loss = min(min_loss, min(losses))

    num_steps = len(next(iter(data_per_file.values())))
    attack_names = sorted(data_per_file.keys())
    
    # 只创建最后一帧的图像
    fig, ax = plt.subplots(figsize=(24, 6))
    colors = plt.cm.get_cmap('tab20')
    color_map = {attack_name: colors(i % 20) for i, attack_name in enumerate(attack_names)}

    # 绘制所有攻击方法的损失曲线
    for attack_name in attack_names:
        if attack_name == "approximated":
            losses = [r[f'{approximated_prefix}loss'] for r in data_per_file[attack_name]]
            final_loss = data_per_file[attack_name][-1][f'{approximated_prefix}loss']
        else:
            losses = [r['loss'] for r in data_per_file[attack_name]]
            final_loss = data_per_file[attack_name][-1]['loss']
        
        # 在图例标签中添加最终loss值
        label = f"{attack_name} (final loss={final_loss:.2f})"
        ax.plot(range(num_steps), losses, 
               label=label, 
               color=color_map[attack_name], 
               marker='o', 
               markersize=3, 
               alpha=0.8, 
               linewidth=2)

    # 解析参数键为更易读的格式
    param_dict = {}
    for param in param_key:
        if '=' in param:
            key, value = param.split('=', 1)
            param_dict[key] = value
    
    # 创建详细的标题
    param_title = "\n".join([f"{key}: {value}" for key, value in param_dict.items()])
    main_title = f"Attack Comparison - Parameters:\n{param_title}"

    # 设置图表标题和标签
    ax.set_title(main_title, pad=20)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_xlim(-0.5, num_steps - 0.5)
    ax.set_ylim(min_loss * 0.9, max_loss * 1.1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

    plt.tight_layout()

    # 保存图像
    output_dir = Path(__file__).parent / "comparison_plots"
    output_dir.mkdir(exist_ok=True)
    
    if approximated_prefix == "surrogate_":
        output_path = output_dir / f"attack_comparison_{'_'.join(param_key)}_surrogate.png"
    elif approximated_prefix == "":
        output_path = output_dir / f"attack_comparison_{'_'.join(param_key)}_true.png"
    else:
        output_path = output_dir / f"attack_comparison_{'_'.join(param_key)}.png"
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"✅ 图像保存到: {output_path}")

if __name__ == "__main__":
    folder_path = Path(__file__).parent / "pickle_files"
    grouped = group_pkl_files(folder_path) 
    print(grouped)
    for key_tuple in grouped:
        visualize_attack_comparison(key_tuple, grouped, dpi=100, approximated_prefix = "surrogate_")

