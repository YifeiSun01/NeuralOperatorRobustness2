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
    分组 .pkl 文件，按 attack base type（即 detached1to5、withsolver）为一级键，按参数为二级键，内部是各种调度方法。
    返回结构：
    {
        'detached1to5': {
            (alpha=..., epsilon=..., steps=..., idx=...): {  # 注意：这里不包含attack_type
                'Adam (Adaptive Moment Estimation)': '路径',
                ...
            }
        },
        ...
    }
    """
    folder = Path(folder_path)
    pkl_files = list(folder.glob("*.pkl"))

    grouped = defaultdict(lambda: defaultdict(dict))

    for path in pkl_files:
        fname = path.stem
        if "|" not in fname:
            print(f"❌ 跳过无效文件名（缺少'|'）: {fname}")
            continue

        base_name, lr_method = fname.split("|", 1)  # 分成主攻击类型和调度器类型
        
        pattern = r"""
            ^pgd_attack_records_
            norm(?P<norm>\d+)_                
            alpha(?P<alpha>[\d\.]+)_          
            epsilon(?P<epsilon>[\d\.]+)_      
            steps(?P<steps>\d+)_              
            idx(?P<idx>\d+)_                  
            (?P<attack_type>.+)$              
        """
        
        match = re.match(pattern, base_name, re.VERBOSE)
        if not match:
            print(f"❗ 无法解析 base 名称: {base_name}")
            continue

        params = match.groupdict()
        attack_type = params['attack_type']  # e.g., detached1to5

        # 创建参数键（不包含attack_type）
        param_key = (
            f"alpha={params['alpha']}",
            f"epsilon={params['epsilon']}",
            f"steps={params['steps']}",
            f"idx={params['idx']}"
        )

        grouped[attack_type][param_key][lr_method] = str(path.resolve())

    return grouped

def visualize_all_lr_methods_for_attack(grouped_dict, fps=5, dpi=100, approximated_prefix="surrogate_"):
    """
    为每个唯一的参数组合（包括攻击类型）绘制所有调度方法的 loss 对比图和 alpha 变化图
    新的参数键格式: (alpha=..., epsilon=..., steps=..., idx=..., attack_type=...)
    """
    # 首先收集所有唯一的参数组合（包括攻击类型）
    all_param_keys = set()
    param_key_to_methods = defaultdict(dict)
    
    # 遍历所有攻击类型和参数组合
    for attack_type, attack_group in grouped_dict.items():
        for param_key, file_map in attack_group.items():
            # 创建新的参数键，包含攻击类型作为第5个元素
            extended_param_key = param_key + (f"attack_type={attack_type}",)
            all_param_keys.add(extended_param_key)
            
            # 收集这个参数组合下的所有方法
            for method, path in file_map.items():
                param_key_to_methods[extended_param_key][method] = path

    # 为每个唯一的参数组合绘制图表
    for param_key in sorted(all_param_keys):
        file_map = param_key_to_methods[param_key]
        attack_type = param_key[-1].split("=")[1]  # 从最后一个元素提取攻击类型
        
        print(f"\n📊 绘图中: 参数组合 {' | '.join(param_key)}")

        # 准备数据
        data_per_method = {}
        max_loss = 0.0
        min_loss = float("inf")
        max_alpha = 0.0
        num_steps = None

        for method, path in file_map.items():
            with open(path, "rb") as f:
                records = pickle.load(f)["steps"]
                data_per_method[method] = records
                losses = [
                    r[f'{approximated_prefix}loss'] if attack_type == "approximated" else r['loss']
                    for r in records
                ]
                # 确保alpha值被正确转换为float
                alphas = [float(r['alpha'].cpu().numpy()) if hasattr(r['alpha'], 'cpu') else float(r['alpha']) for r in records]
                max_loss = max(max_loss, max(losses))
                min_loss = min(min_loss, min(losses))
                max_alpha = max(max_alpha, max(alphas))
                if num_steps is None:
                    num_steps = len(records)

        # 创建包含两个子图的图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))
        # 更新颜色映射获取方式以避免弃用警告
        colors = plt.colormaps['tab20']
        # 固定颜色映射：先排序方法名确保顺序一致
        sorted_methods = sorted(data_per_method.keys())
        color_map = {name: colors(i % 20) for i, name in enumerate(sorted_methods)}

        # 第一个子图：绘制loss曲线
        for method in sorted_methods:
            records = data_per_method[method]
            losses = [
                r[f'{approximated_prefix}loss'] if attack_type == "approximated" else r['loss']
                for r in records
            ]
            last_loss = losses[-1]  # 获取最后一步的loss值
            label = f"{method} (final loss={last_loss:.2f})"  # 在label中添加loss值
            ax1.plot(range(num_steps), losses, 
                   label=label, 
                   color=color_map[method], 
                   marker='o', 
                   markersize=3, 
                   alpha=0.8, 
                   linewidth=2)

        ax1.set_title(f"Loss Comparison - Parameter combination: {' | '.join(param_key)}")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_ylim(min_loss * 0.9, max_loss * 1.1)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)

        # 第二个子图：绘制alpha曲线
        for method in sorted_methods:
            records = data_per_method[method]
            # 确保alpha值被正确转换为float
            alphas = [float(r['alpha'].cpu().numpy()) if hasattr(r['alpha'], 'cpu') else float(r['alpha']) for r in records]
            ax2.plot(range(num_steps), alphas, 
                   label=method, 
                   color=color_map[method], 
                   marker='o', 
                   markersize=3, 
                   alpha=0.8, 
                   linewidth=2)

        ax2.set_title("Alpha Value Progression")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Alpha Value")
        ax2.set_yscale('log')
        ax2.set_ylim(0, max_alpha * 1.1)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True)

        plt.tight_layout()

        # 保存图片
        output_dir = Path(__file__).parent / "comparison_plots"
        output_dir.mkdir(exist_ok=True)
        
        # 创建安全的文件名
        filename_safe_params = "_".join([
            p.replace("=", "").replace(".", "pt") 
            for p in param_key
        ])
        output_path = output_dir / f"lineplot_{filename_safe_params}.png"
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存图像到: {output_path}")

if __name__ == "__main__":
    folder_path = Path(__file__).parent / "pickle_files"
    grouped = group_pkl_files(folder_path)
    visualize_all_lr_methods_for_attack(grouped)