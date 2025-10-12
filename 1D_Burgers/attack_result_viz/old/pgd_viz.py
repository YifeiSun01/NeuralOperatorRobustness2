#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def _sorted_steps(step_update: dict):
    """
    Return a sorted list of step keys like ['step_0','step_1',...]
    """
    def _key(s):
        try:
            return int(s.split('_')[-1])
        except Exception:
            return 10**9
    return sorted([k for k in step_update.keys() if k.startswith('step_')], key=_key)

def _gather_ylim(step_update: dict):
    """
    Scan all steps to compute global y-lims for:
      - 'a' variants
      - outputs (G and g) variants
      - loss metrics
    Returns: (ylim_a, ylim_out, ylim_loss, step_indices, losses_dict)
    where losses_dict = {'solver': list, 'detached': list, 'approximated': list}
    and step_indices = [0,1,...]
    """
    y_min_a, y_max_a = np.inf, -np.inf
    y_min_out, y_max_out = np.inf, -np.inf
    y_min_loss, y_max_loss = np.inf, -np.inf

    step_keys = _sorted_steps(step_update)
    step_indices = []
    losses = {'solver': [], 'detached': [], 'approximated': []}

    for sk in step_keys:
        sdict = step_update.get(sk, {})
        idx = int(sk.split('_')[-1])
        step_indices.append(idx)

        # losses
        lm = sdict.get('loss_metrics', {})
        if isinstance(lm, dict):
            with_solver = lm.get('with_solver', None)
            without_solver = lm.get('without_solver', None)
            approximate = lm.get('approximate', None)

            if with_solver is not None:
                losses['solver'].append(float(with_solver))
                y_min_loss = min(y_min_loss, float(with_solver))
                y_max_loss = max(y_max_loss, float(with_solver))
            else:
                losses['solver'].append(np.nan)

            if without_solver is not None:
                losses['detached'].append(float(without_solver))
                y_min_loss = min(y_min_loss, float(without_solver))
                y_max_loss = max(y_max_loss, float(without_solver))
            else:
                losses['detached'].append(np.nan)

            if approximate is not None:
                losses['approximated'].append(float(approximate))
                y_min_loss = min(y_min_loss, float(approximate))
                y_max_loss = max(y_max_loss, float(approximate))
            else:
                losses['approximated'].append(np.nan)

        # values to set y-lims
        vals = sdict.get('values', {})
        a_vals = vals.get('a_values', {})
        G_vals = vals.get('G_values', {})
        g_vals = vals.get('g_values', {})

        # a variants
        for k in ['a', 'a_with', 'a_without', 'a_approximate']:
            v = a_vals.get(k, None)
            if v is not None:
                v = np.asarray(v).astype(float).ravel()
                if v.size > 0:
                    y_min_a = min(y_min_a, float(np.nanmin(v)))
                    y_max_a = max(y_max_a, float(np.nanmax(v)))

        # outputs
        for k in ['G(a)', 'G(a_with)', 'G(a_without)', 'G(a_approximate)']:
            v = G_vals.get(k, None)
            if v is not None:
                v = np.asarray(v).astype(float).ravel()
                if v.size > 0:
                    y_min_out = min(y_min_out, float(np.nanmin(v)))
                    y_max_out = max(y_max_out, float(np.nanmax(v)))
        for k in ['g(a)', 'g(a_with)', 'g(a_without)', 'g(a_approximate)']:
            v = g_vals.get(k, None)
            if v is not None:
                v = np.asarray(v).astype(float).ravel()
                if v.size > 0:
                    y_min_out = min(y_min_out, float(np.nanmin(v)))
                    y_max_out = max(y_max_out, float(np.nanmax(v)))

    # Handle empty / degenerate cases
    if not np.isfinite(y_min_a) or not np.isfinite(y_max_a):
        y_min_a, y_max_a = -1.0, 1.0
    if not np.isfinite(y_min_out) or not np.isfinite(y_max_out):
        y_min_out, y_max_out = -1.0, 1.0
    if not np.isfinite(y_min_loss) or not np.isfinite(y_max_loss) or y_min_loss == y_max_loss:
        y_min_loss, y_max_loss = 0.0, max(1.0, y_max_loss if np.isfinite(y_max_loss) else 1.0)

    # Add small margins
    def pad(lo, hi, pct=0.05):
        span = hi - lo
        if span <= 0:
            span = max(1.0, abs(hi) + 1.0)
        return lo - pct*span, hi + pct*span

    ylim_a = pad(y_min_a, y_max_a)
    ylim_out = pad(y_min_out, y_max_out)
    ylim_loss = pad(y_min_loss, y_max_loss)

    return ylim_a, ylim_out, ylim_loss, step_indices, losses

# ===== 在文件顶部导入之后，添加一份统一调色板 =====
PALETTE = {
    "black_deep":  "#000000",
    "black_light": "#666666",

    # solver grad backprop（with）
    "orange_deep": "#d35400",
    "orange_light":"#f5b041",

    # detached（without）
    "blue_deep":   "#1f77b4",
    "blue_light":  "#85c1e9",

    # approximated（approximate）
    "red_deep":    "#c0392b",
    "red_light":   "#f1948a",
}

def _plot_frame(axs, idx, sdict, ylim_a, ylim_out, step_indices, losses, ylim_loss):
    """
    Draw one frame (5 panels):
      axs is a list of 5 axes in order:
        0: a-variants
        1: solver grad backprop (with)
        2: detached (without)
        3: approximated (approximate)
        4: loss-progress
    """
    vals = sdict.get('values', {})
    a_vals = vals.get('a_values', {})
    G_vals = vals.get('G_values', {})
    g_vals = vals.get('g_values', {})

    # --- 计算 x 轴长度 ---
    L = None
    for k in ['a', 'a_with', 'a_without', 'a_approximate']:
        if a_vals.get(k, None) is not None:
            L = int(np.asarray(a_vals[k]).size); break
    if L is None:
        for k in ['G(a)', 'G(a_with)', 'G(a_without)', 'G(a_approximate)']:
            if G_vals.get(k, None) is not None:
                L = int(np.asarray(G_vals[k]).size); break
    if L is None:
        for k in ['g(a)', 'g(a_with)', 'g(a_without)', 'g(a_approximate)']:
            if g_vals.get(k, None) is not None:
                L = int(np.asarray(g_vals[k]).size); break
    if L is None:
        L = 1024
    x = np.arange(L)

    # --- 面板 0: Inputs (a variants) ---
    ax = axs[0]; ax.clear()
    plot_input_order = [
        ('a',             'a (clean)',                    PALETTE["black_deep"],   1.6),
        ('a_with',        'a + δ (solver grad backprop)', PALETTE["orange_deep"],  1.2),
        ('a_without',     'a + δ (detached)',             PALETTE["blue_deep"],    1.2),
        ('a_approximate', 'a + δ (approximated)',         PALETTE["red_deep"],     1.2),
    ]
    for key, label, color, lw in plot_input_order:
        v = a_vals.get(key, None)
        if v is not None:
            ax.plot(x, np.asarray(v).astype(float).ravel(), label=label, color=color, linewidth=lw)
    ax.set_title('Inputs (a variants)')
    ax.set_ylim(ylim_a)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 面板 1-3: FNO vs PDE，按三种方法配色（深=FNO，浅=PDE） ---
    def draw_method(ax, method_name, delta_suffix):
        """
        method_name: 'solver grad backprop' | 'detached' | 'approximated'
        delta_suffix: 'with' | 'without' | 'approximate'
        """
        ax.clear()

        # 颜色选择：不同方法对应不同色系
        if delta_suffix == 'with':
            deep_c, light_c = PALETTE["orange_deep"], PALETTE["orange_light"]
            Gk_raw, gk_raw = 'G(a_with)', 'g(a_with)'
        elif delta_suffix == 'without':
            deep_c, light_c = PALETTE["blue_deep"], PALETTE["blue_light"]
            Gk_raw, gk_raw = 'G(a_without)', 'g(a_without)'
        else:
            deep_c, light_c = PALETTE["red_deep"], PALETTE["red_light"]
            Gk_raw, gk_raw = 'G(a_approximate)', 'g(a_approximate)'

        # 基线：FNO(a) & PDE(a) —— 深/浅黑
        base_pairs = [
            ('G(a)', 'FNO(a)', PALETTE["black_deep"], 1.6),
            ('g(a)', 'PDE(a)', PALETTE["black_light"], 1.4),
        ]
        for raw_key, label, color, lw in base_pairs:
            arr = G_vals.get(raw_key) if raw_key.startswith('G') else g_vals.get(raw_key)
            if arr is not None:
                ax.plot(x, np.asarray(arr).astype(float).ravel(), label=label, color=color, linewidth=lw)

        # 扰动后：FNO(a+δ) & PDE(a+δ) —— 同色系深/浅
        vG = G_vals.get(Gk_raw, None)
        vg = g_vals.get(gk_raw, None)
        if vG is not None:
            ax.plot(x, np.asarray(vG).astype(float).ravel(),
                    label='FNO(a + δ)', color=deep_c, linewidth=1.6)
        if vg is not None:
            ax.plot(x, np.asarray(vg).astype(float).ravel(),
                    label='PDE(a + δ)', color=light_c, linewidth=1.4)

        ax.set_title(method_name + ' : FNO vs PDE')
        ax.set_ylim(ylim_out)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    draw_method(axs[1], 'solver grad backprop', 'with')
    draw_method(axs[2], 'detached', 'without')
    draw_method(axs[3], 'approximated', 'approximate')

    # --- 面板 4: Loss progress ---
    ax = axs[4]; ax.clear()
    x_steps = np.array(step_indices, dtype=int)
    l1 = np.array(losses['solver'], dtype=float)
    l2 = np.array(losses['detached'], dtype=float)
    l3 = np.array(losses['approximated'], dtype=float)
    pos = np.where(x_steps == idx)[0]
    end = pos[0] + 1 if pos.size > 0 else len(x_steps)

    ax.plot(x_steps[:end], l1[:end], label='solver grad backprop', color=PALETTE["orange_deep"], linewidth=1.6)
    ax.plot(x_steps[:end], l2[:end], label='detached',            color=PALETTE["blue_deep"],   linewidth=1.6)
    ax.plot(x_steps[:end], l3[:end], label='approximated',        color=PALETTE["red_deep"],    linewidth=1.6)
    ax.set_title(f'Loss progress (up to step {idx})')
    ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.set_ylim(ylim_loss)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

def make_gifs(pickle_path: str, outdir: str, fps: int = 8):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Iterate keys (one attack per key)
    for attack_key, attack_dict in data.items():
        # attack_key can be a tuple of strings; build a safe filename
        key_str = "__".join(attack_key) if isinstance(attack_key, (tuple, list)) else str(attack_key)
        safe_key = "".join(c if c.isalnum() or c in "._-+=" else "_" for c in key_str)
        gif_path = os.path.join(outdir, f"{safe_key}.gif")

        step_update = attack_dict.get('step_update', {})
        step_keys = _sorted_steps(step_update)
        if not step_keys:
            print(f"[WARN] No steps for key={attack_key}, skipping.")
            continue

        # Compute global y-lims and losses arrays
        ylim_a, ylim_out, ylim_loss, step_indices, losses = _gather_ylim(step_update)

        frames = []
        # Prepare a consistent 2x3 layout but only use 5 axes (turn off the last one)
        for sk in tqdm(step_keys, desc=f"Rendering {key_str}"):
            sdict = step_update.get(sk, {})
            idx = int(sk.split('_')[-1])

            fig = plt.figure(figsize=(14, 8))
            # Create 2x3 grid
            gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
            ax0 = fig.add_subplot(gs[0,0])
            ax1 = fig.add_subplot(gs[0,1])
            ax2 = fig.add_subplot(gs[0,2])
            ax3 = fig.add_subplot(gs[1,0])
            ax4 = fig.add_subplot(gs[1,1])
            ax_unused = fig.add_subplot(gs[1,2])
            ax_unused.axis('off')

            axs = [ax0, ax1, ax2, ax3, ax4]
            _plot_frame(axs, idx, sdict, ylim_a, ylim_out, step_indices, losses, ylim_loss)

            # Super title with attack key
            fig.suptitle(f"Attack: {key_str} | Step: {idx}", fontsize=14)
            fig.canvas.draw()

            # Grab buffer as RGB array for GIF
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(frame)
            plt.close(fig)

        # Save GIF
        iio.imwrite(gif_path, frames, fps=fps)
        print(f"[OK] Saved {gif_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', required=True, help='Path to gradient_test_*.pkl')
    parser.add_argument('--outdir', required=True, help='Output directory for GIFs')
    parser.add_argument('--fps', type=int, default=8, help='Frames per second (default: 8)')
    args = parser.parse_args()

    make_gifs(args.pickle, args.outdir, args.fps)

if __name__ == "__main__":
    main()

