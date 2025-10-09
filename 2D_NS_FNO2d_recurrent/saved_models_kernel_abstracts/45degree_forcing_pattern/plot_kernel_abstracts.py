#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量读取 *.analysis.pt，针对每个模型的 layer=0,1,2,3：
1) 绘制 spatial（含原 spatial + spatial_extra；含 TopSingVals 切片与 σ1/σ2 - 1，跳过 U_top2 / V_top2 / S_top2）
2) freq_neg vs freq_pos 并排在同一张图（切片 + ratio）
3) 输出结构化目录；每个 pt 对应模型子文件夹；每个 layer 一个子文件夹；下有 spatial 和 freq_pair 文件夹

增强：
- 打印当前处理的 pt 文件 / layer / 图名
- 如 LogNorm 且数据含负数/非正/NaN，则跳过该图并打印原因
- 复数数据取模长
"""

import argparse
import json
import os
from pathlib import Path
import glob
from typing import Optional, Tuple, List, Union

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ----------------------- 打印工具 -----------------------

def _log(msg: str):
    print(msg, flush=True)

def _log_plot(model_name: str, layer: str, category: str, keyname: str):
    _log(f"[plot] model={model_name} layer={layer} category={category} key={keyname}")

def _log_skip(model_name: str, layer: str, category: str, keyname: str, reason: str):
    _log(f"[skip] model={model_name} layer={layer} category={category} key={keyname} :: {reason}")


# ----------------------- 基础工具 -----------------------

def _tonp(x: torch.Tensor) -> np.ndarray:
    """Torch -> numpy（保留 complex），上层决定是否取模长。"""
    return x.detach().cpu().numpy()

def _real_if_complex(arr: np.ndarray) -> np.ndarray:
    """复数取模长，实数原样返回。"""
    if np.iscomplexobj(arr):
        return np.abs(arr)
    else:
        return arr

def _finite_min_max(arr_list: List[np.ndarray]) -> Optional[Tuple[float, float]]:
    """对一组数组计算（忽略 NaN/Inf 的）全局 vmin/vmax；若均为无效，返回 None。"""
    vals = []
    for a in arr_list:
        if a is None:
            continue
        finite = a[np.isfinite(a)]
        if finite.size:
            vals.append((finite.min(), finite.max()))
    if not vals:
        return None
    vmin = min(v[0] for v in vals)
    vmax = max(v[1] for v in vals)
    return vmin, vmax

def _data_log_transform(arr: np.ndarray, mode: str) -> np.ndarray:
    if arr is None:
        return None
    if mode == "none":
        return arr
    if mode == "log":
        return np.log(arr)
    if mode == "log1p":
        return np.log1p(arr)
    if mode == "signed_log1p":
        return np.sign(arr) * np.log1p(np.abs(arr))
    raise ValueError(f"Unknown data_log_mode: {mode}")

def _imshow(ax, arr: np.ndarray, *, color_log_scale: bool, color_log_eps: Optional[float], cmap):
    """
    统一的 imshow：若 color_log_scale=True，用 LogNorm；否则线性。
    这里不做有效性检查，调用方先判断，避免抛错。
    """
    if color_log_scale:
        plot_vals = arr if color_log_eps is None else (arr + color_log_eps)
        norm = mcolors.LogNorm(vmin=np.nanmin(plot_vals), vmax=np.nanmax(plot_vals))
        return ax.imshow(plot_vals, origin="lower", norm=norm, cmap=cmap)
    else:
        return ax.imshow(arr, origin="lower", cmap=cmap)


# ----------------------- 绘图：Spatial -----------------------

def plot_spatial(
    data: dict,
    layer: str,
    top_singval_k: int,
    save_dir: Path,
    *,
    data_log_mode: str,
    color_log_scale: bool,
    color_log_eps: Optional[float],
    cmap,
    s2_eps: float,
    model_name: str
):
    spatial = data["layers"][layer]["spatial"]
    spatial_extra = data["layers"][layer].get("spatial_extra", {})

    skip_keys = {"U_top2", "V_top2", "S_top2"}

    # 单一 spatial 文件夹
    save_dir.mkdir(parents=True, exist_ok=True)

    for group_name, group in [("spatial", spatial), ("spatial", spatial_extra)]:  # 两者都放入 "spatial"
        for key, tensor in group.items():
            if key in skip_keys:
                continue

            if isinstance(tensor, dict):
                # 例如 Entries / Angles
                for subk, subt in tensor.items():
                    keyname = f"{key}.{subk}"
                    _log_plot(model_name, layer, "spatial", keyname)

                    arr = _tonp(subt)
                    arr = _real_if_complex(arr)
                    if arr.ndim == 2:
                        arr_t = _data_log_transform(arr, data_log_mode)

                        # 检查 LogNorm 条件
                        if color_log_scale:
                            plot_vals = arr_t if color_log_eps is None else (arr_t + color_log_eps)
                            vmm = _finite_min_max([plot_vals])
                            if (vmm is None) or (not np.isfinite(vmm[0])) or (vmm[0] <= 0):
                                _log_skip(model_name, layer, "spatial", keyname, "LogNorm 需要正值：检测到非正/NaN，跳过该图")
                                continue

                        fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
                        try:
                            im = _imshow(ax, arr_t, color_log_scale=color_log_scale,
                                         color_log_eps=color_log_eps, cmap=cmap)
                        except Exception as e:
                            _log_skip(model_name, layer, "spatial", keyname, f"imshow 异常：{e}，跳过")
                            plt.close(fig)
                            continue
                        plt.colorbar(im, ax=ax, shrink=0.9)
                        # 两行标题
                        ax.set_title(f"{model_name}  layer {layer}\nspatial.{keyname}", fontsize=10)
                        fname = f"{model_name}_layer{layer}_spatial_{key}_{subk}.png"
                        fig.savefig(save_dir / fname, dpi=150)
                        plt.close(fig)

                    else:
                        # 高维切片
                        K = arr.shape[-1]
                        for i in range(min(K, top_singval_k)):
                            subkey = f"{key}.{subk}[...,{i}]"
                            _log_plot(model_name, layer, "spatial", subkey)

                            slice_i = arr[..., i]
                            slice_i_t = _data_log_transform(slice_i, data_log_mode)

                            if color_log_scale:
                                plot_vals = slice_i_t if color_log_eps is None else (slice_i_t + color_log_eps)
                                vmm = _finite_min_max([plot_vals])
                                if (vmm is None) or (not np.isfinite(vmm[0])) or (vmm[0] <= 0):
                                    _log_skip(model_name, layer, "spatial", subkey, "LogNorm 需要正值：检测到非正/NaN，跳过该图")
                                    continue

                            fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
                            try:
                                im = _imshow(ax, slice_i_t, color_log_scale=color_log_scale,
                                             color_log_eps=color_log_eps, cmap=cmap)
                            except Exception as e:
                                _log_skip(model_name, layer, "spatial", subkey, f"imshow 异常：{e}，跳过")
                                plt.close(fig)
                                continue
                            plt.colorbar(im, ax=ax, shrink=0.9)
                            ax.set_title(f"{model_name}  layer {layer}\nspatial.{key}.{subk}[...,{i}]", fontsize=10)
                            fname = f"{model_name}_layer{layer}_spatial_{key}_{subk}_slice{i}.png"
                            fig.savefig(save_dir / fname, dpi=150)
                            plt.close(fig)
                continue  # dict 情况已处理

            # 普通张量
            keyname = key
            _log_plot(model_name, layer, "spatial", keyname)

            arr = _tonp(tensor)
            arr = _real_if_complex(arr)
            if arr.ndim == 2:
                arr_t = _data_log_transform(arr, data_log_mode)

                if color_log_scale:
                    plot_vals = arr_t if color_log_eps is None else (arr_t + color_log_eps)
                    vmm = _finite_min_max([plot_vals])
                    if (vmm is None) or (not np.isfinite(vmm[0])) or (vmm[0] <= 0):
                        _log_skip(model_name, layer, "spatial", keyname, "LogNorm 需要正值：检测到非正/NaN，跳过该图")
                        continue

                fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
                try:
                    im = _imshow(ax, arr_t, color_log_scale=color_log_scale,
                                 color_log_eps=color_log_eps, cmap=cmap)
                except Exception as e:
                    _log_skip(model_name, layer, "spatial", keyname, f"imshow 异常：{e}，跳过")
                    plt.close(fig)
                    continue
                plt.colorbar(im, ax=ax, shrink=0.9)
                ax.set_title(f"{model_name}  layer {layer}\nspatial.{key}", fontsize=10)
                fname = f"{model_name}_layer{layer}_spatial_{key}.png"
                fig.savefig(save_dir / fname, dpi=150)
                plt.close(fig)

            else:
                # 多维切片 + ratio
                K = arr.shape[-1]
                for i in range(min(K, top_singval_k)):
                    subkey = f"{key}[...,{i}]"
                    _log_plot(model_name, layer, "spatial", subkey)

                    slice_i = arr[..., i]
                    slice_i_t = _data_log_transform(slice_i, data_log_mode)

                    if color_log_scale:
                        plot_vals = slice_i_t if color_log_eps is None else (slice_i_t + color_log_eps)
                        vmm = _finite_min_max([plot_vals])
                        if (vmm is None) or (not np.isfinite(vmm[0])) or (vmm[0] <= 0):
                            _log_skip(model_name, layer, "spatial", subkey, "LogNorm 需要正值：检测到非正/NaN，跳过该图")
                            continue

                    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
                    try:
                        im = _imshow(ax, slice_i_t, color_log_scale=color_log_scale,
                                     color_log_eps=color_log_eps, cmap=cmap)
                    except Exception as e:
                        _log_skip(model_name, layer, "spatial", subkey, f"imshow 异常：{e}，跳过")
                        plt.close(fig)
                        continue
                    plt.colorbar(im, ax=ax, shrink=0.9)
                    ax.set_title(f"{model_name}  layer {layer}\nspatial.{key}[...,{i}]", fontsize=10)
                    fname = f"{model_name}_layer{layer}_spatial_{key}_slice{i}.png"
                    fig.savefig(save_dir / fname, dpi=150)
                    plt.close(fig)

                if K >= 2 and key.lower().startswith("top"):
                    subkey = f"{key} ratio(σ1/σ2−1)"
                    _log_plot(model_name, layer, "spatial", subkey)

                    s1 = arr[..., 0]
                    s2 = arr[..., 1]
                    ratio_minus1 = (s1 / np.clip(s2, s2_eps, None)) - 1.0
                    rm_t = _data_log_transform(ratio_minus1, data_log_mode)

                    if color_log_scale:
                        plot_vals = rm_t if color_log_eps is None else (rm_t + color_log_eps)
                        vmm = _finite_min_max([plot_vals])
                        if (vmm is None) or (not np.isfinite(vmm[0])) or (vmm[0] <= 0):
                            _log_skip(model_name, layer, "spatial", subkey, "LogNorm 需要正值：检测到非正/NaN，跳过该图")
                            continue

                    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
                    try:
                        im = _imshow(ax, rm_t, color_log_scale=color_log_scale,
                                     color_log_eps=color_log_eps, cmap=cmap)
                    except Exception as e:
                        _log_skip(model_name, layer, "spatial", subkey, f"imshow 异常：{e}，跳过")
                        plt.close(fig)
                        continue
                    plt.colorbar(im, ax=ax, shrink=0.9)
                    ax.set_title(f"{model_name}  layer {layer}\nspatial.{key} ratio(σ1/σ2−1)", fontsize=10)
                    fname = f"{model_name}_layer{layer}_spatial_{key}_ratio.png"
                    fig.savefig(save_dir / fname, dpi=150)
                    plt.close(fig)


# ----------------------- 绘图：频域 / 并排对比 -----------------------

def plot_freq_pair(
    data: dict,
    layer: str,
    tsv_indices: Tuple[int, ...],
    symmetric: bool,
    save_dir: Path,
    *,
    data_log_mode: str,
    color_log_scale: bool,
    color_log_eps: Optional[float],
    cmap,
    s2_eps: float,
    model_name: str
):
    fneg = data["layers"][layer]["freq_neg"]
    fpos = data["layers"][layer]["freq_pos"]

    skip_keys = {"U_top2", "V_top2", "S_top2"}

    save_dir.mkdir(parents=True, exist_ok=True)

    all_keys = set(fneg.keys()) | set(fpos.keys())
    for k in all_keys:
        if k in skip_keys:
            continue
        neg_tensor = fneg.get(k, None)
        pos_tensor = fpos.get(k, None)

        # dict（Entries / Angles）拆子键并排
        if isinstance(neg_tensor, dict) or isinstance(pos_tensor, dict):
            subkeys = set()
            if isinstance(neg_tensor, dict):
                subkeys |= set(neg_tensor.keys())
            if isinstance(pos_tensor, dict):
                subkeys |= set(pos_tensor.keys())
            for subk in subkeys:
                neg_sub = neg_tensor.get(subk, None) if isinstance(neg_tensor, dict) else None
                pos_sub = pos_tensor.get(subk, None) if isinstance(pos_tensor, dict) else None
                _plot_freq_pair_single(
                    neg_sub, pos_sub,
                    model_name, layer, f"{k}.{subk}",
                    save_dir,
                    data_log_mode, color_log_scale, color_log_eps, cmap,
                    tsv_indices, symmetric, s2_eps
                )
        else:
            _plot_freq_pair_single(
                neg_tensor, pos_tensor,
                model_name, layer, k,
                save_dir,
                data_log_mode, color_log_scale, color_log_eps, cmap,
                tsv_indices, symmetric, s2_eps
            )

def _plot_freq_pair_single(
    neg_tensor: Union[torch.Tensor, None],
    pos_tensor: Union[torch.Tensor, None],
    model_name: str,
    layer: str,
    keyname: str,
    save_dir: Path,
    data_log_mode: str,
    color_log_scale: bool,
    color_log_eps: Optional[float],
    cmap,
    tsv_indices: Tuple[int, ...],
    symmetric: bool,
    s2_eps: float
):
    # 转 numpy + 取模长（可为 None）
    A = _real_if_complex(_tonp(neg_tensor)) if neg_tensor is not None else None
    B = _real_if_complex(_tonp(pos_tensor)) if pos_tensor is not None else None

    if (A is None) and (B is None):
        return

    # 3D（TopSingVals 等）
    if (A is not None and A.ndim == 3) or (B is not None and B.ndim == 3):
        KA = A.shape[-1] if (A is not None and A.ndim == 3) else 0
        KB = B.shape[-1] if (B is not None and B.ndim == 3) else 0
        Kmax = max(KA, KB)

        # 切片对比
        for idx in tsv_indices:
            if idx >= Kmax:
                continue
            subkey = f"{keyname}[...,{idx}]"
            _log_plot(model_name, layer, "freq_pair", subkey)

            Ai = A[..., idx] if (A is not None and Ai_valid_dim(A, idx)) else None
            Bi = B[..., idx] if (B is not None and Ai_valid_dim(B, idx)) else None

            Ai_t = _data_log_transform(Ai, data_log_mode) if Ai is not None else None
            Bi_t = _data_log_transform(Bi, data_log_mode) if Bi is not None else None

            # 颜色轴（线性/对数）范围检查
            images = []
            fig, axs = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")

            if color_log_scale:
                # 对数：所有 present 的数组都必须 >0
                P = []
                if Ai_t is not None:
                    P.append(Ai_t if color_log_eps is None else Ai_t + color_log_eps)
                if Bi_t is not None:
                    P.append(Bi_t if color_log_eps is None else Bi_t + color_log_eps)
                vmm = _finite_min_max(P)
                if (vmm is None) or (not np.isfinite(vmm[0])) or (vmm[0] <= 0):
                    _log_skip(model_name, layer, "freq_pair", subkey, "LogNorm 需要正值：检测到非正/NaN，跳过该图")
                    plt.close(fig)
                    continue
                norm = mcolors.LogNorm(vmin=vmm[0], vmax=vmm[1])

                if Ai_t is not None:
                    images.append(axs[0].imshow(Ai_t if color_log_eps is None else (Ai_t + color_log_eps),
                                                origin="lower", norm=norm, cmap=cmap))
                axs[0].set_title("freq_neg\n" + subkey, fontsize=9)
                if Bi_t is not None:
                    images.append(axs[1].imshow(Bi_t if color_log_eps is None else (Bi_t + color_log_eps),
                                                origin="lower", norm=norm, cmap=cmap))
                axs[1].set_title("freq_pos\n" + subkey, fontsize=9)
            else:
                # 线性：合并 present 的 min/max；可选对称
                P = [Ai_t, Bi_t]
                vmm = _finite_min_max([p for p in P if p is not None])
                if vmm is None:
                    _log_skip(model_name, layer, "freq_pair", subkey, "有效数据全为 NaN/Inf，跳过该图")
                    plt.close(fig)
                    continue
                if symmetric:
                    m = max(abs(vmm[0]), abs(vmm[1]))
                    norm = mcolors.Normalize(vmin=-m, vmax=+m)
                else:
                    norm = mcolors.Normalize(vmin=vmm[0], vmax=vmm[1])

                if Ai_t is not None:
                    images.append(axs[0].imshow(Ai_t, origin="lower", norm=norm, cmap=cmap))
                axs[0].set_title("freq_neg\n" + subkey, fontsize=9)
                if Bi_t is not None:
                    images.append(axs[1].imshow(Bi_t, origin="lower", norm=norm, cmap=cmap))
                axs[1].set_title("freq_pos\n" + subkey, fontsize=9)

            if images:
                fig.colorbar(images[-1], ax=axs, shrink=0.9)
            fname = f"{model_name}_layer{layer}_freq_pair_{keyname}_slice{idx}.png"
            fig.savefig(save_dir / fname, dpi=150)
            plt.close(fig)

        # ratio：σ1/σ2-1
        if (KA >= 2) or (KB >= 2):
            subkey = f"{keyname} ratio(σ1/σ2−1)"
            _log_plot(model_name, layer, "freq_pair", subkey)

            Rn = None
            Rp = None
            if KA >= 2:
                Rn = (A[..., 0] / np.clip(A[..., 1], s2_eps, None)) - 1.0
            if KB >= 2:
                Rp = (B[..., 0] / np.clip(B[..., 1], s2_eps, None)) - 1.0

            Rn_t = _data_log_transform(Rn, data_log_mode) if Rn is not None else None
            Rp_t = _data_log_transform(Rp, data_log_mode) if Rp is not None else None

            images = []
            fig, axs = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")

            if color_log_scale:
                P = []
                if Rn_t is not None:
                    P.append(Rn_t if color_log_eps is None else Rn_t + color_log_eps)
                if Rp_t is not None:
                    P.append(Rp_t if color_log_eps is None else Rp_t + color_log_eps)
                vmm = _finite_min_max(P)
                if (vmm is None) or (not np.isfinite(vmm[0])) or (vmm[0] <= 0):
                    _log_skip(model_name, layer, "freq_pair", subkey, "LogNorm 需要正值：检测到非正/NaN，跳过该图")
                    plt.close(fig)
                    return
                norm = mcolors.LogNorm(vmin=vmm[0], vmax=vmm[1])

                if Rn_t is not None:
                    images.append(axs[0].imshow(Rn_t if color_log_eps is None else (Rn_t + color_log_eps),
                                                origin="lower", norm=norm, cmap=cmap))
                axs[0].set_title("freq_neg\n" + subkey, fontsize=9)
                if Rp_t is not None:
                    images.append(axs[1].imshow(Rp_t if color_log_eps is None else (Rp_t + color_log_eps),
                                                origin="lower", norm=norm, cmap=cmap))
                axs[1].set_title("freq_pos\n" + subkey, fontsize=9)
            else:
                P = [Rn_t, Rp_t]
                vmm = _finite_min_max([p for p in P if p is not None])
                if vmm is None:
                    _log_skip(model_name, layer, "freq_pair", subkey, "有效数据全为 NaN/Inf，跳过该图")
                    plt.close(fig)
                    return
                if symmetric:
                    m = max(abs(vmm[0]), abs(vmm[1]))
                    norm = mcolors.Normalize(vmin=-m, vmax=+m)
                else:
                    norm = mcolors.Normalize(vmin=vmm[0], vmax=vmm[1])

                if Rn_t is not None:
                    images.append(axs[0].imshow(Rn_t, origin="lower", norm=norm, cmap=cmap))
                axs[0].set_title("freq_neg\n" + subkey, fontsize=9)
                if Rp_t is not None:
                    images.append(axs[1].imshow(Rp_t, origin="lower", norm=norm, cmap=cmap))
                axs[1].set_title("freq_pos\n" + subkey, fontsize=9)

            if images:
                fig.colorbar(images[-1], ax=axs, shrink=0.9)
            fname = f"{model_name}_layer{layer}_freq_pair_{keyname}_ratio.png"
            fig.savefig(save_dir / fname, dpi=150)
            plt.close(fig)

    else:
        # 2D 直接并排
        _log_plot(model_name, layer, "freq_pair", keyname)

        A_t = _data_log_transform(A, data_log_mode) if A is not None else None
        B_t = _data_log_transform(B, data_log_mode) if B is not None else None

        images = []
        fig, axs = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")

        if color_log_scale:
            P = []
            if A_t is not None:
                P.append(A_t if color_log_eps is None else A_t + color_log_eps)
            if B_t is not None:
                P.append(B_t if color_log_eps is None else B_t + color_log_eps)
            vmm = _finite_min_max(P)
            if (vmm is None) or (not np.isfinite(vmm[0])) or (vmm[0] <= 0):
                _log_skip(model_name, layer, "freq_pair", keyname, "LogNorm 需要正值：检测到非正/NaN，跳过该图")
                plt.close(fig)
                return
            norm = mcolors.LogNorm(vmin=vmm[0], vmax=vmm[1])

            if A_t is not None:
                images.append(axs[0].imshow(A_t if color_log_eps is None else (A_t + color_log_eps),
                                            origin="lower", norm=norm, cmap=cmap))
            axs[0].set_title("freq_neg\n" + keyname, fontsize=9)
            if B_t is not None:
                images.append(axs[1].imshow(B_t if color_log_eps is None else (B_t + color_log_eps),
                                            origin="lower", norm=norm, cmap=cmap))
            axs[1].set_title("freq_pos\n" + keyname, fontsize=9)
        else:
            P = [A_t, B_t]
            vmm = _finite_min_max([p for p in P if p is not None])
            if vmm is None:
                _log_skip(model_name, layer, "freq_pair", keyname, "有效数据全为 NaN/Inf，跳过该图")
                plt.close(fig)
                return
            if symmetric:
                m = max(abs(vmm[0]), abs(vmm[1]))
                norm = mcolors.Normalize(vmin=-m, vmax=+m)
            else:
                norm = mcolors.Normalize(vmin=vmm[0], vmax=vmm[1])

            if A_t is not None:
                images.append(axs[0].imshow(A_t, origin="lower", norm=norm, cmap=cmap))
            axs[0].set_title("freq_neg\n" + keyname, fontsize=9)
            if B_t is not None:
                images.append(axs[1].imshow(B_t, origin="lower", norm=norm, cmap=cmap))
            axs[1].set_title("freq_pos\n" + keyname, fontsize=9)

        if images:
            fig.colorbar(images[-1], ax=axs, shrink=0.9)
        fname = f"{model_name}_layer{layer}_freq_pair_{keyname}.png"
        fig.savefig(save_dir / fname, dpi=150)
        plt.close(fig)

def Ai_valid_dim(A: np.ndarray, idx: int) -> bool:
    return (A.ndim == 3) and (0 <= idx < A.shape[-1])


# ----------------------- 批处理主流程 -----------------------

def process_one_model(
    pt_path: Path,
    out_root: Path,
    layers: List[str],
    *,
    top_singval_k: int,
    data_log_mode: str,
    color_log_scale: bool,
    color_log_eps: Optional[float],
    tsv_indices: Tuple[int, ...],
    symmetric: bool,
    cmap_name: Optional[str],
    s2_eps: float
):
    model_name = pt_path.stem
    model_dir = out_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "pt_file": str(pt_path),
        "layers": layers,
        "top_singval_k": top_singval_k,
        "data_log_mode": data_log_mode,
        "color_log_scale": color_log_scale,
        "color_log_eps": color_log_eps,
        "tsv_indices": list(tsv_indices),
        "symmetric": symmetric,
        "cmap": cmap_name,
        "s2_eps": s2_eps,
    }
    with open(model_dir / "plot_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    _log(f"[model] start {model_name}")
    data = torch.load(pt_path, map_location="cpu")
    cmap = plt.get_cmap(cmap_name) if cmap_name else None

    for L in layers:
        Ls = str(L)
        _log(f"[layer] model={model_name} layer={Ls} ...")
        layer_dir = model_dir / f"layer{Ls}"
        spatial_dir = layer_dir / "spatial"
        freq_dir = layer_dir / "freq_pair"

        plot_spatial(
            data, layer=Ls, top_singval_k=top_singval_k,
            save_dir=spatial_dir,
            data_log_mode=data_log_mode, color_log_scale=color_log_scale,
            color_log_eps=color_log_eps, cmap=cmap, s2_eps=s2_eps,
            model_name=model_name
        )
        plot_freq_pair(
            data, layer=Ls, tsv_indices=tsv_indices, symmetric=symmetric,
            save_dir=freq_dir,
            data_log_mode=data_log_mode, color_log_scale=color_log_scale,
            color_log_eps=color_log_eps, cmap=cmap, s2_eps=s2_eps,
            model_name=model_name
        )
    _log(f"[model] done {model_name}")


def parse_args():
    p = argparse.ArgumentParser(
        description="批量绘制 *.analysis.pt 的 spatial / freq 对比图，结构化输出。"
    )
    p.add_argument("--input-dir", type=str, required=True,
                   help="输入含 *.analysis.pt 的目录（非递归）。")
    p.add_argument("--output-dir", type=str, required=True,
                   help="图片输出根目录。")
    p.add_argument("--run-name", type=str, default=None,
                   help="指定本次运行标签，防止不同设置冲突。")
    p.add_argument("--layers", type=str, default="0,1,2,3",
                   help="要处理的 layers，例如 “0,1,2,3”。")
    p.add_argument("--top-singval-k", type=int, default=4,
                   help="TopSingVals 的前 k 切片画图。")
    p.add_argument("--data-log-mode", type=str, default="none",
                   choices=["none", "log", "log1p", "signed_log1p"],
                   help="对 cell 级别进行对数变换。")
    p.add_argument("--color-log-scale", action="store_true",
                   help="对颜色轴用对数刻度（LogNorm）。")
    p.add_argument("--color-log-eps", type=float, default=None,
                   help="若 color_log_scale=True，用于整体偏移保证正值。")
    p.add_argument("--tsv-indices", type=str, default="0,1,2,3",
                   help="并排对比 TopSingVals 切片索引，逗号分隔。")
    p.add_argument("--symmetric", action="store_true",
                   help="线性色标时用对称区间（vmin=-m, vmax=+m）。")
    p.add_argument("--cmap", type=str, default=None,
                   help="matplotlib colormap 名称，如 'viridis'。")
    p.add_argument("--s2-eps", type=float, default=1e-12,
                   help="计算 σ1/σ2-1 时分母最小值防止除零。")
    return p.parse_args()

def main():
    args = parse_args()
    in_dir = Path(args.input_dir).expanduser().resolve()

    run_tag = args.run_name
    if not run_tag:
        clog = "on" if args.color_log_scale else "off"
        eps_tag = f"__eps={args.color_log_eps}" if (args.color_log_scale and args.color_log_eps is not None) else ""
        run_tag = f"dlog={args.data_log_mode}__clog={clog}{eps_tag}"

    out_dir_root = Path(args.output_dir).expanduser().resolve()
    out_dir = out_dir_root / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = [s.strip() for s in args.layers.split(",") if s.strip() != ""]
    tsv_indices = tuple(int(s.strip()) for s in args.tsv_indices.split(",") if s.strip() != "")

    pt_files = sorted(glob.glob(str(in_dir / "*.analysis.pt")))
    if not pt_files:
        raise FileNotFoundError(f"在 {in_dir} 下找不到 *.analysis.pt 文件。")

    print(f"[info] RunTag = {run_tag}")
    print(f"[info] 发现 {len(pt_files)} 个模型：")
    for p in pt_files:
        print(" -", os.path.basename(p))

    for p in pt_files:
        print(f"[run] 处理 {os.path.basename(p)} ...")
        process_one_model(
            Path(p), out_dir, layers,
            top_singval_k=args.top_singval_k,
            data_log_mode=args.data_log_mode,
            color_log_scale=args.color_log_scale,
            color_log_eps=args.color_log_eps,
            tsv_indices=tsv_indices,
            symmetric=args.symmetric,
            cmap_name=args.cmap,
            s2_eps=args.s2_eps
        )
    print("[done] 全部完成。输出位于：", out_dir)

if __name__ == "__main__":
    main()





