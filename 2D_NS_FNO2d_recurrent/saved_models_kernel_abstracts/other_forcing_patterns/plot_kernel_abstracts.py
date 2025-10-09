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
- 目录与文件名、标题带上 forcing pattern（ringsLinf / sBands / ringsL1 / ringsCos / isoCircles / petals / unknown）
- 支持递归扫描与仅处理部分 pattern
"""

import argparse
import json
import os
from pathlib import Path
import glob
from typing import Optional, Tuple, List, Union, Dict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------------------- 配置：已知 pattern 名 -----------------------
KNOWN_PATTERNS = {"ringsLinf", "sBands", "ringsL1", "ringsCos", "isoCircles", "petals"}

# ----------------------- 打印工具 -----------------------

def _log(msg: str):
    print(msg, flush=True)

def _log_plot(model_name: str, layer: str, category: str, keyname: str):
    _log(f"[plot] model={model_name} layer={layer} category={category} key={keyname}")

def _log_skip(model_name: str, layer: str, category: str, keyname: str, reason: str):
    _log(f"[skip] model={model_name} layer={layer} category={category} key={keyname} :: {reason}")

# ----------------------- 基础工具 -----------------------

def _tonp(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def _real_if_complex(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        return None
    if np.iscomplexobj(arr):
        return np.abs(arr)
    else:
        return arr

def _finite_min_max(arr_list: List[np.ndarray]) -> Optional[Tuple[float, float]]:
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
    if color_log_scale:
        plot_vals = arr if color_log_eps is None else (arr + color_log_eps)
        norm = mcolors.LogNorm(vmin=np.nanmin(plot_vals), vmax=np.nanmax(plot_vals))
        return ax.imshow(plot_vals, origin="lower", norm=norm, cmap=cmap)
    else:
        return ax.imshow(arr, origin="lower", cmap=cmap)

# ----------------------- pattern 识别与命名 -----------------------

def detect_pattern(pt_path: Path, data_obj: Dict) -> str:
    # 1) 从文件内容 meta 里取
    try:
        meta = data_obj.get("model_meta", {})
        p = meta.get("forcing_pattern", None)
        if p and isinstance(p, str):
            return p
    except Exception:
        pass
    # 2) 从路径各级目录名里找
    for part in pt_path.parts:
        if part in KNOWN_PATTERNS:
            return part
    # 3) 不识别则 unknown
    return "unknown"

def _split_model_tag(pt_path: Path, pattern: str) -> str:
    """
    文件名形如: <pattern>__<model_tag>.analysis.pt
    若匹配则返回 <model_tag>，否则返回 stem 原样（去掉 .analysis 后缀）。
    """
    stem = pt_path.stem  # e.g., isoCircles__modes64_width60_epochs500_Tin10_T10.analysis -> "isoCircles__modes64_width60_epochs500_Tin10_T10"
    prefix = f"{pattern}__"
    return stem[len(prefix):] if stem.startswith(prefix) else stem

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
    model_name: str,
    pattern_name: str,
):
    spatial = data["layers"][layer]["spatial"]
    spatial_extra = data["layers"][layer].get("spatial_extra", {})

    skip_keys = {"U_top2", "V_top2", "S_top2"}

    save_dir.mkdir(parents=True, exist_ok=True)

    for group_name, group in [("spatial", spatial), ("spatial", spatial_extra)]:
        for key, tensor in group.items():
            if key in skip_keys:
                continue

            if isinstance(tensor, dict):
                for subk, subt in tensor.items():
                    keyname = f"{key}.{subk}"
                    _log_plot(model_name, layer, "spatial", keyname)

                    arr = _real_if_complex(_tonp(subt))
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
                        ax.set_title(f"[{pattern_name}] {model_name}  layer {layer}\nspatial.{keyname}", fontsize=10)
                        fname = f"{pattern_name}__{model_name}_layer{layer}_spatial_{key}_{subk}.png"
                        fig.savefig(save_dir / fname, dpi=150)
                        plt.close(fig)

                    else:
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
                            ax.set_title(f"[{pattern_name}] {model_name}  layer {layer}\nspatial.{key}.{subk}[...,{i}]", fontsize=10)
                            fname = f"{pattern_name}__{model_name}_layer{layer}_spatial_{key}_{subk}_slice{i}.png"
                            fig.savefig(save_dir / fname, dpi=150)
                            plt.close(fig)
                continue  # dict 情况已处理

            keyname = key
            _log_plot(model_name, layer, "spatial", keyname)

            arr = _real_if_complex(_tonp(tensor))
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
                ax.set_title(f"[{pattern_name}] {model_name}  layer {layer}\nspatial.{key}", fontsize=10)
                fname = f"{pattern_name}__{model_name}_layer{layer}_spatial_{key}.png"
                fig.savefig(save_dir / fname, dpi=150)
                plt.close(fig)

            else:
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
                    ax.set_title(f"[{pattern_name}] {model_name}  layer {layer}\nspatial.{key}[...,{i}]", fontsize=10)
                    fname = f"{pattern_name}__{model_name}_layer{layer}_spatial_{key}_slice{i}.png"
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
                    ax.set_title(f"[{pattern_name}] {model_name}  layer {layer}\nspatial.{key} ratio(σ1/σ2−1)", fontsize=10)
                    fname = f"{pattern_name}__{model_name}_layer{layer}_spatial_{key}_ratio.png"
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
    model_name: str,
    pattern_name: str,
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
                    tsv_indices, symmetric, s2_eps,
                    pattern_name=pattern_name
                )
        else:
            _plot_freq_pair_single(
                neg_tensor, pos_tensor,
                model_name, layer, k,
                save_dir,
                data_log_mode, color_log_scale, color_log_eps, cmap,
                tsv_indices, symmetric, s2_eps,
                pattern_name=pattern_name
            )

def Ai_valid_dim(A: np.ndarray, idx: int) -> bool:
    return (A is not None) and (A.ndim == 3) and (0 <= idx < A.shape[-1])

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
    s2_eps: float,
    *,
    pattern_name: str,
):
    A = _real_if_complex(_tonp(neg_tensor)) if neg_tensor is not None else None
    B = _real_if_complex(_tonp(pos_tensor)) if pos_tensor is not None else None

    if (A is None) and (B is None):
        return

    # 3D: 画切片 + ratio
    if (Ai_valid_dim(A, 0)) or (Ai_valid_dim(B, 0)):
        KA = A.shape[-1] if Ai_valid_dim(A, 0) else 0
        KB = B.shape[-1] if Ai_valid_dim(B, 0) else 0
        Kmax = max(KA, KB)

        for idx in tsv_indices:
            if idx >= Kmax:
                continue
            subkey = f"{keyname}[...,{idx}]"
            _log_plot(model_name, layer, "freq_pair", subkey)

            Ai = A[..., idx] if Ai_valid_dim(A, idx) else None
            Bi = B[..., idx] if Ai_valid_dim(B, idx) else None

            Ai_t = _data_log_transform(Ai, data_log_mode) if Ai is not None else None
            Bi_t = _data_log_transform(Bi, data_log_mode) if Bi is not None else None

            images = []
            fig, axs = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")

            if color_log_scale:
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
                axs[0].set_title(f"freq_neg\n[{pattern_name}] {subkey}", fontsize=9)
                if Bi_t is not None:
                    images.append(axs[1].imshow(Bi_t if color_log_eps is None else (Bi_t + color_log_eps),
                                                origin="lower", norm=norm, cmap=cmap))
                axs[1].set_title(f"freq_pos\n[{pattern_name}] {subkey}", fontsize=9)
            else:
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
                axs[0].set_title(f"freq_neg\n[{pattern_name}] {subkey}", fontsize=9)
                if Bi_t is not None:
                    images.append(axs[1].imshow(Bi_t, origin="lower", norm=norm, cmap=cmap))
                axs[1].set_title(f"freq_pos\n[{pattern_name}] {subkey}", fontsize=9)

            if images:
                fig.colorbar(images[-1], ax=axs, shrink=0.9)
            fname = f"{pattern_name}__{model_name}_layer{layer}_freq_pair_{keyname}_slice{idx}.png"
            fig.savefig(save_dir / fname, dpi=150)
            plt.close(fig)

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
                axs[0].set_title(f"freq_neg\n[{pattern_name}] {subkey}", fontsize=9)
                if Rp_t is not None:
                    images.append(axs[1].imshow(Rp_t if color_log_eps is None else (Rp_t + color_log_eps),
                                                origin="lower", norm=norm, cmap=cmap))
                axs[1].set_title(f"freq_pos\n[{pattern_name}] {subkey}", fontsize=9)
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
                axs[0].set_title(f"freq_neg\n[{pattern_name}] {subkey}", fontsize=9)
                if Rp_t is not None:
                    images.append(axs[1].imshow(Rp_t, origin="lower", norm=norm, cmap=cmap))
                axs[1].set_title(f"freq_pos\n[{pattern_name}] {subkey}", fontsize=9)

            if images:
                fig.colorbar(images[-1], ax=axs, shrink=0.9)
            fname = f"{pattern_name}__{model_name}_layer{layer}_freq_pair_{keyname}_ratio.png"
            fig.savefig(save_dir / fname, dpi=150)
            plt.close(fig)

    # 2D: 直接并排
    else:
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
            axs[0].set_title(f"freq_neg\n[{pattern_name}] {keyname}", fontsize=9)
            if B_t is not None:
                images.append(axs[1].imshow(B_t if color_log_eps is None else (B_t + color_log_eps),
                                            origin="lower", norm=norm, cmap=cmap))
            axs[1].set_title(f"freq_pos\n[{pattern_name}] {keyname}", fontsize=9)
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
            axs[0].set_title(f"freq_neg\n[{pattern_name}] {keyname}", fontsize=9)
            if B_t is not None:
                images.append(axs[1].imshow(B_t, origin="lower", norm=norm, cmap=cmap))
            axs[1].set_title(f"freq_pos\n[{pattern_name}] {keyname}", fontsize=9)

        if images:
            fig.colorbar(images[-1], ax=axs, shrink=0.9)
        fname = f"{pattern_name}__{model_name}_layer{layer}_freq_pair_{keyname}.png"
        fig.savefig(save_dir / fname, dpi=150)
        plt.close(fig)

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
    data = torch.load(pt_path, map_location="cpu")  # 安全地在 CPU 上加载检查点
    pattern = detect_pattern(pt_path, data)
    model_name = _split_model_tag(pt_path, pattern)

    model_dir = out_root / pattern / model_name
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
        "pattern": pattern,
        "model_tag": model_name,
    }
    with open(model_dir / "plot_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    cmap = plt.get_cmap(cmap_name) if cmap_name else None

    _log(f"[model] start {model_name} (pattern={pattern})")
    for L in layers:
        Ls = str(L)
        _log(f"[layer] model={model_name} pattern={pattern} layer={Ls} ...")
        layer_dir = model_dir / f"layer{Ls}"
        spatial_dir = layer_dir / "spatial"
        freq_dir = layer_dir / "freq_pair"

        plot_spatial(
            data, layer=Ls, top_singval_k=top_singval_k,
            save_dir=spatial_dir,
            data_log_mode=data_log_mode, color_log_scale=color_log_scale,
            color_log_eps=color_log_eps, cmap=cmap, s2_eps=s2_eps,
            model_name=model_name, pattern_name=pattern
        )
        plot_freq_pair(
            data, layer=Ls, tsv_indices=tsv_indices, symmetric=symmetric,
            save_dir=freq_dir,
            data_log_mode=data_log_mode, color_log_scale=color_log_scale,
            color_log_eps=color_log_eps, cmap=cmap, s2_eps=s2_eps,
            model_name=model_name, pattern_name=pattern
        )
    _log(f"[model] done {model_name} (pattern={pattern})")

def parse_args():
    p = argparse.ArgumentParser(
        description="批量绘制 *.analysis.pt 的 spatial / freq 对比图，结构化输出（带 forcing pattern 命名）。"
    )
    p.add_argument("--input-dir", type=str, required=True,
                   help="输入根目录（可递归）：包含各 pattern 子目录与 *.analysis.pt。")
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
    p.add_argument("--recursive", action="store_true",
                   help="递归扫描 input-dir 下所有子目录。")
    p.add_argument("--only-patterns", type=str, default=None,
                   help="仅处理这些 pattern（逗号分隔），如 'ringsLinf,sBands'；留空处理全部。")
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

    # 文件发现：递归/非递归 + 仅处理指定 pattern
    if args.recursive:
        pt_files = sorted(in_dir.rglob("*.analysis.pt"))
    else:
        pt_files = sorted(in_dir.glob("*.analysis.pt"))

    if args.only_patterns:
        allow = {s.strip() for s in args.only_patterns.split(",") if s.strip()}
        pt_files = [p for p in pt_files if any(part in allow for part in p.parts)]

    if not pt_files:
        raise FileNotFoundError(f"在 {in_dir} 下找不到 *.analysis.pt 文件（recursive={args.recursive}）。")

    print(f"[info] RunTag = {run_tag}")
    print(f"[info] 发现 {len(pt_files)} 个模型：")
    for p in pt_files:
        print(" -", os.path.relpath(p, in_dir))

    for p in pt_files:
        print(f"[run] 处理 {os.path.relpath(p, in_dir)} ...")
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







