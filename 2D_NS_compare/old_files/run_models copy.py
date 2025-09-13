#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluate FNO models across many datasets (train / test / generalizability / expanded)

功能:
- 递归扫描指定路径下的所有 .pt 数据集文件 (支持文件或文件夹)。
- 对给定的 models_params 中的每个模型、对每个数据集逐一评估。
- 支持 FNO2d 初末帧 (initial_to_final)、FNO2d 自回归 (recurrent)、FNO3d (padding / no_padding)。
- 计算每个 (模型×数据集) 的平均 RMSE / MAE，并保存为一个 CSV。
- CSV 列包含: 模型族/分组/键名、modes/width/epochs/Tin/T 等超参、数据集组别(train/test/generalizability/expanded)、数据集名/路径、样本数/耗时等。
- 控制台实时打印: 当前模型、当前数据集、用时、均值指标。

用法示例:

python batch_eval_fno_datasets.py \
  --train "/path/to/train_or_dir" \
  --test "/path/to/test_or_dir" \
  --generalizability "/path/to/gen_dir" \
  --expanded "/path/to/expanded_dir" \
  --max-samples 50 \
  --out-csv "/path/to/results.csv"

注意:
- 本脚本假设 .pt 数据存成 dict，至少包含键 'x' 和/或 'y'：
  * initial_to_final: 使用 data['x'][i] 作为输入, data['y'][i] 作为目标。
  * recurrent / FNO3d: 使用 data['y'][i] 形状 (H, W, T_total)，从中切出 T_in 和 T_out。
- 若 epochs 未在 models_params 中显式给出，将从 key 或 model_path 中用正则尝试解析。
"""

import os
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import pandas as pd
from tqdm import tqdm

# ============ 你现有的模型类 ============
from FNO2d import FNO2d, RecurrentPredictor
from FNO3d import FNO3d

# ============ 你的现有 models_params 原样可粘贴 ============

base_path = "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO"


# 为了示例，这里提供一个空壳。请把你发来的 models_params 粘贴替换掉下面这个占位。
models_params = {


    "FNO2d":{
        "initial_to_final":{
            "modes32_width40_epochs500":{
                "model_path":f"{base_path}/2D_NS_old/2D_NS_Zongyi_Li/initial_to_final/saved_models/2D/256_256/modes32_width40_epochs500/input_frame_0_output_frame_19/NS_2d_FNO_model_trainedby_NS_data_zongyi_train.pth",
                "data_path":f"{base_path}/2D_NS_old/2D_NS_Zongyi_Li/initial_to_final/datasets/2D/NS/256_256/output_frame_19/NS_data_zongyi_test.pt",
                "modes":32,
                "width":40,
                },
            "modes64_width60_epochs500":{
                "model_path":f"{base_path}/2D_NS_old/2D_NS_Zongyi_Li/initial_to_final/saved_models/2D/256_256/modes64_width60_epochs500/input_frame_0_output_frame_19/NS_2d_FNO_model_trainedby_NS_data_zongyi_train.pth",
                "data_path":f"{base_path}/2D_NS_old/2D_NS_Zongyi_Li/initial_to_final/datasets/2D/NS/256_256/output_frame_19/NS_data_zongyi_test.pt",
                "modes":64,
                "width":60,
                },
            },

        "recurrent":{
            "modes12_width20_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes12_width20_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":12,
                "width":20,
                "Tin":10,
                "T":10,
                },

            "modes16_width60_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes16_width60_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":16,
                "width":60,
                "Tin":10,
                "T":10,
                },

            "modes32_width40_epochs500_Tin1_T19":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes32_width40_epochs500_Tin1_T19/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":32,
                "width":40,
                "Tin":1,
                "T":19,
                },
            "modes32_width40_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes32_width40_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":32,
                "width":40,
                "Tin":10,
                "T":10,
                },

            "modes32_width60_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes32_width60_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":32,
                "width":60,
                "Tin":10,
                "T":10,
                },

            "modes48_width40_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes48_width40_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":48,
                "width":40,
                "Tin":10,
                "T":10,
                },
            "modes48_width60_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes48_width60_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":48,
                "width":60,
                "Tin":10,
                "T":10,
                },
            "modes48_width80_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes48_width80_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":48,
                "width":80,
                "Tin":10,
                "T":10,
                },
            "modes64_width60_epochs500_Tin1_T19":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes64_width60_epochs500_Tin1_T19/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":64,
                "width":60,
                "Tin":1,
                "T":19,
                },
            "modes64_width60_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes64_width60_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":64,
                "width":60,
                "Tin":10,
                "T":10,
                },
            "modes64_width40_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes64_width40_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":64,
                "width":40,
                "Tin":10,
                "T":10,
                },
            "modes64_width80_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes64_width80_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":64,
                "width":80,
                "Tin":10,
                "T":10,
                },
            "modes96_width40_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes96_width40_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":96,
                "width":40,
                "Tin":10,
                "T":10,
                },
            "modes96_width80_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes96_width80_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":96,
                "width":80,
                "Tin":10,
                "T":10,
                },
            "modes96_width80_epochs1500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes96_width80_epochs1500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":96,
                "width":80,
                "Tin":10,
                "T":10,
                },
            "modes128_width40_epochs500_Tin10_T10":{
                "model_path":f"{base_path}/2D_NS_FNO2d_recurrent/saved_models/2D/modes128_width40_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes":128,
                "width":40,
                "Tin":10,
                "T":10,
                },
            },
        },

    "FNO3d":{
        "no_padding":{
            "modes1232_modes36_width40_epochs500_Tin10_T10_no_padding":{
                "model_path":f"{base_path}/2D_NS_FNO3d/saved_models/3D/no_padding/modes1232_modes36_width40_epochs500_Tin10_T10/NS_3d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes12":32,
                "modes3":6,
                "width":40,
                "Tin":10,
                "T":10,
                },
            "modes1232_modes38_width40_epochs500_Tin1_T19_no_padding":{
                "model_path":f"{base_path}/2D_NS_FNO3d/saved_models/3D/no_padding/modes1232_modes38_width40_epochs500_Tin1_T19/NS_3d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes12":32,
                "modes3":8,
                "width":40,
                "Tin":1,
                "T":19,
                },
            "modes1264_modes36_width60_epochs500_Tin10_T10_no_padding":{
                "model_path":f"{base_path}/2D_NS_FNO3d/saved_models/3D/no_padding/modes1264_modes36_width60_epochs500_Tin10_T10/NS_3d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes12":64,
                "modes3":6,
                "width":60,
                "Tin":10,
                "T":10,
                },
        },

        "padding":{
            "modes1232_modes38_width40_epochs500_Tin10_T10_padding":{
                "model_path":f"{base_path}/2D_NS_FNO3d/saved_models/3D/padding/modes1232_modes38_width40_epochs500_Tin10_T10/NS_3d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes12":32,
                "modes3":8,
                "width":40,
                "Tin":10,
                "T":10,
                "padding":6
                },
            "modes1232_modes312_width40_epochs500_Tin1_T19_padding":{
                "model_path":f"{base_path}/2D_NS_FNO3d/saved_models/3D/padding/modes1232_modes312_width40_epochs500_Tin1_T19/NS_3d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth",
                "data_path":f"{base_path}/2D_NS_FNO2d_recurrent/datasets/exponax_datasets/t20/dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt",
                "modes12":32,
                "modes3":12,
                "width":40,
                "Tin":1,
                "T":19,
                "padding":6
                },
            },
        }, 


}

# ============ 工具函数 ============

def is_pt_file(p: Path) -> bool:
    return p.is_file() and p.suffix == ".pt"


def list_pt_files(path_or_dir: Optional[str]) -> List[Path]:
    if not path_or_dir:
        return []
    p = Path(path_or_dir)
    if p.is_file() and p.suffix == ".pt":
        return [p]
    if p.is_dir():
        return sorted([x for x in p.rglob("*.pt") if x.is_file()])
    return []


def base_name_no_ext(p: Path) -> str:
    return p.stem


def parse_hparams_from_text(text: str) -> Dict[str, Optional[int]]:
    """从字符串中尽量解析出 epochs/modes/width/Tin/T/modes12/modes3 等。"""
    if not isinstance(text, str):
        return {}
    d = {}
    # 通用
    m = re.search(r"epochs(\d+)", text)
    if m: d["epochs"] = int(m.group(1))
    m = re.search(r"modes(\d+)", text)  # FNO2d
    if m: d.setdefault("modes", int(m.group(1)))
    m = re.search(r"width(\d+)", text)
    if m: d["width"] = int(m.group(1))
    m = re.search(r"Tin(\d+)", text)
    if m: d["Tin"] = int(m.group(1))
    # 允许 T 或 Tout
    m = re.search(r"(?:Tout|T)(\d+)", text)
    if m: d["T"] = int(m.group(1))
    # FNO3d 的两个 modes
    m = re.search(r"modes12(\d+)", text)
    if m: d["modes12"] = int(m.group(1))
    m = re.search(r"modes3(\d+)", text)
    if m: d["modes3"] = int(m.group(1))
    return d


def to_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


# ============ 单模型×单数据集 评估函数 ============

def eval_initial_to_final(model: FNO2d, data: dict, device: torch.device, max_samples: int) -> Tuple[float, float, int]:
    assert 'x' in data and 'y' in data, "initial_to_final 数据需包含 'x' 和 'y'"
    n = min(max_samples, len(data['y']))
    rmses, maes = [], []
    model.eval()
    with torch.no_grad():
        for i in range(n):
            # x = data['x'][i].to(device)  # (H, W)
            x = data['y'][i][...,0].to(device)
            y = data['y'][i][...,-2].to(device)  # (H, W)
            pred = model(x.unsqueeze(0).unsqueeze(-1)).squeeze(0).squeeze(-1)  # (H, W)
            err = y - pred
            rmses.append(torch.sqrt(torch.mean(err**2)).item())
            maes.append(torch.mean(torch.abs(err)).item())
    return float(sum(rmses)/len(rmses)), float(sum(maes)/len(maes)), n


def eval_recurrent_fno2d(model: FNO2d, Tin: int, T: int, data: dict, device: torch.device, max_samples: int) -> Tuple[float, float, int]:
    assert 'y' in data, "recurrent 数据需包含 'y'，形状 (H, W, T_total)"
    rp = RecurrentPredictor(model, T_out=T, step=1)
    n_total = len(data['y'])
    n = min(max_samples, n_total)
    rmses, maes = [], []
    model.eval()
    with torch.no_grad():
        for i in range(n):
            sample = data['y'][i].to(device)  # (H, W, T_total)
            assert sample.ndim == 3 and sample.shape[-1] >= Tin + T, f"样本 {i} 的时间长度不足 Tin+T"
            current_input = sample[None, ..., :Tin]           # (1, H, W, Tin)
            gt = sample[None, ..., Tin:Tin+T]                 # (1, H, W, T)
            pred = rp(current_input)                          # (1, H, W, T)
            err = gt[..., -1] - pred[..., -1]                 # 对最后一步计算
            rmses.append(torch.sqrt(torch.mean(err**2)).item())
            maes.append(torch.mean(torch.abs(err)).item())
    return float(sum(rmses)/len(rmses)), float(sum(maes)/len(maes)), n


def eval_fno3d(model: FNO3d, Tin: int, T: int, data: dict, device: torch.device, max_samples: int) -> Tuple[float, float, int]:
    assert 'y' in data, "FNO3d 数据需包含 'y'，形状 (H, W, T_total)"
    n_total = len(data['y'])
    n = min(max_samples, n_total)
    rmses, maes = [], []
    model.eval()
    with torch.no_grad():
        for i in range(n):
            sample = data['y'][i].to(device)  # (H, W, T_total)
            assert sample.ndim == 3 and sample.shape[-1] >= Tin + T, f"样本 {i} 的时间长度不足 Tin+T"
            current_input = sample[None, ..., :Tin]           # (1, H, W, Tin)
            gt = sample[None, ..., Tin:Tin+T]                 # (1, H, W, T)
            B, H, W, _ = current_input.shape
            # FNO3d 这里把时间作为第 3 个空间维度, 用长度 T 的重复 (与原脚本一致)
            pred = model(current_input.reshape(1, H, W, 1, Tin).repeat(1, 1, 1, T, 1))  # (1, H, W, T, 1?)
            if pred.shape[-1] == 1:
                pred_last = pred[..., -1, 0]
            else:
                pred_last = pred[..., -1]
            err = gt[..., -1].squeeze(0) - pred_last.squeeze(0)  # (H, W)
            rmses.append(torch.sqrt(torch.mean(err**2)).item())
            maes.append(torch.mean(torch.abs(err)).item())
    return float(sum(rmses)/len(rmses)), float(sum(maes)/len(maes)), n


# ============ 主流程 ============

def build_dataset_catalog(args) -> List[Tuple[str, Path]]:
    """返回 [(group, path_to_pt), ...]"""
    catalog = []
    for group, p in [
        ("train", args.train),
        ("test", args.test),
        ("generalizability", args.generalizability),
        ("expanded", args.expanded),
    ]:
        for fp in list_pt_files(p):
            catalog.append((group, fp))
    return catalog


def load_data_pt(path: Path, device: torch.device) -> Optional[dict]:
    try:
        data = torch.load(str(path), weights_only=False, map_location=device)
        if not isinstance(data, dict):
            print(f"[WARN] {path} 加载结果不是 dict，跳过。")
            return None
        return data
    except Exception as e:
        print(f"[ERROR] 加载 {path} 失败: {e}")
        return None


def ensure_device(device_str: str) -> torch.device:
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，切换到 CPU")
        return torch.device('cpu')
    return torch.device(device_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=None, help='train 数据集文件或文件夹路径')
    parser.add_argument('--test', type=str, default=None, help='test 数据集文件或文件夹路径')
    parser.add_argument('--generalizability', type=str, default=None, help='generalizability 文件夹路径')
    parser.add_argument('--expanded', type=str, default=None, help='expanded 文件夹路径')
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--out-csv', type=str, default='results_batch_eval.csv')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    device = ensure_device(args.device)

    catalog = build_dataset_catalog(args)
    if not catalog:
        print('[ERROR] 未找到任何 .pt 数据集文件，请检查路径。')
        return

    rows = []  # 收集 CSV 行

    # 遍历模型
    for family in models_params.keys():
        family_dict = models_params[family]
        for group in family_dict.keys():
            group_dict = family_dict[group]
            for key in group_dict.keys():
                info = group_dict[key]

                # 解析/准备模型超参
                model_path = info.get('model_path')
                hps = {}
                hps.update(parse_hparams_from_text(key))
                hps.update(parse_hparams_from_text(model_path or ''))
                # 显式字段覆盖
                for k in ['modes', 'width', 'epochs', 'Tin', 'T', 'modes12', 'modes3']:
                    if k in info:
                        hps[k] = info[k]

                t_model_start = time.perf_counter()

                # 实例化模型
                try:
                    if family == 'FNO2d':
                        if group == 'initial_to_final':
                            modes = hps.get('modes')
                            width = hps.get('width')
                            assert modes and width, f"缺少 modes/width: {key}"
                            model = FNO2d(modes1=modes, modes2=modes, width=width, in_channels=1).to(device)
                        elif group == 'recurrent':
                            modes = hps.get('modes')
                            width = hps.get('width')
                            Tin = hps.get('Tin')
                            T = hps.get('T')
                            assert modes and width and Tin and T, f"缺少 modes/width/Tin/T: {key}"
                            model = FNO2d(modes1=modes, modes2=modes, width=width, in_channels=Tin).to(device)
                        else:
                            print(f"[WARN] 未知 FNO2d 分组 {group}，跳过。")
                            continue
                    elif family == 'FNO3d':
                        modes12 = hps.get('modes12') or hps.get('modes')  # 兼容
                        modes3 = hps.get('modes3')
                        width = hps.get('width')
                        Tin = hps.get('Tin')
                        T = hps.get('T')
                        padding = (group == 'padding')
                        assert modes12 and modes3 and width and Tin and T, f"缺少 modes12/modes3/width/Tin/T: {key}"
                        model = FNO3d(modes1=modes12, modes2=modes12, modes3=modes3, width=width, in_channels=Tin, padding=padding).to(device)
                    else:
                        print(f"[WARN] 未知模型族 {family}，跳过。")
                        continue

                    # 加载权重
                    if model_path:
                        state = torch.load(model_path, map_location=device)
                        model.load_state_dict(state)
                    model.eval()
                except Exception as e:
                    print(f"[ERROR] 实例化或加载模型失败: {family} / {group} / {key}: {e}")
                    continue

                t_model_ready = time.perf_counter()
                print("="*70)
                print(f"Model ready: family={family} group={group} key={key}\n  hparams={hps}\n  load_time={t_model_ready - t_model_start:.2f}s")

                # 遍历所有数据集
                for ds_group, ds_path in catalog:
                    t_ds_start = time.perf_counter()
                    ds_data = load_data_pt(ds_path, device)
                    if ds_data is None:
                        continue

                    # 评估
                    try:
                        if family == 'FNO2d' and group == 'initial_to_final':
                            rmse_mean, mae_mean, n = eval_initial_to_final(model, ds_data, device, args.max_samples)
                            Tin = None; T = None; modes12=None; modes3=None; padding=None
                        elif family == 'FNO2d' and group == 'recurrent':
                            Tin = to_int(hps.get('Tin'))
                            T = to_int(hps.get('T'))
                            rmse_mean, mae_mean, n = eval_recurrent_fno2d(model, Tin, T, ds_data, device, args.max_samples)
                            modes12=None; modes3=None; padding=None
                        elif family == 'FNO3d':
                            Tin = to_int(hps.get('Tin'))
                            T = to_int(hps.get('T'))
                            rmse_mean, mae_mean, n = eval_fno3d(model, Tin, T, ds_data, device, args.max_samples)
                            modes12 = to_int(hps.get('modes12') or hps.get('modes'))
                            modes3 = to_int(hps.get('modes3'))
                            padding = (group == 'padding')
                        else:
                            print(f"[WARN] 未知组合，跳过 {family}/{group}")
                            continue
                    except AssertionError as ae:
                        print(f"[SKIP] {ds_path.name}: {ae}")
                        continue
                    except Exception as e:
                        print(f"[ERROR] 评估失败 {family}/{group}/{key} @ {ds_path}: {e}")
                        continue

                    t_ds_end = time.perf_counter()
                    elapsed = t_ds_end - t_ds_start
                    ms_per_sample = (elapsed / max(1, n)) * 1000.0

                    # 记录 CSV 行
                    row = {
                        # 模型信息
                        'model_family': family,
                        'model_group': group,
                        'model_key': key,
                        'model_path': model_path,
                        'modes': to_int(hps.get('modes')),
                        'width': to_int(hps.get('width')),
                        'epochs': to_int(hps.get('epochs')),
                        'Tin': to_int(Tin),
                        'T': to_int(T),
                        'modes12': to_int(modes12),
                        'modes3': to_int(modes3),
                        'padding': padding,
                        # 数据集信息
                        'dataset_group': ds_group,
                        'dataset_name': base_name_no_ext(ds_path),
                        'dataset_path': str(ds_path),
                        # 结果
                        'samples_used': n,
                        'rmse_mean': rmse_mean,
                        'mae_mean': mae_mean,
                        'seconds': elapsed,
                        'ms_per_sample': ms_per_sample,
                    }
                    rows.append(row)

                    print(f"  ▶ {ds_group:<17} | {ds_path.name:<50} | n={n:<3d} | RMSE={rmse_mean:.6f} | MAE={mae_mean:.6f} | time={elapsed:.2f}s ({ms_per_sample:.1f} ms/sample)")

    # 保存 CSV
    if rows:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print("="*70)
        print(f"[OK] 保存结果: {out_path}  (共 {len(rows)} 条记录)")
    else:
        print('[WARN] 没有可保存的结果。')


if __name__ == '__main__':
    main()




