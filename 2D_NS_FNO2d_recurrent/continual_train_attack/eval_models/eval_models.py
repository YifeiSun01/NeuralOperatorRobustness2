#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified batch evaluator for FNO models across families (FNO2d recurrent/expanded,
FNO2d initial_to_final, and FNO3d with/without padding).

- Recursively discover *.pth under multiple --model_dir roots (appendable)
- Auto-detect model family by path + robust hparam parsing:
  * FNO2d recurrent: e.g. modes64_width60_epochs500_Tin10_T10
  * FNO2d initial_to_final: e.g. initial_to_final/.../modes64_width60_epochs500/...
  * FNO3d: e.g. .../modes1232_modes36_width40_epochs500_Tin10_T10_(no_)padding
            → parsed as modes1=32, modes2=32, modes3=6
- State-dict shape-safe partial load (skip size-mismatch keys, allow resume)
- Evaluate against four dataset groups:
  train(.pt), test(.pt), expanded_dir/*.pt, gen_dir/*.pt
- Expanded folder models: prefer matched expanded dataset (from manifest_{add|replace}.json)
- Per-(model,dataset) metrics: RMSE/MAE/MAPE(on last frame) + REL L2 (whole-field)
- Combined CSV + side-car JSON catalog (incremental, resume-safe)
- NEW: At startup, print grouped & numbered lists of found MODELS and DATASETS.
"""

from __future__ import annotations
import os, sys, re, csv, json, math, argparse
from pathlib import Path
from collections import defaultdict
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

# -------------------------- Optional project root injection --------------------------
def add_project_root(levels_up: int = 3):
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    project_root = here
    for _ in range(levels_up):
        project_root = project_root.parent
    sys.path.append(str(project_root))

# 将工程根目录与脚本上级加入 sys.path，兼容多种导入结构
add_project_root(levels_up=3)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ---- Import models (two common layouts) ----
try:
    from models.FNO2d import FNO2d, RecurrentPredictor  # preferred
except Exception:
    from FNO2d import FNO2d, RecurrentPredictor  # fallback

try:
    from models.FNO3d import FNO3d  # optional
except Exception:
    FNO3d = None  # if not available, we won't try FNO3d

# >>> NEW FILTERS: helpers
def _compile_regex_list(patterns):
    if not patterns: return []
    out = []
    for p in patterns:
        p = p.strip()
        if not p: continue
        out.append(re.compile(p, flags=re.IGNORECASE))
    return out

def apply_model_filters(
    models: list[tuple[str, Path]],
    exclude_res: list[re.Pattern],
) -> list[tuple[str, Path]]:
    # models: [(group, path)]
    keep_groups = None
    kept = []
    for grp, p in models:
        if keep_groups and grp not in keep_groups:
            continue
        sp = str(p)
        # exclude: 命中任意则丢弃
        if exclude_res and any(r.search(sp) for r in exclude_res):
            continue
        kept.append((grp, p))
    return kept

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def torch_load_compat(path: Path, map_location=None):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)

def find_models_recursively(roots: List[Path]) -> List[Path]:
    """Find candidate model files under multiple roots."""
    models: List[Path] = []
    for root in roots:
        if (not root) or (not root.exists()) or (not root.is_dir()):
            continue
        cands: List[Path] = []
        for pat in ("**/*.pth", "**/*.pt"):
            cands.extend(p for p in root.rglob(pat) if p.is_file())
        for p in cands:
            s = p.stem.lower()
            if p.suffix.lower() == ".pth":
                models.append(p); continue
            if any(tok in s for tok in ("model", "ckpt", "checkpoint", "state_dict")):
                models.append(p); continue
            if s.endswith("_all_frames"):
                continue
            models.append(p)
    # de-dup + sort
    models = sorted(dict.fromkeys(models))
    return models

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def mape_decimal(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error as a decimal in [0, +inf), not percent.
    """
    denom = np.maximum(np.abs(a), eps)
    return float(np.mean(np.abs((a - b) / denom)))

def pretty_float(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return str(x)
    return f"{x:.6g}"

def get_sequence_tensor(data: Dict[str, Any]) -> torch.Tensor:
    for key in ("y", "x"):
        if key in data and torch.is_tensor(data[key]):
            return data[key]
    for v in data.values():
        if torch.is_tensor(v) and v.dim() >= 3:
            return v
    raise ValueError("Could not find a suitable sequence tensor in dataset (expect 'y' or 'x').")

def ensure_batched_nhwt(seq: torch.Tensor) -> torch.Tensor:
    if seq.dim() == 3:
        return seq.unsqueeze(0)
    if seq.dim() == 4:
        return seq
    raise ValueError(f"Expected 3D or 4D tensor (H,W,T) or (N,H,W,T), got {tuple(seq.shape)}")

# 新增：整体相对 L2（最后一帧，非逐元素）
def rel_l2_whole(pred_last: torch.Tensor, gt_last: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Relative L2 error over the whole field on the last frame:
        ||pred - gt||_2 / ||gt||_2
    pred_last, gt_last: (H, W) or any shape of the last frame.
    """
    # flatten then compute vector norm
    diff = (pred_last - gt_last).reshape(-1).float()
    gvec = gt_last.reshape(-1).float()
    diff_norm = torch.norm(diff, p=2)
    gt_norm = torch.norm(gvec, p=2)
    denom = torch.clamp(gt_norm, min=eps)
    return float((diff_norm / denom).item())

# ------------------------ filename parsers ------------------------
DATASET_KEYS = [
    "type","solver","nx","N","nu","t","ntimepoints",
    "length_scale","variance","alpha","tau","period","exp_factor",
    "vmax","vmin"
]

def parse_dataset_name(stem: str) -> Dict[str, str]:
    info: Dict[str, str] = {}
    patterns = {
        "nx":           r"_nx(\d+)",
        "N":            r"_N(\d+)",
        "nu":          r"_nu([-\d\.]+)",
        "t":            r"_t([-\d\.]+)",
        "ntimepoints":  r"_ntimepoints(\d+)",
        "length_scale": r"_length_scale([-\d\.]+)",
        "variance":     r"_variance([-\d\.]+)",
        "alpha":        r"_alpha([-\d\.]+)",
        "tau":          r"_tau([-\d\.]+)",
        "period":       r"_period([-\d\.]+)",
        "exp_factor":   r"_exp_factor([-\d\.]+)",
        "vmax":         r"_vmax([-\d\.]+)",
        "vmin":         r"_vmin([-\d\.]+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, stem)
        if m: info[k] = m.group(1)
    m = re.search(r"solver=([^_]+)", stem)
    if m: info["solver"] = m.group(1)
    # rough type guess
    m = re.search(r"ntimepoints\d+_(.*)", stem)
    if m:
        tail = m.group(1)
        m2 = re.match(r"([A-Za-z]+)(?:_([A-Za-z]+))?", tail)
        if m2:
            info["type"] = m2.group(1) if not m2.group(2) else f"{m2.group(1)}_{m2.group(2)}"
    return info

# ------------------------ model meta parsers ------------------------
# unified keys kept for CSV
MODEL_KEYS_BASE = ["modes1","modes2","modes3","width","Tin","Tout","step","epochs","seed"]
MODEL_KEYS_EXTRA = [
    "strategy","pct_tag","percent","expanded_stem","expanded_file_name","expanded_file_path",
    "orig_train_dataset","with_replacement","k_requested","N_in_file",
]
MODEL_KEYS = MODEL_KEYS_BASE + MODEL_KEYS_EXTRA

def _parse_common_int(s: str, key: str) -> Optional[int]:
    m = re.search(fr"{key}[_=]?(\d+)", s)
    return int(m.group(1)) if m else None

def parse_model_info_from_path(path: Path) -> Dict[str, str]:
    """Parse modes/width/Tin/Tout/epochs/seed etc. Robust to:
       - modes64_width60_epochs500_Tin10_T10
       - modes64__width60...
       - FNO3d tokens: modes1232 (→ m1=32,m2=32), modes36 (→ m3=6), modes312 (→ m3=12)
       - initial_to_final: input_frame_0_output_frame_19
    """
    s = str(path)
    info: Dict[str, str] = {}

    # ---- FNO3d special tokens ---------------------------------------------------------
    # modes1232 → modes1=32, modes2=32
    for m in re.finditer(r"modes12(\d+)", s):
        val = int(m.group(1))
        info["modes1"] = str(val)
        info["modes2"] = str(val)
    # modes36 / modes38 / modes312 → modes3 = 6/8/12
    m3 = re.search(r"modes3(\d+)", s)
    if m3:
        info["modes3"] = m3.group(1)

    # ---- Generic 2D tokens ------------------------------------------------------------
    # modes64 / modes96 / modes128 → modes1=modes2=val (only set if not already from modes12XX)
    m = re.search(r"modes[_=]?(\d+)(?!\d)", s)
    if m and ("modes1" not in info and "modes2" not in info):
        val = m.group(1)
        info["modes1"] = val
        info["modes2"] = val

    # width / Tin / Tout / epochs / seed / step
    for key in ("width","Tin","T","Tout","epochs","seed","step"):
        v = _parse_common_int(s, key)
        if v is not None:
            # normalize T → Tout
            if key == "T": info["Tout"] = str(v)
            else: info[key] = str(v)

    # initial_to_final: input_frame_0_output_frame_19 (→ Tin=1, Tout=1)
    m_inout = re.search(r"input_frame[_=]?(\d+)_output_frame[_=]?(\d+)", s)
    if m_inout and "Tin" not in info:
        info["Tin"] = "1"
        if "Tout" not in info:
            info["Tout"] = "1"

    # width 再兜底（有的路径写成 width60 或 __width60）
    if "width" not in info:
        m = re.search(r"width[_=]?(\d+)", s)
        if m: info["width"] = m.group(1)

    return info

def enrich_model_meta_extra(path: Path, meta: Dict[str, str]) -> Dict[str, str]:
    """补充 expanded 策略、pct、expanded_stem、原始训练集等。"""
    info = dict(meta) if meta else {}
    parts = list(path.resolve().parts)

    # 策略（add/replace）
    strategy = None
    for p in reversed(parts[:-1]):
        lp = p.lower()
        if lp in ("add","replace"):
            strategy = lp; break
    if strategy:
        info["strategy"] = strategy

    # pct_tag + expanded_stem：.../<expanded_stem>/<XXpct>/<add|replace>/...
    pct_tag = None; expanded_stem = None
    for i, seg in enumerate(parts):
        ls = seg.lower()
        if ls.endswith("pct") and ls[:-3].isdigit():
            pct_tag = seg
            if i-1 >= 0:
                expanded_stem = parts[i-1]
            break
    if pct_tag:
        info["pct_tag"] = pct_tag
        try:
            pct_int = int(pct_tag.lower()[:-3])
            info["percent"] = str(pct_int / 100.0)
        except Exception:
            pass
    if expanded_stem and "expanded_stem" not in info:
        info["expanded_stem"] = expanded_stem

    # manifest_*（与 ckpt 同目录）
    try:
        parent = path.parent
        man_name = "manifest_add.json" if strategy == "add" else ("manifest_replace.json" if strategy == "replace" else None)
        if man_name:
            man_path = parent / man_name
            if man_path.exists():
                with open(man_path, "r") as mf:
                    man = json.load(mf)
                cand = man.get("manifest") if isinstance(man, dict) and "manifest" in man else man
                if isinstance(cand, dict):
                    file_path = cand.get("file", "")
                    if file_path:
                        info["expanded_file_path"] = file_path
                        info["expanded_file_name"] = Path(file_path).name
                        info.setdefault("expanded_stem", Path(file_path).stem)
                    for k in ("k_requested","N_in_file","with_replacement"):
                        if k in cand:
                            info[k] = str(cand[k])
    except Exception:
        pass

    # 原始训练数据集（log 文件名中的 trainedby_*）
    try:
        parent = path.parent
        for lf in parent.glob("NS_2d_FNO_log_trainedby_*.txt"):
            name = lf.name
            prefix = "NS_2d_FNO_log_trainedby_"
            suffix = ".txt"
            if name.startswith(prefix) and name.endswith(suffix):
                info["orig_train_dataset"] = name[len(prefix):-len(suffix)]
                break
    except Exception:
        pass
    return info

# --------------------------------------------------------------------------------------
# Robust model loading (infer arch from checkpoint shapes + shape-safe partial load)
# --------------------------------------------------------------------------------------
def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def _infer_arch_from_state_dict_2d(state_dict: Dict[str, torch.Tensor]) -> Optional[Tuple[int, int, int]]:
    """Infer (modes1,modes2,width) from 4D spectral weight [width,width,m1,m2]."""
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 4:
            w0, w1, m1, m2 = v.shape
            if w0 == w1 and m1 > 0 and m2 > 0:
                return int(m1), int(m2), int(w0)
    return None

def _infer_arch_from_state_dict_3d(state_dict: Dict[str, torch.Tensor]) -> Optional[Tuple[int, int, int, int]]:
    """Infer (modes1,modes2,modes3,width) from 5D [*,*,*,m1,m2?] — heuristic."""
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 5:
            dims = list(v.shape)
            tail = dims[-3:]
            if all(isinstance(x, int) and x > 0 for x in tail):
                m1, m2, m3 = tail
                width = dims[0]
                return int(m1), int(m2), int(m3), int(width)
    return None

def _shape_safe_load(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str], List[str]]:
    model_sd = model.state_dict()
    loaded_keys, skipped_shape, unexpected = [], [], []

    sd = _strip_module_prefix(state_dict)
    compat = {}
    for k, v in sd.items():
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                compat[k] = v
                loaded_keys.append(k)
            else:
                skipped_shape.append(k)
        else:
            unexpected.append(k)
    model.load_state_dict(compat, strict=False)
    missing = [k for k in model_sd.keys() if k not in loaded_keys]
    return loaded_keys, missing, skipped_shape

# --------------------------------------------------------------------------------------
# Evaluators per family (return means + sample stds + N)
# --------------------------------------------------------------------------------------
def _finalize_err_stats4(r_list: List[float], a_list: List[float], m_list: List[float], rel_list: List[float]) -> Tuple[float,float,float,float,float,float,float,float,int]:
    n = len(r_list)
    def _mean_std(x):
        if n == 0:
            return float("nan"), 0.0
        arr = np.array(x, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if n >= 2 else 0.0
        return mean, std
    rmean, rstd = _mean_std(r_list)
    amean, astd = _mean_std(a_list)
    mmean, mstd = _mean_std(m_list)
    relmean, relstd = _mean_std(rel_list)
    return rmean, amean, mmean, relmean, rstd, astd, mstd, relstd, n

def eval_initial_to_final(model: FNO2d, data: dict, device: torch.device, max_samples: int) -> Tuple[float,float,float,float,float,float,float,float,int]:
    assert 'y' in data, "initial_to_final 数据需包含 'y'"
    n_total = len(data['y'])
    n = n_total if (max_samples is None or max_samples <= 0) else min(max_samples, n_total)
    rmses, maes, mapes, rels = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(n):
            # 用 y 的第0帧作为输入、倒数第二帧作为目标
            x = data['y'][i][..., 0].to(device)      # (H, W)
            y = data['y'][i][..., -2].to(device)     # (H, W)
            pred = model(x.unsqueeze(0).unsqueeze(-1)).squeeze(0).squeeze(-1)
            err = y - pred
            rmses.append(torch.sqrt(torch.mean(err**2)).item())
            maes.append(torch.mean(torch.abs(err)).item())
            denom = torch.clamp(torch.abs(y), min=1e-8)
            mapes.append(torch.mean(torch.abs(err) / denom).item())  # decimal
            rels.append(rel_l2_whole(pred, y))
    return _finalize_err_stats4(rmses, maes, mapes, rels)

def eval_recurrent_fno2d(model: FNO2d, Tin: int, T: int, data: dict, device: torch.device, max_samples: int) -> Tuple[float,float,float,float,float,float,float,float,int]:
    assert 'y' in data, "recurrent 数据需包含 'y'，形状 (H, W, T_total)"
    rp = RecurrentPredictor(model, T_out=T, step=1).to(device)
    n_total = len(data['y'])
    n = n_total if (max_samples is None or max_samples <= 0) else min(max_samples, n_total)
    rmses, maes, mapes, rels = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(n):
            sample = data['y'][i].to(device)  # (H, W, T_total)
            assert sample.ndim == 3 and sample.shape[-1] >= Tin + T, f"样本 {i} 的时间长度不足 Tin+T"
            current_input = sample[None, ..., :Tin]           # (1,H,W,Tin)
            gt = sample[None, ..., Tin:Tin+T]                 # (1,H,W,T)
            pred = rp(current_input)                          # (1,H,W,T)
            gt_last = gt[..., -1].squeeze(0)
            pred_last = pred[..., -1].squeeze(0)
            err_last = gt_last - pred_last
            rmses.append(torch.sqrt(torch.mean(err_last**2)).item())
            maes.append(torch.mean(torch.abs(err_last)).item())
            denom = torch.clamp(torch.abs(gt_last), min=1e-8)
            mapes.append(torch.mean(torch.abs(err_last) / denom).item())  # decimal
            rels.append(rel_l2_whole(pred_last, gt_last))
    return _finalize_err_stats4(rmses, maes, mapes, rels)

def eval_fno3d(model: FNO3d, Tin: int, T: int, data: dict, device: torch.device, max_samples: int) -> Tuple[float,float,float,float,float,float,float,float,int]:
    assert 'y' in data, "FNO3d 数据需包含 'y'，形状 (H,W,T_total)"
    n_total = len(data['y'])
    n = n_total if (max_samples is None or max_samples <= 0) else min(max_samples, n_total)
    rmses, maes, mapes, rels = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(n):
            sample = data['y'][i].to(device)
            assert sample.ndim == 3 and sample.shape[-1] >= Tin + T, f"样本 {i} 的时间长度不足 Tin+T"
            current_input = sample[None, ..., :Tin]            # (1,H,W,Tin)
            gt = sample[None, ..., Tin:Tin+T]                  # (1,H,W,T)
            B, H, W, _ = current_input.shape
            # 把时间作为第三个空间维度，长度 T 的重复
            pred = model(current_input.reshape(1, H, W, 1, Tin).repeat(1, 1, 1, T, 1))
            if pred.shape[-1] == 1:
                pred_last = pred[..., -1, 0]
            else:
                pred_last = pred[..., -1]
            gt_last = gt[..., -1].squeeze(0)
            pred_last2d = pred_last.squeeze(0)
            err_last = gt_last - pred_last2d
            rmses.append(torch.sqrt(torch.mean(err_last**2)).item())
            maes.append(torch.mean(torch.abs(err_last)).item())
            denom = torch.clamp(torch.abs(gt_last), min=1e-8)
            mapes.append(torch.mean(torch.abs(err_last) / denom).item())  # decimal
            rels.append(rel_l2_whole(pred_last2d, gt_last))
    return _finalize_err_stats4(rmses, maes, mapes, rels)

# --------------------------------------------------------------------------------------
# Detection & loaders
# --------------------------------------------------------------------------------------
def detect_family(model_path: Path) -> str:
    s = str(model_path).lower()
    if "2d_ns_fno3d" in s or "ns_3d_fno" in s or "/3d/" in s:
        return "FNO3d"
    if "initial_to_final" in s or "input_frame" in s:
        return "FNO2d_initial_to_final"
    return "FNO2d_recurrent"

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

def build_fno2d(modes1: int, modes2: int, width: int, in_channels: Optional[int] = None) -> FNO2d:
    # 兼容不同构造签名（如果你的 FNO2d 需要 in_channels）
    try:
        if in_channels is not None:
            return FNO2d(modes1=modes1, modes2=modes2, width=width, in_channels=in_channels)
        return FNO2d(modes1=modes1, modes2=modes2, width=width)
    except TypeError:
        # 老版不支持 in_channels
        return FNO2d(modes1=modes1, modes2=modes2, width=width)

def build_fno3d(m1: int, m2: int, m3: int, width: int, Tin: int, padding: Optional[int]) -> FNO3d:
    if FNO3d is None:
        raise RuntimeError("FNO3d class not available in your import path.")
    # 兼容不同构造签名
    kwargs = dict(modes1=m1, modes2=m2, modes3=m3, width=width, in_channels=Tin)
    if padding is not None:
        kwargs["padding"] = padding
    try:
        return FNO3d(**kwargs)
    except TypeError:
        kwargs.pop("padding", None)
        return FNO3d(**kwargs)

def load_checkpoint(model_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch_load_compat(model_path, map_location=device)
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("state_dict","model_state_dict")):
        return ckpt.get("state_dict", ckpt.get("model_state_dict"))
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt
    elif hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
    else:
        raise RuntimeError(f"Unrecognized checkpoint format: {model_path}")

def load_model_for_eval(model_path: Path, device: torch.device,
                        defaults: argparse.Namespace) -> Tuple[str, torch.nn.Module, Dict[str,str], Dict[str,int]]:
    """
    Returns: (family, model(nn.Module), meta(dict[str,str]), resolved_hps(dict[str,int]))
    family ∈ {'FNO2d_recurrent','FNO2d_initial_to_final','FNO3d'}
    """
    family = detect_family(model_path)
    meta = enrich_model_meta_extra(model_path, parse_model_info_from_path(model_path))
    # defaults fallback
    meta.setdefault("modes1", str(defaults.modes1))
    meta.setdefault("modes2", str(defaults.modes2))
    meta.setdefault("width",  str(defaults.width))
    if family != "FNO2d_initial_to_final":
        meta.setdefault("Tin",   str(defaults.t_in))
        meta.setdefault("Tout",  str(defaults.t_out))
    meta.setdefault("step",  str(defaults.step))

    # build & shape-safe load
    sd = load_checkpoint(model_path, device)

    if family == "FNO3d":
        inferred = _infer_arch_from_state_dict_3d(sd)
        m1 = int(meta.get("modes1") or (inferred[0] if inferred else defaults.modes1))
        m2 = int(meta.get("modes2") or (inferred[1] if inferred else defaults.modes2))
        m3 = int(meta.get("modes3") or (inferred[2] if inferred else 6))
        width = int(meta.get("width") or (inferred[3] if inferred else defaults.width))
        Tin = int(meta.get("Tin") or defaults.t_in)
        Tout = int(meta.get("Tout") or defaults.t_out)
        padding = None
        s = str(model_path).lower()
        if "padding" in s:
            m_pad = re.search(r"padding[_=]?(\d+)", s)
            padding = int(m_pad.group(1)) if m_pad else 6
        model = build_fno3d(m1, m2, m3, width, Tin, padding).to(device)
        loaded, missing, skipped = _shape_safe_load(model, sd)
        meta["resolved_modes1"] = str(m1)
        meta["resolved_modes2"] = str(m2)
        meta["resolved_modes3"] = str(m3)
        meta["resolved_width"]  = str(width)
        meta["resolved_Tin"]    = str(Tin)
        meta["resolved_Tout"]   = str(Tout)
        return family, model, meta, {"modes1":m1,"modes2":m2,"modes3":m3,"width":width,"Tin":Tin,"Tout":Tout}

    # FNO2d (recurrent or initial_to_final)
    inferred2d = _infer_arch_from_state_dict_2d(sd)
    m1 = int(meta.get("modes1") or (inferred2d[0] if inferred2d else defaults.modes1))
    m2 = int(meta.get("modes2") or (inferred2d[1] if inferred2d else defaults.modes2))
    width = int(meta.get("width")  or (inferred2d[2] if inferred2d else defaults.width))

    if family == "FNO2d_initial_to_final":
        model = build_fno2d(m1, m2, width, in_channels=1).to(device)
        loaded, missing, skipped = _shape_safe_load(model, sd)
        meta["resolved_modes1"] = str(m1)
        meta["resolved_modes2"] = str(m2)
        meta["resolved_width"]  = str(width)
        return family, model, meta, {"modes1":m1,"modes2":m2,"width":width}

    # recurrent
    Tin = int(meta.get("Tin") or defaults.t_in)
    Tout = int(meta.get("Tout") or defaults.t_out)
    # FNO2d 的 in_channels 常常等于 Tin
    model = build_fno2d(m1, m2, width, in_channels=Tin).to(device)
    loaded, missing, skipped = _shape_safe_load(model, sd)
    meta["resolved_modes1"] = str(m1)
    meta["resolved_modes2"] = str(m2)
    meta["resolved_width"]  = str(width)
    meta["resolved_Tin"]    = str(Tin)
    meta["resolved_Tout"]   = str(Tout)
    return family, model, meta, {"modes1":m1,"modes2":m2,"width":width,"Tin":Tin,"Tout":Tout}

# --------------------------------------------------------------------------------------
# Pretty overview printers (GROUPED & NUMBERED)
# --------------------------------------------------------------------------------------
def _print_grouped_numbered(title: str, groups: Dict[str, List[Tuple[str, Path]]], tsv_path: Path):
    print("\n" + "=" * 80)
    print(f"[INFO] {title}")
    print("=" * 80)
    for grp in sorted(groups.keys()):
        items = groups[grp]
        print(f"\n[{grp}]  count={len(items)}")
        for idx, (name, path) in enumerate(items, start=1):
            print(f"  [{idx}] {name}    {path}")
    # write tsv
    try:
        with open(tsv_path, "w", encoding="utf-8", newline="") as f:
            f.write("group\tindex\tname\tpath\n")
            for grp in sorted(groups.keys()):
                for idx, (name, path) in enumerate(groups[grp], start=1):
                    f.write(f"{grp}\t{idx}\t{name}\t{path}\n")
        print(f"\n[SAVED] {tsv_path}\n")
    except Exception as e:
        print(f"[WARN] Failed to write {tsv_path.name}: {e}")

def print_models_overview_grouped(models: List[Tuple[str, Path]], out_root: Path):
    """
    models: list of (group, path)
    group ∈ {'baseline','FNO2d_initial_to_final','FNO2d_recurrent','FNO3d'}
    """
    groups: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for grp, p in models:
        groups[grp].append((p.name, p.resolve()))
    _print_grouped_numbered("MODELS FOUND (grouped & numbered)", groups, out_root / "models_overview.tsv")

def print_datasets_overview_grouped(datasets_single: List[Tuple[str, Path]],
                                    expanded_files: List[Path],
                                    gen_files: List[Path],
                                    out_root: Path):
    groups: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for dg, p in datasets_single:
        groups[dg].append((p.stem, p.resolve()))
    if expanded_files:
        for p in expanded_files:
            groups["expanded"].append((p.stem, p.resolve()))
    if gen_files:
        for p in gen_files:
            groups["generalizability"].append((p.stem, p.resolve()))
    _print_grouped_numbered("DATASETS FOUND (grouped & numbered)", groups, out_root / "datasets_overview.tsv")

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluator for FNO2d/FNO3d models over train/test/expanded/gen datasets."
    )
    # Datasets
    parser.add_argument("--train_pt", type=str, default=None)
    parser.add_argument("--test_pt", type=str, default=None)
    parser.add_argument("--expanded_dir", type=str, default=None)   # folder of .pt
    parser.add_argument("--expanded_pt", type=str, default=None)    # compat: single .pt or dir
    parser.add_argument("--gen_dir", type=str, default=None)        # folder of .pt

    # Models (baseline + many dirs)
    parser.add_argument("--baseline_model", type=str, required=True)
    parser.add_argument("--model_dir", action="append", default=None,
                        help="Can be repeated multiple times or comma-separated; each will be recursively searched for *.pth")
    # Fallback hparams (used when not parsable)
    parser.add_argument("--modes1", type=int, default=64)
    parser.add_argument("--modes2", type=int, default=64)
    parser.add_argument("--width",  type=int, default=60)
    parser.add_argument("--t_in",   type=int, default=10)
    parser.add_argument("--t_out",  type=int, default=10)
    parser.add_argument("--step",   type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    parser.add_argument("--max_samples", type=int, default=0, help="<=0 use all samples; otherwise limit per dataset")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument(
        "--exclude_re", action="append", default=None,
        help="命中任一正则的模型会被排除。可重复多次。"
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if (args.device.lower() == "auto" and torch.cuda.is_available())
                          else (args.device if args.device.lower() != "auto" else "cpu"))
    print(f"[INFO] Device: {device}")

    # 输出目录
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    out_root = Path(args.output_root).expanduser().resolve() if args.output_root else (here / "combined_eval_results").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Collect datasets ---
    datasets_single: List[Tuple[str, Path]] = []
    def push_file(label: str, p: Optional[str]):
        if p:
            pp = Path(p).expanduser().resolve()
            if not pp.exists() or not pp.is_file():
                raise FileNotFoundError(f"{label} file not found: {pp}")
            datasets_single.append((label, pp))
    push_file("train", args.train_pt)
    push_file("test", args.test_pt)

    # Expanded: folder or single file (compat)
    expanded_dir: Optional[Path] = None
    if args.expanded_dir:
        expanded_dir = Path(args.expanded_dir).expanduser().resolve()
        if not expanded_dir.exists() or not expanded_dir.is_dir():
            raise FileNotFoundError(f"expanded_dir not found: {expanded_dir}")
    elif args.expanded_pt:
        ep = Path(args.expanded_pt).expanduser().resolve()
        if ep.is_dir():
            expanded_dir = ep
        elif not ep.exists():
            raise FileNotFoundError(f"expanded_pt not found: {ep}")

    expanded_files: List[Path] = []
    if expanded_dir:
        expanded_files = sorted([p for p in expanded_dir.glob("*.pt") if p.is_file()])
        if not expanded_files:
            print(f"[WARN] No .pt files under {expanded_dir} for expanded.")
    elif args.expanded_pt:
        ep = Path(args.expanded_pt).expanduser().resolve()
        if ep.is_file():
            expanded_files = [ep]

    # Generalizability folder
    gen_files: List[Path] = []
    if args.gen_dir:
        gen_dir = Path(args.gen_dir).expanduser().resolve()
        if not gen_dir.exists() or not gen_dir.is_dir():
            raise FileNotFoundError(f"generalizability dir not found: {gen_dir}")
        gen_files = sorted([p for p in gen_dir.glob("*.pt") if p.is_file()])
        if not gen_files:
            print(f"[WARN] No .pt files under {gen_dir} for generalizability.")

    if not datasets_single and not expanded_files and not gen_files:
        raise ValueError("Provide at least one dataset via --train_pt/--test_pt/--expanded_dir/--expanded_pt/--gen_dir.")

    # --- Collect models: baseline + recursive under many roots ---
    baseline_path = Path(args.baseline_model).expanduser().resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline model not found: {baseline_path}")
    models: List[Tuple[str, Path]] = [("baseline", baseline_path)]

    model_roots: List[Path] = []
    if args.model_dir:
        # support comma-separated inside each repeat
        for ent in args.model_dir:
            if ent is None: continue
            for seg in str(ent).split(","):
                seg = seg.strip()
                if not seg: continue
                model_roots.append(Path(seg).expanduser().resolve())

    folder_models = find_models_recursively(model_roots) if model_roots else []
    folder_models = [m for m in folder_models if m != baseline_path]
    # assign family group for pretty printing
    for m in folder_models:
        fam = detect_family(m)
        models.append((fam, m))

    exclude_res = _compile_regex_list(args.exclude_re)
    before_cnt = len(models)
    models = apply_model_filters(models, exclude_res)
    after_cnt = len(models)
    print(f"[FILTER] models kept: {after_cnt}/{before_cnt} "
    f"(exclude_re={len(exclude_res)})")

    print(f"[INFO] Found {len(folder_models)} models under {len(model_roots)} roots (recursive)")

    # === NEW: print grouped & numbered DATASETS and MODELS at startup ===
    print_datasets_overview_grouped(datasets_single, expanded_files, gen_files, out_root)
    print_models_overview_grouped(models, out_root)

    total_models = len(models)
    total_train_test = len(datasets_single)
    total_expanded = len(expanded_files)
    total_gen = len(gen_files)
    print("\n" + "="*80)
    print(f"[PLAN] Total models: {total_models} | datasets: train/test={total_train_test}, expanded={total_expanded}, generalizability={total_gen}")
    print("="*80)

    # ---- Build dataset catalog list (for JSON catalog) ----
    all_datasets_for_catalog: List[Dict[str, Any]] = []
    for label, p in datasets_single:
        all_datasets_for_catalog.append({"group": label, "path": str(p), "name": p.stem, "parsed": parse_dataset_name(p.stem)})
    for p in expanded_files:
        all_datasets_for_catalog.append({"group": "expanded", "path": str(p), "name": p.stem, "parsed": parse_dataset_name(p.stem)})
    for p in gen_files:
        all_datasets_for_catalog.append({"group": "generalizability", "path": str(p), "name": p.stem, "parsed": parse_dataset_name(p.stem)})

    # CSV header (means/stds with MAPE as decimal, and REL L2)
    header = [
        "dataset_group","dataset_name","dataset_path","n",
        "rmse_mean","mae_mean","mape_mean","rel_mean",
        "rmse_std","mae_std","mape_std","rel_std",
        "model_family","model_group","model_name","model_path","model_source_dir",
    ] + DATASET_KEYS + MODEL_KEYS + ["resolved_modes1","resolved_modes2","resolved_modes3","resolved_width","resolved_Tin","resolved_Tout"]

    # CSV path
    model_dir_tag = "_".join(sorted({r.name for r in model_roots})) if model_roots else "no_model_dir"
    csv_path = out_root / f"combined_eval__roots_{model_dir_tag}.csv"
    print(f"[INFO] Writing CSV → {csv_path}")

    # ===== Read completed pairs to resume =====
    completed: set[tuple[str, str]] = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header_row = next(reader, None)
            if header_row:
                col_idx = {name.strip(): i for i, name in enumerate(header_row)}
                if "model_path" in col_idx and "dataset_path" in col_idx:
                    for row in reader:
                        if not row or len(row) <= max(col_idx.values()):
                            continue
                        completed.add((row[col_idx["model_path"]].strip(),
                                       row[col_idx["dataset_path"]].strip()))
        print(f"[RESUME] Existing CSV with {len(completed)} completed (model,dataset) pairs; will skip them.")

    # Open CSV (append mode)
    write_header = not csv_path.exists() or (csv_path.stat().st_size == 0)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(header)
        csv_file.flush()

    def write_eval_row(row: List[Any]) -> None:
        writer.writerow(row)
        csv_file.flush()

    # ---- helper: find matched expanded for folder model (if any) ----
    def model_has_pending_pairs(model_path: Path, model_group: str, model_meta: Dict[str, str]) -> Tuple[bool, Optional[Path]]:
        match: Optional[Path] = None
        if model_group not in ("baseline","FNO2d_initial_to_final","FNO3d"):
            pass
        if model_group.startswith("FNO2d") and "expanded_stem" in model_meta:
            expanded_stem = model_meta["expanded_stem"]
            for p in expanded_files:
                if p.stem == expanded_stem:
                    match = p; break
            if match is None and expanded_dir:
                for p in expanded_dir.rglob("*.pt"):
                    if p.stem == expanded_stem:
                        match = p; break

        candidates: List[Path] = []
        candidates += [p for _, p in datasets_single]
        candidates += gen_files
        if (model_group == "baseline") or (("expanded_stem" not in model_meta) and model_group != "FNO2d_initial_to_final"):
            candidates += expanded_files
        else:
            if match is not None:
                candidates.append(match)

        for p in candidates:
            if (str(model_path), str(p)) not in completed:
                return True, match
        return False, match

    # ---- JSON catalog scaffold ----
    json_catalog = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "args": vars(args),
        "output_csv": str(csv_path),
        "models": [],
        "datasets": all_datasets_for_catalog,
    }

    # ===== Evaluate (streaming write + resume) =====
    total_pairs_before = len(completed)

    for model_idx, (model_group, model_path) in enumerate(models, start=1):
        # parse + enrich meta
        model_meta = enrich_model_meta_extra(model_path, parse_model_info_from_path(model_path))
        # family detection
        family = detect_family(model_path)
        # fallback defaults
        for k, default in ("modes1", str(args.modes1)), ("modes2", str(args.modes2)), ("width", str(args.width)), ("Tin", str(args.t_in)), ("Tout", str(args.t_out)), ("step", str(args.step)):
            model_meta.setdefault(k, default)

        # Save to catalog
        json_catalog["models"].append({
            "group": model_group,
            "family": family,
            "name": model_path.stem,
            "path": str(model_path),
            "parent": str(model_path.parent),
            "meta": model_meta
        })

        # early skip if everything done
        has_pending, matched_expanded = model_has_pending_pairs(model_path, model_group, model_meta)
        if not has_pending:
            print("\n" + "-"*90)
            print(f"[MODEL {model_idx}/{total_models}] {model_path.name}  (group={model_group} family={family})")
            print(f"   source_dir: {model_path.parent}")
            print("   pending=0 → already completed, skipping.")
            continue

        # candidates in fixed order
        ordered_cands: List[Tuple[str, Path]] = []
        if (model_group == "baseline") or (family in ("FNO3d","FNO2d_initial_to_final")):
            ordered_cands += [(dg, p) for (dg, p) in datasets_single]
            ordered_cands += [("expanded", p) for p in expanded_files]
            ordered_cands += [("generalizability", p) for p in gen_files]
        else:
            # FNO2d recurrent expanded-folder model：expanded 优先用 matched
            if matched_expanded is not None:
                ordered_cands.append(("expanded", matched_expanded))
            ordered_cands += [(dg, p) for (dg, p) in datasets_single]
            ordered_cands += [("generalizability", p) for p in gen_files]

        pending_pairs = [(dg, p) for (dg, p) in ordered_cands if (str(model_path), str(p)) not in completed]
        pending_total = len(pending_pairs)

        # print start
        print("\n" + "-"*90)
        print(f"[MODEL {model_idx}/{total_models}] {model_path.name}  (group={model_group} family={family})")
        print(f"   source_dir: {model_path.parent}")
        print(f"   datasets_pending={pending_total}")

        # Load model lazily + resolve hps
        t_model_load = perf_counter()
        try:
            fam, model, meta_resolved, hps = load_model_for_eval(model_path, device, args)
        except Exception as e:
            print(f"[ERROR] Failed to build/load model: {e}")
            continue
        resolved_m1 = int(meta_resolved.get("resolved_modes1", meta_resolved.get("modes1", args.modes1)))
        resolved_m2 = int(meta_resolved.get("resolved_modes2", meta_resolved.get("modes2", args.modes2)))
        resolved_m3 = int(meta_resolved.get("resolved_modes3", 0))
        resolved_w  = int(meta_resolved.get("resolved_width", args.width)) if isinstance(meta_resolved, dict) else int(model_meta["width"])
        resolved_Tin  = int(meta_resolved.get("resolved_Tin", model_meta.get("Tin", args.t_in)))
        resolved_Tout = int(meta_resolved.get("resolved_Tout", model_meta.get("Tout", args.t_out)))
        print(f"   [LOADED] family={fam} modes({resolved_m1},{resolved_m2}{','+str(resolved_m3) if resolved_m3 else ''}) width={resolved_w} Tin={resolved_Tin if fam!='FNO2d_initial_to_final' else 'N/A'} Tout={resolved_Tout if fam!='FNO2d_initial_to_final' else 'N/A'} in {perf_counter()-t_model_load:.2f}s")

        # helper: eval one dataset & write row
        ds_counter = 0
        def _eval_and_write(dg: str, p: Path):
            nonlocal ds_counter
            pair = (str(model_path), str(p))
            if pair in completed:
                return
            ds_counter += 1
            print(f"    [DATA {ds_counter}/{pending_total}] {dg}: {p.name} ...")
            t_ds = perf_counter()
            data_obj = load_data_pt(p, device)
            if data_obj is None:
                return
            try:
                if fam == "FNO2d_initial_to_final":
                    rmean, amean, mmean, relmean, rstd, astd, mstd, relstd, N = eval_initial_to_final(model, data_obj, device, args.max_samples)
                elif fam == "FNO3d":
                    rmean, amean, mmean, relmean, rstd, astd, mstd, relstd, N = eval_fno3d(model, resolved_Tin, resolved_Tout, data_obj, device, args.max_samples)
                else:
                    rmean, amean, mmean, relmean, rstd, astd, mstd, relstd, N = eval_recurrent_fno2d(model, resolved_Tin, resolved_Tout, data_obj, device, args.max_samples)
            except AssertionError as ae:
                print(f"      └ [SKIP] {dg}:{p.name} → {ae}")
                return
            except Exception as e:
                print(f"      └ [ERROR] {dg}:{p.name} → {e}")
                return
            elapsed = perf_counter() - t_ds
            parsed_ds = parse_dataset_name(p.stem)
            row = [
                dg, p.stem, str(p), N,
                rmean, amean, mmean, relmean,
                rstd, astd, mstd, relstd,
                fam, model_group, model_path.stem, str(model_path), str(model_path.parent),
            ] + [parsed_ds.get(k, "") for k in DATASET_KEYS] + [model_meta.get(k, "") for k in MODEL_KEYS] + [
                resolved_m1, resolved_m2, (resolved_m3 if resolved_m3 else ""), resolved_w,
                (resolved_Tin if fam!="FNO2d_initial_to_final" else ""), (resolved_Tout if fam!="FNO2d_initial_to_final" else "")
            ]
            write_eval_row(row)
            completed.add(pair)
            print(f"      └ done in {elapsed:.2f}s | N={N} | "
                  f"RMSE μ={pretty_float(rmean)} (σ={pretty_float(rstd)}) | "
                  f"MAE  μ={pretty_float(amean)} (σ={pretty_float(astd)}) | "
                  f"MAPE μ={pretty_float(mmean)} (σ={pretty_float(mstd)}) | "
                  f"REL  μ={pretty_float(relmean)} (σ={pretty_float(relstd)})")

        for (dg, p) in ordered_cands:
            _eval_and_write(dg, p)

        print(f"[MODEL {model_idx}] completed | datasets_evaluated={ds_counter}/{pending_total}")

    csv_file.close()
    print("\n" + "="*80)
    print(f"[SAVED] {csv_path} (appended) | new_pairs_added={len(completed)-total_pairs_before}")

    # ---- Write JSON catalog next to CSV ----
    json_path = csv_path.with_suffix(".catalog.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_catalog, f, ensure_ascii=False, indent=2)
        print(f"[SAVED] {json_path}")
    except Exception as e:
        print(f"[WARN] Failed to write JSON catalog: {e}")

if __name__ == "__main__":
    main()





