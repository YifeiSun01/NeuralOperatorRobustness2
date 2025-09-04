#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-evaluate TWO sets of models on FOUR dataset inputs and write ONE combined CSV (no plots).

Datasets (any subset; provide paths via CLI):
  • --train_pt       : single .pt file (train)
  • --test_pt        : single .pt file (test)
  • --expanded_dir   : FOLDER of EXPANDED .pt files
  • --expanded_pt    : [Compat] If dir → treated like --expanded_dir; if file → single expanded file
  • --gen_dir        : folder of many .pt files (generalizability)

Models (both can be used together):
  • --baseline_model : single checkpoint file (.pth or .pt)
  • --model_dir      : folder to search recursively for ALL model checkpoints (.pth/.pt)

For EACH (model_file × dataset_file) combination, compute metrics over ALL samples in that dataset file,
and write one CSV row containing:
  - dataset_group (train/test/expanded/generalizability)
  - dataset_name, dataset_path, N (samples)
  - rmse_mean, rmse_std, mae_mean, mae_std, mape_mean, mape_std (computed on last frame)
  - parsed dataset metadata columns (nx, N, nu, t, ntimepoints, length_scale, variance, alpha, tau, period, exp_factor, vmax, vmin, type, solver)
  - model_group (baseline or folder), model_name, model_path, model_source_dir
  - parsed model metadata columns (modes1, modes2, width, Tin, Tout, step, epochs, seed)
  - extra model metadata from training layout/manifest/logs (strategy, pct_tag, percent, expanded_stem, expanded_file_name, expanded_file_path, orig_train_dataset, with_replacement, k_requested, N_in_file)

The script prints detailed progress with tqdm (models → datasets → samples) and timing/ETA information.
"""

import os
import sys
import re
import csv
import math
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple
from time import perf_counter
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Project imports (assume this file sits somewhere under project tree)
# --------------------------------------------------------------------------------------

def print_models_overview(
    models: List[Tuple[str, Path]],
    model_root: Optional[Path],
    out_root: Path,
    title: str = "MODELS TO EVALUATE"
) -> None:
    """
    清晰打印准备测评的模型清单（baseline + folder），并落盘到 out_root：
      - 表格视图（每行：GROUP / MODEL_NAME / PARENT_DIR / FULL_PATH）
      - 按文件夹分组视图（列出每个目录下的模型名）
      - 另外单独打印 .pth 清单（常用于权重）
      - 写出 models_overview.tsv 与 models_overview_grouped.txt
    """
    # 规范化列表
    items = [(grp, p.resolve()) for grp, p in models]
    if not items:
        print("[INFO] No models to evaluate.")
        return

    # ---- 1) 表格视图（扁平） ----
    print("\n" + "=" * 80)
    print(f"[INFO] {title}")
    print("=" * 80)

    header = f"{'GROUP':9} {'MODEL_NAME':40} {'PARENT_DIR':35} {'FULL_PATH'}"
    print(header)
    print("-" * len(header))
    for grp, path in items:
        model_name  = path.name[:40]
        parent_name = path.parent.name[:35]
        print(f"{grp:9} {model_name:40} {parent_name:35} {str(path)}")

    # ---- 2) 按目录分组视图（仅 folder 组）----
    folder_items = [(g, p) for g, p in items if g == "folder"]
    if folder_items:
        by_dir: Dict[Path, List[Path]] = defaultdict(list)
        for _, p in folder_items:
            by_dir[p.parent].append(p)

        print("\n[INFO] Grouped by parent directory (folder models):")
        for d in sorted(by_dir.keys()):
            print(f"  [DIR] {d}  ({len(by_dir[d])} models)")
            for p in sorted(by_dir[d]):
                print(f"    - {p.name}")

    # ---- 3) 单独打印 .pth 清单 ----
    pth_only = [p for _, p in items if p.suffix.lower() == ".pth"]
    print("\n[INFO] .pth models:")
    if pth_only:
        for i, p in enumerate(sorted(set(pth_only)), 1):
            print(f"   [PTH-{i:04d}] {p}")
    else:
        print("   (none)")

    # ---- 4) 落盘（TSV & 文本）----
    tsv_path = out_root / "models_overview.tsv"
    txt_grouped_path = out_root / "models_overview_grouped.txt"
    try:
        with open(tsv_path, "w", encoding="utf-8", newline="") as f:
            f.write("group\tmodel_name\tparent_dir\tfull_path\n")
            for grp, path in items:
                f.write(f"{grp}\t{path.name}\t{path.parent}\t{path}\n")
        with open(txt_grouped_path, "w", encoding="utf-8") as f:
            if folder_items:
                for d in sorted(by_dir.keys()):
                    f.write(f"[DIR] {d}  ({len(by_dir[d])} models)\n")
                    for p in sorted(by_dir[d]):
                        f.write(f"  - {p.name}\n")
            else:
                f.write("(no folder models)\n")
        print(f"\n[SAVED] {tsv_path}")
        print(f"[SAVED] {txt_grouped_path}\n")
    except Exception as e:
        print(f"[WARN] Failed to write overview files: {e}")

def add_project_root(levels_up: int = 3):
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    project_root = here
    for _ in range(levels_up):
        project_root = project_root.parent
    sys.path.append(str(project_root))

add_project_root(levels_up=3)

# 额外把上级目录加进来, 以便 from models.FNO2d import ...
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO2d import FNO2d, RecurrentPredictor  # noqa: E402

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def torch_load_compat(path: Path, map_location=None):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)

def find_models_recursively(root: Path) -> List[Path]:
    """递归查找 root 下所有可能的模型文件 (.pth/.pt)。
    - .pth 一律视为模型
    - .pt：若文件名包含 {model, ckpt, checkpoint, state_dict} 视为模型；
           若以 _all_frames 结尾（明显像数据集），则跳过；
           其他 .pt 为避免漏检，默认当模型
    """
    if (not root) or (not root.exists()) or (not root.is_dir()):
        return []

    candidates: List[Path] = []
    for pat in ("**/*.pth", "**/*.pt"):
        candidates.extend(p for p in root.rglob(pat) if p.is_file())

    models: List[Path] = []
    for p in candidates:
        s = p.stem.lower()
        if p.suffix.lower() == ".pth":
            models.append(p)
            continue
        # .pt 的判定
        if any(tok in s for tok in ("model", "ckpt", "checkpoint", "state_dict")):
            models.append(p)
            continue
        if s.endswith("_all_frames"):
            continue
        models.append(p)

    # 去重 + 排序
    models = sorted(dict.fromkeys(models))
    return models

def get_sequence_tensor(data: Dict[str, Any]) -> torch.Tensor:
    for key in ("y", "x"):
        if key in data and torch.is_tensor(data[key]):
            return data[key]
    for v in data.values():
        if torch.is_tensor(v) and v.dim() >= 3:
            return v
    raise ValueError("Could not find a suitable sequence tensor in the dataset (looked for 'y' or 'x').")

def ensure_batched_nhwt(seq: torch.Tensor) -> torch.Tensor:
    if seq.dim() == 3:   # (H, W, T) -> (1, H, W, T)
        return seq.unsqueeze(0)
    if seq.dim() == 4:   # (N, H, W, T)
        return seq
    raise ValueError(f"Expected 3D or 4D tensor for sequence, got {seq.shape}")

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def mape(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.where(np.abs(a) < eps, eps, np.abs(a))
    return float(np.mean(np.abs((b - a) / denom)) * 100.0)

def pretty_float(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return str(x)
    return f"{x:.6g}"

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
        "nu":           r"_nu([-\d\.]+)",
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
        if m:
            info[k] = m.group(1)
    m = re.search(r"solver=([^_]+)", stem)
    if m:
        info["solver"] = m.group(1)
    type_str = None
    m = re.search(r"ntimepoints\d+_(.*)", stem)
    if m:
        tail = m.group(1)
        m2 = re.match(r"([A-Za-z]+)(?:_([A-Za-z]+))?", tail)
        if m2:
            type_str = m2.group(1) if not m2.group(2) else f"{m2.group(1)}_{m2.group(2)}"
    if type_str:
        info["type"] = type_str
    return info

MODEL_KEYS_BASE = ["modes1","modes2","width","Tin","Tout","step","epochs","seed"]
MODEL_KEYS_EXTRA = [
    "strategy", "pct_tag", "percent",
    "expanded_stem", "expanded_file_name", "expanded_file_path",
    "orig_train_dataset",
    "with_replacement", "k_requested", "N_in_file",
]
MODEL_KEYS = MODEL_KEYS_BASE + MODEL_KEYS_EXTRA

def parse_model_info_from_path(path: Path) -> Dict[str, str]:
    """从文件名或父目录解析模型超参。"""
    s = str(path)
    info: Dict[str, str] = {}

    # modes1/modes2: 允许 "modes64x64" / "modes64_64" / "modes64"
    m = re.search(r"modes(?:1)?[_=]?(\d+)(?:[xX_](\d+))?", s)
    if m:
        info["modes1"] = m.group(1)
        if m.group(2):
            info["modes2"] = m.group(2)
    # width
    m = re.search(r"width[_=]?(\d+)", s)
    if m:
        info["width"] = m.group(1)
    # Tin / Tout
    m = re.search(r"Tin[_=]?(\d+)", s)
    if m:
        info["Tin"] = m.group(1)
    m = re.search(r"T(?:out|_out)?[_=]?(\d+)(?![\.\d])", s)  # 避免匹配 t20.0（数据集参数）
    if m:
        info["Tout"] = m.group(1)
    # step, epochs, seed
    for key in ("step", "epochs", "seed"):
        m = re.search(fr"{key}[_=]?(\d+)", s)
        if m:
            info[key] = m.group(1)
    return info

def enrich_model_meta_extra(path: Path, meta: Dict[str, str]) -> Dict[str, str]:
    """从目录结构/边侧文件补充训练策略与扩展数据集信息。"""
    info = dict(meta) if meta else {}

    # 训练策略（最近的 add/replace 父目录）
    parts = list(path.resolve().parts)
    strategy = None
    for p in reversed(parts[:-1]):
        lp = p.lower()
        if lp == "add" or lp == "replace":
            strategy = lp
            break
    if strategy:
        info["strategy"] = strategy

    # pct_tag 和 expanded_stem：形如 .../<expanded_stem>/<XXpct>/(add|replace)/...
    pct_tag = None
    expanded_stem = None
    for i, seg in enumerate(parts):
        ls = seg.lower()
        if ls.endswith("pct") and ls[:-3].isdigit():
            pct_tag = seg
            if i - 1 >= 0:
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

class OnlineStats:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    @property
    def var(self) -> float:
        if self.n <= 1:
            return 0.0
        return self.M2 / self.n
    @property
    def std(self) -> float:
        return float(math.sqrt(self.var))

# --------------------------------------------------------------------------------------
# Model loading & evaluation
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Robust model loading: infer arch from checkpoint shapes + shape-safe partial load
# --------------------------------------------------------------------------------------

def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop leading 'module.' prefixes if present (from DataParallel/DDP)."""
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def _infer_arch_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[Tuple[int, int, int]]:
    """
    Try to infer (modes1, modes2, width) from any 4D spectral weight tensor.
    Many FNO2d impls store weights like conv_layers.i.weights{1,2} with shape [width, width, m1, m2].
    """
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 4:
            # Heuristic: [width, width, modes1, modes2]
            w0, w1, m1, m2 = v.shape
            if w0 == w1 and m1 > 0 and m2 > 0:
                return int(m1), int(m2), int(w0)
    return None

def _shape_safe_load(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str], List[str]]:
    """
    Load only parameters whose shapes match exactly; return (loaded, missing, skipped_due_to_shape).
    """
    model_sd = model.state_dict()
    loaded_keys, skipped_shape, unexpected = [], [], []

    # Strip module. prefix if any
    sd = _strip_module_prefix(state_dict)

    # Filter by both key presence and exact same shape
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

    # Actually load
    model.load_state_dict(compat, strict=False)

    # Compute truly missing (present in model but not loaded)
    missing = [k for k in model_sd.keys() if k not in loaded_keys]
    return loaded_keys, missing, skipped_shape

def load_model(model_path: Path, device: torch.device,
               modes1: Optional[int], modes2: Optional[int], width: Optional[int],
               t_out: int, step: int) -> Tuple[FNO2d, RecurrentPredictor]:
    t0 = perf_counter()
    ckpt = torch_load_compat(model_path, map_location=device)

    # Get raw state_dict in a few common formats
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("state_dict", "model_state_dict")):
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict"))
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state_dict = ckpt
    elif hasattr(ckpt, "state_dict"):
        state_dict = ckpt.state_dict()
    else:
        raise RuntimeError(f"Unrecognized checkpoint format: {model_path}")

    # 1) Try infer from checkpoint shapes (highest priority)
    inferred = _infer_arch_from_state_dict(state_dict)

    # 2) Decide final arch: checkpoint-shape > provided values (filename/CLI)
    build_m1 = int(inferred[0]) if inferred else int(modes1) if modes1 is not None else 64
    build_m2 = int(inferred[1]) if inferred else int(modes2) if modes2 is not None else build_m1
    build_w  = int(inferred[2]) if inferred else int(width)  if width  is not None else 60
    if inferred:
        print(f"[AUTO] Inferred from ckpt: modes=({build_m1},{build_m2}), width={build_w}")

    # 3) Build model with the decided arch
    fno = FNO2d(modes1=build_m1, modes2=build_m2, width=build_w).to(device)

    # 4) Shape-safe load (avoids size-mismatch crashes)
    loaded, missing, skipped_shape = _shape_safe_load(fno, state_dict)
    if skipped_shape:
        print(f"[WARN] Skipped {len(skipped_shape)} params due to shape mismatch (likely wrong modes/width).")
    if missing:
        # Often buffers like running stats or non-critical keys; keep it informational
        print(f"[INFO] Missing in checkpoint: {len(missing)} keys (ok if not critical).")

    fno.eval()
    rec = RecurrentPredictor(fno, T_out=t_out, step=step).to(device)
    print(f"[MODEL] Loaded {model_path.name} in {perf_counter()-t0:.3f}s → modes({build_m1},{build_m2}) width={build_w} T_out={t_out} step={step}")
    return fno, rec

def eval_dataset_file(pt_path: Path, rec: RecurrentPredictor, device: torch.device, t_in: int, t_out: int,
                      show_progress: bool = False, quiet: bool = True) -> Tuple[int, float, float, float, float, float, float]:
    """Return (N, rmse_mean, rmse_std, mae_mean, mae_std, mape_mean, mape_std)."""
    t_load = perf_counter()
    data_obj = torch_load_compat(pt_path, map_location="cpu")
    seq = ensure_batched_nhwt(get_sequence_tensor(data_obj))
    N, H, W, T_total = seq.shape
    if t_in + t_out > T_total:
        raise RuntimeError(f"T_in({t_in})+T_out({t_out}) > T_total({T_total}) for {pt_path}")

    if not quiet:
        print(f"      [DATASET] {pt_path.name} | shape=(N={N}, H={H}, W={W}, T={T_total}) | load={perf_counter()-t_load:.3f}s")

    rm, ma, mp = OnlineStats(), OnlineStats(), OnlineStats()
    it = range(N)
    # 关闭进度条或只在非 quiet 情况下显示
    pbar = it if (quiet or not show_progress) else tqdm(it, desc=f"      infer {pt_path.name}", leave=False)
    t0 = perf_counter()
    with torch.no_grad():
        for idx in pbar:
            sample = seq[idx]  # (H,W,T)
            current_input  = sample[..., :t_in].unsqueeze(0).to(device)
            ground_truth   = sample[..., t_in:t_in+t_out].unsqueeze(0)
            pred = rec(current_input)

            gt_last = ground_truth[0, ..., -1].detach().cpu().numpy()
            pr_last = pred[0, ..., -1].detach().cpu().numpy()

            r = rmse(gt_last, pr_last)
            a = mae(gt_last, pr_last)
            m = mape(gt_last, pr_last)

            rm.add(r); ma.add(a); mp.add(m)

    if not quiet:
        elapsed = perf_counter()-t0
        print(f"      [RESULT] {pt_path.name} | N={N} | RMSE μ={rm.mean:.6g} σ={rm.std:.6g} | MAE μ={ma.mean:.6g} σ={ma.std:.6g} | MAPE μ={mp.mean:.6g}% σ={mp.std:.6g} | eval={elapsed:.3f}s")
    return N, rm.mean, rm.std, ma.mean, ma.std, mp.mean, mp.std


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TWO sets of models on FOUR dataset inputs; write ONE combined CSV with per-(model,dataset) stats."
        )
    )
    # Datasets
    parser.add_argument("--train_pt", type=str, default=None, help="Path to TRAIN .pt file")
    parser.add_argument("--test_pt", type=str, default=None, help="Path to TEST .pt file")
    parser.add_argument("--expanded_dir", type=str, default=None, help="FOLDER of EXPANDED .pt files")
    parser.add_argument("--expanded_pt", type=str, default=None, help="[Compat] If dir → treated like --expanded_dir; if file → single expanded file")
    parser.add_argument("--gen_dir", type=str, default=None, help="Folder containing many .pt files for GENERALIZABILITY")

    # Models
    parser.add_argument("--baseline_model", type=str, required=True, help="Path to single baseline checkpoint (.pth/.pt)")
    parser.add_argument("--model_dir", type=str, default=None, help="Folder to recursively search for model checkpoints")

    # Model arch/runtime (fallbacks if not parsable from path)
    parser.add_argument("--modes1", type=int, default=64)
    parser.add_argument("--modes2", type=int, default=64)
    parser.add_argument("--width",  type=int, default=60)
    parser.add_argument("--t_in",   type=int, default=10)
    parser.add_argument("--t_out",  type=int, default=10)
    parser.add_argument("--step",   type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")

    parser.add_argument("--output_root", type=str, default=None, help="Folder to write the combined CSV (default ./combined_eval_results)")

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if (args.device.lower() == "auto" and torch.cuda.is_available()) else (args.device if args.device.lower() != "auto" else "cpu"))
    print(f"[INFO] Device: {device}")

    # 输出目录
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    out_root = Path(args.output_root).expanduser().resolve() if args.output_root else (here / "combined_eval_results").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Collect datasets ---
    datasets_single: List[Tuple[str, Path]] = []  # train/test only
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

    # Models: baseline + recursive folder models (dedup if overlap)
    baseline_path = Path(args.baseline_model).expanduser().resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline model not found: {baseline_path}")
    models: List[Tuple[str, Path]] = [("baseline", baseline_path)]

    folder_models: List[Path] = []
    model_root: Optional[Path] = None
    if args.model_dir:
        model_root = Path(args.model_dir).expanduser().resolve()
        folder_models = find_models_recursively(model_root)
        folder_models = [m for m in folder_models if m != baseline_path]  # 去重 baseline
        models += [("folder", m) for m in folder_models]
        print(f"[INFO] Found {len(folder_models)} models under {model_root} (recursive)")

    print(f"[INFO] Total models to evaluate: {len(models)} (including baseline)")

    # 列出发现的模型并落盘
    if folder_models:
        print(f"[INFO] Listing discovered models under {model_root}:")
        for i, m in enumerate(folder_models, 1):
            print(f"   [MODEL-{i:04d}] {m}")

        discovered_txt = out_root / f"discovered_models__{model_root.name}.txt"
        try:
            with open(discovered_txt, "w") as f:
                for m in folder_models:
                    f.write(str(m) + "\n")
            print(f"[INFO] Wrote model list → {discovered_txt}")
        except Exception as e:
            print(f"[WARN] Could not write model list: {e}")
    else:
        if model_root and model_root.exists():
            peek_pts = list(model_root.rglob("*.pt"))[:10]
            if peek_pts:
                print("[DEBUG] Found .pt files that were skipped (first 10):")
                for p in peek_pts:
                    print(f"   [SKIPPED-PT] {p}")

    # CSV header
    header = [
        "dataset_group","dataset_name","dataset_path","n",
        "rmse_mean","rmse_std","mae_mean","mae_std","mape_mean","mape_std",
        "model_group","model_name","model_path","model_source_dir",
    ] + DATASET_KEYS + MODEL_KEYS

    # CSV path
    baseline_tag = baseline_path.stem[:60]
    model_dir_tag = model_root.name if model_root else "no_model_dir"
    csv_path = out_root / f"combined_eval__baseline_{baseline_tag}__set_{model_dir_tag}.csv"
    print(f"[INFO] Writing CSV → {csv_path}")

    # ===== 读取已完成的 (model_path, dataset_path) 组合 =====
    completed: set[tuple[str, str]] = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header_row = next(reader, None)
            if header_row:
                col_idx = {name.strip(): i for i, name in enumerate(header_row)}
                if "model_path" in col_idx and "dataset_path" in col_idx:
                    for row in reader:
                        # 防御：空行或列数不足
                        if not row or len(row) <= max(col_idx.values()):
                            continue
                        completed.add((row[col_idx["model_path"]].strip(),
                                       row[col_idx["dataset_path"]].strip()))
        print(f"[RESUME] Found existing CSV with {len(completed)} completed pairs; will skip them.")

    # ===== 打开 CSV（追加模式），必要时写表头 =====
    write_header = not csv_path.exists() or (csv_path.stat().st_size == 0)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(header)
        csv_file.flush()

    def write_eval_row(row: List[Any]) -> None:
        writer.writerow(row)
        csv_file.flush()

    # ===== 工具：判断模型是否还有待评估的组合（用于模型级早跳过） =====
    def model_has_pending_pairs(model_path: Path, model_group: str, model_meta: Dict[str, str]) -> Tuple[bool, Optional[Path]]:
        match: Optional[Path] = None
        if model_group != "baseline":
            expanded_stem = model_meta.get("expanded_stem", "")
            if expanded_stem:
                for p in expanded_files:
                    if p.stem == expanded_stem:
                        match = p
                        break
                if match is None and expanded_dir:
                    for p in expanded_dir.rglob("*.pt"):
                        if p.stem == expanded_stem:
                            match = p
                            break

        candidates: List[Path] = []
        candidates += [p for _, p in datasets_single]
        candidates += gen_files
        if model_group == "baseline":
            candidates += expanded_files
        else:
            if match is not None:
                candidates.append(match)

        for p in candidates:
            if (str(model_path), str(p)) not in completed:
                return True, match
        return False, match

    # ===== 开始评测（随算随写 + 跳过已完成） =====
    for (model_group, model_path) in tqdm(models, desc="Models", position=0):
        # 解析 + 补充模型元信息
        model_meta = enrich_model_meta_extra(model_path, parse_model_info_from_path(model_path))
        for k, default in ("modes1", str(args.modes1)), ("modes2", str(args.modes2)), ("width", str(args.width)), ("Tin", str(args.t_in)), ("Tout", str(args.t_out)), ("step", str(args.step)):
            model_meta.setdefault(k, default)

        # 模型级早跳过
        has_pending, matched_expanded = model_has_pending_pairs(model_path, model_group, model_meta)
        if not has_pending:
            print(f"[SKIP MODEL] All pairs already in CSV for: {model_path.name}")
            continue

        # 真需要时才加载模型
        t_model = perf_counter()
        _, rec = load_model(
            model_path, device,
            int(model_meta.get("modes1", args.modes1)),
            int(model_meta.get("modes2", model_meta.get("modes1", args.modes2))),
            int(model_meta.get("width",  args.width)),
            args.t_out, args.step
        )
        print(f"[INFO] Model ready in {perf_counter()-t_model:.3f}s: {model_path}")

        if model_group == "baseline":
            # 1) train/test
            for (dg, p) in tqdm(datasets_single, desc="Datasets(single)", position=1, leave=False):
                pair = (str(model_path), str(p))
                if pair in completed:
                    continue
                try:
                    N, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(p, rec, device, args.t_in, args.t_out)
                except Exception as e:
                    print(f"[ERROR] Skipping {p} for model {model_path.name}: {e}")
                    continue
                parsed_ds = parse_dataset_name(p.stem)
                row = [
                    dg, p.stem, str(p), N,
                    rmean, rstd, amean, astd, mmean, mstd,
                    model_group, model_path.stem, str(model_path), str(model_path.parent),
                ] + [parsed_ds.get(k, "") for k in DATASET_KEYS] + [model_meta.get(k, "") for k in MODEL_KEYS]
                write_eval_row(row)
                completed.add(pair)

            # 2) expanded（baseline 评估所有 expanded 文件）
            for pt_path in tqdm(expanded_files, desc="Datasets(expanded)", position=1, leave=False):
                pair = (str(model_path), str(pt_path))
                if pair in completed:
                    continue
                try:
                    N, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(pt_path, rec, device, args.t_in, args.t_out)
                except Exception as e:
                    print(f"[ERROR] Skipping {pt_path} for model {model_path.name}: {e}")
                    continue
                parsed_ds = parse_dataset_name(pt_path.stem)
                row = [
                    "expanded", pt_path.stem, str(pt_path), N,
                    rmean, rstd, amean, astd, mmean, mstd,
                    model_group, model_path.stem, str(model_path), str(model_path.parent),
                ] + [parsed_ds.get(k, "") for k in DATASET_KEYS] + [model_meta.get(k, "") for k in MODEL_KEYS]
                write_eval_row(row)
                completed.add(pair)

            # 3) generalizability
            for pt_path in tqdm(gen_files, desc="Datasets(generalizability)", position=1, leave=False):
                pair = (str(model_path), str(pt_path))
                if pair in completed:
                    continue
                try:
                    N, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(pt_path, rec, device, args.t_in, args.t_out)
                except Exception as e:
                    print(f"[ERROR] Skipping {pt_path} for model {model_path.name}: {e}")
                    continue
                parsed_ds = parse_dataset_name(pt_path.stem)
                row = [
                    "generalizability", pt_path.stem, str(pt_path), N,
                    rmean, rstd, amean, astd, mmean, mstd,
                    model_group, model_path.stem, str(model_path), str(model_path.parent),
                ] + [parsed_ds.get(k, "") for k in DATASET_KEYS] + [model_meta.get(k, "") for k in MODEL_KEYS]
                write_eval_row(row)
                completed.add(pair)

        else:
            # Folder 组：只评与该模型匹配到的 expanded 文件（若有），并在 train/test/gen 上评
            if matched_expanded is None:
                print(f"[WARN] {model_path.name}: no matching expanded file found; expanded skipped for this model.")

            # expanded（仅匹配的一个）
            if matched_expanded is not None:
                pair = (str(model_path), str(matched_expanded))
                if pair not in completed:
                    try:
                        N, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(matched_expanded, rec, device, args.t_in, args.t_out)
                    except Exception as e:
                        print(f"[ERROR] Skipping {matched_expanded} for model {model_path.name}: {e}")
                    else:
                        parsed_ds = parse_dataset_name(matched_expanded.stem)
                        row = [
                            "expanded", matched_expanded.stem, str(matched_expanded), N,
                            rmean, rstd, amean, astd, mmean, mstd,
                            model_group, model_path.stem, str(model_path), str(model_path.parent),
                        ] + [parsed_ds.get(k, "") for k in DATASET_KEYS] + [model_meta.get(k, "") for k in MODEL_KEYS]
                        write_eval_row(row)
                        completed.add(pair)

            # train / test
            for (dg, p) in tqdm(datasets_single, desc="Datasets(single, folder)", position=1, leave=False):
                pair = (str(model_path), str(p))
                if pair in completed:
                    continue
                try:
                    N, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(p, rec, device, args.t_in, args.t_out)
                except Exception as e:
                    print(f"[ERROR] Skipping {p} for model {model_path.name}: {e}")
                    continue
                parsed_ds = parse_dataset_name(p.stem)
                row = [
                    dg, p.stem, str(p), N,
                    rmean, rstd, amean, astd, mmean, mstd,
                    model_group, model_path.stem, str(model_path), str(model_path.parent),
                ] + [parsed_ds.get(k, "") for k in DATASET_KEYS] + [model_meta.get(k, "") for k in MODEL_KEYS]
                write_eval_row(row)
                completed.add(pair)

            # generalizability
            for pt_path in tqdm(gen_files, desc="Datasets(generalizability, folder)", position=1, leave=False):
                pair = (str(model_path), str(pt_path))
                if pair in completed:
                    continue
                try:
                    N, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(pt_path, rec, device, args.t_in, args.t_out)
                except Exception as e:
                    print(f"[ERROR] Skipping {pt_path} for model {model_path.name}: {e}")
                    continue
                parsed_ds = parse_dataset_name(pt_path.stem)
                row = [
                    "generalizability", pt_path.stem, str(pt_path), N,
                    rmean, rstd, amean, astd, mmean, mstd,
                    model_group, model_path.stem, str(model_path), str(model_path.parent),
                ] + [parsed_ds.get(k, "") for k in DATASET_KEYS] + [model_meta.get(k, "") for k in MODEL_KEYS]
                write_eval_row(row)
                completed.add(pair)

    csv_file.close()
    print(f"[SAVED] {csv_path} (appended) | total_completed_pairs_now={len(completed)}")


if __name__ == "__main__":
    main()


