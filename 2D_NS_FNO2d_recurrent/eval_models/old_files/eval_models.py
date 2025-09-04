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

import numpy as np
import torch
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Project imports (assume this file sits somewhere under project tree)
# --------------------------------------------------------------------------------------

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

def load_model(model_path: Path, device: torch.device, modes1: Optional[int], modes2: Optional[int], width: Optional[int],
               t_out: int, step: int) -> Tuple[FNO2d, RecurrentPredictor]:
    t0 = perf_counter()
    ckpt = torch_load_compat(model_path, map_location=device)
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("state_dict", "model_state_dict")):
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict"))
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state_dict = ckpt
    elif hasattr(ckpt, "state_dict"):
        state_dict = ckpt.state_dict()
    else:
        raise RuntimeError(f"Unrecognized checkpoint format: {model_path}")

    m1 = int(modes1) if modes1 else 64
    m2 = int(modes2) if modes2 else m1
    wd = int(width) if width else 60

    fno = FNO2d(modes1=m1, modes2=m2, width=wd).to(device)
    missing, unexpected = fno.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict(strict=False) -> missing: {missing}, unexpected: {unexpected}")
    fno.eval()
    rec = RecurrentPredictor(fno, T_out=t_out, step=step).to(device)
    print(f"[MODEL] Loaded {model_path.name} in {perf_counter()-t0:.3f}s → modes({m1},{m2}) width={wd} T_out={t_out} step={step}")
    return fno, rec

def eval_dataset_file(pt_path: Path, rec: RecurrentPredictor, device: torch.device, t_in: int, t_out: int,
                      show_progress: bool = True) -> Tuple[int, float, float, float, float, float, float]:
    """Return (N, rmse_mean, rmse_std, mae_mean, mae_std, mape_mean, mape_std)."""
    t_load = perf_counter()
    data_obj = torch_load_compat(pt_path, map_location="cpu")
    seq = ensure_batched_nhwt(get_sequence_tensor(data_obj))
    N, H, W, T_total = seq.shape
    if t_in + t_out > T_total:
        raise RuntimeError(f"T_in({t_in})+T_out({t_out}) > T_total({T_total}) for {pt_path}")
    print(f"      [DATASET] {pt_path.name} | shape=(N={N}, H={H}, W={W}, T={T_total}) | load={perf_counter()-t_load:.3f}s")

    rm, ma, mp = OnlineStats(), OnlineStats(), OnlineStats()
    it = range(N)
    pbar = tqdm(it, desc=f"      infer {pt_path.name}", leave=False, disable=not show_progress)
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
            pbar.set_postfix({
                "rmse": f"{rm.mean:.4f}",
                "mae": f"{ma.mean:.4f}",
                "mape%": f"{mp.mean:.2f}",
                "eta": pbar.format_dict.get('remaining_s', 0.0)
            })
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

    # 提前准备输出目录（后续要写 discovered_models 清单）
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
        # 去掉与 baseline 相同的路径
        folder_models = [m for m in folder_models if m != baseline_path]
        models += [("folder", m) for m in folder_models]
        print(f"[INFO] Found {len(folder_models)} models under {model_root} (recursive)")

    print(f"[INFO] Total models to evaluate: {len(models)} (including baseline)")

    # 打印并落盘所有发现的模型路径
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
        # 若没找到模型，给出一些 .pt 示例帮助排查（仅当提供了 model_root 且其存在）
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

    # Begin evaluation
    rows: List[List[Any]] = []

    for (model_group, model_path) in tqdm(models, desc="Models", position=0):
        # Parse & enrich model metadata
        model_meta = enrich_model_meta_extra(model_path, parse_model_info_from_path(model_path))
        # 默认值回填
        for k, default in ("modes1", str(args.modes1)), ("modes2", str(args.modes2)), ("width", str(args.width)), ("Tin", str(args.t_in)), ("Tout", str(args.t_out)), ("step", str(args.step)):
            model_meta.setdefault(k, default)

        # Load model
        t_model = perf_counter()
        _, rec = load_model(model_path, device,
                            int(model_meta.get("modes1", args.modes1)),
                            int(model_meta.get("modes2", model_meta.get("modes1", args.modes2))),
                            int(model_meta.get("width", args.width)),
                            args.t_out, args.step)
        print(f"[INFO] Model ready in {perf_counter()-t_model:.3f}s: {model_path}")

        if model_group == "baseline":
            # 1) train/test
            for (dg, p) in tqdm(datasets_single, desc="Datasets(single)", position=1, leave=False):
                try:
                    N, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(p, rec, device, args.t_in, args.t_out)
                except Exception as e:
                    print(f"[ERROR] Skipping {p} for model {model_path.name}: {e}")
                    continue
                parsed_ds = parse_dataset_name(p.stem)
                rows.append([
                    dg, p.stem, str(p), N,
                    rmean, rstd, amean, astd, mmean, mstd,
                    model_group, model_path.stem, str(model_path), str(model_path.parent),
                ] + [parsed_ds.get(k, "") for k in DATASET_KEYS] + [model_meta.get(k, "") for k in MODEL_KEYS])

            # 2) expanded（baseline 对 expanded_dir 里的文件全部评估）
            for pt_path in tqdm(expanded_files, desc="Datasets(expanded)", position=1, leave=False):
                try:
                    N, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(pt_path, rec, device, args.t_in, args.t_out)
                except Exception as e:
                    print(f"[ERROR] Skipping {pt_path} for model {model_path.name}: {e}")
                    continue
                parsed_ds = parse_dataset_name(pt_path.stem)
                rows.append([
                    "expanded", pt_path.stem, str(pt_path), N,
                    rmean, rstd, amean, astd, mmean, mstd,
                    model_group, model_path.stem, str(model_path), str(model_path.parent),
                ] + [parsed_ds.get(k, "") for k in DATASET_KEYS] + [model_meta.get(k, "") for k in MODEL_KEYS])

            # 3) generalizability
            for pt_path in tqdm(gen_files, desc="Datasets(generalizability)", position=1, leave=False):
                try:
                    N, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(pt_path, rec, device, args.t_in, args.t_out)
                except Exception as e:
                    print(f"[ERROR] Skipping {pt_path} for model {model_path.name}: {e}")
                    continue
                parsed_ds = parse_dataset_name(pt_path.stem)
                rows.append([
                    "generalizability", pt_path.stem, str(pt_path), N,
                    rmean, rstd, amean, astd, mmean, mstd,
                    model_group, model_path.stem, str(model_path), str(model_path.parent),
                ] + [parsed_ds.get(k, "") for k in DATASET_KEYS] + [model_meta.get(k, "") for k in MODEL_KEYS])

        else:
            # Folder models：expanded 仅评与该模型匹配的那个 .pt，同时也要在 train / test / generalizability 上评
            expanded_stem = model_meta.get("expanded_stem", "")
            if not expanded_stem:
                print(f"[WARN] {model_path.name}: no expanded_stem parsed; skipping expanded evaluation for this model.")
            else:
                match: Optional[Path] = None
                for p in expanded_files:
                    if p.stem == expanded_stem:
                        match = p
                        break
                # 备用：在 expanded_dir 递归找
                if match is None and expanded_dir:
                    for p in expanded_dir.rglob("*.pt"):
                        if p.stem == expanded_stem:
                            match = p
                            break
                if match is None:
                    print(f"[WARN] {model_path.name}: no matching expanded file for stem '{expanded_stem}' under expanded_dir; expanded skipped.")
                else:
                    try:
                        N, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(match, rec, device, args.t_in, args.t_out)
                    except Exception as e:
                        print(f"[ERROR] Skipping {match} for model {model_path.name}: {e}")
                    else:
                        parsed_ds = parse_dataset_name(match.stem)
                        rows.append([
                            "expanded", match.stem, str(match), N,
                            rmean, rstd, amean, astd, mmean, mstd,
                            model_group, model_path.stem, str(model_path), str(model_path.parent),
                        ] + [parsed_ds.get(k, "") for k in DATASET_KEYS] + [model_meta.get(k, "") for k in MODEL_KEYS])

            # 追加：train / test
            for (dg, p) in tqdm(datasets_single, desc="Datasets(single, folder)", position=1, leave=False):
                try:
                    N, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(p, rec, device, args.t_in, args.t_out)
                except Exception as e:
                    print(f"[ERROR] Skipping {p} for model {model_path.name}: {e}")
                    continue
                parsed_ds = parse_dataset_name(p.stem)
                rows.append([
                    dg, p.stem, str(p), N,
                    rmean, rstd, amean, astd, mmean, mstd,
                    model_group, model_path.stem, str(model_path), str(model_path.parent),
                ] + [parsed_ds.get(k, "") for k in DATASET_KEYS] + [model_meta.get(k, "") for k in MODEL_KEYS])

            # 追加：generalizability
            for pt_path in tqdm(gen_files, desc="Datasets(generalizability, folder)", position=1, leave=False):
                try:
                    N, rmean, rstd, amean, astd, mmean, mstd = eval_dataset_file(pt_path, rec, device, args.t_in, args.t_out)
                except Exception as e:
                    print(f"[ERROR] Skipping {pt_path} for model {model_path.name}: {e}")
                    continue
                parsed_ds = parse_dataset_name(pt_path.stem)
                rows.append([
                    "generalizability", pt_path.stem, str(pt_path), N,
                    rmean, rstd, amean, astd, mmean, mstd,
                    model_group, model_path.stem, str(model_path), str(model_path.parent),
                ] + [parsed_ds.get(k, "") for k in DATASET_KEYS] + [model_meta.get(k, "") for k in MODEL_KEYS])

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[SAVED] {csv_path} | rows={len(rows)}")

if __name__ == "__main__":
    main()


