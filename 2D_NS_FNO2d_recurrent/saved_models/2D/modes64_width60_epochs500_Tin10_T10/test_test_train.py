#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import csv
import re
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Project imports
# --------------------------------------------------------------------------------------

def add_project_root(levels_up: int = 3):
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    project_root = here
    for _ in range(levels_up):
        project_root = project_root.parent
    sys.path.append(str(project_root))

add_project_root(levels_up=3)

from models.FNO2d import FNO2d, RecurrentPredictor  # noqa: E402

# --------------------------------------------------------------------------------------
# Defaults & utilities
# --------------------------------------------------------------------------------------

def default_data_dir() -> str:
    base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    return str((base / "../../../datasets/exponax_datasets/t20").resolve())


def torch_load_compat(path: Path, map_location=None):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def find_single_pth_in_dir(directory: Path) -> Path:
    pths = sorted(directory.glob("*.pth"))
    if len(pths) == 0:
        raise FileNotFoundError(f"No .pth model file found in {directory}")
    if len(pths) > 1:
        raise RuntimeError(f"Expected exactly one .pth in {directory}, found {len(pths)}: {pths}")
    return pths[0]


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
    # mean(|(b-a)/a|) * 100; add eps to denom to avoid div-by-zero spikes
    denom = np.where(np.abs(a) < eps, eps, np.abs(a))
    return float(np.mean(np.abs((b - a) / denom)) * 100.0)


def pretty_float(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return str(x)
    return f"{x:.6g}"
# ------------------------ filename parser (robust-ish) ------------------------
KEY_TOKENS = ("length_scale", "variance", "alpha", "exp_factor", "vmax", "vmin", "batch", "all")

def parse_dataset_name(stem: str) -> Dict[str, str]:
    """
    解析数据集文件名中的关键参数，新增支持 split=train/test/val。
    例：
      dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames
      dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames
      ..._GRF_matern_alpha1.500_tau7_vmax-1_vmin-1.5_...
      ..._GRF_rbf_length_scale0.020_variance0.500_...
    """
    info: Dict[str, str] = {}

    # 统一用正则抓取键值
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

    # solver
    m = re.search(r"solver=([^_]+)", stem)
    if m:
        info["solver"] = m.group(1)

    # split：支持 *_train_... / *_test_... / *_val_...（通常在 t 后面）
    m = re.search(r"_(train|test|val)(?:_[^_]*)?$", stem, flags=re.IGNORECASE)
    if m:
        info["split"] = m.group(1).lower()

    # type（最多两段，如 GRF_matern / LogGRF_rq / NegLogGRF_periodic），旧数据里可能存在
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


def short_info_for_title(info: Dict[str, str]) -> str:
    # 把 split 提到比较靠前的位置，便于标题区分
    keys_in_order = [
        "split", "type", "solver", "nx", "N", "nu", "t", "ntimepoints",
        "length_scale", "variance", "alpha", "exp_factor", "vmax", "vmin"
    ]
    parts = []
    for k in keys_in_order:
        if k in info and str(info[k]) != "":
            parts.append(f"{k}={info[k]}")
    return " | ".join(parts)

def sanitize(s: str) -> str: 
    # safe for filenames 
    s = re.sub(r"[^\w\-.]+", "_", s) 
    return s[:200]

def build_unique_png_name(dataset_stem: str, idx: int, parsed: Dict[str, str]) -> str:
    import hashlib

    # 依据 type/或 split 选择前缀；没有 type 时就用 split（train/test/val）
    t = parsed.get("type")
    if not t:
        t = parsed.get("split", "meta")
    t_lower = t.lower()

    def pick(keys):
        out = []
        for k in keys:
            v = parsed.get(k)
            if v is not None and str(v) != "":
                out.append(f"{k}{v}")
        return out

    fields = []
    # 保留一些可选的核参数（旧命名里可能出现）
    if "periodic" in t_lower:
        fields += pick(["length_scale", "period"])
    elif "matern" in t_lower:
        fields += pick(["alpha", "tau"])
    elif "rbf" in t_lower or "gaussian" in t_lower:
        fields += pick(["length_scale", "variance"])
    elif "rq" in t_lower:
        fields += pick(["length_scale", "variance", "alpha"])

    # LogGRF / NegLogGRF 系列额外携带 exp_factor
    if "loggrf" in t_lower:
        fields += pick(["exp_factor"])

    # 振幅范围（如果文件名里有）
    fields += pick(["vmax", "vmin"])

    base = sanitize("_".join([t] + fields)) if fields else sanitize(t)

    # 用哈希保证唯一
    token = hashlib.md5(f"{dataset_stem}|{idx}|{base}".encode("utf-8")).hexdigest()[:12]
    suffix = f"__h{token}.png"

    MAX_FILENAME = 220
    base = base[: max(1, MAX_FILENAME - len(suffix))]

    return base + suffix

# ----------------------------- plotting helper --------------------------------


def plot_sample(
        inp_first: np.ndarray,
        gt_last: np.ndarray,
        pr_last: np.ndarray,
        metrics: Dict[str, float],
        dataset_name: str,
        dataset_info_str: str,
        idx: int,
        out_png: Path,
        dpi: int = 150,
    ) -> None:
    diff = gt_last - pr_last

    # First row uses shared ranges (like before)
    vmin = float(np.nanmin([inp_first.min(), gt_last.min(), pr_last.min()]))
    vmax = float(np.nanmax([inp_first.max(), gt_last.max(), pr_last.max()]))
    dmax = float(np.nanmax(np.abs(diff)))
    dvmin, dvmax = -dmax, dmax

    # 2 rows x 4 cols: row1 = shared ranges; row2 = auto ranges (no vmin/vmax)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=False)
    (ax0, ax1, ax2, ax3) = axes[0]
    (ax4, ax5, ax6, ax7) = axes[1]

    # Row 1 — shared ranges
    im0 = ax0.imshow(inp_first, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    ax0.set_title("Input (frame 0)")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(gt_last, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    ax1.set_title("Ground Truth (last frame)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(pr_last, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    ax2.set_title("Prediction (last frame)")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    im3 = ax3.imshow(diff, origin="lower", vmin=dvmin, vmax=dvmax, cmap="coolwarm")
    ax3.set_title("Difference (GT - Pred)")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Row 2 — auto ranges (independent scales, no vmin/vmax)
    im4 = ax4.imshow(inp_first, origin="lower", cmap="viridis")
    ax4.set_title("Input (auto range)")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    im5 = ax5.imshow(gt_last, origin="lower", cmap="viridis")
    ax5.set_title("Ground Truth (auto range)")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    im6 = ax6.imshow(pr_last, origin="lower", cmap="viridis")
    ax6.set_title("Prediction (auto range)")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    im7 = ax7.imshow(diff, origin="lower", cmap="coolwarm")
    ax7.set_title("Difference (auto range)")
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    rmse_v = metrics.get("rmse", float("nan"))
    mae_v = metrics.get("mae", float("nan"))
    mape_v = metrics.get("mape", float("nan"))

    title_line1 = f"Dataset: {dataset_name} | Index: {idx} | RMSE={rmse_v:.4f}  MAE={mae_v:.4f}  MAPE={mape_v:.2f}%"
    title_line2 = dataset_info_str
    fig.suptitle(title_line1 + ("\n" + title_line2 if title_line2 else ""), fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main evaluation (streaming keep+plot to avoid OOM and to print as we go)
# --------------------------------------------------------------------------------------


def main():
    # ---- 计时工具（放在 main 里免改全局 import）
    from time import perf_counter

    def dur_s(t0):
        return f"{perf_counter() - t0:.3f}s"

    t_script0 = perf_counter()

    # ------------------------- 解析参数 -------------------------
    t0 = perf_counter()
    parser = argparse.ArgumentParser(
        description="Eval FNO2d over multiple .pt datasets; save ONE hierarchical pickle (+ summary CSV + PNG for kept indices)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir(),
        help=("Directory containing dataset .pt files. "
              "Default: ../../../datasets/exponax_datasets/t20 (relative to this .py)."),
    )
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root output dir. Default is ./test_results next to this script.")
    parser.add_argument("--t_in", type=int, default=10, help="Number of input frames.")
    parser.add_argument("--t_out", type=int, default=10, help="Number of output frames to predict.")
    parser.add_argument("--step", type=int, default=1, help="Recurrent step for RecurrentPredictor.")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu | cuda | auto (default auto: cuda if available else cpu)")
    parser.add_argument("--modes1", type=int, default=64)
    parser.add_argument("--modes2", type=int, default=64)
    parser.add_argument("--width", type=int, default=60)
    parser.add_argument("--keep_idx_start", type=int, default=0,
                        help="Inclusive start index to keep in FINAL PICKLE & PLOTS. Default 0.")
    parser.add_argument("--keep_idx_end", type=int, default=0,
                        help="Inclusive end index to keep in FINAL PICKLE & PLOTS. Default 0 (only index 0).")
    args = parser.parse_args()
    print(f"[TIME] argparse: {dur_s(t0)}", flush=True)

    # ------------------------- 目录准备 -------------------------
    t0 = perf_counter()
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve() if args.output_root else (here / "test_train_results").resolve()
    imgs_dir  = out_root / "images"
    pkl_dir   = out_root / "pickle"
    csv_dir   = out_root / "csv"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    pkl_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TIME] make_dirs: {dur_s(t0)}  -> {out_root}", flush=True)

    # ------------------------- 设备/模型加载 -------------------------
    t_model = perf_counter()
    if args.device is None or args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    t0 = perf_counter()
    model_pth = find_single_pth_in_dir(here)
    print(f"[INFO] Using model checkpoint: {model_pth}", flush=True)
    print(f"[TIME] find_checkpoint: {dur_s(t0)}", flush=True)

    t0 = perf_counter()
    ckpt = torch_load_compat(model_pth, map_location=device)
    print(f"[TIME] load_checkpoint: {dur_s(t0)}", flush=True)

    t0 = perf_counter()
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("state_dict", "model_state_dict")):
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict"))
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state_dict = ckpt  # raw state_dict
    elif hasattr(ckpt, "state_dict"):
        print("[WARN] Checkpoint appears to be a full model object; extracting its state_dict.", flush=True)
        state_dict = ckpt.state_dict()
    else:
        raise RuntimeError("Unrecognized checkpoint format; cannot obtain state_dict.")
    print(f"[TIME] parse_state_dict: {dur_s(t0)}", flush=True)

    t0 = perf_counter()
    model = FNO2d(modes1=args.modes1, modes2=args.modes2, width=args.width).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict(strict=False) -> missing keys: {missing}, unexpected: {unexpected}", flush=True)
    model.eval()
    recurrent_model = RecurrentPredictor(model, T_out=args.t_out, step=args.step)
    print(f"[TIME] build_model: {dur_s(t0)}", flush=True)
    print(f"[TIME] model_total: {dur_s(t_model)}", flush=True)

    # ------------------------- 过滤范围 -------------------------
    t0 = perf_counter()
    ks, ke = int(args.keep_idx_start), int(args.keep_idx_end)
    if ke < ks:
        ke = ks
    print(f"[FILTER] Keeping index range [{ks}, {ke}] (inclusive) in the FINAL PICKLE & PLOTS.", flush=True)
    filtered = {
        "model_path": str(model_pth),
        "params": {"T_in": args.t_in, "T_out": args.t_out, "step": args.step},
        "keep_index_range": {"start": ks, "end": ke},
        "datasets_info": {},
        "datasets": {}
    }
    print(f"[TIME] init_structs: {dur_s(t0)}", flush=True)

    # ------------------------- 扫描数据集 -------------------------
    t_scan = perf_counter()
    pt_files = sorted([p for p in data_dir.glob("*.pt") if p.is_file()])
    if not pt_files:
        raise FileNotFoundError(f"No .pt dataset files found in {data_dir}")
    print(f"[INFO] Found {len(pt_files)} dataset file(s) in {data_dir}", flush=True)
    print(f"[INFO] Device: {device}, T_in={args.t_in}, T_out={args.t_out}, step={args.step}", flush=True)
    print("-" * 80, flush=True)
    print(f"[TIME] scan_datasets: {dur_s(t_scan)}", flush=True)

    # For CSV summary (all indices)
    per_dataset_metrics: Dict[str, Dict[str, list]] = {}
    parsed_cache: Dict[str, Dict[str, str]] = {}

    # ------------------------- 主循环：每个数据集 -------------------------
    t_loop_all = perf_counter()
    for fi, pt_path in enumerate(pt_files, 1):
        t_ds = perf_counter()
        print(f"[DATASET {fi}/{len(pt_files)}] Loading: {pt_path}", flush=True)
        dataset_name = pt_path.stem

        # parse name
        t0 = perf_counter()
        parsed = parse_dataset_name(dataset_name)
        parsed_cache[dataset_name] = parsed
        print(f"[TIME][dataset:{fi}] parse_name: {dur_s(t0)}", flush=True)

        # load .pt
        t0 = perf_counter()
        try:
            data_obj = torch_load_compat(pt_path, map_location="cpu")
        except Exception as e:
            print(f"[ERROR] Failed to load {pt_path}: {e}", flush=True)
            print(f"[TIME][dataset:{fi}] load_pt: {dur_s(t0)} (failed)", flush=True)
            continue
        print(f"[TIME][dataset:{fi}] load_pt: {dur_s(t0)}", flush=True)

        # extract tensor
        t0 = perf_counter()
        try:
            seq = get_sequence_tensor(data_obj)
            seq = ensure_batched_nhwt(seq)
        except Exception as e:
            print(f"[ERROR] {pt_path.name}: cannot extract sequence tensor: {e}", flush=True)
            print(f"[TIME][dataset:{fi}] extract_seq: {dur_s(t0)} (failed)", flush=True)
            continue
        print(f"[TIME][dataset:{fi}] extract_seq: {dur_s(t0)}", flush=True)

        # shape check
        N, H, W, T_total = seq.shape
        if args.t_in + args.t_out > T_total:
            print(f"[ERROR] {pt_path.name}: T_in({args.t_in}) + T_out({args.t_out}) > T_total({T_total}). Skipping.", flush=True)
            continue

        filtered["datasets_info"][dataset_name] = {
            "dataset_path": str(pt_path),
            "shape": {"N": N, "H": H, "W": W, "T_total": T_total},
            "parsed_info": parsed
        }
        per_dataset_metrics[dataset_name] = {"rmse": [], "mae": [], "mape": []}

        # --------------------- 子循环：每个 index ---------------------
        for idx in range(N):
            t_idx = perf_counter()
            print(f"  [Index {idx+1}/{N}] ...", end="", flush=True)

            # 准备输入/GT
            t0_prep = perf_counter()
            sample = seq[idx]  # (H, W, T_total)
            current_input = sample[..., :args.t_in].unsqueeze(0).to(device)          # (1,H,W,T_in)
            ground_truth = sample[..., args.t_in:args.t_in+args.t_out].unsqueeze(0)  # (1,H,W,T_out)
            t1_prep = perf_counter()

            # 前向
            t0_fwd = perf_counter()
            with torch.no_grad():
                pred = recurrent_model(current_input)  # (1,H,W,T_out)
            t1_fwd = perf_counter()

            # to numpy + 指标
            t0_np = perf_counter()
            inp_first = current_input[0, ..., 0].detach().cpu().numpy()  # (H,W)
            gt_last  = ground_truth[0, ..., -1].detach().cpu().numpy()   # (H,W)
            pr_last  = pred[0, ..., -1].detach().cpu().numpy()           # (H,W)
            rmse_v = rmse(gt_last, pr_last)
            mae_v  = mae(gt_last, pr_last)
            mape_v = mape(gt_last, pr_last)
            t1_np = perf_counter()

            print(f" RMSE={pretty_float(rmse_v)}  MAE={pretty_float(mae_v)}  MAPE={pretty_float(mape_v)}% | "
                  f"prep={t1_prep - t0_prep:.3f}s fwd={t1_fwd - t0_fwd:.3f}s post={t1_np - t0_np:.3f}s total={dur_s(t_idx)}",
                  flush=True)

            # 收集 CSV 指标
            per_dataset_metrics[dataset_name]["rmse"].append(rmse_v)
            per_dataset_metrics[dataset_name]["mae"].append(mae_v)
            per_dataset_metrics[dataset_name]["mape"].append(mape_v)

            # 命中保留：写入 + 画图（单独计时）
            if ks <= idx <= ke:
                t0_keep = perf_counter()
                filtered["datasets"].setdefault(dataset_name, {})
                filtered["datasets"][dataset_name][idx] = {
                    "input_first_frame": inp_first,
                    "gt_last_frame": gt_last,
                    "pred_last_frame": pr_last,
                    "metrics": {"rmse": rmse_v, "mae": mae_v, "mape": mape_v},
                    "parsed_info": parsed,
                }
                info_str = short_info_for_title(parsed)
                print(
                    f"[PICKLE-KEEP] Dataset={dataset_name} | idx={idx} | "
                    f"RMSE={pretty_float(rmse_v)} MAE={pretty_float(mae_v)} MAPE={pretty_float(mape_v)}% | {info_str}",
                    flush=True,
                )
                t1_keep = perf_counter()

                t0_plot = perf_counter()
                png_name = build_unique_png_name(dataset_name, idx, parsed)
                out_png = imgs_dir / png_name
                plot_sample(
                    inp_first=inp_first,
                    gt_last=gt_last,
                    pr_last=pr_last,
                    metrics={"rmse": rmse_v, "mae": mae_v, "mape": mape_v},
                    dataset_name=dataset_name,
                    dataset_info_str=info_str,
                    idx=idx,
                    out_png=out_png,
                    dpi=150,
                )
                t1_plot = perf_counter()
                print(f"[PLOT] Dataset={dataset_name} | idx={idx} -> {out_png} | keep={t1_keep - t0_keep:.3f}s plot={t1_plot - t0_plot:.3f}s",
                      flush=True)

        # 数据集级别统计 + 耗时
        t0 = perf_counter()
        r = per_dataset_metrics[dataset_name]["rmse"]
        a = per_dataset_metrics[dataset_name]["mae"]
        p = per_dataset_metrics[dataset_name]["mape"]
        print(
            f"  -> Dataset summary: "
            f"RMSE mean={np.mean(r):.6g} var={np.var(r):.6g} | "
            f"MAE mean={np.mean(a):.6g} var={np.var(a):.6g} | "
            f"MAPE mean={np.mean(p):.6g}% var={np.var(p):.6g}",
            flush=True,
        )
        print(f"[TIME][dataset:{fi}] total: {dur_s(t_ds)}", flush=True)
        print("-" * 80, flush=True)

    print(f"[TIME] all_datasets_loop: {dur_s(t_loop_all)}", flush=True)

    # ------------------------- 保存 Pickle -------------------------
    t0 = perf_counter()
    pkl_name = f"ALL_DATASETS__Tin{args.t_in}_Tout{args.t_out}__keep_{ks}-{ke}.pkl"
    pkl_path = pkl_dir / pkl_name
    with open(pkl_path, "wb") as f:
        pickle.dump(filtered, f)
    print(f"[SAVED] {pkl_path}  (datasets: {len(filtered['datasets'])}, kept range: [{ks},{ke}])", flush=True)
    print(f"[TIME] save_pickle: {dur_s(t0)}", flush=True)

    # ------------------------- 保存 CSV -------------------------
    t0 = perf_counter()
    keys_for_csv = ["type","solver","nx","N","nu","t","ntimepoints","length_scale","variance","alpha","exp_factor","vmax","vmin"]
    csv_name = f"SUMMARY__Tin{args.t_in}_Tout{args.t_out}.csv"
    csv_path = csv_dir / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = (["dataset_name","dataset_path","n",
                   "rmse_mean","rmse_var","mae_mean","mae_var","mape_mean","mape_var"]
                  + keys_for_csv)
        writer.writerow(header)
        for dname, m in per_dataset_metrics.items():
            n = len(m["rmse"])
            if n == 0:
                continue
            rmse_mean, rmse_var = float(np.mean(m["rmse"])), float(np.var(m["rmse"]))
            mae_mean,  mae_var  = float(np.mean(m["mae"])),  float(np.var(m["mae"]))
            mape_mean, mape_var = float(np.mean(m["mape"])), float(np.var(m["mape"]))
            dpath = filtered["datasets_info"][dname]["dataset_path"]
            parsed = parsed_cache.get(dname, {})
            row = [dname, dpath, n, rmse_mean, rmse_var, mae_mean, mae_var, mape_mean, mape_var]
            row += [parsed.get(k, "") for k in keys_for_csv]
            writer.writerow(row)
    print(f"[SAVED] {csv_path}", flush=True)
    print(f"[TIME] save_csv: {dur_s(t0)}", flush=True)

    # ------------------------- 全流程耗时 -------------------------
    print(f"[DONE] All datasets processed. total_time={dur_s(t_script0)}", flush=True)

# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()





