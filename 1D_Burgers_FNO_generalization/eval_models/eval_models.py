#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import os
import csv
from pathlib import Path
import subprocess
import re
from datetime import datetime
import json
from collections import Counter
from FNO1d import FNO1d

# ==============================
# 配置区：多来源模型 / 多来源数据集
# ==============================

MODEL_DIRS = [
    "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/1D_Burgers/saved_models_expanded",
]

EXTRA_MODEL_PATHS = [
    "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/1D_Burgers/saved_models/1D/modes16_width64_epochs500/pos/old/unnormalized/burgers_1d_FNO_model_trainedby_dim1d_nx1024_N1500_solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_nu0.0005_t1.0_seed45.pth",
]

DATASET_ROOTS = {
    "generalizability": "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/1D_Burgers_FNO_generalization/datasets",
    "expanded": "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/1D_Burgers/datasets/1D/Burgers/expanded_pos/t1",
    "train_test": "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/1D_Burgers_FNO_generalization/test_train_datasets",
}

BASE_OUTDIR = "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/1D_Burgers_FNO_generalization/eval_models/eval_results"
RESULTS_CSV_NAME = "model_testing_results.csv"
INFO_TXT_WITH_TS = True

# ==== 输入构造 ====
# 重要：本版本 **绝不添加坐标通道**，统一使用单通道 (B,N,1)
# 下面两个变量仅保留占位，不再影响行为
COORD_RANGE_MODE = "0_1"
ADD_COORD_CHANNEL = False

# ==============================
# 设备
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
# 工具函数
# ==============================

def count_special(arr: np.ndarray):
    arr = np.asarray(arr)
    total = arr.size
    nan = int(np.isnan(arr).sum())
    posinf = int(np.isposinf(arr).sum())
    neginf = int(np.isneginf(arr).sum())
    invalid = nan + posinf + neginf
    invalid_pct = (invalid / total * 100.0) if total else 0.0
    return {
        "total": int(total),
        "nan": nan,
        "posinf": posinf,
        "neginf": neginf,
        "invalid": invalid,
        "invalid_pct": invalid_pct,
    }

def safe_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)

    total = y_true.size
    valid_n = int(finite_mask.sum())
    invalid_n = int(total - valid_n)
    invalid_pct = (invalid_n / total * 100.0) if total else 0.0

    if valid_n == 0:
        return {
            "rmse": np.nan, "mae": np.nan, "mape": np.nan,
            "valid_n": valid_n, "invalid_n": invalid_n, "invalid_pct": invalid_pct,
        }

    yt = y_true[finite_mask]
    yp = y_pred[finite_mask]

    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae  = float(np.mean(np.abs(yt - yp)))
    denom = np.maximum(np.abs(yt), epsilon)
    mape = float(np.mean(np.abs((yt - yp) / denom)) * 100.0)

    return {
        "rmse": rmse, "mae": mae, "mape": mape,
        "valid_n": valid_n, "invalid_n": invalid_n, "invalid_pct": invalid_pct,
    }

def infer_modes_width(model_path, checkpoint=None, default_modes=16, default_width=64):
    modes, width = None, None
    if isinstance(checkpoint, dict):
        if 'modes' in checkpoint: modes = int(checkpoint['modes'])
        if 'width' in checkpoint: width = int(checkpoint['width'])
        if modes is None and isinstance(checkpoint.get('config'), dict):
            cfg = checkpoint['config']
            if cfg.get('modes') is not None: modes = int(cfg['modes'])
            if cfg.get('width') is not None: width = int(cfg['width'])
        if modes is None and isinstance(checkpoint.get('model_args'), dict):
            args = checkpoint['model_args']
            if args.get('modes') is not None: modes = int(args['modes'])
            if args.get('width') is not None: width = int(args['width'])
    try:
        cfg_p = Path(model_path).parent / "config.json"
        if cfg_p.exists():
            with open(cfg_p, "r") as f:
                cfg = json.load(f)
                if modes is None and 'modes' in cfg: modes = int(cfg['modes'])
                if width is None and 'width' in cfg: width = int(cfg['width'])
    except Exception:
        pass
    s = str(model_path)
    if modes is None:
        m = re.search(r'modes(\d+)', s)
        if m: modes = int(m.group(1))
    if width is None:
        w = re.search(r'width(\d+)', s)
        if w: width = int(w.group(1))
    return modes or default_modes, width or default_width

def get_gpu_info():
    try:
        partition = os.environ.get("SLURM_JOB_PARTITION", "N/A")
        node_name = os.environ.get("SLURMD_NODENAME", "N/A")
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")
        print("========== GPU Information ==========")
        print(f"Partition: {partition}")
        print(f"Node: {node_name}")
        print(f"GPU IDs: {gpu_ids}")
        try:
            nvidia_smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
                encoding="utf-8"
            ).strip()
            print("GPU Details:")
            print(nvidia_smi)
        except Exception:
            print("nvidia-smi not available")
        print("=====================================")
    except Exception as e:
        print(f"Error getting GPU info: {e}")

def load_model(model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = None
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            modes, width = infer_modes_width(model_path, checkpoint)
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            modes, width = infer_modes_width(model_path, checkpoint)
        elif isinstance(checkpoint, nn.Module):
            try:
                state_dict = checkpoint.state_dict()
            except Exception:
                state_dict = None
            modes, width = infer_modes_width(model_path, None)
        else:
            state_dict = checkpoint
            modes, width = infer_modes_width(model_path, None)

        model = FNO1d(modes=modes, width=width).to(device)
        if isinstance(state_dict, dict):
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:   print(f"[load_model] WARN missing keys: {missing}")
            if unexpected:print(f"[load_model] WARN unexpected keys: {unexpected}")
        else:
            print(f"[load_model] WARN: unsupported checkpoint type for {model_path}, model created with random weights")
        model.eval()
        in_feats = getattr(getattr(model, 'fc0', None), 'in_features', None)
        print(f"[load_model] Loaded: modes={modes}, width={width}, fc0.in_features={in_feats}")
        return model
    except Exception as e:
        print(f"[load_model] Error loading model {model_path}: {e}")
        return None

def collect_models(model_dirs, extra_model_paths):
    found = []
    for d in model_dirs:
        dpath = Path(d).resolve()
        if not dpath.exists():
            print(f"[collect_models] WARN: model dir not found: {dpath}")
            continue
        for p in dpath.rglob("*.pth"):
            found.append(("dir", str(p)))
    for p in extra_model_paths:
        pth = Path(p).resolve()
        if pth.exists() and pth.suffix == ".pth":
            found.append(("extras", str(pth)))
        else:
            print(f"[collect_models] WARN: extra model path invalid: {pth}")
    uniq = {}
    for src, p in found:
        uniq[p] = src
    items = [(uniq[p], p) for p in sorted(uniq.keys())]
    return items

def is_visual_dir(name: str):
    low = name.lower()
    return low.startswith("figs") or "figs_vs" in low or low.endswith("_figs") or low in {"images","plots"}

def collect_datasets(dataset_roots: dict):
    results = []
    for label, root in dataset_roots.items():
        r = Path(root).resolve()
        if not r.exists():
            print(f"[collect_datasets] WARN: dataset root not found: {r}")
            continue
        for dirpath, dirnames, filenames in os.walk(r):
            dirnames[:] = [d for d in dirnames if not is_visual_dir(d)]
            for fn in filenames:
                if fn.endswith(".pt"):
                    results.append((label, str(Path(dirpath) / fn)))
    uniq = {}
    for lab, p in results:
        uniq[p] = lab
    items = [(uniq[p], p) for p in sorted(uniq.keys())]
    return items

def parse_model_info(model_path):
    path = Path(model_path)
    info = {
        'full_path': model_path,
        'model_name': path.name,
        'model_dir': str(path.parent),
        'expanded': 'expanded' in model_path.lower(),
        'normalized': 'normalized' in model_path.lower(),
        'training_config': {}
    }
    s = model_path
    m = re.search(r'modes(\d+)', s);  info['training_config']['modes']  = int(m.group(1)) if m else None
    m = re.search(r'width(\d+)', s);  info['training_config']['width']  = int(m.group(1)) if m else None
    m = re.search(r'epochs(\d+)', s); info['training_config']['epochs'] = int(m.group(1)) if m else None
    cfg = path.parent / "config.json"
    if cfg.exists():
        try:
            with open(cfg, "r") as f: info['training_config'].update(json.load(f))
        except Exception: pass
    return info

def parse_dataset_info(dataset_path):
    path = Path(dataset_path)
    parts = path.parts
    info = {'full_path': dataset_path, 'dataset_name': path.name,
            'dataset_type': 'unknown', 'distribution': 'unknown', 'parameters': {}}
    for part in parts:
        low = part.lower()
        if 'all_pos' in low: info['dataset_type'] = 'all_pos'
        elif 'all_neg' in low: info['dataset_type'] = 'all_neg'
        elif 'pos_neg' in low: info['dataset_type'] = 'pos_neg'
        elif 'expanded_pos' in low: info['dataset_type'] = 'expanded_pos'
        m = re.search(r'range_[\-\d\.]+_[\-\d\.]+', low)
        if m: info['dataset_type'] = m.group(0)
        if 'gaussian' in low:
            info['distribution'] = 'gaussian'
            cl = re.search(r'cl(?:=)?([0-9.]+)', low)
            if cl: info['parameters']['correlation_length'] = float(cl.group(1))
        if 'matern' in low:
            info['distribution'] = 'matern'
            cl = re.search(r'cl(?:=)?([0-9.]+)', low)
            nu = re.search(r'nu(?:=)?([0-9.]+)', low)
            if cl: info['parameters']['correlation_length'] = float(cl.group(1))
            if nu: info['parameters']['nu'] = float(nu.group(1))
        if 'zigzag' in low:
            info['distribution'] = 'zigzag'
            pk = re.search(r'peaks(\d+)', low); amp = re.search(r'amp([0-9.]+)', low)
            if pk:  info['parameters']['peaks'] = int(pk.group(1))
            if amp: info['parameters']['amplitude'] = float(amp.group(1))
        eps = re.search(r'eps(?:=)?([0-9.]+)', low);   alp = re.search(r'alpha(?:=)?([0-9.]+)', low)
        stp = re.search(r'steps(?:=)?(\d+)', low);     Nv  = re.search(r'n=(\d+)', low)
        if eps: info['parameters']['eps'] = float(eps.group(1))
        if alp: info['parameters']['alpha'] = float(alp.group(1))
        if stp: info['parameters']['steps'] = int(stp.group(1))
        if Nv:  info['parameters']['N']    = int(Nv.group(1))
        if part == "t1":
            info['parameters']['t_final'] = 1.0
    return info

# ===== 输入构造：严格单通道 (B,N,1) =====
def _ensure_single_feature_no_coord(x_np):
    """
    只保证 (B,N,1)，绝不加坐标通道：
      - (N,)     -> (1,N,1)
      - (B,N)    -> (B,N,1)
      - (B,N,C)  -> 取第一个通道 (B,N,1)
    """
    x_np = np.asarray(x_np, dtype=np.float32)
    if x_np.ndim == 1:
        return x_np[None, :, None]
    if x_np.ndim == 2:
        return x_np[..., None]
    if x_np.ndim == 3:
        return x_np[..., :1]
    raise ValueError(f"Unsupported x shape (expect 1D/2D/3D), got {x_np.shape}")

def _to_BN1(y_np):
    y_np = np.asarray(y_np)
    if y_np.ndim == 1: return y_np[None, :, None]
    if y_np.ndim == 2: return y_np[..., None]
    if y_np.ndim == 3:
        if y_np.shape[-1] == 1: return y_np
        return y_np[..., :1]
    raise ValueError(f"Unsupported y shape (expect (N,), (B,N) or (B,N,C)), got {y_np.shape}")

def _torchify_to_device(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

# ===== 推理与指标：强制单通道 =====
def forward_with_adaptive_channels(model, x_np, y_np, device):
    """
    严格用单通道：(B,N,1)。若输入多通道，只取第1个通道；绝不添加坐标通道。
    """
    # 统一为 (B,N,1)
    if x_np.ndim == 1:
        a_np = x_np[None, :, None].astype(np.float32)
    elif x_np.ndim == 2:
        a_np = x_np[..., None].astype(np.float32)
    elif x_np.ndim == 3:
        a_np = x_np[..., :1].astype(np.float32)
    else:
        raise ValueError(f"Unsupported x shape: {x_np.shape}")

    y_np_ = y_np
    print(f"[debug] parsed shapes -> x: {a_np.shape}, y: {y_np_.shape}")  # (B,N,1) vs (B,N,1)

    # 前向
    x_t = _torchify_to_device(a_np, device)
    with torch.no_grad():
        pred = model(x_t)
    pred_np = pred.detach().cpu().numpy()
    if pred_np.ndim == 2:
        pred_np = pred_np[..., None]

    assert pred_np.shape[:2] == y_np_.shape[:2], f"pred {pred_np.shape} vs y {y_np_.shape}"

    per_rmse, per_mae, per_mape = [], [], []
    used_valid, used_invalid = 0, 0
    pred_stats_accum = {"total":0, "nan":0, "posinf":0, "neginf":0, "invalid":0}

    for b in range(y_np_.shape[0]):
        yt = y_np_[b].reshape(-1)
        yp = pred_np[b].reshape(-1)
        ps = count_special(yp)
        for k in ["total","nan","posinf","neginf","invalid"]:
            pred_stats_accum[k] += ps[k]
        m = safe_regression_metrics(yt, yp)
        per_rmse.append(m["rmse"]); per_mae.append(m["mae"]); per_mape.append(m["mape"])
        used_valid += m["valid_n"]; used_invalid += m["invalid_n"]

    def _nanmean(a): return float(np.nanmean(a)) if len(a) else np.nan
    def _nanstd(a):  return float(np.nanstd(a))  if len(a) else np.nan

    return {
        'rmse_mean': _nanmean(per_rmse), 'rmse_std': _nanstd(per_rmse),
        'mae_mean':  _nanmean(per_mae),  'mae_std':  _nanstd(per_mae),
        'mape_mean': _nanmean(per_mape), 'mape_std': _nanstd(per_mape),
        'total_samples': int(y_np_.shape[0]),
        'valid_samples': int(used_valid),
        'invalid_samples': int(used_invalid),
        'pred_total': pred_stats_accum["total"],
        'pred_nan': pred_stats_accum["nan"],
        'pred_posinf': pred_stats_accum["posinf"],
        'pred_neginf': pred_stats_accum["neginf"],
        'pred_invalid': pred_stats_accum["invalid"],
        'pred_invalid_pct': (pred_stats_accum["invalid"] / pred_stats_accum["total"] * 100.0) if pred_stats_accum["total"] else 0.0,
        'used_in_channels': 1,  # 明确记录为 1
    }

def test_model_on_dataset(model, dataset_path, device):
    try:
        data = torch.load(dataset_path, map_location='cpu', weights_only=False)
        samples_x, samples_y = None, None

        if isinstance(data, dict) and 'a' in data and 'g(a)' in data:
            a = data['a']; g = data['g(a)']
            a_np = _ensure_single_feature_no_coord(a)   # ← 单通道
            g_np = _to_BN1(g)
            samples_x, samples_y = a_np, g_np

        elif isinstance(data, list) and all(isinstance(it, dict) and 'a' in it and 'g(a)' in it for it in data):
            a_list, g_list = [], []
            for it in data:
                a_list.append(_ensure_single_feature_no_coord(it['a']))   # ← 单通道
                g_list.append(_to_BN1(it['g(a)']))
            samples_x = np.concatenate(a_list, axis=0)
            samples_y = np.concatenate(g_list, axis=0)

        elif isinstance(data, dict) and 'x' in data and 'y' in data:
            x_np = _ensure_single_feature_no_coord(data['x'])             # ← 单通道
            y_np = _to_BN1(data['y'])
            samples_x, samples_y = x_np, y_np

        else:
            print(f"[test_model_on_dataset] Unsupported dataset: {dataset_path}")
            return None

        if samples_x is None:
            print(f"[test_model_on_dataset] No valid samples parsed: {dataset_path}")
            return None

        summary = forward_with_adaptive_channels(model, samples_x, samples_y, device)
        return summary
    except Exception as e:
        print(f"[test_model_on_dataset] Error testing {dataset_path}: {e}")
        return None

# ==============================
# 断点续跑：CSV 读取/写入
# ==============================

def load_done_pairs(csv_path: Path):
    done = set()
    if not csv_path.exists():
        return done
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mp = row.get("model_path")
                dp = row.get("dataset_path")
                if mp and dp:
                    done.add((mp, dp))
    except Exception as e:
        print(f"[load_done_pairs] WARN: cannot read {csv_path}: {e}")
    return done

def open_results_writer(csv_path: Path, fieldnames):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_append = csv_path.exists() and csv_path.stat().st_size > 0
    f = open(csv_path, "a" if is_append else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not is_append:
        writer.writeheader()
        f.flush(); os.fsync(f.fileno())
    return f, writer, is_append

# ==============================
# 主流程
# ==============================

def main():
    get_gpu_info()

    base_dir = Path(BASE_OUTDIR).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_list = collect_models(MODEL_DIRS, EXTRA_MODEL_PATHS)
    dataset_list = collect_datasets(DATASET_ROOTS)

    M = len(model_list); D = len(dataset_list)
    print(f"\nCollected models: {M}")
    mc = Counter([src for src,_ in model_list])
    for k,v in mc.items(): print(f"  - {k}: {v}")
    print(f"Collected datasets: {D}")
    dc = Counter([src for src,_ in dataset_list])
    for k,v in dc.items(): print(f"  - {k}: {v}")

    info_file = (base_dir / f"model_dataset_info_{timestamp}.txt") if INFO_TXT_WITH_TS else (base_dir / "model_dataset_info.txt")
    with open(info_file, "w") as f:
        f.write("MODELS INFORMATION:\n")
        f.write("="*60 + "\n")
        for i, (ms, mp) in enumerate(model_list, 1):
            mi = parse_model_info(mp)
            f.write(f"[{i}/{M}] ({ms}) {mi['full_path']}\n")
            f.write(f"     Name: {mi['model_name']}\n")
            f.write(f"     Expanded: {mi['expanded']}\n")
            f.write(f"     Normalized: {mi['normalized']}\n")
            f.write(f"     Config: {mi['training_config']}\n\n")
        f.write("\nDATASETS INFORMATION:\n")
        f.write("="*60 + "\n")
        for j, (ds, dp) in enumerate(dataset_list, 1):
            di = parse_dataset_info(dp)
            f.write(f"[{j}/{D}] ({ds}) {di['full_path']}\n")
            f.write(f"     Name: {di['dataset_name']}\n")
            f.write(f"     Type: {di['dataset_type']}\n")
            f.write(f"     Distribution: {di['distribution']}\n")
            f.write(f"     Parameters: {di['parameters']}\n\n")
    print(f"\nModel/Dataset inventory saved to: {info_file}")

    results_csv = base_dir / RESULTS_CSV_NAME
    fieldnames = [
        'model_source', 'model_path', 'model_name', 'model_config',
        'dataset_source', 'dataset_path', 'dataset_name', 'dataset_type',
        'dataset_distribution', 'dataset_parameters',
        'rmse_mean', 'rmse_std', 'mae_mean', 'mae_std', 'mape_mean', 'mape_std',
        'total_samples', 'valid_samples', 'invalid_samples', 'test_timestamp',
        # 诊断增强
        'pred_total','pred_nan','pred_posinf','pred_neginf','pred_invalid','pred_invalid_pct',
        'used_in_channels',
    ]

    done_pairs = load_done_pairs(results_csv)
    if done_pairs:
        print(f"\n[Resume] Loaded {len(done_pairs)} completed (model,dataset) pairs from CSV.")

    fcsv, writer, is_append = open_results_writer(results_csv, fieldnames)
    print(f"\nWriting results to: {results_csv} ({'append' if is_append else 'new file'})")

    try:
        for mi, (model_src, model_path) in enumerate(model_list, 1):
            print(f"\n[Model {mi}/{M}] Loading: ({model_src}) {model_path}")
            model_info = parse_model_info(model_path)
            model_path_key = model_info['full_path']

            model = load_model(model_path, device)
            if model is None:
                print("  -> skip (failed to load)")
                continue

            for dj, (ds_src, ds_path) in enumerate(dataset_list, 1):
                pair = (model_path_key, ds_path)
                if pair in done_pairs:
                    print(f"    [Dataset {dj}/{D}] ({ds_src}) {ds_path}  -> already done, skip")
                    continue

                print(f"    [Dataset {dj}/{D}] Testing: ({ds_src}) {ds_path}")
                res = test_model_on_dataset(model, ds_path, device)
                if res is None:
                    print("      -> test failed/unsupported, skip")
                    continue

                di = parse_dataset_info(ds_path)
                row = {
                    'model_source': model_src,
                    'model_path': model_info['full_path'],
                    'model_name': model_info['model_name'],
                    'model_config': str(model_info['training_config']),
                    'dataset_source': ds_src,
                    'dataset_path': di['full_path'],
                    'dataset_name': di['dataset_name'],
                    'dataset_type': di['dataset_type'],
                    'dataset_distribution': di['distribution'],
                    'dataset_parameters': str(di['parameters']),
                    'rmse_mean': res['rmse_mean'],
                    'rmse_std':  res['rmse_std'],
                    'mae_mean':  res['mae_mean'],
                    'mae_std':   res['mae_std'],
                    'mape_mean': res['mape_mean'],
                    'mape_std':  res['mape_std'],
                    'total_samples': res['total_samples'],
                    'valid_samples': res['valid_samples'],
                    'invalid_samples': res['invalid_samples'],
                    'test_timestamp': datetime.now().isoformat(timespec="seconds"),
                    # 诊断增强
                    'pred_total': res.get('pred_total', 0),
                    'pred_nan': res.get('pred_nan', 0),
                    'pred_posinf': res.get('pred_posinf', 0),
                    'pred_neginf': res.get('pred_neginf', 0),
                    'pred_invalid': res.get('pred_invalid', 0),
                    'pred_invalid_pct': res.get('pred_invalid_pct', 0.0),
                    'used_in_channels': res.get('used_in_channels', None),
                }
                writer.writerow(row)
                fcsv.flush(); os.fsync(fcsv.fileno())
                done_pairs.add(pair)

                print(f"      ✓ RMSE {res['rmse_mean']:.4f}±{res['rmse_std']:.4f} | "
                      f"MAE {res['mae_mean']:.4f}±{res['mae_std']:.4f} | "
                      f"MAPE {res['mape_mean']:.2f}%±{res['mape_std']:.2f}% | "
                      f"in_ch={row['used_in_channels']} | pred_invalid={row['pred_invalid_pct']:.2f}%")

    finally:
        fcsv.close()

    print("\nAll done.")
    print(f"Results CSV: {results_csv}")

if __name__ == "__main__":
    main()
