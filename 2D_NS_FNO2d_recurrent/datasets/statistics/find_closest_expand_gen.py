# -*- coding: utf-8 -*-
"""
Compute pairwise similarity (frame 0) between:
  targets = {all 'expanded' + 'train' + 'test'}  vs  generalizability (all)
using stats from:
  - dataset_stats_1440.h5        (spatial-domain)
  - dataset_stats_fft_1440.h5    (FFT-domain, use |F| a.k.a. 'mag')

Outputs (created in the same folder as this script):
  - similarity_spatial_frame0.csv
  - similarity_fft_mag_frame0.csv
"""

from __future__ import annotations
from pathlib import Path
import csv, math
import numpy as np
import h5py
import sys
from typing import Dict, Tuple, List, Optional

# ---------------- user knobs ----------------
from pathlib import Path

# 脚本所在目录（__file__ 在交互环境可能没有，兜底用 cwd）
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

SPATIAL_H5 = HERE / "dataset_stats_1440.h5"
FFT_H5     = HERE / "dataset_stats_fft_1440.h5"

FRAME = "0"                  # we compare frame 0
GRID_BINS = 256              # common rebin grid for histogram metrics
EPS = 1e-12                  # smoothing for divergences
PRINT_EVERY = 50             # progress printing
# -------------------------------------------

# ===== 新增/更新：imports =====
import re  # <--- 需要解析文件名

# ===== 新增：数据集参数键与解析函数（用你给的实现）=====
DATASET_KEYS = [
    "type","solver","nx","N","nu","t","ntimepoints",
    "length_scale","variance","alpha","tau","period","exp_factor",
    "vmax","vmin"
]

def parse_dataset_name(stem: str) -> dict:
    info = {}
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

# 小工具：从路径或组名得到 stem，并解析
def _stem_from_path_or_name(file_path: str, group_name: str) -> str:
    try:
        s = Path(file_path).name
        if s:
            return Path(s).stem
    except Exception:
        pass
    return str(group_name)

def _expand_parsed(prefix: str, parsed: dict) -> dict:
    """
    prefix: 'target' or 'gen'
    把 parsed 的键展开成列名；其中 parsed['type'] → '<prefix>_dtype'，其余 → '<prefix>_<key>'
    未解析到的填空字符串 ""。
    """
    out = {}
    for k in DATASET_KEYS:
        col = f"{prefix}_dtype" if k == "type" else f"{prefix}_{k}"
        out[col] = parsed.get(k, "")
    return out

# ======= utilities =======

def _cdf_at_x_uniform_hist(x: np.ndarray, vmin: float, vmax: float, counts: np.ndarray) -> np.ndarray:
    """
    CDF(x) for an equal-width histogram defined on [vmin, vmax] with integer counts per bin.

    Returns array same shape as x with values in [0,1].
    """
    nb = int(counts.size)
    total = float(counts.sum())
    if total <= 0 or not np.isfinite(total):
        return np.zeros_like(x, dtype=np.float64)

    # Handle degenerate range
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        # everything collapses; place all mass at vmin
        return (x >= vmin).astype(np.float64)

    # Normalize positions to [0, nb]
    pos = (np.clip(x, vmin, vmax) - vmin) / (vmax - vmin) * nb  # [0, nb]
    i = np.floor(pos).astype(int)
    r = pos - i

    # Clamp i to [0, nb-1] for in-bin interpolation; values at vmax map to i=nb
    i_clip = np.clip(i, 0, nb - 1)

    # cumulative counts up to bin i
    cumsum = np.concatenate(([0.0], np.cumsum(counts, dtype=np.float64)))  # len nb+1
    base = cumsum[np.clip(i, 0, nb)] / total  # when pos in [0,nb], i==nb is OK

    # Add partial bin contribution (only when i in [0, nb-1])
    part = np.zeros_like(r, dtype=np.float64)
    in_range = (i >= 0) & (i < nb)
    if in_range.any():
        part[in_range] = (r[in_range] * counts[i_clip[in_range]] / total)

    # Fix x < vmin -> 0, x >= vmax -> 1
    cdf = base + part
    cdf[x < vmin] = 0.0
    cdf[x >= vmax] = 1.0
    return cdf


def _rebin_hist_to_common_grid(counts: np.ndarray, vmin: float, vmax: float,
                               xmin: float, xmax: float, grid_bins: int) -> np.ndarray:
    """
    Rebin a histogram (equal-width bins on [vmin, vmax]) to a common grid on [xmin, xmax]
    with `grid_bins` equal-width bins. Returns a probability mass function over the grid
    (length = grid_bins, sums to 1).
    """
    edges = np.linspace(xmin, xmax, grid_bins + 1, dtype=np.float64)
    cdf_lo = _cdf_at_x_uniform_hist(edges[:-1], vmin, vmax, counts)
    cdf_hi = _cdf_at_x_uniform_hist(edges[1:],  vmin, vmax, counts)
    mass = (cdf_hi - cdf_lo)  # integrated mass per grid bin
    s = mass.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full(grid_bins, 1.0 / grid_bins, dtype=np.float64)
    return (mass / s).astype(np.float64)


def _range_overlap_ratio(vmin1: float, vmax1: float, vmin2: float, vmax2: float) -> float:
    if not all(np.isfinite([vmin1, vmax1, vmin2, vmax2])): return 0.0
    if vmin1 > vmax1 or vmin2 > vmax2: return 0.0
    inter = max(0.0, min(vmax1, vmax2) - max(vmin1, vmin2))
    uni   = max(vmax1, vmax2) - min(vmin1, vmin2)
    if uni <= 0: return 0.0
    return inter / uni


def _js_divergence_bits(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    js_nat = 0.5 * (kl_pm + kl_qm)
    return js_nat / np.log(2.0)  # bits


def _kl_divergence_nat(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _emd_w1(p: np.ndarray, q: np.ndarray, xmin: float, xmax: float) -> Tuple[float, float]:
    """Wasserstein-1 via discrete CDF difference on the common grid."""
    p = p / max(p.sum(), EPS); q = q / max(q.sum(), EPS)
    cp = np.cumsum(p); cq = np.cumsum(q)
    dx = (xmax - xmin) / len(p) if len(p) > 0 else 0.0
    w1 = float(np.sum(np.abs(cp - cq)) * dx)
    # normalized by union range to get a unitless [0,1]-ish measure
    w1_norm = float(w1 / (abs(xmax - xmin) + EPS))
    return w1, w1_norm


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), EPS); q = q / max(q.sum(), EPS)
    return 0.5 * float(np.sum(np.abs(p - q)))


def _overlap_coeff(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), EPS); q = q / max(q.sum(), EPS)
    return float(np.sum(np.minimum(p, q)))


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    sa = a.std(ddof=0); sb = b.std(ddof=0)
    if sa <= 0 or sb <= 0 or not np.isfinite(sa) or not np.isfinite(sb):
        return float('nan')
    # use np.corrcoef for numerical stability
    return float(np.corrcoef(a, b)[0, 1])


def _norm_mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    mae = float(np.mean(np.abs(a - b)))
    denom = float(max(1e-12, (np.nanmax([a.max(), b.max()]) - np.nanmin([a.min(), b.min()]))))
    return float(mae / denom)


# ======= H5 accessors =======

def _enumerate_files_with_types(h5f: h5py.File) -> Dict[str, Dict]:
    out = {}
    g = h5f["files"]
    for name in g.keys():
        grp = g[name]
        fpath = grp.attrs.get("file_path", "")
        ftype = grp.attrs.get("type", "")
        out[name] = {"type": str(ftype), "file_path": str(fpath)}
    return out


def _get_spatial_hist(h5f: h5py.File, name: str, frame: str) -> Optional[Tuple[np.ndarray, float, float, int, Dict[str,float]]]:
    try:
        base = h5f[f"files/{name}/frames/{frame}/global"]
        counts = base["hist"]["counts"][()]
        vmin = float(base["hist"].attrs["vmin"])
        vmax = float(base["hist"].attrs["vmax"])
        nbins = int(base["hist"].attrs["nbins"])
        stats = {
            "mean": float(base["mean"][()]),
            "std":  float(base["std"][()]),
            "q50":  float(base["q50"][()]),
        }
        return counts.astype(np.float64), vmin, vmax, nbins, stats
    except Exception:
        return None


def _get_spatial_maps(h5f: h5py.File, name: str, frame: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        p = f"files/{name}/frames/{frame}/per_cell"
        mean_map = h5f[p + "/mean"][()]
        std_map  = h5f[p + "/std"][()]
        return mean_map, std_map
    except Exception:
        return None


def _get_fft_mag_hist(h5f: h5py.File, name: str, frame: str) -> Optional[Tuple[np.ndarray, float, float, int, Dict[str,float]]]:
    try:
        base = h5f[f"files/{name}/frames/{frame}/fft/mag/global"]
        counts = base["hist"]["counts"][()]
        vmin = float(base["hist"].attrs["vmin"])
        vmax = float(base["hist"].attrs["vmax"])
        nbins = int(base["hist"].attrs["nbins"])
        stats = {
            "mean": float(base["mean"][()]),
            "std":  float(base["std"][()]),
            "q50":  float(base["q50"][()]),
        }
        return counts.astype(np.float64), vmin, vmax, nbins, stats
    except Exception:
        return None


def _get_fft_mag_maps(h5f: h5py.File, name: str, frame: str) -> Optional[np.ndarray]:
    try:
        p = f"files/{name}/frames/{frame}/fft/mag/per_cell"
        mean_map = h5f[p + "/mean"][()]
        return mean_map  # (H_fft, Wc_fft)
    except Exception:
        return None


# ======= main similarity runners =======

def _pair_hist_metrics(h1, h2, grid_bins: int) -> Dict[str, float]:
    (c1, vmin1, vmax1, nb1, s1) = h1
    (c2, vmin2, vmax2, nb2, s2) = h2
    xmin = float(min(vmin1, vmin2))
    xmax = float(max(vmax1, vmax2))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
        # fallback uniform PMFs
        p = np.full(grid_bins, 1.0 / grid_bins, dtype=np.float64)
        q = p.copy()
    else:
        p = _rebin_hist_to_common_grid(c1, vmin1, vmax1, xmin, xmax, grid_bins)
        q = _rebin_hist_to_common_grid(c2, vmin2, vmax2, xmin, xmax, grid_bins)

    js_bits = _js_divergence_bits(p, q)
    tv = _tv_distance(p, q)
    emd, emd_norm = _emd_w1(p, q, xmin, xmax)
    ovl = _overlap_coeff(p, q)
    rng_ovl = _range_overlap_ratio(vmin1, vmax1, vmin2, vmax2)
    kl_pq = _kl_divergence_nat(p, q)
    kl_qp = _kl_divergence_nat(q, p)

    return {
        "hist_js_bits": js_bits,
        "hist_tv": tv,
        "hist_emd": emd,
        "hist_emd_norm": emd_norm,
        "hist_overlap_coeff": ovl,
        "hist_range_overlap_ratio": rng_ovl,
        "hist_kl_pq_nat": kl_pq,
        "hist_kl_qp_nat": kl_qp,
        # scalar deltas (from global stats)
        "global_mean_absdiff": abs(s1["mean"] - s2["mean"]),
        "global_std_absdiff":  abs(s1["std"]  - s2["std"]),
        "global_q50_absdiff":  abs(s1["q50"]  - s2["q50"]),
    }


def _pair_map_metrics(meanA: Optional[np.ndarray], stdA: Optional[np.ndarray],
                      meanB: Optional[np.ndarray], stdB: Optional[np.ndarray]) -> Dict[str, float]:
    out = {
        "mean_map_pearson_r": float('nan'),
        "mean_map_norm_mae":  float('nan'),
        "std_map_pearson_r":  float('nan'),
        "std_map_norm_mae":   float('nan'),
    }
    if meanA is not None and meanB is not None and meanA.shape == meanB.shape:
        out["mean_map_pearson_r"] = _pearson_r(meanA, meanB)
        out["mean_map_norm_mae"]  = _norm_mae(meanA, meanB)
    if stdA is not None and stdB is not None and stdA.shape == stdB.shape:
        out["std_map_pearson_r"]  = _pearson_r(stdA, stdB)
        out["std_map_norm_mae"]   = _norm_mae(stdA, stdB)
    return out


def run_spatial_similarity():
    if not SPATIAL_H5.exists():
        print(f"[ERROR] Missing {SPATIAL_H5}")
        return
    with h5py.File(SPATIAL_H5, "r") as sp:
        index = _enumerate_files_with_types(sp)
        gens    = sorted([k for k,v in index.items() if v["type"] in ("generalizability","train","test")])
        targets = sorted([k for k,v in index.items() if v["type"] in ("expanded",)])

        # ---- header：在原始列之间插入解析出来的 target_* 和 gen_* 列 ----
        parsed_target_cols = ["target_dtype","target_solver","target_nx","target_N","target_nu","target_t",
                              "target_ntimepoints","target_length_scale","target_variance","target_alpha",
                              "target_tau","target_period","target_exp_factor","target_vmax","target_vmin"]
        parsed_gen_cols = ["gen_dtype","gen_solver","gen_nx","gen_N","gen_nu","gen_t",
                           "gen_ntimepoints","gen_length_scale","gen_variance","gen_alpha",
                           "gen_tau","gen_period","gen_exp_factor","gen_vmax","gen_vmin"]

        out_csv = SPATIAL_H5.parent / "similarity_spatial_frame0.csv"
        header = [
            "target_name","target_type","target_path",
            *parsed_target_cols,           # <--- 新增解析列（target）
            "gen_name","gen_type","gen_path",
            *parsed_gen_cols,              # <--- 新增解析列（gen）
            "H","W","grid_bins",
            # histogram metrics
            "hist_js_bits","hist_tv","hist_emd","hist_emd_norm",
            "hist_overlap_coeff","hist_range_overlap_ratio",
            "hist_kl_pq_nat","hist_kl_qp_nat",
            # scalar global deltas
            "global_mean_absdiff","global_std_absdiff","global_q50_absdiff",
            # map metrics
            "mean_map_pearson_r","mean_map_norm_mae",
            "std_map_pearson_r","std_map_norm_mae",
        ]

        # Preload all target maps & hists (small set)
        t_cache = {}
        for t in targets:
            h = _get_spatial_hist(sp, t, FRAME)
            m = _get_spatial_maps(sp, t, FRAME)
            t_cache[t] = {"hist": h, "maps": m}

        rows = []
        for ti, t in enumerate(targets, 1):
            tinfo = index[t]
            th = t_cache[t]["hist"]
            tm = t_cache[t]["maps"]
            if th is None:
                print(f"[WARN] skip target {t}: missing spatial hist")
                continue

            # 解析 target 文件名参数
            t_stem = _stem_from_path_or_name(tinfo.get("file_path",""), t)
            t_parsed = _expand_parsed("target", parse_dataset_name(t_stem))

            # infer shape from mean map if present
            H = W = ""
            if tm is not None:
                H, W = tm[0].shape

            for gi, g in enumerate(gens, 1):
                gh = _get_spatial_hist(sp, g, FRAME)
                if gh is None:
                    continue
                gm = _get_spatial_maps(sp, g, FRAME)

                # 解析 gen 文件名参数
                ginfo = index[g]
                g_stem = _stem_from_path_or_name(ginfo.get("file_path",""), g)
                g_parsed = _expand_parsed("gen", parse_dataset_name(g_stem))

                hist_metrics = _pair_hist_metrics(th, gh, GRID_BINS)
                map_metrics  = _pair_map_metrics(tm[0] if tm else None, tm[1] if tm else None,
                                                 gm[0] if gm else None, gm[1] if gm else None)

                rows.append([
                    t, tinfo["type"], tinfo["file_path"],
                    *[t_parsed[c] for c in parsed_target_cols],   # 按列顺序展开
                    g, ginfo["type"], ginfo["file_path"],
                    *[g_parsed[c] for c in parsed_gen_cols],      # 按列顺序展开
                    H, W, GRID_BINS,
                    hist_metrics["hist_js_bits"],
                    hist_metrics["hist_tv"],
                    hist_metrics["hist_emd"],
                    hist_metrics["hist_emd_norm"],
                    hist_metrics["hist_overlap_coeff"],
                    hist_metrics["hist_range_overlap_ratio"],
                    hist_metrics["hist_kl_pq_nat"],
                    hist_metrics["hist_kl_qp_nat"],
                    hist_metrics["global_mean_absdiff"],
                    hist_metrics["global_std_absdiff"],
                    hist_metrics["global_q50_absdiff"],
                    map_metrics["mean_map_pearson_r"],
                    map_metrics["mean_map_norm_mae"],
                    map_metrics["std_map_pearson_r"],
                    map_metrics["std_map_norm_mae"],
                ])

                if (gi % PRINT_EVERY) == 0:
                    print(f"[spatial] target {ti}/{len(targets)} '{t}' vs gen {gi}/{len(gens)}")

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

        print(f"[OK] wrote {out_csv} with {len(rows)} rows")



def run_fft_similarity():
    if not FFT_H5.exists():
        print(f"[ERROR] Missing {FFT_H5}")
        return
    with h5py.File(FFT_H5, "r") as ff:
        index = _enumerate_files_with_types(ff)
        gens    = sorted([k for k,v in index.items() if v["type"] in ("generalizability","train","test")])
        targets = sorted([k for k,v in index.items() if v["type"] in ("expanded",)])

        parsed_target_cols = ["target_dtype","target_solver","target_nx","target_N","target_nu","target_t",
                              "target_ntimepoints","target_length_scale","target_variance","target_alpha",
                              "target_tau","target_period","target_exp_factor","target_vmax","target_vmin"]
        parsed_gen_cols = ["gen_dtype","gen_solver","gen_nx","gen_N","gen_nu","gen_t",
                           "gen_ntimepoints","gen_length_scale","gen_variance","gen_alpha",
                           "gen_tau","gen_period","gen_exp_factor","gen_vmax","gen_vmin"]

        out_csv = FFT_H5.parent / "similarity_fft_mag_frame0.csv"
        header = [
            "target_name","target_type","target_path",
            *parsed_target_cols,
            "gen_name","gen_type","gen_path",
            *parsed_gen_cols,
            "H_fft","Wc_fft","grid_bins","component",
            # histogram metrics (mag)
            "hist_js_bits","hist_tv","hist_emd","hist_emd_norm",
            "hist_overlap_coeff","hist_range_overlap_ratio",
            "hist_kl_pq_nat","hist_kl_qp_nat",
            # scalar global deltas
            "global_mean_absdiff","global_std_absdiff","global_q50_absdiff",
            # per-cell spectrum mean map metrics
            "spec_mean_map_pearson_r","spec_mean_map_norm_mae",
        ]

        # Preload small target cache
        t_cache = {}
        for t in targets:
            h = _get_fft_mag_hist(ff, t, FRAME)
            m = _get_fft_mag_maps(ff, t, FRAME)
            t_cache[t] = {"hist": h, "mean_map": m}

        rows = []
        for ti, t in enumerate(targets, 1):
            tinfo = index[t]
            th = t_cache[t]["hist"]
            tm = t_cache[t]["mean_map"]
            if th is None:
                print(f"[WARN] skip target {t}: missing FFT mag hist")
                continue

            # 解析 target
            t_stem = _stem_from_path_or_name(tinfo.get("file_path",""), t)
            t_parsed = _expand_parsed("target", parse_dataset_name(t_stem))

            H_fft = Wc_fft = ""
            if tm is not None:
                H_fft, Wc_fft = tm.shape

            for gi, g in enumerate(gens, 1):
                gh = _get_fft_mag_hist(ff, g, FRAME)
                if gh is None:
                    continue
                gm = _get_fft_mag_maps(ff, g, FRAME)

                # 解析 gen
                ginfo = index[g]
                g_stem = _stem_from_path_or_name(ginfo.get("file_path",""), g)
                g_parsed = _expand_parsed("gen", parse_dataset_name(g_stem))

                hist_metrics = _pair_hist_metrics(th, gh, GRID_BINS)
                map_metrics = {
                    "spec_mean_map_pearson_r": float('nan'),
                    "spec_mean_map_norm_mae":  float('nan'),
                }
                if tm is not None and gm is not None and tm.shape == gm.shape:
                    map_metrics["spec_mean_map_pearson_r"] = _pearson_r(tm, gm)
                    map_metrics["spec_mean_map_norm_mae"]  = _norm_mae(tm, gm)

                rows.append([
                    t, tinfo["type"], tinfo["file_path"],
                    *[t_parsed[c] for c in parsed_target_cols],
                    g, ginfo["type"], ginfo["file_path"],
                    *[g_parsed[c] for c in parsed_gen_cols],
                    H_fft, Wc_fft, GRID_BINS, "mag",
                    hist_metrics["hist_js_bits"],
                    hist_metrics["hist_tv"],
                    hist_metrics["hist_emd"],
                    hist_metrics["hist_emd_norm"],
                    hist_metrics["hist_overlap_coeff"],
                    hist_metrics["hist_range_overlap_ratio"],
                    hist_metrics["hist_kl_pq_nat"],
                    hist_metrics["hist_kl_qp_nat"],
                    hist_metrics["global_mean_absdiff"],
                    hist_metrics["global_std_absdiff"],
                    hist_metrics["global_q50_absdiff"],
                    map_metrics["spec_mean_map_pearson_r"],
                    map_metrics["spec_mean_map_norm_mae"],
                ])

                if (gi % PRINT_EVERY) == 0:
                    print(f"[fft|mag] target {ti}/{len(targets)} '{t}' vs gen {gi}/{len(gens)}")

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

        print(f"[OK] wrote {out_csv} with {len(rows)} rows")



def main():
    print("[RUN] spatial similarity (frame 0)")
    run_spatial_similarity()
    print("[RUN] fft(|F|) similarity (frame 0)")
    run_fft_similarity()


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
