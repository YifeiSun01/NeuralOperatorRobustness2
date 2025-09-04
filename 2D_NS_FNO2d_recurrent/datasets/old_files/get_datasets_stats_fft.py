# -*- coding: utf-8 -*-
"""
Compute FFT-domain statistics for frames 0 and 19 and write ONE HDF5 (dataset_stats_fft.h5).
For each dataset and each frame in {0, 19}, we compute stats for:
  - mag = |F|, real = Re(F), imag = Im(F)
Where F = rfft2(y[..., t]) over spatial axes (H, W).
We save:
  /files/{safe_stem}/attrs (attrs: file_path,N,H,W,T,timing_sec,type, H_fft, Wc_fft, fft_kind)
  /files/{safe_stem}/fft_axes/{ky,kx} (frequency coordinates)
  /files/{safe_stem}/frames/{frame}/fft/{comp}/global/{mean,std,min,max,range,q25,q50,q75}
  /files/{safe_stem}/frames/{frame}/fft/{comp}/global/hist/{counts} (attrs: vmin,vmax,nbins)
  /files/{safe_stem}/frames/{frame}/fft/{comp}/per_cell/{mean,std,min,max,range,q25,q50,q75}
  /summary/csv (one row per file×frame×component; includes type, component)
Only scans *.pt in the top-level of SEARCH_DIRS (non-recursive).
"""
from pathlib import Path
import time
import numpy as np
import torch
import h5py

# ======== user params ========
OUTPUT_H5 = Path("./dataset_stats_fft.h5")  # 新的输出文件名
FRAMES    = (0, 19)                         # 只统计 frame 0 和 19
NBINS     = 100                             # 全局直方图 100 bin
GZIP_LVL  = 4
# =============================

SEARCH_DIRS = [
    "expanded_exponax_datasets/t20/N=1150",
    "exponax_datasets/t20/generalizability",
    "exponax_datasets/t20",
]

COMPONENTS = ("mag", "real", "imag")  # 频域统计的三种分量

def _sanitize_name(stem: str) -> str:
    return stem.split("|")[0]

def _infer_type(pt_path: Path) -> str:
    pstr = str(pt_path).lower()
    name = pt_path.name.lower()
    if "/generalizability/" in pstr:
        return "generalizability"
    if "expanded_exponax" in pstr or "/expanded" in pstr:
        return "expanded"
    if "train" in name:
        return "train"
    if "test" in name:
        return "test"
    return "train"  # 不确定就默认 train（可改 "unknown"）

def _hist_edges_from_values(values: np.ndarray, nbins: int) -> np.ndarray:
    vmin = float(np.nanmin(values)); vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = -1.0, 1.0
    if vmin == vmax:
        eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6
        vmin -= eps; vmax += eps
    return np.linspace(vmin, vmax, nbins + 1, dtype=np.float64)

def _global_stats(values: np.ndarray, nbins: int):
    flat = np.asarray(values, dtype=np.float64).ravel()
    mean = float(np.nanmean(flat)); std  = float(np.nanstd(flat))
    vmin = float(np.nanmin(flat));  vmax = float(np.nanmax(flat))
    q25, q50, q75 = np.quantile(flat, [0.25, 0.50, 0.75])
    edges = _hist_edges_from_values(flat, nbins)
    counts, _ = np.histogram(flat, bins=edges)
    return {
        "mean": mean, "std": std, "min": vmin, "max": vmax,
        "range": float(vmax - vmin), "q25": float(q25),
        "q50": float(q50), "q75": float(q75),
        "hist": {"edges": edges, "counts": counts.astype(np.int32)},
    }

def _per_cell_stats(A: np.ndarray, H: int, Wc: int):
    """
    A: shape (N, H*Wc). Compute per-cell stats over axis=0 then reshape(H, Wc).
    """
    mean = np.nanmean(A, axis=0).reshape(H, Wc).astype(np.float32)
    std  = np.nanstd(A, axis=0).reshape(H, Wc).astype(np.float32)
    vmin = np.nanmin(A, axis=0).reshape(H, Wc).astype(np.float32)
    vmax = np.nanmax(A, axis=0).reshape(H, Wc).astype(np.float32)
    rng  = (vmax - vmin).astype(np.float32)
    q25  = np.quantile(A, 0.25, axis=0).reshape(H, Wc).astype(np.float32)
    q50  = np.quantile(A, 0.50, axis=0).reshape(H, Wc).astype(np.float32)
    q75  = np.quantile(A, 0.75, axis=0).reshape(H, Wc).astype(np.float32)
    return {"mean": mean, "std": std, "min": vmin, "max": vmax, "range": rng,
            "q25": q25, "q50": q50, "q75": q75}

def _fft_components(y_t: torch.Tensor):
    """
    y_t: (N,H,W) real tensor
    return dict with torch tensors of shape (N,H,Wc): mag, real, imag
    """
    # rfft2 over spatial dims (H,W)
    F = torch.fft.rfft2(y_t, dim=(1,2))                # complex64/complex128
    real = F.real
    imag = F.imag
    mag  = torch.abs(F)
    return {"mag": mag, "real": real, "imag": imag}

def process_one_file_fft(pt_path: Path):
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "y" not in payload:
        raise ValueError(f"{pt_path.name}: not a dict or missing key 'y'")
    y = payload["y"]
    if not torch.is_tensor(y) or y.ndim != 4 or y.shape[-1] < max(FRAMES) + 1:
        raise ValueError(f"{pt_path.name}: y has wrong shape {tuple(y.shape)} (expect (N,H,W,T>=20))")

    # coerce to float32 for stable FFT dtype (complex64)
    if y.dtype not in (torch.float32, torch.float64):
        y = y.float()

    N, H, W, T = map(int, y.shape)
    Wc = W//2 + 1  # rfft2 width
    per_frame = {}

    with torch.no_grad():
        for t in FRAMES:
            y_t = y[..., t]                  # (N,H,W)
            comps = _fft_components(y_t)     # dict of (N,H,Wc)

            per_comp = {}
            for cname, C in comps.items():
                C_np = C.contiguous().cpu().numpy().astype(np.float64, copy=False)
                A = C_np.reshape(N, H*Wc)  # (N, H*Wc)
                per_cell = _per_cell_stats(A, H, Wc)
                global_stats = _global_stats(A, NBINS)
                per_comp[cname] = {"per_cell": per_cell, "global": global_stats}

            per_frame[int(t)] = {"fft": per_comp}

    return {"N": N, "H": H, "W": W, "T": T, "H_fft": H, "Wc_fft": Wc, "frames": per_frame}

def _fmt_sec(s: float) -> str:
    m, sec = divmod(s, 60); h, m = divmod(m, 60)
    return f"{int(h)}:{int(m):02d}:{sec:06.3f}"

def _collect_pt_files_nonrecursive(base_dir: Path):
    all_files = []; total_count = 0
    print(f"[INFO] Scanning {len(SEARCH_DIRS)} dirs (non-recursive *.pt only):")
    for rel in SEARCH_DIRS:
        d = (base_dir / rel).resolve()
        if not d.exists():
            print(f"  [WARN] {d} (not found)"); continue
        files = sorted(p for p in d.glob("*.pt") if p.is_file())
        total_count += len(files)
        print(f"\n[DIR] {d}  |  {len(files)} files")
        for i, p in enumerate(files, 1):
            print(f"   {i:3d}. {p}")
        all_files.extend(files)
        if not files:
            print("   (empty)")
    print(f"\n[INFO] Found {total_count} .pt files across {len(SEARCH_DIRS)} dirs (non-recursive).")
    return all_files

def main():
    base_dir = Path(__file__).parent.resolve()
    pt_files = _collect_pt_files_nonrecursive(base_dir)
    if not pt_files:
        print("[INFO] No .pt files found."); return

    # summary: one row per file × frame × component (mag/real/imag)
    header = ["file", "type", "frame", "component",
              "N", "H", "W", "T", "H_fft", "Wc_fft",
              "mean", "std", "min", "max", "range", "q25", "q50", "q75",
              "hist_vmin", "hist_vmax", "hist_nbins", "timing_sec"]
    summary_rows = []
    t_total0 = time.perf_counter()

    with h5py.File(OUTPUT_H5, "w") as h5:
        h5.create_group("files")
        sum_grp = h5.create_group("summary")

        # 也保存频率坐标的说明（全局）
        meta = h5.create_group("meta")
        meta.attrs["fft_kind"] = "rfft2"  # 非负频率半谱

        for idx, p in enumerate(pt_files, 1):
            print(f"\n[PROCESS {idx}/{len(pt_files)}] {p}")
            t0 = time.perf_counter()
            try:
                res = process_one_file_fft(p)
            except Exception as e:
                print(f"  [SKIP] {p.name}: {e}"); continue
            t1 = time.perf_counter()
            timing = float(t1 - t0)

            safe_stem = _sanitize_name(p.stem)
            dtype = _infer_type(p)

            fgrp = h5.require_group(f"files/{safe_stem}")
            fgrp.attrs["file_path"]  = str(p.resolve())
            fgrp.attrs["N"]          = res["N"]
            fgrp.attrs["H"]          = res["H"]
            fgrp.attrs["W"]          = res["W"]
            fgrp.attrs["T"]          = res["T"]
            fgrp.attrs["timing_sec"] = timing
            fgrp.attrs["type"]       = dtype
            fgrp.attrs["H_fft"]      = res["H_fft"]
            fgrp.attrs["Wc_fft"]     = res["Wc_fft"]
            fgrp.attrs["fft_kind"]   = "rfft2"

            # 保存频率坐标（便于可视化或下游处理）
            # 这里默认网格步长 d=1（如果你的物理间隔不是 1，可以按需改）
            ky = np.fft.fftfreq(res["H_fft"], d=1.0).astype(np.float64)
            kx = np.fft.rfftfreq(res["W"],     d=1.0).astype(np.float64)
            fft_axes = fgrp.require_group("fft_axes")
            # 避免重复创建
            if "ky" not in fft_axes:
                fft_axes.create_dataset("ky", data=ky)
            if "kx" not in fft_axes:
                fft_axes.create_dataset("kx", data=kx)

            for frame, frame_content in res["frames"].items():
                g_frame = fgrp.require_group(f"frames/{frame}")
                g_fft   = g_frame.require_group("fft")

                for cname in COMPONENTS:
                    content = frame_content["fft"][cname]

                    # ---- global ----
                    g_comp = g_fft.require_group(cname)
                    g_global = g_comp.require_group("global")
                    for key in ("mean","std","min","max","range","q25","q50","q75"):
                        g_global.create_dataset(key, data=np.array(content["global"][key], dtype=np.float32))
                    hist = content["global"]["hist"]
                    g_hist = g_global.require_group("hist")
                    g_hist.attrs["vmin"]  = float(hist["edges"][0])
                    g_hist.attrs["vmax"]  = float(hist["edges"][-1])
                    g_hist.attrs["nbins"] = int(len(hist["counts"]))
                    g_hist.create_dataset("counts", data=hist["counts"].astype(np.int32),
                                          compression="gzip", compression_opts=GZIP_LVL)

                    # ---- per_cell ----
                    g_cell = g_comp.require_group("per_cell")
                    for key in ("mean","std","min","max","range","q25","q50","q75"):
                        g_cell.create_dataset(key, data=content["per_cell"][key],
                                              compression="gzip", compression_opts=GZIP_LVL)

                    # —— summary row ——（用 global 的统计）
                    gs = content["global"]; hist_edges = gs["hist"]["edges"]
                    summary_rows.append([
                        str(p.resolve()), dtype, int(frame), cname,
                        res["N"], res["H"], res["W"], res["T"], res["H_fft"], res["Wc_fft"],
                        float(gs["mean"]), float(gs["std"]),
                        float(gs["min"]), float(gs["max"]),
                        float(gs["range"]), float(gs["q25"]), float(gs["q50"]), float(gs["q75"]),
                        float(hist_edges[0]), float(hist_edges[-1]), int(NBINS),
                        timing
                    ])

            print(f"  [DONE {idx}/{len(pt_files)}] elapsed: {_fmt_sec(timing)}")

        # 写入 summary/csv
        import io, csv
        buf = io.StringIO(); writer = csv.writer(buf); writer.writerow(header)
        for row in summary_rows: writer.writerow(row)
        csv_bytes = buf.getvalue().encode("utf-8")
        sum_grp.create_dataset("csv", data=np.frombuffer(csv_bytes, dtype=np.uint8))

    t_total1 = time.perf_counter()
    total_elapsed = t_total1 - t_total0
    # 每个文件两个帧 × 三个 component
    n_done = max(1, len(summary_rows) // (len(FRAMES) * len(COMPONENTS)))
    avg_elapsed = total_elapsed / n_done
    print(f"\n[OK] All FFT stats written to: {OUTPUT_H5.resolve()}")
    print(f"[SUMMARY] total elapsed: {_fmt_sec(total_elapsed)} | avg per dataset: {_fmt_sec(avg_elapsed)}")

if __name__ == "__main__":
    main()
