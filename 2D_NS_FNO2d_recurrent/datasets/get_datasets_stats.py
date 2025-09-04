# -*- coding: utf-8 -*-
"""
Compute BOTH spatial-domain and FFT-domain statistics for frames 0 and 19.

Outputs (two files in the same run):
  1) dataset_stats.h5
     /files/{safe_stem}/attrs (file_path,N,H,W,T,timing_sec,type)
     /files/{safe_stem}/frames/{frame}/global/{mean,std,min,max,range,q25,q50,q75}
     /files/{safe_stem}/frames/{frame}/global/hist/{counts} (attrs: vmin,vmax,nbins)
     /files/{safe_stem}/frames/{frame}/per_cell/{mean,std,min,max,range,q25,q50,q75}
     /summary/csv  (one row per file×frame, includes type)

  2) dataset_stats_fft.h5
     /files/{safe_stem}/attrs (file_path,N,H,W,T,timing_sec,type,H_fft,Wc_fft,fft_kind)
     /files/{safe_stem}/fft_axes/{ky,kx}
     /files/{safe_stem}/frames/{frame}/fft/{mag|real|imag}/global/{...}
     /files/{safe_stem}/frames/{frame}/fft/{mag|real|imag}/global/hist/{counts} (attrs: vmin,vmax,nbins)
     /files/{safe_stem}/frames/{frame}/fft/{mag|real|imag}/per_cell/{...}
     /summary/csv  (one row per file×frame×component; includes type, component)

Scanning is NON-RECURSIVE: only *.pt directly under each SEARCH_DIR is considered.
For 'generalizability' type, we use a FIXED SEED to randomly sample EXACTLY 100 datasets.
"""

from pathlib import Path
import os, time, re, random
import numpy as np
import torch
import h5py

# ---------------- user params ----------------
OUTPUT_H5_SPATIAL = Path("./dataset_stats_1440.h5")
OUTPUT_H5_FFT     = Path("./dataset_stats_fft_1440.h5")

FRAMES    = (0, 19)     # stats for frame 0 and 19
NBINS     = 100         # histogram bins
GZIP_LVL  = 4

# fixed sampling for generalizability (reproducible)
GEN_SAMPLE_SEED = 3407
GEN_SAMPLE_NUM  = 1440

# non-recursive search roots
SEARCH_DIRS = [
    "expanded_exponax_datasets/t20/N=1150",
    "exponax_datasets/t20/generalizability",
    "exponax_datasets/t20",
]
# ---------------------------------------------

COMPONENTS = ("mag", "real", "imag")

# for HPC parallel filesystems; also safe for local
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

def _sanitize_name(stem: str) -> str:
    return stem.split("|")[0]

def _infer_type(pt_path: Path) -> str:
    """
    Robust type inference with strong path precedence:
      1) If path contains '/generalizability/' -> 'generalizability'
      2) If path contains '/expanded_exponax_datasets/' or '/expanded/' -> 'expanded'
      3) Else (base datasets), use STRICT filename tokens like:
           '_train_all_frames' or suffix '_train'
           '_test_all_frames'  or suffix '_test'
         Avoids generic substring collisions (e.g., generalizability files with 'test_*' in name).
    """
    p = str(pt_path.resolve()).replace("\\", "/").lower()
    s = pt_path.stem.lower()

    # 1) path-based override
    if "/generalizability/" in p:
        return "generalizability"
    if "/expanded_exponax_datasets/" in p or "/expanded/" in p:
        return "expanded"

    # 2) strict filename patterns for base datasets
    # prefer the explicit all_frames tokens
    if "_test_all_frames" in s or s.endswith("_test"):
        return "test"
    if "_train_all_frames" in s or s.endswith("_train"):
        return "train"

    # as a final fallback, accept anchored tokens to avoid partial-word matches
    if re.search(r'(^|_)test($|_)', s):
        return "test"
    if re.search(r'(^|_)train($|_)', s):
        return "train"

    # default bucket (you can change to 'unknown' if you prefer)
    return "train"

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

def _per_cell_stats(A: np.ndarray, H: int, W: int):
    mean = np.nanmean(A, axis=0).reshape(H, W).astype(np.float32)
    std  = np.nanstd(A, axis=0).reshape(H, W).astype(np.float32)
    vmin = np.nanmin(A, axis=0).reshape(H, W).astype(np.float32)
    vmax = np.nanmax(A, axis=0).reshape(H, W).astype(np.float32)
    rng  = (vmax - vmin).astype(np.float32)
    q25  = np.quantile(A, 0.25, axis=0).reshape(H, W).astype(np.float32)
    q50  = np.quantile(A, 0.50, axis=0).reshape(H, W).astype(np.float32)
    q75  = np.quantile(A, 0.75, axis=0).reshape(H, W).astype(np.float32)
    return {"mean": mean, "std": std, "min": vmin, "max": vmax, "range": rng,
            "q25": q25, "q50": q50, "q75": q75}

def _fft_components(y_t: torch.Tensor):
    """
    y_t: (N,H,W) real tensor
    returns dict of (N,H,Wc) torch tensors: mag, real, imag
    """
    F = torch.fft.rfft2(y_t, dim=(1, 2))  # complex64/complex128
    return {
        "mag": torch.abs(F),
        "real": F.real,
        "imag": F.imag,
    }

def _process_one_loaded_tensor(y: torch.Tensor):
    """
    y: (N,H,W,T) torch tensor (float32/64 preferred)
    Returns:
      spatial: {frame->{global, per_cell}}
      fft:     {frame->{fft->{comp->{global,per_cell}}}}, plus H_fft,Wc_fft
    """
    if not torch.is_tensor(y) or y.ndim != 4 or y.shape[-1] < max(FRAMES) + 1:
        raise ValueError(f"y has wrong shape {tuple(y.shape)} (expect (N,H,W,T>=20))")

    # cast for stable FFT dtype if needed
    if y.dtype not in (torch.float32, torch.float64):
        y = y.float()

    N, H, W, T = map(int, y.shape)
    Wc = W // 2 + 1

    spatial_frames = {}
    fft_frames = {}

    with torch.no_grad():
        for t in FRAMES:
            y_t = y[..., t]  # (N,H,W)

            # spatial
            A_sp = y_t.contiguous().cpu().numpy().astype(np.float64, copy=False).reshape(N, H * W)
            sp_per_cell = _per_cell_stats(A_sp, H, W)
            sp_global   = _global_stats(A_sp, NBINS)
            spatial_frames[int(t)] = {"per_cell": sp_per_cell, "global": sp_global}

            # fft
            comps = _fft_components(y_t)  # each (N,H,Wc)
            per_comp = {}
            for cname, C in comps.items():
                C_np = C.contiguous().cpu().numpy().astype(np.float64, copy=False)
                A = C_np.reshape(N, H * Wc)
                per_comp[cname] = {
                    "per_cell": _per_cell_stats(A, H, Wc),
                    "global":   _global_stats(A, NBINS),
                }
            fft_frames[int(t)] = {"fft": per_comp}

    return {
        "N": N, "H": H, "W": W, "T": T,
        "spatial": spatial_frames,
        "fft":     fft_frames,
        "H_fft": H, "Wc_fft": Wc,
    }

def _collect_pt_files_nonrecursive(base_dir: Path):
    all_files = []
    total_count = 0
    print(f"[INFO] Scanning {len(SEARCH_DIRS)} dirs (non-recursive *.pt only):")
    for rel in SEARCH_DIRS:
        d = (base_dir / rel).resolve()
        if not d.exists():
            print(f"  [WARN] {d} (not found)")
            continue
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
    pt_files_all = _collect_pt_files_nonrecursive(base_dir)
    if not pt_files_all:
        print("[INFO] No .pt files found.")
        return

    # ---- classify & sample generalizability ----
    buckets = {"generalizability": [], "expanded": [], "train": [], "test": [], "other": []}
    for p in pt_files_all:
        t = _infer_type(p)
        if t in buckets:
            buckets[t].append(p)
        else:
            buckets["other"].append(p)

    # fixed-seed sampling for generalizability
    gen_all = buckets["generalizability"]
    if len(gen_all) > GEN_SAMPLE_NUM:
        rng = random.Random(GEN_SAMPLE_SEED)
        gen_pick = rng.sample(gen_all, GEN_SAMPLE_NUM)
        # sort to have stable processing order
        gen_pick = sorted(gen_pick, key=lambda x: str(x))
        print(f"\n[INFO] generalizability: {len(gen_all)} found -> sampling {GEN_SAMPLE_NUM} with seed={GEN_SAMPLE_SEED}")
        buckets["generalizability"] = gen_pick
    else:
        print(f"\n[INFO] generalizability: {len(gen_all)} found (<= {GEN_SAMPLE_NUM}), take all.")

    # selected files in a readable, grouped order
    selected = (
        sorted(buckets["expanded"], key=str) +
        sorted(buckets["train"], key=str) +
        sorted(buckets["test"], key=str) +
        sorted(buckets["generalizability"], key=str) +
        sorted(buckets["other"], key=str)
    )
    if not selected:
        print("[INFO] No files selected after filtering.")
        return

    print(f"[INFO] Total selected files: {len(selected)}")
    # --------------------------------------------

    # Prepare summaries
    header_spatial = ["file", "type", "frame", "N", "H", "W", "T",
                      "mean", "std", "min", "max", "range", "q25", "q50", "q75",
                      "hist_vmin", "hist_vmax", "hist_nbins", "timing_sec"]
    header_fft = ["file", "type", "frame", "component",
                  "N", "H", "W", "T", "H_fft", "Wc_fft",
                  "mean", "std", "min", "max", "range", "q25", "q50", "q75",
                  "hist_vmin", "hist_vmax", "hist_nbins", "timing_sec"]

    summary_rows_sp = []
    summary_rows_fft = []

    t0_total = time.perf_counter()

    # open both H5 outputs
    with h5py.File(OUTPUT_H5_SPATIAL, "w") as h5_sp, h5py.File(OUTPUT_H5_FFT, "w") as h5_fft:
        # roots
        h5_sp.create_group("files")
        gsum_sp = h5_sp.create_group("summary")

        h5_fft.create_group("files")
        gsum_fft = h5_fft.create_group("summary")
        gmeta_fft = h5_fft.create_group("meta")
        gmeta_fft.attrs["fft_kind"] = "rfft2"

        # process files
        for idx, p in enumerate(selected, 1):
            print(f"\n[PROCESS {idx}/{len(selected)}] {p}")
            dtype = _infer_type(p)
            safe_stem = _sanitize_name(p.stem)

            t_load0 = time.perf_counter()
            try:
                payload = torch.load(p, map_location="cpu", weights_only=False)
                if not isinstance(payload, dict) or "y" not in payload:
                    print(f"  [SKIP] {p.name}: not a dict or missing key 'y'")
                    continue
                y = payload["y"]
                res = _process_one_loaded_tensor(y)
            except Exception as e:
                print(f"  [SKIP] {p.name}: {e}")
                continue
            t_load1 = time.perf_counter()
            timing = float(t_load1 - t_load0)

            # ---------- write spatial ----------
            fgrp_sp = h5_sp.require_group(f"files/{safe_stem}")
            fgrp_sp.attrs["file_path"]  = str(p.resolve())
            fgrp_sp.attrs["N"]          = res["N"]
            fgrp_sp.attrs["H"]          = res["H"]
            fgrp_sp.attrs["W"]          = res["W"]
            fgrp_sp.attrs["T"]          = res["T"]
            fgrp_sp.attrs["timing_sec"] = timing
            fgrp_sp.attrs["type"]       = dtype

            for frame, content in res["spatial"].items():
                g_frame = fgrp_sp.require_group(f"frames/{frame}")

                # global
                g_global = g_frame.require_group("global")
                for k in ("mean","std","min","max","range","q25","q50","q75"):
                    g_global.create_dataset(k, data=np.array(content["global"][k], dtype=np.float32))
                hist = content["global"]["hist"]
                g_hist = g_global.require_group("hist")
                g_hist.attrs["vmin"]  = float(hist["edges"][0])
                g_hist.attrs["vmax"]  = float(hist["edges"][-1])
                g_hist.attrs["nbins"] = int(len(hist["counts"]))
                g_hist.create_dataset("counts", data=hist["counts"].astype(np.int32),
                                      compression="gzip", compression_opts=GZIP_LVL)

                # per_cell
                g_cell = g_frame.require_group("per_cell")
                for k in ("mean","std","min","max","range","q25","q50","q75"):
                    g_cell.create_dataset(k, data=content["per_cell"][k],
                                          compression="gzip", compression_opts=GZIP_LVL)

                gs = content["global"]; hedges = gs["hist"]["edges"]
                summary_rows_sp.append([
                    str(p.resolve()), dtype, int(frame),
                    res["N"], res["H"], res["W"], res["T"],
                    float(gs["mean"]), float(gs["std"]),
                    float(gs["min"]), float(gs["max"]),
                    float(gs["range"]), float(gs["q25"]), float(gs["q50"]), float(gs["q75"]),
                    float(hedges[0]), float(hedges[-1]), int(NBINS),
                    timing
                ])

            # ---------- write fft ----------
            fgrp_fft = h5_fft.require_group(f"files/{safe_stem}")
            fgrp_fft.attrs["file_path"]  = str(p.resolve())
            fgrp_fft.attrs["N"]          = res["N"]
            fgrp_fft.attrs["H"]          = res["H"]
            fgrp_fft.attrs["W"]          = res["W"]
            fgrp_fft.attrs["T"]          = res["T"]
            fgrp_fft.attrs["timing_sec"] = timing
            fgrp_fft.attrs["type"]       = dtype
            fgrp_fft.attrs["H_fft"]      = res["H_fft"]
            fgrp_fft.attrs["Wc_fft"]     = res["Wc_fft"]
            fgrp_fft.attrs["fft_kind"]   = "rfft2"

            # fft axes (ky full, kx rfftfreq)
            ky = np.fft.fftfreq(res["H_fft"], d=1.0).astype(np.float64)
            kx = np.fft.rfftfreq(res["W"],     d=1.0).astype(np.float64)
            g_axes = fgrp_fft.require_group("fft_axes")
            if "ky" not in g_axes:
                g_axes.create_dataset("ky", data=ky)
            if "kx" not in g_axes:
                g_axes.create_dataset("kx", data=kx)

            for frame, fcontent in res["fft"].items():
                g_frame = fgrp_fft.require_group(f"frames/{frame}")
                g_fft   = g_frame.require_group("fft")
                for cname in COMPONENTS:
                    cstat = fcontent["fft"][cname]
                    g_comp = g_fft.require_group(cname)

                    # global
                    g_global = g_comp.require_group("global")
                    for k in ("mean","std","min","max","range","q25","q50","q75"):
                        g_global.create_dataset(k, data=np.array(cstat["global"][k], dtype=np.float32))
                    hist = cstat["global"]["hist"]
                    g_hist = g_global.require_group("hist")
                    g_hist.attrs["vmin"]  = float(hist["edges"][0])
                    g_hist.attrs["vmax"]  = float(hist["edges"][-1])
                    g_hist.attrs["nbins"] = int(len(hist["counts"]))
                    g_hist.create_dataset("counts", data=hist["counts"].astype(np.int32),
                                          compression="gzip", compression_opts=GZIP_LVL)

                    # per_cell
                    g_cell = g_comp.require_group("per_cell")
                    for k in ("mean","std","min","max","range","q25","q50","q75"):
                        g_cell.create_dataset(k, data=cstat["per_cell"][k],
                                              compression="gzip", compression_opts=GZIP_LVL)

                    gs = cstat["global"]; hedges = gs["hist"]["edges"]
                    summary_rows_fft.append([
                        str(p.resolve()), dtype, int(frame), cname,
                        res["N"], res["H"], res["W"], res["T"], res["H_fft"], res["Wc_fft"],
                        float(gs["mean"]), float(gs["std"]),
                        float(gs["min"]), float(gs["max"]),
                        float(gs["range"]), float(gs["q25"]), float(gs["q50"]), float(gs["q75"]),
                        float(hedges[0]), float(hedges[-1]), int(NBINS),
                        timing
                    ])

            print(f"  [DONE {idx}/{len(selected)}] elapsed: { _fmt_sec(timing) }")

        # ---- write summaries ----
        import io, csv

        buf_sp = io.StringIO(); w_sp = csv.writer(buf_sp); w_sp.writerow(header_spatial)
        for row in summary_rows_sp: w_sp.writerow(row)
        h5_sp["summary"].create_dataset("csv", data=np.frombuffer(buf_sp.getvalue().encode("utf-8"), dtype=np.uint8))

        buf_fft = io.StringIO(); w_fft = csv.writer(buf_fft); w_fft.writerow(header_fft)
        for row in summary_rows_fft: w_fft.writerow(row)
        h5_fft["summary"].create_dataset("csv", data=np.frombuffer(buf_fft.getvalue().encode("utf-8"), dtype=np.uint8))

    t1_total = time.perf_counter()
    total_elapsed = t1_total - t0_total
    # for reporting: average per *dataset* (not per frame or component)
    n_datasets = max(1, len(selected))
    avg_elapsed = total_elapsed / n_datasets

    print(f"\n[OK] Spatial stats written to: {OUTPUT_H5_SPATIAL.resolve()}")
    print(f"[OK] FFT     stats written to: {OUTPUT_H5_FFT.resolve()}")
    print(f"[SUMMARY] total elapsed: { _fmt_sec(total_elapsed) } | avg per dataset: { _fmt_sec(avg_elapsed) }")

def _fmt_sec(s: float) -> str:
    m, sec = divmod(s, 60); h, m = divmod(m, 60)
    return f"{int(h)}:{int(m):02d}:{sec:06.3f}"

if __name__ == "__main__":
    main()



