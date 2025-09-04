# -*- coding: utf-8 -*-
"""
Compute statistics for DELTA tensors: y_delta = y_expanded - y_train
Only processes datasets of type 'expanded', and writes TWO H5 files:
  1) dataset_stats_expanded_diff.h5            (spatial stats)
  2) dataset_stats_fft_expanded_diff.h5        (FFT stats)

H5 layout is the same as the original scripts, so downstream plotting can
reuse the same readers. Attributes mark that values are DIFF:
  - group.attrs['stat_kind'] = 'expanded_minus_train'
  - group.attrs['paired_train'] = '<path to matched train .pt>'
  - type attribute remains 'expanded' for compatibility.
"""

from pathlib import Path
import os, time, re, random
import numpy as np
import torch
import h5py

# ---------------- user params ----------------
OUTPUT_H5_SPATIAL = Path("./dataset_stats_expanded_diff.h5")
OUTPUT_H5_FFT     = Path("./dataset_stats_fft_expanded_diff.h5")

FRAMES    = (0, 19)     # stats for frame 0 and 19
NBINS     = 100         # histogram bins
GZIP_LVL  = 4

# fixed sampling for generalizability (kept but unused here since we only do expanded)
GEN_SAMPLE_SEED = 3407
GEN_SAMPLE_NUM  = 100

# non-recursive search roots
SEARCH_DIRS = [
    "expanded_exponax_datasets/t20/N=1150",
    "exponax_datasets/t20",                       # base train/test live here
    "exponax_datasets/t20/generalizability",
]
# ---------------------------------------------

COMPONENTS = ("mag", "real", "imag")

# for HPC parallel filesystems; also safe for local
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

def _sanitize_name(stem: str) -> str:
    # keep name stable for /files/<safe_stem>
    return stem.split("|")[0]

def _infer_type(pt_path: Path) -> str:
    p = str(pt_path.resolve()).replace("\\", "/").lower()
    s = pt_path.stem.lower()
    if "/generalizability/" in p:
        return "generalizability"
    if "/expanded_exponax_datasets/" in p or "/expanded/" in p:
        return "expanded"
    if "_test_all_frames" in s or s.endswith("_test"):
        return "test"
    if "_train_all_frames" in s or s.endswith("_train"):
        return "train"
    if re.search(r'(^|_)test($|_)', s):
        return "test"
    if re.search(r'(^|_)train($|_)', s):
        return "train"
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
    y: (N,H,W,T) torch tensor
    Returns: spatial + fft stats dict for FRAMES.
    """
    if not torch.is_tensor(y) or y.ndim != 4 or y.shape[-1] < max(FRAMES) + 1:
        raise ValueError(f"y has wrong shape {tuple(y.shape)} (expect (N,H,W,T>=20))")

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

# --------- file discovery & pairing ---------
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
        print(f"  [DIR] {d}  |  {len(files)} files")
        all_files.extend(files)
    print(f"[INFO] Found {total_count} .pt files across {len(SEARCH_DIRS)} dirs.")
    return all_files

def _build_indices(pt_files):
    """Return dicts for quick matching."""
    by_stem = {}
    trains = {}
    expanded = {}
    for p in pt_files:
        s = _sanitize_name(p.stem)
        by_stem.setdefault(s, []).append(p)
        t = _infer_type(p)
        if t == "train":
            trains[s] = p
        elif t == "expanded":
            expanded[s] = p
    return by_stem, trains, expanded

def _guess_train_for_expanded(exp_stem: str, trains_by_stem: dict):
    """Heuristic: strip '_modespec=' tail; try exact, startswith/contains."""
    base = re.split(r"_modespec=", exp_stem, maxsplit=1)[0]
    # exact
    if base in trains_by_stem: return trains_by_stem[base]
    # startswith either way
    for s, p in trains_by_stem.items():
        if s.startswith(base) or base.startswith(s):
            return p
    # contains
    for s, p in trains_by_stem.items():
        if (base in s) or (s in exp_stem):
            return p
    return None

# -------------- main --------------
def _fmt_sec(s: float) -> str:
    m, sec = divmod(s, 60); h, m = divmod(m, 60)
    return f"{int(h)}:{int(m):02d}:{sec:06.3f}"

def main():
    base_dir = Path(__file__).parent.resolve()
    pt_files_all = _collect_pt_files_nonrecursive(base_dir)
    if not pt_files_all:
        print("[INFO] No .pt files found.")
        return

    by_stem, trains_by_stem, expanded_by_stem = _build_indices(pt_files_all)
    expanded_stems = sorted(expanded_by_stem.keys())
    print(f"[INFO] Expanded candidates: {len(expanded_stems)}")

    # open both H5 outputs
    with h5py.File(OUTPUT_H5_SPATIAL, "w") as h5_sp, h5py.File(OUTPUT_H5_FFT, "w") as h5_fft:
        # roots
        g_files_sp = h5_sp.create_group("files")
        gsum_sp    = h5_sp.create_group("summary")
        h5_sp.attrs["stat_kind"] = "expanded_minus_train"

        g_files_fft = h5_fft.create_group("files")
        gsum_fft    = h5_fft.create_group("summary")
        gmeta_fft   = h5_fft.create_group("meta")
        gmeta_fft.attrs["fft_kind"]  = "rfft2"
        h5_fft.attrs["stat_kind"]    = "expanded_minus_train"

        # summaries
        import io, csv
        header_spatial = ["file_expanded", "file_train", "type", "frame",
                          "N","H","W","T",
                          "mean","std","min","max","range","q25","q50","q75",
                          "hist_vmin","hist_vmax","hist_nbins","timing_sec","stat_kind"]
        header_fft = ["file_expanded","file_train","type","frame","component",
                      "N","H","W","T","H_fft","Wc_fft",
                      "mean","std","min","max","range","q25","q50","q75",
                      "hist_vmin","hist_vmax","hist_nbins","timing_sec","stat_kind"]
        buf_sp = io.StringIO(); w_sp = csv.writer(buf_sp); w_sp.writerow(header_spatial)
        buf_fft = io.StringIO(); w_fft = csv.writer(buf_fft); w_fft.writerow(header_fft)

        # process each expanded
        for idx, exp_stem in enumerate(expanded_stems, 1):
            p_exp = expanded_by_stem[exp_stem]
            p_tr  = _guess_train_for_expanded(exp_stem, trains_by_stem)
            print(f"\n[PROCESS {idx}/{len(expanded_stems)}] expanded={p_exp.name}")
            if p_tr is None:
                print("  [SKIP] matched train not found.")
                continue

            print(f"  pair train={p_tr.name}")

            # load tensors
            t0 = time.perf_counter()
            try:
                payE = torch.load(p_exp, map_location="cpu", weights_only=False)
                payT = torch.load(p_tr,  map_location="cpu", weights_only=False)
                if not (isinstance(payE, dict) and "y" in payE and isinstance(payT, dict) and "y" in payT):
                    print("  [SKIP] payload missing 'y'")
                    continue
                yE = payE["y"]; yT = payT["y"]
                if yE.dtype not in (torch.float32, torch.float64): yE = yE.float()
                if yT.dtype not in (torch.float32, torch.float64): yT = yT.float()
                if yE.shape != yT.shape:
                    print(f"  [SKIP] shape mismatch expanded{tuple(yE.shape)} vs train{tuple(yT.shape)}")
                    continue
                y_delta = yE - yT
                res = _process_one_loaded_tensor(y_delta)
            except Exception as e:
                print(f"  [SKIP] failed: {e}")
                continue
            t1 = time.perf_counter()
            timing = float(t1 - t0)

            safe_stem = _sanitize_name(p_exp.stem)

            # ---------- write spatial ----------
            fgrp_sp = g_files_sp.require_group(safe_stem)
            fgrp_sp.attrs["file_path"]        = str(p_exp.resolve())
            fgrp_sp.attrs["paired_train"]     = str(p_tr.resolve())
            fgrp_sp.attrs["N"]                = res["N"]
            fgrp_sp.attrs["H"]                = res["H"]
            fgrp_sp.attrs["W"]                = res["W"]
            fgrp_sp.attrs["T"]                = res["T"]
            fgrp_sp.attrs["timing_sec"]       = timing
            fgrp_sp.attrs["type"]             = "expanded"        # keep for compatibility
            fgrp_sp.attrs["stat_kind"]        = "expanded_minus_train"

            for frame, content in res["spatial"].items():
                g_frame = fgrp_sp.require_group(f"frames/{frame}")

                # global
                g_global = g_frame.require_group("global")
                g_global.attrs["stat_kind"] = "expanded_minus_train"
                for k in ("mean","std","min","max","range","q25","q50","q75"):
                    _create_or_overwrite(g_global, k, np.array(content["global"][k], dtype=np.float32))
                hist = content["global"]["hist"]
                g_hist = g_global.require_group("hist")
                g_hist.attrs["vmin"]  = float(hist["edges"][0])
                g_hist.attrs["vmax"]  = float(hist["edges"][-1])
                g_hist.attrs["nbins"] = int(len(hist["counts"]))
                _create_or_overwrite(g_hist, "counts",
                                     hist["counts"].astype(np.int32),
                                     compression="gzip", compression_opts=GZIP_LVL)

                # per_cell
                g_cell = g_frame.require_group("per_cell")
                g_cell.attrs["stat_kind"] = "expanded_minus_train"
                for k in ("mean","std","min","max","range","q25","q50","q75"):
                    _create_or_overwrite(g_cell, k, content["per_cell"][k],
                                         compression="gzip", compression_opts=GZIP_LVL)

                gs = content["global"]; hedges = gs["hist"]["edges"]
                w_sp.writerow([
                    str(p_exp.resolve()), str(p_tr.resolve()), "expanded", int(frame),
                    res["N"], res["H"], res["W"], res["T"],
                    float(gs["mean"]), float(gs["std"]),
                    float(gs["min"]), float(gs["max"]),
                    float(gs["range"]), float(gs["q25"]), float(gs["q50"]), float(gs["q75"]),
                    float(hedges[0]), float(hedges[-1]), int(NBINS),
                    timing, "expanded_minus_train"
                ])

            # ---------- write fft ----------
            fgrp_fft = g_files_fft.require_group(safe_stem)
            fgrp_fft.attrs["file_path"]        = str(p_exp.resolve())
            fgrp_fft.attrs["paired_train"]     = str(p_tr.resolve())
            fgrp_fft.attrs["N"]                = res["N"]
            fgrp_fft.attrs["H"]                = res["H"]
            fgrp_fft.attrs["W"]                = res["W"]
            fgrp_fft.attrs["T"]                = res["T"]
            fgrp_fft.attrs["timing_sec"]       = timing
            fgrp_fft.attrs["type"]             = "expanded"
            fgrp_fft.attrs["H_fft"]            = res["H_fft"]
            fgrp_fft.attrs["Wc_fft"]           = res["Wc_fft"]
            fgrp_fft.attrs["fft_kind"]         = "rfft2"
            fgrp_fft.attrs["stat_kind"]        = "expanded_minus_train"

            # fft axes (ky full, kx rfftfreq)
            ky = np.fft.fftfreq(res["H_fft"], d=1.0).astype(np.float64)
            kx = np.fft.rfftfreq(res["W"],     d=1.0).astype(np.float64)
            g_axes = fgrp_fft.require_group("fft_axes")
            if "ky" not in g_axes: g_axes.create_dataset("ky", data=ky)
            if "kx" not in g_axes: g_axes.create_dataset("kx", data=kx)

            for frame, fcontent in res["fft"].items():
                g_frame = fgrp_fft.require_group(f"frames/{frame}")
                g_fft   = g_frame.require_group("fft")
                for cname in COMPONENTS:
                    cstat = fcontent["fft"][cname]
                    g_comp = g_fft.require_group(cname)

                    # global
                    g_global = g_comp.require_group("global")
                    g_global.attrs["stat_kind"] = "expanded_minus_train"
                    for k in ("mean","std","min","max","range","q25","q50","q75"):
                        _create_or_overwrite(g_global, k, np.array(cstat["global"][k], dtype=np.float32))
                    hist = cstat["global"]["hist"]
                    g_hist = g_global.require_group("hist")
                    g_hist.attrs["vmin"]  = float(hist["edges"][0])
                    g_hist.attrs["vmax"]  = float(hist["edges"][-1])
                    g_hist.attrs["nbins"] = int(len(hist["counts"]))
                    _create_or_overwrite(g_hist, "counts",
                                         hist["counts"].astype(np.int32),
                                         compression="gzip", compression_opts=GZIP_LVL)

                    # per_cell
                    g_cell = g_comp.require_group("per_cell")
                    g_cell.attrs["stat_kind"] = "expanded_minus_train"
                    for k in ("mean","std","min","max","range","q25","q50","q75"):
                        _create_or_overwrite(g_cell, k, cstat["per_cell"][k],
                                             compression="gzip", compression_opts=GZIP_LVL)

                    gs = cstat["global"]; hedges = gs["hist"]["edges"]
                    w_fft.writerow([
                        str(p_exp.resolve()), str(p_tr.resolve()), "expanded", int(frame), cname,
                        res["N"], res["H"], res["W"], res["T"], res["H_fft"], res["Wc_fft"],
                        float(gs["mean"]), float(gs["std"]),
                        float(gs["min"]), float(gs["max"]),
                        float(gs["range"]), float(gs["q25"]), float(gs["q50"]), float(gs["q75"]),
                        float(hedges[0]), float(hedges[-1]), int(NBINS),
                        timing, "expanded_minus_train"
                    ])

            print(f"  [DONE] elapsed: { _fmt_sec(timing) }")

        # write summaries
        h5_sp["summary"].create_dataset(
            "csv", data=np.frombuffer(buf_sp.getvalue().encode("utf-8"), dtype=np.uint8))
        h5_fft["summary"].create_dataset(
            "csv", data=np.frombuffer(buf_fft.getvalue().encode("utf-8"), dtype=np.uint8))

    print(f"\n[OK] Spatial (DIFF) written to: {OUTPUT_H5_SPATIAL.resolve()}")
    print(f"[OK] FFT     (DIFF) written to: {OUTPUT_H5_FFT.resolve()}")

def _create_or_overwrite(group, name, data, **kw):
    if name in group: del group[name]
    group.create_dataset(name, data=data, **kw)

if __name__ == "__main__":
    main()




