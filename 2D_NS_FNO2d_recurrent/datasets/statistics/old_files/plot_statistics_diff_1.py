# -*- coding: utf-8 -*-
"""
Draw Expanded-minus-Train (per-cell stats diff) from stats HDF5:
  For frames 0 and 19:
    Spatial (diff = expanded - train):
      - Global histogram of per-cell mean diff (all cells)
      - Per-cell heatmaps: mean, std, min, max, q25, q50, q75  (each is diff map)
    FFT (components if available: mag | real | imag), also diff at per-cell level:
      - Histogram of per-cell mean diff
      - Per-cell heatmaps (linear): mean, std, min, max, q25, q50, q75
      - Per-cell heatmaps (log10|.|) for the same set
Outputs live under:
  <OUT_DIR>/expanded/<expanded_dataset>/spatial_diff/*.png
  <OUT_DIR>/expanded/<expanded_dataset>/fourier_diff/{mag,real,imag}/*.png
"""

from pathlib import Path
import os, time, re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple

# ======= user params =======
SCRIPT_DIR = Path(__file__).parent.resolve()

def find_nearby_file(name: str) -> Path:
    p1 = (SCRIPT_DIR / name)
    if p1.exists():
        return p1
    return Path.cwd() / name

SPATIAL_H5 = find_nearby_file("dataset_stats.h5")
FFT_H5     = find_nearby_file("dataset_stats_fft.h5")

OUT_DIR    = SCRIPT_DIR / "dataset_stats_plots_by_type"
FRAMES     = (0, 19)
DPI        = 150

# Only draw expanded (and pair with train)
CATEGORIES = ("generalizability", "expanded", "train", "test")
# ==========================

# ---------- utils ----------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _fmt(v: float) -> str:
    return f"{v:.4g}"

def _safe_get(g: Optional[h5py.Group], key: str):
    return None if g is None else g.get(key, None)

def _attr_int(fgrp: h5py.Group, k: str, default: int = -1) -> int:
    try:
        return int(fgrp.attrs.get(k, default))
    except Exception:
        return default

def _attr_str(fgrp: h5py.Group, k: str, default: str = "") -> str:
    try:
        v = fgrp.attrs.get(k, default)
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="ignore")
        return str(v)
    except Exception:
        return default

def _build_ds_title(dtype: str, ds_name: str, fgrp: h5py.Group, *, is_fft: bool) -> str:
    N = _attr_int(fgrp, "N")
    H = _attr_int(fgrp, "H")
    W = _attr_int(fgrp, "W")
    T = _attr_int(fgrp, "T")
    lines = [f"[{dtype}] {ds_name}", f"N={N}, H={H}, W={W}, T={T}"]
    if is_fft:
        Hf   = _attr_int(fgrp, "H_fft")
        Wcf  = _attr_int(fgrp, "Wc_fft")
        kind = _attr_str(fgrp, "fft_kind", "rfft2")
        lines.append(f"H_fft={Hf}, Wc_fft={Wcf}, kind={kind}")
    return "\n".join(lines)

# ---------- plotting helpers ----------
def _imshow2d(ax, arr, title):
    im = ax.imshow(arr, origin="lower", interpolation="nearest")
    ax.set_title(title, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

def _imshow2d_log(ax, arr, title, eps: float = 1e-12):
    arr_log = np.log10(np.abs(arr) + eps)
    im = ax.imshow(arr_log, origin="lower", interpolation="nearest")
    ax.set_title(title + " (log10|.|)", pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

def _mirror_full_spectrum(arr_half: np.ndarray, full_W: int) -> np.ndarray:
    if arr_half.ndim != 2:
        raise ValueError(f"expect 2D array (H, Wc), got shape {arr_half.shape}")
    H, Wc = arr_half.shape
    if Wc == full_W or full_W <= 0 or Wc > full_W:
        return arr_half
    if full_W % 2 == 0:
        tail = arr_half[:, -2:0:-1] if Wc >= 2 else arr_half[:, :0]
    else:
        tail = arr_half[:, -1:0:-1] if Wc >= 1 else arr_half[:, :0]
    return np.concatenate([arr_half, tail], axis=1)

def _plot_hist_from_array(out_png: Path, frame: int, arr: np.ndarray, title_prefix: str,
                          *, nbins: int = 200):
    mean = float(np.mean(arr))
    std  = float(np.std(arr))
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if out_png.exists():
        print(f"    [skip exists] {out_png.name}")
        return
    counts, edges = np.histogram(arr.ravel(), bins=nbins, range=(vmin, vmax))
    centers = 0.5*(edges[:-1] + edges[1:])
    widths  = (edges[1:] - edges[:-1])

    fig = plt.figure(figsize=(8,5))
    ax = plt.gca()
    ax.bar(centers, counts, width=widths, align="center", edgecolor="none")
    ax.set_xlabel("Value"); ax.set_ylabel("Count")
    title = (f"{title_prefix}\n"
             f"Frame {frame} | Global Histogram of per-cell mean DIFF\n"
             f"mean={_fmt(mean)}  std={_fmt(std)}  range=[{_fmt(vmin)}, {_fmt(vmax)}] (Δ={_fmt(vmax-vmin)})")
    ax.set_title(title); ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {out_png.name}")

def _plot_per_cell_maps_from_arrays(out_png1: Path, out_png2: Path, frame: int,
                                    stat: Dict[str, np.ndarray], title_prefix: str,
                                    *, log_scale: bool = False, mirror_full_W: Optional[int] = None):
    tag = "_log" if log_scale else ""
    out1 = out_png1.with_name(out_png1.stem + tag + out_png1.suffix)
    out2 = out_png2.with_name(out_png2.stem + tag + out_png2.suffix)
    need_1 = not out1.exists()
    need_2 = not out2.exists()
    if not (need_1 or need_2):
        print(f"    [skip exists] {out1.name}, {out2.name}")
        return

    mean = np.array(stat["mean"]); std  = np.array(stat["std"])
    vmin = np.array(stat["min"]);  vmax = np.array(stat["max"])
    q25  = np.array(stat["q25"]);  q50  = np.array(stat["q50"]); q75 = np.array(stat["q75"])

    if mirror_full_W is not None:
        mean = _mirror_full_spectrum(mean, mirror_full_W)
        std  = _mirror_full_spectrum(std,  mirror_full_W)
        vmin = _mirror_full_spectrum(vmin, mirror_full_W)
        vmax = _mirror_full_spectrum(vmax, mirror_full_W)
        q25  = _mirror_full_spectrum(q25,  mirror_full_W)
        q50  = _mirror_full_spectrum(q50,  mirror_full_W)
        q75  = _mirror_full_spectrum(q75,  mirror_full_W)

    imshow_func = _imshow2d_log if log_scale else _imshow2d
    head = f"{title_prefix}\nFrame {frame}"

    if need_1:
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
        imshow_func(axes1[0,0], mean, "Per-cell mean (DIFF)")
        imshow_func(axes1[0,1], std,  "Per-cell std (DIFF)")
        imshow_func(axes1[1,0], vmin, "Per-cell min (DIFF)")
        imshow_func(axes1[1,1], vmax, "Per-cell max (DIFF)")
        fig1.suptitle(f"{head}\nPer-cell statistics (mean/std/min/max) DIFF", y=0.995, fontsize=12)
        fig1.tight_layout(rect=[0,0,1,0.98])
        fig1.savefig(out1, dpi=DPI, bbox_inches="tight")
        plt.close(fig1)
        print(f"    [saved] {out1.name}")

    if need_2:
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        imshow_func(axes2[0], q25, "Per-cell q25 (DIFF)")
        imshow_func(axes2[1], q50, "Per-cell q50 (DIFF)")
        imshow_func(axes2[2], q75, "Per-cell q75 (DIFF)")
        fig2.suptitle(f"{head}\nPer-cell quantiles DIFF{tag}", y=0.98, fontsize=12)
        fig2.tight_layout(rect=[0,0,1,0.95])
        fig2.savefig(out2, dpi=DPI, bbox_inches="tight")
        plt.close(fig2)
        print(f"    [saved] {out2.name}")

# ---------- access helpers ----------
def _collect_names_and_types(h5f: h5py.File) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if "files" not in h5f:
        return out
    for name in h5f["files"].keys():
        fgrp = h5f["files"][name]
        dtype = str(fgrp.attrs.get("type", "train"))
        if dtype not in CATEGORIES:
            dtype = "train"
        out[name] = dtype
    return out

def _frames_group(fgrp: h5py.Group, frame: int) -> Optional[h5py.Group]:
    return _safe_get(_safe_get(fgrp, "frames"), str(frame))

def _get_per_cell_group_spatial(fgrp: h5py.Group, frame: int) -> Optional[h5py.Group]:
    return _safe_get(_frames_group(fgrp, frame), "per_cell")

def _get_fft_comp_groups(fgrp: h5py.Group, frame: int, comp_key: str) -> Tuple[Optional[h5py.Group], Optional[h5py.Group]]:
    """Return (global_group, per_cell_group) for FFT component, compatible with new/old layouts."""
    g_frame = _frames_group(fgrp, frame)
    if g_frame is None:
        return None, None
    # NEW layout: /frames/t/fft/<comp>/{global,per_cell}
    g_fft_root = _safe_get(g_frame, "fft")
    if g_fft_root is not None:
        g_comp = _safe_get(g_fft_root, comp_key)
        if g_comp is not None:
            return _safe_get(g_comp, "global"), _safe_get(g_comp, "per_cell")
    # OLD-A: /frames/t/<comp>/{global,per_cell}
    g_comp_a = _safe_get(g_frame, comp_key)
    if g_comp_a is not None:
        g_global = _safe_get(g_comp_a, "global")
        g_cell   = _safe_get(g_comp_a, "per_cell")
        if g_global is not None or g_cell is not None:
            return g_global, g_cell
    # OLD-B: /frames/t/global_<comp>, per_cell_<comp>
    return _safe_get(g_frame, f"global_{comp_key}"), _safe_get(g_frame, f"per_cell_{comp_key}")

def _group_to_stat_dict(g_cell: h5py.Group) -> Dict[str, np.ndarray]:
    return {
        "mean": np.array(g_cell["mean"]),
        "std":  np.array(g_cell["std"]),
        "min":  np.array(g_cell["min"]),
        "max":  np.array(g_cell["max"]),
        "q25":  np.array(g_cell["q25"]),
        "q50":  np.array(g_cell["q50"]),
        "q75":  np.array(g_cell["q75"]),
    }

def _stat_diff(A: Dict[str, np.ndarray], B: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.array(A[k]) - np.array(B[k]) for k in A.keys()}

def _guess_train_name(h5f: h5py.File, name_exp: str) -> Optional[str]:
    """Heuristics: drop suffix from '_modespec=' onward; try startswith/contains matches."""
    files = h5f.get("files", None)
    if files is None:
        return None
    base = re.split(r"_modespec=", name_exp, maxsplit=1)[0]
    trains = [n for n in files.keys() if str(files[n].attrs.get("type", "")) == "train"]
    # exact
    for n in trains:
        if n == base:
            return n
    # startswith either way
    for n in trains:
        if n.startswith(base) or base.startswith(n):
            return n
    # contains
    for n in trains:
        if (base in n) or (n in name_exp):
            return n
    return None

# ---------- drawing (expanded - train) ----------
def _plot_expanded_minus_train_spatial(h5f: h5py.File, name_exp: str, name_tr: str):
    fexp = h5f["files"][name_exp]
    ftr  = h5f["files"][name_tr]
    base_out = OUT_DIR / "expanded" / name_exp / "spatial_diff"
    _ensure_dir(base_out)
    print(f"  [spatial DIFF] -> {base_out}")

    ds_title = _build_ds_title("expanded", name_exp, fexp, is_fft=False)
    ds_title += f"\nDIFF: expanded - train ({name_tr})\nSpatial"

    for frame in FRAMES:
        gE = _get_per_cell_group_spatial(fexp, frame)
        gT = _get_per_cell_group_spatial(ftr,  frame)
        if gE is None or gT is None:
            print(f"    [warn] per_cell stats missing (frame {frame}); skip.")
            continue

        statE = _group_to_stat_dict(gE)
        statT = _group_to_stat_dict(gT)
        diff  = _stat_diff(statE, statT)

        # Histogram of per-cell mean diff
        hist_png  = base_out / f"frame{frame}_global_hist_percell_mean_diff.png"
        _plot_hist_from_array(hist_png, frame, diff["mean"], ds_title)

        # Per-cell maps
        maps1_png = base_out / f"frame{frame}_percell_mean_std_min_max_diff.png"
        maps2_png = base_out / f"frame{frame}_percell_quantiles_diff.png"
        _plot_per_cell_maps_from_arrays(maps1_png, maps2_png, frame, diff,
                                        title_prefix=ds_title, log_scale=False)

def _plot_expanded_minus_train_fft(h5f: h5py.File, name_exp: str, name_tr: str):
    fexp = h5f["files"][name_exp]
    ftr  = h5f["files"][name_tr]
    root_out = OUT_DIR / "expanded" / name_exp / "fourier_diff"
    _ensure_dir(root_out)
    print(f"  [fft DIFF] -> {root_out}")

    full_W = _attr_int(fexp, "W", -1)  # mirror target
    ds_title_head = _build_ds_title("expanded", name_exp, fexp, is_fft=True)
    ds_title_head += f"\nDIFF: expanded - train ({name_tr})"

    for frame in FRAMES:
        found_any = False
        for comp_key, comp_tag in (("mag", "mag"), ("real","real"), ("imag","imag")):
            gEg, gEc = _get_fft_comp_groups(fexp, frame, comp_key)
            gTg, gTc = _get_fft_comp_groups(ftr,  frame, comp_key)
            if gEc is None or gTc is None:
                continue
            found_any = True

            comp_out = root_out / comp_tag
            _ensure_dir(comp_out)

            # per-cell diff
            statE = _group_to_stat_dict(gEc)
            statT = _group_to_stat_dict(gTc)
            diff  = _stat_diff(statE, statT)

            prefix = f"{ds_title_head}\nFFT-{comp_tag}"

            # Histogram of per-cell mean diff (mirror to full W for display consistency)
            mean_for_hist = _mirror_full_spectrum(np.array(diff["mean"]), full_W) if full_W>0 else np.array(diff["mean"])
            hist_png  = comp_out / f"frame{frame}_global_hist_percell_mean_diff.png"
            _plot_hist_from_array(hist_png, frame, mean_for_hist, prefix)

            # Heatmaps (linear + log)
            maps1_png = comp_out / f"frame{frame}_percell_mean_std_min_max_diff.png"
            maps2_png = comp_out / f"frame{frame}_percell_quantiles_diff.png"
            _plot_per_cell_maps_from_arrays(maps1_png, maps2_png, frame, diff,
                                            title_prefix=prefix, log_scale=False,
                                            mirror_full_W=full_W if full_W>0 else None)
            _plot_per_cell_maps_from_arrays(maps1_png, maps2_png, frame, diff,
                                            title_prefix=prefix, log_scale=True,
                                            mirror_full_W=full_W if full_W>0 else None)
        if not found_any:
            print(f"    [warn] no FFT per_cell components found at frame {frame}.")

# ---------- opening & main ----------
def _open_h5_safely(path: Path, mode="r", retries=6, delay=2.0):
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    last_err = None
    for k in range(retries):
        try:
            return h5py.File(path, mode)
        except (BlockingIOError, OSError) as e:
            last_err = e
            print(f"[WARN] HDF5 lock/IO when opening {path.name} (try {k+1}/{retries}); retry in {delay}s...")
            time.sleep(delay)
    raise last_err

def main():
    # headless
    import matplotlib
    matplotlib.use("Agg")

    # Open
    spatial = h5py.File(SPATIAL_H5, "r") if SPATIAL_H5.exists() else None
    fft     = h5py.File(FFT_H5, "r") if FFT_H5.exists() else None
    if spatial is None and fft is None:
        raise FileNotFoundError("Neither spatial nor FFT HDF5 found; nothing to plot.")

    if spatial is not None:
        print(f"[INFO] Spatial loaded: {SPATIAL_H5.resolve()}")
    else:
        print(f"[WARN] Spatial HDF5 not found: {SPATIAL_H5}")
    if fft is not None:
        print(f"[INFO] FFT loaded: {FFT_H5.resolve()}")
    else:
        print(f"[INFO] FFT HDF5 not present; skip FFT plots.")

    # Names & types
    name2type: Dict[str, str] = {}
    if spatial is not None:
        name2type.update(_collect_names_and_types(spatial))
    if fft is not None:
        for n, t in _collect_names_and_types(fft).items():
            name2type.setdefault(n, t)

    # Only expanded
    expanded_names = sorted([n for n, t in name2type.items() if t == "expanded"])
    print(f"[INFO] Expanded datasets to draw: {len(expanded_names)}")

    for idx, name_exp in enumerate(expanded_names, 1):
        print(f"\n[{idx}/{len(expanded_names)}] Drawing {name_exp}  (expanded-only; using expanded - train)")
        # find train counterpart in each file independently
        name_tr_spatial = _guess_train_name(spatial, name_exp) if (spatial is not None and "files" in spatial) else None
        name_tr_fft     = _guess_train_name(fft,     name_exp) if (fft     is not None and "files" in fft)     else None

        # Spatial
        if spatial is not None and name_exp in spatial["files"] and name_tr_spatial and name_tr_spatial in spatial["files"]:
            print(f"  [do] spatial diff: expanded={name_exp}  minus  train={name_tr_spatial}")
            _plot_expanded_minus_train_spatial(spatial, name_exp, name_tr_spatial)
        else:
            print("  [skip] spatial diff (expanded or matched train not found in spatial H5)")

        # FFT
        if fft is not None and name_exp in fft["files"] and name_tr_fft and name_tr_fft in fft["files"]:
            print(f"  [do] fft diff: expanded={name_exp}  minus  train={name_tr_fft}")
            _plot_expanded_minus_train_fft(fft, name_exp, name_tr_fft)
        else:
            print("  [skip] fft diff (expanded or matched train not found in FFT H5)")

    # Close
    if spatial is not None: spatial.close()
    if fft is not None: fft.close()
    print(f"\n[OK] Plots saved under: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()


