# -*- coding: utf-8 -*-
"""
Read two HDF5 stats files (spatial + FFT) and plot:
  For each dataset, for frames 0 and 19:
    Spatial:
      - Global histogram (all cells x all records) with mean/std/min/max/range annotated
      - Per-cell heatmaps: mean, std, min, max, q25, q50, q75
    FFT (3 components if available): magnitude | real | imag
      - Global histogram
      - Per-cell heatmaps (linear): mean, std, min, max, q25, q50, q75
      - Per-cell heatmaps (log10|.|): mean, std, min, max, q25, q50, q75
Output layout:
  <OUT_DIR>/<type>/<dataset_name>/spatial/*.png
  <OUT_DIR>/<type>/<dataset_name>/fourier/{mag,real,imag}/*.png

If a target plot already exists, it is skipped.

Additionally:
  - Only draw a fixed random subset (size=20) from the 'generalizability' category,
    chosen with a fixed seed so the selection is deterministic across runs.
"""

from pathlib import Path
import os, time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

# ======= user params =======
SCRIPT_DIR = Path(__file__).parent.resolve()

def find_nearby_file(name: str) -> Path:
    p1 = (SCRIPT_DIR / name)
    if p1.exists():
        return p1
    return Path.cwd() / name  # may not exist; we'll check later

SPATIAL_H5 = find_nearby_file("dataset_stats.h5")
FFT_H5     = find_nearby_file("dataset_stats_fft.h5")

OUT_DIR    = SCRIPT_DIR / "dataset_stats_plots_by_type"
FRAMES     = (0, 19)
DPI        = 150

# Sampling for 'generalizability'
LIMIT_GENERALIZABILITY = 20
SAMPLING_SEED          = 20240901   # fixed seed for reproducible selection
# ==========================

CATEGORIES = ("generalizability", "expanded", "train", "test")

# ---------- utils ----------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _fmt(v: float) -> str:
    return f"{v:.4g}"

def _safe_get(g: Optional[h5py.Group], key: str):
    return None if g is None else g.get(key, None)

def _plot_exists_guard(path: Path) -> bool:
    if path.exists():
        print(f"    [skip exists] {path.name}")
        return True
    return False

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
    """
    Multi-line dataset title:
      line1: [type] name
      line2: N/H/W/T
      line3 (FFT only): H_fft/Wc_fft/kind
    """
    N = _attr_int(fgrp, "N")
    H = _attr_int(fgrp, "H")
    W = _attr_int(fgrp, "W")
    T = _attr_int(fgrp, "T")
    lines = [f"[{dtype}] {ds_name}",
             f"N={N}, H={H}, W={W}, T={T}"]
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
    """
    Mirror rfft2 half-spectrum (H, Wc) to full width (H, W) to get a square.
    If width already equals full_W, return as-is.
    Even W: concat [half, half[:, -2:0:-1]]
    Odd  W: concat [half, half[:, -1:0:-1]]
    """
    if arr_half.ndim != 2:
        raise ValueError(f"expect 2D array (H, Wc), got shape {arr_half.shape}")
    H, Wc = arr_half.shape
    if Wc == full_W:
        return arr_half
    if full_W <= 0 or Wc > full_W:
        # Fallback: don't mirror if metadata is inconsistent
        return arr_half
    if full_W % 2 == 0:
        # even W
        tail = arr_half[:, -2:0:-1] if Wc >= 2 else arr_half[:, :0]
    else:
        # odd W
        tail = arr_half[:, -1:0:-1] if Wc >= 1 else arr_half[:, :0]
    return np.concatenate([arr_half, tail], axis=1)

# ---------- plotting ----------
def plot_global_hist(out_png: Path, frame: int, g_global: h5py.Group, title_prefix: str):
    """
    title_prefix is a multi-line string with dataset info + 'Spatial'/'FFT-xxx'.
    """
    if _plot_exists_guard(out_png):
        return
    mean  = float(g_global["mean"][()])
    std   = float(g_global["std"][()])
    vmin  = float(g_global["min"][()])
    vmax  = float(g_global["max"][()])
    grng  = float(g_global["range"][()])

    g_hist = g_global["hist"]
    h_vmin = float(g_hist.attrs["vmin"])
    h_vmax = float(g_hist.attrs["vmax"])
    nbins  = int(g_hist.attrs["nbins"])
    counts = np.array(g_hist["counts"], dtype=np.int64)
    edges  = np.linspace(h_vmin, h_vmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths  = (edges[1:] - edges[:-1])

    fig = plt.figure(figsize=(8,5))
    ax = plt.gca()
    ax.bar(centers, counts, width=widths, align="center", edgecolor="none")
    ax.set_xlabel("Value"); ax.set_ylabel("Count")
    title = (f"{title_prefix}\n"
             f"Frame {frame} | Global Histogram\n"
             f"mean={_fmt(mean)}  std={_fmt(std)}  range=[{_fmt(vmin)}, {_fmt(vmax)}] (Δ={_fmt(grng)})")
    ax.set_title(title); ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {out_png.name}")

def plot_per_cell_maps(out_png1: Path, out_png2: Path, frame: int, g_cell: h5py.Group,
                       title_prefix: str, *, log_scale: bool = False,
                       mirror_full_W: Optional[int] = None):
    """
    Two figures:
      1) mean/std/min/max (2x2)
      2) q25/q50/q75 (1x3)
    If log_scale=True, color by log10(|.|) and append '_log' to filenames.
    If mirror_full_W is not None, horizontally mirror (H,Wc)->(H,mirror_full_W) to show a square spectrum.
    title_prefix should already include dataset info string.
    """
    tag = "_log" if log_scale else ""
    out1 = out_png1.with_name(out_png1.stem + tag + out_png1.suffix)
    out2 = out_png2.with_name(out_png2.stem + tag + out_png2.suffix)

    need_1 = not out1.exists()
    need_2 = not out2.exists()
    if not (need_1 or need_2):
        print(f"    [skip exists] {out1.name}, {out2.name}")
        return

    mean = np.array(g_cell["mean"]); std  = np.array(g_cell["std"])
    vmin = np.array(g_cell["min"]);  vmax = np.array(g_cell["max"])
    q25  = np.array(g_cell["q25"]);  q50  = np.array(g_cell["q50"]); q75 = np.array(g_cell["q75"])

    # Mirror to full width if requested (only used for FFT)
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
        imshow_func(axes1[0,0], mean, "Per-cell mean")
        imshow_func(axes1[0,1], std,  "Per-cell std")
        imshow_func(axes1[1,0], vmin, "Per-cell min")
        imshow_func(axes1[1,1], vmax, "Per-cell max")
        fig1.suptitle(f"{head}\nPer-cell statistics (mean/std/min/max){tag}", y=0.995, fontsize=12)
        fig1.tight_layout(rect=[0,0,1,0.98])
        fig1.savefig(out1, dpi=DPI, bbox_inches="tight")
        plt.close(fig1)
        print(f"    [saved] {out1.name}")

    if need_2:
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        imshow_func(axes2[0], q25, "Per-cell q25")
        imshow_func(axes2[1], q50, "Per-cell q50 (median)")
        imshow_func(axes2[2], q75, "Per-cell q75")
        fig2.suptitle(f"{head}\nPer-cell quantiles{tag}", y=0.98, fontsize=12)
        fig2.tight_layout(rect=[0,0,1,0.95])
        fig2.savefig(out2, dpi=DPI, bbox_inches="tight")
        plt.close(fig2)
        print(f"    [saved] {out2.name}")

# ---------- drivers ----------
def _collect_names_and_types(h5f: h5py.File) -> Dict[str, str]:
    """Return {dataset_name: type}, defaulting to 'train'."""
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

def _plot_for_dataset_spatial(h5f: h5py.File, ds_name: str, dtype: str):
    fgrp = h5f["files"][ds_name]
    # ---- new directory layout ----
    base_out = OUT_DIR / dtype / ds_name / "spatial"
    _ensure_dir(base_out)
    print(f"  [spatial] -> {base_out}")

    ds_title = _build_ds_title(dtype, ds_name, fgrp, is_fft=False)

    for frame in FRAMES:
        g_frame = _frames_group(fgrp, frame)
        if g_frame is None:
            print(f"    [warn] spatial frame {frame} missing; skip.")
            continue
        g_global = _safe_get(g_frame, "global")
        g_cell   = _safe_get(g_frame, "per_cell")
        if g_global is None or g_cell is None:
            print(f"    [warn] spatial stats missing; skip frame {frame}.")
            continue

        hist_png  = base_out / f"frame{frame}_global_hist.png"
        maps1_png = base_out / f"frame{frame}_percell_mean_std_min_max.png"
        maps2_png = base_out / f"frame{frame}_percell_quantiles.png"

        prefix = f"{ds_title}\nSpatial"
        plot_global_hist(hist_png, frame, g_global, title_prefix=prefix)
        plot_per_cell_maps(maps1_png, maps2_png, frame, g_cell, title_prefix=prefix, log_scale=False)

def _plot_for_dataset_fft(h5f: h5py.File, ds_name: str, dtype: str):
    """
    Compatible with three possible layouts:
      NEW (merged writer):
        /files/<name>/frames/<t>/fft/<comp>/{global,per_cell}
      OLD-A:
        /files/<name>/frames/<t>/<comp>/{global,per_cell}
      OLD-B:
        /files/<name>/frames/<t>/global_<comp>  and  per_cell_<comp>
    """
    fgrp = h5f["files"][ds_name]
    # ---- new directory layout ----
    root_out = OUT_DIR / dtype / ds_name / "fourier"
    _ensure_dir(root_out)
    print(f"  [fft] -> {root_out}")

    ds_title = _build_ds_title(dtype, ds_name, fgrp, is_fft=True)
    candidate_sets = [("mag", "mag"), ("real", "real"), ("imag", "imag")]
    full_W = _attr_int(fgrp, "W", -1)  # mirror target width

    for frame in FRAMES:
        g_frame = _frames_group(fgrp, frame)
        if g_frame is None:
            print(f"    [warn] fft frame {frame} missing; skip.")
            continue

        found_any = False
        g_fft_root = _safe_get(g_frame, "fft")  # preferred NEW layout

        for comp_key, comp_tag in candidate_sets:
            g_global = None
            g_cell   = None

            # 1) NEW layout
            if g_fft_root is not None:
                g_comp = _safe_get(g_fft_root, comp_key)
                if g_comp is not None:
                    g_global = _safe_get(g_comp, "global")
                    g_cell   = _safe_get(g_comp, "per_cell")

            # 2) OLD-A layout fallback
            if g_global is None or g_cell is None:
                g_comp_a = _safe_get(g_frame, comp_key)
                if g_comp_a is not None:
                    g_global = _safe_get(g_comp_a, "global") if g_global is None else g_global
                    g_cell   = _safe_get(g_comp_a, "per_cell") if g_cell   is None else g_cell

            # 3) OLD-B layout fallback
            if g_global is None or g_cell is None:
                g_global = _safe_get(g_frame, f"global_{comp_key}") if g_global is None else g_global
                g_cell   = _safe_get(g_frame, f"per_cell_{comp_key}") if g_cell   is None else g_cell

            if g_global is None or g_cell is None:
                continue

            found_any = True

            # component-specific dir
            comp_out = root_out / comp_tag
            _ensure_dir(comp_out)

            hist_png  = comp_out / f"frame{frame}_global_hist.png"
            maps1_png = comp_out / f"frame{frame}_percell_mean_std_min_max.png"
            maps2_png = comp_out / f"frame{frame}_percell_quantiles.png"

            prefix = f"{ds_title}\nFFT-{comp_tag}"

            # histogram
            plot_global_hist(hist_png, frame, g_global, title_prefix=prefix)

            # linear heatmaps (mirror to full width -> square)
            plot_per_cell_maps(maps1_png, maps2_png, frame, g_cell,
                               title_prefix=prefix, log_scale=False,
                               mirror_full_W=full_W if full_W > 0 else None)
            # log10 heatmaps (mirror to full width -> square)
            plot_per_cell_maps(maps1_png, maps2_png, frame, g_cell,
                               title_prefix=prefix, log_scale=True,
                               mirror_full_W=full_W if full_W > 0 else None)

        if not found_any:
            print(f"    [warn] no FFT components found for frame {frame}; checked layouts: new/old-a/old-b.")

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
    # prepare folders
    for cat in CATEGORIES:
        _ensure_dir(OUT_DIR / cat)

    # headless backend
    import matplotlib
    matplotlib.use("Agg")

    # Show what we can see
    print("[INFO] Script dir:", SCRIPT_DIR)
    here_h5 = sorted(p.name for p in SCRIPT_DIR.glob("*.h5"))
    cwd_h5  = sorted(p.name for p in Path.cwd().glob("*.h5"))
    print("[INFO] H5 in script dir:", here_h5)
    print("[INFO] H5 in CWD      :", cwd_h5)

    # Open files if present
    spatial = None
    fft = None

    if SPATIAL_H5.exists():
        try:
            spatial = h5py.File(SPATIAL_H5, "r")
            print(f"[INFO] Spatial file loaded: {SPATIAL_H5.resolve()}")
        except Exception as e:
            print(f"[WARN] Failed to open spatial HDF5: {e}")
            spatial = None
    else:
        print(f"[WARN] Spatial HDF5 not found: {SPATIAL_H5}")

    if FFT_H5.exists():
        try:
            fft = h5py.File(FFT_H5, "r")
            print(f"[INFO] FFT file loaded: {FFT_H5.resolve()}")
        except (BlockingIOError, OSError) as e:
            print(f"[WARN] FFT HDF5 cannot be opened now ({e}); skip FFT plots this run.")
            fft = None
        except Exception as e:
            print(f"[WARN] Failed to open FFT HDF5 ({e}); skip FFT plots.")
            fft = None
    else:
        print(f"[INFO] FFT HDF5 not present ({FFT_H5}); skipping FFT plots.")

    if spatial is None and fft is None:
        raise FileNotFoundError("Neither spatial nor FFT HDF5 found; nothing to plot.")

    # Collect dataset names/types (prefer spatial, then fft)
    name2type: Dict[str, str] = {}
    if spatial is not None:
        name2type.update(_collect_names_and_types(spatial))
    if fft is not None:
        for n, t in _collect_names_and_types(fft).items():
            name2type.setdefault(n, t)

    names_all = sorted(name2type.keys())
    print(f"[INFO] Total datasets discovered: {len(names_all)}")

    # Deterministic sampling for generalizability
    gen_all = sorted([n for n in names_all if name2type.get(n, "train") == "generalizability"])
    non_gen = [n for n in names_all if name2type.get(n, "train") != "generalizability"]

    print(f"[INFO] generalizability datasets available: {len(gen_all)}")
    if len(gen_all) > LIMIT_GENERALIZABILITY:
        rng = np.random.RandomState(SAMPLING_SEED)
        gen_pick = list(rng.choice(gen_all, size=LIMIT_GENERALIZABILITY, replace=False))
        gen_pick_set = set(gen_pick)
        print(f"[INFO] Sampling {LIMIT_GENERALIZABILITY} from generalizability with seed={SAMPLING_SEED}")
        print("       Selected (first 10 shown):", ", ".join(sorted(gen_pick)[:10]), "...")
    else:
        gen_pick = gen_all
        gen_pick_set = set(gen_pick)
        print(f"[INFO] <= {LIMIT_GENERALIZABILITY}; using all generalizability datasets.")

    names = [n for n in names_all if (name2type.get(n) != "generalizability") or (n in gen_pick_set)]
    print(f"[INFO] Total datasets to draw after sampling: {len(names)} "
          f"(non-gen={len(non_gen)}, gen-picked={len(gen_pick)})")

    # Draw
    for idx, name in enumerate(names, 1):
        dtype = name2type.get(name, "train")
        if dtype not in CATEGORIES:
            dtype = "train"
        print(f"\n[{idx}/{len(names)}] Drawing {name}  type={dtype}")

        if spatial is not None and "files" in spatial and name in spatial["files"]:
            print("  [do] spatial plots")
            _plot_for_dataset_spatial(spatial, name, dtype)
        else:
            print("  [skip] spatial (not found)")

        if fft is not None and "files" in fft and name in fft["files"]:
            print("  [do] fft plots (components: mag/real/imag if present)")
            _plot_for_dataset_fft(fft, name, dtype)
        else:
            print("  [skip] fft (not found)")

    # Close files
    if spatial is not None:
        spatial.close()
    if fft is not None:
        fft.close()

    print(f"\n[OK] Plots saved under: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
