#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render 12-panel GIFs (3x4) from attack pickle files produced by the PGD script.

Data layout expected from generator:
-----------------------------------
Top-level pickle: dict[attack_key] -> sample_results
- attack_key: tuple like ('norm_2','index_0','numsteps_100','epsilon_20','alpha_0.01000')
- sample_results: dict[combo] -> records
  - combo is "mse__with_solver" | "mse__without_solver" | "mse__approximate"
            or "softdtw__with_solver" | ... (total 6 combos: 2 losses x 3 modes)
  - records: dict with key "step" (not "step_update"):
      records["step"][f"step_{k}"] = {
          "input": np.ndarray,            # a (+ δ) for THIS combo at step k (k==0 → clean a)
          "G_out": np.ndarray,            # FNO output at this step for THIS combo
          "T_out": np.ndarray,            # PDE output (full solver) at this step for THIS combo
          "loss": {"mse": float, "softdtw": float}
      }

Goal
----
For each attack_key (bundle of 6 combos), render a GIF with 12 panels (3x4):

Row 1
  (1,1) Inputs — optimized by "mse": 4 lines → a(clean, black) + a+δ (with, detached, approximated)
  (1,2) Inputs — optimized by "softdtw": same 4 lines for softdtw objective
  (1,3) FNO vs PDE — method "with" — optimized by "mse"
  (1,4) FNO vs PDE — method "with" — optimized by "softdtw"

Row 2
  (2,1) FNO vs PDE — method "detached" — optimized by "mse"
  (2,2) FNO vs PDE — method "detached" — optimized by "softdtw"
  (2,3) FNO vs PDE — method "approximated" — optimized by "mse"
  (2,4) FNO vs PDE — method "approximated" — optimized by "softdtw"

Row 3 (Loss progress; each with 3 colored lines = methods)
  (3,1) Optimized by "mse" — show MSE loss (with / detached / approximated)
  (3,2) Optimized by "mse" — show SoftDTW loss (with / detached / approximated)
  (3,3) Optimized by "softdtw" — show MSE loss (with / detached / approximated)
  (3,4) Optimized by "softdtw" — show SoftDTW loss (with / detached / approximated)

Notes
-----
- All text should use "detached" instead of "without solver".
- Colors: black for a(clean); orange for "with", blue for "detached", red for "approximated".
- FNO curves use the deep color; PDE curves use the light version of the same hue.
- Global y-lims are computed per GIF (consistent across frames) for inputs, outputs, and each loss range.
"""

import argparse
import os
import pickle
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional

# ---------- Palette ----------
PALETTE = {
    "black_deep":  "#000000",
    "black_light": "#666666",

    # 'with' (solver grad backprop)
    "orange_deep": "#d35400",
    "orange_light":"#f5b041",

    # 'detached' (without)
    "blue_deep":   "#1f77b4",
    "blue_light":  "#85c1e9",

    # 'approximated' (approximate)
    "red_deep":    "#c0392b",
    "red_light":   "#f1948a",
}

METHOD_ORDER = ["with_solver", "without_solver", "approximate"]
METHOD_LABEL = {
    "with_solver": "solver grad backprop",
    "without_solver": "detached",
    "approximate": "approximated",
}

def _sorted_steps(step_dict: Dict[str, Any]) -> List[int]:
    """Return sorted step indices [0,1,2,...] present in records['step']"""
    out = []
    for k in step_dict.keys():
        if not k.startswith("step_"):
            continue
        try:
            out.append(int(k.split("_")[-1]))
        except Exception:
            continue
    return sorted(out)

def _safe_array(x) -> np.ndarray:
    try:
        arr = np.asarray(x).astype(float).ravel()
        return arr
    except Exception:
        return np.array([], dtype=float)

def _pad_ylim(lo: float, hi: float, pct: float = 0.05) -> Tuple[float, float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-1.0, 1.0)
    span = hi - lo
    if span <= 0:
        span = max(1.0, abs(hi) + 1.0)
    return (lo - pct * span, hi + pct * span)

def _extract_bundle(attack_bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    attack_bundle: dict[combo] -> records
    combo: '<loss>__<mode>'  loss in {'mse','softdtw'}, mode in METHOD_ORDER
    returns nested dict: result[loss][mode] = records
    """
    out = {"mse": {}, "softdtw": {}}
    for combo, rec in attack_bundle.items():
        if not isinstance(combo, str) or "__" not in combo:
            # Some users may have tuple keys or other; skip
            continue
        loss_name, mode = combo.split("__", 1)
        if loss_name not in out:
            continue
        out[loss_name][mode] = rec
    return out

def _compute_global_limits(combos: Dict[str, Dict[str, Any]]) -> Tuple[Tuple[float,float], Tuple[float,float], Dict[str,Tuple[float,float]]]:
    """
    Compute global y-lims for inputs, outputs, and per-loss display.
    Returns:
      ylim_a, ylim_out, ylim_loss_map where
        - ylim_a for all 'input' curves
        - ylim_out for all G_out/T_out curves
        - ylim_loss_map: dict key ('mse|softdtw','objective in {mse,softdtw}') → (lo,hi)
          e.g. ('mse','mse'): limits for panel (3,1)
               ('mse','softdtw'): limits for panel (3,2)
               ('softdtw','mse'): limits for panel (3,3)
               ('softdtw','softdtw'): limits for panel (3,4)
    """
    a_lo, a_hi = np.inf, -np.inf
    out_lo, out_hi = np.inf, -np.inf

    loss_bounds = {
        ("mse", "mse"): [np.inf, -np.inf],
        ("mse", "softdtw"): [np.inf, -np.inf],
        ("softdtw", "mse"): [np.inf, -np.inf],
        ("softdtw", "softdtw"): [np.inf, -np.inf],
    }

    for loss_obj in ["mse", "softdtw"]:
        for mode in METHOD_ORDER:
            rec = combos.get(loss_obj, {}).get(mode, None)
            if rec is None:
                continue
            steps = rec.get("step", {})
            for sk, sdict in steps.items():
                # inputs
                inp = _safe_array(sdict.get("input", None))
                if inp.size:
                    a_lo = min(a_lo, float(np.nanmin(inp)))
                    a_hi = max(a_hi, float(np.nanmax(inp)))
                # outputs
                gout = _safe_array(sdict.get("G_out", None))
                tout = _safe_array(sdict.get("T_out", None))
                if gout.size:
                    out_lo = min(out_lo, float(np.nanmin(gout)))
                    out_hi = max(out_hi, float(np.nanmax(gout)))
                if tout.size:
                    out_lo = min(out_lo, float(np.nanmin(tout)))
                    out_hi = max(out_hi, float(np.nanmax(tout)))
                # losses
                ldict = sdict.get("loss", {})
                for show_loss in ["mse", "softdtw"]:
                    val = ldict.get(show_loss, None)
                    if val is None:
                        continue
                    key = (loss_obj, show_loss)
                    loss_bounds[key][0] = min(loss_bounds[key][0], float(val))
                    loss_bounds[key][1] = max(loss_bounds[key][1], float(val))

    ylim_a = _pad_ylim(a_lo, a_hi)
    ylim_out = _pad_ylim(out_lo, out_hi)
    ylim_loss_map = {}
    for k, (lo, hi) in loss_bounds.items():
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = 0.0, max(1.0, hi if np.isfinite(hi) else 1.0)
        ylim_loss_map[k] = _pad_ylim(lo, hi)
    return ylim_a, ylim_out, ylim_loss_map

def _get_clean_a(any_rec: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return 'clean a' from step_0 'input' if available."""
    steps = any_rec.get("step", {})
    s0 = steps.get("step_0", None)
    if s0 is None:
        # try smallest step
        ks = _sorted_steps(steps)
        if ks:
            s0 = steps.get(f"step_{ks[0]}", None)
    if s0 is None:
        return None
    return _safe_array(s0.get("input", None))

def _plot_inputs(ax, clean_a: Optional[np.ndarray], lines: List[Tuple[np.ndarray, str, str, float]], ylim):
    ax.clear()
    if clean_a is not None and clean_a.size:
        x = np.arange(clean_a.size)
        ax.plot(x, clean_a, label="a (clean)", color=PALETTE["black_deep"], linewidth=1.6)
    # Draw three methods' a+δ
    for arr, label, color, lw in lines:
        if arr is None or arr.size == 0:
            continue
        x = np.arange(arr.size)
        ax.plot(x, arr, label=label, color=color, linewidth=lw)
    ax.set_ylim(ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def _plot_fno_vs_pde(ax, G_arr: Optional[np.ndarray], T_arr: Optional[np.ndarray],
                     clean_G: Optional[np.ndarray], clean_T: Optional[np.ndarray],
                     deep_c: str, light_c: str, ylim, method_title: str):
    ax.clear()
    # Baseline curves (clean) if available
    if clean_G is not None and clean_G.size:
        x = np.arange(clean_G.size)
        ax.plot(x, clean_G, label='FNO(a)', color=PALETTE["black_deep"], linewidth=1.6)
    if clean_T is not None and clean_T.size:
        x = np.arange(clean_T.size)
        ax.plot(x, clean_T, label='PDE(a)', color=PALETTE["black_light"], linewidth=1.4)

    # Perturbed curves
    if G_arr is not None and G_arr.size:
        x = np.arange(G_arr.size)
        ax.plot(x, G_arr, label='FNO(a + δ)', color=deep_c, linewidth=1.6)
    if T_arr is not None and T_arr.size:
        x = np.arange(T_arr.size)
        ax.plot(x, T_arr, label='PDE(a + δ)', color=light_c, linewidth=1.4)

    ax.set_title(method_title + " : FNO vs PDE")
    ax.set_ylim(ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def _plot_loss(ax, x_steps: np.ndarray, histories: Dict[str, np.ndarray], ylim, title: str):
    ax.clear()
    ax.plot(x_steps, histories.get("with_solver",  np.full_like(x_steps, np.nan, dtype=float)),
            label='solver grad backprop', color=PALETTE["orange_deep"], linewidth=1.6)
    ax.plot(x_steps, histories.get("without_solver", np.full_like(x_steps, np.nan, dtype=float)),
            label='detached',            color=PALETTE["blue_deep"],   linewidth=1.6)
    ax.plot(x_steps, histories.get("approximate", np.full_like(x_steps, np.nan, dtype=float)),
            label='approximated',        color=PALETTE["red_deep"],    linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.set_ylim(ylim)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

def _build_histories(combos: Dict[str, Dict[str, Any]], show_loss: str, objective: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    For loss panels:
      - objective in {'mse','softdtw'} selects which set of records to read (i.e., which combos were optimized)
      - show_loss in {'mse','softdtw'} selects which metric to plot.
    Returns x_steps, histories dict mapping mode -> np.ndarray aligned on master step indices.
    """
    # Get master step index set (union across three modes for this objective)
    master_steps = set()
    for mode in METHOD_ORDER:
        rec = combos.get(objective, {}).get(mode, None)
        if rec is None:
            continue
        master_steps.update(_sorted_steps(rec.get("step", {})))
    master = np.array(sorted(master_steps), dtype=int)
    if master.size == 0:
        master = np.arange(1, dtype=int)

    histories = {}
    for mode in METHOD_ORDER:
        rec = combos.get(objective, {}).get(mode, None)
        if rec is None:
            histories[mode] = np.full_like(master, np.nan, dtype=float)
            continue
        sdict = rec.get("step", {})
        vals = np.full_like(master, np.nan, dtype=float)
        for i, k in enumerate(master):
            d = sdict.get(f"step_{k}", None)
            if d is None:
                continue
            l = d.get("loss", {})
            v = l.get(show_loss, None)
            vals[i] = float(v) if v is not None else np.nan
        histories[mode] = vals
    return master, histories

def render_gif_for_bundle(attack_key, attack_bundle: Dict[str, Any], outdir: str, fps: int = 8):
    combos = _extract_bundle(attack_bundle)

    # Sanity check we have at least one record
    any_rec = None
    for loss_name in ["mse","softdtw"]:
        for mode in METHOD_ORDER:
            rec = combos.get(loss_name, {}).get(mode, None)
            if rec is not None:
                any_rec = rec
                break
        if any_rec: break
    if any_rec is None:
        print(f"[WARN] Empty bundle for key={attack_key}, skip")
        return

    clean_a = _get_clean_a(any_rec)
    # Clean outputs (from step_0) for plotting baselines
    clean_G, clean_T = None, None
    steps_any = any_rec.get("step", {})
    s0 = steps_any.get("step_0", None)
    if s0 is not None:
        clean_G = _safe_array(s0.get("G_out", None))
        clean_T = _safe_array(s0.get("T_out", None))

    # Global limits
    ylim_a, ylim_out, ylim_loss_map = _compute_global_limits(combos)

    # Build common step list to loop frames (union of all steps across all combos)
    all_steps = set()
    for loss_name in ["mse","softdtw"]:
        for mode in METHOD_ORDER:
            rec = combos.get(loss_name, {}).get(mode, None)
            if rec is None: continue
            all_steps.update(_sorted_steps(rec.get("step", {})))
    step_list = sorted(all_steps)
    if not step_list:
        step_list = [0]

    # Safe filename
    if isinstance(attack_key, (tuple, list)):
        key_str = "__".join(map(str, attack_key))
    else:
        key_str = str(attack_key)
    safe_key = "".join(c if c.isalnum() or c in "._-+=" else "_" for c in key_str)
    gif_path = os.path.join(outdir, f"{safe_key}.gif")

    frames = []
    for k in tqdm(step_list, desc=f"Rendering {key_str}"):
        # Gather arrays for the 12 panels, at step k
        # --- Inputs panels ---
        # For each objective, pull the three modes' 'input' at step k
        input_lines_mse = []
        input_lines_sdtw = []
        # colors per mode
        mode_color = {
            "with_solver": (PALETTE["orange_deep"], 1.2, "a + δ (solver grad backprop)"),
            "without_solver": (PALETTE["blue_deep"], 1.2, "a + δ (detached)"),
            "approximate": (PALETTE["red_deep"], 1.2, "a + δ (approximated)"),
        }
        for mode in METHOD_ORDER:
            rec_m = combos.get("mse", {}).get(mode, None)
            rec_s = combos.get("softdtw", {}).get(mode, None)
            color, lw, label = mode_color[mode]

            # mse objective
            arr_m = None
            if rec_m is not None:
                d = rec_m.get("step", {}).get(f"step_{k}", None)
                if d is not None:
                    arr_m = _safe_array(d.get("input", None))
            input_lines_mse.append((arr_m, label, color, lw))

            # sdtw objective
            arr_s = None
            if rec_s is not None:
                d = rec_s.get("step", {}).get(f"step_{k}", None)
                if d is not None:
                    arr_s = _safe_array(d.get("input", None))
            input_lines_sdtw.append((arr_s, label, color, lw))

        # --- FNO vs PDE panels (6) ---
        def get_outputs(loss_name: str, mode: str):
            rec = combos.get(loss_name, {}).get(mode, None)
            if rec is None:
                return None, None
            d = rec.get("step", {}).get(f"step_{k}", None)
            if d is None:
                return None, None
            return _safe_array(d.get("G_out", None)), _safe_array(d.get("T_out", None))

        # --- Loss histories (4) up to current max step (inclusive) ---
        # pane (3,1): objective=mse, show=mse
        xs_11, hist_11 = _build_histories(combos, show_loss="mse",     objective="mse")
        xs_12, hist_12 = _build_histories(combos, show_loss="softdtw", objective="mse")
        xs_13, hist_13 = _build_histories(combos, show_loss="mse",     objective="softdtw")
        xs_14, hist_14 = _build_histories(combos, show_loss="softdtw", objective="softdtw")

        # Truncate to current frame (<= k)
        def trunc(xs, h):
            mask = xs <= k
            return xs[mask], {m: v[mask] for m, v in h.items()}
        xs_11, hist_11 = trunc(xs_11, hist_11)
        xs_12, hist_12 = trunc(xs_12, hist_12)
        xs_13, hist_13 = trunc(xs_13, hist_13)
        xs_14, hist_14 = trunc(xs_14, hist_14)

        # --- Build figure 3x4 ---
        fig = plt.figure(figsize=(18, 11))
        gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

        ax11 = fig.add_subplot(gs[0,0])  # Inputs (mse objective)
        ax12 = fig.add_subplot(gs[0,1])  # Inputs (softdtw objective)

        ax13 = fig.add_subplot(gs[0,2])  # FNO vs PDE — with — mse
        ax14 = fig.add_subplot(gs[0,3])  # FNO vs PDE — with — softdtw

        ax21 = fig.add_subplot(gs[1,0])  # FNO vs PDE — detached — mse
        ax22 = fig.add_subplot(gs[1,1])  # FNO vs PDE — detached — softdtw
        ax23 = fig.add_subplot(gs[1,2])  # FNO vs PDE — approximated — mse
        ax24 = fig.add_subplot(gs[1,3])  # FNO vs PDE — approximated — softdtw

        ax31 = fig.add_subplot(gs[2,0])  # Loss (mse obj, show mse)
        ax32 = fig.add_subplot(gs[2,1])  # Loss (mse obj, show sdtw)
        ax33 = fig.add_subplot(gs[2,2])  # Loss (sdtw obj, show mse)
        ax34 = fig.add_subplot(gs[2,3])  # Loss (sdtw obj, show sdtw)

        # Row 1 — Inputs
        _plot_inputs(ax11, clean_a, input_lines_mse, ylim_a)
        ax11.set_title("Inputs (objective: MSE)")
        _plot_inputs(ax12, clean_a, input_lines_sdtw, ylim_a)
        ax12.set_title("Inputs (objective: SoftDTW)")

        # Row 1 — with
        G_w_m, T_w_m = get_outputs("mse", "with_solver")
        G_w_s, T_w_s = get_outputs("softdtw", "with_solver")
        _plot_fno_vs_pde(ax13, G_w_m, T_w_m, clean_G, clean_T, PALETTE["orange_deep"], PALETTE["orange_light"],
                         ylim_out, "solver grad backprop (objective: MSE)")
        _plot_fno_vs_pde(ax14, G_w_s, T_w_s, clean_G, clean_T, PALETTE["orange_deep"], PALETTE["orange_light"],
                         ylim_out, "solver grad backprop (objective: SoftDTW)")

        # Row 2 — detached & approximated
        G_d_m, T_d_m = get_outputs("mse", "without_solver")
        G_d_s, T_d_s = get_outputs("softdtw", "without_solver")
        _plot_fno_vs_pde(ax21, G_d_m, T_d_m, clean_G, clean_T, PALETTE["blue_deep"], PALETTE["blue_light"],
                         ylim_out, "detached (objective: MSE)")
        _plot_fno_vs_pde(ax22, G_d_s, T_d_s, clean_G, clean_T, PALETTE["blue_deep"], PALETTE["blue_light"],
                         ylim_out, "detached (objective: SoftDTW)")

        G_a_m, T_a_m = get_outputs("mse", "approximate")
        G_a_s, T_a_s = get_outputs("softdtw", "approximate")
        _plot_fno_vs_pde(ax23, G_a_m, T_a_m, clean_G, clean_T, PALETTE["red_deep"], PALETTE["red_light"],
                         ylim_out, "approximated (objective: MSE)")
        _plot_fno_vs_pde(ax24, G_a_s, T_a_s, clean_G, clean_T, PALETTE["red_deep"], PALETTE["red_light"],
                         ylim_out, "approximated (objective: SoftDTW)")

        # Row 3 — Loss panels
        _plot_loss(ax31, xs_11, hist_11, ylim_loss_map[("mse","mse")], "Loss (objective: MSE, show: MSE)")
        _plot_loss(ax32, xs_12, hist_12, ylim_loss_map[("mse","softdtw")], "Loss (objective: MSE, show: SoftDTW)")
        _plot_loss(ax33, xs_13, hist_13, ylim_loss_map[("softdtw","mse")], "Loss (objective: SoftDTW, show: MSE)")
        _plot_loss(ax34, xs_14, hist_14, ylim_loss_map[("softdtw","softdtw")], "Loss (objective: SoftDTW, show: SoftDTW)")

        fig.suptitle(f"{key_str} | step {k}", fontsize=14)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    os.makedirs(outdir, exist_ok=True)
    iio.imwrite(gif_path, frames, fps=fps)
    print(f"[OK] Saved {gif_path}")

def render_folder(pickle_folder: str, outdir: str, fps: int = 8):
    folder = Path(pickle_folder)
    pkls = sorted([p for p in folder.glob("*.pkl") if p.is_file()])
    print(f"Found {len(pkls)} pickle files in {folder}:")
    for p in pkls:
        print("  -", p.name)

    for pkl in pkls:
        print(f"\n[LOAD] {pkl.name}")
        with open(pkl, "rb") as f:
            top = pickle.load(f)

        # Each top-level key = attack_key; value = attack_bundle (dict of 6 combos)
        out_dir_this = os.path.join(outdir, pkl.stem)
        os.makedirs(out_dir_this, exist_ok=True)
        for attack_key, attack_bundle in top.items():
            render_gif_for_bundle(attack_key, attack_bundle, out_dir_this, fps=fps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder that contains the generated *.pkl files")
    ap.add_argument("--outdir", required=True, help="Output folder for GIFs (will create subfolders per pickle)")
    ap.add_argument("--fps", type=int, default=8, help="Frames per second for GIFs (default: 8)")
    args = ap.parse_args()

    render_folder(args.folder, args.outdir, args.fps)

if __name__ == "__main__":
    main()


