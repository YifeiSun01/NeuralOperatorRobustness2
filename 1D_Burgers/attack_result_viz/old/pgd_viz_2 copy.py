#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render 12-panel GIFs (3x4) from PGD attack pickle files, with robust fallback to
`final_values`, and exhaustive per-step audit prints (NaN/Inf/shape/len).

新增：
- audit 审计模式：顶层、bundle、每一步 step 的数据完整性检查与打印（可控频率）。
- 支持 "step" / "step_update" / "steps" / 直接 step_* 键四种容器。
- finals 模糊键匹配容错（G(a_with) / G_with_solver / G with_solver 等）。
- 若完全无可画数据，跳过写出 GIF 并在控制台报警，避免空白图。

用法示例：
python pgd_viz_2.py --folder /path/pkls --outdir /path/gifs --fps 8 \
  --debug --debug-every 1 --debug-head 9999 \
  --audit --audit-every 1 --audit-head 20
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
    # 'detached'
    "blue_deep":   "#1f77b4",
    "blue_light":  "#85c1e9",
    # 'approximated'
    "red_deep":    "#c0392b",
    "red_light":   "#f1948a",
}

METHOD_ORDER = ["with_solver", "without_solver", "approximate"]

DISPLAY_LABEL = {
    "with_solver": "solver grad backprop",
    "without_solver": "detached",
    "approximate":   "approximated",
}

# === 新增：保存端命名到三种模式的键映射 ===
MODE_TO_SUFFIX = {
    "with_solver":   ("a_with",        "G(a_with)",        "g(a_with)"),
    "without_solver":("a_without",     "G(a_without)",     "g(a_without)"),
    "approximate":   ("a_approximate", "G(a_approximate)", "g(a_approximate)"),
}

# ---------- Debug/audit helpers ----------
def _arr_stats(arr: Optional[np.ndarray]) -> str:
    if arr is None:
        return "None"
    if not isinstance(arr, np.ndarray):
        return f"type={type(arr).__name__}"
    n = arr.size
    if n == 0:
        return "len=0"
    finite_mask = np.isfinite(arr)
    n_fin = int(finite_mask.sum())
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    shape = tuple(arr.shape)
    if n_fin > 0:
        vals = arr[finite_mask]
        mn = float(np.min(vals))
        mx = float(np.max(vals))
        return f"shape={shape}, len={n}, finite={n_fin}, nan={n_nan}, inf={n_inf}, min={mn:.4g}, max={mx:.4g}"
    else:
        return f"shape={shape}, len={n}, finite=0, nan={n_nan}, inf={n_inf}"

def _print_step_debug(step_k: int, loss_name: str, mode: str,
                      inp_step: Optional[np.ndarray],
                      G_step: Optional[np.ndarray], T_step: Optional[np.ndarray],
                      loss_mse: Optional[float], loss_sdtw: Optional[float],
                      G_fin: Optional[np.ndarray], g_fin: Optional[np.ndarray]):
    print(
        f"[DBG][step={step_k:3d}][{loss_name:7s}][{mode:14s}] "
        f"input({_arr_stats(inp_step)}) | "
        f"G_out({_arr_stats(G_step)}) | "
        f"T_out({_arr_stats(T_step)}) | "
        f"loss_mse={loss_mse if loss_mse is not None else 'None'} | "
        f"loss_sdtw={loss_sdtw if loss_sdtw is not None else 'None'} | "
        f"FINAL G({_arr_stats(G_fin)}) g({_arr_stats(g_fin)})"
    )

def _print_step_audit(step_k: int, tag: str, name: str, arr: Optional[np.ndarray]):
    print(f"[AUDIT][step={step_k:3d}][{tag}] {name:18s} :: {_arr_stats(arr)}")

# ---------- Small utils ----------
def _sorted_steps_from_container(step_container: Dict[str, Any]) -> List[int]:
    out = []
    for k in step_container.keys():
        if not isinstance(k, str):
            continue
        if not k.startswith("step_"):
            continue
        try:
            out.append(int(k.split("_")[-1]))
        except Exception:
            continue
    return sorted(out)

def _safe_array(x) -> np.ndarray:
    try:
        arr = np.asarray(x).astype(float)
        return arr
    except Exception:
        return np.array([], dtype=float)

def _as_1d(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if x is None: return None
    if not isinstance(x, np.ndarray) or x.size == 0: return None
    return x.ravel()

def _pad_ylim(lo: float, hi: float, pct: float = 0.05) -> Tuple[float, float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-1.0, 1.0)
    span = hi - lo
    if span <= 0:
        span = max(1.0, abs(hi) + 1.0)
    return (lo - pct * span, hi + pct * span)

# --- detect step container shapes ---
def _detect_step_container(rec: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if isinstance(rec, dict):
        if "step" in rec and isinstance(rec["step"], dict):
            return "step", rec["step"]
        if "step_update" in rec and isinstance(rec["step_update"], dict):
            return "step_update", rec["step_update"]
        if "steps" in rec and isinstance(rec["steps"], dict):
            return "steps", rec["steps"]
        has_direct = any(isinstance(k, str) and k.startswith("step_") for k in rec.keys())
        if has_direct:
            return "direct", rec
    return "none", {}

def _grab_clean_a_G_T(any_rec: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    cont_name, cont = _detect_step_container(any_rec)
    if cont_name == "none" or not isinstance(cont, dict):
        return None, None, None
    steps = cont
    s0 = steps.get("step_0", None)
    if s0 is None:
        ks = _sorted_steps_from_container(steps)
        if ks:
            s0 = steps.get(f"step_{ks[0]}", None)
    if s0 is None or not isinstance(s0, dict):
        return None, None, None
    a0 = _as_1d(_safe_array(s0.get("input", None)))
    G0 = _as_1d(_safe_array(s0.get("G_out", None)))
    T0 = _as_1d(_safe_array(s0.get("T_out", None)))
    return (a0, G0, T0)

# ---------- Data normalization ----------
def _is_expected_combo_bundle(attack_bundle: Dict[str, Any]) -> bool:
    for k, v in attack_bundle.items():
        if isinstance(k, str) and "__" in k and isinstance(v, dict):
            return True
    return False

def _extract_expected_bundle(attack_bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {"mse": {}, "softdtw": {}}
    for combo, rec in attack_bundle.items():
        if not (isinstance(combo, str) and "__" in combo and isinstance(rec, dict)):
            continue
        loss_name, mode = combo.split("__", 1)
        if loss_name not in out:
            continue
        out[loss_name][mode] = rec
    return out

def _loss_from_tuple_key(tpl: Tuple[Any, ...]) -> Optional[str]:
    if not tpl:
        return None
    first = str(tpl[0])
    if first.startswith("loss_"):
        tag = first.split("loss_", 1)[-1].lower()
        if tag in ("mse", "sdtw", "softdtw"):
            return "softdtw" if tag in ("sdtw", "softdtw") else "mse"
    return None

def _base_group_key(tpl: Tuple[Any, ...]) -> Tuple[Any, ...]:
    if not tpl:
        return tpl
    first = str(tpl[0])
    if first.startswith("loss_"):
        return tpl[1:]
    return tpl

def _extract_current_format_into_virtual_bundle(value: Dict[str, Any],
                                                loss_name: str) -> Dict[str, Dict[str, Any]]:
    combos = {"mse": {}, "softdtw": {}}
    target = combos[loss_name]
    found_mode_level = False
    for mode in METHOD_ORDER:
        if mode in value and isinstance(value[mode], dict):
            target[mode] = value[mode]
            found_mode_level = True
    if found_mode_level:
        return combos
    cont_name, _ = _detect_step_container(value)
    if cont_name != "none":
        for mode in METHOD_ORDER:
            target[mode] = value
        return combos
    for candidate in ("records", "data", "bundle"):
        if candidate in value and isinstance(value[candidate], dict):
            sub = value[candidate]
            for mode in METHOD_ORDER:
                if mode in sub and isinstance(sub[mode], dict):
                    target[mode] = sub[mode]
            if target:
                return combos
    return combos

def _merge_top_as_groups(top: Dict[Any, Any]) -> Dict[Any, Dict[str, Dict[str, Any]]]:
    groups: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    for key, val in top.items():
        if not isinstance(key, tuple):
            continue
        loss = _loss_from_tuple_key(key)
        if loss is None:
            continue
        gkey = _base_group_key(key)
        groups.setdefault(gkey, {"mse": {}, "softdtw": {}})
        vb = _extract_current_format_into_virtual_bundle(val, loss)
        for mode, records in vb[loss].items():
            groups[gkey][loss][mode] = records
    return groups

# ---------- FINAL VALUES FALLBACK ----------
def _get_final_pair_from_records(rec: Dict[str, Any], mode: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        fv = rec.get("final_values", {})
        Gv = fv.get("G_values", {}) or {}
        gv = fv.get("g_values", {}) or {}

        if mode == "with_solver":
            candidates = ["a_with", "with_solver"]
        elif mode == "without_solver":
            candidates = ["a_without", "without_solver", "detached"]
        elif mode == "approximate":
            candidates = ["a_approximate", "approximate", "approximated"]
        else:
            candidates = []

        def pick(d, prefix):
            keys = list(d.keys())
            for c in candidates:
                patterns = (f"{prefix}({c})", f"{prefix}_{c}", f"{prefix} {c}", f"{prefix}{c}")
                for k in keys:
                    if any(p in k for p in patterns) or (c in k and (k.startswith(prefix) or prefix in k)):
                        arr = _as_1d(_safe_array(d.get(k, None)))
                        if arr is not None and arr.size:
                            return arr
            for k in keys:
                if k.startswith(prefix) and any(c in k for c in candidates):
                    arr = _as_1d(_safe_array(d.get(k, None)))
                    if arr is not None and arr.size:
                        return arr
            return None

        G_fin = pick(Gv, "G")
        g_fin = pick(gv, "g")
        return G_fin, g_fin
    except Exception:
        return None, None

def _get_final_inputs_from_records(rec: Dict[str, Any], mode: str) -> Optional[np.ndarray]:
    try:
        fv = rec.get("final_values", {})
        av = fv.get("a_values", {}) or {}
        if mode == "with_solver":
            candidates = ["a_with", "with_solver"]
        elif mode == "without_solver":
            candidates = ["a_without", "without_solver", "detached"]
        elif mode == "approximate":
            candidates = ["a_approximate", "approximate", "approximated"]
        else:
            candidates = []

        def pick_a(d):
            keys = list(d.keys())
            for c in candidates:
                patterns = (f"a({c})", f"a_{c}", f"a {c}")
                for k in keys:
                    if any(p in k for p in patterns) or ("a" in k and c in k):
                        arr = _as_1d(_safe_array(d.get(k, None)))
                        if arr is not None and arr.size:
                            return arr
            return None

        a_fin = pick_a(av)
        return a_fin
    except Exception:
        return None

def _collect_all_finals(combos: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]]:
    finals: Dict[str, Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = {"mse": {}, "softdtw": {}}
    for loss in ("mse", "softdtw"):
        for mode in METHOD_ORDER:
            rec = combos.get(loss, {}).get(mode)
            if isinstance(rec, dict):
                G_fin, g_fin = _get_final_pair_from_records(rec, mode)
                finals[loss][mode] = (G_fin, g_fin)
            else:
                finals[loss][mode] = (None, None)
    return finals

# ---------- Plotting ----------
def _plot_inputs(ax, clean_a: Optional[np.ndarray], lines: List[Tuple[Optional[np.ndarray], str, str, float]], ylim):
    ax.clear(); ax.set_facecolor("white")
    has_any = False
    if clean_a is not None and clean_a.size:
        x = np.arange(clean_a.size); ax.plot(x, clean_a, label="a (clean)", color=PALETTE["black_deep"], linewidth=1.6); has_any = True
    for arr, label, color, lw in lines:
        if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0): continue
        x = np.arange(arr.size); ax.plot(x, arr, label=label, color=color, linewidth=lw); has_any = True
    if not has_any:
        ax.text(0.5, 0.5, "NO DATA (per-step & final missing)", ha="center", va="center",
                alpha=0.9, fontsize=12, color="red", transform=ax.transAxes)
    ax.set_ylim(ylim); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

def _plot_fno_vs_pde(ax, G_arr: Optional[np.ndarray], T_arr: Optional[np.ndarray],
                     clean_G: Optional[np.ndarray], clean_T: Optional[np.ndarray],
                     deep_c: str, light_c: str, ylim, method_title: str):
    ax.clear(); ax.set_facecolor("white")
    drew = False
    if clean_G is not None and clean_G.size:
        x = np.arange(clean_G.size); ax.plot(x, clean_G, label='FNO(a)', color=PALETTE["black_deep"], linewidth=1.6); drew = True
    if clean_T is not None and clean_T.size:
        x = np.arange(clean_T.size); ax.plot(x, clean_T, label='PDE(a)', color=PALETTE["black_light"], linewidth=1.4); drew = True
    if G_arr is not None and isinstance(G_arr, np.ndarray) and G_arr.size:
        x = np.arange(G_arr.size); ax.plot(x, G_arr, label='FNO(a + δ)', color=deep_c, linewidth=1.6); drew = True
    if T_arr is not None and isinstance(T_arr, np.ndarray) and T_arr.size:
        x = np.arange(T_arr.size); ax.plot(x, T_arr, label='PDE(a + δ)', color=light_c, linewidth=1.4); drew = True
    ax.set_title(method_title + " : FNO vs PDE")
    if not drew:
        ax.text(0.5, 0.5, "NO DATA (per-step & final missing)", ha="center", va="center",
                alpha=0.9, fontsize=12, color="red", transform=ax.transAxes)
    ax.set_ylim(ylim); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

def _plot_loss(ax, x_steps: np.ndarray, histories: Dict[str, np.ndarray], ylim, title: str):
    ax.clear(); ax.set_facecolor("white")
    ax.plot(x_steps, histories.get("with_solver",  np.full_like(x_steps, np.nan, dtype=float)),
            label=DISPLAY_LABEL["with_solver"], color=PALETTE["orange_deep"], linewidth=1.6)
    ax.plot(x_steps, histories.get("without_solver", np.full_like(x_steps, np.nan, dtype=float)),
            label=DISPLAY_LABEL["without_solver"], color=PALETTE["blue_deep"],   linewidth=1.6)
    ax.plot(x_steps, histories.get("approximate", np.full_like(x_steps, np.nan, dtype=float)),
            label=DISPLAY_LABEL["approximate"],   color=PALETTE["red_deep"],    linewidth=1.6)
    ax.set_title(title); ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.set_ylim(ylim)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ---------- Data access helpers ----------
def _get_step_dict(rec: Dict[str, Any]) -> Dict[str, Any]:
    name, cont = _detect_step_container(rec)
    return cont if isinstance(cont, dict) else {}

def _get_value_at(rec: Dict[str, Any], k: int, field: str, mode_hint: str = "with_solver") -> Optional[np.ndarray]:
    """
    先尝试标准位：step_k/{input,G_out,T_out}；
    若缺失，则从 step_k/values/{a_values,G_values,g_values} 里，按 mode_hint 回退读取。
    """
    steps = _get_step_dict(rec)
    d = steps.get(f"step_{k}", None)
    if not isinstance(d, dict):
        return None

    # 1) 尝试标准键
    if field in ("input", "G_out", "T_out"):
        arr = _as_1d(_safe_array(d.get(field, None)))
        if arr is not None and arr.size:
            return arr

    # 2) 回退读取保存端 values 结构
    vals = d.get("values", {})
    if not isinstance(vals, dict):
        return None
    a_vals = vals.get("a_values", {})
    G_vals = vals.get("G_values", {})
    g_vals = vals.get("g_values", {})

    a_key, G_key, g_key = MODE_TO_SUFFIX.get(mode_hint, (None, None, None))

    if field == "input" and isinstance(a_vals, dict) and a_key in a_vals:
        arr = _as_1d(_safe_array(a_vals.get(a_key)))
        return arr if (arr is not None and arr.size) else None
    if field == "G_out" and isinstance(G_vals, dict) and G_key in G_vals:
        arr = _as_1d(_safe_array(G_vals.get(G_key)))
        return arr if (arr is not None and arr.size) else None
    if field == "T_out" and isinstance(g_vals, dict) and g_key in g_vals:
        arr = _as_1d(_safe_array(g_vals.get(g_key)))
        return arr if (arr is not None and arr.size) else None

    return None

def _get_loss_at(rec: Dict[str, Any], k: int, loss_name: str, mode_hint: str = "with_solver") -> Optional[float]:
    """
    先读标准位 step_k/loss/{mse,softdtw}；没有则从 step_k/loss_metrics 里
    取对应模式（with_solver/without_solver/approximate）的数值。
    """
    steps = _get_step_dict(rec)
    d = steps.get(f"step_{k}", None)
    if not isinstance(d, dict):
        return None

    # 标准位
    l = d.get("loss", {})
    if isinstance(l, dict) and loss_name in l:
        try:
            v = float(l[loss_name])
            return v if np.isfinite(v) else None
        except Exception:
            pass

    # 回退：保存端结构 step_k/loss_metrics/{with_solver,without_solver,approximate}
    lm = d.get("loss_metrics", {})
    if isinstance(lm, dict):
        key_by_mode = {
            "with_solver":   "with_solver",
            "without_solver":"without_solver",
            "approximate":   "approximate",
        }
        mkey = key_by_mode.get(mode_hint)
        if mkey in lm:
            try:
                v = float(lm[mkey])
                return v if np.isfinite(v) else None
            except Exception:
                return None
    return None

def _union_all_steps(combos: Dict[str, Dict[str, Any]]) -> List[int]:
    all_steps = set()
    for loss_name in ("mse", "softdtw"):
        for mode in METHOD_ORDER:
            rec = combos.get(loss_name, {}).get(mode, None)
            if rec is None:
                continue
            steps = _get_step_dict(rec)
            all_steps.update(_sorted_steps_from_container(steps))
    out = sorted(all_steps)
    return out if out else [0]

def _finite_minmax(arr: np.ndarray) -> Tuple[float, float, bool]:
    if arr is None or not isinstance(arr, np.ndarray) or arr.size == 0:
        return (0.0, 1.0, False)
    mask = np.isfinite(arr)
    if not mask.any():
        return (0.0, 1.0, False)
    vals = arr[mask]
    return (float(vals.min()), float(vals.max()), True)

def _compute_global_limits(combos: Dict[str, Dict[str, Any]]) -> Tuple[Tuple[float,float], Tuple[float,float], Dict[Tuple[str,str],Tuple[float,float]]]:
    a_lo, a_hi = np.inf, -np.inf
    out_lo, out_hi = np.inf, -np.inf
    loss_bounds = {
        ("mse", "mse"): [np.inf, -np.inf],
        ("mse", "softdtw"): [np.inf, -np.inf],
        ("softdtw", "mse"): [np.inf, -np.inf],
        ("softdtw", "softdtw"): [np.inf, -np.inf],
    }

    # per-step
    for loss_obj in ("mse", "softdtw"):
        for mode in METHOD_ORDER:
            rec = combos.get(loss_obj, {}).get(mode, None)
            if rec is None:
                continue
            steps = _get_step_dict(rec)
            for sk, sdict in steps.items():
                if not (isinstance(sk, str) and sk.startswith("step_") and isinstance(sdict, dict)):
                    continue
                a_arr = _as_1d(_safe_array(sdict.get("input", None)))
                G_arr = _as_1d(_safe_array(sdict.get("G_out", None)))
                T_arr = _as_1d(_safe_array(sdict.get("T_out", None)))
                lo, hi, ok = _finite_minmax(a_arr);   a_lo = min(a_lo, lo) if ok else a_lo; a_hi = max(a_hi, hi) if ok else a_hi
                lo, hi, ok = _finite_minmax(G_arr);   out_lo = min(out_lo, lo) if ok else out_lo; out_hi = max(out_hi, hi) if ok else out_hi
                lo, hi, ok = _finite_minmax(T_arr);   out_lo = min(out_lo, lo) if ok else out_lo; out_hi = max(out_hi, hi) if ok else out_hi
                ldict = sdict.get("loss", {})
                if isinstance(ldict, dict):
                    for show_loss in ("mse", "softdtw"):
                        v = ldict.get(show_loss, None)
                        try:
                            fv = float(v)
                            if np.isfinite(fv):
                                key = (loss_obj, show_loss)
                                loss_bounds[key][0] = min(loss_bounds[key][0], fv)
                                loss_bounds[key][1] = max(loss_bounds[key][1], fv)
                        except Exception:
                            pass

    # finals
    finals = _collect_all_finals(combos)
    for loss_obj in ("mse", "softdtw"):
        for mode in METHOD_ORDER:
            G_fin, g_fin = finals.get(loss_obj, {}).get(mode, (None, None))
            lo, hi, ok = _finite_minmax(G_fin); out_lo = min(out_lo, lo) if ok else out_lo; out_hi = max(out_hi, hi) if ok else out_hi
            lo, hi, ok = _finite_minmax(g_fin); out_lo = min(out_lo, lo) if ok else out_lo; out_hi = max(out_hi, hi) if ok else out_hi

    ylim_a = _pad_ylim(a_lo, a_hi)
    ylim_out = _pad_ylim(out_lo, out_hi)
    ylim_loss_map: Dict[Tuple[str,str], Tuple[float,float]] = {}
    for k, (lo, hi) in loss_bounds.items():
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = 0.0, max(1.0, hi if np.isfinite(hi) else 1.0)
        ylim_loss_map[k] = _pad_ylim(lo, hi)
    return ylim_a, ylim_out, ylim_loss_map

def _build_histories(combos: Dict[str, Dict[str, Any]], show_loss: str, objective: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    master_steps = set()
    for mode in METHOD_ORDER:
        rec = combos.get(objective, {}).get(mode, None)
        if rec is None:
            continue
        master_steps.update(_sorted_steps_from_container(_get_step_dict(rec)))
    master = np.array(sorted(master_steps), dtype=int)
    if master.size == 0:
        master = np.arange(1, dtype=int)
    histories = {}
    for mode in METHOD_ORDER:
        rec = combos.get(objective, {}).get(mode, None)
        vals = np.full_like(master, np.nan, dtype=float)
        if rec is not None:
            for i, kk in enumerate(master):
                v = _get_loss_at(rec, int(kk), show_loss, mode_hint=mode)  # 传 mode
                vals[i] = v if (v is not None and np.isfinite(v)) else np.nan
        histories[mode] = vals
    return master, histories

# ---------- Rendering ----------
def render_gif_for_bundle(attack_key, combos: Dict[str, Dict[str, Any]], outdir: str,
                          fps: int = 8,
                          debug: bool = True, debug_every: int = 1, debug_head: Optional[int] = None,
                          audit: bool = False, audit_every: int = 1, audit_head: Optional[int] = None):
    # any record?
    any_rec = None
    for ln in ("mse", "softdtw"):
        for m in METHOD_ORDER:
            if combos.get(ln, {}).get(m, None) is not None:
                any_rec = combos[ln][m]; break
        if any_rec: break
    if any_rec is None:
        print(f"[WARN] Empty bundle for key={attack_key}, skip")
        return

    # Inspect finals presence (one-off)
    if debug or audit:
        print(f"[DBG] Inspect final_values availability for key={attack_key}:")
        for loss_name in ("mse","softdtw"):
            for mode in METHOD_ORDER:
                rec = combos.get(loss_name, {}).get(mode)
                tag = f"[{loss_name}][{mode}]"
                if not isinstance(rec, dict):
                    print(f"  {tag}: records=None"); continue
                fv = rec.get("final_values", {})
                Gv = fv.get("G_values", {})
                gv = fv.get("g_values", {})
                aV = fv.get("a_values", {})
                print(f"  {tag}: G_keys={list(Gv.keys())[:3]}{'...' if len(Gv)>3 else ''} | "
                      f"g_keys={list(gv.keys())[:3]}{'...' if len(gv)>3 else ''} | "
                      f"a_keys={list(aV.keys())[:3]}{'...' if len(aV)>3 else ''}")

    clean_a, clean_G, clean_T = _grab_clean_a_G_T(any_rec)
    ylim_a, ylim_out, ylim_loss_map = _compute_global_limits(combos)
    step_list = _union_all_steps(combos)
    finals = _collect_all_finals(combos)

    # safe filename
    if isinstance(attack_key, (tuple, list)):
        key_str = "__".join(map(str, attack_key))
    else:
        key_str = str(attack_key)
    safe_key = "".join(c if c.isalnum() or c in "._-+=" else "_" for c in key_str)
    gif_path = os.path.join(outdir, f"{safe_key}.gif")

    frames = []
    for step_idx, k in enumerate(tqdm(step_list, desc=f"Rendering {key_str}")):
        do_dbg = debug and (debug_every > 0) and (step_idx % debug_every == 0) and (debug_head is None or step_idx < debug_head)
        do_audit = audit and (audit_every > 0) and (step_idx % audit_every == 0) and (audit_head is None or step_idx < audit_head)

        # Inputs per objective
        mode_color = {
            "with_solver": (PALETTE["orange_deep"], 1.2, f"a + δ ({DISPLAY_LABEL['with_solver']})"),
            "without_solver": (PALETTE["blue_deep"],  1.2, f"a + δ ({DISPLAY_LABEL['without_solver']})"),
            "approximate":   (PALETTE["red_deep"],   1.2, f"a + δ ({DISPLAY_LABEL['approximate']})"),
        }
        input_lines_mse, input_lines_sdtw = [], []

        # FNO vs PDE getter (per-step with final fallback)
        def outputs(loss_name, mode):
            rec = combos.get(loss_name, {}).get(mode, None)
            G_step = _get_value_at(rec, k, "G_out", mode_hint=mode) if rec else None
            T_step = _get_value_at(rec, k, "T_out", mode_hint=mode) if rec else None
            if (G_step is not None and G_step.size) or (T_step is not None and T_step.size):
                return G_step, T_step
            G_fin, g_fin = finals.get(loss_name, {}).get(mode, (None, None))
            return G_fin, g_fin

        # build inputs & audit/debug prints (with finals fallback for inputs)
        for mode in METHOD_ORDER:
            c, lw, lab = mode_color[mode]
            rec_m = combos.get("mse", {}).get(mode, None)
            rec_s = combos.get("softdtw", {}).get(mode, None)

            arr_m = _get_value_at(rec_m, k, "input", mode_hint=mode) if rec_m else None
            arr_s = _get_value_at(rec_s, k, "input", mode_hint=mode) if rec_s else None

            if (arr_m is None or (isinstance(arr_m, np.ndarray) and arr_m.size == 0)) and rec_m:
                a_fin_m = _get_final_inputs_from_records(rec_m, mode)
                if a_fin_m is not None and a_fin_m.size:
                    arr_m = a_fin_m
            if (arr_s is None or (isinstance(arr_s, np.ndarray) and arr_s.size == 0)) and rec_s:
                a_fin_s = _get_final_inputs_from_records(rec_s, mode)
                if a_fin_s is not None and a_fin_s.size:
                    arr_s = a_fin_s

            input_lines_mse.append((arr_m, lab, c, lw))
            input_lines_sdtw.append((arr_s, lab, c, lw))

            if do_audit:
                _print_step_audit(k, f"inputs|{mode}|MSE",   "a+δ (obj=MSE)",   arr_m)
                _print_step_audit(k, f"inputs|{mode}|SDTW",  "a+δ (obj=SDTW)",  arr_s)

        # Outputs (per-step or final)
        G_w_m, T_w_m = outputs("mse", "with_solver")
        G_w_s, T_w_s = outputs("softdtw", "with_solver")
        G_d_m, T_d_m = outputs("mse", "without_solver")
        G_d_s, T_d_s = outputs("softdtw", "without_solver")
        G_a_m, T_a_m = outputs("mse", "approximate")
        G_a_s, T_a_s = outputs("softdtw", "approximate")

        if do_audit:
            _print_step_audit(k, "with_solver|MSE",   "G_out", G_w_m); _print_step_audit(k, "with_solver|MSE",   "T_out", T_w_m)
            _print_step_audit(k, "with_solver|SDTW",  "G_out", G_w_s); _print_step_audit(k, "with_solver|SDTW",  "T_out", T_w_s)
            _print_step_audit(k, "detached|MSE",      "G_out", G_d_m); _print_step_audit(k, "detached|MSE",      "T_out", T_d_m)
            _print_step_audit(k, "detached|SDTW",     "G_out", G_d_s); _print_step_audit(k, "detached|SDTW",     "T_out", T_d_s)
            _print_step_audit(k, "approximated|MSE",  "G_out", G_a_m); _print_step_audit(k, "approximated|MSE",  "T_out", T_a_m)
            _print_step_audit(k, "approximated|SDTW", "G_out", G_a_s); _print_step_audit(k, "approximated|SDTW", "T_out", T_a_s)

        # Loss histories (truncate to <= k)
        xs_11, hist_11 = _build_histories(combos, show_loss="mse",     objective="mse")
        xs_12, hist_12 = _build_histories(combos, show_loss="softdtw", objective="mse")
        xs_13, hist_13 = _build_histories(combos, show_loss="mse",     objective="softdtw")
        xs_14, hist_14 = _build_histories(combos, show_loss="softdtw", objective="softdtw")
        def trunc(xs, h):
            mask = xs <= k
            return xs[mask], {m: v[mask] for m, v in h.items()}
        xs_11, hist_11 = trunc(xs_11, hist_11)
        xs_12, hist_12 = trunc(xs_12, hist_12)
        xs_13, hist_13 = trunc(xs_13, hist_13)
        xs_14, hist_14 = trunc(xs_14, hist_14)

        if do_dbg:
            for loss_name in ("mse","softdtw"):
                for mode in METHOD_ORDER:
                    rec = combos.get(loss_name, {}).get(mode)
                    inp = _get_value_at(rec, k, "input", mode_hint=mode) if rec else None
                    Gs  = _get_value_at(rec, k, "G_out",  mode_hint=mode) if rec else None
                    Ts  = _get_value_at(rec, k, "T_out",  mode_hint=mode) if rec else None
                    lm  = _get_loss_at(rec, k, "mse",     mode_hint=mode) if rec else None
                    ls  = _get_loss_at(rec, k, "softdtw", mode_hint=mode) if rec else None
                    Gf, gf = finals.get(loss_name, {}).get(mode, (None, None))
                    _print_step_debug(k, loss_name, mode, inp, Gs, Ts, lm, ls, Gf, gf)

        # Figure 3x4
        plt.rcParams["savefig.facecolor"] = "white"
        plt.rcParams["figure.facecolor"] = "white"
        fig = plt.figure(figsize=(18, 11), facecolor="white")
        gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)
        ax11 = fig.add_subplot(gs[0,0]); ax12 = fig.add_subplot(gs[0,1])
        ax13 = fig.add_subplot(gs[0,2]); ax14 = fig.add_subplot(gs[0,3])
        ax21 = fig.add_subplot(gs[1,0]); ax22 = fig.add_subplot(gs[1,1])
        ax23 = fig.add_subplot(gs[1,2]); ax24 = fig.add_subplot(gs[1,3])
        ax31 = fig.add_subplot(gs[2,0]); ax32 = fig.add_subplot(gs[2,1])
        ax33 = fig.add_subplot(gs[2,2]); ax34 = fig.add_subplot(gs[2,3])

        # Row 1 — Inputs
        _plot_inputs(ax11, clean_a, input_lines_mse,  ylim_a); ax11.set_title("Inputs (objective: MSE)")
        _plot_inputs(ax12, clean_a, input_lines_sdtw, ylim_a); ax12.set_title("Inputs (objective: SoftDTW)")

        # Row 1 — with_solver
        _plot_fno_vs_pde(ax13, G_w_m, T_w_m, clean_G, clean_T, PALETTE["orange_deep"], PALETTE["orange_light"],
                         ylim_out, f"{DISPLAY_LABEL['with_solver']} (objective: MSE)")
        _plot_fno_vs_pde(ax14, G_w_s, T_w_s, clean_G, clean_T, PALETTE["orange_deep"], PALETTE["orange_light"],
                         ylim_out, f"{DISPLAY_LABEL['with_solver']} (objective: SoftDTW)")

        # Row 2 — detached & approximated
        _plot_fno_vs_pde(ax21, G_d_m, T_d_m, clean_G, clean_T, PALETTE["blue_deep"], PALETTE["blue_light"],
                         ylim_out, f"{DISPLAY_LABEL['without_solver']} (objective: MSE)")
        _plot_fno_vs_pde(ax22, G_d_s, T_d_s, clean_G, clean_T, PALETTE["blue_deep"], PALETTE["blue_light"],
                         ylim_out, f"{DISPLAY_LABEL['without_solver']} (objective: SoftDTW)")

        _plot_fno_vs_pde(ax23, G_a_m, T_a_m, clean_G, clean_T, PALETTE["red_deep"], PALETTE["red_light"],
                         ylim_out, f"{DISPLAY_LABEL['approximate']} (objective: MSE)")
        _plot_fno_vs_pde(ax24, G_a_s, T_a_s, clean_G, clean_T, PALETTE["red_deep"], PALETTE["red_light"],
                         ylim_out, f"{DISPLAY_LABEL['approximate']} (objective: SoftDTW)")

        # Row 3 — Loss
        _plot_loss(ax31, xs_11, hist_11, ylim_loss_map[("mse","mse")],         "Loss (objective: MSE, show: MSE)")
        _plot_loss(ax32, xs_12, hist_12, ylim_loss_map[("mse","softdtw")],     "Loss (objective: MSE, show: SoftDTW)")
        _plot_loss(ax33, xs_13, hist_13, ylim_loss_map[("softdtw","mse")],     "Loss (objective: SoftDTW, show: MSE)")
        _plot_loss(ax34, xs_14, hist_14, ylim_loss_map[("softdtw","softdtw")], "Loss (objective: SoftDTW, show: SoftDTW)")

        def _all_nan_hist(h): return all(np.all(np.isnan(v)) for v in h.values())
        if _all_nan_hist(hist_11): ax31.text(0.5, 0.5, "no per-step losses saved", ha="center", va="center", alpha=0.6, fontsize=10, transform=ax31.transAxes)
        if _all_nan_hist(hist_12): ax32.text(0.5, 0.5, "no per-step losses saved", ha="center", va="center", alpha=0.6, fontsize=10, transform=ax32.transAxes)
        if _all_nan_hist(hist_13): ax33.text(0.5, 0.5, "no per-step losses saved", ha="center", va="center", alpha=0.6, fontsize=10, transform=ax33.transAxes)
        if _all_nan_hist(hist_14): ax34.text(0.5, 0.5, "no per-step losses saved", ha="center", va="center", alpha=0.6, fontsize=10, transform=ax34.transAxes)

        fig.suptitle(f"{key_str} | step {k}", fontsize=14)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frame = rgba[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

        # 全局“有没有东西画”的提示
        something_drawn = any([
            (G_w_m is not None and G_w_m.size), (T_w_m is not None and T_w_m.size),
            (G_w_s is not None and G_w_s.size), (T_w_s is not None and T_w_s.size),
            (G_d_m is not None and G_d_m.size), (T_d_m is not None and T_d_m.size),
            (G_d_s is not None and G_d_s.size), (T_d_s is not None and T_d_s.size),
            (G_a_m is not None and G_a_m.size), (T_a_m is not None and T_a_m.size),
            (G_a_s is not None and G_a_s.size), (T_a_s is not None and T_a_s.size),
            (clean_a is not None and isinstance(clean_a, np.ndarray) and clean_a.size),
            (clean_G is not None and isinstance(clean_G, np.ndarray) and clean_G.size),
            (clean_T is not None and isinstance(clean_T, np.ndarray) and clean_T.size),
        ])
        if do_audit and not something_drawn:
            print(f"[AUDIT][WARN] step={k}: no data to draw (per-step and final both missing)")

    os.makedirs(outdir, exist_ok=True)
    if len(frames) == 0:
        print(f"[WARN] key={attack_key}: no frames collected → skip writing GIF: {gif_path}")
        return
    try:
        iio.imwrite(gif_path, frames, fps=fps)
        print(f"[OK] Saved {gif_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write GIF {gif_path}: {e}")

# ---------- Folder driver ----------
def _prepare_bundles_from_top(top: Dict[Any, Any]) -> List[Tuple[Any, Dict[str, Dict[str, Any]]]]:
    items: List[Tuple[Any, Dict[str, Dict[str, Any]]]] = []
    pass_through = False
    for _, v in top.items():
        if isinstance(v, dict) and _is_expected_combo_bundle(v):
            pass_through = True
            break
    if pass_through:
        for attack_key, bundle in top.items():
            combos = _extract_expected_bundle(bundle)
            items.append((attack_key, combos))
        return items
    groups = _merge_top_as_groups(top)
    if groups:
        for gkey, combos in groups.items():
            items.append((gkey, combos))
        return items
    for k, v in top.items():
        if not isinstance(v, dict):
            continue
        loss = None
        if isinstance(k, tuple):
            loss = _loss_from_tuple_key(k)
            base = _base_group_key(k)
        else:
            base = k
        if loss is None:
            for ln in ("mse", "softdtw"):
                if f"loss_{ln}" in v:
                    loss = ln
                    v = v[f"loss_{ln}"]
                    break
        combos = {"mse": {}, "softdtw": {}}
        if loss is not None:
            vb = _extract_current_format_into_virtual_bundle(v, loss)
            combos = vb
        items.append((base, combos))
    return items

def render_folder(pickle_folder: str, outdir: str, fps: int = 8,
                  debug: bool = True, debug_every: int = 1, debug_head: Optional[int] = None,
                  audit: bool = False, audit_every: int = 1, audit_head: Optional[int] = None):
    folder = Path(pickle_folder)
    pkls = sorted([p for p in folder.glob("*.pkl") if p.is_file()])
    print(f"Found {len(pkls)} pickle files in {folder}:")
    for p in pkls:
        print("  -", p.name)

    for idx, path in enumerate(pkls):
        print(f"\n[LOAD] {path.name}")
        try:
            with open(path, "rb") as f:
                top = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load {path.name}: {e}")
            continue

        if audit:
            if isinstance(top, dict):
                ks = list(top.keys())
                print(f"[AUDIT] top-level dict, #keys={len(ks)}; first 3 keys: {ks[:3]}")
                if ks:
                    print(f"[AUDIT] example key type: {type(ks[0]).__name__} → {ks[0]}")
            else:
                print(f"[AUDIT] top-level type={type(top)} (expect dict).")

        out_dir_this = os.path.join(outdir, path.stem)
        os.makedirs(out_dir_this, exist_ok=True)

        pairs = _prepare_bundles_from_top(top)
        if not pairs:
            print(f"[WARN] No renderable bundles in {path.name}")
            continue

        for attack_key, combos in pairs:
            if not any(combos.get(ln, {}) for ln in ("mse", "softdtw")):
                print(f"[WARN] Empty combos for key={attack_key}")

            if audit:
                try:
                    steps = _union_all_steps(combos)
                    head = steps[:5]
                    print(f"[AUDIT] key={attack_key} → steps={head}{'...' if len(steps) > 5 else ''} (total {len(steps)})")
                except Exception as e:
                    print(f"[AUDIT] step union failed for key={attack_key}: {e}")

            render_gif_for_bundle(
                attack_key, combos, out_dir_this, fps=fps,
                debug=debug, debug_every=debug_every, debug_head=debug_head,
                audit=audit, audit_every=audit_every, audit_head=audit_head
            )

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder that contains the generated *.pkl files")
    ap.add_argument("--outdir", required=True, help="Output folder for GIFs (will create subfolders per pickle)")
    ap.add_argument("--fps", type=int, default=8, help="Frames per second for GIFs (default: 8)")
    ap.add_argument("--debug", action="store_true", default=True, help="Enable verbose per-step debug prints")
    ap.add_argument("--debug-every", type=int, default=1, help="Print every N frames when debug enabled")
    ap.add_argument("--debug-head", type=int, default=None, help="Only print for the first N frames when debug enabled")
    ap.add_argument("--audit", action="store_true", default=False, help="Enable strict per-step auditing prints")
    ap.add_argument("--audit-every", type=int, default=1, help="Audit print every N steps")
    ap.add_argument("--audit-head", type=int, default=None, help="Only audit-print for the first N steps")
    args = ap.parse_args()

    render_folder(args.folder, args.outdir, args.fps,
                  debug=args.debug, debug_every=args.debug_every, debug_head=args.debug_head,
                  audit=args.audit, audit_every=args.audit_every, audit_head=args.audit_head)

if __name__ == "__main__":
    main()










