import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from collections import defaultdict
import pickle
import re

# ========== 解析与分组 ==========

def _parse_filename(fname_stem):
    """
    支持两种命名：
    1) 新：pgd_attack_records_modespec=aaaaaaaaaw_norm2_alpha1.0_epsilon13.1072_steps100_idx1|adam_b1...
    2) 旧（可选回退；不含 mode_spec）：
       pgd_attack_records_norm2_alpha1.0_epsilon13.1072_steps100_idx1_with_solver|adam...
    返回:
      ok, info(dict)
      其中 info 至少包含: modespec(可能为None), norm, alpha, epsilon, steps, idx
    """
    # 新命名（带 mode_spec）
    pat_new = re.compile(r"""
        ^pgd_attack_records_
        modespec=(?P<modespec>[adw]{10})_
        norm(?P<norm>\d+)_alpha(?P<alpha>[\d\.+eE+-]+)_
        epsilon(?P<epsilon>[\d\.+eE+-]+)_steps(?P<steps>\d+)_
        idx(?P<idx>\d+)
        (?:\|adam_.*)?$
    """, re.VERBOSE)

    m = pat_new.match(fname_stem)
    if m:
        return True, m.groupdict()

    # 旧命名（无 mode_spec）
    pat_old = re.compile(r"""
        ^pgd_attack_records_
        norm(?P<norm>\d+)_alpha(?P<alpha>[\d\.+eE+-]+)_
        epsilon(?P<epsilon>[\d\.+eE+-]+)_steps(?P<steps>\d+)_
        idx(?P<idx>\d+)
        (?:_.+)?$
    """, re.VERBOSE)

    m2 = pat_old.match(fname_stem)
    if m2:
        info = m2.groupdict()
        info["modespec"] = None
        return True, info

    return False, {}

def group_pkl_files_by_modespec(folder_path):
    """
    返回结构：
    {
      (norm, alpha, epsilon, steps, idx): {
          "<mode_spec or NO_MODE_SPEC>": "/abs/path/to/file.pkl",
          ...
      },
      ...
    }
    """
    folder = Path(folder_path)
    pkl_files = list(folder.glob("*.pkl"))
    grouped = defaultdict(dict)

    for path in pkl_files:
        ok, info = _parse_filename(path.stem)
        if not ok:
            print(f"Warning: 无法解析文件名: {path.name}")
            continue

        key = (
            f"norm={info['norm']}",
            f"alpha={info['alpha']}",
            f"epsilon={info['epsilon']}",
            f"steps={info['steps']}",
            f"idx={info['idx']}",
        )
        ms = info["modespec"] if info["modespec"] is not None else "NO_MODE_SPEC"
        grouped[key][ms] = str(path.resolve())

    return grouped

# ========== 打印 pickle 结构 ==========

def _shape_of(value):
    """返回 (shape_or_desc, dtype_or_None, type_name) """
    import numpy as _np
    if isinstance(value, _np.ndarray):
        return value.shape, str(value.dtype), "numpy.ndarray"
    # 常见标量
    if isinstance(value, (float, int, _np.floating, _np.integer, bool)):
        return "scalar", type(value).__name__, type(value).__name__
    # 具有 shape 属性的张量类（保守处理）
    if hasattr(value, "shape") and not isinstance(value, (list, tuple, dict)):
        try:
            shp = tuple(value.shape)
        except Exception:
            shp = None
        dt  = getattr(value, "dtype", None)
        if dt is not None: dt = str(dt)
        return shp, dt, type(value).__name__
    # 容器
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[len={len(value)}]", None, type(value).__name__
    if isinstance(value, dict):
        return f"dict[{len(value)} keys]", None, "dict"
    # 其他
    return None, None, type(value).__name__

_printed_cache = set()

def print_pickle_schema_once(pkl_path, payload):
    """同一文件只打印一次。打印顶层 keys 与其类型；对 steps[*] 的每个字段打印形状聚合。"""
    abspath = str(Path(pkl_path).resolve())
    if abspath in _printed_cache:
        return
    _printed_cache.add(abspath)

    print("\n" + "="*80)
    print(f"[PKL] {abspath}")
    if isinstance(payload, dict):
        top_keys = list(payload.keys())
        print("Top-level keys:", top_keys)
        for k in top_keys:
            v = payload[k]
            shp, dt, typ = _shape_of(v)
            print(f"  - {k:12s} => type={typ}, shape/desc={shp}, dtype={dt}")
    else:
        print(f"Top-level object type: {type(payload).__name__}")

    # 细化 steps
    steps = payload.get("steps") if isinstance(payload, dict) else None
    if isinstance(steps, list) and len(steps) > 0 and isinstance(steps[0], dict):
        print(f"\n[steps] length = {len(steps)}")
        # 聚合各 key 的形状（可能跨 step 变化）
        union_keys = set()
        for rec in steps:
            union_keys.update(rec.keys())
        union_keys = sorted(list(union_keys))

        for k in union_keys:
            shapes = set()
            dtypes = set()
            types  = set()
            present = 0
            for rec in steps:
                if k in rec:
                    present += 1
                    shp, dt, typ = _shape_of(rec[k])
                    shapes.add(str(shp))
                    if dt is not None: dtypes.add(str(dt))
                    types.add(typ)
            print(f"  * {k:18s} -> present={present}/{len(steps)}, "
                  f"shapes={{{{ {', '.join(sorted(shapes))} }}}}, "
                  f"dtypes={{{{ {', '.join(sorted(dtypes))} }}}}, "
                  f"types={{{{ {', '.join(sorted(types))} }}}}")
    else:
        print("[steps] not found or not a list of dicts.")
    print("="*80 + "\n")

# ========== 可视化 ==========

def visualize_attack_comparison_by_modespec(
    param_key,
    files_dict,
    use_surrogate=False,
    dpi=100,
    only_with_surrogate=True,   # for surrogate plot: only draw when surrogate_loss exists
    mark_fallback=True          # mark [F] if falling back to true loss on surrogate plot
):
    file_map = files_dict[param_key]

    # ---------- load data ----------
    data_per_file = {}
    print(f"Loading files (group: {param_key}) ...")
    for modespec, fpath in file_map.items():
        with open(fpath, "rb") as f:
            payload = pickle.load(f)
        print_pickle_schema_once(fpath, payload)
        steps = payload.get("steps", payload)
        data_per_file[modespec] = steps

    if not data_per_file:
        print("⚠️ No available data.")
        return

    # ---------- common ----------
    num_steps = len(next(iter(data_per_file.values())))
    mode_specs_all = sorted(data_per_file.keys())

    import numpy as _np
    from matplotlib import colors as _mcolors
    import matplotlib.pyplot as _plt

    # validity: exactly 10 chars from a/d/w
    def _valid(ms: str) -> bool:
        return isinstance(ms, str) and re.fullmatch(r"[adw]{10}", ms) is not None

    # selectors (for TRUE-LOSS multi-panel)
    def sel_none_a(ms: str) -> bool:
        return _valid(ms) and ('a' not in ms)

    def sel_last_a_only_once(ms: str) -> bool:
        return _valid(ms) and (ms.count('a') == 1 and ms.endswith('a'))

    def sel_many_a(ms: str) -> bool:
        return _valid(ms) and (ms.count('a') >= 2)

    def sel_last_is_a(ms: str) -> bool:
        return _valid(ms) and ms.endswith('a')

    def sel_last_is_w(ms: str) -> bool:
        return _valid(ms) and ms.endswith('w')

    # ------- dynamic palette chooser -------
    def _concat_listed_colors(names):
        cols = []
        for n in names:
            cmap = _plt.cm.get_cmap(n)
            if hasattr(cmap, "colors"):  # ListedColormap
                cols.extend(list(cmap.colors))
            else:
                # fall back: sample 256 then pick every ~12th to reduce similarity
                arr = cmap(_np.linspace(0, 1, 256))
                cols.extend([tuple(arr[i]) for i in range(0, 256, 12)])
        return cols

    def _hsv_n_colors(N: int):
        # evenly spaced hues, fixed s/v to ensure contrast
        hs = _np.linspace(0, 1, N, endpoint=False)
        rgb = [_mcolors.hsv_to_rgb((h, 0.65, 0.95)) for h in hs]
        return [tuple(c) for c in rgb]

    def _choose_colors(N: int):
        # try to match N with minimal waste and minimal repetition
        if N <= 0:
            return []
        if N <= 10:
            return list(_plt.cm.get_cmap("tab10").colors)[:N]
        if N <= 12:
            # prefer Set3 (12), then Paired (12)
            base = list(_plt.cm.get_cmap("Set3").colors)
            if len(base) < N:
                base = list(_plt.cm.get_cmap("Paired").colors)
            return base[:N]
        if N <= 20:
            return list(_plt.cm.get_cmap("tab20").colors)[:N]
        if N <= 40:
            cols = _concat_listed_colors(["tab20", "tab20b"])
            return cols[:N]
        if N <= 60:
            cols = _concat_listed_colors(["tab20", "tab20b", "tab20c"])
            return cols[:N]
        # very large N: fall back to HSV wheel
        return _hsv_n_colors(N)

    # ============ SURROGATE BRANCH (single-plot) ============
    if use_surrogate:
        # determine which mode_specs will actually be drawn
        used_ms = []
        for ms in mode_specs_all:
            recs = data_per_file[ms]
            has_sur = (isinstance(recs, list) and recs and isinstance(recs[0], dict)
                       and ("surrogate_loss" in recs[0]))
            if only_with_surrogate and not has_sur:
                continue
            used_ms.append(ms)

        # dynamic colors for exactly those
        used_ms = sorted(used_ms)
        colors_list = _choose_colors(len(used_ms))
        color_map = {ms: colors_list[i] for i, ms in enumerate(used_ms)}

        max_loss = -float("inf")
        min_loss = float("inf")
        any_surrogate_missing = False
        skipped = []

        fig, ax = _plt.subplots(figsize=(24, 6))

        for ms in mode_specs_all:
            recs = data_per_file[ms]
            has_sur = (isinstance(recs, list) and recs and isinstance(recs[0], dict)
                       and ("surrogate_loss" in recs[0]))

            if only_with_surrogate and not has_sur:
                skipped.append(ms)
                continue

            if has_sur:
                losses = [r["surrogate_loss"] for r in recs]
                first_loss = losses[0]
                final_loss = losses[-1]
                label = f"{ms} [S] (first={first_loss:.2f}, final={final_loss:.2f})"
            else:
                losses = [r["loss"] for r in recs]
                first_loss = losses[0]
                final_loss = losses[-1]
                any_surrogate_missing = True
                tag = " [F]" if mark_fallback else ""
                label = f"{ms}{tag} (first={first_loss:.2f}, final={final_loss:.2f})"

            ax.plot(
                range(num_steps), losses,
                label=label,
                color=color_map.get(ms, (0.6, 0.6, 0.6)),
                marker="o", markersize=3, alpha=0.9, linewidth=2,
            )
            max_loss = max(max_loss, float(_np.max(losses)))
            min_loss = min(min_loss, float(_np.min(losses)))

        # title and axes
        param_dict = {}
        for p in param_key:
            if "=" in p:
                k, v = p.split("=", 1)
                param_dict[k] = v
        title_lines = [f"{k}: {v}" for k, v in param_dict.items()]
        ax.set_title("ADAM PGD — compare mode_spec (surrogate_loss)\n" + "\n".join(title_lines), pad=18)

        ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_xlim(-0.5, num_steps - 0.5)
        lo = min_loss if _np.isfinite(min_loss) else 0.0
        hi = max_loss if _np.isfinite(max_loss) else 1.0
        if hi == lo: hi = lo + 1.0
        ax.set_ylim(lo - 0.1 * abs(hi - lo), hi + 0.1 * abs(hi - lo))

        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        ax.grid(True)
        _plt.tight_layout()

        # save
        base_dir = Path(__file__).parent / "comparison_plots_modespec"
        out_dir = base_dir / "surrogate"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"modespec_comparison_{'_'.join(param_key)}.png"
        _plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        _plt.close()
        print(f"✅ Figure saved to: {out_path}")
        if only_with_surrogate and skipped:
            print(f"ℹ️ Skipped (no surrogate_loss recorded): {skipped}")
        elif any_surrogate_missing:
            print("⚠️ Some files miss surrogate_loss; curves marked [F] fall back to true loss.")
        return

    # ============ TRUE-LOSS BRANCH (7-row; independent y-lims; dynamic palette) ============

    # selectors (for TRUE-LOSS multi-panel)
    def sel_none_a(ms: str) -> bool:
        return _valid(ms) and ('a' not in ms)

    def sel_last_a_only_once(ms: str) -> bool:
        return _valid(ms) and (ms.count('a') == 1 and ms.endswith('a'))

    def sel_many_a(ms: str) -> bool:
        return _valid(ms) and (ms.count('a') >= 2)

    def sel_last_is_a(ms: str) -> bool:
        return _valid(ms) and ms.endswith('a')

    def sel_last_is_w(ms: str) -> bool:
        return _valid(ms) and ms.endswith('w')

    # NEW: last is 'd' and no 'a' before
    def sel_last_is_d_no_a_before(ms: str) -> bool:
        return _valid(ms) and ms.endswith('d') and ('a' not in ms[:-1])

    # NEW: last is 'd' and has at least one 'a' before
    def sel_last_is_d_with_a_before(ms: str) -> bool:
        return _valid(ms) and ms.endswith('d') and ('a' in ms[:-1])

    panels = [
        ("No 'a' in mode_spec (true_loss)",                  sel_none_a),
        ("Only last char is 'a' (and appears once)",         sel_last_a_only_once),
        ("'a' appears ≥ 2 times",                            sel_many_a),
        ("Last char is 'a' (any count)",                     sel_last_is_a),
        ("Last char is 'w' (any count)",                     sel_last_is_w),
        ("Last char is 'd' and no 'a' before",               sel_last_is_d_no_a_before),   # NEW
        ("Last char is 'd' and has 'a' before",              sel_last_is_d_with_a_before), # NEW
    ]

    # compute union of mode_specs that will actually be drawn across all panels
    used_ms = []
    for ms in mode_specs_all:
        if any(selector(ms) for _, selector in panels):
            used_ms.append(ms)
    used_ms = sorted(set(used_ms))

    # dynamic color selection for exactly those
    colors_list = _choose_colors(len(used_ms))
    color_map = {ms: colors_list[i] for i, ms in enumerate(used_ms)}

    nrows = len(panels)
    fig, axes = _plt.subplots(
        nrows=nrows, ncols=1,
        figsize=(24, 6 * nrows),      # height scales with number of panels
        sharex=True, sharey=False,    # independent y-lims per panel
        constrained_layout=True,
    )
    if not isinstance(axes, (list, _np.ndarray)):
        axes = [axes]

    # figure-level title (more headroom)
    param_dict = {}
    for p in param_key:
        if "=" in p:
            k, v = p.split("=", 1)
            param_dict[k] = v
    title_lines = [f"{k}: {v}" for k, v in param_dict.items()]
    fig.suptitle("ADAM PGD — compare mode_spec (true_loss)\n" + "\n".join(title_lines), y=1.12)

    # extra top padding & inter-panel spacing
    try:
        fig.set_constrained_layout_pads(w_pad=2/72, h_pad=12/72, wspace=0.10, hspace=0.20)
    except Exception:
        pass

    # draw each panel with its own y-range
    for i, (ax, (panel_title, selector)) in enumerate(zip(axes, panels), start=1):
        any_plotted = False
        panel_min = float("inf")
        panel_max = -float("inf")

        for ms in mode_specs_all:
            if not selector(ms):
                continue
            recs = data_per_file[ms]
            if not (isinstance(recs, list) and recs and isinstance(recs[0], dict) and ("loss" in recs[0])):
                continue

            losses = [r["loss"] for r in recs]
            first_loss = losses[0]
            final_loss = losses[-1]
            label = f"{ms} (first={first_loss:.2f}, final={final_loss:.2f})"

            ax.plot(
                range(num_steps),
                losses,
                label=label,
                color=color_map.get(ms, (0.6, 0.6, 0.6)),
                marker="o",
                markersize=3,
                alpha=0.9,
                linewidth=2,
            )
            panel_min = min(panel_min, float(_np.min(losses)))
            panel_max = max(panel_max, float(_np.max(losses)))
            any_plotted = True

        ax.set_title(panel_title, pad=10)
        ax.set_xlim(-0.5, num_steps - 0.5)

        if any_plotted and _np.isfinite(panel_min) and _np.isfinite(panel_max):
            if panel_max == panel_min:
                panel_max = panel_min + 1.0
            pad = 0.10 * abs(panel_max - panel_min)
            ax.set_ylim(panel_min - pad, panel_max + pad)

        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize=9)

        if i == len(panels):
            ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

        if not any_plotted:
            ax.text(
                0.5, 0.5, "(no matching curves in this panel)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, alpha=0.7
            )

    # save
    
    base_dir = Path(__file__).parent / "results" / folder / "comparison_plots_modespec"
    out_dir = base_dir / "true"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"modespec_comparison_{'_'.join(param_key)}.png"
    _plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    _plt.close()
    print(f"✅ Figure saved to: {out_path}")

# ========== 主入口 ==========

if __name__ == "__main__":
    # folder = "perturbation_results_1solver_1sample_all_dt_in_memory"
    folder = "perturbation_results_1solver_1sample_integer_frames"
    folder_path = Path(__file__).parent / "pickle_files" / folder  # 按你的目录放
    grouped = group_pkl_files_by_modespec(folder_path)

    # 概览
    print("分组键数量:", len(grouped))
    for k, v in grouped.items():
        print(k, "=>", list(v.keys()))

    # 为每个分组分别绘制 true_loss 与 surrogate_loss（分别落到不同子文件夹）
    for key_tuple in grouped:
        # true
        visualize_attack_comparison_by_modespec(key_tuple, grouped, use_surrogate=False, dpi=100)
        # surrogate（只画有surrogate_loss的）
        visualize_attack_comparison_by_modespec(key_tuple, grouped, use_surrogate=True, dpi=100, only_with_surrogate=True)
