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
    only_with_surrogate=True,   # 新增：surrogate图默认只画有surrogate_loss的
    mark_fallback=True          # 新增：如果不跳过fallback，则在图例打[F]标记
):
    file_map = files_dict[param_key]

    data_per_file = {}
    max_loss = -float("inf")
    min_loss = float("inf")
    any_surrogate_missing = False
    skipped = []   # 记录因为only_with_surrogate而被跳过的modespec

    print(f"加载文件（组：{param_key}）...")
    for modespec, fpath in file_map.items():
        with open(fpath, "rb") as f:
            payload = pickle.load(f)

        # 只打印一次schema（你已有的函数）
        print_pickle_schema_once(fpath, payload)

        steps = payload.get("steps", payload)
        data_per_file[modespec] = steps

    num_steps = len(next(iter(data_per_file.values())))
    mode_specs = sorted(data_per_file.keys())

    fig, ax = plt.subplots(figsize=(24, 6))
    cmap = plt.cm.get_cmap("tab20")
    color_map = {ms: cmap(i % 20) for i, ms in enumerate(mode_specs)}

    for ms in mode_specs:
        recs = data_per_file[ms]
        has_sur = ("surrogate_loss" in recs[0]) if (isinstance(recs, list) and isinstance(recs[0], dict)) else False

        # 决定是否跳过
        if use_surrogate and only_with_surrogate and not has_sur:
            skipped.append(ms)
            continue

        # 选取要画的loss & legend
        if use_surrogate:
            if has_sur:
                losses = [r["surrogate_loss"] for r in recs]
                final_loss = recs[-1]["surrogate_loss"]
                label = f"{ms} [S] (final={final_loss:.2f})"
            else:
                losses = [r["loss"] for r in recs]
                final_loss = recs[-1]["loss"]
                any_surrogate_missing = True
                label = f"{ms} [F] (final={final_loss:.2f})" if mark_fallback else f"{ms} (final={final_loss:.2f})"
        else:
            losses = [r["loss"] for r in recs]
            final_loss = recs[-1]["loss"]
            label = f"{ms} (final={final_loss:.2f})"

        ax.plot(
            range(num_steps),
            losses,
            label=label,
            color=color_map[ms],
            marker="o",
            markersize=3,
            alpha=0.9,
            linewidth=2,
        )
        max_loss = max(max_loss, max(losses))
        min_loss = min(min_loss, min(losses))

    # 标题
    param_dict = {}
    for p in param_key:
        if "=" in p:
            k, v = p.split("=", 1)
            param_dict[k] = v
    title_lines = [f"{k}: {v}" for k, v in param_dict.items()]
    mode_title = "surrogate_loss" if use_surrogate else "true_loss"
    ax.set_title(f"ADAM PGD — compare mode_spec ({mode_title})\n" + "\n".join(title_lines), pad=16)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_xlim(-0.5, num_steps - 0.5)

    lo = min_loss if np.isfinite(min_loss) else 0.0
    hi = max_loss if np.isfinite(max_loss) else 1.0
    if hi == lo: hi = lo + 1.0
    ax.set_ylim(lo - 0.1 * abs(hi - lo), hi + 0.1 * abs(hi - lo))

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    ax.grid(True)
    plt.tight_layout()

    # 输出目录：true/ 与 surrogate/ 分开
    base_dir = Path(__file__).parent / "comparison_plots_modespec"
    sub_dir = "surrogate" if use_surrogate else "true"
    out_dir = base_dir / sub_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"modespec_comparison_{'_'.join(param_key)}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ 图像保存到: {out_path}")

    if use_surrogate:
        if only_with_surrogate and skipped:
            print(f"ℹ️ surrogate图已跳过无surrogate_loss的文件: {skipped}")
        elif any_surrogate_missing:
            print("⚠️ 有文件未记录 surrogate_loss，图中这些曲线标记为 [F]（fallback 到 loss）。")

# ========== 主入口 ==========

if __name__ == "__main__":
    folder_path = Path(__file__).parent / "pickle_files"  # 按你的目录放
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
