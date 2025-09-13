from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import numpy as _np

_printed_cache = set()

def _shape_of(value):
    if isinstance(value, _np.ndarray):
        return value.shape, str(value.dtype), "numpy.ndarray"
    if isinstance(value, (float, int, _np.floating, _np.integer, bool)):
        return "scalar", type(value).__name__, type(value).__name__
    if hasattr(value, "shape") and not isinstance(value, (list, tuple, dict)):
        try:
            shp = tuple(value.shape)
        except Exception:
            shp = None
        dt  = getattr(value, "dtype", None)
        if dt is not None: dt = str(dt)
        return shp, dt, type(value).__name__
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[len={len(value)}]", None, type(value).__name__
    if isinstance(value, dict):
        return f"dict[{len(value)} keys]", None, "dict"
    return None, None, type(value).__name__

def print_pickle_schema_once(pkl_path, payload):
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

    steps = payload.get("steps") if isinstance(payload, dict) else None
    if isinstance(steps, list) and len(steps) > 0 and isinstance(steps[0], dict):
        print(f"\n[steps] length = {len(steps)}")
        union_keys = set()
        for rec in steps:
            union_keys.update(rec.keys())
        union_keys = sorted(list(union_keys))
        for k in union_keys:
            shapes = set(); dtypes = set(); types = set(); present = 0
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

def _get_last_step(payload):
    steps = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(steps, list) or len(steps) == 0:
        raise ValueError("steps 不存在或为空")
    return steps[-1]

def _get_modespec(payload):
    md = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return md.get("mode_spec", None)

# === 从 metadata 组装 run 目录名；若不全则回退到真实父目录名 ===
def _run_dir_name_from_meta_or_parent(payload, pkl_path: Path, fallback: str):
    md = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(md, dict): md = {}
    fields = {
        "mode_spec": md.get("mode_spec") or md.get("modespec"),
        "norm": md.get("norm"),
        "alpha": md.get("alpha"),
        "epsilon": md.get("epsilon") or md.get("eps"),
        "steps": md.get("num_steps") or md.get("steps"),
        "idx": md.get("index") or md.get("idx_tag") or md.get("idx"),
    }
    if all(v is not None for v in fields.values()):
        return "modespec={}_norm{}_alpha{}_epsilon{}_steps{}_{}".format(
            fields["mode_spec"], fields["norm"], fields["alpha"],
            fields["epsilon"], fields["steps"], fields["idx"]
        )
    # 元数据不全：直接用真实 run 目录名（sample_*.pkl 的父级的父级）
    try:
        return pkl_path.parents[1].name
    except Exception:
        return fallback

def _sym_range(*arrays):
    vmax = 0.0
    for a in arrays:
        if a is None:
            continue
        m = np.nanmax(np.abs(a))
        vmax = max(vmax, float(m))
    if vmax == 0: vmax = 1.0
    return (-vmax, vmax)

def _common_min_max(arrs):
    vmin = +np.inf
    vmax = -np.inf
    for a in arrs:
        if a is None:
            continue
        vmin = min(vmin, float(np.nanmin(a)))
        vmax = max(vmax, float(np.nanmax(a)))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = -1.0, 1.0
    return vmin, vmax

def plot_final_frames_for_file_dual_scales(pkl_path: Path, out_dir_base: Path, print_schema=True):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    if print_schema:
        print_pickle_schema_once(pkl_path, payload)

    last = _get_last_step(payload)

    # 取数据
    x0   = last.get("x0", None)
    pred = last.get("output", None)
    true = last.get("truth", None)
    surr = last.get("surrogate_truth", None)

    if x0 is None or pred is None:
        raise ValueError("缺少 x0 或 output（prediction）")

    # 末位是否为 'a'
    modespec = _get_modespec(payload) or ""
    ends_with_a = (isinstance(modespec, str) and len(modespec) == 10 and modespec.endswith("a"))

    # 统一尺度范围
    if ends_with_a:
        vmin_v, vmax_v = _common_min_max([x0, pred, true, surr])
    else:
        vmin_v, vmax_v = _common_min_max([x0, pred, true])

    diff_true = pred - true if true is not None else None
    diff_surr = pred - surr if surr is not None else None
    dmin, dmax = _sym_range(diff_true, diff_surr)

    # 布局
    nrows = 3 if ends_with_a else 2
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 7*nrows))

    def _imshow(ax, arr, title, cmap, vmin=None, vmax=None):
        ax.set_title(title, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        if arr is None:
            ax.text(0.5, 0.5, "(missing)", ha="center", va="center", fontsize=12, alpha=0.7)
            return
        im = ax.imshow(arr, cmap=cmap, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    # 统一尺度（左两列）
    if ends_with_a:
        _imshow(axes[0,0], x0,   "[Unified] Initial (x0)",          "viridis", vmin_v, vmax_v)
        _imshow(axes[0,1], pred, "[Unified] FNO Prediction (t=19)", "viridis", vmin_v, vmax_v)
        _imshow(axes[1,0], true, "[Unified] Solver Truth (t=19)",   "viridis", vmin_v, vmax_v)
        _imshow(axes[1,1], surr, "[Unified] Surrogate Truth (t=19)","viridis", vmin_v, vmax_v)
        _imshow(axes[2,0], diff_true, "[Unified] Pred - Solver",    "coolwarm", dmin, dmax)
        _imshow(axes[2,1], diff_surr, "[Unified] Pred - Surrogate", "coolwarm", dmin, dmax)
    else:
        _imshow(axes[0,0], x0,   "[Unified] Initial (x0)",          "viridis", vmin_v, vmax_v)
        _imshow(axes[0,1], pred, "[Unified] FNO Prediction (t=19)", "viridis", vmin_v, vmax_v)
        _imshow(axes[1,0], true, "[Unified] Ground Truth (solver)", "viridis", vmin_v, vmax_v)
        _imshow(axes[1,1], diff_true, "[Unified] Pred - Solver",    "coolwarm", dmin, dmax)

    # 自适应尺度（右两列）
    if ends_with_a:
        _imshow(axes[0,2], x0,   "[Auto] Initial (x0)",          "viridis")
        _imshow(axes[0,3], pred, "[Auto] FNO Prediction (t=19)", "viridis")
        _imshow(axes[1,2], true, "[Auto] Solver Truth (t=19)",   "viridis")
        _imshow(axes[1,3], surr, "[Auto] Surrogate Truth (t=19)","viridis")
        _imshow(axes[2,2], diff_true, "[Auto] Pred - Solver",    "coolwarm")
        _imshow(axes[2,3], diff_surr, "[Auto] Pred - Surrogate", "coolwarm")
    else:
        _imshow(axes[0,2], x0,   "[Auto] Initial (x0)",          "viridis")
        _imshow(axes[0,3], pred, "[Auto] FNO Prediction (t=19)", "viridis")
        _imshow(axes[1,2], true, "[Auto] Ground Truth (solver)", "viridis")
        _imshow(axes[1,3], diff_true, "[Auto] Pred - Solver",    "coolwarm")

    # 标题
    ms_txt = f" | modespec={modespec}" if modespec else ""
    fig.suptitle(f"{pkl_path.name}{ms_txt}", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # === 关键：输出目录按照 run 组织，包含 alpha/epsilon 等 ===
    run_dir_name = _run_dir_name_from_meta_or_parent(payload, pkl_path, fallback=pkl_path.stem)
    out_dir = out_dir_base / run_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = pkl_path.stem.split("|")[0] + ".png"   # sample_XXXXXX.png
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ========= 主入口 =========
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    folder = "perturbation_results_batch5_integer_frames"
    # 输入根目录：这一层下面是各个 run 目录
    folder_path = script_dir / "pickle_files" / folder
    # 递归查找 per-sample 的 pkl（关键修改）
    pkl_files = sorted(folder_path.rglob("per_sample/sample_*.pkl"))

    out_dir_base = script_dir / "results" / folder / "final_heatmaps_4col"

    if not pkl_files:
        print(f"未在 {folder_path} 下的 per_sample/ 子目录找到 sample_*.pkl 文件")
        raise SystemExit(0)

    print(f"将处理 {len(pkl_files)} 个 per-sample PKL")
    for p in tqdm(pkl_files, desc="Processing pickles (dual scales)"):
        try:
            out_png = plot_final_frames_for_file_dual_scales(p, out_dir_base, print_schema=False)
            tqdm.write(f"Saved: {out_png}")
        except Exception as e:
            tqdm.write(f"⚠️ Skip {p.name}: {e}")


