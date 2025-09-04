import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# --------- 1) 文件名解析器：尽量宽松但提取可靠 ----------
_num_pat = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

def parse_fname(fname_stem):
    """
    从文件名（不含 .pkl）中解析参数。
    返回: dict(alpha: float, epsilon: float, steps: int|None, idx: int|None, method: 'adam'|'norm', attack_tag: str)
    解析失败返回 None
    """
    s = fname_stem

    # alpha / epsilon / steps / idx
    m_alpha   = re.search(rf"alpha({_num_pat})", s)
    m_epsilon = re.search(rf"epsilon({_num_pat})", s)
    m_steps   = re.search(r"steps(\d+)", s)
    m_idx     = re.search(r"idx(\d+)", s)

    if not (m_alpha and m_epsilon):
        return None

    def norm_float(x):
        return float(x)

    alpha   = round(norm_float(m_alpha.group(1)),   12)
    epsilon = round(norm_float(m_epsilon.group(1)), 12)
    steps   = int(m_steps.group(1)) if m_steps else None
    idx     = int(m_idx.group(1))   if m_idx   else None

    # method: 只要文件名中包含“adam”字样（以下划线/竖线/连字符等分隔），就当作 adam
    method = "adam" if re.search(r"(?:^|[_\-\|])adam(?:$|[_\-\|])", s) or ("adam" in s) else "norm"

    # attack_tag：尽量取 idx 之后到分隔符/末尾的片段；找不到就回退为 'unknown'
    attack_tag = "unknown"
    m_tag = re.search(r"idx\d+[_\|](.+)$", s)
    if m_tag:
        attack_tag = m_tag.group(1)
    else:
        # 再尝试从 steps 后面拿
        m_tag2 = re.search(r"steps\d+[_\|](.+)$", s)
        if m_tag2:
            attack_tag = m_tag2.group(1)
        else:
            # 再退一步，从 norm/alpha/epsilon/steps/idx 的最后一个匹配之后截取
            # 这一步只是兜底，可能拿到较长的后缀
            pass

    return dict(alpha=alpha, epsilon=epsilon, steps=steps, idx=idx, method=method, attack_tag=attack_tag)


# --------- 2) 按 (alpha, epsilon) 聚合：返回 { (alpha,epsilon): { 'adam': {tag: path, ...}, 'norm': {...} } } ----------
def group_by_alpha_eps(folder, method_hint=None, verbose=False, max_print=200):
    folder = Path(folder)
    out = defaultdict(lambda: {'adam': {}, 'norm': {}})

    if verbose:
        print("\n" + "="*80)
        print(f"[SCAN] Folder: {folder.resolve()}")
        if not folder.exists():
            print("  !! 目录不存在")
        files = sorted(folder.glob("*.pkl"))
        print(f"  共发现 {len(files)} 个 .pkl 文件")
        for i, p in enumerate(files[:max_print]):
            print(f"    - {p.name}")
        if len(files) > max_print:
            print(f"    ... 其余 {len(files)-max_print} 个省略")
    else:
        files = list(folder.glob("*.pkl"))

    for p in files:
        stem = p.stem
        info = parse_fname(stem)
        if info is None:
            if verbose:
                print(f"[warn] 跳过（文件名无法解析 alpha/epsilon）: {p.name}")
            continue

        # 如果调用方显式指定 method_hint，以它为准；否则用解析得到的
        method = method_hint if method_hint in ('adam', 'norm') else info['method']
        key = (info['alpha'], info['epsilon'])

        tag = info['attack_tag'] or "unknown"
        # 避免同 tag 冲突：如碰到重名，附加序号
        base_tag = tag
        k = 1
        while tag in out[key][method]:
            tag = f"{base_tag}#{k}"
            k += 1

        out[key][method][tag] = str(p.resolve())

        if verbose:
            print(f"[OK] file={p.name}")
            print(f"     -> parsed: alpha={info['alpha']}, epsilon={info['epsilon']}, steps={info['steps']}, idx={info['idx']}")
            print(f"     -> method={method}, tag={tag}, key={key}")

    if verbose:
        # 打印该目录提取到的键
        keys = sorted(out.keys())
        print(f"[SUMMARY] 目录 {folder.name} 提取到的 (alpha, epsilon) 键数: {len(keys)}")
        for i, k in enumerate(keys[:max_print]):
            n_norm = len(out[k]['norm'])
            n_adam = len(out[k]['adam'])
            print(f"  key={k}  | norm:{n_norm} 条 | adam:{n_adam} 条")
        if len(keys) > max_print:
            print(f"  ... 其余 {len(keys)-max_print} 个键省略")

    return out


def normalize_tag(tag: str) -> str:
    if tag is None:
        return "unknown"
    t = str(tag).lower()
    # 统一命名
    t = t.replace("with_solver", "withsolver")
    # 正确地去掉 adam 的尾巴（不要用 \b）
    t = re.sub(r"\|adam.*$", "", t)
    # 清理首尾符号
    t = t.strip("_- |")
    return t

def build_common_groups(norm_folder, adam_folder, verbose=False):
    norm_groups = group_by_alpha_eps(norm_folder, method_hint="norm", verbose=verbose)
    adam_groups = group_by_alpha_eps(adam_folder, method_hint="adam", verbose=verbose)

    keys_norm = set(norm_groups.keys())
    keys_adam = set(adam_groups.keys())
    common_keys = sorted(keys_norm & keys_adam)

    if verbose:
        print("\n" + "="*80)
        print("[CROSS-CHECK] 键集合统计")
        print(f"  norm 目录键数: {len(keys_norm)}")
        print(f"  adam 目录键数: {len(keys_adam)}")
        print(f"  共有键数(交集): {len(common_keys)}")

    common = {}
    for key in common_keys:
        # 以“归一化后的 tag”为键做映射，便于求交
        norm_tag2path = {normalize_tag(tag): path
                         for tag, path in norm_groups[key]['norm'].items()}
        adam_tag2path = {normalize_tag(tag): path
                         for tag, path in adam_groups[key]['adam'].items()}

        # 共有的 detach/withsolver 等标签
        shared_tags = sorted(set(norm_tag2path.keys()) & set(adam_tag2path.keys()))

        if verbose:
            print(f"\n[COMMON] key={key}")
            print("  -> norm 原始文件：")
            for tag, path in norm_groups[key]['norm'].items():
                print(f"       [{tag}] {Path(path).name}  | normalized='{normalize_tag(tag)}'")
            print("  -> adam 原始文件：")
            for tag, path in adam_groups[key]['adam'].items():
                print(f"       [{tag}] {Path(path).name}  | normalized='{normalize_tag(tag)}'")
            if shared_tags:
                print("  => 共有的 normalized tag：", shared_tags)
            else:
                print("  => 没有共有的 normalized tag")

        if not shared_tags:
            # 没有共有的 detach/withsolver 就跳过该 (alpha, epsilon)
            continue

        # 用“归一化后的 tag”作为 key，分别挑选对应的文件路径
        common[key] = {
            'norm': {t: norm_tag2path[t] for t in shared_tags},
            'adam': {t: adam_tag2path[t] for t in shared_tags},
        }

    return common

# --------- 3) 读取 loss 序列（兼容 surrogate_*） ----------
def extract_losses(records, approximated_prefix="surrogate_"):
    """
    records: pickle['steps']
    返回: list[float] 的 loss 曲线
    """
    losses = []
    for r in records:
        if (approximated_prefix + 'loss') in r:
            losses.append(float(r[approximated_prefix + 'loss']))
        elif 'loss' in r:
            losses.append(float(r['loss']))
        else:
            # 兜底：没有 loss
            losses.append(np.nan)
    return losses

# === 辅助：统一/归一化 tag 名，合并 withsolver / with_solver 等 ===
def canonicalize_tag(tag: str) -> str:
    t = tag.lower()
    t = re.sub(r'#\d+$', '', t)               # 去掉 #1 这类后缀
    t = t.replace('|', '_').replace('-', '_').replace(' ', '_')
    t = re.sub(r'[^a-z0-9_]+', '', t)         # 去掉奇怪字符
    # 同义归并
    t = t.replace('withsolver', 'with_solver')
    t = t.replace('with__solver', 'with_solver')
    # 你还可以在这里按需补充别名映射
    return t

# === 辅助：颜色调亮（给 norm 用浅一点的同色） ===
def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    try:
        c = mc.cnames.get(color, color)
        r, g, b = mc.to_rgb(c)
    except Exception:
        # 如果传进来已经是 (r,g,b)，直接用
        r, g, b = color
    # 和白色按比例混合
    r = 1 - (1 - r) * (1 - amount)
    g = 1 - (1 - g) * (1 - amount)
    b = 1 - (1 - b) * (1 - amount)
    return (r, g, b)

# === 辅助：同一侧某个 tag 有多份时，选择“最优”那一份（步数最多）===
def _choose_best_path(paths):
    # paths: list[str]
    best = None
    best_len = -1
    for fp in paths:
        try:
            with open(fp, 'rb') as f:
                recs = pickle.load(f)['steps']
            n = len(recs)
        except Exception:
            n = -1
        if n > best_len:
            best_len = n
            best = fp
    return best

def visualize_common_alpha_epsilon(key, grouped_common, dpi=120, out_dir=None, title_prefix="PGD vs PGD-Adam"):
    """
    只画共有的 tag（detach/with_solver 等）：每个 tag 画两条线（adam + norm）。
    颜色：每个 tag 分配一种亮色；adam 用原色，norm 用同色浅色+虚线。
    """
    alpha, epsilon = key
    items = grouped_common[key]  # {'norm': {tag:path}, 'adam': {tag:path}}

    # 1) 归一化 tag，构建 {canon_tag: {'adam': [paths], 'norm': [paths], 'disp':原始显示名}}
    bucket = defaultdict(lambda: {'adam': [], 'norm': [], 'disp': None})

    # 收集 norm 侧
    for tag_raw, fp in items.get('norm', {}).items():
        canon = canonicalize_tag(tag_raw)
        bucket[canon]['norm'].append(fp)
        if bucket[canon]['disp'] is None:
            bucket[canon]['disp'] = tag_raw

    # 收集 adam 侧
    for tag_raw, fp in items.get('adam', {}).items():
        canon = canonicalize_tag(tag_raw)
        bucket[canon]['adam'].append(fp)
        if bucket[canon]['disp'] is None:
            bucket[canon]['disp'] = tag_raw

    # 2) 只保留“adam 与 norm 都有”的 canon_tag
    common_tags = [ct for ct, grp in bucket.items() if grp['adam'] and grp['norm']]
    common_tags = sorted(common_tags)
    if not common_tags:
        print(f"[info] (alpha={alpha}, epsilon={epsilon}) 没有共有的 tag，跳过绘图。")
        return

    # 3) 选每侧“最佳路径”（步数最多）
    selected = {}
    for ct in common_tags:
        selected[ct] = {
            'disp': bucket[ct]['disp'] or ct,
            'adam': _choose_best_path(bucket[ct]['adam']),
            'norm': _choose_best_path(bucket[ct]['norm']),
        }

    # 4) 读取曲线，准备全局范围
    curves = []  # [(canon_tag, 'adam'|'norm', losses, final)]
    y_min, y_max = float('inf'), -float('inf')
    x_max = 0

    for ct in common_tags:
        for method in ('adam', 'norm'):
            fp = selected[ct][method]
            if not fp:
                continue
            with open(fp, 'rb') as f:
                payload = pickle.load(f)
            recs = payload['steps']
            losses = extract_losses(recs)  # 兼容 surrogate_
            if len(losses):
                y_min = min(y_min, np.nanmin(losses))
                y_max = max(y_max, np.nanmax(losses))
                x_max = max(x_max, len(losses))
            curves.append((ct, method, losses, losses[-1] if len(losses) else np.nan))

    if not curves:
        print(f"[info] (alpha={alpha}, epsilon={epsilon}) 没有可用曲线，跳过。")
        return

    # 5) 颜色方案：高饱和的明显颜色（可按需扩展）
    base_palette = [
        "#e6194b",  # red
        "#3cb44b",  # green
        "#0082c8",  # blue
        "#f58231",  # orange
        "#911eb4",  # purple
        "#f032e6",  # magenta
        "#46f0f0",  # cyan
        "#000000",  # black
        "#ffe119",  # yellow
        "#008080",  # teal
        "#9a6324",  # brown
        "#fabebe",  # pink
    ]
    color_map = {}
    for i, ct in enumerate(common_tags):
        color_map[ct] = base_palette[i % len(base_palette)]

    # 6) 画图：每个 tag 两条线（adam: 实线+深色；norm: 虚线+同色变浅）
    plt.figure(figsize=(13, 6))
    for ct in common_tags:
        base_color = color_map[ct]
        # adam
        for (ct2, method, losses, final) in [c for c in curves if c[0] == ct and c[1] == 'adam']:
            label = f"{bucket[ct]['disp']} [adam] | final={final:.3g}"
            plt.plot(range(len(losses)), losses,
                     label=label, color=base_color, linewidth=2.4, alpha=0.95)
        # norm
        for (ct2, method, losses, final) in [c for c in curves if c[0] == ct and c[1] == 'norm']:
            label = f"{bucket[ct]['disp']} [norm] | final={final:.3g}"
            plt.plot(range(len(losses)), losses,
                     label=label, color=lighten_color(base_color, 0.55),
                     linewidth=2.4, alpha=0.95)

    # 轴与标题
    plt.xlabel("Step")
    plt.ylabel("Loss (MSE)")
    if np.isfinite(y_min) and np.isfinite(y_max):
        if y_min == y_max:
            y_min, y_max = (y_min - 1.0, y_max + 1.0) if y_max == 0 else (y_min * 0.9, y_max * 1.1)
        pad = 0.05 * max(1e-12, abs(y_max - y_min))
        plt.ylim(y_min - pad, y_max + pad)
    plt.xlim(-0.5, max(0, x_max - 1) + 0.5)
    plt.grid(True, alpha=0.3)
    plt.title(f"{title_prefix}\nalpha={alpha}, epsilon={epsilon}\n(adam or normal PGD on 2D NS Loss)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, frameon=True)

    plt.tight_layout()
    if out_dir is None:
        out_dir = BASE_DIR / "comparison_plots_common_adam_normal"
    else:
        out_dir = BASE_DIR / str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_alpha{alpha}_epsilon{epsilon}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存: {out_path}")


# --------- 5) 示例主程序 ----------
if __name__ == "__main__":
    from pathlib import Path

    try:
        BASE_DIR = Path(__file__).resolve().parent  # 代码文件所在目录
    except NameError:
        # 如果在 Jupyter 里没有 __file__，退化为当前工作目录
        BASE_DIR = Path.cwd()

    # 相对于“这个文件”的路径
    adam_folder = (BASE_DIR / "pickle_files").resolve()
    norm_folder = (BASE_DIR / "../all_attacks_results/pickle_files").resolve()  # 需要几层就写几个 ../

    print("adam_folder:", adam_folder, "| exists:", adam_folder.exists())
    print("norm_folder:", norm_folder, "| exists:", norm_folder.exists())


    # adam_folder = Path("/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/adam_attacks_results/pickle_files")                         # ← 你的 PGD+Adam 目录
    # norm_folder = Path("/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/all_attacks_results/pickle_files")  # ← 你的普通 PGD 目录

    common = build_common_groups(norm_folder, adam_folder, verbose=True)
    print(f"\n共有的 (alpha, epsilon) 组合数: {len(common)}")

    for key in common:
        visualize_common_alpha_epsilon(key, common, dpi=120, out_dir="comparison_plots_common",
                                       title_prefix="PGD (norm) vs PGD-Adam")
