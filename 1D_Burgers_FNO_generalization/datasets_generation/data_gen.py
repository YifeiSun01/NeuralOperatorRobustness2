# -*- coding: utf-8 -*-
# 读取 all_pos 下的每个数据集，把 input 线性缩放到多个目标区间，调用 Burgers 求解器得到新的 output，
# 并按“与原数据集一致的格式”保存到新的顶层目录名（用区间命名）下。
# 兼容两种现有格式：
#   1) .pkl : 列表[{"a": np.ndarray, "g(a)": np.ndarray}, ...]
#   2) .pt  : {'x': Tensor (N,nx), 'y': Tensor (N,nx), 't_final': float}
#
# 用法：
#   run_rescale_and_resolve(base_data_dir=Path(__file__).resolve().parent.parent / "datasets",
#                           solver_name="exponax")
from pathlib import Path
import os
import re
import pickle
import numpy as np
import torch
from tqdm import tqdm

# 你项目里已有：from solvers.burgers1d_solvers import ExponaxBurgersSolver1D
from solvers.burgers1d_solvers import ExponaxBurgersSolver1D


def _linear_rescale(arr: np.ndarray, new_min: float, new_max: float, eps: float = 1e-12) -> np.ndarray:
    """逐样本(min,max)线性缩放到[new_min,new_max]；若常量样本，则映射为区间中点。"""
    a_min = float(np.min(arr))
    a_max = float(np.max(arr))
    if abs(a_max - a_min) < eps:
        mid = (new_min + new_max) * 0.5
        return np.full_like(arr, mid, dtype=np.float32)
    scaled = (arr - a_min) / (a_max - a_min)
    return (scaled * (new_max - new_min) + new_min).astype(np.float32)


def _parse_params_from_name(name: str, verbose: bool = True):
    """
    从文件名解析 Burgers 粘性 nu(=...), t_final, step, bc。
    规则：
      1) 优先在最后一个 'bc=' 之后的片段里找 'nu='（一般 PDE 粘性写在这里）
      2) 若找不到，再在整串中取最后一个 'nu='
      3) 仍找不到则用默认 5e-4
    同时打印解析细节，便于核查。
    """
    import re

    def grab_last_float(pat: str, s: str, default: float) -> float:
        vals = [float(m.group(1)) for m in re.finditer(pat, s)]
        return (vals[-1] if vals else default)

    # 先解析 bc
    m = re.search(r"bc=([A-Za-z]+)", name)
    bc = m.group(1) if m else "periodic"

    # 取“从最后一个 bc= 开始的片段”
    last_bc_pos = name.rfind("bc=")
    segment = name[last_bc_pos:] if last_bc_pos != -1 else name

    # 只认带等号的 nu=...；先在 segment 里搜
    nu_pat = r"(?<![A-Za-z0-9])nu=([0-9eE\.\-]+)"
    nu_candidates_seg = [float(m.group(1)) for m in re.finditer(nu_pat, segment)]
    if nu_candidates_seg:
        nu = nu_candidates_seg[-1]
        nu_source = "after bc= segment"
        nu_all = [float(m.group(1)) for m in re.finditer(nu_pat, name)]
    else:
        # 整串取最后一个
        nu_all = [float(m.group(1)) for m in re.finditer(nu_pat, name)]
        if nu_all:
            nu = nu_all[-1]
            nu_source = "last-in-name fallback"
        else:
            nu = 5e-4
            nu_source = "default(5e-4)"

    # t 与 step 取“最后一次出现”
    t_final = grab_last_float(r"(?:^|[_-])t(?:=)?([0-9eE\.\-]+)", name, 1.0)
    step    = grab_last_float(r"(?:^|[_-])step(?:=)?([0-9eE\.\-]+)", name, 1e-4)

    if verbose:
        print("[PARSE]")
        print(f"  name     : {name}")
        print(f"  bc       : {bc}")
        print(f"  nu(all)  : {nu_all}")
        print(f"  nu(chosen): {nu}   (source: {nu_source})")
        print(f"  t_final  : {t_final}")
        print(f"  step     : {step}")

    return dict(nu=nu, t_final=t_final, step=step, bc=bc)


def _solve_final(u0_np: np.ndarray, nu: float, t_final: float, step: float, bc: str) -> np.ndarray:
    """给定 1D 初值，调用 JAX/Exponax 求解至 t_final，返回最终解（np.ndarray）。"""
    nx = int(u0_np.shape[0])
    solver = ExponaxBurgersSolver1D(nx=nx, nu=nu, bc=bc)
    # 与你生成脚本里的取样方式一致（每 10*step 取一次）
    num_steps = max(int((t_final / step) / 10) + 1, 2)
    t_eval = np.linspace(0.0, t_final, num_steps)
    sol_dict = solver.solve(u0_np.copy(), t_final=t_final, t_eval=t_eval, step=step)
    final_solution = sol_dict[t_eval[-1]][1]  # dict[t] -> (closest_key, array)
    return np.array(final_solution, dtype=np.float32)


def _save_like_existing_format(out_path: Path, fmt: str, samples: list, t_final_for_pt: float = 1.0):
    """
    fmt == 'pkl': samples 是 [{'a': np.ndarray, 'g(a)': np.ndarray}, ...]
    fmt == 'pt' : samples 是 list of (a_np, y_np) -> 保存成 {'x':Tensor,'y':Tensor,'t_final':float}
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pkl":
        with open(out_path, "wb") as f:
            pickle.dump(samples, f)
    elif fmt == "pt":
        x = torch.from_numpy(np.stack([s[0] for s in samples], axis=0))  # (N,nx)
        y = torch.from_numpy(np.stack([s[1] for s in samples], axis=0))  # (N,nx)
        torch.save({"x": x, "y": y, "t_final": t_final_for_pt}, out_path)
    else:
        raise ValueError(f"Unknown format to save: {fmt}")
    print(f"[SAVED] {out_path}")


def _detect_input_format(file_path: Path) -> str:
    return "pkl" if file_path.suffix.lower() == ".pkl" else "pt"


def _iter_pos_datasets(base_data_dir: Path):
    """
    仅遍历 base_data_dir/all_pos，递归收集 *.pkl / *.pt，去重并排序返回。
    """
    root = base_data_dir / "all_pos"
    if not root.exists():
        raise FileNotFoundError(f"未找到 'all_pos' 目录于：{base_data_dir}")

    seen = set()
    for pattern in ("*.pkl", "*.pt"):
        for p in root.rglob(pattern):
            seen.add(p.resolve())

    for p in sorted(seen):
        yield p


def _make_range_dir_name(lo: float, hi: float) -> str:
    """区间目录名直接保留小数点，用紧凑浮点格式避免长尾。"""
    def fmt(v: float) -> str:
        return f"{v:.15g}"
    return f"range_{fmt(lo)}_{fmt(hi)}"


def _print_all_files_abs(base_data_dir: Path) -> None:
    base = base_data_dir.resolve()
    print(f"[LISTING] All existing files under: {base}")
    count = 0
    for p in sorted(base.rglob("*")):
        if p.is_file():
            print(str(p.resolve()))
            count += 1
    print(f"[LISTING] Total files found: {count}")


def run_rescale_and_resolve(base_data_dir: Path,
                            solver_name: str = "exponax",
                            target_ranges=(#(0.0, 2.0),
                                           #(0.0, 0.5),
                                           (0.0, 3.0),
                                           #(-2.0, 2.0),
                                           (-3.0, 0.0),
                                           #(-2.0, 0.0),
                                           (-3.0, 3.0))):
    """
    主流程：
    - 遍历 base_data_dir/all_pos/** 下的每个数据集（pkl 或 pt）
    - 对每个样本的 input 执行 min-max 缩放到多个 target_ranges
    - 用 Burgers 求解器得到新的 output
    - 保存到 base_data_dir/<range_dir_name>/...，格式与原文件保持一致
    """
    assert solver_name.lower() == "exponax", "当前实现使用 ExponaxBurgersSolver1D"

    _print_all_files_abs(base_data_dir)

    # 遍历 all_pos 数据集
    ds_paths = list(_iter_pos_datasets(base_data_dir))
    print(f"[FOUND] 共发现数据集文件 {len(ds_paths)} 个")

    # 小统计：看各家族是否被枚举到（例如 gaussian_diff_distr / matern_diff_distr / zigzag）
    from collections import Counter
    def family(p: Path):
        parts = p.parts
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "all_pos":
                return (parts[i], parts[i + 1] if i + 1 < len(parts) else "")
        return ("", "")
    print("[BREAKDOWN]", Counter(family(Path(p)) for p in ds_paths))

    valid_bc = {"periodic", "nonperiodic"}

    for ds_idx, ds_path in enumerate(ds_paths, start=1):
        fmt = _detect_input_format(ds_path)
        rel = ds_path.relative_to(base_data_dir)   # all_pos/.../<dataset>.<ext>
        subdir = rel.parents[0]                    # e.g. all_pos/gaussian_diff_distr
        dataset_name = ds_path.stem                # 不含扩展名

        # 解析参数（✅ 已修：nu 仅认 nu=...；bc 仅认 bc=...）
        params = _parse_params_from_name(dataset_name)
        if params["bc"] not in valid_bc:
            print(f"[WARN] 未知 bc={params['bc']}，回退为 periodic")
            params["bc"] = "periodic"

        # 读取原数据集
        if fmt == "pkl":
            with open(ds_path, "rb") as f:
                data_list = pickle.load(f)  # list of dicts
            N = len(data_list)
            nx = int(np.array(data_list[0]["a"]).reshape(-1).shape[0])
            inputs_np = np.stack([np.array(rec["a"]).reshape(-1) for rec in data_list], axis=0).astype(np.float32)
        else:  # pt
            data_obj = torch.load(ds_path, map_location="cpu")
            x = data_obj["x"]  # (N,nx)
            N, nx = int(x.shape[0]), int(x.shape[1])
            inputs_np = x.numpy().astype(np.float32)
            # 如果 pt 里自带 t_final，优先用它
            if "t_final" in data_obj:
                params["t_final"] = float(data_obj["t_final"])

        print(
            f"[DATASET {ds_idx}/{len(ds_paths)}] {ds_path}\n"
            f"  -> N={N}, nx={nx}, nu={params['nu']}, t_final={params['t_final']}, step={params['step']}, bc={params['bc']}\n"
            f"  -> 将生成 {len(target_ranges)} 个区间版本（顶层目录用区间命名）"
        )

        # 针对每个目标区间分别生成一个新数据集
        for ridx, (lo, hi) in enumerate(target_ranges, start=1):
            range_dir = _make_range_dir_name(lo, hi)
            # 用新的顶级目录替换 'all_pos'： range_xxx_y/gaussian_diff_distr/...
            out_dir = base_data_dir / range_dir / subdir.relative_to(subdir.parts[0])
            out_path = out_dir / f"{dataset_name}.{'pkl' if fmt=='pkl' else 'pt'}"

            # 先检查目标文件是否已存在
            if out_path.exists():
                print(
                    f"  [CHECK] 目标区间: [{lo}, {hi}] -> 已存在: {out_path}\n"
                    f"  [SKIP] 跳过生成。"
                )
                continue  # 跳过该区间

            # 尚不存在 -> 开始生成
            print(
                f"  [CHECK] 目标区间: [{lo}, {hi}] -> 不存在: {out_path}\n"
                f"  [START] 开始生成..."
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            new_samples = []
            bar_desc = f"{dataset_name} [{lo},{hi}]"
            bar = tqdm(range(N), desc=bar_desc, dynamic_ncols=True, leave=True)
            for i in bar:
                a0 = inputs_np[i]  # (nx,)
                a_scaled = _linear_rescale(a0, lo, hi)  # (nx,)

                y_final = _solve_final(
                    a_scaled,
                    nu=params["nu"],
                    t_final=params["t_final"],
                    step=params["step"],
                    bc=params["bc"]
                )

                if fmt == "pkl":
                    new_samples.append({"a": a_scaled, "g(a)": y_final})
                else:  # pt
                    new_samples.append((a_scaled, y_final))

                # tqdm 实时显示关键参数/进度信息
                bar.set_postfix(
                    sample=i + 1,
                    N=N,
                    nx=nx,
                    nu=params["nu"],
                    t=params["t_final"],
                    step=params["step"],
                    bc=params["bc"]
                )

            _save_like_existing_format(
                out_path,
                fmt=fmt,
                samples=new_samples,
                t_final_for_pt=params["t_final"]
            )

        print(f"[DONE] 数据集完成：{ds_path} -> 生成/跳过 {len(target_ranges)} 个区间版本。\n")

    print("[DONE] All rescaled datasets generated.")


if __name__ == '__main__':
    from pathlib import Path

    # 指向包含 all_pos/ 的目录
    HERE = Path(__file__).resolve().parent
    base = (HERE / "../datasets").resolve()

    # _print_all_files_abs(base_data_dir=base)

    run_rescale_and_resolve(base_data_dir=base)
