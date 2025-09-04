from pathlib import Path
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from GRFs.generateGRFs import GRFGenerator
from solvers.burgers1d_solvers import *
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的是 {'GPU (CUDA)' if device.type == 'cuda' else 'CPU'} 进行计算")
print(f"设备名称: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")


def _build_solver(solver_name: str, nx: int, nu: float, bc: str):
    solver_name = solver_name.lower()
    if solver_name == "scipy":
        return SciPyBurgersSolver1D(nx, nu=nu, bc=bc)
    elif solver_name == "scipy_spectral":
        return SciPySpectralBurgersSolver1D(nx, nu=nu, bc=bc)
    elif solver_name == "exponax":
        return ExponaxBurgersSolver1D(nx, nu=nu, bc=bc, xlim=(0, 1))
    elif solver_name == "phiflow":
        return PhiFlowBurgersSolver1D(nx, nu=nu, bc=bc)
    else:
        raise ValueError(f"Specified solver not implemented: {solver_name}")


def _is_bad(arr) -> bool:
    """检测是否包含 NaN/Inf。"""
    if isinstance(arr, np.ndarray):
        return ~np.isfinite(arr).all()
    # 兜底：转成 numpy 再查
    try:
        arr = np.asarray(arr)
        return ~np.isfinite(arr).all()
    except Exception:
        # 如果类型怪异，也视作 bad
        return True


def _solve_with_adaptive_step(solver, init_field, t_final: float, init_step: float,
                              refine_factor: float = 0.5, min_step: float = None,
                              max_refinements: int = 6):
    """
    自适应步长求解：
      1) 用 init_step 尝试
      2) 若结果含 NaN/Inf，则 step *= refine_factor 继续
      3) 直到无 NaN/Inf，或触达 min_step / 最大细化次数
    返回: (final_solution, used_step, t_eval, solution_obj)
    若失败，抛出 RuntimeError
    """
    if min_step is None:
        # 默认最小步长：init_step / 64
        min_step = max(init_step / 64.0, 1e-6)

    step = float(init_step)
    last_err = None
    for attempt in range(max_refinements + 1):
        # 构造时间网格：确保包含 t=0 和 t=t_final
        # arange 更直观，避免浮点误差漏掉终点，这里 +1e-12 冗余以包含 t_final
        t_eval = np.arange(0.0, t_final + 1e-12, step)
        if t_eval[-1] < t_final:  # 保险：若因为浮点误差没到终点，补上
            t_eval = np.append(t_eval, t_final)

        try:
            solution = solver.solve(init_field.copy(), t_final, t_eval, step)
            # 维持你原来的取法：solution[t_eval[-1]][1]
            final_solution = solution[t_eval[-1]][1]

            if _is_bad(final_solution):
                print(f"[Adaptive] step={step:g} 结果含 NaN/Inf，细化步长…")
                last_err = RuntimeError("结果含 NaN/Inf")
            else:
                print(f"[Adaptive] 成功：使用 step={step:g}（尝试次数 {attempt+1}）")
                return final_solution, step, t_eval, solution

        except Exception as e:
            print(f"[Adaptive] step={step:g} 求解异常：{e}；细化步长…")
            last_err = e

        # 细化条件判断
        new_step = step * refine_factor
        if new_step < min_step - 1e-15:
            break
        step = new_step

    raise RuntimeError(f"自适应步长求解失败。最后错误：{last_err}")


def generate_burgers_dataset(params, save_dir="datasets/1D/Burgers", solver_name="exponax"):
    grf_params = params["GRF"]
    dataset_params = params["dataset"]
    burgers_params = params["burgers"]
    np.random.seed(grf_params['seed'])

    num_record = dataset_params['num_record']
    nu = burgers_params['nu']
    t_final = burgers_params["simulation_time"]

    # ✓ 新增可选自适应参数
    init_step = float(burgers_params.get("step", 0.01))
    refine_factor = float(burgers_params.get("refine_factor", 0.5))
    max_refinements = int(burgers_params.get("max_refinements", 6))
    min_step = burgers_params.get("min_step", None)
    if min_step is not None:
        min_step = float(min_step)

    nx = grf_params['nx']
    dim = grf_params['dim']

    if dim == 1:
        shape = (grf_params['nx'],)
        folder = '1D'
    else:
        raise ValueError("dim must be 1")

    solver = _build_solver(solver_name, nx, nu=nu, bc=grf_params['bc'])

    input_tensors = []
    output_tensors = []
    used_steps = []  # 记录每个样本最终使用的步长，方便追踪

    for i in tqdm(range(num_record), desc="Generating samples"):
        try:
            grf = GRFGenerator.generate_grf(
                shape=shape,
                kernel=grf_params['kernel'],
                kernel_params=grf_params['kernel_params'],
                bc=grf_params['bc'],
                seed=grf_params['seed'] + i,  # 每个样本不同随机种子
                zero_mean=grf_params['zero_mean']
            )

            final_solution, used_step, t_eval, _ = _solve_with_adaptive_step(
                solver, grf, t_final, init_step,
                refine_factor=refine_factor,
                min_step=min_step,
                max_refinements=max_refinements
            )

            print(f"index: {i}    initial condition (GRF): shape={np.shape(grf)}")
            print(f"index: {i}    final_solution: shape={np.shape(final_solution)}    used_step={used_step:g}")

            # 转 tensor（在 CPU 上累计，避免 GPU 挤压）
            input_tensor = torch.tensor(grf.copy(), dtype=torch.float32)  # 不上 GPU
            if isinstance(final_solution, np.ndarray):
                output_tensor = torch.tensor(final_solution, dtype=torch.float32)
            else:
                output_tensor = torch.from_numpy(np.array(final_solution)).float()

            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            used_steps.append(used_step)

        except Exception as e:
            print(f"Error processing sample {i}: {e} —— 已跳过该样本")
            continue

    if not input_tensors:
        raise RuntimeError("没有成功的样本，停止保存。")

    input_tensor = torch.stack(input_tensors).cpu()
    output_tensor = torch.stack(output_tensors).cpu()
    print({"input tensor shape": input_tensor.shape, "output tensor shape": output_tensor.shape})

    # —— 文件命名中加入最终步长信息（若每个样本步长不同，则标记为“adaptive”）——
    if len(set(np.round(used_steps, 12))) == 1:
        step_tag = f"step{used_steps[0]:g}"
    else:
        step_tag = "step_adaptive"

    param_names = [
        f"dim{dim}d",
        f"nx{nx}",
        f"N{len(input_tensors)}",  # 实际成功样本数
        f"solver={solver_name}",
        f"kernel={grf_params['kernel']}",
        *[f"{k}{v}" for k, v in grf_params['kernel_params'].items()],
        f"bc{grf_params['bc']}",
        f"nu{nu}",
        f"t{t_final}",
        f"seed{grf_params['seed']}",
    ]

    current_file_path = Path(__file__).resolve().parent
    save_path = current_file_path / folder / save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    filename = "_".join(param_names) + ".pt"
    input_filename = f"inputs_{filename}"
    output_filename = f"outputs_{filename}"
    steps_filename = f"used_steps_{filename.replace('.pt','.npy')}"

    torch.save(input_tensor, save_path / input_filename)
    torch.save(output_tensor, save_path / output_filename)
    # np.save(save_path / steps_filename, np.array(used_steps, dtype=np.float64))
    print(f"Saved input tensor to {save_path / input_filename}")
    print(f"Saved output tensor to {save_path / output_filename}")
    # print(f"Saved per-sample used steps to {save_path / steps_filename}")


# ======== 示例参数 ========
grf_params = {
    'dim': 1,
    'nx': 1024,
    'kernel': 'gaussian',
    'kernel_params': {'correlation_length': 0.03},
    'bc': 'periodic',
    'seed': 1000045,
    'zero_mean': False
}

burger_params = {
    'nu': 0.0005,
    'simulation_time': 1.0,
    "step": 0.002,           # 初始步长（可较大，失败再细化）
    "min_step": 0.001,      # 最小步长（可选）
    "refine_factor": 0.5,   # 细化倍率：每次 step *= 0.5
    "max_refinements": 6    # 最多细化 6 次
}

dataset_params = {
    "num_record": 200000
}

params = {
    "GRF": grf_params,
    "dataset": dataset_params,
    "burgers": burger_params,
}

for solver_name in ["exponax"]:
    generate_burgers_dataset(params, save_dir="Burgers", solver_name=solver_name)
