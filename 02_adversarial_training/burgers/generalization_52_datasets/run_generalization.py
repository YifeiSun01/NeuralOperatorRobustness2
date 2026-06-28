import torch
from FNO1d import FNO1d
import utilities3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import csv
import pickle
import subprocess
import numpy as np


def count_special(arr: np.ndarray):
    """统计 NaN / +Inf / -Inf 数量与占比"""
    arr = np.asarray(arr)
    total = arr.size
    nan = int(np.isnan(arr).sum())
    posinf = int(np.isposinf(arr).sum())
    neginf = int(np.isneginf(arr).sum())
    invalid = nan + posinf + neginf
    invalid_pct = (invalid / total * 100.0) if total else 0.0
    return {
        "total": int(total),
        "nan": nan,
        "posinf": posinf,
        "neginf": neginf,
        "invalid": invalid,
        "invalid_pct": invalid_pct,
    }


def safe_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8):
    """
    仅在 y_true 与 y_pred 同时为有限值的位置上计算 RMSE/MAE/MAPE。
    返回：metrics + 有效/无效统计，用于 CSV。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)

    total = y_true.size
    valid_n = int(finite_mask.sum())
    invalid_n = int(total - valid_n)
    invalid_pct = (invalid_n / total * 100.0) if total else 0.0

    if valid_n == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "valid_n": valid_n,
            "invalid_n": invalid_n,
            "invalid_pct": invalid_pct,
            "finite_mask": finite_mask,  # 供绘图用
        }

    yt = y_true[finite_mask]
    yp = y_pred[finite_mask]

    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae  = float(np.mean(np.abs(yt - yp)))
    denom = np.maximum(np.abs(yt), epsilon)
    mape = float(np.mean(np.abs((yt - yp) / denom)) * 100.0)

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "valid_n": valid_n,
        "invalid_n": invalid_n,
        "invalid_pct": invalid_pct,
        "finite_mask": finite_mask,
    }


def to_nan_where_invalid(arr: np.ndarray):
    """把非有限值变成 np.nan，便于折线自动断开"""
    arr = np.asarray(arr)
    out = arr.copy()
    out[~np.isfinite(out)] = np.nan
    return out


def get_gpu_info():
    try:
        # 获取当前分区（Slurm 环境变量）
        partition = os.environ.get("SLURM_JOB_PARTITION", "N/A")

        # 获取当前节点名
        node_name = os.environ.get("SLURMD_NODENAME", "N/A")

        # 获取当前使用的 GPU ID（如果用 --gres）
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")

        # 调用 nvidia-smi 获取详细信息
        try:
            nvidia_smi_output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free",
                    "--format=csv,noheader",
                ],
                encoding="utf-8",
            ).strip()
        except FileNotFoundError:
            nvidia_smi_output = "nvidia-smi 未找到，可能此节点没有 NVIDIA GPU"

        print("========== 当前作业 GPU 信息 ==========")
        print(f"分区: {partition}")
        print(f"节点: {node_name}")
        print(f"GPU ID (CUDA_VISIBLE_DEVICES): {gpu_ids}")
        print("\nGPU 详细信息:")
        print(nvidia_smi_output)
        print("=====================================")

    except Exception as e:
        print(f"获取 GPU 信息时出错: {e}")


def list_pickle_files(folder_path):
    folder = Path(folder_path)
    pickle_files = list(folder.rglob("*.pkl"))
    return [str(f.resolve()) for f in pickle_files]


def load_checkpoint(path, device):
    torch.serialization.add_safe_globals([utilities3.UnitGaussianNormalizer])
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = FNO1d(modes=checkpoint["modes"], width=checkpoint["width"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # 将 normalizer 的参数转到 device
        x_norm = checkpoint["x_normalizer"]
        y_norm = checkpoint["y_normalizer"]
        x_norm.mean = x_norm.mean.to(device)
        x_norm.std = x_norm.std.to(device)
        y_norm.mean = y_norm.mean.to(device)
        y_norm.std = y_norm.std.to(device)

        return model, x_norm, y_norm
    else:
        model = FNO1d(modes=16, width=64).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model


def infer_two_models(input_tensor, model_norm, x_norm, y_norm, model_unnorm, device):
    input_tensor = input_tensor.to(device).to(torch.float32)
    # 确保归一化器的mean和std和input_tensor同设备
    x_norm.mean = x_norm.mean.to(device)
    x_norm.std = x_norm.std.to(device)
    y_norm.mean = y_norm.mean.to(device)
    y_norm.std = y_norm.std.to(device)

    x_norm_encoded = x_norm.encode(input_tensor).to(torch.float32).to(device)
    x_unnorm = input_tensor.to(torch.float32).to(device)

    with torch.no_grad():
        out_norm = model_norm(x_norm_encoded)
        out_norm = y_norm.decode(out_norm)[..., 0]
        out_unnorm = model_unnorm(x_unnorm)

    return out_norm.cpu(), out_unnorm.cpu()


def squeeze_to_1d(t):
    t = t.squeeze()
    if t.ndim > 1:
        t = t.view(-1)
    return t


def compute_mape(y_true, y_pred, epsilon=1e-8):
    # 避免除零，epsilon防止分母0
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


from pathlib import Path
import re


def parse_info_from_path(save_path):
    p = Path(save_path)

    # 取倒数第4级目录: all_pos / all_neg / pos_neg
    try:
        dataset_type = p.parts[-4]
    except IndexError:
        dataset_type = "unknown_type"

    # 取倒数第3级目录：子文件夹名
    try:
        dataset_subfolder = p.parts[-3]
    except IndexError:
        dataset_subfolder = "unknown_subfolder"

    # 取倒数第2级目录，dataset_name
    try:
        dataset_name = p.parts[-2]
    except IndexError:
        dataset_name = "unknown_name"

    # 简单提取pattern
    if "matern" in dataset_name.lower():
        pattern = "matern"
    elif "gaussian" in dataset_name.lower():
        pattern = "gaussian"
    elif "zigzag" in dataset_name.lower():
        pattern = "zigzag"
    else:
        pattern = "other"

    # 根据pattern提取参数
    cl_str = nu_str = peaks_str = None
    if pattern == "matern":
        cl = re.search(r"cl([0-9\.]+)", dataset_name)
        nu = re.search(r"nu([0-9\.]+)", dataset_name)
        cl_str = cl.group(1) if cl else "?"
        nu_str = nu.group(1) if nu else "?"
    elif pattern == "gaussian":
        cl = re.search(r"cl([0-9\.]+)", dataset_name)
        cl_str = cl.group(1) if cl else "?"
    elif pattern == "zigzag":
        peaks = re.search(r"peaks([0-9]+)", dataset_name)
        peaks_str = peaks.group(1) if peaks else "?"

    # 构造简洁信息字符串
    param_str = ""
    if pattern == "matern":
        param_str = f"cl={cl_str}, nu={nu_str}"
    elif pattern == "gaussian":
        param_str = f"cl={cl_str}"
    elif pattern == "zigzag":
        param_str = f"peaks={peaks_str}"

    range_map = {
        "all_pos": "range=(0,1)",
        "all_neg": "range=(-1,0)",
        "pos_neg": "range=(-0.5,0.5)",
        "range_-2_0": "range=(-2,0)",
        "range_-2_2": "range=(-2,2)",
        "range_-3_0": "range=(-3,0)",
        "range_-3_3": "range=(-3,3)",
        "range_0_0.5": "range=(0,0.5)",
        "range_0_2": "range=(0,2)",
        "range_0_3": "range=(0,3)",
    }
    readable_type = range_map.get(dataset_type, dataset_type)

    return f"Dataset: {readable_type} | Pattern: {pattern} | Params: {param_str}"


def plot_inputs_and_outputs(
    x,
    pred_norm,
    pred_unnorm,
    y_true,
    rmse1,
    rmse2,
    mae1,
    mae2,
    mape1,
    mape2,
    save_path=None,
):
    import matplotlib.pyplot as plt

    x = squeeze_to_1d(x)
    pred_norm = squeeze_to_1d(pred_norm)
    pred_unnorm = squeeze_to_1d(pred_unnorm)
    y_true = squeeze_to_1d(y_true)

    # 为了断线显示，把非有限值统一置为 NaN
    x_plot = to_nan_where_invalid(x)
    y_plot = to_nan_where_invalid(y_true)
    pred_norm_plot = to_nan_where_invalid(pred_norm)
    pred_unnorm_plot = to_nan_where_invalid(pred_unnorm)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    axs[0].plot(x_plot, label="Input a")
    axs[0].set_title("Input a")
    axs[0].legend()

    axs[1].plot(y_plot, label="Ground Truth g(a)", linestyle="--", color="black")
    axs[1].plot(
        pred_norm_plot,
        label=f"Norm Model (RMSE={rmse1:.4f}, MAE={mae1:.4f}, MAPE={mape1:.2f}%)",
        alpha=0.8,
    )
    axs[1].plot(
        pred_unnorm_plot,
        label=f"Unnorm Model (RMSE={rmse2:.4f}, MAE={mae2:.4f}, MAPE={mape2:.2f}%)",
        alpha=0.8,
    )
    axs[1].set_title("Model Predictions vs Ground Truth")
    axs[1].legend()

    if save_path is not None:
        info_title = parse_info_from_path(save_path)
        fig.suptitle(info_title, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved figure: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    get_gpu_info()

    base_path = Path(__file__).parent

    # 模型路径
    normalized_path = base_path / (
        "normalized/burgers_1d_FNO_model_trainedby_dim1d_nx1024_N1500_"
        "solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_"
        "nu0.0005_t1.0_seed45.pth"
    )
    unnormalized_path = base_path / (
        "unnormalized/burgers_1d_FNO_model_trainedby_dim1d_nx1024_N1500_"
        "solver=exponax_kernel=gaussian_correlation_length0.03_bcperiodic_"
        "nu0.0005_t1.0_seed45.pth"
    )

    model_norm, x_norm, y_norm = load_checkpoint(normalized_path, device)
    model_unnorm = load_checkpoint(unnormalized_path, device)

    data_folder = base_path.parent / "datasets"
    all_pkl_files = list_pickle_files(data_folder)

    save_base_dir = base_path / "generalization_plots"
    os.makedirs(save_base_dir, exist_ok=True)

    # 记录所有数据的指标，最终写csv
    csv_records = []
    csv_fieldnames = [
        "dataset_type",
        "dataset_name",
        "sample_index",
        "rmse_norm",
        "mae_norm",
        "mape_norm",
        "rmse_unnorm",
        "mae_unnorm",
        "mape_unnorm",
        # y_true 自身的 NaN/Inf 统计
        "y_true_total",
        "y_true_nan",
        "y_true_posinf",
        "y_true_neginf",
        "y_true_invalid",
        "y_true_invalid_pct",
        # norm 预测自身的 NaN/Inf 统计
        "pred_norm_total",
        "pred_norm_nan",
        "pred_norm_posinf",
        "pred_norm_neginf",
        "pred_norm_invalid",
        "pred_norm_invalid_pct",
        # unnorm 预测自身的 NaN/Inf 统计
        "pred_unn_total",
        "pred_unn_nan",
        "pred_unn_posinf",
        "pred_unn_neginf",
        "pred_unn_invalid",
        "pred_unn_invalid_pct",
        # 参与指标计算的有效点统计（与 y_true 的交集）
        "valid_used_norm",
        "invalid_used_norm",
        "used_invalid_pct_norm",
        "valid_used_unnorm",
        "invalid_used_unnorm",
        "used_invalid_pct_unnorm",
    ]

    for pkl_file in all_pkl_files:
        print(pkl_file)
        # 解析路径，提取类别和名称
        pkl_path = Path(pkl_file)
        relative_path = pkl_path.relative_to(data_folder)
        dataset_type = relative_path.parts[0]  # e.g. all_pos
        dataset_subfolder = relative_path.parent.name  # e.g. gaussian_diff_distr
        dataset_name = pkl_path.stem  # 文件名，无后缀

        save_dir = save_base_dir / dataset_type / dataset_subfolder / dataset_name
        os.makedirs(save_dir, exist_ok=True)

        with open(pkl_file, "rb") as f:
            data_list = pickle.load(f)

        for idx, record in enumerate(data_list):
            print("         ", idx)
            a_np = np.array(record["a"], copy=True)
            g_a_np = np.array(record["g(a)"], copy=True)

            # 转成tensor，并且加batch维度，模型输入是 (batch, nx)
            a_tensor = torch.tensor(a_np.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            g_a_tensor = torch.tensor(g_a_np.copy(), dtype=torch.float32).unsqueeze(0)

            # 推理
            pred_norm, pred_unnorm = infer_two_models(
                a_tensor, model_norm, x_norm, y_norm, model_unnorm, device
            )

            # 计算指标
            y_true_np = g_a_tensor.detach().cpu().numpy().reshape(-1)
            pred_norm_np = pred_norm.detach().cpu().numpy().reshape(-1)
            pred_unnorm_np = pred_unnorm.detach().cpu().numpy().reshape(-1)

            stats_true = count_special(y_true_np)
            stats_norm = count_special(pred_norm_np)
            stats_unn = count_special(pred_unnorm_np)

            m_norm = safe_regression_metrics(y_true_np, pred_norm_np, epsilon=1e-8)
            m_unn = safe_regression_metrics(y_true_np, pred_unnorm_np, epsilon=1e-8)

            rmse_norm, mae_norm, mape_norm = (
                m_norm["rmse"],
                m_norm["mae"],
                m_norm["mape"],
            )
            rmse_unnorm, mae_unnorm, mape_unnorm = (
                m_unn["rmse"],
                m_unn["mae"],
                m_unn["mape"],
            )

            # 记录结果（CSV 逻辑保持不变）
            csv_records.append(
                {
                    "dataset_type": dataset_type,
                    "dataset_name": dataset_name,
                    "sample_index": idx,
                    "rmse_norm": rmse_norm,
                    "mae_norm": mae_norm,
                    "mape_norm": mape_norm,
                    "rmse_unnorm": rmse_unnorm,
                    "mae_unnorm": mae_unnorm,
                    "mape_unnorm": mape_unnorm,
                    "y_true_total": stats_true["total"],
                    "y_true_nan": stats_true["nan"],
                    "y_true_posinf": stats_true["posinf"],
                    "y_true_neginf": stats_true["neginf"],
                    "y_true_invalid": stats_true["invalid"],
                    "y_true_invalid_pct": stats_true["invalid_pct"],
                    "pred_norm_total": stats_norm["total"],
                    "pred_norm_nan": stats_norm["nan"],
                    "pred_norm_posinf": stats_norm["posinf"],
                    "pred_norm_neginf": stats_norm["neginf"],
                    "pred_norm_invalid": stats_norm["invalid"],
                    "pred_norm_invalid_pct": stats_norm["invalid_pct"],
                    "pred_unn_total": stats_unn["total"],
                    "pred_unn_nan": stats_unn["nan"],
                    "pred_unn_posinf": stats_unn["posinf"],
                    "pred_unn_neginf": stats_unn["neginf"],
                    "pred_unn_invalid": stats_unn["invalid"],
                    "pred_unn_invalid_pct": stats_unn["invalid_pct"],
                    "valid_used_norm": m_norm["valid_n"],
                    "invalid_used_norm": m_norm["invalid_n"],
                    "used_invalid_pct_norm": m_norm["invalid_pct"],
                    "valid_used_unnorm": m_unn["valid_n"],
                    "invalid_used_unnorm": m_unn["invalid_n"],
                    "used_invalid_pct_unnorm": m_unn["invalid_pct"],
                }
            )

            # 保存图：先根据 rmse 生成文件名，然后判断是否已存在，存在则跳过生成
            img_name = (
                f"sample_{idx}_rmseNorm{rmse_norm:.4f}_rmseUnnorm{rmse_unnorm:.4f}.png"
            )
            save_path = save_dir / img_name

            if save_path.exists():
                print(f"Skip existing image: {save_path}")
            else:
                plot_inputs_and_outputs(
                    a_tensor,
                    pred_norm,
                    pred_unnorm,
                    g_a_tensor,
                    rmse_norm,
                    rmse_unnorm,
                    mae_norm,
                    mae_unnorm,
                    mape_norm,
                    mape_unnorm,
                    save_path=str(save_path),
                )

    # 最后写csv文件（保持不变）
    csv_path = save_base_dir / "metrics_summary.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(csv_records)

    print(f"All metrics saved to {csv_path}")