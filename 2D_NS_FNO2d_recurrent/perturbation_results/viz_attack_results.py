import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from PIL import Image
import pickle
import numpy as np

def compute_gradient(u, h=1.0):
    """计算二维数组的梯度（带周期性边界条件）"""
    grad_x = np.zeros_like(u)
    grad_y = np.zeros_like(u)
    
    # x方向梯度（周期性边界）
    grad_x[:, :] = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * h)
    
    # y方向梯度（周期性边界）
    grad_y[:, :] = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * h)
    
    return grad_x, grad_y

def compute_hessian(u, h=1.0):
    """计算二维数组的Hessian矩阵（带周期性边界条件）"""
    # 二阶偏导数（使用周期性边界）
    u_xx = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / (h**2)
    u_yy = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / (h**2)
    u_xy = (np.roll(np.roll(u, -1, axis=0), -1, axis=1) - 
           np.roll(np.roll(u, -1, axis=0), 1, axis=1) - 
           np.roll(np.roll(u, 1, axis=0), -1, axis=1) + 
           np.roll(np.roll(u, 1, axis=0), 1, axis=1)) / (4 * h**2)
    
    return u_xx, u_yy, u_xy

def periodic_boundary_error(u):
    """重组后的周期性误差计算（标准化误差单独分组）"""
    assert u.ndim == 2 and u.shape[0] == u.shape[1], "输入必须是N×N的二维数组"
    
    # ========================================================================
    # 0阶误差计算
    # ========================================================================
    top, bottom = u[0,:], u[-1,:]
    left, right = u[:,0], u[:,-1]
    
    # 原始差异
    diff_top_bottom = np.mean(np.abs(top - bottom))
    diff_left_right = np.mean(np.abs(left - right))
    diff_all_0th = (diff_top_bottom + diff_left_right) / 2
    
    # 绝对值统计
    abs_u = np.abs(u)
    mean_abs_u = np.mean(abs_u)
    max_abs_u = np.max(abs_u)
    
    # ========================================================================
    # 1阶误差计算
    # ========================================================================
    grad_x, grad_y = compute_gradient(u)
    
    # x方向梯度差异
    diff_grad_x_top_bottom = np.mean(np.abs(grad_x[0,:] - grad_x[-1,:]))
    diff_grad_x_left_right = np.mean(np.abs(grad_x[:,0] - grad_x[:,-1]))
    diff_grad_x_all = (diff_grad_x_top_bottom + diff_grad_x_left_right) / 2
    
    # y方向梯度差异
    diff_grad_y_top_bottom = np.mean(np.abs(grad_y[0,:] - grad_y[-1,:]))
    diff_grad_y_left_right = np.mean(np.abs(grad_y[:,0] - grad_y[:,-1]))
    diff_grad_y_all = (diff_grad_y_top_bottom + diff_grad_y_left_right) / 2
    
    # 组合梯度差异
    diff_grad_xy_all = (diff_grad_x_top_bottom + diff_grad_x_left_right + 
                       diff_grad_y_top_bottom + diff_grad_y_left_right) / 4
    
    # 梯度绝对值统计
    abs_grad_x, abs_grad_y = np.abs(grad_x), np.abs(grad_y)
    mean_abs_grad_x, max_abs_grad_x = np.mean(abs_grad_x), np.max(abs_grad_x)
    mean_abs_grad_y, max_abs_grad_y = np.mean(abs_grad_y), np.max(abs_grad_y)
    mean_abs_grad = (mean_abs_grad_x + mean_abs_grad_y) / 2
    max_abs_grad = max(max_abs_grad_x, max_abs_grad_y)
    
    # ========================================================================
    # 2阶误差计算
    # ========================================================================
    hess_xx, hess_yy, hess_xy = compute_hessian(u)
    
    # 计算边界差异（不再跳过任何位置）
    diff_hess_xx_top_bottom = np.mean(np.abs(hess_xx[0,:] - hess_xx[-1,:]))
    diff_hess_xx_left_right = np.mean(np.abs(hess_xx[:,0] - hess_xx[:,-1]))
    diff_hess_xx_all = (diff_hess_xx_top_bottom + diff_hess_xx_left_right) / 2

    diff_hess_yy_top_bottom = np.mean(np.abs(hess_yy[0,:] - hess_yy[-1,:]))
    diff_hess_yy_left_right = np.mean(np.abs(hess_yy[:,0] - hess_yy[:,-1]))
    diff_hess_yy_all = (diff_hess_yy_top_bottom + diff_hess_yy_left_right) / 2

    diff_hess_xy_top_bottom = np.mean(np.abs(hess_xy[0,:] - hess_xy[-1,:]))
    diff_hess_xy_left_right = np.mean(np.abs(hess_xy[:,0] - hess_xy[:,-1]))
    diff_hess_xy_all = (diff_hess_xy_top_bottom + diff_hess_xy_left_right) / 2

    diff_hess_all = (diff_hess_xx_top_bottom + diff_hess_xx_left_right +
                    diff_hess_yy_top_bottom + diff_hess_yy_left_right +
                    diff_hess_xy_top_bottom + diff_hess_xy_left_right) / 6
    
    # Hessian绝对值统计（忽略NaN）
    def nan_stats(arr):
        """计算数组的统计量，自动处理NaN/Inf"""
        valid = arr[np.isfinite(arr)]
        if valid.size > 0:
            return np.mean(valid), np.max(valid)
        else:
            return 0.0, 0.0 

    def enhanced_nan_stats(arr, name=""):
        """返回包含NaN/Inf统计的字典"""
        total = arr.size
        isnan = np.isnan(arr)
        isposinf = np.isposinf(arr)
        isneginf = np.isneginf(arr)
        isinf = isposinf | isneginf
        valid = ~(isnan | isinf)
        
        stats = {
            'count': {
                'total': total,
                'nan': np.sum(isnan),
                'inf': np.sum(isinf),
                'posinf': np.sum(isposinf),
                'neginf': np.sum(isneginf),
                'valid': np.sum(valid)
            },
            'percentage': {
                'nan': np.mean(isnan) * 100,
                'inf': np.mean(isinf) * 100,
                'posinf': np.mean(isposinf) * 100,
                'neginf': np.mean(isneginf) * 100,
                'valid': np.mean(valid) * 100
            }
        }
        
        # 有效值的统计
        if np.any(valid):
            valid_data = arr[valid]
            stats.update({
                'mean_abs': np.mean(np.abs(valid_data)),
                'max_abs': np.max(np.abs(valid_data)),
                'min_abs': np.min(np.abs(valid_data)),
                'std_abs': np.std(np.abs(valid_data))
            })
        else:
            stats.update({
                'mean_abs': 0,
                'max_abs': 0,
                'min_abs': 0,
                'std_abs': 0
            })
            
        return stats
    
    mean_abs_hess_xx, max_abs_hess_xx = nan_stats(hess_xx)
    mean_abs_hess_yy, max_abs_hess_yy = nan_stats(hess_yy)
    mean_abs_hess_xy, max_abs_hess_xy = nan_stats(hess_xy)
    mean_abs_hess = (mean_abs_hess_xx + mean_abs_hess_yy + mean_abs_hess_xy) / 3
    max_abs_hess = max(max_abs_hess_xx, max_abs_hess_yy, max_abs_hess_xy)
    
    # ========================================================================
    # 重组数据结构
    # ========================================================================
    return {
        # 第一层：原始差异
        'raw_differences': {
            'zeroth_order': {
                'top_bottom': diff_top_bottom,
                'left_right': diff_left_right,
                'all': diff_all_0th
            },
            'first_order': {
                'grad_x': {
                    'top_bottom': diff_grad_x_top_bottom,
                    'left_right': diff_grad_x_left_right,
                    'all': diff_grad_x_all
                },
                'grad_y': {
                    'top_bottom': diff_grad_y_top_bottom,
                    'left_right': diff_grad_y_left_right,
                    'all': diff_grad_y_all
                },
                'grad_xy': {
                    'all': diff_grad_xy_all
                }
            },
            'second_order': {
                'hess_xx': {
                    'top_bottom': diff_hess_xx_top_bottom,
                    'left_right': diff_hess_xx_left_right,
                    'all': diff_hess_xx_all
                },
                'hess_yy': {
                    'top_bottom': diff_hess_yy_top_bottom,
                    'left_right': diff_hess_yy_left_right,
                    'all': diff_hess_yy_all
                },
                'hess_xy': {
                    'top_bottom': diff_hess_xy_top_bottom,
                    'left_right': diff_hess_xy_left_right,
                    'all': diff_hess_xy_all
                },
                'hess_all': diff_hess_all
            }
        },

        # 新增数据质量统计层
        'data_quality': {
            'first_order': {
                'grad_x': enhanced_nan_stats(grad_x, "grad_x"),
                'grad_y': enhanced_nan_stats(grad_y, "grad_y")
            },
            'second_order': {
                'hess_xx': enhanced_nan_stats(hess_xx, "hess_xx"),
                'hess_yy': enhanced_nan_stats(hess_yy, "hess_yy"),
                'hess_xy': enhanced_nan_stats(hess_xy, "hess_xy")
            }
        },
        
        # 第一层：绝对值统计
        'absolute_stats': {
            'zeroth_order': {
                'mean_abs': mean_abs_u,
                'max_abs': max_abs_u
            },
            'first_order': {
                'grad_x': {'mean_abs': mean_abs_grad_x, 'max_abs': max_abs_grad_x},
                'grad_y': {'mean_abs': mean_abs_grad_y, 'max_abs': max_abs_grad_y},
                'grad_xy': {'mean_abs': mean_abs_grad, 'max_abs': max_abs_grad}
            },
            'second_order': {
                'hess_xx': {'mean_abs': mean_abs_hess_xx, 'max_abs': max_abs_hess_xx},
                'hess_yy': {'mean_abs': mean_abs_hess_yy, 'max_abs': max_abs_hess_yy},
                'hess_xy': {'mean_abs': mean_abs_hess_xy, 'max_abs': max_abs_hess_xy},
                'hess_all': {'mean_abs': mean_abs_hess, 'max_abs': max_abs_hess}
            }
        },
        
        # 第一层：标准化误差（新增）
        'normalized_differences': {
            'zeroth_order': {
                'top_bottom_by_mean': safe_divide(diff_top_bottom, mean_abs_u),
                'top_bottom_by_max': safe_divide(diff_top_bottom, max_abs_u),
                'left_right_by_mean': safe_divide(diff_left_right, mean_abs_u),
                'left_right_by_max': safe_divide(diff_left_right, max_abs_u),
                'all_by_mean': safe_divide(diff_all_0th, mean_abs_u),
                'all_by_max': safe_divide(diff_all_0th, max_abs_u)
            },
            'first_order': {
                'grad_x': {
                    'top_bottom_by_mean': safe_divide(diff_grad_x_top_bottom, mean_abs_grad_x),
                    'top_bottom_by_max': safe_divide(diff_grad_x_top_bottom, max_abs_grad_x),
                    'left_right_by_mean': safe_divide(diff_grad_x_left_right, mean_abs_grad_x),
                    'left_right_by_max': safe_divide(diff_grad_x_left_right, max_abs_grad_x),
                    'all_by_mean': safe_divide(diff_grad_x_all, mean_abs_grad_x),
                    'all_by_max': safe_divide(diff_grad_x_all, max_abs_grad_x)
                },
                'grad_y': {
                    'top_bottom_by_mean': safe_divide(diff_grad_y_top_bottom, mean_abs_grad_y),
                    'top_bottom_by_max': safe_divide(diff_grad_y_top_bottom, max_abs_grad_y),
                    'left_right_by_mean': safe_divide(diff_grad_y_left_right, mean_abs_grad_y),
                    'left_right_by_max': safe_divide(diff_grad_y_left_right, max_abs_grad_y),
                    'all_by_mean': safe_divide(diff_grad_y_all, mean_abs_grad_y),
                    'all_by_max': safe_divide(diff_grad_y_all, max_abs_grad_y)
                },
                'grad_xy': {
                    'all_by_mean': safe_divide(diff_grad_xy_all, mean_abs_grad),
                    'all_by_max': safe_divide(diff_grad_xy_all, max_abs_grad)
                }
            },
            'second_order': {
                'hess_xx': {
                    'top_bottom_by_mean': safe_divide(diff_hess_xx_top_bottom, mean_abs_hess_xx),
                    'top_bottom_by_max': safe_divide(diff_hess_xx_top_bottom, max_abs_hess_xx),
                    'left_right_by_mean': safe_divide(diff_hess_xx_left_right, mean_abs_hess_xx),
                    'left_right_by_max': safe_divide(diff_hess_xx_left_right, max_abs_hess_xx),
                    'all_by_mean': safe_divide(diff_hess_xx_all, mean_abs_hess_xx),
                    'all_by_max': safe_divide(diff_hess_xx_all, max_abs_hess_xx)
                },
                'hess_yy': {
                    'top_bottom_by_mean': safe_divide(diff_hess_yy_top_bottom, mean_abs_hess_yy),
                    'top_bottom_by_max': safe_divide(diff_hess_yy_top_bottom, max_abs_hess_yy),
                    'left_right_by_mean': safe_divide(diff_hess_yy_left_right, mean_abs_hess_yy),
                    'left_right_by_max': safe_divide(diff_hess_yy_left_right, max_abs_hess_yy),
                    'all_by_mean': safe_divide(diff_hess_yy_all, mean_abs_hess_yy),
                    'all_by_max': safe_divide(diff_hess_yy_all, max_abs_hess_yy)
                },
                'hess_xy': {
                    'top_bottom_by_mean': safe_divide(diff_hess_xy_top_bottom, mean_abs_hess_xy),
                    'top_bottom_by_max': safe_divide(diff_hess_xy_top_bottom, max_abs_hess_xy),
                    'left_right_by_mean': safe_divide(diff_hess_xy_left_right, mean_abs_hess_xy),
                    'left_right_by_max': safe_divide(diff_hess_xy_left_right, max_abs_hess_xy),
                    'all_by_mean': safe_divide(diff_hess_xy_all, mean_abs_hess_xy),
                    'all_by_max': safe_divide(diff_hess_xy_all, max_abs_hess_xy)
                },
                'hess_all': {
                    'by_mean': safe_divide(diff_hess_all, mean_abs_hess),
                    'by_max': safe_divide(diff_hess_all, max_abs_hess)
                }
            }
        }
    }

def safe_divide(a, b):
    """安全除法（处理除零和无效值），支持数组输入"""
    if isinstance(b, (np.ndarray, list)):  # 如果是数组
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(a, b)
            result[~np.isfinite(result)] = np.nan
        return result
    else:  # 如果是标量
        if b == 0 or np.isnan(b) or np.isinf(b):
            return float('nan')
        return a / b

def remove_nan(arr):
    """移除包含 NaN 的行和列"""
    mask = ~np.isnan(arr)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return arr[rows][:, cols]

def create_attack_visualization(records, output_path="attack_progress.gif", fps=2, dpi=100, metadata=None):
    """
    Create a GIF visualization of the attack progress with proper animation
    
    Args:
        records: List of attack step dictionaries
        output_path: Path to save the GIF
        fps: Frames per second
        dpi: Dots per inch for output quality
        metadata: Dictionary containing attack parameters (e.g., {'alpha': 0.01, 'epsilon': 0.1})
    """
    
    # Prepare loss data
    steps = [r['step'] for r in records]
    losses = [r['loss'] for r in records]
    
    # Calculate global min/max for consistent color scaling
    vmax_diff = max(np.max(np.abs(r['output'] - r['truth'])) for r in records)
    vmax_grad = max(np.max(np.abs(r['gradient'])) for r in records)
    
    # Create parameter string for title
    param_str = ""
    if metadata:
        param_str = " | " + ", ".join([f"{k}={v}" for k,v in metadata.items()])
    script_dir = Path(__file__).parent.resolve()
    
    # Create paths relative to script location
    temp_dir = script_dir / "temp_frames"
    temp_dir.mkdir(exist_ok=True)
    
    output_path = str(script_dir / output_path)  # Full output path
    frame_files = []
    
    for i, record in enumerate(records):
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        # Main title with parameters
        main_title = f'PGD Attack Progress - Step {i}{param_str}'
        fig.suptitle(main_title, fontsize=16)
        
        # Get data for current step
        x0 = record['x0']
        grad = record['gradient']
        output = record['output']
        truth = record['truth']
        reach_boundary = record['reach_boundary']
        diff = output - truth
        mean_abs_diff = np.mean(np.abs(diff))
        
        # Plot 1: x0 (input)
        im1 = axs[0,0].imshow(x0, cmap='viridis')
        axs[0,0].set_title(f'Adversarial Input (x0) \n delta reach boundary={reach_boundary}')
        fig.colorbar(im1, ax=axs[0,0])
        
        # Plot 2: Gradient
        im2 = axs[0,1].imshow(grad, cmap='viridis', vmin=-vmax_grad, vmax=vmax_grad)
        axs[0,1].set_title('Attack Gradient')
        fig.colorbar(im2, ax=axs[0,1])
        
        # Plot 3: Model Output
        im3 = axs[0,2].imshow(output, cmap='viridis')
        axs[0,2].set_title('Model Output')
        fig.colorbar(im3, ax=axs[0,2])
        
        # Plot 4: Ground Truth
        im4 = axs[1,0].imshow(truth, cmap='viridis')
        axs[1,0].set_title('Ground Truth')
        fig.colorbar(im4, ax=axs[1,0])
        
        # Plot 5: Difference (Output - Truth)
        im5 = axs[1,1].imshow(diff, cmap='coolwarm', 
                             norm=Normalize(vmin=-vmax_diff, vmax=vmax_diff))
        axs[1,1].set_title(f'Output-Truth\nMean Abs Diff: {mean_abs_diff:.4f}')
        fig.colorbar(im5, ax=axs[1,1])
        
        # Plot 6: Loss progression
        axs[1,2].plot(steps[:i+1], losses[:i+1], 'b-', marker='o')
        axs[1,2].set_title('Loss Progression')
        axs[1,2].set_xlabel('Attack Step')
        axs[1,2].set_ylabel('Loss')
        axs[1,2].grid(True)
        axs[1,2].set_xlim(-0.5, len(records)-0.5)
        axs[1,2].set_ylim(min(losses)*0.9, max(losses)*1.1)
        
        plt.tight_layout()
        
        # Save frame
        frame_file = str(temp_dir / f"frame_{i:04d}.png")  # <-- Updated path
        plt.savefig(frame_file, dpi=dpi, bbox_inches='tight')
        frame_files.append(frame_file)
        plt.close()
    
    # Create GIF
    images = [Image.open(f) for f in frame_files]
    images[0].save(output_path, save_all=True, append_images=images[1:],
                  duration=1000//fps, loop=0)
    
    # Cleanup
    for f in frame_files:
        os.remove(f)
    temp_dir.rmdir()
    print(f"GIF saved to: {output_path}")
    
    print(f"GIF successfully saved to {output_path}")

def create_compact_attack_visualization(records, output_path="compact_attack_progress.gif", fps=2, dpi=100, metadata=None):
    param_str = ""
    if metadata:
        param_str = " | " + ", ".join([f"{k}={v}" for k,v in metadata.items()])
    script_dir = Path(__file__).parent.resolve()
    temp_dir = script_dir / "temp_frames"
    temp_dir.mkdir(exist_ok=True)
    
    output_path = str(script_dir / output_path)
    frame_files = []
    
    # 预计算所有步骤的指标
    all_metrics = []
    for record in records:
        metrics = {
            'x0': periodic_boundary_error(record['x0']),
            'grad': periodic_boundary_error(record['gradient']),
            'output': periodic_boundary_error(record['output']),
            'truth': periodic_boundary_error(record['truth'])
        }
        all_metrics.append(metrics)
    
    for i, (record, metrics) in enumerate(zip(records, all_metrics)):
        # 创建7个子图（4个原始图 + 3个折线图）
        fig = plt.figure(figsize=(18, 18))
        gs = fig.add_gridspec(3, 3)
        axs = [
            fig.add_subplot(gs[0, 0]),  # x0
            fig.add_subplot(gs[0, 1]),  # grad
            fig.add_subplot(gs[1, 0]),  # output
            fig.add_subplot(gs[1, 1]),  # truth
            fig.add_subplot(gs[0, 2]),  # 0阶误差折线
            fig.add_subplot(gs[1, 2]),  # 1阶误差折线
            fig.add_subplot(gs[2, 0])   # 2阶误差折线
        ]
        
        main_title = f'Enhanced Attack Progress - Step {i}{param_str}'
        fig.suptitle(main_title, fontsize=16)
        
        # 原始四个图像（代码保持不变）
        def tile_array(arr):
            return np.tile(arr, (2, 2))
        
        # Plot 1: x0
        tiled_x0 = tile_array(record['x0'])
        im1 = axs[0].imshow(tiled_x0, cmap='viridis')
        axs[0].set_title(f'Adversarial Input (x0) \n delta reach boundary={record["reach_boundary"]}', fontsize=9)
        fig.colorbar(im1, ax=axs[0])
        
        # Plot 2: Gradient
        tiled_grad = tile_array(record['gradient'])
        im2 = axs[1].imshow(tiled_grad, cmap='viridis')
        axs[1].set_title('Attack Gradient', fontsize=9)
        fig.colorbar(im2, ax=axs[1])
        
        # Plot 3: Model Output
        tiled_output = tile_array(record['output'])
        im3 = axs[2].imshow(tiled_output, cmap='viridis')
        axs[2].set_title('Model Output', fontsize=9)
        fig.colorbar(im3, ax=axs[2])
        
        # Plot 4: PDE Output (Truth)
        tiled_truth = tile_array(record['truth'])
        im4 = axs[3].imshow(tiled_truth, cmap='viridis')
        axs[3].set_title('PDE Solution (Truth)', fontsize=9)
        fig.colorbar(im4, ax=axs[3])
        
        # ========================================================================
        # 新增的三个折线图
        # ========================================================================
        
        # Plot 5: 0阶误差演化
        axs[4].set_title('0th Order Boundary Error (%)', fontsize=10)
        for field in ['x0', 'grad', 'output', 'truth']:
            errors = [m[field]['normalized_differences']['zeroth_order']['all_by_mean']*100 
                    for m in all_metrics[:i+1]]
            axs[4].plot(errors, label=f'{field} (0th)')
        axs[4].set_xlabel('Attack Step')
        axs[4].set_ylabel('Error %')
        axs[4].legend()
        axs[4].grid(True)

        # Plot 6: 1阶误差演化
        axs[5].set_title('1st Order Gradient Error (%)', fontsize=10)
        for field in ['x0', 'grad', 'output', 'truth']:
            errors_x = [m[field]['normalized_differences']['first_order']['grad_x']['all_by_mean']*100 
                        for m in all_metrics[:i+1]]
            errors_y = [m[field]['normalized_differences']['first_order']['grad_y']['all_by_mean']*100 
                        for m in all_metrics[:i+1]]
            combined_errors = [(x+y)/2 for x,y in zip(errors_x, errors_y)]
            axs[5].plot(combined_errors, label=f'{field} (1st)')
        axs[5].set_xlabel('Attack Step')
        axs[5].set_ylabel('Error %')
        axs[5].legend()
        axs[5].grid(True)

        # Plot 7: 2阶误差演化
        axs[6].set_title('2nd Order Hessian Error (%)', fontsize=10)
        for field in ['x0', 'grad', 'output', 'truth']:
            errors = [m[field]['normalized_differences']['second_order']['hess_all']['by_mean']*100 
                    for m in all_metrics[:i+1]]
            axs[6].plot(errors, label=f'{field} (2nd)')
        axs[6].set_xlabel('Attack Step')
        axs[6].set_ylabel('Error %')
        axs[6].legend()
        axs[6].grid(True)
        axs[6].set_ylim([0,200])
        
        plt.tight_layout()
        
        # Save frame
        frame_file = str(temp_dir / f"frame_{i:04d}.png")
        plt.savefig(frame_file, dpi=dpi, bbox_inches='tight')
        frame_files.append(frame_file)
        plt.close()
    
    # Create GIF
    images = [Image.open(f) for f in frame_files]
    images[0].save(output_path, save_all=True, append_images=images[1:],
                  duration=1000//fps, loop=0, optimize=True, quality=95)
    
    # Cleanup
    for f in frame_files:
        os.remove(f)
    temp_dir.rmdir()
    print(f"Enhanced GIF saved to: {output_path}")

def create_compact_attack_visualization_grad(records, output_path="compact_attack_progress.gif", fps=2, dpi=100, metadata=None):
    param_str = ""
    if metadata:
        param_str = " | " + ", ".join([f"{k}={v}" for k,v in metadata.items()])
    script_dir = Path(__file__).parent.resolve()
    temp_dir = script_dir / "temp_frames"
    temp_dir.mkdir(exist_ok=True)
    
    output_path = str(script_dir / output_path)
    frame_files = []
    
    for i, record in enumerate(records):
        # 创建3x3的子图网格（共9个图）
        fig, axs = plt.subplots(3, 3, figsize=(18, 18))
        axs = axs.ravel()  # 展平为1D数组方便索引
        
        main_title = f'Enhanced Attack Progress - Step {i}{param_str}'
        fig.suptitle(main_title, fontsize=16)
        
        # 原始四个图像
        def tile_array(arr):
            return np.tile(arr, (2, 2))
        
        # Plot 1: x0 (input)
        tiled_x0 = tile_array(record['x0'])
        im1 = axs[0].imshow(tiled_x0, cmap='viridis')
        axs[0].set_title(f'Adversarial Input (x0)\nreach_boundary={record["reach_boundary"]}', fontsize=9)
        fig.colorbar(im1, ax=axs[0])
        
        # Plot 2: Gradient (原gradient图)
        tiled_grad = tile_array(record['gradient'])
        im2 = axs[1].imshow(tiled_grad, cmap='viridis')
        axs[1].set_title('Attack Gradient (Magnitude)', fontsize=9)
        fig.colorbar(im2, ax=axs[1])
        
        # Plot 3: Model Output
        tiled_output = tile_array(record['output'])
        im3 = axs[2].imshow(tiled_output, cmap='viridis')
        axs[2].set_title('Model Output', fontsize=9)
        fig.colorbar(im3, ax=axs[2])
        
        # Plot 4: PDE Output (Truth)
        tiled_truth = tile_array(record['truth'])
        im4 = axs[3].imshow(tiled_truth, cmap='viridis')
        axs[3].set_title('PDE Solution (Truth)', fontsize=9)
        fig.colorbar(im4, ax=axs[3])
        
        # ========================================================================
        # 新增的5个导数图
        # ========================================================================
        
        # 计算梯度和Hessian
        grad_x, grad_y = compute_gradient(record['gradient'])
        hess_xx, hess_yy, hess_xy = compute_hessian(record['gradient'])

        hess_xx = remove_nan(hess_xx)
        hess_yy = remove_nan(hess_yy)
        hess_xy = remove_nan(hess_xy)
        
        # Plot 5: Gradient X分量
        tiled_grad_x = tile_array(grad_x)
        im5 = axs[4].imshow(tiled_grad_x, cmap='viridis')
        axs[4].set_title('gradient Gradient X Component', fontsize=9)
        fig.colorbar(im5, ax=axs[4])
        
        # Plot 6: Gradient Y分量
        tiled_grad_y = tile_array(grad_y)
        im6 = axs[5].imshow(tiled_grad_y, cmap='viridis')
        axs[5].set_title('gradient Gradient Y Component', fontsize=9)
        fig.colorbar(im6, ax=axs[5])
        
        # Plot 7: Hessian XX分量
        tiled_hess_xx = tile_array(hess_xx)
        im7 = axs[6].imshow(tiled_hess_xx, cmap='viridis')
        axs[6].set_title('gradient Hessian XX Component', fontsize=9)
        fig.colorbar(im7, ax=axs[6])
        
        # Plot 8: Hessian YY分量
        tiled_hess_yy = tile_array(hess_yy)
        im8 = axs[7].imshow(tiled_hess_yy, cmap='viridis')
        axs[7].set_title('gradient Hessian YY Component', fontsize=9)
        fig.colorbar(im8, ax=axs[7])
        
        # Plot 9: Hessian XY分量
        tiled_hess_xy = tile_array(hess_xy)
        im9 = axs[8].imshow(tiled_hess_xy, cmap='viridis')
        axs[8].set_title('gradient Hessian XY Component', fontsize=9)
        fig.colorbar(im9, ax=axs[8])
        
        plt.tight_layout()
        
        # Save frame
        frame_file = str(temp_dir / f"frame_{i:04d}.png")
        plt.savefig(frame_file, dpi=dpi, bbox_inches='tight')
        frame_files.append(frame_file)
        plt.close()
    
    # Create GIF
    images = [Image.open(f) for f in frame_files]
    images[0].save(output_path, save_all=True, append_images=images[1:],
                  duration=1000//fps, loop=0, optimize=True, quality=95)
    
    # Cleanup
    for f in frame_files:
        os.remove(f)
    temp_dir.rmdir()
    print(f"Enhanced GIF saved to: {output_path}")

def create_compact_attack_visualization_x0(records, output_path="compact_attack_progress.gif", fps=2, dpi=100, metadata=None):
    param_str = ""
    if metadata:
        param_str = " | " + ", ".join([f"{k}={v}" for k,v in metadata.items()])
    script_dir = Path(__file__).parent.resolve()
    temp_dir = script_dir / "temp_frames"
    temp_dir.mkdir(exist_ok=True)
    
    output_path = str(script_dir / output_path)
    frame_files = []
    
    for i, record in enumerate(records):
        # 创建3x3的子图网格（共9个图）
        fig, axs = plt.subplots(3, 3, figsize=(18, 18))
        axs = axs.ravel()  # 展平为1D数组方便索引
        
        main_title = f'Enhanced Attack Progress - Step {i}{param_str}'
        fig.suptitle(main_title, fontsize=16)
        
        # 原始四个图像
        def tile_array(arr):
            return np.tile(arr, (2, 2))
        
        # Plot 1: x0 (input)
        tiled_x0 = tile_array(record['x0'])
        im1 = axs[0].imshow(tiled_x0, cmap='viridis')
        axs[0].set_title(f'Adversarial Input (x0)\nreach_boundary={record["reach_boundary"]}', fontsize=9)
        fig.colorbar(im1, ax=axs[0])
        
        # Plot 2: Gradient (原gradient图)
        tiled_grad = tile_array(record['gradient'])
        im2 = axs[1].imshow(tiled_grad, cmap='viridis')
        axs[1].set_title('Attack Gradient (Magnitude)', fontsize=9)
        fig.colorbar(im2, ax=axs[1])
        
        # Plot 3: Model Output
        tiled_output = tile_array(record['output'])
        im3 = axs[2].imshow(tiled_output, cmap='viridis')
        axs[2].set_title('Model Output', fontsize=9)
        fig.colorbar(im3, ax=axs[2])
        
        # Plot 4: PDE Output (Truth)
        tiled_truth = tile_array(record['truth'])
        im4 = axs[3].imshow(tiled_truth, cmap='viridis')
        axs[3].set_title('PDE Solution (Truth)', fontsize=9)
        fig.colorbar(im4, ax=axs[3])
        
        # ========================================================================
        # 新增的5个导数图
        # ========================================================================
        
        # 计算梯度和Hessian
        grad_x, grad_y = compute_gradient(record['x0'])
        hess_xx, hess_yy, hess_xy = compute_hessian(record['x0'])

        hess_xx = remove_nan(hess_xx)
        hess_yy = remove_nan(hess_yy)
        hess_xy = remove_nan(hess_xy)

        # Plot 5: Gradient X分量
        tiled_grad_x = tile_array(grad_x)
        im5 = axs[4].imshow(tiled_grad_x, cmap='viridis')
        axs[4].set_title('x0 Gradient X Component', fontsize=9)
        fig.colorbar(im5, ax=axs[4])
        
        # Plot 6: Gradient Y分量
        tiled_grad_y = tile_array(grad_y)
        im6 = axs[5].imshow(tiled_grad_y, cmap='viridis')
        axs[5].set_title('x0 Gradient Y Component', fontsize=9)
        fig.colorbar(im6, ax=axs[5])
        
        # Plot 7: Hessian XX分量
        tiled_hess_xx = tile_array(hess_xx)
        im7 = axs[6].imshow(tiled_hess_xx, cmap='viridis')
        axs[6].set_title('x0 Hessian XX Component', fontsize=9)
        fig.colorbar(im7, ax=axs[6])
        
        # Plot 8: Hessian YY分量
        tiled_hess_yy = tile_array(hess_yy)
        im8 = axs[7].imshow(tiled_hess_yy, cmap='viridis')
        axs[7].set_title('x0 Hessian YY Component', fontsize=9)
        fig.colorbar(im8, ax=axs[7])
        
        # Plot 9: Hessian XY分量
        tiled_hess_xy = tile_array(hess_xy)
        im9 = axs[8].imshow(tiled_hess_xy, cmap='viridis')
        axs[8].set_title('x0 Hessian XY Component', fontsize=9)
        fig.colorbar(im9, ax=axs[8])
        
        plt.tight_layout()
        
        # Save frame
        frame_file = str(temp_dir / f"frame_{i:04d}.png")
        plt.savefig(frame_file, dpi=dpi, bbox_inches='tight')
        frame_files.append(frame_file)
        plt.close()
    
    # Create GIF
    images = [Image.open(f) for f in frame_files]
    images[0].save(output_path, save_all=True, append_images=images[1:],
                  duration=1000//fps, loop=0, optimize=True, quality=95)
    
    # Cleanup
    for f in frame_files:
        os.remove(f)
    temp_dir.rmdir()
    print(f"Enhanced GIF saved to: {output_path}")

if __name__ == "__main__":
    import os
    from pathlib import Path
    from tqdm import tqdm
    import time

    # Get the directory where the current Python file is located
    current_dir = Path(__file__).parent

    # Find all pickle files in this directory (non-recursive)
    # pickle_files = [str(file.resolve()) for file in current_dir.iterdir() if file.is_file() and file.suffix.lower() in ('.pkl', '.pickle')]
    
    """
    pickle_files = [
        "pgd_attack_records_norm2_alpha10.0_epsilon65.536_steps100_idx1.pkl",
        "pgd_attack_records_norm2_alpha10.0_epsilon65.536_steps100_idx2.pkl",
        "pgd_attack_records_norm2_alpha10.0_epsilon65.536_steps100_idx3.pkl",
        "pgd_attack_records_norm2_alpha10.0_epsilon655.36_steps100_idx1.pkl",
        "pgd_attack_records_norm2_alpha10.0_epsilon655.36_steps100_idx2.pkl",
        "pgd_attack_records_norm2_alpha10.0_epsilon655.36_steps100_idx3.pkl",
    ]
    """

    """
    pickle_files = [
        "pgd_attack_records_norm2_alpha1.0_epsilon6.5536_steps100_idx1.pkl",
        "pgd_attack_records_norm2_alpha1.0_epsilon6.5536_steps100_idx2.pkl",
        "pgd_attack_records_norm2_alpha1.0_epsilon6.5536_steps100_idx3.pkl",
        "pgd_attack_records_norm2_alpha2.0_epsilon6.5536_steps100_idx1.pkl",
        "pgd_attack_records_norm2_alpha2.0_epsilon6.5536_steps100_idx2.pkl",
        "pgd_attack_records_norm2_alpha2.0_epsilon6.5536_steps100_idx3.pkl",
        "pgd_attack_records_norm2_alpha5.0_epsilon6.5536_steps100_idx1.pkl",
        "pgd_attack_records_norm2_alpha5.0_epsilon6.5536_steps100_idx2.pkl",
        "pgd_attack_records_norm2_alpha5.0_epsilon6.5536_steps100_idx3.pkl",
    ]
    """

    pickle_files = [
        "pgd_attack_records_norm2_alpha1.0_epsilon65.536_steps100_idx1.pkl",
        "pgd_attack_records_norm2_alpha1.0_epsilon65.536_steps100_idx2.pkl",
        "pgd_attack_records_norm2_alpha1.0_epsilon65.536_steps100_idx3.pkl",
        "pgd_attack_records_norm2_alpha2.0_epsilon65.536_steps100_idx1.pkl",
        "pgd_attack_records_norm2_alpha2.0_epsilon65.536_steps100_idx2.pkl",
        "pgd_attack_records_norm2_alpha2.0_epsilon65.536_steps100_idx3.pkl",
        "pgd_attack_records_norm2_alpha5.0_epsilon65.536_steps100_idx1.pkl",
        "pgd_attack_records_norm2_alpha5.0_epsilon65.536_steps100_idx2.pkl",
        "pgd_attack_records_norm2_alpha5.0_epsilon65.536_steps100_idx3.pkl",
    ]

    pickle_files = [current_dir / file for file in pickle_files] 
    
    for filename in tqdm(pickle_files, desc="Processing attack visualizations"):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        metadata_str = "_".join([f"{key}{val}" for key, val in data["metadata"].items()])

        # 第一个GIF - 常规可视化
        start_time = time.time()
        output_filename = f"attack_progress_{metadata_str}.gif" if metadata_str else f"attack_progress_{os.path.splitext(filename)[0]}.gif"
        create_attack_visualization(
            data["steps"],
            output_path=output_filename,
            fps=5,
            metadata=data["metadata"]
        )
        elapsed = time.time() - start_time
        print(f"Created visualization: {output_filename} \n    | Time: {elapsed:.2f}s")

        """
        # 第二个GIF - 紧凑可视化
        start_time = time.time()
        output_filename = f"attack_progress_{metadata_str}_periodic.gif" if metadata_str else f"attack_progress_{os.path.splitext(filename)[0]}.gif"
        create_compact_attack_visualization(
            data["steps"],
            output_path=output_filename,
            fps=5,
            metadata=data["metadata"]
        )
        elapsed = time.time() - start_time
        print(f"Created visualization: {output_filename} \n    | Time: {elapsed:.2f}s")

        # 第三个GIF - 梯度可视化
        start_time = time.time()
        output_filename = f"attack_progress_{metadata_str}_periodic_grad.gif" if metadata_str else f"attack_progress_{os.path.splitext(filename)[0]}.gif"
        create_compact_attack_visualization_grad(
            data["steps"],
            output_path=output_filename,
            fps=5,
            metadata=data["metadata"]
        )
        elapsed = time.time() - start_time
        print(f"Created visualization: {output_filename} \n    | Time: {elapsed:.2f}s")

        # 第四个GIF - x0可视化
        start_time = time.time()
        output_filename = f"attack_progress_{metadata_str}_periodic_x0.gif" if metadata_str else f"attack_progress_{os.path.splitext(filename)[0]}.gif"
        create_compact_attack_visualization_x0(
            data["steps"],
            output_path=output_filename,
            fps=5,
            metadata=data["metadata"]
        )
        elapsed = time.time() - start_time
        print(f"Created visualization: {output_filename} \n    | Time: {elapsed:.2f}s")
        """


