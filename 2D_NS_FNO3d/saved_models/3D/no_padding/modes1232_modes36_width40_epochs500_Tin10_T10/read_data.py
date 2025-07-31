import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image
import os
import sys
from pathlib import Path
import torch
from tqdm import tqdm

def create_heatmap_gif(array1, array2, 
                      title1='Heatmap 1', title2='Heatmap 2', 
                      cmap='viridis', figsize=(18, 6),
                      colorbar=True, 
                      fps=10, dpi=100, text="training"):
    # Create temporary directory for frames
    os.makedirs('temp_frames', exist_ok=True)
    
    frame_files = []
    
    for idx in range(array1.shape[0]):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot first heatmap
        im1 = ax1.imshow(array1[idx], cmap=cmap, origin='lower')
        ax1.set_title(title1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        if colorbar:
            fig.colorbar(im1, ax=ax1, label='Value')
        
        # Plot second heatmap
        im2 = ax2.imshow(array2[idx], cmap=cmap, origin='lower')
        ax2.set_title(title2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        if colorbar:
            fig.colorbar(im2, ax=ax2, label='Value')

        # Plot second heatmap
        diff = array2[idx]-array1[idx]
        im3 = ax3.imshow(diff, cmap="coolwarm", origin='lower')
        ax3.set_title(f"{title2}-{title1} difference \n mean abs={np.mean(np.abs(diff))}")
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        if colorbar:
            fig.colorbar(im3, ax=ax3, label='Value')
        
        plt.suptitle(f"Navier-Stokes Vorticity Transport (frame {idx+10}) ({text})")
        plt.tight_layout()
        
        # Save frame
        frame_file = f"temp_frames/frame_{idx:04d}.png"
        plt.savefig(frame_file, dpi=dpi, bbox_inches='tight')
        frame_files.append(frame_file)
        plt.close()
    
    # Create GIF from frames
    images = [Image.open(f) for f in frame_files]
    os.makedirs(f"{Path(__file__).parent}/gifs_{text.split(' ')[1]}", exist_ok=True)
    images[0].save(f"{Path(__file__).parent}/gifs_{text.split(' ')[1]}/{text}.gif", 
                  save_all=True, 
                  append_images=images[1:], 
                  duration=1000//fps, 
                  loop=0)
    
    # Clean up temporary files
    for f in frame_files:
        os.remove(f)
    os.rmdir('temp_frames')
    
    print(f"GIF saved as {text}.gif")


project_root = Path(__file__).parent.parent.parent  # 根据实际情况调整
utilities_path = project_root / "models" 
sys.path.append(str(utilities_path))

T_in = 10
T = 10
s = 256
step = 1
train_test = "test"
# train_test = "train"

notebook_path = Path(os.getcwd())
project_root = notebook_path.parent.parent.parent
sys.path.append(str(project_root))

from models.FNO3d import FNO3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the data file
current_dir = Path(__file__).parent

# Construct paths relative to the script location
data_test_path = current_dir.parent.parent.parent.parent / "2D_NS_FNO2d_recurrent" / "datasets" / "exponax_datasets" / "t20" / f"dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_{train_test}_all_frames.pt"
model_path = current_dir / "NS_3d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth"

# Load files
data_test = torch.load(data_test_path, weights_only=False, map_location=device)
model_instance = FNO3d(modes1=32, modes2=32, modes3=6, width=40).to(device)
state_dict = torch.load(model_path, map_location=device)

model_instance.load_state_dict(state_dict)
model_instance.eval() 

for idx in tqdm(range(0, 5)):
    # 加载测试数据和模型
    sample = data_test["y"][idx]  # shape: (x, y, T_total)
    sample = sample.to(device)

    # 初始输入 (1, x, y, T_in)
    current_input = sample[None, ..., :T_in]

    # 真实输出 (1, x, y, T)
    ground_truth = sample[None, ..., T_in:T_in+T]

    # 执行 autoregressive 推理
    with torch.no_grad():
        pred = model_instance(current_input.reshape(1,s,s,1,T_in).repeat([1,1,1,T,1]))  # (1, x, y, T)
        pred_np = pred.squeeze(0).cpu().numpy().transpose(2, 0, 1, 3).squeeze()  # (T, x, y)
        ground_truth_np = ground_truth.squeeze(0).cpu().numpy().transpose(2, 0, 1)  # (T, x, y)

    # 可视化预测和真实值
    create_heatmap_gif(
        ground_truth_np,  # (T, x, y)
        pred_np,          # (T, x, y)
        title1='PDE Solution',
        title2='FNO Prediction',
        fps=5,
        text=f"modes{model_instance.modes1}-width{model_instance.width} {train_test} (Autoregressive T_in={T_in}, T_out={T}) index={idx}"
    )





