import torch
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
                           dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.mlp(x)

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_layers=4, in_channels=10):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        # self.padding = 9

        self.p = nn.Linear(in_channels+2, self.width)
        
        # Initialize layers dynamically
        self.conv_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.conv_layers.append(SpectralConv2d(self.width, self.width, modes1, modes2))
            self.mlp_layers.append(MLP(self.width, self.width, self.width))
            self.w_layers.append(nn.Conv2d(self.width, self.width, 1))
        
        self.q = MLP(self.width, 1, self.width * 4)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.num_layers):
            x1 = self.conv_layers[i](x)
            x1 = self.mlp_layers[i](x1)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:  # No GELU after last layer
                x = F.gelu(x)

        # x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        return x.permute(0, 2, 3, 1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)


class RecurrentPredictor(torch.nn.Module):
    def __init__(self, model, T_out=10, step=1):
        super().__init__()
        self.model = model
        self.T_out = T_out
        self.step = step

    def forward(self, x_init):
        # x_init: (batch, s, s, T_in=10)
        batch_size, s1, s2, T_in = x_init.shape
        outputs = []

        x = x_init
        for _ in range(0, self.T_out, self.step):
            y_pred = self.model(x)  # 预测后面1帧，形状(batch, s, s, step)
            outputs.append(y_pred)
            # 滚动输入：删掉最前面step帧，拼接新预测帧
            x = torch.cat([x[..., self.step:], y_pred], dim=-1)

        # 拼接所有预测帧，变成(batch, s, s, T_out)
        return torch.cat(outputs, dim=-1) 

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def load_y_data():
    log("🔷 开始加载数据...")
    t0 = time.time()
    all_y_list, all_meta_list = [], []
    folder1 = Path(__file__).parent.parent / "generalizability"
    folder2 = Path(__file__).parent.parent
    pt_files = list(folder1.glob("*.pt")) + list(folder2.glob("*.pt"))

    for fpath in pt_files:
        data = torch.load(fpath, weights_only=False)
        y, meta = data["y"], data["metadata"]

        if y.ndim == 2: y = y.unsqueeze(0)
        all_y_list.append(y)

        dtype = ("generalizability" if folder1 in fpath.parents else 
                 "train" if "train" in fpath.name.lower() else 
                 "test" if "test" in fpath.name.lower() else "unknown")

        if isinstance(meta, list):
            for m in meta: m["data_type"] = dtype
            all_meta_list.extend(meta)
        else:
            meta["data_type"] = dtype
            all_meta_list.extend([meta] * y.shape[0])

    all_y = torch.cat(all_y_list, dim=0).numpy()
    assert len(all_meta_list) == all_y.shape[0]
    log(f"✅ 加载完成，用时 {time.time()-t0:.1f}s，形状: {all_y.shape}")
    return all_y, all_meta_list

# 已定义：
# - FNO2d
# - RecurrentPredictor
# - load_data() （改为返回 y）
# - log() （如果有）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 加载模型 ==========
script_dir = Path(__file__).parent
fno = FNO2d(modes1=64, modes2=64, width=60, in_channels=10).to(device)

state_dict = torch.load(
    f'{script_dir}/saved_models/2D/modes64_width60_epochs500_Tin10_T10/NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pth',
    map_location=device
)
fno.load_state_dict(state_dict)
fno.eval()

model = RecurrentPredictor(fno, T_out=10, step=1).to(device)
model.eval()

# ========== 加载数据 ==========
all_y, all_meta = load_y_data()   # all_y: numpy (N, 256, 256, 21)

N = all_y.shape[0]
print(f"总共 {N} 条样本")

# ========== 记录指标 ==========
all_rmse, all_mae, all_mape = [], [], []

# ========== 遍历每一条样本 ==========
start_time = time.time()

with tqdm(range(N), desc="Processing", unit="sample") as pbar:
    for idx in pbar:
        y_np = all_y[idx]  # (256, 256, 21)

        # 输入：前 10 帧
        x0 = y_np[..., 0:10]  # (256, 256, 10)
        true_19 = y_np[..., 19]  # (256, 256)

        # 转成 torch
        x_tensor = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 256, 256, 10)
        true_19_tensor = torch.tensor(true_19, dtype=torch.float32, device=device)    # (256, 256)

        with torch.no_grad():
            pred_seq = model(x_tensor)  # (1, 256, 256, 10)
            pred_19 = pred_seq[0, ..., -1]  # (256, 256)

        # 计算指标
        diff = pred_19 - true_19_tensor
        rmse = torch.sqrt(torch.mean(diff**2)).item()
        mae = torch.mean(torch.abs(diff)).item()
        mape = torch.mean(torch.abs(diff / (true_19_tensor + 1e-8))).item() * 100

        all_rmse.append(rmse)
        all_mae.append(mae)
        all_mape.append(mape)

        pbar.set_postfix({
            "RMSE": f"{rmse:.4f}",
            "MAE": f"{mae:.4f}",
            "MAPE": f"{mape:.2f}%"
        })

end_time = time.time()
print(f"\n✅ 全部样本完成！总耗时: {end_time-start_time:.1f}s")

# ========== 统计整体指标 ==========
print("\n📊 统计结果:")
print(f"平均 RMSE: {np.mean(all_rmse):.4f}")
print(f"平均 MAE:  {np.mean(all_mae):.4f}")
print(f"平均 MAPE: {np.mean(all_mape):.2f}%")
