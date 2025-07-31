from pathlib import Path
import torch, numpy as np, pandas as pd
from sklearn.decomposition import PCA
import umap
from tensorly.decomposition import tucker
from tqdm import tqdm
import time

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

# ========== 读取数据 ========== #
def load_x_data():
    log("🔷 开始加载数据...")
    t0 = time.time()
    all_x_list, all_meta_list = [], []
    folder1 = Path(__file__).parent.parent / "generalizability"
    folder2 = Path(__file__).parent.parent
    pt_files = list(folder1.glob("*.pt")) + list(folder2.glob("*.pt"))

    for fpath in pt_files:
        data = torch.load(fpath, weights_only=False)
        x, meta = data["x"], data["metadata"]

        if x.ndim == 2: x = x.unsqueeze(0)
        all_x_list.append(x)

        dtype = ("generalizability" if folder1 in fpath.parents else 
                 "train" if "train" in fpath.name.lower() else 
                 "test" if "test" in fpath.name.lower() else "unknown")

        if isinstance(meta, list):
            for m in meta: m["data_type"] = dtype
            all_meta_list.extend(meta)
        else:
            meta["data_type"] = dtype
            all_meta_list.extend([meta] * x.shape[0])

    all_x = torch.cat(all_x_list, dim=0).numpy()
    assert len(all_meta_list) == all_x.shape[0]
    log(f"✅ 加载完成，用时 {time.time()-t0:.1f}s，形状: {all_x.shape}")
    return all_x, all_meta_list


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

# ========== 预处理输入 ========== #
def prepare_inputs(all_x):
    log("🔷 开始预处理...")
    t0 = time.time()
    N = all_x.shape[0]
    X_time = all_x#.reshape(N, -1)

    F = np.fft.fft2(all_x, axes=(-2,-1))
    amp = np.abs(F)#.reshape(N, -1)

    phase = np.angle(F)
    phase_cos, phase_sin = np.cos(phase), np.sin(phase)

    def norm(x):
        x_min = x.min(axis=(-2,-1), keepdims=True)
        x_max = x.max(axis=(-2,-1), keepdims=True)
        return (x - x_min) / (x_max - x_min + 1e-8)

    amp_n, cos_n, sin_n = norm(np.abs(F)), norm(phase_cos), norm(phase_sin)
    X_freq3 = np.stack([amp_n, cos_n, sin_n], axis=1)#.reshape(N, -1)

    log(f"✅ 预处理完成，用时 {time.time()-t0:.1f}s")
    return {"time": X_time, "freq": np.expand_dims(amp, axis=1), "freq3": X_freq3}



# ========== 降维函数 ========== #
def pca_umap(X, n_pca=30, umap_dim=2):
    X_pca = PCA(n_components=min(n_pca, X.shape[1])).fit_transform(X)
    return umap.UMAP(n_components=umap_dim).fit_transform(X_pca)

def tucker_sample(X, ranks=[1,30,30]):
    log("🔷 Tucker per-sample...")
    t0 = time.time()
    feats = []
    for i in tqdm(range(X.shape[0]), desc="Tucker-per-sample"):
        core, _ = tucker(X[i], rank=ranks)
        feats.append(core.flatten())
    log(f"✅ Tucker per-sample完成，用时 {time.time()-t0:.1f}s")
    return np.array(feats)

def tucker_overall(X, ranks_sample=50):
    log("🔷 Tucker overall...")
    t0 = time.time()
    core, factors = tucker(X, rank=[ranks_sample, X.shape[1], 30,30])
    log(f"✅ Tucker overall完成，用时 {time.time()-t0:.1f}s")
    return factors[0]

# ========== 主逻辑 ========== #
all_x, all_meta = load_x_data()
inputs = prepare_inputs(all_x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 加载模型 ==========
script_dir = Path(__file__).parent.parent.parent.parent.parent
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

start_time = time.time()

print(f"\n predict all data from generalization dataset")
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

# -- 你的 UMAP 结果合并部分 --

records = []
umap_results = {name:{} for name in inputs.keys()}
methods = ["pca", "tucker1", "tucker2"]
dims = [2,3]
N = all_x.shape[0]

for name, X in inputs.items():
    log(f"=== 开始处理输入: {name} ===")
    
    # 方法1：pca_umap
    for dim in dims:
        t_start = time.time()
        umap_results[name][f"pca_{dim}d"] = pca_umap(X.reshape(N,-1), umap_dim=dim)
        t_end = time.time()
        print(f"[{name}] pca_umap {dim}d 用时: {t_end - t_start:.2f}s")
    
    # 方法2：tucker_sample + UMAP
    features1 = tucker_sample(X)
    n_pca1 = min(50, features1.shape[0], features1.shape[1])
    for dim in dims:
        t_start = time.time()
        umap_results[name][f"tucker1_{dim}d"] = umap.UMAP(n_components=dim).fit_transform(
            PCA(n_components=n_pca1).fit_transform(features1))
        t_end = time.time()
        print(f"[{name}] tucker1 UMAP {dim}d 用时: {t_end - t_start:.2f}s")

    # 方法3：tucker_overall + UMAP
    features2 = tucker_overall(X)
    n_pca2 = min(50, features2.shape[0], features2.shape[1])
    for dim in dims:
        t_start = time.time()
        umap_results[name][f"tucker2_{dim}d"] = umap.UMAP(n_components=dim).fit_transform(
            PCA(n_components=n_pca2).fit_transform(features2))
        t_end = time.time()
        print(f"[{name}] tucker2 UMAP {dim}d 用时: {t_end - t_start:.2f}s")

    log(f"✅ {name} 全部维度完成")

log("🔷 开始合并结果...")
t0 = time.time()

for i in tqdm(range(all_x.shape[0]), desc="Merging results"):
    record = dict(all_meta[i])
    for name in inputs.keys():
        for method in methods:
            for dim in dims:
                key = f"{method}_{dim}d"
                coords = umap_results[name][key][i]
                for j, axis in enumerate("xyz"[:dim]):
                    record[f"{name}_{method}_{axis}_{dim}d"] = coords[j]

    # 追加FNO指标（与样本索引一一对应）
    record['fno_rmse'] = all_rmse[i]
    record['fno_mae'] = all_mae[i]
    record['fno_mape'] = all_mape[i]

    records.append(record)

df = pd.DataFrame(records)
df.to_csv("umap_all_methods_results_with_fno_metrics.csv", index=False)

log(f"✅ 所有完成，总用时 {time.time()-t0:.1f}s，结果保存到 umap_all_methods_results_with_fno_metrics.csv")
print(df.head())


