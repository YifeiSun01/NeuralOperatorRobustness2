from pathlib import Path
import torch, numpy as np, pandas as pd
from sklearn.decomposition import PCA
import umap, tensorly as tl
from tensorly.decomposition import tucker
from tqdm import tqdm
import time

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ========== 读取数据 ========== #
def load_data():
    log("🔷 开始加载数据...")
    t0 = time.time()
    all_x_list, all_meta_list = [], []
    folder1 = Path(__file__).parent.parent / "generalizability"
    folder2 = Path(__file__).parent.parent
    pt_files = list(folder1.glob("*.pt")) + list(folder2.glob("*.pt"))

    for fpath in pt_files:
        data = torch.load(fpath)
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

# ========== 预处理输入 ========== #
def prepare_inputs(all_x):
    log("🔷 开始预处理...")
    t0 = time.time()
    N = all_x.shape[0]
    X_time = all_x.reshape(N, -1)

    F = np.fft.fft2(all_x, axes=(2,3))
    amp = np.abs(F).reshape(N, -1)

    phase = np.angle(F)
    phase_cos, phase_sin = np.cos(phase), np.sin(phase)

    def norm(x):
        x_min = x.min(axis=(2,3), keepdims=True)
        x_max = x.max(axis=(2,3), keepdims=True)
        return (x - x_min) / (x_max - x_min + 1e-8)

    amp_n, cos_n, sin_n = norm(np.abs(F)), norm(phase_cos), norm(phase_sin)
    X_freq3 = np.stack([amp_n, cos_n, sin_n], axis=1).reshape(N, -1)

    log(f"✅ 预处理完成，用时 {time.time()-t0:.1f}s")
    return {"time": X_time, "freq": amp, "freq3": X_freq3}

# ========== 降维函数 ========== #
def pca_umap(X, n_pca=30, umap_dim=2):
    X_pca = PCA(n_components=min(n_pca, X.shape[1])).fit_transform(X)
    return umap.UMAP(n_components=umap_dim).fit_transform(X_pca)

def tucker_sample(X, ranks=[1,30,30]):
    log("🔷 Tucker per-sample...")
    t0 = time.time()
    feats = []
    for i in tqdm(range(X.shape[0]), desc="Tucker-per-sample"):
        core, _ = tucker(X[i], ranks=ranks)
        feats.append(core.flatten())
    log(f"✅ Tucker per-sample完成，用时 {time.time()-t0:.1f}s")
    return np.array(feats)

def tucker_overall(X, ranks_sample=50):
    log("🔷 Tucker overall...")
    t0 = time.time()
    core, factors = tucker(X, ranks=[ranks_sample, X.shape[1], 30,30])
    log(f"✅ Tucker overall完成，用时 {time.time()-t0:.1f}s")
    return factors[0]

# ========== 主逻辑 ========== #
all_x, all_meta = load_data()
inputs = prepare_inputs(all_x)

records = []
umap_results = {name:{} for name in inputs.keys()}
methods = ["pca", "tucker1", "tucker2"]
dims = [2,3]

for name, X in inputs.items():
    log(f"=== 开始处理输入: {name} ===")
    t_start = time.time()
    # 方法1
    for dim in dims:
        umap_results[name][f"pca_{dim}d"] = pca_umap(X, umap_dim=dim)

    # 方法2
    features1 = tucker_sample(all_x)  # 注意这里输入是 (N,C,H,W)
    for dim in dims:
        umap_results[name][f"tucker1_{dim}d"] = umap.UMAP(n_components=dim).fit_transform(
            PCA(50).fit_transform(features1))

    # 方法3
    features2 = tucker_overall(all_x)
    for dim in dims:
        umap_results[name][f"tucker2_{dim}d"] = umap.UMAP(n_components=dim).fit_transform(
            PCA(10).fit_transform(features2))
    log(f"✅ {name} 完成，用时 {time.time()-t_start:.1f}s")

# ========== 合并结果 ========== #
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
    records.append(record)

df = pd.DataFrame(records)
df.to_csv("umap_all_methods_results.csv", index=False)
log(f"✅ 所有完成，总用时 {time.time()-t0:.1f}s，结果保存到 umap_all_methods_results.csv")
print(df.head())















from pathlib import Path
import torch
import cupy as cp
import cudf
from cuml.decomposition import PCA
from cuml.manifold import UMAP
import tensorly as tl
from tensorly.decomposition import tucker
from tqdm import tqdm
import time

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tl.set_backend('pytorch')  # 使用PyTorch后端

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ========== GPU加速数据加载 ========== #
def load_data():
    log("🔷 开始加载数据(GPU)...")
    t0 = time.time()
    all_x_list, all_meta_list = [], []
    folder1 = Path(__file__).parent.parent / "generalizability"
    folder2 = Path(__file__).parent.parent
    
    pt_files = list(folder1.glob("*.pt")) + list(folder2.glob("*.pt"))

    for fpath in pt_files:
        data = torch.load(fpath, map_location=device)  # 直接加载到GPU
        x, meta = data["x"], data["metadata"]

        if x.ndim == 2: 
            x = x.unsqueeze(0)
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

    all_x = torch.cat(all_x_list, dim=0)  # 保持GPU张量
    log(f"✅ 加载完成，用时 {time.time()-t0:.1f}s，形状: {all_x.shape}")
    return all_x, all_meta_list

# ========== GPU预处理 ========== #
def prepare_inputs(all_x):
    log("🔷 GPU预处理...")
    t0 = time.time()
    N = all_x.shape[0]
    
    # 时域处理
    X_time = all_x.reshape(N, -1)
    
    # GPU加速FFT
    F = torch.fft.fft2(all_x, dim=(2,3))
    amp = torch.abs(F).reshape(N, -1)
    
    # 相位处理
    phase = torch.angle(F)
    phase_cos, phase_sin = torch.cos(phase), torch.sin(phase)
    
    # GPU归一化
    def norm(x):
        x_min = x.amin(dim=(2,3), keepdim=True)
        x_max = x.amax(dim=(2,3), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-8)
    
    amp_n, cos_n, sin_n = norm(torch.abs(F)), norm(phase_cos), norm(phase_sin)
    X_freq3 = torch.cat([amp_n, cos_n, sin_n], dim=1).reshape(N, -1)
    
    log(f"✅ 预处理完成，用时 {time.time()-t0:.1f}s")
    return {
        "time": X_time.cpu().numpy(),  # cuML需要numpy/cupy输入
        "freq": amp.cpu().numpy(),
        "freq3": X_freq3.cpu().numpy()
    }

# ========== GPU加速降维 ========== #
def pca_umap_gpu(X, n_pca=30, umap_dim=2):
    X_cp = cp.asarray(X)  # 转换为cupy数组
    pca = PCA(n_components=min(n_pca, X.shape[1]))
    umap = UMAP(n_components=umap_dim)
    return umap.fit_transform(pca.fit_transform(X_cp))

def tucker_sample_gpu(X, ranks=[1,30,30]):
    log("🔷 GPU Tucker per-sample...")
    t0 = time.time()
    feats = []
    for i in tqdm(range(X.shape[0]), desc="GPU Tucker"):
        core, _ = tucker(X[i].to(device), ranks=ranks)
        feats.append(core.flatten().cpu())
    log(f"✅ 完成，用时 {time.time()-t0:.1f}s")
    return torch.stack(feats).numpy()

def tucker_overall_gpu(X, ranks_sample=50):
    log("🔷 GPU Tucker overall...")
    t0 = time.time()
    core, factors = tucker(X.to(device), ranks=[ranks_sample, X.shape[1], 30, 30])
    log(f"✅ 完成，用时 {time.time()-t0:.1f}s")
    return factors[0].cpu().numpy()

# ========== 主逻辑 ========== #
if __name__ == "__main__":
    torch.cuda.empty_cache()
    all_x, all_meta = load_data()
    inputs = prepare_inputs(all_x)
    
    records = []
    umap_results = {name: {} for name in inputs.keys()}
    
    for name, X in inputs.items():
        log(f"=== 处理 {name} ===")
        t_start = time.time()
        
        # 方法1: PCA+UMAP
        for dim in [2, 3]:
            umap_results[name][f"pca_{dim}d"] = pca_umap_gpu(X, umap_dim=dim)
        
        # 方法2: 单样本Tucker
        features1 = tucker_sample_gpu(all_x)
        for dim in [2, 3]:
            umap_results[name][f"tucker1_{dim}d"] = pca_umap_gpu(features1, umap_dim=dim)
        
        # 方法3: 整体Tucker
        features2 = tucker_overall_gpu(all_x)
        for dim in [2, 3]:
            umap_results[name][f"tucker2_{dim}d"] = pca_umap_gpu(features2, umap_dim=dim)
        
        log(f"✅ {name} 完成，用时 {time.time()-t_start:.1f}s")
    
    # 合并结果
    log("🔷 合并结果...")
    t0 = time.time()
    for i in tqdm(range(all_x.shape[0]), desc="Merging"):
        record = dict(all_meta[i])
        for name in inputs.keys():
            for method in ["pca", "tucker1", "tucker2"]:
                for dim in [2, 3]:
                    key = f"{method}_{dim}d"
                    coords = umap_results[name][key][i]
                    for j, axis in enumerate("xyz"[:dim]):
                        record[f"{name}_{method}_{axis}_{dim}d"] = float(coords[j])
        records.append(record)
    
    # 使用cudf加速DataFrame构建
    gdf = cudf.DataFrame(records)
    gdf.to_csv("umap_all_methods_results_gpu.csv", index=False)
    log(f"✅ 所有完成，总用时 {time.time()-t0:.1f}s")


    