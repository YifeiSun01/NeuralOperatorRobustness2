from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm
import tensorly as tl
from tensorly.decomposition import tucker

# ========== 读取数据 ==========
all_x_list = []
all_meta_list = []

folder1 = Path(__file__).parent.parent / "generalizability"
folder2 = Path(__file__).parent.parent

generalizability_pt_files = [f.resolve() for f in folder1.glob("*.pt")]
train_test_pt_files = [f.resolve() for f in folder2.glob("*.pt")]

pt_files = generalizability_pt_files + train_test_pt_files

print(f"\nTotal {len(pt_files)} .pt files found.\n")

for fpath in pt_files:
    print(f"Loading {fpath}")
    data = torch.load(fpath)

    x = data["x"]
    meta = data["metadata"]

    if fpath in generalizability_pt_files:
        data_type = "generalizability"
    elif "train" in fpath.name.lower():
        data_type = "train"
    elif "test" in fpath.name.lower():
        data_type = "test"
    else:
        data_type = "unknown"

    if x.ndim == 2:
        x = x.unsqueeze(0)

    all_x_list.append(x)

    if isinstance(meta, list):
        for m in meta:
            m["data_type"] = data_type
        all_meta_list.extend(meta)
    else:
        meta["data_type"] = data_type
        all_meta_list.extend([meta] * x.shape[0])

all_x = torch.cat(all_x_list, dim=0).numpy()
N = all_x.shape[0]
print(f"\nAll data shape: {all_x.shape}")
assert len(all_meta_list) == N

# ========== 定义PCA+UMAP函数 ==========
def pca_umap(X, n_pca=30, n_neighbors=15, min_dist=0.1, random_state=42, n_components=2):
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X)
    um = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, n_components=n_components)
    X_umap = um.fit_transform(X_pca)
    return X_umap

# ========== 1. 时域方法 ==========
print("\n[1] Time domain flatten + PCA + UMAP (2D) ...")
X_time = all_x.reshape(N, -1)
umap_time = pca_umap(X_time, n_components=2)

print("\n[1-3D] Time domain flatten + PCA + UMAP (3D) ...")
umap_time_3d = pca_umap(X_time, n_components=3)

# ========== 2. 频域方法（幅值） ==========
print("\n[2] Frequency domain amplitude only + PCA + UMAP (2D) ...")
F = np.fft.fft2(all_x, axes=(1, 2))
amplitude = np.abs(F)
X_freq_amp = amplitude.reshape(N, -1)
umap_freq = pca_umap(X_freq_amp, n_components=2)

print("\n[2-3D] Frequency domain amplitude only + PCA + UMAP (3D) ...")
umap_freq_3d = pca_umap(X_freq_amp, n_components=3)

# ========== 3. 频域三通道幅值+cos相位+sin相位 ==========
print("\n[3] Frequency domain amplitude+cos(phase)+sin(phase) + normalization + PCA + UMAP (2D) ...")

phase = np.angle(F)
phase_cos = np.cos(phase)
phase_sin = np.sin(phase)

def min_max_normalize_channel(X):
    X_min = X.reshape(X.shape[0], -1).min(axis=1, keepdims=True)
    X_max = X.reshape(X.shape[0], -1).max(axis=1, keepdims=True)
    return ((X.reshape(X.shape[0], -1) - X_min) / (X_max - X_min + 1e-8)).reshape(X.shape)

amplitude_norm = min_max_normalize_channel(amplitude)
phase_cos_norm = min_max_normalize_channel(phase_cos)
phase_sin_norm = min_max_normalize_channel(phase_sin)

X_freq_three_channel = np.stack([amplitude_norm, phase_cos_norm, phase_sin_norm], axis=1)
X_freq_three_channel = X_freq_three_channel.reshape(N, -1)
umap_freq_three = pca_umap(X_freq_three_channel, n_components=2)

print("\n[3-3D] Frequency domain amplitude+cos(phase)+sin(phase) + normalization + PCA + UMAP (3D) ...")
umap_freq_three_3d = pca_umap(X_freq_three_channel, n_components=3)

# ========== 4. 方法1：单样本 Tucker + PCA + UMAP ==========
print("\n[4] Per-sample Tucker + PCA + UMAP ...")

def reduce_sample_tucker(x_sample, ranks=[3,30,30]):
    core, factors = tucker(x_sample, ranks=ranks)
    return core.flatten()

def reduce_batch_tucker(X, ranks=[3,30,30]):
    features = []
    for i in tqdm(range(X.shape[0]), desc="Per-sample Tucker"):
        f = reduce_sample_tucker(X[i], ranks)
        features.append(f)
    return np.array(features)

# 对三种输入分别处理
# 4.1 原始时域数据 (N,1,256,256)
print("\n[4.1] Time domain Tucker ...")
tucker_time_features = reduce_batch_tucker(all_x)
umap_tucker_time_2d = pca_umap(tucker_time_features, n_components=2)
umap_tucker_time_3d = pca_umap(tucker_time_features, n_components=3)

# 4.2 频域幅值数据 (N,1,256,256)
print("\n[4.2] Frequency amplitude Tucker ...")
tucker_freq_amp_features = reduce_batch_tucker(amplitude)
umap_tucker_freq_amp_2d = pca_umap(tucker_freq_amp_features, n_components=2)
umap_tucker_freq_amp_3d = pca_umap(tucker_freq_amp_features, n_components=3)

# 4.3 频域三通道数据 (N,3,256,256)
print("\n[4.3] Frequency three-channel Tucker ...")
tucker_freq_three_features = reduce_batch_tucker(np.stack([amplitude_norm, phase_cos_norm, phase_sin_norm], axis=1))
umap_tucker_freq_three_2d = pca_umap(tucker_freq_three_features, n_components=2)
umap_tucker_freq_three_3d = pca_umap(tucker_freq_three_features, n_components=3)

# ========== 5. 方法2：整体4阶张量 Tucker + PCA + UMAP ==========
print("\n[5] Batch Tucker + PCA + UMAP ...")

def batch_tucker(X, ranks=[50, None, 30, 30]):
    # Set the sample dimension rank automatically if None
    if ranks[1] is None:
        ranks[1] = X.shape[1]
    core, factors = tucker(X, ranks=ranks)
    sample_factors = factors[0]  # (N, ranks[0])
    return sample_factors

# 对三种输入分别处理
# 5.1 原始时域数据 (N,1,256,256)
print("\n[5.1] Time domain batch Tucker ...")
batch_tucker_time_features = batch_tucker(all_x, ranks=[50, 1, 30, 30])
umap_batch_tucker_time_2d = pca_umap(batch_tucker_time_features, n_pca=10, n_components=2)
umap_batch_tucker_time_3d = pca_umap(batch_tucker_time_features, n_pca=10, n_components=3)

# 5.2 频域幅值数据 (N,1,256,256)
print("\n[5.2] Frequency amplitude batch Tucker ...")
batch_tucker_freq_amp_features = batch_tucker(amplitude, ranks=[50, 1, 30, 30])
umap_batch_tucker_freq_amp_2d = pca_umap(batch_tucker_freq_amp_features, n_pca=10, n_components=2)
umap_batch_tucker_freq_amp_3d = pca_umap(batch_tucker_freq_amp_features, n_pca=10, n_components=3)

# 5.3 频域三通道数据 (N,3,256,256)
print("\n[5.3] Frequency three-channel batch Tucker ...")
batch_tucker_freq_three_features = batch_tucker(np.stack([amplitude_norm, phase_cos_norm, phase_sin_norm], axis=1), ranks=[50, 3, 30, 30])
umap_batch_tucker_freq_three_2d = pca_umap(batch_tucker_freq_three_features, n_pca=10, n_components=2)
umap_batch_tucker_freq_three_3d = pca_umap(batch_tucker_freq_three_features, n_pca=10, n_components=3)

# ========== 合并所有结果和metadata ==========
print("\n[Merging results and metadata...]")
records = []
for i in tqdm(range(N), desc="Creating records"):
    record = {
        # 原始3种方法的2D和3D结果 (6个)
        "time_x": umap_time[i, 0],
        "time_y": umap_time[i, 1],
        "time_x_3d": umap_time_3d[i, 0],
        "time_y_3d": umap_time_3d[i, 1],
        "time_z_3d": umap_time_3d[i, 2],

        "freq_x": umap_freq[i, 0],
        "freq_y": umap_freq[i, 1],
        "freq_x_3d": umap_freq_3d[i, 0],
        "freq_y_3d": umap_freq_3d[i, 1],
        "freq_z_3d": umap_freq_3d[i, 2],

        "freq3_x": umap_freq_three[i, 0],
        "freq3_y": umap_freq_three[i, 1],
        "freq3_x_3d": umap_freq_three_3d[i, 0],
        "freq3_y_3d": umap_freq_three_3d[i, 1],
        "freq3_z_3d": umap_freq_three_3d[i, 2],

        # 方法1: 单样本Tucker (3输入×2D/3D = 6个)
        "tucker_time_x": umap_tucker_time_2d[i, 0],
        "tucker_time_y": umap_tucker_time_2d[i, 1],
        "tucker_time_x_3d": umap_tucker_time_3d[i, 0],
        "tucker_time_y_3d": umap_tucker_time_3d[i, 1],
        "tucker_time_z_3d": umap_tucker_time_3d[i, 2],

        "tucker_freq_amp_x": umap_tucker_freq_amp_2d[i, 0],
        "tucker_freq_amp_y": umap_tucker_freq_amp_2d[i, 1],
        "tucker_freq_amp_x_3d": umap_tucker_freq_amp_3d[i, 0],
        "tucker_freq_amp_y_3d": umap_tucker_freq_amp_3d[i, 1],
        "tucker_freq_amp_z_3d": umap_tucker_freq_amp_3d[i, 2],

        "tucker_freq_three_x": umap_tucker_freq_three_2d[i, 0],
        "tucker_freq_three_y": umap_tucker_freq_three_2d[i, 1],
        "tucker_freq_three_x_3d": umap_tucker_freq_three_3d[i, 0],
        "tucker_freq_three_y_3d": umap_tucker_freq_three_3d[i, 1],
        "tucker_freq_three_z_3d": umap_tucker_freq_three_3d[i, 2],

        # 方法2: 批量Tucker (3输入×2D/3D = 6个)
        "batch_tucker_time_x": umap_batch_tucker_time_2d[i, 0],
        "batch_tucker_time_y": umap_batch_tucker_time_2d[i, 1],
        "batch_tucker_time_x_3d": umap_batch_tucker_time_3d[i, 0],
        "batch_tucker_time_y_3d": umap_batch_tucker_time_3d[i, 1],
        "batch_tucker_time_z_3d": umap_batch_tucker_time_3d[i, 2],

        "batch_tucker_freq_amp_x": umap_batch_tucker_freq_amp_2d[i, 0],
        "batch_tucker_freq_amp_y": umap_batch_tucker_freq_amp_2d[i, 1],
        "batch_tucker_freq_amp_x_3d": umap_batch_tucker_freq_amp_3d[i, 0],
        "batch_tucker_freq_amp_y_3d": umap_batch_tucker_freq_amp_3d[i, 1],
        "batch_tucker_freq_amp_z_3d": umap_batch_tucker_freq_amp_3d[i, 2],

        "batch_tucker_freq_three_x": umap_batch_tucker_freq_three_2d[i, 0],
        "batch_tucker_freq_three_y": umap_batch_tucker_freq_three_2d[i, 1],
        "batch_tucker_freq_three_x_3d": umap_batch_tucker_freq_three_3d[i, 0],
        "batch_tucker_freq_three_y_3d": umap_batch_tucker_freq_three_3d[i, 1],
        "batch_tucker_freq_three_z_3d": umap_batch_tucker_freq_three_3d[i, 2],
    }
    record.update(all_meta_list[i])
    records.append(record)

df = pd.DataFrame(records)

# ========== 保存结果 ==========
output_path = "umap_all_methods_results_2d_3d.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Saved to {output_path}")
print("\nColumns in the output DataFrame:")
print(df.columns.tolist())