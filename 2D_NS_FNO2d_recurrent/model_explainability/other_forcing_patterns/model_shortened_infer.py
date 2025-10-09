import os
import re
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ---------------- 模型定义（与你之前一致） ---------------- #
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
        # rfft2/irfft2 的频域形状与“一侧频谱”约定：W -> W//2+1
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(x.shape[0], self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
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

        self.p = nn.Linear(in_channels + 2, self.width)
        self.conv_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.w_layers   = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(SpectralConv2d(self.width, self.width, modes1, modes2))
            self.mlp_layers.append(MLP(self.width, self.width, self.width))
            self.w_layers.append(nn.Conv2d(self.width, self.width, 1))
        self.q = MLP(self.width, 1, self.width * 4)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device).reshape(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float, device=device).reshape(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        for i in range(self.num_layers):
            x1 = self.conv_layers[i](x)
            x1 = self.mlp_layers[i](x1)
            x2 = self.w_layers[i](x)
            x  = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)
        x = self.q(x)
        return x.permute(0, 2, 3, 1)

# ---------------- 工具函数 ---------------- #
def strip_prefix(sd, prefixes=("module.", "model.", "model.module.")):
    out = OrderedDict()
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
                break
        out[kk] = v
    return out

def infer_arch_hparams(sd):
    width = sd[[k for k in sd if k.startswith("w_layers.0.weight")][0]].shape[0]
    cand = [k for k in sd if re.search(r"conv_layers\.0\.weights1$", k)]
    if not cand:
        raise KeyError("Cannot find conv_layers.0.weights1 in state_dict.")
    modes1, modes2 = sd[cand[0]].shape[-2:]
    Tin = sd["p.weight"].shape[1] - 2
    layer_idxs = [int(re.findall(r"conv_layers\.(\d+)\.", k)[0]) for k in sd if k.startswith("conv_layers.")]
    num_layers = max(layer_idxs) + 1
    return modes1, modes2, width, num_layers, Tin

def build_variant_from_core(core: FNO2d, n_keep: int) -> FNO2d:
    assert 0 <= n_keep <= core.num_layers
    variant = FNO2d(core.modes1, core.modes2, core.width, num_layers=n_keep, in_channels=core.p.in_features - 2)
    variant.p.load_state_dict(core.p.state_dict())
    variant.q.load_state_dict(core.q.state_dict())
    for i in range(n_keep):
        variant.conv_layers[i].load_state_dict(core.conv_layers[i].state_dict())
        variant.mlp_layers[i].load_state_dict(core.mlp_layers[i].state_dict())
        variant.w_layers[i].load_state_dict(core.w_layers[i].state_dict())
    return variant

@torch.no_grad()
def predict_one_step(model: FNO2d, x_in: torch.Tensor, device: torch.device, batch_size: int = 4, desc: str = ""):
    model.eval().to(device)
    N = x_in.shape[0]
    outs = []
    total_batches = (N + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc=desc, unit="batch")
    for i in range(0, N, batch_size):
        xb = x_in[i:i+batch_size].to(device, non_blocking=True)
        yb = model(xb)
        outs.append(yb.cpu())
        pbar.update(1)
    pbar.close()
    return torch.cat(outs, dim=0)

def load_dataset(dataset_path: Path, s_target: int = 256, T_in: int = 10):
    data = torch.load(dataset_path, weights_only=False)
    Y = data['y']  # 形状假设: (N, S, S, T_total)
    S = Y.shape[1]
    sub = S // s_target
    if S % s_target != 0:
        raise ValueError(f"Dataset spatial size {S} not divisible by {s_target}.")
    x_all = Y[:, ::sub, ::sub, :T_in].contiguous()
    return x_all

def save_stacked(out_tensor: torch.Tensor, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_tensor, out_path)
    print(f"[SAVE] tensor saved to: {out_path}")

def find_ckpt_for_pattern(model_root: Path, pattern: str) -> Path:
    # 路径结构：<root>/<pattern>/modes64_width60_epochs500_Tin10_T10/NS_2d_FNO_trained_on_..._train_forcingPattern<pattern>.pth
    cand = list((model_root / pattern).rglob(f"*train_forcingPattern{pattern}.pth"))
    if not cand:
        raise FileNotFoundError(f"No checkpoint for pattern={pattern} under {model_root}")
    # 取最深层/最近修改的一个
    cand.sort(key=lambda p: (len(p.parts), p.stat().st_mtime))
    return cand[-1]

def dataset_for_pattern(dataset_root: Path, pattern: str) -> Path:
    # 路径结构：<root>/<pattern>/dim2d_nx256_N1150_solver=..._train_forcingPattern<pattern>.pt
    cand = list((dataset_root / pattern).glob(f"dim2d_nx256_N1150_solver=*_train_forcingPattern{pattern}.pt"))
    if not cand:
        raise FileNotFoundError(f"No train-1150 dataset for pattern={pattern} under {dataset_root}")
    return cand[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", type=str,
                        default="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/external_forcing_patterns/saved_models",
                        help="root directory containing model subfolders by pattern")
    parser.add_argument("--dataset_root", type=str,
                        default="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/external_forcing_patterns/datasets/all_patterns",
                        help="root directory containing datasets by pattern")
    parser.add_argument("--out_dir", type=str,
                        default="/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/model_explainability/other_forcing_patterns/infer_results",
                        help="where to save stacked prediction tensors")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_keep_layers", type=int, default=4)
    args = parser.parse_args()

    model_root = Path(args.model_root)
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    batch_size = args.batch_size
    max_keep = args.max_keep_layers

    # 只循环这 6 个 pattern（与目录一致）
    patterns = ["isoCircles", "petals", "ringsCos", "ringsL1", "ringsLinf", "sBands"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[SETUP] device={device}, batch_size={batch_size}, max_keep={max_keep}")
    print(f"[ROOTS] model_root={model_root}")
    print(f"[ROOTS] dataset_root={dataset_root}")
    print(f"[OUT]   out_dir={out_dir}")

    for pattern in patterns:
        print("\n" + "="*80)
        print(f"[PAIR] pattern = {pattern}")

        # --- 绑定：一个模型 ↔ 一个 N=1150 训练集 --- #
        ds_path = dataset_for_pattern(dataset_root, pattern)
        ckpt_path = find_ckpt_for_pattern(model_root, pattern)

        print(f"[DATASET] {ds_path}")
        print(f"[CKPT]    {ckpt_path}")

        # 加载数据（只用 T_in=10）
        x_all = load_dataset(ds_path, s_target=256, T_in=10)
        N, S1, S2, T_in = x_all.shape
        print(f"[DATA] shape={tuple(x_all.shape)}")

        # 加载模型
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, OrderedDict):
                state_dict = ckpt
            elif isinstance(ckpt, dict):
                for k in ["state_dict", "model_state_dict", "model", "net", "module"]:
                    if k in ckpt and isinstance(ckpt[k], (dict, OrderedDict)):
                        state_dict = ckpt[k]; break
                else:
                    raise ValueError(f"Unexpected checkpoint keys: {list(ckpt.keys())[:10]}")
            else:
                raise TypeError(f"Unexpected ckpt type: {type(ckpt)}")
            state_dict = strip_prefix(state_dict)
            modes1, modes2, width, num_layers, Tin = infer_arch_hparams(state_dict)
            print(f"[MODEL] inferred: modes1={modes1}, modes2={modes2}, width={width}, num_layers={num_layers}, Tin={Tin}")

            core = FNO2d(modes1, modes2, width, num_layers=num_layers, in_channels=Tin)
            missing, unexpected = core.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[WARN] missing keys: {len(missing)} (first few): {missing[:5]}")
            if unexpected:
                print(f"[WARN] unexpected keys: {len(unexpected)} (first few): {unexpected[:5]}")

            # 依次生成 keep=0..max_keep 的变体并推理
            variant_preds = []
            for n_keep in range(0, min(max_keep, num_layers) + 1):
                desc = f"{pattern} | keep{n_keep}"
                variant = build_variant_from_core(core, n_keep=n_keep)
                y_pred = predict_one_step(variant, x_all, device=device, batch_size=batch_size, desc=desc)
                variant_preds.append(y_pred)

            # 若 max_keep 超过实际层数，尾部复制最后一个
            while len(variant_preds) < (max_keep + 1):
                variant_preds.append(variant_preds[-1].clone())

            stacked = torch.cat(variant_preds, dim=-1)  # (N, S1, S2, max_keep+1)

            # 保存到 pattern 子目录
            sub_out = out_dir / pattern
            sub_out.mkdir(parents=True, exist_ok=True)
            model_tag = ckpt_path.parent.name  # e.g., modes64_width60_epochs500_Tin10_T10
            save_name = f"{model_tag}__{pattern}__variants_keep0to{max_keep}__pred11.pt"
            out_path = sub_out / save_name
            save_stacked(stacked, out_path)

        except Exception as e:
            print(f"[ERROR] pattern={pattern} failed:\n{e}")
            print("Continue to next pattern.")
            continue

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    np.random.seed(0)
    torch.manual_seed(0)
    main()

