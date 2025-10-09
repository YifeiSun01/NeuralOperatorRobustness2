# === Notebook cell: 批量加载 .pth → 自动选 spatial_hw → 逐层分析（tqdm）→ 区分保存 ===
from __future__ import annotations
import os, re, math, json, time
from time import perf_counter
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

torch.manual_seed(0)
np.random.seed(0)

# ----------------- FNO 定义（与你之前一致） -----------------
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
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        # irfft2 的输出尺寸完全由 s=(H,W) 控制；输入是 rfft2 的一侧谱。:contentReference[oaicite:3]{index=3}
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 1)
        )
    def forward(self, x): return self.mlp(x)

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_layers=4, in_channels=10):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.width, self.num_layers = width, num_layers
        self.p = nn.Linear(in_channels+2, self.width)
        self.conv_layers = nn.ModuleList()
        self.mlp_layers  = nn.ModuleList()
        self.w_layers    = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(SpectralConv2d(self.width, self.width, modes1, modes2))
            self.mlp_layers.append(MLP(self.width, self.width, self.width))
            self.w_layers.append(nn.Conv2d(self.width, self.width, 1))
        self.q = MLP(self.width, 1, self.width * 4)
    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x).permute(0,3,1,2)
        for i in range(self.num_layers):
            x1 = self.mlp_layers[i](self.conv_layers[i](x))
            x2 = self.w_layers[i](x)
            x  = x1 + x2
            if i < self.num_layers - 1: x = F.gelu(x)
        x = self.q(x)
        return x.permute(0,2,3,1)
    def get_grid(self, shape, device):
        b, H, W = shape[0], shape[1], shape[2]
        gx = torch.linspace(0,1,H,device=device).reshape(1,H,1,1).repeat(b,1,W,1)
        gy = torch.linspace(0,1,W,device=device).reshape(1,1,W,1).repeat(b,H,1,1)
        return torch.cat((gx,gy), dim=-1)

class RecurrentPredictor(nn.Module):
    def __init__(self, model, T_out=10, step=1):
        super().__init__()
        self.model, self.T_out, self.step = model, T_out, step
    def forward(self, x_init):
        b, s1, s2, T_in = x_init.shape
        outs, x = [], x_init
        for _ in range(0, self.T_out, self.step):
            y = self.model(x); outs.append(y)
            x = torch.cat([x[..., self.step:], y], dim=-1)
        return torch.cat(outs, dim=-1)

# ----------------- 工具函数：核心提取 / 统计 / 频域→空间核 -----------------
def get_fno_core(m):
    return m.model if hasattr(m, "model") and hasattr(m.model, "conv_layers") else m

@torch.no_grad()
def _svd_stats_from_S(S: torch.Tensor, top_ks=(1,5,10)) -> Dict[str, torch.Tensor]:
    r = S.shape[-1]; S2 = S**2
    fro2 = S2.sum(-1); fro=fro2.sqrt()
    spec = S[...,0]; nuc=S.sum(-1)
    last = S[...,-1].clamp_min(1e-12); cond = spec/last
    tot  = S2.sum(-1, keepdim=True).clamp_min(1e-30)
    p    = S2 / tot; entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
    out = {"Fro":fro,"Spec":spec,"Nuclear":nuc,"Cond":cond,"Entropy":entropy}
    if r>=2:
        s1=S[...,0]; s2=S[...,1].clamp_min(1e-12)
        out["Gap12"]=s1-s2; out["Anisotropy"]=(s1-s2)/s1.clamp_min(1e-12)
    for k in top_ks:
        kk=min(k,r); out[f"ER{k}"]=S2[...,:kk].sum(-1)/tot.squeeze(-1)
    return out

def _reshape_map(vals: torch.Tensor, hw: Tuple[int,int]):
    return vals.view(hw[0], hw[1], *vals.shape[1:]).cpu()

def spectral_weights_to_spatial_kernels(sconv, size_x: int, size_y: int, norm=None) -> torch.Tensor:
    """
    把频域权重（两侧 kx 带）合成一侧谱，再 irfft2 → 空域核，输出 [Cin, Cout, H, W]。
    irfft2 输出尺寸由 s=(H,W) 完全决定。:contentReference[oaicite:4]{index=4}
    """
    m1,m2 = sconv.modes1, sconv.modes2
    cin,cout = sconv.in_channels, sconv.out_channels
    spec = torch.zeros(cin, cout, size_x, size_y//2 + 1, dtype=torch.cfloat, device=sconv.weights1.device)
    spec[:,:, :m1,:m2]  = sconv.weights1
    spec[:,:, -m1:,:m2] = sconv.weights2
    return torch.fft.irfft2(spec, s=(size_x, size_y), norm=norm)

def _svd_top2(A: torch.Tensor):
    """返回 Top-2 左/右奇异向量与奇异值：U2:[...,cout,2], S2:[...,2], V2:[...,cin,2]。奇异值降序。:contentReference[oaicite:5]{index=5}"""
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U[..., :, :2], S[..., :2], Vh[..., :2, :].conj().mT

def _principal_angle_field(vecs_all: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    计算与参考向量的主角（批量）：cosθ = |<x, x0>| / (||x|| ||x0||) ∈ [0,1]，θ∈[0,π/2]。:contentReference[oaicite:6]{index=6}
    vecs_all: [N, dim] 或 [N, dim, K]；ref: [dim] 或 [dim, K]
    返回：若输入 [N,dim] -> [N]；若 [N,dim,K] -> [N,K]
    """
    eps = 1e-12
    if vecs_all.ndim == 2:  # [N,dim]
        vn = vecs_all / vecs_all.norm(dim=1, keepdim=True).clamp_min(eps)
        r  = ref / ref.norm(dim=0, keepdim=False).clamp_min(eps)      # [dim]
        cos = (vn.conj() * r).sum(dim=1).abs().clamp(0, 1)
        return torch.arccos(cos)
    elif vecs_all.ndim == 3:  # [N,dim,K]
        vn = vecs_all / vecs_all.norm(dim=1, keepdim=True).clamp_min(eps)
        r  = ref / ref.norm(dim=0, keepdim=False).clamp_min(eps)      # [dim,K]
        cos = (vn.conj() * r).sum(dim=1).abs().clamp(0, 1)            # [N,K]
        return torch.arccos(cos)
    else:
        raise ValueError("vecs_all must be [N,dim] or [N,dim,K]")

# ------ 条目抽取（含负索引） ------
# 取超集（含你两版提到的坐标）
ENTRY_IDXS = [(0,0),(10,0),(0,10),(10,10),(30,30),(30,0),(0,30),(-10,-10),(-10,0),(0,-10)]

def _wrap_idx(idx: int, dim: int) -> Optional[int]:
    j = idx if idx >= 0 else dim + idx
    return j if (0 <= j < dim) else None

def _extract_entries_batch(B: torch.Tensor, idxs, hw: Tuple[int,int]) -> Dict[str, torch.Tensor]:
    """
    从批量矩阵 B:[N, cout, cin] 抽取若干 (r,c) 条目，映射成 [H,W]（或 [m1,m2]）。
    越界填 NaN。支持复数。
    """
    N, cout, cin = B.shape
    out = {}
    for (r, c) in idxs:
        rr = _wrap_idx(r, cout); cc = _wrap_idx(c, cin)
        if rr is None or cc is None:
            vals = torch.full((N,), float('nan'), dtype=B.dtype, device=B.device)
        else:
            vals = B[:, rr, cc]  # [N]
        out[f"A[{r},{c}]"] = _reshape_map(vals, hw)
    return out

# ----------------- 单层/整模分析（含 tqdm） -----------------
@torch.no_grad()
def analyze_spectral_layer(
    sconv,
    spatial_hw: Tuple[int,int],
    device: torch.device,
    top_ks=(1,5,10),
    spatial_chunk: int = 4096,
    keep_top_sv: int = 8,
    spatial_dirs: bool = True,
    store_vectors: bool = False,   # 默认 False，避免巨量存储；如需可设 True
    polar_stats: bool = True,
    layer_index: int = 0,
    n_layers: int = 1,
    on_info = print,
    pbar_factory = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    m1,m2 = sconv.modes1, sconv.modes2
    cin,cout = sconv.in_channels, sconv.out_channels

    # ---------- 频域统计（+kx / -kx） ----------
    def _freq_block_stats(w: torch.Tensor):
        """
        频域一个权重块（weights1 或 weights2）：
         - 原有 SVD 统计（TopSingVals 等）
         - 新增：抽取 ENTRY_IDXS 指定条目（freq Entries）
         - 新增：u1/v1/u2/v2 相对 (0,0) 参考点的主角场（freq Angles）
        """
        # 组批： [m1*m2, cout, cin]
        Bf = w.permute(2, 3, 1, 0).reshape(m1 * m2, cout, cin).contiguous().to(device)

        # (1) 奇异值统计
        S = torch.linalg.svdvals(Bf)                                               # [m1*m2, r]（降序）:contentReference[oaicite:7]{index=7}
        stats = _svd_stats_from_S(S, top_ks=top_ks)
        out = {k: _reshape_map(v, (m1, m2)) for k, v in stats.items()}
        out["TopSingVals"] = _reshape_map(S[:, :min(keep_top_sv, S.shape[-1])], (m1, m2))

        # (2) 条目抽取（频域）
        out["Entries"] = _extract_entries_batch(Bf, ENTRY_IDXS, (m1, m2))

        # (3) 与 (0,0) 参考的主角场（Top-2）
        U2f, S2f, V2f = _svd_top2(Bf)                                              # U2:[N,cout,2], V2:[N,cin,2]
        uref = U2f[0]       # [cout,2]，(0,0) 参考
        vref = V2f[0]       # [cin, 2]
        angU = _principal_angle_field(U2f, uref)                                   # [N,2]
        angV = _principal_angle_field(V2f, vref)                                   # [N,2]
        out["Angles"] = {
            "U1_to_ref": _reshape_map(angU[:, 0], (m1, m2)),
            "U2_to_ref": _reshape_map(angU[:, 1], (m1, m2)),
            "V1_to_ref": _reshape_map(angV[:, 0], (m1, m2)),
            "V2_to_ref": _reshape_map(angV[:, 1], (m1, m2)),
        }
        # （如需保存向量本体，可在此加 out["U_top2_freq"] 等键）
        return out

    on_info(f"[Layer {layer_index+1}/{n_layers}] freq (+kx) ..."); t0=perf_counter()
    freq_pos = _freq_block_stats(sconv.weights1); on_info(f"[+] done {perf_counter()-t0:.2f}s")
    on_info(f"[Layer {layer_index+1}/{n_layers}] freq (-kx) ..."); t0=perf_counter()
    freq_neg = _freq_block_stats(sconv.weights2); on_info(f"[-] done {perf_counter()-t0:.2f}s")

    # ---------- 时域核与统计 ----------
    H,W = spatial_hw
    total_mats = H*W
    on_info(f"[Layer {layer_index+1}/{n_layers}] spatial HxW={H}x{W} => {total_mats} mats")
    kernels = spectral_weights_to_spatial_kernels(sconv, H, W).to(device)          # [Cin,Cout,H,W]
    B = kernels.permute(2,3,1,0).reshape(total_mats, cout, cin).contiguous()       # [N, cout, cin]

    # svdvals（分块）
    S_list=[]; chunk=spatial_chunk; n_chunks=(total_mats+chunk-1)//chunk
    pbar = pbar_factory(total=n_chunks, desc=f"[Layer {layer_index+1}] spatial SVDvals")
    for i0 in range(0,total_mats,chunk):
        i1=min(total_mats,i0+chunk); ts=perf_counter()
        S_chunk = torch.linalg.svdvals(B[i0:i1])                                    # 降序:contentReference[oaicite:8]{index=8}
        S_list.append(S_chunk); te=perf_counter()
        pbar.set_postfix({"chunk":f"{i0//chunk+1}/{n_chunks}","Δt(s)":f"{te-ts:.2f}"}); pbar.update(1)
    pbar.close()
    S_all=torch.cat(S_list,0)
    sstats = _svd_stats_from_S(S_all, top_ks=top_ks)
    spatial_stats = {k:_reshape_map(v,(H,W)) for k,v in sstats.items()}
    spatial_stats["TopSingVals"] = _reshape_map(S_all[:, :min(keep_top_sv, S_all.shape[-1])], (H,W))

    extra={}
    # ---- 时域：条目抽取（一次性） ----
    extra["Entries"] = _extract_entries_batch(B, ENTRY_IDXS, (H, W))

    # ---- 时域：Top-2 与 (0,0) 参考的主角场（分块以省显存） ----
    uref, vref = None, None
    # 先算参考向量（(0,0) -> 扁平索引 0）
    U2_ref, S2_ref, V2_ref = _svd_top2(B[0:1])            # [1,cout,2], [1,2], [1,cin,2]
    uref = U2_ref[0]                                       # [cout,2]
    vref = V2_ref[0]                                       # [cin, 2]

    angU_chunks=[]; angV_chunks=[]
    pbar = pbar_factory(total=n_chunks, desc=f"[Layer {layer_index+1}] spatial Angles vs (0,0)")
    for i0 in range(0,total_mats,chunk):
        i1=min(total_mats,i0+chunk); ts=perf_counter()
        U2_c, S2_c, V2_c = _svd_top2(B[i0:i1])            # [m,cout,2], [m,2], [m,cin,2]
        angU_c = _principal_angle_field(U2_c, uref)       # [m,2]
        angV_c = _principal_angle_field(V2_c, vref)       # [m,2]
        angU_chunks.append(angU_c); angV_chunks.append(angV_c)
        te=perf_counter(); pbar.set_postfix({"chunk":f"{i0//chunk+1}/{n_chunks}","Δt(s)":f"{te-ts:.2f}"}); pbar.update(1)
    pbar.close()
    angU = torch.cat(angU_chunks, 0)                      # [N,2]
    angV = torch.cat(angV_chunks, 0)                      # [N,2]

    extra.setdefault("Angles", {})
    extra["Angles"].update({
        "U1_to_ref": _reshape_map(angU[:, 0], (H, W)),
        "U2_to_ref": _reshape_map(angU[:, 1], (H, W)),
        "V1_to_ref": _reshape_map(angV[:, 0], (H, W)),
        "V2_to_ref": _reshape_map(angV[:, 1], (H, W)),
    })

    # ---- 可选：保留你原来的方向角（仅 2 通道才有意义） ----
    if spatial_dirs:
        if cout==2:
            # 用 atan2 得“方位角”，再 mod π（方向等价），与“主角”不同，保留兼容
            def _angles_from_vec2(v: torch.Tensor):
                theta = torch.atan2(v[...,1], v[...,0]); return (theta % math.pi)
            # 为了避免再做一次全量 SVD，这里从已经计算过的 U2_ref/分块 U2_c 里取第一列（u1）
            # 简便起见：再分块重算 u1（开销与上面一致数量级）
            u1_all=[]
            pbar = pbar_factory(total=n_chunks, desc=f"[Layer {layer_index+1}] spatial U1 angle(atan2)")
            for i0 in range(0,total_mats,chunk):
                i1=min(total_mats,i0+chunk)
                U2_c, _, _ = _svd_top2(B[i0:i1])
                u1_all.append(U2_c[..., :, 0])
                pbar.update(1)
            pbar.close()
            u1_all = torch.cat(u1_all, 0)
            extra["U1_Angle"] = _reshape_map(_angles_from_vec2(u1_all), (H, W))
        if cin==2:
            v1_all=[]
            pbar = pbar_factory(total=n_chunks, desc=f"[Layer {layer_index+1}] spatial V1 angle(atan2)")
            for i0 in range(0,total_mats,chunk):
                i1=min(total_mats,i0+chunk)
                _, _, V2_c = _svd_top2(B[i0:i1])
                v1_all.append(V2_c[..., :, 0])
                pbar.update(1)
            pbar.close()
            v1_all = torch.cat(v1_all, 0)
            def _angles_from_vec2(v: torch.Tensor):
                theta = torch.atan2(v[...,1], v[...,0]); return (theta % math.pi)
            extra["V1_Angle"] = _reshape_map(_angles_from_vec2(v1_all), (H, W))

    # ---- 如需把向量本体也落盘（量很大），可打开下列开关 ----
    if store_vectors:
        # 注意：这些张量可能非常大（H*W*channels*2）
        # 这里仅保存 Top-2 的 U/V 与对应奇异值
        # 再分块拿一次（避免存全部中间结果）
        U2_store=[]; V2_store=[]; S2_store=[]
        pbar = pbar_factory(total=n_chunks, desc=f"[Layer {layer_index+1}] store U/V top2")
        for i0 in range(0,total_mats,chunk):
            i1=min(total_mats,i0+chunk)
            U2_c, S2_c, V2_c = _svd_top2(B[i0:i1])
            U2_store.append(U2_c.cpu()); V2_store.append(V2_c.cpu()); S2_store.append(S2_c.cpu())
            pbar.update(1)
        pbar.close()
        U2_all = torch.cat(U2_store, 0); V2_all = torch.cat(V2_store, 0); S2_all = torch.cat(S2_store, 0)
        extra["U_top2"] = _reshape_map(U2_all, (H, W))    # [H,W,cout,2]
        extra["V_top2"] = _reshape_map(V2_all, (H, W))    # [H,W,cin,2]
        extra["S_top2"] = _reshape_map(S2_all, (H, W))    # [H,W,2]

    out = {"freq_pos":freq_pos, "freq_neg":freq_neg, "spatial":spatial_stats,
           "meta":{"cin":cin,"cout":cout,"modes1":m1,"modes2":m2,
                   "spatial_hw":list(spatial_hw),"top_ks":list(top_ks),"keep_top_sv":keep_top_sv,
                   "spatial_dirs":spatial_dirs,"store_vectors":store_vectors,"polar_stats":polar_stats}}
    out["spatial_extra"]=extra
    return out

@torch.no_grad()
def analyze_fno_model(
    model,
    spatial_hw: Tuple[int,int],
    out_path: str,
    device: str | None = None,
    top_ks=(1,5,10),
    spatial_chunk: int = 4096,
    keep_top_sv: int = 8,
    spatial_dirs: bool = True,
    store_vectors: bool = False,
    polar_stats: bool = True,
    on_info = print,
    pbar_factory = None,
):
    core = get_fno_core(model).eval()
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {"model_meta":{"width":getattr(core,"width",None),
                             "num_layers":len(core.conv_layers),
                             "modes1":core.modes1, "modes2":core.modes2},
               "layers":{}}

    t0 = perf_counter(); n_layers = len(core.conv_layers)
    for li, sconv in enumerate(core.conv_layers):
        sconv = sconv.to(dev)
        if dev.type=="cuda": torch.cuda.synchronize()
        lt0 = perf_counter()
        layer_res = analyze_spectral_layer(
            sconv, spatial_hw, dev, top_ks=top_ks, spatial_chunk=spatial_chunk,
            keep_top_sv=keep_top_sv, spatial_dirs=spatial_dirs, store_vectors=store_vectors,
            polar_stats=polar_stats, layer_index=li, n_layers=n_layers, on_info=on_info, pbar_factory=pbar_factory
        )
        results["layers"][str(li)] = layer_res
        sconv = sconv.cpu(); torch.cuda.empty_cache()
        on_info(f"[Layer {li+1}/{n_layers}] done in {perf_counter()-lt0:.2f}s")

    if dev.type=="cuda": torch.cuda.synchronize()
    results["elapsed_sec"] = perf_counter() - t0
    torch.save(results, out_path)
    on_info(f"[SAVE] {out_path}  (elapsed {results['elapsed_sec']:.2f}s)")
    return out_path, results["elapsed_sec"]

# ----------------- (A) 自动选择 spatial_hw -----------------
def _parse_hw_from_path(p: Path) -> tuple[Optional[int], Optional[int]]:
    s = p.as_posix()
    nx = re.search(r"nx(\d+)", s)
    ny = re.search(r"ny(\d+)", s)
    return (int(nx.group(1)) if nx else None, int(ny.group(1)) if ny else None)

def pick_spatial_hw_for_model(model, pth_path: Path, fallback_hw=(256,256)) -> tuple[int,int]:
    """
    从 modes1/modes2 推断最小一致尺寸：
      H_min = 2*modes1
      W_min = 2*(modes2 - 1)   （rfft2 的最后一维是 W//2+1）:contentReference[oaicite:9]{index=9}
    若 fallback 更大，则取更大者。
    """
    core = get_fno_core(model)
    if hasattr(core,"modes1") and hasattr(core,"modes2"):
        m1, m2 = int(core.modes1), int(core.modes2)
        H_min = 2 * m1
        W_min = 2 * max(m2 - 1, 1)
        H = max(H_min, fallback_hw[0])
        W = max(W_min, fallback_hw[1])
        return (H, W)
    return fallback_hw

# ----------------- (B) 智能模型加载：整模型 / state_dict 都支持 -----------------
def _strip_prefix(sd: Dict[str, torch.Tensor], prefixes=("module.", "model.")) -> Dict[str, torch.Tensor]:
    out={}
    for k,v in sd.items():
        kk=k
        for pre in prefixes:
            if kk.startswith(pre): kk=kk[len(pre):]
        out[kk]=v
    return out

def _infer_fno_hparams_from_state_dict(sd: Dict[str, torch.Tensor]):
    sd = _strip_prefix(sd)
    # modes1/modes2：找第一层 weights1 的后两维
    w1 = None
    for k,v in sd.items():
        if k.startswith("conv_layers.0.") and k.endswith("weights1"):
            w1=v; break
    if w1 is None:
        for k,v in sd.items():
            if k.endswith("weights1"): w1=v; break
    if w1 is None:
        raise RuntimeError("无法从 state_dict 推断 modes（未找到 weights1）")
    m1, m2 = int(w1.shape[-2]), int(w1.shape[-1])

    # width：优先从 1x1 卷积的 out_channels
    width = None
    for k,v in sd.items():
        if k.startswith("w_layers.0.weight"):
            width = int(v.shape[0]); break
    if width is None:
        width = int(w1.shape[0])  # in_channels
    # in_channels：从线性 p.weight 推断（[width, in_ch+2]）
    in_ch = None
    for k,v in sd.items():
        if k=="p.weight":
            in_ch = int(v.shape[1])-2; break
    if in_ch is None: in_ch = 10

    # num_layers：统计 conv_layers.<idx>.weights1 的 idx 个数
    idxs=set()
    for k in sd.keys():
        m=re.match(r"conv_layers\.(\d+)\.weights1", k)
        if m: idxs.add(int(m.group(1)))
    num_layers = (max(idxs)+1) if idxs else 4

    return dict(modes1=m1, modes2=m2, width=width, num_layers=num_layers, in_channels=in_ch)

def smart_model_loader(pth_path: Path):
    try:
        obj = torch.load(pth_path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(pth_path, map_location="cpu")

    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("model", None), nn.Module):
        return obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd_raw = obj["state_dict"]
        sd = {k: v for k,v in sd_raw.items() if isinstance(v, torch.Tensor)}
        hp = _infer_fno_hparams_from_state_dict(sd)
        model = FNO2d(**hp)
        sd_norm = _strip_prefix(sd)
        model.load_state_dict(sd_norm, strict=False)
        return model
    if isinstance(obj, dict) and all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in obj.items()):
        sd = obj
        hp = _infer_fno_hparams_from_state_dict(sd)
        model = FNO2d(**hp)
        sd_norm = _strip_prefix(sd)
        model.load_state_dict(sd_norm, strict=False)
        return model
    raise RuntimeError(f"无法识别的模型文件格式：{pth_path}")

# ----------------- 批量：遍历 .pth → 加载模型 → 自动 spatial_hw → 分析 → 区分命名保存 -----------------
def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

def find_pth_models(root: str | Path) -> List[Path]:
    return sorted(Path(root).rglob("*.pth"))

def run_batch_analysis(
    root_dir: str | Path,
    out_dir: str | Path,
    *,
    force_hw: Optional[Tuple[int,int]] = None,
    fallback_hw: Tuple[int,int] = (256,256),
    device=None,
    spatial_chunk=4096,
    keep_top_sv=8,
    spatial_dirs=True,
    store_vectors=False,
    polar_stats=True,
    top_ks=(1,5,10),
    ignore_patterns: Optional[List[str]] = None,
    model_loader=smart_model_loader,
):
    root_dir = Path(root_dir)
    out_dir  = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    all_pths = find_pth_models(root_dir)
    if ignore_patterns:
        kept=[]
        for p in all_pths:
            rel=p.relative_to(root_dir).as_posix()
            if any(re.fullmatch(fn.replace("**",".*").replace("*","[^/]*"), rel) for fn in ignore_patterns):
                continue
            kept.append(p)
        all_pths=kept

    model_pbar = tqdm(all_pths, desc="[Models]", unit="model")
    for pth in model_pbar:
        subdir_name = pth.parent.name
        base = _safe_name(subdir_name)
        out_file = out_dir / f"{base}.analysis.pt"
        if out_file.exists():
            out_file = out_dir / f"{base}__{_safe_name(pth.stem)}.analysis.pt"

        model_pbar.set_postfix({"subdir": subdir_name, "out": out_file.name})
        t0 = perf_counter()
        print(f"\n==== Model ====\n{pth}\nSave -> {out_file}")

        # 加载
        model = model_loader(pth)

        # 决定该模型的 (H,W)
        if force_hw is not None:
            spatial_hw_model = force_hw
        else:
            spatial_hw_model = pick_spatial_hw_for_model(model, pth, fallback_hw=fallback_hw)

        def pbar_factory(total:int, desc:str): return tqdm(total=total, desc=desc, leave=False)
        def on_info(msg:str): print(msg)

        out_path, secs = analyze_fno_model(
            model,
            spatial_hw=spatial_hw_model,
            out_path=str(out_file),
            device=device,
            top_ks=top_ks,
            spatial_chunk=spatial_chunk,
            keep_top_sv=keep_top_sv,
            spatial_dirs=spatial_dirs,
            store_vectors=store_vectors,
            polar_stats=polar_stats,
            on_info=on_info,
            pbar_factory=pbar_factory,
        )
        print(f"[DONE] {pth.name}  elapsed {perf_counter()-t0:.2f}s  → {out_path}")

    print("\nAll models processed.")

# ----------------- 使用示例 -----------------
root_dir = "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/external_forcing_patterns/saved_models"
out_dir  = "/blue/shiboli.fsu/yifeisun.umich/adversarial_robustness_FNO/2D_NS_FNO2d_recurrent/saved_models_kernel_abstracts/other_forcing_patterns/kernel_results/"
run_batch_analysis(
    root_dir=root_dir,
    out_dir=out_dir,
    force_hw=None,           # 想固定网格就填 (H,W)，否则自动解析（由 modes 推断）
    fallback_hw=(256,256),   # 仅当 modes 给出的最小尺寸更小才会使用更大的 fallback
    device=None,             # "cuda" 或 None(自动)
    spatial_chunk=4096,      # 调大/调小可权衡速度/显存
    keep_top_sv=8,
    spatial_dirs=True,
    store_vectors=True,     # 若要把 U/V top-2 全量落盘，设 True（注意体积）
    polar_stats=True,
)



