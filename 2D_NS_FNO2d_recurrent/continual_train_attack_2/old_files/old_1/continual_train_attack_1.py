# ====== 新增 import ======
from PGD_attack_batch import attack_batch, PGDAdamConfig
import torch
from pathlib import Path
from FNO2d import FNO2d, RecurrentPredictor
from utilities3 import *
from tqdm import tqdm                     # FIX: use tqdm from tqdm import tqdm
from timeit import default_timer
from solver import DifferentiablePDESolver

modes = 96
width = 80
epochs = 2000
T_in = 10
T = 10
step = 1

ntrain = 1000
ntest = 100

batch_size = 5
learning_rate = 0.001

current_file_path = Path(__file__)        # ok; .resolve() optional

# ====== 设备与预训练模型路径（按需调整）======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model_name = "NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames"
pretrained_path = current_file_path.parent / f"original_model/{pretrained_model_name}.pth"
dataset_name = "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames"
dataset_path = current_file_path.parent / f"dataset/{dataset_name}.pt"  # NOTE: confirm extension (.pt vs .pth)

# ====== PDE solver（你自己的实现/构造方式）======
pde_solver = DifferentiablePDESolver(nu=1e-5, device=device)

# ====== 攻击配置（可按你之前的缩放习惯设置）======
size = 256
attack_cfg = PGDAdamConfig(
    epsilon = 0.0006 * (size * size),
    alpha   = 1.0,
    num_steps = 100,
    norm = 2,
    mode_spec = "wwwwwwwwww",
    beta1=0.9, beta2=0.999, adam_eps=1e-8, amsgrad=False, use_sign_for_linf=True
)

# ====== 加载底层模型与递归封装，并加载预训练权重 ======
model = FNO2d(modes, modes, width, in_channels=T_in).to(device)
recurrent_model = RecurrentPredictor(model, T_out=T, step=step).to(device)
state = torch.load(pretrained_path, map_location=device)
model.load_state_dict(state)
print(f"[Resume] Loaded pretrained weights from: {pretrained_path}")

model_parameters = count_params(model)
print("params:", model_parameters)

optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
iterations = epochs * (ntrain // batch_size)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
myloss     = LpLoss(size_average=False)

# ====== 路径 & 日志 ======
snap_dir = current_file_path.parent / f"saved_models/modes{modes}_width{width}_epochs{epochs}_Tin{T_in}_T{T}"
snap_dir.mkdir(parents=True, exist_ok=True)
log_save_path = snap_dir / f"{pretrained_model_name}.txt"
log_file = open(log_save_path, "a")
log_file.write("\n\n=== Adversarial-Continued Training ===\n")

data = torch.load(dataset_path, map_location=device, weights_only=False)

# ====== 构造数据集/loader（保持你原有逻辑）======
sub = data["x"].shape[1] // size
x_train = data['y'][:ntrain,::sub,::sub,:T_in].contiguous().to(device)
y_train = data['y'][:ntrain,::sub,::sub,T_in:T+T_in].contiguous().to(device)
x_test  = data['y'][-ntest:,::sub,::sub,:T_in].contiguous().to(device)
y_test  = data['y'][-ntest:,::sub,::sub,T_in:T+T_in].contiguous().to(device)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader  = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=batch_size, shuffle=False, drop_last=False
)

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ====== 辅助：对一个 batch（T_in 帧）逐帧做攻击 ======
def adversarialize_inputs(xx: torch.Tensor) -> torch.Tensor:
    """
    xx: (B, s, s, T_in)
    返回 adv_xx: (B, s, s, T_in)
    逐帧攻击：对每个 t=0..T_in-1 的 (B,H,W) 做 attack_batch。
    """
    B, H, W, Tin = xx.shape
    adv_frames = []
    # 冻结参数 —— 仅对输入求梯度
    for p in recurrent_model.parameters():
        p.requires_grad_(False)

    for t in range(Tin):
        x_t = xx[..., t].contiguous()
        x_adv_t = attack_batch(x_t, recurrent_model, pde_solver, cfg=attack_cfg)
        adv_frames.append(x_adv_t)

    # 恢复可训练
    for p in recurrent_model.parameters():
        p.requires_grad_(True)

    adv_xx = torch.stack(adv_frames, dim=-1)  # (B,H,W,T_in)
    return adv_xx

# ====== 训练循环（每个 batch 先攻击，再训练；测试仍用干净集）======
first_train_batch_cache = None  # 缓存“整个数据集里的第一个 batch”，用于定期日志/快照

for ep in tqdm(range(epochs), desc="Adversarial Continued Training"):
    model.train()
    epoch_t0 = default_timer()

    # NEW: accumulators for timing & loss
    perturb_time = 0.0
    train_time   = 0.0
    eval_time    = 0.0
    train_l2     = 0.0

    # —— 提前缓存第一个 batch（干净，用于日志对比/快照）——
    if first_train_batch_cache is None:
        with torch.no_grad():
            xx0, yy0 = next(iter(train_loader))
            first_train_batch_cache = (xx0.to(device), yy0.to(device))

    # ---- train pass ----
    num_batches = len(train_loader)
    for batch_idx, (xx, yy) in enumerate(train_loader, start=1):
        xx = xx.to(device)  # (B,s,s,T_in)
        yy = yy.to(device)  # (B,s,s,T)

        # 1) 对抗生成 (per-batch)
        _sync()
        t0 = default_timer()
        adv_xx = adversarialize_inputs(xx)
        _sync()
        per_batch_perturb = default_timer() - t0
        perturb_time += per_batch_perturb

        # 2) 训练步 (per-batch)
        _sync()
        t1 = default_timer()
        optimizer.zero_grad(set_to_none=True)
        pred = recurrent_model(adv_xx)                     # (B,s,s,T)
        loss = myloss(pred.reshape(pred.shape[0], -1), yy.reshape(yy.shape[0], -1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        _sync()
        per_batch_train = default_timer() - t1
        train_time += per_batch_train
        train_l2 += float(loss.item())

        # —— 每个 batch 打印 + 记日志 —— 
        msg_batch = (f"[ep {ep:04d} | batch {batch_idx:04d}/{num_batches:04d}] "
                    f"perturb_time:{per_batch_perturb:.3f}s | train_time:{per_batch_train:.3f}s "
                    f"| loss:{float(loss.item()):.6f}")
        print(msg_batch)
        log_file.write(msg_batch + "\n")

    # ---- eval pass (clean test set) ----
    model.eval()
    t2 = default_timer()
    test_l2 = 0.0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            pred = recurrent_model(xx)
            loss = myloss(pred.reshape(pred.shape[0], -1), yy.reshape(yy.shape[0], -1))
            test_l2 += float(loss.item())
    eval_time += (default_timer() - t2)

    # ---- epoch timings ----
    total_time = default_timer() - epoch_t0

    # —— 定期打印：每 50 个 epoch 打印第一个 batch 的 clean/adv loss（仅日志，不更新梯度）——
    if (ep % 500) == 0:
        with torch.no_grad():
            xx0, yy0 = first_train_batch_cache
            # clean
            pred_clean = recurrent_model(xx0)
            clean_loss = myloss(pred_clean.reshape(pred_clean.shape[0], -1),
                                yy0.reshape(yy0.shape[0], -1))
            # adv（用冻结参数生成 adv，再用当前模型评估 loss）
            for p in recurrent_model.parameters(): p.requires_grad_(False)
            adv_xx0 = adversarialize_inputs(xx0)
            for p in recurrent_model.parameters(): p.requires_grad_(True)
            pred_adv0 = recurrent_model(adv_xx0)
            adv_loss0 = myloss(pred_adv0.reshape(pred_adv0.shape[0], -1),
                               yy0.reshape(yy0.shape[0], -1))
        log_file.write(
            f"[ep {ep:04d}] first-batch clean_l2={float(clean_loss):.6f} "
            f"| adv_l2={float(adv_loss0):.6f}\n"
        )

    # —— 定期快照：每 100 个 epoch 保存“第一个 batch 的扰动结果”——
    if (ep % 250) == 0:
        with torch.no_grad():
            xx0, yy0 = first_train_batch_cache
            for p in recurrent_model.parameters(): p.requires_grad_(False)
            adv_xx0 = adversarialize_inputs(xx0)
            for p in recurrent_model.parameters(): p.requires_grad_(True)
        batch_snap_path = snap_dir / f"adv_first_batch_epoch{ep:04d}.pt"
        torch.save(
            {"epoch": ep, "x_clean": xx0.detach().cpu(),
             "x_adv": adv_xx0.detach().cpu(),
             "y": yy0.detach().cpu()},
            batch_snap_path
        )

    # —— 定期保存模型：每 200 个 epoch —— 
    if (ep % 250) == 0 and ep > 0:
        model_snap = snap_dir / f"NS_2d_FNO_model_epoch{ep:04d}.pth"
        torch.save(model.state_dict(), model_snap)

    # —— 逐 epoch 打印/日志 —— 
    msg = (f"epoch:{ep:04d} | perturb_time:{perturb_time:.3f}s "
           f"| train_time:{train_time:.3f}s | eval_time:{eval_time:.3f}s "
           f"| total_time:{total_time:.3f}s | train_l2:{train_l2:.6f} | test_l2:{test_l2:.6f}")
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

# —— 训练结束：保存最终模型 —— 
final_path = snap_dir / f"NS_2d_FNO_model_trainedby_{Path(dataset_name).stem}_final.pth"
torch.save(model.state_dict(), final_path)
log_file.write(f"\n[Done] saved final to {final_path}\n")
log_file.close()
print(f"Model saved to {final_path}")

