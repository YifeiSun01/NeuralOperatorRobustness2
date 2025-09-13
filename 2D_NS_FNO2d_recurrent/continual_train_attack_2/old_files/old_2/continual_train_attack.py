# ====== 新增 import ======
from PGD_attack_batch import attack_and_rollout_seq, PGDAdamConfig
import torch
from pathlib import Path
from FNO2d import FNO2d, RecurrentPredictor
from utilities3 import *
from tqdm import tqdm
from timeit import default_timer
from solver import DifferentiablePDESolver
import jax

jax.config.update("jax_compilation_cache_dir", "/blue/shiboli.fsu/yifeisun.umich/.jax_cache")
# 可选：允许小函数也入缓存（按需）
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# 非本地文件系统时，JAX 文档建议安装 etils: pip install etils

modes = 96
width = 80
epochs = 2000
T_in = 10
T = 10
step = 1

ntrain = 1000
ntest = 100

batch_size = 2
learning_rate = 0.001

current_file_path = Path(__file__)

# ====== 设备与预训练模型路径（按需调整）======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model_name = "NS_2d_FNO_model_trainedby_dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames"
pretrained_path = current_file_path.parent / f"original_model/{pretrained_model_name}.pth"
dataset_name = "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames"
dataset_path = current_file_path.parent / f"dataset/{dataset_name}.pt"

# ====== PDE solver ======
pde_solver = DifferentiablePDESolver(nu=1e-5, device=device)

# ====== 攻击配置（与你之前缩放一致）======
size = 256
attack_cfg = PGDAdamConfig(
    epsilon = 0.0080 * (size * size),
    alpha   = 100.0,
    num_steps = 1,
    norm = 2,
    mode_spec = "wwwwwwwwww",
    beta1=0.9, beta2=0.999, adam_eps=1e-8, amsgrad=False, use_sign_for_linf=True
)

# ====== 打印所有内容 ======
print("=" * 40 + " 参数配置 " + "=" * 40)
print(f"modes={modes}, width={width}, epochs={epochs}")
print(f"T_in={T_in}, T={T}, step={step}")
print(f"ntrain={ntrain}, ntest={ntest}")
print(f"batch_size={batch_size}, learning_rate={learning_rate}")
print()

print("=" * 40 + " 路径配置 " + "=" * 40)
print(f"current_file_path={current_file_path}")
print(f"pretrained_model_name={pretrained_model_name}")
print(f"pretrained_path={pretrained_path}")
print(f"dataset_name={dataset_name}")
print(f"dataset_path={dataset_path}")
print()

print("=" * 40 + " 设备与PDE solver " + "=" * 40)
print(f"device={device}")
print(f"pde_solver={pde_solver}")
print()

print("=" * 40 + " 攻击配置 " + "=" * 40)
print(attack_cfg)


# ====== 加载模型 ======
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
snap_dir = current_file_path.parent / f"saved_models/modes{modes}_width{width}_epochs{epochs}_Tin{T_in}_T{T}_batch{batch_size}_2"
snap_dir.mkdir(parents=True, exist_ok=True)
log_save_path = snap_dir / f"{pretrained_model_name}.txt"
log_file = open(log_save_path, "a")
log_file.write("\n\n=== Adversarial-Continued Training (attack first frame + PDE labels) ===\n")

data = torch.load(dataset_path, map_location=device, weights_only=False)

# ====== 构造数据集/loader ======
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

# ====== 训练循环（每个 batch：只攻第0帧 + PDE 生成 Tin/Tout）======
first_train_batch_cache = None  # 缓存“干净 batch”用于日志/快照（会用新 API 生成 adv 与标签）

# ---- 预热（无PGD、无backward）----
with torch.no_grad():
    # 1) 触发 JAX jit + cuFFT 计划（形状与训练一致）
    dummy = torch.zeros((batch_size, size, size), device=device, dtype=torch.float32)
    _ = pde_solver.rollout_seconds(dummy, T_seconds=T_in + T - 1)

    # 2) 触发一次 FNO+Recurrent 的卷积/前向内核加载
    dummy_xx = torch.zeros((batch_size, size, size, T_in), device=device, dtype=torch.float32)
    _ = recurrent_model(dummy_xx)

for ep in tqdm(range(epochs), desc="Adversarial Continued Training"):
    model.train()
    epoch_t0 = default_timer()

    perturb_time = 0.0
    train_time   = 0.0
    eval_time    = 0.0
    train_l2     = 0.0

    # 缓存第一个 batch（干净，用于周期性日志/快照；注意：评估时用新 API 产出对齐标签）
    if first_train_batch_cache is None:
        with torch.no_grad():
            xx0, yy0 = next(iter(train_loader))
            first_train_batch_cache = (xx0.to(device), yy0.to(device))

    # ---- train pass ----
    num_batches = len(train_loader)
    for batch_idx, (xx, yy) in enumerate(train_loader, start=1):
        xx = xx.to(device)  # (B,s,s,T_in)
        # yy (loader 给的) 不再用作训练标签，我们用 PDE rollout 的 yy

        # 1) 对抗 + PDE 标签
        _sync()
        t0 = default_timer()
        adv_xx, yy_pde = attack_and_rollout_seq(xx, recurrent_model, pde_solver, Tout=T, cfg=attack_cfg)
        _sync()
        per_batch_perturb = default_timer() - t0
        perturb_time += per_batch_perturb

        # 2) 训练步
        _sync()
        t1 = default_timer()
        optimizer.zero_grad(set_to_none=True)
        pred = recurrent_model(adv_xx)                     # (B,s,s,T)
        loss = myloss(pred.reshape(pred.shape[0], -1), yy_pde.reshape(yy_pde.shape[0], -1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        _sync()
        per_batch_train = default_timer() - t1
        train_time += per_batch_train
        train_l2 += float(loss.item())

        msg_batch = (f"[ep {ep:04d} | batch {batch_idx:04d}/{num_batches:04d}] "
                     f"perturb_time:{per_batch_perturb:.3f}s | train_time:{per_batch_train:.3f}s "
                     f"| loss:{float(loss.item()):.6f}")
        print(msg_batch)
        log_file.write(msg_batch + "\n")

    # ---- eval pass (clean set; 仍然用干净 xx / 干净标签 y_test 评估基准泛化) ----
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

    total_time = default_timer() - epoch_t0

    if (ep % 10) == 0:
        xx0, _ = first_train_batch_cache

        # 1) 先在开启梯度的上下文里生成对抗序列（PGD里要反传到 x_adv）
        with torch.enable_grad():
            adv_xx0, yy_adv0 = attack_and_rollout_seq(
                xx0, recurrent_model, pde_solver, Tout=T, cfg=attack_cfg
            )

        # 2) 再在 no_grad 里做纯推理与日志/快照，省显存/更快
        with torch.no_grad():
            pred_clean = recurrent_model(xx0)
            clean_loss = myloss(pred_clean.reshape(pred_clean.shape[0], -1),
                                y_train[:xx0.shape[0]].reshape(xx0.shape[0], -1))

            pred_adv0 = recurrent_model(adv_xx0)
            adv_loss0 = myloss(pred_adv0.reshape(pred_adv0.shape[0], -1),
                            yy_adv0.reshape(yy_adv0.shape[0], -1))

            log_file.write(
                f"[ep {ep:04d}] first-batch clean_l2={float(clean_loss):.6f} "
                f"| adv_l2={float(adv_loss0):.6f}\n"
            )
            torch.save(
                {"epoch": ep,
                "x_adv": adv_xx0.detach().cpu(),
                "y": yy_adv0.detach().cpu()},
                snap_dir / f"adv_first_batch_epoch{ep:04d}.pt"
            )

    # —— 周期性存模型 —— 
    if (ep % 10) == 0 and ep > 0:
        model_snap = snap_dir / f"NS_2d_FNO_model_epoch{ep:04d}.pth"
        torch.save(model.state_dict(), model_snap)

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


