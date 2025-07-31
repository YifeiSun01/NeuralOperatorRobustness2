import re
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer
from utilities3 import *
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO2d_2 import FNO2d
from sklearn.model_selection import train_test_split

################################################################
#  configurations
################################################################
ntrain = 4800
ntest = 100

batch_size = 20
learning_rate = 0.001

for epochs in [500]:
    iterations = epochs*(ntrain//batch_size)

    modes = 12
    width = 32
    
    current_file_path = Path(__file__).resolve().parent.parent
    dataset_name = "dim2d_nx256_N5000_solver=exponax_kernel=matern_correlation_length0.50_bcperiodic_nu0.010_t40.0_seed0.pt"
    file_path = current_file_path / "datasets" / "2D" / "NS" / dataset_name
    data = torch.load(file_path, weights_only=False)

    s = 64
    sub = data["x"].shape[1] // s

    x_data = data['x'][:,::sub,::sub]
    y_data = data['y'][:,::sub,::sub]

    x_train_all = x_data[:ntrain,:]
    y_train_all = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]

    x_train_all = x_train_all.reshape(ntrain,s,s,1)
    x_test = x_test.reshape(ntest,s,s,1)
    y_train_all = y_train_all.reshape(ntrain,s,s,1)
    y_test = y_test.reshape(ntest,s,s,1)

    # 划分训练集和验证集 (80%训练，20%验证)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all, y_train_all, test_size=0.2, random_state=42
    )

    # 修改数据加载器
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # 修改2：训练循环中的调度器使用
    best_val_loss = float('inf')
    patience = 20
    no_improve = 0

    # model
    model = FNO2d(modes, modes, width).cuda()
    model_parameters = count_params(model)
    print(model_parameters)

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    file_name = Path(file_path).stem
    log_save_path = current_file_path / f"saved_models/2D/modes{modes}_width{width}_epochs{epochs}/NS_2d_FNO_log_trainedby_{file_name}.txt"
    log_save_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_save_path, "w")
    log_file.write("NS 2d FNO training log\n\n\n\n") 
    log_file.write(f"training and testing dataset: {file_name}\n")
    log_file.write(f"modes: {modes}\n")
    log_file.write(f"width: {width}\n")
    log_file.write(f"model parameters: {model_parameters}\n\n\n\n")

    model_save_path = current_file_path / f"saved_models/2D/modes{modes}_width{width}_epochs{epochs}/NS_2d_FNO_model_trainedby_{file_name}.pth"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    myloss = LpLoss(size_average=False)
    best_val_loss = float('inf')
    patience = 20  # 设置早停耐心值
    no_improve = 0  # 记录未改善的epoch数

    for ep in tqdm(range(epochs), desc="Training FNO 2d"):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        
        # 训练阶段
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x)

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward()  # 使用L2相对损失反向传播
            
            optimizer.step()  # 参数更新
            # 注意：这里移除了 scheduler.step()，因为应该在验证后更新学习率
            
            train_mse += mse.item()
            train_l2 += l2.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                val_loss += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
        
        # 计算平均损失
        train_mse /= len(train_loader)
        train_l2 /= len(train_loader)  # 改为按batch数平均
        val_loss /= len(val_loader)    # 按验证集batch数平均
        
        # 更新学习率调度器（关键修改点）
        scheduler.step(val_loss)  # 使用验证损失更新学习率
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), model_save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {ep}")
                break

        t2 = default_timer()
        elapsed_time = t2 - t1

        # 记录日志（修正了val_loss的标签，原代码错误地标记为test_l2）
        log_file.write(f"epoch:{ep}, time taken:{elapsed_time:.6f}, train mse:{train_mse:.8f}, train l2:{train_l2:.8f}, val l2:{val_loss:.8f}\n")

    log_file.close()

    model_save_path = current_file_path / f"saved_models/2D/modes{modes}_width{width}_epochs{epochs}/NS_2d_FNO_model_trainedby_{file_name}.pth"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")