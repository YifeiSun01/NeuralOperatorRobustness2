import re
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer
from utilities3 import *
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO3d import FNO3d
import torch

################################################################
#  configurations
################################################################
ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 0.001

modes12 = 32
modes3 = 8
width = 40
s = 256

T_in = 10
T = 10
step = 1

print("Training FNO3d")

print(f"{'Variable':<15} {'Value':<10}")
print("-" * 25)
print(f"{'ntrain':<15} {ntrain:<10}")
print(f"{'ntest':<15} {ntest:<10}")
print(f"{'batch_size':<15} {batch_size:<10}")
print(f"{'learning_rate':<15} {learning_rate:<10}")
print(f"{'modes12':<15} {modes12:<10}")
print(f"{'modes3':<15} {modes3:<10}")
print(f"{'width':<15} {width:<10}")
print(f"{'s':<15} {s:<10}")
print(f"{'T_in':<15} {T_in:<10}")
print(f"{'T':<15} {T:<10}")
print(f"{'step':<15} {step:<10}")

for epochs in [500]:
    iterations = epochs*(ntrain//batch_size)
    
    current_file_path = Path(__file__).resolve().parent.parent
    dataset_name = "dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pt"
    file_path = current_file_path.parent / "2D_NS_FNO2d_recurrent" / "datasets" / "exponax_datasets" / "t20" / dataset_name
    data = torch.load(file_path, weights_only=False)

    sub = data["x"].shape[1] // s

    print(f"{'iterations':<15} {iterations:<10}")
    print(f"{'sub':<15} {sub:<10}")

    x_train = data['y'][:ntrain,::sub,::sub,:T_in]
    y_train = data['y'][:ntrain,::sub,::sub,T_in:T+T_in]
    x_test = data['y'][-ntest:,::sub,::sub,:T_in]
    y_test = data['y'][-ntest:,::sub,::sub,T_in:T+T_in]

    x_train = x_train.reshape(ntrain,s,s,1,T_in).repeat([1,1,1,T,1])
    x_test = x_test.reshape(ntest,s,s,1,T_in).repeat([1,1,1,T,1])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # 初始化底层模型和递归预测器
    model = FNO3d(modes12, modes12, modes3, width).cuda()
    model_parameters = count_params(model)
    print(model_parameters)

    # 优化器只需要优化 FNO2d 的参数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    file_name = Path(file_path).stem
    log_save_path = current_file_path / f"saved_models/3D/modes12{modes12}_modes3{modes3}_width{width}_epochs{epochs}_Tin{T_in}_T{T}/NS_3d_FNO_log_trainedby_{file_name}.txt"
    log_save_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_save_path, "w")
    log_file.write("NS 3d FNO training log\n\n\n\n") 
    log_file.write(f"training and testing dataset: {file_name}\n")
    log_file.write(f"modes1 and modes2: {modes12}\n")
    log_file.write(f"modes3: {modes3}\n")
    log_file.write(f"width: {width}\n")
    log_file.write(f"model parameters: {model_parameters}\n\n\n\n")

    myloss = LpLoss(size_average=False)
    # for ep in range(epochs):
    for ep in tqdm(range(epochs), desc="Training FNO 3d Recurrent"):
        model.train()
        t1 = default_timer()
        train_l2 = 0

        for xx, yy in train_loader:
            xx = xx.to(device)  # shape (batch, s, s, 10)
            yy = yy.to(device)  # shape (batch, s, s, 10)

            pred = model(xx)  # shape (batch, s, s, 10)
            loss = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_l2 += loss.item()

        test_l2 = 0
        with torch.no_grad():
            model.eval()
            for xx, yy in test_loader:
                xx = xx.to(device)
                yy = yy.to(device)

                pred = model(xx)
                loss = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
                test_l2 += loss.item()

        t2 = default_timer()
        elapsed_time = t2 - t1
        log_file.write(f"epoch:{ep}, time taken:{elapsed_time:.6f}, train l2:{train_l2:.8f}, test l2:{test_l2:.8f}\n")

    log_file.close()

    model_save_path = current_file_path / f"saved_models/3D/modes12{modes12}_modes3{modes3}_width{width}_epochs{epochs}_Tin{T_in}_T{T}/NS_3d_FNO_model_trainedby_{file_name}.pth"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")