import re
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer
from utilities3 import *
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FNO2d import FNO2d
from torch.cuda.amp import autocast, GradScaler

################################################################
#  configurations
################################################################
ntrain = 1400
ntest = 100

batch_size = 10
learning_rate = 0.001

for epochs in [500]:
    iterations = epochs*(ntrain//batch_size)

    modes = 128
    width = 64
    
    current_file_path = Path(__file__).resolve().parent.parent
    dataset_name = "dim2d_nx256_N1500_solver=exponax_kernel=matern_correlation_length0.50_bcperiodic_nu0.010_t40.0_seed0_all_frames.pt"
    file_path = current_file_path / "datasets" / "2D" / "NS" / dataset_name
    data = torch.load(file_path, weights_only=False)

    s = 256
    sub = data["x"].shape[1] // s
    T_in = 10
    T = 10
    step = 1
    nsamples = data["x"].shape[0]

    x_train = data['y'][...,::2][:ntrain,::sub,::sub,:T_in]
    y_train = data['y'][...,::2][:ntrain,::sub,::sub,T_in:T+T_in]
    x_test = data['y'][...,::2][-ntest:,::sub,::sub,:T_in]
    y_test = data['y'][...,::2][-ntest:,::sub,::sub,T_in:T+T_in]

    x_train = x_train.reshape(ntrain,s,s,T_in)
    x_test = x_test.reshape(ntest,s,s,T_in)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, pin_memory=False)

    # model
    model = FNO2d(modes, modes, width).cuda()
    model_parameters = count_params(model)
    print(model_parameters)

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)\

    file_name = Path(file_path).stem
    log_save_path = current_file_path / f"saved_models/2D/modes{modes}_width{width}_epochs{epochs}_T{T}_Tin{T_in}_trainN{nsamples}/NS_2d_FNO_log_trainedby_{file_name}.txt"
    log_save_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_save_path, "w")
    log_file.write("NS 2d FNO training log\n\n\n\n") 
    log_file.write(f"training and testing dataset: {file_name}\n")
    log_file.write(f"modes: {modes}\n")
    log_file.write(f"width: {width}\n")
    log_file.write(f"model parameters: {model_parameters}\n\n\n\n")

    myloss = LpLoss(size_average=False)
    # for ep in range(epochs):
    for ep in tqdm(range(epochs), desc="Training FNO 2d"):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0

        for xx, yy in train_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            train_l2_step += loss.item()
            l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            train_l2_full += l2_full.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        test_l2_step = 0
        test_l2_full = 0
        with torch.no_grad():
            for xx, yy in test_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)

                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    im = model(xx)
                    loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., step:], im), dim=-1)

                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

        t2 = default_timer()
        elapsed_time = t2 - t1

        # print(ep, elapsed_time, train_mse, train_l2, test_l2)
        log_file.write(f"epoch:{ep}, time taken:{elapsed_time:.6f}, train l2_full:{train_l2_full:.8f}, test l2_full:{test_l2_full:.8f}\n")

    log_file.close()

    model_save_path = current_file_path / f"saved_models/2D/modes{modes}_width{width}_epochs{epochs}_T{T}_Tin{T_in}_trainN{nsamples}/NS_2d_FNO_model_trainedby_{file_name}.pth"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")