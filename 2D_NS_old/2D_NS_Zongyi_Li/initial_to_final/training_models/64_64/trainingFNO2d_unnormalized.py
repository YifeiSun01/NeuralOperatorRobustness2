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

################################################################
#  configurations
################################################################
ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 0.001

for epochs in [500]:
    iterations = epochs*(ntrain//batch_size)

    modes = 12
    width = 16
    
    current_file_path = Path(__file__).resolve().parent.parent
    dataset_name = "NS_data_zongyi_train.pt"
    file_path = current_file_path / "datasets" / "2D" / "NS" / "input_frame_19" / dataset_name
    data = torch.load(file_path, weights_only=False)

    s = 64
    sub = data["x"].shape[1] // s

    x_data = data['x'][:,::sub,::sub]
    y_data = data['y'][:,::sub,::sub]

    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]

    x_train = x_train.reshape(ntrain,s,s,1)
    x_test = x_test.reshape(ntest,s,s,1)
    y_train = y_train.reshape(ntrain,s,s,1)
    y_test = y_test.reshape(ntest,s,s,1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

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
    log_save_path = current_file_path / f"saved_models/2D/modes{modes}_width{width}_epochs{epochs}/input_frame_19/NS_2d_FNO_log_trainedby_{file_name}.txt"
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
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x).reshape(batch_size, s, s)

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() # use the l2 relative loss

            optimizer.step()
            scheduler.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x).reshape(batch_size, s, s)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        elapsed_time = t2 - t1

        # print(ep, elapsed_time, train_mse, train_l2, test_l2)
        log_file.write(f"epoch:{ep}, time taken:{elapsed_time:.6f}, train mse:{train_mse:.8f}, train l2:{train_l2:.8f}, test l2:{test_l2:.8f}\n")

    log_file.close()

    model_save_path = current_file_path / f"saved_models/2D/modes{modes}_width{width}_epochs{epochs}/input_frame_19/NS_2d_FNO_model_trainedby_{file_name}.pth"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")