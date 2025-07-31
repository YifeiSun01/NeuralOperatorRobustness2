import deepxde as dde
import numpy as np
from deepxde.backend import tf
from scipy import io
from sklearn.preprocessing import StandardScaler
import os
import torch
from datetime import datetime
import joblib

# subfolder = "nu0.1_mean0"

for subfolder in ["nu0.1","nu0.01","nu0.001","nu0.05","nu0.005"]:
    def periodic(x):
        x = x * (2 * np.pi)
        return torch.cat(
            [torch.cos(x), torch.sin(x), torch.cos(2 * x), torch.sin(2 * x)], dim=1
        )

    def get_data(ntrain, ntest):
        sub_x = 1
        sub_y = 1

        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        train_data_path = os.path.join(current_dir, f"../datasets/{subfolder}/dim1d_nx1024_N1400_train.pt")
        data = torch.load(train_data_path)
        x_data = data["x"][:, ::sub_x].to(torch.float32)  # [N, 1024]
        y_data = data["y"][:, ::sub_y].to(torch.float32)  # [N, 1024]

        x_branch_train = x_data[:ntrain]  # [ntrain, 1024]
        y_train = y_data[:ntrain]         # [ntrain, 1024]
        x_branch_test = x_data[-ntest:]  # [ntest, 1024]
        y_test = y_data[-ntest:]         # [ntest, 1024]

        grid = np.linspace(0, 1, x_data.shape[1], dtype=np.float32)[:, None]  # [1024, 1]

        # Return non-expanded (original) data
        x_train = (x_branch_train.numpy(), grid)
        x_test = (x_branch_test.numpy(), grid)
        return x_train, y_train.numpy(), x_test, y_test.numpy()


    def train(model, lr, epochs):
        decay = ("inverse time", epochs // 5, 0.5)
        model.compile("adam", lr=lr, metrics=["mean l2 relative error"], decay=decay)
        losshistory, train_state = model.train(epochs=epochs, batch_size=None)
        # dde.postprocessing.save_loss_history(losshistory, "loss.dat")
        print("\nTraining done ...\n")

    def main():
        # 1. 数据加载
        x_train, y_train, x_test, y_test = get_data(1000, 200)

        # 2. 标准化处理
        scaler = StandardScaler().fit(y_train)
        std = np.sqrt(scaler.var_.astype(np.float32))
        mean = scaler.mean_.astype(np.float32)

        def output_transform(inputs, outputs):
            device = outputs.device
            std_t = torch.tensor(std, device=device)
            mean_t = torch.tensor(mean, device=device)
            return outputs * std_t + mean_t

        # 3. 模型初始化
        m = 2 ** 10  # 1024
        net = dde.maps.DeepONetCartesianProd(
            [m, 128, 128, 128, 128], 
            [4, 128, 128, 128], 
            "tanh", 
            "Glorot normal"
        )
        net.apply_feature_transform(periodic)
        net.apply_output_transform(output_transform)

        # 4. 准备训练数据
        data = dde.data.TripleCartesianProd(
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test
        )
        model = dde.Model(data, net)

        # 5. 训练配置
        lr = 0.001
        epochs = 500000
        
        # 6. 训练前创建临时目录（用于保存训练日志等）
        # temp_dir = "temp_training_logs"
        # os.makedirs(temp_dir, exist_ok=True)

        # 7. 执行训练
        train(model, lr, epochs)

        # 8. 训练完成后保存最终模型和标准化器
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"saved_models/{subfolder}/deepxde_2/DeepONet_b{m}_lr{lr}_ep{epochs}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        model.save(os.path.join(save_dir, "model.pt"))
        # 保存标准化器
        joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
        
        print(f"训练完成！模型和标准化器已保存到: {save_dir}")

    if __name__ == "__main__":
        main()