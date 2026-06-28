import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
import os
from datetime import datetime

subfolder = "nu0.1_mean0"

current_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DeepONet定义
class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim):
        super(DeepONet, self).__init__()
        # branch net: 输入是初始条件 (batch, branch_input_dim)
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        # trunk net: 输入是空间坐标 (batch, trunk_input_dim, 1) 这里 trunk_input_dim=1024
        self.trunk_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
    def forward(self, branch_input, trunk_input):
        # branch_input: (batch_size, branch_input_dim)
        # trunk_input: (batch_size, num_points, 1)
        
        branch_out = self.branch_net(branch_input)  # (batch_size, hidden_dim)
        
        # trunk_input是坐标，逐点计算trunk net输出
        batch_size, num_points, _ = trunk_input.shape
        trunk_out = self.trunk_net(trunk_input.view(-1, 1))  # (batch_size*num_points, hidden_dim)
        trunk_out = trunk_out.view(batch_size, num_points, -1)  # (batch_size, num_points, hidden_dim)
        
        # 计算内积：branch_out与trunk_out的每个点内积，得到预测函数值
        # 先扩展branch_out维度以广播
        branch_out = branch_out.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # 内积
        y_pred = torch.bmm(trunk_out, branch_out.transpose(1, 2)).squeeze(-1)  # (batch_size, num_points)
        
        return y_pred

# 只加载训练集文件，里面有1400条数据
train_data_path = os.path.join(current_dir, f"../datasets/{subfolder}/dim1d_nx1024_N1400_train.pt")
train_data = torch.load(train_data_path)
all_x = train_data["x"]  # [1400, 1024]
all_y = train_data["y"]  # [1400, 1024]

# 划分训练和测试（比如最后100条作为测试）
train_a = all_x[:-100].to(device)   # 前1300条训练
train_u = all_y[:-100].to(device)
test_a = all_x[-100:].to(device)    # 后100条测试
test_u = all_y[-100:].to(device)

# 构建空间坐标网格
grid = torch.linspace(0, 1, steps=1024, device=device).unsqueeze(0).unsqueeze(-1)  # (1,1024,1)
train_grid = grid.repeat(train_a.size(0), 1, 1)
test_grid = grid.repeat(test_a.size(0), 1, 1)

# 模型参数
branch_input_dim = 1024
trunk_input_dim = 1024
hidden_dim = 1024
learning_rate = 1e-3
batch_size = 100
num_epochs = 50000

# 初始化模型
model = DeepONet(branch_input_dim, trunk_input_dim, hidden_dim).to(device)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 创建保存模型的文件夹
model_info = f"DeepONet_b{branch_input_dim}_t{trunk_input_dim}_h{hidden_dim}_lr{learning_rate}_bs{batch_size}_ep{num_epochs}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(current_dir, f"../saved_models/{subfolder}/no_deepxde/{model_info}")

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# 训练
best_test_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(train_a.size(0))
    epoch_loss = 0
    for i in range(0, train_a.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_a = train_a[indices]
        batch_u = train_u[indices]
        batch_grid = grid.repeat(batch_a.size(0), 1, 1)
        
        optimizer.zero_grad()
        output = model(batch_a, batch_grid)  # 预测 (batch, 1024)
        loss = criterion(output, batch_u)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_a.size(0)
    epoch_loss /= train_a.size(0)
    
    # 测试集评估
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_output = model(test_a, test_grid)
            test_loss = criterion(test_output, test_u).item()
            
            # 保存最佳模型
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
                
        print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {epoch_loss:.6f} Test Loss: {test_loss:.6f}")

# 训练循环末尾保存最终模型
model_save_path = os.path.join(save_dir, "final_model.pt")
torch.save(model.state_dict(), model_save_path)
print(f"Models saved in directory: {save_dir}")
