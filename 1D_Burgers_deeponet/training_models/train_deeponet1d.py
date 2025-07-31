import deepxde as dde
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add the current file's directory to sys.path
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))

# 获取当前工作目录（运行脚本时的终端路径）
print("当前工作目录 (cwd):", os.getcwd())

# 获取脚本所在的绝对目录
script_dir = Path(__file__).parent.resolve()
print("脚本所在目录:", script_dir)

# 检查文件是否存在
target_file = Path("../datasets/dim1d_nx1024_N1400_train.pt").resolve()
print("目标文件路径:", target_file)
print("文件是否存在:", target_file.exists())

# Load dataset
data_path = Path(__file__).parent.parent / "datasets" / "dim1d_nx1024_N1400_train.pt"
d = torch.load(data_path, weights_only=False)

# Convert tensors to float32 using .to() instead of .astype()
X_train = (
    d["x"][:1300].to(torch.float32), 
    torch.linspace(0, 1, d["x"].shape[1]).unsqueeze(1).to(torch.float32)
)
y_train = d["y"][:1300].to(torch.float32)
X_test = (
    d["x"][1300:].to(torch.float32), 
    torch.linspace(0, 1, d["x"].shape[1]).unsqueeze(1).to(torch.float32)
)
y_test = d["y"][1300:].to(torch.float32)

data = dde.data.TripleCartesianProd(
    X_train=X_train, 
    y_train=y_train, 
    X_test=X_test, 
    y_test=y_test
)

# Choose a network
m = d["x"].shape[1]
dim_x = 1
# branch_hidden_layer_dim = [640,320,160,80,40,20,20,40,80,160,320,640]
# trunk_hidden_layer_dim = [640,320,160,80,40,20,20,40,80,160,320,640]
# branch_hidden_layer_dim = [640,320,160,160,320,640]
# trunk_hidden_layer_dim = [640,320,160,160,320,640]
# branch_hidden_layer_dim = [640,320,320,640]
# trunk_hidden_layer_dim = [640,320,320,640]
# branch_hidden_layer_dim = [640,640]
# trunk_hidden_layer_dim = [640,640]
# branch_hidden_layer_dim = [400,200,100,200,400]
# trunk_hidden_layer_dim = [400,200,100,200,400]
branch_hidden_layer_dim = [80,40,20,20,40,80]
trunk_hidden_layer_dim = [80,40,20,20,40,80]
net = dde.nn.DeepONetCartesianProd(
    [m, *branch_hidden_layer_dim],  # Unpacks branch dimensions
    [dim_x, *trunk_hidden_layer_dim],  # Unpacks trunk dimensions
    "relu",
    "Glorot normal",
)

# Define a Model
model = dde.Model(data, net)

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

branch_hidden_layer_dim = [str(item) for item in branch_hidden_layer_dim]
trunk_hidden_layer_dim = [str(item) for item in trunk_hidden_layer_dim]

print(f"Branch net sizes: {', '.join(branch_hidden_layer_dim)}")
print(f"Trunk net sizes: {', '.join(trunk_hidden_layer_dim)}")

print(f"Branch net layers: {len(net.branch.linears)}")
print(f"Trunk net layers: {len(net.trunk.linears)}")

n_params = count_parameters(net)
print(f"Total trainable parameters: {n_params:,}")

branch_params = sum(p.numel() for p in net.branch.parameters() if p.requires_grad)
trunk_params = sum(p.numel() for p in net.trunk.parameters() if p.requires_grad)

print(f"Branch net params: {branch_params:,}")
print(f"Trunk net params: {trunk_params:,}")
print(f"Total trainable parameters: {branch_params + trunk_params:,}")

try:
    train_samples = model.data.train_x[0].shape[0]
    print(f"Number of training samples: {train_samples}")
except AttributeError as e:
    print(f"Error accessing training data shape: {e}")
    print("Please check if 'model.data.train_x' is properly initialized")

try:
    batch_size = model.data.batch_size
    print(f"Current batch size: {batch_size}")
    if batch_size is None:
        print("Note: Batch size is None, which means full-batch training is being used")
        print(f"This implies each iteration processes all {train_samples} training samples")
except AttributeError as e:
    print(f"Error accessing batch size: {e}")
    print("The data loader may not have batch_size attribute")

# Compile and Train
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
iterations = 500
losshistory, train_state = model.train(iterations=iterations)

# Plot the loss trajectory
# de.utils.plot_loss_history(losshistory)
# plt.show()

name_text = "branch" + "_".join(branch_hidden_layer_dim) + "trunk" + "_".join(trunk_hidden_layer_dim) + f"_iter{iterations}"
os.makedirs(Path(__file__).parent.parent / "saved_models" / name_text, exist_ok=True)
model_path = Path(__file__).parent.parent / "saved_models" / name_text / f"trained_model_{name_text}.pt"
model.save(model_path)

log_path = Path(__file__).parent.parent / "saved_models" / name_text / f"loss_history_{name_text}.txt"

# Save loss history to a .txt file
np.savetxt(
    log_path,
    np.column_stack((
        losshistory.steps,
        losshistory.loss_train,
        losshistory.loss_test,
        losshistory.metrics_test,
    )),
    header="step, train_loss, test_loss, test_metric",
    comments="",
    fmt="%d, %.6e, %.6e, %.6e",  # Format: integer steps, scientific notation for losses
)



