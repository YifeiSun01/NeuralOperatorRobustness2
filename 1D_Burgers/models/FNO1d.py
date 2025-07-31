import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, dtype=torch.float64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        # self.dtype = torch.float64
        
        # 根据dtype确定复数类型
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 
                                  dtype=complex_dtype))
        
    def compl_mul1d(self, input, weights):
        # 自动类型匹配
        return torch.einsum("bix,iox->box", input, weights.type_as(input))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        
        # 输出类型与输入保持一致
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 
                           device=x.device, dtype=x_ft.dtype)
        
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        return torch.fft.irfft(out_ft, n=x.size(-1))

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, dtype=torch.float32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 1, dtype=dtype),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, 1, dtype=dtype)
        )

    def forward(self, x):
        return self.mlp(x.type_as(next(self.mlp.parameters())))

class FNO1d(nn.Module):
    def __init__(self, modes, width, num_layers=4, dtype=torch.float32):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.num_layers = num_layers
        self.padding = 8
        self.dtype = dtype

        self.p = nn.Linear(2, self.width, dtype=dtype)
        
        # Initialize layers with specified dtype
        self.conv_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.conv_layers.append(SpectralConv1d(self.width, self.width, self.modes1, dtype))
            self.mlp_layers.append(MLP(self.width, self.width, self.width, dtype))
            self.w_layers.append(nn.Conv1d(self.width, self.width, 1, dtype=dtype))
        
        self.q = MLP(self.width, 1, self.width*2, dtype)

    def forward(self, x):
        # 确保输入类型与模型一致
        x = x.to(self.dtype)
        grid = self.get_grid(x.shape, x.device).to(self.dtype)
        
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 2, 1)

        for i in range(self.num_layers):
            x1 = self.conv_layers[i](x)
            x1 = self.mlp_layers[i](x1)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)

        x = self.q(x)
        return x.permute(0, 2, 1)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, dtype=self.dtype, device=device)
        return gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])

    def to_dtype(self, dtype):
        """完整精度转换方法"""
        self.dtype = dtype
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        
        # 转换所有参数
        for param in self.parameters():
            if param.is_complex():
                param.data = param.data.to(complex_dtype)
            elif param.is_floating_point():
                param.data = param.data.to(dtype)
        
        # 转换所有缓冲区
        for buffer in self.buffers():
            if buffer.is_complex():
                buffer.data = buffer.data.to(complex_dtype)
            elif buffer.is_floating_point():
                buffer.data = buffer.data.to(dtype)
        
        # 特殊处理线性层和卷积层（确保权重和偏置都转换）
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                module.weight.data = module.weight.data.to(dtype)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(dtype)
            elif isinstance(module, SpectralConv1d):
                module.weights1.data = module.weights1.data.to(complex_dtype)
        
        return self