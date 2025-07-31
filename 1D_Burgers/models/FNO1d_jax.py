import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        return torch.fft.irfft(out_ft, n=x.size(-1))

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 1),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.mlp(x)

class FNO1d(nn.Module):
    def __init__(self, modes, width, num_layers=4):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.num_layers = num_layers
        self.padding = 8

        self.p = nn.Linear(2, self.width)
        
        # Initialize layers dynamically
        self.conv_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.conv_layers.append(SpectralConv1d(self.width, self.width, self.modes1))
            self.mlp_layers.append(MLP(self.width, self.width, self.width))
            self.w_layers.append(nn.Conv1d(self.width, self.width, 1))
        
        self.q = MLP(self.width, 1, self.width*2)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 2, 1)

        for i in range(self.num_layers):
            x1 = self.conv_layers[i](x)
            x1 = self.mlp_layers[i](x1)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:  # No GELU after last layer
                x = F.gelu(x)

        x = self.q(x)
        return x.permute(0, 2, 1)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        return gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])

