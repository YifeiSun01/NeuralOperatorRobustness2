from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)
np.random.seed(0)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(x, weights):
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        m1 = min(self.modes1, x_ft.size(-2))
        m2 = min(self.modes2, x_ft.size(-1))
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.mlp(x)


class FNO2d(nn.Module):
    """Static 2D FNO mapping a Darcy coefficient field A to a pressure field U."""

    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int,
        num_layers: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.padding = padding

        self.p = nn.Linear(in_channels + 2, width)
        self.conv_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(SpectralConv2d(width, width, modes1, modes2))
            self.mlp_layers.append(MLP(width, width, width))
            self.w_layers.append(nn.Conv2d(width, width, 1))
        self.q = MLP(width, out_channels, width * 4)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        grid = self.get_grid(x.shape, x.device, x.dtype)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x).permute(0, 3, 1, 2)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.num_layers):
            x = self.mlp_layers[i](self.conv_layers[i](x)) + self.w_layers[i](x)
            if i < self.num_layers - 1:
                x = F.gelu(x)

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        return x.permute(0, 2, 3, 1)

    @staticmethod
    def get_grid(shape, device, dtype):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, dtype=dtype, device=device)
        gridy = torch.linspace(0, 1, size_y, dtype=dtype, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)
