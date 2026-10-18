"""Fourier Neural Operator 1D (FNO-1D) implementation.

Reference:
    Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations",
    ICLR 2021. https://arxiv.org/abs/2010.08895
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Fourier Spectral Convolution Layer.

    Computes:
        v_out = F^(-1) ( R * F(v_in) )
    where F is the Fourier transform, R is a learnable complex weight tensor
    acting on the lowest `modes` frequencies.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Learnable complex weights for positive Fourier modes
        # Shape: (in_channels, out_channels, modes)
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, spatial_res)

        Returns:
            Output tensor of shape (batch_size, out_channels, spatial_res)
        """
        batch_size, _, length = x.shape

        # 1. Real Fast Fourier Transform along spatial dimension
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch_size, in_channels, length // 2 + 1)

        # 2. Multiply relevant Fourier modes by learnable complex weights
        num_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )

        out_ft[:, :, :num_modes] = torch.einsum(
            "bix,iox->box",
            x_ft[:, :, :num_modes],
            self.weights[:, :, :num_modes],
        )

        # 3. Inverse Real FFT back to spatial domain
        x_out = torch.fft.irfft(out_ft, n=length, dim=-1)
        return x_out


class FNO1d(nn.Module):
    """1D Fourier Neural Operator.

    Architecture:
        1. Lifting Layer: Projects input channels (e.g. [u0(x), x, t]) to hidden width.
        2. Fourier Layers: Sequence of SpectralConv1d + 1x1 Conv residual + GELU.
        3. Projection Layer: Two-layer MLP projecting hidden width to output channels (u(x, t)).

    Mesh-independent: The same weights can evaluate on any spatial grid resolution.
    """

    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        in_channels: int = 3,
        out_channels: int = 1,
        num_layers: int = 4,
    ):
        """
        Args:
            modes: Number of low-frequency Fourier modes kept.
            width: Channel capacity (hidden width) of Fourier layers.
            in_channels: Number of input channels (e.g. 3 for [u0, x, t]).
            out_channels: Number of output channels (e.g. 1 for u(x, t)).
            num_layers: Number of Fourier operator layers.
        """
        super().__init__()
        self.modes = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        # 1. Lifting layer P: R^in_channels -> R^width
        self.lifting = nn.Linear(in_channels, width)

        # 2. Fourier Layers: Spectral Convolution + 1x1 Conv (linear channel bypass)
        self.spectral_convs = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(num_layers)]
        )
        self.residual_convs = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1) for _ in range(num_layers)]
        )

        # 3. Projection layer Q: R^width -> R^128 -> R^out_channels
        self.projection1 = nn.Linear(width, 128)
        self.projection2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, spatial_res, in_channels)

        Returns:
            Output tensor of shape (batch_size, spatial_res, out_channels)
        """
        # 1. Lifting layer
        # (batch, spatial_res, in_channels) -> (batch, spatial_res, width)
        x = self.lifting(x)

        # Permute to (batch, width, spatial_res) for Conv1d and rfft along dim=-1
        x = x.permute(0, 2, 1)

        # 2. Fourier Layers
        for i in range(self.num_layers):
            x_spectral = self.spectral_convs[i](x)
            x_residual = self.residual_convs[i](x)
            x = x_spectral + x_residual
            x = F.gelu(x)

        # Permute back to (batch, spatial_res, width)
        x = x.permute(0, 2, 1)

        # 3. Projection layer
        x = self.projection1(x)
        x = F.gelu(x)
        x = self.projection2(x)

        return x
