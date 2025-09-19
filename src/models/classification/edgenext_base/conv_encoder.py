# This script were from the following GitHub repository:
# https://github.com/mmaaz60/EdgeNeXt/blob/main/models/edgenext_bn_hs.py
# Original license: MIT License (see repository for details)
# Retrieved on: September 12, 2025


import torch
from torch import nn
from timm.models.layers import DropPath
from .layers import LayerNorm


class ConvEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        expan_ratio: int = 4,
        kernel_size: int = 7,
        dw_dilation: int = 1,  # ← edgenext.py から渡される想定に対応
    ):
        super().__init__()
        # SAME 風 padding（odd kernel 前提）
        pad = (kernel_size // 2) * dw_dilation
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=pad,
            groups=dim,
            dilation=dw_dilation,
            bias=True,
        )
        # LayerNorm は channels_last を明示
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) \
            if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.dwconv(x)                     # (N, C, H, W)
        x = x.permute(0, 2, 3, 1)              # (N, H, W, C)  ← ここが重要
        x = self.norm(x)                       # channels_last で正規化
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)              # (N, C, H, W)
        x = inp + self.drop_path(x)
        return x


class ConvEncoderBNHS(nn.Module):
    """
    Conv Encoder with BatchNorm2d + Hardswish
    """
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        expan_ratio: int = 4,
        kernel_size: int = 7,
        dw_dilation: int = 1,  # 互換のため追加
    ):
        super().__init__()
        pad = (kernel_size // 2) * dw_dilation
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=pad,
            groups=dim,
            dilation=dw_dilation,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.Hardswish()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) \
            if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.dwconv(x)                     # (N, C, H, W)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)              # (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)              # (N, C, H, W)
        x = inp + self.drop_path(x)
        return x
