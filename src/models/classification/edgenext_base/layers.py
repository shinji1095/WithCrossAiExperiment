# layers.py
# This script were from the following GitHub repository:
# https://github.com/mmaaz60/EdgeNeXt/blob/main/models/edgenext_bn_hs.py
# Original license: MIT License (see repository for details)
# Retrieved on: September 12, 2025


import math
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        assert data_format in ("channels_last", "channels_first")
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PositionalEncodingFourier(nn.Module):
    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        device = self.token_projection.weight.device
        mask = torch.zeros(B, H, W, device=device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return self.token_projection(pos)


# ---------- Anti-aliased downsampling primitives ----------

class BlurPoolDW(nn.Module):
    """Depthwise 3x3 Gaussian-like blur with stride (fixed weights)."""
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        assert stride in (1, 2)
        self.dw = nn.Conv2d(channels, channels, 3, stride=stride, padding=1,
                            groups=channels, bias=False)
        with torch.no_grad():
            k = torch.tensor([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]], dtype=torch.float32) / 16.0
            w = torch.zeros(channels, 1, 3, 3)
            w[:, 0, :, :] = k
            self.dw.weight.copy_(w)
        for p in self.dw.parameters():
            p.requires_grad = False  # 固定カーネル

    def forward(self, x):
        return self.dw(x)


class DownsampleAAMix(nn.Module):
    """
    Anti-aliased mix: BlurPool と MaxPool を学習可能ゲートで混合。
    出力Cは入力Cと同じ。後段で 1x1 Conv などを接続して使用。
    """
    def __init__(self, channels: int, stride: int = 2, init_alpha: float = 0.5):
        super().__init__()
        assert stride in (1, 2)
        self.blur = BlurPoolDW(channels, stride=stride)
        self.maxp = nn.MaxPool2d(kernel_size=stride, stride=stride)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        gate = torch.sigmoid(self.alpha)
        y_blur = self.blur(x)
        y_max  = self.maxp(x)
        return gate * y_max + (1.0 - gate) * y_blur
