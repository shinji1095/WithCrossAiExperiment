# This script were from the following GitHub repository:
# https://github.com/mmaaz60/EdgeNeXt/blob/main/models/edgenext_bn_hs.py
# Original license: MIT License (see repository for details)
# Retrieved on: September 12, 2025


# src/models/classification/edgenext_base/sdta_encoder.py
import torch
from torch import nn
from timm.models.layers import DropPath
from .layers import LayerNorm, PositionalEncodingFourier
import math


class SDTAEncoder(nn.Module):
    def __init__(
        self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4,
        use_pos_emb=True, num_heads=8, qkv_bias=True, attn_drop=0., drop=0., scales=1,
        conv_dilation: int = 1,  # ← edgenext.py の引数に対応
    ):
        super().__init__()
        width = max(int(math.ceil(dim / scales)), int(math.floor(dim // scales)))
        self.width = width
        self.nums = 1 if scales == 1 else (scales - 1)

        convs = []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3,
                                   padding=conv_dilation, dilation=conv_dilation,
                                   groups=width))
        self.convs = nn.ModuleList(convs)

        self.pos_embd = PositionalEncodingFourier(dim=dim) if use_pos_emb else None
        self.norm_xca = LayerNorm(dim, eps=1e-6)  # (B, N, C) にそのまま適用可
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) \
            if layer_scale_init_value > 0 else None
        self.xca = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = LayerNorm(dim, eps=1e-6)  # NHWC 前提
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) \
            if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        # Multi-Scale DW-Conv branch
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else (sp + spx[i])
            sp = self.convs[i](sp)
            out = sp if i == 0 else torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.nums]), 1)

        # XCA (token mixing in (B, N, C))
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        if self.pos_embd:
            pos = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos
        x = x + self.drop_path(self.gamma_xca * self.xca(self.norm_xca(x)))
        x = x.reshape(B, H, W, C)  # NHWC

        # Inverted bottleneck (NHWC)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = inp + self.drop_path(x)
        return x


class SDTAEncoderBNHS(nn.Module):
    """
    SDTA Encoder with BatchNorm2d + Hardswish
    """
    def __init__(
        self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4,
        use_pos_emb=True, num_heads=8, qkv_bias=True, attn_drop=0., drop=0., scales=1,
        conv_dilation: int = 1,  # 互換のため追加
    ):
        super().__init__()
        width = max(int(math.ceil(dim / scales)), int(math.floor(dim // scales)))
        self.width = width
        self.nums = 1 if scales == 1 else (scales - 1)

        convs = []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3,
                                   padding=conv_dilation, dilation=conv_dilation,
                                   groups=width))
        self.convs = nn.ModuleList(convs)

        self.pos_embd = PositionalEncodingFourier(dim=dim) if use_pos_emb else None
        self.norm_xca = nn.BatchNorm2d(dim)
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) \
            if layer_scale_init_value > 0 else None
        self.xca = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.Hardswish()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) \
            if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x

        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else (sp + spx[i])
            sp = self.convs[i](sp)
            out = sp if i == 0 else torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.nums]), 1)

        # XCA（BN2d 前処理で安定化）
        x = self.norm_xca(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        if self.pos_embd:
            pos = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos
        x = x + self.drop_path(self.gamma_xca * self.xca(x))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Inverted bottleneck
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # NHWC
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = inp + self.drop_path(x)
        return x


class XCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = torch.nn.functional.normalize(q.transpose(-2, -1), dim=-1)
        k = torch.nn.functional.normalize(k.transpose(-2, -1), dim=-1)
        v = v.transpose(-2, -1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
