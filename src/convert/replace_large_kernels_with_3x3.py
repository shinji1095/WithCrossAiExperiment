from __future__ import annotations
from typing import Tuple, Optional, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_target_depthwise(m: nn.Module, kset: Iterable[int] = (5, 7, 9)) -> bool:
    """kernel ∈ {5,7,9}, depthwise(groups=C) かつ stride=1, dilation=1 の Conv2d を対象にする。"""
    if not isinstance(m, nn.Conv2d):
        return False
    if m.groups != m.in_channels:
        return False
    if m.out_channels != m.in_channels:
        return False
    kh, kw = m.kernel_size if isinstance(m.kernel_size, Tuple) else (m.kernel_size, m.kernel_size)
    if kh != kw or kh not in kset:
        return False
    if m.stride != (1, 1) or m.dilation != (1, 1):
        return False
    return True


def _make_dirac_3x3(ch: int, device: torch.device) -> torch.Tensor:
    """中心1の3x3カーネル（[C,1,3,3]）。"""
    w = torch.zeros(ch, 1, 3, 3, device=device)
    w[:, 0, 1, 1] = 1.0
    return w


def _factorize_dwkernel_to_3x3(
    K: torch.Tensor,
    iters: int = 600,
    lr: float = 5e-2,
    wd: float = 1e-6,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> list[torch.Tensor]:
    """
    単一 KxK depthwise カーネル（チャネル毎）を、3x3 depthwise の直列 M=(K-1)//2 段へ近似分解。
    K: [C, K, K] 返り値: list[[C,1,3,3]]（長さ M）
    """
    C, Kh, Kw = K.shape
    assert Kh == Kw and Kh % 2 == 1, "K must be odd and square"
    M = (Kh - 1) // 2
    if device is None:
        device = K.device

    # 学習パラメータ：3x3×M
    ks = [torch.nn.Parameter(_make_dirac_3x3(C, device)) for _ in range(M)]
    opt = torch.optim.Adam(ks, lr=lr, weight_decay=wd)

    # インパルス画像
    H = W = Kh
    target = torch.zeros(1, C, H, W, device=device)
    target[:, :, H // 2, W // 2] = 1.0

    # 目標応答（元の KxK DW）
    KK = K.to(device).view(C, 1, Kh, Kw)
    with torch.no_grad():
        tgt = F.conv2d(target, KK, bias=None, stride=1, padding=Kh // 2, groups=C)

    # 最適化ループ
    for i in range(iters):
        x = target
        for p in ks:
            x = F.conv2d(x, p, bias=None, stride=1, padding=1, groups=C)
        loss = F.mse_loss(x, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (i % 100 == 0 or i == iters - 1):
            print(f"[dw 3x3 factorize] step={i:03d} loss={loss.item():.6e}")

    return [p.detach().clone() for p in ks]


def _replace_single_dw_with_stack(
    dw: nn.Conv2d,
    iters: int = 600,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> nn.Sequential:
    """
    depthwise KxK(5/7/9) を、3x3 depthwise の Sequential に置換（最終段にのみ bias を付与）。
    """
    assert _is_target_depthwise(dw)
    if device is None:
        device = next(dw.parameters()).device

    K = dw.weight.detach().to(device).squeeze(1)  # [C,Kh,Kw]
    bias = None if dw.bias is None else dw.bias.detach().to(device)
    C = dw.in_channels
    M = (K.shape[-1] - 1) // 2

    # 既に 3x3 ならコピー
    if M == 1:
        new = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, groups=C, bias=(bias is not None))
        new.weight.data.copy_(dw.weight.data)
        if bias is not None:
            new.bias.data.copy_(bias.cpu())
        return nn.Sequential(new)

    # 3x3×M へ近似分解
    ks = _factorize_dwkernel_to_3x3(K, iters=iters, device=device, verbose=verbose)

    layers: list[nn.Conv2d] = []
    for i, w3 in enumerate(ks):
        use_bias = (i == len(ks) - 1) and (bias is not None)
        conv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, groups=C, bias=use_bias)
        conv.weight.data.copy_(w3.detach().cpu())
        if use_bias:
            conv.bias.data.copy_(bias.detach().cpu())
        layers.append(conv)
        # 等価性を保つため、この直列内には BN / Act を挟まない

    return nn.Sequential(*layers)


def _walk_and_replace(
    module: nn.Module,
    iters: int = 600,
    verbose: bool = False,
    _depth: int = 0,
):
    """
    子モジュールを再帰走査して、対象 DW Conv をその場で置換。
    """
    for name, child in list(module.named_children()):
        if _is_target_depthwise(child):
            if verbose:
                kh = child.kernel_size[0] if isinstance(child.kernel_size, tuple) else child.kernel_size
                print(f"[dw replace] {'  '*_depth}{name}: DW {kh}x{kh} → 3x3×{(kh-1)//2}")
            new_seq = _replace_single_dw_with_stack(child, iters=iters, device=next(child.parameters()).device, verbose=verbose)
            # 親に差し戻す
            if isinstance(module, nn.Sequential):
                module[int(name)] = new_seq
            else:
                setattr(module, name, new_seq)
        else:
            _walk_and_replace(child, iters=iters, verbose=verbose, _depth=_depth + 1)


def apply_depthwise_3x3_factorization(
    root: nn.Module,
    iters: int = 600,
    verbose: bool = False,
):
    """
    公開API：モデル（root）内の kernel=5/7/9 の depthwise Conv を 3x3×M に置換（近似）。
    """
    was_training = root.training
    root.eval()
    _walk_and_replace(root, iters=iters, verbose=verbose)
    if was_training:
        root.train()


__all__ = [
    "apply_depthwise_3x3_factorization",
]
