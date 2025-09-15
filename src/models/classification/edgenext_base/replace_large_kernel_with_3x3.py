#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
replace_large_kernels_with_3x3.py

Pretrained Conv2d layers with k in {5,7,9} are replaced by a stack of 3x3 convs
(5->2 layers, 7->3, 9->4). The new stack is calibrated by minimizing MSE between
the original conv output and the stack output, using random inputs (no dataset).

- Supports groups (incl. depthwise).
- Keeps original stride on the first 3x3; others use stride=1.
- Auto padding to preserve output shape.
- Dilation != 1 is allowed but warned (approximation quality may degrade).
- Bias is enabled on all new convs; optimized by Adam.

USAGE (library):
    from replace_large_kernels_with_3x3 import replace_large_kernels_in_model
    model = ...  # your torch.nn.Module (weights loaded)
    replace_large_kernels_in_model(model, calib_batches=8, iters=400, device='cuda')

USAGE (CLI, example with timm):
    python replace_large_kernels_with_3x3.py --timm resnet50 --weights my.pth --out replaced.pth
"""
from __future__ import annotations
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- helpers ----------------------

def _eff_kernel_from_stack(n_layers: int) -> int:
    # stacking n of 3x3 (stride=1,dilation=1) gives effective k = 2*n+1
    return 2 * n_layers + 1

def _layers_for_kernel(k: int) -> int:
    if k not in (5, 7, 9):
        raise ValueError(f"Unsupported kernel size: {k}")
    return (k - 1) // 2  # 5->2, 7->3, 9->4

def _first_padding_for_shape(orig_k: int, orig_d: Tuple[int,int], orig_p: Tuple[int,int]) -> Tuple[int,int]:
    """
    Choose padding for the first 3x3 so that the output spatial size matches the original conv.
    For original (k, dilation d, padding p), and our first layer (k'=3, d'=1):
      Rough heuristic: p1 = p - floor((d*(k-3))/2).
    Clip at >=0.
    """
    dy = max(1, int(orig_d[0])); dx = max(1, int(orig_d[1]))
    delta_y = (dy * (orig_k - 3)) // 2
    delta_x = (dx * (orig_k - 3)) // 2
    p1y = max(0, int(orig_p[0]) - delta_y)
    p1x = max(0, int(orig_p[1]) - delta_x)
    return p1y, p1x

@dataclass
class CalibConfig:
    H: int = 320
    W: int = 320
    batch: int = 8
    batches: int = 8       # how many random batches
    iters: int = 400       # Adam steps per layer
    lr: float = 1e-2
    wd: float = 0.0
    seed: int = 42
    verbose: bool = True
    amp: bool = True

class Conv3x3Stack(nn.Module):
    """
    Stack of 3x3 convs approximating a larger kxk conv.
    - First layer takes original stride; others stride=1.
    - Padding: first layer is chosen to preserve output geometry; others use padding=1.
    - Groups preserved.
    - Dilation fixed to 1 to keep 3x3 semantics.
    """
    def __init__(self, orig: nn.Conv2d, n_layers: int):
        super().__init__()
        assert n_layers >= 2
        in_ch  = orig.in_channels
        out_ch = orig.out_channels
        groups = orig.groups
        stride = orig.stride
        # Compute first-layer padding to keep output shape same as original
        p1y, p1x = _first_padding_for_shape(orig.kernel_size[0], orig.dilation, orig.padding)

        layers: List[nn.Conv2d] = []
        last_ch = in_ch

        for i in range(n_layers):
            # on the last layer, channels must be out_ch; otherwise middle layers keep width=out_ch to keep capacity high
            next_ch = out_ch if i == n_layers - 1 else out_ch
            # stride only on the first layer
            s = stride if i == 0 else (1, 1)
            # padding
            p = (p1y, p1x) if i == 0 else (1, 1)
            conv = nn.Conv2d(
                in_channels=last_ch,
                out_channels=next_ch,
                kernel_size=3,
                stride=s,
                padding=p,
                dilation=1,
                groups=groups if last_ch % groups == 0 and next_ch % groups == 0 else 1,
                bias=True,
            )
            nn.init.kaiming_normal_(conv.weight, nonlinearity='linear')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            layers.append(conv)
            last_ch = next_ch

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for conv in self.layers:
            y = conv(y)  # linear stack (no BN/ReLU)
        return y

def _calibrate_single(original: nn.Conv2d, stacked: Conv3x3Stack, cfg: CalibConfig, device: torch.device) -> float:
    """
    Optimize stacked's parameters to fit original's output on random inputs.
    Returns final average MSE.
    """
    torch.manual_seed(cfg.seed)
    stacked.train()
    original.eval()
    original.requires_grad_(False)

    # pick spatial size to be large enough for stride/downsample
    H = max(cfg.H, 16)
    W = max(cfg.W, 16)
    B = cfg.batch
    C = original.in_channels

    opt = torch.optim.Adam(stacked.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and cfg.amp))

    loss_avg = 0.0
    steps = cfg.iters
    batches = cfg.batches

    for b in range(batches):
        # regenerate random inputs per batch to improve coverage
        x = torch.randn(B, C, H, W, device=device)

        with torch.no_grad():
            y_tgt = original(x)

        for it in range(steps // batches):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type=='cuda' and cfg.amp)):
                y_hat = stacked(x)
                loss = F.mse_loss(y_hat, y_tgt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_avg += float(loss.detach().item())

    loss_avg /= max(1, (steps))
    return loss_avg

# ---------------------- traversal & replacement ----------------------

def _get_parent_and_key(root: nn.Module, target_name: str):
    """
    Given 'layer1.0.conv1' returns (parent_module=layer1.0, key='conv1')
    """
    parts = target_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def replace_large_kernels_in_model(
    model: nn.Module,
    calib: CalibConfig = CalibConfig(),
    device: Optional[str] = None,
    target_kernels: Tuple[int, ...] = (5, 7, 9),
) -> nn.Module:
    """
    In-place replacement. Returns model (for chaining).
    """
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()

    # Collect target convs (named)
    targets: List[Tuple[str, nn.Conv2d]] = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size
            if isinstance(k, tuple): k = k[0]
            if k in target_kernels:
                targets.append((name, m))

    if calib.verbose:
        print(f"[INFO] Found {len(targets)} Conv2d layers with kernel in {target_kernels}")

    # Replace one by one
    for name, conv in targets:
        k = conv.kernel_size[0]
        n_layers = _layers_for_kernel(k)

        # Warn for dilation
        if any(d != 1 for d in conv.dilation):
            print(f"[WARN] {name}: dilation={conv.dilation} -> approximation quality may degrade.")

        # Build stack
        stack = Conv3x3Stack(conv, n_layers=n_layers).to(dev)

        # Calibrate
        mse = _calibrate_single(conv, stack, calib, dev)
        if calib.verbose:
            print(f"[OK] Replaced {name} (k={k}) with {n_layers}x3x3; calib MSE ~ {mse:.6f}")

        # Swap in the model
        parent, key = _get_parent_and_key(model, name)
        setattr(parent, key, stack)

    return model

# ---------------------- CLI (optional) ----------------------

def _load_timm(model_name: str, weights: Optional[str]) -> nn.Module:
    import timm
    m = timm.create_model(model_name, pretrained=(weights is None), num_classes=1000)
    if weights:
        state = torch.load(weights, map_location="cpu")
        # accept both pure state_dict or checkpoint-like dicts
        sd = state.get("state_dict", state)
        # strip "module." if present
        sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
        m.load_state_dict(sd, strict=False)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm", type=str, default="", help="timm model name (e.g., 'resnet50')")
    ap.add_argument("--weights", type=str, default="", help="optional path to .pth")
    ap.add_argument("--out", type=str, default="", help="optional path to save replaced state_dict")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    args = ap.parse_args()

    if not args.timm:
        print("Provide --timm MODEL (e.g., resnet50). For library use, import replace_large_kernels_in_model().")
        return

    model = _load_timm(args.timm, args.weights if args.weights else None)
    calib = CalibConfig(
        H=args.H, W=args.W, batch=args.batch, batches=args.batches,
        iters=args.iters, lr=args.lr, verbose=True
    )
    replace_large_kernels_in_model(model, calib=calib)

    if args.out:
        torch.save(model.state_dict(), args.out)
        print(f"[DONE] saved: {args.out}")

if __name__ == "__main__":
    main()
