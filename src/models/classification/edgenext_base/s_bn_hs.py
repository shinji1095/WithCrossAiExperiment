from __future__ import annotations
import torch
import torch.nn as nn

from .edgenext_bn_hs import EdgeNeXtBNHS 
from .replace_large_kernel_with_3x3 import replace_large_kernels_in_model, CalibConfig

DEFAULT_IMAGE_SIZE = 320  
BACKBONE_NAME = "edgenext_base.s_bn_hs"


def edgenext_small_bn_hs(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    元の実装と同じ関数名・引数でラッパー提供。
    - EdgeNeXtBNHS を small 設定で構築します。
    - **kwargs には num_classes / in_chans / drop_rate / drop_path_rate 等を渡せます。
    """
    model = EdgeNeXtBNHS(
        depths=[3, 3, 9, 3],
        dims=[48, 96, 160, 304],
        expan_ratio=4,
        global_block=[0, 1, 1, 1],
        global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
        use_pos_embd_xca=[False, True, False, False],
        kernel_sizes=[3, 5, 7, 9],
        d2_scales=[2, 2, 3, 4],
        classifier_dropout=0.0,
        **kwargs,
    )
    return model

def build_model(
    num_classes: int,
    backbone_name: str = BACKBONE_NAME,
    in_chans: int = 3,
    dropout_rate=0.0,
    drop_path_rate=0.0,
    **kwargs
) -> nn.Module:
    """
    get_model('classification', 'edgenext_bn_hs', num_classes=..., ...) から呼ばれる想定。
    """
    cfg = kwargs["cfg"]
    model = edgenext_small_bn_hs(
            num_classes=1000,
            in_chans=in_chans
        )
    # print(model)
    checkpoint = torch.load('weight\pytorch\edgenext_small_bn_hs.state_dict.pth', weights_only=False)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    
    if  cfg.replace_large_kernel:
        calib = CalibConfig(H=cfg.image_size[0], W=cfg.image_size[1], batch=8, batches=8, iters=400, lr=1e-2, verbose=True)
        replace_large_kernels_in_model(model, calib=calib, device="cuda")
    model.head = nn.Linear(304, num_classes)


    return model


Model = EdgeNeXtBNHS

__all__ = ["DEFAULT_IMAGE_SIZE", "ClassificationModel", "Model", "build_model"]
