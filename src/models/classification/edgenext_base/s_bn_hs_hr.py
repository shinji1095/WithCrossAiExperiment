from __future__ import annotations
import torch
import torch.nn as nn

from .edgenext_bn_hs import EdgeNeXtBNHS 
from utils.load import load_partial_state_dict

DEFAULT_IMAGE_SIZE = 320  
BACKBONE_NAME = "edgenext_base.s_bn_hs_hr"


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
        use_pos_embd_xca=[False, True, False, True],
        kernel_sizes=[3, 5, 7, 9],
        d2_scales=[2, 2, 3, 4],
        classifier_dropout=0.0,
        downsample_strides=[4, 2, 2, 1],
        stage_dilations=[1, 1, 1, 1],
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
    ckpt_path = 'weight\pytorch\edgenext_small_bn_hs.state_dict.pth'
    # checkpoint = torch.load('weight\pytorch\edgenext_small_bn_hs.state_dict.pth', weights_only=False)
    # state_dict = checkpoint["model"]
    load_partial_state_dict(model, ckpt_path)
    
    model.head = nn.Linear(304, num_classes)


    return model


Model = EdgeNeXtBNHS

__all__ = ["DEFAULT_IMAGE_SIZE", "ClassificationModel", "Model", "build_model"]
