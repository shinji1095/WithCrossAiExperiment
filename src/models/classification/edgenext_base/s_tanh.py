from __future__ import annotations
import math
import torch
import torch.nn as nn
import timm

from .edgenext import EdgeNeXt
from convert.replace_gelu_with_tanh import replace_gelu_with_tanh

DEFAULT_IMAGE_SIZE = 320  
BACKBONE_NAME = "edgenext_base.s_tanh"


def edgenext_small(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    元の実装と同じ関数名・引数でラッパー提供。
    - EdgeNeXtBNHS を small 設定で構築します。
    - **kwargs には num_classes / in_chans / drop_rate / drop_path_rate 等を渡せます。
    """
    model = EdgeNeXt(
        depths=[3, 3, 9, 3],
        dims=[48, 96, 160, 304],
        expan_ratio=4,
        global_block=[0, 1, 1, 1],
        global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
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
    model = edgenext_small(
            num_classes=1000,
            in_chans=in_chans
        )
    # print(model)
    checkpoint = torch.load(r'weight\pytorch\edgenext_small.usi_in1k.pth', weights_only=False)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    replace_gelu_with_tanh(model)
    
    model.head = nn.Linear(304, num_classes)


    return model


Model = EdgeNeXt

__all__ = ["DEFAULT_IMAGE_SIZE", "ClassificationModel", "Model", "build_model"]
