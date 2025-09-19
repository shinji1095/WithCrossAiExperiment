from __future__ import annotations
import torch
import torch.nn as nn
import timm
from .edgenext import EdgeNeXt

DEFAULT_IMAGE_SIZE = 320  # get_default_image_size() で参照可能
BACKBONE_NAME = "edgenext_base.usi_in1k"

# class ClassificationModel(nn.Module):
#     """
#     timm の EdgeNeXt など任意バックボーンを使った薄い分類ヘッド。
#     - backbone は num_classes=0 で最終FCを外した特徴量ベクトルを返す前提。
#     """
#     def __init__(
#         self,
#         backbone_name: str,
#         num_classes: int,
#         dropout_rate: float = 0.0,
#         drop_path_rate: float = 0.0,
#         in_chans: int = 3,
#         pretrained: bool = True,
#     ):
#         super().__init__()
#         self.backbone = timm.create_model(
#             backbone_name,
#             pretrained=pretrained,
#             num_classes=num_classes, 
#             drop_path_rate=drop_path_rate,
#             in_chans=in_chans,
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.backbone(x)

# def build_model(
#     num_classes: int,
#     backbone_name: str = BACKBONE_NAME,
#     dropout_rate: float = 0.0,
#     drop_path_rate: float = 0.0,
#     in_chans: int = 3,
#     pretrained: bool = False,
#     **kwargs
# ) -> nn.Module:
#     """
#     get_model('classification', 'edgenext', num_classes=..., ...) から呼ばれる想定。
#     """
#     return ClassificationModel(
#         backbone_name=backbone_name,
#         num_classes=num_classes,
#         dropout_rate=dropout_rate,
#         drop_path_rate=drop_path_rate,
#         in_chans=in_chans,
#         pretrained=pretrained,
#     )

def edgenext_small(pretrained=False, **kwargs):
    # 5.59M & 1260.59M @ 256 resolution
    # 79.43% Top-1 accuracy
    # AA=True, No Mixup & Cutmix, DropPath=0.1, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=20.47 versus 18.86 for MobileViT_S
    # For A100: FPS @ BS=1: 172.33 & @ BS=256: 3010.25 versus FPS @ BS=1: 93.84 & @ BS=256: 1785.92 for MobileViT_S
    model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 2, 3, 4],
                     downsample_strides=[4, 2, 2, 2],
                     stage_dilations=[1, 1, 1, 1],
                     **kwargs)

    return model

class ClassificationModel(nn.Module):
    """
    timm の EdgeNeXt など任意バックボーンを使った薄い分類ヘッド。
    - backbone は num_classes=0 で最終FCを外した特徴量ベクトルを返す前提。
    """
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.3,
        in_chans: int = 3,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__()
        self.backbone = EdgeNeXt(in_chans=in_chans, 
                                 num_classes=num_classes,
                                 drop_path_rate=drop_path_rate,
                                 depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
                                 global_block=[0, 1, 1, 1],
                                 global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                                 use_pos_embd_xca=[False, True, False, False],
                                 kernel_sizes=[3, 5, 7, 9],
                                 d2_scales=[2, 2, 3, 4], 
                                 downsample_strides=[4, 2, 2, 2],
                                 stage_dilations=[1, 1, 1, 1],
                                 **kwargs)
        
        checkpoint = torch.load(r'weight\pytorch\edgenext_small.usi_in1k.pth', weights_only=False)
        state_dict = checkpoint
        self.backbone.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def build_model(
    num_classes: int,
    backbone_name: str = BACKBONE_NAME,
    dropout_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    in_chans: int = 3,
    pretrained: bool = False,
    **kwargs
) -> nn.Module:
    """
    get_model('classification', 'edgenext', num_classes=..., ...) から呼ばれる想定。
    """
    return ClassificationModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        in_chans=in_chans,
        pretrained=pretrained,
    )

Model = ClassificationModel

__all__ = ["DEFAULT_IMAGE_SIZE", "ClassificationModel", "Model", "build_model"]
