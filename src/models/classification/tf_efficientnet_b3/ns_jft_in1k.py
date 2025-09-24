from __future__ import annotations
import torch
import torch.nn as nn
import timm

DEFAULT_IMAGE_SIZE = 300  
BACKBONE_NAME = "tf_efficientnet_b3.ns_jft_in1k"

class ClassificationModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        in_chans: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes, 
            drop_path_rate=drop_path_rate,
            in_chans=in_chans,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def build_model(
    num_classes: int,
    backbone_name: str = BACKBONE_NAME,
    dropout_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    in_chans: int = 3,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:

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
