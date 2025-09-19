from __future__ import annotations
import torch
import torch.nn as nn
import timm

from convert.replace_large_kernels_with_3x3 import apply_depthwise_3x3_factorization

DEFAULT_IMAGE_SIZE = 320  # get_default_image_size() で参照可能
BACKBONE_NAME = "edgenext_base.lkdw_cv3x3"


class ClassificationModel(nn.Module):
    """
    timm の EdgeNeXt など任意バックボーンを使った薄い分類ヘッド。
    - backbone は timm に任せる（num_classes は引数の値をそのまま渡す）。
    - 追加オプションで kernel=5/7/9 の depthwise conv を 3x3×M の近似に置換可能。
    """
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        in_chans: int = 3,
        pretrained: bool = False,
        dw_factorize: bool = True,
        dw_factorize_iters: int = 600,
        dw_factorize_verbose: bool = False,
        **kwargs
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "edgenext_base.usi_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            in_chans=in_chans,
        )

        if dw_factorize:
            was_training = self.backbone.training
            self.backbone.eval()
            apply_depthwise_3x3_factorization(
                self.backbone,
                iters=dw_factorize_iters,
                verbose=dw_factorize_verbose,
            )
            if was_training:
                self.backbone.train()

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
    kwargs で dw_factorize などを受け付ける（既定は有効）。
    """
    return ClassificationModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        in_chans=in_chans,
        pretrained=pretrained,
        **kwargs,
    )


Model = ClassificationModel

__all__ = ["DEFAULT_IMAGE_SIZE", "ClassificationModel", "Model", "build_model"]
