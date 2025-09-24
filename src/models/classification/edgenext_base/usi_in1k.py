from __future__ import annotations
import torch
import torch.nn as nn
import timm

DEFAULT_IMAGE_SIZE = 320  # get_default_image_size() で参照可能
BACKBONE_NAME = "edgenext_base.usi_in1k"

@torch.no_grad()
def _get_in_features(backbone: nn.Module) -> int:
    """Get final feature dim by dummy forward."""
    backbone.eval()
    # timm の default_cfg から推奨サイズを取得（無ければ 3x224x224）
    c, h, w = backbone.default_cfg.get("input_size", (3, 320, 320))
    device = next(backbone.parameters()).device
    dummy  = torch.zeros(1, c, h, w, device=device)
    out    = backbone(dummy)
    # 多くのtimm分類モデルは (B, C) を返すが、一部 (B, C, h, w) の場合もあるため潰す
    if out.ndim == 4:
        out = out.mean(dim=[2, 3])  # GAP 相当
    return out.shape[1]

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
        # in_features  = _get_in_features(self.backbone)
        # self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        # self.head    = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # feat = self.backbone(x)
        # if feat.ndim == 4:
        #     feat = feat.mean(dim=[2, 3])
        # feat = self.dropout(feat)
        return self.backbone(x)

def build_model(
    num_classes: int,
    backbone_name: str = BACKBONE_NAME,
    dropout_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    in_chans: int = 3,
    pretrained: bool = True,
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

