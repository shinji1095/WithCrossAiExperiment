# src/models/segmentation/edgenext.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

DEFAULT_IMAGE_SIZE = 320

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class FPNDecoder(nn.Module):
    """
    Timmのfeatures_onlyエンコーダの多段特徴（高解像度→低解像度の順）を入力。
    lateral 1x1で整形し、最下段からトップダウンにupsample+加算。
    """
    def __init__(self, enc_channels: list[int], fpn_channels: int = 256,
                 out_channels: int = 256, dropout: float = 0.0):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, kernel_size=1, bias=False) for c in enc_channels
        ])
        self.smooth   = ConvBNAct(fpn_channels, out_channels, k=3, s=1, p=1)
        self.dropout  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, feats: list[torch.Tensor], out_hw: tuple[int, int]) -> torch.Tensor:
        lats = [lat(f) for lat, f in zip(self.laterals, feats)]
        y = lats[-1]  # 最も低解像度（最深）の特徴
        for i in range(len(lats) - 2, -1, -1):
            y = F.interpolate(y, size=lats[i].shape[-2:], mode="bilinear", align_corners=False)
            y = y + lats[i]
        y = self.smooth(y)
        y = self.dropout(y)
        y = F.interpolate(y, size=out_hw, mode="bilinear", align_corners=False)
        return y

class SegmentationModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        backbone_name: str = "edgenext_small",
        in_chans: int = 3,
        pretrained: bool = True,
        out_indices = (0, 1, 2, 3),   # ← ここを修正（EdgeNeXtは0..3が有効）
        fpn_channels: int = 256,
        dec_channels: int = 256,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        # 一部モデルは drop_path_rate 未対応なので try/except
        try:
            self.encoder = timm.create_model(
                backbone_name,
                features_only=True,
                out_indices=out_indices,
                in_chans=in_chans,
                pretrained=pretrained,
                drop_path_rate=drop_path_rate,
            )
        except TypeError:
            self.encoder = timm.create_model(
                backbone_name,
                features_only=True,
                out_indices=out_indices,
                in_chans=in_chans,
                pretrained=pretrained,
            )

        enc_channels = self.encoder.feature_info.channels()

        self.decoder = FPNDecoder(enc_channels, fpn_channels=fpn_channels,
                                  out_channels=dec_channels, dropout=dropout_rate)
        self.head = nn.Conv2d(dec_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        feats = self.encoder(x)   # List[Tensor]（高解像度→低解像度）
        y = self.decoder(feats, (H, W))
        logits = self.head(y)
        return logits

def build_model(
    num_classes: int,
    backbone_name: str = "edgenext_small",
    in_chans: int = 3,
    pretrained: bool = True,
    out_indices = (0, 1, 2, 3),   # ← 同様に修正
    fpn_channels: int = 256,
    dec_channels: int = 256,
    dropout_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    return SegmentationModel(
        num_classes=num_classes,
        backbone_name=backbone_name,
        in_chans=in_chans,
        pretrained=pretrained,
        out_indices=out_indices,
        fpn_channels=fpn_channels,
        dec_channels=dec_channels,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
    )

Model = SegmentationModel
__all__ = ["DEFAULT_IMAGE_SIZE", "SegmentationModel", "Model", "build_model"]
