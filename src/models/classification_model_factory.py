import torch
import torch.nn as nn
import timm

@torch.no_grad()
def _get_in_features(backbone: nn.Module) -> int:
    """Get final feature dim by dummy forward."""
    backbone.eval()
    c, h, w = backbone.default_cfg.get("input_size", (3, 224, 224))
    dummy   = torch.zeros(1, c, h, w, device=next(backbone.parameters()).device)
    return backbone(dummy).shape[1]

class ClassificationModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int,
                 dropout_rate: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(backbone_name,
                                          pretrained=True,
                                          num_classes=0,
                                          drop_path_rate=drop_path_rate)
        in_features   = _get_in_features(self.backbone)
        self.dropout  = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head     = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.head(feat)
