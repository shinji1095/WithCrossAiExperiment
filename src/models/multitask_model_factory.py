import timm
import torch.nn as nn

class MultitaskModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int,
                 dropout_rate: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(backbone_name,
                                          pretrained=True,
                                          num_classes=0,
                                          drop_path_rate=drop_path_rate)
        in_features = self.backbone.num_features

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.class_head = nn.Linear(in_features, num_classes)
        self.regress_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        signal_pred = self.class_head(feat)
        slope_pred  = self.regress_head(feat).squeeze(1)
        return signal_pred, slope_pred
