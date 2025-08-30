import timm
import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, backbone_name: str,
                 dropout_rate: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(backbone_name,
                                          pretrained=True,
                                          num_classes=0,
                                          drop_path_rate=drop_path_rate)
        in_features = self.backbone.num_features
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.head(feat).squeeze(1)
