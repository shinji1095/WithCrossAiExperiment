import torch
import torch.nn as nn
import timm
import pathlib

# ====== パス設定 ======
weights_dir = pathlib.Path(r"data\ex001\weights")
tflite_dir  = pathlib.Path(r"data\ex001\tflite")
tflite_dir.mkdir(parents=True, exist_ok=True)

import torch
import torch.nn as nn
import timm
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os
from pathlib import Path

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# モデル定義（ユーザーのコードを再利用）
def _get_in_features(backbone: nn.Module) -> int:
    backbone.eval()
    cfg = backbone.default_cfg
    c, h, w = cfg.get("input_size", (3, 160, 160))
    dummy = torch.zeros(1, c, h, w, device=next(backbone.parameters()).device)
    feat = backbone(dummy)
    return feat.shape[1]

class ClassificationModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(backbone_name,
                                        pretrained=True,
                                        num_classes=0)
        in_features = _get_in_features(self.backbone)
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


# Paths
weights_dir = r"data\ex001\weights"
tflite_dir = r"data\ex001\tflite"
os.makedirs(tflite_dir, exist_ok=True)

# Number of classes (adjust based on your model)
num_classes = 1000  # Example: ImageNet classes, modify as needed

# Process each .pth file
for pth_file in Path(weights_dir).glob("*.pth"):
    # Extract model name from filename
    model_name = pth_file.stem.replace("_best", "")
    
    # Load the model
    model = ClassificationModel(model_name, num_classes)
    state_dict = torch.load(pth_file, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    # Convert to TorchScript
    example_input = torch.randn(1, 3, 224, 224)  # Standard input size
    traced_model = torch.jit.trace(model, example_input)
    
    # Save TorchScript model (optional, for debugging)
    torchscript_path = os.path.join(tflite_dir, f"{model_name}_traced.pt")
    traced_model.save(torchscript_path)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_torchscript(torchscript_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()

    # Save TFLite model
    tflite_path = os.path.join(tflite_dir, f"{model_name}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Converted {model_name} to {tflite_path}")