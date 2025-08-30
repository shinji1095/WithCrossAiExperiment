import os
from pathlib import Path
import torch
import torch.nn as nn
import timm
import tensorflow as tf
from models.model import get_model


@torch.no_grad()
def _get_in_features(backbone: nn.Module) -> int:
    """
    ダミー入力で最終特徴ベクトル長を直接測定。
    どの timm バックボーンでも正確。
    """
    backbone.eval()                     # ← 念のため
    cfg = backbone.default_cfg
    c, h, w = cfg.get("input_size", (3, 224, 224))
    dummy   = torch.zeros(1, c, h, w, device=next(backbone.parameters()).device)
    feat    = backbone(dummy)
    return feat.shape[1]


class ClassificationModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(backbone_name,
                                          pretrained=True,
                                          num_classes=0)  # ヘッドなし
        in_features   = _get_in_features(self.backbone)   # ← 新ユーティリティ
        self.head     = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.backbone(x)        # shape: (B, in_features)
        return self.head(feat)

# =============================================
# Colab: timm → ONNX → Keras/SavedModel → TFLite
# モデルごとに入力サイズを指定
# (c) 2025
# =============================================

# --- モデルと入力サイズの対応表（必要に応じて追加） ---
MODELS = {
    # 'convnext_tiny.in12k_ft_in1k_best': 288,
    # 'convnext_nano.in12k_ft_in1k_best': 288,
    # 'hgnetv2_b4.ssld_stage2_ft_in1k_best': 288,
    # 'vit_medium_patch16_gap_256.sw_in12k_ft_in1k_best': 256,
    'edgenext_base.usi_in1k': 320,
    # 'hgnetv2_b3.ssld_stage2_ft_in1k_best': 288,
    # 'edgenext_base.in21k_ft_in1k_best': 320,
    # 'vit_mediumd_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k_best': 256,
    # 'tf_efficientnet_b4.ap_in1k_best': 380,
}

WEIGHTS_DIR = Path('weight/pytorch')
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)

# --- 失敗記録用リスト ---
errors = []

# --- 変換ループ ---
for model_name, image_size in MODELS.items():
    print(f'\n==================== {model_name} 変換開始 ====================')
    try:
        ROOT_DIR = Path(f'weight')
        ROOT_DIR.mkdir(exist_ok=True, parents=True)

        ONNX_MODEL_PATH = ROOT_DIR/ 'onnx' / f'{model_name}.onnx'
        ONNX2TF_OUTPUT_DIR = ROOT_DIR / 'tflite' / model_name
        TFLITE_MODEL_PATH = ROOT_DIR / f'{model_name}.tflite'

        # --- 1️⃣ timm → ONNX ---
        # model = ClassificationModel(model_name.replace("_best", ""), 3).eval()
        model = get_model(
            'classification',
            model_name,
            num_classes=3,
            dropout_rate=0.0,
            drop_path_rate=0.0,
        ).eval()

        # --- 重みロード ---
        weights_path = WEIGHTS_DIR / f'{model_name}.pth'
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f'[✓] 重みロード完了: {weights_path}')
        else:
            raise FileNotFoundError(f'重みファイルが見つかりません: {weights_path}')

        dummy_input = torch.randn(1, 3, image_size, image_size)
        torch.onnx.export(
            model,
            dummy_input,
            str(ONNX_MODEL_PATH),
            input_names=['input'],
            output_names=['output'],
            opset_version=14
        )
        print(f'[✓] ONNXエクスポート完了: {ONNX_MODEL_PATH}')


        # --- 2️⃣ ONNX → Keras/SavedModel ---
        result = os.system(
            f'onnx2tf -i {ONNX_MODEL_PATH} '
            f'-osd -oh5 -b 1 -kt /input '
            f'-ois 1,3,{image_size},{image_size} '
            f'-o {ONNX2TF_OUTPUT_DIR}'
        )
        if result != 0:
            raise RuntimeError('onnx2tfコマンドで失敗')

        print(f'[✓] Keras/SavedModel生成完了: {ONNX2TF_OUTPUT_DIR}')

        # --- 3️⃣ SavedModel → TFLite ---
        converter = tf.lite.TFLiteConverter.from_saved_model(str(ONNX2TF_OUTPUT_DIR))

        # 安定化オプション
        converter.experimental_new_converter = True
        converter._experimental_lower_tensor_list_ops = True

        # Flex を許可（内蔵OPで足りない分だけ Flex に回し、大半は XNNPACK に乗せる）
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

        # （任意）サイズ/帯域を下げたい場合は “重みだけ” FP16。入出力は float32 のまま。
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()
        TFLITE_MODEL_PATH.write_bytes(tflite_model)

        print(f'[✓] TFLite変換完了: {TFLITE_MODEL_PATH}')

        
    except Exception as e:
        print(f'[✗] {model_name} 変換失敗: {e}')
        errors.append({'model': model_name, 'error': str(e)})

# --- 結果報告 ---
if errors:
    print('\n==================== 変換失敗モデル一覧 ====================')
    for entry in errors:
        print(f"・{entry['model']}: {entry['error']}")
else:
    print('\n🎉 全モデル正常変換完了 🎉')
