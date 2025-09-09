from __future__ import annotations
import argparse
import subprocess
import sys
import os
from pathlib import Path

import torch
import tensorflow as tf

THIS_FILE = Path(__file__).resolve()
SRC_DIR   = THIS_FILE.parents[1]           # .../src
ROOT_DIR  = THIS_FILE.parents[2]
print(SRC_DIR)
print(ROOT_DIR)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.model import get_model, get_default_image_size

def run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def main():
    ap = argparse.ArgumentParser(description="PyTorch -> ONNX -> onnx2tf -> TFLite converter")
    ap.add_argument("--task", required=True, choices=["classification", "regression", "segmentation"],
                    help="タスク種別（classification / regression）")
    ap.add_argument("--model", required=True,
                    help="モデル名（models/<task>/<model>.py） 例: edgenext")
    ap.add_argument("--image_size", type=int, default=None,
                    help="入力解像度（省略時はモジュールのDEFAULT_IMAGE_SIZEか320）")
    ap.add_argument("--num_classes", type=int, default=3,
                    help="分類タスク時のクラス数（デフォルト3）")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--drop_path", type=float, default=0.0)
    ap.add_argument("--weights", type=str, default=None,
                    help="明示的に.pthを指定する場合。未指定時は weight/pytorch/<task>/<model>.pth")
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--extra_onnx2tf_args", type=str, default="",
                    help='onnx2tf追加引数をそのまま渡す（例: "--keep_input_tensor"）')
    args = ap.parse_args()

    task       = args.task
    model_name = args.model

    weight_root     = ROOT_DIR / "weight"
    onnx_dir        = weight_root / "onnx" / task
    keras_dir       = weight_root / "keras" / task / model_name
    tflite_dir      = weight_root / "tflite" / task
    onnx_dir.mkdir(parents=True, exist_ok=True)
    keras_dir.mkdir(parents=True, exist_ok=True)
    tflite_dir.mkdir(parents=True, exist_ok=True)

    onnx_path   = onnx_dir / f"{model_name}.onnx"
    tflite_path = tflite_dir / f"{model_name}.tflite"

    image_size = args.image_size or get_default_image_size(task, model_name, fallback=320)

    build_kwargs = {
        "num_classes": args.num_classes,
        "dropout_rate": args.dropout,
        "drop_path_rate": args.drop_path,
    }
    if task == "regression":
        build_kwargs.pop("num_classes", None)

    print(f"[INFO] Build model: task={task}, name={model_name}, image_size={image_size}")
    model = get_model(task, model_name, **build_kwargs).eval()

    weights_path = Path(args.weights) if args.weights else (weight_root / "pytorch" / task / f"{model_name}.pth")
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state, strict=True)
    print(f"[✓] Loaded weights: {weights_path}")

    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[✓] Exported ONNX: {onnx_path}")

    onnx2tf_cmd = [
        "onnx2tf", "-i", str(onnx_path),
        "-osd", "-b", "1",
        "-ois", f"1,3,{image_size},{image_size}",
        "-o", str(keras_dir),
    ]
    if args.extra_onnx2tf_args:
        onnx2tf_cmd.extend(args.extra_onnx2tf_args.split(" "))

    run_cmd(onnx2tf_cmd)
    print(f"[✓] Built SavedModel/Keras: {keras_dir}")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(keras_dir))
    # 安定化オプション
    converter.experimental_new_converter = True
    # 内蔵OP不足は SELECT_TF_OPS に回しつつ、可能な限り XNNPACK 実行
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    # 最適化（必要に応じてコメント解除）
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]   # 重みのみFP16化したい場合

    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    print(f"[✓] Exported TFLite: {tflite_path}")

if __name__ == "__main__":
    main()
