#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test.py (Torch + TFLite + ONNX + TF Keras/SavedModel, robust loaders)
- Evaluate *.pth/*.pt/*.onnx/*.tflite/*.h5/*.keras and SavedModel directories.
- Torch: uses training builder (models.model.get_model).
- ONNX: auto NCHW/NHWC; RGB/norm auto-fallback if predictions collapse.
- TFLite: quantized I/O handled; RGB/norm auto-fallback.
- TF Keras/SavedModel:
    * .h5/.keras -> load_model with custom_object_scope({'TFOpLambda': Lambda})
      + safe_mode=False で再挑戦
    * SavedModel dir -> keras.layers.TFSMLayer(call_endpoint=...), 出力は自動で key 取得
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- Project modules ----
from config.config import load_all_training_configs
from dataset_signalmix import SignalMixClassificationDataset
from metrics.metrics import evaluate_classification, evaluate_regression
from models.model import get_model


# ===== Dataset Transform & Collate (aligned to training) =====
class SimpleTransform:
    """Resize -> ToTensor(0..1) -> Normalize(mean,std) (default mean=std=0.5 -> -1..1)"""
    def __init__(self, size_hw=None, mean=0.5, std=0.5):
        self.size_hw = tuple(size_hw) if size_hw is not None else None  # (H, W)
        self.mean = float(mean); self.std = float(std)

    def base_transform(self, image):
        import cv2, torch
        if self.size_hw is not None:
            H, W = self.size_hw
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        t = (t - self.mean) / self.std
        return {"image": t}


def signalmix_collate(batch):
    if len(batch[0]) == 3:
        imgs, labels, bboxes = zip(*batch)
        return torch.stack(imgs, 0), torch.as_tensor(labels, dtype=torch.long), list(bboxes)
    else:
        imgs, labels = zip(*batch)
        return torch.stack(imgs, 0), torch.as_tensor(labels, dtype=torch.long)


# ===== Utilities =====
def _first_not_none(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

def _safe_int(x, default: int):
    try:
        return int(x) if x is not None else default
    except Exception:
        return default

def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    ez = np.exp(z); return ez / np.sum(ez, axis=-1, keepdims=True)

def _degenerate(preds: np.ndarray, acc: float) -> bool:
    if len(preds) == 0: return True
    dom = np.bincount(preds).max() / len(preds)
    return dom >= 0.95 or acc < 0.2


# ===== Config & Torch helpers =====
def pick_cfg_for_ckpt(cfg_list, ckpt: Dict[str, Any]):
    ck_name = str(ckpt.get("model_name", "") or "").lower()
    ck_task = str(ckpt.get("task", "") or "").lower()
    ck_ncls = _safe_int(ckpt.get("num_classes", None), -1)
    def score(c):
        s = 0
        if ck_name and ck_name == str(getattr(c, "model_name", "")).lower(): s += 2
        if ck_task and ck_task == str(getattr(c, "task", "")).lower(): s += 1
        if ck_ncls > 0 and ck_ncls == int(getattr(c, "num_classes", 3)): s += 1
        return s
    return max(cfg_list, key=score) if cfg_list else None

def build_model_from_cfg_via_builder(cfg, override_ckpt: Optional[Dict[str, Any]] = None,
                                     ds_num_classes: Optional[int] = None):
    ck = override_ckpt or {}
    if "model_name" in ck and ck["model_name"]: cfg.model_name = ck["model_name"]
    if "task" in ck and ck["task"]: cfg.task = ck["task"]
    cfg.num_classes = _safe_int(_first_not_none(ck.get("num_classes"),
                                               getattr(cfg, "num_classes", None),
                                               ds_num_classes, 3), 3)
    model = get_model(cfg.task, cfg.model_name,
                      num_classes=int(cfg.num_classes),
                      dropout_rate=getattr(cfg, "dropout_rate", 0.0),
                      drop_path_rate=getattr(cfg, "drop_path_rate", 0.0))
    return model, str(cfg.task).lower(), int(cfg.num_classes), str(cfg.model_name)

def load_state_dict_safely(model: torch.nn.Module, ckpt: Dict[str, Any]) -> None:
    if isinstance(ckpt, dict) and "state_dict" in ckpt: state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:     state = ckpt["model"]
    else:                                                state = ckpt
    new_state = {}
    for k, v in state.items():
        new_state[k[7:] if isinstance(k, str) and k.startswith("module.") else k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:   print(f"[WARN] missing keys: {missing[:12]}{'...' if len(missing)>12 else ''}")
    if unexpected:print(f"[WARN] unexpected keys: {unexpected[:12]}{'...' if len(unexpected)>12 else ''}")


# ===== Torch evaluation =====
def evaluate_model_torch(model, loader, device, task: str, num_classes: int):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing (Torch)", leave=False):
            images, labels = batch[:2]
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                logits = model(images).float()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_prob.extend(torch.softmax(logits, 1).cpu().numpy())
    return evaluate_classification(y_true, y_pred, np.array(y_prob), num_classes=num_classes)


# ===== TFLite evaluation =====
def _load_tflite_interpreter(model_path: Path):
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        it = Interpreter(model_path=str(model_path)); it.allocate_tensors(); return it
    except Exception:
        pass
    try:
        import tensorflow as tf  # type: ignore
        it = tf.lite.Interpreter(model_path=str(model_path)); it.allocate_tensors(); return it
    except Exception:
        pass
    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        it = Interpreter(model_path=str(model_path)); it.allocate_tensors(); return it
    except Exception as e:
        raise ImportError("TFLite interpreter not found. Install tflite-runtime or tensorflow.") from e

def _get_scale_zero(detail: Dict[str, Any]) -> Tuple[float, int]:
    q = detail.get("quantization", None)
    if q and len(q) == 2 and q[0] not in (None, 0.0): return float(q[0]), int(q[1])
    qp = detail.get("quantization_parameters", None)
    if qp:
        scales = qp.get("scales", []); zeros = qp.get("zero_points", [])
        if len(scales) > 0: return float(scales[0]), int(zeros[0] if len(zeros)>0 else 0)
    return 1.0, 0

def _quantize_if_needed(x: np.ndarray, detail: Dict[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    if dtype == np.float32: return x.astype(np.float32, copy=False)
    scale, zero = _get_scale_zero(detail); scale = 1.0 if scale == 0 else scale
    xq = np.round(x / scale + zero)
    if dtype == np.int8:  return np.clip(xq, -128, 127).astype(np.int8)
    if dtype == np.uint8: return np.clip(xq, 0, 255).astype(np.uint8)
    return xq.astype(dtype)

def _dequantize_if_needed(y: np.ndarray, detail: Dict[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    if dtype == np.float32: return y.astype(np.float32, copy=False)
    scale, zero = _get_scale_zero(detail); scale = 1.0 if scale == 0 else scale
    return (y.astype(np.float32) - float(zero)) * float(scale)

def _prep_image_from_chw(x_chw: np.ndarray, orig_mean: float, orig_std: float,
                         swap_rgb: bool, norm_mode: str) -> np.ndarray:
    x01 = x_chw * float(orig_std) + float(orig_mean)
    x_hwc = np.transpose(x01, (1, 2, 0))  # BGR
    if swap_rgb: x_hwc = x_hwc[..., ::-1]  # BGR->RGB
    if norm_mode == 'neg11':
        x_hwc = (x_hwc - 0.5) / 0.5
    elif norm_mode == 'imagenet':
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std  = np.array([0.229, 0.224, 0.225], np.float32)
        x_hwc = (x_hwc - mean) / std
    return x_hwc.astype(np.float32, copy=False)

def _eval_tflite_once(tflite_path: Path, loader: DataLoader, num_classes: int,
                      orig_mean: float, orig_std: float,
                      swap_rgb: bool, norm_mode: str):
    it = _load_tflite_interpreter(tflite_path)
    in_d = it.get_input_details()[0]; out_d = it.get_output_details()[0]
    in_idx = in_d["index"]; in_shape = tuple(in_d["shape"])
    y_true, y_pred, y_prob = [], [], []
    for batch in tqdm(loader, desc=f"TFLite {tflite_path.name} [{('RGB' if swap_rgb else 'BGR')},{norm_mode}]", leave=False):
        images, labels = batch[:2]
        for i in range(images.shape[0]):
            x = images[i].cpu().numpy()
            x = _prep_image_from_chw(x, orig_mean, orig_std, swap_rgb, norm_mode)  # HWC
            if len(in_shape) == 4:
                _, H, W, C = in_shape
                if H>0 and W>0 and C>0 and x.shape[:2] != (H, W):
                    import cv2; x = cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR)
                x = x[None, ...]
            it.set_tensor(in_idx, _quantize_if_needed(x, in_d)); it.invoke()
            y = _dequantize_if_needed(it.get_tensor(out_d["index"]), out_d)
            probs = _softmax_np(y.reshape(1, -1))[0]
            y_true.append(int(labels[i].item())); y_pred.append(int(np.argmax(probs))); y_prob.append(probs)
    m = evaluate_classification(y_true, y_pred, np.array(y_prob), num_classes=num_classes)
    return m, np.array(y_pred)

def evaluate_model_tflite_auto(tflite_path: Path, loader: DataLoader, num_classes: int,
                               orig_mean: float, orig_std: float,
                               force_rgb: Optional[bool] = None, use_imagenet_norm: Optional[bool] = None):
    swap_rgb = bool(force_rgb) if force_rgb is not None else False
    norm_mode = 'imagenet' if use_imagenet_norm else 'neg11'
    m, p = _eval_tflite_once(tflite_path, loader, num_classes, orig_mean, orig_std, swap_rgb, norm_mode)
    if _degenerate(p, float(m.get("accuracy", 0.0))) and (force_rgb is None or use_imagenet_norm is None):
        cands = [(True, 'neg11'), (True, 'imagenet')]
        best_m, best = m, (swap_rgb, norm_mode); best_acc = float(m.get("accuracy", 0.0))
        for cand in cands:
            m2, _ = _eval_tflite_once(tflite_path, loader, num_classes, orig_mean, orig_std, *cand)
            acc2 = float(m2.get("accuracy", 0.0))
            if acc2 > best_acc: best_m, best, best_acc = m2, cand, acc2
        if best != (swap_rgb, norm_mode):
            print(f"[INFO] TFLite auto-selected RGB={best[0]}, norm={best[1]} (acc={best_acc:.4f})")
            m = best_m
    return m


# ===== ONNX evaluation =====
def _load_onnx_session(onnx_path: Path):
    import onnxruntime as ort  # type: ignore
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=['CPUExecutionProvider'])

def _detect_layout_from_onnx(sess) -> str:
    inp = sess.get_inputs()[0]; shp = list(inp.shape)
    try:
        if len(shp) == 4:
            c2 = int(shp[1]) if isinstance(shp[1], (int, np.integer)) else None
            cL = int(shp[-1]) if isinstance(shp[-1], (int, np.integer)) else None
            if c2 == 3: return 'nchw'
            if cL == 3: return 'nhwc'
    except Exception:
        pass
    return 'nchw'

def _eval_onnx_once(onnx_path: Path, loader: DataLoader, num_classes: int,
                    orig_mean: float, orig_std: float,
                    layout: str, swap_rgb: bool, norm_mode: str):
    sess = _load_onnx_session(onnx_path)
    inp = sess.get_inputs()[0]; out = sess.get_outputs()[0]
    in_name, out_name = inp.name, out.name
    y_true, y_pred, y_prob = [], [], []
    for batch in tqdm(loader, desc=f"ONNX {onnx_path.name} [{layout},{'RGB' if swap_rgb else 'BGR'},{norm_mode}]", leave=False):
        images, labels = batch[:2]
        for i in range(images.shape[0]):
            x_chw = images[i].cpu().numpy()
            if layout == 'nchw':
                if not swap_rgb and norm_mode == 'neg11':
                    xin = x_chw[None, ...].astype(np.float32)
                else:
                    x = _prep_image_from_chw(x_chw, orig_mean, orig_std, swap_rgb, norm_mode)  # HWC
                    xin = np.transpose(x, (2,0,1))[None, ...].astype(np.float32)
            else:
                x = _prep_image_from_chw(x_chw, orig_mean, orig_std, swap_rgb, norm_mode)
                xin = x[None, ...].astype(np.float32)
            outs = sess.run([out_name], {in_name: xin})[0]
            probs = _softmax_np(outs.reshape(1, -1))[0]
            y_true.append(int(labels[i].item())); y_pred.append(int(np.argmax(probs))); y_prob.append(probs)
    m = evaluate_classification(y_true, y_pred, np.array(y_prob), num_classes=num_classes)
    return m, np.array(y_pred)

def evaluate_model_onnx_auto(onnx_path: Path, loader: DataLoader, num_classes: int,
                             orig_mean: float, orig_std: float,
                             force_layout: Optional[str] = None,
                             force_rgb: Optional[bool] = None,
                             use_imagenet_norm: Optional[bool] = None):
    try:
        sess = _load_onnx_session(onnx_path)
    except Exception as e:
        print(f"[WARN] Skipping ONNX (cannot open): {onnx_path.name}: {e}")
        return None
    layout = force_layout or _detect_layout_from_onnx(sess)
    swap_rgb = bool(force_rgb) if force_rgb is not None else (layout == 'nhwc')
    norm_mode = 'imagenet' if use_imagenet_norm else 'neg11'
    m, p = _eval_onnx_once(onnx_path, loader, num_classes, orig_mean, orig_std, layout, swap_rgb, norm_mode)
    if _degenerate(p, float(m.get("accuracy", 0.0))) and (force_rgb is None or use_imagenet_norm is None or force_layout is None):
        cands = [(layout, True, 'neg11'), (layout, True, 'imagenet')]
        if layout == 'nchw':
            cands += [('nhwc', True, 'neg11'), ('nhwc', True, 'imagenet')]
        best_m, best = m, (layout, swap_rgb, norm_mode); best_acc = float(m.get("accuracy", 0.0))
        for cand in cands:
            m2, _ = _eval_onnx_once(onnx_path, loader, num_classes, orig_mean, orig_std, *cand)
            acc2 = float(m2.get("accuracy", 0.0))
            if acc2 > best_acc: best_m, best, best_acc = m2, cand, acc2
        if best != (layout, swap_rgb, norm_mode):
            print(f"[INFO] ONNX auto-selected layout={best[0]}, RGB={best[1]}, norm={best[2]} (acc={best_acc:.4f})")
            m = best_m
    return m


# ===== TF Keras & SavedModel evaluation =====
def _is_saved_model_dir(p: Path) -> bool:
    return p.is_dir() and (p / "saved_model.pb").exists()

def _pick_tensor_from_tf_output(out: Any, prefer_key: Optional[str] = None):
    if isinstance(out, dict):
        if prefer_key and prefer_key in out: return out[prefer_key]
        for k in ("logits","predictions","output","outputs","probs","probabilities","Identity","Softmax","dense","dense_1"):
            if k in out: return out[k]
        return next(iter(out.values()))
    return out

def _load_tf_model_file(model_path: Path):
    last_err = None
    try:
        import tensorflow as tf  # type: ignore
        with tf.keras.utils.custom_object_scope({'TFOpLambda': tf.keras.layers.Lambda}):
            # Keras3 互換: safe_mode が存在すれば False に
            try:
                return tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
            except TypeError:
                return tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as e:
        last_err = e
    try:
        import keras  # type: ignore
        with keras.utils.custom_object_scope({'TFOpLambda': keras.layers.Lambda}):
            try:
                return keras.models.load_model(str(model_path), compile=False, safe_mode=False)
            except TypeError:
                return keras.models.load_model(str(model_path), compile=False)
    except Exception as e:
        last_err = e
    raise last_err

def _load_tf_savedmodel_as_layer(saved_dir: Path, call_endpoint: Optional[str] = "serving_default"):
    """
    Return a callable (x, **kwargs) -> np.ndarray
    """
    import numpy as np
    try:
        import keras  # type: ignore
        from keras.layers import TFSMLayer  # type: ignore
        layer = TFSMLayer(str(saved_dir), call_endpoint=call_endpoint or "serving_default")
        def _call(x, **kwargs):
            y = layer(x)  # training は渡さない（TFSMLayer は未対応のことがある）
            y = _pick_tensor_from_tf_output(y, None)
            return np.array(y) if hasattr(y, "numpy") else y
        return _call
    except Exception:
        import tensorflow as tf  # type: ignore
        sm = tf.saved_model.load(str(saved_dir))
        fn = sm.signatures.get(call_endpoint or "serving_default", None)
        if fn is None and len(sm.signatures) > 0:
            fn = list(sm.signatures.values())[0]
        if fn is None:
            raise RuntimeError("No callable signature found in SavedModel.")
        def _call(x, **kwargs):
            y = fn(tf.convert_to_tensor(x))
            y = _pick_tensor_from_tf_output(y, None)
            return y.numpy() if hasattr(y, "numpy") else np.array(y)
        return _call

def _eval_tf_once(model_or_callable, loader: DataLoader, num_classes: int,
                  orig_mean: float, orig_std: float,
                  swap_rgb: bool, norm_mode: str,
                  output_key: Optional[str] = None):
    """
    model_or_callable: Keras Model もしくは Python callable
    - まず training=False 付きで呼び、TypeError なら引数なしで再試行
    """
    y_true, y_pred, y_prob = [], [], []
    for batch in tqdm(loader, desc=f"TF [{('RGB' if swap_rgb else 'BGR')},{norm_mode}]", leave=False):
        images, labels = batch[:2]
        for i in range(images.shape[0]):
            x_chw = images[i].cpu().numpy()
            x = _prep_image_from_chw(x_chw, orig_mean, orig_std, swap_rgb, norm_mode)  # HWC
            x = x[None, ...].astype(np.float32)
            try:
                y = model_or_callable(x, training=False)
            except TypeError:
                y = model_or_callable(x)
            y = _pick_tensor_from_tf_output(y, output_key)
            y = np.array(y)
            probs = _softmax_np(y.reshape(1, -1))[0]
            y_true.append(int(labels[i].item())); y_pred.append(int(np.argmax(probs))); y_prob.append(probs)
    m = evaluate_classification(y_true, y_pred, np.array(y_prob), num_classes=num_classes)
    return m, np.array(y_pred)

def evaluate_model_tf_auto(model_path: Path, loader: DataLoader, num_classes: int,
                           orig_mean: float, orig_std: float,
                           force_rgb: Optional[bool] = None,
                           use_imagenet_norm: Optional[bool] = None,
                           call_endpoint: Optional[str] = None,
                           output_key: Optional[str] = None):
    try:
        if _is_saved_model_dir(model_path):
            model = _load_tf_savedmodel_as_layer(model_path, call_endpoint or "serving_default")
        else:
            # .h5 / .keras のみ対応。それ以外の拡張子はここに来ない
            model = _load_tf_model_file(model_path)
    except Exception as e:
        print(f"[WARN] Skipping TF model (cannot load): {model_path.name}: {e}")
        return None
    swap_rgb = bool(force_rgb) if force_rgb is not None else True  # TF系は基本RGB
    norm_mode = 'imagenet' if use_imagenet_norm else 'neg11'
    m, p = _eval_tf_once(model, loader, num_classes, orig_mean, orig_std, swap_rgb, norm_mode, output_key)
    if _degenerate(p, float(m.get("accuracy", 0.0))) and (force_rgb is None or use_imagenet_norm is None):
        cands = [(True, 'imagenet'), (True, 'neg11')]
        best_m, best = m, (swap_rgb, norm_mode); best_acc = float(m.get("accuracy", 0.0))
        for cand in cands:
            m2, _ = _eval_tf_once(model, loader, num_classes, orig_mean, orig_std, *cand, output_key)
            acc2 = float(m2.get("accuracy", 0.0))
            if acc2 > best_acc: best_m, best, best_acc = m2, cand, acc2
        if best != (swap_rgb, norm_mode):
            print(f"[INFO] TF auto-selected RGB={best[0]}, norm={best[1]} (acc={best_acc:.4f})")
            m = best_m
    return m


# ===== Main =====
def main():
    ap = argparse.ArgumentParser(description="Evaluate Torch/TFLite/ONNX/TF models and write metrics to CSV.")
    ap.add_argument("--config", required=True, help="YAML used in training (experiment format)")
    ap.add_argument("--weights_dir", required=True, help="Directory containing models")
    ap.add_argument("--out_csv", default="test_results.csv")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Optional hints (TFLite / TF / ONNX)
    ap.add_argument("--tflite_force_rgb", action="store_true")
    ap.add_argument("--tflite_imagenet_norm", action="store_true")
    ap.add_argument("--tf_force_rgb", action="store_true")
    ap.add_argument("--tf_imagenet_norm", action="store_true")
    ap.add_argument("--tf_call_endpoint", default="serving_default",
                    help="SavedModel call endpoint name for TFSMLayer (default: serving_default)")
    ap.add_argument("--tf_output_key", default="",
                    help="If TF outputs a dict, pick this key if present (default: auto)")
    ap.add_argument("--onnx_force_rgb", action="store_true")
    ap.add_argument("--onnx_imagenet_norm", action="store_true")
    ap.add_argument("--onnx_force_nhwc", action="store_true", help="Force ONNX layout as NHWC (default: auto/NCHW)")
    args = ap.parse_args()

    cfg_list = load_all_training_configs(args.config)
    if not cfg_list: raise RuntimeError(f"No training configs parsed from: {args.config}")

    device = torch.device(args.device)
    out_rows: List[Dict[str, Any]] = []
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)

    wdir = Path(args.weights_dir)
    # gather files
    model_paths: List[Path] = []
    model_paths += sorted(list(wdir.glob("*.pth")) + list(wdir.glob("*.pt")))
    model_paths += sorted(list(wdir.glob("*.tflite")))
    model_paths += sorted(list(wdir.glob("*.onnx")))
    model_paths += sorted(list(wdir.glob("*.h5")) + list(wdir.glob("*.keras")))
    # SavedModel directoriesのみ追加
    model_paths += sorted([p for p in wdir.iterdir() if _is_saved_model_dir(p)])

    if not model_paths:
        print(f"[WARN] No model found in: {wdir}")

    for mpath in model_paths:
        suffix = mpath.suffix.lower()
        # ===== Choose config =====
        chosen_cfg = None; ckpt = None
        if suffix in [".pth", ".pt"]:
            ckpt = torch.load(str(mpath), map_location="cpu")
            chosen_cfg = pick_cfg_for_ckpt(cfg_list, ckpt) or cfg_list[0]
        else:
            chosen_cfg = cfg_list[0]

        # ===== Dataset / Loaders =====
        image_size = tuple(getattr(chosen_cfg, "image_size", (480, 480)))
        mean = float(getattr(chosen_cfg, "normalize_mean", 0.5)) if hasattr(chosen_cfg, "normalize_mean") else 0.5
        std  = float(getattr(chosen_cfg, "normalize_std", 0.5))  if hasattr(chosen_cfg, "normalize_std")  else 0.5
        tfm = SimpleTransform(size_hw=image_size, mean=mean, std=std)

        valid_csv = getattr(chosen_cfg, "valid_file_dir", None)
        valid_img_dir = getattr(chosen_cfg, "valid_img_dir", "")
        if valid_csv is None or not Path(valid_csv).exists():
            raise FileNotFoundError(f"valid_file_dir not found in cfg: {valid_csv}")

        ds = SignalMixClassificationDataset(img_dir=valid_img_dir, annotation_csv=valid_csv, transform=tfm)

        bs = args.batch_size or int(getattr(chosen_cfg, "batch_size", 16))
        pin = (device.type == "cuda")
        loader_torch = DataLoader(ds, batch_size=bs, shuffle=False,
                                  num_workers=int(getattr(chosen_cfg, "num_workers", 4)),
                                  pin_memory=pin, collate_fn=signalmix_collate, drop_last=False)
        loader1 = DataLoader(ds, batch_size=1, shuffle=False,
                             num_workers=int(getattr(chosen_cfg, "num_workers", 2)),
                             pin_memory=False, collate_fn=signalmix_collate, drop_last=False)

        num_classes = int(getattr(chosen_cfg, "num_classes", getattr(ds, "num_classes", 3)))
        task = "classification"  # SignalMix is classification

        # ===== Evaluate by type =====
        row = {"model_file": mpath.name, "model_name": getattr(chosen_cfg, "model_name", mpath.stem), "task": task}

        if suffix in [".pth", ".pt"]:
            model, task_eff, ncls_eff, model_name = build_model_from_cfg_via_builder(chosen_cfg, ckpt, getattr(ds,"num_classes",None))
            model = model.to(device); load_state_dict_safely(model, ckpt)
            metrics = evaluate_model_torch(model, loader_torch, device, task_eff, ncls_eff)
            row["model_name"] = model_name
            row.update(metrics)
            print(f"[OK][Torch] {mpath.name} -> acc: {row.get('accuracy','NA')}")

        elif suffix == ".tflite":
            metrics = evaluate_model_tflite_auto(
                tflite_path=mpath, loader=loader1, num_classes=num_classes,
                orig_mean=mean, orig_std=std,
                force_rgb=args.tflite_force_rgb if hasattr(args, "tflite_force_rgb") else None,
                use_imagenet_norm=args.tflite_imagenet_norm if hasattr(args, "tflite_imagenet_norm") else None
            )
            if metrics is None: metrics = {}
            row.update(metrics or {})
            print(f"[OK][TFLite] {mpath.name} -> acc: {row.get('accuracy','NA')}")

        elif suffix == ".onnx":
            try:
                import onnxruntime  # noqa: F401
            except Exception as e:
                print(f"[WARN] ONNXRuntime not available, skipping {mpath.name}: {e}")
                continue
            metrics = evaluate_model_onnx_auto(
                onnx_path=mpath, loader=loader1, num_classes=num_classes,
                orig_mean=mean, orig_std=std,
                force_layout=('nhwc' if args.onnx_force_nhwc else None),
                force_rgb=args.onnx_force_rgb if hasattr(args, "onnx_force_rgb") else None,
                use_imagenet_norm=args.onnx_imagenet_norm if hasattr(args, "onnx_imagenet_norm") else None
            )
            if metrics is None: metrics = {}
            row.update(metrics or {})
            print(f"[OK][ONNX] {mpath.name} -> acc: {row.get('accuracy','NA')}")

        elif suffix in [".h5", ".keras"] or _is_saved_model_dir(mpath):
            # TF: Keras(.h5/.keras) or SavedModel dir のみ対応
            try:
                import tensorflow as tf  # noqa: F401
            except Exception as e:
                print(f"[WARN] TensorFlow not available, skipping {mpath.name}: {e}")
                continue
            metrics = evaluate_model_tf_auto(
                model_path=mpath, loader=loader1, num_classes=num_classes,
                orig_mean=mean, orig_std=std,
                force_rgb=args.tf_force_rgb if hasattr(args, "tf_force_rgb") else None,
                use_imagenet_norm=args.tf_imagenet_norm if hasattr(args, "tf_imagenet_norm") else None,
                call_endpoint=args.tf_call_endpoint if hasattr(args, "tf_call_endpoint") else "serving_default",
                output_key=args.tf_output_key if hasattr(args, "tf_output_key") and args.tf_output_key else None
            )
            if metrics is None: metrics = {}
            row.update(metrics or {})
            print(f"[OK][TF] {mpath.name} -> acc: {row.get('accuracy','NA')}")

        else:
            print(f"[WARN] Unsupported file type, skipping: {mpath.name}")
            continue

        out_rows.append(row)

    # ===== Write CSV =====
    if out_rows:
        keys = list(sorted({k for r in out_rows for k in r.keys()}))
        head = ["model_file", "model_name", "task"] + [k for k in keys if k not in ("model_file","model_name","task")]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=head); w.writeheader()
            for r in out_rows: w.writerow({k: r.get(k, "") for k in head})
    else:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            f.write("model_file,model_name,task\n")
    print(f"[DONE] wrote: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
