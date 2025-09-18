#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified evaluator for classification / regression across PyTorch / ONNX / TFLite / Keras.

- Uses user-provided evaluate_classification / evaluate_regression to compute metrics
- Robust checkpoint unwrapping:
    * recursively extract real state_dict from common wrappers
    * if 'state_dict' shows up as an unexpected key, retry with ckpt['state_dict']
- Pick config by filename: exact > prefix(with delimiter) > substring, then longest match
- Task via CLI: --task {classification,regression}
- Supports: .pth/.pt, .onnx, .tflite, .h5/.keras or SavedModel dir
- CSV output per model file
- Windows-friendly DataLoader (default workers=0) and top-level transform to avoid pickling errors.

Usage:
  python src/test.py --task classification --config src/config/test.yaml --weights_dir weight/test
"""

from __future__ import annotations
import argparse
import csv
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- import path: add src/ to sys.path ---
import sys
THIS = Path(__file__).resolve()
SRC  = THIS.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- project modules ---
from config.config import load_all_training_configs                 # YAML loader
from models.model import get_model                                  # model builder
from dataset_signalmix import SignalMixClassificationDataset        # classification dataset
from dataset import SignalSlopeDataset                               # regression dataset
from metrics.metrics import evaluate_classification, evaluate_regression

# ====== Top-level transforms (Windows pickling-safe) ======
class ClassifyTransform:
    """Resize -> to CHW float32 in [-1,1] (mean=std=0.5 by default)."""
    def __init__(self, size_hw: Tuple[int, int], mean=0.5, std=0.5):
        self.H, self.W = int(size_hw[0]), int(size_hw[1])
        self.mean = float(mean); self.std = float(std)
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        import cv2
        img = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t = (t - self.mean) / self.std
        return t

class EvalTransform:
    """Dataset 側が期待する base_transform(image)->{'image':tensor} を提供。"""
    def __init__(self, size_hw: Tuple[int,int], mean=0.5, std=0.5):
        self._tfm = ClassifyTransform(size_hw, mean, std)
    def base_transform(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        return {"image": self._tfm(image)}

class RegrTransformAdapter:
    """SignalSlopeDataset 期待の (image_np, slope)->(tensor, slope)。"""
    def __init__(self, size_hw: Tuple[int,int], mean=0.5, std=0.5):
        self._tfm = ClassifyTransform(size_hw, mean, std)
    def __call__(self, image_np: np.ndarray, slope_deg: float):
        return self._tfm(image_np), slope_deg

def _collate_xy(batch):
    imgs, ys = [], []
    for b in batch:
        imgs.append(b[0]); ys.append(b[1])
    return torch.stack(imgs, 0), torch.stack(ys, 0) if torch.is_tensor(ys[0]) else torch.as_tensor(ys)

# ====== File / cfg helpers ======
def _is_saved_model_dir(p: Path) -> bool:
    return p.is_dir() and (p / "saved_model.pb").exists()

def _gather_models(wdir: Path) -> List[Path]:
    files: List[Path] = []
    files += sorted(list(wdir.glob("*.pth")) + list(wdir.glob("*.pt")))
    files += sorted(list(wdir.glob("*.onnx")))
    files += sorted(list(wdir.glob("*.tflite")))
    files += sorted(list(wdir.glob("*.h5")) + list(wdir.glob("*.keras")))
    files += [p for p in wdir.iterdir() if _is_saved_model_dir(p)]
    return files

def _score_name_match(cfg_name: str, stem: str) -> Tuple[int, int]:
    """Return (rank, length). Higher is better. 3 exact, 2 prefix(delimited), 1 substring, 0 no match."""
    cfg = cfg_name.strip().lower()
    s   = stem.strip().lower()
    if s == cfg:
        return (3, len(cfg))
    if s.startswith(cfg) and (len(s) == len(cfg) or s[len(cfg)] in ('.', '-', '_')):
        return (2, len(cfg))
    if cfg in s:
        return (1, len(cfg))
    return (0, 0)

def _pick_cfg(cfgs, task: str, filename_stem: str):
    task = str(task).lower()
    scored: List[Tuple[Tuple[int,int], Any]] = []
    for c in cfgs:
        c_task = str(getattr(c, "task", task)).lower()
        if c_task != task:
            continue
        c_name = str(getattr(c, "model_name", ""))
        score  = _score_name_match(c_name, filename_stem)
        if score[0] > 0:
            scored.append((score, c))
    if scored:
        scored.sort(key=lambda x: (x[0][0], x[0][1]), reverse=True)
        return scored[0][1]
    for c in cfgs:
        if str(getattr(c, "task", task)).lower() == task:
            return c
    return cfgs[0]

def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        out[nk] = v
    return out

# ====== Checkpoint unwrapping ======
def _looks_like_state_dict(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    hit_dot = 0
    hit_tensor = 0
    for k, v in d.items():
        if isinstance(k, str) and ('.' in k or k.endswith(('weight','bias'))):
            hit_dot += 1
        if torch.is_tensor(v) or isinstance(v, (torch.nn.Parameter,)):
            hit_tensor += 1
    return (hit_dot >= 3 and hit_tensor >= 3)

def _extract_state_dict_like(obj: Any, depth: int = 0) -> Optional[Dict[str, Any]]:
    if obj is None or depth > 4:
        return None
    if isinstance(obj, dict) and _looks_like_state_dict(obj):
        return obj
    if isinstance(obj, dict):
        for key in ["state_dict", "model", "model_state", "model_state_dict",
                    "module", "net", "weights", "params"]:
            if key in obj:
                got = _extract_state_dict_like(obj[key], depth + 1)
                if got is not None:
                    return got
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        try:
            sd = obj.state_dict()
            if isinstance(sd, dict) and _looks_like_state_dict(sd):
                return sd
        except Exception:
            pass
    return None

# ====== Common helpers ======
def _softmax_np(y):
    y = np.array(y)
    y = y - y.max(axis=-1, keepdims=True)
    e = np.exp(y)
    return e / e.sum(axis=-1, keepdims=True)

def _load_state_flex(model: torch.nn.Module, state_in: Dict[str, Any]) -> Dict[str, Any]:
    own = model.state_dict()
    loaded = _strip_module_prefix(state_in)
    missing = [k for k in own.keys() if k not in loaded]
    unexpected = [k for k in loaded.keys() if k not in own]
    mismatched = []
    filtered = {}
    for k, v in loaded.items():
        if k in own and own[k].shape != v.shape:
            mismatched.append((k, tuple(v.shape), tuple(own[k].shape)))
        elif k in own:
            filtered[k] = v
    msg = (f"[partial load] loaded={len(filtered)} missing={len(missing)} "
           f"mismatched={len(mismatched)} unexpected={len(unexpected)}")
    if mismatched:
        ex = ", ".join([f"('{k}', {src}, {dst})" for k, src, dst in mismatched[:3]])
        msg += f"\n  mismatched examples: [{ex}]"
    print(msg)
    model.load_state_dict(filtered, strict=False)
    return {
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
        "loaded_count": len(filtered),
    }

# ====== Framework runners ======
def run_pytorch(pth: Path, task: str, cfg, num_classes: int, device: torch.device,
                loader: DataLoader) -> Dict[str, Any]:
    kwargs = dict(dropout_rate=getattr(cfg, "dropout_rate", 0.0),
                  drop_path_rate=getattr(cfg, "drop_path_rate", 0.0))
    if task == "classification":
        kwargs["num_classes"] = int(num_classes)
    model = get_model(task, getattr(cfg, "model_name", ""), cfg=cfg, **kwargs).to(device)

    ckpt = torch.load(str(pth), map_location="cpu", weights_only=False)
    state = _extract_state_dict_like(ckpt)
    if state is None:
        state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    summary = _load_state_flex(model, state)
    if "state_dict" in summary["unexpected"] and isinstance(ckpt, dict) and isinstance(ckpt.get("state_dict", None), dict):
        print("[info] Retrying with ckpt['state_dict'] because 'state_dict' was unexpected.")
        summary = _load_state_flex(model, ckpt["state_dict"])

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    y_reg_true, y_reg_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"[Torch] {pth.name}", leave=False):
            images = images.to(device, non_blocking=True)
            if task == "classification":
                logits = model(images).float()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(np.argmax(probs, axis=1).tolist())
                y_prob.extend(probs)
            else:
                out = model(images).float().squeeze()
                y_reg_true.extend(labels.cpu().view(-1).tolist())
                y_reg_pred.extend(out.cpu().view(-1).tolist())

    if task == "classification":
        y_prob_np = np.asarray(y_prob) if len(y_prob) else np.zeros((0, int(num_classes or 1)))
        ncls_eff = int(num_classes or (y_prob_np.shape[1] if y_prob_np.size else 1))
        return evaluate_classification(y_true, y_pred, y_prob_np, ncls_eff)
    return evaluate_regression(y_reg_true, y_reg_pred)

def _detect_onnx_layout(sess) -> str:
    inp = sess.get_inputs()[0]; shp = list(inp.shape)
    try:
        if len(shp) == 4:
            c2 = int(shp[1]) if isinstance(shp[1], (int, np.integer)) else None
            cL = int(shp[-1]) if isinstance(shp[-1], (int, np.integer)) else None
            if c2 == 3: return "nchw"
            if cL == 3: return "nhwc"
    except Exception:
        pass
    return "nchw"

def run_onnx(onnx_path: Path, task: str, layout_opt: str,
             rgb: bool, loader1: DataLoader, num_classes: Optional[int]) -> Optional[Dict[str, Any]]:
    try:
        import onnxruntime as ort
    except Exception as e:
        print(f"[WARN] ONNXRuntime not available: {e}")
        return None
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    layout = _detect_onnx_layout(sess) if layout_opt == "auto" else layout_opt
    inp = sess.get_inputs()[0].name; out = sess.get_outputs()[0].name

    y_true, y_pred, y_prob = [], [], []
    y_reg_true, y_reg_pred = [], []

    for images, labels in tqdm(loader1, desc=f"[ONNX] {onnx_path.name}", leave=False):
        x = images[0].cpu().numpy()   # CHW, [-1,1]
        if layout == "nchw":
            xin = x[None, ...].astype(np.float32)
            if rgb:  # BGR->RGB
                xin = xin[:, ::-1, ...]
        else:
            x_hwc = np.transpose(x, (1, 2, 0)).astype(np.float32)
            if rgb: x_hwc = x_hwc[..., ::-1]
            xin = x_hwc[None, ...]
        y = sess.run([out], {inp: xin})[0]
        if task == "classification":
            probs = _softmax_np(y)
            y_true.append(int(labels[0].item()))
            y_pred.append(int(np.argmax(probs)))
            y_prob.append(probs.reshape(-1))
        else:
            y_reg_true.append(float(labels.view(-1)[0].item()))
            y_reg_pred.append(float(np.array(y).reshape(-1)[0]))

    if task == "classification":
        y_prob_np = np.vstack(y_prob) if len(y_prob) else np.zeros((0, int(num_classes or 1)))
        ncls_eff = int(num_classes or (y_prob_np.shape[1] if y_prob_np.size else 1))
        return evaluate_classification(y_true, y_pred, y_prob_np, ncls_eff)
    return evaluate_regression(y_reg_true, y_reg_pred)

def _tflite_interpreter(path: Path):
    try:
        from tflite_runtime.interpreter import Interpreter
        it = Interpreter(model_path=str(path)); it.allocate_tensors(); return it
    except Exception:
        import tensorflow as tf
        it = tf.lite.Interpreter(model_path=str(path)); it.allocate_tensors(); return it

def run_tflite(tfl: Path, task: str, rgb: bool, loader1: DataLoader, num_classes: Optional[int]) -> Optional[Dict[str, Any]]:
    try:
        it = _tflite_interpreter(tfl)
    except Exception as e:
        print(f"[WARN] TFLite interpreter not available: {e}")
        return None
    in_d = it.get_input_details()[0]; out_d = it.get_output_details()[0]
    in_idx = in_d["index"]; out_idx = out_d["index"]
    in_shape = tuple(in_d["shape"])  # [1,H,W,C] or [1,C,H,W]
    nhwc = (len(in_shape) == 4 and in_shape[-1] in (1,3))

    def _q(x: np.ndarray, d: Dict[str, Any]) -> np.ndarray:
        if d["dtype"] == np.float32: return x.astype(np.float32)
        qp = d.get("quantization_parameters", {})
        scales = qp.get("scales", [1.0]); zeros = qp.get("zero_points", [0])
        scale, zero = float(scales[0] if len(scales)>0 else 1.0), int(zeros[0] if len(zeros)>0 else 0)
        xq = np.round(x / (scale if scale!=0 else 1.0) + zero)
        if d["dtype"] == np.int8:  return np.clip(xq, -128, 127).astype(np.int8)
        if d["dtype"] == np.uint8: return np.clip(xq, 0, 255).astype(np.uint8)
        return xq.astype(d["dtype"])

    y_true, y_pred, y_prob = [], [], []
    y_reg_true, y_reg_pred = [], []

    for images, labels in tqdm(loader1, desc=f"[TFLite] {tfl.name}", leave=False):
        x = images[0].cpu().numpy()  # CHW, [-1,1]
        if nhwc:
            x = np.transpose(x, (1, 2, 0))
            if rgb: x = x[..., ::-1]
            x = x[None, ...].astype(np.float32)
        else:
            x = x[None, ...].astype(np.float32)
            if rgb: x = x[:, ::-1, ...]
        it.set_tensor(in_idx, _q(x, in_d))
        it.invoke()
        y = it.get_tensor(out_idx)
        if task == "classification":
            probs = _softmax_np(y)
            y_true.append(int(labels[0].item()))
            y_pred.append(int(np.argmax(probs)))
            y_prob.append(probs.reshape(-1))
        else:
            y_reg_true.append(float(labels.view(-1)[0].item()))
            y_reg_pred.append(float(np.array(y).reshape(-1)[0]))
    if task == "classification":
        y_prob_np = np.vstack(y_prob) if len(y_prob) else np.zeros((0, int(num_classes or 1)))
        ncls_eff = int(num_classes or (y_prob_np.shape[1] if y_prob_np.size else 1))
        return evaluate_classification(y_true, y_pred, y_prob_np, ncls_eff)
    return evaluate_regression(y_reg_true, y_reg_pred)

def run_keras(model_path: Path, task: str, rgb: bool, loader1: DataLoader, num_classes: Optional[int]) -> Optional[Dict[str, Any]]:
    try:
        import tensorflow as tf
    except Exception as e:
        print(f"[WARN] TensorFlow not available: {e}")
        return None

    model = None
    call = None
    if _is_saved_model_dir(model_path):
        sm = tf.saved_model.load(str(model_path))
        fn = sm.signatures.get("serving_default", None)
        if fn is None and len(sm.signatures)>0:
            fn = list(sm.signatures.values())[0]
        if fn is None:
            print("[WARN] No callable signature in SavedModel.")
            return None
        def call(x):
            y = fn(tf.convert_to_tensor(x))
            if isinstance(y, dict): y = list(y.values())[0]
            return y.numpy()
    else:
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
        except Exception:
            model = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
        def call(x):
            y = model(x, training=False)
            if isinstance(y, dict): y = list(y.values())[0]
            return y.numpy()

    y_true, y_pred, y_prob = [], [], []
    y_reg_true, y_reg_pred = [], []

    for images, labels in tqdm(loader1, desc=f"[TF] {model_path.name}", leave=False):
        x = images[0].cpu().numpy()  # CHW [-1,1]
        x = np.transpose(x, (1,2,0))  # -> HWC
        if rgb: x = x[..., ::-1]
        y = call(x[None, ...].astype(np.float32))
        if task == "classification":
            probs = _softmax_np(y)
            y_true.append(int(labels[0].item()))
            y_pred.append(int(np.argmax(probs)))
            y_prob.append(probs.reshape(-1))
        else:
            y_reg_true.append(float(labels.view(-1)[0].item()))
            y_reg_pred.append(float(np.array(y).reshape(-1)[0]))

    if task == "classification":
        y_prob_np = np.vstack(y_prob) if len(y_prob) else np.zeros((0, int(num_classes or 1)))
        ncls_eff = int(num_classes or (y_prob_np.shape[1] if y_prob_np.size else 1))
        return evaluate_classification(y_true, y_pred, y_prob_np, ncls_eff)
    return evaluate_regression(y_reg_true, y_reg_pred)

# ====== Main ======
def main():
    ap = argparse.ArgumentParser(description="Unified simple tester")
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--out_csv", default="test_results.csv")

    ap.add_argument("--image_size", type=int, default=None, help="Square size (fallback: YAML image_size or 320)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=None, help="None: auto (Windows=0, else=2)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_classes", type=int, default=None, help="classification only (fallback: YAML / infer)")

    ap.add_argument("--mean", type=float, default=0.5)
    ap.add_argument("--std",  type=float, default=0.5)

    ap.add_argument("--onnx_layout", choices=["auto","nchw","nhwc"], default="auto")
    ap.add_argument("--tf_rgb",     action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tflite_rgb", action=argparse.BooleanOptionalAction, default=True)

    args = ap.parse_args()

    # Load YAML configs
    cfgs = load_all_training_configs(args.config)

    # Collect model files
    wdir = Path(args.weights_dir)
    files = _gather_models(wdir)
    if not files:
        print(f"[WARN] no model files under: {wdir}")
        return

    # Output CSV
    out_path = Path(args.out_csv); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict[str, Any]] = []

    # workers(auto)
    if args.num_workers is None:
        is_win = (platform.system().lower() == "windows")
        num_workers = 0 if is_win else 2
    else:
        num_workers = max(0, int(args.num_workers))

    device = torch.device(args.device)

    for mpath in files:
        stem = mpath.stem  # extension removed; dots in name remain
        cfg = _pick_cfg(cfgs, args.task, stem)

        # image size
        if args.image_size is not None:
            H = W = int(args.image_size)
        else:
            img_sz = getattr(cfg, "image_size", (320, 320))
            H = int(img_sz[0] if isinstance(img_sz, (list,tuple)) else img_sz)
            W = int(img_sz[1] if isinstance(img_sz, (list,tuple)) else img_sz)
        size_hw = (H, W)

        # datasets
        if args.task == "classification":
            ds = SignalMixClassificationDataset(
                img_dir=getattr(cfg, "valid_img_dir", ""),
                annotation_csv=getattr(cfg, "valid_file_dir", ""),
                transform=EvalTransform(size_hw, args.mean, args.std),
                is_train=False
            )
            ncls = int(args.num_classes or getattr(cfg, "num_classes", getattr(ds, "num_classes", 3)))
        else:
            ds = SignalSlopeDataset(
                csv_path=getattr(cfg, "valid_file_dir", ""),
                image_dir=getattr(cfg, "valid_img_dir", ""),
                task="regression",
                transform=RegrTransformAdapter(size_hw, args.mean, args.std),
                shuffle=False
            )
            ncls = 0

        # loaders
        pin = device.type == "cuda"
        loader_bs = DataLoader(ds, batch_size=max(1, args.batch_size), shuffle=False,
                               num_workers=num_workers, pin_memory=pin, collate_fn=_collate_xy)
        loader_1  = DataLoader(ds, batch_size=1, shuffle=False,
                               num_workers=num_workers, pin_memory=False, collate_fn=_collate_xy)

        row: Dict[str, Any] = {
            "model_file": mpath.name,
            "task": args.task,
            "model_name": getattr(cfg, "model_name", "")
        }

        suffix = mpath.suffix.lower()
        if suffix in [".pth", ".pt"]:
            metrics = run_pytorch(mpath, args.task, cfg, ncls, device, loader_bs)
            row.update(metrics or {})
            print(f"[OK][Torch] {mpath.name}  ->  {row}")

        elif suffix == ".onnx":
            metrics = run_onnx(mpath, args.task, args.onnx_layout, rgb=False, loader1=loader_1, num_classes=ncls if ncls else None)
            row.update(metrics or {})
            print(f"[OK][ONNX]  {mpath.name}  ->  {row}")

        elif suffix == ".tflite":
            metrics = run_tflite(mpath, args.task, rgb=args.tflite_rgb, loader1=loader_1, num_classes=ncls if ncls else None)
            row.update(metrics or {})
            print(f"[OK][TFL]   {mpath.name}  ->  {row}")

        elif suffix in [".h5", ".keras"] or _is_saved_model_dir(mpath):
            metrics = run_keras(mpath, args.task, rgb=args.tf_rgb, loader1=loader_1, num_classes=ncls if ncls else None)
            row.update(metrics or {})
            print(f"[OK][TF]    {mpath.name}  ->  {row}")

        else:
            print(f("[SKIP] Unsupported: {mpath.name}"))
            continue

        out_rows.append(row)

    # write CSV
    keys = sorted({k for r in out_rows for k in r.keys()})
    head = ["model_file", "model_name", "task"] + [k for k in keys if k not in ("model_file","model_name","task")]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=head)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (r.get(k, "").tolist() if hasattr(r.get(k, ""), "tolist") else r.get(k, "")) for k in head})
    print(f"[DONE] wrote: {out_path.resolve()}")

if __name__ == "__main__":
    main()
