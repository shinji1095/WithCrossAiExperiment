#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test.py (Torch + TFLite, robust preprocessing)
- Evaluate *.pth/*.pt/*.tflite and write metrics to CSV.
- Uses the same builder as training (models.model.get_model) for Torch.
- TFLite path now matches training preprocessing and can auto-recover
  from RGB/BGR mismatch (and optionally try ImageNet mean/std).
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


# ===== Transforms & Collate (aligned with training) =====
class SimpleTransform:
    """
    Training-consistent preprocessing:
      - Optional resize to (H, W)
      - ToTensor (0..1) -> Normalize(mean, std)  (default mean=std=0.5, i.e. -1..1)
    """
    def __init__(self, size_hw=None, mean=0.5, std=0.5):
        self.size_hw = tuple(size_hw) if size_hw is not None else None  # (H, W)
        self.mean = float(mean)
        self.std = float(std)

    def base_transform(self, image):
        import cv2
        import torch
        if self.size_hw is not None:
            H, W = self.size_hw
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        t = (t - self.mean) / self.std
        return {"image": t}


def signalmix_collate(batch):
    if len(batch[0]) == 3:
        imgs, labels, bboxes = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return imgs, labels, list(bboxes)
    else:
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return imgs, labels


# ===== Utility =====
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


def pick_cfg_for_ckpt(cfg_list, ckpt: Dict[str, Any]):
    ck_name = str(ckpt.get("model_name", "") or "").lower()
    ck_task = str(ckpt.get("task", "") or "").lower()
    ck_ncls = _safe_int(ckpt.get("num_classes", None), -1)

    def score(c):
        s = 0
        if ck_name and ck_name == str(getattr(c, "model_name", "")).lower():
            s += 2
        if ck_task and ck_task == str(getattr(c, "task", "")).lower():
            s += 1
        if ck_ncls > 0 and ck_ncls == int(getattr(c, "num_classes", 3)):
            s += 1
        return s

    return max(cfg_list, key=score) if cfg_list else None


def build_model_from_cfg_via_builder(
    cfg,
    override_ckpt: Optional[Dict[str, Any]] = None,
    ds_num_classes: Optional[int] = None
):
    ck = override_ckpt or {}

    if "model_name" in ck and ck["model_name"]:
        cfg.model_name = ck["model_name"]
    if "task" in ck and ck["task"]:
        cfg.task = ck["task"]

    cfg.num_classes = _safe_int(
        _first_not_none(ck.get("num_classes"), getattr(cfg, "num_classes", None), ds_num_classes, 3),
        3
    )

    model = get_model(
        cfg.task,
        cfg.model_name,
        num_classes=int(cfg.num_classes),
        dropout_rate=getattr(cfg, "dropout_rate", 0.0),
        drop_path_rate=getattr(cfg, "drop_path_rate", 0.0),
    )
    return model, str(cfg.task).lower(), int(cfg.num_classes), str(cfg.model_name)


def load_state_dict_safely(model: torch.nn.Module, ckpt: Dict[str, Any]) -> None:
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[WARN] missing keys: {missing[:12]}{'...' if len(missing)>12 else ''}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected[:12]}{'...' if len(unexpected)>12 else ''}")


# ===== Torch Evaluation =====
def evaluate_model_torch(model, loader, device, task: str, num_classes: int):
    model.eval()
    y_true_cls, y_pred_cls, y_prob_cls = [], [], []
    y_true_reg, y_pred_reg = [], []

    autocast_enabled = (device.type == "cuda")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing (Torch)", leave=False):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            images = images.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', enabled=autocast_enabled):
                out = model(images)

            if task == "multitask":
                signal_pred, slope_pred = out
                logits = signal_pred.float()
                y_true_cls.extend(labels.cpu().tolist())
                y_pred_cls.extend(logits.argmax(dim=1).cpu().tolist())
                y_prob_cls.extend(torch.softmax(logits, dim=1).cpu().numpy())
                y_pred_reg.extend(torch.tanh(slope_pred).flatten().cpu().tolist())
            elif task == "classification":
                logits = out.float()
                y_true_cls.extend(labels.cpu().tolist())
                y_pred_cls.extend(logits.argmax(dim=1).cpu().tolist())
                y_prob_cls.extend(torch.softmax(logits, dim=1).cpu().numpy())
            else:  # regression
                y_true_reg.extend(labels.cpu().tolist())
                y_pred_reg.extend(torch.tanh(out).flatten().cpu().tolist())

    if task == "classification":
        return evaluate_classification(y_true_cls, y_pred_cls, np.array(y_prob_cls), num_classes=num_classes)
    elif task == "regression":
        return evaluate_regression(y_true_reg, y_pred_reg)
    else:
        cls_metrics = {}
        if len(set(y_true_cls)) >= 2:
            cls_metrics = evaluate_classification(y_true_cls, y_pred_cls, np.array(y_prob_cls), num_classes=num_classes)
        reg_metrics = evaluate_regression(y_true_reg, y_pred_reg) if len(y_pred_reg) else {}
        return {"classification": cls_metrics, "regression": reg_metrics}


# ===== TFLite helpers & evaluation =====
def _load_tflite_interpreter(model_path: Path):
    """
    Robust TFLite import:
      1) tflite_runtime
      2) tensorflow (tf.lite.Interpreter)
      3) tensorflow.lite.python.interpreter.Interpreter
    """
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        interp = Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
        return interp
    except Exception:
        pass
    try:
        import tensorflow as tf  # type: ignore
        interp = tf.lite.Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
        return interp
    except Exception:
        pass
    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        interp = Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
        return interp
    except Exception as e:
        raise ImportError(
            "Could not import a TFLite Interpreter from either tflite_runtime or TensorFlow. "
            "Install either:\n"
            "  pip install tflite-runtime\n"
            "  pip install tensorflow\n"
            f"Original error: {e}"
        )


def _get_scale_zero(detail: Dict[str, Any]) -> Tuple[float, int]:
    q = detail.get("quantization", None)
    if q and len(q) == 2 and q[0] not in (None, 0.0):
        return float(q[0]), int(q[1])
    qp = detail.get("quantization_parameters", None)
    if qp:
        scales = qp.get("scales", [])
        zeros = qp.get("zero_points", [])
        if len(scales) > 0:
            return float(scales[0]), int(zeros[0] if len(zeros) > 0 else 0)
    return 1.0, 0


def _quantize_if_needed(x: np.ndarray, detail: Dict[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    if dtype == np.float32:
        return x.astype(np.float32, copy=False)
    scale, zero = _get_scale_zero(detail)
    if scale == 0:
        scale = 1.0
    xq = np.round(x / scale + zero)
    if dtype == np.int8:
        xq = np.clip(xq, -128, 127).astype(np.int8)
    elif dtype == np.uint8:
        xq = np.clip(xq, 0, 255).astype(np.uint8)
    else:
        xq = xq.astype(dtype)
    return xq


def _dequantize_if_needed(y: np.ndarray, detail: Dict[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    if dtype == np.float32:
        return y.astype(np.float32, copy=False)
    scale, zero = _get_scale_zero(detail)
    if scale == 0:
        scale = 1.0
    return (y.astype(np.float32) - float(zero)) * float(scale)


def _prep_image_from_chw(
    x_chw: np.ndarray,
    orig_mean: float,
    orig_std: float,
    swap_rgb: bool,
    norm_mode: str  # 'neg11' (x-0.5)/0.5, 'imagenet', 'none'
) -> np.ndarray:
    """
    x_chw : normalized CHW from dataset (already (x - orig_mean)/orig_std)
    returns HWC float32 ready for TFLite (pre-normalized as requested)
    """
    # back to 0..1
    x01 = x_chw * float(orig_std) + float(orig_mean)  # CHW, in [0,1] roughly
    x_hwc = np.transpose(x01, (1, 2, 0))  # HWC, still BGR as dataset came from cv2

    # Optional BGR->RGB
    if swap_rgb:
        x_hwc = x_hwc[..., ::-1]  # BGR->RGB

    if norm_mode == 'neg11':
        x_hwc = (x_hwc - 0.5) / 0.5
    elif norm_mode == 'imagenet':
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x_hwc = (x_hwc - mean) / std
    elif norm_mode == 'none':
        # keep 0..1
        pass
    return x_hwc.astype(np.float32, copy=False)


def evaluate_model_tflite_once(
    tflite_path: Path,
    loader: DataLoader,
    num_classes: int,
    # original dataset normalization
    orig_mean: float,
    orig_std: float,
    # how to feed into TFLite
    swap_rgb: bool,
    norm_mode: str  # 'neg11' / 'imagenet' / 'none'
):
    interpreter = _load_tflite_interpreter(tflite_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_detail = input_details[0]
    in_index = in_detail["index"]
    in_shape = tuple(in_detail["shape"])  # expected (1,H,W,C)

    y_true_cls, y_pred_cls, y_prob_cls = [], [], []

    for batch in tqdm(loader, desc=f"TFLite {tflite_path.name} [{('RGB' if swap_rgb else 'BGR')}, {norm_mode}]", leave=False):
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        bs = images.shape[0]
        for i in range(bs):
            x = images[i].detach().cpu().numpy()  # CHW, normalized by dataset(mean=orig_mean,std=orig_std), BGR-channel order
            x = _prep_image_from_chw(x, orig_mean, orig_std, swap_rgb=swap_rgb, norm_mode=norm_mode)  # HWC float

            # Resize to interpreter's H,W if needed
            if len(in_shape) == 4:
                _, H, W, C = in_shape
                if H > 0 and W > 0 and C > 0 and (x.shape[:2] != (H, W)):
                    import cv2
                    x = cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR)
                x = x[None, ...]  # (1,H,W,C)
            else:
                x = x[None, ...]

            xin = _quantize_if_needed(x, in_detail)
            interpreter.set_tensor(in_index, xin)
            interpreter.invoke()

            y = interpreter.get_tensor(output_details[0]["index"])
            y = _dequantize_if_needed(y, output_details[0]).squeeze()
            if y.ndim == 2:
                y = y[0]
            probs = np.exp(y - np.max(y))
            probs = probs / np.sum(probs)

            y_true_cls.append(int(labels[i].item()))
            y_pred_cls.append(int(np.argmax(probs)))
            y_prob_cls.append(probs.astype(np.float32))

    # metrics
    if num_classes <= 1:
        return {"accuracy": 0.0}, np.array(y_pred_cls)
    m = evaluate_classification(y_true_cls, y_pred_cls, np.array(y_prob_cls), num_classes=num_classes)
    return m, np.array(y_pred_cls)


def evaluate_model_tflite_auto(
    tflite_path: Path,
    loader: DataLoader,
    num_classes: int,
    orig_mean: float,
    orig_std: float,
    force_rgb: Optional[bool] = None,
    use_imagenet_norm: Optional[bool] = None
):
    """
    Run once with the user's hints (or default BGR,-1..1), then
    auto-fallback to RGB(/ImageNet) if predictions collapse to one class.
    """
    # Primary run (defaults to BGR + -1..1 if not specified)
    swap_rgb = bool(force_rgb) if force_rgb is not None else False
    norm_mode = 'imagenet' if use_imagenet_norm else 'neg11'
    metrics, preds = evaluate_model_tflite_once(
        tflite_path, loader, num_classes, orig_mean, orig_std, swap_rgb, norm_mode
    )

    # Degeneracy check: >95% same label or very low accuracy
    dom_ratio = (np.bincount(preds).max() / len(preds)) if len(preds) else 1.0
    acc = float(metrics.get("accuracy", 0.0))
    if len(preds) > 0 and (dom_ratio >= 0.95 or acc < 0.2) and (force_rgb is None or use_imagenet_norm is None):
        print(f"[WARN] Suspicious TFLite outputs (dom={dom_ratio:.3f}, acc={acc:.3f}). Trying RGB/ImageNet variants...")

        candidates = []
        tried = set()
        # try RGB with -1..1
        cand = (True, 'neg11')
        if cand not in tried:
            m2, p2 = evaluate_model_tflite_once(tflite_path, loader, num_classes, orig_mean, orig_std, *cand)
            candidates.append((m2, p2, cand))
            tried.add(cand)
        # try RGB with ImageNet mean/std
        cand = (True, 'imagenet')
        if cand not in tried:
            m3, p3 = evaluate_model_tflite_once(tflite_path, loader, num_classes, orig_mean, orig_std, *cand)
            candidates.append((m3, p3, cand))
            tried.add(cand)

        # pick the best accuracy
        best_m, best_cand = metrics, (swap_rgb, norm_mode)
        best_acc = acc
        for m, p, cand in candidates:
            a = float(m.get("accuracy", 0.0))
            if a > best_acc:
                best_acc = a
                best_m = m
                best_cand = cand
        if best_cand != (swap_rgb, norm_mode):
            print(f"[INFO] Auto-selected TFLite preprocessing: RGB={best_cand[0]}, norm={best_cand[1]} (acc={best_acc:.4f})")
            metrics = best_m

    return metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluate Torch/TFLite models in a folder and write metrics to CSV.")
    ap.add_argument("--config", required=True, help="YAML used in training (experiment format)")
    ap.add_argument("--weights_dir", required=True, help="Directory containing *.pth / *.pt / *.tflite")
    ap.add_argument("--out_csv", default="test_results.csv")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # optional TFLite hints
    ap.add_argument("--tflite_force_rgb", action="store_true", help="Force BGR->RGB for TFLite inputs")
    ap.add_argument("--tflite_imagenet_norm", action="store_true", help="Use ImageNet mean/std for TFLite inputs")
    args = ap.parse_args()

    # ---- YAML -> TrainingConfig ----
    cfg_list = load_all_training_configs(args.config)
    if not cfg_list:
        raise RuntimeError(f"No training configs parsed from: {args.config}")

    device = torch.device(args.device)

    out_rows: List[Dict[str, Any]] = []
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    wdir = Path(args.weights_dir)
    paths = sorted(list(wdir.glob("*.pth")) + list(wdir.glob("*.pt")) + list(wdir.glob("*.tflite")))
    if not paths:
        print(f"[WARN] No checkpoint/model found in: {wdir}")

    for wpath in paths:
        suffix = wpath.suffix.lower()

        # ===== Choose cfg =====
        chosen_cfg = None
        ckpt = None
        if suffix in [".pth", ".pt"]:
            ckpt = torch.load(str(wpath), map_location="cpu")
            chosen_cfg = pick_cfg_for_ckpt(cfg_list, ckpt) or cfg_list[0]
        else:
            chosen_cfg = cfg_list[0]

        # ===== Dataset / DataLoader =====
        image_size = tuple(getattr(chosen_cfg, "image_size", (480, 480)))
        mean = float(getattr(chosen_cfg, "normalize_mean", 0.5)) if hasattr(chosen_cfg, "normalize_mean") else 0.5
        std = float(getattr(chosen_cfg, "normalize_std", 0.5)) if hasattr(chosen_cfg, "normalize_std") else 0.5

        tf = SimpleTransform(size_hw=image_size, mean=mean, std=std)

        valid_csv = getattr(chosen_cfg, "valid_file_dir", None)
        valid_img_dir = getattr(chosen_cfg, "valid_img_dir", "")
        if valid_csv is None or not Path(valid_csv).exists():
            raise FileNotFoundError(f"valid_file_dir not found in cfg: {valid_csv}")

        ds = SignalMixClassificationDataset(
            img_dir=valid_img_dir,
            annotation_csv=valid_csv,
            transform=tf,
        )

        bs = args.batch_size or int(getattr(chosen_cfg, "batch_size", 16))
        is_pin_memory = (device.type == "cuda")
        loader_torch = DataLoader(
            ds, batch_size=bs, shuffle=False,
            num_workers=int(getattr(chosen_cfg, "num_workers", 4)),
            pin_memory=is_pin_memory, collate_fn=signalmix_collate, drop_last=False
        )
        loader_tflite = DataLoader(
            ds, batch_size=1, shuffle=False,
            num_workers=int(getattr(chosen_cfg, "num_workers", 2)),
            pin_memory=False, collate_fn=signalmix_collate, drop_last=False
        )

        # ===== Determine task & num_classes =====
        task = str(getattr(chosen_cfg, "task", "classification")).lower()
        num_classes = int(getattr(chosen_cfg, "num_classes", getattr(ds, "num_classes", 3)))

        # ===== Evaluate =====
        if suffix in [".pth", ".pt"]:
            model, task_eff, num_classes_eff, model_name = build_model_from_cfg_via_builder(
                chosen_cfg, override_ckpt=ckpt, ds_num_classes=getattr(ds, "num_classes", None)
            )
            model = model.to(device)
            load_state_dict_safely(model, ckpt)
            metrics = evaluate_model_torch(model, loader_torch, device, task_eff, num_classes_eff)

            row = {"model_file": wpath.name, "model_name": model_name, "task": task_eff}
            if task_eff == "multitask":
                for k, v in (metrics.get("classification", {}) or {}).items():
                    row[f"classification_{k}"] = v
                for k, v in (metrics.get("regression", {}) or {}).items():
                    row[f"regression_{k}"] = v
            else:
                row.update(metrics)
            out_rows.append(row)
            acc_print = row.get("accuracy", row.get("classification_accuracy", "NA"))
            print(f"[OK][Torch] {wpath.name} -> accuracy: {acc_print}")

        else:  # .tflite
            # run with hints (or default) + auto-fallback if collapsed
            metrics = evaluate_model_tflite_auto(
                tflite_path=wpath,
                loader=loader_tflite,
                num_classes=num_classes,
                orig_mean=mean,
                orig_std=std,
                force_rgb=args.tflite_force_rgb if hasattr(args, "tflite_force_rgb") else None,
                use_imagenet_norm=args.tflite_imagenet_norm if hasattr(args, "tflite_imagenet_norm") else None
            )
            row = {"model_file": wpath.name, "model_name": getattr(chosen_cfg, "model_name", "tflite_model"), "task": task}
            if task == "multitask":
                for k, v in (metrics.get("classification", {}) or {}).items():
                    row[f"classification_{k}"] = v
                for k, v in (metrics.get("regression", {}) or {}).items():
                    row[f"regression_{k}"] = v
            else:
                row.update(metrics)
            out_rows.append(row)
            acc_print = row.get("accuracy", row.get("classification_accuracy", "NA"))
            print(f"[OK][TFLite] {wpath.name} -> accuracy: {acc_print}")

    # ===== Write CSV =====
    if out_rows:
        keys = list(sorted({k for r in out_rows for k in r.keys()}))
        head = ["model_file", "model_name", "task"] + [k for k in keys if k not in ("model_file", "model_name", "task")]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=head)
            w.writeheader()
            for r in out_rows:
                w.writerow({k: r.get(k, "") for k in head})
    else:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            f.write("model_file,model_name,task\n")

    print(f"[DONE] wrote: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
