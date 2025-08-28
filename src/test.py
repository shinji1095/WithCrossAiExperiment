#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test.py (project-aligned)
- 指定フォルダ内の *.pth / *.pt をすべてロードして検証
- モデル構築は train_ddp.py と同様: from models.model import get_model
- YAML は config.py 経由で TrainingConfig を構築（学習と同じ前処理・データパス）
- 指標は validation と同じ metrics.metrics の evaluate_* を使用
- 出力は CSV（1行=1モデル）
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- プロジェクト内モジュール（学習時と同じもの）----
from config.config import load_all_training_configs
from dataset_signalmix import SignalMixClassificationDataset
from metrics.metrics import evaluate_classification, evaluate_regression
from models.model import get_model  # ★ train_ddp と同じビルダを使用


# ===== train_ddp.py と同じ Transform / collate を再現 =====
class SimpleTransform:
    """
    train_ddp.py の前処理と整合:
      - Resize to (H,W)
      - ToTensor(0..1) -> Normalize(mean=0.5,std=0.5) => 値域 -1..1
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
    """
    Dataset -> (img, label_idx, bboxes) を受け取り、bboxesはlistのまま返す
    """
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
    """
    複数 model を含む experiment の場合、ckpt 情報に最も合致する cfg を選ぶ。
    """
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


def build_model_from_cfg_via_builder(cfg, override_ckpt: Optional[Dict[str, Any]] = None,
                                     ds_num_classes: Optional[int] = None):
    """
    train_ddp.py と同様に get_model(cfg) を使ってモデルを構築。
    - ckpt 側に model_name / task / num_classes があれば cfg に反映してから get_model を呼ぶ
    - num_classes は ckpt -> cfg -> ds -> 3 の順で決定（None 安全）
    - get_model の戻りが (model, ...) でも model 単体でも受けられるように対応
    """
    ck = override_ckpt or {}

    # cfg を上書き（学習時の保存メタと矛盾しないよう優先度は ckpt > cfg）
    if "model_name" in ck and ck["model_name"]:
        cfg.model_name = ck["model_name"]
    if "task" in ck and ck["task"]:
        cfg.task = ck["task"]

    cfg.num_classes = _safe_int(
        _first_not_none(ck.get("num_classes"), getattr(cfg, "num_classes", None), ds_num_classes, 3),
        3
    )

    # get_model はプロジェクト実装に合わせてそのまま呼ぶ
    model = get_model(
            cfg.task,
            cfg.model_name,
            num_classes=3,
            dropout_rate=cfg.dropout_rate,
            drop_path_rate=cfg.drop_path_rate,
        ).to('cuda')
    return model, str(cfg.task).lower(), int(cfg.num_classes), str(cfg.model_name)


def load_state_dict_safely(model: torch.nn.Module, ckpt: Dict[str, Any]) -> None:
    """
    学習側の保存形式に幅広く対応:
      - {'state_dict': ..., ...}
      - {'model': ...}
      - それ以外は dict 自体を state_dict とみなす
    'module.' プレフィックスを剥がしてから load
    """
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


def evaluate_model(model, loader, device, task: str, num_classes: int):
    """
    validation と同じメトリクスを返す:
      - classification: evaluate_classification
      - regression:     evaluate_regression
      - multitask:      {'classification':..., 'regression':...}
    """
    model.eval()
    y_true_cls, y_pred_cls, y_prob_cls = [], [], []
    y_true_reg, y_pred_reg = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", leave=False):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            images = images.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type):
                out = model(images)

            if task == "multitask":
                signal_pred, slope_pred = out
                # classification
                logits = signal_pred
                y_true_cls.extend(labels.cpu().tolist())
                y_pred_cls.extend(logits.argmax(dim=1).cpu().tolist())
                y_prob_cls.extend(torch.softmax(logits, dim=1).cpu().numpy())
                # regression（テストCSVに回帰のGTが無ければ空のまま）
                y_pred_reg.extend(torch.tanh(slope_pred).flatten().cpu().tolist())

            elif task == "classification":
                logits = out
                y_true_cls.extend(labels.cpu().tolist())
                y_pred_cls.extend(logits.argmax(dim=1).cpu().tolist())
                y_prob_cls.extend(torch.softmax(logits, dim=1).cpu().numpy())

            else:  # regression
                y_true_reg.extend(labels.cpu().tolist())
                y_pred_reg.extend(torch.tanh(out).flatten().cpu().tolist())

    if task == "classification":
        metrics = evaluate_classification(y_true_cls, y_pred_cls, np.array(y_prob_cls), num_classes=num_classes)
    elif task == "regression":
        metrics = evaluate_regression(y_true_reg, y_pred_reg)
    else:
        cls_metrics = {}
        if len(set(y_true_cls)) >= 2:
            cls_metrics = evaluate_classification(y_true_cls, y_pred_cls, np.array(y_prob_cls), num_classes=num_classes)
        reg_metrics = evaluate_regression(y_true_reg, y_pred_reg) if len(y_pred_reg) else {}
        metrics = {"classification": cls_metrics, "regression": reg_metrics}
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluate all checkpoints in a folder and write metrics to CSV.")
    ap.add_argument("--config", required=True, help="YAML used in training (experiment format)")
    ap.add_argument("--weights_dir", required=True, help="Directory containing *.pth / *.pt")
    ap.add_argument("--out_csv", default="test_results.csv")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ---- YAML -> TrainingConfig（学習時と同じローダ）----
    cfg_list = load_all_training_configs(args.config)
    if not cfg_list:
        raise RuntimeError(f"No training configs parsed from: {args.config}")

    device = torch.device(args.device)

    out_rows: List[Dict[str, Any]] = []
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    wdir = Path(args.weights_dir)
    paths = sorted(list(wdir.glob("*.pth")) + list(wdir.glob("*.pt")))
    if not paths:
        print(f"[WARN] No checkpoint found in: {wdir}")

    for wpath in paths:
        ckpt = torch.load(str(wpath), map_location="cpu")
        cfg = pick_cfg_for_ckpt(cfg_list, ckpt) or cfg_list[0]

        # ===== Dataset / DataLoader =====
        image_size = tuple(getattr(cfg, "image_size", (480, 480)))
        mean = float(getattr(cfg, "normalize_mean", 0.5)) if hasattr(cfg, "normalize_mean") else 0.5
        std  = float(getattr(cfg, "normalize_std", 0.5))  if hasattr(cfg, "normalize_std")  else 0.5

        tf = SimpleTransform(size_hw=image_size, mean=mean, std=std)

        valid_csv = getattr(cfg, "valid_file_dir", None)
        valid_img_dir = getattr(cfg, "valid_img_dir", "")
        if valid_csv is None or not Path(valid_csv).exists():
            raise FileNotFoundError(f"valid_file_dir not found in cfg: {valid_csv}")

        ds = SignalMixClassificationDataset(
            img_dir=valid_img_dir,
            annotation_csv=valid_csv,
            transform=tf,
        )
        bs = args.batch_size or int(getattr(cfg, "batch_size", 16))
        loader = DataLoader(
            ds, batch_size=bs, shuffle=False,
            num_workers=int(getattr(cfg, "num_workers", 4)),
            pin_memory=True, collate_fn=signalmix_collate, drop_last=False
        )

        # ===== Model: train_ddp.py と同じ経路で構築 =====
        model, task, num_classes, model_name = build_model_from_cfg_via_builder(
            cfg, override_ckpt=ckpt, ds_num_classes=getattr(ds, "num_classes", None)
        )
        model = model.to(device)
        load_state_dict_safely(model, ckpt)

        # ===== Eval =====
        metrics = evaluate_model(model, loader, device, task, num_classes)

        # ===== Row =====
        row = {
            "model_file": wpath.name,
            "model_name": model_name,
            "task": task,
        }
        if task == "multitask":
            for k, v in (metrics.get("classification", {}) or {}).items():
                row[f"classification_{k}"] = v
            for k, v in (metrics.get("regression", {}) or {}).items():
                row[f"regression_{k}"] = v
        else:
            row.update(metrics)

        out_rows.append(row)
        print(f"[OK] {wpath.name} -> accuracy: {row.get('accuracy', 'NA')}")

    # ===== CSV 出力 =====
    if out_rows:
        keys = list(sorted({k for r in out_rows for k in r.keys()}))
        head = ["model_file", "model_name", "task"] + [k for k in keys if k not in ("model_file","model_name","task")]
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
