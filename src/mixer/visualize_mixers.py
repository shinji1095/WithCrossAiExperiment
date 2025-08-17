"""
make_augmented_samples.py

1. dataset_dir 以下を再帰走査して               *.jpg と対応 .txt アノテーションを検索
2. class_id == 15 (赤) or 16 (青) が 1 つでもある画像をランダムに N 枚抽出
3. advanced_mixers.py に実装されている全拡張
      CutMix / AttentiveCutMix / SaliencyMix / PuzzleMix /
      SnapMix / KeepAugment / SignalMix
   を適用（SignalMix は信号 bbox へ signal_dir の画像 1 枚を貼り付け）
4. out_dir/＜AugName＞/ に保存して目視確認できるようにする
"""

import argparse, random, itertools, shutil, re
from pathlib import Path
from typing import List, Tuple

import cv2
import torch, numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models

from advanced_mixers import get_mixer

backbone = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V2
).eval().cpu()

# ------------------- util -------------------------------------------------
def read_yolo_txt(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """YOLO txt -> [(cls, cx, cy, w, h), ...]"""
    rows = []
    with open(txt_path, "r") as f:
        for ln in f:
            if ln.strip():
                rows.append(tuple(map(float, ln.split())))
    return rows                                                      # type: ignore


def to_xyxy(rel_box, img_w, img_h):
    """cx,cy,w,h [0-1] -> x1y1x2y2 [pixel]"""
    cx, cy, w, h = rel_box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_tensor(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size))
    return TF.to_tensor(img)  # (3,H,W) 0-1 float


def save_img(tensor: torch.Tensor, text: str, path: Path):
    img = TF.to_pil_image(tensor.clamp(0, 1))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, (255, 0, 0), spacing=10, align='left')
    img.save(path)


# -------------------------------------------------------------------------
def collect_signal_images(dataset_dir: Path, num_samples: int) -> List[Tuple[Path, torch.Tensor, List[List[int]]]]:
    """
    Returns list of (img_path, img_tensor(3,224,224), bboxes[list[x1y1x2y2...]])
    """
    candidates: List[Tuple[Path, List[List[int]]]] = []

    for txt in dataset_dir.rglob("labels/*.txt"):
        img_path = txt.parent.parent / "images" / (txt.stem + ".jpg")
        if not img_path.exists():
            continue
        rows = read_yolo_txt(txt)
        boxes = []
        for cls, cx, cy, w, h in rows:
            if int(cls) in (15, 16):  # red or blue signal
                boxes.append((cx, cy, w, h))
        if boxes:
            candidates.append((img_path, boxes))

    if len(candidates) < num_samples:
        raise RuntimeError(f"信号付き画像が {len(candidates)} 枚しか見つかりません")

    chosen = random.sample(candidates, num_samples)
    outputs = []
    for img_path, rel_boxes in chosen:
        tensor = load_tensor(img_path, 224)             # (3,224,224)
        H, W = 224, 224
        abs_boxes = [to_xyxy(b, W, H) for b in rel_boxes]
        outputs.append((img_path, tensor, abs_boxes))
    return outputs


# -------------------------------------------------------------------------
def main(args):
    random.seed(0)

    dataset_dir = Path(args.dataset_dir)
    signal_dir = Path(args.signal_dir)
    out_root = Path(args.out_dir)
    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) 元画像 + bbox を収集
    samples = collect_signal_images(dataset_dir, args.num_samples)

    # 2) Augmentation 定義
    aug_cfgs = [
        {"name": "cutmix"},
        {"name": "attentive_cutmix"},
        {"name": "saliencymix"},
        {"name": "puzzlemix"},
        {"name": "snapmix"},
        {"name": "keepaugment", "keepaugment_tau": 0.15},
        {"name": "signalmix", "signal_dir": str(signal_dir), "mix_prob": 1.0},
    ]

    # 3) 画像ごとに Aug → 保存
    for idx, (img_path, img_t, bboxes) in tqdm(enumerate(samples, 1)):
        if idx >= 10:
            break

        for cfg in aug_cfgs:
            aug_name = cfg["name"]
            mixer = get_mixer(cfg, backbone)
            # バッチ: 2 枚必要な Aug があるので同じ画像を複製
            if aug_name == "signalmix":
                batch_imgs = torch.stack([img_t, img_t.clone()])
                batch_lbls = torch.tensor([0, 0])
                signal_bboxes = [bboxes, bboxes]
            else:
                # その他はランダムに「別のサンプル」をもう１枚選んで混合
                other_path, other_t, other_bboxes = random.choice(samples)

                batch_imgs = torch.stack([img_t, other_t])
                batch_lbls = torch.tensor([0, 1])
                signal_bboxes = None

            bbox_list = [bboxes, bboxes]                             # 2 枚とも同じ bbox
            if aug_name == "signalmix":
                out_imgs, labels = mixer(batch_imgs.clone(), batch_lbls, signal_bboxes=bbox_list)
            else:
                out_imgs, labels = mixer(batch_imgs.clone(), batch_lbls)

            save_dir = out_root / aug_name
            save_dir.mkdir(exist_ok=True)
            save_img(out_imgs[0], f"Label: {labels[0]}" , save_dir / f"{img_path.stem}_{idx:02d}.png")

    print(f"Augmented images saved under: {out_root.resolve()}")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir",  default=r"D:\Datasets\VIDVIP")
    ap.add_argument("--signal_dir",  default=r"D:\Datasets\WithCross Dataset\signal")
    ap.add_argument("--out_dir", default="aug_samples")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--force", action="store_true",
                    help="既存 out_dir を削除して作り直す")
    main(ap.parse_args())
