#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize SignalMix (CutMix-contained) and others on samples from annotation.csv.

- Reads preprocessed/annotation.csv (filename,label,bbox,width,height,...)
- Builds a small batch and applies a chosen mixer (signalmix|cutmix|none|...).
- Saves original & mixed images with decoded labels. (optionally draws bbox)

Notes
-----
* Labels are expected as {"NONE","RED","GREEN"} mapped to {0,1,2}.
* BBoxes in annotation.csv are in original pixel coords; this script rescales
  them to the resized image size before passing to SignalMix.
* Normalization assumes mean=0.5, std=0.5 by default (configurable via CLI).
"""

import argparse
import csv as _csv
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# === Your project mixer factory ===
try:
    from advanced_mixers import get_mixer
except Exception as e:
    raise RuntimeError("advanced_mixers.py with get_mixer(...) is required.") from e

LABEL2IDX = {"NONE": 0, "RED": 1, "GREEN": 2}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


# ------------------------------
# Utils
# ------------------------------
def to_tensor_chw01(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    """HWC uint8 [0..255] -> CHW float32 [0..1]"""
    t = torch.from_numpy(img_rgb_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
    return t


def normalize_(t: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Inplace normalize: (t - mean)/std for each channel."""
    return (t - mean) / std


def denorm(t: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Reverse normalize: t * std + mean, clip to 0..1"""
    return torch.clamp(t * std + mean, 0, 1)


def parse_bbox_string(s: str) -> List[Tuple[int, int, int, int]]:
    """'x1,y1,x2,y2' -> [(x1,y1,x2,y2)] or []"""
    if not isinstance(s, str) or not s.strip():
        return []
    parts = s.split(",")
    if len(parts) != 4:
        return []
    try:
        x1, y1, x2, y2 = map(int, parts)
        return [(x1, y1, x2, y2)]
    except Exception:
        return []


def scale_bboxes(bboxes: List[Tuple[int, int, int, int]],
                 src_w: int, src_h: int, dst_w: int, dst_h: int) -> List[Tuple[int, int, int, int]]:
    """Scale pixel bboxes from (src_w,src_h) -> (dst_w,dst_h)."""
    if not bboxes:
        return []
    sx = dst_w / max(1, src_w)
    sy = dst_h / max(1, src_h)
    out = []
    for x1, y1, x2, y2 in bboxes:
        nx1 = int(round(x1 * sx))
        ny1 = int(round(y1 * sy))
        nx2 = int(round(x2 * sx))
        ny2 = int(round(y2 * sy))
        # clamp & fix invalids
        nx1 = max(0, min(nx1, dst_w - 1))
        ny1 = max(0, min(ny1, dst_h - 1))
        nx2 = max(1, min(nx2, dst_w))
        ny2 = max(1, min(ny2, dst_h))
        if nx2 > nx1 and ny2 > ny1:
            out.append((nx1, ny1, nx2, ny2))
    return out


def decode_label(y: torch.Tensor) -> int:
    """Accepts hard (scalar) or one-hot/prob (vector). Returns class index int."""
    if y.dim() == 0:
        return int(y.item())
    return int(torch.argmax(y).item())


def put_label(img_rgb: np.ndarray, text: str, top_left=(5, 18)) -> np.ndarray:
    out = img_rgb.copy()
    cv2.putText(out, text, top_left, cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def draw_bboxes(img_rgb: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    out = img_rgb.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


# ------------------------------
# Dataset loader (CSV based)
# ------------------------------
def read_annotation_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = _csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def make_batch_from_csv(
    csv_rows: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    mean: float, std: float,
    only_with_signal: bool,
    batch_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[Tuple[int, int, int, int]]], List[Dict[str, Any]]]:
    """
    Returns:
      imgs:  (B,3,H,W) normalized
      labels: (B,) int
      bboxes_list: list of list[bbox] scaled to HxW
      picked_rows: list of CSV rows used (for debugging/printing)
    """
    rng = random.Random(seed)
    H, W = image_size

    # filter rows
    pool = []
    for row in csv_rows:
        label_name = row.get("label", "NONE")
        if only_with_signal and label_name == "NONE":
            continue
        pool.append(row)
    if not pool:
        raise RuntimeError("No rows matched the filter (try --only_with_signal 0).")

    picked = rng.sample(pool, k=min(batch_size, len(pool)))
    imgs = []
    labels = []
    bbox_list = []

    for row in picked:
        img_path = Path(row["filename"])
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        src_h, src_w = img_rgb.shape[:2]

        # resize to (H,W)
        if (src_h, src_w) != (H, W):
            img_rgb_rs = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            img_rgb_rs = img_rgb

        # parse & scale bbox
        bxs = parse_bbox_string(row.get("bbox", ""))
        if bxs:
            # if width/height columns exist, use them; else use src shape
            try:
                ow = int(row.get("width", src_w))
                oh = int(row.get("height", src_h))
            except Exception:
                ow, oh = src_w, src_h
            bxs_s = scale_bboxes(bxs, ow, oh, W, H)
        else:
            bxs_s = []

        # to tensor + normalize
        t = to_tensor_chw01(img_rgb_rs)
        t = normalize_(t, mean, std)
        imgs.append(t)

        labels.append(LABEL2IDX.get(row.get("label", "NONE"), 0))
        bbox_list.append(bxs_s)

    imgs_t = torch.stack(imgs, dim=0)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return imgs_t, labels_t, bbox_list, picked


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualize SignalMix/CutMix via annotation.csv")
    ap.add_argument("--annotation_csv", default=r"D:\Datasets\WithCross Dataset\vidvip_signal/annotation.csv", help="Path to preprocessed/annotation.csv")
    ap.add_argument("--signal_dir",     default=r"D:\Datasets\WithCross Dataset\vidvip_signal/signal",        help="signal crop images dir (for signalmix)")
    ap.add_argument("--signal_csv",     default=r"D:\Datasets\WithCross Dataset\vidvip_signal/signal.csv",    help="signal.csv (0:RED,1:GREEN)")
    ap.add_argument("--out_dir",        default="viz_out",                    help="output dir")

    ap.add_argument("--mixer",
                    choices=["signalmix", "cutmix", "attentive_cutmix", "saliencymix",
                             "puzzlemix", "snapmix", "keepaugment", "none"],
                    default="signalmix")
    ap.add_argument("--prob", type=float, default=0.1, help="SignalMix probability when label != NONE")
    ap.add_argument("--beta", type=float, default=1.0, help="Beta(alpha) for CutMix-like samplers")
    ap.add_argument("--use_cutmix", action="store_true", help="Use CutMix inside SignalMix before patching")
    ap.add_argument("--no-use_cutmix", dest="use_cutmix", action="store_false")
    ap.set_defaults(use_cutmix=True)
    ap.add_argument("--none_index", type=int, default=0, help="Index of NONE class in your training (default 0)")

    ap.add_argument("--image_size", type=int, nargs=2, default=[224, 224], metavar=("H", "W"))
    ap.add_argument("--mean", type=float, default=0.5, help="Normalize mean")
    ap.add_argument("--std",  type=float, default=0.5, help="Normalize std")

    ap.add_argument("--batch_size", type=int, default=8, help="batch size for visualization")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--only_with_signal", type=int, default=1, help="1: sample only RED/GREEN rows, 0: include NONE rows")
    ap.add_argument("--draw_bbox", action="store_true", help="draw bbox on outputs for sanity-check")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load CSV
    rows = read_annotation_csv(Path(args.annotation_csv))
    if len(rows) == 0:
        raise RuntimeError(f"No rows in CSV: {args.annotation_csv}")

    # 2) Make batch (resizing & bbox scaling done here)
    H, W = args.image_size
    imgs, labels, bboxes, picked = make_batch_from_csv(
        rows, (H, W), args.mean, args.std,
        only_with_signal=bool(args.only_with_signal),
        batch_size=args.batch_size,
        seed=args.seed
    )
    imgs = imgs.to(device)
    labels = labels.to(device)

    # 3) Mixer config & build
    aug_cfg = dict(
        name=args.mixer.lower(),
        prob=args.prob,
        beta=args.beta,
        use_cutmix=args.use_cutmix,
        signal_dir=args.signal_dir,
        signal_csv=args.signal_csv,
        none_index=args.none_index,
    )

    # Some mixers require a backbone; advanced_mixers.get_mixer handles it if needed.
    mixer = get_mixer(aug_cfg, backbone=None)

    # 4) Forward mixer
    y_one = F.one_hot(labels, num_classes=3).float()
    if mixer is None or args.mixer.lower() == "none":
        mixed_imgs, mixed_y = imgs, y_one
    else:
        if getattr(mixer, "needs_bboxes", False):
            mixed_imgs, mixed_y = mixer(imgs.clone(), y_one.clone(), bboxes)
        else:
            mixed_imgs, mixed_y = mixer(imgs.clone(), y_one.clone())

    # 5) Save originals & mixed with decoded labels (and optional bbox)
    orig_vis  = denorm(imgs.detach().cpu(), args.mean, args.std)
    mixed_vis = denorm(mixed_imgs.detach().cpu(), args.mean, args.std)

    for i in range(orig_vis.size(0)):
        # decode labels
        y0_idx = decode_label(y_one[i].cpu())
        y1_idx = decode_label(mixed_y[i].cpu())
        y0_txt = f"orig: {IDX2LABEL.get(y0_idx, str(y0_idx))}"
        y1_txt = f"{args.mixer}: {IDX2LABEL.get(y1_idx, str(y1_idx))}"

        # to uint8 RGB
        o_rgb = (orig_vis[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        m_rgb = (mixed_vis[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        if args.draw_bbox and bboxes[i]:
            o_rgb = draw_bboxes(o_rgb, bboxes[i])
            m_rgb = draw_bboxes(m_rgb, bboxes[i])

        o_rgb = put_label(o_rgb, y0_txt, (5, 20))
        m_rgb = put_label(m_rgb, y1_txt, (5, 20))

        cv2.imwrite(str(out_dir / f"orig_{i:02d}.jpg"), cv2.cvtColor(o_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{args.mixer.lower()}_{i:02d}.jpg"),
                    cv2.cvtColor(m_rgb, cv2.COLOR_RGB2BGR))

    print(f"[OK] Saved {orig_vis.size(0)} originals as orig_*.jpg")
    print(f"[OK] Saved {mixed_vis.size(0)} mixed as {args.mixer.lower()}_*.jpg")
    print(f"Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
