#!/usr/bin/env python3
"""
Extract every image whose label file contains class **13**, pad it to a centered
square using black pixels, resize to **480 × 480**, and save the result under *VIDVIP_crosswalk/*
(keeping the same sub-folder structure).  
Usage:
    python extract_crosswalk.py --src ./VIDVIP --dst ./VIDVIP_crosswalk
"""

from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
import argparse

# --------------------------------------------------------------------------- #
TARGET_CLASS = 13           # class-id to keep
TARGET_SIZE  = 480          # final edge length in px
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
# --------------------------------------------------------------------------- #

def pad_to_square(img: Image.Image) -> Image.Image:
    """
    Pad the image to a square using black pixels (zero padding), centering the original image.
    """
    w, h = img.size
    max_side = max(w, h)
    pad_left   = (max_side - w) // 2
    pad_right  = max_side - w - pad_left
    pad_top    = (max_side - h) // 2
    pad_bottom = max_side - h - pad_top
    return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

def matching_image_path(label_path: Path) -> Path | None:
    img_dir = label_path.parent.parent / "images"
    stem    = label_path.stem
    for ext in IMG_EXTS:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None

def label_has_target(label_path: Path) -> bool:
    with label_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            if int(line.split()[0]) == TARGET_CLASS:
                return True
    return False

def process_image(src_img: Path, dst_img: Path) -> None:
    with Image.open(src_img) as im:
        im_padded = pad_to_square(im).resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        im_padded.save(dst_img, quality=95)

def collect_label_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("labels/*.txt") if p.is_file()]

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract class-13 images, pad & resize.")
    parser.add_argument("--src",  type=Path, default=r"D:\Datasets\VIDVIP")
    parser.add_argument("--dst",  type=Path, default=r"D:\Datasets\WithCross Dataset\vidvip/")
    args = parser.parse_args()

    if not args.src.is_dir():
        raise NotADirectoryError(f"{args.src} is not a directory")
    args.dst.mkdir(parents=True, exist_ok=True)

    label_files = collect_label_files(args.src)

    kept = 0

    for lbl in tqdm(label_files, desc="Scanning labels", unit="file"):
        if not label_has_target(lbl):
            continue

        img_path = matching_image_path(lbl)
        if img_path is None:
            tqdm.write(f"⚠  image missing for {lbl}")
            continue

        dst_img = args.dst / img_path.name
        process_image(img_path, dst_img)

if __name__ == "__main__":
    main()