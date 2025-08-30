from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

TARGET_SIZE = 480
SUPPORTED_EXTS = {".jpg", ".JPG", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}


def center_square_crop(img: Image.Image) -> Image.Image:
    """Crop the largest possible centered square from `img`."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    upper = (h - side) // 2
    right = left + side
    lower = upper + side
    return img.crop((left, upper, right, lower))


def process_image(src_path: Path, dst_path: Path) -> None:
    """Crop, resize, and save a single image."""
    with Image.open(src_path) as img:
        img = center_square_crop(img).resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        img.save(dst_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Square-crop and resize images.")
    parser.add_argument("--src", required=True, type=Path, help="Input directory")
    parser.add_argument("--dst", required=True, type=Path, help="Output directory")
    args = parser.parse_args()

    if not args.src.is_dir():
        raise NotADirectoryError(f"Source folder not found: {args.src}")

    args.dst.mkdir(parents=True, exist_ok=True)

    images = [p for p in args.src.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    if not images:
        print("No supported image files found.")
        return

    for img_path in tqdm(images, desc="Processing", unit="img"):
        rel = img_path.relative_to(args.src)
        out_path = args.dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        process_image(img_path, out_path)

    print(f"✔ Done! {len(images)} images saved to {args.dst}")


if __name__ == "__main__":
    main()

"""
python crop_resize.py --src ./raw_images --dst ./processed_images
"""