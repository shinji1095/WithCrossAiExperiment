#!/usr/bin/env python3
# convert_vidvip_annotations.py
"""
Usage:
    python convert_vidvip_annotations.py --root ./VIDVIP --out vidvip_converted.csv
"""

from pathlib import Path
import pandas as pd
import argparse

# ── 定数 ─────────────────────────────────────────────
SIGNAL_MAP = {15: 0, 16: 1}  # それ以外 → 2
IMAGE_EXTS = [".jpg", ".JPG", ".jpeg", ".png"]

# ── 補助関数 ─────────────────────────────────────────
def find_image(path_labels: Path) -> str:
    """同名の画像ファイルを labels/ の兄弟 images/ ディレクトリから探し，ファイル名を返す。"""
    stem = path_labels.stem
    images_dir = path_labels.parent.parent / "images"
    for ext in IMAGE_EXTS:
        img_path = images_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path.name
    raise FileNotFoundError(f"Image for {path_labels} not found")

def parse_label_file(path_labels: Path) -> list[int]:
    """ラベルファイルを読み込み，整数クラス ID のリストを返す。"""
    with path_labels.open() as f:
        return [int(line.split()[0]) for line in f if line.strip()]

# ── 変換処理 ─────────────────────────────────────────
def convert(root: Path) -> pd.DataFrame:
    label_files = list(root.rglob("labels/*.txt"))
    rows = []

    for lbl_path in label_files:
        class_ids = parse_label_file(lbl_path)

        # crosswalk (13) を含まない場合はスキップ
        if 13 not in class_ids:
            continue

        # signal 判定
        signal = 2  # デフォルト None
        if 15 in class_ids:
            signal = 0
        elif 16 in class_ids:
            signal = 1

        rows.append({
            "filename": find_image(lbl_path),
            "signal": signal,
            "state": 0,           # Normal
            "slope_deg": 0.0,     # 指定により固定
        })

    return pd.DataFrame(rows)

# ── main ────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VIDVIP annotations to required CSV format")
    parser.add_argument("--root", required=True, type=Path, help="Path to VIDVIP directory")
    parser.add_argument("--out",  required=True, type=Path, help="Output CSV path")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise NotADirectoryError(args.root)

    df = convert(args.root)
    if df.empty:
        print("No valid annotations found (class 13 missing).")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✔ Converted {len(df)} entries → {args.out}")

if __name__ == "__main__":
    main()
