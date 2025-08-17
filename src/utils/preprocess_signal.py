# preprocess_signals.py
import argparse
from pathlib import Path
import csv
import cv2

CLS_RED = 15
CLS_GREEN = 16

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w/2) * img_w)
    y1 = int((cy - h/2) * img_h)
    x2 = int((cx + w/2) * img_w)
    y2 = int((cy + h/2) * img_h)
    # clamp
    x1 = max(0, min(x1, img_w-1)); x2 = max(0, min(x2, img_w-1))
    y1 = max(0, min(y1, img_h-1)); y2 = max(0, min(y2, img_h-1))
    if x2 <= x1: x2 = min(img_w-1, x1+1)
    if y2 <= y1: y2 = min(img_h-1, y1+1)
    return x1, y1, x2, y2

def parse_label_file(txt_path):
    sig_boxes = []  # (cls_id, (cx,cy,w,h))
    others = []
    if not txt_path.exists():
        return sig_boxes, others
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        if cls_id in (CLS_RED, CLS_GREEN):
            sig_boxes.append((cls_id, (cx, cy, w, h)))
        else:
            others.append((cls_id, (cx, cy, w, h)))
    return sig_boxes, others

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True, help="root dir of dataset/")
    ap.add_argument("--out_dir", type=str, default="preprocessed", help="output dir")
    ap.add_argument("--signal_prefix", type=str, default="sig", help="cropped signal filename prefix")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_images_dir = out_dir / "images_15_16"
    out_signal_dir = out_dir / "signal"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_signal_dir.mkdir(parents=True, exist_ok=True)

    annotation_rows = []
    signal_rows = []

    # 3階層（1-999, 1000-1999, 2000-2999）を総なめ
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    sig_counter = 0

    for sub in subdirs:
        # images と labels の場所がディレクトリ構成で微妙に違うパターンにも耐性
        img_dir = sub / "images"
        if not img_dir.exists():  # 2000-2999 形式
            img_dir = sub
        lbl_dir = sub / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            base = img_path.stem  # img_XXXXX
            txt_path = lbl_dir / f"{base}.txt"

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            sig_boxes, _ = parse_label_file(txt_path)
            has_signal = len(sig_boxes) > 0

            # 1) 15/16 を含む画像だけ out_images_dir にコピー保存
            if has_signal:
                out_img_path = out_images_dir / img_path.name
                cv2.imwrite(str(out_img_path), img)

            # 2) 15/16 のみクロップして signal/ に保存、signal.csv に記録
            signal_label_list = []
            max_area = -1
            max_xyxy = None
            max_label = None

            for cls_id, (cx, cy, ww, hh) in sig_boxes:
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, ww, hh, w, h)
                crop = img[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue
                sig_counter += 1
                sig_name = f"{args.signal_prefix}_{base}_{sig_counter:06d}.jpg"
                cv2.imwrite(str(out_signal_dir / sig_name), crop)

                # 3) signal.csv のラベルは 15→0, 16→1
                mapped = 0 if cls_id == CLS_RED else 1
                signal_rows.append({
                    "signal_file": sig_name,
                    "label": mapped,
                    "src_image": str(img_path)
                })

                signal_label_list.append(str(cls_id))
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    max_xyxy = (x1, y1, x2, y2)
                    max_label = "RED" if cls_id == CLS_RED else "GREEN"

            # 4) annotation.csv 生成
            if max_xyxy is None:
                # NONE のとき
                annotation_rows.append({
                    "filename": str(img_path),
                    "label": "NONE",
                    "signal_labels": "",
                    "bbox": "",
                    "width": w,
                    "height": h
                })
            else:
                annotation_rows.append({
                    "filename": str(img_path),
                    "label": max_label,  # 最大面積の信号のラベル
                    "signal_labels": ",".join(signal_label_list),  # 画像内の 15/16 を列挙
                    "bbox": "{},{},{},{}".format(*max_xyxy),
                    "width": w,
                    "height": h
                })

    # 書き出し
    with open(out_dir / "annotation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "signal_labels", "bbox", "width", "height"])
        writer.writeheader()
        writer.writerows(annotation_rows)

    with open(out_dir / "signal.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["signal_file", "label", "src_image"])
        writer.writeheader()
        writer.writerows(signal_rows)

    print(f"[OK] images_15_16: {len(list(out_images_dir.glob('*.jpg')))}")
    print(f"[OK] signal crops : {len(signal_rows)}")
    print(f"[OK] CSVs written : {out_dir/'annotation.csv'}, {out_dir/'signal.csv'}")

if __name__ == "__main__":
    main()
