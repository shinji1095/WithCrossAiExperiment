# preprocess_signals.py
import argparse
from pathlib import Path
import csv
import cv2
from typing import List, Tuple

CLS_RED = 15
CLS_GREEN = 16
CLS_INT = 13                  # 追加：抽出対象に含めるクラス

# ----------------------------
# YOLO (cx,cy,w,h) -> pixel xyxy
# ----------------------------
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

# ----------------------------
# レターボックス（縮小→パディング）
#  - out_size: (H,W)
#  - downscale_only=True の場合、拡大しない（<=480なら等倍でパディングのみ）
#  - 返り値: img_out, scale, pad_left, pad_top
# ----------------------------
def letterbox_pad(img, out_h=480, out_w=480, pad_color=0, downscale_only=True):
    h, w = img.shape[:2]
    if downscale_only:
        scale = min(1.0, out_w / w, out_h / h)
    else:
        scale = min(out_w / w, out_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w != w or new_h != h:
        img_rs = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_rs = img

    canvas = (pad_color if isinstance(pad_color, int) else tuple(pad_color))
    out = cv2.copyMakeBorder(
        img_rs,
        top=(out_h - new_h) // 2,
        bottom=out_h - new_h - (out_h - new_h) // 2,
        left=(out_w - new_w) // 2,
        right=out_w - new_w - (out_w - new_w) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=canvas
    )
    pad_left = (out_w - new_w) // 2
    pad_top  = (out_h - new_h) // 2
    return out, scale, pad_left, pad_top

# ----------------------------
# ラベルファイルのパース
#  - 15/16（信号）を sig_boxes に、13 を cls13_boxes に
# ----------------------------
def parse_label_file(txt_path: Path):
    sig_boxes = []   # [(cls_id,(cx,cy,w,h)), ...] for 15/16
    cls13_boxes = [] # ditto for 13
    if not txt_path.exists():
        return sig_boxes, cls13_boxes
    txt = txt_path.read_text(encoding="utf-8").strip()
    if not txt:
        return sig_boxes, cls13_boxes
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0]); cx, cy, w, h = map(float, parts[1:])
        if cls_id in (CLS_RED, CLS_GREEN):
            sig_boxes.append((cls_id, (cx, cy, w, h)))
        elif cls_id == CLS_INT:
            cls13_boxes.append((cls_id, (cx, cy, w, h)))
    return sig_boxes, cls13_boxes

# ----------------------------
# アスペクト比フィルタ
# ----------------------------
def aspect_ok(h: int, w: int, amin: float, amax: float) -> bool:
    if h <= 0 or w <= 0:
        return False
    ar = h / float(w)
    return (amin <= ar <= amax)

# ----------------------------
# bbox のスケール・平行移動（レターボックス用）
#  - scale, pad_left, pad_top に基づいて xyxy を変換
# ----------------------------
def transform_bbox_xyxy(xyxy: Tuple[int,int,int,int],
                        scale: float, pad_left: int, pad_top: int,
                        out_w: int, out_h: int) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = xyxy
    nx1 = int(round(x1 * scale)) + pad_left
    ny1 = int(round(y1 * scale)) + pad_top
    nx2 = int(round(x2 * scale)) + pad_left
    ny2 = int(round(y2 * scale)) + pad_top
    # clamp
    nx1 = max(0, min(nx1, out_w-1)); nx2 = max(1, min(nx2, out_w))
    ny1 = max(0, min(ny1, out_h-1)); ny2 = max(1, min(ny2, out_h))
    if nx2 <= nx1: nx2 = min(out_w-1, nx1+1)
    if ny2 <= ny1: ny2 = min(out_h-1, ny1+1)
    return nx1, ny1, nx2, ny2

# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Crop signals (15/16) with AR screening, and build padded 480x480 dataset.")
    ap.add_argument("--dataset_root", type=str, required=True, help="root dir of dataset/")
    ap.add_argument("--out_dir", type=str, default="preprocessed", help="output dir root")
    ap.add_argument("--signal_prefix", type=str, default="sig", help="cropped signal filename prefix")

    # 画像整形
    ap.add_argument("--out_size", type=int, default=480, help="output square size (HxW)")
    ap.add_argument("--pad_color", type=int, nargs="+", default=[0], help="padding color (scalar or BGR triplet)")
    ap.add_argument("--downscale_only", action="store_true",
                    help="do not upscale before padding (default ON if flag present)")
    ap.set_defaults(downscale_only=True)

    # アスペクト比フィルタ（提供サンプル由来：H/W ∈ [1.48, 2.05]）
    ap.add_argument("--aspect_min", type=float, default=1.48, help="min allowed aspect ratio (H/W)")
    ap.add_argument("--aspect_max", type=float, default=2.05, help="max allowed aspect ratio (H/W)")
    ap.add_argument("--save_rejects", action="store_true", help="save filtered-out crops to out_dir/rejects")
    ap.add_argument("--min_side", type=int, default=8, help="reject if min(h, w) < min_side")
    ap.add_argument("--min_area", type=int, default=16, help="reject if (h*w) < min_area")

    args = ap.parse_args()

    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_images_dir = out_dir / f"images_13_15_16_{args.out_size}"
    out_signal_dir = out_dir / "signal"
    out_rejects_dir = out_dir / "rejects"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_signal_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rejects:
        out_rejects_dir.mkdir(parents=True, exist_ok=True)

    annotation_rows = []
    signal_rows = []

    kept_cnt = rej_cnt = 0
    kept_cnt_red = kept_cnt_green = 0
    rej_cnt_red = rej_cnt_green = 0

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    sig_counter = 0

    for sub in subdirs:
        # images / labels の構成差に耐性
        img_dir = sub / "images"
        if not img_dir.exists():  # 2000-2999 形式
            img_dir = sub
        lbl_dir = sub / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            base = img_path.stem
            txt_path = lbl_dir / f"{base}.txt"

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            sig_boxes, cls13_boxes = parse_label_file(txt_path)

            # --- ここがポイント：13/15/16 を含む画像だけ対象 ---
            has_15_16 = len(sig_boxes) > 0
            has_13 = len(cls13_boxes) > 0
            if not (has_15_16 or has_13):
                # 何も対象が無ければスキップ（annotation.csv にも入れない）
                continue

            # ---------- 画像を 480x480 に整形（縮小→パディング） ----------
            pad_color = args.pad_color if len(args.pad_color) == 3 else [args.pad_color[0]]*3
            img_pad, scale, pad_left, pad_top = letterbox_pad(
                img, out_h=args.out_size, out_w=args.out_size,
                pad_color=pad_color, downscale_only=args.downscale_only
            )
            # 保存（抽出対象のみ）
            out_img_path = out_images_dir / img_path.name
            cv2.imwrite(str(out_img_path), img_pad)

            # ---------- 信号クロップ(15/16)：ARスクリーニングして保存 ----------
            signal_label_list: List[str] = []
            max_area = -1
            max_xyxy = None
            max_label = None

            for cls_id, (cx, cy, ww, hh) in sig_boxes:
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, ww, hh, w, h)
                crop = img[y1:y2, x1:x2].copy()
                ch, cw = crop.shape[:2]
                if crop.size == 0:
                    continue

                # スクリーニング
                ok_aspect = aspect_ok(ch, cw, args.aspect_min, args.aspect_max)
                ok_size = (min(ch, cw) >= args.min_side) and (ch * cw >= args.min_area)
                accept = ok_aspect and ok_size

                sig_counter += 1
                sig_name = f"{args.signal_prefix}_{base}_{sig_counter:06d}.jpg"

                if accept:
                    cv2.imwrite(str(out_signal_dir / sig_name), crop)
                    mapped = 0 if cls_id == CLS_RED else 1  # 15->0, 16->1
                    signal_rows.append({
                        "signal_file": sig_name,
                        "label": mapped,
                        "src_image": str(img_path)
                    })
                    kept_cnt += 1
                    if mapped == 0: kept_cnt_red += 1
                    else: kept_cnt_green += 1
                else:
                    if args.save_rejects:
                        dbg = crop.copy()
                        try:
                            cv2.putText(dbg, f"AR={ch/cw:.2f} {cw}x{ch}",
                                        (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                        except Exception:
                            pass
                        cv2.imwrite(str(out_rejects_dir / sig_name), dbg)
                    rej_cnt += 1
                    if cls_id == CLS_RED: rej_cnt_red += 1
                    elif cls_id == CLS_GREEN: rej_cnt_green += 1

                # 最大面積の信号bbox（元画像座標系）
                signal_label_list.append(str(cls_id))
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    max_xyxy = (x1, y1, x2, y2)
                    max_label = "0" if cls_id == CLS_RED else "1"

            # ---------- annotation.csv 行を作成 ----------
            if max_xyxy is None:
                # 15/16 が無い（13のみ）→ label=NONE, bbox=""
                annotation_rows.append({
                    "filename": str(img_path.name),   # ← パディング済み画像のパス
                    "label": "NONE",
                    "signal_labels": "",             # 15/16が無いので空
                    "bbox": "",
                    "width": args.out_size,
                    "height": args.out_size
                })
            else:
                # 15/16 の最大bboxをパディング後座標に変換
                px1, py1, px2, py2 = transform_bbox_xyxy(
                    max_xyxy, scale, pad_left, pad_top,
                    out_w=args.out_size, out_h=args.out_size
                )
                annotation_rows.append({
                    "filename": str(img_path.name),
                    "label": max_label,  # 最大面積の信号
                    "signal_labels": ",".join(signal_label_list),  # 画像内の 15/16 を列挙
                    "bbox": "{},{},{},{}".format(px1, py1, px2, py2),
                    "width": args.out_size,
                    "height": args.out_size
                })

    # ---------- CSV 書き出し ----------
    with open(out_dir / "annotation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "signal_labels", "bbox", "width", "height"])
        writer.writeheader()
        writer.writerows(annotation_rows)

    with open(out_dir / "signal.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["signal_file", "label", "src_image"])
        writer.writeheader()
        writer.writerows(signal_rows)

    # ---------- ログ ----------
    print(f"[OK] extracted images (13/15/16 only): {len(list(out_images_dir.glob('*.jpg')))}")
    print(f"[OK] kept signals (AR/size passed):   {kept_cnt}  [RED:{kept_cnt_red}  GREEN:{kept_cnt_green}]")
    print(f"[OK] rejected signals:                 {rej_cnt}  [RED:{rej_cnt_red} GREEN:{rej_cnt_green}]")
    print(f"[OK] CSVs: {out_dir/'annotation.csv'}, {out_dir/'signal.csv'}")

if __name__ == "__main__":
    main()
