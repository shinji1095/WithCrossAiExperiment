import random
from pathlib import Path

import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from typing import List, Tuple, Optional

class SignalCutMixDataset(Dataset):
    """
    A Dataset that performs 'SignalMix': with some probability, swaps the traffic-signal
    region in each image with a randomly chosen signal crop (RED/GREEN) and updates the label.
    """
    def __init__(self,
                 data_csv: str,
                 signal_csv: str,
                 signal_dir: str,
                 transform=None,
                 mix_prob: float = 0.5):
        """
        Args:
          data_csv    Path to dataset.csv (preprocessed)
          signal_csv  Path to signal.csv (preprocessed)
          signal_dir  Directory containing cropped signal images
          transform   Albumentations or other transform: called as transform(image=...)
          mix_prob    Probability of applying SignalMix when label != 'NONE'
        """
        self.df = pd.read_csv(data_csv)
        self.signal_df = pd.read_csv(signal_csv)
        self.signal_dir = Path(signal_dir)
        self.transform = transform
        self.mix_prob = mix_prob

        # group signal crops by label
        self.signals = {'RED': [], 'GREEN': []}
        for _, row in self.signal_df.iterrows():
            self.signals[row['signal']].append(row['signal_file'])

        # label → int mapping
        self.label_map = {'NONE': 0, 'RED': 1, 'GREEN': 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['filename'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = row['signal']
        bbox_str = row['bbox']
        if label != 'NONE' and bbox_str:
            x1, y1, x2, y2 = map(int, bbox_str.split(','))
        else:
            x1 = y1 = x2 = y2 = None

        # --- SignalMix ---
        if label != 'NONE' and random.random() < self.mix_prob:
            # choose new signal class at random (or could restrict to same class)
            new_label = random.choice(['RED', 'GREEN'])
            choice_list = self.signals[new_label]
            if choice_list:
                sig_file = random.choice(choice_list)
                sig_img = cv2.imread(str(self.signal_dir / sig_file))
                # resize crop to current bbox size
                h, w = y2 - y1, x2 - x1
                sig_resized = cv2.resize(sig_img, (w, h))
                img[y1:y2, x1:x2] = sig_resized
                label = new_label  # pasted label

        # apply other transforms (e.g. Albumentations → ToTensorV2)
        if self.transform:
            out = self.transform(image=img)
            img = out['image']

        # convert label to tensor
        label_idx = self.label_map[label]
        return img, torch.tensor(label_idx, dtype=torch.long)

class SignalMixClassificationDataset(Dataset):
    """
    preprocessed/annotation.csv 互換の分類用データセット。
    返り値: (image_tensor, label_idx(int 0..2), bboxes[List[(x1,y1,x2,y2)]])
    - 列の自動検出: 画像パスは {filename,image,img,img_path,path,filepath}、ラベルは {signal,label,class,target,y,y_cls}
    - ラベルは 'NONE/RED/GREEN' も 0/1/2 も受け付けて 0..2 に正規化
    - bbox は 'bbox' 文字列 or x1,y1,x2,y2 の両対応
    - transform.size_hw=(H,W) があれば bbox を自動スケーリング
    - is_train=True のときだけ学習用の拡張を適用
      * transform に以下の属性があれば優先的に利用します:
        - train_transform(image=ndarray)->{'image':tensor}
        - val_transform(image=ndarray)->{'image':tensor}
        - base_transform(image=ndarray)->{'image':tensor}
      * 何もなければ transform(image=ndarray) を呼びます（validationで拡張を避けたい場合は val 用の transform を渡してください）
    """
    _LABEL_CANDS = ["signal", "label", "class", "target", "y", "y_cls"]
    _PATH_CANDS  = ["filename", "image", "img", "img_path", "path", "filepath"]

    def __init__(self,
                 img_dir: str,
                 annotation_csv: str,
                 transform=None,
                 is_train: bool = True):
        self.img_dir = Path(img_dir) if img_dir else Path(".")
        self.df = pd.read_csv(annotation_csv)
        self.transform = transform
        self.is_train = bool(is_train)
        self.num_classes = 3

        # 画像パス列の検出
        self.path_col = None
        for c in self._PATH_CANDS:
            if c in self.df.columns:
                self.path_col = c
                break
        if self.path_col is None:
            raise KeyError(f"Image path column not found. Tried {self._PATH_CANDS}, "
                           f"but got {list(self.df.columns)}")

        # ラベル列の検出
        self.label_col = None
        for c in self._LABEL_CANDS:
            if c in self.df.columns:
                self.label_col = c
                break
        if self.label_col is None:
            raise KeyError(f"Label column not found. Tried {self._LABEL_CANDS}, "
                           f"but got {list(self.df.columns)}")

        # ラベルを 0..2 に正規化（数値/文字列の両方対応）
        self.labels_idx = self._normalize_labels(self.df[self.label_col])

        # クラス分布（損失の重み計算用）
        cnt = Counter(self.labels_idx)
        self.class_counts = [cnt.get(i, 0) for i in range(self.num_classes)]

        # 目的サイズ（transform.size_hw）を拾う
        self.out_size: Optional[Tuple[int, int]] = None
        if self.transform is not None and hasattr(self.transform, "size_hw"):
            sz = getattr(self.transform, "size_hw")
            if isinstance(sz, (tuple, list)) and len(sz) == 2:
                self.out_size = (int(sz[0]), int(sz[1]))  # (H, W)

    # ---------- helpers ----------
    def _normalize_labels(self, s: pd.Series) -> List[int]:
        table = {"NONE": 0, "RED": 1, "GREEN": 2}
        idxs: List[int] = []
        for v in s:
            if pd.isna(v):
                idxs.append(0); continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                idxs.append(max(0, min(2, int(v)))); continue
            sv = str(v).strip()
            if sv.isdigit():
                idxs.append(max(0, min(2, int(sv)))); continue
            idxs.append(table.get(sv.upper(), 0))
        return idxs

    def _load_rgb(self, any_path: str):
        p = str(any_path)
        image_path = p if os.path.isabs(p) else str(self.img_dir / p)
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _parse_bboxes(self, row: pd.Series) -> List[Tuple[int, int, int, int]]:
        bbox_str = row.get("bbox", "")
        if isinstance(bbox_str, str) and bbox_str.strip():
            parts = bbox_str.split(",")
            if len(parts) == 4:
                try:
                    x1, y1, x2, y2 = map(float, parts)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if x2 > x1 and y2 > y1:
                        return [(x1, y1, x2, y2)]
                except Exception:
                    pass
        cols = row.index
        if all(k in cols for k in ["x1", "y1", "x2", "y2"]):
            try:
                x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                if x2 > x1 and y2 > y1:
                    return [(x1, y1, x2, y2)]
            except Exception:
                pass
        return []

    @staticmethod
    def _scale_bboxes(bxs: List[Tuple[int, int, int, int]],
                      src_w: int, src_h: int,
                      dst_w: int, dst_h: int) -> List[Tuple[int, int, int, int]]:
        if not bxs:
            return []
        sx = dst_w / max(1, src_w)
        sy = dst_h / max(1, src_h)
        out: List[Tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in bxs:
            nx1 = int(round(x1 * sx)); ny1 = int(round(y1 * sy))
            nx2 = int(round(x2 * sx)); ny2 = int(round(y2 * sy))
            nx1 = max(0, min(nx1, dst_w - 1))
            ny1 = max(0, min(ny1, dst_h - 1))
            nx2 = max(1, min(nx2, dst_w))
            ny2 = max(1, min(ny2, dst_h))
            if nx2 > nx1 and ny2 > ny1:
                out.append((nx1, ny1, nx2, ny2))
        return out

    # ---------- Dataset API ----------
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 画像
        img = self._load_rgb(row[self.path_col])
        src_h, src_w = img.shape[:2]

        # ラベル index（前計算）
        label_idx = int(self.labels_idx[idx])

        # bbox（必要ならスケール）
        bboxes = self._parse_bboxes(row)
        if self.out_size is not None:
            H, W = self.out_size
            if (src_h, src_w) != (H, W):
                bboxes = self._scale_bboxes(bboxes, src_w, src_h, W, H)

        # 画像変換（is_train のときだけ拡張を使う）
        if self.transform is not None:
            if self.is_train and hasattr(self.transform, "train_transform"):
                img = self.transform.train_transform(image=img)["image"]
                if hasattr(self.transform, "base_transform"):
                    img = self.transform.base_transform(image=img)["image"]
            elif (not self.is_train) and hasattr(self.transform, "val_transform"):
                img = self.transform.val_transform(image=img)["image"]
                if hasattr(self.transform, "base_transform"):
                    img = self.transform.base_transform(image=img)["image"]
            elif hasattr(self.transform, "base_transform"):
                img = self.transform.base_transform(image=img)["image"]
            else:
                # transform が Compose 等のとき。validationで拡張を避けたい場合は
                # val 用に別の transform を渡してください。
                img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, torch.tensor(label_idx, dtype=torch.long), bboxes