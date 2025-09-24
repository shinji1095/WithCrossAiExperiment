import cv2
import json
import torch
import random
import logging
import numpy as np
import pandas as pd
import albumentations as A

from pathlib import Path
from typing import Tuple, Optional
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

class SimpleTransform:
    """
    - Optional resize to (H,W)
    - ToTensor (CHW, float32, 0..1)
    - Normalize: (x-mean)/std
    SignalMixClassificationDataset が期待する .base_transform(image=...) を提供。
    """
    def __init__(self, size_hw: Optional[Tuple[int,int]]=None, mean=0.5, std=0.5):
        self.size_hw = tuple(size_hw) if size_hw is not None else None
        self.mean = float(mean); self.std = float(std)

    def base_transform(self, image):
        import cv2, torch
        if self.size_hw is not None:
            H, W = self.size_hw
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        t = (t - self.mean) / self.std
        return {"image": t}

class AlbumentationTransform:
    def __init__(self, size_hw=(224, 224)):
        self.size_hw = size_hw
        self.base_transform = A.Compose([
            A.Resize(*size_hw),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            # A.MotionBlur(blur_limit=(3, 7), p=0.3),  
            # A.GaussNoise(std_range=(0.1, 0.2), p=0.3),  
            # A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),  
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def __call__(self, image_np, slope_deg):
        angle = 0.0
        flipped = False

        rotate_transform = A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=1.0)
        rotated = rotate_transform(image=image_np)
        angle = rotate_transform.params.get('angle', 0.0)
        image_np = rotated['image']

        if random.random() < 0.5:
            image_np = cv2.flip(image_np, 1)
            flipped = True
            slope_deg = -slope_deg

        slope_deg -= angle  # 傾き補正（時計回りが正）

        transformed = self.base_transform(image=image_np)
        image_tensor = transformed['image']

        return image_tensor, slope_deg

class SignalSlopeDataset(Dataset):
    def __init__(self, csv_path, image_dir, task='multitask', transform=None, state_filter=None, shuffle=False):
        """
        Args:
            csv_path (str): path to CSV
            image_dir (str or Path): path to image directory
            task (str): 'classification', 'regression', or 'multitask'
            transform (callable): albumentations-based transform
            state_filter (str or None): filter by 'Normal', 'Faded', 'Occlusion', or 'Soiled'
            shuffle (bool): whether to shuffle the data on load
        """
        df = pd.read_csv(csv_path)

        if state_filter is not None:
            df = df[df['state'] == state_filter].reset_index(drop=True)

        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)

        self.df = df
        self.image_dir = Path(image_dir)
        self.task = task
        self.transform = transform
        self.signal2id = {'RED': 0, 'GREEN': 1, 'NONE': 2}

        self.num_classes = len(self.signal2id)
        # “signal” 列を数値ラベルへ変換（欠損や未知は 'None' 扱い）
        self.labels = self.df['signal'].values
        self.class_counts = np.bincount(self.labels, minlength=self.num_classes).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = str(self.image_dir / row['filename'])
        image_np = cv2.imread(image_path)
        if image_np is None:
            logger.error(f"[Dataset] cv2.imread failed: {image_path}")
            raise RuntimeError("cv2.imread returned None")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        signal_label = row['signal']
        slope_label = row['slope_deg'] if pd.notna(row['slope_deg']) else 0.0

        if self.transform is not None:
            image_tensor, slope_label = self.transform(image_np, slope_label)
        else:
            # transform が無い場合のフォールバック
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        if self.task == 'classification':
            return image_tensor, torch.tensor(signal_label, dtype=torch.long)
        elif self.task == 'regression':
            return image_tensor, torch.tensor(slope_label, dtype=torch.float32)
        elif self.task == 'multitask':
            return image_tensor, torch.tensor(signal_label, dtype=torch.long), torch.tensor(slope_label, dtype=torch.float32)
        else:
            raise ValueError("task must be 'classification', 'regression', or 'multitask'")

    def _load_bbox(self, img_path: Path):
        """
        例: 画像と同名 .json から {x1,y1,x2,y2} を取得
        無い場合はゼロ矩形を返す
        """
        js = img_path.with_suffix(".json")
        if js.exists():
            with open(js, "r") as f:
                box = json.load(f)["bbox"]          # [x1,y1,x2,y2]
            return torch.tensor(box, dtype=torch.float32)
        return torch.zeros(4, dtype=torch.float32)