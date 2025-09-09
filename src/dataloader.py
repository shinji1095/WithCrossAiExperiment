import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import torch
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random


class SegmentationAugment:
    """
    画像とマスクを同期で変換する Albumentations パイプライン。
    - 幾何変換のみマスクに適用（Albumentationsが自動で制御）
    - 最後に Normalize & ToTensorV2
    """
    def __init__(self, image_size=(320, 320), mean=0.5, std=0.5):
        H, W = image_size
        self.tfm = A.Compose([
            A.Resize(H, W),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                               border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.Normalize(mean=(mean, mean, mean), std=(std, std, std)),
            ToTensorV2()
        ])

    def __call__(self, image_np, mask_np):
        out = self.tfm(image=image_np, mask=mask_np)
        img_t = out['image']                    # (3,H,W) float32
        msk_t = torch.from_numpy(out['mask']).long()  # (H,W) long
        return img_t, msk_t


class SegmentationDataset(Dataset):
    """
    汎用セグメンテーションDataset。
    - 画像: img_dir 内の *.jpg|*.png 等
    - マスク: mask_dir 内で同stemの .png 等（uint8 のクラスID想定）
    - list_csv が与えられた場合は filename 列に従う（mask列があれば優先）
    """
    def __init__(self, img_dir, mask_dir, list_csv=None, transform=None, num_classes=None):
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.transform = transform
        self.num_classes = int(num_classes) if num_classes is not None else None

        self.samples = []
        if list_csv and Path(list_csv).exists():
            import pandas as pd
            df = pd.read_csv(list_csv)
            for _, r in df.iterrows():
                img = str(self.img_dir / r['filename'])
                if 'mask' in df.columns and isinstance(r['mask'], str):
                    msk = str(self.mask_dir / r['mask']) if self.mask_dir else None
                else:
                    stem = Path(r['filename']).stem
                    msk = str(self.mask_dir / f"{stem}.png") if self.mask_dir else None
                self.samples.append((img, msk))
        else:
            # 画像を走査して対応マスクを推定
            exts = ('.jpg','.jpeg','.png','.bmp')
            for p in sorted(self.img_dir.rglob('*')):
                if p.suffix.lower() in exts:
                    stem = p.stem
                    msk = str(self.mask_dir / f"{stem}.png") if self.mask_dir else None
                    self.samples.append((str(p), msk))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found under: {self.img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, msk_path = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if msk_path is None or not os.path.exists(msk_path):
            # マスクが無い場合は全0
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            mask = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise RuntimeError(f"cv2.imread failed: {msk_path}")
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.transform is not None:
            img_t, mask_t = self.transform(img, mask)
        else:
            # 最低限の Tensor 化
            img_t = torch.from_numpy(img).permute(2,0,1).float()/255.0
            mask_t = torch.from_numpy(mask).long()

        return img_t, mask_t


class AlbumentationTransform:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.base_transform = A.Compose([
            A.Resize(*image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),  
            A.GaussNoise(std_range=(0.1, 0.2), p=0.3),  
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),  
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


"""
from torch.utils.data import DataLoader
from dataloader import SignalSlopeDataset, AlbumentationTransform

transform = AlbumentationTransform()

dataset = SignalSlopeDataset(
    csv_path='csv/training.csv',
    image_dir='image',
    task='multitask',
    transform=AlbumentationTransform(),
    state_filter='Faded',
    shuffle=True 
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
"""