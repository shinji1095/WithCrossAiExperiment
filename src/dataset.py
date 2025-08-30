import cv2
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import json

import logging

# logger を用意
logger = logging.getLogger(__name__)

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