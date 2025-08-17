import random
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

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
            self.signals[row['label']].append(row['signal_file'])

        # label → int mapping
        self.label_map = {'NONE': 0, 'RED': 1, 'GREEN': 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['filename'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = row['label']
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
