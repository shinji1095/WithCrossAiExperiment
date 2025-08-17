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