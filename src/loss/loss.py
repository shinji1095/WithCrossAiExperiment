"""
損失関数ユーティリティ
* 通常 CE
* クラス数不均衡に対する Class-Balanced CrossEntropy (CB-Loss)
* FocalLoss
* multitask（分類 + 回帰）
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 補助 ----------
def _cb_weights(class_counts: list[int] | np.ndarray, beta: float = 0.9999) -> torch.Tensor:
    """Class-Balanced Loss で使う重み ω = (1-β)/(1-β^{n_i})"""
    counts = np.asarray(class_counts, dtype=np.float32)
    effective_num = 1.0 - np.power(beta, counts)
    effective_num = np.where(effective_num == 0, 1.0, effective_num)
    weights = (1.0 - beta) / effective_num
    weights /= weights.sum() * len(counts)  # Σω_i = C
    return torch.as_tensor(weights, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0,
                 weight: torch.Tensor | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if getattr(self, "weight", None) is not None else None
        ce = F.cross_entropy(logits, target, weight=w, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# ---------- メインエントリ ----------
def get_loss_fn(cfg_loss: dict,
                task: str = "classification",
                class_counts: list[int] | None = None) -> nn.Module:
    """cfg_loss は config.LOSS（yaml でも可）をそのまま渡す"""
    name = cfg_loss.get("name", "ce")
    label_smoothing = cfg_loss.get("label_smoothing", 0.0)
    alpha, beta_mse = cfg_loss.get("alpha", 1.0), cfg_loss.get("beta_mse", 1.0)

    if task == "multitask":  # 分類 + 回帰
        ce = get_loss_fn(cfg_loss | {"name": "ce"}, "classification", class_counts)
        mse = nn.MSELoss()
        def _multitask(signal_pred, signal_target, slope_pred, slope_target):
            return alpha * ce(signal_pred, signal_target) + beta_mse * mse(slope_pred, slope_target)
        return _multitask

    # ----------- 純分類/回帰 -----------
    if task == "regression":
        return nn.MSELoss()

    # 以下 classification
    weight = None
    if cfg_loss.get("apply_class_balance") and class_counts is not None:
        weight = _cb_weights(class_counts, beta=cfg_loss.get("beta", 0.9999))

    if name in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weight)
    elif name == "weighted_ce":
        if weight is None:
            raise ValueError("apply_class_balance=True が必要です")
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weight)
    elif name == "focal":
        return FocalLoss(gamma=cfg_loss.get("focal_gamma", 2.0), weight=weight)
    else:
        raise ValueError(f"Unsupported loss name: {name}")