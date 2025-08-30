import torchvision.transforms.functional as TF

import csv
from collections import defaultdict
import random, math, torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as T


__all__ = [
    "CutMix", "AttentiveCutMix", "SaliencyMix",
    "PuzzleMix", "SnapMix", "KeepAugment",
    "get_mixer", "SignalMix"
]

# ------------------------------------------------------------------------- #
# ヘルパ：長方形を β 分布からサンプリング
# ------------------------------------------------------------------------- #
def _rand_bbox(W: int, H: int, lam: float) -> Tuple[int, int, int, int]:
    cut_ratio = math.sqrt(1. - lam)
    # 最低でも1px は切り出す
    cut_w = max(1, int(W * cut_ratio))
    cut_h = max(1, int(H * cut_ratio))    
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1 = int(torch.clamp(torch.tensor(cx - cut_w // 2), 0, W))
    y1 = int(torch.clamp(torch.tensor(cy - cut_h // 2), 0, H))
    x2 = int(torch.clamp(torch.tensor(cx + cut_w // 2), 0, W))
    y2 = int(torch.clamp(torch.tensor(cy + cut_h // 2), 0, H))
    return x1, y1, x2, y2

def _rand_bbox_wh(W: int, H: int, lam: float,
                  min_lam: float = 0.3, max_lam: float = 0.7) -> Tuple[int,int,int,int]:
    lam = float(max(min(lam, max_lam), min_lam))
    cut_rat = (1.0 - lam) ** 0.5
    cut_w = max(1, int(W * cut_rat))
    cut_h = max(1, int(H * cut_rat))
    cx = random.randint(0, W - 1)
    cy = random.randint(0, H - 1)
    x1 = int(max(0, cx - cut_w // 2)); x2 = int(min(W, cx + cut_w // 2))
    y1 = int(max(0, cy - cut_h // 2)); y2 = int(min(H, cy + cut_h // 2))
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


def _rect_overlap(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _any_overlap(rect: Tuple[int,int,int,int], boxes: List[Tuple[int,int,int,int]]) -> bool:
    for bx in boxes:
        if _rect_overlap(rect, bx):
            return True
    return False

# ------------------------------------------------------------------------- #
# 1. CutMix（ベースライン）
# ------------------------------------------------------------------------- #
class CutMix:
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """
        x: (B,3,H,W), y: (B,C) or (B,)
        ※ “コピー元は clone() から読む” ことで伝播混入を防ぐ
        """
        B, _, H, W = x.size()
        if B < 2:                         # 1枚だけなら何もしない
            return x, y
        lam = torch.distributions.Beta(self.beta, self.beta).sample().item()
        idx = torch.randperm(B, device=x.device)
        x1, y1, x2, y2 = _rand_bbox_wh(W, H, lam, 0.3, 0.7)

        src = x.clone()                   # ← 重要：コピー元のスナップショット
        x[:, :, y1:y2, x1:x2] = src[idx, :, y1:y2, x1:x2]

        lam_adj = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        if y.dim() == 2:
            y = lam_adj * y + (1 - lam_adj) * y[idx]
        else:
            y = y[idx]
        return x, y

# ------------------------------------------------------------------------- #
# 2. Attentive CutMix – CAM 最大領域を貼り付ける
#    参考: Uddin et al. 2020 
# ------------------------------------------------------------------------- #
class AttentiveCutMix(CutMix):
    def __init__(self, model, beta=1.0, max_trials: int = 3):
        super().__init__(beta)
        self.model      = model.eval()
        self.max_trials = max_trials

    @torch.no_grad()
    def _top_box(self, img: torch.Tensor) -> Tuple[int,int,int,int]:
        """単一画像 (1,C,H,W) から CAM 最大位置を矩形で取る (簡易版)"""
        cam = self._get_cam(img)          # -> (H,W)
        thresh = cam.mean() + cam.std()   # Max5% 相当
        ys, xs = torch.where(cam >= thresh)
        y1, y2 = ys.min().item(), ys.max().item()
        x1, x2 = xs.min().item(), xs.max().item()
        return x1, y1, x2, y2

    def _get_cam(self, img):
        img = transforms.functional.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
        feat = None
        def hook(_, __, output): nonlocal feat; feat = output
        handle = self.model.layer4[-1].register_forward_hook(hook)
        _ = self.model(img)
        handle.remove()
        cam = feat.squeeze(0).mean(0)     # GAP -> (H,W)
        cam = (cam - cam.min())/(cam.max()-cam.min() + 1e-5)
        return cam

    def __call__(self, x, y):
        batch, _, H, W = x.size()
        index = torch.randperm(batch, device=x.device)

        # 複数回試して mix 領域を探す
        for _ in range(self.max_trials):
            lam   = torch.distributions.Beta(self.beta,self.beta).sample().item()
            cam_boxes = [ self._top_box(x[i:i+1]) for i in range(batch) ]

            # 全サンプルで有効な bbox があるか？
            valid = [(x2>x1 and y2>y1) 
                     for (x1,y1,x2,y2) in cam_boxes]
            if any(valid):
                # 有効サンプルだけ mix 実行
                for i,(x1,y1,x2,y2) in enumerate(cam_boxes):
                    if not (x2>x1 and y2>y1): continue
                    patch = F.interpolate(
                        x[index[i]:index[i]+1],
                        size=(y2-y1, x2-x1),
                        mode="bilinear", align_corners=False
                    ).squeeze(0)
                    x[i,:,y1:y2,x1:x2] = patch
                    # ラベルも一応補正
                    y[i] = lam*y[i] + (1-lam)*y[index[i]]
                return x, y

        # 何度試しても領域が取れなかった → CutMix にフォールバック
        return CutMix(self.beta)(x, y)

# ------------------------------------------------------------------------- #
# 3. SaliencyMix – 勾配サリエンシーマップ最大領域
#    参考: Jiang et al. 2021
# ------------------------------------------------------------------------- #
class SaliencyMix(CutMix):
    def __init__(self, model, beta=1.0, method="grad"):
        super().__init__(beta)
        self.model, self.method = model.eval(), method

    def _saliency(self, img, label):
        img.requires_grad_()
        out = self.model(img)
        loss = F.cross_entropy(out, label)
        loss.backward()
        sal = img.grad.data.abs().max(dim=1)[0]        # (1,H,W)
        return (sal - sal.min())/(sal.max()-sal.min()+1e-5)

    def __call__(self, x, y):
        batch, _, H, W = x.size()
        index = torch.randperm(batch, device=x.device)
        # 各画像の最 saliency bbox (半面 CutMix と同サイズ)
        lam = torch.distributions.Beta(self.beta, self.beta).sample().item()
        x1, y1, x2, y2 = _rand_bbox(W, H, lam)

        for i in range(batch):
            if y.dim() == 1:
                lbl = y[i:i+1]                    # shape (1,)
            else:
                lbl = y[i:i+1].argmax(dim=1)      # shape (1,)
            sal = self._saliency(x[i:i+1], lbl)
            ys, xs = torch.where(sal.squeeze(0) > sal.mean())
            if len(xs) == 0: continue
            # bbox 内に収まる最大領域 / fallback=rand
            ys = ys.clamp(y1, y2-1); xs = xs.clamp(x1, x2-1)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            x[i,:,y_min:y_max,x_min:x_max] = x[index[i],:,y_min:y_max,x_min:x_max]

        lam_adj = 1 - (x2-x1)*(y2-y1)/(W*H)
        y = lam_adj*y + (1-lam_adj)*y[index]
        return x, y

# ------------------------------------------------------------------------- #
# 4. PuzzleMix – gradient-based mask & optimal transport
#    公式実装が見つかった場合のみ簡易ラッパ
# ------------------------------------------------------------------------- #
class PuzzleMix(CutMix):
    def __init__(self, beta=1.2):
        super().__init__(beta)
        try:
            from puzzlemix import PuzzleMixAug
            self.pm = PuzzleMixAug(beta=beta)
        except ImportError:
            self.pm = None

    def __call__(self, x, y):
        if self.pm is None:               # fallback = CutMix
            return super().__call__(x, y)
        return self.pm(x, y)              # (img, target)

# ------------------------------------------------------------------------- #
# 5. SnapMix – CAM でラベル重み付け
#    参考: Huang et al. 2021
# ------------------------------------------------------------------------- #
class SnapMix(CutMix):
    def __init__(self, model, beta=0.5, max_trials=3):
        super().__init__(beta)
        self.model = model.eval()
        self.max_trials = max_trials

    @torch.no_grad()
    def _cam(self, img, cls_idx):
        img = transforms.functional.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
        feat = None
        def hook(_, __, output): nonlocal feat; feat = output
        h = self.model.layer4[-1].register_forward_hook(hook)
        logit = self.model(img)[0, cls_idx]
        h.remove()
        cam = (feat.squeeze(0) * self.model.fc.weight[cls_idx][:, None, None]).sum(0)
        cam = F.relu(cam)
        cam = (cam - cam.min())/(cam.max()-cam.min()+1e-5)
        return cam
    

    def _get_snap_boxes(self,
                        x: torch.Tensor,
                       y_idx: torch.Tensor
                        ) -> List[Tuple[int,int,int,int]]:
        """
        バッチ x, 各画像のクラス idx y_idx から
        CAM 最大領域の bbox を返す (リスト長=batch)
        """
        batch, _, H, W = x.size()
        boxes: List[Tuple[int,int,int,int]] = []
        for i in range(batch):
            cam = self._cam(x[i:i+1], int(y_idx[i]))
            # 閾値は mean+std のままでもOKですが、調整可
            thresh = cam.mean() + cam.std()
            ys, xs = torch.where(cam >= thresh)
            if len(xs) == 0:
                # 領域が取れなかったらランダム bbox にフォールバック
                boxes.append(_rand_bbox(W, H,
                                        lam=0.5))  # lam=0.5 で中間サイズ
            else:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                # 座標クリップ
                x1, x2 = max(0,x1), min(W, x2)
                y1, y2 = max(0,y1), min(H, y2)
                boxes.append((x1, y1, x2, y2))
        return boxes

    def __call__(self, x, y):
        if y.dim()>1: y_idx = y.argmax(dim=1)
        else:          y_idx = y

        for _ in range(self.max_trials):
            y_idx = y.argmax(dim=1) if y.dim()>1 else y
            cam_boxes = self._get_snap_boxes(x, y_idx)
            # 有効 bbox があるか？
            valid = [(x1<x2 and y1<y2) for x1,y1,x2,y2 in cam_boxes]
            if any(valid):
                # mix 実行
                for i,(x1,y1,x2,y2) in enumerate(cam_boxes):
                    if not (x1<x2 and y1<y2): continue
                    patch = x[i:i+1, :, y1:y2, x1:x2]
                    x[i,:,y1:y2,x1:x2] = patch                      # ここは SnapMix の独自論理に置き換え
                    y[i] = y_idx[i]                                 # ラベル上書き等
                return x, y

        # 全試行失敗 → Fallback
        return CutMix(self.beta)(x, y)

# ------------------------------------------------------------------------- #
# 6. KeepAugment – Saliency で “重要領域を切らない” CutMix
#    参考: Gong et al. 2021
# ------------------------------------------------------------------------- #
class KeepAugment(CutMix):
    """
    CAM（Saliency）に基づいて重要領域だけ残し、
    残りを他サンプルのパッチで埋めるミキサー。
    max_trials 回だけ有効領域を探し、それでもなければ CutMix にフォールバック。
    """
    def __init__(self,
                 model,
                 beta: float = 1.0,
                 tau: float = 0.15,
                 max_trials: int = 3):
        super().__init__(beta)
        self.model      = model.eval()
        self.tau        = tau
        self.max_trials = max_trials

    @torch.no_grad()
    def _saliency(self,
                  img: torch.Tensor,
                  label: torch.Tensor) -> torch.Tensor:
        """
        1枚 (1,C,H,W) から CAM を計算し、
        正規化した (H,W) の saliency map を返す。
        """
        # ImageNet 標準化
        img = transforms.functional.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
        feat = None
        def hook(_, __, out):
            nonlocal feat
            feat = out
        handle = self.model.layer4[-1].register_forward_hook(hook)
        _ = self.model(img)
        handle.remove()

        cam = feat.squeeze(0).mean(0)  # (H,W)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-5)
        return cam

    def __call__(self,
                 x: torch.Tensor,   # (B,3,H,W)
                 y: torch.Tensor    # (B,) or (B,C)
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, H, W = x.size()
        indices = torch.randperm(batch, device=x.device)

        # max_trials 回だけ有効領域を探す
        for _ in range(self.max_trials):
            lam = torch.distributions.Beta(self.beta, self.beta).sample().item()
            bx1, by1, bx2, by2 = _rand_bbox(W, H, lam)
            # 無効領域はスキップ
            if bx2 <= bx1 or by2 <= by1:
                continue

            # 最初のサンプルで Saliency をチェック
            if y.dim() > 1:
                lbl0 = y[0:1].argmax(dim=1)
            else:
                lbl0 = y[0:1]
            s_map = self._saliency(x[0:1], lbl0)

            # 閾値以上の画素が一つでもあれば mix 実行
            if (s_map[by1:by2, bx1:bx2] > self.tau).any():
                for i in range(batch):
                    idx = int(indices[i].item())  # Python int に
                    h = by2 - by1
                    w = bx2 - bx1

                    patch = F.interpolate(
                        x[idx:idx+1],
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)  # (3,h,w)

                    x[i, :, by1:by2, bx1:bx1+w] = patch

                    # ラベルも線形補間（整数 or one-hot 共に対応）
                    if y.dim() == 1:
                        y[i] = lam * y[i] + (1 - lam) * y[idx]
                    else:
                        y[i] = lam * y[i] + (1 - lam) * y[idx]

                return x, y

        # 全試行失敗 → CutMix にフォールバック
        return CutMix(self.beta)(x, y)

# ------------------------------------------------------------------------- #
# 7. SignalMix –  src 画像の「信号 bbox」へ別フォルダの信号画像を 1 枚貼り付け
# ------------------------------------------------------------------------- #
class SignalMix:
    """
    SignalMix:
      1) （任意）CutMix
         - 矩形は *dst* と *src* の両方の signal bbox を避けて選ぶ
         - さらに CutMix 後に dst の signal 中心半径 r を元画素で復元（保護）
      2) signal crops を dst の signal bbox へ貼付
      3) ラベルは signal.csv の 0/1 → {1:RED, 2:GREEN} に上書き

    cfg 主要パラメータ:
      mix_prob, use_cutmix, cutmix_prob, beta, min_lam, max_lam,
      protect_radius, max_trials, none_index,
      signal_dir, signal_csv
    """
    needs_bboxes = True  # visualize/train 側で bbox を渡すトリガー

    def __init__(self,
                 signal_dir: str | Path,
                 csv_path: str | Path,
                 mix_prob: float = 0.5,
                 use_cutmix: bool = True,
                 cutmix_prob: float = 1.0,
                 beta: float = 1.0,
                 min_lam: float = 0.3,
                 max_lam: float = 0.7,
                 protect_radius: int = 50,
                 max_trials: int = 25,
                 none_index: int = 0):
        self.mix_prob = float(mix_prob)
        self.use_cutmix = bool(use_cutmix)
        self.cutmix_prob = float(cutmix_prob)
        self.beta = float(beta)
        self.min_lam = float(min_lam)
        self.max_lam = float(max_lam)
        self.protect_radius = int(protect_radius)
        self.max_trials = int(max_trials)
        self.none_index = int(none_index)

        # signal crops を読み込み（0..1, CHW）
        self.signal_buf: Dict[str, torch.Tensor] = {}
        for p in sorted(Path(signal_dir).glob("*.[jp][pn]g")):
            self.signal_buf[p.name] = TF.to_tensor(Image.open(p).convert("RGB"))
        if not self.signal_buf:
            raise FileNotFoundError(f"No images found in {signal_dir}")

        # signal.csv: {signal_file -> 0/1}
        self.sig_label: Dict[str, int] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.sig_label[row["signal_file"]] = int(row["label"])
        if not self.sig_label:
            raise FileNotFoundError(f"No valid entries in {csv_path}")

        # 0/1 -> {1:RED, 2:GREEN}
        self.sig_to_train = {0: 1, 1: 2}

        # 可視化用: 直近の CutMix 矩形（各サンプル）
        self.last_cutmix_boxes: Optional[List[Tuple[int,int,int,int]]] = None

    @staticmethod
    def _resize_tensor(img_chw: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        return F.interpolate(img_chw.unsqueeze(0), size=(out_h, out_w),
                             mode="bilinear", align_corners=False).squeeze(0)

    def _centers_from_bboxes(self, bxs: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int]]:
        cs = []
        for (x1, y1, x2, y2) in bxs:
            cs.append(((x1 + x2) // 2, (y1 + y2) // 2))
        return cs

    def _protect_dst_centers(self, imgs: torch.Tensor,
                             before: torch.Tensor,
                             bboxes_batch: List[List[Tuple[int,int,int,int]]]) -> None:
        """
        CutMix 後に、dst 側の signal 中心（半径 protect_radius）を before の画素で復元
        """
        if self.protect_radius <= 0:
            return
        B, C, H, W = imgs.shape
        r2 = self.protect_radius * self.protect_radius
        for i in range(B):
            cs = self._centers_from_bboxes(bboxes_batch[i])
            if not cs:
                continue
            for (cx, cy) in cs:
                x0 = max(0, cx - self.protect_radius)
                y0 = max(0, cy - self.protect_radius)
                x1 = min(W, cx + self.protect_radius)
                y1 = min(H, cy + self.protect_radius)
                if x1 <= x0 or y1 <= y0:
                    continue
                yy = torch.arange(y0, y1, device=imgs.device) - cy
                xx = torch.arange(x0, x1, device=imgs.device) - cx
                Y, X = torch.meshgrid(yy, xx, indexing='ij')
                mask = (X*X + Y*Y) <= r2  # (h,w) bool
                # 画素復元（元の dst を優先）
                patch = imgs[i, :, y0:y1, x0:x1]
                patch0 = before[i, :, y0:y1, x0:x1]
                patch[:, mask] = patch0[:, mask]

    def _maybe_cutmix(self,
                      imgs: torch.Tensor,
                      bboxes_batch: Optional[List[List[Tuple[int,int,int,int]]]]) -> torch.Tensor:
        """
        CutMix:
          - 画像クローン src をコピー元に使用して“連鎖混入”を防止
          - 矩形は dst と src の signal bbox のどちらとも非交差となるまで試行
          - 施工後に dst の signal 中心を復元
        """
        self.last_cutmix_boxes = None
        if not self.use_cutmix or self.cutmix_prob <= 0.0:
            return imgs
        B, C, H, W = imgs.shape
        if B < 2:
            return imgs
        if random.random() > self.cutmix_prob:
            return imgs

        idx = torch.roll(torch.arange(B, device=imgs.device), shifts=1, dims=0)
        src = imgs.clone()      # ← コピー元のスナップショット
        before = imgs.clone()   # ← 後で中心を復元するための“元の dst”

        boxes_record: List[Tuple[int,int,int,int]] = []

        for i in range(B):
            # dst/src の bbox 群
            dst_bxs = bboxes_batch[i] if bboxes_batch is not None else []
            src_bxs = bboxes_batch[int(idx[i].item())] if bboxes_batch is not None else []

            # 矩形サンプリング（非交差まで）
            rect = None
            for _ in range(self.max_trials):
                lam = torch.distributions.Beta(self.beta, self.beta).sample().item()
                x1, y1, x2, y2 = _rand_bbox_wh(W, H, lam, self.min_lam, self.max_lam)
                candidate = (x1, y1, x2, y2)
                if _any_overlap(candidate, dst_bxs) or _any_overlap(candidate, src_bxs):
                    continue  # signal を含むのでやり直し
                rect = candidate
                break

            if rect is None:
                # 適切な矩形が見つからなければこのサンプルはスキップ
                boxes_record.append((0, 0, 0, 0))
                continue

            x1, y1, x2, y2 = rect
            imgs[i, :, y1:y2, x1:x2] = src[idx[i], :, y1:y2, x1:x2]
            boxes_record.append(rect)

        # signal の中心を保護（半径 r）
        self._protect_dst_centers(imgs, before, bboxes_batch or [[] for _ in range(B)])

        self.last_cutmix_boxes = boxes_record
        return imgs

    def __call__(self,
                 imgs: torch.Tensor,             # (B,3,H,W) すでに学習正規化域
                 labels: torch.Tensor,           # (B, num_classes) or (B,)
                 signal_bboxes: Optional[List[List[Tuple[int,int,int,int]]]] = None):

        device = imgs.device
        B, C, H, W = imgs.shape

        # (1) CutMix（signal 非交差 & 中心保護）
        imgs = self._maybe_cutmix(imgs, signal_bboxes)

        # (2) bbox へ signal を貼付 & ラベル上書き（labelがNONEのときはスキップ）
        for i in range(B):
            cur_idx = int(labels[i].argmax().item()) if labels.dim() == 2 else int(labels[i].item())
            if cur_idx == self.none_index:
                continue
            if random.random() > self.mix_prob:
                continue
            bxs = signal_bboxes[i] if signal_bboxes is not None else []
            if not bxs:
                continue

            # ランダム signal crop を選択
            sig_name, sig_src01 = random.choice(list(self.signal_buf.items()))
            sig_lbl_rg = self.sig_label.get(sig_name, None)
            if sig_lbl_rg is None:
                continue
            sig_norm = (sig_src01.to(device) - 0.5) / 0.5  # 0..1 -> -1..1

            # 各 bbox へリサイズ貼付
            for (x1, y1, x2, y2) in bxs:
                x1 = max(0, min(int(x1), W - 1)); x2 = max(1, min(int(x2), W))
                y1 = max(0, min(int(y1), H - 1)); y2 = max(1, min(int(y2), H))
                if x2 <= x1 or y2 <= y1:
                    continue
                patch = self._resize_tensor(sig_norm, y2 - y1, x2 - x1)
                imgs[i, :, y1:y2, x1:x2] = patch

            # ラベル上書き（0/1 → 1/2）
            mapped_idx = self.sig_to_train[int(sig_lbl_rg)]
            if labels.dim() == 2:
                labels[i].zero_()
                labels[i, mapped_idx] = 1.0
            else:
                labels[i] = mapped_idx

        return imgs, labels
    
# ------------------------------------------------------------------------- #

# 8. Factory
def get_mixer(cfg: Dict, backbone=None, max_trials=3):
    name = cfg["name"].lower()
    beta = cfg.get("beta", 1.0)
    if name == "none":            return None
    if name == "cutmix":          return CutMix(beta)
    if name == "attentive_cutmix":return AttentiveCutMix(backbone, beta, max_trials)
    if name == "saliencymix":     return SaliencyMix(backbone, beta)
    if name == "puzzlemix":       return PuzzleMix(beta)
    if name == "snapmix":         return SnapMix(backbone, beta, max_trials)
    if name == "keepaugment":     return KeepAugment(backbone, beta,
                                                     cfg.get("keepaugment_tau", .15), max_trials)
    if name == "signalmix":
        return SignalMix(
            signal_dir    = cfg["signal_dir"],
            csv_path      = cfg.get("signal_csv", r"D:\Datasets\WithCross Dataset\vidvip_signal/signal.csv"),
            mix_prob      = cfg.get("prob", 0.5),
            use_cutmix    = cfg.get("use_cutmix", True),
            cutmix_prob   = cfg.get("cutmix_prob", 1.0),
            beta          = cfg.get("beta", 1.0),
            min_lam       = cfg.get("min_lam", 0.3),
            max_lam       = cfg.get("max_lam", 0.7),
            protect_radius= cfg.get("protect_radius", 50),   # ← 追加
            max_trials    = cfg.get("max_trials", 25),       # ← 追加
            none_index    = cfg.get("none_index", 0),
        )
    raise ValueError(f"Unknown augmentation: {name}")