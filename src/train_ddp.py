import os
import sys
import time
import json
import random
import logging
import datetime
from typing import Dict, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------------- third‑party utils -----------------
from sklearn.metrics import accuracy_score
import wandb  # ⬅️  NEW: Weights & Biases

# ------------------- project utils ------------------
from loss.loss import get_loss_fn
from models.model import get_model
from config.config import load_all_training_configs
from dataset import SignalSlopeDataset
from dataset_signalmix import SignalMixClassificationDataset
from dataloader import AlbumentationTransform
from metrics.metrics import evaluate_classification, evaluate_regression
from mixer.advanced_mixers import get_mixer
from utils.visualize import write_tensorboard
from utils.early_stopping import EarlyStopping


# --------- helpers ---------
def _is_dist():
    return dist.is_available() and dist.is_initialized()

def _get_world_size():
    return dist.get_world_size() if _is_dist() else 1

def _get_rank():
    return dist.get_rank() if _is_dist() else 0

def _num_classes_from_cfg(cfg, default=3):
    return int(getattr(cfg, "num_classes", default))

def _to_index_labels(y):
    """
    y: (B,) int もしくは (B,C) one-hot/prob
    -> (B,) int
    """
    if y.ndim == 1:
        return y
    return torch.argmax(y, dim=1)

def _distributed_sums(device, loss_sum, correct_sum, sample_sum):
    """
    全プロセスで合計を加算して返す
    """
    if not _is_dist():
        return loss_sum, correct_sum, sample_sum
    t = torch.tensor([loss_sum, correct_sum, sample_sum], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t[0].item(), t[1].item(), t[2].item()

def signalmix_collate(batch):
    """
    batch: [(img(C,H,W), label:int, bboxes:list[xyxy]), ...]
    -> (imgs(B,3,H,W), labels(B,), bboxes:list[list[xyxy]])
    """
    imgs, labels, bboxes = zip(*batch)  # 長さBのタプル
    imgs = torch.stack(imgs, dim=0)
    labels = torch.as_tensor(labels, dtype=torch.long)
    return imgs, labels, list(bboxes)

# -------------------------------------------------
# DDP utility
# -------------------------------------------------

def setup_ddp(rank: int, world_size: int):
    """Initialize torch.distributed."""
    if rank == 0:
        print(f"[DDP] master = {os.getenv('MASTER_ADDR')}:{os.getenv('MASTER_PORT')}")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup_ddp():
    """Terminate torch.distributed."""
    dist.destroy_process_group()


# -------------------------------------------------
# GPU memory utility
# -------------------------------------------------

def free_gpu_memory(*objs):
    for o in objs:
        del o
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# -------------------------------------------------
# DataLoader factory
# -------------------------------------------------


# ===== 追加: シンプルな前処理 Transform =====
class SimpleTransform:
    """
    dataset_signalmix.SignalMixClassificationDataset が期待する
    .base_transform(image=...) -> {'image': torch.Tensor} 形式の軽量Transform。
    - optional resize to (H,W)
    - ToTensor (CHW, float32, 0..1)
    - Normalize: (x - mean)/std  ← SignalMixのパッチ（-1..1）と整合
    """
    def __init__(self, size_hw=None, mean=0.5, std=0.5):
        self.size_hw = tuple(size_hw) if size_hw is not None else None  # (H, W)
        self.mean = float(mean)
        self.std = float(std)

    def base_transform(self, image):
        import cv2
        import torch
        if self.size_hw is not None:
            H, W = self.size_hw
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        t = (t - self.mean) / self.std
        return {"image": t}

def _worker_init_fn(worker_id: int):
    """DataLoader worker の起動時に呼ばれる。乱数・スレッド数を抑制して起動を安定化。"""
    import os
    import random
    import numpy as np
    try:
        import cv2
        cv2.setNumThreads(0)  # OpenCV の内部スレッドを無効化（ワーカー多重起動と相性悪い）
    except Exception:
        pass
    # 各ワーカーで決定的乱数（必要なら）
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    # BLAS / OMP 過剰スレッド抑制
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def get_dataloaders(cfg, rank, world_size):

    # === あなたの元のデータセット生成（そのまま） ===
    train_ds = SignalMixClassificationDataset(
        img_dir=cfg.train_img_dir,
        annotation_csv=cfg.train_file_dir,
        transform=SimpleTransform(),
        is_train=True
    )
    valid_ds = SignalMixClassificationDataset(
        img_dir=cfg.valid_img_dir,
        annotation_csv=cfg.valid_file_dir,
        transform=SimpleTransform(),
        is_train=False
    )

    # DDP Sampler（そのまま）
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    # ===== Windows 既定の workers を穏当にする =====
    import platform
    default_workers = 0 if platform.system() == "Windows" else 4
    num_workers = int(getattr(cfg, "num_workers", default_workers))

    # persistent_workers / prefetch_factor は num_workers>0 の時のみ有効
    loader_common_kwargs = dict(
        pin_memory=True,
        collate_fn=signalmix_collate,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
    )
    if num_workers > 0:
        loader_common_kwargs.update(
            dict(persistent_workers=True, prefetch_factor=getattr(cfg, "prefetch_factor", 2))
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        **loader_common_kwargs,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        **loader_common_kwargs,
    )

    return train_loader, valid_loader, train_sampler


# -------------------------------------------------
# Training helpers (AMP) — FIXED & COMPLETE
# -------------------------------------------------
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

def _is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    loss_fn,
    device,
    task: str,
    mixer,                # None | callable; SignalMix は needs_bboxes 属性あり
    num_classes: int,
    cfg,
    logger,
):
    """
    - 分類: mixer が有効なら one-hot soft targets を用いた soft-CE を使用
      → accuracy は mixer で書き換えられたターゲット（y_soft の argmax）で算出
    - マルチタスク/回帰: 既存仕様を維持
    - DDP rank 安全化 / AMP / NaN ガードは元の動作を踏襲
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    use_mixer = mixer is not None and str(cfg.AUGMENTATION.get("name", "none")).lower() != "none"
    pbar_disable = not _is_main_process()

    for batch in tqdm(loader, desc="Training", disable=pbar_disable):
        # -------------------------
        # Unpack batch safely
        # -------------------------
        # 期待される形式:
        #  - 分類 (SignalMix DS): (imgs, labels_idx, bboxes)
        #  - 分類 (標準DS):       (imgs, labels_idx)
        #  - マルチタスク:        (imgs, cls_labels, slope_targets)
        #  - 回帰:                (imgs, targets)
        def to_dev(x):
            return x.to(device, non_blocking=True) if torch.is_tensor(x) else x

        if isinstance(batch, (list, tuple)):
            batch = list(batch)
        else:
            batch = [batch]

        # 画像テンソル
        images = to_dev(batch[0])

        # 2番目以降はタスクにより可変
        cls_labels = None
        slope_targets = None
        bboxes = None

        if task == "multitask":
            # (imgs, cls_labels, slope_targets)
            cls_labels = to_dev(batch[1])
            slope_targets = to_dev(batch[2])
        elif task == "classification":
            cls_labels = to_dev(batch[1])
            # bboxes があればそのまま（list のまま）使う
            if len(batch) > 2 and not torch.is_tensor(batch[2]):
                bboxes = batch[2]
        else:  # regression
            # (imgs, targets)
            slope_targets = to_dev(batch[1])

        # -------------------------
        # Advanced-Mixers (Classification only)
        # -------------------------
        target_cls_soft = None
        if use_mixer and task == "classification":
            # one-hot soft target を mixer に渡す
            y_one = F.one_hot(cls_labels, num_classes=num_classes).float()
            try:
                if getattr(mixer, "needs_bboxes", False):
                    images, y_soft = mixer(images, y_one, bboxes)
                else:
                    images, y_soft = mixer(images, y_one)
            except Exception as e:
                logger.error(f"Mixer failed: {e}; skipping mixing this batch.")
                y_soft = y_one  # フォールバック
            target_cls_soft = y_soft  # 以後の loss/acc に使用

        optimizer.zero_grad(set_to_none=True)

        # -------------------------
        # Forward & Loss (AMP)
        # -------------------------
        with torch.autocast(device_type=device.type):
            output = model(images)

            if task == "multitask":
                # output: (signal_pred, slope_pred)
                signal_pred, slope_pred = output
                if (not torch.isfinite(signal_pred).all()) or (not torch.isfinite(slope_pred).all()):
                    logger.warning("NaN/Inf detected in model output – skipping batch")
                    continue
                loss = loss_fn(signal_pred, cls_labels, slope_pred, slope_targets)
                if not torch.isfinite(loss):
                    logger.warning("NaN/Inf detected in loss – skipping batch")
                    continue

                # accuracy for classification head
                _, predicted = torch.max(signal_pred, 1)
                correct += (predicted == cls_labels).sum().item()
                total += cls_labels.size(0)

            elif task == "classification":
                if not torch.isfinite(output).all():
                    logger.warning("NaN/Inf detected in output – skipping batch")
                    continue

                if target_cls_soft is not None:
                    # Soft cross-entropy: −Σ q log p
                    logp = F.log_softmax(output, dim=1)
                    loss = -(target_cls_soft * logp).sum(dim=1).mean()
                    # accuracy は「現在のターゲット」に合わせる（mixer がラベルを書き換えるため）
                    hard_targets_for_acc = torch.argmax(target_cls_soft, dim=1)
                else:
                    loss = loss_fn(output, cls_labels)
                    hard_targets_for_acc = cls_labels

                if not torch.isfinite(loss):
                    logger.warning("NaN/Inf detected in loss – skipping batch")
                    continue

                _, predicted = torch.max(output, 1)
                correct += (predicted == hard_targets_for_acc).sum().item()
                total += hard_targets_for_acc.size(0)

            else:  # regression
                if not torch.isfinite(output).all():
                    logger.warning("NaN/Inf detected in regression output – skipping batch")
                    continue
                loss = loss_fn(output, slope_targets)
                if not torch.isfinite(loss):
                    logger.warning("NaN/Inf detected in loss – skipping batch")
                    continue

        # -------------------------
        # Backward (AMP)
        # -------------------------
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().item())

    acc = (correct / total * 100.0) if (task != "regression" and total > 0) else 0.0
    denom = max(1, len(loader))
    return total_loss / denom, acc


def validate_one_epoch(model, loader, loss_fn, device, task: str, logger):
    """
    - 評価時は mixer は使わず、従来どおり hard target で計測
    - 既存のメトリクス構築（classification/regression/multitask）はそのまま
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true_cls: List[int] = []
    y_pred_cls: List[int] = []
    y_prob_cls: List[np.ndarray] = []
    y_true_reg: List[float] = []
    y_pred_reg: List[float] = []

    pbar_disable = not _is_main_process()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", disable=pbar_disable):
            def to_dev(x):
                return x.to(device, non_blocking=True) if torch.is_tensor(x) else x

            if isinstance(batch, (list, tuple)):
                batch = list(batch)
            else:
                batch = [batch]

            images = to_dev(batch[0])

            # 2番目以降（タスク別）
            cls_labels = None
            slope_targets = None
            if task == "multitask":
                cls_labels = to_dev(batch[1])
                slope_targets = to_dev(batch[2])
            elif task == "classification":
                cls_labels = to_dev(batch[1])
            else:
                slope_targets = to_dev(batch[1])

            with torch.autocast(device_type=device.type):
                output = model(images)

                if task == "multitask":
                    signal_pred, slope_pred = output
                    if (not torch.isfinite(signal_pred).all()) or (not torch.isfinite(slope_pred).all()):
                        logger.warning("NaN/Inf detected in output – skipping sample")
                        continue
                    loss = loss_fn(signal_pred, cls_labels, slope_pred, slope_targets)
                    if not torch.isfinite(loss):
                        logger.warning("NaN/Inf detected in loss – skipping sample")
                        continue

                    # collect classification metrics
                    y_true_cls.extend(cls_labels.cpu().tolist())
                    y_pred_cls.extend(signal_pred.argmax(dim=1).cpu().tolist())
                    y_prob_cls.extend(torch.softmax(signal_pred, dim=1).cpu().numpy())
                    # collect regression metrics
                    y_true_reg.extend(slope_targets.cpu().tolist())
                    y_pred_reg.extend(torch.tanh(slope_pred).cpu().tolist())

                    # accuracy for classification head
                    _, predicted = torch.max(signal_pred, 1)
                    correct += (predicted == cls_labels).sum().item()
                    total += cls_labels.size(0)

                elif task == "classification":
                    if not torch.isfinite(output).all():
                        logger.warning("NaN/Inf detected in output – skipping sample")
                        continue
                    loss = loss_fn(output, cls_labels)
                    if not torch.isfinite(loss):
                        logger.warning("NaN/Inf detected in loss – skipping sample")
                        continue

                    y_true_cls.extend(cls_labels.cpu().tolist())
                    y_pred_cls.extend(output.argmax(dim=1).cpu().tolist())
                    y_prob_cls.extend(torch.softmax(output, dim=1).cpu().numpy())

                    _, predicted = torch.max(output, 1)
                    correct += (predicted == cls_labels).sum().item()
                    total += cls_labels.size(0)

                else:  # regression
                    if not torch.isfinite(output).all():
                        logger.warning("NaN/Inf detected in output – skipping sample")
                        continue
                    loss = loss_fn(output, slope_targets)
                    if not torch.isfinite(loss):
                        logger.warning("NaN/Inf detected in loss – skipping sample")
                        continue

                    y_true_reg.extend(slope_targets.cpu().tolist())
                    y_pred_reg.extend(torch.tanh(output).cpu().tolist())

            total_loss += float(loss.detach().item())

    # Accuracy (classification / multitask のみ)
    acc = (correct / total * 100.0) if (task != "regression" and total > 0) else 0.0

    # ---- build metrics dict（既存の evaluate_* をそのまま使用）----
    metrics: Dict[str, Any] = {}
    if task == "classification":
        if len(set(y_true_cls)) >= 2:
            metrics = evaluate_classification(y_true_cls, y_pred_cls, np.array(y_prob_cls), num_classes=3)
    elif task == "regression":
        metrics = evaluate_regression(y_true_reg, y_pred_reg)
    else:  # multitask
        cls_metrics = {}
        if len(set(y_true_cls)) >= 2:
            cls_metrics = evaluate_classification(y_true_cls, y_pred_cls, np.array(y_prob_cls), num_classes=3)
        reg_metrics = evaluate_regression(y_true_reg, y_pred_reg)
        metrics = {"classification": cls_metrics, "regression": reg_metrics}

    denom = max(1, len(loader))
    return total_loss / denom, acc, metrics

# -------------------------------------------------
# Main worker
# -------------------------------------------------

def main_worker(rank: int, world_size: int):
    """Entry point for each DDP process."""

    # ---------------- logger ----------------
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    yaml_path = os.environ.get("TRAIN_CONFIG_PATH", "training_setting.yaml")
    configs = load_all_training_configs(yaml_path)

    for cfg in configs:
        # ----------------‑ wandb init (rank 0 only) ----------------
        time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.run_prefix}.{cfg.model_name}.{time_tag}"

        if rank == 0:
            run = wandb.init(
                project=cfg.wandb_project,   # ← ここが可変に
                name=run_name,
                config={
                    "task": cfg.task,
                    "model_name": cfg.model_name,
                    # 必要なら他のcfgも記録
                },
                reinit=True,
            )
            # Watch gradients & params (optional)
            # wandb.watch(
            #     None,  # we will set later when model is ready
            #     log="gradients",
            #     log_freq=100,
            # )
            logger.info("wandb run initialised")

        # ---------- model ----------
        t0 = time.time()
        logger.info(f"[R{rank}] Loading model {cfg.model_name} …")
        model = get_model(
            cfg.task,
            cfg.model_name,
            num_classes=3,
            dropout_rate=cfg.dropout_rate,
            drop_path_rate=cfg.drop_path_rate,
        ).to(device)
        logger.info(f"[R{rank}] Model loaded ({time.time() - t0:.1f}s)")
        dist.barrier()

        # DDP wrapper
        model = DDP(model, device_ids=[rank])

        if rank == 0:
            wandb.watch(model.module)

        # ---------- data ----------
        train_loader, valid_loader, train_sampler = get_dataloaders(cfg, rank, world_size)

        # ---------- Advanced Mixer ----------
        mixer = get_mixer(cfg.AUGMENTATION, backbone=model.module)

        # ---------- optimizer / loss ----------
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scaler = torch.cuda.amp.GradScaler()
        loss_fn = get_loss_fn(cfg.LOSS, task=cfg.task, class_counts=train_loader.dataset.class_counts)
        if hasattr(loss_fn, "to"):
            loss_fn = loss_fn.to(device)

        # ---------- logging ----------
        if rank == 0:
            os.makedirs(cfg.save_path, exist_ok=True)
            os.makedirs(cfg.result_path, exist_ok=True)
            writer = SummaryWriter(log_dir=os.path.join(cfg.result_path, "tensorboard", cfg.model_name))

        # ---------- early stopping ----------
        early_stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

        best_val_loss = float("inf")
        best_val_acc  = -1.0
        best_ckpt_path = None
        result_log = []

        for epoch in range(cfg.epochs):
            train_sampler.set_epoch(epoch)

            # ----- train -----
            tr_loss, tr_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                loss_fn,
                device,
                cfg.task,
                mixer=mixer,
                num_classes=train_loader.dataset.num_classes,
                cfg=cfg,
                logger=logger,
            )

            # ----- validation (rank 0 only) -----
            if rank == 0:
                va_loss, va_acc, va_metrics = validate_one_epoch(model.module, valid_loader, loss_fn, device, cfg.task, logger)
                early_stopper.step(va_loss)
            else:
                va_loss, va_acc, va_metrics = 0.0, 0.0, {}

            # ----- early‑stop flag sync -----
            stop_tensor = torch.tensor([1 if (rank == 0 and early_stopper.early_stop) else 0], dtype=torch.int, device=device)
            dist.broadcast(stop_tensor, src=0)

            # ----- rank 0: logging -----
            if rank == 0:
                logger.info(
                    f"[{cfg.model_name}] epoch {epoch + 1}/{cfg.epochs} | "
                    f"train {tr_loss:.4f}/{tr_acc:.2f}% | val {va_loss:.4f}/{va_acc:.2f}%"
                )

                # summary for CSV
                result_log.append([epoch + 1, tr_loss, va_loss, tr_acc, va_acc])
                # TensorBoard
                write_tensorboard(writer, epoch + 1, tr_loss, va_loss, tr_acc, va_acc, va_metrics, cfg.task)

                # wandb
                log_data = {
                    "epoch": epoch + 1,
                    "train/loss": tr_loss,
                    "train/acc": tr_acc,
                    "val/loss": va_loss,
                    "val/acc": va_acc,
                }
                if cfg.task in ["classification", "multitask"] and va_metrics:
                    cls = va_metrics["classification"] if cfg.task == "multitask" else va_metrics
                    for k in ["macro_f1", "micro_f1", "cohen_kappa", "mcc"]:
                        if k in cls and cls[k] is not None:
                            log_data[f"val/{k}"] = cls[k]
                if cfg.task in ["regression", "multitask"] and va_metrics:
                    reg = va_metrics["regression"] if cfg.task == "multitask" else va_metrics
                    for k in ["rmse", "mae"]:
                        if k in reg and reg[k] is not None:
                            log_data[f"val/{k}"] = reg[k]
                wandb.log(log_data)

                # best checkpoint
                if va_acc > best_val_acc:
                    best_val_acc = va_acc
                    ckpt_name = f"{cfg.run_prefix}.{cfg.model_name}.e{epoch:03d}.acc{va_acc:.2f}.pth"
                    best_path = os.path.join(cfg.save_path, ckpt_name)
                    torch.save(model.module.state_dict(), best_path)
                    # save to wandb artifact
                    artifact = wandb.Artifact(f"{cfg.run_prefix}", type="model")
                    artifact.add_file(best_path)
                    wandb.log_artifact(artifact)
                    wandb.run.summary["best_val_acc"] = best_val_acc
                    os.remove(best_path)

            # ----- early stopping -----
            if stop_tensor.item():
                if rank == 0:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # ---------- rank 0: CSV output ----------
        if rank == 0:
            df = pd.DataFrame(result_log, columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
            csv_path = os.path.join(cfg.result_path, f"{cfg.model_name}_result.csv")
            df.to_csv(csv_path, index=False)
            wandb.save(csv_path)
            writer.close()
            wandb.finish()

        dist.barrier()
        torch.cuda.empty_cache()

    cleanup_ddp()
