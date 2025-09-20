# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import random
import logging
import datetime
import platform
from typing import Any, List

import torch
from torch.amp import GradScaler
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import wandb

# ==== project ====
from loss.loss import get_loss_fn
from models.model import get_model
from config.config import load_all_training_configs
from dataset import SignalSlopeDataset, SimpleTransform, AlbumentationTransform
from dataset_signalmix import SignalMixClassificationDataset
from metrics.metrics import evaluate_classification, evaluate_regression
from mixer.advanced_mixers import get_mixer
from utils.visualize import write_tensorboard
from utils.early_stopping import EarlyStopping

from dataloader import SegmentationDataset, SegmentationAugment  # type: ignore
from models.segmentation_model_factory import create_segmentation_model  # type: ignore


# -------------------------------------------------
# Small helpers
# -------------------------------------------------
def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0

def _world() -> int:
    return dist.get_world_size() if _is_dist() else 1

def _is_main() -> bool:
    return _rank() == 0

def _env_sanity_log(logger: logging.Logger):
    if not _is_main():
        return
    logger.info(
        f"[ENV] pid={os.getpid()} "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"device_count={torch.cuda.device_count()} "
        f"os={platform.system()}-{platform.release()}"
    )
    for k in ["CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","MKL_NUM_THREADS","TRAIN_CONFIG_PATH"]:
        if os.getenv(k) is not None:
            logger.info(f"[ENV] {k}={os.getenv(k)}")

def setup_ddp(rank: int, world_size: int):
    if rank == 0:
        print(f"[DDP] master = {os.getenv('MASTER_ADDR')}:{os.getenv('MASTER_PORT')}")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    dist.barrier()

def cleanup_ddp():
    if _is_dist():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()

def free_gpu_memory(*objs):
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def signalmix_collate(batch):
    """
    Robust collate for (img, label) or (img, label, bboxes).
    - bboxes が無いサンプルは [] で埋める
    - 画像テンソル形状の不一致を検出してエラーメッセージ化
    """
    imgs, labels, bboxes_all = [], [], []
    for i, sample in enumerate(batch):
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            raise RuntimeError(f"Unexpected sample format at idx {i}: {type(sample)}")
        img = sample[0]
        lab = sample[1]
        bbx = sample[2] if len(sample) >= 3 else []
        imgs.append(img)
        labels.append(int(lab))
        bboxes_all.append(bbx)
    # 形状チェック（ここで揃っていないと後段で積めない）
    first_shape = tuple(imgs[0].shape)
    for i, t in enumerate(imgs):
        if tuple(t.shape) != first_shape:
            raise RuntimeError(f"Image shape mismatch in batch: {first_shape} vs {tuple(t.shape)} at local idx {i}")
    return torch.stack(imgs, 0), torch.as_tensor(labels, dtype=torch.long), bboxes_all

def _worker_init_fn(worker_id: int):
    """各 DataLoader worker 起動時に呼ばれる。乱数/スレッド数を抑制。"""
    try:
        import cv2
        cv2.setNumThreads(0)  # OpenCV 内部スレッド無効化
    except Exception:
        pass
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


# -------------------------------------------------
# Dataloaders (with robust logging & timeouts)
# -------------------------------------------------
def _build_loader(dataset, batch_size: int, sampler, num_workers: int, logger: logging.Logger, collate_fn=None) -> DataLoader:
    dl_timeout = 0 if num_workers == 0 else 120
    persistent_ok = (num_workers > 0) and (platform.system() != "Windows")
    kwargs = dict(
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        timeout=dl_timeout,           
    )
    if persistent_ok:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    logger.info(f"[Loader] bs={batch_size}, num_workers={num_workers}, "
                f"persistent={persistent_ok}, timeout={dl_timeout}s")
    return DataLoader(dataset, **kwargs)

def get_dataloaders(cfg, rank, world_size, logger: logging.Logger):
    """
    既存仕様を保ちつつ、OS毎の安全な既定値・詳細ログ・timeout を付与。
    """
    is_windows = (platform.system() == "Windows")
    default_workers = 0 if is_windows else 4
    num_workers = int(getattr(cfg, "num_workers", default_workers))
    image_size = tuple(getattr(cfg, "image_size", (320, 320)))
    mean = float(getattr(cfg, "normalize_mean", 0.5)) if hasattr(cfg, "normalize_mean") else 0.5
    std  = float(getattr(cfg, "normalize_std", 0.5))  if hasattr(cfg, "normalize_std")  else 0.5

    task = str(getattr(cfg, "task", "classification")).lower()
    logger.info(f"[Data] task={task}, image_size={image_size}, mean/std=({mean},{std}), num_workers={num_workers}")

    if task == "segmentation":
        tfm = SegmentationAugment(image_size=image_size, mean=mean, std=std)
        train_ds = SegmentationDataset(
            img_dir=cfg.train_img_dir,
            mask_dir=cfg.train_mask_dir,
            list_csv=getattr(cfg, "train_file_dir", None),
            transform=tfm,
            num_classes=getattr(cfg, "num_classes", None),
        )
        valid_ds = SegmentationDataset(
            img_dir=cfg.valid_img_dir,
            mask_dir=cfg.valid_mask_dir,
            list_csv=getattr(cfg, "valid_file_dir", None),
            transform=tfm,
            num_classes=getattr(cfg, "num_classes", None),
        )

        logger.info(f"[Data] train_ds={len(train_ds)} images, valid_ds={len(valid_ds)} images")

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

        train_loader = _build_loader(train_ds, cfg.batch_size, train_sampler, num_workers, logger)
        valid_loader = _build_loader(valid_ds, cfg.batch_size, valid_sampler, num_workers, logger)
        return train_loader, valid_loader, train_sampler
    
    # ---- classification / multitask / regression ----
    tfm_type = cfg.AUGMENTATION.get("transform", "simple")
    logger.info(f"[Tfm] transformer={tfm_type}")
    if tfm_type == "simple":
        tfm = SimpleTransform(size_hw=image_size, mean=mean, std=std)
    elif tfm_type == "albumentation":
        tfm = AlbumentationTransform(size_hw=image_size)
        
    train_ds = SignalMixClassificationDataset(
        img_dir=cfg.train_img_dir, annotation_csv=cfg.train_file_dir, transform=tfm, is_train=True
    )
    valid_ds = SignalMixClassificationDataset(
        img_dir=cfg.valid_img_dir, annotation_csv=cfg.valid_file_dir, transform=tfm, is_train=False
    )
    logger.info(f"[Data] train_ds={len(train_ds)} images, valid_ds={len(valid_ds)} images")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = _build_loader(train_ds, cfg.batch_size, train_sampler, num_workers, logger,
                                 collate_fn=signalmix_collate)
    valid_loader = _build_loader(valid_ds, cfg.batch_size, valid_sampler, num_workers, logger,
                                 collate_fn=signalmix_collate)

    return train_loader, valid_loader, train_sampler


# -------------------------------------------------
# Train / Validate (AMP + robust logs)
# -------------------------------------------------
def train_one_epoch(
    model, loader, optimizer, scaler, loss_fn, device,
    task: str, mixer, num_classes: int, cfg, logger: logging.Logger,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_mixer = mixer is not None and str(cfg.AUGMENTATION.get("name", "none")).lower() != "none"
    pbar_disable = not _is_main()
    desc = f"Training[R{_rank()}]"

    seg_ignore = int(cfg.LOSS.get("ignore_index", 255))
    seg_w = cfg.LOSS.get("class_weights", None)
    seg_weight = torch.tensor(seg_w, dtype=torch.float32, device=device) if (task=="segmentation" and seg_w) else None

    t_iter = time.time()
    for ib, batch in enumerate(tqdm(loader, desc=desc, disable=pbar_disable)):
        try:
            def to_dev(x): return x.to(device, non_blocking=True) if torch.is_tensor(x) else x

            # --- unpack (タスク別に可変) ---
            if isinstance(batch, (list, tuple)):
                batch = list(batch)
            else:
                batch = [batch]
            images = to_dev(batch[0])

            cls_labels = None
            slope_targets = None
            bboxes = None

            if task == "multitask":
                cls_labels = to_dev(batch[1]); slope_targets = to_dev(batch[2])
            elif task == "classification":
                cls_labels = to_dev(batch[1])
                if len(batch) > 2 and not torch.is_tensor(batch[2]):
                    bboxes = batch[2]
            elif task == "regression":
                slope_targets = to_dev(batch[1])
            else:  # segmentation
                # segmentation の場合、get_dataloaders で (img, mask) にしている想定
                images = to_dev(batch[0]); masks = to_dev(batch[1])

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type):
                if task == "segmentation":
                    logits = model(images)  # [B,C,H,W]
                    if not torch.isfinite(logits).all():
                        logger.warning("NaN/Inf in logits – skip batch"); continue
                    loss = F.cross_entropy(logits, masks, ignore_index=seg_ignore, weight=seg_weight)
                elif task == "multitask":
                    signal_pred, slope_pred = model(images)
                    if (not torch.isfinite(signal_pred).all()) or (not torch.isfinite(slope_pred).all()):
                        logger.warning("NaN/Inf in output – skip batch"); continue
                    loss = loss_fn(signal_pred, cls_labels, slope_pred, slope_targets)
                elif task == "classification":
                    if use_mixer:
                        y_one = F.one_hot(cls_labels, num_classes=num_classes).float()
                        try:
                            if getattr(mixer, "needs_bboxes", False):
                                images, y_soft = mixer(images, y_one, bboxes)
                            else:
                                images, y_soft = mixer(images, y_one)
                        except Exception as e:
                            logger.error(f"Mixer failed: {e}; use hard labels")
                            y_soft = y_one
                        logits = model(images)  # ★ Mix後の画像で再forward
                        if not torch.isfinite(logits).all():
                            logger.warning("NaN/Inf in logits – skip batch"); continue
                        logp = F.log_softmax(logits, dim=1)
                        loss = -(y_soft * logp).sum(dim=1).mean()
                        hard_t = torch.argmax(y_soft, dim=1)
                        correct += (logits.argmax(1) == hard_t).sum().item()
                        total += hard_t.numel()
                    else:
                        logits = model(images)
                        if not torch.isfinite(logits).all():
                            logger.warning("NaN/Inf in logits – skip batch"); continue
                        loss = loss_fn(logits, cls_labels)
                        correct += (logits.argmax(1) == cls_labels).sum().item()
                        total += cls_labels.numel()
                else:  # regression
                    out = model(images)
                    if not torch.isfinite(out).all():
                        logger.warning("NaN/Inf in output – skip batch"); continue
                    loss = loss_fn(out, slope_targets)

            scaler.scale(loss).backward()

            if ib == 0 and _is_main():
                names = (model.module.named_parameters() if hasattr(model, "module") else model.named_parameters())
                unused = [n for n, p in names if p.requires_grad and p.grad is None]
                if unused:
                    logger.warning(f"[GradCheck] Unused params in first iter (n={len(unused)}): {unused[:20]}{' ...' if len(unused)>20 else ''}")
            scaler.step(optimizer); scaler.update()
            total_loss += float(loss.detach().item())

            if ib == 0 and _is_main():
                try:
                    if task in ("classification","multitask","regression"):
                        logger.info(f"[Batch0] images={tuple(images.shape)} dtype={images.dtype}")
                    elif task == "segmentation":
                        logger.info(f"[Batch0] images={tuple(images.shape)} masks={tuple(masks.shape)}")
                except Exception:
                    pass

        except Exception as e:
            logger.exception(f"[Train] Exception at iter {ib}: {e}")
            raise

    acc = (correct / total * 100.0) if (task != "regression" and total > 0) else 0.0
    denom = max(1, len(loader))
    return total_loss / denom, acc

def validate_one_epoch(model, loader, loss_fn, device, task: str, logger: logging.Logger):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true_cls: List[int] = []; y_pred_cls: List[int] = []; y_prob_cls: List[np.ndarray] = []
    y_true_reg: List[float] = []; y_pred_reg: List[float] = []

    pbar_disable = not _is_main()
    desc = f"Validating[R{_rank()}]"

    # segmentation 集計
    seg_inter = None; seg_union = None
    seg_ignore = 255; seg_weight = None

    with torch.no_grad():
        for ib, batch in enumerate(tqdm(loader, desc=desc, disable=pbar_disable)):
            try:
                def to_dev(x): return x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                if isinstance(batch, (list, tuple)): batch = list(batch)
                images = to_dev(batch[0])

                if task == "multitask":
                    cls_labels = to_dev(batch[1]); slope_targets = to_dev(batch[2])
                    signal_pred, slope_pred = model(images)
                    loss = loss_fn(signal_pred, cls_labels, slope_pred, slope_targets)
                    y_true_cls.extend(cls_labels.cpu().tolist())
                    y_pred_cls.extend(signal_pred.argmax(1).cpu().tolist())
                    y_prob_cls.extend(torch.softmax(signal_pred, 1).cpu().numpy())
                    y_true_reg.extend(slope_targets.cpu().tolist())
                    y_pred_reg.extend(torch.tanh(slope_pred).cpu().tolist())
                    correct += (signal_pred.argmax(1) == cls_labels).sum().item()
                    total += cls_labels.numel()

                elif task == "classification":
                    logits = model(images); loss = loss_fn(logits, to_dev(batch[1]))
                    y_true = to_dev(batch[1])
                    y_true_cls.extend(y_true.cpu().tolist())
                    y_pred_cls.extend(logits.argmax(1).cpu().tolist())
                    y_prob_cls.extend(torch.softmax(logits, 1).cpu().numpy())
                    correct += (logits.argmax(1) == y_true).sum().item()
                    total += y_true.numel()

                elif task == "segmentation":
                    images = to_dev(batch[0]); masks = to_dev(batch[1])
                    logits = model(images); loss = F.cross_entropy(logits, masks, ignore_index=seg_ignore, weight=seg_weight)
                    preds = logits.argmax(1); valid = (masks != seg_ignore)
                    correct += (preds[valid] == masks[valid]).sum().item(); total += int(valid.sum().item())
                    # IoU
                    num_classes = logits.shape[1]
                    pnp = preds.cpu().numpy(); mnp = masks.cpu().numpy()
                    for c in range(num_classes):
                        inter = ((pnp == c) & (mnp == c)).sum()
                        union = ((pnp == c) | (mnp == c)).sum()
                        if seg_inter is None:
                            seg_inter = np.zeros(num_classes, np.int64); seg_union = np.zeros(num_classes, np.int64)
                        seg_inter[c] += inter; seg_union[c] += union

                else:  # regression
                    out = model(images); loss = loss_fn(out, to_dev(batch[1]))
                    y_true_reg.extend(to_dev(batch[1]).cpu().tolist())
                    y_pred_reg.extend(torch.tanh(out).cpu().tolist())

                total_loss += float(loss.detach().item())

            except Exception as e:
                logger.exception(f"[Val] Exception at iter {ib}: {e}")
                raise

    acc = (correct / total * 100.0) if (task != "regression" and total > 0) else 0.0
    if task == "classification":
        metrics = {}
        if len(set(y_true_cls)) >= 2:
            metrics = evaluate_classification(y_true_cls, y_pred_cls, np.array(y_prob_cls), num_classes=3)
        return total_loss / max(1, len(loader)), acc, metrics
    if task == "regression":
        metrics = evaluate_regression(y_true_reg, y_pred_reg)
        return total_loss / max(1, len(loader)), 0.0, metrics
    if task == "segmentation":
        pix_acc = acc
        miou = 0.0; per_class_iou = {}
        if seg_inter is not None and seg_union is not None:
            iou = np.divide(seg_inter, np.maximum(1, seg_union), dtype=np.float64)
            valid_cls = [i for i in range(len(iou)) if seg_union[i] > 0]
            miou = float(iou[valid_cls].mean()) if valid_cls else 0.0
            per_class_iou = {f"class_{i}_iou": float(iou[i]) for i in valid_cls}
        metrics = {"pixel_acc": pix_acc, "miou": miou, **per_class_iou}
        return total_loss / max(1, len(loader)), pix_acc, metrics
    # multitask
    cls_metrics = {}
    if len(set(y_true_cls)) >= 2:
        cls_metrics = evaluate_classification(y_true_cls, y_pred_cls, np.array(y_prob_cls), num_classes=3)
    reg_metrics = evaluate_regression(y_true_reg, y_pred_reg)
    return total_loss / max(1, len(loader)), acc, {"classification": cls_metrics, "regression": reg_metrics}


# -------------------------------------------------
# Main worker
# -------------------------------------------------
def main_worker(rank: int, world_size: int):
    # ---- logger ----
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s [R{rank}] [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    try:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
        _env_sanity_log(logger)

        yaml_path = os.environ.get("TRAIN_CONFIG_PATH", "training_setting.yaml")
        if _is_main():
            logger.info(f"[CFG] YAML: {yaml_path}")

        configs = load_all_training_configs(yaml_path)
        if _is_main():
            logger.info(f"[CFG] parsed {len(configs)} model entries")

        for cfg in configs:
            # ---- wandb (rank0 only) ----
            time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{cfg.run_prefix}.{cfg.model_name}.{time_tag}"
            if rank == 0:
                run = wandb.init(
                    project=cfg.wandb_project,
                    name=run_name,
                    config={"task": cfg.task, "model_name": cfg.model_name},
                    reinit=True,
                )
                logger.info("wandb run initialised")

            # ---- model ----
            logger.info(f"[Model] building {cfg.model_name} (task={cfg.task})")
            if str(cfg.task).lower() == "segmentation":
                model = create_segmentation_model(
                    cfg.model_name,
                    num_classes=int(getattr(cfg, "num_classes", 2)),
                    dropout_rate=cfg.dropout_rate,
                    drop_path_rate=cfg.drop_path_rate,
                ).to(device)
            else:
                model = get_model(
                    cfg.task,
                    cfg.model_name,
                    num_classes=int(getattr(cfg, "num_classes", 3)),
                    dropout_rate=cfg.dropout_rate,
                    drop_path_rate=cfg.drop_path_rate,
                    cfg=cfg
                ).to(device)

            model = DDP(
                model,
                device_ids=[rank] if device.type == "cuda" else None,
                find_unused_parameters=False,         # 未使用パラメータ許可＆検知
                gradient_as_bucket_view=True,         # 軽微な最適化
                static_graph=False                    # 動的分岐の可能性に備える
            )
            if _is_main():
                logger.info("[DDP] find_unused_parameters=True / static_graph=False")
            if rank == 0:
                wandb.watch(model.module if hasattr(model, "module") else model)

            # ---- data ----
            logger.info("[Data] building dataloaders …")
            train_loader, valid_loader, train_sampler = get_dataloaders(cfg, rank, world_size, logger)

            # ---- 事前プローブ: 最初の1バッチを rank0 で確認（形・dtype をログ）----
            if _is_main():
                try:
                    logger.info("[Probe] fetching first train batch …")
                    _it = iter(train_loader)
                    b = next(_it)  # DataLoader.timeout により最大120秒で例外
                    if isinstance(b, (list, tuple)):
                        x = b[0]
                    else:
                        x = b
                    if torch.is_tensor(x):
                        logger.info(f"[Probe] OK: batch0 images={tuple(x.shape)} dtype={x.dtype}")
                    else:
                        logger.info(f"[Probe] OK: batch0 type={type(x)}")
                    del _it, b, x
                except Exception as e:
                    logger.exception(f"[Probe] FAILED to get first batch: {e}")
                    raise

            # ---- mixer / optim / loss ----
            mixer = None if str(cfg.task).lower()=="segmentation" else get_mixer(cfg.AUGMENTATION, backbone=model.module if hasattr(model,"module") else model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            scaler = GradScaler('cuda', enabled=(device.type=="cuda"))
            loss_fn = get_loss_fn(cfg.LOSS, task=cfg.task, class_counts=getattr(train_loader.dataset, "class_counts", None))
            if hasattr(loss_fn, "to"): loss_fn = loss_fn.to(device)

            # ---- logging / dirs (rank0) ----
            if _is_main():
                os.makedirs(cfg.save_path, exist_ok=True)
                os.makedirs(cfg.result_path, exist_ok=True)
                writer = SummaryWriter(log_dir=os.path.join(cfg.result_path, "tensorboard", cfg.model_name))

            early_stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
            best_val_acc = -1.0
            result_log: List[List[Any]] = []

            for epoch in range(cfg.epochs):
                train_sampler.set_epoch(epoch)

                tr_loss, tr_acc = train_one_epoch(
                    model, train_loader, optimizer, scaler, loss_fn, device,
                    cfg.task, mixer, getattr(train_loader.dataset, "num_classes", 3), cfg, logger
                )

                if _is_main():
                    va_loss, va_acc, va_metrics = validate_one_epoch(
                        model.module if hasattr(model,"module") else model,
                        valid_loader, loss_fn, device, cfg.task, logger
                    )
                    early_stopper.step(va_loss)
                else:
                    va_loss, va_acc, va_metrics = 0.0, 0.0, {}

                # sync stop flag
                stop_tensor = torch.tensor([1 if (_is_main() and early_stopper.early_stop) else 0],
                                           dtype=torch.int, device=device)
                dist.broadcast(stop_tensor, src=0)

                if _is_main():
                    logger.info(
                        f"[{cfg.model_name}] epoch {epoch+1}/{cfg.epochs} | "
                        f"train {tr_loss:.4f}/{tr_acc:.2f}% | val {va_loss:.4f}/{va_acc:.2f}%"
                    )
                    result_log.append([epoch+1, tr_loss, va_loss, tr_acc, va_acc])

                    log_data = {"epoch": epoch+1, "train/loss": tr_loss, "train/acc": tr_acc,
                                "val/loss": va_loss, "val/acc": va_acc}
                    if str(cfg.task).lower() == "segmentation" and va_metrics:
                        for k in ["pixel_acc","miou"]:
                            if k in va_metrics: log_data[f"val/{k}"] = va_metrics[k]
                    if cfg.task in ["classification","multitask"] and va_metrics:
                        cls = va_metrics["classification"] if cfg.task == "multitask" else va_metrics
                        for k in ["macro_f1","micro_f1","cohen_kappa","mcc"]:
                            if k in cls and cls[k] is not None:
                                log_data[f"val/{k}"] = cls[k]
                    if cfg.task in ["regression","multitask"] and va_metrics:
                        reg = va_metrics["regression"] if cfg.task == "multitask" else va_metrics
                        for k in ["rmse","mae"]:
                            if k in reg and reg[k] is not None:
                                log_data[f"val/{k}"] = reg[k]
                    wandb.log(log_data)

                    if va_acc > best_val_acc:
                        best_val_acc = va_acc
                        ckpt_name = f"{cfg.run_prefix}.{cfg.model_name}.e{epoch:03d}.acc{va_acc:.2f}.pth"
                        best_path = os.path.join(cfg.save_path, ckpt_name)
                        torch.save((model.module if hasattr(model,"module") else model).state_dict(), best_path)
                        artifact = wandb.Artifact(f"{cfg.run_prefix}", type="model")
                        artifact.add_file(best_path)
                        wandb.log_artifact(artifact)
                        wandb.run.summary["best_val_acc"] = best_val_acc
                        os.remove(best_path)

                if stop_tensor.item():
                    if _is_main():
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # ---- rank0: CSV / finish ----
            if _is_main():
                df = pd.DataFrame(result_log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])
                csv_path = os.path.join(cfg.result_path, f"{cfg.model_name}_result.csv")
                df.to_csv(csv_path, index=False)
                wandb.save(csv_path)
                writer.close()
                wandb.finish()

            dist.barrier()
            free_gpu_memory(model)

    except KeyboardInterrupt:
        if _is_main():
            logging.getLogger(__name__).warning("KeyboardInterrupt received. Cleaning up …")
        try:
            wandb.finish()
        except Exception:
            pass
        raise
    except Exception as e:
        logging.getLogger(__name__).exception(f"Fatal error: {e}")
        try:
            wandb.finish()
        except Exception:
            pass
        raise
    finally:
        cleanup_ddp()
        free_gpu_memory()
