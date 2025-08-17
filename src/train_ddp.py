import os
import sys
import time
import json
import random
import logging
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
from dataloader import AlbumentationTransform
from metrics.metrics import evaluate_classification, evaluate_regression
from mixer.advanced_mixers import get_mixer
from utils.visualize import write_tensorboard
from utils.early_stopping import EarlyStopping

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

def get_dataloaders(cfg, rank, world_size):
    transform = AlbumentationTransform(image_size=cfg.image_size)

    train_dataset = SignalSlopeDataset(
        csv_path=cfg.train_file_dir,
        image_dir=cfg.train_img_dir,
        task=cfg.task,
        transform=transform,
        state_filter=cfg.state_filter,
        shuffle=True,
    )
    valid_dataset = SignalSlopeDataset(
        csv_path=cfg.valid_file_dir,
        image_dir=cfg.valid_img_dir,
        task=cfg.task,
        transform=transform,
        state_filter=cfg.state_filter,
        shuffle=False,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=0,
        persistent_workers=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        sampler=valid_sampler,
        num_workers=0,
        persistent_workers=False,
    )

    return train_loader, valid_loader, train_sampler


# -------------------------------------------------
# Training helpers (AMP)
# -------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    loss_fn,
    device,
    task,
    mixer,
    num_classes: int,
    cfg,
    logger,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    use_mixer = mixer is not None and cfg.AUGMENTATION["name"] != "none"

    for batch in tqdm(loader, desc="Training", disable=dist.get_rank() != 0):
        batch = [b.to(device) for b in batch]
        images = batch[0]

        # ----- Advanced‑Mixers -----
        if use_mixer and task == "classification":
            y_one = F.one_hot(batch[1], num_classes=num_classes).float()
            images, y_soft = mixer(images, y_one)
            target_cls_soft = y_soft
        else:
            target_cls_soft = None

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type):
            output = model(images)

            if task == "multitask":
                signal_pred, slope_pred = output
                if not torch.isfinite(signal_pred).all() or not torch.isfinite(slope_pred).all():
                    logger.warning("NaN/Inf detected in model output – skipping batch")
                    continue
                loss = loss_fn(signal_pred, batch[1], slope_pred, batch[2])
                if not torch.isfinite(loss):
                    logger.warning("NaN/Inf detected in loss – skipping batch")
                    continue
                _, predicted = torch.max(signal_pred, 1)
                correct += (predicted == batch[1]).sum().item()
                total += batch[1].size(0)

            elif task == "classification":
                if not torch.isfinite(output).all():
                    logger.warning("NaN/Inf detected in output – skipping batch")
                    continue

                if target_cls_soft is not None:
                    logp = F.log_softmax(output, dim=1)
                    loss = -(target_cls_soft * logp).sum(dim=1).mean()
                else:
                    loss = loss_fn(output, batch[1])

                _, predicted = torch.max(output, 1)
                correct += (predicted == batch[1]).sum().item()
                total += batch[1].size(0)

            else:  # regression
                if not torch.isfinite(output).all():
                    logger.warning("NaN/Inf detected in regression output – skipping batch")
                    continue
                loss = loss_fn(output, batch[1])
                if not torch.isfinite(loss):
                    logger.warning("NaN/Inf detected in loss – skipping batch")
                    continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    acc = (correct / total * 100.0) if task != "regression" else 0.0
    return total_loss / len(loader), acc


def validate_one_epoch(model, loader, loss_fn, device, task, logger):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true_cls, y_pred_cls, y_prob_cls = [], [], []
    y_true_reg, y_pred_reg = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", disable=dist.get_rank() != 0):
            batch = [b.to(device) for b in batch]
            images = batch[0]

            with torch.autocast(device_type=device.type):
                output = model(images)

                if task == "multitask":
                    signal_pred, slope_pred = output
                    if not torch.isfinite(signal_pred).all() or not torch.isfinite(slope_pred).all():
                        logger.warning("NaN/Inf detected in output – skipping sample")
                        continue
                    loss = loss_fn(signal_pred, batch[1], slope_pred, batch[2])
                    if not torch.isfinite(loss):
                        logger.warning("NaN/Inf detected in loss – skipping sample")
                        continue
                    y_true_cls.extend(batch[1].cpu().tolist())
                    y_pred_cls.extend(signal_pred.argmax(dim=1).cpu().tolist())
                    y_prob_cls.extend(torch.softmax(signal_pred, dim=1).cpu().numpy())
                    y_true_reg.extend(batch[2].cpu().tolist())
                    y_pred_reg.extend(torch.tanh(slope_pred).cpu().tolist())

                elif task == "classification":
                    if not torch.isfinite(output).all():
                        logger.warning("NaN/Inf detected in output – skipping sample")
                        continue
                    loss = loss_fn(output, batch[1])
                    if not torch.isfinite(loss):
                        logger.warning("NaN/Inf detected in loss – skipping sample")
                        continue
                    y_true_cls.extend(batch[1].cpu().tolist())
                    y_pred_cls.extend(output.argmax(dim=1).cpu().tolist())
                    y_prob_cls.extend(torch.softmax(output, dim=1).cpu().numpy())

                else:  # regression
                    if not torch.isfinite(output).all():
                        logger.warning("NaN/Inf detected in output – skipping sample")
                        continue
                    loss = loss_fn(output, batch[1])
                    if not torch.isfinite(loss):
                        logger.warning("NaN/Inf detected in loss – skipping sample")
                        continue
                    y_true_reg.extend(batch[1].cpu().tolist())
                    y_pred_reg.extend(torch.tanh(output).cpu().tolist())

            total_loss += loss.item()

    acc = accuracy_score(y_true_cls, y_pred_cls) * 100 if task != "regression" else 0.0

    # build metrics dict
    metrics: Dict[str, Any] = {}
    if task == "classification":
        if len(set(y_true_cls)) >= 2:
            metrics = evaluate_classification(y_true_cls, y_pred_cls, np.array(y_prob_cls), num_classes=3)
    elif task == "regression":
        metrics = evaluate_regression(y_true_reg, y_pred_reg)
    else:  # multitask
        if len(set(y_true_cls)) >= 2:
            metrics = {
                "classification": evaluate_classification(
                    y_true_cls, y_pred_cls, np.array(y_prob_cls), num_classes=3
                ),
                "regression": evaluate_regression(y_true_reg, y_pred_reg),
            }
    return total_loss / len(loader), acc, metrics


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
        if rank == 0:
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "withcross-training"),
                name=f"{cfg.model_name}_{time.strftime('%Y%m%d_%H%M%S')}",
                config=cfg.__dict__,  # save all hyper‑params
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

        # ---------- logging ----------
        if rank == 0:
            os.makedirs(cfg.save_path, exist_ok=True)
            os.makedirs(cfg.result_path, exist_ok=True)
            writer = SummaryWriter(log_dir=os.path.join(cfg.result_path, "tensorboard", cfg.model_name))

        # ---------- early stopping ----------
        early_stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

        best_val_loss = float("inf")
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
                if va_loss < best_val_loss:
                    best_val_loss = va_loss
                    best_path = os.path.join(cfg.save_path, f"{cfg.model_name}_best.pth")
                    torch.save(model.module.state_dict(), best_path)
                    # save to wandb artifact
                    artifact = wandb.Artifact(f"{cfg.model_name}_best", type="model")
                    artifact.add_file(best_path)
                    wandb.log_artifact(artifact)
                    wandb.run.summary["best_val_loss"] = best_val_loss

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
