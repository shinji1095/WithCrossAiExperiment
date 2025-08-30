import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
import os


def plot_roc_curve(roc_auc_dict, fpr_dict, tpr_dict, save_path=None):
    plt.figure(figsize=(8, 6))
    for i in roc_auc_dict.keys():
        fpr = fpr_dict[i]
        tpr = tpr_dict[i]
        auc_val = roc_auc_dict[i]
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_val:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()



def plot_pr_curve(pr_auc_dict, prc_dict, save_path=None):
    """
    Plot Precision-Recall Curve (one-vs-rest)
    """
    plt.figure(figsize=(8, 6))
    for i, (prec, recall) in prc_dict.items():
        auc_val = pr_auc_dict[i]
        plt.plot(recall, prec, label=f'Class {i} (AUC = {auc_val:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (One-vs-Rest)")
    plt.legend(loc="upper right")
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names=None, save_path=None, normalize=False):
    """
    Confusion matrix visualization
    """
    plt.figure(figsize=(6, 5))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", square=True, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def write_tensorboard(writer, epoch, tr_loss, va_loss, tr_acc, va_acc, metrics, task):
    writer.add_scalar("Loss/Train", tr_loss, epoch)
    writer.add_scalar("Loss/Valid", va_loss, epoch)
    writer.add_scalar("Accuracy/Train", tr_acc, epoch)
    writer.add_scalar("Accuracy/Valid", va_acc, epoch)

    if task in ["classification", "multitask"]:
        cls = metrics["classification"] if task == "multitask" else metrics
        # ---- 有無を確認してから書き込む（1 クラスしか無い場合など） ----
        for tag, key in [
            ("F1/Macro",  "macro_f1"),
            ("F1/Micro",  "micro_f1"),
            ("Cohen_Kappa", "cohen_kappa"),
            ("MCC",       "mcc"),
            # ("FPS",     "fps"),   # FPS は未実装ならコメントのまま
        ]:
            if key in cls and cls[key] is not None:
                writer.add_scalar(tag, cls[key], epoch)
        # writer.add_scalar("FPS", cls["fps"], epoch)

    if task in ["regression", "multitask"]:
        reg = metrics["regression"] if task == "multitask" else metrics
        writer.add_scalar("RMSE", reg["rmse"], epoch)
        writer.add_scalar("MAE", reg["mae"], epoch)
