import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    cohen_kappa_score, matthews_corrcoef, mean_squared_error,
    mean_absolute_error
)
import torch

def evaluate_classification(y_true, y_pred, y_prob, num_classes):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

    # ROC & PR curve (One-vs-Rest)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    prc = dict()
    pr_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve((np.array(y_true) == i).astype(int), y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        prc[i] = precision_recall_curve((np.array(y_true) == i).astype(int), y_prob[:, i])
        pr_auc[i] = auc(prc[i][1], prc[i][0])

    metrics["roc_auc"] = roc_auc
    metrics["pr_auc"] = pr_auc

    return metrics


def evaluate_regression(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred)
    }
