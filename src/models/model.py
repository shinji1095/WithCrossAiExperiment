import torch
import torch.nn.functional as F
from models.classification_model_factory import ClassificationModel
from models.regression_model_factory import RegressionModel
from models.multitask_model_factory import MultitaskModel

def get_model(task: str, backbone_name: str, num_classes: int = 3,
              dropout_rate: float = 0.0, drop_path_rate: float = 0.0):
    if task == "classification":
        return ClassificationModel(backbone_name, num_classes, dropout_rate, drop_path_rate)
    elif task == "regression":
        return RegressionModel(backbone_name, dropout_rate, drop_path_rate)
    elif task == "multitask":
        return MultitaskModel(backbone_name, num_classes, dropout_rate, drop_path_rate)
    else:
        raise ValueError(f"Unsupported task: {task}")

def apply_postprocessing(task, output):
    if task == "classification":
        return F.softmax(output, dim=1)
    elif task == "regression":
        return torch.tanh(output)
    elif task == "multitask":
        logits, slope = output
        return F.softmax(logits, dim=1), torch.tanh(slope)
    else:
        raise ValueError(f"Unsupported task: {task}")
