from __future__ import annotations
import importlib
from typing import Any, Optional

_TASK_DIR = {
    "classification": "models.classification",
    "regression"    : "models.regression",
    "segmentation"  : "models.segmentation"
}

class ModelNotFoundError(ImportError):
    pass

def _import_task_module(task: str, model_name: str):
    """models/<task>/<model_name>.py を動的 import"""
    if task not in _TASK_DIR:
        raise ValueError(f"Unsupported task: {task}")
    pkg = _TASK_DIR[task]
    try:
        return importlib.import_module(f"{pkg}.{model_name}")
    except ImportError as e:
        raise ModelNotFoundError(
            f"Cannot import model module: {pkg}.{model_name}"
        ) from e

def get_model(
    task: str,
    model_name: str,
    **kwargs: Any,
):
    """
    指定タスク/モデル名でモデルを組み立てて返す。
    - 各モジュールは `build_model(**kwargs)` か `Model(**kwargs)` を公開していること。
    例:
      models/classification/edgenext.py:
        - DEFAULT_IMAGE_SIZE = 320 (任意)
        - def build_model(num_classes: int, dropout_rate: float=0.0, drop_path_rate: float=0.0): ...
          または
        - class Model(nn.Module): ...
    """
    mod = _import_task_module(task, model_name)
    if hasattr(mod, "build_model"):
        return mod.build_model(**kwargs)
    if hasattr(mod, "Model"):
        return mod.Model(**kwargs)
    raise AttributeError(
        f"Module {mod.__name__} must expose `build_model` or `Model`."
    )

def get_default_image_size(task: str, model_name: str, fallback: int = 320) -> int:
    """
    モジュールが `DEFAULT_IMAGE_SIZE` を持っていればそれを返す。無ければ fallback。
    """
    mod = _import_task_module(task, model_name)
    return int(getattr(mod, "DEFAULT_IMAGE_SIZE", fallback))