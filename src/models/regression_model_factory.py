# src/models/regression_model_factory.py
from __future__ import annotations
import importlib
import torch.nn as nn

def create_regression_model(model_name: str, **kwargs) -> nn.Module:
    """
    models/regression/<model_name>.py を import してモデルを生成。
    モジュールは `build_model(**kwargs)` もしくは `Model(**kwargs)` を提供すること。
    """
    mod = importlib.import_module(f"models.regression.{model_name}")
    if hasattr(mod, "build_model"):
        return mod.build_model(**kwargs)
    if hasattr(mod, "Model"):
        return mod.Model(**kwargs)
    raise AttributeError(
        f"Module {mod.__name__} must expose `build_model` or `Model`."
    )
