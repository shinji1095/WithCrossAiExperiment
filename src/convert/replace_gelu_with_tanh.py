import math
import torch
import torch.nn as nn

class TanhGELU(nn.Module):
    """GELU tanh approximation:
       0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
       Fallback for environments that don't support nn.GELU(approximate="tanh").
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x * x * x))
        ))


def _make_tanh_gelu_module() -> nn.Module:
    """Return the fastest available tanh-approx GELU module for this runtime."""
    try:
        # PyTorch 1.12+ supports approximate="tanh"
        return nn.GELU(approximate="tanh")
    except TypeError:
        # Older PyTorch fallback
        return TanhGELU()


def _is_timm_gelu(m: nn.Module) -> bool:
    """Detect timm.layers.activations.GELU without importing internals hard."""
    cls = m.__class__
    return cls.__name__ == "GELU" and getattr(cls, "__module__", "").startswith("timm.")


def replace_gelu_with_tanh(module: nn.Module) -> None:
    """Recursively replace all GELU variants with tanh-approximation versions.

    - Replaces torch.nn.GELU (any variant) with nn.GELU(approximate="tanh") or TanhGELU fallback.
    - Replaces timm.layers.activations.GELU with the same.
    - Leaves other activations (SiLU, ReLU, etc.) intact.
    """
    for name, child in list(module.named_children()):
        replace = False

        # torch.nn.GELU
        if isinstance(child, nn.GELU):
            replace = True

        # timm's own GELU class
        elif _is_timm_gelu(child):
            replace = True

        if replace:
            setattr(module, name, _make_tanh_gelu_module())
        else:
            replace_gelu_with_tanh(child)  # recurse