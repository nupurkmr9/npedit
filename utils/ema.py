"""
Exponential Moving Average (EMA) utilities for PyTorch models.

This module provides utilities for maintaining and updating EMA models,
which are commonly used to improve model stability and generalization
in training deep neural networks. It supports both regular tensors and
DTensors (from FSDP-wrapped models).
"""

import torch
from torch.distributed._tensor import DTensor
import torch.nn as nn


@torch.no_grad()
def copy_params(model: nn.Module, ema: nn.Module, model_to_ema: bool = True) -> None:
    """
    Copy parameters between main model and EMA model.

    Args:
        model: Main model
        ema: EMA model
        model_to_ema: Direction of copy. If True, copy model -> ema.
                     If False, copy ema -> model.

    Examples:
        >>> # Initialize EMA with model parameters
        >>> copy_params(model, ema, model_to_ema=True)

        >>> # Replace model with EMA parameters for inference
        >>> copy_params(model, ema, model_to_ema=False)
    """
    if model_to_ema:
        src_model, dst_model = model, ema
    else:
        src_model, dst_model = ema, model

    # Extract parameters and convert DTensors to local tensors
    src_params = [p._local_tensor if isinstance(p, DTensor) else p for p in src_model.parameters()]
    dst_params = [p._local_tensor if isinstance(p, DTensor) else p for p in dst_model.parameters()]

    # Ensure parameter counts match
    if len(src_params) != len(dst_params):
        raise ValueError(f"Parameter count mismatch: src has {len(src_params)}, dst has {len(dst_params)}")

    # Use efficient multi-tensor copy
    torch._foreach_copy_(dst_params, src_params)


@torch.no_grad()
def update_ema(model: nn.Module, ema: nn.Module, decay: float = 0.999) -> None:
    """
    Update EMA model parameters using current model parameters.

    This performs: ema_param = decay * ema_param + (1 - decay) * model_param
    for all parameters in the models.

    Args:
        model: Main model (source of current parameters)
        ema: EMA model to update (target)
        decay: EMA decay factor (typically 0.999)
    """
    # Extract parameters and convert DTensors to local tensors
    model_params = [p._local_tensor if isinstance(p, DTensor) else p for p in model.parameters()]
    ema_params = [p._local_tensor if isinstance(p, DTensor) else p for p in ema.parameters()]

    # Ensure parameter counts match
    if len(model_params) != len(ema_params):
        raise ValueError(f"Parameter count mismatch: model has {len(model_params)}, ema has {len(ema_params)}")

    # Use efficient multi-tensor lerp operation
    # lerp performs: out = input + alpha * (other - input)
    # Which is equivalent to: out = (1 - alpha) * input + alpha * other
    # So we use alpha = (1 - decay) to get: out = decay * ema + (1 - decay) * model
    torch._foreach_lerp_(ema_params, model_params, 1 - decay)
