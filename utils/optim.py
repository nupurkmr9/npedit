"""
Optimizer utilities for training.
"""

from torch import nn

from utils.log import get_logger

logger = get_logger(__name__)


def create_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """
    Create parameter groups for optimizer with selective weight decay.

    Weight decay is only applied to parameters with ndim > 1 (e.g., weight matrices).
    Parameters with ndim <= 1 (e.g., biases, layer norm parameters) get no weight decay.

    Args:
        model: The model to create parameter groups for
        weight_decay: Weight decay value to apply to parameters with ndim > 1

    Returns:
        List of parameter group dictionaries suitable for optimizer initialization
    """
    # Separate parameters by dimensionality for selective weight decay
    decay_params = []
    no_decay_params = []

    for param in model.parameters():
        if param.requires_grad:
            if param.ndim > 1:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

    # Count parameters for logging
    decay_param_count = sum(p.numel() for p in decay_params)
    no_decay_param_count = sum(p.numel() for p in no_decay_params)
    total_params = decay_param_count + no_decay_param_count

    logger.info(
        f"Parameter groups: {decay_param_count:,} parameters with weight decay, "
        f"{no_decay_param_count:,} without weight decay "
        f"(total: {total_params:,} parameters)"
    )

    # Create parameter groups with different weight decay settings
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return param_groups
