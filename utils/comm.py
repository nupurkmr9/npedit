"""
Communication utilities for distributed training.

This module provides utilities for distributed communication operations,
particularly for aggregating metrics and losses across multiple GPUs.
"""

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


def dist_reduce(tensors: list[torch.Tensor], reduction: str = "mean") -> float:
    """
    Efficiently compute reduction operations across distributed tensors.

    Takes a list of tensors (typically from different GPUs) and applies the
    specified reduction operation across all elements in all tensors globally.

    Args:
        tensors: List of tensors with the same shape from different GPUs
        reduction: Reduction method. One of "mean", "sum", "var", "std"

    Returns:
        Reduced value across all tensors and all GPUs

    Example:
        # Each GPU has tensor of shape (batch_size,)
        local_losses = torch.tensor([0.5, 0.3, 0.8])  # Local batch losses
        global_mean = dist_reduce([local_losses], "mean")
        global_std = dist_reduce([local_losses], "std")
    """
    assert len(tensors) > 0, "No tensors provided"
    assert reduction in ["mean", "sum", "var", "std"], f"Unsupported reduction: {reduction}"

    # Concatenate all tensors to avoid many small kernel launches
    all_data = torch.cat(tensors, dim=0)
    device = all_data.device

    if reduction == "sum":
        # For sum, just all-reduce the local sum (already a tensor)
        local_sum = all_data.sum().unsqueeze(0)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        return float(local_sum[0])

    elif reduction == "mean":
        # Pack sum and count tensors
        local_sum = all_data.sum()
        local_count = torch.tensor(all_data.numel(), dtype=local_sum.dtype, device=device)
        stats = torch.stack([local_sum, local_count])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        stats_list = stats.tolist()
        return float(stats_list[0]) / float(stats_list[1])

    elif reduction in ["var", "std"]:
        # Pack sum, sum_of_squares, and count tensors
        local_sum = all_data.sum()
        local_sum_sq = (all_data**2).sum()
        local_count = torch.tensor(all_data.numel(), dtype=local_sum.dtype, device=device)
        stats = torch.stack([local_sum, local_sum_sq, local_count])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        # Single GPU-to-CPU sync and unpack
        global_sum, global_sum_sq, global_count = stats.tolist()
        global_mean = float(global_sum) / float(global_count)
        variance = (float(global_sum_sq) / float(global_count)) - (global_mean**2)

        return float(variance) if reduction == "var" else float(variance**0.5)

    # This should never be reached due to assertion above
    raise ValueError(f"Unsupported reduction: {reduction}")


def all_gather_with_padding(tensor: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """
    Gather tensors with potentially different first dimensions across GPUs.
    Pads to max dimension, gathers, then slices back to original shapes.

    Args:
        tensor: Input tensor with shape (N, ...)
        group: Process group for the all_gather operation

    Returns:
        Concatenated tensor containing all original tensors from all GPUs
    """
    # Get all first dimension sizes
    local_size = tensor.shape[0]
    sizes_gathered = all_gather(torch.tensor([local_size], device=tensor.device), group=group)
    # all_gather returns a tuple, convert to tensor and get sizes
    if isinstance(sizes_gathered, tuple | list):
        sizes_gathered = torch.cat(sizes_gathered, dim=0)

    sizes = sizes_gathered.tolist()
    max_size = max(sizes)

    # Pad to max size and gather
    if local_size < max_size:
        pad_shape = (max_size - local_size,) + tuple(tensor.shape[1:])
        tensor = torch.cat([tensor, torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)], dim=0)

    # all_gather returns a tuple of tensors, one from each rank
    gathered_tensors = all_gather(tensor, group=group)

    # Handle both tuple/list and single tensor cases
    if not isinstance(gathered_tensors, tuple | list):
        gathered_tensors = [gathered_tensors]

    # Slice back to original sizes and concatenate
    result = []
    for gathered_tensor, size in zip(gathered_tensors, sizes, strict=True):
        result.append(gathered_tensor[:size])

    return torch.cat(result, dim=0)
