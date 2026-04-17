import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
import torch.nn as nn


@torch.no_grad()
def clip_grad(model: nn.Module, max_norm: float) -> torch.Tensor:
    """
    Clip gradients of model.
    """
    shard_size, replicate_size = 1, dist.get_world_size()
    if isinstance(model, FSDPModule):
        shard_size = model._get_fsdp_state()._fsdp_param_group.mesh_info.shard_mesh_size
        replicate_size = dist.get_world_size() // shard_size

    # Separate DTensor and non-DTensor parameters
    all_param_grads = []
    dtensor_param_grads = []
    regular_param_grads = []

    for p in model.parameters():
        if (not p.requires_grad) or (p.grad is None):
            continue

        if isinstance(p.grad, DTensor):
            local_p_grad = p.grad._local_tensor
            dtensor_param_grads.append(local_p_grad.ravel())
        else:
            local_p_grad = p.grad
            regular_param_grads.append(local_p_grad.ravel())

        all_param_grads.append(local_p_grad)

    # Compute local square sum for each group separately
    local_sq_sum = torch.tensor(0.0, device=all_param_grads[0].device)

    if dtensor_param_grads:
        dtensor_sq_sum = (torch.cat(dtensor_param_grads, dim=0) ** 2).float().sum()
        local_sq_sum = local_sq_sum + dtensor_sq_sum

    if regular_param_grads:
        regular_sq_sum = (torch.cat(regular_param_grads, dim=0) ** 2).float().sum()
        local_sq_sum = local_sq_sum + regular_sq_sum / shard_size

    # Single all-reduce operation
    global_sq_sum = local_sq_sum.clone()
    dist.all_reduce(global_sq_sum, op=dist.ReduceOp.SUM)
    global_sq_sum = global_sq_sum / replicate_size

    total_norm = global_sq_sum.sqrt()

    # Only apply clipping when exceeding threshold
    if total_norm > max_norm:
        torch._foreach_mul_(all_param_grads, max_norm / total_norm)

    return total_norm
