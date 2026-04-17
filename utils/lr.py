import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR


class LinearWarmupCosineDecayScheduler:
    """
    Learning rate scheduler with linear warmup, cosine decay, and fixed minimum learning rate.

    Uses SequentialLR with three phases:
    1. LinearLR for warmup (if warmup_steps > 0)
    2. CosineAnnealingLR for decay
    3. LambdaLR to maintain min_lr after decay completes
    """

    def __init__(
        self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float = 0.0
    ):
        """
        Initialize the learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of steps for linear warmup (can be 0)
            total_steps: Total number of training steps for warmup + cosine decay
            max_lr: Maximum learning rate
            min_lr: Minimum learning rate at the end of cosine decay (maintained afterwards)
        """
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if total_steps <= warmup_steps:
            raise ValueError("total_steps must be greater than warmup_steps")
        if max_lr < min_lr:
            raise ValueError("max_lr must be greater than or equal to min_lr")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_steps = total_steps - warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

        # Set optimizer's initial lr to max_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = max_lr

        # Create warmup scheduler (no-op if warmup_steps is 0)
        if warmup_steps > 0:
            # Use a very small but non-zero start_factor (PyTorch requires > 0)
            warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        else:
            # No-op warmup: start and end at max_lr for 1 step
            warmup_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=1)

        # Create cosine decay scheduler
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.decay_steps, eta_min=min_lr)

        # Create fixed learning rate scheduler to maintain min_lr after decay completes
        # LambdaLR with constant function that returns min_lr/max_lr ratio
        fixed_lr_factor = min_lr / max_lr
        fixed_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: fixed_lr_factor)

        # Create SequentialLR with three phases
        warmup_milestone = max(warmup_steps, 1)  # Use 1 if warmup_steps is 0
        decay_milestone = total_steps

        self.scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler, fixed_scheduler],
            milestones=[warmup_milestone, decay_milestone],
        )

    def step(self) -> None:
        """Update learning rate for the next step."""
        self.scheduler.step()

    def get_last_lr(self) -> list:
        """Get the last computed learning rate."""
        return self.scheduler.get_last_lr()

    def state_dict(self) -> dict:
        """Return scheduler state."""
        return dict(self.scheduler.state_dict())

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state."""
        self.scheduler.load_state_dict(state_dict)
