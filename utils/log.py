"""
Shared logging utilities for npedit.
"""

from contextlib import contextmanager
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
from tqdm import tqdm
import wandb


def human_readable_number(n: int | float) -> str:
    """Pretty-format *n* using T/B/M/K suffixes."""
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return f"{n:,}"


def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with distributed training-aware formatting.

    This function configures a logger on a per-instance basis without
    affecting the root logger, making it safe for use in environments
    like pytest where root logging may already be configured.

    It will clear any existing handlers on the logger to ensure the
    configuration is applied correctly every time.

    Args:
        name: Logger name (defaults to caller's module name if None)
        level: Logging level (defaults to INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Clear any existing handlers to ensure we are starting fresh.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Check if we're in distributed training environment
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Create a handler to stream logs to the console
    handler = logging.StreamHandler()

    # Create a formatter with rank information
    if world_size > 1:
        format_str = f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)

    # Add the configured handler to this specific logger
    logger.addHandler(handler)
    logger.setLevel(level)

    # Do not propagate messages to the root logger to avoid duplication
    # in environments like pytest.
    logger.propagate = False

    return logger


def get_pbar(max_steps: int, start_step: int) -> tqdm | None:
    pbar = None
    if dist.get_rank() == 0:
        pbar = tqdm(
            total=max_steps,
            initial=start_step,
            desc="Training",
            unit="step",
            file=sys.stderr,
            dynamic_ncols=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
    return pbar


class WandbLogger:
    """Simple wandb logger that only logs from rank 0."""

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict | None = None,
        entity: str | None = None,
        host: str | None = None,
        save_dir: str | None = None,
        mode: str = "online",
    ):
        # Check if we're in distributed training and get rank
        if dist.is_initialized():
            self.is_rank_zero = dist.get_rank() == 0
        else:
            self.is_rank_zero = True  # Single GPU case

        self.enabled = mode != "disabled"

        if self.enabled and self.is_rank_zero:
            # Force login with the API key from environment to override cached credentials
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            wandb.login(key=wandb_api_key, relogin=True, host=host)

            wandb.init(project=project, entity=entity, name=name, config=config, mode=mode, dir=save_dir)

            # Log and save wandb URL
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "wandb_url.log"), "w") as f:
                    f.write(wandb.run.get_url())

    def log(self, metrics: dict[str, float | int | torch.Tensor], step: int) -> None:
        """Log metrics to wandb. Only rank 0 actually logs."""
        if not self.enabled or not self.is_rank_zero:
            return

        # Convert tensor values to scalars
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = value.item()
            else:
                processed_metrics[key] = value

        wandb.log(processed_metrics, step=step)

    def __del__(self) -> None:
        """Close the wandb run."""
        if self.enabled and self.is_rank_zero and wandb.run is not None:
            wandb.finish()


class TrackingLogger:
    def __init__(self):
        self.buffers_gpu = {}
        self.buffers_cpu = {}
        self.stats = {}

        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1

    @contextmanager
    def log_time(self, tag: str, cuda_sync: bool = False):
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        try:
            yield
        finally:
            if cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            duration = time.time() - start
            self.log({f"{tag}": duration})

    def time_fn(self, tag: str, cuda_sync: bool = False):
        def decorator(fn):
            def wrapped(*args, **kwargs):
                if cuda_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                try:
                    return fn(*args, **kwargs)
                finally:
                    if cuda_sync and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    duration = time.time() - start
                    self.log({f"{tag}": duration})

            return wrapped

        return decorator

    def log(self, metrics: dict[str, torch.Tensor | float | int]):
        assert isinstance(metrics, dict), "`log()` requires a dictionary"

        # # we always insert current timestamp as a metric
        # metrics["_timestamp"] = time.time()

        for name, value in metrics.items():
            if torch.is_tensor(value):
                value = value.detach().float()
                if value.ndim == 0:
                    value = value.view(1)
                elif value.ndim != 1:
                    raise ValueError(f"Only 1D or scalar tensors are supported. Got shape: {value.shape}")

                if value.device.type == "cuda":
                    self.buffers_gpu.setdefault(name, []).append(value)
                else:
                    self.buffers_cpu.setdefault(name, []).append(value)
            else:
                tensor = torch.tensor([value], dtype=torch.float32, device="cpu")
                self.buffers_cpu.setdefault(name, []).append(tensor)

    def flush(self):
        key_order = []
        local_stats = []

        for name, values in self.buffers_gpu.items():
            if values:
                x = torch.cat(values, dim=0)
                key_order.append(name)
                local_stats.append(self._local_stats_vector(x))
        self.buffers_gpu = {}

        for name, values in self.buffers_cpu.items():
            if values:
                x = torch.cat(values, dim=0)
                key_order.append(name)
                local_stats.append(self._local_stats_vector(x))
        self.buffers_cpu = {}

        if not key_order:
            return

        local_tensor = torch.stack(local_stats, dim=0).cuda()  # [num_keys, num_stats]
        num_keys, num_stats = local_tensor.shape
        flat_local = local_tensor.ravel()

        if self.distributed:
            flat_gathered = torch.empty(self.world_size * flat_local.numel(), dtype=torch.float32, device="cuda")
            dist.all_gather_into_tensor(flat_gathered, flat_local)
            gathered_tensor = flat_gathered.view(self.world_size, num_keys, num_stats)
        else:
            gathered_tensor = flat_local.view(1, num_keys, num_stats)

        gathered_tensor = gathered_tensor.cpu()

        for i, name in enumerate(key_order):
            rows = gathered_tensor[:, i, :]  # [world_size, num_stats]
            count = rows[:, 0].sum()
            sum_ = rows[:, 1].sum()
            sum_sq = rows[:, 2].sum()
            min_ = rows[:, 3].min()
            max_ = rows[:, 4].max()
            mean = sum_ / count
            var = (sum_sq / count) - mean**2
            std = var.sqrt() if count > 1 else torch.tensor(0.0)

            self.stats[name] = {
                "count": int(count),
                "sum": sum_.item(),
                "mean": mean.item(),
                "std": std.item(),
                "min": min_.item(),
                "max": max_.item(),
            }

    def _local_stats_vector(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().cpu()
        return torch.tensor(
            [
                float(x.numel()),  # count
                x.sum().item(),  # sum
                (x**2).sum().item(),  # sum_sq
                x.min().item(),  # min
                x.max().item(),  # max
            ],
            dtype=torch.float32,
        )

    def __getitem__(self, key: tuple[str, str]):
        assert isinstance(key, tuple) and len(key) == 2, "Use logger[metric_name, stat_name]"
        metric_name, stat_name = key
        return self.stats.get(metric_name, {}).get(stat_name, None)

    def get_stats(self, metric_name: str):
        return self.stats.get(metric_name, {})
