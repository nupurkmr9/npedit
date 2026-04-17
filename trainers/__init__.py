"""
Trainers package for npedit.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any, cast

import torch
import torch.distributed as dist
import yaml  # type: ignore

from utils.config import BaseParams, create_component
from utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class BaseTrainerParams(BaseParams):
    """Base parameters for all trainers - contains common training configuration."""

    # Learning rate and optimizer settings
    max_lr: float = 0.0001
    min_lr: float = 0.00001
    warmup_steps: int = 2000
    max_steps: int = 1_000_000
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = field(default_factory=lambda: (0.9, 0.95))

    max_lr_aux: float = 0.0001
    min_lr_aux: float = 0.00001
    weight_decay_aux: float = 0.0
    adam_betas_aux: tuple[float, float] = field(default_factory=lambda: (0.9, 0.95))

    # Gradient accumulation settings
    total_batch_size: int = -1

    # Training settings
    gradient_clip_norm: float = 1.0
    grad_norm_spike_threshold: float = 2.0
    grad_norm_spike_detection_start_step: int = 2000

    # EMA settings
    ema_decay: float = 0.999

    # Checkpointing settings
    init_ckpt: str | None = None
    init_ckpt_load_plan: str | None = (
        "ckpt_model:mem_model,ckpt_ema:mem_ema,ckpt_optimizer:mem_optimizer,ckpt_scheduler:mem_scheduler,ckpt_step:mem_step"
    )
    ckpt_freq: int = 1000
    exp_dir: str = "./experiments/default"

    # Logging
    log_freq: int = 20
    wandb_project: str = "npedit"
    wandb_name: str | None = None
    wandb_entity: str | None = None
    wandb_host: str | None = None
    wandb_mode: str = "online"  # online, offline, or disabled

    # Validation settings
    val_freq: int = 1000
    val_num_samples: int = 10_000

    # Inference settings
    inference_at_start: bool = False
    inference_then_exit: bool = False
    inference_freq: int = 1_000_000_000

    # Image logging
    image_log_freq: int = 100

    # Garbage collection settings
    gc_freq: int = 10000000  # Run GC and sync every N steps to avoid stragglers


def load_config(yaml_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Recursively expand environment variables in all string values
    def _expand_env_vars(value: Any) -> Any:
        """Recursively expand $VAR and ${VAR} in strings within nested structures."""
        if isinstance(value, dict):
            return {k: _expand_env_vars(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_expand_env_vars(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_expand_env_vars(v) for v in value)
        if isinstance(value, str):
            # os.path.expandvars leaves unknown vars unchanged, which is desired
            return os.path.expandvars(value)
        return value

    expanded_config = _expand_env_vars(config)

    return cast(dict[str, Any], expanded_config)


def setup_distributed() -> tuple[torch.device, int, int, int]:
    """Initialize distributed training environment for multi-GPU setup."""
    # Get distributed environment variables
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    # Set CUDA device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    logger.info(f"Initialized distributed training: rank {global_rank}/{world_size}, device {device}")
    return device, local_rank, global_rank, world_size


def run_trainer_from_config(config_path: str, mode: str = "train") -> None:
    """Generic function to run any trainer from config file using the structured pattern."""
    # Load config from file
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Create trainer from config with parameter validation
    logger.info("Creating trainer from config...")
    trainer_config = config["trainer"]
    trainer_component = create_component(trainer_config["module"], trainer_config["params"])

    # Run training
    if mode == "inference":
        trainer_component.inference(config)
    elif mode == "train":
        trainer_component.train(config)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def setup_experiment_dirs(exp_dir: str, config: dict[str, Any]) -> tuple[str, str]:
    """
    Setup experiment directories for training.
    """
    # === Experiment Lifecycle Management ===
    # 1. Create a synchronized timestamp for the run
    global_rank = dist.get_rank()
    local_rank = os.environ["LOCAL_RANK"]
    print(f"global_rank: {global_rank}, local_rank: {local_rank}")
    
    # Use tensor-based broadcasting which is more reliable than object serialization
    if global_rank == 0:
        run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Convert string to fixed-size byte tensor (32 bytes should be enough for timestamp)
        timestamp_bytes = run_timestamp.encode('utf-8')
        timestamp_tensor = torch.zeros(32, dtype=torch.uint8, device=f'cuda:{global_rank}')
        timestamp_tensor[:len(timestamp_bytes)] = torch.frombuffer(timestamp_bytes, dtype=torch.uint8)
        print(f"run_timestamp: {run_timestamp}")
    else:
        timestamp_tensor = torch.zeros(32, dtype=torch.uint8, device=f'cuda:{local_rank}')
    
    # Broadcast the tensor instead of using object serialization (use CPU for simplicity)
    dist.broadcast(timestamp_tensor, src=0)
    
    # Convert tensor back to string on all ranks
    if global_rank != 0:
        # Find the actual length by looking for the first zero byte
        timestamp_length = (timestamp_tensor != 0).sum().item()
        timestamp_bytes = timestamp_tensor[:timestamp_length].cpu().numpy().tobytes()
        run_timestamp = timestamp_bytes.decode('utf-8')
        print(f"received run_timestamp: {run_timestamp}")

    # 2. Define experiment directories
    run_dir = os.path.join(exp_dir, "runs", run_timestamp)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    # 3. Create directories and save config on rank 0
    if global_rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        logger.info(f"Experiment directory: {exp_dir}")
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Checkpoints directory: {ckpt_dir}")

        # Save the complete config for this run
        config_path = os.path.join(run_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        logger.info(f"Saved run config to {config_path}")

    # Wait for all processes to catch up
    dist.barrier()

    return run_dir, ckpt_dir
