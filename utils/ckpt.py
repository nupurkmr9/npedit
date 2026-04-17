"""
FSDP2 Distributed Checkpointing Utilities

This module provides a comprehensive checkpointing system for distributed training
with FSDP2-wrapped models, EMA models, optimizers, and learning rate schedulers.
"""

from pathlib import Path
import time
from typing import Any, cast

import torch
import os
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner, DefaultLoadPlanner
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
import torch.nn as nn

from utils.log import get_logger

logger = get_logger(__name__)


def load_checkpoint_for_inference(checkpoint_path: str | Path, model: nn.Module, model_key: str = "model") -> None:
    """
    Load a checkpoint for inference.
    """
    assert model_key in ["model", "ema"], f"model_key must be either 'model' or 'ema', got {model_key}"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_state_dict = get_model_state_dict(
        model,
        options=StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
        ),
    )
    dist_state_dict = {model_key: model_state_dict}

    dcp.load(dist_state_dict, checkpoint_id=str(checkpoint_path))

    set_model_state_dict(
        model,
        cast(dict[str, Any], dist_state_dict[model_key]),
        options=StateDictOptions(full_state_dict=False, cpu_offload=True, strict=True),
    )


class FSDPCheckpointer:
    """
    Distributed checkpointer for FSDP-wrapped models with EMA, optimizer, and scheduler support.

    This checkpointer handles:
    - FSDP-wrapped main model using distributed checkpointing
    - FSDP-wrapped EMA model (also a nn.Module)
    - Optimizer state with proper FSDP handling
    - Learning rate scheduler state
    - Training metadata (step, loss, etc.)

    The checkpointer uses PyTorch's distributed checkpoint (DCP) API for efficient
    saving and loading across multiple processes.
    """

    def __init__(self, checkpoint_dir: str | Path):
        """
        Initialize the FSDP2 checkpointer.

        Args:
            checkpoint_dir: Base directory for saving checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Verify distributed environment
        if not dist.is_initialized():
            raise RuntimeError("Distributed environment not initialized. Call dist.init_process_group() first.")

        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = dist.get_world_size()

        logger.info(f"Initialized FSDPCheckpointer with world size {self.world_size}")

        self.async_futures: dict[str, torch.futures.Future] = {}
        self.async_pg: dist.ProcessGroup | None = None

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        ema: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        metadata: dict[str, Any] | None = None,
        async_save: bool = True,
    ) -> Path:
        """
        Save a distributed checkpoint with all training components.

        This method uses a hybrid approach:
        - Model, EMA, and optimizer use FSDP2 distributed checkpointing
        - Scheduler and metadata are saved by rank 0 to shared.pth

        Args:
            step: Current training step
            model: FSDP2-wrapped main model to save
            ema: FSDP2-wrapped EMA model to save
            optimizer: Optimizer to save (should be created with main model parameters)
            scheduler: Learning rate scheduler (optional)
            metadata: Additional metadata to save (loss, config, etc.)

        Returns:
            Path to the saved checkpoint directory
        """
        start_time = time.time()

        checkpoint_path = self.checkpoint_dir / f"step_{step:08d}"

        # Create checkpoint directory
        if self.rank == 0:
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Wait for directory creation
        dist.barrier()

        # Prepare state dictionaries with proper FSDP2 handling
        logger.info("Preparing model state dict")

        dist_state_dict: dict[str, Any] = {}
        shared_state_dict: dict[str, Any] = {}

        # Main model state dict with FSDP2 handling
        model_state_dict = get_model_state_dict(
            model,
            options=StateDictOptions(
                full_state_dict=False,  # Keep sharded for efficient distributed saving
                cpu_offload=True,  # Offload to CPU to save GPU memory
            ),
        )
        dist_state_dict["model"] = model_state_dict

        # EMA state dict with FSDP2 handling
        if ema is not None:
            logger.info("Preparing ema state dict")
            ema_state_dict = get_model_state_dict(
                ema,
                options=StateDictOptions(
                    full_state_dict=False,
                    cpu_offload=True,
                ),
            )
            dist_state_dict["ema"] = ema_state_dict

        # Optimizer state dict with FSDP2 handling
        logger.info("Preparing optimizer state dict")
        optim_state_dict = get_optimizer_state_dict(
            model,
            optimizer,
            options=StateDictOptions(
                full_state_dict=False,
                cpu_offload=True,
            ),
        )
        dist_state_dict["optimizer"] = optim_state_dict

        # Scheduler state (only rank 0 saves to shared.pth)
        shared_state_dict["scheduler"] = scheduler.state_dict()

        # Metadata (only rank 0 saves to shared.pth)
        shared_state_dict["metadata"] = {
            "step": step,
            "world_size": self.world_size,
            "pytorch_version": torch.__version__,
            "rng_states": self._gather_rng_states(),
        }
        if metadata:
            shared_state_dict["metadata"].update(metadata)

        # Save using distributed checkpoint
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        if async_save:
            if self.async_pg is None:
                self.async_pg = dist.new_group(backend="gloo")

            future = dcp.async_save(
                dist_state_dict,
                checkpoint_id=str(checkpoint_path),
                process_group=self.async_pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                planner=DefaultSavePlanner(enable_plan_caching=True),
            )
            self.async_futures[str(checkpoint_path)] = future
            # Print out how many futures have been done
            total_futures = len(self.async_futures)
            done_futures = sum(1 for future in self.async_futures.values() if future.done())
            logger.info(
                f"Checkpointer async_save futures - total: {total_futures}, done: {done_futures}, "
                f"remaining: {total_futures - done_futures}"
            )
        else:
            dcp.save(dist_state_dict, checkpoint_id=str(checkpoint_path))

        # Additionally save shared data to shared.pth (rank 0 only)
        # This file can contain any arbitrary data that needs to be shared across all ranks
        if self.rank == 0:
            torch.save(shared_state_dict, checkpoint_path / "shared.pth")
            logger.info("Saved shared data (scheduler and metadata) to shared.pth")

        elapsed = time.time() - start_time
        logger.info(f"Successfully saved checkpoint to {checkpoint_path} in {elapsed:.3f}s")

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        ema: nn.Module,
        optimizer: torch.optim.Optimizer=None,
        scheduler: torch.optim.lr_scheduler._LRScheduler=None,
        loading_plan: str | None = None,
    ) -> dict[str, Any]:
        """
        Load a distributed checkpoint and restore all training components.

        This method uses a hybrid approach:
        - Model, EMA, and optimizer use FSDP2 distributed checkpointing
        - Scheduler and metadata are loaded by all ranks from shared.pth

        Args:
            checkpoint_path: Path to the checkpoint directory
            model: FSDP2-wrapped main model to load into
            ema: FSDP2-wrapped EMA model to load into
            optimizer: Optimizer to load into
            scheduler: Learning rate scheduler to load into (optional)
            loading_plan: Selective loading plan string specifying what to load where.
                         Format: 'ckpt_source:mem_target, ckpt_source:mem_target, ...'
                         Example: 'ckpt_model:mem_model, ckpt_ema:mem_ema, ckpt_optimizer:mem_optimizer'
                         If None, loads all components to their default targets.

        Returns:
            Dictionary containing loaded metadata (step, loss, etc.)
        """
        start_time = time.time()

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Parse loading plan
        plan_mapping = self._parse_loading_plan(loading_plan)
        logger.info(f"Loading plan: {plan_mapping}")

        # Determine which checkpoint components need to be loaded
        ckpt_components_needed = set(plan_mapping.keys())

        # Prepare state dicts for distributed and shared components
        dist_state_dict: dict[str, Any] = {}
        shared_state_dict: dict[str, Any] = {}

        print("ckpt_components_needed", ckpt_components_needed, checkpoint_path)

        if "ckpt_model" in ckpt_components_needed:
            model_state_dict = get_model_state_dict(
                model,
                options=StateDictOptions(
                    full_state_dict=False,
                    cpu_offload=True,
                ),
            )
            dist_state_dict["model"] = model_state_dict

        if "ckpt_ema" in ckpt_components_needed:
            ema_state_dict = get_model_state_dict(
                ema,
                options=StateDictOptions(
                    full_state_dict=False,
                    cpu_offload=True,
                ),
            )
            dist_state_dict["ema"] = ema_state_dict

        if "ckpt_optimizer" in ckpt_components_needed:
            optim_state_dict = get_optimizer_state_dict(
                model,
                optimizer,
                options=StateDictOptions(
                    full_state_dict=False,
                    cpu_offload=True,
                ),
            )
            dist_state_dict["optimizer"] = optim_state_dict

        # Load distributed components
        dcp.load(dist_state_dict, checkpoint_id=str(checkpoint_path), planner=DefaultLoadPlanner(allow_partial_load=True))


        # Load shared data from shared.pth (all ranks)
        shared_file = checkpoint_path / "shared.pth"
        if shared_file.exists():
            shared_state_dict = torch.load(shared_file, map_location="cpu", weights_only=False)
            logger.info("Loaded shared data from shared.pth")
        else:
            logger.info("No shared.pth file found, not loading shared data")

        # Combine state dicts for applying the loading plan
        state_dict = {**dist_state_dict, **shared_state_dict}

        # Apply loaded states according to the loading plan
        self._apply_loading_plan(plan_mapping, state_dict, model, ema, optimizer, scheduler)

        # Restore RNG states if they are present in the checkpoint
        state_dict["metadata"] = state_dict["metadata"] if "metadata" in state_dict else {'step': 0} ## added new
        self._restore_rng_state(state_dict["metadata"].get("rng_states", None))

        elapsed = time.time() - start_time
        logger.info(f"Successfully loaded checkpoint from {checkpoint_path} in {elapsed:.3f}s")

        # Return metadata
        return cast(dict[str, Any], state_dict["metadata"])

    def _parse_loading_plan(self, loading_plan: str | None) -> dict[str, str]:
        """
        Parse the loading plan string into a mapping dictionary.

        Args:
            loading_plan: Loading plan string or None for default loading

        Returns:
            Dictionary mapping checkpoint components to memory targets
        """
        if loading_plan is None:
            # Default: load all components to their standard targets
            return {
                "ckpt_model": "mem_model",
                # "ckpt_ema": "mem_ema",
                "ckpt_optimizer": "mem_optimizer",
                "ckpt_scheduler": "mem_scheduler",
                "ckpt_step": "mem_step",
            }

        plan_mapping = {}
        # Split by comma and parse each mapping
        for mapping in loading_plan.split(","):
            mapping = mapping.strip()
            if ":" not in mapping:
                raise ValueError(f"Invalid loading plan format: '{mapping}'. Expected 'source:target'")

            source, target = mapping.split(":", 1)
            source = source.strip()
            target = target.strip()

            # Validate source and target names
            valid_sources = ["ckpt_model", "ckpt_ema", "ckpt_optimizer", "ckpt_scheduler", "ckpt_step"]
            valid_targets = ["mem_model", "mem_ema", "mem_optimizer", "mem_scheduler", "mem_step"]

            if source not in valid_sources:
                raise ValueError(f"Invalid source component: '{source}'. Valid sources: {valid_sources}")
            if target not in valid_targets:
                raise ValueError(f"Invalid target component: '{target}'. Valid targets: {valid_targets}")

            plan_mapping[source] = target

        return plan_mapping

    def _apply_loading_plan(
        self,
        plan_mapping: dict[str, str],
        full_state_dict: dict[str, Any],
        model: nn.Module,
        ema: nn.Module,
        optimizer: torch.optim.Optimizer=None,
        scheduler: torch.optim.lr_scheduler._LRScheduler=None,
    ) -> None:
        """
        Apply the loading plan by setting state dicts according to the mapping.

        Args:
            plan_mapping: Dictionary mapping checkpoint components to memory targets
            full_state_dict: Combined state dictionary from checkpoint (distributed + shared)
            model: FSDP2-wrapped main model
            ema: FSDP2-wrapped EMA model
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
        """
        if "ckpt_step" not in plan_mapping:
            full_state_dict["metadata"]["step"] = 0

        for ckpt_source, mem_target in plan_mapping.items():
            if ckpt_source == "ckpt_model" and mem_target == "mem_model":
                # Standard model loading
                set_model_state_dict(
                    model,
                    model_state_dict=cast(dict[str, Any], full_state_dict["model"]),
                    options=StateDictOptions(strict=True),
                )
                logger.info("Loaded model state from ckpt_model to mem_model")

            elif ckpt_source == "ckpt_model" and mem_target == "mem_ema":
                # Load model weights into EMA model
                set_model_state_dict(
                    ema,
                    model_state_dict=cast(dict[str, Any], full_state_dict["model"]),
                    options=StateDictOptions(strict=True),
                )
                logger.info("Loaded model state from ckpt_model to mem_ema")

            elif ckpt_source == "ckpt_ema" and mem_target == "mem_model":
                # Load EMA weights into main model
                set_model_state_dict(
                    model,
                    model_state_dict=cast(dict[str, Any], full_state_dict["ema"]),
                    options=StateDictOptions(strict=True),
                )
                logger.info("Loaded EMA state from ckpt_ema to mem_model")

            elif ckpt_source == "ckpt_ema" and mem_target == "mem_ema":
                # Standard EMA loading
                set_model_state_dict(
                    ema,
                    model_state_dict=cast(dict[str, Any], full_state_dict["ema"]),
                    options=StateDictOptions(strict=True),
                )
                logger.info("Loaded EMA state from ckpt_ema to mem_ema")

            elif ckpt_source == "ckpt_optimizer" and mem_target == "mem_optimizer":
                if optimizer is None:
                    raise ValueError("Optimizer is not provided")
                # Standard optimizer loading
                set_optimizer_state_dict(
                    model,
                    optimizer,
                    optim_state_dict=cast(dict[str, Any], full_state_dict["optimizer"]),
                    options=StateDictOptions(strict=False),
                )
                logger.info("Loaded optimizer state from ckpt_optimizer to mem_optimizer")

            elif ckpt_source == "ckpt_scheduler" and mem_target == "mem_scheduler":
                if scheduler is None:
                    raise ValueError("Scheduler is not provided")
                # Standard scheduler loading
                scheduler.load_state_dict(full_state_dict["scheduler"])
                logger.info("Loaded scheduler state from ckpt_scheduler to mem_scheduler")

            elif ckpt_source == "ckpt_step" and mem_target == "mem_step":
                pass  # No need to do anything; we'll set the step to 0 if it's not part of the loading plan

            else:
                raise ValueError(f"Unsupported loading plan mapping: {ckpt_source} -> {mem_target}")

    def _gather_rng_states(self) -> list[dict[str, torch.Tensor]]:
        """Gather the current CPU and CUDA RNG states from every rank.

        Returns:
            List where ``i``-th element holds the RNG state dict for rank ``i``.
        """
        local_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),  # this is already on cpu
        }
        gathered_states: list[dict[str, torch.Tensor] | None] = [None] * self.world_size  # type: ignore[var-annotated]
        dist.all_gather_object(gathered_states, local_state)
        return gathered_states

    def _restore_rng_state(self, rng_states: list[dict[str, torch.Tensor]] | None) -> None:
        """Restore the RNG state for *this* rank from a list collected at save-time."""
        if rng_states is None:
            logger.warning("No RNG states found in checkpoint, keeping existing RNG state.")
            return

        if self.rank >= len(rng_states):
            logger.warning(
                f"Checkpoint only contains RNG states for {len(rng_states)} ranks but current rank is {self.rank}; "
                "keeping existing RNG state."
            )
            return

        state_dict = rng_states[self.rank]
        torch.set_rng_state(state_dict["cpu"])
        torch.cuda.set_rng_state(state_dict["cuda"])  # no need to move to gpu
        logger.info(f"Restored RNG state for current rank {self.rank}")

    def list_checkpoints(self) -> list[Path]:
        """
        List all available checkpoints in the checkpoint directory.

        Returns:
            List of checkpoint directory paths
        """
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = [
            item
            for item in self.checkpoint_dir.iterdir()
            if item.is_dir() and (item / ".metadata").exists() and (item / "shared.pth").exists()
        ]

        # Sort by descending names (newest first)
        checkpoints.sort(reverse=True)
        return checkpoints

    def get_latest_checkpoint(self) -> Path | None:
        """
        Get the path to the most recent checkpoint.

        Returns:
            Path to latest checkpoint directory, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None

    def cleanup_old_checkpoints(self, keep_last: int = 5) -> None:
        """
        Remove old checkpoints, keeping only the most recent ones.

        Args:
            keep_last: Number of recent checkpoints to keep
        """
        if self.rank != 0:
            return  # Only cleanup on rank 0

        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep_last:
            return

        # Remove old checkpoints
        for checkpoint_path in checkpoints[keep_last:]:
            try:
                # Remove checkpoint directory
                import shutil

                shutil.rmtree(checkpoint_path)
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")

    def resume_latest(
        self,
        model: nn.Module,
        ema: nn.Module,
        optimizer: torch.optim.Optimizer=None,
        scheduler: torch.optim.lr_scheduler._LRScheduler=None,
        init_ckpt: str | None = None,
        init_ckpt_load_plan: str | None = None,
    ) -> int:
        """
        Resume training from the latest available checkpoint.

        Args:
            model: FSDP2-wrapped main model to load into
            ema: FSDP2-wrapped EMA model to load into
            optimizer: Optimizer to load into
            scheduler: Learning rate scheduler to load into (optional)
            loading_plan: Selective loading plan string (optional)

        Returns:
            Step number to resume from (0 if no checkpoint found)
        """
        # Each rank independently finds and loads the latest checkpoint
        latest_checkpoint = self.get_latest_checkpoint()
        step = 0

        candidate_checkpoints = [latest_checkpoint, init_ckpt]
        for checkpoint in candidate_checkpoints:
            if checkpoint:
                logger.info(f"Loading checkpoint from {checkpoint}")
                metadata = self.load_checkpoint(
                    checkpoint_path=checkpoint,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loading_plan=init_ckpt_load_plan if checkpoint == init_ckpt else None,
                )
                step = metadata["step"]
                logger.info(f"Loaded checkpoint from {checkpoint} with step={step}")
                break

        # Synchronize decisions across all ranks to detect inconsistencies
        # Use tensor-based gathering to avoid pickle serialization issues
        
        # Convert checkpoint path to string representation (max 512 bytes should be enough)
        path_str = str(latest_checkpoint) if latest_checkpoint is not None else ""
        path_bytes = path_str.encode('utf-8')
        path_tensor = torch.zeros(512, dtype=torch.uint8, device=f'cuda:{self.local_rank}')
        if len(path_bytes) > 0:
            path_tensor[:len(path_bytes)] = torch.frombuffer(path_bytes, dtype=torch.uint8)
        
        # Create step tensor
        step_tensor = torch.tensor(step, dtype=torch.int64, device=f'cuda:{self.local_rank}')
        
        # Gather path tensors from all ranks
        gathered_path_tensors = [torch.zeros_like(path_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_path_tensors, path_tensor)
        # dist.gather_object(path_tensor, object_gather_list=gathered_path_tensors, dst=0)
        
        # Gather step tensors from all ranks  
        gathered_step_tensors = [torch.zeros_like(step_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_step_tensors, step_tensor)
        # dist.gather_object(step_tensor, object_gather_list=gathered_step_tensors, dst=0)
        
        # Convert tensors back to original format
        gathered_paths = []
        for path_t in gathered_path_tensors:
            # Find actual length by looking for first zero byte
            path_length = (path_t != 0).sum().item()
            if path_length == 0:
                gathered_paths.append(None)
            else:
                path_bytes_recovered = path_t[:path_length].cpu().numpy().tobytes()
                path_str_recovered = path_bytes_recovered.decode('utf-8')
                gathered_paths.append(Path(path_str_recovered))
        
        gathered_steps = [tensor.cpu().item() for tensor in gathered_step_tensors]

        # Check if all ranks agree on checkpoint path
        unique_paths = set(gathered_paths)
        if len(unique_paths) > 1:
            path_summary = {path: [i for i, p in enumerate(gathered_paths) if p == path] for path in unique_paths}
            error_msg = (
                f"Checkpoint path inconsistency detected across ranks: {path_summary}. "
                f"This indicates file system inconsistency that could lead to training divergence."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Check if all ranks agree on step number
        unique_steps = set(gathered_steps)
        if len(unique_steps) > 1:
            step_summary = {
                step_val: [i for i, s in enumerate(gathered_steps) if s == step_val] for step_val in unique_steps
            }
            error_msg = (
                f"Step number inconsistency detected across ranks: {step_summary}. "
                f"This indicates checkpoint corruption or loading inconsistency."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # All ranks are synchronized
        if latest_checkpoint:
            logger.info(f"All ranks synchronized on checkpoint: {latest_checkpoint}, step: {step}")
        else:
            logger.info("All ranks synchronized on no checkpoint, starting from step 0")

        return step

    def finish(self) -> None:
        """
        Wait for all async_save futures to complete.
        """
        for future in self.async_futures.values():
            if not future.done():
                future.result()
        self.async_futures.clear()
