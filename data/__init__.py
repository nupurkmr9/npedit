from collections.abc import Iterator
from typing import Any, cast
import time
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.distributed as dist

from models.latent_fm import FMDataContext
from utils.config import create_component
from utils.log import get_logger
logger = get_logger(__name__)


class DataStreamer:
    """Base class for data streamers."""

    def __init__(
        self, config: dict[str, Any], data_seed: int = 42, data_process_group: dist.ProcessGroup | None = None
    ):
        self.config = config

        self.data_module = create_component(
            config["module"], config["params"], data_seed=data_seed, data_process_group=data_process_group
        )

    def train_dataloader(self) -> Iterator[FMDataContext]:
        """Return the training dataloader with a precise type for static analysis."""
        return cast(Iterator[FMDataContext], self.data_module.train_dataloader())

    def val_dataloader(self) -> Iterator[FMDataContext]:
        """Return the validation dataloader with a precise type for static analysis."""
        return cast(Iterator[FMDataContext], self.data_module.val_dataloader())
    
    def prepare_batch(self, data_raw, device) -> FMDataContext:
        fm_data_context = FMDataContext(
            prompts=data_raw["prompts"],
            images=data_raw["images"].to(device, non_blocking=True)  * 2. -1. ,
            hash_key=data_raw["hash_key"],
            idx=data_raw["idx"] if "idx" in data_raw else None,
            edited_prompts=data_raw["edited_prompts"],
            input_ids=data_raw["input_ids"].to(device, non_blocking=True) if "input_ids" in data_raw else None,
            labels=data_raw["labels"].to(device, non_blocking=True) if "labels" in data_raw else None,
            input_ids_identity=data_raw["input_ids_identity"].to(device, non_blocking=True) if "input_ids_identity" in data_raw else None,
            labels_identity=data_raw["labels_identity"].to(device, non_blocking=True) if "labels_identity" in data_raw else None,
            negative_prompts=data_raw.get("negative_prompts"),
            sub_task_type=data_raw["sub_task_type"] if "sub_task_type" in data_raw else None,
            task_type=data_raw["task_type"] if "task_type" in data_raw else None,
            cfg=data_raw["cfg"].to(device, non_blocking=True) if "cfg" in data_raw else None,
        )
        fm_data_context.reference_images = fm_data_context.images
        return fm_data_context


@dataclass
class MiniBatchWrapper:
    bs: int
    mini_bs: int
    fm_data_context: FMDataContext


class InstructWrapper:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iterator = iter(self.dataloader)

    def __iter__(self):
        """Return the same iterator each time"""
        return self._iterator

    def __len__(self):
        """Return the length of the dataloader"""
        return len(self.dataloader)

    def _get_next_batch(self):
        """Get the next batch, restarting the iterator if needed"""
        try:
            return next(self._iterator)
        except StopIteration:
            # Iterator is exhausted, create a new one
            self.dataloader.sampler.set_epoch(self.dataloader.sampler.epoch + 1)
            if hasattr(self.dataloader.dataset, 'set_epoch'):
                self.dataloader.dataset.set_epoch(self.dataloader.sampler.epoch)
            if dist.get_rank() == 0:
                logger.info("Restarting dataloader iterator")
            self._iterator = iter(self.dataloader)
            return next(self._iterator)


class MiniBatchDataLoader:
    def __init__(self, data_module, dataloader, embed_fn=None, start_step=0, tracking_logger=None, total_batch_size=-1, txt_drop_prob=0.0):
        self.data_module = data_module

        if not isinstance(dataloader, InstructWrapper):
            dataloader = InstructWrapper(dataloader)

        self.dataloader = dataloader
        self.device = torch.cuda.current_device()
        self.minibatch_step = 0
        self.start_step = start_step
        self.total_batch_size = total_batch_size
        self.txt_drop_prob = txt_drop_prob
        # Ensure we can get at least one batch
        fm_data_context = None
        for fm_data_context in self.dataloader:
            fm_data_context = self.data_module.prepare_batch(fm_data_context, self.device)
            break
        
        if fm_data_context is None:
            if dist.get_rank() == 0:
                logger.error("Could not get initial batch from dataloader!")
            raise RuntimeError("Dataloader is empty or exhausted")

        self.mini_bs = fm_data_context.images.shape[0]

        bs = torch.tensor(self.mini_bs, device=self.device)
        dist.all_reduce(bs, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        self.bs = int(bs.item())
        self.embed_fn = embed_fn
        if self.total_batch_size == -1:
            self.total_batch_size = self.bs
        self.num_accum_steps = self.total_batch_size // self.bs

    def step(self, tracking_logger=None) -> Iterator[MiniBatchWrapper]:
        # construct an iterator of minibatches which sum up to full_bs across all GPUs
        accum_bs = 0
        while accum_bs < self.total_batch_size:
            with tracking_logger.log_time(f"time/data"):
                fm_data_context = self.dataloader._get_next_batch()
            self.minibatch_step += 1

            fm_data_context = self.data_module.prepare_batch(fm_data_context, self.device)
            accum_bs += self.bs

            if self.embed_fn is not None:
                with tracking_logger.log_time(f"time/frozen_ops"):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):  # <-- ADD THIS
                        fm_data_context = self.embed_fn(fm_data_context, txt_drop_prob=self.txt_drop_prob)
            
            yield MiniBatchWrapper(
                bs=self.bs,
                mini_bs=self.mini_bs,
                fm_data_context=fm_data_context,
            )