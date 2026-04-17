"""Distributed bucket sampler for variable-resolution training.

Adapted from nunchaku-train-dev DistributedBucketSampler.
Ensures every batch contains samples from the same resolution bucket
so that torch.stack works without padding.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


def _reverse_index_map(bucket_ids: list[int]) -> dict[int, list[int]]:
    """Build a mapping from bucket_id to the list of dataset indices in that bucket."""
    bucket_map: dict[int, list[int]] = {}
    for data_id, bucket_id in enumerate(bucket_ids):
        bucket_map.setdefault(bucket_id, []).append(data_id)
    return bucket_map


class DistributedBucketSampler(Sampler):
    """Distributed sampler that groups samples by ``bucket_id``.

    Guarantees every batch contains samples from the **same bucket**
    (i.e. the same resolution), so ``torch.stack`` works without padding.

    Parameters
    ----------
    dataset : Dataset
        Must expose a ``bucket_ids`` attribute (``list[int]``, one per sample).
    samples_per_gpu : int
        Batch size per replica.  Batches are formed from a single bucket.
    num_replicas : int | None
        Number of distributed processes.  Defaults to ``world_size``.
    rank : int | None
        Rank of the current process.  Defaults to ``get_rank``.
    shuffle : bool
        Shuffle within each bucket and shuffle the global batch order.
    seed : int
        Base random seed (combined with ``epoch`` for reproducibility).
    """

    def __init__(
        self,
        dataset: Dataset,
        samples_per_gpu: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu

        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        bucket_ids: list[int] = dataset.bucket_ids  # type: ignore[attr-defined]
        self.bucket_map = dict(sorted(_reverse_index_map(bucket_ids).items()))

        # Compute padded sizes per bucket and total
        self.total_size_per_bucket: dict[int, int] = {}
        total_padded = 0
        for bucket_id, indices in self.bucket_map.items():
            if len(indices) < samples_per_gpu:
                raise ValueError(
                    f"Bucket {bucket_id} has only {len(indices)} samples, "
                    f"which is less than samples_per_gpu={samples_per_gpu}. "
                    f"Use fewer GPUs or a smaller batch size."
                )
            num_batches = int(np.ceil(len(indices) / samples_per_gpu))
            padded = num_batches * samples_per_gpu
            self.total_size_per_bucket[bucket_id] = padded
            total_padded += padded

        total_num_batches = total_padded // samples_per_gpu
        # Pad total to be divisible by num_replicas (in batch units)
        padded_total_batches = int(np.ceil(total_num_batches / self.num_replicas)) * self.num_replicas
        self.total_size = padded_total_batches * samples_per_gpu
        self.num_samples = self.total_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches: list[torch.Tensor] = []

        for bucket_id, data_indices in self.bucket_map.items():
            indices = torch.tensor(data_indices, dtype=torch.long)
            if self.shuffle:
                indices = indices[torch.randperm(len(indices), generator=g)]

            # Pad to make divisible by samples_per_gpu
            padded_size = self.total_size_per_bucket[bucket_id]
            if padded_size > indices.numel():
                indices = torch.cat([indices, indices[: padded_size - indices.numel()]])

            total_batches_in_bucket = padded_size // self.samples_per_gpu
            full_rounds = total_batches_in_bucket // self.num_replicas
            leftover_batches = total_batches_in_bucket % self.num_replicas

            # Full round-robin batches: reshape so each replica gets the same count
            full_count = full_rounds * self.num_replicas * self.samples_per_gpu
            if full_count > 0:
                batches_a = (
                    indices[:full_count]
                    .reshape(full_rounds, self.samples_per_gpu, self.num_replicas)
                    .permute(0, 2, 1)
                    .reshape(full_rounds * self.num_replicas, self.samples_per_gpu)
                )
                all_batches.append(batches_a)

            # Leftover partial round-robin
            if leftover_batches > 0:
                batches_b = (
                    indices[full_count:]
                    .reshape(self.samples_per_gpu, leftover_batches)
                    .permute(1, 0)
                )
                all_batches.append(batches_b)

        # Concatenate all batches: (total_num_batches, samples_per_gpu)
        all_batches_tensor = torch.cat(all_batches, dim=0)

        # Shuffle the global batch order
        if self.shuffle:
            all_batches_tensor = all_batches_tensor[
                torch.randperm(all_batches_tensor.size(0), generator=g)
            ]

        # Pad total batches to be divisible by num_replicas
        total_num_batches = self.total_size // self.samples_per_gpu
        pad_batches = total_num_batches - all_batches_tensor.size(0)
        if pad_batches > 0:
            all_batches_tensor = torch.cat(
                [all_batches_tensor, all_batches_tensor[:pad_batches]], dim=0
            )

        # Slice for this rank: take every num_replicas-th batch
        rank_batches = all_batches_tensor[self.rank :: self.num_replicas]
        indices = rank_batches.flatten().tolist()
        return iter(indices)
