import json, os, random
from typing import Any, Callable, List
from dataclasses import dataclass
from PIL import Image, ImageOps
from copy import deepcopy
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from torch.distributed import ReduceOp
import torchvision.transforms as T
from datasets import load_dataset
import torch.distributed as dist
import torch


from transformers import AutoTokenizer

from utils.config import BaseParams, ConfigurableModule
from utils.log import get_logger
from critic_models.prompts import SELECTION_PROMPT

from .utils_ import *


Image.MAX_IMAGE_PIXELS = 933120000
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Resolution bucketing helpers
# ---------------------------------------------------------------------------

DEFAULT_RESOLUTION_BUCKETS = [
    (512, 512),    # 1:1
    (576, 448),    # ~1.29:1 landscape
    (448, 576),    # portrait
    (640, 384),    # ~1.67:1 wide
    (384, 640),    # tall
    (512, 384),    # 4:3
    (384, 512),    # 3:4
]


def find_nearest_bucket(w: int, h: int, buckets: list) -> int:
    """Return index of bucket whose aspect ratio is closest to w/h."""
    aspect = w / h
    # buckets are (H, W) tuples
    return min(range(len(buckets)), key=lambda i: abs((buckets[i][1] / buckets[i][0]) - aspect))


def build_bucket_transform(height: int, width: int):
    """Resize shortest side then center-crop to (height, width)."""
    return T.Compose([
        T.Resize(min(height, width), interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop((height, width)),
        T.ToTensor(),
    ])


class GEditBenchDataset(Dataset):
    def __init__(
        self,
        resolution: int = 512,
        transform: Callable = None,
        bucketize: bool = False,
        resolution_buckets: list = None,
    ):
        dataset = load_dataset("stepfun-ai/GEdit-Bench")
        # Define filter function
        def filter_func(item):
            if item['instruction_language'] != 'en':
                return False
            return True

        self.base_dataset = dataset["train"].filter(filter_func)
        self.resolution = resolution

        # --- Resolution bucketing ---
        self.bucket_ids = None
        self._bucket_transforms = {}

        if bucketize:
            buckets = [tuple(b) for b in (resolution_buckets or DEFAULT_RESOLUTION_BUCKETS)]
            self._bucket_resolutions = {i: b for i, b in enumerate(buckets)}
            self._bucket_transforms = {
                i: build_bucket_transform(h, w) for i, (h, w) in enumerate(buckets)
            }
            self.bucket_ids = []
            for item in self.base_dataset:
                img = item["input_image"]
                w, h = img.size
                bid = find_nearest_bucket(w, h, buckets)
                self.bucket_ids.append(bid)

            from collections import Counter
            bucket_counts = Counter(self.bucket_ids)
            print(f"[GEditBenchDataset] Bucket distribution: { {buckets[k]: v for k, v in sorted(bucket_counts.items())} }")

        self.transform = transform
        if self.transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.base_dataset)

    def getitem(self, idx):
        item = self.base_dataset[idx]

        image_pil = item["input_image"]

        if self.bucket_ids is not None:
            bid = self.bucket_ids[idx]
            image = self._bucket_transforms[bid](image_pil)
        else:
            w, h = image_pil.size
            new_h = self.resolution
            new_w = int(w * new_h / h)
            # Round width to nearest multiple of 16 (8 VAE compression × 2 denoiser patch size)
            new_w = max(16, round(new_w / 16) * 16)
            image_pil = image_pil.resize((new_w, new_h), Image.LANCZOS)
            image = self.transform(image_pil)

        edit_instruction = item["instruction"]

        return {
            "images": [image, image],
            "prompts": edit_instruction,
            "edited_prompts": edit_instruction,
            "task_type": "editing",
            "hash_key": item['key'],
            'sub_task_type': item['task_type'],
            "pil_images": [image_pil, image_pil],
            "cfg": torch.tensor(7.5)
        }

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            my_bucket = self.bucket_ids[idx] if self.bucket_ids is not None else None
            while True:
                try:
                    alt_idx = random.randint(0, self.__len__() - 1)
                    if my_bucket is not None and self.bucket_ids[alt_idx] != my_bucket:
                        continue
                    return self.getitem(alt_idx)
                except Exception:
                    continue


class LaionAesthetics(Dataset):
    def __init__(
        self,
        resolution: int = 512,
        transform: Callable = None,
    ):
        ds = load_dataset("laion/aesthetics_v2_4.5")
        self.base_dataset = ds["train"]
        self.transform = transform
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize(resolution, interpolation=T.InterpolationMode.LANCZOS),
                T.CenterCrop((resolution, resolution)),
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.base_dataset)

    def getitem(self, idx):
        item = self.base_dataset[idx]

        # Crop the image to target and condition
        image_url = item["URL"]
        image_pil = load_from_url(image_url)
        max_dim = max(image_pil.size)
        image_pil = ImageOps.pad(image_pil, (max_dim, max_dim), color='white')

        image = self.transform(image_pil)
        edit_instruction = item["TEXT"]

        return {
            "images": [image, image],
            "prompts": edit_instruction,
            "edited_prompts": edit_instruction,
            "original_prompts": edit_instruction,
            "task_type": "editing",
            "hash_key": item['hash'],
            'sub_task_type': 'none',
            "pil_images": [image_pil, image_pil],
            "cfg": torch.tensor(7.5)
        }
    
    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            while True:
                try:
                    return self.getitem(random.randint(0, self.__len__() - 1))
                except Exception:
                    continue


class EditDataset(Dataset):
    """Unified editing dataset that reads from a single pre-merged JSON file.

    The JSON is produced by ``scripts/merge_edit_datasets.py`` and contains
    pre-resolved captions, edit types, and relative image paths.
    """

    def __init__(
        self,
        data_path: str,
        root_dir: str,
        resolution: int = 512,
        transform: Callable = None,
        num_samples_ratio: float = 1.0,
        bucketize: bool = False,
        resolution_buckets: list = None,
    ):
        self.root_dir = root_dir

        with open(data_path, "r") as f:
            self.data = json.load(f)

        if num_samples_ratio < 1.0:
            random.shuffle(self.data)
            self.data = self.data[:int(len(self.data) * num_samples_ratio)]

        print(f"[EditDataset] Loaded {len(self.data)} samples from {data_path}")

        # --- Resolution bucketing ---
        self.bucket_ids = None
        self._bucket_transforms = {}

        if bucketize:
            buckets = [tuple(b) for b in (resolution_buckets or DEFAULT_RESOLUTION_BUCKETS)]
            self._bucket_resolutions = {i: b for i, b in enumerate(buckets)}
            self._bucket_transforms = {
                i: build_bucket_transform(h, w) for i, (h, w) in enumerate(buckets)
            }
            self.bucket_ids = []
            for item in self.data:
                if 'width' in item and 'height' in item:
                    w, h = item['width'], item['height']
                else:
                    img_path = os.path.join(self.root_dir, item["input_image_path"])
                    with Image.open(img_path) as img:
                        w, h = img.size
                bid = find_nearest_bucket(w, h, buckets)
                self.bucket_ids.append(bid)

            from collections import Counter
            bucket_counts = Counter(self.bucket_ids)
            print(f"[EditDataset] Bucket distribution: { {buckets[k]: v for k, v in sorted(bucket_counts.items())} }")

        self.transform = transform
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize(resolution, interpolation=T.InterpolationMode.LANCZOS),
                T.CenterCrop((resolution, resolution)),
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.data)

    def getitem(self, idx):
        item = self.data[idx]

        img_path = os.path.join(self.root_dir, item["input_image_path"])
        image_pil = Image.open(img_path).convert("RGB")

        if self.bucket_ids is not None:
            # Bucket-aware transform: resize + crop to bucket resolution
            bid = self.bucket_ids[idx]
            image = self._bucket_transforms[bid](image_pil)
        else:
            # Original: center-crop to square
            min_dim = min(image_pil.size)
            w, h = image_pil.size
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            image_pil = image_pil.crop((left, top, left + min_dim, top + min_dim))
            image = self.transform(image_pil)

        edit_type = item.get("edit_type", "")

        return {
            "images": [image, image],
            "prompts": item["edit_instruction"],
            "edited_prompts": item["edited_caption"],
            "original_prompts": item.get("input_description", ""),
            "task_type": "editing",
            "sub_task_type": edit_type,
            "dataset_source": item.get("dataset_source", ""),
            "instruction_type": item.get("instruction_type", ""),
            "hash_key": item["id"],
            "category": item.get("category", ""),
            "pil_images": [image_pil, image_pil],
            "cfg": torch.tensor(7.5) if edit_type in ['remove', 'shape change', 'style', 'action'] else torch.tensor(3.5),
        }

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            my_bucket = self.bucket_ids[idx] if self.bucket_ids is not None else None
            while True:
                try:
                    alt_idx = random.randint(0, self.__len__() - 1)
                    if my_bucket is not None and self.bucket_ids[alt_idx] != my_bucket:
                        continue
                    return self.getitem(alt_idx)
                except Exception:
                    continue


class CombinedDataset(Dataset):
    def __init__(self, datasets: List[Dataset], val=False):
        self.datasets = datasets
        # Filter out datasets that don't have __len__ method
        valid_datasets = [ds for ds in datasets if hasattr(ds, '__len__')]
        self.lengths = [len(ds) for ds in valid_datasets]
        self.cumulative_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        for i, dataset_len in enumerate(self.cumulative_lengths):
            if idx < dataset_len:
                if i > 0:
                    idx = idx - self.cumulative_lengths[i-1]
                batch = self.datasets[i][idx]
                return batch


def get_category(prompt, sub_task_type):
    if sub_task_type in prompt:
        return prompt.lower().split(sub_task_type)[1]
    else:
        return prompt


class CriticCombinedDataset(Dataset):
    """
    A dataset that combines multiple datasets and samples from each while populating the critic questions and answers.
    """
    def __init__(
        self,
        datasets: List[Dataset],
        critic_resolution=512,
        critic_identity_resolution=512,
        critic_model_name='',
        critic_identity_model_name='',
        do_nothing_prob=0.0,
        combined_edit_customization=False,
        val=False,
        edit_type_probs=None,
    ):
        self.combined_edit_customization = combined_edit_customization
        self.do_nothing_prob = do_nothing_prob
        self.datasets = datasets
        self.val = val
        self.critic_resolution = critic_resolution
        self.critic_identity_resolution = critic_identity_resolution
        self.critic_model_name = critic_model_name
        self.critic_identity_model_name = critic_identity_model_name
        if critic_model_name != '':
            self.processor = AutoTokenizer.from_pretrained(
                critic_model_name, trust_remote_code=True, use_fast=False
            )
            # InternVL: 256 image tokens per 448x448 tile after pixel unshuffle
            self.num_image_token = 256
            self.img_context_token_id = self.processor.convert_tokens_to_ids('<IMG_CONTEXT>')
        else:
            self.processor = None

        if critic_identity_model_name != '':
            self.processor_identity_eval = AutoTokenizer.from_pretrained(
                critic_identity_model_name, trust_remote_code=True, use_fast=False
            )
        else:
            self.processor_identity_eval = None

        # Calculate total length
        self.total_length = sum(len(ds)-32 for ds in self.datasets) if not self.val else sum(32 for _ in self.datasets)
        self.lengths = [len(ds) - 32 if not self.val else 32 for ds in self.datasets]
        self.cumulative_lengths = np.cumsum(self.lengths)

        # Edit-type based sampling
        self.edit_type_probs = edit_type_probs
        if edit_type_probs is not None and not val:
            self._build_edit_type_index()
            self.total_length = self.editing_unique_total + self.other_total

        # --- Build unified bucket_ids across all sub-datasets ---
        self.bucket_ids = None
        has_buckets = any(getattr(ds, 'bucket_ids', None) is not None for ds in self.datasets)
        if has_buckets:
            self.bucket_ids = self._build_bucket_ids()

    def _build_bucket_ids(self) -> list[int]:
        """Build a flat bucket_ids list aligned with __getitem__ index space."""
        bucket_ids = []

        if self.edit_type_probs is not None and not self.val:
            # edit_type_probs active: indices [0, editing_unique_total) come from
            # the epoch schedule, indices [editing_unique_total, ...) from "other" datasets.
            for ds_idx, sample_idx in self._epoch_schedule:
                ds = self.datasets[ds_idx]
                ds_bids = getattr(ds, 'bucket_ids', None)
                bucket_ids.append(ds_bids[sample_idx] if ds_bids is not None else 0)
            # "Other" dataset samples appended after editing samples
            for ds_idx in self.other_ds_indices:
                ds = self.datasets[ds_idx]
                ds_len = self.lengths[ds_idx]
                ds_bids = getattr(ds, 'bucket_ids', None)
                if ds_bids is not None:
                    bucket_ids.extend(ds_bids[:ds_len])
                else:
                    bucket_ids.extend([0] * ds_len)
        else:
            # Sequential concatenation: indices map to sub-datasets in order
            for ds_idx, ds in enumerate(self.datasets):
                ds_len = self.lengths[ds_idx]
                ds_bids = getattr(ds, 'bucket_ids', None)
                if ds_bids is not None:
                    if self.val:
                        bucket_ids.extend(ds_bids[-32:])
                    else:
                        bucket_ids.extend(ds_bids[:ds_len])
                else:
                    bucket_ids.extend([0] * ds_len)

        return bucket_ids

    def _build_edit_type_index(self):
        """Pre-build an index mapping canonical edit_types to (dataset_idx, sample_idx) pairs."""
        from collections import defaultdict
        self.edit_type_index = defaultdict(list)
        self.editing_dataset_indices = set()
        self.other_dataset_indices = set()

        for ds_idx, ds in enumerate(self.datasets):
            if isinstance(ds, EditDataset):
                self.editing_dataset_indices.add(ds_idx)
                max_idx = len(ds) - 32  # exclude last 32 for val
                for sample_idx in range(max_idx):
                    edit_type = self._get_edit_type_from_metadata(ds, sample_idx)
                    canonical = match_edit_type(edit_type)
                    if canonical is not None and canonical in self.edit_type_probs:
                        self.edit_type_index[canonical].append((ds_idx, sample_idx))
            else:
                self.other_dataset_indices.add(ds_idx)

        # Build per-type shuffled queues and position pointers
        self.edit_type_queues = {}
        self.edit_type_positions = {}
        for et, indices in self.edit_type_index.items():
            arr = indices.copy()
            np.random.shuffle(arr)
            self.edit_type_queues[et] = arr
            self.edit_type_positions[et] = 0

        # Compute lengths for editing vs other datasets
        self.editing_total = sum(self.lengths[i] for i in self.editing_dataset_indices)
        self.other_total = sum(self.lengths[i] for i in self.other_dataset_indices)

        # Precompute cumulative lengths for other datasets only
        self.other_ds_indices = sorted(self.other_dataset_indices)
        self.other_lengths = [self.lengths[i] for i in self.other_ds_indices]
        self.other_cumulative_lengths = np.cumsum(self.other_lengths) if self.other_lengths else np.array([])

        # Filter to edit types that actually have data
        edit_types_with_data = [et for et in self.edit_type_probs if et in self.edit_type_index and len(self.edit_type_index[et]) > 0]
        raw_probs = np.array([self.edit_type_probs[et] for et in edit_types_with_data])
        self._sampling_edit_types = edit_types_with_data
        self._sampling_probs = raw_probs / raw_probs.sum()

        # Compute editing_unique_total before generating the schedule
        self.editing_unique_total = sum(
            len(self.edit_type_index[et]) for et in self._sampling_edit_types
        )

        # Pre-generate the epoch schedule: a flat list of (ds_idx, sample_idx) in shuffled order
        # that respects the target edit_type proportions exactly.
        self._epoch_schedule = self._generate_epoch_schedule()

        print(f"[CriticCombinedDataset] Edit-type index built (epoch size={len(self._epoch_schedule)}):")
        for et in self._sampling_edit_types:
            print(f"  {et}: {len(self.edit_type_index[et])} samples, prob={self.edit_type_probs[et]:.3f}")

    def _generate_epoch_schedule(self):
        """
        Pre-generate a full epoch of editing samples.

        Allocates exactly N * prob_i slots to each edit_type i (where N = editing_total),
        fills them by drawing sequentially from that type's shuffled queue, then shuffles
        the whole schedule. This guarantees:
          - Exact proportions per edit_type (up to integer rounding)
          - Within each type, samples are seen in shuffled order without repeats
          - When a type's queue is exhausted, it reshuffles before continuing
        """
        N = self.editing_unique_total
        schedule = []

        # Allocate slots per edit_type proportional to probs
        counts = {}
        remaining = N
        for i, et in enumerate(self._sampling_edit_types):
            if i == len(self._sampling_edit_types) - 1:
                # Last type gets the remainder to ensure exact total
                counts[et] = remaining
            else:
                counts[et] = int(round(self._sampling_probs[i] * N))
                remaining -= counts[et]

        # Fill each type's slots from its shuffled queue
        for et, n_samples in counts.items():
            queue = self.edit_type_queues[et]
            pos = self.edit_type_positions[et]
            for _ in range(n_samples):
                if pos >= len(queue):
                    np.random.shuffle(queue)
                    pos = 0
                schedule.append(queue[pos])
                pos += 1
            self.edit_type_positions[et] = pos

        # Shuffle the full schedule so types are interleaved
        np.random.shuffle(schedule)
        return schedule

    def _get_edit_type_from_metadata(self, ds, sample_idx):
        """Extract edit_type from dataset metadata without loading images."""
        if isinstance(ds, EditDataset):
            return ds.data[sample_idx].get("edit_type", "")
        return ''

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if self.edit_type_probs is not None and not self.val:
            return self._getitem_edit_type_sampling(idx)

        # Original sequential logic
        for dataset_idx, dataset_len in enumerate(self.cumulative_lengths):
            if idx < dataset_len:
                if dataset_idx > 0:
                    idx = idx - self.cumulative_lengths[dataset_idx-1]
                break
        if self.val:
            idx = len(self.datasets[dataset_idx]) - 32 + idx
        batch = self.datasets[dataset_idx][idx]
        return self.get_data(batch)

    def set_epoch(self, epoch: int):
        """Regenerate the epoch schedule so that idx-to-sample mapping changes
        each epoch, complementing DistributedSampler's index shuffling."""
        if self.edit_type_probs is not None and not self.val:
            if self.bucket_ids is not None:
                # When bucketing is active, the DistributedBucketSampler handles
                # per-epoch shuffling. The schedule (and bucket_ids) must stay
                # fixed so that bucket_ids[idx] always matches __getitem__(idx).
                return
            self._epoch_schedule = self._generate_epoch_schedule()

    def _getitem_edit_type_sampling(self, idx):
        """Sample using pre-generated epoch schedule for editing datasets.

        Uses idx to index directly into the schedule.  This is safe for
        multi-worker DataLoaders (each worker receives *different* idx
        values from the sampler) and for DistributedSampler (which
        guarantees unique idx values across ranks).
        """
        if idx < self.editing_unique_total:
            ds_idx, sample_idx = self._epoch_schedule[idx]

            batch = self.datasets[ds_idx][sample_idx]
            return self.get_data(batch)
        else:
            # Other datasets: use sequential logic
            other_idx = idx - self.editing_unique_total
            for i, cum_len in enumerate(self.other_cumulative_lengths):
                if other_idx < cum_len:
                    if i > 0:
                        other_idx = other_idx - self.other_cumulative_lengths[i - 1]
                    real_ds_idx = self.other_ds_indices[i]
                    batch = self.datasets[real_ds_idx][other_idx]
                    return self.get_data(batch)
            # Fallback: shouldn't reach here
            batch = self.datasets[self.other_ds_indices[-1]][0]
            return self.get_data(batch)

    def get_data(self, batch):
        category = batch['category'] if 'category' in batch else ''
        batch['negative_prompts'] = category
        task_type = batch['task_type']
        sub_task_type = batch['sub_task_type'] if 'sub_task_type' in batch else ''
        prompt = batch['prompts']

        if np.random.rand() < self.do_nothing_prob and task_type == 'editing':
            batch['task_type'] = 'do nothing'
            batch['prompts'] = prompt = 'do nothing.'

        if self.critic_model_name != '' and self.critic_identity_model_name != '':
            keys = [task_type  + '_identity', task_type]
            output_keys = ['_identity', '']
            for _, (key, output_key) in enumerate(zip(keys, output_keys)):

                # make sure user_prompt and answer have appropriate placeholders
                dict_ = SELECTION_PROMPT[key]
                dict_ = dict_[sub_task_type] if sub_task_type in dict_ else dict_['default']

                user_prompt = dict_['question']
                user_prompt = user_prompt.replace("<instruction>", prompt).strip()
                user_prompt = user_prompt.replace("<object_name>", category).strip()
                answer = dict_['answer']
                answer = answer.replace("<instruction>", prompt).strip()
                answer = answer.replace("<object_name>", category).strip()
                single_image = dict_['single_image']

                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{user_prompt}"},
                            {"type": "image", "image": "<image>"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer}
                        ],
                    }
                ]

                if not single_image:
                    message[0]['content'] = [
                            {"type": "text", "text": f"{user_prompt}"},
                            {"type": "image", "image": "<image>"},
                            {"type": "image", "image": "<image>"},
                        ]

                processor = self.processor_identity_eval if 'identity' in key else self.processor

                # InternVL tokenization path
                num_tiles = 1  # single 448x448 tile per image
                img_placeholder = '<img>' + '<IMG_CONTEXT>' * (self.num_image_token * num_tiles) + '</img>'
                if single_image:
                    img_section = img_placeholder + '\n'
                else:
                    img_section = f'Image-1: {img_placeholder}\nImage-2: {img_placeholder}\n'

                # Build full chat text
                query = f'<|im_start|>user\n{img_section}{user_prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>'
                input_ids = processor.encode(query, add_special_tokens=True, return_tensors='pt')

                # Build labels: mask everything before the answer
                labels = torch.full_like(input_ids, -100)
                im_start_id = processor.convert_tokens_to_ids('<|im_start|>')
                assistant_prefix = processor.encode('<|im_start|>assistant\n', add_special_tokens=False)
                assistant_prefix_len = len(assistant_prefix)
                assistant_positions = (input_ids[0] == im_start_id).nonzero(as_tuple=True)[0]
                if len(assistant_positions) >= 2:
                    answer_start = assistant_positions[1].item() + assistant_prefix_len
                else:
                    answer_start = assistant_positions[0].item() + assistant_prefix_len
                labels[:, answer_start:] = input_ids[:, answer_start:]
                labels[:, -1:] = -100  # mask the final <|im_end|> token

                batch[f'input_ids{output_key}'] = input_ids
                batch[f'labels{output_key}'] = labels

        del batch['pil_images']
        if self.combined_edit_customization:
            batch['prompts'] = 'Task type: ' + task_type + '.\n' + prompt

        return batch


@dataclass
class InstructDataModuleParams(BaseParams):
    """
    Configuration parameters for the instruct data module.
    """

    resolution: int = 512
    batch_size: int = 1
    num_workers: int = 8
    data_seed: int = 1750701990
    drop_last: bool = True
    prefetch_factor: int = 16
    p_horizon_flip: float = 0.5
    num_samples_ratio: float = 1.0
    critic_resolution: int = 384
    critic_identity_resolution: int = 384
    do_nothing_prob: float = 0.0
    identity_mode: bool = False

    edit_data_path: str = None
    edit_data_root_dir: str = None
    gedit_bench_data_path: str = None
    laion_aesthetics: str = None

    combined_edit_customization: bool = False
    critic_model_name: str = ''
    critic_identity_model_name: str = ''
    edit_type_probs: dict = None  # e.g., {"style": 0.15, "insertion": 0.1, ...}

    bucketize: bool = False
    resolution_buckets: list = None  # list of [H, W] pairs; None = use DEFAULT_RESOLUTION_BUCKETS



class InstructDataModule(ConfigurableModule[InstructDataModuleParams]):
    @classmethod
    def get_default_params(cls) -> InstructDataModuleParams:
        return InstructDataModuleParams()

    def __init__(self, params: InstructDataModuleParams, data_seed: int = 52, data_process_group=None):
        self.params = deepcopy(params)
        # Sync the data seed across the data group.
        if dist.is_initialized():
            data_seed_tensor: torch.Tensor = torch.tensor(data_seed, dtype=torch.int64).cuda()
            dist.all_reduce(data_seed_tensor, op=ReduceOp.MIN, group=data_process_group)
            data_seed = int(data_seed_tensor.cpu())

        self.params.data_seed = data_seed
        print(f"rank {dist.get_rank()} set data seed to: {data_seed}")

        # Initialize the dataset
        datasets = []
        if self.params.edit_data_path is not None:
            datasets.append(EditDataset(
                self.params.edit_data_path,
                root_dir=self.params.edit_data_root_dir,
                resolution=self.params.resolution,
                num_samples_ratio=self.params.num_samples_ratio,
                bucketize=self.params.bucketize,
                resolution_buckets=self.params.resolution_buckets,
            ))
        if self.params.gedit_bench_data_path is not None:
            datasets.append(GEditBenchDataset(
                resolution=self.params.resolution,
                bucketize=self.params.bucketize,
                resolution_buckets=self.params.resolution_buckets,
            ))
        if self.params.laion_aesthetics is not None:
            datasets.append(LaionAesthetics(self.params.laion_aesthetics, resolution=self.params.resolution))

        if len(datasets) == 0:
            raise ValueError("No dataset specified")

        self.datasets = {
            split: CriticCombinedDataset(
                datasets,
                critic_resolution=self.params.critic_resolution,
                critic_identity_resolution=self.params.critic_identity_resolution,
                critic_model_name=self.params.critic_model_name,
                critic_identity_model_name=self.params.critic_identity_model_name,
                do_nothing_prob=self.params.do_nothing_prob,
                combined_edit_customization=self.params.combined_edit_customization,
                val=split == "val",
                edit_type_probs=self.params.edit_type_probs,
            )
            for split in ["train", "val"]
        }

        print(f"  Total training samples: {len(self.datasets['train'])}")
        print(f"  Total validation samples: {len(self.datasets['val'])}")

        # Set up data loader parameters
        self.data_world_size = dist.get_world_size() if data_process_group is None else data_process_group.size()
        self.data_rank = dist.get_rank() if data_process_group is None else data_process_group.rank()
        print(f"rank {self.data_rank} world size {self.data_world_size}")

    def train_dataloader(self) -> Any:
        sampler = None
        dataloader_shuffle = None
        dataset = self.datasets["train"]

        if self.params.bucketize and getattr(dataset, 'bucket_ids', None) is not None:
            from data.samplers import DistributedBucketSampler
            logger.info("Using DistributedBucketSampler for train dataloader")
            sampler = DistributedBucketSampler(
                dataset,
                samples_per_gpu=self.params.batch_size,
                num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
                rank=dist.get_rank() if dist.is_initialized() else 0,
                shuffle=True,
                seed=self.params.data_seed,
            )
        elif dist.is_initialized():
            logger.info("Using DistributedSampler for train dataloader")
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                seed=self.params.data_seed,
                shuffle=True,
            )
        else:
            logger.info("Using regular DataLoader for train dataloader")
            dataloader_shuffle = True

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            collate_fn=self._collate_fn,
            shuffle=dataloader_shuffle if sampler is None else False,
            pin_memory=True,
            prefetch_factor=self.params.prefetch_factor,
            persistent_workers=True,
            drop_last=self.params.drop_last,
        )
        return dataloader


    def val_dataloader(self) -> Any:
        sampler = None
        dataloader_shuffle = None
        if dist.is_initialized():
            logger.info("Using DistributedSampler for val dataloader")
            sampler = DistributedSampler(
                self.datasets["val"],
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                seed=self.params.data_seed,
                shuffle=False,
            )
        else:
            logger.info("Using regular DataLoader for val dataloader")
            dataloader_shuffle = True

        dataloader = DataLoader(
            self.datasets["val"],
            sampler=sampler,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            collate_fn=self._collate_fn,
            shuffle=dataloader_shuffle,
            pin_memory=True,
            prefetch_factor=self.params.prefetch_factor,
            persistent_workers=True,
            drop_last=self.params.drop_last,
        )
        return dataloader

    def _collate_fn(self, batch):
        """Optimized custom collate function to handle the batch properly."""
        batch_size = len(batch)

        images = []

        prompts = []
        edited_prompts = []
        negative_prompts = []
        task_types = []
        sub_task_types = []

        pad_token_id = 2  # InternVL pad token (</s>)

        if 'input_ids' in batch[0]:
            input_ids = torch.nn.utils.rnn.pad_sequence([item['input_ids'][0] for item in batch], batch_first=True, padding_value=pad_token_id, padding_side='right')
            labels = torch.nn.utils.rnn.pad_sequence([item['labels'][0] for item in batch], batch_first=True, padding_value=-100, padding_side='right')
            input_ids_identity = torch.nn.utils.rnn.pad_sequence([item['input_ids_identity'][0] for item in batch], batch_first=True, padding_value=pad_token_id, padding_side='right')
            labels_identity = torch.nn.utils.rnn.pad_sequence([item['labels_identity'][0] for item in batch], batch_first=True, padding_value=-100, padding_side='right')

        for _, item in enumerate(batch):
            images.append(item['images'][1]) # Target image and reference image are the same.
            if self.params.identity_mode:
                prompts.append(item.get('original_prompts'))
            else:
                prompts.append(item['prompts'])

            edited_prompts.append(item['edited_prompts'])
            negative_prompts.append(item.get('negative_prompts', ''))
            if 'task_type' in item:
                task_types.append(item['task_type'])
            if 'sub_task_type' in item:
                sub_task_types.append(item['sub_task_type'])


        collated_batch = {
            'images': torch.stack(images).unsqueeze(2),
            'prompts': prompts,
            'edited_prompts': edited_prompts,
            'negative_prompts': negative_prompts,
            'hash_key': [item['hash_key'] for item in batch],
            'cfg': torch.stack([item['cfg'] for item in batch], 0)
        }

        if 'input_ids' in batch[0]:
            collated_batch['input_ids'] = input_ids
            collated_batch['labels'] = labels

        if 'input_ids_identity' in batch[0]:
            collated_batch['input_ids_identity'] = input_ids_identity
            collated_batch['labels_identity'] = labels_identity

        if len(task_types) > 0:
            collated_batch['task_type'] = task_types
        if len(sub_task_types) > 0:
            collated_batch['sub_task_type'] = sub_task_types
        if 'idx' in batch[0]:
            collated_batch['idx'] = torch.stack([item['idx'] for item in batch], 0)

        return collated_batch
