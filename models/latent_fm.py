from dataclasses import dataclass, fields
import itertools
import json
import os
import threading
from typing import Any, Literal, cast
from einops import rearrange
from collections.abc import Iterator

# from knapformer import SequenceBalancer
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.fsdp import fwd_only_mode
from utils.log import get_logger, human_readable_number
from utils.pack import pack_reduce
from utils.vis import generate_html_gallery
from utils_fm.noiser import NoiserProtocol
from utils_fm.sampler import FlowSampler, energy_preserve_cfg
from utils_fm.noiser import match_dims
from critic_models.critic import BaseCritic
from utils.log import TrackingLogger
logger = get_logger(__name__)


def _log_images_to_wandb(image_dict: dict, step: int):
    """Log images to wandb in a background thread to avoid blocking training."""
    import wandb
    for key, (np_image, caption) in image_dict.items():
        pil_image = Image.fromarray(np_image)
        wandb.log({key: wandb.Image(pil_image, caption=caption)}, step=step)


@dataclass
class FMDataContext:
    """
    Flow Matching data context for managing (txt, img) pairs during training.
    All fields are optional to support progressive data flow where each field
    is filled in by different operators (frozen and trainable).

    Attributes:
        prompts: List of text prompts as strings
        images: Raw image tensors (b, c, h, w)
        txt: Text embeddings (l1+l2+...+ln, d_txt)
        txt_datum_lens: Length of each text sequence (n,)

        img_clean: Clean image patches/latents (l1+l2+...+ln, d_img)
        img: Noised image patches/latents (l1+l2+...+ln, d_img)
        img_datum_lens: Length of each image sequence (n,)

        timesteps: Diffusion timesteps (n,)
        vec: Guidance vector for conditioning (n, d_vec)
    """

    """Filled by Dataloader"""
    prompts: list[str] | None = None
    edited_prompts: list[str] | None = None
    images: torch.Tensor | None = None  # (n, c, f, h, w)
    reference_images: torch.Tensor | None = None  # (n, c, f, h, w)
    idx: torch.Tensor | None = None  # (n,)
    task_type: list[str] | None = None  # (n,)
    cfg: torch.Tensor | None = None  # (n,)
    negative_prompts: list[str] | None = None  # (n,)

    """Input to Critic"""
    input_ids: torch.Tensor | None = None  # (n, l)
    labels: torch.Tensor | None = None  # (n, l)
    input_ids_identity: torch.Tensor | None = None  # (n, l)
    labels_identity: torch.Tensor | None = None  # (n, l)
    hash_key: list[str] | None = None  # (n,)
    sub_task_type: list[str] | None = None  # (n,)

    """Filled by FrozenOps"""
    """Frozen T5 Ops"""
    txt: torch.Tensor | None = None  # (l1+l2+...+ln, d_txt)
    txt_datum_lens: torch.Tensor | None = None  # (n,)
    """Frozen CLIP Ops"""
    vec: torch.Tensor | None = None  # (n, d_vec)
    """Frozen VAE Ops"""
    img_clean: torch.Tensor | None = None  # (b, c, f, h, w)
    img_datum_lens: torch.Tensor | None = None  # (n,)
    """Frozen Flow Noiser Ops"""
    img: torch.Tensor | None = None  # (l1+l2+...+ln, d_img)
    timesteps: torch.Tensor | None = None  # (n,)
    timestep_weights: torch.Tensor | None = None  # (n,)
    img_v_truth: torch.Tensor | None = None  # (l1+l2+...+ln, d_img)
    """Filled by TrainableOps"""
    img_v_pred: torch.Tensor | None = None  # (l1+l2+...+ln, d_img)
    loss_vec: torch.Tensor | None = None  # (n,)
    loss: torch.Tensor | None = None  # (1,)
    """Stats"""
    num_tokens: int | None = None

    def summarize(self) -> str:
        """Return a human-readable table summarizing tensor attributes.

        For every tensor attribute that is not ``None`` this method lists:
        1. Shape;
        2. Mean;
        3. Standard deviation;
        4. Minimum value;
        5. Maximum value.
        6. Dtype.

        The summary string is also logged for quick inspection.
        """

        headers = ("Tensor", "Shape", "Mean", "Std", "Min", "Max", "Dtype")
        line_sep = "-" * 105
        summary_lines: list[str] = [
            f"{headers[0]:20} | {headers[1]:30} | "
            f"{headers[2]:>12} | {headers[3]:>12} | {headers[4]:>12} | {headers[5]:>12} | "
            f"{headers[6]:>12}",
            line_sep,
        ]

        # Iterate over dataclass fields and collect stats for tensor attributes
        for field in fields(self):
            name = field.name
            value = getattr(self, name)

            # Skip non-tensor values that are not None (e.g., prompts list)
            if value is None or not torch.is_tensor(value):
                continue

            tensor = value
            # If the tensor is a DTensor, operate on its local shard for stats
            if isinstance(tensor, torch.distributed.tensor.DTensor):
                tensor = tensor._local_tensor

            shape_str = "x".join(map(str, tensor.shape))
            tensor_f = tensor.to(torch.float32)
            mean_v = tensor_f.mean().item()
            std_v = tensor_f.std().item()
            min_v = tensor_f.min().item()
            max_v = tensor_f.max().item()
            dtype_str = str(tensor.dtype).replace("torch.", "")

            summary_lines.append(
                f"{name:20} | {shape_str:30} | {mean_v:12.4f} | {std_v:12.4f} | {min_v:12.4f} | {max_v:12.4f} | {dtype_str:12}"
            )

        summary_str = "\n".join(summary_lines)
        logger.info("\n" + summary_str)
        return summary_str


@dataclass
class LatentFM:
    """
    Dataclass for managing all components in a latent flow matching model.

    Attributes:
        text_encoder: Text encoder for processing text prompts (any nn.Module implementation)
        clip_encoder: CLIP encoder for visual-textual understanding (any nn.Module implementation)
        vae: Variational autoencoder for encoding/decoding images

        denoiser: Main denoiser transformer model (supports any nn.Module implementation)
        ema_denoiser: EMA (Exponential Moving Average) version of the denoiser for inference
        flow_noiser: FlowNoiser for applying noise to clean data during training
        time_sampler: TimeSampler for sampling timesteps
        time_warper: TimeWarper for adaptive timestep scheduling based on sequence length
        time_weighter: TimeWeighter for loss weighting based on timestep distribution
        dit_balancer: SequenceBalancer for managing sequence lengths in batch processing
    """

    text_encoder: nn.Module | None = None
    clip_encoder: nn.Module | None = None
    vae: nn.Module | None = None
    denoiser: nn.Module | None = None
    ema_denoiser: nn.Module | None = None
    flow_noiser: nn.Module | None = None
    time_sampler: nn.Module | None = None
    time_warper: nn.Module | None = None
    time_weighter: nn.Module | None = None
    # dit_balancer: SequenceBalancer | None = None

    def summarize(self) -> str:
        """Return a human-readable table of parameter counts for the main sub-modules.

        For each available component (``text_encoder``, ``clip_encoder``, ``vae``,
        ``denoiser``, ``ema_denoiser``) the function lists:
        1. total parameters
        2. trainable parameters (``requires_grad=True``)
        3. frozen parameters (``requires_grad=False``)

        The summary is returned as a string and also printed to stdout so that
        users can quickly inspect it.
        """

        def _count_params(module: nn.Module) -> tuple[int, int, int, int, str, str]:
            """Return (global_total, global_trainable, local_total, local_trainable, device, dtype).

            *global_* counts refer to the full parameter sizes (``p.numel()``).
            *local_* counts consider only the local shards when ``p`` is a
            ``DTensor`` produced by FSDP2; otherwise they equal the global counts.
            """
            first_param = next(module.parameters())
            device_str = first_param.device.type
            dtype_str = str(first_param.dtype).replace("torch.", "")

            global_total = 0
            global_trainable = 0
            local_total = 0
            local_trainable = 0

            for p in module.parameters():
                # Global counts (always available)
                numel_global = p.numel()
                global_total += numel_global
                if p.requires_grad:
                    global_trainable += numel_global

                # Local counts (handle DTensor)
                if isinstance(p, torch.distributed.tensor.DTensor):
                    local_view = p._local_tensor  # Tensor representing the local shard
                    numel_local = local_view.numel()
                else:
                    numel_local = numel_global
                local_total += numel_local
                if p.requires_grad:
                    local_trainable += numel_local

            return global_total, global_trainable, local_total, local_trainable, device_str, dtype_str

        headers = ("Module", "Global", "Trainable", "Frozen", "Local", "Device", "Dtype")
        line_sep = "-" * 105
        summary_lines: list[str] = [
            f"{headers[0]:20} | {headers[1]:>12} | "
            f"{headers[2]:>12} | {headers[3]:>12} | "
            f"{headers[4]:>12} | {headers[5]:>12} | "
            f"{headers[6]:>12}",
            line_sep,
        ]

        modules_to_check: list[tuple[str, nn.Module | None]] = [
            ("text_encoder", self.text_encoder),
            ("clip_encoder", self.clip_encoder),
            ("vae", self.vae),
            ("denoiser", self.denoiser),
        ]
        if self.ema_denoiser is not None:
            modules_to_check.append(("ema_denoiser", self.ema_denoiser))

        for name, module in modules_to_check:
            if module is None:
                continue
            g_total, g_train, l_total, _, device_str, dtype_str = _count_params(module)
            g_frozen = g_total - g_train

            summary_lines.append(
                f"{name:20} | "
                f"{human_readable_number(g_total):>12} | "
                f"{human_readable_number(g_train):>12} | "
                f"{human_readable_number(g_frozen):>12} | "
                f"{human_readable_number(l_total):>12} | "
                f"{device_str:>12} | "
                f"{dtype_str:>12}"
            )

        summary_str = "\n".join(summary_lines)

        # Print for convenience
        logger.info("\n" + summary_str)
        return summary_str


@dataclass
class FrozenOps:
    lfm: LatentFM
    _cached_uncond: dict | None = None  # cached negative prompt encoding

    @torch.no_grad()
    def __call__(self, data_batch: FMDataContext, txt_drop_prob: float = 0.0, mode: str = 'student') -> FMDataContext:
        """Drop raw text with a certain probability in i.i.d. manner (in-place)."""
        assert data_batch.prompts is not None, "prompts must be provided"

        device = torch.device("cuda")

        if txt_drop_prob > 0.0:
            drop_probs = torch.rand(len(data_batch.prompts)).tolist()
            for i, drop_prob in enumerate(drop_probs):
                if drop_prob < txt_drop_prob:
                    data_batch.prompts[i] = ""

        """Encode text into embeddings (in-place)."""
        vec_embed = None
        vec_embed_original = None
        uncond_vec_embed = None
        data_batch.vec = None
        data_batch.original_vec = None
        data_batch.uncond_vec = None
        # negative_prompts = data_batch.negative_prompts if data_batch.negative_prompts is not None else 
        negative_prompts = ["low quality, artifacts"]*len(data_batch.prompts)

        # Check if negative prompts are uniform (same string for all samples) — cacheable
        neg_is_uniform = len(set(negative_prompts)) == 1
        use_cached_uncond = (
            neg_is_uniform
            and self._cached_uncond is not None
            and self._cached_uncond['batch_size'] == len(data_batch.prompts)
            and self._cached_uncond['neg_prompt'] == negative_prompts[0]
        )

        if use_cached_uncond:
            # Only encode prompts + edited_prompts (skip negative prompts)
            all_prompts = data_batch.prompts + data_batch.edited_prompts
        else:
            all_prompts = data_batch.prompts + data_batch.edited_prompts + negative_prompts

        if self.lfm.clip_encoder is not None:
            # clip is (b, d)
            with fwd_only_mode(self.lfm.clip_encoder):
                all_embeds, all_vecs = self.lfm.clip_encoder(all_prompts)
                # split it now
                if vec_embed is not None:
                    vec_embed = all_embeds[:len(data_batch.prompts)]
                    vec_embed_original = all_embeds[len(data_batch.prompts):len(data_batch.prompts) + len(data_batch.edited_prompts)]
                    uncond_vec_embed = all_embeds[len(data_batch.prompts) + len(data_batch.edited_prompts):]
                data_batch.vec = all_vecs[:len(data_batch.prompts)]
                data_batch.original_vec = all_vecs[len(data_batch.prompts):len(data_batch.prompts) + len(data_batch.edited_prompts)]
                if use_cached_uncond:
                    data_batch.uncond_vec = self._cached_uncond['uncond_vec']
                else:
                    data_batch.uncond_vec = all_vecs[len(data_batch.prompts) + len(data_batch.edited_prompts):]

        if self.lfm.text_encoder is not None:
            with fwd_only_mode(self.lfm.text_encoder):

                all_txts, all_txt_datum_lens, all_txt_embedding_mask = self.lfm.text_encoder(all_prompts)
                # split it now
                n_prompts = len(data_batch.prompts)
                n_edited = len(data_batch.edited_prompts)
                data_batch.txt = all_txts[:n_prompts]
                data_batch.txt_datum_lens = all_txt_datum_lens[:n_prompts]
                data_batch.txt_embedding_mask = all_txt_embedding_mask[:n_prompts]
                data_batch.original_txt = all_txts[n_prompts:n_prompts + n_edited]
                data_batch.original_txt_datum_lens = all_txt_datum_lens[n_prompts:n_prompts + n_edited]
                data_batch.original_txt_embedding_mask = all_txt_embedding_mask[n_prompts:n_prompts + n_edited]

                if use_cached_uncond:
                    data_batch.uncond_txt = self._cached_uncond['uncond_txt']
                    data_batch.uncond_txt_datum_lens = self._cached_uncond['uncond_txt_datum_lens']
                    data_batch.uncond_txt_embedding_mask = self._cached_uncond['uncond_txt_embedding_mask']
                else:
                    data_batch.uncond_txt = all_txts[n_prompts + n_edited:]
                    data_batch.uncond_txt_datum_lens = all_txt_datum_lens[n_prompts + n_edited:]
                    data_batch.uncond_txt_embedding_mask = all_txt_embedding_mask[n_prompts + n_edited:]

                    # Cache for future steps if negative prompts are uniform
                    if neg_is_uniform:
                        self._cached_uncond = {
                            'neg_prompt': negative_prompts[0],
                            'batch_size': len(data_batch.prompts),
                            'uncond_txt': data_batch.uncond_txt.clone(),
                            'uncond_txt_datum_lens': data_batch.uncond_txt_datum_lens.clone(),
                            'uncond_txt_embedding_mask': data_batch.uncond_txt_embedding_mask.clone(),
                            'uncond_vec': data_batch.uncond_vec.clone() if data_batch.uncond_vec is not None else None,
                        }

            # Note: removed torch.cuda.empty_cache() here — it forces a full CUDA sync
            # that stalls the GPU pipeline. Memory is freed naturally by PyTorch's allocator.

        """Encode image into embeddings (in-place)."""
        if data_batch.images is not None and self.lfm.vae is not None:
            with fwd_only_mode(self.lfm.vae):
                latents = self.lfm.vae.encode(data_batch.images)

            data_batch.img_clean = latents
            b, c, f, h, w = latents.shape
            data_batch.img_datum_lens = torch.full((b,), f * (h // 2) * (w // 2), device=latents.device, dtype=torch.int32)

        if data_batch.reference_images is not None and self.lfm.vae is not None:
            data_batch.reference_img_clean = data_batch.img_clean # only works because we basically have the same image as input and reference
            

        """Add noise to the image (in-place)."""
        if (
            data_batch.img_clean is not None
            and data_batch.img_datum_lens is not None
            and self.lfm.time_sampler is not None
            and self.lfm.time_warper is not None
            and self.lfm.time_weighter is not None
            and self.lfm.flow_noiser is not None
        ):
            timesteps = self.lfm.time_sampler((len(data_batch.prompts),), device=device, multi_step=False)
            timesteps = self.lfm.time_warper(timesteps, data_batch.img_datum_lens)
            data_batch.timesteps = timesteps

            data_batch.timestep_weights = self.lfm.time_weighter(data_batch.timesteps)

            data_batch.img, data_batch.img_v_truth = self.lfm.flow_noiser(
                data_batch.img_clean, data_batch.img_datum_lens, data_batch.timesteps
            )

            if data_batch.reference_images is not None:
                data_batch.img = torch.cat([data_batch.reference_img_clean, data_batch.img], dim=-2) # CHECK THIS LATER
                data_batch.img_datum_lens = data_batch.img_datum_lens * 2

        if data_batch.txt is not None and data_batch.img is not None:
            data_batch.num_tokens = data_batch.txt.shape[0] + data_batch.img.shape[0]

        return data_batch


@dataclass
class TrainableOps:
    lfm: LatentFM
    global_batch_size: int | None = None
    image_log_freq: int = 100

    def __call__(self, data_batch: FMDataContext, global_step: int) -> FMDataContext:
        """Pass through the denoiser with correct interface"""
        assert data_batch.txt is not None, "txt must be provided"
        assert data_batch.txt_datum_lens is not None, "txt_datum_lens must be provided"

        assert data_batch.img is not None, "img must be provided"
        assert data_batch.img_datum_lens is not None, "img_datum_lens must be provided"

        assert data_batch.timesteps is not None, "timesteps must be provided"
        assert self.lfm.denoiser is not None, "denoiser must be provided"

        data_batch.img_v_pred = self.lfm.denoiser(
            txt=data_batch.txt,
            txt_datum_lens=data_batch.txt_datum_lens,
            img=data_batch.img,
            t=data_batch.timesteps,
            txt_embedding_mask=data_batch.txt_embedding_mask,
        )
        bs = len(data_batch.img_datum_lens)
        if data_batch.reference_images is not None:
            h = data_batch.img.shape[-2]
            data_batch.img = data_batch.img[:, :, :, h // 2:, :]
            data_batch.img_datum_lens = data_batch.img_datum_lens // 2

        assert data_batch.img_v_pred is not None, "img_v_pred must be provided"
        assert data_batch.img_v_truth is not None, "img_v_truth must be provided"
        assert data_batch.img_datum_lens is not None, "img_datum_lens must be provided"
        assert data_batch.timestep_weights is not None, "timestep_weights must be provided"

        mse_loss = (data_batch.img_v_pred.float() - data_batch.img_v_truth.float()) ** 2
        loss_vec = mse_loss.mean(dim=[1, 2, 3, 4])
        loss_vec = loss_vec * data_batch.timestep_weights

        if self.global_batch_size is None:
            device = torch.device("cuda")
            bs = torch.tensor(len(data_batch.img_datum_lens), device=device)
            dist.all_reduce(bs, op=dist.ReduceOp.SUM)
            self.global_batch_size = int(bs.item())
        data_batch.loss = loss_vec.sum() * dist.get_world_size() / self.global_batch_size
        data_batch.loss_vec = loss_vec

        if global_step % self.image_log_freq == 0 and dist.get_rank() == 0:
            import wandb
            # log images 
            with torch.no_grad():
                x_0_pred = data_batch.img - match_dims(data_batch.timesteps, data_batch.img.shape) * data_batch.img_v_pred
                x_0_pred_decoded = self.lfm.vae.decode(x_0_pred[:1])
                x_0_pred_decoded = torch.clamp(x_0_pred_decoded * 0.5+0.5, min=0.0, max=1.0)
                output_image = (x_0_pred_decoded.detach()[0,:,0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                output_image = Image.fromarray(output_image)
                images = wandb.Image(output_image, caption= data_batch.prompts[0])
                wandb.log({"generated_images x0":images}, step=global_step)

                x_0_decoded = self.lfm.vae.decode(data_batch.img[:1])
                x_0_decoded = torch.clamp(x_0_decoded * 0.5+0.5, min=0.0, max=1.0)
                gt_images = (x_0_decoded.detach()[0,:,0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                gt_images = Image.fromarray(gt_images)
                gt_images = wandb.Image(gt_images, caption=data_batch.prompts[0])
                wandb.log({"gt_noisy_images":gt_images}, step=global_step)

                if data_batch.reference_images is not None:
                    x_ref_decoded = data_batch.reference_images.detach()*0.5+0.5
                    ref_images = (x_ref_decoded.detach()[0,:,0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                    ref_images = Image.fromarray(ref_images)
                    ref_images = wandb.Image(ref_images, caption=data_batch.prompts[0])
                    wandb.log({"ref_images":ref_images}, step=global_step)

        return data_batch


@dataclass
class TrainableOpsDMDAux:
    lfm: LatentFM
    aux_lfm: LatentFM
    global_batch_size: int | None = None
    image_log_freq: int = 100
    start_multistep_sampling_step: bool = False

    @torch.no_grad()
    def get_student_image(self, data_batch: FMDataContext, global_step: int) -> torch.Tensor:
        """Pass through the denoiser with correct interface"""
        assert data_batch.txt is not None, "txt must be provided"
        assert data_batch.txt_datum_lens is not None, "txt_datum_lens must be provided"

        assert data_batch.img is not None, "img must be provided"
        assert data_batch.img_datum_lens is not None, "img_datum_lens must be provided"

        assert data_batch.timesteps is not None, "timesteps must be provided"
        assert self.lfm.denoiser is not None, "denoiser must be provided"

        device = data_batch.img.device
        generator = torch.Generator(device=device)
        generator.manual_seed(global_step)
        sample_method = "ddim"

        num_step_backward_unroll = torch.randint(1, len(self.lfm.time_sampler.dmd_time_steps)+1, (1,), device=device, generator=generator)

        # randomly sample the intermediate timesteps from the dmd_time_steps
        timesteps = self.lfm.time_sampler.dmd_time_steps[:1]
        if num_step_backward_unroll > 1 and global_step > self.start_multistep_sampling_step:
            timesteps += self.lfm.time_sampler.dmd_time_steps[num_step_backward_unroll-1:num_step_backward_unroll]
        timesteps += [0.]

        h = data_batch.img.shape[-2]
        x = data_batch.img[:, :, :, h // 2:, :]
        with fwd_only_mode(self.lfm.denoiser):
            for t, next_t in zip(timesteps[:-1], timesteps[1:], strict=True):
                t = torch.ones(data_batch.timesteps.shape, dtype=torch.float32, device=device) * t
                next_t = torch.ones(data_batch.timesteps.shape, dtype=torch.float32, device=device) * next_t
                img = torch.cat([data_batch.reference_img_clean.detach(), x], dim=-2).contiguous()
                pred_v = self.lfm.denoiser(
                    txt=data_batch.txt,
                    txt_datum_lens=data_batch.txt_datum_lens,
                    img=img,
                    t=t,
                    txt_embedding_mask=data_batch.txt_embedding_mask,
                )
                x_0 = x - match_dims(t, pred_v.shape) * pred_v
                x = x + pred_v * match_dims(next_t - t, pred_v.shape)
                
                if sample_method == "ddim":
                    x, _ = self.lfm.flow_noiser(x_0, data_batch.img_datum_lens // 2, next_t)
        
        return x
    
    def __call__(self, data_batch: FMDataContext, global_step: int, log_images: bool) -> FMDataContext:
        x_0_pred = self.get_student_image(data_batch, global_step)

        timesteps = self.aux_lfm.time_sampler((len(data_batch.prompts),), device=data_batch.img.device)
        data_batch.timesteps = timesteps

        data_batch.timestep_weights = self.aux_lfm.time_weighter(data_batch.timesteps)

        data_batch.img, data_batch.img_v_truth = self.aux_lfm.flow_noiser(
            x_0_pred, data_batch.img_datum_lens // 2, data_batch.timesteps
        )

        data_batch.img_v_pred = self.aux_lfm.denoiser(
            txt=data_batch.original_txt,
            txt_datum_lens=data_batch.original_txt_datum_lens,
            img=data_batch.img,
            t=data_batch.timesteps,
            txt_embedding_mask=data_batch.original_txt_embedding_mask,
        )

        mse_loss = (data_batch.img_v_pred.float() - data_batch.img_v_truth.float()) ** 2
        loss_vec = mse_loss.mean(dim=[1, 2, 3, 4])
        loss_vec = loss_vec * data_batch.timestep_weights

        data_batch.loss = loss_vec.mean()
        data_batch.loss_vec = loss_vec

        if self.global_batch_size is None:
            device = torch.device("cuda")
            bs = torch.tensor(len(data_batch.img_datum_lens), device=device)
            dist.all_reduce(bs, op=dist.ReduceOp.SUM)
            self.global_batch_size = int(bs.item())
        
        x_0_pred_decoded = None
        if global_step % self.image_log_freq == 0 and dist.get_rank() == 0 and log_images:
            # Prepare images on GPU/CPU synchronously, then log to wandb in background thread
            with torch.no_grad():

                x_0_pred_decoded = self.lfm.vae.decode(x_0_pred)
                x_0_pred_decoded = torch.clamp(x_0_pred_decoded * 0.5 + 0.5, min=0.0, max=1.0)
                output_image_np = (x_0_pred_decoded.detach()[0,:,0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)

                x_ref_decoded = data_batch.reference_images.detach()*0.5+0.5
                ref_image_np = (x_ref_decoded.detach()[0,:,0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)

                del x_0_pred_decoded, x_ref_decoded

            image_dict = {
                "generated_images during aux x0": (output_image_np, data_batch.prompts[0]),
                "ref_images during aux": (ref_image_np, data_batch.edited_prompts[0]),
            }
            threading.Thread(target=_log_images_to_wandb, args=(image_dict, global_step), daemon=True).start()

        return data_batch


@dataclass
class TrainableOpsDMD:
    lfm: LatentFM
    aux_lfm: LatentFM
    teacher_lfm: LatentFM
    critic: BaseCritic | None = None
    global_batch_size: int | None = None
    real_guidance_scale: float = 1.0
    critic_loss_weight: float = 0.05
    dmd_loss_weight: float = 0.5
    start_multistep_sampling_step: int = 4000
    image_log_freq: int = 100
    mse_loss_fn: torch.nn.Module = torch.nn.MSELoss(reduction="none")

    def __call__(self, data_batch: FMDataContext, global_step: int) -> FMDataContext:
        """Pass through the denoiser with correct interface"""
        assert data_batch.txt is not None, "txt must be provided"
        assert data_batch.txt_datum_lens is not None, "txt_datum_lens must be provided"

        assert data_batch.img is not None, "img must be provided"
        assert data_batch.img_datum_lens is not None, "img_datum_lens must be provided"

        assert data_batch.timesteps is not None, "timesteps must be provided"
        assert self.lfm.denoiser is not None, "denoiser must be provided"

        # Sample timesteps from the denoising step list
        sample_method = "ddim"
        device = data_batch.img.device
        generator = torch.Generator(device=device)
        generator.manual_seed(global_step)

        num_step_backward_unroll = torch.randint(1, len(self.lfm.time_sampler.dmd_time_steps)+1, (1,), device=device, generator=generator)

        # randomly sample the intermediate timesteps from the dmd_time_steps
        timesteps = self.lfm.time_sampler.dmd_time_steps[:1]
        if num_step_backward_unroll > 1 and global_step > self.start_multistep_sampling_step:
            timesteps += self.lfm.time_sampler.dmd_time_steps[num_step_backward_unroll-1:num_step_backward_unroll]
        timesteps += [0.]

        h = data_batch.img.shape[-2]
        x = data_batch.img[:, :, :, h // 2:, :]
        for counter, (t, next_t) in enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)):
            t = torch.ones(data_batch.timesteps.shape, dtype=torch.float32, device=device) * t
            next_t = torch.ones(data_batch.timesteps.shape, dtype=torch.float32, device=device) * next_t
            img = torch.cat([data_batch.reference_img_clean.detach(), x], dim=-2).contiguous()
            pred_v = self.lfm.denoiser(
                txt=data_batch.txt,
                txt_datum_lens=data_batch.txt_datum_lens,
                img=img,
                t=t,
                txt_embedding_mask=data_batch.txt_embedding_mask,
            )
            x_0 = x - match_dims(t, pred_v.shape) * pred_v
            x = x + pred_v * match_dims(next_t - t, pred_v.shape)
            
            if sample_method == "ddim":
                x, _ = self.lfm.flow_noiser(x_0, data_batch.img_datum_lens // 2, next_t)
            
        
        x_0_pred = x
        generator_data_dict = {
                "x_0_gen": x_0_pred,
                "batch": data_batch,
            }
        dmd_loss = self.compute_distribution_matching_loss(generator_data_dict)
        dmd_loss = dmd_loss.mean(dim=[1, 2, 3, 4])
        data_batch.dmd_loss_vec = dmd_loss
        dmd_loss = self.dmd_loss_weight * dmd_loss

        # get the do_nothing regularization loss
        regularization_mask = torch.tensor([t == 'do nothing' for t in data_batch.task_type], dtype=torch.bool, device=x_0_pred.device)
        mse_loss = self.mse_loss_fn(x_0_pred, data_batch.reference_img_clean.detach())
        mse_loss = mse_loss.mean(dim=tuple(range(1, mse_loss.ndim)))
        mse_loss[~regularization_mask] = mse_loss[~regularization_mask] * 0.0
        data_batch.mse_loss_vec = mse_loss

        x_0_pred_decoded = None
        output_text_ = [''] * len(data_batch.prompts)
        critic_loss = torch.zeros_like(dmd_loss)
        if self.critic is not None:
            x_0_pred_decoded = self.lfm.vae.decode(x_0_pred)

            pixel_values_edit = self.critic.image_prep(x_0_pred_decoded)
            pixel_values_ref = self.critic.image_prep(data_batch.reference_images)
            temperature = min(1.0, 0.1 + 0.9 * global_step / self.critic.max_steps_temp)

            # Extract ViT features ONCE; for both edit-verification and identity-evaluation 
            with torch.no_grad():
                vit_embeds_ref = self.critic.extract_vit_features(pixel_values_ref.detach().contiguous())
            vit_embeds_ref = vit_embeds_ref.detach()
            vit_embeds_edit = self.critic.extract_vit_features(pixel_values_edit.contiguous())

            # Determine if score eval uses single image or ref+edit pair.
            single_image_eval = (data_batch.input_ids[0] == self.critic.image_token_id).sum().item() == 1
            edit_vit = vit_embeds_edit if single_image_eval else torch.cat([vit_embeds_ref, vit_embeds_edit], dim=0)

            identity_vit = torch.cat([vit_embeds_ref, vit_embeds_edit], dim=0)

            # Batch both LLM calls into a single forward pass to amortize FSDP all-gather
            ids_edit = data_batch.input_ids        # [1, N_s]
            ids_identity = data_batch.input_ids_identity  # [1, N_i]
            N_s, N_i = ids_edit.shape[1], ids_identity.shape[1]
            max_len = max(N_s, N_i)
            pad_id = self.critic.pad_token_id if self.critic.pad_token_id is not None else 0

            # Right-pad to same length
            ids_edit_pad = F.pad(ids_edit, (0, max_len - N_s), value=pad_id)
            ids_identity_pad = F.pad(ids_identity, (0, max_len - N_i), value=pad_id)
            batched_input_ids = torch.cat([ids_edit_pad, ids_identity_pad], dim=0)  # [2, max_len]

            # Attention mask (1 for real tokens, 0 for padding)
            attn_edit = F.pad(torch.ones(1, N_s, device=ids_edit.device, dtype=torch.long), (0, max_len - N_s), value=0)
            attn_identity = F.pad(torch.ones(1, N_i, device=ids_identity.device, dtype=torch.long), (0, max_len - N_i), value=0)
            batched_attn_mask = torch.cat([attn_edit, attn_identity], dim=0)  # [2, max_len]

            # Pad labels similarly
            labels_edit_pad = F.pad(data_batch.labels, (0, max_len - N_s), value=-100)
            labels_identity_pad = F.pad(data_batch.labels_identity, (0, max_len - N_i), value=-100)
            batched_labels = torch.cat([labels_edit_pad, labels_identity_pad], dim=0)  # [2, max_len]

            # Concat ViT embeds
            batched_vit = torch.cat([edit_vit, identity_vit], dim=0)

            # Build inputs dict for InternVL critic
            batched_flags = torch.ones(batched_vit.shape[0], device=batched_vit.device, dtype=torch.long)
            inputs = {
                'input_ids': batched_input_ids,
                'labels': batched_labels,
                'image_flags': batched_flags,
            }

            # Single batched critic call (1 FSDP all-gather instead of 2)
            batched_loss, batched_text, batched_score_logits = self.critic(
                inputs, temperature=temperature, vit_embeds=batched_vit,
                attention_mask=batched_attn_mask)

            # Split results: batch item 0 = edit eval, batch item 1 = identity eval
            critic_loss = batched_loss[0:1]
            critic_loss_identity = batched_loss[1:2]
            output_text_ = [batched_text[0]]
            output_text_identity = [batched_text[1]]
            score_logits_edit_eval = batched_score_logits[0:1]
            score_logits_identity_eval = batched_score_logits[1:2]

            critic_loss = (critic_loss + critic_loss_identity) / 2.
            output_text_ = [output_text1 + ';' + output_text2 for output_text1, output_text2 in zip(output_text_, output_text_identity)]

            data_batch.critic_loss_vec = critic_loss
            critic_loss = self.critic_loss_weight * critic_loss

            data_batch.score_logits_edit_eval = (score_logits_edit_eval.softmax(-1)[:,-1]).detach()
            data_batch.score_logits_identity_eval = (score_logits_identity_eval.softmax(-1)[:,-1]).detach()

        if self.global_batch_size is None:
            device = torch.device("cuda")
            bs = torch.tensor(len(data_batch.img_datum_lens), device=device)
            dist.all_reduce(bs, op=dist.ReduceOp.SUM)
            self.global_batch_size = int(bs.item())

        data_batch.loss_vec = critic_loss + mse_loss  + dmd_loss
        data_batch.loss = data_batch.loss_vec.mean()


        if global_step % self.image_log_freq == 0 and dist.get_rank() == 0:
            # Prepare images on GPU, then log to wandb in background thread
            with torch.no_grad():
                if x_0_pred_decoded is None:
    
                    x_0_pred_decoded = self.lfm.vae.decode(x_0_pred)
                x_0_pred_decoded = torch.clamp(x_0_pred_decoded * 0.5 + 0.5, min=0.0, max=1.0)
                output_image_np = (x_0_pred_decoded.detach()[0,:,0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)

                x_0_decoded = data_batch.images.detach()*0.5+0.5
                gt_image_np = (x_0_decoded.detach()[0,:,0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)

                x_ref_decoded = data_batch.reference_images.detach()*0.5+0.5
                ref_image_np = (x_ref_decoded.detach()[0,:,0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)

                del x_0_pred_decoded, x_0_decoded, x_ref_decoded

            image_dict = {
                "generated_images x0": (output_image_np, output_text_[0] + data_batch.sub_task_type[0]),
                "gt_images": (gt_image_np, data_batch.prompts[0]),
                "ref_images": (ref_image_np, data_batch.edited_prompts[0]),
            }
            threading.Thread(target=_log_images_to_wandb, args=(image_dict, global_step), daemon=True).start()

        return data_batch

    def compute_distribution_matching_loss(self, generator_data_dict):
        """DMD loss."""
        latents = generator_data_dict["x_0_gen"]
        data_batch = generator_data_dict["batch"]
        original_latents = latents

        batch_size = latents.shape[0]
        bs = len(data_batch.img_datum_lens)

        with torch.no_grad():
            # Sample timesteps
            timesteps = self.teacher_lfm.time_sampler((len(data_batch.prompts),), device=data_batch.img.device)
            timesteps = self.teacher_lfm.time_warper(timesteps, data_batch.img_datum_lens // 2)

            # Add noise to the latents
            x_t, _ = self.lfm.flow_noiser(latents, data_batch.img_datum_lens // 2, timesteps)

            # Fake score: aux model (conditional, no CFG)
            velocity_model = VelocityModel(
                denoiser=self.aux_lfm.denoiser,
                txt=data_batch.original_txt,
                txt_datum_lens=data_batch.original_txt_datum_lens,
                cfg_scale=torch.ones_like(data_batch.cfg),
                txt_embedding_mask=data_batch.original_txt_embedding_mask,
            )
            v = velocity_model(x_t, data_batch.img_datum_lens // 2, timesteps)
            pred_fake_x0 = x_t - match_dims(timesteps, v.shape) * v

            # Real score: teacher model (with CFG) - conditional pass
            cond_velocity_model = VelocityModel(
                denoiser=self.teacher_lfm.denoiser,
                txt=data_batch.original_txt,
                txt_datum_lens=data_batch.original_txt_datum_lens,
                cfg_scale=1.0,
                txt_embedding_mask=data_batch.original_txt_embedding_mask,
            )
            v_cond = cond_velocity_model(x_t, data_batch.img_datum_lens // 2, timesteps)

            # Real score: teacher model (with CFG) - unconditional pass
            uncond_velocity_model = VelocityModel(
                denoiser=self.teacher_lfm.denoiser,
                txt=data_batch.uncond_txt,
                txt_datum_lens=data_batch.uncond_txt_datum_lens,
                cfg_scale=1.0,
                txt_embedding_mask=data_batch.uncond_txt_embedding_mask,
            )
            v_uncond = uncond_velocity_model(x_t, data_batch.img_datum_lens // 2, timesteps)

            # Combine with classifier-free guidance
            cfg_scale = data_batch.cfg if data_batch.cfg is not None else self.real_guidance_scale
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            pred_real_x0 = x_t - match_dims(timesteps, v.shape) * v

            # Compute gradients for DMD
            p_real = (latents - pred_real_x0)
            p_fake = (latents - pred_fake_x0)
            w = rearrange(torch.abs((latents.float() - pred_real_x0.float())) , "(b c) ... -> b c ...", b=bs).mean(dim=[1, 2], keepdim=True).clamp(min=1e-8)

            p_real = rearrange(p_real, "(b c) ... -> b c ...", b=bs)
            p_fake = rearrange(p_fake, "(b c) ... -> b c ...", b=bs)

            grad = (p_real - p_fake) / w
            grad = rearrange(torch.nan_to_num(grad), "b c ... -> (b c) ...", b=bs)

        loss = 0.5 * torch.nn.functional.mse_loss(original_latents, (original_latents-grad).detach(), reduction="none")
        return loss    


class VelocityModel:
    """Wrapper for velocity prediction models with support for classifier-free guidance.

    This class encapsulates a velocity prediction model (typically a transformer) along with
    text conditioning. For classifier-free guidance, users should provide concatenated
    positive and negative conditioning in the initialization parameters.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        txt: torch.Tensor,
        txt_datum_lens: torch.Tensor,
        txt_embedding_mask: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        reference_img_clean: torch.Tensor | None = None,
        energy_preserve_cfg: bool = False,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.txt = txt
        self.txt_datum_lens = txt_datum_lens
        self.cfg_scale = cfg_scale
        self.reference_img_clean = reference_img_clean
        self.energy_preserve_cfg = energy_preserve_cfg
        self.txt_embedding_mask = txt_embedding_mask

    def __call__(
        self,
        img: torch.Tensor,
        img_datum_lens: torch.Tensor,
        t: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        txt, txt_datum_lens, txt_embedding_mask = self.txt, self.txt_datum_lens, self.txt_embedding_mask

        if self.reference_img_clean is not None:
            img = torch.cat([self.reference_img_clean, img], dim=-2)

        # check if any of the cfg_scale is greater than 1.0
        if (isinstance(self.cfg_scale, float) and self.cfg_scale > 1.0) or (isinstance(self.cfg_scale, torch.Tensor) and torch.any((self.cfg_scale > 1.0)*1.0)):
            if len(txt_datum_lens) != 2 * len(img_datum_lens):
                raise ValueError(
                    f"For classifier-free guidance (cfg_scale > 1.0), txt_datum_lens must have "
                    f"2*n elements where n = len(img_datum_lens). Got txt_datum_lens length: "
                    f"{len(txt_datum_lens)}, expected: {2 * len(img_datum_lens)} (2 * {len(img_datum_lens)})"
                )

            # Duplicate img data for positive and negative conditioning
            img = torch.cat([img, img], dim=0)
            img_datum_lens = torch.cat([img_datum_lens, img_datum_lens], dim=0)
            t = torch.cat([t, t], dim=0)

        pred_v: torch.Tensor = self.denoiser(
            txt=txt,
            txt_datum_lens=txt_datum_lens,
            img=img,
            t=t,
            txt_embedding_mask=txt_embedding_mask,
        )

        img_v = pred_v

        if (isinstance(self.cfg_scale, float) and self.cfg_scale > 1.0) or (isinstance(self.cfg_scale, torch.Tensor) and torch.any((self.cfg_scale > 1.0)*1.0)):
            # Split results into conditional and unconditional components
            img_datum_lens = img_datum_lens.chunk(2, dim=0)[0]
            pos_img_v, neg_img_v = img_v.chunk(2, dim=0)

            if return_components:
                return pos_img_v, neg_img_v

            if self.energy_preserve_cfg:
                img_v = energy_preserve_cfg(pos_img_v, neg_img_v, img_datum_lens, self.cfg_scale)
            else:
                img_v = neg_img_v + self.cfg_scale * (pos_img_v - neg_img_v)

        return img_v


@dataclass
class InferenceTask:
    img_fhw: tuple[int, int, int]  # frame, height, width
    prompts: list[str]
    neg_prompts: list[str]
    cfg_scale: float
    num_steps: int
    eta: float
    seed: int
    output_names: list[str]
    guidance: float | None = None


def load_prompts_as_tasks(
    img_fhw: tuple[int, int, int],
    prompt_file: str,
    samples_per_prompt: int = 2,
    neg_prompt: str = "",
    cfg_scale: float = 5,
    num_steps: int = 50,
    eta: float = 1.0,
    file_ext: str = "jpg",
    per_gpu_bs: int = 16,
    guidance: float | None = None,
) -> list[InferenceTask]:
    prompts = []
    with open(prompt_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    # add dummy prompt so that it's divisible by dist.get_world_size()
    num_gpus = dist.get_world_size()
    while len(prompts) % (num_gpus * per_gpu_bs) != 0:
        prompts.append("DUMMY_PROMPT")

    tasks = []
    for seed in range(samples_per_prompt):
        for i in range(0, len(prompts), per_gpu_bs):
            tasks.append(
                InferenceTask(
                    img_fhw=img_fhw,
                    prompts=prompts[i : i + per_gpu_bs],
                    neg_prompts=[neg_prompt] * per_gpu_bs,
                    cfg_scale=cfg_scale,
                    num_steps=num_steps,
                    eta=eta,
                    seed=seed,
                    output_names=[f"p{k:06d}-s{seed:06d}.{file_ext}" for k in range(i, i + per_gpu_bs)],
                    guidance=guidance,
                )
            )

    # divide to each gpu
    num_tasks_per_gpu = len(tasks) // num_gpus
    gpu_rank = dist.get_rank()
    tasks = tasks[gpu_rank * num_tasks_per_gpu : (gpu_rank + 1) * num_tasks_per_gpu]
    return tasks


@dataclass
class InferenceOps:
    lfm: LatentFM
    train_dataloader: Any
    data_module: Any = None

    @torch.no_grad()
    # @torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    def __call__(
        self,
        output_dir: str,
        img_fhw: tuple[int, int, int],
        prompt_file: str,
        samples_per_prompt: int = 2,
        neg_prompt: str = "",
        cfg_scale: float = 5,
        num_steps: int = 50,
        eta: float = 1.0,
        file_ext: str = "jpg",
        per_gpu_bs: int = 16,
        use_ema: bool = True,
        guidance: float | None = None,
        sample_method: Literal["euler", "ddim"] = "ddim",
        min_timestep: float = 0.0,
        save_as_npz: bool = False,
        seed: int = 0,
        energy_preserve_cfg: bool = False,
        timesteps: list[float] = None,
        system_prompt: str = '',
    ) -> None:
        """
        Perform flow matching inference sampling.

        Args:
            save_as_npz: If True, saves all images as a uint8 npz file instead of individual image files.
                        Also saves metadata and sampling info as a JSON file.
        """
        # Select denoiser based on use_ema flag
        selected_denoiser = self.lfm.denoiser
        if use_ema and self.lfm.ema_denoiser is not None:
            if self.lfm.ema_denoiser is None:
                raise ValueError("EMA denoiser is not available, please use non-EMA denoiser")
            selected_denoiser = self.lfm.ema_denoiser
            logger.info("Using EMA denoiser for inference")

        assert selected_denoiser is not None, "denoiser must be provided"
        assert self.lfm.vae is not None, "vae must be provided"
        assert self.lfm.text_encoder is not None, "text_encoder must be provided"
        assert self.lfm.flow_noiser is not None, "flow_noiser must be provided"
        assert self.lfm.time_warper is not None, "time_warper must be provided"

        with fwd_only_mode(selected_denoiser):
            device = next(selected_denoiser.parameters()).device
            if timesteps is not None:
                timesteps = torch.tensor(timesteps, dtype=torch.float32, device=device)

            if self.train_dataloader is not None:
                tasks = []
                for batch in self.train_dataloader:
                    task = self.data_module.prepare_batch(batch, device)
                    tasks.append(task)
            else:
                tasks = load_prompts_as_tasks(
                    img_fhw,
                    prompt_file,
                    samples_per_prompt,
                    neg_prompt,
                    cfg_scale,
                    num_steps,
                    eta,
                    file_ext,
                    per_gpu_bs,
                    guidance=guidance,
                )

            os.makedirs(os.path.join(output_dir, "assets"), exist_ok=True)
            asset_metadata_list: list[dict[str, Any]] = []
            all_images_for_npz: list[torch.Tensor] = []
            counter = 0
            for task in tqdm(tasks, desc="Generating images"): #, disable=dist.get_rank() != 0):
                # Set random seed for reproducible generation
                generator = torch.Generator(device=device).manual_seed(seed)

                # Add CLIP conditioning if available
                vec = None
                vec_embed = None
                neg_vec_embed = None
                if self.lfm.clip_encoder is not None:
                    with fwd_only_mode(self.lfm.clip_encoder):
                        vec_embed, vec = self.lfm.clip_encoder(task.prompts)
                        if cfg_scale > 1.0:
                            neg_vec_embed, neg_vec = self.lfm.clip_encoder([neg_prompt]* len(task.prompts))
                            vec = torch.cat([vec, neg_vec], dim=0)

                # Encode text prompt
                with fwd_only_mode(self.lfm.text_encoder):
                    txt, txt_datum_lens, txt_embedding_mask = self.lfm.text_encoder(task.prompts)

                # Z-Image VAE properties
                vae_c = 16
                vae_cf, vae_ch, vae_cw = 1, 8, 8

                reference_img_clean = None
                if self.train_dataloader is not None and task.reference_images is not None:
                    reference_img_clean = self.lfm.vae.encode(task.reference_images.to(device))
                    # Use reference image latent shape for noise tensor so dimensions match
                    _, _, latent_f, latent_h, latent_w = reference_img_clean.shape
                else:
                    img_f, img_h, img_w = img_fhw
                    latent_f = int(img_f / vae_cf)
                    latent_h = int(img_h / vae_ch)
                    latent_w = int(img_w / vae_cw)

                # Create noise tensor in latent space shape (b, vae_c, f/vae_cf, h/vae_ch, w/vae_cw)
                noise_tensor = torch.randn(
                    len(task.prompts),
                    vae_c,
                    latent_f,
                    latent_h,
                    latent_w,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )

                # Compute datum_lens from noise tensor shape (patch_size = (1,2,2))
                x_noise = noise_tensor
                b_noise = noise_tensor.shape[0]
                img_datum_lens = torch.full((b_noise,), latent_f * (latent_h // 2) * (latent_w // 2), device=device, dtype=torch.int32)

                if reference_img_clean is not None:
                    img_datum_lens = img_datum_lens * 2

                # Add negative prompts if cfg_scale > 1.0
                if cfg_scale > 1.0:
                    neg_txt, neg_txt_datum_lens, neg_txt_embedding_mask = self.lfm.text_encoder([neg_prompt]* len(task.prompts))
                    txt = torch.cat([txt, neg_txt], dim=0)
                    txt_datum_lens = torch.cat([txt_datum_lens, neg_txt_datum_lens], dim=0)
                    txt_embedding_mask = torch.cat([txt_embedding_mask, neg_txt_embedding_mask], dim=0)

                # Create velocity model wrapper
                velocity_model = VelocityModel(
                    denoiser=selected_denoiser,
                    txt=txt,
                    txt_datum_lens=txt_datum_lens,
                    txt_embedding_mask=txt_embedding_mask,
                    cfg_scale=cfg_scale,
                    reference_img_clean=reference_img_clean,
                    energy_preserve_cfg=energy_preserve_cfg,
                )

                # Create flow sampler
                flow_sampler = FlowSampler(
                    velocity_model=velocity_model,
                    noiser=cast(NoiserProtocol, self.lfm.flow_noiser),
                    t_warper=self.lfm.time_warper,
                    sample_method=sample_method,
                    min_timestep=min_timestep,
                )

                # Perform sampling with appropriate parameters
                warp_len = int(img_datum_lens[0].item())  # Use patch count for time warping

                # Run the denoising loop
                # with torch.autocast("cuda", torch.bfloat16):
                latents, _ = flow_sampler(
                    x=x_noise,
                    x_datum_lens=img_datum_lens,
                    num_steps=num_steps,
                    warp_len=warp_len,
                    rng=generator,
                    eta=eta,
                    timesteps=timesteps,
                )
                # Decode with VAE to get final images
                with fwd_only_mode(self.lfm.vae):
                    images = self.lfm.vae.decode(latents)  # type: ignore

                # Convert to uint8 [0, 255] range
                images = (images.float() + 1) * 127.5  # Convert to [0, 255]
                images = images.round().clamp(0, 255).to(torch.uint8)

                reference_images = [None] * len(task.prompts) if (not hasattr(task, "reference_images") or task.reference_images is None) else task.reference_images

                for prompt, img, reference_img in zip(task.prompts, images, reference_images, strict=True):
                    output_name = f"image_{counter}_{dist.get_rank()}.jpg"
                    counter += 1
                    if prompt == "DUMMY_PROMPT":
                        continue

                    # (c, f, h, w) -> (f, h, w, c) -> (h, w, c)
                    img_processed = img.permute(1, 2, 3, 0)[0]

                    if not save_as_npz:
                        # Save individual image files (default behavior)
                        img_pil = Image.fromarray(img_processed.cpu().numpy())
                        img_pil.save(os.path.join(output_dir, "assets", output_name))
                        if reference_img is not None:
                            reference_img = (reference_img.permute(1, 2, 3, 0)[0] + 1 ) * 127.5
                            img_pil_ref = Image.fromarray(reference_img.round().clamp(0, 255).to(torch.uint8).cpu().numpy())
                            img_pil_ref.save(os.path.join(output_dir, "assets", "reference_" + output_name))
                    else:
                        # Additionally collect for NPZ if requested
                        all_images_for_npz.append(img_processed.cpu())

                    asset_metadata_list.append(
                        {
                            "path": os.path.join("assets", output_name),
                            "prompt": prompt,
                        }
                    )

        # Make a index html file
        gathered_asset_metadata_list = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_asset_metadata_list, asset_metadata_list) #asset_metadata_list, object_gather_list=gathered_asset_metadata_list, dst=0)
        gathered_asset_metadata_list_flat: list[dict[str, Any]] = list(
            itertools.chain.from_iterable(gathered_asset_metadata_list)
        )
        gathered_asset_metadata_list_flat.sort(key=lambda x: x["path"])  # sort by path

        # Prepare sampling info
        sampling_info = {
            "use_ema": use_ema,
            "prompt_file": prompt_file,
            "img_fhw": img_fhw,
            "samples_per_prompt": samples_per_prompt,
            "neg_prompt": neg_prompt,
            "cfg_scale": cfg_scale,
            "num_steps": num_steps,
            "eta": eta,
            "per_gpu_bs": per_gpu_bs,
            "file_ext": file_ext,
            "save_as_npz": save_as_npz,
        }
        metadata = {
            "sampling_info": sampling_info,
            "asset_metadata": gathered_asset_metadata_list_flat,
            "total_images": len(gathered_asset_metadata_list_flat),
        }

        if save_as_npz:
            gathered_images = [None] * dist.get_world_size()
            dist.gather_object(all_images_for_npz, object_gather_list=gathered_images, dst=0)

        # Generate HTML gallery (default behavior)
        if dist.get_rank() == 0:
            # Save metadata as JSON
            with open(os.path.join(output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Saved metadata to metadata.json")

            # Generate HTML gallery
            if not save_as_npz:
                generate_html_gallery(
                    output_dir=output_dir,
                    asset_metadata_list=gathered_asset_metadata_list_flat,
                    sampling_info=sampling_info,
                    images_per_row=samples_per_prompt * 2,
                )
            else:
                all_images = list(itertools.chain.from_iterable(gathered_images))
                images_array = torch.stack(all_images, dim=0).numpy().astype(np.uint8)  # (n, h, w, c)
                np.savez_compressed(os.path.join(output_dir, "assets/images.npz"), images_array, allow_pickle=False)
                logger.info(f"Saved {images_array.shape[0]} images to {os.path.join(output_dir, 'assets/images.npz')}")

        dist.barrier()
        torch.cuda.empty_cache()
