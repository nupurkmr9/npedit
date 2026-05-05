"""
Model factory for creating configurable LatentFM components.
"""

from copy import deepcopy
from typing import Any
from peft import LoraConfig, get_peft_model
import torch

from models.latent_fm import LatentFM
from utils.config import create_component
from utils.fsdp import dist_model_setup
from utils.log import get_logger
from critic_models.critic import BaseCritic
from utils.misc import DTYPE_MAP
# from knapformer import SequenceBalancer
from torch.distributed.fsdp import CPUOffload
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import functools
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

logger = get_logger(__name__)

from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
import torch.nn as nn
def move_non_fsdp_to_cuda(root: torch.nn.Module, device: torch.device, dtype: torch.dtype | None = None):

    def move_leaf_tensors(mod: torch.nn.Module):
        # parameters
        for k, p in list(mod._parameters.items()):
            if p is None: 
                continue
            mod._parameters[k] = p.to(device=device, dtype=(dtype or p.dtype))
        # buffers
        for k, b in list(mod._buffers.items()):
            if b is None: 
                continue
            mod._buffers[k] = b.to(device=device)

    stack = [root]
    while stack:
        m = stack.pop()
        if isinstance(m, FSDPModule):
            # do not touch this subtree (FSDP manages it)
            continue
        move_leaf_tensors(m)
        for child in m.children():
            stack.append(child)


def create_critic(critic_config: dict[str, Any], device: torch.device) -> BaseCritic:
    """Create Critic with all components based on config."""
    logger.info("Creating Critic...")
    fsdp_spec = critic_config.get("fsdp", None)
    critic = create_component(critic_config["module"], critic_config["params"], fsdp_spec)
    if fsdp_spec is not None and fsdp_spec.get("offload_params", False):
        if not fsdp_spec["shard_root"]:
            move_non_fsdp_to_cuda(critic, device, DTYPE_MAP[critic_config["params"]["dtype"]])
    if fsdp_spec is None or not fsdp_spec.get("offload_params", False):
        critic = critic.to(device, dtype=DTYPE_MAP[critic_config["params"]["dtype"]])
    critic.model.eval()
    critic.model.requires_grad = False
    
    if torch.distributed.get_rank() == 0:
        print(f"Summarizing Critic...")
        critic.summarize()
    return critic


def create_latent_fm(config: dict[str, Any], device: torch.device, create_ema: bool = True, mode: str = 'student', lora_rank: int = 0) -> LatentFM:
    """Create LatentFM with all components based on config."""
    logger.info("Creating LatentFM components...")

    ########## VAE ##########
    vae = None
    if "vae" in config["model"]:
        logger.info("Creating VAE...")
        vae_config = config["model"]["vae"]
        fsdp_spec = vae_config.get("fsdp", None)
        vae = create_component(vae_config["module"], vae_config["params"], fsdp_spec)
        vae = vae.to(device=device, dtype=DTYPE_MAP[vae_config["params"]["dtype"]])
        for param in vae.parameters():
            param.requires_grad = False

    ########## Text Encoder ##########
    text_encoder = None
    if "text_encoder" in config["model"]:
        logger.info("Creating Text Encoder...")
        text_encoder_config = config["model"]["text_encoder"]
        fsdp_spec = text_encoder_config.get("fsdp", None)
        text_encoder = create_component(text_encoder_config["module"], text_encoder_config["params"], fsdp_spec)
        text_encoder = text_encoder.to(device, dtype=DTYPE_MAP[text_encoder_config["params"]["dtype"]])  # noops
        for param in text_encoder.parameters():
            param.requires_grad = False

    ########## Denoiser ##########
    logger.info("Creating Denoiser...")
    denoiser_config = config["model"]["denoiser"]
    denoiser_fsdp_spec = denoiser_config.get("fsdp", None)
    if lora_rank > 0:
        # Create denoiser WITHOUT FSDP first — LoRA must be applied on unsharded
        # weights so that peft picks up full dimensions (not local shard sizes).
        # FSDP is applied after LoRA wrapping below.
        denoiser = create_component(denoiser_config["module"], denoiser_config["params"])
    else:
        denoiser = create_component(denoiser_config["module"], denoiser_config["params"], denoiser_fsdp_spec)
    if denoiser_fsdp_spec is None or not denoiser_fsdp_spec.get("offload_params", False):
        denoiser = denoiser.to(device)
    denoiser.init_weights()
    if lora_rank > 0:
        logger.info(f"Creating LoRA for Denoiser with rank {lora_rank}...")
        denoiser_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"]
        )
        denoiser = get_peft_model(denoiser, denoiser_lora_config)
        # Cast LoRA params (initialized as float32 by peft) to match base model dtype
        denoiser = denoiser.to(dtype=DTYPE_MAP[denoiser_config["params"]["dtype"]])
        # Apply FSDP after LoRA so peft sees full (unsharded) weight dimensions
        if denoiser_fsdp_spec is not None:
            fsdp_spec_copy = deepcopy(denoiser_fsdp_spec)
            fsdp_spec_copy.pop("meta_device_init", None)
            denoiser = dist_model_setup(denoiser, **fsdp_spec_copy)
    else:
        for name, param in denoiser.named_parameters():
            if 'pos_embed' in name or 'x_pad_token' in name or 'cap_pad_token' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True if (mode in ['student', 'aux'] and not lora_rank > 0) else False

    ########## EMA Denoiser ##########
    ema_denoiser = None
    if create_ema:
        logger.info("Creating EMA Denoiser...")
        ema_denoiser_config = config["model"]["denoiser"]
        fsdp_spec = ema_denoiser_config.get("fsdp", None)
        if fsdp_spec is not None:
            # we need to always reshard after forward for ema_denoiser, as it's never back-propagated through
            fsdp_spec = deepcopy(fsdp_spec)
            fsdp_spec["reshard_after_forward_policy"] = "always"

        ema_denoiser = create_component(ema_denoiser_config["module"], ema_denoiser_config["params"], fsdp_spec)
        if fsdp_spec is None or not fsdp_spec.get("offload_params", False):
            ema_denoiser = ema_denoiser.to(device)
        # No need to init weights for ema_denoiser, as it will be copied from denoiser at training start
        for param in ema_denoiser.parameters():
            param.requires_grad = False  # we will directly modify ema_denoiser parameters in training

    ########## Time Sampler ##########
    logger.info("Creating Time Sampler...")
    time_sampler_config = config["model"]["time_sampler"]
    time_sampler = create_component(time_sampler_config["module"], time_sampler_config["params"])

    ########## Time Warper ##########
    logger.info("Creating Time Warper...")
    time_warper_config = config["model"]["time_warper"]
    time_warper = create_component(time_warper_config["module"], time_warper_config["params"])

    ########## Time Weighter ##########
    logger.info("Creating Time Weighter...")
    time_weighter_config = config["model"]["time_weighter"]
    time_weighter = create_component(time_weighter_config["module"], time_weighter_config["params"])

    ########## Flow Noiser ##########
    logger.info("Creating Flow Noiser...")
    flow_noiser_config = config["model"]["flow_noiser"]
    flow_noiser = create_component(flow_noiser_config["module"], flow_noiser_config["params"])

    ########## LatentFM ##########
    latent_fm = LatentFM(
        text_encoder=text_encoder,
        vae=vae,
        denoiser=denoiser,
        ema_denoiser=ema_denoiser,
        time_sampler=time_sampler,
        time_warper=time_warper,
        time_weighter=time_weighter,
        flow_noiser=flow_noiser,
    )
    # only summarize after lora has been taken care of 
    if torch.distributed.get_rank() == 0:
        print(f"Summarizing LatentFM...{mode}")
        latent_fm.summarize()
    return latent_fm
