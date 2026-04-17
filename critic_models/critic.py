from einops import rearrange
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torchvision
from transformers import AutoTokenizer
import torch.nn.functional as F

from utils.config import BaseParams, ConfigurableModule
from utils.misc import Float32MatmulPrecision
from utils.log import get_logger, human_readable_number
import torch.distributed as dist
logger = get_logger(__name__)


@dataclass
class BaseCriticParams(BaseParams):
    model_name: str = "OpenGVLab/InternVL3-14B"
    device: str = "cuda"
    image_size: int = 256
    max_steps_temp: int = 5000
    float32_matmul_precision: str = "highest"
    compile: bool = True
    dtype: str = "bf16"

class BaseCritic(nn.Module, ConfigurableModule[BaseCriticParams]):
    def __init__(self, params: BaseCriticParams):
        super().__init__()
        self.params = params

        self.setup_model(params.model_name)
        self.image_size = params.image_size
        self.max_steps_temp = params.max_steps_temp
        self.normalized_logits_over = torch.tensor([2753, 9454])
        self.float32_matmul_precision = params.float32_matmul_precision
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        if params.compile:
            with Float32MatmulPrecision(self.float32_matmul_precision):
                # Compile individual decoder layers instead of the full language_model
                # to avoid torch.compile issues with embedding layers under FSDP2
                for layer in self._get_decoder_layers():
                    layer.compile()


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
            # collect fsdp model parameters on the local rank
            if isinstance(module, torch.distributed.fsdp.FullyShardedDataParallel):
                module = module.module
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
            ("model", self.model),
        ]

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

    @classmethod
    def get_default_params(cls) -> BaseCriticParams:
        """Return the default parameters for BaseCritic."""
        return BaseCriticParams()

    def _get_decoder_layers(self):
        """Return iterable of decoder layers for torch.compile. Override per model."""
        return self.model.model.language_model.layers

    def setup_model(self, model_name):
        pass

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens=100, early_stop=True):
        start_idx = inputs['input_ids'].shape[1]
        for i in range(max_new_tokens):
            current_output = self.model(**inputs)
            predicted_tokens = current_output.logits.argmax(-1)
            if early_stop and predicted_tokens[0, -1] == self.processor.tokenizer.eos_token_id:
                break
            inputs['input_ids'] = torch.cat([inputs['input_ids'], predicted_tokens[:, -1:]], dim=1)
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        current_output_text = self.processor.batch_decode(
            inputs['input_ids'][:, start_idx:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor([[self.processor.tokenizer.eos_token_id, self.processor.tokenizer.pad_token_id]], device=inputs['input_ids'].device)], dim=1)
        return inputs['input_ids'], current_output_text, start_idx

    def __call__(self, inputs, temperature=0.1, vit_embeds=None, attention_mask=None):
        labels = inputs['labels']
        shift_labels = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = shift_labels[..., 1:].contiguous()
        mask = (shift_labels > 0)
        output_ids_start = torch.argmax(mask.float(), dim=1)
        output_ids_end = torch.argmax(torch.cumsum(mask.float(), dim=1), dim=1)
        target_indices = torch.gather(shift_labels, 1, output_ids_start.unsqueeze(1)).squeeze(-1) #[..., output_ids_start]
        del inputs['labels']
        with Float32MatmulPrecision(self.float32_matmul_precision):
            if vit_embeds is not None:
                generated_ids = self.forward_with_cached_vit(
                    vit_embeds, inputs['input_ids'],
                    image_sizes=inputs.get('image_sizes'),
                    batch_num_images=inputs.get('batch_num_images'),
                    image_flags=inputs.get('image_flags'),
                    attention_mask=attention_mask)
            else:
                generated_ids = self.model(**inputs)[0]

        ## target index is 0 if the answer is No and 1 if the answer is Yes
        target_indices = torch.where(target_indices == self.normalized_logits_over[1], 1, 0)
        target_indices = target_indices.squeeze(-1)

        ## get output text
        output_ids = generated_ids.argmax(-1)
        generated_ids_trimmed = [out_[out_ids_:out_ids_end+1] for out_, out_ids_, out_ids_end in zip(output_ids, output_ids_start, output_ids_end)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        ## get normalized score logits
        mask = [(shift_labels == self.normalized_logits_over[i])  for i in range(self.normalized_logits_over.shape[0])]  # [B, T]
        
        # take | over the list in mask
        mask = torch.stack(mask, dim=1)
        mask = torch.any(mask, dim=1)
        score_indices = torch.argmax(mask.float(), dim=1)  # [B], gets first True index for each batch item
        
        # Gather logits for scores Yes/No at the found indices
        score_logits = torch.gather(generated_ids[..., self.normalized_logits_over], 1, score_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.normalized_logits_over.shape[0])).squeeze(1)
    
        ce_loss = self.ce_loss_fn(score_logits * temperature, target_indices.reshape(-1,)).reshape((-1,))

        return ce_loss, output_text, score_logits


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class InternVLCritic(BaseCritic):

    def setup_model(self, model_name):
        from critic_models.internvl import InternVLChatModel, InternVLChatConfig
        local_config = InternVLChatConfig.from_pretrained(model_name, local_files_only=False)
        self.model = InternVLChatModel.from_pretrained(
            model_name,
            config=local_config,
            torch_dtype=torch.bfloat16,
            use_flash_attn=True,
            local_files_only=False,
            low_cpu_mem_usage=False,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False, local_files_only=False,
        )
        self.processor = self.tokenizer
        self.model_name = model_name
        self.model = self.model.eval()
        self.model.requires_grad_(False)

        # Cache pad token ID for sequence padding
        self.pad_token_id = self.tokenizer.pad_token_id

        # Set Yes/No token IDs for InternLM2.5 tokenizer
        yes_ids = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode("No", add_special_tokens=False)
        self.normalized_logits_over = torch.tensor([no_ids[0], yes_ids[0]])

        # Set img_context_token_id needed by the model's forward pass
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')

        # Image token ID for counting images in input_ids (used by unified critic path)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids('<img>')

        # Silence the verbose print in InternVL's forward()
        _orig_forward = self.model.forward
        def _quiet_forward(*args, **kwargs):
            import builtins
            _orig_print = builtins.print
            builtins.print = lambda *a, **kw: None
            try:
                return _orig_forward(*args, **kwargs)
            finally:
                builtins.print = _orig_print
        self.model.forward = _quiet_forward

    def _get_decoder_layers(self):
        return self.model.language_model.model.layers

    def extract_vit_features(self, pixel_values):
        """Run ViT encoder + MLP connector, return vision embeddings [N_images, 256, hidden]."""
        return self.model.extract_feature(pixel_values)

    def forward_with_cached_vit(self, vit_embeds, input_ids, image_sizes=None,
                                batch_num_images=None, image_flags=None, attention_mask=None):
        """Run LLM forward using pre-computed ViT embeddings, bypassing the vision encoder.

        Replicates the token-replacement logic from InternVLChatModel.forward()
        (modeling_internvl_chat.py lines 105-140) but skips extract_feature().
        """
        model = self.model
        input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
        filtered_vit_embeds = vit_embeds[image_flags.squeeze(-1) == 1]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids_flat = input_ids.reshape(B * N)
        selected = (input_ids_flat == model.img_context_token_id)
        input_embeds[selected] = input_embeds[selected] * 0.0 + filtered_vit_embeds.reshape(-1, C)

        input_embeds = input_embeds.reshape(B, N, C)
        outputs = model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        return outputs.logits

    def image_prep(self, x):
        """Input: [B, C, num_images, H, W] in [-1,1]. Output: [B, 3, 448, 448]."""
        torch_dtype = x.dtype
        x = torch.clamp(x.to(torch.float32) * 0.5 + 0.5, 0, 1.)
        x = x[:, :, 0]  # [B, C, H, W]
        x = torchvision.transforms.functional.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        x = F.interpolate(x, size=(448, 448), mode="bicubic", antialias=True, align_corners=False)
        return x.to(torch_dtype)


