from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from utils.config import BaseParams, ConfigurableModule
from utils.log import get_logger
from utils.misc import DTYPE_MAP, Float32MatmulPrecision

logger = get_logger(__name__)


@dataclass
class ZImageTextParams(BaseParams):
    version: str = "Tongyi-MAI/Z-Image"
    max_length: int = 512
    dtype: str = "bf16"
    compile: bool = False
    float32_matmul_precision: str = "high"
    padding_side: str = "right"
    attn_mask_padding: bool = True
    output_exclude_padding: bool = True
    flatten_output: bool = True
    apply_chat_template: bool = True


class ZImageTextEmbedder(nn.Module, ConfigurableModule[ZImageTextParams]):
    def __init__(self, params: ZImageTextParams) -> None:
        nn.Module.__init__(self)
        self.params = params
        self.max_length = params.max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            params.version,
            subfolder="tokenizer",
        )
        self.tokenizer.padding_side = params.padding_side

        self.hf_module = AutoModel.from_pretrained(
            params.version,
            subfolder="text_encoder",
            torch_dtype=DTYPE_MAP[params.dtype],
        )
        self.hf_module = self.hf_module.eval().requires_grad_(False)

        # Hook the second-to-last layer to capture its output.
        # .clone() in the hook detaches from FSDP's unshard context so the
        # layer's parameters can be resharded/freed immediately.
        self._hooked_hidden_state: torch.Tensor | None = None
        target_layer = self.hf_module.layers[-2]
        target_layer.register_forward_hook(self._capture_hidden_state)

        self.float32_matmul_precision = params.float32_matmul_precision
        if params.compile:
            with Float32MatmulPrecision(self.float32_matmul_precision):
                for layer in self.hf_module.layers:
                    layer.compile()

    def _capture_hidden_state(self, module, input, output):
        """Forward hook: clone the second-to-last layer's output.

        .clone() is critical — without it the tensor holds a reference into
        FSDP's unsharded parameter buffer, preventing resharding and causing OOM.
        """
        hidden = output[0] if isinstance(output, tuple) else output
        self._hooked_hidden_state = hidden.clone()

    @classmethod
    def get_default_params(cls) -> ZImageTextParams:
        return ZImageTextParams()

    def _apply_chat_template(self, text: list[str]) -> list[str]:
        """Wrap each prompt in Qwen3's chat template, matching the official ZImagePipeline."""
        templated = []
        for prompt in text:
            messages = [{"role": "user", "content": prompt}]
            templated.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        return templated

    def forward(
        self, text: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.params.apply_chat_template:
            text = self._apply_chat_template(text)

        batch_encoding = self.tokenizer(
            text,
            padding=True,  # pad to longest in batch, not max_length
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = batch_encoding["input_ids"]  # (b, l) where l = longest in batch
        attention_mask = batch_encoding["attention_mask"]  # (b, l)

        device = self.hf_module.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        original_attention_mask = attention_mask.clone()

        if not self.params.attn_mask_padding:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        with Float32MatmulPrecision(self.float32_matmul_precision):
            self.hf_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Grabbed by the hook on layers[-2] — no output_hidden_states needed,
        # so we don't materialize all 36 hidden states in memory.
        text_embedding = self._hooked_hidden_state  # (b, l, d)
        self._hooked_hidden_state = None

        # Determine output mask
        if self.params.output_exclude_padding:
            output_mask = original_attention_mask
        else:
            output_mask = torch.ones_like(original_attention_mask)

        text_embedding_mask = output_mask.bool()  # (b, l)
        text_datum_lens = output_mask.sum(dim=-1)  # (b,)

        if self.params.flatten_output:
            text_embedding = text_embedding.flatten(0, 1)  # (b*l, d)
            text_embedding_mask = text_embedding_mask.flatten(0, 1)  # (b*l,)
            text_embedding = text_embedding[text_embedding_mask]  # (l1+l2+...+ln, d)

        return text_embedding, text_datum_lens, text_embedding_mask
