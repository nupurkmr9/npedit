from dataclasses import dataclass

from einops import rearrange
import torch
from torch import Tensor, nn

from utils.misc import DTYPE_MAP
from utils.config import BaseParams, ConfigurableModule


@dataclass
class ZImageVAEParams(BaseParams):
    version: str = "Tongyi-MAI/Z-Image"
    dtype: str = "fp32"
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159
    chunk_size: int = 64


class ZImageVAE(nn.Module, ConfigurableModule[ZImageVAEParams]):
    def __init__(self, params: ZImageVAEParams, from_pretrained: bool = True):
        nn.Module.__init__(self)
        self.params = params
        self.runtime_dtype: torch.dtype | None = None

        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(
            params.version,
            subfolder="vae",
            torch_dtype=DTYPE_MAP[params.dtype],
        )
        self.vae = self.vae.eval().requires_grad_(False)

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor
        self.chunk_size = params.chunk_size

    @classmethod
    def get_default_params(cls) -> ZImageVAEParams:
        return ZImageVAEParams()

    def encode(self, x: Tensor) -> Tensor:
        if self.runtime_dtype is None:
            self.runtime_dtype = next(self.vae.encoder.parameters()).dtype

        input_shape, input_dtype = x.shape, x.dtype
        if len(input_shape) == 5:
            x = rearrange(x, "b c f h w -> (b f) c h w")

        x = x.type(self.runtime_dtype)
        z_chunks = []
        for i in range(0, x.shape[0], self.chunk_size):
            z_chunks.append(self.vae.encode(x[i : i + self.chunk_size]).latent_dist.mean)
        z = torch.cat(z_chunks, dim=0)
        z = self.scale_factor * (z - self.shift_factor)

        if len(input_shape) == 5:
            z = rearrange(z, "(b f) c h w -> b c f h w", b=input_shape[0])

        return z.type(input_dtype)

    def decode(self, z: Tensor) -> Tensor:
        if self.runtime_dtype is None:
            self.runtime_dtype = next(self.vae.decoder.parameters()).dtype

        input_shape, input_dtype = z.shape, z.dtype
        if len(input_shape) == 5:
            z = rearrange(z, "b c f h w -> (b f) c h w")

        z = z.type(self.runtime_dtype) / self.scale_factor + self.shift_factor
        x_chunks = []
        for i in range(0, z.shape[0], self.chunk_size):
            x_chunks.append(self.vae.decode(z[i : i + self.chunk_size]).sample)
        x = torch.cat(x_chunks, dim=0)

        if len(input_shape) == 5:
            x = rearrange(x, "(b f) c h w -> b c f h w", b=input_shape[0])

        return x.type(input_dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))
