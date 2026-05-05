from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from diffusers import ZImageTransformer2DModel
from diffusers.models.attention_processor import Attention

try:
    from flash_attn.cute.interface import flash_attn_func as _fa4_func
    _HAS_FA4 = True
except ImportError:
    _HAS_FA4 = False

from utils.config import BaseParams, ConfigurableModule
from utils.misc import DTYPE_MAP

SEQ_MULTI_OF = 32


# ---------------------------------------------------------------------------
# Batched patchify / unpatchify (replaces diffusers list-based versions)
# ---------------------------------------------------------------------------

def _batched_patchify(img: torch.Tensor, patch_size: int, f_patch_size: int) -> torch.Tensor:
    """Patchify a batch of images.

    Args:
        img: (B, C, F, H, W)
        patch_size: spatial patch size (pH = pW)
        f_patch_size: temporal patch size (pF)

    Returns:
        patches: (B, N, pF*pH*pW*C) where N = (F//pF)*(H//pH)*(W//pW), padded to multiple of SEQ_MULTI_OF
    """
    B, C, F, H, W = img.shape
    pH = pW = patch_size
    pF = f_patch_size
    Ft, Ht, Wt = F // pF, H // pH, W // pW
    N = Ft * Ht * Wt

    # (B, C, Ft, pF, Ht, pH, Wt, pW) -> (B, Ft, Ht, Wt, pF, pH, pW, C) -> (B, N, pF*pH*pW*C)
    x = img.view(B, C, Ft, pF, Ht, pH, Wt, pW)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, N, pF * pH * pW * C)

    # Pad to multiple of SEQ_MULTI_OF
    pad_len = (-N) % SEQ_MULTI_OF
    if pad_len > 0:
        x = torch.cat([x, x[:, -1:].expand(-1, pad_len, -1)], dim=1)

    return x, N, pad_len, (Ft, Ht, Wt)


def _batched_unpatchify(
    x: torch.Tensor, ori_len: int, grid: Tuple[int, int, int],
    patch_size: int, f_patch_size: int, out_channels: int,
) -> torch.Tensor:
    """Unpatchify a batch of token sequences back to images.

    Args:
        x: (B, S, D) — padded token sequences
        ori_len: number of real (non-padding) tokens
        grid: (Ft, Ht, Wt) token-space grid dimensions
        patch_size, f_patch_size, out_channels: patch / channel config

    Returns:
        (B, out_channels, F, H, W)
    """
    pH = pW = patch_size
    pF = f_patch_size
    Ft, Ht, Wt = grid
    B = x.shape[0]

    # (B, Ft*Ht*Wt, pF*pH*pW*C) -> (B, Ft, Ht, Wt, pF, pH, pW, C) -> (B, C, F, H, W)
    x = x[:, :ori_len].view(B, Ft, Ht, Wt, pF, pH, pW, out_channels)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # (B, C, Ft, pF, Ht, pH, Wt, pW)
    return x.reshape(B, out_channels, Ft * pF, Ht * pH, Wt * pW)


def _batched_cap_pad(cap_feats: torch.Tensor, cap_lens: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Pad caption features to a common length that is a multiple of SEQ_MULTI_OF.

    Args:
        cap_feats: (B, L, D) — already padded to max cap length L
        cap_lens: (B,) — original lengths per sample

    Returns:
        padded: (B, L_padded, D)
        max_ori_len: max original caption length
        padded_len: total padded sequence length (multiple of SEQ_MULTI_OF)
    """
    max_ori_len = int(cap_lens.max().item())
    padded_len = max_ori_len + ((-max_ori_len) % SEQ_MULTI_OF)

    # Trim or expand to padded_len
    if padded_len <= cap_feats.shape[1]:
        padded = cap_feats[:, :padded_len]
    else:
        extra = padded_len - cap_feats.shape[1]
        padded = torch.cat([cap_feats, cap_feats[:, -1:].expand(-1, extra, -1)], dim=1)

    return padded, max_ori_len, padded_len


def _make_image_pos_ids(
    cap_padded_len: int, Ft: int, Ht: int, Wt: int, pad_len: int, device: torch.device,
) -> torch.Tensor:
    """Create batched image position IDs for RoPE.

    Matches diffusers create_coordinate_grid with start=(cap_padded_len + 1, 0, 0).
    Returns: (N + pad_len, 3) — shared across all batch items.
    """
    N = Ft * Ht * Wt
    f_ax = torch.arange(cap_padded_len + 1, cap_padded_len + 1 + Ft, dtype=torch.int32, device=device)
    h_ax = torch.arange(Ht, dtype=torch.int32, device=device)
    w_ax = torch.arange(Wt, dtype=torch.int32, device=device)
    grid_f, grid_h, grid_w = torch.meshgrid(f_ax, h_ax, w_ax, indexing="ij")
    pos_ids = torch.stack([grid_f, grid_h, grid_w], dim=-1).reshape(N, 3)

    if pad_len > 0:
        pad_ids = torch.zeros((pad_len, 3), dtype=torch.int32, device=device)
        pos_ids = torch.cat([pos_ids, pad_ids], dim=0)

    return pos_ids


def _make_cap_pos_ids(cap_padded_len: int, device: torch.device) -> torch.Tensor:
    """Create batched caption position IDs for RoPE.

    Matches diffusers create_coordinate_grid with start=(1, 0, 0), size=(cap_padded_len, 1, 1).
    Returns: (cap_padded_len, 3)
    """
    pos = torch.zeros((cap_padded_len, 3), dtype=torch.int32, device=device)
    pos[:, 0] = torch.arange(1, cap_padded_len + 1, dtype=torch.int32, device=device)
    return pos


# ---------------------------------------------------------------------------
# Dual-adaln helpers
# ---------------------------------------------------------------------------

def _expand_mod(mod: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Expand (B, D) modulation to (B, n_tokens, D) without copying."""
    return mod.unsqueeze(1).expand(-1, n_tokens, -1)


def _dual_adaln_block_forward(layer, x, attn_mask, freqs_cis, adaln_noisy, adaln_clean, n_clean):
    """Custom forward for ZImageTransformerBlock with per-token adaln.

    Instead of torch.where with a mask, we split the sequence at `n_clean`,
    apply clean modulation to [:n_clean] and noisy modulation to [n_clean:],
    then concatenate. This eliminates 4 torch.where broadcast ops per layer.

    Args:
        n_clean: number of clean (reference) tokens at the start of the sequence.
                 Tokens [n_clean:] get noisy modulation (target + text tokens).
    """
    seq_len = x.shape[1]
    n_noisy = seq_len - n_clean

    mod_noisy = layer.adaLN_modulation(adaln_noisy)
    mod_clean = layer.adaLN_modulation(adaln_clean)

    scale_msa_n, gate_msa_n, scale_mlp_n, gate_mlp_n = mod_noisy.chunk(4, dim=1)
    scale_msa_c, gate_msa_c, scale_mlp_c, gate_mlp_c = mod_clean.chunk(4, dim=1)

    gate_msa_n, gate_mlp_n = gate_msa_n.tanh(), gate_mlp_n.tanh()
    gate_msa_c, gate_mlp_c = gate_msa_c.tanh(), gate_mlp_c.tanh()
    scale_msa_n, scale_mlp_n = 1.0 + scale_msa_n, 1.0 + scale_mlp_n
    scale_msa_c, scale_mlp_c = 1.0 + scale_msa_c, 1.0 + scale_mlp_c

    # Build per-token modulation via concat instead of torch.where
    scale_msa = torch.cat([_expand_mod(scale_msa_c, n_clean), _expand_mod(scale_msa_n, n_noisy)], dim=1)
    gate_msa = torch.cat([_expand_mod(gate_msa_c, n_clean), _expand_mod(gate_msa_n, n_noisy)], dim=1)
    scale_mlp = torch.cat([_expand_mod(scale_mlp_c, n_clean), _expand_mod(scale_mlp_n, n_noisy)], dim=1)
    gate_mlp = torch.cat([_expand_mod(gate_mlp_c, n_clean), _expand_mod(gate_mlp_n, n_noisy)], dim=1)

    attn_out = layer.attention(
        layer.attention_norm1(x) * scale_msa,
        attention_mask=attn_mask,
        freqs_cis=freqs_cis,
    )
    x = x + gate_msa * layer.attention_norm2(attn_out)
    x = x + gate_mlp * layer.ffn_norm2(layer.feed_forward(layer.ffn_norm1(x) * scale_mlp))
    return x


def _dual_adaln_final_forward(final_layer, x, adaln_noisy, adaln_clean, n_clean):
    """Custom forward for FinalLayer with per-token adaln."""
    n_noisy = x.shape[1] - n_clean
    scale_noisy = 1.0 + final_layer.adaLN_modulation(adaln_noisy)
    scale_clean = 1.0 + final_layer.adaLN_modulation(adaln_clean)
    scale = torch.cat([_expand_mod(scale_clean, n_clean), _expand_mod(scale_noisy, n_noisy)], dim=1)
    x = final_layer.norm_final(x) * scale
    x = final_layer.linear(x)
    return x


# ---------------------------------------------------------------------------
# Bound method replacements (replaces per-call monkey-patching)
# ---------------------------------------------------------------------------

def replace_forward_function(module, new_forward):
    """Permanently bind a new forward method to a module instance."""
    module.forward = new_forward.__get__(module, module.__class__)


def _bound_dual_adaln_block_forward(self, x, attn_mask, freqs_cis, adaln_noisy, adaln_clean=None, n_clean=None):
    """Bound replacement for ZImageTransformerBlock.forward with dual-adaln.

    `self` is the layer instance (bound via __get__).
    """
    if not self.modulation:
        return self._original_forward(x, attn_mask, freqs_cis)
    return _dual_adaln_block_forward(
        self, x, attn_mask, freqs_cis,
        adaln_noisy, adaln_clean, n_clean,
    )


def _bound_dual_adaln_final_forward(self, x, adaln_noisy, adaln_clean, n_clean):
    """Bound replacement for FinalLayer.forward with dual-adaln."""
    return _dual_adaln_final_forward(
        self, x, adaln_noisy, adaln_clean, n_clean,
    )


# ---------------------------------------------------------------------------
# FA4 attention processor (replaces ZSingleStreamAttnProcessor)
# ---------------------------------------------------------------------------

def _apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding (same math as ZSingleStreamAttnProcessor)."""
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


class FA4AttnProcessor:
    """Drop-in replacement for ZSingleStreamAttnProcessor that calls Flash
    Attention 4 directly via ``flash_attn_func``.

    FA4's ``flash_attn_func`` takes (B, S, H, D) natively — no packing or
    cu_seqlens needed.  Padding tokens (at most 31 of them) participate in
    attention but have negligible effect on outputs.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # (B, S, dim) -> (B, S, H, D)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if freqs_cis is not None:
            query = _apply_rotary_emb(query, freqs_cis)
            key = _apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype

        # FA4 flash_attn_func: (B, S, H, D) in, (B, S, H, D) out
        hidden_states, _ = _fa4_func(query, key, value, causal=False)

        hidden_states = hidden_states.flatten(2, 3).to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)

        return output


# ---------------------------------------------------------------------------
# Main denoiser
# ---------------------------------------------------------------------------

@dataclass
class ZImageDenoiserParams(BaseParams):
    version: str = "Tongyi-MAI/Z-Image"
    dtype: str = "bf16"
    reference_image: bool = False


class ZImageDenoiser(nn.Module, ConfigurableModule[ZImageDenoiserParams]):
    def __init__(self, params: ZImageDenoiserParams):
        nn.Module.__init__(self)
        self.params = params
        self.reference_image = params.reference_image

        # Always load pretrained weights eagerly (on CPU).
        # FSDP2 will shard these after construction.
        # We override the default device context to ensure from_pretrained
        # allocates on CPU even if called inside `with torch.device("meta"):`.
        with torch.device("cpu"):
            self.transformer = ZImageTransformer2DModel.from_pretrained(
                params.version,
                subfolder="transformer",
                device_map=None,
                low_cpu_mem_usage=False,
                torch_dtype=DTYPE_MAP[params.dtype],
            )
        # Replace attention processors with FA4
        if _HAS_FA4:
            fa4_processor = FA4AttnProcessor()
            for module in self.transformer.modules():
                if isinstance(module, Attention):
                    module.processor = fa4_processor

        self.runtime_parameter_dtype: torch.dtype | None = None

        if self.reference_image:
            self._setup_dual_adaln_forwards()

    def _setup_dual_adaln_forwards(self):
        """Replace layer forwards once for dual-adaln support.

        Called in __init__ when reference_image=True. Each layer gets a permanent
        bound method whose adaln args are threaded positionally.
        """
        tf = self.transformer

        for layer in list(tf.noise_refiner) + list(tf.layers):
            layer._original_forward = layer.forward
            replace_forward_function(layer, _bound_dual_adaln_block_forward)

        for key, final_layer in tf.all_final_layer.items():
            final_layer._original_forward = final_layer.forward
            replace_forward_function(final_layer, _bound_dual_adaln_final_forward)

    @classmethod
    def get_default_params(cls) -> ZImageDenoiserParams:
        return ZImageDenoiserParams()

    def init_weights(self) -> None:
        rope = getattr(self.transformer, 'rope_embedder', None)
        if rope is not None:
            print("Re-initializing Z-Image rope embedder freqs_cis")
            rope.freqs_cis = rope.precompute_freqs_cis(
                rope.axes_dims, rope.axes_lens, rope.theta,
            )

    def forward(
        self,
        txt: torch.Tensor,
        txt_datum_lens: torch.Tensor,
        img: torch.Tensor,
        t: torch.Tensor,
        txt_embedding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Adapts the repo's denoiser interface to ZImageTransformer2DModel.forward(x, t, cap_feats).

        With dummy patchifier:
            img: (b, c, f, h, w) — raw VAE latents
            txt: (total_tokens, d) flattened or (b, L, d) non-flattened
            txt_datum_lens: (b,) — per-sample valid text token counts
            t: (b,) — timesteps in [0,1], where 0=clean, 1=noise

        With reference_image=True:
            img arrives as (b, c, f=1, 2h, w) — ref+target concatenated along height by FrozenOps.
            We split and re-stack along the frame dim to get (b, c, f=2, h, w) so that Z-Image's
            internal patchifier assigns aligned spatial RoPE coords but separates ref/target by a
            unit offset in the temporal axis (per the Z-Image paper, arXiv 2511.22699).
            Per-frame adaln conditioning: reference tokens get t_clean=1.0, target tokens get actual t.
            Output is cropped to the target frame only: (b, c, 1, h, w).
        """
        device, orig_img_dtype = img.device, img.dtype
        if self.runtime_parameter_dtype is None:
            self.runtime_parameter_dtype = next(self.parameters()).dtype

        txt = txt.type(self.runtime_parameter_dtype)
        img = img.type(self.runtime_parameter_dtype)

        # Z-Image uses t=1 for clean, t=0 for noise (opposite of repo convention)
        t_zimage = (1.0 - t.float()).type(self.runtime_parameter_dtype)

        # Handle reference image: convert height-concat to frame-concat
        if self.reference_image:
            h = img.shape[-2] // 2
            ref = img[:, :, :, :h, :]
            noisy = img[:, :, :, h:, :]
            img = torch.cat([ref, noisy], dim=2)  # (b, c, 2, h, w)

        if self.reference_image:
            noise_pred = self._forward_dual_adaln_batched(img, t_zimage, txt, txt_datum_lens)
        else:
            # Non-dual-adaln path: still needs lists for diffusers API
            if txt.dim() == 3:
                cap_feats = [txt[i, :txt_datum_lens[i].item()] for i in range(txt.shape[0])]
            else:
                cap_feats = list(txt.split(txt_datum_lens.tolist(), dim=0))
            x_list = list(img.unbind(dim=0))
            output = self.transformer(x=x_list, t=t_zimage, cap_feats=cap_feats, return_dict=False)
            noise_pred = torch.stack(output[0], dim=0)

        if self.reference_image:
            noise_pred = noise_pred[:, :, 1:, :, :]  # target frame only

        return (-noise_pred).type(orig_img_dtype)

    def _forward_dual_adaln_batched(
        self,
        img: torch.Tensor,
        t: torch.Tensor,
        txt: torch.Tensor,
        txt_datum_lens: torch.Tensor,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> torch.Tensor:
        """Fully batched dual-adaln forward — no list conversions, no monkey-patching.

        Layer forwards were replaced once in __init__ via _setup_dual_adaln_forwards.
        adaln_noisy/adaln_clean/n_clean are passed positionally so activation
        checkpointing can recompute layers independently.
        """
        tf = self.transformer
        B = img.shape[0]
        device = img.device

        # Move datum_lens to CPU once to avoid repeated CUDA syncs from .item()/.tolist()
        txt_datum_lens_cpu = txt_datum_lens.tolist() if txt_datum_lens.is_cuda else txt_datum_lens.tolist()
        max_cap_ori_len = max(txt_datum_lens_cpu)
        cap_padded_len = max_cap_ori_len + ((-max_cap_ori_len) % SEQ_MULTI_OF)

        # --- Timestep embeddings ---
        adaln_noisy = tf.t_embedder(t * tf.t_scale)
        adaln_clean = tf.t_embedder(torch.ones_like(t) * tf.t_scale)

        # --- Patchify image (batched) ---
        x_patches, x_ori_len, x_pad_len, grid = _batched_patchify(img, patch_size, f_patch_size)
        x_seq_len = x_ori_len + x_pad_len

        # Image embedding: (B, S, patch_flat) -> (B, S, dim)
        x_flat = x_patches.reshape(B * x_seq_len, -1)
        x_flat = tf.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_flat)

        # Apply pad token to padding positions
        x_flat = x_flat.view(B, x_seq_len, -1)
        if x_pad_len > 0:
            x_flat[:, x_ori_len:] = tf.x_pad_token

        adaln_noisy = adaln_noisy.type_as(x_flat)
        adaln_clean = adaln_clean.type_as(x_flat)

        # Image RoPE position IDs (shared across batch)
        img_pos_ids = _make_image_pos_ids(cap_padded_len, grid[0], grid[1], grid[2], x_pad_len, device)
        x_freqs_cis = tf.rope_embedder(img_pos_ids).unsqueeze(0).expand(B, -1, -1)

        # Image attention mask
        x_attn_mask = torch.ones((B, x_seq_len), dtype=torch.bool, device=device)
        if x_pad_len > 0:
            x_attn_mask[:, x_ori_len:] = False

        # n_clean = number of reference (frame 0) tokens at start of sequence
        n_ref_tokens = grid[1] * grid[2]

        # --- Noise refiner (image-only) ---
        for layer in tf.noise_refiner:
            x_flat = layer(x_flat, x_attn_mask, x_freqs_cis, adaln_noisy, adaln_clean, n_ref_tokens)

        # --- Caption embedding (batched) ---
        if txt.dim() == 3:
            cap_feats_batched = txt[:, :max_cap_ori_len]
        else:
            # Use pre-computed CPU list — no .tolist() sync here
            splits = txt.split(txt_datum_lens_cpu, dim=0)
            cap_feats_batched = torch.zeros(B, max_cap_ori_len, txt.shape[-1], dtype=txt.dtype, device=device)
            for i, s in enumerate(splits):
                L = min(s.shape[0], max_cap_ori_len)
                cap_feats_batched[i, :L] = s[:L]

        cap_seq_len = cap_padded_len  # already computed above, no second .item() call

        # Trim or pad caption features to cap_seq_len
        if cap_seq_len <= cap_feats_batched.shape[1]:
            cap_feats_padded = cap_feats_batched[:, :cap_seq_len]
        else:
            extra = cap_seq_len - cap_feats_batched.shape[1]
            cap_feats_padded = torch.cat([cap_feats_batched, cap_feats_batched[:, -1:].expand(-1, extra, -1)], dim=1)

        cap_flat = cap_feats_padded.reshape(B * cap_seq_len, -1)
        cap_flat = tf.cap_embedder(cap_flat)
        cap_flat = cap_flat.view(B, cap_seq_len, -1)

        # Apply pad token to caption padding positions (use CPU lens to build mask without sync)
        cap_lens_t = torch.tensor(txt_datum_lens_cpu, dtype=torch.long, device=device)
        cap_lens_clamped = cap_lens_t.clamp(max=max_cap_ori_len)
        cap_pad_mask = torch.arange(cap_seq_len, device=device).unsqueeze(0) >= cap_lens_clamped.unsqueeze(1)
        cap_flat[cap_pad_mask] = tf.cap_pad_token

        # Caption RoPE
        cap_pos_ids = _make_cap_pos_ids(cap_seq_len, device)
        cap_freqs_cis = tf.rope_embedder(cap_pos_ids).unsqueeze(0).expand(B, -1, -1)

        cap_attn_mask = ~cap_pad_mask

        # --- Context refiner (caption-only, no adaln) ---
        for layer in tf.context_refiner:
            cap_flat = layer(cap_flat, cap_attn_mask, cap_freqs_cis)

        # --- Unify image + caption tokens ---
        unified = torch.cat([x_flat, cap_flat], dim=1)
        unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_cis], dim=1)
        unified_attn_mask = torch.cat([x_attn_mask, cap_attn_mask], dim=1)

        # n_clean stays the same: only the first n_ref_tokens (frame 0 image tokens)
        # are clean. All other tokens (frame 1 image + padding + text) get noisy modulation.
        # --- Main transformer layers ---
        for layer in tf.layers:
            unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_noisy, adaln_clean, n_ref_tokens)

        # --- Final layer ---
        unified = tf.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_noisy, adaln_clean, n_ref_tokens)

        # --- Unpatchify (batched) ---
        x_out = unified[:, :x_seq_len]
        return _batched_unpatchify(x_out, x_ori_len, grid, patch_size, f_patch_size, tf.out_channels)
