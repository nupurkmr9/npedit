from dataclasses import dataclass
import itertools

import torch


@dataclass
class PackMeta:
    """
    Metadata for interleaving multimodal sequences.

    Transforms sequences from modality-grouped to interleaved format:
    [a1, a2, ..., an, b1, b2, ..., bn, c1, c2, ..., cn]
    -> [a1, b1, c1, ..., an, bn, cn]

    Where a_i, b_i, c_i denote per-modality token chunks for datum i.

    Attributes:
        permute_indices: Indices to reorder tokens from modality-grouped to interleaved
        datum_lens: (num_datums,) - total tokens per datum (may contain 0s)
        datum_ids: (sum(datum_lens),) - datum ID for each token
        modality_ids: (sum(datum_lens),) - modality ID for each token
        modality_id2indices: Maps modality_id to token indices for that modality
    """

    permute_indices: torch.Tensor
    datum_lens: torch.Tensor
    datum_ids: torch.Tensor
    modality_ids: torch.Tensor
    modality_id2indices: dict[int, torch.Tensor]


def build_modality_id2indices(
    modality_ids: torch.Tensor,
    unique_modality_ids: torch.Tensor | list[int] | None = None,
) -> dict[int, torch.Tensor]:
    """
    Build mappings from modality_id to token indices for multimodal sequences.

    Args:
        modality_ids: (N,) tensor of modality IDs for each token
        unique_modality_ids: Optional list of unique modality IDs to include

    Returns:
        Dictionary mapping modality_id to tensor of token indices for that modality

    Note:
        Sorts tokens by modality_id (primary) and token index (secondary) for efficient grouping.
    """
    if unique_modality_ids is None:
        unique_modality_ids_list = torch.unique(modality_ids).tolist()
    elif isinstance(unique_modality_ids, torch.Tensor):
        unique_modality_ids_list = unique_modality_ids.tolist()
    else:
        unique_modality_ids_list = unique_modality_ids

    if len(unique_modality_ids_list) == 0:
        raise ValueError("No unique modality IDs provided")

    assert min(unique_modality_ids_list) >= 0, f"Modality IDs must be non-negative, got {min(unique_modality_ids_list)}"

    # Sort by modality_id (primary), then by token index (secondary)
    max_modality_id = max(unique_modality_ids_list)
    num_tokens = modality_ids.shape[0]
    sort_keys = modality_ids * (num_tokens + 1) + torch.arange(num_tokens, device=modality_ids.device)
    sorted_indices = torch.argsort(sort_keys)

    # Count tokens per modality and find split points
    num_tokens_per_modality = torch.bincount(modality_ids, minlength=max_modality_id + 1)

    modality_indices = sorted_indices.split(num_tokens_per_modality.tolist())
    modality_id2indices: dict[int, torch.Tensor] = {}
    for modality_id in unique_modality_ids_list:
        modality_id2indices[modality_id] = modality_indices[modality_id]

    return modality_id2indices


def plan_interleave(
    unimodal_datum_lens: list[torch.Tensor | list[int]],
    unimodal_modality_ids: list[int],
    device: torch.device = torch.device("cpu"),
) -> PackMeta:
    """
    Plan the interleaving of multimodal sequences.

    Creates metadata needed to transform modality-grouped sequences into interleaved format.

    Args:
        unimodal_datum_lens: List of sequence lengths for each modality
        unimodal_modality_ids: List of modality IDs corresponding to each length list

    Returns:
        PackMeta containing indices and metadata for interleaving

    Example:
        >>> lengths = [[2, 3], [1, 2]]  # 2 modalities, 2 datums each
        >>> modality_ids = [0, 1]       # modality 0 and 1
        >>> meta = plan_interleave(lengths, modality_ids)
    """
    if not (len(unimodal_datum_lens) == len(unimodal_modality_ids)):
        raise ValueError(
            f"Input unimodal_datum_lens ({len(unimodal_datum_lens)}) must match "
            f"unimodal_modality_ids ({len(unimodal_modality_ids)})"
        )
    if len(unimodal_datum_lens) == 0:
        raise ValueError("No unimodal datum lengths provided")

    # Convert unimodal_datum_lens to list[torch.Tensor]
    unimodal_datum_lens_tensor: list[torch.Tensor] = []
    for lens in unimodal_datum_lens:
        if not isinstance(lens, torch.Tensor):
            lens = torch.tensor(lens, dtype=torch.int32)
        else:
            lens = lens.type(torch.int32)
        unimodal_datum_lens_tensor.append(lens)

    unimodal_datum_lens_tensor_cpu = torch.stack(unimodal_datum_lens_tensor, dim=0).cpu()
    multimodal_datum_lens = unimodal_datum_lens_tensor_cpu.sum(dim=0)
    num_datums, num_tokens = multimodal_datum_lens.shape[0], int(multimodal_datum_lens.sum().item())
    multimodal_datum_lens = multimodal_datum_lens.to(device=device)
    multimodal_datum_ids: torch.Tensor = torch.repeat_interleave(
        torch.arange(num_datums, device=device, dtype=torch.int32), multimodal_datum_lens, output_size=num_tokens
    )

    original_indices_list: list[list[torch.Tensor]] = []
    original_modality_ids_list: list[int] = []
    unique_modality_ids_list: list[int] = []
    global_offset = 0
    for modality_id, lens_list in zip(unimodal_modality_ids, unimodal_datum_lens_tensor_cpu.tolist(), strict=True):
        num_tokens = sum(lens_list)
        chunk_indices: list[torch.Tensor] = list(
            torch.arange(global_offset, global_offset + num_tokens, dtype=torch.int32).split(lens_list)
        )  # cpu
        original_indices_list.append(chunk_indices)
        original_modality_ids_list.extend([modality_id] * num_tokens)
        if modality_id not in unique_modality_ids_list:
            unique_modality_ids_list.append(modality_id)
        global_offset += num_tokens

    permute_indices: torch.Tensor = torch.cat(
        list(itertools.chain.from_iterable(zip(*original_indices_list, strict=True)))  # type: ignore[arg-type]
    ).to(device=device, dtype=torch.int32)

    original_modality_ids: torch.Tensor = torch.tensor(original_modality_ids_list, device=device, dtype=torch.int32)
    multimodal_modality_ids: torch.Tensor = original_modality_ids[permute_indices]
    multimodal_modality_id2indices = build_modality_id2indices(multimodal_modality_ids, unique_modality_ids_list)
    return PackMeta(
        permute_indices=permute_indices,
        datum_lens=multimodal_datum_lens,
        datum_ids=multimodal_datum_ids,
        modality_ids=multimodal_modality_ids,
        modality_id2indices=multimodal_modality_id2indices,
    )


def apply_interleave(unimodal_sequences: list[torch.Tensor], pack_meta: PackMeta) -> torch.Tensor:
    """
    Apply interleaving to multimodal sequences using pre-computed metadata.

    Args:
        unimodal_sequences: List of tensors, one per modality
        pack_meta: PackMeta containing interleaving indices and metadata

    Returns:
        Interleaved tensor with tokens from all modalities mixed together
    """
    unimodal_sequences_tensor = torch.cat(unimodal_sequences, dim=0)
    permute_indices = pack_meta.permute_indices
    return unimodal_sequences_tensor[permute_indices]


def pack_reduce(packed_tensor: torch.Tensor, lengths: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """
    Reduce packed sequences by summing or averaging all elements within each sequence.

    Args:
        packed_tensor: Tensor with shape (sum(lengths), *features) containing concatenated sequences
        lengths: (N,) tensor of sequence lengths
        reduction: "sum" or "mean"

    Returns:
        (N,) tensor with reduced values for each sequence

    Example:
        >>> packed = torch.tensor([1., 2., 3., 4., 5.])
        >>> lengths = torch.tensor([2, 3])  # sequences: [1,2] and [3,4,5]
        >>> pack_reduce(packed, lengths, "sum")
        tensor([3., 12.])  # [1+2, 3+4+5]
    """
    if reduction not in ["sum", "mean", "none"]:
        raise ValueError(f"reduction must be 'sum' or 'mean', or 'none', got '{reduction}'")

    if packed_tensor.ndim < 1:
        raise RuntimeError(f"packed_tensor must be at least 1D, got {packed_tensor.ndim}D")

    if packed_tensor.shape[0] != lengths.sum():
        raise RuntimeError(
            f"First dimension of packed_tensor ({packed_tensor.shape[0]}) must equal sum of lengths ({lengths.sum()})"
        )

    device, dtype, shape = packed_tensor.device, packed_tensor.dtype, packed_tensor.shape
    num_sequences = len(lengths)

    packed_tensor = packed_tensor.float().ravel()
    numel_per_seq = lengths * (1 if len(shape) == 1 else torch.tensor(shape[1:]).prod().item())
    numel_total = packed_tensor.numel()
    segment_ids = torch.repeat_interleave(
        torch.arange(num_sequences, device=device), numel_per_seq, output_size=numel_total
    )
    result = torch.zeros(num_sequences, device=device, dtype=torch.float32)
    result = result.scatter_add(0, segment_ids, packed_tensor)

    if reduction == "mean":
        result = result / (numel_per_seq + 1e-6)

    return result.type(dtype)
