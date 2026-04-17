from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch


@contextmanager
def Float32MatmulPrecision(precision: str) -> Generator[None, None, None]:
    # Read more about the precision here:
    # https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    assert precision in ["highest", "high", "medium"]
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(old_precision)


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def load_pt_data_from_path(path: str) -> Any:
    """
    Load data from a pytorch file.

    Args:
        path: Path to the pytorch file
        weights_only: Whether to load only the weights of the file

    Returns:
        Loaded data

    Example:
        >>> data = load_pt_data_from_path("xxx/ilsvrc2012_meta.pt::clip_embeddings")
    """
    if "::" in path:
        file_path, keys = path.split("::")
        data = torch.load(file_path, weights_only=False)
        for key in keys.split("."):
            data = data[key]
    else:
        data = torch.load(path, weights_only=False)
    return data
