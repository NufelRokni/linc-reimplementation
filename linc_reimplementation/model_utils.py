"""Model loading utilities.

This module centralises logic related to loading HuggingFace models
and tokenisers with appropriate precision and device mapping.  In the
current implementation only auto‑cast precision and a single device are
supported, but this module provides a convenient hook for future
enhancements (e.g. quantisation, model parallelism).
"""

from __future__ import annotations

__all__ = ["torch_dtype_from_precision"]


def torch_dtype_from_precision(arg: str):
    """Map a precision string onto a torch dtype or ``None``.

    If the argument is ``'auto'`` then ``None`` is returned, allowing
    HuggingFace to infer the appropriate dtype when loading the model.
    Supported explicit precisions are ``fp32``, ``fp16`` and ``bf16``.

    :param arg: Precision string from the command line.
    :returns: torch.dtype or ``None``.
    :raises ValueError: for unknown precision strings.
    """
    if arg is None or arg == "auto":
        return None
    # Import torch lazily here to avoid a hard dependency at module import time
    import torch  # type: ignore

    if arg == "fp16":
        return torch.float16
    if arg == "bf16":
        return torch.bfloat16
    if arg == "fp32":
        return torch.float32
    raise ValueError(f"Unknown precision: {arg}")