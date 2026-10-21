"""Moondream's paged CUDA attention must not change reduction on capture."""
import inspect

import torch
from kestrel_kernels import get_runtime


def validate_attention_policy(device: torch.device) -> None:
    """Reject an old CUDA wrapper before allocating/loading the model.

    The companion kestrel-kernels change adds the explicit selector argument.
    Silently retrying without it would reintroduce capture-dependent dispatch.
    CPU/MPS backends do not use the CUDA persistent-split selector.
    """
    if device.type != "cuda":
        return
    attention = get_runtime(device).attention.flash_attn_fwd
    if "allow_persistent_split" not in inspect.signature(attention).parameters:
        raise RuntimeError(
            "Moondream CUDA requires kestrel-kernels with the "
            "allow_persistent_split attention policy. Install the companion "
            "kernel update; the unpatched 0.7.5 wrapper is unsupported."
        )
