"""
Connectivity tensors for the specs and MaskedLinear.
"""

import torch


def as_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    """
    Return an independent float32 copy of ``mask``.

    A plain ``torch.Tensor``: detached, contiguous, not tracking
    gradients, and never a tensor subclass, so nothing custom
    runs in the forward path.

    The copy is what gives connectivity its stability. Whoever
    stores the result owns it, and mutating the tensor that was
    passed in does not reach the stored one. Storing modules and
    dataclasses document their masks as read-only, but nothing
    enforces that at runtime, exactly as for any other PyTorch
    buffer.
    """
    return mask.detach().to(dtype=torch.float32).contiguous().clone()
