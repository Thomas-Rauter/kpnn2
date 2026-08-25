"""
Read-only connectivity tensors for GraphSpec and MaskedLinear.
"""

import torch

from .errors import Kpnn2Error

_READ_ONLY_MSG = (
    "Connectivity mask is read-only. Rebuild from the edgelist "
    "GraphSpec if the wiring should change."
)


class FrozenMask(torch.Tensor):
    """
    Float32 connectivity tensor that rejects in-place writes.
    """

    @classmethod
    def __torch_function__(
        cls,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        if kwargs is None:
            kwargs = {}
        name = getattr(func, "__name__", "")
        inplace = name.endswith("_") and not name.startswith("__")
        if inplace and args and isinstance(args[0], cls):
            raise Kpnn2Error(_READ_ONLY_MSG)
        result = super().__torch_function__(
            func,
            types,
            args,
            kwargs,
        )
        return _as_plain(result)

    def __setitem__(
        self,
        key,
        value,
    ):
        raise Kpnn2Error(_READ_ONLY_MSG)

    def __getitem__(
        self,
        key,
    ):
        item = torch.Tensor.__getitem__(
            self,
            key,
        )
        return _as_plain(item)


def _as_plain(result):
    """
    Drop FrozenMask so arithmetic does not leak the subclass.
    """
    if isinstance(result, tuple):
        return tuple(_as_plain(item) for item in result)
    if not isinstance(result, FrozenMask):
        return result
    plain = torch.Tensor.as_subclass(
        result,
        torch.Tensor,
    )
    if result._is_view():
        return torch.Tensor.clone(plain)
    return plain


def freeze_mask(mask: torch.Tensor) -> FrozenMask:
    """
    Return a float32 FrozenMask copy of ``mask``.
    """
    if isinstance(mask, FrozenMask) and mask.dtype == torch.float32:
        return mask
    data = mask.detach().to(dtype=torch.float32).contiguous().clone()
    data.requires_grad_(False)
    return data.as_subclass(FrozenMask)
