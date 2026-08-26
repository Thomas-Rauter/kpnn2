"""
Read-only connectivity tensors for GraphSpec and MaskedLinear.
"""

import numpy as np
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
        if _out_targets_frozen(kwargs.get("out")):
            raise Kpnn2Error(_READ_ONLY_MSG)
        result = super().__torch_function__(
            func,
            types,
            args,
            kwargs,
        )
        if isinstance(result, np.ndarray):
            return _readonly_ndarray(result)
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

    def numpy(
        self,
        *,
        force: bool = False,
    ) -> np.ndarray:
        plain = torch.Tensor.clone(
            torch.Tensor.as_subclass(
                self,
                torch.Tensor,
            )
        )
        arr = torch.Tensor.numpy(
            plain,
            force=force,
        )
        return _readonly_ndarray(arr)

    def __array__(
        self,
        dtype=None,
    ):
        arr = self.numpy()
        if dtype is None:
            return arr
        converted = arr.astype(
            dtype,
            copy=True,
        )
        converted.setflags(write=False)
        return converted


def _out_targets_frozen(out) -> bool:
    if isinstance(out, FrozenMask):
        return True
    if isinstance(out, (tuple, list)):
        return any(isinstance(item, FrozenMask) for item in out)
    return False


def _readonly_ndarray(arr: np.ndarray) -> np.ndarray:
    copied = np.array(
        arr,
        copy=True,
    )
    copied.setflags(write=False)
    return copied


def _as_plain(result):
    """
    Drop FrozenMask so arithmetic does not leak the subclass.

    Always clone so callers never receive a writable alias of
    stored connectivity storage.
    """
    if isinstance(result, tuple):
        return tuple(_as_plain(item) for item in result)
    if not isinstance(result, FrozenMask):
        return result
    plain = torch.Tensor.as_subclass(
        result,
        torch.Tensor,
    )
    return torch.Tensor.clone(plain)


def freeze_mask(mask: torch.Tensor) -> FrozenMask:
    """
    Return a new float32 FrozenMask copy of ``mask``.
    """
    if isinstance(mask, FrozenMask):
        mask = torch.Tensor.as_subclass(
            mask,
            torch.Tensor,
        )
    data = mask.detach().to(dtype=torch.float32).contiguous().clone()
    data.requires_grad_(False)
    return data.as_subclass(FrozenMask)
