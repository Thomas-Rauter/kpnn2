"""Packed live-pair indices shared by the packed layers."""

from collections.abc import Iterable

import torch

from ._errors import Kpnn2Error
from ._validate import is_integer


def copy_index(
    value: object,
    name: str,
) -> torch.Tensor:
    """
    Copy ``value`` to a 1-D int64 tensor.

    Accepts a 1-D integer ``torch.Tensor`` or a sequence of
    ``int``. The result is contiguous and independent of
    ``value``.
    """
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise Kpnn2Error(
                f"'{name}' must be a 1-dimensional integer "
                "tensor or a sequence of int."
            )
        if value.is_floating_point() or value.dtype == torch.bool:
            raise Kpnn2Error(
                f"'{name}' must be a 1-dimensional integer "
                "tensor or a sequence of int."
            )
        return value.detach().to(dtype=torch.int64).contiguous().clone()

    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise Kpnn2Error(
            f"'{name}' must be a 1-dimensional integer tensor "
            "or a sequence of int."
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise Kpnn2Error(
            f"'{name}' must be a 1-dimensional integer tensor "
            "or a sequence of int."
        ) from exc
    for item in items:
        if not is_integer(item):
            raise Kpnn2Error(
                f"'{name}' must be a 1-dimensional integer "
                "tensor or a sequence of int."
            )
    return torch.tensor(
        items,
        dtype=torch.int64,
    )


def as_packed_pairs(
    source_index: object,
    target_index: object,
    *,
    source_bound: int,
    target_bound: int,
    source_bound_name: str,
    target_bound_name: str,
    owner: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Copy and check the packed ``(source, target)`` index pair.

    Checks run in a fixed order: each index is a 1-D integer
    tensor or sequence of ``int``, the two have the same length,
    they hold at least one entry, every entry is in range of its
    bound, and no ``(source, target)`` pair repeats.

    Parameters
    ----------
    source_index, target_index : object
        Caller indices, as passed to the layer constructor.
    source_bound, target_bound : int
        Exclusive upper bounds of the source and target entries.
    source_bound_name, target_bound_name : str
        Constructor argument names of the two bounds, used in
        error messages.
    owner : str
        Layer class name, used in the duplicate-pair message.

    Returns
    -------
    tuple of torch.Tensor
        ``(source, target)`` as contiguous int64 copies,
        independent of the caller's objects.

    Raises
    ------
    Kpnn2Error
        If any of the checks above fails.
    """
    source = copy_index(
        source_index,
        "source_index",
    )
    target = copy_index(
        target_index,
        "target_index",
    )
    if source.shape != target.shape:
        raise Kpnn2Error(
            "'source_index' and 'target_index' must have the same length."
        )
    if source.numel() == 0:
        raise Kpnn2Error(
            "'source_index' and 'target_index' must contain at least one index."
        )
    if torch.any(source < 0) or torch.any(source >= source_bound):
        raise Kpnn2Error(
            "'source_index' entries must satisfy "
            f"0 <= source_index < {source_bound_name}."
        )
    if torch.any(target < 0) or torch.any(target >= target_bound):
        raise Kpnn2Error(
            "'target_index' entries must satisfy "
            f"0 <= target_index < {target_bound_name}."
        )
    pairs = list(
        zip(
            source.tolist(),
            target.tolist(),
        )
    )
    if len(set(pairs)) != len(pairs):
        raise Kpnn2Error(
            f"{owner} indices contain duplicate (source, target) pair(s)."
        )
    return source, target


def digest_matches(
    saved: object,
    current: torch.Tensor,
) -> bool:
    if not isinstance(saved, torch.Tensor):
        return False
    saved_flat = saved.detach().cpu().contiguous().reshape(-1)
    if saved_flat.shape != current.shape or saved_flat.dtype != current.dtype:
        return False
    return bool(
        torch.equal(
            saved_flat,
            current,
        )
    )
