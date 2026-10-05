"""Packed live-pair indices shared by the packed layers."""

from collections import Counter
from collections.abc import Iterable

import numpy as np
import torch

from ._errors import Kpnn2Error
from ._validate import describe, is_integer

_NOT_INDEX = "must be a 1-dimensional integer tensor or a sequence of int."

# A duplicate-pair message lists at most this many pairs.
_DUPLICATES_SHOWN = 5


def _not_an_index(
    name: str,
    received: str,
) -> Kpnn2Error:
    return Kpnn2Error(f"'{name}' {_NOT_INDEX} Got {received}.")


def copy_index(
    value: object,
    name: str,
) -> torch.Tensor:
    """
    Copy ``value`` to a 1-D int64 tensor.

    Accepts a 1-D integer ``torch.Tensor``, a 1-D
    ``numpy.ndarray`` of an integer dtype, or a sequence of
    ``int`` or numpy integer scalars. ``bool`` and floating
    tensors or arrays are rejected. The result is contiguous and
    independent of ``value``.
    """
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise _not_an_index(
                name,
                describe(value),
            )
        if value.is_floating_point() or value.dtype == torch.bool:
            raise _not_an_index(
                name,
                describe(value),
            )
        return value.detach().to(dtype=torch.int64).contiguous().clone()

    if isinstance(value, np.ndarray):
        if value.ndim != 1 or not np.issubdtype(
            value.dtype,
            np.integer,
        ):
            raise _not_an_index(
                name,
                describe(value),
            )
        return torch.from_numpy(
            value.astype(
                np.int64,
                order="C",
            )
        )

    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise _not_an_index(
            name,
            describe(value),
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise _not_an_index(
            name,
            describe(value),
        ) from exc
    for position, item in enumerate(items):
        if not is_integer(item):
            raise _not_an_index(
                name,
                f"{describe(item)} at position {position}",
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
    tensor, a 1-D integer numpy array, or a sequence of ``int``
    or numpy integer scalars, the two have the same length,
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
            "'source_index' and 'target_index' must have the same length. "
            f"Got {source.numel()} and {target.numel()}."
        )
    if source.numel() == 0:
        raise Kpnn2Error(
            "'source_index' and 'target_index' must contain at least one index."
        )
    _check_in_range(
        source,
        "source_index",
        source_bound,
        source_bound_name,
    )
    _check_in_range(
        target,
        "target_index",
        target_bound,
        target_bound_name,
    )
    pairs = list(
        zip(
            source.tolist(),
            target.tolist(),
        )
    )
    if len(set(pairs)) != len(pairs):
        raise Kpnn2Error(
            f"{owner} indices contain duplicate (source, target) pair(s). "
            f"Got {_duplicated_pairs(pairs)}."
        )
    return source, target


def _check_in_range(
    index: torch.Tensor,
    name: str,
    bound: int,
    bound_name: str,
) -> None:
    """
    Raise unless every entry of ``index`` is in ``[0, bound)``.

    The message names the bound and its value, the first entry
    outside the range and its position, and how many there are.
    """
    outside = (index < 0) | (index >= bound)
    if not bool(outside.any()):
        return
    positions = outside.nonzero().flatten()
    first = int(positions[0])
    count = int(positions.numel())
    entries = "entry" if count == 1 else "entries"
    raise Kpnn2Error(
        f"'{name}' entries must satisfy 0 <= {name} < {bound_name}. "
        f"Got {count} {entries} out of range with {bound_name}={bound}; "
        f"the first is {int(index[first])} at position {first}."
    )


def _duplicated_pairs(pairs: list[tuple[int, int]]) -> str:
    """Count the pairs that repeat and list the first few, sorted."""
    counts = Counter(pairs)
    duplicated = sorted(pair for pair, count in counts.items() if count > 1)
    text = ", ".join(str(pair) for pair in duplicated[:_DUPLICATES_SHOWN])
    hidden = len(duplicated) - _DUPLICATES_SHOWN
    if hidden > 0:
        text = f"{text} (and {hidden} more)"
    noun = "pair" if len(duplicated) == 1 else "pairs"
    return f"{len(duplicated)} duplicated {noun}: {text}"


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
