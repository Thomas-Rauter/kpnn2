"""Packed live-pair indices shared by the packed layers."""

from collections import Counter
from collections.abc import Iterable

import numpy as np
import torch

from ._errors import Kpnn2Error, caller_error, listed
from ._validate import describe, is_integer, reject_unordered

_INDEX_FORMS = "a 1-dimensional integer tensor or a sequence of int"


def _not_an_index(
    name: str,
    received: str,
) -> Kpnn2Error:
    return caller_error(
        f"'{name}' must be {_INDEX_FORMS}.",
        got=received,
    )


def copy_index(
    value: object,
    name: str,
) -> torch.Tensor:
    """
    Copy ``value`` to a 1-D int64 tensor.

    Accepts a 1-D integer ``torch.Tensor``, a 1-D
    ``numpy.ndarray`` of an integer dtype, or a sequence of
    ``int`` or numpy integer scalars. ``bool``, floating, and
    complex tensors or arrays are rejected, and so are sets and
    mappings, whose order is not the caller's. The result is
    contiguous and independent of ``value``.
    """
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise _not_an_index(
                name,
                describe(value),
            )
        if (
            value.is_floating_point()
            or value.is_complex()
            or value.dtype == torch.bool
        ):
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

    reject_unordered(
        value,
        name,
        _INDEX_FORMS,
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
    tensor, a 1-D integer numpy array, or a sequence (not a set
    or mapping) of ``int`` or numpy integer scalars, the two
    have the same length, they hold at least one entry, every
    entry is in range of its bound, and no ``(source, target)``
    pair repeats.

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
        raise caller_error(
            "'source_index' and 'target_index' must have the same length.",
            got=f"lengths {source.numel()} and {target.numel()}",
        )
    if source.numel() == 0:
        raise caller_error(
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
        counts = Counter(pairs)
        duplicated = sorted(pair for pair, count in counts.items() if count > 1)
        raise caller_error(
            f"{owner} indices contain {len(duplicated)} duplicate "
            "(source, target) pair(s).",
            got=listed(duplicated),
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

    The message names how many entries are outside the range, the
    first of them and its position, and the valid range.
    """
    outside = (index < 0) | (index >= bound)
    if not bool(outside.any()):
        return
    positions = outside.nonzero().flatten()
    first = int(positions[0])
    count = int(positions.numel())
    entries = "entry" if count == 1 else "entries"
    raise caller_error(
        f"'{name}' entries must satisfy 0 <= {name} < {bound_name}.",
        got=(
            f"{count} {entries} out of range, the first "
            f"{int(index[first])} at position {first}"
        ),
        expected=f"entries from 0 to {bound - 1}, as {bound_name}={bound}",
    )


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
