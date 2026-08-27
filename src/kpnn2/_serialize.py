"""
Private reconstruction of spec edges as sorted pairs.
"""

import torch

from ._adjacency_spec import AdjacencySpec
from ._errors import Kpnn2Error
from ._spec import LayeredSpec


def canonical_edges(
    spec: LayeredSpec | AdjacencySpec,
) -> tuple[tuple[str, str], ...]:
    """
    Return every original edge as a sorted ``(source, target)``
    tuple.

    Pairs come from mask entries that equal ``1.0``. A
    ``LayeredSpec`` is read from hop masks only; skip metadata
    is not consulted. An ``AdjacencySpec`` is read from the
    square mask, including the diagonal.

    Parameters
    ----------
    spec : LayeredSpec or AdjacencySpec
        Spec whose masks store the original edges.

    Returns
    -------
    tuple of (str, str)
        Each original edge once, sorted lexicographically by
        ``(source, target)``.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is neither a ``LayeredSpec`` nor an
        ``AdjacencySpec``.
    """
    if isinstance(spec, LayeredSpec):
        return _layered_edges(spec)
    if isinstance(spec, AdjacencySpec):
        return _adjacency_edges(spec)
    raise Kpnn2Error("'spec' must be a LayeredSpec or an AdjacencySpec.")


def _layered_edges(
    spec: LayeredSpec,
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for hop in spec.hops:
        pairs.extend(
            _pairs_from_mask(
                hop.mask,
                hop.source_nodes,
                spec.layer_nodes[hop.target_layer],
            )
        )
    return tuple(sorted(pairs))


def _adjacency_edges(
    spec: AdjacencySpec,
) -> tuple[tuple[str, str], ...]:
    pairs = _pairs_from_mask(
        spec.mask,
        spec.nodes,
        spec.nodes,
    )
    return tuple(sorted(pairs))


def _pairs_from_mask(
    mask: torch.Tensor,
    source_names: tuple[str, ...],
    target_names: tuple[str, ...],
) -> list[tuple[str, str]]:
    rows, cols = (mask == 1.0).nonzero(as_tuple=True)
    pairs: list[tuple[str, str]] = []
    for row, col in zip(
        rows.tolist(),
        cols.tolist(),
        strict=True,
    ):
        pairs.append(
            (
                source_names[col],
                target_names[row],
            )
        )
    return pairs
