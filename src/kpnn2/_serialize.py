"""
Private reconstruction of spec edges as sorted pairs,
tagged dicts, and fingerprints.
"""

import hashlib
import json
from collections.abc import Sequence

import pandas as pd
import torch

from ._adjacency_spec import AdjacencySpec
from ._errors import Kpnn2Error
from ._spec import LayeredSpec

_SPEC_VERSION = 1
_LAYOUT_LAYERED = "layered"
_LAYOUT_ADJACENCY = "adjacency"
_KNOWN_LAYOUTS = (
    _LAYOUT_LAYERED,
    _LAYOUT_ADJACENCY,
)
_LAYOUT_CLASS_NAME = {
    _LAYOUT_LAYERED: "LayeredSpec",
    _LAYOUT_ADJACENCY: "AdjacencySpec",
}
_NON_PAIR_SEQUENCES = (
    str,
    bytes,
    bytearray,
)


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


def spec_to_edgelist(
    spec: LayeredSpec | AdjacencySpec,
) -> pd.DataFrame:
    """
    Build a two-column edgelist from a spec's masks.

    Rows follow ``canonical_edges(spec)``: sorted
    lexicographically by ``(source, target)``, one row per
    original edge, names as strings. Columns are exactly
    ``source`` then ``target``. Extra columns from a pre-parse
    DataFrame are not reproduced.

    Parameters
    ----------
    spec : LayeredSpec or AdjacencySpec
        Spec whose masks store the original edges.

    Returns
    -------
    pandas.DataFrame
        One row per original edge.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is neither a ``LayeredSpec`` nor an
        ``AdjacencySpec``.
    """
    edges = canonical_edges(spec)
    return pd.DataFrame(
        edges,
        columns=["source", "target"],
    )


def spec_to_dict(
    spec: LayeredSpec | AdjacencySpec,
) -> dict:
    """
    Build a JSON-safe tagged dict from a spec.

    Keys are ``kpnn2_spec`` (integer ``1``), ``layout``
    (``"layered"`` or ``"adjacency"``), and ``edges`` (list of
    ``[source, target]`` lists in ``canonical_edges`` order).

    Parameters
    ----------
    spec : LayeredSpec or AdjacencySpec
        Spec whose masks store the original edges.

    Returns
    -------
    dict
        A new dict. Nested ``edges`` lists are new as well.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is neither a ``LayeredSpec`` nor an
        ``AdjacencySpec``.
    """
    edges = [list(pair) for pair in canonical_edges(spec)]
    if isinstance(spec, LayeredSpec):
        layout = _LAYOUT_LAYERED
    else:
        layout = _LAYOUT_ADJACENCY
    return {
        "kpnn2_spec": _SPEC_VERSION,
        "layout": layout,
        "edges": edges,
    }


def spec_fingerprint(
    spec: LayeredSpec | AdjacencySpec,
) -> str:
    """
    SHA-256 hex digest of the canonical ``spec_to_dict`` JSON.

    The payload is ``json.dumps(..., sort_keys=True,
    separators=(",", ":"), ensure_ascii=False)`` encoded as
    UTF-8. The result is 64 lowercase hex characters.

    Parameters
    ----------
    spec : LayeredSpec or AdjacencySpec
        Spec to hash.

    Returns
    -------
    str
        Hex digest of the tagged spec dict.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is neither a ``LayeredSpec`` nor an
        ``AdjacencySpec``.
    """
    payload = json.dumps(
        spec_to_dict(spec),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def layered_spec_from_dict(payload: object) -> LayeredSpec:
    """
    Rebuild a ``LayeredSpec`` by parsing ``payload["edges"]``.

    Calls ``parse_layered`` on a DataFrame built from the tagged
    dict. Extra keys are ignored.

    Parameters
    ----------
    payload : dict
        Output of ``LayeredSpec.to_dict()``, or a compatible
        dict with ``kpnn2_spec``, ``layout``, and ``edges``.

    Returns
    -------
    LayeredSpec
        The parsed spec.

    Raises
    ------
    Kpnn2Error
        If ``payload`` is invalid or ``layout`` is not
        ``"layered"``.
    """
    from ._parse import parse_layered

    table = _edgelist_from_payload(
        payload,
        expected_layout=_LAYOUT_LAYERED,
    )
    return parse_layered(table)


def adjacency_spec_from_dict(payload: object) -> AdjacencySpec:
    """
    Rebuild an ``AdjacencySpec`` by parsing ``payload["edges"]``.

    Calls ``parse_adjacency`` on a DataFrame built from the
    tagged dict. Extra keys are ignored.

    Parameters
    ----------
    payload : dict
        Output of ``AdjacencySpec.to_dict()``, or a compatible
        dict with ``kpnn2_spec``, ``layout``, and ``edges``.

    Returns
    -------
    AdjacencySpec
        The parsed spec.

    Raises
    ------
    Kpnn2Error
        If ``payload`` is invalid or ``layout`` is not
        ``"adjacency"``.
    """
    from ._parse_adjacency import parse_adjacency

    table = _edgelist_from_payload(
        payload,
        expected_layout=_LAYOUT_ADJACENCY,
    )
    return parse_adjacency(table)


def _edgelist_from_payload(
    payload: object,
    expected_layout: str,
) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise Kpnn2Error("'payload' must be a dict.")
    version = payload.get("kpnn2_spec")
    if type(version) is not int or version != _SPEC_VERSION:
        raise Kpnn2Error("'kpnn2_spec' must be 1.")
    if "layout" not in payload:
        raise Kpnn2Error("'layout' must be 'layered' or 'adjacency'.")
    layout = payload["layout"]
    if layout not in _KNOWN_LAYOUTS:
        raise Kpnn2Error("'layout' must be 'layered' or 'adjacency'.")
    if layout != expected_layout:
        class_name = _LAYOUT_CLASS_NAME[expected_layout]
        raise Kpnn2Error(
            f"{class_name}.from_dict received layout "
            f"'{layout}'; expected '{expected_layout}'."
        )
    if "edges" not in payload:
        raise Kpnn2Error("'edges' is missing.")
    pairs = _pairs_from_edges(payload["edges"])
    return pd.DataFrame(
        pairs,
        columns=["source", "target"],
    )


def _pairs_from_edges(edges: object) -> list[list[str]]:
    if not isinstance(edges, Sequence) or isinstance(
        edges,
        _NON_PAIR_SEQUENCES,
    ):
        raise Kpnn2Error(
            "'edges' must be a sequence of [source, target] pairs."
        )
    pairs: list[list[str]] = []
    for pair in edges:
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, _NON_PAIR_SEQUENCES)
            or len(pair) != 2
        ):
            raise Kpnn2Error("Each edge must be a pair of two nonempty names.")
        source, target = pair
        source_name = str(source)
        target_name = str(target)
        if source_name == "" or target_name == "":
            raise Kpnn2Error("Each edge must be a pair of two nonempty names.")
        pairs.append(
            [
                source_name,
                target_name,
            ]
        )
    return pairs


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
