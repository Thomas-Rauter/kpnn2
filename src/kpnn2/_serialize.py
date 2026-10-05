"""
Private reconstruction of spec edges as sorted pairs,
tagged dicts, and fingerprints.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence

import pandas as pd

from ._adjacency_spec import AdjacencySpec
from ._errors import Kpnn2Error, internal_error
from ._layout import build_layout, hop_axis_layouts
from ._spec import LayeredSpec
from ._validate import describe, require_node_mapping

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

    A ``LayeredSpec`` is read from packed hop index pairs;
    skip metadata is not consulted. An
    ``AdjacencySpec`` is read from packed
    ``(nodes[source_index[i]], nodes[target_index[i]])``
    pairs, including cycles and self-loops, never from a dense
    mask.

    Parameters
    ----------
    spec : LayeredSpec or AdjacencySpec
        Spec whose edges are stored as packed hop indices or
        packed state-vector indices.

    Returns
    -------
    tuple of (str, str)
        Each original edge once, sorted lexicographically by
        ``(source, target)``.

    Raises
    ------
    AssertionError
        From ``internal_error``, if ``spec`` is neither a
        ``LayeredSpec`` nor an ``AdjacencySpec``. Every caller
        passes a spec kpnn2 built.
    """
    if isinstance(spec, LayeredSpec):
        return _layered_edges(spec)
    if isinstance(spec, AdjacencySpec):
        return _adjacency_edges(spec)
    raise internal_error(
        f"canonical_edges received type {type(spec).__name__}, "
        "not a LayeredSpec or an AdjacencySpec"
    )


def spec_to_edgelist(
    spec: LayeredSpec | AdjacencySpec,
) -> pd.DataFrame:
    """
    Build a two-column edgelist from a spec's edges.

    Rows follow ``canonical_edges(spec)``: sorted
    lexicographically by ``(source, target)``, one row per
    original edge, names as strings. Columns are exactly
    ``source`` then ``target``. Extra columns from a pre-parse
    DataFrame are not reproduced.

    Parameters
    ----------
    spec : LayeredSpec or AdjacencySpec
        Spec whose edges are stored as packed indices.

    Returns
    -------
    pandas.DataFrame
        One row per original edge.

    Raises
    ------
    AssertionError
        From ``internal_error``, if ``spec`` is neither a
        ``LayeredSpec`` nor an ``AdjacencySpec``.
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
    A ``LayeredSpec`` with any node wider than 1 also includes
    ``"widths"``. When compacted layers differ from longest-path
    on the same edges, a ``LayeredSpec`` also includes
    ``"ranks"`` (every node, compacted 0-based index). An
    ``AdjacencySpec`` with any node wider than 1 also includes
    ``"widths"``; adjacency payloads never include ``"ranks"``.

    Parameters
    ----------
    spec : LayeredSpec or AdjacencySpec
        Spec whose edges are stored as packed indices.

    Returns
    -------
    dict
        A new dict. Nested ``edges`` lists are new as well.

    Raises
    ------
    AssertionError
        From ``internal_error``, if ``spec`` is neither a
        ``LayeredSpec`` nor an ``AdjacencySpec``.
    """
    edges = [list(pair) for pair in canonical_edges(spec)]
    if isinstance(spec, LayeredSpec):
        layout = _LAYOUT_LAYERED
        payload = {
            "kpnn2_spec": _SPEC_VERSION,
            "layout": layout,
            "edges": edges,
        }
        widths = _layered_widths_payload(spec)
        if widths:
            payload["widths"] = widths
        ranks = _layered_ranks_payload(spec)
        if ranks:
            payload["ranks"] = ranks
        return payload
    adjacency_payload: dict = {
        "kpnn2_spec": _SPEC_VERSION,
        "layout": _LAYOUT_ADJACENCY,
        "edges": edges,
    }
    adjacency_widths = {
        name: width
        for name, width in zip(
            spec.nodes,
            spec.node_widths,
            strict=True,
        )
        if width != 1
    }
    if adjacency_widths:
        adjacency_payload["widths"] = adjacency_widths
    return adjacency_payload


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
    AssertionError
        From ``internal_error``, if ``spec`` is neither a
        ``LayeredSpec`` nor an ``AdjacencySpec``.
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
    dict, passing ``payload["widths"]`` and ``payload["ranks"]``
    when present. Extra unknown keys are ignored.

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
    widths = _widths_from_payload(payload)
    ranks = _ranks_from_payload(payload)
    return parse_layered(
        table,
        widths=widths,
        ranks=ranks,
    )


def adjacency_spec_from_dict(payload: object) -> AdjacencySpec:
    """
    Rebuild an ``AdjacencySpec`` by parsing ``payload["edges"]``.

    Calls ``parse_adjacency`` on a DataFrame built from the
    tagged dict, passing ``payload["widths"]`` when present.
    Extra keys are ignored, including a stray ``"ranks"``.

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
    return parse_adjacency(
        table,
        widths=_widths_from_payload(payload),
    )


def _edgelist_from_payload(
    payload: object,
    expected_layout: str,
) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise Kpnn2Error(f"'payload' must be a dict. Got {describe(payload)}.")
    if "kpnn2_spec" not in payload:
        raise Kpnn2Error(
            "'kpnn2_spec' must be 1. The payload has no 'kpnn2_spec' key."
        )
    version = payload["kpnn2_spec"]
    if type(version) is not int or version != _SPEC_VERSION:
        raise Kpnn2Error(f"'kpnn2_spec' must be 1. Got {describe(version)}.")
    if "layout" not in payload:
        raise Kpnn2Error(
            "'layout' must be 'layered' or 'adjacency'. The payload has "
            "no 'layout' key."
        )
    layout = payload["layout"]
    if layout not in _KNOWN_LAYOUTS:
        raise Kpnn2Error(
            "'layout' must be 'layered' or 'adjacency'. Got "
            f"{describe(layout)}."
        )
    if layout != expected_layout:
        class_name = _LAYOUT_CLASS_NAME[expected_layout]
        raise Kpnn2Error(
            f"{class_name}.from_dict received layout "
            f"'{layout}'; expected '{expected_layout}'."
        )
    if "edges" not in payload:
        raise Kpnn2Error("'edges' is missing.")
    pairs = _pairs_from_edges(payload["edges"])
    # Names stay raw, so the parser rejects a missing one (None,
    # NaN) as it does in a caller's edgelist instead of reading
    # "None". Object dtype keeps 1 from becoming 1.0.
    return pd.DataFrame(
        pairs,
        columns=["source", "target"],
        dtype=object,
    )


def _pairs_from_edges(edges: object) -> list[list[object]]:
    if not isinstance(edges, Sequence) or isinstance(
        edges,
        _NON_PAIR_SEQUENCES,
    ):
        raise Kpnn2Error(
            "'edges' must be a sequence of [source, target] pairs. Got "
            f"{describe(edges)}."
        )
    pairs: list[list[object]] = []
    for position, pair in enumerate(edges):
        is_pair = (
            isinstance(pair, Sequence)
            and not isinstance(pair, _NON_PAIR_SEQUENCES)
            and len(pair) == 2
        )
        names = [str(name) for name in pair] if is_pair else []
        if not is_pair or "" in names:
            raise Kpnn2Error(
                "Each edge must be a pair of two nonempty names. Got "
                f"edges[{position}] = {describe(pair)}."
            )
        pairs.append(list(pair))
    return pairs


def _layered_widths_payload(spec: LayeredSpec) -> dict[str, int]:
    """Node name -> width for every node whose width is not 1."""
    widths: dict[str, int] = {}
    for names, sizes in zip(
        spec.layer_nodes,
        spec.layer_widths,
        strict=True,
    ):
        for name, width in zip(
            names,
            sizes,
            strict=True,
        ):
            if width != 1:
                widths[name] = width
    return widths


def _widths_from_payload(payload: object) -> Mapping[str, int] | None:
    """Read optional ``payload["widths"]``. Absent or empty is all 1."""
    if not isinstance(payload, dict):
        return None
    if "widths" not in payload:
        return None
    widths = payload["widths"]
    if widths is None:
        return None
    require_node_mapping(
        widths,
        "widths",
    )
    if len(widths) == 0:
        return None
    return widths


def _layered_ranks_payload(spec: LayeredSpec) -> dict[str, int] | None:
    """
    Compacted depth of every node, or ``None`` when that equals
    longest-path on the same edges.

    ``_rank_layers`` validates caller edgelists inside
    ``parse_layered``. Here it re-ranks the spec's own edges, so
    its ``Kpnn2Error`` would be a kpnn2 bug and is raised through
    ``internal_error`` instead.
    """
    from ._parse import (
        _build_adjacency,
        _rank_layers,
    )

    table = spec_to_edgelist(spec)
    (
        nodes,
        children,
        parents,
        in_degree,
        _,
    ) = _build_adjacency(table)
    try:
        default_layers = _rank_layers(
            nodes,
            children,
            parents,
            in_degree,
        )
    except Kpnn2Error as error:
        raise internal_error(
            f"re-ranking a LayeredSpec's own edges failed: {error}"
        ) from error
    default = tuple(tuple(layer) for layer in default_layers)
    if default == spec.layer_nodes:
        return None
    ranks: dict[str, int] = {}
    for depth, names in enumerate(spec.layer_nodes):
        for name in names:
            ranks[name] = depth
    return ranks


def _ranks_from_payload(payload: object) -> Mapping[str, int] | None:
    """Read optional ``payload["ranks"]``. Absent is longest-path."""
    if not isinstance(payload, dict):
        return None
    if "ranks" not in payload:
        return None
    ranks = payload["ranks"]
    if ranks is None:
        return None
    require_node_mapping(
        ranks,
        "ranks",
    )
    return ranks


def _layered_edges(
    spec: LayeredSpec,
) -> tuple[tuple[str, str], ...]:
    pairs: set[tuple[str, str]] = set()
    for hop in spec.hops:
        source_layout, target_layout = hop_axis_layouts(
            spec.layer_nodes,
            spec.layer_widths,
            hop.source_layers,
            hop.target_layer,
        )
        for source_unit, target_unit in zip(
            hop.source_index,
            hop.target_index,
            strict=True,
        ):
            source_name = source_layout.slot_containing(
                source_unit,
            ).name
            target_name = target_layout.slot_containing(
                target_unit,
            ).name
            pairs.add(
                (
                    source_name,
                    target_name,
                )
            )
    return tuple(sorted(pairs))


def _adjacency_edges(
    spec: AdjacencySpec,
) -> tuple[tuple[str, str], ...]:
    layout = build_layout(
        spec.nodes,
        spec.node_widths,
    )
    pairs: set[tuple[str, str]] = set()
    for source, target in zip(
        spec.source_index,
        spec.target_index,
        strict=True,
    ):
        pairs.add(
            (
                layout.slot_containing(source).name,
                layout.slot_containing(target).name,
            )
        )
    return tuple(sorted(pairs))
