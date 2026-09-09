"""
Private reconstruction of spec edges as sorted pairs,
tagged dicts, and fingerprints.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence

import pandas as pd

from ._adjacency_spec import AdjacencySpec
from ._errors import Kpnn2Error
from ._layout import build_layout, concat_layouts
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
    A ``LayeredSpec`` with any node wider than 1 also includes
    ``"widths"``. Adjacency payloads never include ``"widths"``.

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
    Kpnn2Error
        If ``spec`` is neither a ``LayeredSpec`` nor an
        ``AdjacencySpec``.
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
        return payload
    return {
        "kpnn2_spec": _SPEC_VERSION,
        "layout": _LAYOUT_ADJACENCY,
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
    dict, passing ``payload["widths"]`` when present. Extra
    unknown keys are ignored.

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
    return parse_layered(
        table,
        widths=widths,
    )


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
    if not isinstance(widths, Mapping):
        raise Kpnn2Error("'widths' must be a mapping of node name to int.")
    if len(widths) == 0:
        return None
    return widths


def _layered_edges(
    spec: LayeredSpec,
) -> tuple[tuple[str, str], ...]:
    pairs: set[tuple[str, str]] = set()
    for hop in spec.hops:
        source_layout = concat_layouts(
            [
                build_layout(
                    spec.layer_nodes[layer],
                    spec.layer_widths[layer],
                )
                for layer in hop.source_layers
            ]
        )
        target_layout = build_layout(
            spec.layer_nodes[hop.target_layer],
            spec.layer_widths[hop.target_layer],
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
    pairs: list[tuple[str, str]] = []
    for source, target in zip(
        spec.source_index,
        spec.target_index,
    ):
        pairs.append(
            (
                spec.nodes[source],
                spec.nodes[target],
            )
        )
    return tuple(sorted(pairs))
