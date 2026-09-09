"""
Edgelist parsing for kpnn2.
"""

from collections import deque
from collections.abc import Mapping

import pandas as pd

from ._errors import Kpnn2Error
from ._layout import (
    Layout,
    NodeSlot,
    build_layout,
    concat_layouts,
    iter_block_pairs,
)
from ._spec import Hop, LayeredSpec, Skip

_SOURCE = "source"
_TARGET = "target"


def _validate_edgelist(edgelist: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a source/target edgelist and return a normalized copy.

    Node names are converted to strings. Only ``source`` and ``target``
    are kept. Extra columns are ignored.

    Parameters
    ----------
    edgelist
        Edge table with required columns ``source`` and ``target``.

    Returns
    -------
    pd.DataFrame
        Copy with string ``source`` and ``target`` columns.

    Raises
    ------
    Kpnn2Error
        If ``edgelist`` is not a DataFrame, required columns are
        missing, values are missing or empty, the table has no rows,
        or ``(source, target)`` pairs are duplicated (message
        names the unique pairs, sorted).
    """
    if not isinstance(edgelist, pd.DataFrame):
        raise Kpnn2Error("'edgelist' must be a pandas DataFrame.")

    missing_columns = [
        name for name in (_SOURCE, _TARGET) if name not in edgelist.columns
    ]
    if missing_columns:
        missing_str = ", ".join(missing_columns)
        raise Kpnn2Error(
            "Edgelist must contain columns 'source' and 'target'. "
            f"Missing: {missing_str}."
        )

    if len(edgelist) == 0:
        raise Kpnn2Error("Edgelist must contain at least one edge.")

    source = edgelist[_SOURCE]
    target = edgelist[_TARGET]
    if source.isna().any() or target.isna().any():
        raise Kpnn2Error(
            "Edgelist contains missing values in 'source' or 'target'."
        )

    normalized = pd.DataFrame(
        {
            _SOURCE: source.map(str),
            _TARGET: target.map(str),
        }
    )

    empty_source = normalized[_SOURCE] == ""
    empty_target = normalized[_TARGET] == ""
    if empty_source.any() or empty_target.any():
        raise Kpnn2Error(
            "Edgelist contains empty node names in 'source' or 'target'."
        )

    n_duplicates = int(normalized.duplicated().sum())
    if n_duplicates > 0:
        duplicated_rows = normalized[normalized.duplicated(keep=False)]
        unique_pairs = duplicated_rows.drop_duplicates().sort_values(
            by=[_SOURCE, _TARGET],
        )
        pair_labels = [
            f"{source} -> {target}"
            for source, target in zip(
                unique_pairs[_SOURCE],
                unique_pairs[_TARGET],
            )
        ]
        pairs_str = ", ".join(pair_labels)
        raise Kpnn2Error(
            f"Edgelist contains {n_duplicates} duplicate edge(s): "
            f"{pairs_str}. At most one connection is allowed for each "
            "source-target pair."
        )

    return pd.DataFrame(
        {
            _SOURCE: normalized[_SOURCE].tolist(),
            _TARGET: normalized[_TARGET].tolist(),
        }
    )


def _reject_self_loops(edgelist: pd.DataFrame) -> None:
    """
    Raise if any edge has the same source and target.

    Parameters
    ----------
    edgelist
        Normalized edgelist with string ``source`` and ``target``.

    Raises
    ------
    Kpnn2Error
        If one or more self-loops are present. The message names
        the unique self-loop nodes, sorted alphabetically.
    """
    self_loops = edgelist[_SOURCE] == edgelist[_TARGET]
    n_self_loops = int(self_loops.sum())
    if n_self_loops > 0:
        loop_nodes = sorted(set(edgelist[_SOURCE][self_loops].tolist()))
        nodes_str = ", ".join(loop_nodes)
        raise Kpnn2Error(
            f"Edgelist contains {n_self_loops} self-loop(s): "
            f"{nodes_str}. Self-loops are not allowed."
        )


def _build_adjacency(
    edgelist: pd.DataFrame,
) -> tuple[
    set[str],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, int],
    dict[str, int],
]:
    """
    Build parent/child adjacency and degree maps.

    Parameters
    ----------
    edgelist
        Normalized edgelist with string ``source`` and ``target``.

    Returns
    -------
    tuple
        ``nodes``, ``children``, ``parents``, ``in_degree``,
        ``out_degree``.
    """
    sources = edgelist[_SOURCE].tolist()
    targets = edgelist[_TARGET].tolist()
    nodes = set(sources) | set(targets)
    children: dict[str, list[str]] = {node: [] for node in nodes}
    parents: dict[str, list[str]] = {node: [] for node in nodes}
    for source, target in zip(
        sources,
        targets,
    ):
        children[source].append(target)
        parents[target].append(source)
    in_degree = {node: len(parents[node]) for node in nodes}
    out_degree = {node: len(children[node]) for node in nodes}
    return (
        nodes,
        children,
        parents,
        in_degree,
        out_degree,
    )


def _node_sets(
    nodes: set[str],
    in_degree: dict[str, int],
    out_degree: dict[str, int],
) -> tuple[list[str], list[str], list[str]]:
    """
    Infer sorted input, output, and hidden node lists.

    Parameters
    ----------
    nodes
        All node names in the edgelist.
    in_degree
        Incoming edge counts.
    out_degree
        Outgoing edge counts.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        ``input_nodes``, ``output_nodes``, ``hidden_nodes``.

    Raises
    ------
    Kpnn2Error
        If there is no in-degree-0 node or no out-degree-0 node.
    """
    input_nodes = sorted(node for node in nodes if in_degree[node] == 0)
    output_nodes = sorted(node for node in nodes if out_degree[node] == 0)
    if not input_nodes:
        raise Kpnn2Error(
            "Edgelist must contain at least one input node (in-degree 0)."
        )
    if not output_nodes:
        raise Kpnn2Error(
            "Edgelist must contain at least one output node (out-degree 0)."
        )
    input_set = set(input_nodes)
    output_set = set(output_nodes)
    hidden_nodes = sorted(
        node
        for node in nodes
        if node not in input_set and node not in output_set
    )
    return (
        input_nodes,
        output_nodes,
        hidden_nodes,
    )


def _rank_layers(
    nodes: set[str],
    children: dict[str, list[str]],
    parents: dict[str, list[str]],
    in_degree: dict[str, int],
) -> list[list[str]]:
    """
    Assign Kahn depths and return alphabetically sorted layers.

    ``depth(input) = 0``. For every other node,
    ``depth = 1 + max(parent depths)``.

    Parameters
    ----------
    nodes
        All node names.
    children
        Adjacency list from source to targets.
    parents
        Adjacency list from target to sources.
    in_degree
        Incoming edge counts (not mutated).

    Returns
    -------
    list[list[str]]
        ``layer_nodes[d]`` is sorted names with depth ``d``.

    Raises
    ------
    Kpnn2Error
        If a cycle leaves some nodes unranked. The message names
        every leftover node (``nodes`` minus keys of ``depths``),
        sorted alphabetically.
    """
    remaining = dict(in_degree)
    depths: dict[str, int] = {}
    ready: deque[str] = deque(node for node in nodes if remaining[node] == 0)
    while ready:
        node = ready.popleft()
        if not parents[node]:
            depths[node] = 0
        else:
            depths[node] = 1 + max(depths[parent] for parent in parents[node])
        for child in children[node]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)

    if len(depths) != len(nodes):
        unranked = sorted(nodes - depths.keys())
        unranked_str = ", ".join(unranked)
        raise Kpnn2Error(
            "Edgelist contains a cycle. Only DAGs are supported. "
            f"Unranked nodes: {unranked_str}."
        )

    n_layers = max(depths.values()) + 1
    layer_nodes: list[list[str]] = [[] for _ in range(n_layers)]
    for node, depth in depths.items():
        layer_nodes[depth].append(node)
    for layer in layer_nodes:
        layer.sort()
    return layer_nodes


def _normalize_widths(
    widths: Mapping[str, int] | None,
    nodes: set[str],
) -> dict[str, int]:
    """
    Map each graph node to a positive unit count.

    ``None`` or omitted keys are width 1. Keys are matched after
    ``str(...)``. Unknown names and non-positive or non-int
    values raise ``Kpnn2Error``. ``bool`` is rejected.
    """
    if widths is None:
        return {node: 1 for node in nodes}
    if not isinstance(widths, Mapping):
        raise Kpnn2Error("'widths' must be a mapping of node name to int.")
    requested: dict[str, int] = {}
    for key, value in widths.items():
        name = str(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise Kpnn2Error(
                f"Width for node {name!r} must be a positive int. "
                f"Got {value!r}."
            )
        if value < 1:
            raise Kpnn2Error(
                f"Width for node {name!r} must be a positive int. "
                f"Got {value!r}."
            )
        requested[name] = value
    unknown = sorted(set(requested) - nodes)
    if unknown:
        names_str = ", ".join(unknown)
        raise Kpnn2Error(f"Unknown node name(s) in 'widths': {names_str}.")
    return {node: requested.get(node, 1) for node in nodes}


def _layer_layouts(
    layer_nodes: list[list[str]],
    width_of: dict[str, int],
) -> list[Layout]:
    """
    Place each layer's nodes on that layer's unit axis.

    One layout per depth. ``layout.n_units`` is the sum of the
    node widths at that depth.
    """
    return [
        build_layout(
            names,
            [width_of[name] for name in names],
        )
        for names in layer_nodes
    ]


def _node_placement(
    layouts: list[Layout],
) -> dict[str, tuple[int, NodeSlot]]:
    """
    Map each node name to its ``(depth, slot)``.
    """
    placement: dict[str, tuple[int, NodeSlot]] = {}
    for depth, layout in enumerate(layouts):
        for slot in layout.slots:
            placement[slot.name] = (depth, slot)
    return placement


def _parent_layers(
    edgelist: pd.DataFrame,
    placement: dict[str, tuple[int, NodeSlot]],
    n_layers: int,
) -> list[set[int]]:
    """
    Collect, per depth, the depths that feed it.

    Entry ``d`` is every depth with at least one edge into depth
    ``d``. Longest-path ranking guarantees ``d - 1`` is in it for
    every ``d > 0``, and that entry 0 is empty.
    """
    parents: list[set[int]] = [set() for _ in range(n_layers)]
    for source, target in zip(
        edgelist[_SOURCE].tolist(),
        edgelist[_TARGET].tolist(),
    ):
        source_layer, _ = placement[source]
        target_layer, _ = placement[target]
        parents[target_layer].add(source_layer)
    return parents


def _build_hops(
    edgelist: pd.DataFrame,
    layouts: list[Layout],
    placement: dict[str, tuple[int, NodeSlot]],
) -> list[Hop]:
    """
    Build one incoming packed hop per depth after the first.

    ``hops[i]`` targets depth ``i + 1`` and its packed indices
    hold every edge entering that depth, adjacent or skip. The
    column axis is the source depths concatenated in ascending
    order. Each named edge expands into every unit pair of the
    ``(target.width, source.width)`` block. At width 1 that is
    one pair per original edge.

    Only depths that really feed the target become columns, so a
    graph without skips gives exactly one source depth per hop.
    Does not allocate an ``(out, in)`` tensor.

    Parameters
    ----------
    edgelist
        Normalized edgelist with string ``source`` and ``target``.
    layouts
        Unit placement per depth.
    placement
        Node name to ``(depth, slot)``.

    Returns
    -------
    list[Hop]
        One hop per depth from 1 upwards, in depth order.
    """
    n_layers = len(layouts)
    parents = _parent_layers(
        edgelist,
        placement,
        n_layers,
    )

    source_layers: dict[int, tuple[int, ...]] = {}
    source_layouts: dict[int, Layout] = {}
    packed: dict[int, list[tuple[str, str, NodeSlot, NodeSlot]]] = {}
    for target_layer in range(1, n_layers):
        ordered = tuple(sorted(parents[target_layer]))
        source_layout = concat_layouts(
            [layouts[layer] for layer in ordered],
        )
        source_layers[target_layer] = ordered
        source_layouts[target_layer] = source_layout
        packed[target_layer] = []

    for source, target in zip(
        edgelist[_SOURCE].tolist(),
        edgelist[_TARGET].tolist(),
        strict=True,
    ):
        target_layer, target_slot = placement[target]
        source_slot = source_layouts[target_layer].slot(source)
        packed[target_layer].append(
            (
                source,
                target,
                source_slot,
                target_slot,
            )
        )

    hops: list[Hop] = []
    for target_layer in range(1, n_layers):
        ordered = source_layers[target_layer]
        rows = sorted(
            packed[target_layer],
            key=lambda row: (row[0], row[1]),
        )
        source_index: list[int] = []
        target_index: list[int] = []
        for _, _, source_slot, target_slot in rows:
            for source_unit, target_unit in iter_block_pairs(
                source_slot,
                target_slot,
            ):
                source_index.append(source_unit)
                target_index.append(target_unit)
        hops.append(
            Hop(
                target_layer=target_layer,
                source_layers=ordered,
                source_dims=tuple(layouts[layer].n_units for layer in ordered),
                source_nodes=source_layouts[target_layer].names,
                target_dim=layouts[target_layer].n_units,
                source_index=tuple(source_index),
                target_index=tuple(target_index),
            )
        )
    return hops


def _build_skips(
    edgelist: pd.DataFrame,
    placement: dict[str, tuple[int, NodeSlot]],
) -> list[Skip]:
    """
    Collect original edges with depth gap greater than 1.

    These records are metadata: the named edges themselves are
    already unit-pair blocks in the target depth's hop. Adjacent
    edges (gap exactly 1) are omitted, since nothing
    distinguishes them. Each recorded index is the first unit
    its node owns, which equals the node's ordinal in
    ``layer_nodes`` while every node is one unit wide.

    Parameters
    ----------
    edgelist
        Normalized edgelist with string ``source`` and ``target``.
    placement
        Node name to ``(depth, slot)``.

    Returns
    -------
    list[Skip]
        One record per skip edge, in edgelist order.
    """
    skips: list[Skip] = []
    sources = edgelist[_SOURCE].tolist()
    targets = edgelist[_TARGET].tolist()
    for source, target in zip(
        sources,
        targets,
    ):
        source_layer, source_slot = placement[source]
        target_layer, target_slot = placement[target]
        gap = target_layer - source_layer
        if gap > 1:
            skips.append(
                Skip(
                    source=source,
                    target=target,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    source_in_layer=source_slot.start,
                    target_in_layer=target_slot.start,
                )
            )
    return skips


def parse_layered(
    edgelist: pd.DataFrame,
    *,
    widths: Mapping[str, int] | None = None,
) -> LayeredSpec:
    """
    Parse a source/target edgelist into a ``LayeredSpec``.

    Prior-knowledge edges, such as genes into pathways, become
    depth-ranked layers plus one packed hop per layer, ready for
    a ``PackedLinear`` stack. Reach for it when a DAG should
    become one hop per layer; ``parse_adjacency`` is the
    shared-state packed alternative, which also allows cycles.
    Depth is longest path from inputs, names sort alphabetically
    within a layer, and no hop mask is allocated. Optional
    ``widths`` lets a named node own several units.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge table with required columns ``source`` and ``target``,
        one row per directed edge in the direction of computation.
        Names are converted with ``str(...)``; extra columns are
        ignored. The frame is read, never modified.
    widths : mapping of str to int, optional
        Units per named node. Omitted names, ``None``, and an
        empty mapping are width 1. Keys are matched after
        ``str(...)``. Unknown names raise ``Kpnn2Error``. Values
        must be positive ints; ``bool``, ``0``, and negatives
        are rejected.

    Returns
    -------
    LayeredSpec
        Frozen structure: layers, one ``Hop`` per layer after the
        first, and ``skips`` metadata. ``skips`` is empty when no
        edge spans more than one layer; ``hops`` never is, because
        a valid edgelist always yields at least two layers.

    Raises
    ------
    Kpnn2Error
        If ``edgelist`` is not a DataFrame; ``source`` or ``target``
        is absent, missing, or an empty name; the table has no rows;
        a ``(source, target)`` pair is duplicated; any edge is a
        self-loop; the graph has a cycle; there is no in-degree-0
        node or no out-degree-0 node; or ``widths`` names an
        unknown node or is not a mapping of positive ints. Each
        message names the offending pairs or nodes, sorted.

    See Also
    --------
    parse_adjacency : Pack the same table into one state vector;
        allows cycles and self-loops. Has no ``widths`` argument.
    PackedLinear : Apply one hop from its packed indices.
    gather_hop_inputs : Build one hop's input from the saved layer
        tensors.

    Notes
    -----
    Every named edge belongs to exactly one hop, the one of its
    target layer, whether its depth gap is 1 or larger. Packed
    indices are in unit space: named edge ``A -> B`` expands
    into a ``(k_B, k_A)`` block of live pairs. At default width
    1 that is one pair per named edge. A hop whose target has
    parents further back reads several layers, and its source
    columns are those layers concatenated in ascending order, so
    a skip edge is an ordinary weight rather than a dummy neuron
    or a second mechanism; ``skips`` only reports it. Terminals
    below maximum depth (early outputs) are allowed, and isolated
    nodes cannot appear, since the node set is the union of
    ``source`` and ``target``.

    Examples
    --------
    A chain ``A -> H -> C`` plus the skip ``A -> C``. The hop into
    ``C`` reads both earlier layers, so the skip is a packed pair
    of that hop:

    >>> import pandas as pd
    >>> import kpnn2
    >>> edgelist = pd.DataFrame(
    ...     {"source": ["A", "H", "A"], "target": ["H", "C", "C"]}
    ... )
    >>> spec = kpnn2.parse_layered(edgelist)
    >>> spec.layer_nodes
    (('A',), ('H',), ('C',))
    >>> spec.hops[1].source_nodes
    ('A', 'H')
    >>> spec.hops[1].source_index, spec.hops[1].target_index
    ((0, 1), (0, 0))
    >>> spec.hops[1].to_mask().tolist()
    [[1.0, 1.0]]
    >>> spec.skips[0].source, spec.skips[0].target
    ('A', 'C')

    A hidden node of width 2 owns two units; the named edge
    ``A -> H`` becomes a 2-by-1 block:

    >>> spec = kpnn2.parse_layered(
    ...     edgelist,
    ...     widths={"H": 2},
    ... )
    >>> spec.layer_dims
    (1, 2, 1)
    >>> spec.layer_widths
    ((1,), (2,), (1,))
    >>> spec.hops[0].to_mask().tolist()
    [[1.0], [1.0]]
    """
    normalized = _validate_edgelist(edgelist)
    _reject_self_loops(normalized)
    (
        nodes,
        children,
        parents,
        in_degree,
        out_degree,
    ) = _build_adjacency(normalized)
    input_nodes, output_nodes, hidden_nodes = _node_sets(
        nodes,
        in_degree,
        out_degree,
    )
    layer_nodes = _rank_layers(
        nodes,
        children,
        parents,
        in_degree,
    )
    width_of = _normalize_widths(
        widths,
        nodes,
    )
    layouts = _layer_layouts(
        layer_nodes,
        width_of,
    )
    placement = _node_placement(layouts)
    layer_dims = [layout.n_units for layout in layouts]
    hops = _build_hops(
        normalized,
        layouts,
        placement,
    )
    skips = _build_skips(
        normalized,
        placement,
    )
    return LayeredSpec(
        input_nodes=tuple(input_nodes),
        output_nodes=tuple(output_nodes),
        hidden_nodes=tuple(hidden_nodes),
        layer_nodes=tuple(tuple(layer) for layer in layer_nodes),
        layer_dims=tuple(layer_dims),
        layer_widths=tuple(layout.widths() for layout in layouts),
        hops=tuple(hops),
        skips=tuple(skips),
    )
