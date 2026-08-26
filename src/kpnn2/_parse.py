"""
Edgelist parsing for kpnn2.
"""

from collections import deque

import pandas as pd
import torch

from ._errors import Kpnn2Error
from ._layout import (
    Layout,
    NodeSlot,
    build_layout,
    concat_layouts,
    fill_block,
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


def _layer_layouts(
    layer_nodes: list[list[str]],
) -> list[Layout]:
    """
    Place each layer's nodes on that layer's unit axis.

    One layout per depth. Every node is ``DEFAULT_NODE_WIDTH``
    units wide, so ``layout.n_units`` is the node count and a
    node's slot start is its column index.
    """
    return [build_layout(names) for names in layer_nodes]


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
    Build one incoming mask per depth after the first.

    ``hops[i]`` targets depth ``i + 1`` and its mask holds every
    edge entering that depth, adjacent or skip, with shape
    ``(target units, sum of source units)``. The column axis is
    the source depths concatenated in ascending order, so an
    edge fills the block its endpoints own on that axis: one
    entry per edge while nodes are one unit wide.

    Only depths that really feed the target become columns, so a
    graph without skips gives exactly one source depth per hop
    and the same masks a per-adjacent-hop layout would.

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
    masks: dict[int, torch.Tensor] = {}
    for target_layer in range(1, n_layers):
        ordered = tuple(sorted(parents[target_layer]))
        source_layout = concat_layouts(
            [layouts[layer] for layer in ordered],
        )
        source_layers[target_layer] = ordered
        source_layouts[target_layer] = source_layout
        masks[target_layer] = torch.zeros(
            (
                layouts[target_layer].n_units,
                source_layout.n_units,
            ),
            dtype=torch.float32,
        )

    for source, target in zip(
        edgelist[_SOURCE].tolist(),
        edgelist[_TARGET].tolist(),
    ):
        target_layer, target_slot = placement[target]
        fill_block(
            masks[target_layer],
            target_slot,
            source_layouts[target_layer].slot(source),
        )

    hops: list[Hop] = []
    for target_layer in range(1, n_layers):
        ordered = source_layers[target_layer]
        hops.append(
            Hop(
                target_layer=target_layer,
                source_layers=ordered,
                source_dims=tuple(layouts[layer].n_units for layer in ordered),
                source_nodes=source_layouts[target_layer].names,
                mask=masks[target_layer],
            )
        )
    return hops


def _build_skips(
    edgelist: pd.DataFrame,
    placement: dict[str, tuple[int, NodeSlot]],
) -> list[Skip]:
    """
    Collect original edges with depth gap greater than 1.

    These records are metadata: the edges themselves are already
    ones in the target depth's hop mask. Adjacent edges (gap
    exactly 1) are omitted, since nothing distinguishes them.
    Each recorded index is the first unit its node owns, which
    is the node's column index while nodes are one unit wide.

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
                    source_index=source_slot.start,
                    target_index=target_slot.start,
                )
            )
    return skips


def parse_layered(edgelist: pd.DataFrame) -> LayeredSpec:
    """
    Parse a source/target edgelist into a ``LayeredSpec``.

    The graph must be a DAG. Nodes are ranked with Kahn's algorithm:
    input nodes (in-degree 0) have depth 0, and every other node has
    ``depth = 1 + max(parent depths)``. Names are sorted alphabetically
    inside each layer. That ranking defines ``LayeredSpec.layer_nodes``
    and one ``Hop`` per layer after the first.

    **Every** edge lands in exactly one hop mask, the one of its
    target layer, whether its depth gap is 1 or larger. A hop
    whose target has parents further back reads several layers:
    its mask columns are those layers concatenated. Edges with a
    gap greater than 1 are additionally listed in ``skips`` as
    metadata, so they can be reported, but they are not a
    separate computation and are not expanded into dummy
    neurons.

    Terminals that are not at maximum depth (early outputs) are
    allowed. Cycles, self-loops, and graphs with no input or no
    output are not. Isolated nodes cannot appear: the node set is
    the union of ``source`` and ``target`` values only.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge table with required columns ``source`` and ``target``.
        Node names are stored as strings via ``str(...)``. Extra
        columns are ignored.

    Returns
    -------
    LayeredSpec
        Frozen structure: layers, hops, and skip metadata.

    Raises
    ------
    Kpnn2Error
        If ``edgelist`` is not a DataFrame; required columns are
        missing; the table is empty; names are missing or empty;
        source-target pairs are duplicated (message names the
        unique pairs, sorted); any edge is a self-loop (message
        names the unique nodes, sorted); the graph has a cycle
        (message names unranked leftover nodes); or there is no
        in-degree-0 node or no out-degree-0 node.

    Notes
    -----
    A cycle is detected when the Kahn sweep leaves some nodes
    unranked. The error still says the edgelist has a cycle and
    that only DAGs are supported, and lists every leftover name
    (``nodes`` minus keys of ``depths``), sorted alphabetically.
    That set may include nodes downstream of a cycle, not only
    vertices on a directed cycle.

    Duplicate ``(source, target)`` pairs name the unique pairs as
    ``{source} -> {target}``, sorted lexicographically.
    Self-loops name the unique nodes, sorted alphabetically.

    ``len(hops)`` is ``len(layer_nodes) - 1`` and
    ``hops[i].target_layer`` is ``i + 1``. ``hops[i].mask`` has
    shape ``(layer_dims[i + 1], sum(hops[i].source_dims))``,
    matching ``nn.Linear.weight``, and dtype float32. Its rows
    are ``layer_nodes[i + 1]`` and its columns are
    ``hops[i].source_nodes``, the source layers concatenated in
    ascending order. An entry is ``1.0`` only for an original
    edge between the node naming that row and the node naming
    that column. ``hops[0]`` always reads layer 0 alone, so its
    mask is what an ``align_inputs`` tensor feeds directly.

    Every original edge with depth gap greater than 1 appears once
    in ``skips``. Each record has ``source``, ``target``,
    ``source_layer``, ``target_layer``, ``source_index``, and
    ``target_index``. Adjacent edges never appear in ``skips``.
    Membership in ``skips`` changes nothing about how the edge is
    computed; it is already in its target's hop mask.

    Examples
    --------
    A chain plus one skip ``A -> C``:

    >>> import pandas as pd
    >>> import kpnn2 as k2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H", "A"],
    ...         "target": ["H", "C", "C"],
    ...     }
    ... )
    >>> spec = k2.parse_layered(edgelist)
    >>> spec.input_nodes
    ('A',)
    >>> spec.hidden_nodes
    ('H',)
    >>> spec.output_nodes
    ('C',)
    >>> spec.layer_nodes
    (('A',), ('H',), ('C',))
    >>> spec.layer_dims
    (1, 1, 1)

    The hop into ``C`` reads both earlier layers, so the skip
    ``A -> C`` is a column of its mask:

    >>> spec.hops[1].source_layers
    (0, 1)
    >>> spec.hops[1].source_nodes
    ('A', 'H')
    >>> spec.hops[1].mask.tolist()
    [[1.0, 1.0]]
    >>> spec.skips[0].source, spec.skips[0].target
    ('A', 'C')
    >>> spec.skips[0].source_layer, spec.skips[0].target_layer
    (0, 2)
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
    layouts = _layer_layouts(layer_nodes)
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
        hops=tuple(hops),
        skips=tuple(skips),
    )
