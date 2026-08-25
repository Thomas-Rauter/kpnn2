"""
Edgelist parsing for kpnn2.
"""

from collections import deque

import pandas as pd
import torch

from .errors import Kpnn2Error
from .graph_spec import GraphSpec, Skip

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
        or ``(source, target)`` pairs are duplicated.
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
        raise Kpnn2Error(
            f"Edgelist contains {n_duplicates} duplicate edge(s). "
            "At most one connection is allowed for each "
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
        If one or more self-loops are present.
    """
    self_loops = edgelist[_SOURCE] == edgelist[_TARGET]
    n_self_loops = int(self_loops.sum())
    if n_self_loops > 0:
        raise Kpnn2Error(
            f"Edgelist contains {n_self_loops} self-loop(s). "
            "Self-loops are not allowed."
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
        If a cycle leaves some nodes unranked.
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
        raise Kpnn2Error("Edgelist contains a cycle. Only DAGs are supported.")

    n_layers = max(depths.values()) + 1
    layer_nodes: list[list[str]] = [[] for _ in range(n_layers)]
    for node, depth in depths.items():
        layer_nodes[depth].append(node)
    for layer in layer_nodes:
        layer.sort()
    return layer_nodes


def _node_locations(
    layer_nodes: list[list[str]],
) -> dict[str, tuple[int, int]]:
    """
    Map each node name to ``(layer, index)`` in ``layer_nodes``.
    """
    locations: dict[str, tuple[int, int]] = {}
    for layer, names in enumerate(layer_nodes):
        for index, name in enumerate(names):
            locations[name] = (layer, index)
    return locations


def _build_masks(
    edgelist: pd.DataFrame,
    layer_nodes: list[list[str]],
) -> list[torch.Tensor]:
    """
    Build adjacent-hop mask tensors from original edges.

    ``masks[i]`` has shape ``(n_{i+1}, n_i)``, dtype float32.
    Entries are ``1.0`` only for depth-gap-1 edges. Skip edges
    (gap greater than 1) are omitted.

    Parameters
    ----------
    edgelist
        Normalized edgelist with string ``source`` and ``target``.
    layer_nodes
        Ranked node names, one list per depth.

    Returns
    -------
    list[torch.Tensor]
        One mask per hop from layer ``i`` to layer ``i + 1``.
    """
    locations = _node_locations(layer_nodes)
    masks: list[torch.Tensor] = []
    for i in range(len(layer_nodes) - 1):
        mask = torch.zeros(
            (
                len(layer_nodes[i + 1]),
                len(layer_nodes[i]),
            ),
            dtype=torch.float32,
        )
        masks.append(mask)

    sources = edgelist[_SOURCE].tolist()
    targets = edgelist[_TARGET].tolist()
    for source, target in zip(
        sources,
        targets,
    ):
        source_layer, source_index = locations[source]
        target_layer, target_index = locations[target]
        gap = target_layer - source_layer
        if gap == 1:
            masks[source_layer][target_index, source_index] = 1.0
    return masks


def _build_skips(
    edgelist: pd.DataFrame,
    layer_nodes: list[list[str]],
) -> list[Skip]:
    """
    Collect original edges with depth gap greater than 1.

    Adjacent edges (gap exactly 1) are omitted; they belong in
    masks, not here.

    Parameters
    ----------
    edgelist
        Normalized edgelist with string ``source`` and ``target``.
    layer_nodes
        Ranked node names, one list per depth.

    Returns
    -------
    list[Skip]
        One record per skip edge, in edgelist order.
    """
    locations = _node_locations(layer_nodes)
    skips: list[Skip] = []
    sources = edgelist[_SOURCE].tolist()
    targets = edgelist[_TARGET].tolist()
    for source, target in zip(
        sources,
        targets,
    ):
        source_layer, source_index = locations[source]
        target_layer, target_index = locations[target]
        gap = target_layer - source_layer
        if gap > 1:
            skips.append(
                Skip(
                    source=source,
                    target=target,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    source_index=source_index,
                    target_index=target_index,
                )
            )
    return skips


def parse_edgelist(edgelist: pd.DataFrame) -> GraphSpec:
    """
    Parse a source/target edgelist into a layered ``GraphSpec``.

    The graph must be a DAG. Nodes are ranked with Kahn's algorithm:
    input nodes (in-degree 0) have depth 0, and every other node has
    ``depth = 1 + max(parent depths)``. Names are sorted alphabetically
    inside each layer. That ranking defines ``GraphSpec.layer_nodes``,
    adjacent-hop ``masks``, and skip-edge records.

    Adjacent edges (depth gap exactly 1) become ones in ``masks``.
    Edges with depth gap greater than 1 are stored in ``skips`` and
    do not appear in any mask. Do not expand skips into dummy
    neurons; add them as residuals in ``forward()``.

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
    GraphSpec
        Frozen structure: layers, masks, and skips.

    Raises
    ------
    Kpnn2Error
        If ``edgelist`` is not a DataFrame; required columns are
        missing; the table is empty; names are missing or empty;
        source-target pairs are duplicated; any edge is a self-loop;
        the graph has a cycle; or there is no in-degree-0 node or no
        out-degree-0 node.

    Notes
    -----
    ``len(masks)`` is ``len(layer_nodes) - 1``. ``masks[i]`` is the
    hop from layer ``i`` to ``i + 1``. Shape is
    ``(layer_dims[i + 1], layer_dims[i])``, matching
    ``nn.Linear.weight``. Dtype is float32.
    ``masks[i][target_index, source_index]`` is ``1.0`` only for an
    original edge from ``layer_nodes[i][source_index]`` to
    ``layer_nodes[i + 1][target_index]`` with depth gap exactly 1.

    Every original edge with depth gap greater than 1 appears once
    in ``skips``. Each record has ``source``, ``target``,
    ``source_layer``, ``target_layer``, ``source_index``, and
    ``target_index``. Adjacent edges never appear in ``skips``.

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
    >>> spec = k2.parse_edgelist(edgelist)
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
    >>> spec.masks[0].tolist()
    [[1.0]]
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
    layer_dims = [len(layer) for layer in layer_nodes]
    masks = _build_masks(
        normalized,
        layer_nodes,
    )
    skips = _build_skips(
        normalized,
        layer_nodes,
    )
    return GraphSpec(
        input_nodes=tuple(input_nodes),
        output_nodes=tuple(output_nodes),
        hidden_nodes=tuple(hidden_nodes),
        layer_nodes=tuple(tuple(layer) for layer in layer_nodes),
        layer_dims=tuple(layer_dims),
        masks=tuple(masks),
        skips=tuple(skips),
    )
