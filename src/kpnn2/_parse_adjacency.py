"""
Adjacency parsing for kpnn2.
"""

import pandas as pd

from ._adjacency_spec import AdjacencySpec
from ._layout import Layout, build_layout
from ._parse import (
    _SOURCE,
    _TARGET,
    _build_adjacency,
    _node_sets,
    _validate_edgelist,
)


def _packed_edge_indices(
    edgelist: pd.DataFrame,
    layout: Layout,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Record live edges as source and target unit indices.

    Walks the normalized edgelist, records ``layout.start_of``
    for each endpoint, and sorts lexicographically by
    ``(source name, target name)``. Does not allocate an
    ``(n, n)`` tensor.

    Parameters
    ----------
    edgelist
        Normalized edgelist with string ``source`` and ``target``.
    layout
        Unit placement of every node in the state vector.

    Returns
    -------
    tuple[tuple[int, ...], tuple[int, ...]]
        Canonical ``source_index`` then ``target_index``.
    """
    sources = edgelist[_SOURCE].tolist()
    targets = edgelist[_TARGET].tolist()
    rows = sorted(
        zip(
            sources,
            targets,
        )
    )
    source_index = tuple(layout.start_of(source) for source, _ in rows)
    target_index = tuple(layout.start_of(target) for _, target in rows)
    return (
        source_index,
        target_index,
    )


def parse_adjacency(edgelist: pd.DataFrame) -> AdjacencySpec:
    """
    Parse a source/target edgelist into an ``AdjacencySpec``.

    Every node goes into one state vector, sorted alphabetically,
    and every edge goes into packed source/target index tuples.
    Nothing is ranked, so cycles and self-loops are allowed. Use
    this layout for recurrent networks; use ``parse_layered``
    for a DAG that should become one mask per layer.

    A DAG is valid input to both parsers. The layout is a choice,
    not a property of the graph, so this function never inspects
    the graph to decide which spec to return.

    Isolated nodes cannot appear: the node set is the union of
    ``source`` and ``target`` values only. Graphs with no
    in-degree-0 node or no out-degree-0 node are rejected, which
    also rejects a pure ring and a lone self-loop.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge table with required columns ``source`` and ``target``.
        Node names are stored as strings via ``str(...)``. Extra
        columns are ignored.

    Returns
    -------
    AdjacencySpec
        Frozen structure: node names, packed edge indices, and
        the input and output positions in the state vector.

    Raises
    ------
    Kpnn2Error
        If ``edgelist`` is not a DataFrame; required columns are
        missing; the table is empty; names are missing or empty;
        source-target pairs are duplicated (message names the
        unique pairs, sorted); or there is no in-degree-0 node or
        no out-degree-0 node.

    See Also
    --------
    parse_layered : Rank a DAG into one incoming mask per layer.

    Notes
    -----
    Self-loops are allowed here and rejected by ``parse_layered``.
    That is the only edgelist rule the two parsers disagree on;
    every other validation is shared, so the messages match.

    A self-loop removes its node from both the input set and the
    output set, so an edgelist of only ``A -> A`` raises for
    having no input node. A pure ring such as ``A -> B, B -> A``
    raises for the same reason.

    This function does not allocate an ``(n, n)`` tensor. Packed
    indices have the same length as the edge count. Order is
    canonical: lexicographic by ``(source name, target name)``.
    A dense square would have ``1.0`` at
    ``[target_index[i], source_index[i]]``, matching the
    ``nn.Linear.weight`` layout used by the layered hop masks.
    Call ``spec.to_mask()`` to materialize that square.

    This function does not build an ``nn.Module``, unroll time,
    choose a step count, or re-inject inputs between steps. The
    recurrence stays in user ``forward()`` code.

    Examples
    --------
    An input feeding a two-node feedback core plus one output:

    >>> import pandas as pd
    >>> import kpnn2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["x", "a", "b", "a"],
    ...         "target": ["a", "b", "a", "y"],
    ...     }
    ... )
    >>> spec = kpnn2.parse_adjacency(edgelist)
    >>> spec.nodes
    ('a', 'b', 'x', 'y')
    >>> spec.input_nodes
    ('x',)
    >>> spec.input_index
    (2,)
    >>> spec.source_index
    (0, 0, 1, 2)
    >>> spec.target_index
    (1, 3, 0, 0)
    >>> tuple(spec.to_mask().shape)
    (4, 4)

    The feedback edge ``b -> a`` and the forward edge ``a -> b``
    are both present:

    >>> mask = spec.to_mask()
    >>> mask[0, 1].item(), mask[1, 0].item()
    (1.0, 1.0)
    """
    normalized = _validate_edgelist(edgelist)
    (
        node_set,
        children,
        parents,
        in_degree,
        out_degree,
    ) = _build_adjacency(normalized)
    input_nodes, output_nodes, hidden_nodes = _node_sets(
        node_set,
        in_degree,
        out_degree,
    )
    nodes = sorted(node_set)
    layout = build_layout(nodes)
    source_index, target_index = _packed_edge_indices(
        normalized,
        layout,
    )
    return AdjacencySpec(
        nodes=tuple(nodes),
        input_nodes=tuple(input_nodes),
        output_nodes=tuple(output_nodes),
        hidden_nodes=tuple(hidden_nodes),
        source_index=source_index,
        target_index=target_index,
        input_index=tuple(layout.start_of(name) for name in input_nodes),
        output_index=tuple(layout.start_of(name) for name in output_nodes),
    )
