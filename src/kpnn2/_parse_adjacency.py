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

    Prior-knowledge edges, such as genes into pathways, become one
    alphabetical state vector plus packed source/target index
    tuples, ready for ``PackedLinear`` or packed attention. Reach
    for it when the graph has cycles or self-loops, or when a
    shared-state loop is the update; ``parse_layered`` ranks a DAG
    into one packed hop per layer instead. Nothing is ranked here,
    and no ``(n, n)`` tensor is allocated.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge table with required columns ``source`` and ``target``,
        one row per directed edge in the direction of computation.
        Names are converted with ``str(...)``; extra columns are
        ignored. The frame is read, never modified.

    Returns
    -------
    AdjacencySpec
        Frozen structure: every node name alphabetically, one
        ``source_index`` / ``target_index`` entry per edge, and the
        positions of the input and output nodes in that state
        vector. Packed order is canonical, lexicographic by
        ``(source name, target name)``. ``hidden_nodes`` is empty
        when every node is an input or an output; the other tuples
        never are.

    Raises
    ------
    Kpnn2Error
        If ``edgelist`` is not a DataFrame; ``source`` or ``target``
        is absent, missing, or an empty name; the table has no rows;
        a ``(source, target)`` pair is duplicated; or there is no
        in-degree-0 node or no out-degree-0 node. Each message names
        the offending pairs or nodes, sorted.

    See Also
    --------
    parse_layered : Rank a DAG into one incoming mask per layer;
        rejects cycles and self-loops.
    PackedLinear : One trainable weight per packed edge, for large
        node counts.
    PackedMultiheadAttention : Score only the packed pairs.

    Notes
    -----
    Self-loops are allowed here and rejected by ``parse_layered``.
    That is the only edgelist rule the two parsers disagree on;
    every other validation is shared, so the messages match. A
    self-loop takes its node out of both the input and the output
    set, so an edgelist of only ``A -> A`` raises for having no
    input node, as does a pure ring such as ``A -> B, B -> A``.
    Isolated nodes cannot appear: the node set is the union of
    ``source`` and ``target``.

    The layout is your choice, not a property of the graph. A DAG
    is valid input to both parsers, and neither inspects the graph
    to decide which spec to return. A dense square would carry
    ``1.0`` at ``[target_index[i], source_index[i]]``, the
    ``nn.Linear.weight`` orientation the layered hops also use
    after ``to_mask()``; ``spec.to_mask()`` materializes it.
    Nothing here builds an ``nn.Module``, unrolls time, or
    re-injects inputs between steps: that update stays in user
    ``forward()`` code.

    Examples
    --------
    A feedback edge ``b -> a``, which ``parse_layered`` would
    reject, packed alongside the forward edges:

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
    >>> spec.input_nodes, spec.input_index
    (('x',), (2,))
    >>> spec.source_index, spec.target_index
    ((0, 0, 1, 2), (1, 3, 0, 0))
    >>> spec.to_mask()[0, 1].item()
    1.0
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
