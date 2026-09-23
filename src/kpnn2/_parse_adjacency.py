"""
Adjacency parsing for kpnn2.
"""

from collections.abc import Mapping

import pandas as pd

from ._adjacency_spec import AdjacencySpec
from ._layout import Layout, build_layout, iter_block_pairs
from ._parse import (
    _SOURCE,
    _TARGET,
    _build_adjacency,
    _node_sets,
    _normalize_widths,
    _validate_edgelist,
)


def _packed_edge_indices(
    edgelist: pd.DataFrame,
    layout: Layout,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Record live edges as source and target unit indices.

    Sorts the normalized edgelist lexicographically by
    ``(source name, target name)`` and expands each named edge
    ``A -> B`` into every unit pair of its ``(k_B, k_A)`` block,
    target-unit outer, source-unit inner, as ``parse_layered``
    does on a hop. At width 1 that is one pair per named edge.
    Does not allocate an ``(n, n)`` tensor.

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
    source_index: list[int] = []
    target_index: list[int] = []
    for source, target in rows:
        for source_unit, target_unit in iter_block_pairs(
            layout.slot(source),
            layout.slot(target),
        ):
            source_index.append(source_unit)
            target_index.append(target_unit)
    return (
        tuple(source_index),
        tuple(target_index),
    )


def _units_of(
    names: list[str],
    layout: Layout,
) -> tuple[int, ...]:
    """
    Return every unit of each named node, in ``names`` order.
    """
    units: list[int] = []
    for name in names:
        slot = layout.slot(name)
        units.extend(range(slot.start, slot.stop))
    return tuple(units)


def parse_adjacency(
    edgelist: pd.DataFrame,
    *,
    widths: Mapping[str, int] | None = None,
) -> AdjacencySpec:
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
    widths : mapping of str to int, optional
        Units per named node, for a node backed by several
        neurons (DCell-style), exactly as in ``parse_layered``.
        Keys are matched after ``str(...)``; omitted names are
        width 1, and ``None`` makes every node width 1. A node
        of width ``k`` owns a contiguous block of ``k`` units of
        the state vector (``node_units``), and a named edge
        ``A -> B`` becomes every unit pair of its
        ``(k_B, k_A)`` block.

    Returns
    -------
    AdjacencySpec
        Frozen structure: every node name alphabetically with its
        width, the packed unit pairs of every edge, and the unit
        positions of the input and output nodes in that state
        vector. Packed order is canonical, lexicographic by
        ``(source name, target name)``, then target-unit outer,
        source-unit inner. ``hidden_nodes`` is empty when every
        node is an input or an output; the other tuples never
        are.

    Raises
    ------
    Kpnn2Error
        If ``edgelist`` is not a DataFrame; ``source`` or ``target``
        is absent, missing, or an empty name; the table has no rows;
        a ``(source, target)`` pair is duplicated; there is no
        in-degree-0 node or no out-degree-0 node; or ``widths`` is
        not a mapping, names an unknown node, or holds a value that
        is not a positive int (``bool`` included). Each message
        names the offending pairs or nodes, sorted.

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

    A node can own several units, as in ``parse_layered``. Its
    named edges become blocks of unit pairs:

    >>> wide = kpnn2.parse_adjacency(
    ...     edgelist,
    ...     widths={"a": 2},
    ... )
    >>> wide.node_widths, wide.state_dim
    ((2, 1, 1, 1), 5)
    >>> wide.node_units("a")
    slice(0, 2, None)
    >>> wide.edge_location("a", "b")
    (0, 1)
    >>> wide.input_index
    (3,)
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
    width_of = _normalize_widths(
        widths,
        node_set,
    )
    layout = build_layout(
        nodes,
        [width_of[name] for name in nodes],
    )
    source_index, target_index = _packed_edge_indices(
        normalized,
        layout,
    )
    return AdjacencySpec(
        nodes=tuple(nodes),
        node_widths=layout.widths(),
        input_nodes=tuple(input_nodes),
        output_nodes=tuple(output_nodes),
        hidden_nodes=tuple(hidden_nodes),
        source_index=source_index,
        target_index=target_index,
        input_index=_units_of(
            input_nodes,
            layout,
        ),
        output_index=_units_of(
            output_nodes,
            layout,
        ),
    )
