"""
Adjacency parsing for kpnn2.
"""

from collections.abc import Iterable, Mapping

import pandas as pd

from ._adjacency_spec import AdjacencySpec
from ._errors import internal_error
from ._layout import Layout, build_layout, iter_block_pairs
from ._parse import (
    _SOURCE,
    _TARGET,
    _build_adjacency,
    _check_edges_conserved,
    _confirm_outputs,
    _declared_outputs,
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


def _check_input_units(spec: AdjacencySpec) -> None:
    """
    Raise unless ``input_index`` is the units of ``input_nodes``.

    ``align_inputs`` and ``map_node_attributions`` name the input
    units from ``input_nodes`` and ``node_widths`` and trust
    ``input_index`` to hold exactly those units, in that order.
    The expected units are rebuilt here from the stored ``nodes``
    and ``node_widths``, once, instead of on each call.

    Raises
    ------
    AssertionError
        From ``internal_error``, if ``input_index`` differs.
    """
    layout = build_layout(
        spec.nodes,
        spec.node_widths,
    )
    slots = [layout.slot(name) for name in spec.input_nodes]
    expected = tuple(
        unit for slot in slots for unit in range(slot.start, slot.stop)
    )
    if spec.input_index != expected:
        raise internal_error(
            f"parse_adjacency input_index ({len(spec.input_index)} "
            "units) differs from the units of input_nodes "
            f"({len(expected)} units) in values or order"
        )


def _adjacency_from_edgelist(
    edgelist: pd.DataFrame,
    *,
    widths: Mapping[str, int] | None,
) -> AdjacencySpec:
    """
    Parse ``edgelist`` into an ``AdjacencySpec`` without ``outputs``.

    Runs every check of ``parse_adjacency`` except the output
    confirmation, which neither raises nor warns here.
    ``AdjacencySpec.from_dict`` uses it: it rebuilds a spec that
    was parsed before, so several outputs are not news.
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
        input_nodes,
    )
    layout = build_layout(
        nodes,
        [width_of[name] for name in nodes],
    )
    source_index, target_index = _packed_edge_indices(
        normalized,
        layout,
    )
    spec = AdjacencySpec(
        nodes=tuple(nodes),
        node_widths=layout.widths(),
        input_nodes=tuple(input_nodes),
        output_nodes=tuple(output_nodes),
        hidden_nodes=tuple(hidden_nodes),
        source_index=source_index,
        target_index=target_index,
        input_index=layout.units_of(input_nodes),
        output_index=layout.units_of(output_nodes),
    )
    _check_edges_conserved(
        spec,
        normalized,
        width_of,
        [(spec.source_index, spec.target_index)],
    )
    _check_input_units(spec)
    return spec


def parse_adjacency(  # numpydoc ignore=PR06
    edgelist: pd.DataFrame,
    *,
    widths: Mapping[str, int] | None = None,
    outputs: Iterable[str] | None = None,
) -> AdjacencySpec:
    """
    Parse a source/target edgelist into an ``AdjacencySpec``.

    Prior-knowledge edges, such as genes into pathways, become one
    alphabetical state vector plus packed source/target index
    tuples, ready for ``PackedLinear`` or packed attention. Reach
    for it when the graph has cycles or self-loops, or when one
    shared update, applied for several steps, is the model;
    ``parse_layered`` ranks a DAG into one packed hop per layer
    instead. Nothing is ranked here, and no ``(n, n)`` tensor is
    allocated. Every node is a single unit, one neuron, unless
    ``widths`` makes it a block of several.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge table with required columns ``source`` and ``target``,
        one row per directed edge in the direction of computation.
        Node names are text: a ``str`` is kept, and an integer is
        converted with ``str(...)``, so ``1`` and ``"1"`` name the
        same node. A float or ``bool`` is converted too but warns,
        since ``1.0`` becomes ``"1.0"``; any other value raises.
        Extra columns are ignored. The frame is read, never
        modified.
    widths : mapping of str to int or numpy integer, optional
        Number of units per named node. A unit is one neuron: it
        holds one number per sample. A node of width ``k`` is a
        block of ``k`` units, a contiguous slice of the state
        vector (``node_units``), so it can carry ``k`` numbers
        instead of one. An edge ``A -> B`` then connects every
        unit of ``A`` to every unit of ``B``, its ``(k_B, k_A)``
        block of unit pairs, which is ``k_B * k_A`` weights
        instead of one.

        Widths change what the network can compute. A node of
        width 1 squeezes everything its parents send into one
        number, and every node it feeds sees that same number.
        So wider nodes matter where one number per node is too
        little:

        - a hidden node that combines many parents and should
          pass on more than one summary of them
        - a node on a cycle that should carry more than
          one number from one update to the next
        - capacity that grows with the node, such as a width set
          by its number of parents
        - an output that is several numbers, such as class
          scores or the parameters of a distribution

        The price is ``k_B * k_A`` weights per edge, and a score
        per node has to combine the scores of its units. A wide
        node also shifts every node after it in the state
        vector, so read positions from ``node_units``,
        ``unit_index``, ``input_index``, and ``output_index``
        instead of counting nodes.

        Omitted names, ``None``, and an empty mapping are width 1.
        Keys are matched after ``str(...)``. Unknown names raise
        ``Kpnn2Error``. Values must be positive ints or numpy
        integers, stored as ``int``; ``bool``, ``numpy.bool_``,
        floats, ``0``, and negatives are rejected. Input nodes
        must stay width 1: each reads one feature column, so
        separate measurements belong on separate input nodes.
    outputs : iterable of str, optional
        The output nodes you read, to confirm the ones the parser
        infers. Every node with no outgoing edge is an output
        node, so a node whose onward rows are missing from the
        edgelist is one too, and ``output_index`` would hand it
        to the head as a prediction. With ``outputs``, any other
        node with no outgoing edge raises, and the message names
        every node with no path to a name in ``outputs``,
        including the nodes of a cycle that never reaches one:
        drop the rows whose target is one of those and parse
        again.
        ``outputs`` checks and never assigns. It does not make a
        node an output, and the spec is the same as without it.

        ``None`` or omitted skips the check, and the parse warns
        when there is more than one output node. Names are
        matched after ``str(...)`` and compared as a set, so a
        list, tuple, set, one-dimensional numpy array, or
        ``pandas.Index`` works in any order. A string, a mapping,
        an empty collection, a repeated name, an unknown name, or
        a node with an outgoing edge raises ``Kpnn2Error``. To
        read a node that feeds others, use
        ``AdjacencySpec.unit_index``.

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
        is absent, missing, or an empty name, or holds a value that
        is not a ``str``, integer, float, or ``bool`` (a list, a
        tensor, bytes, any other object); the table has no rows;
        two node names differ only by leading or trailing whitespace
        (``'A'`` and ``'A '``); a ``(source, target)`` pair is
        duplicated; there is no in-degree-0 node or no out-degree-0
        node; or ``widths`` is not a mapping, names an unknown node,
        holds a value that is not a positive integer (``bool``
        included), or gives an input node a width other than 1;
        or ``outputs`` is a string, a mapping, or empty, repeats a
        name, names a node that is not an output node, or leaves
        out an output node. Each message names the offending
        pairs or nodes, sorted.

    Warns
    -----
    UserWarning
        If ``source`` or ``target`` holds a float or a ``bool``,
        converted to a name such as ``"1.0"`` or ``"True"``.
        Convert the column to the intended strings first to
        silence it. Also if ``outputs`` is omitted and the graph
        has more than one output node; the message names them.
        Pass the output nodes you read as ``outputs`` to silence
        it.

    See Also
    --------
    parse_layered : Rank a DAG into one incoming mask per layer;
        rejects cycles and self-loops.
    PackedLinear.from_adjacency : One trainable weight per packed
        edge, for large node counts.
    PackedMultiheadAttention.from_adjacency : Score only the
        packed pairs.

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
    The cycle ``a -> b -> a``, which ``parse_layered`` would
    reject, packed like every other edge:

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

    ``x`` moves from unit 2 to unit 3, because ``a`` now takes two:

    >>> wide.input_index
    (3,)
    """
    declared = _declared_outputs(outputs)
    spec = _adjacency_from_edgelist(
        edgelist,
        widths=widths,
    )
    _confirm_outputs(
        spec,
        declared,
    )
    return spec
