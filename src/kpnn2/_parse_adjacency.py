"""
Adjacency parsing for kpnn2.
"""

import pandas as pd
import torch

from ._adjacency_spec import AdjacencySpec
from ._layout import Layout, build_layout, fill_block
from ._parse import (
    _SOURCE,
    _TARGET,
    _build_adjacency,
    _node_sets,
    _validate_edgelist,
)


def _build_square_mask(
    edgelist: pd.DataFrame,
    layout: Layout,
) -> torch.Tensor:
    """
    Build one square mask over every node.

    Shape is ``(n, n)`` with ``n == layout.n_units``, dtype
    float32. Each edge fills the block its endpoints own, which
    is the single entry ``[target_index, source_index]`` while
    nodes are one unit wide. Self-loops land on the diagonal.

    Parameters
    ----------
    edgelist
        Normalized edgelist with string ``source`` and ``target``.
    layout
        Unit placement of every node in the state vector.

    Returns
    -------
    torch.Tensor
        Square float32 connectivity mask.
    """
    mask = torch.zeros(
        (
            layout.n_units,
            layout.n_units,
        ),
        dtype=torch.float32,
    )
    sources = edgelist[_SOURCE].tolist()
    targets = edgelist[_TARGET].tolist()
    for source, target in zip(
        sources,
        targets,
    ):
        fill_block(
            mask,
            layout.slot(target),
            layout.slot(source),
        )
    return mask


def parse_adjacency(edgelist: pd.DataFrame) -> AdjacencySpec:
    """
    Parse a source/target edgelist into an ``AdjacencySpec``.

    Every node goes into one state vector, sorted alphabetically,
    and every edge goes into one square mask. Nothing is ranked,
    so cycles and self-loops are allowed. Use this layout for
    recurrent networks; use ``parse_layered`` for a DAG that
    should become per-hop masks and skip records.

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
        Frozen structure: node names, one square mask, and the
        input and output positions in that mask.

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
    parse_layered : Rank a DAG into per-hop masks and skips.

    Notes
    -----
    Self-loops are allowed here and rejected by ``parse_layered``.
    That is the only edgelist rule the two parsers disagree on;
    every other validation is shared, so the messages match.

    A self-loop removes its node from both the input set and the
    output set, so an edgelist of only ``A -> A`` raises for
    having no input node. A pure ring such as ``A -> B, B -> A``
    raises for the same reason.

    ``mask`` has shape ``(n, n)`` with ``n == len(nodes)`` and
    dtype float32. ``mask[target_index, source_index]`` is ``1.0``
    for an original edge from ``nodes[source_index]`` to
    ``nodes[target_index]``, matching the ``nn.Linear.weight``
    layout used by the layered masks.

    This function does not build an ``nn.Module``, unroll time,
    choose a step count, or re-inject inputs between steps. The
    recurrence stays in user ``forward()`` code.

    Examples
    --------
    An input feeding a two-node feedback core plus one output:

    >>> import pandas as pd
    >>> import kpnn2 as k2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["x", "a", "b", "a"],
    ...         "target": ["a", "b", "a", "y"],
    ...     }
    ... )
    >>> spec = k2.parse_adjacency(edgelist)
    >>> spec.nodes
    ('a', 'b', 'x', 'y')
    >>> spec.input_nodes
    ('x',)
    >>> spec.input_index
    (2,)
    >>> tuple(spec.mask.shape)
    (4, 4)

    The feedback edge ``b -> a`` and the forward edge ``a -> b``
    are both present:

    >>> spec.mask[0, 1].item(), spec.mask[1, 0].item()
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
    mask = _build_square_mask(
        normalized,
        layout,
    )
    return AdjacencySpec(
        nodes=tuple(nodes),
        input_nodes=tuple(input_nodes),
        output_nodes=tuple(output_nodes),
        hidden_nodes=tuple(hidden_nodes),
        mask=mask,
        input_index=tuple(layout.start_of(name) for name in input_nodes),
        output_index=tuple(layout.start_of(name) for name in output_nodes),
    )
