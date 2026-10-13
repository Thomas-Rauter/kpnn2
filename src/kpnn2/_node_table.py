"""Per-node graph counts behind ``node_table()`` on both specs."""

from collections.abc import Iterable, Sequence

import pandas as pd

from ._adjacency_spec import AdjacencySpec
from ._errors import internal_error
from ._serialize import canonical_edges
from ._spec import LayeredSpec


def spec_node_table(
    spec: LayeredSpec | AdjacencySpec,
) -> pd.DataFrame:
    """
    Build the per-node count table of a spec.

    Rows follow the spec's unit order: the flattened
    ``layer_nodes`` of a ``LayeredSpec``, ``nodes`` of an
    ``AdjacencySpec``. A ``LayeredSpec`` table starts with
    ``layer``; both continue with ``width`` and the graph
    counts over named edges.

    Parameters
    ----------
    spec : LayeredSpec or AdjacencySpec
        Spec built by a kpnn2 parser.

    Returns
    -------
    pandas.DataFrame
        One ``int64`` row per node, indexed by name under the
        index name ``node``.

    Raises
    ------
    AssertionError
        From ``internal_error``, if ``spec`` is neither spec
        type, or if the zero in-degree or zero out-degree nodes
        differ from ``input_nodes`` or ``output_nodes``.
    """
    columns: dict[str, list[int]] = {}
    if isinstance(spec, LayeredSpec):
        nodes = tuple(name for layer in spec.layer_nodes for name in layer)
        columns["layer"] = [
            layer for layer, names in enumerate(spec.layer_nodes) for _ in names
        ]
        columns["width"] = [
            width for layer in spec.layer_widths for width in layer
        ]
    elif isinstance(spec, AdjacencySpec):
        nodes = spec.nodes
        columns["width"] = list(spec.node_widths)
    else:
        raise internal_error(
            f"spec_node_table received type {type(spec).__name__}, "
            "not a LayeredSpec or an AdjacencySpec"
        )
    counts = _graph_counts(
        nodes,
        canonical_edges(spec),
        spec.input_nodes,
    )
    _check_roles(
        nodes,
        counts,
        spec,
    )
    columns.update(counts)
    return pd.DataFrame(
        columns,
        index=pd.Index(
            nodes,
            name="node",
        ),
        dtype="int64",
    )


def _graph_counts(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    input_nodes: Sequence[str],
) -> dict[str, list[int]]:
    """
    Count degrees and reachable nodes for every node, in order.

    A node is never its own ancestor or descendant, even on a
    cycle or a self-loop; the other members of its cycle are
    both.
    """
    position = {name: index for index, name in enumerate(nodes)}
    parents: list[list[int]] = [[] for _ in nodes]
    children: list[list[int]] = [[] for _ in nodes]
    for source, target in edges:
        parents[position[target]].append(position[source])
        children[position[source]].append(position[target])

    components = _strong_components(children)
    component_of = [0] * len(nodes)
    for component, members in enumerate(components):
        for node in members:
            component_of[node] = component
    order = range(len(components))
    ancestors = _reach_from_outside(
        components,
        component_of,
        parents,
        order,
    )
    descendants = _reach_from_outside(
        components,
        component_of,
        children,
        reversed(order),
    )

    input_bits = 0
    for name in input_nodes:
        input_bits |= 1 << position[name]
    n_ancestors = []
    n_descendants = []
    n_input_ancestors = []
    for node in range(len(nodes)):
        component = component_of[node]
        # The other members of its cycle; 0 for a lone node.
        same_cycle = len(components[component]) - 1
        n_ancestors.append(ancestors[component].bit_count() + same_cycle)
        n_descendants.append(
            descendants[component].bit_count() + same_cycle,
        )
        n_input_ancestors.append(
            (ancestors[component] & input_bits).bit_count(),
        )
    return {
        "in_degree": [len(node_parents) for node_parents in parents],
        "out_degree": [len(node_children) for node_children in children],
        "n_ancestors": n_ancestors,
        "n_descendants": n_descendants,
        "n_input_ancestors": n_input_ancestors,
    }


def _strong_components(
    children: Sequence[Sequence[int]],
) -> list[list[int]]:
    """
    Group nodes into cycles, ordered sources first.

    Iterative Tarjan. Each group is a strongly connected
    component: a single node, or every node of one cycle.
    Tarjan closes a group only after every group it reaches,
    so the reversed closing order puts sources first.
    """
    n_nodes = len(children)
    visit_index = [-1] * n_nodes
    low_link = [0] * n_nodes
    on_stack = [False] * n_nodes
    stack: list[int] = []
    components: list[list[int]] = []
    counter = 0
    for root in range(n_nodes):
        if visit_index[root] != -1:
            continue
        visit_index[root] = low_link[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        work = [(root, 0)]
        while work:
            node, next_child = work[-1]
            if next_child < len(children[node]):
                work[-1] = (node, next_child + 1)
                child = children[node][next_child]
                if visit_index[child] == -1:
                    visit_index[child] = low_link[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack[child] = True
                    work.append((child, 0))
                elif on_stack[child]:
                    low_link[node] = min(
                        low_link[node],
                        visit_index[child],
                    )
                continue
            work.pop()
            if work:
                caller = work[-1][0]
                low_link[caller] = min(
                    low_link[caller],
                    low_link[node],
                )
            if low_link[node] == visit_index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                components.append(component)
    components.reverse()
    return components


def _reach_from_outside(
    components: Sequence[Sequence[int]],
    component_of: Sequence[int],
    neighbors: Sequence[Sequence[int]],
    order: Iterable[int],
) -> list[int]:
    """
    Return, per component, the nodes outside it that reach it.

    Each result is a bitset over node positions. With
    ``neighbors`` as parents and sources-first ``order`` these
    are ancestors; with children and the reversed order,
    descendants. ``order`` must visit every neighbor component
    before the component itself.
    """
    reach = [0] * len(components)
    members = [0] * len(components)
    for component, nodes in enumerate(components):
        for node in nodes:
            members[component] |= 1 << node
    for component in order:
        bits = 0
        for node in components[component]:
            for neighbor in neighbors[node]:
                other = component_of[neighbor]
                if other != component:
                    bits |= members[other] | reach[other]
        reach[component] = bits
    return reach


def _check_roles(
    nodes: Sequence[str],
    counts: dict[str, list[int]],
    spec: LayeredSpec | AdjacencySpec,
) -> None:
    """Raise if zero degrees disagree with the spec's node roles."""
    no_parents = {
        name
        for name, degree in zip(
            nodes,
            counts["in_degree"],
            strict=True,
        )
        if degree == 0
    }
    no_children = {
        name
        for name, degree in zip(
            nodes,
            counts["out_degree"],
            strict=True,
        )
        if degree == 0
    }
    if no_parents != set(spec.input_nodes):
        raise internal_error(
            "node_table's zero in-degree nodes differ from spec.input_nodes"
        )
    if no_children != set(spec.output_nodes):
        raise internal_error(
            "node_table's zero out-degree nodes differ from spec.output_nodes"
        )
