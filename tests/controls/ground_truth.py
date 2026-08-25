"""
Live-path solver for structural control tests.

A name is important if and only if some path of live edges runs from
an input through that name to at least one attributed output. Dead
edges are ``(source, target)`` pairs pinned at weight 0 in tests;
they remain in the edgelist so node roles do not change.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from .graphs import Edge


@dataclass(frozen=True)
class StructuralGroundTruth:
    """
    Features and hidden nodes that can influence attributed outputs.

    Output nodes are not classified. Inputs appear only in the
    feature sets; hidden nodes only in the node sets.
    """

    important_features: frozenset[str]
    unimportant_features: frozenset[str]
    important_nodes: frozenset[str]
    unimportant_nodes: frozenset[str]

    @property
    def important_names(self) -> frozenset[str]:
        """Return every important feature and hidden node."""
        return self.important_features | self.important_nodes

    @property
    def unimportant_names(self) -> frozenset[str]:
        """Return every unimportant feature and hidden node."""
        return self.unimportant_features | self.unimportant_nodes


def live_edges(
    edgelist: pd.DataFrame,
    dead_edges: Iterable[Edge] = (),
) -> set[Edge]:
    """
    Return ``(source, target)`` pairs that can carry nonzero signal.

    Every ``dead_edges`` pair must exist in the edgelist.
    """
    edges = {
        (str(source), str(target))
        for source, target in zip(
            edgelist["source"],
            edgelist["target"],
            strict=True,
        )
    }
    dead = {(str(source), str(target)) for source, target in dead_edges}
    unknown = sorted(dead - edges)
    if unknown:
        raise ValueError(
            "These dead_edges are not in the edgelist: "
            f"{unknown}. Available edges: {sorted(edges)}."
        )
    return edges - dead


def node_roles(
    edgelist: pd.DataFrame,
) -> tuple[set[str], set[str], set[str]]:
    """
    Infer inputs, hidden nodes, and outputs from the full edgelist.

    Roles use every row, including edges that tests will pin to 0.
    """
    sources = [str(value) for value in edgelist["source"]]
    targets = [str(value) for value in edgelist["target"]]
    nodes = set(sources) | set(targets)
    in_degree = dict.fromkeys(nodes, 0)
    out_degree = dict.fromkeys(nodes, 0)
    for source, target in zip(
        sources,
        targets,
        strict=True,
    ):
        out_degree[source] += 1
        in_degree[target] += 1
    inputs = {node for node in nodes if in_degree[node] == 0}
    outputs = {node for node in nodes if out_degree[node] == 0}
    hidden = nodes - inputs - outputs
    return inputs, hidden, outputs


def solve_structural_ground_truth(
    *,
    edgelist: pd.DataFrame,
    dead_edges: Iterable[Edge] = (),
    attributed_outputs: Sequence[str],
) -> StructuralGroundTruth:
    """
    Label features and hidden nodes by live-path reachability.

    Forward from inputs intersected with backward from attributed
    outputs, over edges not listed in ``dead_edges``.
    """
    inputs, hidden, outputs = node_roles(edgelist)
    _validate_attributed_outputs(
        output_nodes=outputs,
        attributed_outputs=attributed_outputs,
    )
    edges = live_edges(
        edgelist,
        dead_edges,
    )
    successors: dict[str, set[str]] = {}
    predecessors: dict[str, set[str]] = {}
    for source, target in edges:
        successors.setdefault(source, set()).add(target)
        predecessors.setdefault(target, set()).add(source)
    reachable_from_inputs = _traverse(
        successors,
        inputs,
    )
    reaches_outputs = _traverse(
        predecessors,
        attributed_outputs,
    )
    influential = reachable_from_inputs & reaches_outputs
    important_features = inputs & influential
    important_nodes = hidden & influential
    return StructuralGroundTruth(
        important_features=frozenset(important_features),
        unimportant_features=frozenset(inputs - important_features),
        important_nodes=frozenset(important_nodes),
        unimportant_nodes=frozenset(hidden - important_nodes),
    )


def assert_all_structurally_live(
    *,
    edgelist: pd.DataFrame,
    dead_edges: Iterable[Edge] = (),
    attributed_outputs: Sequence[str],
    names: Sequence[str],
) -> None:
    """
    Assert every name can structurally reach the attributed outputs.

    Trained-tier guard: a decoy declared unimportant by the data
    must still be wired, or rank tests pass on structural zeros.
    """
    ground_truth = solve_structural_ground_truth(
        edgelist=edgelist,
        dead_edges=dead_edges,
        attributed_outputs=attributed_outputs,
    )
    classified = ground_truth.important_names | ground_truth.unimportant_names
    unknown = sorted(name for name in names if name not in classified)
    assert not unknown, (
        f"These names are neither input features nor hidden nodes: {unknown}."
    )
    dead = sorted(
        name for name in names if name not in ground_truth.important_names
    )
    assert not dead, (
        "These names cannot structurally influence the attributed "
        f"outputs {sorted(attributed_outputs)}: {dead}. A trained-"
        "tier scenario must wire both groups to the same output."
    )


def _traverse(
    adjacency: dict[str, set[str]],
    sources: Iterable[str],
) -> set[str]:
    """
    Return every node reachable from ``sources``, including sources.
    """
    seen: set[str] = set()
    queue: deque[str] = deque()
    for source in sources:
        source_name = str(source)
        if source_name not in seen:
            seen.add(source_name)
            queue.append(source_name)
    while queue:
        node = queue.popleft()
        for neighbour in adjacency.get(node, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return seen


def _validate_attributed_outputs(
    *,
    output_nodes: set[str],
    attributed_outputs: Sequence[str],
) -> None:
    """
    Require a nonempty subset of graph output nodes.
    """
    if not attributed_outputs:
        raise ValueError(
            "'attributed_outputs' must name at least one output "
            "node. An empty selection would mark every node "
            "unimportant."
        )
    unknown = sorted(
        name for name in attributed_outputs if str(name) not in output_nodes
    )
    if unknown:
        raise ValueError(
            "'attributed_outputs' contains names that are not "
            f"graph output nodes: {unknown}. Available output "
            f"nodes: {sorted(output_nodes)}."
        )
