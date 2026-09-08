"""
T-bounded live-path solver for adjacency unroll controls.

A name is important if and only if some walk of live edges from
an input through that name to an attributed output has length
``<= n_steps``. That matches write-inputs-then-one-hop, repeated
``n_steps`` times: unbounded DAG reachability is the wrong
ground truth once the graph can cycle.

Do not add ``n_steps`` to ``solve_structural_ground_truth``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd
import torch

from kpnn2 import AdjacencySpec, map_node_attributions

from .graphs import (
    PREDICTION_OUTPUT,
    Edge,
    wrap_cycle_graph,
)
from .ground_truth import (
    StructuralGroundTruth,
    _validate_attributed_outputs,
    live_edges,
    node_roles,
)


@dataclass(frozen=True)
class UnrolledScenario:
    """
    One unroll control: graph, step count, attributed outputs,
    labels.
    """

    id: str
    n_steps: int
    edgelist: pd.DataFrame
    attributed_outputs: tuple[str, ...]
    important_features: tuple[str, ...]
    unimportant_features: tuple[str, ...]
    important_nodes: tuple[str, ...]
    unimportant_nodes: tuple[str, ...]
    dead_edges: frozenset[Edge] = frozenset()


def declared_unrolled_ground_truth(
    scenario: UnrolledScenario,
) -> StructuralGroundTruth:
    """
    Wrap the scenario's declared labels as a ground-truth object.
    """
    return StructuralGroundTruth(
        important_features=frozenset(scenario.important_features),
        unimportant_features=frozenset(scenario.unimportant_features),
        important_nodes=frozenset(scenario.important_nodes),
        unimportant_nodes=frozenset(scenario.unimportant_nodes),
    )


def solver_unrolled_ground_truth(
    scenario: UnrolledScenario,
) -> StructuralGroundTruth:
    """
    Derive labels from T-bounded live-path reachability.
    """
    return solve_unrolled_ground_truth(
        edgelist=scenario.edgelist,
        dead_edges=scenario.dead_edges,
        attributed_outputs=scenario.attributed_outputs,
        n_steps=scenario.n_steps,
    )


def solve_unrolled_ground_truth(
    *,
    edgelist: pd.DataFrame,
    attributed_outputs: Sequence[str],
    n_steps: int,
    dead_edges: Iterable[Edge] = (),
) -> StructuralGroundTruth:
    """
    Label features and hidden nodes by walks of length ``<= T``.

    Roles use every edgelist row, including edges that tests will
    pin to 0. Distances use live edges only. A name is important
    iff the shortest path from some input to that name plus the
    shortest path from that name to some attributed output is at
    most ``n_steps``.
    """
    if n_steps < 1:
        raise ValueError(f"'n_steps' must be at least 1, got {n_steps}.")
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
    dist_from_inputs = _shortest_distances(
        successors,
        inputs,
    )
    dist_to_outputs = _shortest_distances(
        predecessors,
        attributed_outputs,
    )
    important_features = frozenset(
        name
        for name in inputs
        if _walk_length(
            dist_from_inputs,
            dist_to_outputs,
            name,
        )
        <= n_steps
    )
    important_nodes = frozenset(
        name
        for name in hidden
        if _walk_length(
            dist_from_inputs,
            dist_to_outputs,
            name,
        )
        <= n_steps
    )
    return StructuralGroundTruth(
        important_features=important_features,
        unimportant_features=frozenset(inputs - important_features),
        important_nodes=important_nodes,
        unimportant_nodes=frozenset(hidden - important_nodes),
    )


def feature_grad_table_adjacency(
    input_grad: torch.Tensor,
    spec: AdjacencySpec,
) -> pd.DataFrame:
    """
    Name input gradients by ``spec.input_nodes``.

    The aligned tensor is only ``len(input_nodes)`` wide, so
    ``map_node_attributions`` on an ``AdjacencySpec`` cannot take
    it: that mapper wants the full state axis.
    """
    return pd.DataFrame(
        input_grad.detach().cpu().numpy(),
        columns=list(spec.input_nodes),
    )


def hidden_score_table_unrolled(
    step_states: Sequence[torch.Tensor],
    spec: AdjacencySpec,
) -> pd.DataFrame:
    """
    Max-over-steps |activation × grad| for hidden names.

    Each post-step state should have ``retain_grad``. Missing
    grads count as zero. Output and input names are dropped.
    """
    hidden = set(spec.hidden_nodes)
    if not hidden or not step_states:
        return pd.DataFrame()
    peak: pd.DataFrame | None = None
    for state in step_states:
        grad = state.grad
        if grad is None:
            grad = torch.zeros_like(state)
        scores = (state.detach() * grad).abs()
        mapped = map_node_attributions(
            attributions=scores,
            spec=spec,
        ).to_pandas()
        subset = mapped.loc[:, [name for name in hidden]]
        if peak is None:
            peak = subset
        else:
            peak = pd.DataFrame(
                {
                    name: peak[name].combine(
                        subset[name],
                        max,
                    )
                    for name in hidden
                }
            )
    assert peak is not None
    return peak


def attributed_output_sum_adjacency(
    output: torch.Tensor,
    spec: AdjacencySpec,
    attributed_outputs: tuple[str, ...],
) -> torch.Tensor:
    """
    Sum the output-node columns named in ``attributed_outputs``.

    ``output`` is already sliced to ``spec.output_nodes``.
    """
    indices = [spec.output_nodes.index(name) for name in attributed_outputs]
    return output[:, indices].sum()


def _walk_length(
    dist_from_inputs: dict[str, int],
    dist_to_outputs: dict[str, int],
    name: str,
) -> int:
    """
    Shortest input-to-name-to-output walk, or a sentinel.
    """
    incoming = dist_from_inputs.get(name)
    outgoing = dist_to_outputs.get(name)
    if incoming is None or outgoing is None:
        return 10**9
    return incoming + outgoing


def _shortest_distances(
    adjacency: dict[str, set[str]],
    sources: Iterable[str],
) -> dict[str, int]:
    """
    Distance from the nearest source. Sources have distance 0.
    """
    dist: dict[str, int] = {}
    queue: deque[str] = deque()
    for source in sources:
        name = str(source)
        if name not in dist:
            dist[name] = 0
            queue.append(name)
    while queue:
        node = queue.popleft()
        for neighbour in adjacency.get(node, ()):
            if neighbour not in dist:
                dist[neighbour] = dist[node] + 1
                queue.append(neighbour)
    return dist


def _wrap_scenario(
    *,
    scenario_id: str,
    n_steps: int,
    important_features: tuple[str, ...],
    unimportant_features: tuple[str, ...],
    important_nodes: tuple[str, ...],
    unimportant_nodes: tuple[str, ...],
) -> UnrolledScenario:
    graph = wrap_cycle_graph()
    return UnrolledScenario(
        id=scenario_id,
        n_steps=n_steps,
        edgelist=graph.edgelist,
        attributed_outputs=(PREDICTION_OUTPUT,),
        important_features=important_features,
        unimportant_features=unimportant_features,
        important_nodes=important_nodes,
        unimportant_nodes=unimportant_nodes,
    )


UNROLLED_SCENARIOS: tuple[UnrolledScenario, ...] = (
    _wrap_scenario(
        scenario_id="wrap_cycle_t2",
        n_steps=2,
        important_features=("fast_in",),
        unimportant_features=("wrap_in", "decoy_in"),
        important_nodes=("fast_h",),
        unimportant_nodes=("wrap_u", "wrap_v", "decoy_h"),
    ),
    _wrap_scenario(
        scenario_id="wrap_cycle_t3",
        n_steps=3,
        important_features=("fast_in", "wrap_in"),
        unimportant_features=("decoy_in",),
        important_nodes=("fast_h", "wrap_u", "wrap_v"),
        unimportant_nodes=("decoy_h",),
    ),
)
