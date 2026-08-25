"""
Structural-tier scenario records.

Each scenario declares important and unimportant names. Tests compare
that declaration to the live-path solver so a bug in either side
cannot cancel.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .graphs import (
    PREDICTION_OUTPUT,
    Edge,
    ImportanceGraph,
    dead_edge_graph,
    multi_output_graph,
    skip_edge_graph,
    two_tower_feedforward,
)
from .ground_truth import (
    StructuralGroundTruth,
    solve_structural_ground_truth,
)


@dataclass(frozen=True)
class StructuralScenario:
    """
    One structural control: graph, pins, attributed outputs, labels.
    """

    id: str
    edgelist: pd.DataFrame
    attributed_outputs: tuple[str, ...]
    important_features: tuple[str, ...]
    unimportant_features: tuple[str, ...]
    important_nodes: tuple[str, ...]
    unimportant_nodes: tuple[str, ...]
    dead_edges: frozenset[Edge]
    bias: bool = False


def declared_ground_truth(
    scenario: StructuralScenario,
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


def solver_ground_truth(
    scenario: StructuralScenario,
) -> StructuralGroundTruth:
    """
    Derive labels from live-path reachability.
    """
    return solve_structural_ground_truth(
        edgelist=scenario.edgelist,
        dead_edges=scenario.dead_edges,
        attributed_outputs=scenario.attributed_outputs,
    )


def _from_importance_graph(
    *,
    scenario_id: str,
    graph: ImportanceGraph,
    attributed_outputs: tuple[str, ...] | None = None,
    bias: bool = False,
) -> StructuralScenario:
    """
    Wrap an ``ImportanceGraph`` as a scenario.
    """
    outputs = (
        attributed_outputs if attributed_outputs is not None else graph.outputs
    )
    return StructuralScenario(
        id=scenario_id,
        edgelist=graph.edgelist,
        attributed_outputs=outputs,
        important_features=graph.important_features,
        unimportant_features=graph.unimportant_features,
        important_nodes=graph.important_nodes,
        unimportant_nodes=graph.unimportant_nodes,
        dead_edges=graph.dead_edges,
        bias=bias,
    )


def _disconnected_two_tower(
    *,
    scenario_id: str,
    bias: bool,
) -> StructuralScenario:
    graph = two_tower_feedforward(shared_output=False)
    return StructuralScenario(
        id=scenario_id,
        edgelist=graph.edgelist,
        attributed_outputs=(PREDICTION_OUTPUT,),
        important_features=graph.tower_a_features,
        unimportant_features=graph.tower_b_features,
        important_nodes=graph.tower_a_nodes,
        unimportant_nodes=graph.tower_b_nodes,
        dead_edges=frozenset(),
        bias=bias,
    )


STRUCTURAL_SCENARIOS: tuple[StructuralScenario, ...] = (
    _disconnected_two_tower(
        scenario_id="disconnected_feedforward",
        bias=False,
    ),
    _disconnected_two_tower(
        scenario_id="disconnected_feedforward_with_bias",
        bias=True,
    ),
    _from_importance_graph(
        scenario_id="dead_edge_feedforward",
        graph=dead_edge_graph(),
    ),
    _from_importance_graph(
        scenario_id="skip_edge_feedforward",
        graph=skip_edge_graph(),
    ),
    _from_importance_graph(
        scenario_id="multi_output_output_1",
        graph=multi_output_graph(),
        attributed_outputs=("output_1",),
    ),
)
