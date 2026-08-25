"""
Unit tests for the live-path solver.

These pin solver behavior without building an nn.Module.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.controls.graphs import (
    dead_edge_graph,
    multi_output_graph,
    skip_edge_graph,
)
from tests.controls.ground_truth import (
    assert_all_structurally_live,
    live_edges,
    solve_structural_ground_truth,
)
from tests.controls.scenario import (
    STRUCTURAL_SCENARIOS,
    declared_ground_truth,
    solver_ground_truth,
)


def _tiny_live_and_dead_edgelist() -> pd.DataFrame:
    """
    One live path and one dead first hop, both into prediction.
    """
    return pd.DataFrame(
        {
            "source": [
                "signal_in",
                "live_hidden",
                "dead_in",
                "dead_hidden",
            ],
            "target": [
                "live_hidden",
                "prediction",
                "dead_hidden",
                "prediction",
            ],
        }
    )


def test_tiny_graph_dead_edge_cuts_that_branch() -> None:
    edgelist = _tiny_live_and_dead_edgelist()
    dead_edges = {("dead_in", "dead_hidden")}
    ground_truth = solve_structural_ground_truth(
        edgelist=edgelist,
        dead_edges=dead_edges,
        attributed_outputs=["prediction"],
    )
    assert ("dead_in", "dead_hidden") not in live_edges(
        edgelist,
        dead_edges,
    )
    assert ground_truth.important_features == frozenset({"signal_in"})
    assert ground_truth.unimportant_features == frozenset({"dead_in"})
    assert ground_truth.important_nodes == frozenset({"live_hidden"})
    assert ground_truth.unimportant_nodes == frozenset({"dead_hidden"})


def test_tiny_graph_without_pins_marks_both_branches_live() -> None:
    edgelist = _tiny_live_and_dead_edgelist()
    ground_truth = solve_structural_ground_truth(
        edgelist=edgelist,
        dead_edges=(),
        attributed_outputs=["prediction"],
    )
    assert ground_truth.unimportant_features == frozenset()
    assert ground_truth.unimportant_nodes == frozenset()


def test_disconnected_decoy_is_unimportant() -> None:
    edgelist = pd.DataFrame(
        {
            "source": [
                "sig_in",
                "sig_hidden",
                "dec_in",
                "dec_hidden",
            ],
            "target": [
                "sig_hidden",
                "prediction",
                "dec_hidden",
                "decoy_readout",
            ],
        }
    )
    ground_truth = solve_structural_ground_truth(
        edgelist=edgelist,
        attributed_outputs=["prediction"],
    )
    assert ground_truth.important_features == frozenset({"sig_in"})
    assert ground_truth.unimportant_features == frozenset({"dec_in"})
    assert ground_truth.important_nodes == frozenset({"sig_hidden"})
    assert ground_truth.unimportant_nodes == frozenset({"dec_hidden"})


def test_dead_skip_makes_only_its_source_unimportant() -> None:
    edgelist = pd.DataFrame(
        {
            "source": ["gene_a", "tf_1", "module_1", "gene_b"],
            "target": ["tf_1", "module_1", "output_1", "output_1"],
        }
    )
    ground_truth = solve_structural_ground_truth(
        edgelist=edgelist,
        dead_edges={("gene_b", "output_1")},
        attributed_outputs=["output_1"],
    )
    assert ground_truth.important_features == frozenset({"gene_a"})
    assert ground_truth.unimportant_features == frozenset({"gene_b"})
    assert ground_truth.important_nodes == frozenset({"tf_1", "module_1"})


def test_unknown_dead_edge_raises() -> None:
    edgelist = _tiny_live_and_dead_edgelist()
    with pytest.raises(
        ValueError,
        match="not in the edgelist",
    ):
        live_edges(
            edgelist,
            {("missing", "prediction")},
        )


def test_empty_attributed_outputs_raises() -> None:
    edgelist = _tiny_live_and_dead_edgelist()
    with pytest.raises(
        ValueError,
        match="at least one output node",
    ):
        solve_structural_ground_truth(
            edgelist=edgelist,
            attributed_outputs=[],
        )


def test_unknown_attributed_output_raises() -> None:
    edgelist = _tiny_live_and_dead_edgelist()
    with pytest.raises(
        ValueError,
        match="not graph output",
    ):
        solve_structural_ground_truth(
            edgelist=edgelist,
            attributed_outputs=["live_hidden"],
        )


def test_assert_all_structurally_live_rejects_cut_off_names() -> None:
    edgelist = _tiny_live_and_dead_edgelist()
    with pytest.raises(
        AssertionError,
        match="dead_hidden",
    ):
        assert_all_structurally_live(
            edgelist=edgelist,
            dead_edges={("dead_in", "dead_hidden")},
            attributed_outputs=["prediction"],
            names=["dead_hidden"],
        )


@pytest.mark.parametrize(
    "scenario",
    STRUCTURAL_SCENARIOS,
    ids=[item.id for item in STRUCTURAL_SCENARIOS],
)
def test_declared_labels_match_solver(scenario) -> None:
    assert declared_ground_truth(scenario) == solver_ground_truth(scenario)


def test_structural_scenario_ids_are_unique() -> None:
    ids = [scenario.id for scenario in STRUCTURAL_SCENARIOS]
    assert len(ids) == len(set(ids))


def test_graph_builders_keep_source_target_only() -> None:
    graphs = [
        dead_edge_graph(),
        skip_edge_graph(),
        multi_output_graph(),
    ]
    for graph in graphs:
        assert list(graph.edgelist.columns) == ["source", "target"]
