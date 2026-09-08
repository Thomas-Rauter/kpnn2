"""
Unit tests for the T-bounded unroll solver.

These pin solver behavior without building an nn.Module.
"""

from __future__ import annotations

import pytest

from tests.controls.graphs import (
    memory_self_loop_graph,
    wrap_cycle_graph,
)
from tests.controls.unroll import (
    UNROLLED_SCENARIOS,
    declared_unrolled_ground_truth,
    solve_unrolled_ground_truth,
    solver_unrolled_ground_truth,
)


@pytest.mark.parametrize(
    "scenario",
    UNROLLED_SCENARIOS,
    ids=[item.id for item in UNROLLED_SCENARIOS],
)
def test_declared_unrolled_labels_match_solver(scenario) -> None:
    assert declared_unrolled_ground_truth(
        scenario,
    ) == solver_unrolled_ground_truth(scenario)


def test_wrap_cycle_t2_keeps_wrap_path_dead() -> None:
    scenario = next(
        item for item in UNROLLED_SCENARIOS if item.id == "wrap_cycle_t2"
    )
    ground_truth = solver_unrolled_ground_truth(scenario)
    assert ground_truth.important_features == frozenset({"fast_in"})
    assert ground_truth.unimportant_features == frozenset(
        {
            "wrap_in",
            "decoy_in",
        }
    )
    assert ground_truth.important_nodes == frozenset({"fast_h"})
    assert ground_truth.unimportant_nodes == frozenset(
        {
            "wrap_u",
            "wrap_v",
            "decoy_h",
        }
    )


def test_wrap_cycle_t3_lights_the_wrap_path() -> None:
    scenario = next(
        item for item in UNROLLED_SCENARIOS if item.id == "wrap_cycle_t3"
    )
    ground_truth = solver_unrolled_ground_truth(scenario)
    assert ground_truth.important_features == frozenset(
        {
            "fast_in",
            "wrap_in",
        }
    )
    assert ground_truth.unimportant_features == frozenset({"decoy_in"})
    assert ground_truth.important_nodes == frozenset(
        {
            "fast_h",
            "wrap_u",
            "wrap_v",
        }
    )
    assert ground_truth.unimportant_nodes == frozenset({"decoy_h"})


def test_self_loop_does_not_make_input_live_at_t1() -> None:
    edgelist = memory_self_loop_graph(self_loop=True)
    ground_truth = solve_unrolled_ground_truth(
        edgelist=edgelist,
        attributed_outputs=["output"],
        n_steps=1,
    )
    assert ground_truth.important_features == frozenset()
    assert ground_truth.unimportant_features == frozenset(
        {
            "input_signal",
            "input_noise",
        }
    )
    t2 = solve_unrolled_ground_truth(
        edgelist=edgelist,
        attributed_outputs=["output"],
        n_steps=2,
    )
    assert "input_signal" in t2.important_features
    assert "node_a" in t2.important_nodes


def test_n_steps_less_than_one_raises() -> None:
    edgelist = memory_self_loop_graph()
    with pytest.raises(
        ValueError,
        match="n_steps",
    ):
        solve_unrolled_ground_truth(
            edgelist=edgelist,
            attributed_outputs=["output"],
            n_steps=0,
        )


def test_unrolled_scenario_ids_are_unique() -> None:
    ids = [scenario.id for scenario in UNROLLED_SCENARIOS]
    assert len(ids) == len(set(ids))


def test_unroll_graph_builders_keep_source_target_only() -> None:
    graphs = [
        wrap_cycle_graph(),
    ]
    for graph in graphs:
        assert list(graph.edgelist.columns) == [
            "source",
            "target",
        ]
    memory = memory_self_loop_graph()
    assert list(memory.columns) == ["source", "target"]
    broken = memory_self_loop_graph(self_loop=False)
    assert list(broken.columns) == ["source", "target"]
    assert ("node_a", "node_a") not in list(
        zip(
            broken["source"],
            broken["target"],
            strict=True,
        )
    )
