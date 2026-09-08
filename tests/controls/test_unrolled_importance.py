"""
Pinned-weight unrolled importance, no training.

Scores are autograd magnitudes. Feature scores are |input grad|
on the aligned ``x``. Hidden scores are max-over-steps
|activation × grad| named by ``map_node_attributions`` without
``layer``. ``AdjacencyNet`` is linear so a live walk of length
``<= T`` has a deterministic nonzero score.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
import torch

from kpnn2 import parse_adjacency
from tests.controls.diagnostics import score_report
from tests.controls.scoring import (
    DEAD_TOLERANCE,
    LIVE_FLOOR,
    SEED,
    align_and_enable_grad,
    independent_gaussian_features,
    max_abs_scores,
    median_abs_scores,
)
from tests.controls.unroll import (
    UNROLLED_SCENARIOS,
    UnrolledScenario,
    attributed_output_sum_adjacency,
    declared_unrolled_ground_truth,
    feature_grad_table_adjacency,
    hidden_score_table_unrolled,
    solver_unrolled_ground_truth,
)
from tests.helpers.adjacency_net import (
    AdjacencyNet,
    pin_all_weights,
)


def _set_seeds() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _run_scenario(
    scenario: UnrolledScenario,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pin weights, unroll T steps, return feature and hidden scores.
    """
    _set_seeds()
    spec = parse_adjacency(scenario.edgelist)
    model = AdjacencyNet(
        spec,
        n_steps=scenario.n_steps,
        bias=False,
        relu=False,
    )
    pin_all_weights(
        model,
        value=1.0,
    )
    features = independent_gaussian_features(spec)
    x = align_and_enable_grad(
        features,
        spec,
    )
    output = model(x)
    attributed = attributed_output_sum_adjacency(
        output,
        spec,
        scenario.attributed_outputs,
    )
    attributed.backward()
    assert x.grad is not None
    feature_table = feature_grad_table_adjacency(
        x.grad,
        spec,
    )
    node_table = hidden_score_table_unrolled(
        model.step_states,
        spec,
    )
    return feature_table, node_table


@pytest.mark.parametrize(
    "scenario",
    UNROLLED_SCENARIOS,
    ids=[item.id for item in UNROLLED_SCENARIOS],
)
def test_unrolled_importance_matches_ground_truth(
    scenario: UnrolledScenario,
) -> None:
    solved = solver_unrolled_ground_truth(scenario)
    assert declared_unrolled_ground_truth(scenario) == solved

    feature_table, node_table = _run_scenario(scenario)
    _assert_unimportant_near_zero(
        scenario=scenario,
        label="features",
        table=feature_table,
        names=scenario.unimportant_features,
    )
    _assert_unimportant_near_zero(
        scenario=scenario,
        label="hidden nodes",
        table=node_table,
        names=scenario.unimportant_nodes,
    )
    _assert_important_above_floor(
        scenario=scenario,
        label="features",
        table=feature_table,
        names=scenario.important_features,
    )
    _assert_important_above_floor(
        scenario=scenario,
        label="hidden nodes",
        table=node_table,
        names=scenario.important_nodes,
    )


def test_swapped_t_labels_fail_unrolled_criterion() -> None:
    """
    Mutation: T=2 labels on a T=3 run, and the reverse, must fail.
    """
    t2 = next(item for item in UNROLLED_SCENARIOS if item.id == "wrap_cycle_t2")
    t3 = next(item for item in UNROLLED_SCENARIOS if item.id == "wrap_cycle_t3")
    feature_t3, node_t3 = _run_scenario(t3)
    with pytest.raises(
        AssertionError,
        match="declared unimportant",
    ):
        _assert_unimportant_near_zero(
            scenario=t2,
            label="features",
            table=feature_t3,
            names=t2.unimportant_features,
        )
    with pytest.raises(
        AssertionError,
        match="declared unimportant",
    ):
        _assert_unimportant_near_zero(
            scenario=t2,
            label="hidden nodes",
            table=node_t3,
            names=t2.unimportant_nodes,
        )
    feature_t2, node_t2 = _run_scenario(t2)
    with pytest.raises(
        AssertionError,
        match="declared important",
    ):
        _assert_important_above_floor(
            scenario=t3,
            label="features",
            table=feature_t2,
            names=t3.important_features,
        )
    with pytest.raises(
        AssertionError,
        match="declared important",
    ):
        _assert_important_above_floor(
            scenario=t3,
            label="hidden nodes",
            table=node_t2,
            names=t3.important_nodes,
        )


def _assert_unimportant_near_zero(
    *,
    scenario: UnrolledScenario,
    label: str,
    table: pd.DataFrame,
    names: Sequence[str],
) -> None:
    important = (
        scenario.important_features
        if label == "features"
        else scenario.important_nodes
    )
    scores = max_abs_scores(
        table,
        tuple(important) + tuple(names),
    )
    offenders = [name for name in names if scores[name] > DEAD_TOLERANCE]
    report = score_report(
        scenario_id=scenario.id,
        label=label,
        scores=scores,
        important=important,
        unimportant=names,
    )
    assert not offenders, (
        f"Scenario '{scenario.id}': declared unimportant {label} "
        f"must have max |autograd| <= {DEAD_TOLERANCE}, but "
        f"{offenders} did not.\n{report}"
    )


def _assert_important_above_floor(
    *,
    scenario: UnrolledScenario,
    label: str,
    table: pd.DataFrame,
    names: Sequence[str],
) -> None:
    unimportant = (
        scenario.unimportant_features
        if label == "features"
        else scenario.unimportant_nodes
    )
    scores = median_abs_scores(
        table,
        tuple(names) + tuple(unimportant),
    )
    too_small = [name for name in names if scores[name] <= LIVE_FLOOR]
    report = score_report(
        scenario_id=scenario.id,
        label=label,
        scores=scores,
        important=names,
        unimportant=unimportant,
    )
    assert not too_small, (
        f"Scenario '{scenario.id}': declared important {label} "
        f"must have median |autograd| > {LIVE_FLOOR}, but "
        f"{too_small} did not.\n{report}"
    )
