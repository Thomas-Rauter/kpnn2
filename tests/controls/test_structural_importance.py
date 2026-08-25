"""
Tier 1: pinned-weight structural importance, no training.

Scores are autograd magnitudes, not Captum. Feature scores are
|input grad|. Hidden scores are |activation × layer grad| at that
node's GraphSpec layer. ``LayeredNet`` is linear (relu=False) so
live paths have deterministic nonzero scores.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
import torch

from kpnn2 import parse_edgelist
from tests.controls.diagnostics import score_report
from tests.controls.scenario import (
    STRUCTURAL_SCENARIOS,
    StructuralScenario,
    declared_ground_truth,
    solver_ground_truth,
)
from tests.controls.scoring import (
    DEAD_TOLERANCE,
    LIVE_FLOOR,
    SEED,
    align_and_enable_grad,
    attributed_output_sum,
    feature_grad_table,
    hidden_score_table,
    independent_gaussian_features,
    max_abs_scores,
    median_abs_scores,
    pin_scenario_weights,
)
from tests.helpers.layered_net import LayeredNet


def _set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _run_scenario(
    scenario: StructuralScenario,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pin weights, forward, backward, return feature and hidden grads.
    """
    _set_seeds()
    spec = parse_edgelist(scenario.edgelist)
    model = LayeredNet(
        spec,
        bias=scenario.bias,
        relu=False,
    )
    pin_scenario_weights(
        model,
        scenario,
    )
    features = independent_gaussian_features(spec)
    x = align_and_enable_grad(
        features,
        spec,
    )
    output = model(x)
    attributed = attributed_output_sum(
        output,
        spec,
        scenario.attributed_outputs,
    )
    attributed.backward()
    assert x.grad is not None
    feature_table = feature_grad_table(
        x.grad,
        spec,
    )
    node_table = hidden_score_table(
        model,
        spec,
    )
    return feature_table, node_table


@pytest.mark.parametrize(
    "scenario",
    STRUCTURAL_SCENARIOS,
    ids=[item.id for item in STRUCTURAL_SCENARIOS],
)
def test_structural_importance_matches_ground_truth(
    scenario: StructuralScenario,
) -> None:
    solved = solver_ground_truth(scenario)
    assert declared_ground_truth(scenario) == solved

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


def test_swapped_ground_truth_fails_structural_criterion() -> None:
    """
    Mutation test: swapping important and unimportant must fail.

    The live tower is not near zero, so the dead-node assertion is
    sensitive to the declared ground truth.
    """
    scenario = next(
        item
        for item in STRUCTURAL_SCENARIOS
        if item.id == "disconnected_feedforward"
    )
    feature_table, node_table = _run_scenario(scenario)
    swapped = dataclasses.replace(
        scenario,
        important_features=scenario.unimportant_features,
        unimportant_features=scenario.important_features,
        important_nodes=scenario.unimportant_nodes,
        unimportant_nodes=scenario.important_nodes,
    )
    with pytest.raises(
        AssertionError,
        match="declared unimportant",
    ):
        _assert_unimportant_near_zero(
            scenario=swapped,
            label="features",
            table=feature_table,
            names=swapped.unimportant_features,
        )
    with pytest.raises(
        AssertionError,
        match="declared unimportant",
    ):
        _assert_unimportant_near_zero(
            scenario=swapped,
            label="hidden nodes",
            table=node_table,
            names=swapped.unimportant_nodes,
        )


def _assert_unimportant_near_zero(
    *,
    scenario: StructuralScenario,
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
    scenario: StructuralScenario,
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
