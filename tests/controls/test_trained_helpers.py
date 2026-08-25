"""
Fast tests for trained-tier helpers: no model fitting.

Both registered scenarios share the matched two-tower graph and
linear tower-A labels. They differ only in LayeredNet relu/bias.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.controls.simulate import linear_logit_labels
from tests.controls.training import (
    RELU_BIAS_INIT,
    TRAINED_SCENARIO_IDS,
    assert_decoys_structurally_live,
    linear_coefficients,
    linear_tower_simulator,
    matched_linear_towers,
    trained_scenario,
)

SEED = 42
N_SAMPLES = 64


@pytest.mark.parametrize(
    "scenario_id",
    TRAINED_SCENARIO_IDS,
    ids=list(TRAINED_SCENARIO_IDS),
)
def test_matched_tower_decoys_are_structurally_live(
    scenario_id: str,
) -> None:
    trained_scenario(scenario_id)
    graph = matched_linear_towers()
    assert_decoys_structurally_live(graph)
    assert graph.outputs == ("prediction",)


@pytest.mark.parametrize(
    "scenario_id",
    TRAINED_SCENARIO_IDS,
    ids=list(TRAINED_SCENARIO_IDS),
)
def test_labels_follow_tower_a_not_tower_b(
    scenario_id: str,
) -> None:
    trained_scenario(scenario_id)
    graph = matched_linear_towers()
    simulate = linear_tower_simulator(graph)
    rng = np.random.default_rng(SEED)
    features, labels = simulate(
        rng,
        N_SAMPLES,
    )
    coefficients = linear_coefficients(len(graph.tower_a_features))

    flipped_a = features.copy()
    flipped_a.loc[:, list(graph.tower_a_features)] = -flipped_a.loc[
        :,
        list(graph.tower_a_features),
    ]
    labels_flip_a = linear_logit_labels(
        flipped_a,
        causal_features=graph.tower_a_features,
        coefficients=coefficients,
    )
    assert not np.array_equal(
        labels,
        labels_flip_a,
    )
    n_flipped = int((labels != labels_flip_a).sum())
    assert n_flipped >= int(0.9 * N_SAMPLES)

    flipped_b = features.copy()
    flipped_b.loc[:, list(graph.tower_b_features)] = -flipped_b.loc[
        :,
        list(graph.tower_b_features),
    ]
    labels_flip_b = linear_logit_labels(
        flipped_b,
        causal_features=graph.tower_a_features,
        coefficients=coefficients,
    )
    assert np.array_equal(
        labels,
        labels_flip_b,
    )


def test_linear_feedforward_is_linear_without_bias() -> None:
    scenario = trained_scenario("linear_feedforward")
    assert scenario.bias is False
    assert scenario.relu is False
    assert scenario.bias_init is None
    assert scenario.include_hidden is True


def test_relu_feedforward_uses_relu_with_bias() -> None:
    scenario = trained_scenario("relu_feedforward")
    assert scenario.bias is True
    assert scenario.relu is True
    assert scenario.bias_init == RELU_BIAS_INIT
    assert scenario.include_hidden is False
