"""
Fast tests for trained-tier helpers: no model fitting.

Registered scenarios share the matched two-tower graph and
independent features. Linear-label ids differ only in LayeredNet
relu/bias. ``relu_product_feedforward`` uses the product of the
two tower-A features.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.controls.simulate import (
    linear_logit_labels,
    product_logit_labels,
)
from tests.controls.training import (
    LINEAR_MEASURED_PASSING,
    MIN_VAL_ROC_AUC,
    N_SEEDS,
    PRODUCT_HIDDEN_WIDTH,
    PRODUCT_MAX_RATIO,
    PRODUCT_MEASURED_PASSING,
    PRODUCT_MIN_PER_EXAMPLE_CONSISTENCY,
    RELU_BIAS_INIT,
    RELU_MEASURED_PASSING,
    SEEDS,
    TRAINED_SCENARIO_IDS,
    assert_decoys_structurally_live,
    linear_coefficients,
    linear_tower_simulator,
    matched_linear_towers,
    product_causal_features,
    product_tower_simulator,
    scenario_graph,
    seeds_passing_floor,
    trained_scenario,
)

SEED = 42
N_SAMPLES = 64
LINEAR_LABEL_SCENARIO_IDS = tuple(
    scenario_id
    for scenario_id in TRAINED_SCENARIO_IDS
    if not trained_scenario(scenario_id).product_labels
)


@pytest.mark.parametrize(
    "scenario_id",
    TRAINED_SCENARIO_IDS,
    ids=list(TRAINED_SCENARIO_IDS),
)
def test_matched_tower_decoys_are_structurally_live(
    scenario_id: str,
) -> None:
    scenario = trained_scenario(scenario_id)
    graph = scenario_graph(scenario)
    assert_decoys_structurally_live(graph)
    assert graph.outputs == ("prediction",)


@pytest.mark.parametrize(
    "scenario_id",
    LINEAR_LABEL_SCENARIO_IDS,
    ids=list(LINEAR_LABEL_SCENARIO_IDS),
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


def test_product_labels_follow_either_causal_feature_not_tower_b() -> None:
    graph = matched_linear_towers()
    causal = product_causal_features(graph)
    assert causal == graph.tower_a_features
    simulate = product_tower_simulator(graph)
    rng = np.random.default_rng(SEED)
    features, labels = simulate(
        rng,
        N_SAMPLES,
    )
    min_flipped = int(0.9 * N_SAMPLES)
    for name in causal:
        flipped = features.copy()
        flipped.loc[:, name] = -flipped.loc[:, name]
        labels_flipped = product_logit_labels(
            flipped,
            causal_features=causal,
        )
        n_flipped = int((labels != labels_flipped).sum())
        assert n_flipped >= min_flipped, (
            f"flipping causal feature {name!r} changed "
            f"{n_flipped}/{N_SAMPLES} labels, need >= {min_flipped}."
        )

    flipped_b = features.copy()
    flipped_b.loc[:, list(graph.tower_b_features)] = -flipped_b.loc[
        :,
        list(graph.tower_b_features),
    ]
    labels_flip_b = product_logit_labels(
        flipped_b,
        causal_features=causal,
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
    assert scenario.product_labels is False
    assert scenario.hidden_width is None
    assert scenario.min_val_roc_auc == MIN_VAL_ROC_AUC


def test_relu_feedforward_uses_relu_with_bias() -> None:
    scenario = trained_scenario("relu_feedforward")
    assert scenario.bias is True
    assert scenario.relu is True
    assert scenario.bias_init == RELU_BIAS_INIT
    assert scenario.include_hidden is False
    assert scenario.product_labels is False
    assert scenario.hidden_width is None
    assert scenario.min_val_roc_auc == MIN_VAL_ROC_AUC


def test_relu_product_feedforward_uses_relu_with_bias() -> None:
    scenario = trained_scenario("relu_product_feedforward")
    assert scenario.bias is True
    assert scenario.relu is True
    assert scenario.bias_init == RELU_BIAS_INIT
    assert scenario.include_hidden is False
    assert scenario.product_labels is True
    assert scenario.hidden_width == PRODUCT_HIDDEN_WIDTH
    assert scenario.max_ratio == PRODUCT_MAX_RATIO
    assert (
        scenario.min_per_example_consistency
        == PRODUCT_MIN_PER_EXAMPLE_CONSISTENCY
    )
    assert scenario.min_val_roc_auc == MIN_VAL_ROC_AUC
    graph = scenario_graph(scenario)
    causal = product_causal_features(graph)
    assert causal == graph.tower_a_features
    assert len(causal) == 2


def test_trained_seeds_are_a_pre_registered_block() -> None:
    assert N_SEEDS == 30
    assert SEEDS == tuple(range(42, 72))
    assert len(SEEDS) == N_SEEDS


def test_seeds_passing_floor_sits_below_the_measured_count() -> None:
    assert seeds_passing_floor(22) == 15
    assert seeds_passing_floor(19) == 12
    assert seeds_passing_floor(18) == 11
    assert seeds_passing_floor(0) == 0
    with pytest.raises(
        ValueError,
        match="n_passing",
    ):
        seeds_passing_floor(31)
    with pytest.raises(
        ValueError,
        match="n_seeds",
    ):
        seeds_passing_floor(
            1,
            n_seeds=0,
        )


def test_trained_scenario_bars_are_rate_floors() -> None:
    measured = {
        "linear_feedforward": LINEAR_MEASURED_PASSING,
        "relu_feedforward": RELU_MEASURED_PASSING,
        "relu_product_feedforward": PRODUCT_MEASURED_PASSING,
    }
    for scenario_id, n_passing in measured.items():
        scenario = trained_scenario(scenario_id)
        floor = seeds_passing_floor(n_passing)
        assert scenario.min_seeds_passing == floor
        assert 0 < scenario.min_seeds_passing < N_SEEDS
        assert scenario.min_seeds_passing < n_passing
