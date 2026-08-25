"""
Fast tests for trained-tier helpers: no model fitting.
"""

from __future__ import annotations

import numpy as np

from tests.controls.simulate import linear_logit_labels
from tests.controls.training import (
    assert_decoys_structurally_live,
    linear_coefficients,
    linear_tower_simulator,
    matched_linear_towers,
)

SEED = 42
N_SAMPLES = 64


def test_matched_tower_decoys_are_structurally_live() -> None:
    graph = matched_linear_towers()
    assert_decoys_structurally_live(graph)
    assert graph.outputs == ("prediction",)


def test_labels_follow_tower_a_not_tower_b() -> None:
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
