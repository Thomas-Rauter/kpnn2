"""
rauter_mangano_2026 on real Integrated Gradients.

Seven groups of five Gaussian features (spread 1) differ only in
their class means. A one-layer logit net is trained on 50 seeds,
Captum IG (``target=None``) scores the held-out observations, and
the seed-stacked attributions go through
``aggregate_node_attributions``. The expected signs and ranks are
fixed by the simulation design, not read off a run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import xarray as xr

from kpnn2 import aggregate_node_attributions

pytest.importorskip("captum")

from captum.attr import IntegratedGradients  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# (class 0 mean, class 1 mean) per group of five features.
_GROUP_MEANS = (
    (5.0, 1.0),
    (1.0, 5.0),
    (3.0, 3.0),
    (-5.0, -1.0),
    (-1.0, -5.0),
    (5.0, -3.0),
    (-3.0, 5.0),
)
_EXPECTED_SIGN = (-1, 1, 0, -1, 1, -1, 1)
_GROUP_SIZE = 5
_N_PER_CLASS = 500
_N_TEST = 200
_N_SEEDS = 50
_N_EPOCHS = 100
_LEARNING_RATE = 0.001
# Group 3 counts as 0 below this fraction of the 4-gap groups.
_NULL_FRACTION = 0.1
# Groups 1, 2, 4, and 5 are comparable within this ratio.
_COMPARABLE_RATIO = 1.5


def _simulate():
    np.random.seed(42)
    blocks0 = []
    blocks1 = []
    for mean0, mean1 in _GROUP_MEANS:
        blocks0.append(
            np.random.normal(
                loc=mean0,
                scale=1.0,
                size=(_N_PER_CLASS, _GROUP_SIZE),
            )
        )
        blocks1.append(
            np.random.normal(
                loc=mean1,
                scale=1.0,
                size=(_N_PER_CLASS, _GROUP_SIZE),
            )
        )
    features = np.vstack(
        [
            np.hstack(blocks0),
            np.hstack(blocks1),
        ]
    )
    labels = np.array([0] * _N_PER_CLASS + [1] * _N_PER_CLASS)
    order = np.random.permutation(len(labels))
    test = order[:_N_TEST]
    train = order[_N_TEST:]
    return (
        features[train],
        labels[train],
        features[test],
        labels[test],
    )


def _train_and_attribute(x_train, y_train, x_test, seed):
    torch.manual_seed(seed)
    model = torch.nn.Linear(x_train.shape[1], 1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=_LEARNING_RATE,
    )
    inputs = torch.tensor(x_train, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(_N_EPOCHS):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
    model.eval()
    attributions = IntegratedGradients(model).attribute(
        torch.tensor(x_test, dtype=torch.float32),
        target=None,
    )
    return attributions.detach().numpy()


@pytest.fixture(scope="module")
def group_scores():
    x_train, y_train, x_test, y_test = _simulate()
    per_seed = [
        _train_and_attribute(
            x_train,
            y_train,
            x_test,
            seed,
        )
        for seed in range(_N_SEEDS)
    ]
    nodes = [
        f"group{group + 1}_feature{feature + 1}"
        for group in range(len(_GROUP_MEANS))
        for feature in range(_GROUP_SIZE)
    ]
    attributions = xr.DataArray(
        np.stack(per_seed),
        dims=("seed", "observation", "node"),
        coords={"node": nodes},
    )
    out = aggregate_node_attributions(
        attributions,
        y_test,
        class_0=0,
        class_1=1,
    )
    return out["score"].values.reshape(
        len(_GROUP_MEANS),
        _GROUP_SIZE,
    )


def test_every_feature_has_its_group_sign(group_scores):
    for group, sign in enumerate(_EXPECTED_SIGN):
        if sign == 0:
            continue
        scores = group_scores[group]
        assert np.all(np.sign(scores) == sign), (group + 1, scores)


def test_non_predictive_group_scores_near_zero(group_scores):
    gap4 = np.abs(group_scores[[0, 1, 3, 4]].mean(axis=1))
    null = np.abs(group_scores[2])
    assert null.max() < _NULL_FRACTION * gap4.min(), (null, gap4)


def test_equal_gap_groups_are_comparable(group_scores):
    gap4 = np.abs(group_scores[[0, 1, 3, 4]].mean(axis=1))
    assert gap4.max() / gap4.min() < _COMPARABLE_RATIO, gap4


def test_opposite_sign_groups_rank_above_equal_sign_groups(
    group_scores,
):
    gap4 = np.abs(group_scores[[0, 1, 3, 4]].mean(axis=1))
    gap8 = np.abs(group_scores[[5, 6]].mean(axis=1))
    assert gap8.min() > gap4.max(), (gap8, gap4)
