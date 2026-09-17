"""
rauter_mangano_2026 on real Integrated Gradients.

Two simulations train one-layer logit nets on several seeds, score
held-out observations with Captum IG (``target=None``), and pass the
seed-stacked attributions to ``aggregate_node_attributions``. The
expected signs, ranks, and flags are fixed by the simulation
design, not read off a run.

Feature groups: seven groups of five Gaussian features (spread 1)
differ only in their class means. The expected signs and ranks must
hold with and without ``correct_sign``. Individual seeds can still
flag predictive features: with 35 redundant features, some weights
keep a wrong sign from their initialization while the net predicts
perfectly.

Batch effect: two features share a nuisance ``z`` without class
information. The net subtracts ``x2`` from ``x1`` to cancel ``z``,
so ``x2`` gets a negative weight although it is higher in class 1.
Its class 1 mean attribution is then below its class 0 mean in
every seed, which is the case where winner minus loser and
``correct_sign=True`` disagree in sign.
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

_FLAG = "mean_class1_below_class0"

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
_NULL_GROUP = 2
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

_BATCH_N_PER_CLASS = 1000
_BATCH_N_TEST = 400
_BATCH_SD = 3.0
_BATCH_NOISE_SD = 0.5
_BATCH_N_SEEDS = 20
_BATCH_N_EPOCHS = 300
_BATCH_LEARNING_RATE = 0.05
# About 0.85 is the best reachable accuracy for this noise level.
_BATCH_MIN_ACCURACY = 0.8


def _split(features, labels, n_test):
    order = np.random.permutation(len(labels))
    test = order[:n_test]
    train = order[n_test:]
    return (
        features[train],
        labels[train],
        features[test],
        labels[test],
    )


def _simulate_groups():
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
    return _split(
        features,
        labels,
        _N_TEST,
    )


def _simulate_batch_effect():
    np.random.seed(42)
    n = 2 * _BATCH_N_PER_CLASS
    labels = np.array([0] * _BATCH_N_PER_CLASS + [1] * _BATCH_N_PER_CLASS)
    batch = np.random.normal(0.0, _BATCH_SD, n)
    x1 = (
        5.0
        + batch
        + 2.0 * labels
        + np.random.normal(
            0.0,
            _BATCH_NOISE_SD,
            n,
        )
    )
    x2 = (
        5.0
        + batch
        + 0.5 * labels
        + np.random.normal(
            0.0,
            _BATCH_NOISE_SD,
            n,
        )
    )
    return _split(
        np.column_stack([x1, x2]),
        labels,
        _BATCH_N_TEST,
    )


def _train_and_attribute(
    x_train,
    y_train,
    x_test,
    y_test,
    *,
    seed,
    n_epochs,
    learning_rate,
):
    torch.manual_seed(seed)
    model = torch.nn.Linear(x_train.shape[1], 1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    inputs = torch.tensor(x_train, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
    model.eval()
    test_inputs = torch.tensor(x_test, dtype=torch.float32)
    with torch.no_grad():
        predicted = (model(test_inputs) > 0).squeeze(1).numpy()
    accuracy = float((predicted == y_test).mean())
    attributions = IntegratedGradients(model).attribute(
        test_inputs,
        target=None,
    )
    return attributions.detach().numpy(), accuracy


def _stack_seeds(
    x_train,
    y_train,
    x_test,
    y_test,
    *,
    nodes,
    n_seeds,
    n_epochs,
    learning_rate,
):
    per_seed = []
    accuracies = []
    for seed in range(n_seeds):
        attributions, accuracy = _train_and_attribute(
            x_train,
            y_train,
            x_test,
            y_test,
            seed=seed,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
        )
        per_seed.append(attributions)
        accuracies.append(accuracy)
    stacked = xr.DataArray(
        np.stack(per_seed),
        dims=("seed", "observation", "node"),
        coords={"node": nodes},
    )
    return stacked, np.array(accuracies)


def _aggregate_both(attributions, labels):
    plain = aggregate_node_attributions(
        attributions,
        labels,
        class_0=0,
        class_1=1,
    )
    corrected = aggregate_node_attributions(
        attributions,
        labels,
        class_0=0,
        class_1=1,
        correct_sign=True,
    )
    return plain, corrected


@pytest.fixture(scope="module")
def groups():
    x_train, y_train, x_test, y_test = _simulate_groups()
    nodes = [
        f"group{group + 1}_feature{feature + 1}"
        for group in range(len(_GROUP_MEANS))
        for feature in range(_GROUP_SIZE)
    ]
    attributions, _ = _stack_seeds(
        x_train,
        y_train,
        x_test,
        y_test,
        nodes=nodes,
        n_seeds=_N_SEEDS,
        n_epochs=_N_EPOCHS,
        learning_rate=_LEARNING_RATE,
    )
    return _aggregate_both(attributions, y_test)


@pytest.fixture(scope="module")
def batch_effect():
    x_train, y_train, x_test, y_test = _simulate_batch_effect()
    attributions, accuracies = _stack_seeds(
        x_train,
        y_train,
        x_test,
        y_test,
        nodes=["x1", "x2"],
        n_seeds=_BATCH_N_SEEDS,
        n_epochs=_BATCH_N_EPOCHS,
        learning_rate=_BATCH_LEARNING_RATE,
    )
    plain, corrected = _aggregate_both(attributions, y_test)
    return plain, corrected, accuracies


def _by_group(values):
    return values.reshape(len(_GROUP_MEANS), _GROUP_SIZE)


def _group_scores(groups, correct_sign):
    plain, corrected = groups
    dataset = corrected if correct_sign else plain
    return _by_group(dataset["score"].values)


_BOTH_RULES = pytest.mark.parametrize(
    "correct_sign",
    [False, True],
)


@_BOTH_RULES
def test_every_feature_has_its_group_sign(groups, correct_sign):
    scores = _group_scores(
        groups,
        correct_sign,
    )
    for group, sign in enumerate(_EXPECTED_SIGN):
        if sign == 0:
            continue
        assert np.all(np.sign(scores[group]) == sign), (
            group + 1,
            scores[group],
        )


@_BOTH_RULES
def test_non_predictive_group_scores_near_zero(groups, correct_sign):
    scores = _group_scores(
        groups,
        correct_sign,
    )
    gap4 = np.abs(scores[[0, 1, 3, 4]].mean(axis=1))
    null = np.abs(scores[_NULL_GROUP])
    assert null.max() < _NULL_FRACTION * gap4.min(), (null, gap4)


@_BOTH_RULES
def test_equal_gap_groups_are_comparable(groups, correct_sign):
    scores = _group_scores(
        groups,
        correct_sign,
    )
    gap4 = np.abs(scores[[0, 1, 3, 4]].mean(axis=1))
    assert gap4.max() / gap4.min() < _COMPARABLE_RATIO, gap4


@_BOTH_RULES
def test_opposite_sign_groups_rank_above_equal_sign_groups(
    groups, correct_sign
):
    scores = _group_scores(
        groups,
        correct_sign,
    )
    gap4 = np.abs(scores[[0, 1, 3, 4]].mean(axis=1))
    gap8 = np.abs(scores[[5, 6]].mean(axis=1))
    assert gap8.min() > gap4.max(), (gap8, gap4)


def test_batch_effect_model_predicts(batch_effect):
    _, _, accuracies = batch_effect
    assert accuracies.min() > _BATCH_MIN_ACCURACY, accuracies


def test_batch_effect_flags_the_correction_feature(batch_effect):
    plain, _, _ = batch_effect
    flags = plain[_FLAG]
    assert flags.sel(node="x2").values.all()
    assert not flags.sel(node="x1").values.any()


def test_batch_effect_scores_disagree_only_for_the_flagged_feature(
    batch_effect,
):
    plain, corrected, _ = batch_effect
    x1_plain = plain["score"].sel(node="x1").item()
    x2_plain = plain["score"].sel(node="x2").item()
    assert x1_plain > 0
    assert x2_plain < 0
    assert corrected["score"].sel(node="x1").item() == pytest.approx(
        x1_plain,
    )
    assert corrected["score"].sel(node="x2").item() == pytest.approx(
        -x2_plain,
    )
