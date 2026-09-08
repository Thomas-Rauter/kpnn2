"""
No-memory prior: a self-loop is what carries an early pulse.

``G_true`` has ``node_a -> node_a``. ``G_broken`` keeps the same
feature and hidden names but drops that self-loop. Labels are an
early pulse on ``input_signal`` at times 0 and 1; the last frame
is not enough. ``G_true`` must fit. ``G_broken`` must stay at
chance.

Measured on seed 42: G_true val ROC-AUC = 0.991; G_broken =
0.450. Marked integration/slow.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from kpnn2 import align_inputs, parse_adjacency
from tests.controls.graphs import memory_self_loop_graph
from tests.controls.training import MIN_VAL_ROC_AUC
from tests.helpers.adjacency_net import AdjacencyNet

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_SEED = 42
_N_STEPS = 6
_N_EPOCHS = 150
_N_PER_CLASS_TRAIN = 80
_N_PER_CLASS_EVAL = 40
_LEARNING_RATE = 0.05
_PULSE = 2.5
_PULSE_TIMES = (0, 1)
# Same ceiling as shuffled-label / rewired-prior.
_MAX_CHANCE_ROC_AUC = 0.6


def test_no_memory_prior_cannot_learn_without_self_loop() -> None:
    """
    Training on an early pulse needs ``node_a -> node_a``.
    """
    graph_true = memory_self_loop_graph(self_loop=True)
    graph_broken = memory_self_loop_graph(self_loop=False)
    spec_true = parse_adjacency(graph_true)
    spec_broken = parse_adjacency(graph_broken)
    assert spec_true.input_nodes == spec_broken.input_nodes
    assert spec_true.output_nodes == spec_broken.output_nodes
    assert spec_true.hidden_nodes == spec_broken.hidden_nodes

    rng = np.random.default_rng(_SEED)
    x_train_np, y_train = _simulate_sequences(
        rng,
        n_per_class=_N_PER_CLASS_TRAIN,
        feature_names=list(spec_true.input_nodes),
    )
    x_eval_np, y_eval = _simulate_sequences(
        rng,
        n_per_class=_N_PER_CLASS_EVAL,
        feature_names=list(spec_true.input_nodes),
    )
    x_train = _align_sequences(
        x_train_np,
        spec_true,
    )
    x_eval = _align_sequences(
        x_eval_np,
        spec_true,
    )
    y_train_tensor = torch.as_tensor(
        y_train,
        dtype=torch.float32,
    ).reshape(-1, 1)
    y_eval_tensor = torch.as_tensor(
        y_eval,
        dtype=torch.float32,
    ).reshape(-1, 1)

    run_true = _train_and_eval(
        spec_true,
        x_train=x_train,
        y_train=y_train_tensor,
        x_eval=x_eval,
        y_eval=y_eval_tensor,
    )
    run_broken = _train_and_eval(
        spec_broken,
        x_train=x_train,
        y_train=y_train_tensor,
        x_eval=x_eval,
        y_eval=y_eval_tensor,
    )
    reports = (
        f"G_true val_roc_auc={run_true:.3f} "
        f"G_broken val_roc_auc={run_broken:.3f}"
    )
    assert run_true >= MIN_VAL_ROC_AUC, (
        "learnability PRECONDITION failed: G_true did not fit "
        "early-pulse labels (val ROC-AUC="
        f"{run_true:.3f}, need >= {MIN_VAL_ROC_AUC}). "
        "The no-memory check is not meaningful if the self-loop "
        f"graph cannot learn.\n{reports}"
    )
    assert run_broken < _MAX_CHANCE_ROC_AUC, (
        "no-memory prior still learned early-pulse labels, so a "
        "model can fit without a self-loop to carry the pulse "
        "(val ROC-AUC="
        f"{run_broken:.3f}, need < {_MAX_CHANCE_ROC_AUC}).\n"
        f"{reports}"
    )


def _simulate_sequences(
    rng: np.random.Generator,
    *,
    n_per_class: int,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    n_features = len(feature_names)
    n_samples = 2 * n_per_class
    x = rng.normal(
        0.0,
        1.0,
        size=(n_samples, _N_STEPS, n_features),
    )
    y = np.array([0] * n_per_class + [1] * n_per_class)
    signal_idx = feature_names.index("input_signal")
    class_1 = y == 1
    for time in _PULSE_TIMES:
        x[class_1, time, signal_idx] += _PULSE
    return x, y


def _align_sequences(
    x_np: np.ndarray,
    spec,
) -> torch.Tensor:
    feature_names = list(spec.input_nodes)
    n_samples, n_steps, _n_features = x_np.shape
    rows = []
    for sample in range(n_samples):
        for time in range(n_steps):
            rows.append(
                {
                    name: float(x_np[sample, time, index])
                    for index, name in enumerate(feature_names)
                }
            )
    flat = align_inputs(
        pd.DataFrame(rows),
        spec,
    )
    return flat.reshape(
        n_samples,
        n_steps,
        len(spec.input_nodes),
    )


def _train_and_eval(
    spec,
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
) -> float:
    _set_seed(_SEED)
    model = AdjacencyNet(
        spec,
        n_steps=_N_STEPS,
        bias=True,
        relu=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=_LEARNING_RATE,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _epoch in range(_N_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = loss_fn(
            logits,
            y_train,
        )
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_eval))
    return float(
        roc_auc_score(
            y_eval.detach().cpu().numpy().reshape(-1),
            probabilities.detach().cpu().numpy().reshape(-1),
        )
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
