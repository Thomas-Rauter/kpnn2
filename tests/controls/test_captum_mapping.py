"""
Captum mapping workflow: IG tensors labeled by map_node_attributions.

kpnn2 does not import Captum. These tests run Captum here, then
pass the tensor to the public mapper. Captum is a dev extra, not
a core dependency.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from kpnn2 import (
    align_inputs,
    map_node_attributions,
    parse_layered,
)
from tests.controls.scenario import STRUCTURAL_SCENARIOS
from tests.controls.scoring import (
    DEAD_TOLERANCE,
    LIVE_FLOOR,
    SEED,
    independent_gaussian_features,
    median_abs_scores,
    pin_scenario_weights,
)
from tests.helpers.layered_net import LayeredNet

pytest.importorskip("captum")

_SCENARIO_ID = "dead_edge_feedforward"


def _set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _dead_edge_scenario():
    return next(
        item for item in STRUCTURAL_SCENARIOS if item.id == _SCENARIO_ID
    )


def _pinned_model(
    spec,
    scenario,
):
    model = LayeredNet(
        spec,
        bias=False,
        relu=False,
    )
    pin_scenario_weights(
        model,
        scenario,
    )
    model.eval()
    return model


def test_integrated_gradients_maps_input_node_order() -> None:
    """
    IG on inputs maps to spec.input_nodes, not a permutation.

    Graph: dead_edge_graph (scenario dead_edge_feedforward).
    Live path: signal_in -> live_h1 -> live_h2 -> prediction.
    Dead path: dead_in hop pinned at 0.
    """
    _set_seeds()
    scenario = _dead_edge_scenario()
    spec = parse_layered(scenario.edgelist)
    model = _pinned_model(
        spec,
        scenario,
    )
    features = independent_gaussian_features(spec)
    x = align_inputs(
        features,
        spec,
    )

    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        x,
        target=0,
    )
    mapped = map_node_attributions(
        attributions=attributions,
        spec=spec,
        layer=0,
    )

    expected_nodes = list(spec.layer_nodes[0])
    assert expected_nodes == list(spec.input_nodes)
    assert mapped["node"].values.tolist() == expected_nodes
    assert int(mapped.coords["layer"]) == 0

    table = mapped.to_pandas()
    raw = attributions.detach().cpu().numpy()
    for column, name in enumerate(expected_nodes):
        np.testing.assert_allclose(
            table[name].to_numpy(),
            raw[:, column],
        )

    scores = median_abs_scores(
        table,
        spec.input_nodes,
    )
    dead_name = scenario.unimportant_features[0]
    live_name = scenario.important_features[0]
    assert scores[dead_name] <= DEAD_TOLERANCE, (
        f"Dead input {dead_name!r} median |IG| was "
        f"{scores[dead_name]}, expected <= {DEAD_TOLERANCE}."
    )
    assert scores[live_name] > LIVE_FLOOR, (
        f"Live input {live_name!r} median |IG| was "
        f"{scores[live_name]}, expected > {LIVE_FLOOR}."
    )


def test_synthetic_captum_tensor_maps_hidden_layer_names() -> None:
    """
    Captum-shaped (observation, node) scores map at layer i+1.

    IntegratedGradients attributes inputs only. Hidden-layer
    LayerConductance on MaskedLinear is not required here: the
    mapper only needs a tensor whose node axis matches
    spec.layer_nodes[layer]. Hop 0 output is layer 1.
    """
    scenario = _dead_edge_scenario()
    spec = parse_layered(scenario.edgelist)
    hop = 0
    layer = hop + 1
    names = list(spec.layer_nodes[layer])
    n_obs = 4
    # Distinct values so a permutation of names would fail.
    values = torch.arange(
        n_obs * len(names),
        dtype=torch.float32,
    ).reshape(n_obs, len(names))

    mapped = map_node_attributions(
        attributions=values,
        spec=spec,
        layer=layer,
    )

    assert mapped["node"].values.tolist() == names
    assert int(mapped.coords["layer"]) == layer
    table = mapped.to_pandas()
    raw = values.numpy()
    for column, name in enumerate(names):
        np.testing.assert_allclose(
            table[name].to_numpy(),
            raw[:, column],
        )
