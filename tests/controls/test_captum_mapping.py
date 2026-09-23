"""
Captum mapping workflow: IG tensors labeled by map_node_attributions.

kpnn2 does not import Captum. These tests run Captum here, then
pass the tensor to the public mapper. Captum is a dev extra, not
a core dependency.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch

from kpnn2 import (
    map_node_attributions,
    parse_layered,
)
from tests.controls.scenario import STRUCTURAL_SCENARIOS
from tests.controls.scoring import (
    DEAD_TOLERANCE,
    LIVE_FLOOR,
    SEED,
    aligned_feature_tensor,
    independent_gaussian_features,
    max_abs_scores,
    median_abs_scores,
    pin_scenario_weights,
)
from tests.helpers.layered_net import (
    LayeredNet,
    pin_all_weights,
    pin_edge,
)

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
    x = aligned_feature_tensor(
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

    Hop output scores map with layer= (node axis matches that
    depth's units) or hop_output= on the same hop. Hop 0
    output is layer 1. Captum runs on real modules in the
    LayerConductance tests below.
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


def _square_hop_spec():
    """
    ``hops[0]`` reads A, B and writes H1, H2: equal widths.

    Either side of that hop fits a ``(batch, 2)`` tensor, so the
    node axis length cannot tell the sides apart.
    """
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B", "A", "B", "H1", "H2"],
            "target": ["H1", "H1", "H2", "H2", "C", "C"],
        }
    )
    return parse_layered(edgelist)


def _square_hop_conductance(
    dead_edges: tuple[tuple[str, str], ...],
    attribute_to_layer_input: bool,
):
    """
    LayerConductance on ``hops[0]`` with ``dead_edges`` pinned at 0.
    """
    _set_seeds()
    spec = _square_hop_spec()
    model = LayeredNet(
        spec,
        bias=False,
        relu=False,
    )
    pin_all_weights(model)
    for source, target in dead_edges:
        pin_edge(
            model,
            source,
            target,
            0.0,
        )
    model.eval()
    x = torch.randn(16, 2) + 2.0

    from captum.attr import LayerConductance

    conductance = LayerConductance(
        model,
        model.layers[0],
    )
    attributions = conductance.attribute(
        x,
        target=0,
        attribute_to_layer_input=attribute_to_layer_input,
    )
    return spec, attributions


def test_layer_conductance_hop_output_names_dead_hidden_node() -> None:
    """
    Captum's default layer side is the module output.

    H1 has no path to C (H1 -> C pinned at 0), so its
    conductance is zero. hop_output=spec.hops[0] must put that
    zero column under H1, the unit the hop module writes.
    """
    spec, attributions = _square_hop_conductance(
        dead_edges=(("H1", "C"),),
        attribute_to_layer_input=False,
    )
    hop = spec.hops[0]

    mapped = map_node_attributions(
        attributions=attributions,
        spec=spec,
        hop_output=hop,
    )

    assert mapped["node"].values.tolist() == ["H1", "H2"]
    assert int(mapped.coords["layer"]) == hop.target_layer
    table = mapped.to_pandas()
    assert max_abs_scores(table, ("H1",))["H1"] <= DEAD_TOLERANCE
    assert median_abs_scores(table, ("H2",))["H2"] > LIVE_FLOOR


def test_layer_conductance_hop_input_names_dead_input() -> None:
    """
    attribute_to_layer_input=True scores what the module reads.

    A feeds nothing (A -> H1 and A -> H2 pinned at 0), so its
    conductance at the input of hops[0] is zero.
    hop_input=spec.hops[0] must put that zero column under A.
    """
    spec, attributions = _square_hop_conductance(
        dead_edges=(
            ("A", "H1"),
            ("A", "H2"),
        ),
        attribute_to_layer_input=True,
    )
    hop = spec.hops[0]

    mapped = map_node_attributions(
        attributions=attributions,
        spec=spec,
        hop_input=hop,
    )

    assert mapped["node"].values.tolist() == ["A", "B"]
    assert "layer" not in mapped.coords
    table = mapped.to_pandas()
    assert max_abs_scores(table, ("A",))["A"] <= DEAD_TOLERANCE
    assert median_abs_scores(table, ("B",))["B"] > LIVE_FLOOR
