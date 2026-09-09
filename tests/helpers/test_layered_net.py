import pandas as pd
import pytest
import torch

import kpnn2
from kpnn2 import Kpnn2Error, parse_layered
from tests.helpers.layered_net import (
    LayeredNet,
    pin_all_weights,
    pin_edge,
)


def _chain_plus_skip():
    return pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )


def test_layered_net_layers_carry_spec_fingerprint():
    spec = parse_layered(_chain_plus_skip())
    model = LayeredNet(
        spec,
        bias=False,
    )
    for layer in model.layers:
        assert layer.identity == spec.fingerprint
    assert "LayeredNet" not in kpnn2.__all__
    assert not hasattr(
        kpnn2,
        "LayeredNet",
    )
    assert "pin_all_weights" not in kpnn2.__all__
    assert "pin_edge" not in kpnn2.__all__


def test_layered_net_constructs_runs_and_pin_edge_zeros_skip():
    torch.manual_seed(42)
    spec = parse_layered(_chain_plus_skip())
    model = LayeredNet(
        spec,
        bias=False,
    )
    pin_all_weights(
        model,
        value=1.0,
    )
    x = torch.tensor(
        [
            [2.0],
            [-1.0],
            [0.5],
        ],
        dtype=torch.float32,
    )
    y_with_skip = model(x)
    assert y_with_skip.shape == (3, 1)

    pin_edge(
        model,
        "A",
        "C",
        0.0,
    )
    y_without_skip = model(x)
    assert not torch.equal(
        y_with_skip,
        y_without_skip,
    )
    skip = spec.skips[0]
    source = x[:, skip.source_in_layer]
    expected = y_with_skip.clone()
    expected[:, skip.target_in_layer] = (
        y_with_skip[:, skip.target_in_layer] - 1.0 * source
    )
    torch.testing.assert_close(
        y_without_skip,
        expected,
    )


def test_pin_edge_raises_for_missing_edge():
    spec = parse_layered(_chain_plus_skip())
    model = LayeredNet(
        spec,
        bias=False,
    )
    with pytest.raises(
        Kpnn2Error,
        match=r"H -> A",
    ):
        pin_edge(
            model,
            "H",
            "A",
            0.0,
        )


def test_pin_edge_pins_every_unit_pair_when_wide():
    spec = parse_layered(
        pd.DataFrame(
            {
                "source": ["A", "B"],
                "target": ["H", "H"],
            }
        ),
        widths={
            "A": 2,
            "H": 3,
        },
    )
    model = LayeredNet(
        spec,
        bias=False,
    )
    pin_all_weights(
        model,
        value=1.0,
    )
    pin_edge(
        model,
        "A",
        "H",
        0.25,
    )
    hop_index, packed = spec.edge_location(
        "A",
        "H",
    )
    weights = model.layers[hop_index].weight.detach()
    assert len(packed) == 6
    assert weights[list(packed)].tolist() == [0.25] * 6
    _, other = spec.edge_location(
        "B",
        "H",
    )
    assert weights[list(other)].tolist() == [1.0] * len(other)
