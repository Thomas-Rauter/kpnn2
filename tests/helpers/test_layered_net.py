import pandas as pd
import pytest
import torch

import kpnn2
from kpnn2 import parse_edgelist
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


def test_layered_net_is_not_public_api():
    assert "LayeredNet" not in kpnn2.__all__
    assert not hasattr(
        kpnn2,
        "LayeredNet",
    )
    assert "pin_all_weights" not in kpnn2.__all__
    assert "pin_edge" not in kpnn2.__all__


def test_layered_net_constructs_runs_and_pin_edge_zeros_skip():
    torch.manual_seed(42)
    spec = parse_edgelist(_chain_plus_skip())
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
    source = x[:, skip.source_index]
    expected = y_with_skip.clone()
    expected[:, skip.target_index] = (
        y_with_skip[:, skip.target_index] - 1.0 * source
    )
    torch.testing.assert_close(
        y_without_skip,
        expected,
    )


def test_pin_edge_raises_for_missing_edge():
    spec = parse_edgelist(_chain_plus_skip())
    model = LayeredNet(
        spec,
        bias=False,
    )
    with pytest.raises(
        ValueError,
        match="No edge",
    ):
        pin_edge(
            model,
            "H",
            "A",
            0.0,
        )
