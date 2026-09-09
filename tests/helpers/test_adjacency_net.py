import pandas as pd
import pytest
import torch

import kpnn2
from kpnn2 import Kpnn2Error, parse_adjacency
from tests.helpers.adjacency_net import (
    AdjacencyNet,
    pin_all_weights,
    pin_edge,
)


def _tiny_cycle():
    return pd.DataFrame(
        {
            "source": ["A", "H", "K", "H"],
            "target": ["H", "K", "H", "C"],
        }
    )


def test_adjacency_net_is_not_public_api():
    assert "AdjacencyNet" not in kpnn2.__all__
    assert not hasattr(
        kpnn2,
        "AdjacencyNet",
    )
    assert "pin_all_weights" not in kpnn2.__all__
    assert "pin_edge" not in kpnn2.__all__


def test_adjacency_net_constructs_runs_and_pin_edge():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_cycle())
    model = AdjacencyNet(
        spec,
        n_steps=2,
        bias=False,
        relu=False,
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
    y_with_cycle = model(x)
    assert y_with_cycle.shape == (3, 1)

    pin_edge(
        model,
        "H",
        "C",
        0.0,
    )
    y_without = model(x)
    assert not torch.equal(
        y_with_cycle,
        y_without,
    )


def test_pin_edge_raises_for_missing_edge():
    spec = parse_adjacency(_tiny_cycle())
    model = AdjacencyNet(
        spec,
        n_steps=1,
        bias=False,
    )
    with pytest.raises(
        Kpnn2Error,
        match=r"C -> A",
    ):
        pin_edge(
            model,
            "C",
            "A",
            0.0,
        )


def test_n_steps_less_than_one_raises():
    spec = parse_adjacency(_tiny_cycle())
    with pytest.raises(
        ValueError,
        match="n_steps",
    ):
        AdjacencyNet(
            spec,
            n_steps=0,
        )


def test_sequence_length_must_match_n_steps():
    spec = parse_adjacency(_tiny_cycle())
    model = AdjacencyNet(
        spec,
        n_steps=2,
        bias=False,
        relu=False,
    )
    x = torch.zeros(1, 3, 1)
    with pytest.raises(
        ValueError,
        match="n_steps",
    ):
        model(x)
