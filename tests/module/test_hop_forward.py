"""
Forward numerics of the grouped-hop wiring.

These are the cases a separate skip module used to cover: a skip
must reach its target's pre-activation, must not travel through
the adjacent weight matrix, and must survive a dead adjacent
path. Here they are ordinary entries of a hop mask.
"""

import pandas as pd
import torch
import torch.nn.functional as F

from kpnn2 import MaskedLinear, gather_hop_inputs, parse_layered
from tests.helpers.layered_net import (
    LayeredNet,
    pin_all_weights,
    pin_edge,
)


def _one_skip_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )


def _three_hop_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "B", "H1", "H2", "A", "H1", "A"],
            "target": ["H1", "H1", "H2", "C", "H2", "C", "C"],
        }
    )


def _pinned_three_hop_net():
    spec = parse_layered(_three_hop_edgelist())
    net = LayeredNet(
        spec,
        bias=False,
    )
    pin_all_weights(
        net,
        value=1.0,
    )
    for source, target, value in (
        ("A", "H2", 0.3),
        ("H1", "C", 0.2),
        ("A", "C", 0.1),
    ):
        pin_edge(
            net,
            source,
            target,
            value,
        )
    return spec, net


def test_hop_forward_matches_pinned_arithmetic():
    spec, net = _pinned_three_hop_net()
    assert spec.layer_nodes == (
        ("A", "B"),
        ("H1",),
        ("H2",),
        ("C",),
    )

    x_pos = torch.tensor(
        [[2.0, 0.0]],
        dtype=torch.float32,
    )
    c_pos = net(x_pos)
    torch.testing.assert_close(
        net.layer_tensors[1],
        torch.tensor(
            [[2.0]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        net.layer_tensors[2],
        torch.tensor(
            [[2.6]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        c_pos,
        torch.tensor(
            [[3.2]],
            dtype=torch.float32,
        ),
    )


def test_hop_forward_skip_survives_a_relu_zeroed_chain():
    _spec, net = _pinned_three_hop_net()

    x_neg = torch.tensor(
        [[-1.0, 0.0]],
        dtype=torch.float32,
    )
    c_neg = net(x_neg)
    torch.testing.assert_close(
        net.layer_tensors[1],
        torch.tensor(
            [[0.0]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        net.layer_tensors[2],
        torch.tensor(
            [[0.0]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        c_neg,
        torch.tensor(
            [[-0.1]],
            dtype=torch.float32,
        ),
    )


def test_hop_forward_skip_column_alone_drives_the_output():
    spec = parse_layered(_one_skip_edgelist())
    net = LayeredNet(
        spec,
        bias=False,
    )
    pin_all_weights(
        net,
        value=0.0,
    )
    pin_edge(
        net,
        "A",
        "C",
        0.3,
    )
    y_low = net(
        torch.zeros(
            1,
            1,
        )
    )
    y_high = net(
        torch.tensor(
            [[4.0]],
            dtype=torch.float32,
        )
    )
    assert y_low.item() == 0.0
    assert y_high.item() == torch.tensor(1.2).item()


def test_hop_forward_zeroing_the_skip_column_removes_only_that_term():
    spec = parse_layered(_one_skip_edgelist())
    net = LayeredNet(
        spec,
        bias=False,
    )
    pin_all_weights(
        net,
        value=1.0,
    )
    x = torch.tensor(
        [[2.0], [-1.0], [0.5]],
        dtype=torch.float32,
    )
    with_skip = net(x)
    pin_edge(
        net,
        "A",
        "C",
        0.0,
    )
    without_skip = net(x)

    torch.testing.assert_close(
        without_skip,
        with_skip - x,
    )
    torch.testing.assert_close(
        without_skip,
        F.relu(x),
    )


def test_hop_forward_equals_a_hand_written_residual_add():
    spec = parse_layered(_one_skip_edgelist())
    torch.manual_seed(42)
    layer0 = MaskedLinear(
        spec.hops[0].mask,
        bias=False,
    )
    layer1 = MaskedLinear(
        spec.hops[1].mask,
        bias=False,
    )
    weights = layer1.parametrizations.weight.original
    with torch.no_grad():
        weights.copy_(
            torch.tensor(
                [[0.3, 0.7]],
                dtype=torch.float32,
            )
        )
    x = torch.tensor(
        [[2.0], [-1.0]],
        dtype=torch.float32,
    )

    hidden = F.relu(layer0(x))
    saved = {
        0: x,
        1: hidden,
    }
    grouped = layer1(
        gather_hop_inputs(
            saved,
            spec.hops[1],
        )
    )
    adjacent = 0.7 * hidden
    residual = 0.3 * x
    torch.testing.assert_close(
        grouped,
        adjacent + residual,
    )


def test_hop_forward_gradient_reaches_the_skip_source_directly():
    spec = parse_layered(_one_skip_edgelist())
    net = LayeredNet(
        spec,
        bias=False,
    )
    pin_all_weights(
        net,
        value=1.0,
    )
    x = torch.tensor(
        [[-1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    net(x).sum().backward()

    assert x.grad is not None
    torch.testing.assert_close(
        x.grad,
        torch.tensor(
            [[1.0]],
            dtype=torch.float32,
        ),
    )


def test_hop_forward_ignores_saved_layers_a_hop_does_not_read():
    spec = parse_layered(_one_skip_edgelist())
    hop = spec.hops[0]
    saved = {
        0: torch.tensor([[3.0]]),
        1: torch.tensor([[99.0]]),
    }
    gathered = gather_hop_inputs(
        saved,
        hop,
    )
    assert gathered is saved[0]
