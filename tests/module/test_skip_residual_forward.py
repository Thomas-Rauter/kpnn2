import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from kpnn2 import MaskedLinear, parse_layered


class SkipResidualNet(nn.Module):
    """
    User-owned module: adjacent hops plus skip residuals.
    """

    def __init__(
        self,
        spec,
        bias=True,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.layer0 = MaskedLinear(
            spec.masks[0],
            bias=bias,
        )
        self.layer1 = MaskedLinear(
            spec.masks[1],
            bias=bias,
        )
        self.w_skip = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        a: torch.Tensor,
    ) -> torch.Tensor:
        h = F.relu(self.layer0(a))
        c = self.layer1(h)
        saved = {
            0: a,
            1: h,
        }
        for skip in self.spec.skips:
            source = saved[skip.source_layer][
                :,
                skip.source_index,
            ]
            addition = torch.zeros_like(c)
            addition[:, skip.target_index] = self.w_skip * source
            c = c + addition
        return c


def _skip_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )


def _known_input():
    return torch.tensor(
        [
            [2.0],
            [-1.0],
            [0.5],
        ],
        dtype=torch.float32,
    )


def _pinned_adjacent_net(
    spec,
    w_skip,
):
    model = SkipResidualNet(
        spec,
        bias=False,
    )
    with torch.no_grad():
        model.layer0.parametrizations.weight.original.fill_(1.0)
        model.layer1.parametrizations.weight.original.fill_(1.0)
        model.w_skip.fill_(w_skip)
    return model


def _adjacent_only(
    model,
    a,
):
    hidden = F.relu(model.layer0(a))
    return model.layer1(hidden)


def _skip_addition(
    spec,
    a,
    hidden,
    w_skip,
    like,
):
    saved = {
        0: a,
        1: hidden,
    }
    addition = torch.zeros_like(like)
    for skip in spec.skips:
        source = saved[skip.source_layer][
            :,
            skip.source_index,
        ]
        addition[:, skip.target_index] = w_skip * source
    return addition


def test_skip_residual_forward_shape_and_skip_effect():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    spec = parse_layered(_skip_edgelist())
    model = SkipResidualNet(spec)
    x = torch.randn(
        4,
        1,
    )

    with torch.no_grad():
        model.w_skip.fill_(0.0)
    y_without_skip = model(x)

    with torch.no_grad():
        model.w_skip.fill_(1.0)
    y_with_skip = model(x)

    assert y_without_skip.shape == (4, 1)
    assert y_with_skip.shape == (4, 1)
    assert not torch.equal(
        y_without_skip,
        y_with_skip,
    )


def test_skip_residual_matches_w_times_saved_source():
    spec = parse_layered(_skip_edgelist())
    w_skip = 0.3
    model = _pinned_adjacent_net(
        spec,
        w_skip,
    )
    a = _known_input()
    hidden = F.relu(model.layer0(a))
    adjacent = model.layer1(hidden)
    expected = adjacent + _skip_addition(
        spec,
        a,
        hidden,
        w_skip,
        adjacent,
    )
    y = model(a)
    torch.testing.assert_close(
        y,
        expected,
    )
    skip = spec.skips[0]
    n_out = y.shape[1]
    for out_index in range(n_out):
        if out_index == skip.target_index:
            continue
        torch.testing.assert_close(
            y[:, out_index],
            adjacent[:, out_index],
        )


def test_skip_weight_zero_matches_adjacent_only_forward():
    spec = parse_layered(_skip_edgelist())
    model = _pinned_adjacent_net(
        spec,
        0.0,
    )
    a = _known_input()
    torch.testing.assert_close(
        model(a),
        _adjacent_only(
            model,
            a,
        ),
    )


def test_skip_ac_is_absent_from_masks():
    spec = parse_layered(_skip_edgelist())
    n_ones = sum(int(mask.sum().item()) for mask in spec.masks)
    assert n_ones == 2
    skip = spec.skips[0]
    assert skip.source == "A"
    assert skip.target == "C"


def test_skip_path_changes_output_when_adjacent_weights_are_zero():
    spec = parse_layered(_skip_edgelist())
    model = _pinned_adjacent_net(
        spec,
        0.3,
    )
    with torch.no_grad():
        model.layer0.parametrizations.weight.original.zero_()
        model.layer1.parametrizations.weight.original.zero_()
    skip = spec.skips[0]
    a_low = torch.zeros(
        1,
        1,
        dtype=torch.float32,
    )
    a_high = torch.zeros(
        1,
        1,
        dtype=torch.float32,
    )
    a_high[0, skip.source_index] = 4.0
    y_low = model(a_low)
    y_high = model(a_high)
    assert (
        y_low[0, skip.target_index].item()
        != y_high[0, skip.target_index].item()
    )
