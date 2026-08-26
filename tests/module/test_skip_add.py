import copy
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kpnn2 import (
    Kpnn2Error,
    MaskedLinear,
    SkipAdd,
    parse_edgelist,
)
from kpnn2._frozen_mask import FrozenMask


def _chain_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
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
            "source": [
                "A",
                "B",
                "H1",
                "H2",
                "A",
                "H1",
                "A",
            ],
            "target": [
                "H1",
                "H1",
                "H2",
                "C",
                "H2",
                "C",
                "C",
            ],
        }
    )


def _pin_skip_weights(
    module,
    values,
):
    with torch.no_grad():
        for weight, value in zip(
            module.skip_weights,
            values,
            strict=True,
        ):
            weight.fill_(value)


class _ThreeHopNet(nn.Module):
    """
    Test-only: adjacent hops, SkipAdd, ReLU except last hop.
    """

    def __init__(
        self,
        spec,
    ) -> None:
        super().__init__()
        self.lin0 = MaskedLinear(
            spec.masks[0],
            bias=False,
        )
        self.lin1 = MaskedLinear(
            spec.masks[1],
            bias=False,
        )
        self.lin2 = MaskedLinear(
            spec.masks[2],
            bias=False,
        )
        self.skips = SkipAdd(spec)

    def forward(
        self,
        x,
    ):
        saved = {0: x}
        h1 = self.skips(
            self.lin0(x),
            saved,
            target_layer=1,
        )
        h1 = F.relu(h1)
        saved[1] = h1
        h2 = self.skips(
            self.lin1(h1),
            saved,
            target_layer=2,
        )
        h2 = F.relu(h2)
        saved[2] = h2
        c = self.skips(
            self.lin2(h2),
            saved,
            target_layer=3,
        )
        return c, h1, h2


def test_skip_add_empty_skips_is_identity():
    spec = parse_edgelist(_chain_edgelist())
    module = SkipAdd(spec)
    assert len(module.skip_weights) == 0
    hidden = torch.tensor(
        [
            [1.5],
            [-0.25],
        ],
        dtype=torch.float32,
    )
    saved = {
        0: torch.ones(
            2,
            1,
        ),
    }
    before = hidden.clone()
    out = module(
        hidden,
        saved,
        target_layer=1,
    )
    torch.testing.assert_close(
        out,
        before,
    )
    torch.testing.assert_close(
        hidden,
        before,
    )


def test_skip_add_one_skip_injects_weight_times_source():
    spec = parse_edgelist(_one_skip_edgelist())
    module = SkipAdd(spec)
    assert len(module.skip_weights) == 1
    assert module.skip_weights[0].item() == 0.0
    hidden = torch.tensor(
        [
            [1.0],
            [2.0],
        ],
        dtype=torch.float32,
    )
    a = torch.tensor(
        [
            [4.0],
            [-1.0],
        ],
        dtype=torch.float32,
    )
    saved = {0: a}
    skip = spec.skips[0]
    _pin_skip_weights(
        module,
        [0.3],
    )
    out = module(
        hidden,
        saved,
        target_layer=skip.target_layer,
    )
    expected = hidden.clone()
    expected[:, skip.target_index] = (
        hidden[:, skip.target_index] + 0.3 * a[:, skip.source_index]
    )
    torch.testing.assert_close(
        out,
        expected,
    )
    torch.testing.assert_close(
        saved[0],
        a,
    )


def test_skip_add_zero_weight_matches_adjacent_hidden():
    spec = parse_edgelist(_one_skip_edgelist())
    module = SkipAdd(spec)
    hidden = torch.tensor(
        [[2.0], [-1.0], [0.5]],
        dtype=torch.float32,
    )
    saved = {
        0: torch.ones_like(hidden) * 9.0,
    }
    out = module(
        hidden,
        saved,
        target_layer=2,
    )
    torch.testing.assert_close(
        out,
        hidden,
    )


def test_skip_add_several_skips_to_different_layers():
    spec = parse_edgelist(_three_hop_edgelist())
    assert spec.layer_nodes == (
        ("A", "B"),
        ("H1",),
        ("H2",),
        ("C",),
    )
    assert [(skip.source, skip.target) for skip in spec.skips] == [
        ("A", "H2"),
        ("H1", "C"),
        ("A", "C"),
    ]
    assert len(spec.skips) == 3
    module = SkipAdd(spec)
    _pin_skip_weights(
        module,
        [0.3, 0.2, 0.1],
    )
    a = torch.tensor(
        [[2.0, 0.0]],
        dtype=torch.float32,
    )
    h1 = torch.tensor(
        [[2.0]],
        dtype=torch.float32,
    )
    h2 = torch.tensor(
        [[2.0]],
        dtype=torch.float32,
    )
    c = torch.tensor(
        [[2.6]],
        dtype=torch.float32,
    )
    saved = {
        0: a,
        1: h1,
        2: h2,
    }
    at_h1 = module(
        h1,
        saved,
        target_layer=1,
    )
    torch.testing.assert_close(
        at_h1,
        h1,
    )
    at_h2 = module(
        h2,
        saved,
        target_layer=2,
    )
    torch.testing.assert_close(
        at_h2,
        torch.tensor(
            [[2.6]],
            dtype=torch.float32,
        ),
    )
    at_c = module(
        c,
        saved,
        target_layer=3,
    )
    torch.testing.assert_close(
        at_c,
        torch.tensor(
            [[3.2]],
            dtype=torch.float32,
        ),
    )


def test_skip_add_numerical_pin_relu_then_last_hop_linear():
    spec = parse_edgelist(_three_hop_edgelist())
    net = _ThreeHopNet(spec)
    with torch.no_grad():
        net.lin0.raw_weight.fill_(1.0)
        net.lin1.raw_weight.fill_(1.0)
        net.lin2.raw_weight.fill_(1.0)
    _pin_skip_weights(
        net.skips,
        [0.3, 0.2, 0.1],
    )

    x_pos = torch.tensor(
        [[2.0, 0.0]],
        dtype=torch.float32,
    )
    c_pos, h1_pos, h2_pos = net(x_pos)
    torch.testing.assert_close(
        h1_pos,
        torch.tensor(
            [[2.0]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        h2_pos,
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

    x_neg = torch.tensor(
        [[-1.0, 0.0]],
        dtype=torch.float32,
    )
    c_neg, h1_neg, h2_neg = net(x_neg)
    torch.testing.assert_close(
        h1_neg,
        torch.tensor(
            [[0.0]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        h2_neg,
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

    a_neg = x_neg[:, 0:1]
    adjacent_h2 = net.lin1(h1_neg)
    h2_pre = net.skips(
        adjacent_h2,
        {0: x_neg, 1: h1_neg},
        target_layer=2,
    )
    torch.testing.assert_close(
        h2_pre,
        adjacent_h2 + 0.3 * a_neg,
    )
    torch.testing.assert_close(
        h2_neg,
        F.relu(h2_pre),
    )
    adjacent_c = net.lin2(h2_neg)
    c_pre = net.skips(
        adjacent_c,
        {
            0: x_neg,
            1: h1_neg,
            2: h2_neg,
        },
        target_layer=3,
    )
    torch.testing.assert_close(
        c_pre,
        adjacent_c + 0.1 * a_neg,
    )
    torch.testing.assert_close(
        c_neg,
        c_pre,
    )


def test_skip_add_shape_and_device_match_hidden():
    spec = parse_edgelist(_one_skip_edgelist())
    module = SkipAdd(spec)
    _pin_skip_weights(
        module,
        [0.5],
    )
    hidden = torch.ones(
        4,
        1,
        dtype=torch.float32,
        device="cpu",
    )
    saved = {
        0: torch.arange(
            4,
            dtype=torch.float32,
        ).reshape(4, 1),
    }
    out = module(
        hidden,
        saved,
        target_layer=2,
    )
    assert out.shape == hidden.shape
    assert out.dtype == hidden.dtype
    assert out.device == hidden.device

    module = module.double()
    hidden64 = hidden.double()
    saved32 = {
        0: saved[0],
    }
    out64 = module(
        hidden64,
        saved32,
        target_layer=2,
    )
    assert out64.dtype == torch.float64
    assert out64.device == hidden64.device
    expected = hidden64 + 0.5 * saved[0].double()
    torch.testing.assert_close(
        out64,
        expected,
    )


def test_skip_add_deepcopy_independent_params_frozen_spec():
    spec = parse_edgelist(_one_skip_edgelist())
    module = SkipAdd(spec)
    _pin_skip_weights(
        module,
        [0.4],
    )
    before_skips = spec.skips
    before_mask = spec.masks[0].tolist()
    copied = copy.deepcopy(module)

    assert copied is not module
    assert copied.spec is not module.spec
    assert copied.skip_weights[0] is not module.skip_weights[0]
    torch.testing.assert_close(
        copied.skip_weights[0],
        module.skip_weights[0],
    )
    with torch.no_grad():
        copied.skip_weights[0].fill_(1.0)
    assert module.skip_weights[0].item() == pytest.approx(0.4)
    assert spec.skips is before_skips
    assert copied.spec.skips == spec.skips

    hidden = torch.ones(
        2,
        1,
    )
    saved = {
        0: torch.ones(
            2,
            1,
        ),
    }
    torch.testing.assert_close(
        module(
            hidden,
            saved,
            target_layer=2,
        ),
        hidden + 0.4,
    )
    torch.testing.assert_close(
        copied(
            hidden,
            saved,
            target_layer=2,
        ),
        hidden + 1.0,
    )

    with pytest.raises(FrozenInstanceError):
        module.spec.skips = ()
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        module.spec.masks[0].fill_(0.0)
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        copied.spec.masks[0].fill_(0.0)
    assert spec.masks[0].tolist() == before_mask
    assert copied.spec.masks[0].dtype == torch.float32
    assert isinstance(
        copied.spec.masks[0],
        FrozenMask,
    )


def test_skip_add_rejects_bad_spec():
    with pytest.raises(
        Kpnn2Error,
        match="GraphSpec",
    ):
        SkipAdd(None)


def test_skip_add_rejects_invalid_target_layer():
    spec = parse_edgelist(_one_skip_edgelist())
    module = SkipAdd(spec)
    hidden = torch.ones(
        1,
        1,
    )
    saved = {
        0: torch.ones(
            1,
            1,
        ),
    }
    with pytest.raises(
        Kpnn2Error,
        match="target_layer",
    ):
        module(
            hidden,
            saved,
            target_layer=99,
        )
    with pytest.raises(
        Kpnn2Error,
        match="must be an int",
    ):
        module(
            hidden,
            saved,
            target_layer=True,
        )


def test_skip_add_rejects_missing_saved_layer():
    spec = parse_edgelist(_one_skip_edgelist())
    module = SkipAdd(spec)
    hidden = torch.ones(
        1,
        1,
    )
    with pytest.raises(
        Kpnn2Error,
        match="missing layer",
    ):
        module(
            hidden,
            {},
            target_layer=2,
        )


def test_skip_add_rejects_width_mismatch():
    spec = parse_edgelist(_one_skip_edgelist())
    module = SkipAdd(spec)
    hidden = torch.ones(
        1,
        2,
    )
    saved = {
        0: torch.ones(
            1,
            1,
        ),
    }
    with pytest.raises(
        Kpnn2Error,
        match="wrong number of units",
    ):
        module(
            hidden,
            saved,
            target_layer=2,
        )
    hidden_ok = torch.ones(
        1,
        1,
    )
    saved_wide = {
        0: torch.ones(
            1,
            2,
        ),
    }
    with pytest.raises(
        Kpnn2Error,
        match="wrong number of units",
    ):
        module(
            hidden_ok,
            saved_wide,
            target_layer=2,
        )


def test_skip_add_does_not_mutate_saved_or_hidden():
    spec = parse_edgelist(_one_skip_edgelist())
    module = SkipAdd(spec)
    _pin_skip_weights(
        module,
        [0.5],
    )
    hidden = torch.ones(
        2,
        1,
    )
    saved_source = torch.ones(
        2,
        1,
    )
    saved = {0: saved_source}
    hidden_before = hidden.clone()
    source_before = saved_source.clone()
    module(
        hidden,
        saved,
        target_layer=2,
    )
    torch.testing.assert_close(
        hidden,
        hidden_before,
    )
    torch.testing.assert_close(
        saved_source,
        source_before,
    )
    assert saved[0] is saved_source
