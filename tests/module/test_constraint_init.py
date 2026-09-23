"""
Degree-aware init under constraint= and init_bound().

reset_parameters draws the degree-aware value for the effective
weight. A constraint that defines right_inverse gets
right_inverse(draw) stored, so the effective weight keeps that
scale; one without it gets the draw itself, unchanged from
earlier releases.
"""

import math

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kpnn2 import Kpnn2Error, MaskedLinear, PackedLinear, parse_layered


def _inverse_softplus_of_magnitude(weight):
    magnitude = weight.abs().clamp_min(1e-6)
    return magnitude + torch.log(-torch.expm1(-magnitude))


class _PositiveEdges(nn.Module):
    """
    Softplus with the inverse that keeps the degree-aware scale.
    """

    def forward(self, weight):
        return F.softplus(weight)

    def right_inverse(self, weight):
        return _inverse_softplus_of_magnitude(weight)


class _SignedEdges(nn.Module):
    """
    Per-entry sign: +1 non-negative, -1 non-positive, 0 free.
    """

    def __init__(self, sign):
        super().__init__()
        self.register_buffer(
            "sign",
            sign,
        )

    def forward(self, weight):
        magnitude = F.softplus(weight)
        return torch.where(
            self.sign > 0,
            magnitude,
            torch.where(
                self.sign < 0,
                -magnitude,
                weight,
            ),
        )

    def right_inverse(self, weight):
        return torch.where(
            self.sign == 0,
            weight,
            _inverse_softplus_of_magnitude(weight),
        )


class _Inverse(nn.Module):
    """
    Identity forward with a caller-supplied right_inverse.
    """

    def __init__(self, inverse):
        super().__init__()
        self._inverse = inverse

    def forward(self, weight):
        return weight

    def right_inverse(self, weight):
        return self._inverse(weight)


def _fan_in_spec(fan_in):
    """
    Node T with ``fan_in`` parents and node U with 2, in one hop.
    """
    sources = [f"g{i}" for i in range(fan_in)] + ["g0", "g1"]
    targets = ["T"] * fan_in + ["U", "U"]
    return parse_layered(
        pd.DataFrame(
            {
                "source": sources,
                "target": targets,
            }
        )
    )


def _packed(
    hop,
    constraint=None,
    generator=None,
    bias=True,
):
    return PackedLinear(
        hop.source_index,
        hop.target_index,
        hop.out_features,
        hop.in_features,
        bias=bias,
        constraint=constraint,
        generator=generator,
    )


def _masked(
    hop,
    constraint=None,
    generator=None,
    bias=True,
):
    return MaskedLinear(
        hop.to_mask(),
        bias=bias,
        constraint=constraint,
        generator=generator,
    )


_BUILDERS = pytest.mark.parametrize(
    "build",
    [_packed, _masked],
    ids=["packed", "masked"],
)


@_BUILDERS
def test_constraint_without_inverse_stores_the_draw(build):
    hop = _fan_in_spec(6).hops[0]
    torch.manual_seed(42)
    plain = build(hop)
    torch.manual_seed(42)
    softplus = build(
        hop,
        constraint=nn.Softplus(),
    )

    assert torch.equal(
        softplus.weight,
        plain.weight,
    )
    assert torch.equal(
        softplus.bias,
        plain.bias,
    )


@_BUILDERS
def test_right_inverse_stores_the_inverse_of_the_same_draw(build):
    hop = _fan_in_spec(6).hops[0]
    torch.manual_seed(42)
    plain = build(hop)
    after_plain = torch.rand(4)
    torch.manual_seed(42)
    positive = build(
        hop,
        constraint=_PositiveEdges(),
    )
    after_positive = torch.rand(4)

    torch.testing.assert_close(
        positive.effective_weight(),
        plain.effective_weight().abs().clamp_min(1e-6)
        * (plain.init_bound() > 0),
    )
    assert torch.equal(
        positive.bias,
        plain.bias,
    )
    assert torch.equal(
        after_positive,
        after_plain,
    )


@_BUILDERS
def test_right_inverse_keeps_the_degree_aware_scale(build):
    """
    Fan-in 2000: pre-activation spread must match the
    unconstrained init, not grow with fan-in.

    Plain softplus starts every edge near ln 2, so the spread of
    the fan-in-2000 node is about ln 2 * sqrt(2000) instead of
    about 1 / sqrt(3).
    """
    spec = _fan_in_spec(2000)
    hop = spec.hops[0]
    _, t_units = spec.node_units("T")
    # A different seed from the weights, so x and w are unrelated.
    x = torch.randn(
        4096,
        hop.in_features,
        generator=torch.Generator().manual_seed(0),
    )

    def spread(constraint):
        layer = build(
            hop,
            constraint=constraint,
            generator=torch.Generator().manual_seed(42),
            bias=False,
        )
        with torch.no_grad():
            return layer(x)[:, t_units].std().item()

    plain = spread(None)
    positive = spread(_PositiveEdges())
    softplus = spread(nn.Softplus())

    assert plain == pytest.approx(
        1.0 / math.sqrt(3.0),
        rel=0.2,
    )
    assert 0.7 < positive / plain < 1.4
    assert softplus / plain > 10.0


@_BUILDERS
def test_signed_constraint_init_respects_signs_and_bounds(build):
    hop = _fan_in_spec(8).hops[0]
    torch.manual_seed(42)
    template = build(hop)
    shape = template.weight.shape
    sign = torch.zeros(shape)
    flat = sign.view(-1)
    flat[0::3] = 1.0
    flat[1::3] = -1.0

    layer = build(
        hop,
        constraint=_SignedEdges(sign),
        generator=torch.Generator().manual_seed(42),
    )

    effective = layer.effective_weight().detach()
    bound = layer.init_bound()
    live = bound > 0
    assert torch.all(effective[(sign > 0) & live] >= 0)
    assert torch.all(effective[(sign < 0) & live] <= 0)
    assert torch.all(effective.abs() <= bound + 1e-6)


def test_masked_right_inverse_is_checked_on_live_entries_only():
    hop = _fan_in_spec(4).hops[0]
    blocked = hop.to_mask() == 0
    assert blocked.any()

    layer = _masked(
        hop,
        constraint=_Inverse(lambda weight: torch.log(weight.abs())),
    )

    assert torch.equal(
        layer.weight[blocked],
        torch.zeros(int(blocked.sum())),
    )
    assert torch.isfinite(layer.weight).all()


@_BUILDERS
def test_non_finite_right_inverse_on_a_live_edge_raises(build):
    hop = _fan_in_spec(6).hops[0]

    with pytest.raises(
        Kpnn2Error,
        match="non-finite",
    ):
        build(
            hop,
            constraint=_Inverse(lambda weight: weight + float("nan")),
        )


@_BUILDERS
def test_right_inverse_of_wrong_shape_raises(build):
    hop = _fan_in_spec(6).hops[0]

    with pytest.raises(
        Kpnn2Error,
        match="same shape",
    ):
        build(
            hop,
            constraint=_Inverse(lambda weight: weight.reshape(-1)[:1]),
        )


@_BUILDERS
def test_reset_parameters_replays_right_inverse_with_a_generator(build):
    hop = _fan_in_spec(6).hops[0]
    layer = build(
        hop,
        constraint=_PositiveEdges(),
        generator=torch.Generator().manual_seed(7),
    )
    first = layer.weight.detach().clone()

    layer.reset_parameters(torch.Generator().manual_seed(7))

    assert torch.equal(
        layer.weight,
        first,
    )


def test_packed_init_bound_is_the_row_fan_in_bound():
    spec = _fan_in_spec(9)
    hop = spec.hops[0]
    torch.manual_seed(42)
    layer = _packed(hop)
    fan_in = torch.bincount(
        torch.tensor(hop.target_index),
        minlength=hop.out_features,
    )

    bound = layer.init_bound()

    expected = torch.tensor(
        [1.0 / math.sqrt(fan_in[row]) for row in hop.target_index],
    )
    torch.testing.assert_close(
        bound,
        expected,
    )
    assert torch.all(layer.weight.abs() <= bound + 1e-6)


def test_masked_init_bound_is_zero_on_blocked_entries_and_empty_rows():
    mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    torch.manual_seed(42)
    layer = MaskedLinear(mask)

    bound = layer.init_bound()

    expected = torch.tensor(
        [
            [1.0 / math.sqrt(3.0)] * 2 + [0.0, 1.0 / math.sqrt(3.0)],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(
        bound,
        expected,
    )
    assert torch.all(layer.weight.abs() <= bound + 1e-6)


@_BUILDERS
def test_init_bound_follows_the_weight_dtype(build):
    hop = _fan_in_spec(4).hops[0]
    layer = build(hop).double()

    assert layer.init_bound().dtype == torch.float64
    assert layer.init_bound().shape == layer.weight.shape


def test_transpose_copy_with_right_inverse_keeps_encoder_values():
    hop = _fan_in_spec(6).hops[0]
    torch.manual_seed(42)
    encoder = _packed(
        hop,
        constraint=_PositiveEdges(),
        bias=False,
    )

    decoder = encoder.transpose(
        bias=False,
        tie=False,
    )

    assert torch.equal(
        decoder.weight,
        encoder.weight,
    )
    torch.testing.assert_close(
        decoder.effective_weight(),
        encoder.effective_weight(),
    )
