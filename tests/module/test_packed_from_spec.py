"""
``from_hop`` and ``from_adjacency`` build the constructor's layer.

They read the packed indices and widths off a spec, so the caller
cannot swap source and target. The tests pin that the result equals
the explicit constructor call (sizes, buffers, init draws), that every
option reaches the constructor, and that edges run from source to
target. Direction is checked against the dense ``to_mask()`` map and
against named nodes, which do not go through the packed layer.
"""

import inspect
import re

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kpnn2 import (
    Kpnn2Error,
    PackedLinear,
    PackedMultiheadAttention,
    parse_adjacency,
    parse_layered,
)


def _layered_spec():
    """
    ``hops[0]`` is square (2 by 2); ``hops[1]`` reads two layers.

    A swap of source and target on a square hop passes every range
    check, so it is the case these methods guard.
    """
    return parse_layered(
        pd.DataFrame(
            {
                "source": [
                    "gene_a",
                    "gene_b",
                    "gene_a",
                    "tf_1",
                    "tf_2",
                    "gene_b",
                ],
                "target": ["tf_1", "tf_2", "tf_2", "out", "out", "out"],
            }
        ),
        widths={"out": 2},
    )


def _adjacency_spec(**kwargs):
    """``x`` feeds a two-node cycle ``a <-> b``, read by ``y``."""
    return parse_adjacency(
        pd.DataFrame(
            {
                "source": ["x", "a", "b", "a"],
                "target": ["a", "b", "a", "y"],
            }
        ),
        **kwargs,
    )


def _generator():
    return torch.Generator().manual_seed(42)


def _assert_same_state(
    built,
    expected,
):
    built_state = built.state_dict()
    expected_state = expected.state_dict()
    assert list(built_state) == list(expected_state)
    for key, value in expected_state.items():
        if value.is_floating_point():
            torch.testing.assert_close(
                built_state[key],
                value,
            )
        else:
            assert torch.equal(
                built_state[key],
                value,
            ), key


def test_square_hop_exists_in_the_layered_fixture():
    spec = _layered_spec()
    hop = spec.hops[0]
    assert hop.out_features == hop.in_features


def test_from_hop_matches_the_constructor_on_every_hop():
    spec = _layered_spec()
    for hop in spec.hops:
        built = PackedLinear.from_hop(
            hop,
            identity=spec.fingerprint,
            generator=_generator(),
        )
        expected = PackedLinear(
            hop.source_index,
            hop.target_index,
            hop.out_features,
            hop.in_features,
            identity=spec.fingerprint,
            generator=_generator(),
        )
        assert (built.out_features, built.in_features, built.nnz) == (
            expected.out_features,
            expected.in_features,
            expected.nnz,
        )
        _assert_same_state(
            built,
            expected,
        )


@pytest.mark.parametrize(
    "widths",
    [
        None,
        {"a": 2, "y": 3},
    ],
)
def test_from_adjacency_matches_the_constructor(widths):
    spec = _adjacency_spec(widths=widths)
    built = PackedLinear.from_adjacency(
        spec,
        identity=spec.fingerprint,
        generator=_generator(),
    )
    expected = PackedLinear(
        spec.source_index,
        spec.target_index,
        spec.state_dim,
        spec.state_dim,
        identity=spec.fingerprint,
        generator=_generator(),
    )
    assert built.in_features == built.out_features == spec.state_dim
    assert built.nnz == expected.nnz == len(spec.source_index)
    _assert_same_state(
        built,
        expected,
    )


def test_from_hop_runs_every_hop_from_source_to_target():
    torch.manual_seed(42)
    spec = _layered_spec()
    for hop in spec.hops:
        layer = PackedLinear.from_hop(
            hop,
            generator=_generator(),
        )
        dense = torch.zeros(
            hop.out_features,
            hop.in_features,
        )
        with torch.no_grad():
            dense[
                list(hop.target_index),
                list(hop.source_index),
            ] = layer.weight
        assert torch.equal(
            dense != 0,
            hop.to_mask().bool(),
        )
        x = torch.randn(
            5,
            hop.in_features,
        )
        torch.testing.assert_close(
            layer(x),
            F.linear(
                x,
                dense,
                layer.bias,
            ),
        )


def test_from_adjacency_sends_a_node_only_to_its_targets():
    spec = _adjacency_spec()
    core = PackedLinear.from_adjacency(
        spec,
        bias=False,
        generator=_generator(),
    )
    with torch.no_grad():
        core.weight.fill_(1.0)
    state = torch.zeros(
        1,
        spec.state_dim,
    )
    state[0, spec.node_units("x")] = 1.0
    expected = torch.zeros(
        1,
        spec.state_dim,
    )
    expected[0, spec.node_units("a")] = 1.0
    torch.testing.assert_close(
        core(state),
        expected,
    )


def test_from_hop_forwards_every_option():
    spec = _layered_spec()
    constraint = nn.Softplus()
    layer = PackedLinear.from_hop(
        spec.hops[1],
        False,
        identity=spec.fingerprint,
        constraint=constraint,
        generator=_generator(),
    )
    assert layer.bias is None
    assert layer.identity == spec.fingerprint
    assert layer.constraint is constraint


def test_from_adjacency_forwards_every_option():
    spec = _adjacency_spec()
    constraint = nn.Softplus()
    core = PackedLinear.from_adjacency(
        spec,
        False,
        identity=spec.fingerprint,
        constraint=constraint,
        generator=_generator(),
    )
    assert core.bias is None
    assert core.identity == spec.fingerprint
    assert core.constraint is constraint


def test_from_adjacency_points_a_layered_spec_at_from_hop():
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("call PackedLinear.from_hop(hop) for each hop"),
    ):
        PackedLinear.from_adjacency(_layered_spec())


def test_from_spec_methods_leave_identity_to_the_caller():
    spec = _adjacency_spec()
    assert PackedLinear.from_adjacency(spec).identity is None
    assert PackedLinear.from_hop(_layered_spec().hops[0]).identity is None


def test_from_spec_methods_take_only_leading_arguments_by_position():
    for method, leading in (
        (PackedLinear.from_hop, {"hop", "bias"}),
        (PackedLinear.from_adjacency, {"spec", "bias"}),
        (
            PackedMultiheadAttention.from_adjacency,
            {"spec", "embed_dim", "num_heads"},
        ),
    ):
        parameters = inspect.signature(method).parameters.values()
        positional = {
            parameter.name
            for parameter in parameters
            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        }
        assert positional == leading


def test_attention_from_adjacency_matches_the_constructor():
    spec = _adjacency_spec(widths={"a": 2})
    built = PackedMultiheadAttention.from_adjacency(
        spec,
        8,
        2,
        add_self_loops=True,
        identity=spec.fingerprint,
        generator=_generator(),
    )
    expected = PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        spec.state_dim,
        spec.state_dim,
        8,
        2,
        add_self_loops=True,
        identity=spec.fingerprint,
        generator=_generator(),
    )
    assert built.query_features == built.key_features == spec.state_dim
    assert built.nnz == expected.nnz
    _assert_same_state(
        built,
        expected,
    )


def test_attention_from_adjacency_forwards_every_option():
    spec = _adjacency_spec()
    attn = PackedMultiheadAttention.from_adjacency(
        spec,
        8,
        2,
        dropout=0.25,
        bias=False,
        batch_first=False,
        add_self_loops=True,
        identity=spec.fingerprint,
        generator=_generator(),
        chunk_size=3,
    )
    assert attn.dropout == 0.25
    assert attn.q_proj.bias is None
    assert attn.batch_first is False
    assert attn.add_self_loops is True
    assert attn.nnz == len(spec.source_index) + spec.state_dim
    assert attn.identity == spec.fingerprint
    assert attn.chunk_size == 3


def test_attention_from_adjacency_lets_a_target_attend_to_its_sources():
    torch.manual_seed(42)
    spec = _adjacency_spec()
    attn = PackedMultiheadAttention.from_adjacency(
        spec,
        8,
        2,
        generator=_generator(),
    )
    assert attn.source_index.tolist() == list(spec.source_index)
    assert attn.target_index.tolist() == list(spec.target_index)
    tokens = torch.randn(
        3,
        spec.state_dim,
        8,
    )
    _, packed = attn(
        tokens,
        tokens,
        tokens,
        need_weights=True,
    )
    # y has one parent, a, so query y puts all its weight on that key.
    # Reversed, a would query y and b, and the weight would drop.
    (only_key,) = spec.edge_location(
        "a",
        "y",
    )
    torch.testing.assert_close(
        packed[:, only_key],
        torch.ones(3),
    )
