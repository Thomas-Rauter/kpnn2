import copy

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kpnn2 import (
    Kpnn2Error,
    PackedLinear,
    gather_hop_inputs,
    parse_adjacency,
    parse_layered,
    scatter_hop_outputs,
)


def _dense_from_packed(packed):
    weight = packed.weight
    if packed.constraint is not None:
        weight = packed.constraint(weight)
    dense = torch.zeros(
        packed.out_features,
        packed.in_features,
        dtype=weight.dtype,
        device=weight.device,
    )
    dense[
        packed.target_index,
        packed.source_index,
    ] = weight
    return dense


def test_transpose_swaps_indices_and_sizes():
    layer = PackedLinear(
        [0, 1, 2],
        [0, 0, 1],
        2,
        3,
        bias=False,
    )
    mirrored = layer.transpose(bias=False)
    assert mirrored.in_features == 2
    assert mirrored.out_features == 3
    assert mirrored.nnz == layer.nnz
    assert torch.equal(
        mirrored.source_index,
        layer.target_index,
    )
    assert torch.equal(
        mirrored.target_index,
        layer.source_index,
    )


def test_transpose_enforces_its_own_in_features():
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 1, 2],
        [0, 0, 1],
        2,
        3,
    )
    mirrored = layer.transpose()

    assert mirrored(torch.randn(4, 2)).shape == (4, 3)
    with pytest.raises(
        Kpnn2Error,
        match="in_features=2",
    ):
        mirrored(torch.randn(4, 3))


def test_transpose_ties_weight_by_default():
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
    )
    mirrored = layer.transpose(bias=False)
    assert mirrored.weight is layer.weight


def test_transpose_tie_false_copies_values():
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
    )
    mirrored = layer.transpose(
        bias=False,
        tie=False,
    )
    assert mirrored.weight is not layer.weight
    assert torch.equal(
        mirrored.weight,
        layer.weight,
    )
    with torch.no_grad():
        layer.weight[0] += 1.0
    assert not torch.equal(
        mirrored.weight,
        layer.weight,
    )


def test_transpose_does_not_tie_bias():
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
    )
    mirrored = layer.transpose()
    assert layer.bias is not None
    assert mirrored.bias is not None
    assert mirrored.bias is not layer.bias
    assert tuple(layer.bias.shape) == (1,)
    assert tuple(mirrored.bias.shape) == (2,)


def test_transpose_bias_false():
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
    )
    mirrored = layer.transpose(bias=False)
    assert mirrored.bias is None
    assert layer.bias is not None


def test_transpose_matches_dense_weight_transpose():
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 1, 2],
        [0, 0, 1],
        2,
        3,
        bias=False,
    )
    mirrored = layer.transpose(bias=False)
    dense = _dense_from_packed(layer)
    x = torch.randn(
        4,
        3,
    )
    hidden = layer(x)
    torch.testing.assert_close(
        hidden,
        F.linear(
            x,
            dense,
        ),
    )
    torch.testing.assert_close(
        mirrored(hidden),
        F.linear(
            hidden,
            dense.T,
        ),
    )


def test_transpose_with_bias_matches_dense_plus_new_bias():
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
    )
    mirrored = layer.transpose(bias=True)
    dense = _dense_from_packed(layer)
    hidden = torch.randn(
        3,
        1,
    )
    torch.testing.assert_close(
        mirrored(hidden),
        F.linear(
            hidden,
            dense.T,
            mirrored.bias,
        ),
    )


def test_tied_gradients_accumulate_on_encoder_weight():
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
    )
    mirrored = layer.transpose(bias=False)
    x = torch.randn(
        2,
        2,
        requires_grad=True,
    )
    loss = mirrored(layer(x)).sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert mirrored.weight.grad is layer.weight.grad


def test_transpose_keeps_encoder_weight_values():
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
    )
    before = layer.weight.detach().clone()
    layer.transpose(bias=False)
    assert torch.equal(
        layer.weight,
        before,
    )


def test_transpose_rejects_non_bool_tie():
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
    )
    with pytest.raises(
        Kpnn2Error,
        match="tie",
    ):
        layer.transpose(tie=1)


def test_transpose_does_not_copy_identity_by_default():
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
        identity="graph-a",
    )
    mirrored = layer.transpose(bias=False)
    assert layer.identity == "graph-a"
    assert mirrored.identity is None
    named = layer.transpose(
        bias=False,
        identity="graph-a",
    )
    assert named.identity == "graph-a"


def test_transpose_state_dict_does_not_load_into_encoder():
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
    )
    mirrored = layer.transpose(bias=False)
    with pytest.raises(
        Kpnn2Error,
        match="indices do not match",
    ):
        mirrored.load_state_dict(layer.state_dict())


class _KeepMask(nn.Module):
    """
    Per-slot keep mask: a pruning constraint with state.
    """

    def __init__(
        self,
        nnz,
        dtype=torch.float32,
    ):
        super().__init__()
        self.register_buffer(
            "keep",
            torch.ones(
                nnz,
                dtype=dtype,
            ),
        )

    def forward(self, weight):
        return weight * self.keep.to(dtype=weight.dtype)


class _LearnedGate(nn.Module):
    """
    Per-slot learnable gate, as in self-pruning BINNs.
    """

    def __init__(self, nnz):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(nnz))

    def forward(self, weight):
        return weight * torch.sigmoid(self.gate)


class _TiedPair(nn.Module):
    def __init__(self, constraint):
        super().__init__()
        self.enc = PackedLinear(
            [0, 1, 2],
            [0, 0, 1],
            2,
            3,
            bias=False,
            constraint=constraint,
        )
        self.dec = self.enc.transpose(bias=False)

    def forward(self, x):
        return self.dec(torch.tanh(self.enc(x)))


def test_transpose_tie_shares_constraint_and_keeps_encoder_child():
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
        constraint=nn.Softplus(),
    )
    mirrored = layer.transpose(bias=False)
    assert "constraint" in dict(layer.named_children())
    assert "constraint" in dict(mirrored.named_children())
    assert mirrored.constraint is layer.constraint
    dense = _dense_from_packed(layer)
    x = torch.randn(
        2,
        2,
    )
    hidden = layer(x)
    torch.testing.assert_close(
        mirrored(hidden),
        F.linear(
            hidden,
            dense.T,
        ),
    )


def test_transpose_tie_keeps_a_pruned_edge_pruned_in_the_decoder():
    """
    Pruning after the transpose must reach the decoder.

    The decoder's live map must stay the transpose of the
    encoder's live map, so an edge the encoder no longer uses
    contributes nothing in the decoder either.
    """
    torch.manual_seed(42)
    enc = PackedLinear(
        [0, 1, 2],
        [0, 0, 1],
        2,
        3,
        bias=False,
        constraint=_KeepMask(3),
    )
    dec = enc.transpose(bias=False)

    enc.constraint.keep[1] = 0.0

    enc_dense = _dense_from_packed(enc)
    assert enc_dense[0, 1].item() == 0.0
    torch.testing.assert_close(
        _dense_from_packed(dec),
        enc_dense.T,
    )
    hidden = torch.randn(
        4,
        2,
    )
    torch.testing.assert_close(
        dec(hidden),
        F.linear(
            hidden,
            enc_dense.T,
        ),
    )


def test_transpose_tie_trains_one_constraint_parameter():
    torch.manual_seed(42)
    pair = _TiedPair(_LearnedGate(3))
    names = [name for name, _ in pair.named_parameters()]
    assert names == [
        "enc.weight",
        "enc.constraint.gate",
    ]
    optimizer = torch.optim.Adam(
        pair.parameters(),
        lr=0.1,
    )
    x = torch.randn(
        8,
        3,
    )
    for _ in range(5):
        optimizer.zero_grad()
        loss = F.mse_loss(
            pair(x),
            x,
        )
        loss.backward()
        optimizer.step()

    assert not torch.equal(
        pair.enc.constraint.gate.detach(),
        torch.ones(3),
    )
    torch.testing.assert_close(
        _dense_from_packed(pair.dec),
        _dense_from_packed(pair.enc).T,
    )


def test_transpose_tie_false_deepcopies_constraint():
    torch.manual_seed(42)
    enc = PackedLinear(
        [0, 1, 2],
        [0, 0, 1],
        2,
        3,
        bias=False,
        constraint=_KeepMask(3),
    )
    dec = enc.transpose(
        bias=False,
        tie=False,
    )
    assert dec.constraint is not enc.constraint
    torch.testing.assert_close(
        _dense_from_packed(dec),
        _dense_from_packed(enc).T,
    )

    enc.constraint.keep[1] = 0.0

    assert dec.constraint.keep[1].item() == 1.0
    assert _dense_from_packed(dec)[1, 0].item() != 0.0


def test_transpose_tie_does_not_modify_the_encoder_constraint():
    torch.manual_seed(42)
    enc = PackedLinear(
        [0, 1, 2],
        [0, 0, 1],
        2,
        3,
        bias=False,
        constraint=_KeepMask(
            3,
            dtype=torch.float64,
        ),
    )
    keep = enc.constraint.keep

    dec = enc.transpose(bias=False)

    assert enc.constraint.keep is keep
    assert keep.dtype == torch.float64
    assert dec.constraint.keep is keep


def test_tied_pair_state_dict_roundtrip_keeps_constraint_shared():
    torch.manual_seed(42)
    trained = _TiedPair(_KeepMask(3))
    trained.enc.constraint.keep[1] = 0.0

    restored = _TiedPair(_KeepMask(3))
    restored.load_state_dict(trained.state_dict())

    assert restored.dec.constraint is restored.enc.constraint
    assert restored.enc.constraint.keep.tolist() == [1.0, 0.0, 1.0]
    torch.testing.assert_close(
        _dense_from_packed(restored.dec),
        _dense_from_packed(trained.enc).T,
    )


def test_parent_deepcopy_keeps_tied_weights():
    torch.manual_seed(42)
    pair = _TiedPair(_KeepMask(3))
    cloned = copy.deepcopy(pair)
    assert cloned.enc.weight is cloned.dec.weight
    assert cloned.enc.weight is not pair.enc.weight
    assert cloned.enc.constraint is cloned.dec.constraint
    assert cloned.enc.constraint is not pair.enc.constraint


def test_transpose_follows_weight_dtype():
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
    )
    layer = layer.to(dtype=torch.float64)
    mirrored = layer.transpose(bias=False)
    assert mirrored.weight.dtype == torch.float64
    assert mirrored.source_index.dtype == torch.int64
    x = torch.ones(
        1,
        1,
        dtype=torch.float64,
    )
    y = mirrored(x)
    assert y.dtype == torch.float64
    assert y.shape == (1, 2)


def test_transpose_on_a_skip_hop_matches_dense_and_scatters():
    torch.manual_seed(42)
    spec = parse_layered(
        pd.DataFrame(
            {
                "source": ["A", "H", "A"],
                "target": ["H", "C", "C"],
            }
        )
    )
    hop = spec.hops[1]
    enc = PackedLinear(
        hop.source_index,
        hop.target_index,
        hop.out_features,
        hop.in_features,
        bias=False,
    )
    dec = enc.transpose(bias=False)
    dense = _dense_from_packed(enc)
    saved = {
        0: torch.randn(
            3,
            spec.layer_dims[0],
        ),
        1: torch.randn(
            3,
            spec.layer_dims[1],
        ),
    }
    sources = gather_hop_inputs(
        saved,
        hop,
    )
    hidden = enc(sources)
    recon = dec(hidden)
    torch.testing.assert_close(
        recon,
        F.linear(
            hidden,
            dense.T,
        ),
    )
    parts = scatter_hop_outputs(
        recon,
        hop,
    )
    assert list(parts) == list(hop.source_layers)
    torch.testing.assert_close(
        torch.cat(
            list(parts.values()),
            dim=-1,
        ),
        recon,
    )


def test_transpose_on_wide_units_matches_dense():
    torch.manual_seed(42)
    spec = parse_layered(
        pd.DataFrame(
            {
                "source": ["A", "H"],
                "target": ["H", "C"],
            }
        ),
        widths={"A": 2, "H": 3},
    )
    hop = spec.hops[0]
    enc = PackedLinear(
        hop.source_index,
        hop.target_index,
        hop.out_features,
        hop.in_features,
        bias=False,
    )
    dec = enc.transpose(bias=False)
    dense = _dense_from_packed(enc)
    x = torch.randn(
        2,
        hop.in_features,
    )
    hidden = enc(x)
    torch.testing.assert_close(
        dec(hidden),
        F.linear(
            hidden,
            dense.T,
        ),
    )


def test_transpose_on_adjacency_spec_matches_dense():
    torch.manual_seed(42)
    spec = parse_adjacency(
        pd.DataFrame(
            {
                "source": ["x", "a", "b", "a"],
                "target": ["a", "b", "a", "y"],
            }
        )
    )
    n = len(spec.nodes)
    enc = PackedLinear(
        spec.source_index,
        spec.target_index,
        n,
        n,
        bias=False,
    )
    dec = enc.transpose(bias=False)
    dense = _dense_from_packed(enc)
    state = torch.randn(
        4,
        n,
    )
    hidden = enc(state)
    torch.testing.assert_close(
        dec(hidden),
        F.linear(
            hidden,
            dense.T,
        ),
    )


def test_transpose_generator_is_keyword_only():
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
    )
    with pytest.raises(TypeError):
        layer.transpose(
            True,
            True,
            None,
            torch.Generator(),
        )


def test_transpose_generator_rejects_non_generator():
    layer = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
    )
    with pytest.raises(
        Kpnn2Error,
        match="generator",
    ):
        layer.transpose(generator=42)


def test_transpose_generator_none_matches_omitted():
    torch.manual_seed(42)
    enc = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
    )
    torch.manual_seed(42)
    omitted = enc.transpose()
    torch.manual_seed(42)
    explicit = enc.transpose(generator=None)
    torch.testing.assert_close(
        omitted.bias,
        explicit.bias,
    )


def test_transpose_generator_isolates_bias_init():
    enc = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
    )
    with torch.no_grad():
        enc.weight.fill_(1.0)
    torch.manual_seed(0)
    torch.randn(8)
    g = torch.Generator().manual_seed(42)
    polluted = enc.transpose(generator=g)
    g2 = torch.Generator().manual_seed(42)
    clean = enc.transpose(generator=g2)
    torch.testing.assert_close(
        polluted.bias,
        clean.bias,
    )


def test_transpose_omitted_generator_follows_global_stream():
    enc = PackedLinear(
        [0, 1],
        [0, 0],
        1,
        2,
        bias=False,
    )
    torch.manual_seed(42)
    first = enc.transpose()
    torch.manual_seed(42)
    torch.randn(8)
    second = enc.transpose()
    assert not torch.equal(
        first.bias,
        second.bias,
    )
