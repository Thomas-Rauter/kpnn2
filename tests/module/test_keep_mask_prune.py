"""
In-training prune is a keep-mask inside constraint=.

The spec stays the one that was parsed. A zeroed slot
contributes nothing, Adam state and index_digest stay, the
buffer reloads into a layer rebuilt from the same spec, and
a tied transpose sees the zero. optimizer.load_state_dict
across a reparse is accepted when nnz matches.
"""

import warnings

import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn.utils import prune

from kpnn2 import (
    Kpnn2Error,
    LayeredSpec,
    MaskedLinear,
    PackedLinear,
    align_inputs,
    parse_layered,
)


class _KeepMask(nn.Module):
    """
    Persistent per-entry keep buffer, multiplied into weight.
    """

    def __init__(
        self,
        shape,
    ):
        super().__init__()
        self.register_buffer(
            "keep",
            torch.ones(shape),
        )

    def forward(
        self,
        weight,
    ):
        return weight * self.keep


def _two_parent_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "B", "H"],
            "target": ["H", "H", "C"],
        }
    )


def _star_edgelist(sources):
    return pd.DataFrame(
        {
            "source": list(sources),
            "target": ["T"] * len(sources),
        }
    )


def _packed(
    spec,
    hop,
    constraint,
):
    return PackedLinear(
        hop.source_index,
        hop.target_index,
        hop.out_features,
        hop.in_features,
        bias=False,
        identity=spec.fingerprint,
        constraint=constraint,
    )


def _first_hop_layer(spec):
    return _packed(
        spec,
        spec.hops[0],
        _KeepMask(len(spec.hops[0].source_index)),
    )


def test_packed_keep_mask_zero_contributes_nothing():
    torch.manual_seed(42)
    spec = parse_layered(_two_parent_edgelist())
    hop_index, packed = spec.edge_location(
        "A",
        "H",
    )
    assert hop_index == 0
    assert packed == (0,)
    layer = _first_hop_layer(spec)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [10.0, 3.0],
            )
        )
    layer.constraint.keep[packed[0]] = 0.0
    x = torch.tensor(
        [[7.0, 1.0]],
    )
    torch.testing.assert_close(
        layer.effective_weight(),
        torch.tensor(
            [0.0, 3.0],
        ),
    )
    torch.testing.assert_close(
        layer(x),
        torch.tensor(
            [[3.0]],
        ),
    )
    layer(x).sum().backward()
    assert layer.weight.grad[0].item() == 0.0
    assert layer.weight.grad[1].item() != 0.0
    with torch.no_grad():
        layer.weight[0] = 100.0
    torch.testing.assert_close(
        layer(x),
        torch.tensor(
            [[3.0]],
        ),
    )
    assert layer.effective_weight()[0].item() == 0.0


def test_packed_keep_mask_leaves_optimizer_state_in_place():
    torch.manual_seed(42)
    spec = parse_layered(_two_parent_edgelist())
    layer = _first_hop_layer(spec)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [0.2, -0.4],
            )
        )
    optimizer = torch.optim.Adam(
        layer.parameters(),
        lr=0.1,
    )
    x = torch.tensor(
        [[1.0, 2.0]],
    )
    for _ in range(3):
        optimizer.zero_grad()
        layer(x).sum().backward()
        optimizer.step()
    weight = layer.weight
    state = optimizer.state[weight]
    exp_avg = state["exp_avg"]
    assert not torch.equal(
        exp_avg,
        torch.zeros_like(exp_avg),
    )
    exp_avg_before = exp_avg.clone()
    exp_avg_sq_before = state["exp_avg_sq"].clone()
    step_before = int(state["step"].item())
    stored_before = weight.detach().clone()
    layer.constraint.keep[0] = 0.0
    assert optimizer.state[weight] is state
    assert state["exp_avg"] is exp_avg
    torch.testing.assert_close(
        exp_avg,
        exp_avg_before,
    )
    torch.testing.assert_close(
        state["exp_avg_sq"],
        exp_avg_sq_before,
    )
    assert int(state["step"].item()) == step_before
    optimizer.zero_grad()
    layer(x).sum().backward()
    assert layer.weight.grad[0].item() == 0.0
    optimizer.step()
    assert int(state["step"].item()) == step_before + 1
    assert weight[0].item() != stored_before[0].item()
    assert layer.effective_weight()[0].item() == 0.0


def test_packed_keep_mask_leaves_index_digest_and_roles():
    torch.manual_seed(42)
    spec = parse_layered(_two_parent_edgelist())
    layer = _first_hop_layer(spec)
    digest = layer.state_dict()["index_digest"].clone()
    source = layer.source_index.clone()
    target = layer.target_index.clone()
    fingerprint = spec.fingerprint
    roles = (
        spec.input_nodes,
        spec.hidden_nodes,
        spec.output_nodes,
        spec.layer_nodes,
    )
    columns = align_inputs(
        ["B", "A"],
        spec,
    ).copy()
    location = spec.edge_location(
        "A",
        "H",
    )
    layer.constraint.keep[0] = 0.0
    assert torch.equal(
        layer.state_dict()["index_digest"],
        digest,
    )
    assert torch.equal(
        layer.source_index,
        source,
    )
    assert torch.equal(
        layer.target_index,
        target,
    )
    assert spec.fingerprint == fingerprint
    assert (
        spec.input_nodes,
        spec.hidden_nodes,
        spec.output_nodes,
        spec.layer_nodes,
    ) == roles
    assert (
        align_inputs(
            ["B", "A"],
            spec,
        ).tolist()
        == columns.tolist()
    )
    assert (
        spec.edge_location(
            "A",
            "H",
        )
        == location
    )


def test_packed_keep_mask_state_dict_loads_into_the_same_spec():
    torch.manual_seed(42)
    spec = parse_layered(_two_parent_edgelist())
    layer = _first_hop_layer(spec)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [10.0, 3.0],
            )
        )
    layer.constraint.keep[0] = 0.0
    blob = layer.state_dict()
    assert blob["constraint.keep"][0].item() == 0.0
    restored_spec = LayeredSpec.from_dict(spec.to_dict())
    assert restored_spec.fingerprint == spec.fingerprint
    hop = restored_spec.hops[0]
    rebuilt = _packed(
        restored_spec,
        hop,
        _KeepMask(len(hop.source_index)),
    )
    result = rebuilt.load_state_dict(blob)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    x = torch.tensor(
        [[7.0, 1.0]],
    )
    torch.testing.assert_close(
        rebuilt.constraint.keep,
        layer.constraint.keep,
    )
    torch.testing.assert_close(
        rebuilt(x),
        layer(x),
    )
    assert torch.equal(
        rebuilt.state_dict()["index_digest"],
        layer.state_dict()["index_digest"],
    )


def test_packed_keep_mask_reaches_a_tied_transpose():
    torch.manual_seed(42)
    spec = parse_layered(_two_parent_edgelist())
    layer = _first_hop_layer(spec)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [10.0, 3.0],
            )
        )
    decoder = layer.transpose(bias=False)
    assert decoder.constraint is layer.constraint
    layer.constraint.keep[0] = 0.0
    encoder_dense = torch.zeros(
        layer.out_features,
        layer.in_features,
    )
    encoder_dense[
        layer.target_index,
        layer.source_index,
    ] = layer.effective_weight()
    decoder_dense = torch.zeros(
        decoder.out_features,
        decoder.in_features,
    )
    decoder_dense[
        decoder.target_index,
        decoder.source_index,
    ] = decoder.effective_weight()
    torch.testing.assert_close(
        decoder_dense,
        encoder_dense.T,
    )
    assert encoder_dense[0, 0].item() == 0.0
    hidden = torch.tensor(
        [[4.0]],
    )
    torch.testing.assert_close(
        decoder(hidden),
        torch.tensor(
            [[0.0, 12.0]],
        ),
    )


def test_masked_keep_mask_zero_contributes_nothing():
    torch.manual_seed(42)
    spec = parse_layered(_two_parent_edgelist())
    hop = spec.hops[0]
    mask = hop.to_mask()
    layer = MaskedLinear(
        mask,
        bias=False,
        identity=spec.fingerprint,
        constraint=_KeepMask(mask.shape),
    )
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [[4.0, 5.0]],
            )
        )
    layer.constraint.keep[0, 0] = 0.0
    x = torch.tensor(
        [[2.0, 3.0]],
    )
    torch.testing.assert_close(
        layer(x),
        torch.tensor(
            [[15.0]],
        ),
    )
    layer(x).sum().backward()
    assert layer.weight.grad[0, 0].item() == 0.0
    assert layer.weight.grad[0, 1].item() != 0.0


def test_masked_keep_mask_round_trips_on_the_same_spec():
    torch.manual_seed(42)
    spec = parse_layered(_two_parent_edgelist())
    hop = spec.hops[0]
    mask = hop.to_mask()
    layer = MaskedLinear(
        mask,
        bias=False,
        identity=spec.fingerprint,
        constraint=_KeepMask(mask.shape),
    )
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [[0.2, -0.4]],
            )
        )
    optimizer = torch.optim.Adam(
        layer.parameters(),
        lr=0.1,
    )
    x = torch.tensor(
        [[1.0, 2.0]],
    )
    for _ in range(3):
        optimizer.zero_grad()
        layer(x).sum().backward()
        optimizer.step()
    digest = layer.state_dict()["mask_digest"].clone()
    state = optimizer.state[layer.weight]
    exp_avg = state["exp_avg"]
    exp_avg_before = exp_avg.clone()
    layer.constraint.keep[0, 0] = 0.0
    assert torch.equal(
        layer.state_dict()["mask_digest"],
        digest,
    )
    assert state["exp_avg"] is exp_avg
    torch.testing.assert_close(
        exp_avg,
        exp_avg_before,
    )
    restored_spec = LayeredSpec.from_dict(spec.to_dict())
    restored_mask = restored_spec.hops[0].to_mask()
    rebuilt = MaskedLinear(
        restored_mask,
        bias=False,
        identity=restored_spec.fingerprint,
        constraint=_KeepMask(restored_mask.shape),
    )
    result = rebuilt.load_state_dict(layer.state_dict())
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    torch.testing.assert_close(
        rebuilt(x),
        layer(x),
    )
    assert rebuilt.effective_weight()[0, 0].item() == 0.0
    assert torch.equal(
        rebuilt.state_dict()["mask_digest"],
        digest,
    )


def test_custom_from_mask_renames_packed_state_dict_keys():
    torch.manual_seed(42)
    spec = parse_layered(_two_parent_edgelist())
    hop = spec.hops[0]
    layer = PackedLinear(
        hop.source_index,
        hop.target_index,
        hop.out_features,
        hop.in_features,
        bias=False,
    )
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [1.5, 2.5],
            )
        )
    prune.custom_from_mask(
        layer,
        name="weight",
        mask=torch.tensor(
            [0.0, 1.0],
        ),
    )
    keys = set(layer.state_dict())
    assert "weight" not in keys
    assert "weight_orig" in keys
    assert "weight_mask" in keys
    torch.testing.assert_close(
        layer.effective_weight(),
        torch.tensor(
            [0.0, 2.5],
        ),
    )
    torch.testing.assert_close(
        layer(
            torch.tensor(
                [[1.0, 1.0]],
            )
        ),
        torch.tensor(
            [[2.5]],
        ),
    )


def test_reparse_optimizer_load_is_silent_when_nnz_matches():
    """
    Dropping g3->T and adding g0->T keeps nnz and the index
    pattern. The layer load raises on identity. The optimizer
    load is accepted and parks g1's moment on g0.
    """
    torch.manual_seed(42)
    old_spec = parse_layered(
        _star_edgelist(
            ["g1", "g2", "g3"],
        )
    )
    new_spec = parse_layered(
        _star_edgelist(
            ["g0", "g1", "g2"],
        )
    )
    old_hop = old_spec.hops[0]
    new_hop = new_spec.hops[0]
    old_layer = PackedLinear(
        old_hop.source_index,
        old_hop.target_index,
        old_hop.out_features,
        old_hop.in_features,
        identity=old_spec.fingerprint,
    )
    new_layer = PackedLinear(
        new_hop.source_index,
        new_hop.target_index,
        new_hop.out_features,
        new_hop.in_features,
        identity=new_spec.fingerprint,
    )
    assert old_layer.nnz == new_layer.nnz
    assert old_layer.bias.shape == new_layer.bias.shape
    assert torch.equal(
        old_layer.state_dict()["index_digest"],
        new_layer.state_dict()["index_digest"],
    )
    assert old_spec.edge_location(
        "g1",
        "T",
    ) == (
        0,
        (0,),
    )
    assert new_spec.edge_location(
        "g1",
        "T",
    ) == (
        0,
        (1,),
    )
    assert new_spec.edge_location(
        "g0",
        "T",
    ) == (
        0,
        (0,),
    )
    with torch.no_grad():
        old_layer.weight.copy_(
            torch.tensor(
                [0.2, -0.4, 0.7],
            )
        )
    old_opt = torch.optim.Adam(
        old_layer.parameters(),
        lr=0.1,
    )
    x = torch.tensor(
        [[1.0, 2.0, 3.0]],
    )
    for _ in range(3):
        old_opt.zero_grad()
        old_layer(x).sum().backward()
        old_opt.step()
    moments = old_opt.state[old_layer.weight]["exp_avg"]
    assert moments[0].item() != moments[1].item()
    assert moments[1].item() != moments[2].item()
    new_opt = torch.optim.Adam(
        new_layer.parameters(),
        lr=0.1,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new_opt.load_state_dict(old_opt.state_dict())
    assert caught == []
    torch.testing.assert_close(
        new_opt.state[new_layer.weight]["exp_avg"],
        moments,
    )
    with pytest.raises(
        Kpnn2Error,
        match="identity does not match",
    ):
        new_layer.load_state_dict(old_layer.state_dict())
