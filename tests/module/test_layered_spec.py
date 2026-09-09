import copy
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest
import torch

from kpnn2 import Hop, MaskedLinear, parse_layered


def _chain_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )


def test_layered_spec_rejects_field_assignment():
    spec = parse_layered(_chain_edgelist())

    with pytest.raises(FrozenInstanceError):
        spec.input_nodes = ("X",)


def test_layered_spec_hop_rejects_field_assignment():
    spec = parse_layered(_chain_edgelist())

    with pytest.raises(FrozenInstanceError):
        spec.hops[0].target_layer = 5


def test_layered_spec_sequences_are_tuples():
    spec = parse_layered(_chain_edgelist())

    assert isinstance(spec.input_nodes, tuple)
    assert isinstance(spec.layer_nodes, tuple)
    assert isinstance(spec.layer_nodes[0], tuple)
    assert isinstance(spec.layer_widths, tuple)
    assert isinstance(spec.layer_widths[0], tuple)
    assert isinstance(spec.hops, tuple)
    assert isinstance(spec.skips, tuple)
    for hop in spec.hops:
        assert isinstance(hop, Hop)
        assert isinstance(hop.source_layers, tuple)
        assert isinstance(hop.source_dims, tuple)
        assert isinstance(hop.source_nodes, tuple)
        assert isinstance(hop.source_index, tuple)
        assert isinstance(hop.target_index, tuple)
    with pytest.raises(AttributeError):
        spec.input_nodes.append("X")


def test_layered_spec_hop_count_matches_layers():
    spec = parse_layered(_chain_edgelist())

    assert len(spec.hops) == len(spec.layer_nodes) - 1
    for index, hop in enumerate(spec.hops):
        assert hop.target_layer == index + 1
        assert isinstance(hop.source_index, tuple)
        assert isinstance(hop.target_index, tuple)
        assert hop.out_features == spec.layer_dims[hop.target_layer]
        assert hop.in_features == sum(hop.source_dims)
        assert hop.target_dim == spec.layer_dims[hop.target_layer]
        assert len(hop.source_index) == len(hop.target_index)
        assert len(hop.source_dims) == len(hop.source_layers)
        assert len(hop.column_offsets) == len(hop.source_layers)


def test_layered_spec_hops_have_no_stored_mask():
    spec = parse_layered(_chain_edgelist())

    for hop in spec.hops:
        assert not hasattr(
            hop,
            "mask",
        )
        mask = hop.to_mask()
        assert type(mask) is torch.Tensor
        assert mask.dtype == torch.float32
        assert not mask.requires_grad
        assert mask.is_contiguous()
        assert mask.numpy().flags.writeable


def test_layered_spec_to_mask_does_not_alias_the_hop():
    edgelist = _chain_edgelist()
    spec = parse_layered(edgelist)
    mask = spec.hops[0].to_mask()
    layer = MaskedLinear(mask)
    layer_before = layer.mask.tolist()
    other = parse_layered(edgelist)

    mask.fill_(0.0)

    assert layer.mask.tolist() == layer_before
    assert other.hops[0].to_mask().tolist() == layer_before
    assert spec.hops[0].to_mask().tolist() == layer_before


def test_layered_spec_deepcopy_independent_hops():
    spec = parse_layered(_chain_edgelist())
    before = [(hop.source_index, hop.target_index) for hop in spec.hops]
    copied = copy.deepcopy(spec)

    assert copied is not spec
    assert copied.input_nodes == spec.input_nodes
    assert copied.layer_nodes == spec.layer_nodes
    assert copied.layer_widths == spec.layer_widths
    assert copied.layer_dims == spec.layer_dims
    assert len(copied.hops) == len(spec.hops)
    for original, duplicate in zip(
        spec.hops,
        copied.hops,
        strict=True,
    ):
        assert duplicate.target_layer == original.target_layer
        assert duplicate.source_layers == original.source_layers
        assert duplicate.source_nodes == original.source_nodes
        assert duplicate.source_index == original.source_index
        assert duplicate.target_index == original.target_index
        assert duplicate.target_dim == original.target_dim
        first = original.to_mask()
        second = duplicate.to_mask()
        assert first.tolist() == second.tolist()
        assert first is not second
        second.fill_(0.0)

    assert [(hop.source_index, hop.target_index) for hop in spec.hops] == before
    assert spec.hops[0].to_mask().tolist() != [[0.0]]
