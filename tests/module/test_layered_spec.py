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
    assert isinstance(spec.hops, tuple)
    assert isinstance(spec.skips, tuple)
    for hop in spec.hops:
        assert isinstance(hop, Hop)
        assert isinstance(hop.source_layers, tuple)
        assert isinstance(hop.source_dims, tuple)
        assert isinstance(hop.source_nodes, tuple)
    with pytest.raises(AttributeError):
        spec.input_nodes.append("X")


def test_layered_spec_hop_count_matches_layers():
    spec = parse_layered(_chain_edgelist())

    assert len(spec.hops) == len(spec.layer_nodes) - 1
    for index, hop in enumerate(spec.hops):
        assert hop.target_layer == index + 1
        assert hop.mask.shape[0] == spec.layer_dims[hop.target_layer]
        assert hop.mask.shape[1] == sum(hop.source_dims)
        assert len(hop.source_dims) == len(hop.source_layers)
        assert len(hop.column_offsets) == len(hop.source_layers)


def test_layered_spec_masks_are_plain_float32_tensors():
    spec = parse_layered(_chain_edgelist())

    for hop in spec.hops:
        mask = hop.mask
        assert type(mask) is torch.Tensor
        assert mask.dtype == torch.float32
        assert not mask.requires_grad
        assert mask.is_contiguous()
        assert mask.numpy().flags.writeable


def test_layered_spec_masks_do_not_alias_the_parsed_tensors():
    edgelist = _chain_edgelist()
    spec = parse_layered(edgelist)
    layer = MaskedLinear(spec.hops[0].mask)
    layer_before = layer.mask.tolist()
    other = parse_layered(edgelist)

    spec.hops[0].mask.fill_(0.0)

    assert layer.mask.tolist() == layer_before
    assert other.hops[0].mask.tolist() == layer_before


def test_layered_spec_deepcopy_independent_masks():
    spec = parse_layered(_chain_edgelist())
    before = [hop.mask.tolist() for hop in spec.hops]
    copied = copy.deepcopy(spec)

    assert copied is not spec
    assert copied.input_nodes == spec.input_nodes
    assert copied.layer_nodes == spec.layer_nodes
    assert len(copied.hops) == len(spec.hops)
    for original, duplicate in zip(
        spec.hops,
        copied.hops,
        strict=True,
    ):
        assert duplicate.target_layer == original.target_layer
        assert duplicate.source_layers == original.source_layers
        assert duplicate.source_nodes == original.source_nodes
        assert type(original.mask) is torch.Tensor
        assert type(duplicate.mask) is torch.Tensor
        assert original.mask.dtype == torch.float32
        assert duplicate.mask.dtype == torch.float32
        assert original.mask.tolist() == duplicate.mask.tolist()
        assert original.mask is not duplicate.mask
        assert original.mask.data_ptr() != duplicate.mask.data_ptr()
        duplicate.mask.fill_(0.0)

    assert [hop.mask.tolist() for hop in spec.hops] == before
