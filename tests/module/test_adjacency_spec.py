import copy
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest
import torch

from kpnn2 import MaskedLinear, parse_adjacency


def _cyclic_edgelist():
    return pd.DataFrame(
        {
            "source": ["x", "a", "b", "a"],
            "target": ["a", "b", "a", "y"],
        }
    )


def test_adjacency_spec_rejects_field_assignment():
    spec = parse_adjacency(_cyclic_edgelist())

    with pytest.raises(FrozenInstanceError):
        spec.nodes = ("X",)


def test_adjacency_spec_sequences_are_tuples():
    spec = parse_adjacency(_cyclic_edgelist())

    assert isinstance(spec.nodes, tuple)
    assert isinstance(spec.input_nodes, tuple)
    assert isinstance(spec.output_nodes, tuple)
    assert isinstance(spec.hidden_nodes, tuple)
    assert isinstance(spec.input_index, tuple)
    assert isinstance(spec.output_index, tuple)
    with pytest.raises(AttributeError):
        spec.nodes.append("X")


def test_adjacency_spec_has_no_layered_fields():
    spec = parse_adjacency(_cyclic_edgelist())

    assert not hasattr(spec, "layer_nodes")
    assert not hasattr(spec, "layer_dims")
    assert not hasattr(spec, "masks")
    assert not hasattr(spec, "skips")


def test_adjacency_spec_mask_is_a_plain_float32_tensor():
    spec = parse_adjacency(_cyclic_edgelist())

    assert type(spec.mask) is torch.Tensor
    assert spec.mask.dtype == torch.float32
    assert not spec.mask.requires_grad
    assert spec.mask.is_contiguous()
    assert spec.mask.numpy().flags.writeable


def test_adjacency_spec_mask_does_not_alias_the_parsed_tensor():
    spec = parse_adjacency(_cyclic_edgelist())
    layer = MaskedLinear(spec.mask)
    layer_before = layer.mask.tolist()
    other = parse_adjacency(_cyclic_edgelist())

    spec.mask.fill_(0.0)

    assert layer.mask.tolist() == layer_before
    assert other.mask.tolist() == layer_before


def test_adjacency_spec_deepcopy_independent_mask():
    spec = parse_adjacency(_cyclic_edgelist())
    before = spec.mask.tolist()
    copied = copy.deepcopy(spec)

    assert copied is not spec
    assert copied.nodes == spec.nodes
    assert copied.input_index == spec.input_index
    assert copied.output_index == spec.output_index

    assert type(copied.mask) is torch.Tensor
    assert copied.mask.dtype == torch.float32
    assert copied.mask.tolist() == before
    assert copied.mask is not spec.mask
    assert copied.mask.data_ptr() != spec.mask.data_ptr()

    copied.mask.fill_(0.0)

    assert spec.mask.tolist() == before
