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
    assert isinstance(spec.source_index, tuple)
    assert isinstance(spec.target_index, tuple)
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


def test_adjacency_spec_has_no_mask_field():
    spec = parse_adjacency(_cyclic_edgelist())

    assert not hasattr(spec, "mask")


def test_adjacency_spec_packed_indices_are_canonical():
    spec = parse_adjacency(_cyclic_edgelist())

    assert spec.source_index == (0, 0, 1, 2)
    assert spec.target_index == (1, 3, 0, 0)
    assert len(spec.source_index) == len(spec.target_index)


def test_to_mask_is_a_plain_float32_tensor():
    spec = parse_adjacency(_cyclic_edgelist())
    mask = spec.to_mask()

    assert type(mask) is torch.Tensor
    assert mask.dtype == torch.float32
    assert not mask.requires_grad
    assert mask.is_contiguous()
    assert mask.numpy().flags.writeable
    n_nodes = len(spec.nodes)
    assert tuple(mask.shape) == (n_nodes, n_nodes)


def test_to_mask_does_not_alias_the_layer_or_later_calls():
    spec = parse_adjacency(_cyclic_edgelist())
    mask = spec.to_mask()
    layer = MaskedLinear(mask)
    before = spec.to_mask().tolist()

    mask.fill_(0.0)
    layer.mask.fill_(0.0)

    later = spec.to_mask()
    assert later.tolist() == before
    assert later.data_ptr() != mask.data_ptr()


def test_mutating_to_mask_does_not_change_the_spec():
    spec = parse_adjacency(_cyclic_edgelist())
    source_before = spec.source_index
    target_before = spec.target_index
    first = spec.to_mask()
    expected = first.clone()

    first.fill_(0.0)

    assert spec.source_index == source_before
    assert spec.target_index == target_before
    assert torch.equal(
        spec.to_mask(),
        expected,
    )


def test_two_to_mask_results_do_not_share_storage():
    spec = parse_adjacency(_cyclic_edgelist())
    first = spec.to_mask()
    second = spec.to_mask()

    assert torch.equal(
        first,
        second,
    )
    assert first is not second
    assert first.data_ptr() != second.data_ptr()


def test_adjacency_spec_deepcopy_independent_to_mask():
    spec = parse_adjacency(_cyclic_edgelist())
    copied = copy.deepcopy(spec)

    assert copied is not spec
    assert copied.nodes == spec.nodes
    assert copied.source_index == spec.source_index
    assert copied.target_index == spec.target_index
    assert copied.input_index == spec.input_index
    assert copied.output_index == spec.output_index
    assert not hasattr(copied, "mask")

    original_mask = spec.to_mask()
    copied_mask = copied.to_mask()
    assert torch.equal(
        original_mask,
        copied_mask,
    )
    assert original_mask.data_ptr() != copied_mask.data_ptr()

    copied_mask.fill_(0.0)

    assert torch.equal(
        spec.to_mask(),
        original_mask,
    )
