import copy
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest
import torch

from kpnn2 import MaskedLinear, parse_layered


def test_layered_spec_rejects_field_assignment():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_layered(edgelist)

    with pytest.raises(FrozenInstanceError):
        spec.input_nodes = ("X",)


def test_layered_spec_sequences_are_tuples():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_layered(edgelist)

    assert isinstance(spec.input_nodes, tuple)
    assert isinstance(spec.layer_nodes, tuple)
    assert isinstance(spec.layer_nodes[0], tuple)
    assert isinstance(spec.masks, tuple)
    assert isinstance(spec.skips, tuple)
    with pytest.raises(AttributeError):
        spec.input_nodes.append("X")


def test_layered_spec_masks_are_plain_float32_tensors():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_layered(edgelist)

    for mask in spec.masks:
        assert type(mask) is torch.Tensor
        assert mask.dtype == torch.float32
        assert not mask.requires_grad
        assert mask.is_contiguous()
        assert mask.numpy().flags.writeable


def test_layered_spec_masks_do_not_alias_the_parsed_tensors():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_layered(edgelist)
    layer = MaskedLinear(spec.masks[0])
    layer_before = layer.mask.tolist()
    other = parse_layered(edgelist)

    spec.masks[0].fill_(0.0)

    assert layer.mask.tolist() == layer_before
    assert other.masks[0].tolist() == layer_before


def test_layered_spec_deepcopy_independent_masks():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_layered(edgelist)
    before = [mask.tolist() for mask in spec.masks]
    copied = copy.deepcopy(spec)

    assert copied is not spec
    assert copied.input_nodes == spec.input_nodes
    assert copied.layer_nodes == spec.layer_nodes
    assert len(copied.masks) == len(spec.masks)
    for orig, dup in zip(
        spec.masks,
        copied.masks,
        strict=True,
    ):
        assert type(orig) is torch.Tensor
        assert type(dup) is torch.Tensor
        assert orig.dtype == torch.float32
        assert dup.dtype == torch.float32
        assert orig.tolist() == dup.tolist()
        assert orig is not dup
        assert orig.data_ptr() != dup.data_ptr()
        dup.fill_(0.0)

    assert [mask.tolist() for mask in spec.masks] == before
