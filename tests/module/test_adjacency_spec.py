import copy
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest
import torch

from kpnn2 import Kpnn2Error, parse_adjacency
from kpnn2._frozen_mask import FrozenMask


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


def test_adjacency_spec_mask_is_frozen_float32():
    spec = parse_adjacency(_cyclic_edgelist())

    assert isinstance(
        spec.mask,
        FrozenMask,
    )
    assert spec.mask.dtype == torch.float32


def test_adjacency_spec_mask_rejects_in_place_writes():
    spec = parse_adjacency(_cyclic_edgelist())

    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        spec.mask.fill_(0.0)

    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        spec.mask[0, 0] = 1.0


def test_adjacency_spec_mask_rejects_out_kwarg_write():
    spec = parse_adjacency(_cyclic_edgelist())
    before = spec.mask.tolist()

    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        torch.add(
            spec.mask,
            1,
            out=spec.mask,
        )
    assert spec.mask.tolist() == before


def test_adjacency_spec_mask_numpy_cannot_change_values():
    spec = parse_adjacency(_cyclic_edgelist())
    before = spec.mask.tolist()
    arr = spec.mask.numpy()
    try:
        arr[:] = 0
    except (ValueError, Kpnn2Error):
        pass
    assert spec.mask.tolist() == before
    assert not arr.flags.writeable


def test_adjacency_spec_deepcopy_independent_frozen_mask():
    spec = parse_adjacency(_cyclic_edgelist())
    before = spec.mask.tolist()
    copied = copy.deepcopy(spec)

    assert copied is not spec
    assert copied.nodes == spec.nodes
    assert copied.input_index == spec.input_index
    assert copied.output_index == spec.output_index

    assert isinstance(
        copied.mask,
        FrozenMask,
    )
    assert copied.mask.dtype == torch.float32
    assert copied.mask.tolist() == before
    assert copied.mask is not spec.mask
    assert copied.mask.data_ptr() != spec.mask.data_ptr()

    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        spec.mask.fill_(0.0)
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        copied.mask.fill_(0.0)

    assert spec.mask.tolist() == before
    assert copied.mask.tolist() == before
