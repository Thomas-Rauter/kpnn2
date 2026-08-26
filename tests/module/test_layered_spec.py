import copy
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest
import torch

from kpnn2 import Kpnn2Error, parse_layered
from kpnn2._frozen_mask import FrozenMask


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


def test_layered_spec_masks_reject_in_place_writes():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_layered(edgelist)

    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        spec.masks[0].fill_(0.0)

    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        spec.masks[0][0, 0] = 0.0


def test_layered_spec_masks_reject_out_kwarg_write():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_layered(edgelist)
    before = spec.masks[0].tolist()

    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        torch.add(
            spec.masks[0],
            1,
            out=spec.masks[0],
        )
    assert spec.masks[0].tolist() == before


def test_layered_spec_masks_numpy_cannot_change_values():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_layered(edgelist)
    before = spec.masks[0].tolist()
    arr = spec.masks[0].numpy()
    try:
        arr[:] = 0
    except (ValueError, Kpnn2Error):
        pass
    assert spec.masks[0].tolist() == before
    assert not arr.flags.writeable


def test_layered_spec_deepcopy_independent_frozen_masks():
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
        assert isinstance(
            orig,
            FrozenMask,
        )
        assert isinstance(
            dup,
            FrozenMask,
        )
        assert orig.dtype == torch.float32
        assert dup.dtype == torch.float32
        assert orig.tolist() == dup.tolist()
        assert orig is not dup
        assert orig.data_ptr() != dup.data_ptr()
        with pytest.raises(
            Kpnn2Error,
            match="read-only",
        ):
            orig.fill_(0.0)
        with pytest.raises(
            Kpnn2Error,
            match="read-only",
        ):
            dup.fill_(0.0)
        assert orig.tolist() == dup.tolist()

    assert [mask.tolist() for mask in spec.masks] == before
    assert [mask.tolist() for mask in copied.masks] == before
