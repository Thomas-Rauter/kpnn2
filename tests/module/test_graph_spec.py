from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from kpnn2 import Kpnn2Error, parse_edgelist


def test_graph_spec_rejects_field_assignment():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_edgelist(edgelist)

    with pytest.raises(FrozenInstanceError):
        spec.input_nodes = ("X",)


def test_graph_spec_sequences_are_tuples():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_edgelist(edgelist)

    assert isinstance(spec.input_nodes, tuple)
    assert isinstance(spec.layer_nodes, tuple)
    assert isinstance(spec.layer_nodes[0], tuple)
    assert isinstance(spec.masks, tuple)
    assert isinstance(spec.skips, tuple)
    with pytest.raises(AttributeError):
        spec.input_nodes.append("X")


def test_graph_spec_masks_reject_in_place_writes():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_edgelist(edgelist)

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
