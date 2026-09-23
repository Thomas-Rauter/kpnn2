"""
parse_adjacency(widths=): several units per named node.

The same widths contract as parse_layered: a named edge A -> B is
the (k_B, k_A) block of unit pairs, target-unit outer,
source-unit inner, and every lookup stays at the named-node level.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from kpnn2 import (
    AdjacencySpec,
    Kpnn2Error,
    MaskedLinear,
    PackedLinear,
    align_inputs,
    map_node_attributions,
    parse_adjacency,
    parse_layered,
)


def _dcell_edgelist():
    """
    Genes feed GO terms T1, T2; the terms and a gene feed R.
    """
    return pd.DataFrame(
        {
            "source": ["g1", "g2", "g3", "g4", "T1", "T2", "g5"],
            "target": ["T1", "T1", "T2", "T2", "R", "R", "R"],
        }
    )


def _dcell_widths():
    """
    Two units per child, DCell-style.
    """
    children = _dcell_edgelist().groupby("target").size()
    return {name: 2 * int(count) for name, count in children.items()}


def _cycle_edgelist():
    return pd.DataFrame(
        {
            "source": ["x", "a", "b", "a"],
            "target": ["a", "b", "a", "y"],
        }
    )


def test_width_one_is_unchanged():
    edgelist = _cycle_edgelist()
    default = parse_adjacency(edgelist)
    explicit = parse_adjacency(
        edgelist,
        widths={},
    )
    ones = parse_adjacency(
        edgelist,
        widths={name: 1 for name in default.nodes},
    )

    assert default == explicit == ones
    assert default.node_widths == (1, 1, 1, 1)
    assert default.state_dim == len(default.nodes)
    assert default.source_index == (0, 0, 1, 2)
    assert default.target_index == (1, 3, 0, 0)
    assert default.input_index == (2,)
    assert default.output_index == (3,)
    assert "widths" not in default.to_dict()


def test_dcell_named_edges_keep_their_names():
    widths = _dcell_widths()
    spec = parse_adjacency(
        _dcell_edgelist(),
        widths=widths,
    )

    assert widths == {"R": 6, "T1": 4, "T2": 4}
    assert spec.nodes == (
        "R",
        "T1",
        "T2",
        "g1",
        "g2",
        "g3",
        "g4",
        "g5",
    )
    assert spec.state_dim == 6 + 4 + 4 + 5
    assert spec.node_units("T1") == slice(6, 10)
    slots = spec.edge_location("T1", "R")
    assert len(slots) == 6 * 4
    assert len(spec.source_index) == sum(
        widths.get(source, 1) * widths.get(target, 1)
        for source, target in _dcell_edgelist().itertuples(index=False)
    )
    da = map_node_attributions(
        torch.zeros(2, spec.state_dim),
        spec,
    )
    assert da["node"].values.tolist() == (
        ["R"] * 6 + ["T1"] * 4 + ["T2"] * 4 + ["g1", "g2", "g3", "g4", "g5"]
    )


def test_edge_location_is_the_full_unit_block():
    spec = parse_adjacency(
        _dcell_edgelist(),
        widths=_dcell_widths(),
    )
    source_units = spec.node_units("T1")
    target_units = spec.node_units("R")

    slots = spec.edge_location("T1", "R")

    pairs = [(spec.target_index[i], spec.source_index[i]) for i in slots]
    expected = [
        (target, source)
        for target in range(target_units.start, target_units.stop)
        for source in range(source_units.start, source_units.stop)
    ]
    assert pairs == expected


def test_to_mask_blocks_match_the_layered_hop():
    """
    On a DAG both layouts expand each named edge into the same
    block, so the dense squares agree block by block.
    """
    widths = _dcell_widths()
    adjacency = parse_adjacency(
        _dcell_edgelist(),
        widths=widths,
    )
    layered = parse_layered(
        _dcell_edgelist(),
        widths=widths,
    )
    square = adjacency.to_mask()
    assert square.shape == (adjacency.state_dim, adjacency.state_dim)
    assert int(square.sum()) == len(adjacency.source_index)

    for source, target in _dcell_edgelist().itertuples(index=False):
        hop_index, _ = layered.edge_location(
            source,
            target,
        )
        hop = layered.hops[hop_index]
        rectangle = hop.to_mask()
        block = square[
            adjacency.node_units(target),
            adjacency.node_units(source),
        ]
        _, target_units = layered.node_units(target)
        layered_block = rectangle[
            target_units,
            layered.hop_units(
                hop,
                source,
            ),
        ]
        assert torch.equal(
            block,
            layered_block,
        )
        assert bool(block.eq(1.0).all())


def test_input_and_output_index_cover_every_unit():
    spec = parse_adjacency(
        _cycle_edgelist(),
        widths={
            "x": 2,
            "y": 3,
        },
    )

    assert spec.node_units("x") == slice(2, 4)
    assert spec.input_index == (2, 3)
    assert spec.output_index == (4, 5, 6)


def test_align_inputs_and_input_index_scatter_each_input():
    spec = parse_adjacency(
        pd.DataFrame(
            {
                "source": ["u", "v", "a"],
                "target": ["a", "a", "y"],
            }
        ),
        widths={"u": 2},
    )
    features = pd.DataFrame(
        {
            "v": [5.0],
            "noise": [9.0],
            "u": [3.0],
        }
    )

    col = align_inputs(
        features.columns,
        spec,
    )

    assert len(col) == len(spec.input_index)
    x = torch.as_tensor(
        features.to_numpy()[:, col],
        dtype=torch.float32,
    )
    state = torch.zeros(
        1,
        spec.state_dim,
    )
    state[:, spec.input_index] = x
    assert state[0, spec.node_units("u")].tolist() == [3.0, 3.0]
    assert state[0, spec.node_units("v")].tolist() == [5.0]


def test_packed_linear_matches_masked_linear_with_widths():
    torch.manual_seed(42)
    spec = parse_adjacency(
        _cycle_edgelist(),
        widths={
            "a": 2,
            "b": 3,
        },
    )
    n = spec.state_dim
    dense = MaskedLinear(spec.to_mask())
    packed = PackedLinear(
        spec.source_index,
        spec.target_index,
        n,
        n,
    )
    with torch.no_grad():
        packed.weight.copy_(dense.weight[spec.target_index, spec.source_index])
        packed.bias.copy_(dense.bias)
    state = torch.randn(
        4,
        n,
    )

    torch.testing.assert_close(
        packed(state),
        dense(state),
    )


def test_to_dict_roundtrips_widths_and_changes_the_fingerprint():
    narrow = parse_adjacency(_cycle_edgelist())
    wide = parse_adjacency(
        _cycle_edgelist(),
        widths={"a": 2},
    )

    payload = wide.to_dict()

    assert payload["widths"] == {"a": 2}
    assert AdjacencySpec.from_dict(payload) == wide
    assert wide.fingerprint != narrow.fingerprint
    assert set(narrow.to_dict()) == {"kpnn2_spec", "layout", "edges"}


def test_to_edgelist_keeps_named_edges_once():
    wide = parse_adjacency(
        _cycle_edgelist(),
        widths={"a": 2},
    )

    table = wide.to_edgelist()

    assert list(zip(table["source"], table["target"])) == [
        ("a", "b"),
        ("a", "y"),
        ("b", "a"),
        ("x", "a"),
    ]


@pytest.mark.parametrize(
    ("widths", "message"),
    [
        ({"nope": 2}, "Unknown node name"),
        ({"a": 0}, "positive int"),
        ({"a": True}, "positive int"),
        ({"a": 1.5}, "positive int"),
        (["a"], "mapping"),
    ],
)
def test_widths_are_validated_like_parse_layered(
    widths,
    message,
):
    with pytest.raises(
        Kpnn2Error,
        match=message,
    ) as adjacency_error:
        parse_adjacency(
            _cycle_edgelist(),
            widths=widths,
        )
    edgelist = pd.DataFrame(
        {
            "source": ["x", "a"],
            "target": ["a", "y"],
        }
    )
    with pytest.raises(Kpnn2Error) as layered_error:
        parse_layered(
            edgelist,
            widths=widths,
        )
    assert str(adjacency_error.value) == str(layered_error.value)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("", "empty"),
        ("nope", "Unknown node name"),
    ],
)
def test_node_units_rejects_unknown_names(
    name,
    message,
):
    spec = parse_adjacency(_cycle_edgelist())

    with pytest.raises(
        Kpnn2Error,
        match=message,
    ):
        spec.node_units(name)


def test_node_units_matches_names_after_str():
    spec = parse_adjacency(
        pd.DataFrame(
            {
                "source": [1, 2],
                "target": [2, 3],
            }
        ),
        widths={2: 3},
    )

    assert spec.node_units(2) == slice(1, 4)
    assert spec.node_widths == (1, 3, 1)
    assert np.array_equal(
        align_inputs(
            ["1"],
            spec,
        ),
        np.array([0]),
    )
