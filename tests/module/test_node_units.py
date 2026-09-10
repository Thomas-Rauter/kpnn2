from dataclasses import replace

import pandas as pd
import pytest
import torch

from kpnn2 import Kpnn2Error, parse_layered


def _chain_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
    )


def _chain_plus_skip():
    return pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )


def _two_parents():
    return pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["H", "H"],
        }
    )


def test_width_one_slice_is_the_layer_ordinal():
    spec = parse_layered(_chain_edgelist())
    for depth, names in enumerate(spec.layer_nodes):
        for ordinal, name in enumerate(names):
            layer, units = spec.node_units(name)
            assert layer == depth
            assert units.start == ordinal
            assert units.stop == ordinal + 1
            assert units.step is None


def test_wide_node_is_a_contiguous_block():
    spec = parse_layered(
        _chain_edgelist(),
        widths={"H": 3},
    )
    layer, units = spec.node_units("H")
    assert layer == 1
    assert units.start == 0
    assert units.stop == 3
    assert spec.layer_dims[layer] == 3
    _, a_units = spec.node_units("A")
    assert a_units.start == 0
    assert a_units.stop == 1


def test_two_nodes_in_a_layer_use_prefix_widths():
    spec = parse_layered(
        _two_parents(),
        widths={
            "A": 2,
            "B": 3,
        },
    )
    assert spec.layer_nodes[0] == ("A", "B")
    saved = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0, 4.0]],
    )
    _, a_units = spec.node_units("A")
    _, b_units = spec.node_units("B")
    assert saved[:, a_units].tolist() == [[0.0, 1.0]]
    assert saved[:, b_units].tolist() == [[2.0, 3.0, 4.0]]


def test_skip_block_starts_match_node_units():
    spec = parse_layered(
        _chain_plus_skip(),
        widths={
            "A": 2,
            "C": 4,
        },
    )
    skip = spec.skips[0]
    source_layer, source_units = spec.node_units(skip.source)
    target_layer, target_units = spec.node_units(skip.target)
    assert source_layer == skip.source_layer
    assert target_layer == skip.target_layer
    assert source_units.start == skip.source_in_layer
    assert target_units.start == skip.target_in_layer
    assert source_units.stop == skip.source_in_layer + 2
    assert target_units.stop == skip.target_in_layer + 4


def test_hop_units_skip_axis_concatenates_source_layers():
    spec = parse_layered(_chain_plus_skip())
    hop = spec.hops[1]
    a_units = spec.hop_units(
        hop,
        "A",
    )
    h_units = spec.hop_units(
        hop,
        "H",
    )
    assert a_units.start == 0
    assert a_units.stop == 1
    assert h_units.start == 1
    assert h_units.stop == 2
    assert hop.column_offsets == (0, 1)


def test_hop_units_wide_skip_matches_concatenated_blocks():
    spec = parse_layered(
        _chain_plus_skip(),
        widths={
            "A": 2,
            "H": 3,
        },
    )
    hop = spec.hops[1]
    a_units = spec.hop_units(
        hop,
        "A",
    )
    h_units = spec.hop_units(
        hop,
        "H",
    )
    assert a_units == slice(0, 2)
    assert h_units == slice(2, 5)
    assert hop.in_features == 5
    layer, h_layer_units = spec.node_units("H")
    offset = hop.column_offsets[hop.source_layers.index(layer)]
    assert h_units.start == offset + h_layer_units.start
    assert h_units.stop == offset + h_layer_units.stop


def test_hop_units_first_hop_is_the_input_layer():
    spec = parse_layered(
        _two_parents(),
        widths={
            "A": 2,
            "B": 3,
        },
    )
    hop = spec.hops[0]
    assert spec.hop_units(hop, "A") == spec.node_units("A")[1]
    assert spec.hop_units(hop, "B") == spec.node_units("B")[1]


def test_str_matching_accepts_integer_names():
    edgelist = pd.DataFrame(
        {
            "source": [1, 2],
            "target": [2, 3],
        }
    )
    spec = parse_layered(edgelist)
    layer, units = spec.node_units(1)
    as_string = spec.node_units("1")
    assert layer == as_string[0]
    assert units == as_string[1]
    hop = spec.hops[0]
    assert spec.hop_units(hop, 1) == spec.hop_units(hop, "1")


def test_empty_and_unknown_name_raise():
    spec = parse_layered(_chain_edgelist())
    with pytest.raises(
        Kpnn2Error,
        match="empty",
    ):
        spec.node_units("")
    with pytest.raises(
        Kpnn2Error,
        match="Unknown node name: Z",
    ):
        spec.node_units("Z")
    with pytest.raises(
        Kpnn2Error,
        match="empty",
    ):
        spec.hop_units(
            spec.hops[0],
            "",
        )
    with pytest.raises(
        Kpnn2Error,
        match="Unknown node name: Z",
    ):
        spec.hop_units(
            spec.hops[0],
            "Z",
        )


def test_hop_units_rejects_a_target_only_node():
    spec = parse_layered(_chain_plus_skip())
    with pytest.raises(
        Kpnn2Error,
        match="not on this hop's source axis",
    ):
        spec.hop_units(
            spec.hops[0],
            "C",
        )
    with pytest.raises(
        Kpnn2Error,
        match="not on this hop's source axis",
    ):
        spec.hop_units(
            spec.hops[1],
            "C",
        )
    with pytest.raises(
        Kpnn2Error,
        match="not on this hop's source axis",
    ):
        spec.hop_units(
            spec.hops[0],
            "H",
        )


def test_hop_units_rejects_non_hop_and_foreign_hop():
    spec = parse_layered(_chain_plus_skip())
    hop = spec.hops[1]
    other = replace(
        hop,
        source_index=hop.source_index + (0,),
        target_index=hop.target_index + (0,),
    )
    with pytest.raises(
        Kpnn2Error,
        match="must be a Hop",
    ):
        spec.hop_units(
            1,
            "A",
        )
    with pytest.raises(
        Kpnn2Error,
        match="must match an entry of spec.hops",
    ):
        spec.hop_units(
            other,
            "A",
        )
    foreign = parse_layered(_chain_edgelist()).hops[1]
    with pytest.raises(
        Kpnn2Error,
        match="must match an entry of spec.hops",
    ):
        spec.hop_units(
            foreign,
            "A",
        )
