import pandas as pd
import pytest
import torch

from kpnn2 import Kpnn2Error, parse_adjacency, parse_layered
from kpnn2._layout import (
    DEFAULT_NODE_WIDTH,
    Layout,
    NodeSlot,
    build_layout,
    concat_layouts,
    expand_columns,
    fill_block,
)
from kpnn2._parse import _build_hops, _build_skips, _node_placement
from kpnn2._parse_adjacency import _build_square_mask


def _wide_layouts():
    return [
        build_layout(
            ["A", "B"],
            [2, 3],
        ),
        build_layout(
            ["H"],
            [2],
        ),
    ]


def test_default_width_is_one():
    assert DEFAULT_NODE_WIDTH == 1


def test_build_layout_defaults_to_one_unit_per_node():
    layout = build_layout(["A", "B", "C"])
    assert layout.n_units == 3
    assert layout.names == ("A", "B", "C")
    assert layout.widths() == (1, 1, 1)
    assert layout.unit_names() == ["A", "B", "C"]


def test_width_one_slot_start_is_the_column_index():
    names = ["A", "B", "C"]
    layout = build_layout(names)
    for index, name in enumerate(names):
        slot = layout.slot(name)
        assert slot.start == index
        assert slot.stop == index + 1
        assert slot.width == 1
        assert layout.start_of(name) == index
        assert layout.slot_at(index) is slot


def test_wider_layout_tiles_the_axis_without_gaps():
    layout = build_layout(
        ["A", "B", "C"],
        [2, 1, 3],
    )
    assert layout.n_units == 6
    assert layout.widths() == (2, 1, 3)
    assert layout.slot("A").units == slice(0, 2)
    assert layout.slot("B").units == slice(2, 3)
    assert layout.slot("C").units == slice(3, 6)
    assert layout.unit_names() == [
        "A",
        "A",
        "B",
        "C",
        "C",
        "C",
    ]
    assert layout.slot_at(3) is layout.slot("C")


def test_layout_rejects_unknown_name_and_unknown_start():
    layout = build_layout(["A", "B"])
    with pytest.raises(
        Kpnn2Error,
        match="Unknown node name",
    ):
        layout.slot("missing")
    with pytest.raises(
        Kpnn2Error,
        match="No node begins at unit index",
    ):
        layout.slot_at(7)


def test_layout_rejects_gaps_overlaps_duplicates_and_zero_width():
    with pytest.raises(
        Kpnn2Error,
        match="without gaps",
    ):
        Layout(
            slots=(
                NodeSlot(
                    name="A",
                    start=1,
                    width=1,
                ),
            )
        )
    with pytest.raises(
        Kpnn2Error,
        match="without gaps",
    ):
        Layout(
            slots=(
                NodeSlot(
                    name="A",
                    start=0,
                    width=2,
                ),
                NodeSlot(
                    name="B",
                    start=1,
                    width=1,
                ),
            )
        )
    with pytest.raises(
        Kpnn2Error,
        match="Duplicate node name",
    ):
        Layout(
            slots=(
                NodeSlot(
                    name="A",
                    start=0,
                    width=1,
                ),
                NodeSlot(
                    name="A",
                    start=1,
                    width=1,
                ),
            )
        )
    with pytest.raises(
        Kpnn2Error,
        match="at least one unit",
    ):
        Layout(
            slots=(
                NodeSlot(
                    name="A",
                    start=0,
                    width=0,
                ),
            )
        )


def test_build_layout_rejects_width_count_mismatch():
    with pytest.raises(
        Kpnn2Error,
        match="one entry per node",
    ):
        build_layout(
            ["A", "B"],
            [1],
        )


def test_fill_block_at_width_one_sets_a_single_entry():
    layout = build_layout(["A", "B"])
    target_layout = build_layout(["H", "K"])
    mask = torch.zeros(
        2,
        2,
    )
    fill_block(
        mask,
        target_layout.slot("K"),
        layout.slot("A"),
    )
    assert mask.tolist() == [
        [0.0, 0.0],
        [1.0, 0.0],
    ]


def test_fill_block_marks_every_unit_pair_of_one_edge():
    source_layout = build_layout(
        ["A", "B"],
        [2, 1],
    )
    target_layout = build_layout(
        ["H"],
        [2],
    )
    mask = torch.zeros(
        target_layout.n_units,
        source_layout.n_units,
    )
    fill_block(
        mask,
        target_layout.slot("H"),
        source_layout.slot("A"),
    )
    assert mask.tolist() == [
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]


def test_expand_columns_is_a_no_op_at_width_one():
    layout = build_layout(["A", "B"])
    values = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": [3.0, 4.0],
        }
    ).to_numpy()
    out = expand_columns(
        values,
        layout,
    )
    assert out is values


def test_expand_columns_repeats_each_node_across_its_units():
    layout = build_layout(
        ["A", "B"],
        [2, 1],
    )
    values = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": [3.0, 4.0],
        }
    ).to_numpy()
    out = expand_columns(
        values,
        layout,
    )
    assert out.tolist() == [
        [1.0, 1.0, 3.0],
        [2.0, 2.0, 4.0],
    ]


def test_concat_layouts_shifts_each_layout_onto_one_axis():
    joined = concat_layouts(_wide_layouts())
    assert joined.n_units == 7
    assert joined.names == ("A", "B", "H")
    assert joined.slot("A").units == slice(0, 2)
    assert joined.slot("B").units == slice(2, 5)
    assert joined.slot("H").units == slice(5, 7)
    assert joined.unit_names() == [
        "A",
        "A",
        "B",
        "B",
        "B",
        "H",
        "H",
    ]


def test_concat_layouts_of_nothing_is_an_empty_axis():
    joined = concat_layouts([])
    assert joined.n_units == 0
    assert joined.names == ()


def test_concat_layouts_rejects_a_repeated_name():
    layout = build_layout(["A"])
    with pytest.raises(
        Kpnn2Error,
        match="Duplicate node name",
    ):
        concat_layouts(
            [
                layout,
                layout,
            ]
        )


def test_build_hops_block_expands_a_wider_layout():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["H", "H"],
        }
    )
    layouts = _wide_layouts()
    hops = _build_hops(
        edgelist,
        layouts,
        _node_placement(layouts),
    )
    assert len(hops) == 1
    assert hops[0].source_layers == (0,)
    assert hops[0].source_dims == (5,)
    assert tuple(hops[0].mask.shape) == (2, 5)
    assert hops[0].mask.tolist() == [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ]


def test_build_hops_leaves_unconnected_blocks_zero():
    edgelist = pd.DataFrame(
        {
            "source": ["A"],
            "target": ["H"],
        }
    )
    layouts = _wide_layouts()
    hops = _build_hops(
        edgelist,
        layouts,
        _node_placement(layouts),
    )
    assert hops[0].mask.tolist() == [
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
    ]


def test_build_hops_block_expands_a_skip_edge_too():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H", "B"],
            "target": ["H", "C", "C"],
        }
    )
    layouts = [
        build_layout(
            ["A", "B"],
            [2, 3],
        ),
        build_layout(
            ["H"],
            [2],
        ),
        build_layout(
            ["C"],
            [4],
        ),
    ]
    hops = _build_hops(
        edgelist,
        layouts,
        _node_placement(layouts),
    )
    assert len(hops) == 2
    skip_hop = hops[1]
    assert skip_hop.target_layer == 2
    assert skip_hop.source_layers == (0, 1)
    assert skip_hop.source_dims == (5, 2)
    assert skip_hop.column_offsets == (0, 5)
    assert tuple(skip_hop.mask.shape) == (4, 7)
    expected_row = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert skip_hop.mask.tolist() == [expected_row] * 4


def test_build_skips_records_the_block_start():
    edgelist = pd.DataFrame(
        {
            "source": ["B"],
            "target": ["C"],
        }
    )
    layouts = [
        build_layout(
            ["A", "B"],
            [2, 3],
        ),
        build_layout(
            ["H"],
            [2],
        ),
        build_layout(
            ["C"],
            [4],
        ),
    ]
    skips = _build_skips(
        edgelist,
        _node_placement(layouts),
    )
    assert len(skips) == 1
    assert skips[0].source_index == 2
    assert skips[0].target_index == 0
    assert skips[0].source_layer == 0
    assert skips[0].target_layer == 2


def test_build_square_mask_block_expands_a_wider_layout():
    edgelist = pd.DataFrame(
        {
            "source": ["a", "b"],
            "target": ["b", "b"],
        }
    )
    layout = build_layout(
        ["a", "b"],
        [1, 2],
    )
    mask = _build_square_mask(
        edgelist,
        layout,
    )
    assert tuple(mask.shape) == (3, 3)
    assert mask.tolist() == [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]


def test_parsers_still_place_one_unit_per_node():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B", "H", "A"],
            "target": ["H", "H", "C", "C"],
        }
    )
    spec = parse_layered(edgelist)
    assert spec.layer_dims == tuple(len(names) for names in spec.layer_nodes)
    for depth, names in enumerate(spec.layer_nodes):
        layout = build_layout(names)
        assert layout.n_units == spec.layer_dims[depth]
    skip = spec.skips[0]
    assert skip.source_index == spec.layer_nodes[0].index(skip.source)
    assert skip.target_index == spec.layer_nodes[2].index(skip.target)

    state_spec = parse_adjacency(edgelist)
    assert tuple(state_spec.mask.shape) == (
        len(state_spec.nodes),
        len(state_spec.nodes),
    )
    assert state_spec.input_index == tuple(
        state_spec.nodes.index(name) for name in state_spec.input_nodes
    )
