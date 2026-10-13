import re

import numpy as np
import pandas as pd
import pytest
import torch

from kpnn2 import Kpnn2Error, map_node_attributions, parse_layered

# Wide hidden and output nodes. Input nodes always have width 1.
_WIDTHS = {
    "a": 2,
    "b": 3,
    "c": 2,
    "y": 2,
}

# Longest-path depths; ranks= that lift output z to y's layer.
_RANKS = {
    "u": 0,
    "x": 0,
    "a": 1,
    "b": 1,
    "c": 2,
    "y": 3,
    "z": 3,
}


def _edgelist():
    """
    Inputs u and x feed a and b, which feed c, which feeds output y.
    Output z hangs off a, so longest-path puts it at depth 2, below y.
    """
    return pd.DataFrame(
        {
            "source": ["u", "x", "x", "a", "b", "c", "a"],
            "target": ["a", "a", "b", "c", "c", "y", "z"],
        }
    )


def _spec(widths=None, ranks=None):
    return parse_layered(
        _edgelist(),
        widths=widths,
        ranks=ranks,
    )


def _both_widths():
    return pytest.mark.parametrize(
        "widths",
        [
            pytest.param(
                None,
                id="width_1",
            ),
            pytest.param(
                _WIDTHS,
                id="wide",
            ),
        ],
    )


@_both_widths()
def test_every_node_of_a_layer_gives_every_unit(widths):
    spec = _spec(widths)
    for layer, names in enumerate(spec.layer_nodes):
        assert spec.unit_index(
            names,
            layer=layer,
        ) == tuple(range(spec.layer_dims[layer]))


@_both_widths()
def test_input_nodes_give_input_columns(widths):
    spec = _spec(widths)
    assert spec.input_nodes == ("u", "x")
    assert spec.unit_index(
        spec.input_nodes,
        layer=0,
    ) == (0, 1)


def test_width_one_gives_positions_in_the_layer():
    spec = _spec()
    assert spec.layer_nodes == (
        ("u", "x"),
        ("a", "b"),
        ("c", "z"),
        ("y",),
    )
    for layer, names in enumerate(spec.layer_nodes):
        for position, name in enumerate(names):
            assert spec.unit_index(
                [name],
                layer=layer,
            ) == (position,)
    assert spec.unit_index(
        ["z", "c"],
        layer=2,
    ) == (1, 0)


@_both_widths()
def test_each_node_gives_its_node_units_block(widths):
    spec = _spec(widths)
    for names in spec.layer_nodes:
        for name in names:
            layer, units = spec.node_units(name)
            assert spec.unit_index(
                [name],
                layer=layer,
            ) == tuple(
                range(
                    units.start,
                    units.stop,
                )
            )


def test_wide_nodes_give_blocks_in_names_order():
    spec = _spec(_WIDTHS)
    assert spec.layer_widths[1] == (2, 3)
    assert spec.unit_index(
        ["b", "a"],
        layer=1,
    ) == (2, 3, 4, 0, 1)
    assert spec.layer_widths[2] == (2, 1)
    assert spec.unit_index(
        ["z", "c"],
        layer=2,
    ) == (2, 0, 1)


def test_wide_node_is_not_its_position_in_the_layer():
    # The hand-built index layer_nodes[i].index(name) counts nodes,
    # not units, so it is wrong after a wide node.
    spec = _spec(_WIDTHS)
    assert spec.layer_nodes[1].index("b") == 1
    assert spec.unit_index(
        ["b"],
        layer=1,
    ) == (2, 3, 4)


@_both_widths()
def test_units_match_the_attribution_labels(widths):
    spec = _spec(widths)
    for layer, names in enumerate(spec.layer_nodes):
        scores = map_node_attributions(
            torch.zeros(
                1,
                spec.layer_dims[layer],
            ),
            spec,
            layer,
        )
        labels = scores.coords["node"].values.tolist()
        for name in names:
            labelled = tuple(
                unit for unit, label in enumerate(labels) if label == name
            )
            assert labelled == spec.unit_index(
                [name],
                layer=layer,
            )


@_both_widths()
def test_index_reads_the_node_units_slices(widths):
    spec = _spec(widths)
    saved = torch.arange(
        2.0 * spec.layer_dims[1],
    ).reshape(
        2,
        spec.layer_dims[1],
    )
    names = ["b", "a", "b"]
    index = torch.as_tensor(
        spec.unit_index(
            names,
            layer=1,
        ),
        dtype=torch.int64,
    )
    expected = torch.cat(
        [saved[..., spec.node_units(name)[1]] for name in names],
        dim=-1,
    )
    torch.testing.assert_close(
        saved[..., index],
        expected,
    )


@_both_widths()
def test_outputs_ranked_to_one_layer_read_in_label_order(widths):
    labels = ["z", "y"]
    longest = _spec(widths)
    assert longest.node_units("z")[0] == 2
    assert longest.node_units("y")[0] == 3
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("Node(s) not in layer 3: z (layer 2)."),
    ):
        longest.unit_index(
            labels,
            layer=3,
        )
    ranked = _spec(
        widths,
        _RANKS,
    )
    top = len(ranked.layer_nodes) - 1
    assert ranked.layer_nodes[top] == ("y", "z")
    index = ranked.unit_index(
        labels,
        layer=top,
    )
    assert index == ranked.unit_index(
        ["z"],
        layer=top,
    ) + ranked.unit_index(
        ["y"],
        layer=top,
    )
    assert sorted(index) == list(range(ranked.layer_dims[top]))


def test_repeated_name_repeats_its_units():
    spec = _spec(_WIDTHS)
    a_units = spec.unit_index(
        ["a"],
        layer=1,
    )
    b_units = spec.unit_index(
        ["b"],
        layer=1,
    )
    assert a_units == (0, 1)
    assert (
        spec.unit_index(
            ["a", "b", "a"],
            layer=1,
        )
        == a_units + b_units + a_units
    )


def test_empty_names_give_an_empty_tuple():
    spec = _spec(_WIDTHS)
    assert (
        spec.unit_index(
            [],
            layer=1,
        )
        == ()
    )
    assert (
        spec.unit_index(
            np.array([], dtype=str),
            layer=1,
        )
        == ()
    )
    index = torch.as_tensor(
        spec.unit_index(
            (),
            layer=1,
        ),
        dtype=torch.int64,
    )
    saved = torch.zeros(
        2,
        spec.layer_dims[1],
    )
    assert tuple(saved[..., index].shape) == (2, 0)


def test_integer_names_match_after_str():
    spec = parse_layered(
        pd.DataFrame(
            {
                "source": [1, 1, 2, 3],
                "target": [2, 3, 4, 4],
            }
        )
    )
    assert spec.layer_nodes[1] == ("2", "3")
    assert spec.unit_index(
        [3, "2"],
        layer=1,
    ) == (1, 0)
    assert spec.unit_index(
        np.array([3, 2]),
        layer=1,
    ) == (1, 0)


@pytest.mark.parametrize(
    "integer_type",
    [
        pytest.param(
            np.int64,
            id="int64",
        ),
        pytest.param(
            np.uint8,
            id="uint8",
        ),
    ],
)
def test_numpy_integer_layer_matches_int(integer_type):
    spec = _spec(_WIDTHS)
    assert spec.unit_index(
        ["b", "a"],
        layer=integer_type(1),
    ) == spec.unit_index(
        ["b", "a"],
        layer=1,
    )


@_both_widths()
def test_items_are_python_int(widths):
    spec = _spec(widths)
    for names in (
        list(spec.layer_nodes[1]),
        np.array(spec.layer_nodes[1]),
    ):
        index = spec.unit_index(
            names,
            layer=np.int64(1),
        )
        assert index
        assert all(type(unit) is int for unit in index)


@pytest.mark.parametrize(
    "make_names",
    [
        pytest.param(
            lambda: ["b", "a"],
            id="list",
        ),
        pytest.param(
            lambda: ("b", "a"),
            id="tuple",
        ),
        pytest.param(
            lambda: np.array(["b", "a"]),
            id="numpy",
        ),
        pytest.param(
            lambda: pd.Index(["b", "a"]),
            id="pandas_index",
        ),
        pytest.param(
            lambda: (name for name in ["b", "a"]),
            id="generator",
        ),
    ],
)
def test_accepted_containers(make_names):
    spec = _spec(_WIDTHS)
    assert spec.unit_index(
        make_names(),
        layer=1,
    ) == (2, 3, 4, 0, 1)


def test_caller_names_are_not_mutated():
    spec = _spec(_WIDTHS)
    names = ["b", "a", "a"]
    array = np.array(names)
    spec.unit_index(
        names,
        layer=1,
    )
    spec.unit_index(
        array,
        layer=1,
    )
    assert names == ["b", "a", "a"]
    np.testing.assert_array_equal(
        array,
        np.array(["b", "a", "a"]),
    )


def test_layer_is_keyword_only_and_required():
    spec = _spec()
    with pytest.raises(TypeError):
        spec.unit_index(
            ["a"],
            1,
        )
    with pytest.raises(TypeError):
        spec.unit_index(["a"])


@pytest.mark.parametrize(
    ("names", "layer", "message"),
    [
        pytest.param(
            "a",
            1,
            "'names' is a string; pass a sequence of node names.",
            id="str",
        ),
        pytest.param(
            b"a",
            1,
            "'names' is bytes; pass a sequence of node names.",
            id="bytes",
        ),
        pytest.param(
            torch.tensor([0]),
            1,
            "'names' is a tensor; pass a sequence of node names.",
            id="tensor",
        ),
        pytest.param(
            5,
            1,
            "Unsupported names type. Expected a sequence of node names. "
            "Got 5 (int).",
            id="int",
        ),
        pytest.param(
            {"a"},
            1,
            "'names' must be a sequence of node names, not a set: its "
            "items are matched by position. Got {'a'} (set). Pass a list "
            "or tuple in the intended order.",
            id="set",
        ),
        pytest.param(
            {"a": 0},
            1,
            "'names' must be a sequence of node names, not a mapping: its "
            "items are matched by position. Got {'a': 0} (dict). Pass a "
            "list or tuple in the intended order.",
            id="mapping",
        ),
        pytest.param(
            np.array([["a"]]),
            1,
            "'names' must be one-dimensional. Got ndarray of shape (1, 1), "
            "dtype <U1. A matrix is not a name list; pass the node names "
            "and index the matrix yourself.",
            id="matrix",
        ),
        pytest.param(
            ["a"],
            1.0,
            "'layer' must be an int. Got 1.0 (float).",
            id="layer_float",
        ),
        pytest.param(
            ["a"],
            True,
            "'layer' must be an int. Got True (bool).",
            id="layer_bool",
        ),
        pytest.param(
            ["a"],
            np.bool_(True),
            "'layer' must be an int. Got True (bool).",
            id="layer_numpy_bool",
        ),
        pytest.param(
            ["a"],
            None,
            "'layer' must be an int. Got None.",
            id="layer_none",
        ),
        pytest.param(
            ["a"],
            -1,
            "'layer' must be in range [0, 4). Got -1.",
            id="layer_negative",
        ),
        pytest.param(
            ["a"],
            4,
            "'layer' must be in range [0, 4). Got 4.",
            id="layer_past_the_end",
        ),
        pytest.param(
            ["q"],
            4,
            "'layer' must be in range [0, 4). Got 4.",
            id="layer_before_unknown_name",
        ),
        pytest.param(
            ["a", ""],
            1,
            "Node name is empty.",
            id="empty_name",
        ),
        pytest.param(
            ["q", ""],
            1,
            "Node name is empty.",
            id="empty_name_before_unknown",
        ),
        pytest.param(
            ["q", "a", "Q", "q"],
            1,
            "Unknown node name(s): Q, q. Valid names are in spec.layer_nodes.",
            id="unknown_names",
        ),
        pytest.param(
            ["q", "y"],
            1,
            "Unknown node name(s): q. Valid names are in spec.layer_nodes.",
            id="unknown_before_other_layer",
        ),
        pytest.param(
            ["y", "a", "u", "y", "c"],
            1,
            "Node(s) not in layer 1: c (layer 2), u (layer 0), y (layer 3). "
            "Valid names are in spec.layer_nodes[1].",
            id="other_layers",
        ),
    ],
)
def test_rejection_messages_are_pinned(
    names,
    layer,
    message,
):
    with pytest.raises(
        Kpnn2Error,
        match=f"^{re.escape(message)}$",
    ):
        _spec().unit_index(
            names,
            layer=layer,
        )
