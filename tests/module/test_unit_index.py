import re

import numpy as np
import pandas as pd
import pytest
import torch

from kpnn2 import Kpnn2Error, map_node_attributions, parse_adjacency

# Nodes of a width-k block: a hidden, an input, and an output node.
_WIDTHS = {
    "a": 2,
    "u": 3,
    "y": 2,
}


def _edgelist():
    """
    Inputs u and x feed a feedback core a <-> b; outputs y and z.
    """
    return pd.DataFrame(
        {
            "source": ["x", "u", "a", "b", "a", "b"],
            "target": ["a", "b", "b", "a", "y", "z"],
        }
    )


def _spec(widths=None):
    return parse_adjacency(
        _edgelist(),
        widths=widths,
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
def test_input_nodes_give_input_index(widths):
    spec = _spec(widths)
    assert spec.input_nodes == ("u", "x")
    assert spec.unit_index(spec.input_nodes) == spec.input_index


@_both_widths()
def test_output_nodes_give_output_index(widths):
    spec = _spec(widths)
    assert spec.output_nodes == ("y", "z")
    assert spec.unit_index(spec.output_nodes) == spec.output_index


def test_width_one_gives_positions_in_nodes():
    spec = _spec()
    assert spec.nodes == ("a", "b", "u", "x", "y", "z")
    for position, name in enumerate(spec.nodes):
        assert spec.unit_index([name]) == (position,)
    backwards = list(reversed(spec.nodes))
    assert spec.unit_index(backwards) == (5, 4, 3, 2, 1, 0)


@_both_widths()
def test_each_node_gives_its_whole_block(widths):
    spec = _spec(widths)
    for name in spec.nodes:
        units = spec.node_units(name)
        assert spec.unit_index([name]) == tuple(
            range(
                units.start,
                units.stop,
            )
        )


def test_wide_nodes_give_blocks_in_names_order():
    spec = _spec(_WIDTHS)
    assert spec.node_widths == (2, 1, 3, 1, 2, 1)
    assert spec.unit_index(["y", "b", "a"]) == (7, 8, 2, 0, 1)
    assert spec.unit_index(spec.nodes) == tuple(range(spec.state_dim))


@_both_widths()
def test_units_match_the_attribution_labels(widths):
    spec = _spec(widths)
    scores = map_node_attributions(
        torch.zeros(
            1,
            spec.state_dim,
        ),
        spec,
    )
    labels = scores.coords["node"].values.tolist()
    for name in spec.nodes:
        labelled = tuple(
            unit for unit, label in enumerate(labels) if label == name
        )
        assert labelled == spec.unit_index([name])


@_both_widths()
def test_index_reads_the_node_units_slices(widths):
    spec = _spec(widths)
    state = torch.arange(
        2.0 * spec.state_dim,
    ).reshape(
        2,
        spec.state_dim,
    )
    names = ["y", "a", "u", "b"]
    index = torch.as_tensor(
        spec.unit_index(names),
        dtype=torch.int64,
    )
    expected = torch.cat(
        [state[..., spec.node_units(name)] for name in names],
        dim=-1,
    )
    torch.testing.assert_close(
        state[..., index],
        expected,
    )


def test_repeated_name_repeats_its_units():
    spec = _spec(_WIDTHS)
    a_units = spec.unit_index(["a"])
    y_units = spec.unit_index(["y"])
    assert a_units == (0, 1)
    assert spec.unit_index(["a", "y", "a"]) == a_units + y_units + a_units


def test_empty_names_give_an_empty_tuple():
    spec = _spec(_WIDTHS)
    assert spec.unit_index([]) == ()
    assert spec.unit_index(()) == ()
    assert spec.unit_index(np.array([], dtype=str)) == ()
    index = torch.as_tensor(
        spec.unit_index([]),
        dtype=torch.int64,
    )
    state = torch.zeros(
        2,
        spec.state_dim,
    )
    assert tuple(state[..., index].shape) == (2, 0)


def test_integer_names_match_after_str():
    spec = parse_adjacency(
        pd.DataFrame(
            {
                "source": [1, 2],
                "target": [2, 3],
            }
        )
    )
    assert spec.nodes == ("1", "2", "3")
    assert spec.unit_index([1, "3"]) == (0, 2)
    assert spec.unit_index(np.array([3, 1])) == (2, 0)


@_both_widths()
def test_items_are_python_int(widths):
    spec = _spec(widths)
    for names in (
        list(spec.nodes),
        np.array(spec.nodes),
    ):
        index = spec.unit_index(names)
        assert index
        assert all(type(unit) is int for unit in index)


@pytest.mark.parametrize(
    "make_names",
    [
        pytest.param(
            lambda: ["y", "a"],
            id="list",
        ),
        pytest.param(
            lambda: ("y", "a"),
            id="tuple",
        ),
        pytest.param(
            lambda: np.array(["y", "a"]),
            id="numpy",
        ),
        pytest.param(
            lambda: pd.Index(["y", "a"]),
            id="pandas_index",
        ),
        pytest.param(
            lambda: (name for name in ["y", "a"]),
            id="generator",
        ),
    ],
)
def test_accepted_containers(make_names):
    spec = _spec(_WIDTHS)
    assert spec.unit_index(make_names()) == (7, 8, 0, 1)


def test_caller_names_are_not_mutated():
    spec = _spec(_WIDTHS)
    names = ["y", "a", "a"]
    array = np.array(names)
    spec.unit_index(names)
    spec.unit_index(array)
    assert names == ["y", "a", "a"]
    np.testing.assert_array_equal(
        array,
        np.array(["y", "a", "a"]),
    )


@pytest.mark.parametrize(
    ("names", "message"),
    [
        pytest.param(
            "a",
            "'names' is a string; pass a sequence of node names.",
            id="str",
        ),
        pytest.param(
            b"a",
            "'names' is bytes; pass a sequence of node names.",
            id="bytes",
        ),
        pytest.param(
            torch.tensor([0]),
            "'names' is a tensor; pass a sequence of node names.",
            id="tensor",
        ),
        pytest.param(
            5,
            "Unsupported names type. Expected a sequence of node names. "
            "Got 5 (int).",
            id="int",
        ),
        pytest.param(
            {"a"},
            "'names' must be a sequence of node names, not a set: its "
            "items are matched by position. Got {'a'} (set). Pass a list "
            "or tuple in the intended order.",
            id="set",
        ),
        pytest.param(
            {"a": 0},
            "'names' must be a sequence of node names, not a mapping: its "
            "items are matched by position. Got {'a': 0} (dict). Pass a "
            "list or tuple in the intended order.",
            id="mapping",
        ),
        pytest.param(
            np.array([["a"]]),
            "'names' must be one-dimensional. Got ndarray of shape (1, 1), "
            "dtype <U1. A matrix is not a name list; pass the node names "
            "and index the matrix yourself.",
            id="matrix",
        ),
        pytest.param(
            ["a", ""],
            "Node name is empty.",
            id="empty_name",
        ),
        pytest.param(
            ["q", ""],
            "Node name is empty.",
            id="empty_name_before_unknown",
        ),
        pytest.param(
            ["q", "a", "Q", "q"],
            "Unknown node name(s): Q, q. Valid names are in spec.nodes.",
            id="unknown_names",
        ),
    ],
)
def test_rejection_messages_are_pinned(
    names,
    message,
):
    with pytest.raises(
        Kpnn2Error,
        match=f"^{re.escape(message)}$",
    ):
        _spec().unit_index(names)
