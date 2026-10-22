import re

import numpy as np
import pandas as pd
import pytest
import torch

from kpnn2 import Kpnn2Error, PackedLinear, parse_adjacency

# Nodes of a width-k block: two hidden nodes and an output node.
# Input nodes always have width 1.
_WIDTHS = {
    "a": 2,
    "b": 3,
    "y": 2,
}


def _edgelist():
    """
    Inputs u and x feed a cycle a <-> b, a has a self-loop,
    and the core feeds outputs y and z.
    """
    return pd.DataFrame(
        {
            "source": ["x", "u", "a", "b", "a", "a", "b"],
            "target": ["a", "b", "b", "a", "a", "y", "z"],
        }
    )


def _spec(widths=None):
    return parse_adjacency(
        _edgelist(),
        widths=widths,
        outputs=["y", "z"],
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
def test_every_edge_gives_every_slot_in_order(widths):
    spec = _spec(widths)
    table = spec.to_edgelist()
    assert spec.edge_index(
        table["source"],
        table["target"],
    ) == tuple(range(len(spec.source_index)))


def test_width_one_gives_rows_of_the_edgelist():
    spec = _spec()
    table = spec.to_edgelist()
    assert table.values.tolist() == [
        ["a", "a"],
        ["a", "b"],
        ["a", "y"],
        ["b", "a"],
        ["b", "z"],
        ["u", "b"],
        ["x", "a"],
    ]
    backwards = table.iloc[::-1]
    assert spec.edge_index(
        backwards["source"],
        backwards["target"],
    ) == (6, 5, 4, 3, 2, 1, 0)


@_both_widths()
def test_each_edge_gives_its_edge_location(widths):
    spec = _spec(widths)
    for source, target in spec.to_edgelist().itertuples(index=False):
        assert spec.edge_index(
            [source],
            [target],
        ) == spec.edge_location(
            source,
            target,
        )


@_both_widths()
def test_slots_join_the_units_of_the_named_nodes(widths):
    spec = _spec(widths)
    for source, target in spec.to_edgelist().itertuples(index=False):
        pairs = {
            (
                spec.source_index[slot],
                spec.target_index[slot],
            )
            for slot in spec.edge_index(
                [source],
                [target],
            )
        }
        assert pairs == {
            (
                source_unit,
                target_unit,
            )
            for source_unit in spec.unit_index([source])
            for target_unit in spec.unit_index([target])
        }


def test_wide_edges_give_blocks_in_edge_order():
    spec = _spec(_WIDTHS)
    assert spec.edge_location("x", "a") == (26, 27)
    assert spec.edge_location("a", "y") == (10, 11, 12, 13)
    assert spec.edge_index(
        ["x", "a"],
        ["a", "y"],
    ) == (26, 27, 10, 11, 12, 13)


@_both_widths()
def test_weight_on_the_slots_carries_signal_along_the_edge(widths):
    spec = _spec(widths)
    layer = PackedLinear.from_adjacency(
        spec,
        bias=False,
        generator=torch.Generator().manual_seed(42),
    )
    index = torch.as_tensor(
        spec.edge_index(
            ["a"],
            ["b"],
        ),
        dtype=torch.int64,
    )
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[index] = 1.0
    state = torch.zeros(
        1,
        spec.state_dim,
    )
    state[..., spec.node_units("a")] = 1.0
    out = layer(state)
    # Every unit of b sums the ones on every unit of a, and only b
    # receives anything: the self-loop a -> a and a -> y stay at 0.
    expected = torch.zeros(
        1,
        spec.state_dim,
    )
    expected[..., spec.node_units("b")] = float(len(spec.unit_index(["a"])))
    torch.testing.assert_close(
        out,
        expected,
    )


def test_values_on_a_caller_table_land_on_their_slots():
    spec = _spec()
    signs = pd.DataFrame(
        {
            "source": ["b", "x", "a"],
            "target": ["a", "a", "b"],
            "sign": [-1.0, 1.0, 1.0],
        }
    )
    index = torch.as_tensor(
        spec.edge_index(
            signs["source"],
            signs["target"],
        ),
        dtype=torch.int64,
    )
    sign = torch.zeros(len(spec.source_index))
    sign[index] = torch.tensor(signs["sign"].tolist())
    for source, target, value in signs.itertuples(index=False):
        (slot,) = spec.edge_location(
            source,
            target,
        )
        assert sign[slot].item() == value
    assert int(sign.ne(0).sum()) == len(signs)


def test_repeated_edge_repeats_its_slots():
    spec = _spec(_WIDTHS)
    ab = spec.edge_index(
        ["a"],
        ["b"],
    )
    xa = spec.edge_index(
        ["x"],
        ["a"],
    )
    assert len(ab) == 6
    assert (
        spec.edge_index(
            ["a", "x", "a"],
            ["b", "a", "b"],
        )
        == ab + xa + ab
    )


def test_self_loop_gives_its_block():
    spec = _spec(_WIDTHS)
    slots = spec.edge_index(
        ["a"],
        ["a"],
    )
    assert slots == spec.edge_location(
        "a",
        "a",
    )
    assert len(slots) == 4


def test_no_edges_give_an_empty_tuple():
    spec = _spec(_WIDTHS)
    for sources, targets in (
        (
            [],
            [],
        ),
        (
            (),
            (),
        ),
        (
            np.array(
                [],
                dtype=str,
            ),
            pd.Series(
                [],
                dtype=str,
            ),
        ),
    ):
        assert (
            spec.edge_index(
                sources,
                targets,
            )
            == ()
        )
    index = torch.as_tensor(
        spec.edge_index(
            [],
            [],
        ),
        dtype=torch.int64,
    )
    weight = torch.zeros(len(spec.source_index))
    assert tuple(weight[index].shape) == (0,)


def test_integer_names_match_after_str():
    spec = parse_adjacency(
        pd.DataFrame(
            {
                "source": [1, 2],
                "target": [2, 3],
            }
        )
    )
    assert spec.edge_index(
        [2, "1"],
        ["3", 2],
    ) == (1, 0)
    assert spec.edge_index(
        np.array([2, 1]),
        np.array([3, 2]),
    ) == (1, 0)


@_both_widths()
def test_items_are_python_int(widths):
    spec = _spec(widths)
    table = spec.to_edgelist()
    for sources, targets in (
        (
            list(table["source"]),
            list(table["target"]),
        ),
        (
            table["source"].to_numpy(),
            table["target"].to_numpy(),
        ),
    ):
        index = spec.edge_index(
            sources,
            targets,
        )
        assert index
        assert all(type(slot) is int for slot in index)


@pytest.mark.parametrize(
    "make_names",
    [
        pytest.param(
            lambda names: list(names),
            id="list",
        ),
        pytest.param(
            lambda names: tuple(names),
            id="tuple",
        ),
        pytest.param(
            lambda names: np.array(names),
            id="numpy",
        ),
        pytest.param(
            lambda names: pd.Series(names),
            id="pandas_series",
        ),
        pytest.param(
            lambda names: pd.Index(names),
            id="pandas_index",
        ),
        pytest.param(
            lambda names: (name for name in names),
            id="generator",
        ),
    ],
)
def test_accepted_containers(make_names):
    spec = _spec(_WIDTHS)
    assert spec.edge_index(
        make_names(["x", "a"]),
        make_names(["a", "y"]),
    ) == (26, 27, 10, 11, 12, 13)


def test_caller_names_are_not_mutated():
    spec = _spec(_WIDTHS)
    sources = ["b", "a", "a"]
    targets = pd.Series(["a", "b", "b"])
    spec.edge_index(
        sources,
        targets,
    )
    assert sources == ["b", "a", "a"]
    pd.testing.assert_series_equal(
        targets,
        pd.Series(["a", "b", "b"]),
    )


@pytest.mark.parametrize(
    ("sources", "targets", "message"),
    [
        pytest.param(
            "a",
            ["b"],
            "'sources' must be a sequence of source node names, not a str.\n"
            "  Got: 'a' (str)\n"
            "  Fix: Wrap a single name in a list.",
            id="str",
        ),
        pytest.param(
            ["a"],
            b"b",
            "'targets' must be a sequence of target node names, not bytes.\n"
            "  Got: b'b' (bytes)",
            id="bytes",
        ),
        pytest.param(
            torch.tensor([0]),
            ["b"],
            "'sources' must be a sequence of source node names, not a "
            "tensor.\n"
            "  Got: Tensor of shape (1,), dtype torch.int64",
            id="tensor",
        ),
        pytest.param(
            ["a"],
            5,
            "'targets' must be a sequence of target node names.\n"
            "  Got: 5 (int)",
            id="int",
        ),
        pytest.param(
            {"a"},
            ["b"],
            "'sources' must be a sequence of source node names, not a set: "
            "its items are matched by position.\n"
            "  Got: {'a'} (set)\n"
            "  Fix: Pass a list or tuple in the intended order.",
            id="set",
        ),
        pytest.param(
            ["a"],
            {"a": "b"},
            "'targets' must be a sequence of target node names, not a "
            "mapping: its items are matched by position.\n"
            "  Got: {'a': 'b'} (dict)\n"
            "  Fix: Pass a list or tuple in the intended order.",
            id="mapping",
        ),
        pytest.param(
            np.array([["a"]]),
            ["b"],
            "'sources' must be one-dimensional: a matrix is not a name list.\n"
            "  Got: ndarray of shape (1, 1), dtype <U1\n"
            "  Fix: Pass the source node names, and index the matrix yourself.",
            id="matrix",
        ),
        pytest.param(
            ["a", "b"],
            ["b"],
            "'sources' and 'targets' must have the same length: edge i is "
            "sources[i] -> targets[i]. Got 2 source(s) and 1 target(s).",
            id="lengths_differ",
        ),
        pytest.param(
            ["a", ""],
            ["b", "a"],
            "Node name is empty.",
            id="empty_source",
        ),
        pytest.param(
            ["q", "a"],
            ["b", ""],
            "Node name is empty.",
            id="empty_target_before_unknown",
        ),
        pytest.param(
            ["q", "a", "Q"],
            ["b", "q", "a"],
            "Unknown node name(s): Q, q. Valid names are in spec.nodes.",
            id="unknown_names",
        ),
        pytest.param(
            ["y", "a", "x", "y"],
            ["a", "b", "b", "a"],
            "No edge(s): x -> b, y -> a. Valid edges are the rows of "
            "spec.to_edgelist().",
            id="absent_edges",
        ),
        pytest.param(
            ["b"],
            ["b"],
            "No edge(s): b -> b. Valid edges are the rows of "
            "spec.to_edgelist().",
            id="absent_self_loop",
        ),
    ],
)
def test_rejection_messages_are_pinned(
    sources,
    targets,
    message,
):
    with pytest.raises(
        Kpnn2Error,
        match=f"^{re.escape(message)}$",
    ):
        _spec().edge_index(
            sources,
            targets,
        )
