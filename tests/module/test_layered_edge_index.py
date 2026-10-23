import re

import numpy as np
import pandas as pd
import pytest
import torch

from kpnn2 import Kpnn2Error, PackedLinear, parse_layered

# Wide hidden and output nodes. Input nodes always have width 1.
_WIDTHS = {
    "a": 2,
    "b": 3,
    "c": 2,
    "y": 2,
}


def _edgelist():
    """
    Inputs u and x feed a and b, which feed c, which feeds output y.
    Output z hangs off a, and the skip x -> c spans two layers.

    Layers are (u, x), (a, b), (c, z), (y,), so hops[1] carries the
    adjacent edges a -> c, b -> c, a -> z and the skip x -> c.
    """
    return pd.DataFrame(
        {
            "source": ["u", "x", "x", "a", "b", "c", "a", "x"],
            "target": ["a", "a", "b", "c", "c", "y", "z", "c"],
        }
    )


def _spec(widths=None):
    return parse_layered(
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


def _hop_edges(spec, hop_index):
    """
    Rows of ``to_edgelist()`` whose edge is in ``hops[hop_index]``.
    """
    table = spec.to_edgelist()
    keep = [
        spec.edge_location(
            source,
            target,
        )[0]
        == hop_index
        for source, target in table.itertuples(index=False)
    ]
    return table[keep]


def test_layers_and_hops_of_the_fixture():
    spec = _spec()
    assert spec.layer_nodes == (
        ("u", "x"),
        ("a", "b"),
        ("c", "z"),
        ("y",),
    )
    assert _hop_edges(spec, 1).values.tolist() == [
        ["a", "c"],
        ["a", "z"],
        ["b", "c"],
        ["x", "c"],
    ]


@_both_widths()
def test_every_edge_of_a_hop_gives_every_slot_in_order(widths):
    spec = _spec(widths)
    for hop_index, hop in enumerate(spec.hops):
        edges = _hop_edges(
            spec,
            hop_index,
        )
        assert spec.edge_index(
            edges["source"],
            edges["target"],
            hop=hop,
        ) == tuple(range(len(hop.source_index)))


@_both_widths()
def test_each_edge_gives_its_edge_location(widths):
    spec = _spec(widths)
    for source, target in spec.to_edgelist().itertuples(index=False):
        hop_index, slots = spec.edge_location(
            source,
            target,
        )
        assert (
            spec.edge_index(
                [source],
                [target],
                hop=spec.hops[hop_index],
            )
            == slots
        )


@_both_widths()
def test_slots_join_the_units_of_the_named_nodes(widths):
    spec = _spec(widths)
    for source, target in spec.to_edgelist().itertuples(index=False):
        hop = spec.hops[
            spec.edge_location(
                source,
                target,
            )[0]
        ]
        source_units = spec.hop_units(
            hop,
            source,
        )
        pairs = {
            (
                hop.source_index[slot],
                hop.target_index[slot],
            )
            for slot in spec.edge_index(
                [source],
                [target],
                hop=hop,
            )
        }
        assert pairs == {
            (
                source_unit,
                target_unit,
            )
            for source_unit in range(
                source_units.start,
                source_units.stop,
            )
            for target_unit in spec.unit_index(
                [target],
                layer=hop.target_layer,
            )
        }


def test_wide_edges_give_blocks_in_edge_order():
    spec = _spec(_WIDTHS)
    hop = spec.hops[1]
    assert spec.edge_location("x", "c") == (1, (12, 13))
    assert spec.edge_location("b", "c") == (1, (6, 7, 8, 9, 10, 11))
    assert spec.edge_index(
        ["x", "b"],
        ["c", "c"],
        hop=hop,
    ) == (12, 13, 6, 7, 8, 9, 10, 11)


@_both_widths()
def test_weight_on_the_slots_carries_signal_along_a_skip(widths):
    spec = _spec(widths)
    hop = spec.hops[1]
    layer = PackedLinear.from_hop(
        hop,
        bias=False,
        generator=torch.Generator().manual_seed(42),
    )
    index = torch.as_tensor(
        spec.edge_index(
            ["x"],
            ["c"],
            hop=hop,
        ),
        dtype=torch.int64,
    )
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[index] = 1.0
    sources = torch.ones(
        1,
        hop.in_features,
    )
    out = layer(sources)
    # Only c reads x through these slots; every other edge into
    # layer 2 has weight 0, so z gets nothing.
    expected = torch.zeros(
        1,
        hop.out_features,
    )
    expected[..., spec.node_units("c")[1]] = 1.0
    torch.testing.assert_close(
        out,
        expected,
    )


@_both_widths()
def test_values_on_a_caller_table_land_on_their_slots(widths):
    spec = _spec(widths)
    signs = _edgelist().assign(
        sign=[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    )
    per_hop = []
    for hop_index, hop in enumerate(spec.hops):
        rows = signs[
            [
                spec.edge_location(
                    source,
                    target,
                )[0]
                == hop_index
                for source, target in zip(
                    signs["source"],
                    signs["target"],
                )
            ]
        ]
        index = torch.as_tensor(
            spec.edge_index(
                rows["source"],
                rows["target"],
                hop=hop,
            ),
            dtype=torch.int64,
        )
        # One value per edge, repeated across each edge's slots.
        counts = torch.as_tensor(
            [
                len(
                    spec.edge_location(
                        source,
                        target,
                    )[1]
                )
                for source, target in zip(
                    rows["source"],
                    rows["target"],
                )
            ]
        )
        sign = torch.zeros(len(hop.source_index))
        sign[index] = torch.repeat_interleave(
            torch.tensor(rows["sign"].tolist()),
            counts,
        )
        per_hop.append(sign)
    for hop_index, hop in enumerate(spec.hops):
        for source, target, value in signs.itertuples(index=False):
            located_hop, slots = spec.edge_location(
                source,
                target,
            )
            if located_hop == hop_index:
                assert per_hop[hop_index][list(slots)].eq(value).all()
        assert bool(per_hop[hop_index].ne(0).all())


def test_one_value_per_edge_at_width_one():
    spec = _spec()
    hop = spec.hops[1]
    signs = pd.DataFrame(
        {
            "source": ["x", "b", "a"],
            "target": ["c", "c", "z"],
            "sign": [-1.0, 1.0, -1.0],
        }
    )
    index = torch.as_tensor(
        spec.edge_index(
            signs["source"],
            signs["target"],
            hop=hop,
        ),
        dtype=torch.int64,
    )
    sign = torch.zeros(len(hop.source_index))
    sign[index] = torch.tensor(signs["sign"].tolist())
    assert sign.tolist() == [0.0, -1.0, 1.0, -1.0]


def test_repeated_edge_repeats_its_slots():
    spec = _spec(_WIDTHS)
    hop = spec.hops[1]
    bc = spec.edge_index(
        ["b"],
        ["c"],
        hop=hop,
    )
    xc = spec.edge_index(
        ["x"],
        ["c"],
        hop=hop,
    )
    assert (
        spec.edge_index(
            ["b", "x", "b"],
            ["c", "c", "c"],
            hop=hop,
        )
        == bc + xc + bc
    )


def test_no_edges_give_an_empty_tuple():
    spec = _spec(_WIDTHS)
    for hop in spec.hops:
        assert (
            spec.edge_index(
                [],
                [],
                hop=hop,
            )
            == ()
        )
    index = torch.as_tensor(
        spec.edge_index(
            (),
            (),
            hop=spec.hops[0],
        ),
        dtype=torch.int64,
    )
    weight = torch.zeros(len(spec.hops[0].source_index))
    assert tuple(weight[index].shape) == (0,)


def test_integer_names_match_after_str():
    spec = parse_layered(
        pd.DataFrame(
            {
                "source": [1, 1, 2],
                "target": [2, 3, 3],
            }
        )
    )
    assert spec.edge_index(
        [2, "1"],
        ["3", 3],
        hop=spec.hops[1],
    ) == (1, 0)
    assert spec.edge_index(
        np.array([1]),
        np.array([2]),
        hop=spec.hops[0],
    ) == (0,)


@_both_widths()
def test_items_are_python_int(widths):
    spec = _spec(widths)
    for hop_index, hop in enumerate(spec.hops):
        edges = _hop_edges(
            spec,
            hop_index,
        )
        for sources, targets in (
            (
                list(edges["source"]),
                list(edges["target"]),
            ),
            (
                edges["source"].to_numpy(),
                edges["target"].to_numpy(),
            ),
        ):
            index = spec.edge_index(
                sources,
                targets,
                hop=hop,
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
        make_names(["x", "b"]),
        make_names(["c", "c"]),
        hop=spec.hops[1],
    ) == (12, 13, 6, 7, 8, 9, 10, 11)


def test_caller_names_are_not_mutated():
    spec = _spec(_WIDTHS)
    sources = ["x", "b", "b"]
    targets = pd.Series(["c", "c", "c"])
    spec.edge_index(
        sources,
        targets,
        hop=spec.hops[1],
    )
    assert sources == ["x", "b", "b"]
    pd.testing.assert_series_equal(
        targets,
        pd.Series(["c", "c", "c"]),
    )


def test_hop_is_keyword_only_and_required():
    spec = _spec()
    with pytest.raises(TypeError):
        spec.edge_index(
            ["x"],
            ["c"],
            spec.hops[1],
        )
    with pytest.raises(TypeError):
        spec.edge_index(
            ["x"],
            ["c"],
        )


_OTHER_HOP = parse_layered(
    pd.DataFrame(
        {
            "source": ["p"],
            "target": ["q"],
        }
    )
).hops[0]


@pytest.mark.parametrize(
    ("sources", "targets", "hop_index", "message"),
    [
        pytest.param(
            "x",
            ["c"],
            1,
            "'sources' must be a sequence of source node names, not a single "
            "string.\n"
            "  Got: 'x' (str)\n"
            "  Fix: Wrap it in a list.",
            id="str",
        ),
        pytest.param(
            ["x"],
            torch.tensor([0]),
            1,
            "'targets' must be a sequence of target node names, not a "
            "tensor.\n"
            "  Got: Tensor of shape (1,), dtype torch.int64",
            id="tensor",
        ),
        pytest.param(
            {"x"},
            ["c"],
            1,
            "'sources' must be a sequence of source node names, not a set: "
            "its items are matched by position.\n"
            "  Got: {'x'} (set)\n"
            "  Fix: Pass a list or tuple in the intended order.",
            id="set",
        ),
        pytest.param(
            ["x"],
            np.array([["c"]]),
            1,
            "'targets' must be a 1-dimensional sequence of target node "
            "names.\n"
            "  Got: ndarray of shape (1, 1), dtype <U1",
            id="matrix",
        ),
        pytest.param(
            ["x"],
            ["c"],
            1.0,
            "'hop' must be a Hop from spec.hops.\n  Got: 1.0 (float)",
            id="hop_not_a_hop",
        ),
        pytest.param(
            ["x"],
            ["c"],
            "other",
            "'hop' must match an entry of spec.hops: a Hop from another spec "
            "does not match.\n"
            "  Got: a Hop into layer 1 that differs from spec.hops[0]\n"
            "  Fix: Take it from this spec's hops.",
            id="hop_from_another_spec",
        ),
        pytest.param(
            ["x", "b"],
            ["c"],
            1,
            "'sources' and 'targets' must have the same length: edge i is "
            "sources[i] -> targets[i].\n"
            "  Got: 2 source(s) and 1 target(s)",
            id="lengths_differ",
        ),
        pytest.param(
            ["x", "q"],
            ["", "c"],
            1,
            "'sources' or 'targets' contains empty node names.\n"
            "  Got: 'targets': 1 name(s) at position 0",
            id="empty_name_before_unknown",
        ),
        pytest.param(
            ["q", "x", "Q"],
            ["c", "p", "c"],
            1,
            "Unknown node name(s).\n"
            "  Got: 'Q', 'p', 'q'\n"
            "  Expected: names in spec.layer_nodes",
            id="unknown_names",
        ),
        pytest.param(
            ["c", "u", "c", "y"],
            ["a", "b", "a", "a"],
            1,
            "No edge(s) for some source-target pairs.\n"
            "  Got: c -> a, u -> b, y -> a\n"
            "  Expected: rows of spec.to_edgelist()\n"
            "  Fix: Swap source and target for c -> a: the reverse is an edge.",
            id="absent_edges_before_other_hop",
        ),
        pytest.param(
            ["c", "x", "u", "c"],
            ["y", "c", "a", "y"],
            1,
            "Edge(s) not in spec.hops[1].\n"
            "  Got: c -> y (spec.hops[2]), u -> a (spec.hops[0])\n"
            "  Fix: Call edge_index once per hop; edge_location gives the hop "
            "of an edge.",
            id="edges_in_other_hops",
        ),
    ],
)
def test_rejection_messages_are_pinned(
    sources,
    targets,
    hop_index,
    message,
):
    spec = _spec()
    if hop_index == "other":
        hop = _OTHER_HOP
    elif isinstance(hop_index, int):
        hop = spec.hops[hop_index]
    else:
        hop = hop_index
    with pytest.raises(
        Kpnn2Error,
        match=f"^{re.escape(message)}$",
    ):
        spec.edge_index(
            sources,
            targets,
            hop=hop,
        )
