import pandas as pd
import pytest
import torch

from kpnn2 import PackedLinear, parse_adjacency, parse_layered

# Wide hidden and output nodes. Input nodes always have width 1.
_ADJACENCY_WIDTHS = {
    "a": 2,
    "b": 3,
    "y": 2,
}
_LAYERED_WIDTHS = {
    "a": 2,
    "b": 3,
    "c": 2,
    "y": 2,
}


def _adjacency_edgelist():
    """
    Inputs u and x feed a feedback core a <-> b, a has a self-loop,
    and the core feeds outputs y and z.
    """
    return pd.DataFrame(
        {
            "source": ["x", "u", "a", "b", "a", "a", "b"],
            "target": ["a", "b", "b", "a", "a", "y", "z"],
        }
    )


def _layered_edgelist():
    """
    Inputs u and x feed a and b, which feed c, which feeds output y.
    Output z hangs off a, and the skip x -> c spans two layers.
    """
    return pd.DataFrame(
        {
            "source": ["u", "x", "x", "a", "b", "c", "a", "x"],
            "target": ["a", "a", "b", "c", "c", "y", "z", "c"],
        }
    )


def _adjacency(widths=None):
    return parse_adjacency(
        _adjacency_edgelist(),
        widths=widths,
    )


def _layered(widths=None):
    return parse_layered(
        _layered_edgelist(),
        widths=widths,
    )


def _widths(wide):
    return pytest.mark.parametrize(
        "widths",
        [
            pytest.param(
                None,
                id="width_1",
            ),
            pytest.param(
                wide,
                id="wide",
            ),
        ],
    )


def _generator():
    return torch.Generator().manual_seed(42)


@_widths(_ADJACENCY_WIDTHS)
def test_adjacency_has_one_row_per_slot(widths):
    spec = _adjacency(widths)
    table = spec.slot_table()
    assert list(table.columns) == [
        "source",
        "target",
    ]
    assert table.index.name == "slot"
    assert table.index.dtype == "int64"
    assert table.index.tolist() == list(range(len(spec.source_index)))


def test_adjacency_width_one_rows_are_the_edgelist():
    spec = _adjacency()
    pd.testing.assert_frame_equal(
        spec.slot_table().reset_index(drop=True),
        spec.to_edgelist(),
    )


@_widths(_ADJACENCY_WIDTHS)
def test_adjacency_rows_name_the_nodes_that_own_the_units(widths):
    spec = _adjacency(widths)
    for slot, source, target in spec.slot_table().itertuples():
        assert spec.source_index[slot] in spec.unit_index([source])
        assert spec.target_index[slot] in spec.unit_index([target])


@_widths(_ADJACENCY_WIDTHS)
def test_adjacency_rows_of_an_edge_are_its_edge_index(widths):
    spec = _adjacency(widths)
    table = spec.slot_table()
    widths_of = spec.node_table()["width"]
    for (source, target), rows in table.groupby(
        [
            "source",
            "target",
        ]
    ):
        assert tuple(rows.index) == spec.edge_index(
            [source],
            [target],
        )
        assert len(rows) == widths_of[source] * widths_of[target]


@_widths(_ADJACENCY_WIDTHS)
def test_adjacency_weight_of_a_row_joins_its_source_to_its_target(
    widths,
):
    spec = _adjacency(widths)
    layer = PackedLinear.from_adjacency(
        spec,
        bias=False,
        generator=_generator(),
    )
    table = spec.slot_table()
    for slot, source, target in table.itertuples():
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[slot] = 1.0
        state = torch.zeros(
            1,
            spec.state_dim,
        )
        state[..., spec.node_units(source)] = 1.0
        with torch.no_grad():
            out = layer(state)
        # One unit of target receives 1; every other unit gets 0.
        assert float(out.sum()) == pytest.approx(1.0)
        assert float(out[..., spec.node_units(target)].sum()) == (
            pytest.approx(1.0)
        )


def test_adjacency_assigned_weights_are_named_by_slot():
    spec = _adjacency(_ADJACENCY_WIDTHS)
    layer = PackedLinear.from_adjacency(
        spec,
        generator=_generator(),
    )
    weight = layer.effective_weight().detach()
    table = spec.slot_table()
    table["weight"] = weight.numpy()
    for source, target in spec.to_edgelist().itertuples(index=False):
        named = table.loc[
            (table["source"] == source) & (table["target"] == target),
            "weight",
        ]
        slots = torch.as_tensor(
            spec.edge_location(
                source,
                target,
            ),
            dtype=torch.int64,
        )
        torch.testing.assert_close(
            torch.as_tensor(named.to_numpy()),
            weight[slots],
        )


def test_adjacency_self_loop_has_its_rows():
    spec = _adjacency(_ADJACENCY_WIDTHS)
    table = spec.slot_table()
    loop = table[(table["source"] == "a") & (table["target"] == "a")]
    assert tuple(loop.index) == spec.edge_location(
        "a",
        "a",
    )


@_widths(_LAYERED_WIDTHS)
def test_layered_has_one_row_per_slot_of_every_hop(widths):
    spec = _layered(widths)
    table = spec.slot_table()
    assert list(table.columns) == [
        "source",
        "target",
    ]
    assert list(table.index.names) == [
        "hop",
        "slot",
    ]
    assert all(dtype == "int64" for dtype in table.index.dtypes)
    assert table.index.tolist() == [
        (
            hop_index,
            slot,
        )
        for hop_index, hop in enumerate(spec.hops)
        for slot in range(len(hop.source_index))
    ]


def test_layered_width_one_rows_of_a_hop_are_its_edgelist_rows():
    spec = _layered()
    table = spec.slot_table()
    edgelist = spec.to_edgelist()
    hop_of = [
        spec.edge_location(
            source,
            target,
        )[0]
        for source, target in edgelist.itertuples(index=False)
    ]
    for hop_index in range(len(spec.hops)):
        pd.testing.assert_frame_equal(
            table.loc[hop_index].reset_index(drop=True),
            edgelist[[hop == hop_index for hop in hop_of]].reset_index(
                drop=True
            ),
        )


@_widths(_LAYERED_WIDTHS)
def test_layered_rows_name_the_nodes_that_own_the_units(widths):
    spec = _layered(widths)
    for (hop_index, slot), source, target in spec.slot_table().itertuples():
        hop = spec.hops[hop_index]
        source_units = spec.hop_units(
            hop,
            source,
        )
        assert source_units.start <= hop.source_index[slot]
        assert hop.source_index[slot] < source_units.stop
        assert hop.target_index[slot] in spec.unit_index(
            [target],
            layer=hop.target_layer,
        )


@_widths(_LAYERED_WIDTHS)
def test_layered_rows_of_an_edge_are_its_edge_location(widths):
    spec = _layered(widths)
    table = spec.slot_table().reset_index()
    for (source, target), rows in table.groupby(
        [
            "source",
            "target",
        ]
    ):
        hop_index, slots = spec.edge_location(
            source,
            target,
        )
        assert set(rows["hop"]) == {hop_index}
        assert tuple(rows["slot"]) == slots
        assert tuple(rows["slot"]) == spec.edge_index(
            [source],
            [target],
            hop=spec.hops[hop_index],
        )


@_widths(_LAYERED_WIDTHS)
def test_layered_concatenated_hop_weights_are_named_by_slot(widths):
    spec = _layered(widths)
    layers = [
        PackedLinear.from_hop(
            hop,
            generator=_generator(),
        )
        for hop in spec.hops
    ]
    table = spec.slot_table()
    table["weight"] = (
        torch.cat([layer.effective_weight() for layer in layers])
        .detach()
        .numpy()
    )
    for source, target in spec.to_edgelist().itertuples(index=False):
        hop_index, slots = spec.edge_location(
            source,
            target,
        )
        named = table.loc[hop_index]
        named = named[(named["source"] == source) & (named["target"] == target)]
        torch.testing.assert_close(
            torch.as_tensor(named["weight"].to_numpy()),
            layers[hop_index].effective_weight().detach()[list(slots)],
        )


@_widths(_LAYERED_WIDTHS)
def test_layered_weight_of_a_row_joins_its_source_to_its_target(widths):
    spec = _layered(widths)
    for (hop_index, slot), source, target in spec.slot_table().itertuples():
        hop = spec.hops[hop_index]
        layer = PackedLinear.from_hop(
            hop,
            bias=False,
            generator=_generator(),
        )
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[slot] = 1.0
        sources = torch.zeros(
            1,
            hop.in_features,
        )
        sources[
            ...,
            spec.hop_units(
                hop,
                source,
            ),
        ] = 1.0
        with torch.no_grad():
            out = layer(sources)
        assert float(out.sum()) == pytest.approx(1.0)
        assert float(out[..., spec.node_units(target)[1]].sum()) == (
            pytest.approx(1.0)
        )


def test_layered_ranks_keep_rows_on_their_hops():
    spec = parse_layered(
        _layered_edgelist(),
        ranks={
            "u": 0,
            "x": 0,
            "a": 1,
            "b": 1,
            "c": 2,
            "y": 3,
            "z": 3,
        },
    )
    table = spec.slot_table()
    assert table.loc[2].values.tolist() == [
        ["a", "z"],
        ["c", "y"],
    ]


@pytest.mark.parametrize(
    "make_spec",
    [
        pytest.param(
            lambda: _adjacency(_ADJACENCY_WIDTHS),
            id="adjacency",
        ),
        pytest.param(
            lambda: _layered(_LAYERED_WIDTHS),
            id="layered",
        ),
    ],
)
def test_each_call_returns_a_new_table(make_spec):
    spec = make_spec()
    first = spec.slot_table()
    expected = first.copy()
    first["source"] = "changed"
    first["weight"] = 0.0
    pd.testing.assert_frame_equal(
        spec.slot_table(),
        expected,
    )
