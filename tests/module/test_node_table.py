import random

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import kpnn2

_COUNTS = [
    "in_degree",
    "out_degree",
    "n_ancestors",
    "n_descendants",
    "n_input_ancestors",
]


def _layered_edgelist():
    """
    Inputs x1-x3 feed h1 and h2, which feed y; x3 also feeds y.
    """
    return pd.DataFrame(
        {
            "source": ["x1", "x2", "x2", "x3", "h1", "h2", "x3"],
            "target": ["h1", "h1", "h2", "h2", "y", "y", "y"],
        }
    )


def _cyclic_edgelist():
    """
    Input x feeds a <-> b, b has a self-loop, and a feeds output y.
    """
    return pd.DataFrame(
        {
            "source": ["x", "a", "b", "a", "b"],
            "target": ["a", "b", "a", "y", "b"],
        }
    )


def _expected(
    nodes,
    columns,
):
    return pd.DataFrame(
        columns,
        index=pd.Index(
            nodes,
            name="node",
        ),
        dtype="int64",
    )


def _random_dag():
    """
    Random DAG over a shuffled order; edges point forward.
    """
    n_nodes = random.randint(2, 12)
    order = [f"n{index}" for index in range(n_nodes)]
    random.shuffle(order)
    edges = {(order[0], order[1])}
    for low in range(n_nodes):
        for high in range(low + 1, n_nodes):
            if random.random() < 0.3:
                edges.add((order[low], order[high]))
    return sorted(edges)


def _random_cyclic():
    """
    Random digraph with cycles and self-loops, plus an input and an
    output so parse_adjacency accepts it.
    """
    n_core = random.randint(1, 10)
    core = [f"c{index}" for index in range(n_core)]
    edges = set()
    for source in core:
        for target in core:
            if random.random() < 0.2:
                edges.add((source, target))
    for index in range(random.randint(1, 3)):
        edges.add((f"in{index}", random.choice(core)))
    for index in range(random.randint(1, 3)):
        edges.add((random.choice(core), f"out{index}"))
    return sorted(edges)


def _random_widths(edges):
    targets = sorted({target for _, target in edges})
    return {name: random.randint(1, 3) for name in targets}


def _networkx_counts(
    edges,
    nodes,
):
    graph = nx.DiGraph(edges)
    inputs = {name for name in graph if graph.in_degree(name) == 0}
    return _expected(
        nodes,
        {
            "in_degree": [graph.in_degree(name) for name in nodes],
            "out_degree": [graph.out_degree(name) for name in nodes],
            "n_ancestors": [len(nx.ancestors(graph, name)) for name in nodes],
            "n_descendants": [
                len(nx.descendants(graph, name)) for name in nodes
            ],
            "n_input_ancestors": [
                len(nx.ancestors(graph, name) & inputs) for name in nodes
            ],
        },
    )


def test_layered_table_matches_hand_counts():
    table = kpnn2.parse_layered(_layered_edgelist()).node_table()
    expected = _expected(
        ["x1", "x2", "x3", "h1", "h2", "y"],
        {
            "layer": [0, 0, 0, 1, 1, 2],
            "width": [1, 1, 1, 1, 1, 1],
            "in_degree": [0, 0, 0, 2, 2, 3],
            "out_degree": [1, 2, 2, 1, 1, 0],
            "n_ancestors": [0, 0, 0, 2, 2, 5],
            "n_descendants": [2, 3, 2, 1, 1, 0],
            "n_input_ancestors": [0, 0, 0, 2, 2, 3],
        },
    )
    pd.testing.assert_frame_equal(
        table,
        expected,
    )


def test_adjacency_table_matches_hand_counts():
    table = kpnn2.parse_adjacency(_cyclic_edgelist()).node_table()
    # b's self-loop adds 1 to both its degrees, but b is not its
    # own ancestor or descendant; a and b, one cycle, are both.
    expected = _expected(
        ["a", "b", "x", "y"],
        {
            "width": [1, 1, 1, 1],
            "in_degree": [2, 2, 0, 1],
            "out_degree": [2, 2, 1, 0],
            "n_ancestors": [2, 2, 0, 3],
            "n_descendants": [2, 2, 3, 0],
            "n_input_ancestors": [1, 1, 0, 1],
        },
    )
    pd.testing.assert_frame_equal(
        table,
        expected,
    )


def test_layered_counts_match_networkx_on_random_dags():
    random.seed(42)
    for _ in range(100):
        edges = _random_dag()
        spec = kpnn2.parse_layered(
            pd.DataFrame(
                edges,
                columns=["source", "target"],
            ),
            widths=_random_widths(edges),
        )
        table = spec.node_table()
        pd.testing.assert_frame_equal(
            table[_COUNTS],
            _networkx_counts(
                edges,
                list(table.index),
            ),
        )


def test_adjacency_counts_match_networkx_on_random_cyclic_graphs():
    random.seed(42)
    n_cyclic = 0
    for _ in range(200):
        edges = _random_cyclic()
        spec = kpnn2.parse_adjacency(
            pd.DataFrame(
                edges,
                columns=["source", "target"],
            ),
            widths=_random_widths(edges),
        )
        table = spec.node_table()
        pd.testing.assert_frame_equal(
            table[_COUNTS],
            _networkx_counts(
                edges,
                list(spec.nodes),
            ),
        )
        n_cyclic += not nx.is_directed_acyclic_graph(nx.DiGraph(edges))
    # The generator must exercise cycles, not only DAGs.
    assert n_cyclic > 100


@pytest.mark.parametrize(
    "make_spec",
    [
        pytest.param(
            lambda: kpnn2.parse_layered(_layered_edgelist()),
            id="layered",
        ),
        pytest.param(
            lambda: kpnn2.parse_adjacency(_cyclic_edgelist()),
            id="adjacency_self_loop",
        ),
        pytest.param(
            lambda: kpnn2.parse_adjacency(
                pd.DataFrame(
                    {
                        "source": ["x", "a", "a"],
                        "target": ["a", "a", "y"],
                    }
                )
            ),
            id="adjacency_self_loop_before_output",
        ),
    ],
)
def test_zero_degrees_select_input_and_output_nodes(make_spec):
    spec = make_spec()
    table = spec.node_table()
    assert tuple(table.index[table["in_degree"] == 0]) == spec.input_nodes
    assert tuple(table.index[table["out_degree"] == 0]) == spec.output_nodes


def test_layered_rows_follow_unit_order():
    spec = kpnn2.parse_layered(
        _layered_edgelist(),
        widths={"h1": 3, "y": 2},
    )
    table = spec.node_table()
    assert list(table.index) == [
        name for names in spec.layer_nodes for name in names
    ]
    assert table["width"].tolist() == [
        width for widths in spec.layer_widths for width in widths
    ]
    per_unit = np.repeat(
        table["in_degree"],
        table["width"],
    )
    assert len(per_unit) == sum(spec.layer_dims)
    for layer, names in enumerate(spec.layer_nodes):
        rows = table[table["layer"] == layer]
        assert tuple(rows.index) == names
        layer_units = np.repeat(
            rows["in_degree"].to_numpy(),
            rows["width"].to_numpy(),
        )
        assert len(layer_units) == spec.layer_dims[layer]
        for name in names:
            node_layer, units = spec.node_units(name)
            assert node_layer == layer
            assert set(layer_units[units]) == {table.loc[name, "in_degree"]}


def test_adjacency_rows_follow_state_order():
    spec = kpnn2.parse_adjacency(
        _cyclic_edgelist(),
        widths={"a": 2, "y": 3},
    )
    table = spec.node_table()
    assert tuple(table.index) == spec.nodes
    assert tuple(table["width"]) == spec.node_widths
    per_unit = np.repeat(
        table["out_degree"].to_numpy(),
        table["width"].to_numpy(),
    )
    assert len(per_unit) == spec.state_dim
    for name in spec.nodes:
        units = spec.node_units(name)
        assert set(per_unit[units]) == {table.loc[name, "out_degree"]}


@pytest.mark.parametrize(
    ("parse", "edgelist", "widths"),
    [
        pytest.param(
            kpnn2.parse_layered,
            _layered_edgelist(),
            {"h1": 3, "h2": 2, "y": 4},
            id="layered",
        ),
        pytest.param(
            kpnn2.parse_adjacency,
            _cyclic_edgelist(),
            {"a": 3, "b": 2},
            id="adjacency",
        ),
    ],
)
def test_widths_change_only_the_width_column(
    parse,
    edgelist,
    widths,
):
    narrow = parse(edgelist).node_table()
    wide = parse(
        edgelist,
        widths=widths,
    ).node_table()
    pd.testing.assert_frame_equal(
        wide.drop(columns="width"),
        narrow.drop(columns="width"),
    )
    assert wide["width"].to_dict() == {
        name: widths.get(name, 1) for name in wide.index
    }


def test_ranks_change_only_layers_and_row_order():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "A", "Mid"],
            "target": ["Short", "Mid", "Long"],
        }
    )
    default = kpnn2.parse_layered(edgelist).node_table()
    ranked = kpnn2.parse_layered(
        edgelist,
        ranks={"A": 0, "Mid": 1, "Short": 2, "Long": 2},
    ).node_table()
    assert list(default.index) == ["A", "Mid", "Short", "Long"]
    assert list(ranked.index) == ["A", "Mid", "Long", "Short"]
    assert default.loc["Short", "layer"] == 1
    assert ranked.loc["Short", "layer"] == 2
    pd.testing.assert_frame_equal(
        ranked[_COUNTS].sort_index(),
        default[_COUNTS].sort_index(),
    )


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(
            kpnn2.parse_layered(
                _layered_edgelist(),
                widths={"h2": 2},
            ),
            id="layered",
        ),
        pytest.param(
            kpnn2.parse_adjacency(
                _cyclic_edgelist(),
                widths={"b": 2},
            ),
            id="adjacency",
        ),
    ],
)
def test_from_dict_gives_the_same_table(spec):
    rebuilt = type(spec).from_dict(spec.to_dict())
    pd.testing.assert_frame_equal(
        rebuilt.node_table(),
        spec.node_table(),
    )


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(
            kpnn2.parse_layered(_layered_edgelist()),
            id="layered",
        ),
        pytest.param(
            kpnn2.parse_adjacency(_cyclic_edgelist()),
            id="adjacency",
        ),
    ],
)
def test_each_call_returns_a_new_frame(spec):
    first = spec.node_table()
    expected = first.copy()
    first.loc[:, "in_degree"] = -1
    first["extra"] = 0
    second = spec.node_table()
    assert second is not first
    pd.testing.assert_frame_equal(
        second,
        expected,
    )
