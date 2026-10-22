"""
``outputs=`` confirms the inferred output nodes; without it, several warn.

Output nodes are every node with no outgoing edge, which a node whose
onward rows are missing also has. ``outputs`` must name exactly those
nodes. Otherwise the parse raises and names every node with no path to
a declared output, and dropping the rows whose target is one of them
leaves exactly the declared outputs. The check never changes the spec.
"""

import random
import re
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

import kpnn2
from kpnn2 import Kpnn2Error, parse_adjacency, parse_layered
from kpnn2._parse import _without_path_to

_PARSERS = [
    pytest.param(
        parse_layered,
        id="layered",
    ),
    pytest.param(
        parse_adjacency,
        id="adjacency",
    ),
]


def _dangling_edgelist():
    """
    Genes feed TF1 and TF2, which feed ``out``. TF3 has parents, one
    of them through K, but no row leaves it.
    """
    return pd.DataFrame(
        {
            "source": ["g1", "g2", "g3", "g4", "g5", "K", "TF1", "TF2"],
            "target": ["TF1", "TF2", "TF3", "TF3", "K", "TF3", "out", "out"],
        }
    )


def _two_output_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "A", "H"],
            "target": ["H", "E", "C"],
        }
    )


def _frame(edges):
    return pd.DataFrame(
        edges,
        columns=["source", "target"],
    )


def _sinks(edges):
    """
    Nodes that are never a source: the graph's output nodes.
    """
    sources = {source for source, _ in edges}
    return sorted({target for _, target in edges} - sources)


def _random_dag():
    """
    Edges only from a lower to a higher index, so no cycle.
    """
    n_nodes = random.randint(4, 12)
    edges = set()
    for target in range(1, n_nodes):
        edges.add((random.randrange(target), target))
        for source in range(target):
            if random.random() < 0.2:
                edges.add((source, target))
    return sorted((f"n{source}", f"n{target}") for source, target in edges)


def _random_cyclic():
    """
    Random edges among core nodes, plus an input and an output.

    Self-loops and cycles are allowed. Core nodes without an edge
    out of them are extra output nodes.
    """
    core = [f"c{index}" for index in range(random.randint(3, 10))]
    edges = {(source, target) for source in core for target in core}
    edges = {edge for edge in edges if random.random() < 0.15}
    edges.add(("in", random.choice(core)))
    edges.add((random.choice(core), "out"))
    return sorted(edges)


def _random_graphs():
    """
    Yield ``(parse, edges)``: DAGs for both parsers, cyclic graphs
    for ``parse_adjacency``, each with more than one output node.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    n_dags = 0
    while n_dags < 100:
        edges = _random_dag()
        if len(_sinks(edges)) > 1:
            n_dags += 1
            yield parse_layered, edges
            yield parse_adjacency, edges
    n_cyclic = 0
    while n_cyclic < 100:
        edges = _random_cyclic()
        if len(_sinks(edges)) > 1:
            n_cyclic += 1
            yield parse_adjacency, edges


def _no_path_independently(
    edges,
    declared,
):
    """
    Nodes from which no forward path reaches a declared output.
    """
    graph = nx.DiGraph(edges)
    return sorted(
        node
        for node in graph.nodes
        if node not in declared
        and nx.descendants(graph, node).isdisjoint(declared)
    )


@pytest.mark.parametrize(
    "parse",
    _PARSERS,
)
def test_single_output_parses_silently(parse):
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = parse(edgelist)

    assert spec.output_nodes == ("C",)


@pytest.mark.parametrize(
    "parse",
    _PARSERS,
)
def test_several_outputs_warn_and_name_them(parse):
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "The edgelist has 2 output nodes, nodes with no outgoing "
            "edge: TF3, out. A node whose onward rows are missing from "
            "the edgelist is an output node too. Pass the output nodes "
            "you read as 'outputs' to confirm them: this warning then "
            "stops, and parsing raises on any other node with no "
            "outgoing edge."
        ),
    ) as record:
        parse(_dangling_edgelist())

    assert len(record) == 1


@pytest.mark.parametrize(
    "parse",
    _PARSERS,
)
def test_several_outputs_warning_points_at_the_caller(parse):
    with pytest.warns(UserWarning) as record:
        parse(_two_output_edgelist())

    assert record[0].filename == __file__


@pytest.mark.parametrize(
    "parse",
    _PARSERS,
)
def test_confirmed_outputs_parse_silently_and_change_nothing(parse):
    edgelist = _dangling_edgelist()
    with pytest.warns(UserWarning):
        inferred = parse(edgelist)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        confirmed = parse(
            edgelist,
            outputs=["out", "TF3"],
        )

    assert confirmed == inferred
    assert confirmed.to_dict() == inferred.to_dict()
    assert confirmed.fingerprint == inferred.fingerprint


@pytest.mark.parametrize(
    "parse",
    _PARSERS,
)
@pytest.mark.parametrize(
    "outputs",
    [
        pytest.param(
            ["C", "E"],
            id="list",
        ),
        pytest.param(
            ("E", "C"),
            id="tuple_in_another_order",
        ),
        pytest.param(
            {"C", "E"},
            id="set",
        ),
        pytest.param(
            frozenset({"C", "E"}),
            id="frozenset",
        ),
        pytest.param(
            np.array(["E", "C"]),
            id="numpy_array",
        ),
        pytest.param(
            pd.Index(["C", "E"]),
            id="pandas_index",
        ),
    ],
)
def test_outputs_accept_any_collection_in_any_order(
    parse,
    outputs,
):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = parse(
            _two_output_edgelist(),
            outputs=outputs,
        )

    assert spec.output_nodes == ("C", "E")


@pytest.mark.parametrize(
    "parse",
    _PARSERS,
)
def test_outputs_match_integer_names_after_str(parse):
    edgelist = pd.DataFrame(
        {
            "source": [1, 1],
            "target": [2, 3],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = parse(
            edgelist,
            outputs=[np.int64(2), 3],
        )

    assert spec.output_nodes == ("2", "3")


@pytest.mark.parametrize(
    "parse",
    _PARSERS,
)
def test_left_out_output_raises_with_every_node_without_a_path(parse):
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "Node(s) with no outgoing edge that are not in 'outputs': "
            "TF3. The edgelist has no row leaving them, so they are "
            "output nodes too. Node(s) with no path to a node in "
            "'outputs': K, TF3, g3, g4, g5. Drop the edgelist rows whose "
            "target is one of these and parse again, or add a node to "
            "'outputs' if you read it."
        ),
    ):
        parse(
            _dangling_edgelist(),
            outputs=["out"],
        )


@pytest.mark.parametrize(
    "parse",
    _PARSERS,
)
def test_dropping_the_named_rows_leaves_the_declared_outputs(parse):
    edgelist = _dangling_edgelist()
    dead = ["K", "TF3", "g3", "g4", "g5"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = parse(
            edgelist[~edgelist["target"].isin(dead)],
            outputs=["out"],
        )

    assert spec.output_nodes == ("out",)
    assert spec.input_nodes == ("g1", "g2")


def test_cycle_without_a_path_to_an_output_is_named():
    edgelist = pd.DataFrame(
        {
            "source": ["x", "a", "x", "c", "d", "d"],
            "target": ["a", "y", "c", "d", "c", "z"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match=re.escape("no path to a node in 'outputs': c, d, z."),
    ):
        parse_adjacency(
            edgelist,
            outputs=["y"],
        )


def test_self_loop_does_not_make_a_path():
    edgelist = pd.DataFrame(
        {
            "source": ["x", "a", "x", "b", "b"],
            "target": ["a", "y", "b", "b", "z"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match=re.escape("no path to a node in 'outputs': b, z."),
    ):
        parse_adjacency(
            edgelist,
            outputs=["y"],
        )


def test_named_nodes_match_an_independent_walk_on_random_graphs():
    n_checked = 0
    for parse, edges in _random_graphs():
        outputs = _sinks(edges)
        declared = sorted(
            random.sample(
                outputs,
                random.randint(1, len(outputs) - 1),
            )
        )
        dead = _without_path_to(
            declared,
            _frame(edges),
        )
        assert dead == _no_path_independently(
            edges,
            declared,
        )
        with pytest.raises(
            Kpnn2Error,
            match=re.escape(
                f"no path to a node in 'outputs': {', '.join(dead)}."
            ),
        ):
            parse(
                _frame(edges),
                outputs=declared,
            )
        n_checked += 1
    assert n_checked == 300


def test_one_drop_pass_leaves_exactly_the_declared_outputs():
    """
    The kept rows have exactly the declared outputs as sinks. They
    parse unless no input reaches a declared output, which only a
    cycle that no input feeds allows; then they have no input node.
    """
    n_parsed = 0
    n_without_input = 0
    for parse, edges in _random_graphs():
        outputs = _sinks(edges)
        declared = sorted(
            random.sample(
                outputs,
                random.randint(1, len(outputs) - 1),
            )
        )
        dead = set(
            _without_path_to(
                declared,
                _frame(edges),
            )
        )
        kept = [edge for edge in edges if edge[1] not in dead]
        original = parse(
            _frame(edges),
            outputs=outputs,
        )
        assert _sinks(kept) == declared
        if set(original.input_nodes) <= dead:
            assert nx.is_directed_acyclic_graph(nx.DiGraph(edges)) is False
            with pytest.raises(
                Kpnn2Error,
                match="at least one input node",
            ):
                parse(
                    _frame(kept),
                    outputs=declared,
                )
            n_without_input += 1
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pruned = parse(
                _frame(kept),
                outputs=declared,
            )

        assert pruned.output_nodes == tuple(declared)
        assert set(pruned.input_nodes) == set(original.input_nodes) - dead
        n_parsed += 1
    assert n_parsed > 250
    assert n_without_input > 0


@pytest.mark.parametrize(
    "parse",
    _PARSERS,
)
def test_unknown_and_non_output_names_raise_in_one_message(parse):
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'outputs' names node(s) that are not in the edgelist: oot. "
            "'outputs' names node(s) with an outgoing edge, which are not "
            "output nodes: TF1 (hidden), g1 (input). To read a node that "
            "feeds others, index it with unit_index()."
        ),
    ):
        parse(
            _dangling_edgelist(),
            outputs=["out", "oot", "TF1", "g1", "TF3"],
        )


@pytest.mark.parametrize(
    "parse",
    _PARSERS,
)
@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        pytest.param(
            "C",
            "'outputs' must be a sequence of output node names, not a "
            "str.\n"
            "  Got: 'C' (str)\n"
            "  Fix: Wrap a single name in a list.",
            id="string",
        ),
        pytest.param(
            b"C",
            "'outputs' must be a sequence of output node names, not "
            "bytes.\n"
            "  Got: b'C' (bytes)",
            id="bytes",
        ),
        pytest.param(
            torch.tensor([1]),
            "'outputs' must be a sequence of output node names, not a "
            "tensor.\n"
            "  Got: Tensor of shape (1,), dtype torch.int64",
            id="tensor",
        ),
        pytest.param(
            {"C": 1},
            "'outputs' must be a collection of node names, not a mapping. "
            "Got {'C': 1} (dict). Pass the names as a list.",
            id="mapping",
        ),
        pytest.param(
            np.array([["C", "E"]]),
            "'outputs' must be one-dimensional: a matrix is not a name list.",
            id="matrix",
        ),
        pytest.param(
            5,
            "'outputs' must be a sequence of output node names.\n"
            "  Got: 5 (int)",
            id="int",
        ),
        pytest.param(
            [],
            "'outputs' must name at least one node. Got [] (list). Pass the "
            "output nodes you read, or omit 'outputs'.",
            id="empty",
        ),
        pytest.param(
            ["C", "E", "C"],
            "'outputs' names a node more than once: C. Name each output "
            "node once.",
            id="repeated",
        ),
        pytest.param(
            ["C", 1, "1"],
            "'outputs' names a node more than once: 1.",
            id="repeated_after_str",
        ),
    ],
)
def test_bad_outputs_raise(
    parse,
    outputs,
    message,
):
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(message),
    ):
        parse(
            _two_output_edgelist(),
            outputs=outputs,
        )


@pytest.mark.parametrize(
    "parse",
    _PARSERS,
)
def test_outputs_type_is_checked_before_the_edgelist(parse):
    with pytest.raises(
        Kpnn2Error,
        match="'outputs' must be a sequence of output node names, not a str",
    ):
        parse(
            "not a frame",
            outputs="C",
        )


@pytest.mark.parametrize(
    "spec_type",
    [
        pytest.param(
            kpnn2.LayeredSpec,
            id="layered",
        ),
        pytest.param(
            kpnn2.AdjacencySpec,
            id="adjacency",
        ),
    ],
)
def test_from_dict_rebuilds_several_outputs_silently(spec_type):
    parse = parse_layered if spec_type is kpnn2.LayeredSpec else parse_adjacency
    spec = parse(
        _dangling_edgelist(),
        outputs=["TF3", "out"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rebuilt = spec_type.from_dict(spec.to_dict())

    assert rebuilt == spec


@pytest.mark.parametrize(
    "parse",
    _PARSERS,
)
def test_caller_data_is_not_modified(parse):
    edgelist = _dangling_edgelist()
    before = edgelist.copy()
    outputs = ["out"]

    with pytest.raises(Kpnn2Error):
        parse(
            edgelist,
            outputs=outputs,
        )

    pd.testing.assert_frame_equal(
        edgelist,
        before,
    )
    assert outputs == ["out"]
