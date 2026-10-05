import dataclasses
import random
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest
import torch

import kpnn2
from kpnn2 import (
    Kpnn2Error,
    _packed_multihead_attention,
    _parse,
    _parse_adjacency,
    _serialize,
)
from kpnn2._errors import _ISSUES_URL, internal_error, warn_at_caller

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _skip_edgelist():
    """
    Chain ``A -> H -> C`` plus the skip ``A -> C``.
    """
    return pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )


def _cyclic_edgelist():
    """
    Input x feeds a two-node feedback core; a feeds output y.
    """
    return pd.DataFrame(
        {
            "source": ["x", "a", "b", "a"],
            "target": ["a", "b", "a", "y"],
        }
    )


def _two_input_cyclic_edgelist():
    """
    Inputs x1 and x2 feed a two-node feedback core; a feeds y.
    """
    return pd.DataFrame(
        {
            "source": ["x1", "x2", "a", "b", "a"],
            "target": ["a", "a", "b", "a", "y"],
        }
    )


def _internal_check(what: str) -> str:
    return re.escape(f"internal check failed: {what}")


def test_internal_error_is_assertion_error_not_kpnn2_error():
    error = internal_error("x")

    assert isinstance(error, AssertionError)
    assert not isinstance(error, Kpnn2Error)


def test_internal_error_message_names_bug_version_and_tracker():
    message = str(internal_error("hop edges are not conserved"))

    assert "kpnn2 internal check failed" in message
    assert "hop edges are not conserved" in message
    assert "bug in kpnn2" in message
    assert kpnn2.__version__ in message
    assert _ISSUES_URL in message


def test_issues_url_matches_pyproject():
    text = _PYPROJECT.read_text(encoding="utf-8")

    assert f'Issues = "{_ISSUES_URL}"' in text


def test_warn_at_caller_points_at_the_first_frame_outside_kpnn2():
    with pytest.warns(
        FutureWarning,
        match="^soon$",
    ) as record:
        warn_at_caller(
            "soon",
            FutureWarning,
        )

    assert len(record) == 1
    assert record[0].filename == __file__


def test_raised_internal_error_is_caught_as_assertion_error():
    with pytest.raises(
        AssertionError,
        match="internal check failed",
    ):
        raise internal_error("x")


@pytest.mark.parametrize(
    ("widths", "what"),
    [
        # Width 1: the dropped pair was the whole named edge H -> C.
        (
            None,
            "parse_layered packed pairs encode 2 named edges, not the 3",
        ),
        # C is two units wide: H -> C keeps one of its two pairs.
        (
            {"C": 2},
            "parse_layered packed 4 unit pairs, expected 5",
        ),
    ],
)
def test_parse_layered_raises_on_dropped_pair(
    monkeypatch,
    widths,
    what,
):
    build_hops = _parse._build_hops

    def drop_last_pair(*args, **kwargs):
        hops = build_hops(
            *args,
            **kwargs,
        )
        last = hops[-1]
        hops[-1] = dataclasses.replace(
            last,
            source_index=last.source_index[:-1],
            target_index=last.target_index[:-1],
        )
        return hops

    monkeypatch.setattr(
        _parse,
        "_build_hops",
        drop_last_pair,
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(what),
    ):
        kpnn2.parse_layered(
            _skip_edgelist(),
            widths=widths,
        )


def test_parse_layered_raises_on_repeated_pair_with_totals_unchanged(
    monkeypatch,
):
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    build_hops = _parse._build_hops

    def repeat_pair(*args, **kwargs):
        hops = build_hops(
            *args,
            **kwargs,
        )
        first = hops[0]
        # hops[0] is the 2x2 block of A -> B; pair 3 becomes pair 0.
        assert len(first.source_index) == 4
        hops[0] = dataclasses.replace(
            first,
            source_index=first.source_index[:3] + first.source_index[:1],
            target_index=first.target_index[:3] + first.target_index[:1],
        )
        return hops

    monkeypatch.setattr(
        _parse,
        "_build_hops",
        repeat_pair,
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "parse_layered packed 1 unit pair(s) more than once"
        ),
    ):
        kpnn2.parse_layered(
            edgelist,
            widths={"A": 2, "B": 2},
        )


def test_parse_layered_raises_when_inputs_differ_from_layer_zero(
    monkeypatch,
):
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["C", "C"],
        }
    )
    node_sets = _parse._node_sets

    def reverse_inputs(*args, **kwargs):
        input_nodes, output_nodes, hidden_nodes = node_sets(
            *args,
            **kwargs,
        )
        return (
            input_nodes[::-1],
            output_nodes,
            hidden_nodes,
        )

    monkeypatch.setattr(
        _parse,
        "_node_sets",
        reverse_inputs,
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check("parse_layered input_nodes (2 names)"),
    ):
        kpnn2.parse_layered(edgelist)


@pytest.mark.parametrize(
    ("widths", "what"),
    [
        # Width 1: the dropped pair was the whole named edge x -> a.
        (
            None,
            "parse_adjacency packed pairs encode 3 named edges, not the 4",
        ),
        # x is two units wide: x -> a keeps one of its two pairs.
        (
            {"x": 2},
            "parse_adjacency packed 4 unit pairs, expected 5",
        ),
    ],
)
def test_parse_adjacency_raises_on_dropped_pair(
    monkeypatch,
    widths,
    what,
):
    packed_edge_indices = _parse_adjacency._packed_edge_indices

    def drop_last_pair(*args, **kwargs):
        source_index, target_index = packed_edge_indices(
            *args,
            **kwargs,
        )
        return (
            source_index[:-1],
            target_index[:-1],
        )

    monkeypatch.setattr(
        _parse_adjacency,
        "_packed_edge_indices",
        drop_last_pair,
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(what),
    ):
        kpnn2.parse_adjacency(
            _cyclic_edgelist(),
            widths=widths,
        )


_DROPPED_PAIR_UNDER_OPTIMIZE = textwrap.dedent(
    """
    import dataclasses
    import sys

    import pandas as pd

    import kpnn2
    from kpnn2 import _parse

    if sys.flags.optimize < 1:
        sys.exit("not running under python -O")

    build_hops = _parse._build_hops


    def drop_last_pair(*args, **kwargs):
        hops = build_hops(*args, **kwargs)
        last = hops[-1]
        hops[-1] = dataclasses.replace(
            last,
            source_index=last.source_index[:-1],
            target_index=last.target_index[:-1],
        )
        return hops


    _parse._build_hops = drop_last_pair
    edgelist = pd.DataFrame(
        {"source": ["A", "H", "A"], "target": ["H", "C", "C"]}
    )
    try:
        kpnn2.parse_layered(edgelist)
    except AssertionError as error:
        if "internal check failed" not in str(error):
            sys.exit(f"unexpected AssertionError: {error}")
    else:
        sys.exit("no AssertionError under python -O")
    """
)


def test_internal_checks_run_under_python_optimize():
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            _DROPPED_PAIR_UNDER_OPTIMIZE,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


# The parser checks once that the spec fields consumers read agree;
# the specs below break one field each, as a parser bug would.
def _broken_layer_axes_cases():
    spec = kpnn2.parse_layered(_skip_edgelist())
    assert spec.layer_dims == (1, 1, 1)
    skip_hop = spec.hops[1]
    assert skip_hop.target_layer == 2
    assert skip_hop.source_nodes == ("A", "H")
    assert skip_hop.source_dims == (1, 1)
    return [
        pytest.param(
            dataclasses.replace(
                spec,
                input_nodes=("C",),
            ),
            "parse_layered input_nodes (1 names) differ from "
            "layer_nodes[0] (1 names) in names or order",
            id="input_nodes",
        ),
        pytest.param(
            dataclasses.replace(
                spec,
                layer_dims=(1, 2, 1),
            ),
            "parse_layered layer 1 holds 1 unit(s) by layer_widths, but "
            "layer_dims[1] is 2",
            id="layer_dims",
        ),
        pytest.param(
            dataclasses.replace(
                spec,
                layer_dims=(1, 1),
            ),
            "parse_layered stored 3 layer_nodes, 3 layer_widths, and 2 "
            "layer_dims",
            id="layer_count",
        ),
        pytest.param(
            dataclasses.replace(
                spec,
                hops=(
                    spec.hops[0],
                    dataclasses.replace(
                        skip_hop,
                        source_nodes=("H", "A"),
                    ),
                ),
            ),
            "parse_layered hop into layer 2 lists 2 source_nodes that "
            "differ from the 2 names of its source layers in names or "
            "order",
            id="hop_source_nodes",
        ),
        pytest.param(
            dataclasses.replace(
                spec,
                hops=(
                    spec.hops[0],
                    dataclasses.replace(
                        skip_hop,
                        source_dims=(1, 2),
                    ),
                ),
            ),
            "parse_layered hop into layer 2 has source_dims (1, 2), but "
            "its source layers are (1, 1) units wide",
            id="hop_source_dims",
        ),
    ]


@pytest.mark.parametrize(
    ("broken", "what"),
    _broken_layer_axes_cases(),
)
def test_layer_axes_check_raises_when_a_field_disagrees(
    broken,
    what,
):
    with pytest.raises(
        AssertionError,
        match=_internal_check(what),
    ):
        _parse._check_layer_axes(broken)


def test_parse_layered_runs_the_layer_axes_check(monkeypatch):
    build_hops = _parse._build_hops

    def reverse_source_nodes(*args, **kwargs):
        hops = build_hops(
            *args,
            **kwargs,
        )
        last = hops[-1]
        hops[-1] = dataclasses.replace(
            last,
            source_nodes=last.source_nodes[::-1],
        )
        return hops

    monkeypatch.setattr(
        _parse,
        "_build_hops",
        reverse_source_nodes,
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "parse_layered hop into layer 2 lists 2 source_nodes"
        ),
    ):
        kpnn2.parse_layered(_skip_edgelist())


@pytest.mark.parametrize(
    ("input_index", "what"),
    [
        pytest.param(
            (3, 2),
            "parse_adjacency input_index (2 units) differs from the "
            "units of input_nodes (2 units) in values or order",
            id="reordered",
        ),
        pytest.param(
            (2,),
            "parse_adjacency input_index (1 units) differs from the "
            "units of input_nodes (2 units) in values or order",
            id="short",
        ),
        pytest.param(
            (2, 99),
            "parse_adjacency input_index (2 units) differs from the "
            "units of input_nodes (2 units) in values or order",
            id="out_of_range",
        ),
    ],
)
def test_input_units_check_raises_when_input_index_disagrees(
    input_index,
    what,
):
    spec = kpnn2.parse_adjacency(_two_input_cyclic_edgelist())
    assert spec.input_nodes == ("x1", "x2")
    assert spec.input_index == (2, 3)
    broken = dataclasses.replace(
        spec,
        input_index=input_index,
    )

    with pytest.raises(
        AssertionError,
        match=_internal_check(what),
    ):
        _parse_adjacency._check_input_units(broken)


def test_parse_adjacency_runs_the_input_units_check(monkeypatch):
    units_of = _parse_adjacency._units_of

    def reverse_units(*args, **kwargs):
        return units_of(
            *args,
            **kwargs,
        )[::-1]

    monkeypatch.setattr(
        _parse_adjacency,
        "_units_of",
        reverse_units,
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check("parse_adjacency input_index (2 units) differs"),
    ):
        kpnn2.parse_adjacency(_two_input_cyclic_edgelist())


def test_parse_layered_out_of_range_unit_fails_an_internal_check(
    monkeypatch,
):
    build_hops = _parse._build_hops

    def push_unit_out_of_range(*args, **kwargs):
        hops = build_hops(
            *args,
            **kwargs,
        )
        last = hops[-1]
        hops[-1] = dataclasses.replace(
            last,
            source_index=last.source_index[:-1] + (last.in_features,),
        )
        return hops

    monkeypatch.setattr(
        _parse,
        "_build_hops",
        push_unit_out_of_range,
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "unit index 2 is out of the layout's range [0, 2)"
        ),
    ):
        kpnn2.parse_layered(_skip_edgelist())


def test_layered_edge_location_out_of_range_unit_fails_an_internal_check():
    spec = kpnn2.parse_layered(_skip_edgelist())
    skip_hop = spec.hops[1]
    assert skip_hop.in_features == 2
    broken_hop = dataclasses.replace(
        skip_hop,
        source_index=skip_hop.source_index[:-1] + (2,),
    )
    broken = dataclasses.replace(
        spec,
        hops=(spec.hops[0], broken_hop),
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "unit index 2 is out of the layout's range [0, 2)"
        ),
    ):
        broken.edge_location(
            "A",
            "H",
        )


def test_adjacency_edge_location_out_of_range_unit_fails_an_internal_check():
    spec = kpnn2.parse_adjacency(_cyclic_edgelist())
    broken = dataclasses.replace(
        spec,
        source_index=spec.source_index[:-1] + (spec.state_dim,),
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "unit index 4 is out of the layout's range [0, 4)"
        ),
    ):
        broken.edge_location(
            "a",
            "b",
        )


def test_to_dict_re_ranking_failure_fails_an_internal_check(monkeypatch):
    spec = kpnn2.parse_layered(_skip_edgelist())

    def refuse_to_rank(*args, **kwargs):
        raise Kpnn2Error("Edgelist contains a cycle.")

    monkeypatch.setattr(
        _parse,
        "_rank_layers",
        refuse_to_rank,
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "re-ranking a LayeredSpec's own edges failed: "
            "Edgelist contains a cycle."
        ),
    ) as caught:
        spec.to_dict()
    assert isinstance(
        caught.value.__cause__,
        Kpnn2Error,
    )


def test_canonical_edges_on_a_non_spec_fails_an_internal_check():
    """
    Only spec methods and the parsers call it, with a spec kpnn2
    built, so a caller cannot reach this branch.
    """
    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "canonical_edges received type object, "
            "not a LayeredSpec or an AdjacencySpec"
        ),
    ):
        _serialize.canonical_edges(object())


@pytest.mark.parametrize(
    ("kind", "method", "args"),
    [
        ("layered", "node_units", ("missing",)),
        ("layered", "node_units", ("",)),
        ("layered", "hop_units", ("missing",)),
        ("layered", "hop_units", ("C",)),
        ("layered", "edge_location", ("A", "missing")),
        ("layered", "edge_location", ("C", "A")),
        ("adjacency", "node_units", ("missing",)),
        ("adjacency", "edge_location", ("x", "missing")),
        ("adjacency", "edge_location", ("y", "x")),
    ],
)
def test_public_name_misses_still_raise_kpnn2_error(
    kind,
    method,
    args,
):
    """
    Layout lookups fail as internal checks, so every public path
    that takes a caller's node or edge name must validate first.
    """
    if kind == "layered":
        spec = kpnn2.parse_layered(
            _skip_edgelist(),
            widths={"H": 2},
        )
    else:
        spec = kpnn2.parse_adjacency(
            _cyclic_edgelist(),
            widths={"a": 2},
        )
    if method == "hop_units":
        args = (spec.hops[1], *args)
    with pytest.raises(Kpnn2Error):
        getattr(
            spec,
            method,
        )(*args)


def _shuffled_frame(edges):
    rows = list(edges)
    random.shuffle(rows)
    return pd.DataFrame(
        rows,
        columns=["source", "target"],
    )


def _random_dag():
    """
    Forward edges along a random topological order of 3-12 nodes.
    """
    n_nodes = random.randint(3, 12)
    order = random.sample(
        [f"n{index}" for index in range(n_nodes)],
        n_nodes,
    )
    density = random.uniform(0.15, 0.6)
    edges = [
        (source, target)
        for position, source in enumerate(order)
        for target in order[position + 1 :]
        if random.random() < density
    ]
    if not edges:
        edges = [(order[0], order[-1])]
    return order, edges


def _random_ranks(order, edges):
    """
    Inputs at 0, every other node 1-3 above its highest parent.
    """
    parents: dict[str, list[str]] = {}
    for source, target in edges:
        parents.setdefault(target, []).append(source)
    nodes = {name for edge in edges for name in edge}
    ranks: dict[str, int] = {}
    for node in order:
        if node not in nodes:
            continue
        if node in parents:
            ranks[node] = max(
                ranks[parent] for parent in parents[node]
            ) + random.randint(1, 3)
        else:
            ranks[node] = 0
    return ranks


def _random_widths(edges):
    nodes = sorted({name for edge in edges for name in edge})
    chosen = random.sample(
        nodes,
        random.randint(0, len(nodes)),
    )
    return {name: random.randint(1, 3) for name in chosen}


def _random_cyclic():
    """
    Input ``in`` feeds a core that holds a ring (a self-loop when
    the core is one node), and the core feeds output ``out``.
    """
    n_core = random.randint(1, 6)
    core = [f"c{index}" for index in range(n_core)]
    edges = {
        (core[index], core[(index + 1) % n_core]) for index in range(n_core)
    }
    for source in core:
        for target in core:
            if random.random() < 0.3:
                edges.add((source, target))
    edges.add(("in", random.choice(core)))
    edges.add((random.choice(core), "out"))
    return sorted(edges)


def test_parsers_pass_internal_checks_on_random_graphs():
    random.seed(42)
    n_reranked = 0
    for _ in range(200):
        order, edges = _random_dag()
        widths = _random_widths(edges)
        edgelist = _shuffled_frame(edges)
        default = kpnn2.parse_layered(
            edgelist,
            widths=widths,
        )
        ranked = kpnn2.parse_layered(
            edgelist,
            widths=widths,
            ranks=_random_ranks(
                order,
                edges,
            ),
        )
        kpnn2.parse_adjacency(
            edgelist,
            widths=widths,
        )
        n_reranked += ranked.layer_nodes != default.layer_nodes
    for _ in range(100):
        edges = _random_cyclic()
        kpnn2.parse_adjacency(
            _shuffled_frame(edges),
            widths=_random_widths(edges),
        )

    # ranks= must place nodes off longest-path on a fair share.
    assert n_reranked > 20


def test_softmax_row_check_raises_on_rows_off_one():
    row_sum = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.5],
        ]
    )
    degree = torch.full(
        (2, 1),
        4.0,
    )

    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "packed softmax weights of a query do not sum to 1"
        ),
    ):
        _packed_multihead_attention._check_softmax_rows(
            row_sum,
            torch.zeros_like(
                row_sum,
                dtype=torch.bool,
            ),
            degree,
            torch.float32,
        )


def test_softmax_row_check_passes_one_zero_and_non_finite_rows():
    row_sum = torch.tensor(
        [
            [1.0, 0.0],
            [float("nan"), float("inf")],
        ]
    )
    # Only the zero row has no live score above the fill.
    no_live_score = torch.tensor(
        [
            [False, True],
            [False, False],
        ]
    )
    degree = torch.full(
        (2, 1),
        4.0,
    )

    _packed_multihead_attention._check_softmax_rows(
        row_sum,
        no_live_score,
        degree,
        torch.float32,
    )


def test_softmax_row_check_raises_on_a_zero_row_with_a_live_score():
    row_sum = torch.tensor(
        [
            [1.0, 0.0],
        ]
    )
    degree = torch.full(
        (1, 1),
        4.0,
    )

    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "packed softmax weights of a query do not sum to 1"
        ),
    ):
        _packed_multihead_attention._check_softmax_rows(
            row_sum,
            torch.zeros_like(
                row_sum,
                dtype=torch.bool,
            ),
            degree,
            torch.float32,
        )


def test_chunked_attention_catches_a_bug_that_zeroes_live_queries(
    monkeypatch,
):
    # Simulate a bug: the max is formed from the real scores, but every
    # weight is then zeroed. Each live query sums to 0, not 1.
    original = _packed_multihead_attention._apply_participate_to_scores

    def zeroing(
        scores,
        participate,
        fill,
    ):
        scores_for_max, _ = original(
            scores,
            participate,
            fill,
        )
        return scores_for_max, torch.full_like(
            scores,
            float("-inf"),
        )

    monkeypatch.setattr(
        _packed_multihead_attention,
        "_apply_participate_to_scores",
        zeroing,
    )
    torch.manual_seed(42)
    spec = kpnn2.parse_adjacency(_cyclic_edgelist())
    n = spec.state_dim
    layer = kpnn2.PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        n,
        n,
        4,
        2,
        chunk_size=2,
    )
    x = torch.randn(
        2,
        n,
        4,
    )

    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "packed softmax weights of a query do not sum to 1"
        ),
    ):
        layer(
            x,
            x,
            x,
        )


def test_chunked_attention_accepts_a_live_query_whose_scores_overflow():
    # Query 0 has two live keys, but in float16 every one of its scores
    # overflows to -inf, so its row sums to 0 from the caller's data.
    query = torch.tensor(
        [[[300.0, 300.0]], [[1.0, 1.0]]],
        dtype=torch.float16,
    )
    key = torch.full(
        (2, 1, 2),
        -300.0,
        dtype=torch.float16,
    )
    value = torch.ones(
        2,
        1,
        2,
        dtype=torch.float16,
    )
    source = torch.tensor([0, 1, 0])
    target = torch.tensor([0, 0, 1])

    chunked, chunked_attn = (
        _packed_multihead_attention._packed_attention_chunked(
            query,
            key,
            value,
            source,
            target,
            0.0,
            False,
            2,
        )
    )
    dense, dense_attn = _packed_multihead_attention._packed_attention(
        query,
        key,
        value,
        source,
        target,
        0.0,
        False,
        None,
    )

    assert chunked_attn[..., :2, :].abs().sum() == 0
    torch.testing.assert_close(
        chunked,
        dense,
    )
    torch.testing.assert_close(
        chunked_attn,
        dense_attn,
    )


def test_softmax_grad_row_check_raises_on_a_row_off_zero():
    grad_sum = torch.tensor(
        [
            [0.0, 1.0],
        ]
    )
    scale = torch.ones(
        1,
        2,
    )
    degree = torch.full(
        (1, 1),
        4.0,
    )

    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "packed softmax gradient of a query does not sum to 0"
        ),
    ):
        _packed_multihead_attention._check_softmax_grad_rows(
            grad_sum,
            scale,
            degree,
            torch.float32,
        )


def test_softmax_grad_row_check_passes_zero_and_non_finite_rows():
    grad_sum = torch.tensor(
        [
            [0.0, 1e-7],
            [float("nan"), float("inf")],
        ]
    )
    scale = torch.ones(
        2,
        2,
    )
    degree = torch.full(
        (2, 1),
        4.0,
    )

    _packed_multihead_attention._check_softmax_grad_rows(
        grad_sum,
        scale,
        degree,
        torch.float32,
    )


def _count_calls(
    monkeypatch,
    name,
):
    calls = []
    check = getattr(
        _packed_multihead_attention,
        name,
    )

    def counted(*args):
        calls.append(args)
        return check(*args)

    monkeypatch.setattr(
        _packed_multihead_attention,
        name,
        counted,
    )
    return calls


@pytest.mark.parametrize(
    ("chunk_size", "n_calls"),
    [
        (None, 0),
        (2, 1),
    ],
)
def test_attention_softmax_checks_run_once_per_chunked_pass(
    monkeypatch,
    chunk_size,
    n_calls,
):
    torch.manual_seed(42)
    row_calls = _count_calls(
        monkeypatch,
        "_check_softmax_rows",
    )
    grad_calls = _count_calls(
        monkeypatch,
        "_check_softmax_grad_rows",
    )
    spec = kpnn2.parse_adjacency(_cyclic_edgelist())
    n = spec.state_dim
    layer = kpnn2.PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        n,
        n,
        4,
        2,
        chunk_size=chunk_size,
    )
    x = torch.randn(
        2,
        n,
        4,
        requires_grad=True,
    )

    out, _ = layer(
        x,
        x,
        x,
    )
    assert len(row_calls) == n_calls
    assert len(grad_calls) == 0
    out.sum().backward()

    assert len(row_calls) == n_calls
    assert len(grad_calls) == n_calls


_STAR_KEYS = 1000


def _star_attention(
    dtype,
    **kwargs,
):
    """
    Query 0 attends over every other node, query 1 over three of
    them, and the remaining queries have no live key.
    """
    n = _STAR_KEYS + 1
    source = list(range(1, n)) + [2, 3, 4]
    target = [0] * _STAR_KEYS + [1, 1, 1]
    layer = kpnn2.PackedMultiheadAttention(
        source,
        target,
        n,
        n,
        4,
        2,
        chunk_size=64,
        **kwargs,
    )
    return layer.to(dtype=dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    ("layer_kwargs", "padding", "nan_input"),
    [
        ({}, False, False),
        ({}, True, False),
        ({"dropout": 0.5}, False, False),
        ({"add_self_loops": True}, True, False),
        ({}, False, True),
    ],
    ids=[
        "star",
        "padding",
        "dropout",
        "self_loops",
        "nan_input",
    ],
)
def test_chunked_attention_passes_internal_checks(
    dtype,
    layer_kwargs,
    padding,
    nan_input,
):
    torch.manual_seed(42)
    layer = _star_attention(
        dtype,
        **layer_kwargs,
    )
    layer.train()
    n = _STAR_KEYS + 1
    x = torch.randn(
        3,
        n,
        4,
        dtype=dtype,
    )
    if nan_input:
        x[0, 3, 0] = float("nan")
    x.requires_grad_(True)
    key_padding_mask = None
    if padding:
        key_padding_mask = torch.rand(3, n) < 0.5
        key_padding_mask[1] = True

    out, weights = layer(
        x,
        x,
        x,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    (out.sum() + weights.sum()).backward()

    # Non-finite rows skip the checks, so the clean runs must be finite.
    if nan_input:
        assert torch.isnan(out).any()
    else:
        assert torch.isfinite(weights).all()
        assert torch.isfinite(x.grad).all()
