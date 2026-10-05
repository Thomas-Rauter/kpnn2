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
from kpnn2._errors import _ISSUES_URL, internal_error

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


def test_align_inputs_raises_when_input_nodes_leave_layer_zero_order():
    spec = kpnn2.parse_layered(
        pd.DataFrame(
            {
                "source": ["A", "B"],
                "target": ["C", "C"],
            }
        )
    )
    broken = dataclasses.replace(
        spec,
        input_nodes=spec.input_nodes[::-1],
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "align_inputs column 0 holds 'B', but the spec's input "
            "axis has 'A' there"
        ),
    ):
        kpnn2.align_inputs(
            ["A", "B"],
            broken,
        )


def test_align_inputs_raises_when_layer_dims_disagree():
    spec = kpnn2.parse_layered(_skip_edgelist())
    broken = dataclasses.replace(
        spec,
        layer_dims=(spec.layer_dims[0] + 1,) + spec.layer_dims[1:],
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "align_inputs returned 1 column(s), but spec.layer_dims[0] is 2"
        ),
    ):
        kpnn2.align_inputs(
            ["A"],
            broken,
        )


def test_align_inputs_raises_when_input_index_is_reordered():
    spec = kpnn2.parse_adjacency(_two_input_cyclic_edgelist())
    assert spec.input_nodes == ("x1", "x2")
    broken = dataclasses.replace(
        spec,
        input_index=spec.input_index[::-1],
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "align_inputs column 0 holds 'x1', but the spec's input "
            "axis has 'x2' there"
        ),
    ):
        kpnn2.align_inputs(
            ["x1", "x2"],
            broken,
        )


# Width 1 is the layer's real width; width 2 is what the broken
# layer_dims claims. Neither caller is wrong.
@pytest.mark.parametrize(
    "n_units",
    [1, 2],
)
def test_map_node_attributions_raises_when_layer_dims_disagree(n_units):
    spec = kpnn2.parse_layered(_skip_edgelist())
    assert spec.layer_dims == (1, 1, 1)
    broken = dataclasses.replace(
        spec,
        layer_dims=(1, 2, 1),
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "map_node_attributions named 1 unit(s) of layer 1, but "
            "spec.layer_dims[1] is 2"
        ),
    ):
        kpnn2.map_node_attributions(
            torch.zeros(2, n_units),
            broken,
            layer=1,
        )


@pytest.mark.parametrize(
    ("field", "what"),
    [
        (
            "source_nodes",
            "map_node_attributions hop_input names (2 nodes) differ "
            "from hop.source_nodes (2 nodes) in names or order",
        ),
        (
            "source_dims",
            "map_node_attributions named 2 unit(s) on the hop_input "
            "axis, but hop.in_features is 3",
        ),
    ],
)
def test_map_node_attributions_raises_when_hop_input_disagrees(
    field,
    what,
):
    spec = kpnn2.parse_layered(_skip_edgelist())
    skip_hop = spec.hops[1]
    assert skip_hop.source_nodes == ("A", "H")
    assert skip_hop.source_dims == (1, 1)
    broken_value = {
        "source_nodes": ("H", "A"),
        "source_dims": (1, 2),
    }[field]
    broken_hop = dataclasses.replace(
        skip_hop,
        **{field: broken_value},
    )
    broken = dataclasses.replace(
        spec,
        hops=(spec.hops[0], broken_hop),
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(what),
    ):
        kpnn2.map_node_attributions(
            torch.zeros(2, 2),
            broken,
            hop_input=broken_hop,
        )


def test_map_node_attributions_raises_when_input_index_is_short():
    spec = kpnn2.parse_adjacency(
        _cyclic_edgelist(),
        widths={"x": 2},
    )
    assert len(spec.input_index) == 2
    broken = dataclasses.replace(
        spec,
        input_index=spec.input_index[:-1],
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "map_node_attributions named 2 input unit(s), but "
            "spec.input_index has 1"
        ),
    ):
        kpnn2.map_node_attributions(
            torch.zeros(2, 2),
            broken,
            axis="inputs",
        )


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


def test_align_inputs_out_of_range_input_index_fails_an_internal_check():
    spec = kpnn2.parse_adjacency(_cyclic_edgelist())
    assert spec.state_dim == 4
    broken = dataclasses.replace(
        spec,
        input_index=(spec.state_dim,),
    )
    with pytest.raises(
        AssertionError,
        match=_internal_check(
            "unit index 4 is out of the layout's range [0, 4)"
        ),
    ):
        kpnn2.align_inputs(
            ["x"],
            broken,
        )


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
    degree = torch.full(
        (2, 1),
        4.0,
    )

    _packed_multihead_attention._check_softmax_rows(
        row_sum,
        degree,
        torch.float32,
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
