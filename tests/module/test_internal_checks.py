import dataclasses
import random
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

import kpnn2
from kpnn2 import Kpnn2Error, _parse, _parse_adjacency
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
