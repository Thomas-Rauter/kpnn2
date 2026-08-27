import inspect

import pandas as pd
import pytest

import kpnn2
import kpnn2._serialize as serialize_mod
from kpnn2 import (
    Kpnn2Error,
    parse_adjacency,
    parse_layered,
)
from kpnn2._serialize import canonical_edges


def _chain_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )


def _skip_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )


def _cycle_and_self_loop_edgelist():
    return pd.DataFrame(
        {
            "source": ["x", "a", "b", "a", "a"],
            "target": ["a", "b", "a", "a", "y"],
        }
    )


def _n_ones_layered(spec):
    total = 0
    for hop in spec.hops:
        total += int((hop.mask == 1.0).sum().item())
    return total


def test_canonical_edges_is_not_a_public_name():
    assert "canonical_edges" not in kpnn2.__all__
    assert not hasattr(
        kpnn2,
        "canonical_edges",
    )


def test_layered_chain_is_sorted_source_target_pairs():
    spec = parse_layered(_chain_edgelist())

    edges = canonical_edges(spec)

    assert edges == (
        ("A", "B"),
        ("B", "C"),
    )
    assert isinstance(edges, tuple)
    assert all(isinstance(pair, tuple) for pair in edges)


def test_layered_skip_graph_reads_hop_masks_not_skips():
    spec = parse_layered(_skip_edgelist())

    edges = canonical_edges(spec)

    assert edges == (
        ("A", "C"),
        ("A", "H"),
        ("H", "C"),
    )
    assert len(edges) == _n_ones_layered(spec)
    assert len(spec.skips) == 1
    assert spec.skips[0].source == "A"
    assert spec.skips[0].target == "C"


def test_unsorted_layered_input_matches_sorted_copy():
    unsorted = pd.DataFrame(
        {
            "source": ["H", "A", "A"],
            "target": ["C", "C", "H"],
        }
    )
    sorted_copy = unsorted.sort_values(
        ["source", "target"],
    ).reset_index(drop=True)

    unsorted_edges = canonical_edges(parse_layered(unsorted))
    sorted_edges = canonical_edges(parse_layered(sorted_copy))

    assert unsorted_edges == sorted_edges
    assert unsorted_edges == (
        ("A", "C"),
        ("A", "H"),
        ("H", "C"),
    )


def test_adjacency_includes_cycle_and_self_loop():
    spec = parse_adjacency(_cycle_and_self_loop_edgelist())

    edges = canonical_edges(spec)

    assert edges == (
        ("a", "a"),
        ("a", "b"),
        ("a", "y"),
        ("b", "a"),
        ("x", "a"),
    )
    assert ("a", "a") in edges
    assert ("a", "b") in edges
    assert ("b", "a") in edges
    n_ones = int((spec.mask == 1.0).sum().item())
    assert len(edges) == n_ones


def test_dag_parsed_both_ways_yields_the_same_tuple():
    edgelist = _skip_edgelist()

    layered = canonical_edges(parse_layered(edgelist))
    adjacency = canonical_edges(parse_adjacency(edgelist))

    assert layered == adjacency
    assert layered == (
        ("A", "C"),
        ("A", "H"),
        ("H", "C"),
    )


def test_non_spec_argument_raises_kpnn2_error():
    with pytest.raises(
        Kpnn2Error,
        match="must be a LayeredSpec or an AdjacencySpec",
    ) as caught:
        canonical_edges(object())

    message = str(caught.value)
    assert "LayeredSpec" in message
    assert "AdjacencySpec" in message


def test_only_ones_count_as_edges():
    spec = parse_layered(_chain_edgelist())
    spec.hops[0].mask[0, 0] = 0.5
    spec.hops[1].mask[0, 0] = 2.0

    edges = canonical_edges(spec)

    assert edges == ()
    assert _n_ones_layered(spec) == 0


def test_serialize_module_does_not_import_parsers():
    source = inspect.getsource(serialize_mod)
    assert "parse_layered" not in source
    assert "parse_adjacency" not in source
