import inspect

import pandas as pd
import pytest
import torch

from kpnn2 import (
    Kpnn2Error,
    gather_hop_inputs,
    parse_adjacency,
    parse_layered,
)


def _chain_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
    )


def _unequal_sibling_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "A", "Mid"],
            "target": ["Short", "Mid", "Long"],
        }
    )


def _unequal_sibling_ranks():
    return {
        "A": 0,
        "Mid": 1,
        "Short": 2,
        "Long": 2,
    }


def _skip_only_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "A"],
            "target": ["B", "C"],
        }
    )


def test_ranks_none_matches_default_longest_path():
    edgelist = _chain_edgelist()
    default = parse_layered(edgelist)
    omitted = parse_layered(
        edgelist,
        ranks=None,
    )

    assert omitted.layer_nodes == default.layer_nodes
    assert omitted.layer_dims == default.layer_dims
    for left, right in zip(
        default.hops,
        omitted.hops,
        strict=True,
    ):
        assert left.source_layers == right.source_layers
        assert left.source_index == right.source_index
        assert left.target_index == right.target_index
    assert set(default.to_dict()) == {
        "kpnn2_spec",
        "layout",
        "edges",
    }
    assert "ranks" not in default.to_dict()
    assert default.fingerprint == omitted.fingerprint


def test_unequal_siblings_share_a_layer_with_equal_ranks():
    edgelist = _unequal_sibling_edgelist()
    default = parse_layered(edgelist)
    ranked = parse_layered(
        edgelist,
        ranks=_unequal_sibling_ranks(),
    )

    assert default.layer_nodes == (
        ("A",),
        ("Mid", "Short"),
        ("Long",),
    )
    assert default.layer_dims == (1, 2, 1)
    assert ranked.layer_nodes == (
        ("A",),
        ("Mid",),
        ("Long", "Short"),
    )
    assert ranked.layer_dims == (1, 1, 2)
    assert ranked.hops[0].source_layers == (0,)
    assert ranked.hops[0].target_layer == 1
    assert ranked.hops[1].source_layers == (0, 1)
    assert ranked.hops[1].target_layer == 2
    assert ranked.hops[1].source_nodes == ("A", "Mid")
    assert 0 not in default.hops[1].source_layers


def test_user_ranks_are_compacted_to_dense_layers():
    spec = parse_layered(
        _chain_edgelist(),
        ranks={
            "A": 0,
            "H": 10,
            "C": 20,
        },
    )
    default = parse_layered(_chain_edgelist())

    assert spec.layer_nodes == (("A",), ("H",), ("C",))
    assert spec.layer_nodes == default.layer_nodes
    assert spec.hops[0].target_layer == 1
    assert spec.hops[1].target_layer == 2
    assert "ranks" not in spec.to_dict()
    assert spec.fingerprint == default.fingerprint


def test_same_rank_edge_raises():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
    )
    with pytest.raises(
        Kpnn2Error,
        match="strictly forward",
    ) as caught:
        parse_layered(
            edgelist,
            ranks={
                "A": 0,
                "H": 1,
                "C": 1,
            },
        )
    assert "H -> C" in str(caught.value)


def test_backward_edge_raises():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
    )
    with pytest.raises(
        Kpnn2Error,
        match="strictly forward",
    ) as caught:
        parse_layered(
            edgelist,
            ranks={
                "A": 0,
                "H": 2,
                "C": 1,
            },
        )
    assert "H -> C" in str(caught.value)


def test_missing_rank_names_raise_sorted():
    with pytest.raises(
        Kpnn2Error,
        match="Missing node name",
    ) as caught:
        parse_layered(
            _chain_edgelist(),
            ranks={"A": 0},
        )
    assert "C, H" in str(caught.value)


def test_unknown_rank_names_raise_sorted():
    with pytest.raises(
        Kpnn2Error,
        match="Unknown node name",
    ) as caught:
        parse_layered(
            _chain_edgelist(),
            ranks={
                "A": 0,
                "H": 1,
                "C": 2,
                "Z": 3,
                "Q": 3,
            },
        )
    assert "Q, Z" in str(caught.value)


def test_bool_and_non_int_ranks_raise():
    edgelist = _chain_edgelist()
    complete = {
        "A": 0,
        "H": 1,
        "C": 2,
    }
    for value in (True, False, -1, 1.5, "0"):
        ranks = dict(complete)
        ranks["H"] = value
        with pytest.raises(
            Kpnn2Error,
            match="non-negative int",
        ):
            parse_layered(
                edgelist,
                ranks=ranks,
            )


def test_ranks_rejects_non_mapping():
    with pytest.raises(
        Kpnn2Error,
        match="mapping of node name",
    ):
        parse_layered(
            _chain_edgelist(),
            ranks=["A"],
        )


def test_inputs_not_all_at_min_rank_raise():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["C", "C"],
        }
    )
    with pytest.raises(
        Kpnn2Error,
        match="minimum rank",
    ) as caught:
        parse_layered(
            edgelist,
            ranks={
                "A": 0,
                "B": 1,
                "C": 2,
            },
        )
    message = str(caught.value)
    assert "B" in message
    assert "input" in message


def test_hidden_node_at_min_rank_raises():
    edgelist = _chain_edgelist()
    with pytest.raises(
        Kpnn2Error,
        match="minimum rank",
    ) as caught:
        parse_layered(
            edgelist,
            ranks={
                "A": 0,
                "H": 0,
                "C": 1,
            },
        )
    message = str(caught.value)
    assert "H" in message
    assert "non-input" in message


def test_skip_only_parent_omits_previous_layer():
    spec = parse_layered(
        _skip_only_edgelist(),
        ranks={
            "A": 0,
            "B": 1,
            "C": 2,
        },
    )
    default = parse_layered(_skip_only_edgelist())

    assert default.layer_nodes == (("A",), ("B", "C"))
    assert spec.layer_nodes == (("A",), ("B",), ("C",))
    hop = spec.hops[1]
    assert hop.target_layer == 2
    assert hop.source_layers == (0,)
    assert 1 not in hop.source_layers
    assert spec.skips[0].source == "A"
    assert spec.skips[0].target == "C"

    saved = {
        0: torch.tensor([[3.0]]),
        1: torch.tensor([[9.0]]),
    }
    gathered = gather_hop_inputs(
        saved,
        hop,
    )
    assert gathered is saved[0]
    gathered_without_adjacent = gather_hop_inputs(
        {0: saved[0]},
        hop,
    )
    assert gathered_without_adjacent is saved[0]


def test_widths_and_ranks_together():
    spec = parse_layered(
        _unequal_sibling_edgelist(),
        widths={
            "Short": 2,
            "Long": 2,
        },
        ranks=_unequal_sibling_ranks(),
    )

    assert spec.layer_nodes == (
        ("A",),
        ("Mid",),
        ("Long", "Short"),
    )
    assert spec.layer_widths == ((1,), (1,), (2, 2))
    assert spec.layer_dims == (1, 1, 4)
    assert spec.hops[1].source_layers == (0, 1)
    assert spec.hops[1].in_features == 2
    assert spec.hops[1].out_features == 4
    payload = spec.to_dict()
    assert payload["widths"] == {
        "Long": 2,
        "Short": 2,
    }
    assert payload["ranks"] == {
        "A": 0,
        "Mid": 1,
        "Long": 2,
        "Short": 2,
    }


def test_parse_adjacency_has_no_ranks_argument():
    signature = inspect.signature(parse_adjacency)
    assert "ranks" not in signature.parameters
    assert "widths" not in signature.parameters
    edgelist = _chain_edgelist()
    with pytest.raises(TypeError):
        parse_adjacency(
            edgelist,
            ranks={"A": 0, "H": 1, "C": 2},
        )
