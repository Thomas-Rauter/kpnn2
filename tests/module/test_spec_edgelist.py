import ast
import inspect

import pandas as pd
import torch

import kpnn2
import kpnn2._adjacency_spec as adjacency_spec_mod
import kpnn2._spec as spec_mod
from kpnn2 import (
    AdjacencySpec,
    LayeredSpec,
    parse_adjacency,
    parse_layered,
)
from kpnn2._serialize import canonical_edges


def _skip_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )


def _unsorted_skip_edgelist():
    return pd.DataFrame(
        {
            "source": ["H", "A", "A"],
            "target": ["C", "C", "H"],
        }
    )


def _multi_skip_edgelist():
    return pd.DataFrame(
        {
            "source": [
                "H2",
                "A",
                "H1",
                "B",
                "A",
                "H1",
                "A",
            ],
            "target": [
                "C",
                "H1",
                "H2",
                "H1",
                "C",
                "C",
                "H2",
            ],
        }
    )


def _cycle_edgelist():
    return pd.DataFrame(
        {
            "source": ["x", "a", "b", "a", "a"],
            "target": ["a", "b", "a", "a", "y"],
        }
    )


def _skip_identity(spec):
    return {
        (
            skip.source,
            skip.target,
            skip.source_layer,
            skip.target_layer,
            skip.source_index,
            skip.target_index,
        )
        for skip in spec.skips
    }


def _assert_layered_structure(
    original,
    roundtrip,
):
    assert roundtrip.input_nodes == original.input_nodes
    assert roundtrip.output_nodes == original.output_nodes
    assert roundtrip.hidden_nodes == original.hidden_nodes
    assert roundtrip.layer_nodes == original.layer_nodes
    assert roundtrip.layer_dims == original.layer_dims
    assert len(roundtrip.hops) == len(original.hops)
    for original_hop, roundtrip_hop in zip(
        original.hops,
        roundtrip.hops,
        strict=True,
    ):
        assert roundtrip_hop.source_layers == original_hop.source_layers
        assert roundtrip_hop.source_dims == original_hop.source_dims
        assert roundtrip_hop.source_nodes == original_hop.source_nodes
        assert torch.equal(
            roundtrip_hop.mask,
            original_hop.mask,
        )
    assert _skip_identity(roundtrip) == _skip_identity(original)


def _assert_adjacency_structure(
    original,
    roundtrip,
):
    assert roundtrip.nodes == original.nodes
    assert roundtrip.input_nodes == original.input_nodes
    assert roundtrip.output_nodes == original.output_nodes
    assert roundtrip.hidden_nodes == original.hidden_nodes
    assert roundtrip.source_index == original.source_index
    assert roundtrip.target_index == original.target_index
    assert torch.equal(
        roundtrip.to_mask(),
        original.to_mask(),
    )
    assert roundtrip.input_index == original.input_index
    assert roundtrip.output_index == original.output_index


def _top_level_imported_modules(module):
    tree = ast.parse(inspect.getsource(module))
    imported = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    return imported


def test_to_edgelist_is_not_a_public_name():
    assert "to_edgelist" not in kpnn2.__all__
    assert not hasattr(
        kpnn2,
        "to_edgelist",
    )
    assert hasattr(
        LayeredSpec,
        "to_edgelist",
    )
    assert hasattr(
        AdjacencySpec,
        "to_edgelist",
    )


def test_layered_skip_graph_round_trip():
    spec = parse_layered(_skip_edgelist())

    roundtrip = parse_layered(spec.to_edgelist())

    _assert_layered_structure(
        spec,
        roundtrip,
    )


def test_adjacency_cycle_round_trip():
    spec = parse_adjacency(_cycle_edgelist())

    roundtrip = parse_adjacency(spec.to_edgelist())

    _assert_adjacency_structure(
        spec,
        roundtrip,
    )


def test_unsorted_layered_edgelist_is_sorted_and_round_trips():
    spec = parse_layered(_unsorted_skip_edgelist())
    table = spec.to_edgelist()

    assert table["source"].tolist() == ["A", "A", "H"]
    assert table["target"].tolist() == ["C", "H", "C"]

    roundtrip = parse_layered(table)
    _assert_layered_structure(
        spec,
        roundtrip,
    )


def test_unsorted_multi_skip_compares_skip_set():
    spec = parse_layered(_multi_skip_edgelist())
    table = spec.to_edgelist()
    roundtrip = parse_layered(table)

    assert list(
        zip(
            table["source"],
            table["target"],
            strict=True,
        )
    ) == list(canonical_edges(spec))
    _assert_layered_structure(
        spec,
        roundtrip,
    )


def test_to_edgelist_has_only_source_and_target():
    wide = pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "weight": [0.1, 0.2, 0.3],
            "target": ["H", "C", "C"],
            "note": ["x", "y", "z"],
        }
    )
    spec = parse_layered(wide)
    table = spec.to_edgelist()

    assert list(table.columns) == ["source", "target"]
    assert "weight" not in table.columns
    assert "note" not in table.columns


def test_to_edgelist_rows_match_canonical_edges():
    layered = parse_layered(_skip_edgelist())
    adjacency = parse_adjacency(_cycle_edgelist())

    layered_table = layered.to_edgelist()
    adjacency_table = adjacency.to_edgelist()

    assert list(
        zip(
            layered_table["source"],
            layered_table["target"],
            strict=True,
        )
    ) == list(canonical_edges(layered))
    assert list(
        zip(
            adjacency_table["source"],
            adjacency_table["target"],
            strict=True,
        )
    ) == list(canonical_edges(adjacency))
    assert all(isinstance(name, str) for name in layered_table["source"])
    assert all(isinstance(name, str) for name in layered_table["target"])


def test_integer_node_names_come_back_as_strings():
    edgelist = pd.DataFrame(
        {
            "source": [1, 2],
            "target": [2, 3],
        }
    )
    spec = parse_layered(edgelist)
    table = spec.to_edgelist()

    assert table["source"].tolist() == ["1", "2"]
    assert table["target"].tolist() == ["2", "3"]


def test_spec_modules_do_not_import_serialize_at_top_level():
    assert "_serialize" not in _top_level_imported_modules(spec_mod)
    assert "_serialize" not in _top_level_imported_modules(adjacency_spec_mod)
    layered_src = inspect.getsource(LayeredSpec.to_edgelist)
    adjacency_src = inspect.getsource(AdjacencySpec.to_edgelist)
    assert "from ._serialize import spec_to_edgelist" in layered_src
    assert "from ._serialize import spec_to_edgelist" in adjacency_src
