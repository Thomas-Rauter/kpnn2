import pandas as pd
import torch

from kpnn2 import parse_layered


def _chain_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )


def _one_skip_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )


def _three_hop_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "B", "H1", "H2", "A", "H1", "A"],
            "target": ["H1", "H1", "H2", "C", "H2", "C", "C"],
        }
    )


def _edge_pairs(edgelist):
    return list(
        zip(
            edgelist["source"].tolist(),
            edgelist["target"].tolist(),
            strict=True,
        )
    )


def _mask_entry(
    spec,
    hop,
    source,
    target,
):
    row = spec.layer_nodes[hop.target_layer].index(target)
    column = hop.source_nodes.index(source)
    return hop.mask[row, column].item()


def test_parse_layered_hops_chain_shapes_and_dtype():
    spec = parse_layered(_chain_edgelist())

    assert len(spec.hops) == 2
    ones = torch.tensor(
        [[1.0]],
        dtype=torch.float32,
    )
    for index, hop in enumerate(spec.hops):
        assert hop.target_layer == index + 1
        assert hop.mask.shape == (1, 1)
        assert hop.mask.dtype == torch.float32
        assert torch.equal(
            hop.mask,
            ones,
        )


def test_parse_layered_hops_without_skips_read_one_layer():
    spec = parse_layered(_chain_edgelist())

    for index, hop in enumerate(spec.hops):
        assert hop.source_layers == (index,)
        assert hop.source_dims == (spec.layer_dims[index],)
        assert hop.source_nodes == spec.layer_nodes[index]
        assert hop.column_offsets == (0,)
    assert spec.skips == ()


def test_parse_layered_hops_include_the_skip_as_a_column():
    spec = parse_layered(_one_skip_edgelist())

    assert spec.layer_nodes == (("A",), ("H",), ("C",))
    assert len(spec.hops) == 2

    first, second = spec.hops
    assert first.source_layers == (0,)
    assert first.source_nodes == ("A",)
    assert first.mask.tolist() == [[1.0]]

    assert second.target_layer == 2
    assert second.source_layers == (0, 1)
    assert second.source_dims == (1, 1)
    assert second.source_nodes == ("A", "H")
    assert second.column_offsets == (0, 1)
    assert second.mask.shape == (1, 2)
    assert second.mask.tolist() == [[1.0, 1.0]]


def test_parse_layered_hops_first_hop_reads_only_the_input_layer():
    for edgelist in (
        _chain_edgelist(),
        _one_skip_edgelist(),
        _three_hop_edgelist(),
    ):
        spec = parse_layered(edgelist)
        first = spec.hops[0]
        assert first.source_layers == (0,)
        assert first.source_nodes == spec.input_nodes
        assert first.mask.shape[1] == len(spec.input_nodes)


def test_parse_layered_hops_always_read_the_layer_below():
    spec = parse_layered(_three_hop_edgelist())

    for hop in spec.hops:
        assert hop.target_layer - 1 in hop.source_layers
        assert hop.source_layers == tuple(sorted(hop.source_layers))
        assert all(layer < hop.target_layer for layer in hop.source_layers)


def test_parse_layered_hops_cover_every_edge_exactly_once():
    edgelist = _three_hop_edgelist()
    spec = parse_layered(edgelist)

    n_ones = sum(int(hop.mask.sum().item()) for hop in spec.hops)
    assert n_ones == len(edgelist)

    for source, target in _edge_pairs(edgelist):
        found = [
            hop
            for hop in spec.hops
            if target in spec.layer_nodes[hop.target_layer]
        ]
        assert len(found) == 1
        assert (
            _mask_entry(
                spec,
                found[0],
                source,
                target,
            )
            == 1.0
        )


def test_parse_layered_hops_absent_pair_stays_zero():
    spec = parse_layered(_three_hop_edgelist())
    hop = spec.hops[1]

    assert hop.source_nodes == ("A", "B", "H1")
    assert (
        _mask_entry(
            spec,
            hop,
            "B",
            "H2",
        )
        == 0.0
    )


def test_parse_layered_hops_row_degree_counts_skip_parents():
    spec = parse_layered(_three_hop_edgelist())
    last = spec.hops[-1]

    assert last.target_layer == 3
    assert last.source_layers == (0, 1, 2)
    assert last.source_nodes == ("A", "B", "H1", "H2")
    assert last.mask.tolist() == [[1.0, 0.0, 1.0, 1.0]]
    assert last.mask.sum(dim=1).tolist() == [3.0]


def test_parse_layered_hops_early_output_indexing():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "A", "H"],
            "target": ["H", "E", "C"],
        }
    )

    spec = parse_layered(edgelist)

    assert spec.layer_nodes == (("A",), ("E", "H"), ("C",))
    first, second = spec.hops
    assert first.source_layers == (0,)
    assert first.mask.shape == (2, 1)
    assert first.mask.tolist() == [[1.0], [1.0]]
    assert second.source_layers == (1,)
    assert second.source_nodes == ("E", "H")
    assert second.mask.shape == (1, 2)
    assert second.mask.tolist() == [[0.0, 1.0]]


def test_parse_layered_hops_skip_out_of_a_wide_layer():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B", "H1", "A"],
            "target": ["H1", "H1", "H2", "H2"],
        }
    )

    spec = parse_layered(edgelist)

    assert spec.layer_nodes == (("A", "B"), ("H1",), ("H2",))
    assert spec.hops[0].source_layers == (0,)
    assert spec.hops[0].mask.tolist() == [[1.0, 1.0]]

    hop = spec.hops[1]
    assert hop.source_layers == (0, 1)
    assert hop.source_dims == (2, 1)
    assert hop.source_nodes == ("A", "B", "H1")
    assert hop.column_offsets == (0, 2)
    assert hop.mask.tolist() == [[1.0, 0.0, 1.0]]
