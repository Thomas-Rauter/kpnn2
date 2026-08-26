import pandas as pd
import pytest
import torch

from kpnn2 import (
    Kpnn2Error,
    MaskedLinear,
    parse_adjacency,
    parse_layered,
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


def _dag_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
    )


def test_parse_adjacency_accepts_a_cycle_that_parse_layered_rejects():
    edgelist = _cyclic_edgelist()

    spec = parse_adjacency(edgelist)
    assert spec.nodes == ("a", "b", "x", "y")

    with pytest.raises(
        Kpnn2Error,
        match="cycle",
    ):
        parse_layered(edgelist)


def test_parse_adjacency_accepts_a_dag_that_parse_layered_accepts():
    edgelist = _dag_edgelist()

    adjacency = parse_adjacency(edgelist)
    layered = parse_layered(edgelist)

    assert adjacency.nodes == ("A", "C", "H")
    assert adjacency.input_nodes == layered.input_nodes
    assert adjacency.output_nodes == layered.output_nodes
    assert adjacency.hidden_nodes == layered.hidden_nodes


def test_parse_adjacency_accepts_a_self_loop_layered_rejects():
    edgelist = pd.DataFrame(
        {
            "source": ["x", "a", "a"],
            "target": ["a", "a", "y"],
        }
    )

    spec = parse_adjacency(edgelist)
    loop = spec.nodes.index("a")
    assert spec.mask[loop, loop].item() == 1.0

    with pytest.raises(
        Kpnn2Error,
        match="self-loop",
    ):
        parse_layered(edgelist)


def test_parse_adjacency_mask_shape_and_dtype():
    spec = parse_adjacency(_cyclic_edgelist())

    n_nodes = len(spec.nodes)
    assert n_nodes == 4
    assert tuple(spec.mask.shape) == (n_nodes, n_nodes)
    assert spec.mask.dtype == torch.float32


def test_parse_adjacency_mask_uses_target_source_indexing():
    spec = parse_adjacency(_cyclic_edgelist())
    position = {name: i for i, name in enumerate(spec.nodes)}

    # x -> a is asymmetric: only [a, x] is set, never [x, a].
    assert spec.mask[position["a"], position["x"]].item() == 1.0
    assert spec.mask[position["x"], position["a"]].item() == 0.0

    # The feedback pair sets both directions, one per edge.
    assert spec.mask[position["b"], position["a"]].item() == 1.0
    assert spec.mask[position["a"], position["b"]].item() == 1.0

    assert spec.mask[position["y"], position["a"]].item() == 1.0
    assert spec.mask.sum().item() == 4.0


def test_parse_adjacency_input_rows_are_all_zero():
    spec = parse_adjacency(_cyclic_edgelist())

    for index in spec.input_index:
        assert spec.mask[index].sum().item() == 0.0


def test_parse_adjacency_node_roles_are_alphabetical():
    spec = parse_adjacency(_cyclic_edgelist())

    assert spec.nodes == ("a", "b", "x", "y")
    assert spec.input_nodes == ("x",)
    assert spec.output_nodes == ("y",)
    assert spec.hidden_nodes == ("a", "b")
    assert list(spec.nodes) == sorted(spec.nodes)


def test_parse_adjacency_index_tuples_point_into_nodes():
    spec = parse_adjacency(_cyclic_edgelist())

    assert spec.input_index == (2,)
    assert spec.output_index == (3,)
    for name, index in zip(
        spec.input_nodes,
        spec.input_index,
        strict=True,
    ):
        assert spec.nodes[index] == name
    for name, index in zip(
        spec.output_nodes,
        spec.output_index,
        strict=True,
    ):
        assert spec.nodes[index] == name


def test_parse_adjacency_ignores_extra_columns():
    edgelist = _cyclic_edgelist()
    edgelist["weight"] = [0.1, 0.2, 0.3, 0.4]

    spec = parse_adjacency(edgelist)

    assert spec.nodes == ("a", "b", "x", "y")
    assert spec.mask.sum().item() == 4.0


def test_parse_adjacency_converts_names_with_str():
    edgelist = pd.DataFrame(
        {
            "source": [1, 2],
            "target": [2, 3],
        }
    )

    spec = parse_adjacency(edgelist)

    assert spec.nodes == ("1", "2", "3")


def test_parse_adjacency_rejects_non_dataframe():
    with pytest.raises(
        Kpnn2Error,
        match="must be a pandas DataFrame",
    ):
        parse_adjacency([["a", "b"]])


def test_parse_adjacency_rejects_missing_columns():
    edgelist = pd.DataFrame({"source": ["a"]})

    with pytest.raises(
        Kpnn2Error,
        match="Missing: target",
    ):
        parse_adjacency(edgelist)


def test_parse_adjacency_rejects_empty_table():
    edgelist = pd.DataFrame(
        {
            "source": pd.Series(dtype="object"),
            "target": pd.Series(dtype="object"),
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="at least one edge",
    ):
        parse_adjacency(edgelist)


def test_parse_adjacency_rejects_missing_values():
    edgelist = pd.DataFrame(
        {
            "source": ["a", None],
            "target": ["b", "c"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="missing values",
    ):
        parse_adjacency(edgelist)


def test_parse_adjacency_rejects_empty_names():
    edgelist = pd.DataFrame(
        {
            "source": ["a", ""],
            "target": ["b", "c"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="empty node names",
    ):
        parse_adjacency(edgelist)


def test_parse_adjacency_rejects_duplicate_edges():
    edgelist = pd.DataFrame(
        {
            "source": ["a", "a"],
            "target": ["b", "b"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="1 duplicate edge",
    ) as exc_info:
        parse_adjacency(edgelist)

    assert "a -> b" in str(exc_info.value)


def test_parse_adjacency_rejects_a_pure_ring_for_having_no_input():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "A"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="at least one input node",
    ):
        parse_adjacency(edgelist)


def test_parse_adjacency_rejects_a_lone_self_loop():
    edgelist = pd.DataFrame(
        {
            "source": ["A"],
            "target": ["A"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="at least one input node",
    ):
        parse_adjacency(edgelist)


def test_parse_adjacency_rejects_a_graph_with_no_output():
    edgelist = pd.DataFrame(
        {
            "source": ["x", "a", "b"],
            "target": ["a", "b", "a"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="at least one output node",
    ):
        parse_adjacency(edgelist)


def test_parse_adjacency_mask_drives_a_masked_linear():
    spec = parse_adjacency(_cyclic_edgelist())
    layer = MaskedLinear(spec.mask)

    n_nodes = len(spec.nodes)
    assert layer.mask.tolist() == spec.mask.tolist()
    assert layer.weight.shape == (n_nodes, n_nodes)

    state = torch.zeros(
        2,
        n_nodes,
    )
    state[:, spec.input_index] = 1.0
    out = layer(state)

    assert out.shape == (2, n_nodes)


def test_parse_adjacency_does_not_build_a_module():
    spec = parse_adjacency(_cyclic_edgelist())

    assert not isinstance(
        spec,
        torch.nn.Module,
    )
    assert not isinstance(
        spec.mask,
        torch.nn.Module,
    )
