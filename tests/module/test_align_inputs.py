import numpy as np
import pandas as pd
import pytest
import torch

from kpnn2 import (
    AdjacencySpec,
    Kpnn2Error,
    align_inputs,
    parse_adjacency,
    parse_layered,
)


def _tiny_spec():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B", "H"],
            "target": ["H", "H", "C"],
        }
    )
    return parse_layered(edgelist)


def _cyclic_edgelist():
    """
    Inputs A and B feed a two-node feedback core; H feeds C.
    """
    return pd.DataFrame(
        {
            "source": ["A", "B", "H", "K", "H"],
            "target": ["H", "K", "K", "H", "C"],
        }
    )


def _tiny_adjacency_spec():
    return parse_adjacency(_cyclic_edgelist())


def _gather_values(values, names, spec):
    col = align_inputs(
        names,
        spec,
    )
    return values[:, col]


def test_align_inputs_reorders_list_names():
    spec = _tiny_spec()
    assert spec.input_nodes == ("A", "B")
    names = ["B", "A"]
    values = np.array(
        [
            [2.0, 1.0],
            [4.0, 3.0],
        ]
    )

    col = align_inputs(
        names,
        spec,
    )

    assert col.dtype == np.int64
    assert col.tolist() == [1, 0]
    gathered = _gather_values(
        values,
        names,
        spec,
    )
    expected = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    np.testing.assert_array_equal(
        gathered,
        expected,
    )


def test_align_inputs_ignores_extra_names():
    spec = _tiny_spec()
    names = ["extra", "B", "A"]
    values = np.array(
        [
            [9.0, 2.0, 1.0],
            [8.0, 4.0, 3.0],
        ]
    )

    col = align_inputs(
        names,
        spec,
    )

    assert col.tolist() == [2, 1]
    gathered = values[:, col]
    expected = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    np.testing.assert_array_equal(
        gathered,
        expected,
    )


def test_align_inputs_accepts_dataframe_columns():
    spec = _tiny_spec()
    data = pd.DataFrame(
        {
            "B": [2.0, 4.0],
            "A": [1.0, 3.0],
        }
    )

    col = align_inputs(
        data.columns,
        spec,
    )

    gathered = torch.as_tensor(
        data.to_numpy()[:, col],
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(
        gathered,
        expected,
    )


def test_align_inputs_accepts_numpy_names():
    spec = _tiny_spec()
    names = np.array(["B", "A"])
    col = align_inputs(
        names,
        spec,
    )
    assert col.tolist() == [1, 0]


def test_align_inputs_matches_integer_labels_after_str():
    edgelist = pd.DataFrame(
        {
            "source": ["1", "H"],
            "target": ["H", "C"],
        }
    )
    spec = parse_layered(edgelist)
    col = align_inputs(
        [1],
        spec,
    )
    assert col.tolist() == [0]


def test_align_inputs_rejects_missing_name():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match="missing required",
    ):
        align_inputs(
            ["A"],
            spec,
        )


def _tensor_reject_cases():
    return [
        pytest.param(
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                dtype=torch.float32,
            ),
            id="matching_2d",
        ),
        pytest.param(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                dtype=torch.float32,
            ),
            id="wrong_width",
        ),
        pytest.param(
            torch.tensor([1.0, 2.0]),
            id="not_2d",
        ),
    ]


@pytest.mark.parametrize(
    "names",
    _tensor_reject_cases(),
)
def test_align_inputs_rejects_tensor(names):
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match=r"tensor.*feature names",
    ):
        align_inputs(
            names,
            spec,
        )


def test_align_inputs_rejects_dataframe_values():
    spec = _tiny_spec()
    data = pd.DataFrame(
        {
            "A": [1.0],
            "B": [2.0],
        }
    )
    with pytest.raises(
        Kpnn2Error,
        match=r"DataFrame.*feature names",
    ):
        align_inputs(
            data,
            spec,
        )


class _FakeAnnData:
    var_names = pd.Index(["A", "B"])
    X = np.array([[1.0, 2.0]])


def test_align_inputs_rejects_anndata_like():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match=r"AnnData.*var_names",
    ):
        align_inputs(
            _FakeAnnData(),
            spec,
        )


def test_align_inputs_rejects_2d_numpy_matrix():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match="one-dimensional",
    ):
        align_inputs(
            np.array([[1.0, 2.0]]),
            spec,
        )


def _invalid_align_cases():
    spec = _tiny_spec()
    names = ["A", "B"]
    return [
        pytest.param(
            names,
            object(),
            "LayeredSpec",
            id="non_layered_spec",
        ),
        pytest.param(
            object(),
            spec,
            "Unsupported names type",
            id="unsupported_type",
        ),
        pytest.param(
            ["A", "B", "A"],
            spec,
            "duplicate",
            id="duplicate_labels",
        ),
        pytest.param(
            ["A", "B", 1, "1"],
            spec,
            "duplicate",
            id="duplicate_after_str",
        ),
        pytest.param(
            "AB",
            spec,
            "string",
            id="string",
        ),
        pytest.param(
            b"AB",
            spec,
            "bytes",
            id="bytes",
        ),
        pytest.param(
            {"A": 0, "B": 1},
            spec,
            "mapping",
            id="mapping",
        ),
        pytest.param(
            {"A", "B"},
            spec,
            "set",
            id="set",
        ),
    ]


@pytest.mark.parametrize(
    "names, spec, match",
    _invalid_align_cases(),
)
def test_align_inputs_rejects_invalid_names_and_spec(
    names,
    spec,
    match,
):
    with pytest.raises(
        Kpnn2Error,
        match=match,
    ):
        align_inputs(
            names,
            spec,
        )


def test_align_inputs_duplicate_labels_name_a():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match="duplicate",
    ) as caught:
        align_inputs(
            ["A", "B", "A"],
            spec,
        )

    message = str(caught.value)
    assert "A" in message
    assert "converting labels to strings" in message


def test_align_inputs_duplicate_after_str_names_label_1():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match="duplicate",
    ) as caught:
        align_inputs(
            ["A", "B", 1, "1"],
            spec,
        )

    message = str(caught.value)
    assert "1" in message
    assert "converting labels to strings" in message


def test_align_inputs_duplicate_labels_sorted_comma_separated():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match="duplicate",
    ) as caught:
        align_inputs(
            ["A", "B", "A", 1, "1"],
            spec,
        )

    message = str(caught.value)
    assert "1, A" in message
    assert "converting labels to strings" in message


def test_align_inputs_rejects_bytes():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match=r"bytes.*feature names",
    ) as caught:
        align_inputs(
            b"AB",
            spec,
        )

    message = str(caught.value)
    assert "string" not in message


def test_align_inputs_empty_names_when_no_input_nodes():
    spec = AdjacencySpec(
        nodes=(),
        input_nodes=(),
        output_nodes=(),
        hidden_nodes=(),
        source_index=(),
        target_index=(),
        input_index=(),
        output_index=(),
    )
    col = align_inputs(
        [],
        spec,
    )
    assert col.dtype == np.int64
    assert col.tolist() == []
    unused = align_inputs(
        ["extra"],
        spec,
    )
    assert unused.dtype == np.int64
    assert unused.tolist() == []


def test_align_inputs_accepts_adjacency_spec():
    spec = _tiny_adjacency_spec()
    assert spec.input_nodes == ("A", "B")
    names = ["A", "B"]
    values = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    col = align_inputs(
        names,
        spec,
    )

    assert col.dtype == np.int64
    assert col.tolist() == [0, 1]
    gathered = values[:, col]
    expected = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    np.testing.assert_array_equal(
        gathered,
        expected,
    )


def test_align_inputs_reorders_names_for_adjacency_spec():
    spec = _tiny_adjacency_spec()
    names = ["extra", "B", "A"]
    values = np.array(
        [
            [9.0, 2.0, 1.0],
            [8.0, 4.0, 3.0],
        ]
    )

    col = align_inputs(
        names,
        spec,
    )

    assert col.tolist() == [2, 1]
    gathered = values[:, col]
    expected = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    np.testing.assert_array_equal(
        gathered,
        expected,
    )


def test_align_inputs_adjacency_width_is_narrower_than_state():
    spec = _tiny_adjacency_spec()
    names = ["A", "B"]
    values = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    col = align_inputs(
        names,
        spec,
    )
    gathered = torch.as_tensor(
        values[:, col],
        dtype=torch.float32,
    )

    n_nodes = len(spec.nodes)
    assert spec.hidden_nodes == ("H", "K")
    assert n_nodes == len(spec.nodes)
    assert col.shape == (len(spec.input_nodes),)
    assert col.shape[0] < n_nodes

    state = torch.zeros(
        gathered.shape[0],
        n_nodes,
    )
    state[:, spec.input_index] = gathered
    assert state[:, spec.input_index].tolist() == gathered.tolist()


def test_align_inputs_repeats_a_wide_layered_input():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
    )
    spec = parse_layered(
        edgelist,
        widths={"A": 3},
    )
    assert spec.layer_dims[0] == 3
    col = align_inputs(
        ["A"],
        spec,
    )
    assert col.tolist() == [0, 0, 0]
    values = np.array([[1.0], [2.0]])
    gathered = values[:, col]
    assert tuple(gathered.shape) == (2, spec.layer_dims[0])
    assert gathered.tolist() == [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]


def _invalid_adjacency_align_cases():
    spec = _tiny_adjacency_spec()
    tensor = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    frame = pd.DataFrame(
        {
            "A": [1.0],
            "B": [2.0],
        }
    )
    return [
        pytest.param(
            ["A"],
            spec,
            "missing required",
            id="missing_name",
        ),
        pytest.param(
            ["A", "B", "A"],
            spec,
            "duplicate",
            id="duplicate_labels",
        ),
        pytest.param(
            tensor,
            spec,
            r"tensor.*feature names",
            id="tensor",
        ),
        pytest.param(
            frame,
            spec,
            r"DataFrame.*feature names",
            id="dataframe",
        ),
        pytest.param(
            object(),
            spec,
            "Unsupported names type",
            id="unsupported_type",
        ),
    ]


@pytest.mark.parametrize(
    "names, spec, match",
    _invalid_adjacency_align_cases(),
)
def test_align_inputs_rejects_invalid_names_for_adjacency_spec(
    names,
    spec,
    match,
):
    with pytest.raises(
        Kpnn2Error,
        match=match,
    ):
        align_inputs(
            names,
            spec,
        )


def test_align_inputs_spec_error_names_both_spec_types():
    with pytest.raises(
        Kpnn2Error,
        match="LayeredSpec",
    ) as caught:
        align_inputs(
            ["A", "B"],
            object(),
        )

    message = str(caught.value)
    assert "LayeredSpec" in message
    assert "AdjacencySpec" in message
