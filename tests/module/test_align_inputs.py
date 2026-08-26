import pandas as pd
import pytest
import torch

from kpnn2 import (
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


def test_align_inputs_reorders_dataframe_columns():
    spec = _tiny_spec()
    assert spec.input_nodes == ("A", "B")
    data = pd.DataFrame(
        {
            "B": [2.0, 4.0],
            "A": [1.0, 3.0],
        }
    )

    aligned = align_inputs(
        data,
        spec,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    assert aligned.dtype == torch.float32
    assert torch.equal(
        aligned,
        expected,
    )


def test_align_inputs_ignores_extra_dataframe_columns():
    spec = _tiny_spec()
    data = pd.DataFrame(
        {
            "extra": [9.0, 8.0],
            "B": [2.0, 4.0],
            "A": [1.0, 3.0],
        }
    )

    aligned = align_inputs(
        data,
        spec,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(
        aligned,
        expected,
    )


def test_align_inputs_rejects_missing_dataframe_column():
    spec = _tiny_spec()
    data = pd.DataFrame(
        {
            "A": [1.0, 3.0],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="missing required feature",
    ):
        align_inputs(
            data,
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
    "data",
    _tensor_reject_cases(),
)
def test_align_inputs_rejects_tensor(data):
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match=r"tensor.*pandas DataFrame",
    ):
        align_inputs(
            data,
            spec,
        )


def _invalid_align_cases():
    spec = _tiny_spec()
    numeric = pd.DataFrame(
        {
            "A": [1.0],
            "B": [2.0],
        }
    )
    duplicate = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        columns=["A", "B", "A"],
    )
    duplicate_after_str = pd.DataFrame(
        {
            "A": [1.0],
            "B": [2.0],
            1: [3.0],
            "1": [4.0],
        }
    )
    non_numeric = pd.DataFrame(
        {
            "A": ["x"],
            "B": [2.0],
        }
    )
    return [
        pytest.param(
            numeric,
            object(),
            "LayeredSpec",
            id="non_layered_spec",
        ),
        pytest.param(
            [[1.0, 2.0]],
            spec,
            "Unsupported input data type",
            id="unsupported_type",
        ),
        pytest.param(
            duplicate,
            spec,
            "duplicate column",
            id="duplicate_columns",
        ),
        pytest.param(
            duplicate_after_str,
            spec,
            "duplicate column",
            id="duplicate_after_str",
        ),
        pytest.param(
            non_numeric,
            spec,
            "non-numeric",
            id="non_numeric",
        ),
    ]


@pytest.mark.parametrize(
    "data, spec, match",
    _invalid_align_cases(),
)
def test_align_inputs_rejects_invalid_data_and_spec(
    data,
    spec,
    match,
):
    with pytest.raises(
        Kpnn2Error,
        match=match,
    ):
        align_inputs(
            data,
            spec,
        )


def test_align_inputs_duplicate_columns_name_label_a():
    spec = _tiny_spec()
    data = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        columns=["A", "B", "A"],
    )

    with pytest.raises(
        Kpnn2Error,
        match="duplicate column",
    ) as caught:
        align_inputs(
            data,
            spec,
        )

    message = str(caught.value)
    assert "A" in message
    assert "converting labels to strings" in message


def test_align_inputs_duplicate_after_str_names_label_1():
    spec = _tiny_spec()
    data = pd.DataFrame(
        {
            "A": [1.0],
            "B": [2.0],
            1: [3.0],
            "1": [4.0],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="duplicate column",
    ) as caught:
        align_inputs(
            data,
            spec,
        )

    message = str(caught.value)
    assert "1" in message
    assert "converting labels to strings" in message


def test_align_inputs_duplicate_labels_sorted_comma_separated():
    spec = _tiny_spec()
    data = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]],
        columns=["A", "B", "A", 1, "1"],
    )

    with pytest.raises(
        Kpnn2Error,
        match="duplicate column",
    ) as caught:
        align_inputs(
            data,
            spec,
        )

    message = str(caught.value)
    assert "1, A" in message
    assert "converting labels to strings" in message


def test_align_inputs_accepts_adjacency_spec():
    spec = _tiny_adjacency_spec()
    assert spec.input_nodes == ("A", "B")
    data = pd.DataFrame(
        {
            "A": [1.0, 3.0],
            "B": [2.0, 4.0],
        }
    )

    aligned = align_inputs(
        data,
        spec,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    assert aligned.dtype == torch.float32
    assert tuple(aligned.shape) == (2, len(spec.input_nodes))
    assert torch.equal(
        aligned,
        expected,
    )


def test_align_inputs_reorders_columns_for_adjacency_spec():
    spec = _tiny_adjacency_spec()
    data = pd.DataFrame(
        {
            "extra": [9.0, 8.0],
            "B": [2.0, 4.0],
            "A": [1.0, 3.0],
        }
    )

    aligned = align_inputs(
        data,
        spec,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(
        aligned,
        expected,
    )


def test_align_inputs_adjacency_width_is_narrower_than_mask():
    spec = _tiny_adjacency_spec()
    data = pd.DataFrame(
        {
            "A": [1.0, 3.0],
            "B": [2.0, 4.0],
        }
    )

    aligned = align_inputs(
        data,
        spec,
    )

    n_nodes = spec.mask.shape[0]
    assert spec.hidden_nodes == ("H", "K")
    assert n_nodes == len(spec.nodes)
    assert aligned.shape[1] == len(spec.input_nodes)
    assert aligned.shape[1] < n_nodes

    # The aligned tensor only reaches the mask after a scatter.
    state = torch.zeros(
        aligned.shape[0],
        n_nodes,
    )
    state[:, spec.input_index] = aligned
    assert state[:, spec.input_index].tolist() == aligned.tolist()


def _invalid_adjacency_align_cases():
    spec = _tiny_adjacency_spec()
    missing = pd.DataFrame(
        {
            "A": [1.0],
        }
    )
    duplicate = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        columns=["A", "B", "A"],
    )
    non_numeric = pd.DataFrame(
        {
            "A": ["x"],
            "B": [2.0],
        }
    )
    tensor = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    return [
        pytest.param(
            missing,
            spec,
            "missing required feature",
            id="missing_column",
        ),
        pytest.param(
            duplicate,
            spec,
            "duplicate column",
            id="duplicate_columns",
        ),
        pytest.param(
            non_numeric,
            spec,
            "non-numeric",
            id="non_numeric",
        ),
        pytest.param(
            tensor,
            spec,
            r"tensor.*pandas DataFrame",
            id="tensor",
        ),
        pytest.param(
            [[1.0, 2.0]],
            spec,
            "Unsupported input data type",
            id="unsupported_type",
        ),
    ]


@pytest.mark.parametrize(
    "data, spec, match",
    _invalid_adjacency_align_cases(),
)
def test_align_inputs_rejects_invalid_data_for_adjacency_spec(
    data,
    spec,
    match,
):
    with pytest.raises(
        Kpnn2Error,
        match=match,
    ):
        align_inputs(
            data,
            spec,
        )


def test_align_inputs_spec_error_names_both_spec_types():
    data = pd.DataFrame(
        {
            "A": [1.0],
            "B": [2.0],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="LayeredSpec",
    ) as caught:
        align_inputs(
            data,
            object(),
        )

    message = str(caught.value)
    assert "LayeredSpec" in message
    assert "AdjacencySpec" in message
