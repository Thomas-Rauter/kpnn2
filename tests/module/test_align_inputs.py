import pandas as pd
import pytest
import torch

from kpnn2 import align_inputs, parse_edgelist
from kpnn2.errors import Kpnn2Error


def _tiny_spec():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B", "H"],
            "target": ["H", "H", "C"],
        }
    )
    return parse_edgelist(edgelist)


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
            "GraphSpec",
            id="non_graph_spec",
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
