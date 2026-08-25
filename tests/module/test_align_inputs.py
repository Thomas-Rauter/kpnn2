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


def test_align_inputs_accepts_matching_2d_tensor():
    spec = _tiny_spec()
    data = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )

    aligned = align_inputs(
        data,
        spec,
    )

    assert aligned.shape == (2, 2)
    assert aligned.dtype == torch.float32
    assert torch.equal(
        aligned,
        data,
    )


def test_align_inputs_rejects_wrong_tensor_width():
    spec = _tiny_spec()
    data = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(
        Kpnn2Error,
        match="wrong number of features",
    ):
        align_inputs(
            data,
            spec,
        )
