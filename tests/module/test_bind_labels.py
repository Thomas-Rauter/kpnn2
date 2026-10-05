import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from kpnn2 import Kpnn2Error
from kpnn2._aggregation._bind import bind_labels


def _attributions():
    return xr.DataArray(
        np.zeros((3, 1)),
        dims=("observation", "node"),
        coords={
            "observation": ["a", "b", "c"],
            "node": ["n"],
        },
    )


def test_series_with_repeated_index_raises_kpnn2_error():
    labels = pd.Series(
        [0, 1, 2, 3, 4],
        index=["a", "b", "b", "c", "c"],
    )

    with pytest.raises(Kpnn2Error) as excinfo:
        bind_labels(
            _attributions(),
            labels,
        )

    message = str(excinfo.value)
    assert "'labels' index must be unique" in message
    assert "Repeated: 'b', 'c'." in message


def test_series_is_reindexed_to_observation_order():
    labels = pd.Series(
        [2, 0, 1],
        index=["c", "a", "b"],
    )

    bound = bind_labels(
        _attributions(),
        labels,
    )

    np.testing.assert_array_equal(
        bound.coords["label"].values,
        [0, 1, 2],
    )
    assert list(labels.index) == ["c", "a", "b"]


@pytest.mark.parametrize(
    ("index", "missing", "extra"),
    [
        (["a", "b"], "'c'", "(none)"),
        (["a", "b", "c", "d"], "(none)", "'d'"),
    ],
)
def test_series_with_missing_or_extra_index_raises(
    index,
    missing,
    extra,
):
    labels = pd.Series(
        np.arange(len(index)),
        index=index,
    )

    with pytest.raises(Kpnn2Error) as excinfo:
        bind_labels(
            _attributions(),
            labels,
        )

    assert str(excinfo.value) == (
        "'labels' index does not match the 'observation' "
        f"coordinate. Missing: {missing}. Extra: {extra}."
    )


def test_missing_labels_message_names_the_argument_to_pass():
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'labels' is required for this aggregation method. Pass "
            "labels= with one label per observation."
        ),
    ):
        bind_labels(
            _attributions(),
            None,
        )


def test_two_dimensional_labels_message_reports_the_shape():
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("'labels' must be 1-dimensional. Got shape (3, 1)."),
    ):
        bind_labels(
            _attributions(),
            np.zeros((3, 1)),
        )


def test_missing_observation_dim_message_reports_the_dims():
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "Attribution DataArray must have an 'observation' dim. "
            "Got dims ('sample', 'node')."
        ),
    ):
        bind_labels(
            _attributions().rename(observation="sample"),
            [0, 1, 2],
        )
