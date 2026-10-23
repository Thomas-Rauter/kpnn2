import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from kpnn2 import Kpnn2Error
from kpnn2._aggregation._bind import (
    bind_labels,
    require_named_dims,
    require_unique_nodes,
)


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

    assert str(excinfo.value) == (
        "'labels' index must be unique to align it to the 'observation' "
        "coordinate.\n"
        "  Got: repeated 'b', 'c'"
    )


def test_repeated_observation_coordinate_raises_for_a_series():
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "The 'observation' coordinate must be unique to align a "
            "pandas Series of labels.\n"
            "  Got: repeated 'a'\n"
        ),
    ):
        bind_labels(
            _attributions().assign_coords(observation=["a", "a", "c"]),
            pd.Series(
                [0, 1, 2],
                index=["a", "b", "c"],
            ),
        )


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
    ("index", "fields"),
    [
        pytest.param(
            ["a", "b"],
            "  Got: missing 'c'",
            id="missing",
        ),
        pytest.param(
            ["a", "b", "c", "d"],
            "  Got: extra 'd'",
            id="extra",
        ),
        pytest.param(
            ["a", "b", "C"],
            "  Got: missing 'c'; extra 'C'\n  Did you mean: 'c'",
            id="case",
        ),
    ],
)
def test_series_with_missing_or_extra_index_raises(
    index,
    fields,
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
        f"'labels' index does not match the 'observation' coordinate.\n{fields}"
    )


def test_missing_labels_message_names_the_argument_to_pass():
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'labels' is required for this aggregation method.\n"
            "  Fix: Pass labels= with one label per observation."
        ),
    ):
        bind_labels(
            _attributions(),
            None,
        )


def test_two_dimensional_labels_message_reports_the_shape():
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("'labels' must be 1-dimensional.\n  Got: shape (3, 1)"),
    ):
        bind_labels(
            _attributions(),
            np.zeros((3, 1)),
        )


@pytest.mark.parametrize(
    ("name", "fix"),
    [
        pytest.param(
            "sample",
            "",
            id="far",
        ),
        pytest.param(
            "observations",
            "\n  Fix: Rename dim 'observations' to 'observation'.",
            id="close",
        ),
    ],
)
def test_missing_observation_dim_message_reports_the_dims(
    name,
    fix,
):
    with pytest.raises(Kpnn2Error) as excinfo:
        bind_labels(
            _attributions().rename(observation=name),
            [0, 1, 2],
        )
    assert str(excinfo.value) == (
        "'attributions' is missing required dim(s): 'observation'.\n"
        f"  Got: dims {name!r}, 'node'{fix}"
    )


@pytest.mark.parametrize(
    ("labels", "got", "fix"),
    [
        pytest.param(
            xr.DataArray(np.zeros(3)),
            "DataArray of shape (3,), dtype float64",
            "Pass labels.to_series(), which aligns the labels by their "
            "coordinate.",
            id="DataArray",
        ),
        pytest.param(
            pd.DataFrame({"label": [0, 1, 2]}),
            "DataFrame of shape (3, 1)",
            "Pass the column that holds the labels, as a Series.",
            id="DataFrame",
        ),
    ],
)
def test_labels_table_or_dataarray_is_rejected(
    labels,
    got,
    fix,
):
    with pytest.raises(Kpnn2Error) as excinfo:
        bind_labels(
            _attributions(),
            labels,
        )
    assert str(excinfo.value) == (
        "'labels' must be a 1-d array or pandas Series, not a "
        f"{type(labels).__name__}.\n"
        f"  Got: {got}\n"
        f"  Fix: {fix}"
    )


def test_labels_length_message_reports_both_lengths():
    with pytest.raises(Kpnn2Error) as excinfo:
        bind_labels(
            _attributions(),
            [0, 1],
        )
    assert str(excinfo.value) == (
        "'labels' length must match the 'observation' dim.\n"
        "  Got: length 2\n"
        "  Expected: length 3"
    )


def test_missing_required_dim_is_reported_with_the_dims():
    with pytest.raises(Kpnn2Error) as excinfo:
        require_named_dims(
            _attributions(),
            required=("observation", "node", "seed"),
        )
    assert str(excinfo.value) == (
        "'attributions' is missing required dim(s): 'seed'.\n"
        "  Got: dims 'observation', 'node'"
    )


def test_missing_and_unexpected_dims_raise_in_one_message():
    with pytest.raises(Kpnn2Error) as excinfo:
        require_named_dims(
            _attributions().rename(observation="observations"),
            required=("observation", "node"),
        )
    assert str(excinfo.value) == (
        "'attributions' has unexpected dim(s) and is missing required "
        "dim(s).\n"
        "  Got: unexpected 'observations'; missing 'observation'\n"
        "  Expected: only dims 'node', 'observation'\n"
        "  Did you mean: 'observation'\n"
        "  Fix: Reduce or rename the unexpected dims first."
    )


def test_unexpected_dim_suggests_a_close_optional_dim():
    with pytest.raises(Kpnn2Error) as excinfo:
        require_named_dims(
            _attributions().expand_dims(seeds=2),
            required=("observation", "node"),
            optional=("seed",),
        )
    assert str(excinfo.value) == (
        "'attributions' has unexpected dim(s).\n"
        "  Got: 'seeds'\n"
        "  Expected: only dims 'node', 'observation', 'seed'\n"
        "  Did you mean: 'seed'\n"
        "  Fix: Reduce or rename the unexpected dims first."
    )


def test_repeated_node_names_raise():
    attributions = _attributions().reindex(node=["n", "n"])
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "The 'node' coordinate repeats node names.\n  Got: 'n'\n"
        ),
    ):
        require_unique_nodes(attributions)


def test_labels_index_sharing_no_value_says_to_check_the_identifier():
    labels = pd.Series(
        [0, 1, 0],
        index=["s1", "s2", "s3"],
    )
    with pytest.raises(Kpnn2Error) as excinfo:
        bind_labels(
            _attributions().assign_coords(observation=[0, 1, 2]),
            labels,
        )
    assert str(excinfo.value) == (
        "'labels' index does not match the 'observation' coordinate.\n"
        "  Got: missing 0, 1, 2; extra 's1', 's2', 's3'\n"
        "  Fix: No index value names an observation; check the kind of "
        "identifier, or name the observations with coords= in "
        "map_node_attributions()."
    )


def test_labels_index_differing_only_by_case_gets_suggestions():
    labels = pd.Series(
        [0, 1, 0],
        index=["A", "B", "C"],
    )
    with pytest.raises(Kpnn2Error) as excinfo:
        bind_labels(
            _attributions(),
            labels,
        )
    assert str(excinfo.value) == (
        "'labels' index does not match the 'observation' coordinate.\n"
        "  Got: missing 'a', 'b', 'c'; extra 'A', 'B', 'C'\n"
        "  Did you mean: 'a' for 'A'; 'b' for 'B'; 'c' for 'C'"
    )


@pytest.mark.parametrize(
    ("labels", "got"),
    [
        pytest.param(
            {0, 1, 2},
            "{0, 1, 2} (set)",
            id="set",
        ),
        pytest.param(
            "abc",
            "'abc' (str)",
            id="str",
        ),
    ],
)
def test_labels_that_convert_to_zero_dims_are_described(
    labels,
    got,
):
    with pytest.raises(Kpnn2Error) as excinfo:
        bind_labels(
            _attributions(),
            labels,
        )
    assert str(excinfo.value) == (
        f"'labels' must be 1-dimensional.\n  Got: {got}"
    )
