import pandas as pd
import pytest

from kpnn2.errors import Kpnn2Error
from kpnn2.parse_edgelist import _validate_edgelist


def test_validate_edgelist_returns_string_source_target_copy():
    edgelist = pd.DataFrame(
        {
            "source": ["a", 1],
            "target": ["b", 2],
            "extra": [10, 20],
        }
    )

    normalized = _validate_edgelist(edgelist)

    assert list(normalized.columns) == ["source", "target"]
    assert normalized["source"].tolist() == ["a", "1"]
    assert normalized["target"].tolist() == ["b", "2"]
    assert "extra" in edgelist.columns


def test_validate_edgelist_rejects_non_dataframe():
    with pytest.raises(
        Kpnn2Error,
        match="must be a pandas DataFrame",
    ):
        _validate_edgelist(
            [["a", "b"]],
        )


def test_validate_edgelist_rejects_missing_columns():
    edgelist = pd.DataFrame({"source": ["a"]})

    with pytest.raises(
        Kpnn2Error,
        match="Missing: target",
    ):
        _validate_edgelist(edgelist)


def test_validate_edgelist_rejects_empty_table():
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
        _validate_edgelist(edgelist)


def test_validate_edgelist_rejects_missing_values():
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
        _validate_edgelist(edgelist)


def test_validate_edgelist_rejects_empty_names():
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
        _validate_edgelist(edgelist)


def test_validate_edgelist_rejects_duplicate_edges():
    edgelist = pd.DataFrame(
        {
            "source": ["a", "a"],
            "target": ["b", "b"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="1 duplicate edge",
    ):
        _validate_edgelist(edgelist)
