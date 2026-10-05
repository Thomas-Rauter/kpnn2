import pandas as pd
import pytest

from kpnn2 import (
    AdjacencySpec,
    Kpnn2Error,
    LayeredSpec,
    parse_adjacency,
    parse_layered,
)
from kpnn2._parse import _validate_edgelist


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
    ) as exc_info:
        _validate_edgelist(edgelist)

    assert "a -> b" in str(exc_info.value)


def test_validate_edgelist_rejects_two_duplicate_pairs():
    edgelist = pd.DataFrame(
        {
            "source": ["z", "x", "x", "x", "z"],
            "target": ["w", "y", "y", "y", "w"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="3 duplicate edge",
    ) as exc_info:
        _validate_edgelist(edgelist)

    message = str(exc_info.value)
    assert "x -> y" in message
    assert "z -> w" in message
    assert "x -> y, z -> w" in message


def test_validate_edgelist_rejects_duplicate_self_loops_as_duplicates():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "A"],
            "target": ["A", "A"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="1 duplicate edge",
    ) as exc_info:
        _validate_edgelist(edgelist)

    message = str(exc_info.value)
    assert "A -> A" in message
    assert "self-loop" not in message


def _whitespace_collision_edgelist():
    """
    ``"A"`` and ``" A"`` both feed ``Y``: one node spelled twice.
    """
    return pd.DataFrame(
        {
            "source": ["A", " A"],
            "target": ["Y", "Y"],
        }
    )


def test_validate_edgelist_rejects_whitespace_collision_in_source():
    with pytest.raises(
        Kpnn2Error,
        match="differ only by leading or trailing whitespace",
    ) as exc_info:
        _validate_edgelist(_whitespace_collision_edgelist())

    assert "' A', 'A'" in str(exc_info.value)


def test_validate_edgelist_rejects_whitespace_collision_across_columns():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H "],
            "target": ["H", "Y"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="differ only by leading or trailing whitespace",
    ) as exc_info:
        _validate_edgelist(edgelist)

    assert "'H', 'H '" in str(exc_info.value)


def test_validate_edgelist_lists_every_whitespace_group_in_order():
    edgelist = pd.DataFrame(
        {
            "source": ["Z ", "B", "Z", " B", "B "],
            "target": ["Y", "Y", "Y", "Y", "Y"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="differ only by leading or trailing whitespace",
    ) as exc_info:
        _validate_edgelist(edgelist)

    message = str(exc_info.value)
    assert "' B', 'B', 'B '; 'Z', 'Z '." in message
    assert "'Y'" not in message


@pytest.mark.parametrize(
    "padded",
    [
        pytest.param(
            "A\t",
            id="tab",
        ),
        pytest.param(
            "\nA",
            id="newline",
        ),
        pytest.param(
            "A\xa0",
            id="non-breaking-space",
        ),
    ],
)
def test_validate_edgelist_rejects_non_space_whitespace_collision(padded):
    edgelist = pd.DataFrame(
        {
            "source": ["A", padded],
            "target": ["Y", "Y"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="differ only by leading or trailing whitespace",
    ) as exc_info:
        _validate_edgelist(edgelist)

    assert repr(padded) in str(exc_info.value)


def test_validate_edgelist_whitespace_message_quotes_names_and_names_fix():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "A "],
            "target": ["Y", "Y"],
        }
    )

    with pytest.raises(Kpnn2Error) as exc_info:
        _validate_edgelist(edgelist)

    message = str(exc_info.value)
    assert "'A', 'A '" in message
    assert 'edgelist["source"].str.strip()' in message
    assert "distinct names" in message


@pytest.mark.parametrize(
    ("source", "target"),
    [
        pytest.param(
            [" A", "B"],
            ["Y", "Y"],
            id="lone-padded-name",
        ),
        pytest.param(
            ["a", "A"],
            ["Y", "Y"],
            id="case-only",
        ),
        pytest.param(
            ["A B", "A  B"],
            ["Y", "Y"],
            id="internal-whitespace",
        ),
        pytest.param(
            [" ", "B"],
            ["Y", "Y"],
            id="lone-whitespace-only-name",
        ),
    ],
)
def test_validate_edgelist_accepts_names_without_whitespace_collision(
    source,
    target,
):
    edgelist = pd.DataFrame(
        {
            "source": source,
            "target": target,
        }
    )

    normalized = _validate_edgelist(edgelist)

    assert normalized["source"].tolist() == source
    assert normalized["target"].tolist() == target


def test_validate_edgelist_rejects_two_whitespace_only_names():
    edgelist = pd.DataFrame(
        {
            "source": [" ", "\t"],
            "target": ["Y", "Y"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="differ only by leading or trailing whitespace",
    ) as exc_info:
        _validate_edgelist(edgelist)

    assert "'\\t', ' '" in str(exc_info.value)


def test_validate_edgelist_whitespace_check_leaves_caller_frame_unchanged():
    edgelist = _whitespace_collision_edgelist()
    expected = edgelist.copy()

    with pytest.raises(Kpnn2Error):
        _validate_edgelist(edgelist)

    pd.testing.assert_frame_equal(
        edgelist,
        expected,
    )

    lone = pd.DataFrame(
        {
            "source": [" A"],
            "target": ["Y"],
        }
    )
    lone_expected = lone.copy()
    _validate_edgelist(lone)
    pd.testing.assert_frame_equal(
        lone,
        lone_expected,
    )


def _layered_payload(edgelist):
    return {
        "kpnn2_spec": 1,
        "layout": "layered",
        "edges": edgelist.values.tolist(),
    }


def _adjacency_payload(edgelist):
    return {
        "kpnn2_spec": 1,
        "layout": "adjacency",
        "edges": edgelist.values.tolist(),
    }


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            parse_layered,
            id="parse_layered",
        ),
        pytest.param(
            parse_adjacency,
            id="parse_adjacency",
        ),
        pytest.param(
            lambda edgelist: LayeredSpec.from_dict(
                _layered_payload(edgelist),
            ),
            id="LayeredSpec.from_dict",
        ),
        pytest.param(
            lambda edgelist: AdjacencySpec.from_dict(
                _adjacency_payload(edgelist),
            ),
            id="AdjacencySpec.from_dict",
        ),
    ],
)
def test_public_entry_points_reject_whitespace_collision(build):
    edgelist = _whitespace_collision_edgelist()
    expected = edgelist.copy()

    with pytest.raises(
        Kpnn2Error,
        match="differ only by leading or trailing whitespace",
    ) as exc_info:
        build(edgelist)

    assert "' A', 'A'" in str(exc_info.value)
    pd.testing.assert_frame_equal(
        edgelist,
        expected,
    )
