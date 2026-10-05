import re
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

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


def test_validate_edgelist_non_dataframe_message_names_its_type():
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'edgelist' must be a pandas DataFrame. Got [['a', 'b']] (list)."
        ),
    ):
        _validate_edgelist(
            [["a", "b"]],
        )


def test_validate_edgelist_missing_column_message_lists_present_columns():
    edgelist = pd.DataFrame(
        {
            "Source": ["a"],
            "target": ["b"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "Missing: source. Present columns: 'Source', 'target'."
        ),
    ):
        _validate_edgelist(edgelist)


def test_validate_edgelist_present_columns_are_capped_at_ten():
    edgelist = pd.DataFrame({f"c{i:02d}": ["a"] for i in range(13)})

    with pytest.raises(
        Kpnn2Error,
        match=re.escape("'c08', 'c09', and 3 more."),
    ) as exc_info:
        _validate_edgelist(edgelist)

    message = str(exc_info.value)
    assert "Present columns: 'c00', 'c01'," in message
    assert "'c10'" not in message


def test_validate_edgelist_frame_without_columns_says_none_present():
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("Missing: source, target. Present columns: none."),
    ):
        _validate_edgelist(pd.DataFrame())


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


_BAD_ROW_CASES = [
    pytest.param(
        None,
        "missing values",
        id="missing-values",
    ),
    pytest.param(
        "",
        "empty node names",
        id="empty-names",
    ),
]


@pytest.mark.parametrize(
    ("bad", "problem"),
    _BAD_ROW_CASES,
)
def test_validate_edgelist_bad_rows_named_by_index_label_and_counted(
    bad,
    problem,
):
    edgelist = pd.DataFrame(
        {
            "source": ["a", bad, "c", bad],
            "target": ["b", "c", "d", "e"],
        },
        index=["r0", "r1", "r2", "r3"],
    )

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            f"Edgelist contains {problem} in 'source' or 'target'. "
            "'source': 2 row(s) at index 'r1', 'r3'."
        ),
    ):
        _validate_edgelist(edgelist)


@pytest.mark.parametrize(
    ("bad", "problem"),
    _BAD_ROW_CASES,
)
def test_validate_edgelist_bad_rows_are_counted_per_column(
    bad,
    problem,
):
    edgelist = pd.DataFrame(
        {
            "source": ["a", bad, "c"],
            "target": ["b", "c", bad],
        },
        index=["r0", "r1", "r2"],
    )

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'source': 1 row(s) at index 'r1'; "
            "'target': 1 row(s) at index 'r2'."
        ),
    ):
        _validate_edgelist(edgelist)


@pytest.mark.parametrize(
    ("bad", "problem"),
    _BAD_ROW_CASES,
)
def test_validate_edgelist_bad_rows_show_only_the_first_five_labels(
    bad,
    problem,
):
    edgelist = pd.DataFrame(
        {
            "source": ["a"] * 7,
            "target": [bad] * 7,
        },
        index=list("abcdefg"),
    )

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'target': 7 row(s) at index 'a', 'b', 'c', 'd', 'e', and 2 more."
        ),
    ) as exc_info:
        _validate_edgelist(edgelist)

    message = str(exc_info.value)
    assert "'f'" not in message
    assert "'g'" not in message


@pytest.mark.parametrize(
    ("bad", "problem"),
    _BAD_ROW_CASES,
)
def test_validate_edgelist_bad_rows_use_labels_not_positions(
    bad,
    problem,
):
    edgelist = pd.DataFrame(
        {
            "source": ["a", bad],
            "target": ["b", "c"],
        },
        index=[10, 20],
    )

    with pytest.raises(
        Kpnn2Error,
        match=re.escape("'source': 1 row(s) at index 20."),
    ):
        _validate_edgelist(edgelist)


@pytest.mark.parametrize(
    "parse",
    [
        pytest.param(
            parse_layered,
            id="parse_layered",
        ),
        pytest.param(
            parse_adjacency,
            id="parse_adjacency",
        ),
    ],
)
def test_parsers_name_bad_rows_by_index_label(parse):
    edgelist = pd.DataFrame(
        {
            "source": ["a", None],
            "target": ["b", "c"],
        },
        index=["first", "second"],
    )

    with pytest.raises(
        Kpnn2Error,
        match=re.escape("'source': 1 row(s) at index 'second'."),
    ):
        parse(edgelist)


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


_ENTRY_POINTS = [
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
]

_NOT_NAMES = (
    "Edgelist 'source' and 'target' must hold node names: strings, "
    "or integers that are read through str(). "
)


class _Node:
    def __repr__(self):
        return "_Node()"


def _with_target(value):
    """Edges a -> c and b -> ``value``, rows labelled r1 and r2."""
    return pd.DataFrame(
        {
            "source": ["a", "b"],
            "target": pd.Series(
                ["c", value],
                index=["r1", "r2"],
                dtype=object,
            ),
        },
        index=["r1", "r2"],
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            [1, 2],
            ["1", "2"],
            id="int",
        ),
        pytest.param(
            np.array(
                [1, 2],
                dtype=np.int64,
            ),
            ["1", "2"],
            id="numpy_int64_column",
        ),
        pytest.param(
            [np.int32(1), "b"],
            ["1", "b"],
            id="numpy_int32_in_object_column",
        ),
        pytest.param(
            pd.array(
                ["a", "b"],
                dtype="string",
            ),
            ["a", "b"],
            id="string_dtype",
        ),
        pytest.param(
            pd.Categorical(["a", "b"]),
            ["a", "b"],
            id="categorical",
        ),
        pytest.param(
            np.array(["a", "b"]),
            ["a", "b"],
            id="numpy_str",
        ),
    ],
)
def test_validate_edgelist_reads_strings_and_integers_silently(
    source,
    expected,
):
    edgelist = pd.DataFrame(
        {
            "source": source,
            "target": ["c", "c"],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        normalized = _validate_edgelist(edgelist)

    assert normalized["source"].tolist() == expected


def test_integer_and_string_with_the_same_text_are_one_node():
    edgelist = pd.DataFrame(
        {
            "source": [1, "1"],
            "target": ["b", "c"],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = parse_layered(edgelist)

    assert spec.input_nodes == ("1",)
    assert spec.output_nodes == ("b", "c")


@pytest.mark.parametrize(
    ("source", "target", "example", "rows"),
    [
        pytest.param(
            [1.0, 2.0],
            ["c", "c"],
            "1.0 became '1.0'",
            "'source': 2 row(s) at index 0, 1",
            id="float",
        ),
        pytest.param(
            ["a", "b"],
            [True, False],
            "True became 'True'",
            "'target': 2 row(s) at index 0, 1",
            id="bool",
        ),
        pytest.param(
            [np.float32(1.5), "b"],
            ["c", np.bool_(True)],
            "1.5 became '1.5'",
            "'source': 1 row(s) at index 0; 'target': 1 row(s) at index 1",
            id="numpy_scalars",
        ),
    ],
)
def test_validate_edgelist_warns_on_float_and_bool_names(
    source,
    target,
    example,
    rows,
):
    edgelist = pd.DataFrame(
        {
            "source": source,
            "target": target,
        }
    )

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Edgelist holds float or bool values in 'source' or "
            "'target'; they become node names through str(), so "
            f"{example}. {rows}. Node names are text, so 1.0 and 1 "
            "name different nodes. To silence this, convert the "
            "column to the intended strings first, for example with "
            ".astype(int).astype(str) for float IDs."
        ),
    ):
        normalized = _validate_edgelist(edgelist)

    assert normalized["source"].tolist() == [str(name) for name in source]


@pytest.mark.parametrize(
    "build",
    _ENTRY_POINTS,
)
def test_float_name_warning_points_at_the_caller(build):
    edgelist = pd.DataFrame(
        {
            "source": [1.5],
            "target": ["b"],
        }
    )

    with pytest.warns(UserWarning) as record:
        build(edgelist)

    assert len(record) == 1
    assert record[0].filename == __file__


@pytest.mark.parametrize(
    ("value", "described"),
    [
        pytest.param(
            ["x", "y"],
            "['x', 'y'] (list)",
            id="list",
        ),
        pytest.param(
            ("x",),
            "('x',) (tuple)",
            id="tuple",
        ),
        pytest.param(
            {"x": 1},
            "{'x': 1} (dict)",
            id="dict",
        ),
        pytest.param(
            b"x",
            "b'x' (bytes)",
            id="bytes",
        ),
        pytest.param(
            1 + 2j,
            "(1+2j) (complex)",
            id="complex",
        ),
        pytest.param(
            torch.tensor([12345, 67890]),
            "Tensor of shape (2,), dtype torch.int64",
            id="tensor",
        ),
        pytest.param(
            np.array(
                [12345],
                dtype=np.int64,
            ),
            "ndarray of shape (1,), dtype int64",
            id="array",
        ),
        pytest.param(
            _Node(),
            "_Node() (_Node)",
            id="object",
        ),
    ],
)
def test_validate_edgelist_rejects_values_that_are_not_names(
    value,
    described,
):
    with pytest.raises(Kpnn2Error) as exc_info:
        _validate_edgelist(_with_target(value))

    message = str(exc_info.value)
    assert message == (
        f"{_NOT_NAMES}Got {described} in 'target' at index 'r2'. "
        "'target': 1 row(s) at index 'r2'. Replace each with the name "
        "it stands for."
    )
    assert "12345" not in message


def test_validate_edgelist_counts_rejected_rows_per_column():
    edgelist = pd.DataFrame(
        {
            "source": pd.Series(
                ["a", ["x"], ["y"]],
                dtype=object,
            ),
            "target": pd.Series(
                [b"c", "d", "d"],
                dtype=object,
            ),
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            f"{_NOT_NAMES}Got ['x'] (list) in 'source' at index 1. "
            "'source': 2 row(s) at index 1, 2; 'target': 1 row(s) at "
            "index 0."
        ),
    ):
        _validate_edgelist(edgelist)


def test_rejected_name_is_raised_before_any_float_warning():
    edgelist = pd.DataFrame(
        {
            "source": pd.Series(
                [1.5, ["x"]],
                dtype=object,
            ),
            "target": ["c", "c"],
        }
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(
            Kpnn2Error,
            match="must hold node names",
        ):
            _validate_edgelist(edgelist)

    assert record == []


def test_missing_name_is_reported_before_its_float_type():
    edgelist = pd.DataFrame(
        {
            "source": [1.0, np.nan],
            "target": ["c", "c"],
        }
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(
            Kpnn2Error,
            match="missing values",
        ):
            _validate_edgelist(edgelist)

    assert record == []


@pytest.mark.parametrize(
    "build",
    _ENTRY_POINTS,
)
def test_public_entry_points_reject_values_that_are_not_names(build):
    edgelist = _with_target(["x"])
    expected = edgelist.copy()

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(f"{_NOT_NAMES}Got ['x'] (list)"),
    ):
        build(edgelist)

    pd.testing.assert_frame_equal(
        edgelist,
        expected,
    )
