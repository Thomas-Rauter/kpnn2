import re

import pytest

import kpnn2
from kpnn2 import Kpnn2Error
from kpnn2._errors import (
    _ISSUES_URL,
    caller_error,
    close_matches,
    internal_error,
    listed,
    warn_at_caller,
)


def _internal_check(what: str) -> str:
    return re.escape(f"internal check failed: {what}")


def test_caller_error_with_only_a_headline_is_one_line():
    error = caller_error("'bias' must be True or False.")

    assert isinstance(error, Kpnn2Error)
    assert str(error) == "'bias' must be True or False."


def test_caller_error_renders_fields_in_the_fixed_order():
    error = caller_error(
        "Unknown node name in 'ranks'.",
        fix="Rename it.",
        candidates="'TP53'",
        expected="a node of the edgelist",
        got="'TP35'",
    )

    assert str(error) == (
        "Unknown node name in 'ranks'.\n"
        "  Got: 'TP35'\n"
        "  Expected: a node of the edgelist\n"
        "  Did you mean: 'TP53'\n"
        "  Fix: Rename it."
    )


def test_caller_error_leaves_out_fields_not_given():
    error = caller_error(
        "'constraint' must be a torch.nn.Module or None.",
        got="'softplus' (str)",
        fix="Pass an instance such as nn.Softplus().",
    )

    assert str(error) == (
        "'constraint' must be a torch.nn.Module or None.\n"
        "  Got: 'softplus' (str)\n"
        "  Fix: Pass an instance such as nn.Softplus()."
    )


def test_line_breaks_in_parts_are_escaped():
    error = caller_error(
        "Unknown node name: a\nb.",
        got="'x\r\ny'",
        fix="Rename\u2028it.",
    )

    assert str(error).splitlines() == [
        "Unknown node name: a\\nb.",
        "  Got: 'x\\r\\ny'",
        "  Fix: Rename\\u2028it.",
    ]


@pytest.mark.parametrize(
    ("kwargs", "what"),
    [
        (
            {"headline": "No period"},
            "message headline 'No period' does not end with a period",
        ),
        (
            {"headline": ""},
            "message headline '' does not end with a period",
        ),
        (
            {
                "headline": "Bad.",
                "fix": "Do this",
            },
            "Fix line 'Do this' does not end with a period",
        ),
        (
            {
                "headline": "Bad.",
                "got": "",
            },
            "the Got line of a message is empty",
        ),
        (
            {
                "headline": "Bad.",
                "candidates": "",
            },
            "the Did you mean line of a message is empty",
        ),
    ],
)
def test_malformed_message_is_an_internal_error(
    kwargs,
    what,
):
    with pytest.raises(
        AssertionError,
        match=_internal_check(what),
    ):
        caller_error(**kwargs)


def test_warn_at_caller_renders_the_same_format():
    with pytest.warns(
        UserWarning,
        match=re.escape("Odd edgelist.\n  Fix: Check it."),
    ) as record:
        warn_at_caller(
            "Odd edgelist.",
            UserWarning,
            fix="Check it.",
        )

    assert record[0].filename == __file__


def test_warn_at_caller_checks_the_headline():
    with pytest.raises(
        AssertionError,
        match=_internal_check("message headline 'soon' does not end"),
    ):
        warn_at_caller(
            "soon",
            FutureWarning,
        )


def test_internal_error_has_a_headline_and_a_fix_line():
    message = str(internal_error("hop edges are not conserved"))

    assert message.splitlines() == [
        "kpnn2 internal check failed: hop edges are not conserved. "
        f"This is a bug in kpnn2 {kpnn2.__version__}, not a problem "
        "with your call.",
        f"  Fix: Report it with this message at {_ISSUES_URL}.",
    ]


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        (
            ["A"],
            "A",
        ),
        (
            ["A", "B", "C"],
            "A, B, C",
        ),
        (
            [str(i) for i in range(10)],
            "0, 1, 2, 3, 4, 5, 6, 7, 8, 9",
        ),
        (
            [str(i) for i in range(11)],
            "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, and 1 more",
        ),
        (
            (i for i in range(25)),
            "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, and 15 more",
        ),
    ],
)
def test_listed_shows_at_most_ten_items(
    items,
    expected,
):
    assert listed(items) == expected


def test_close_matches_suggests_a_close_name():
    suggestion = close_matches(
        ["TP35"],
        ["TP53", "MYC", "EGFR"],
    )

    assert suggestion == "'TP53'"


def test_close_matches_prefers_a_case_only_match():
    suggestion = close_matches(
        ["tp53"],
        ["TP53", "TP5", "TP53X"],
    )

    assert suggestion == "'TP53'"


def test_close_matches_returns_none_when_nothing_is_close():
    assert (
        close_matches(
            ["ZZZZ"],
            ["TP53", "MYC"],
        )
        is None
    )


def test_close_matches_names_the_unknown_name_when_several_miss():
    suggestion = close_matches(
        ["MCY", "TP35", "ZZZZ"],
        ["TP53", "MYC", "EGFR"],
    )

    assert suggestion == "'MYC' for 'MCY'; 'TP53' for 'TP35'"


def test_close_matches_suggests_at_most_three_names():
    suggestion = close_matches(
        ["GENE"],
        ["GENE1", "GENE2", "GENE3", "GENE4"],
    )

    assert suggestion is not None
    assert suggestion.count("'GENE") == 3


def test_close_matches_looks_up_only_the_first_five_names():
    unknown = [f"Q{i}" for i in range(5)] + ["TP35"]

    assert (
        close_matches(
            unknown,
            ["TP53"],
        )
        is None
    )
