import re

import pytest
import torch

from kpnn2 import Kpnn2Error
from kpnn2._validate import (
    as_bool,
    as_positive_int,
    describe,
    is_integer,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            3,
            True,
            id="int",
        ),
        pytest.param(
            0,
            True,
            id="zero",
        ),
        pytest.param(
            True,
            False,
            id="bool",
        ),
        pytest.param(
            1.0,
            False,
            id="float",
        ),
        pytest.param(
            "1",
            False,
            id="str",
        ),
        pytest.param(
            None,
            False,
            id="None",
        ),
    ],
)
def test_is_integer(
    value,
    expected,
):
    assert is_integer(value) is expected


def test_as_positive_int_returns_one():
    assert (
        as_positive_int(
            1,
            "n",
        )
        == 1
    )


@pytest.mark.parametrize(
    ("value", "got"),
    [
        pytest.param(
            0,
            "0 (int)",
            id="zero",
        ),
        pytest.param(
            -1,
            "-1 (int)",
            id="negative",
        ),
        pytest.param(
            True,
            "True (bool)",
            id="bool",
        ),
        pytest.param(
            1.0,
            "1.0 (float)",
            id="float",
        ),
        pytest.param(
            "1",
            "'1' (str)",
            id="str",
        ),
    ],
)
def test_as_positive_int_rejects_and_reports_value(
    value,
    got,
):
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(f"'n' must be a positive int. Got {got}."),
    ):
        as_positive_int(
            value,
            "n",
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ],
)
def test_as_bool_returns_the_flag(value):
    assert (
        as_bool(
            value,
            "flag",
        )
        is value
    )


@pytest.mark.parametrize(
    ("value", "got"),
    [
        pytest.param(
            "False",
            "'False' (str)",
            id="str",
        ),
        pytest.param(
            0,
            "0 (int)",
            id="zero",
        ),
        pytest.param(
            1,
            "1 (int)",
            id="one",
        ),
        pytest.param(
            None,
            "None",
            id="None",
        ),
    ],
)
def test_as_bool_rejects_and_reports_value(
    value,
    got,
):
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(f"'flag' must be True or False. Got {got}."),
    ):
        as_bool(
            value,
            "flag",
        )


def test_describe_tensor_by_shape_and_dtype_not_values():
    tensor = torch.tensor(
        [[12345.0, 67890.0, 13579.0]],
        dtype=torch.float64,
    )

    text = describe(tensor)

    assert text == "Tensor of shape (1, 3), dtype torch.float64"
    assert "12345" not in text


def test_describe_truncates_long_repr():
    text = describe("a" * 200)

    assert text.endswith("... (str)")
    assert len(text) <= 70
    assert text.startswith("'aaa")


def test_describe_keeps_one_line():
    text = describe(torch.nn.Sequential(torch.nn.Identity()))

    assert "\n" not in text
    assert text == "Sequential(... (Sequential)"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            "no",
            "'no' (str)",
            id="str",
        ),
        pytest.param(
            1.5,
            "1.5 (float)",
            id="float",
        ),
        pytest.param(
            None,
            "None",
            id="None",
        ),
    ],
)
def test_describe_short_values(
    value,
    expected,
):
    assert describe(value) == expected
