import re
from types import MappingProxyType

import numpy as np
import pytest
import torch

from kpnn2 import Kpnn2Error
from kpnn2._validate import (
    as_bool,
    as_positive_int,
    describe,
    is_integer,
    reject_unordered,
    require_floating,
    require_node_mapping,
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
        pytest.param(
            np.int64(3),
            True,
            id="numpy_int64",
        ),
        pytest.param(
            np.int32(0),
            True,
            id="numpy_int32",
        ),
        pytest.param(
            np.uint8(1),
            True,
            id="numpy_uint8",
        ),
        pytest.param(
            np.bool_(True),
            False,
            id="numpy_bool",
        ),
        pytest.param(
            np.float64(1.0),
            False,
            id="numpy_float64",
        ),
        pytest.param(
            np.array(1),
            False,
            id="zero_dim_array",
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
    "value",
    [
        np.int64(2),
        np.int32(2),
        np.uint16(2),
    ],
)
def test_as_positive_int_returns_python_int_for_numpy(value):
    result = as_positive_int(
        value,
        "n",
    )

    assert type(result) is int
    assert result == 2


@pytest.mark.parametrize(
    "value",
    [
        np.int64(0),
        np.int64(-1),
        np.bool_(True),
        np.float64(2.0),
    ],
)
def test_as_positive_int_rejects_numpy_non_positive_or_non_integer(value):
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("'n' must be a positive int. Got "),
    ):
        as_positive_int(
            value,
            "n",
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            [
                torch.tensor([12345.0]),
                torch.tensor([67890.0]),
            ],
            "list of length 2",
            id="list_of_tensors",
        ),
        pytest.param(
            (np.array([12345.0]),),
            "tuple of length 1",
            id="tuple_of_arrays",
        ),
        pytest.param(
            {"scores": torch.tensor([12345.0])},
            "dict of length 1",
            id="dict_of_tensors",
        ),
        pytest.param(
            {torch.tensor([12345.0])},
            "set of length 1",
            id="set_of_tensors",
        ),
        pytest.param(
            frozenset({torch.tensor([67890.0])}),
            "frozenset of length 1",
            id="frozenset_of_tensors",
        ),
    ],
)
def test_describe_container_of_data_by_length_not_values(
    value,
    expected,
):
    text = describe(value)

    assert text == expected
    assert "12345" not in text
    assert "67890" not in text


def test_describe_container_without_data_keeps_its_repr():
    assert describe([0.0, 1.0]) == "[0.0, 1.0] (list)"


def test_require_node_mapping_accepts_any_mapping():
    require_node_mapping(
        {"A": 2},
        "widths",
    )
    require_node_mapping(
        MappingProxyType({}),
        "ranks",
    )


@pytest.mark.parametrize(
    ("value", "got"),
    [
        pytest.param(
            [("A", 2)],
            "[('A', 2)] (list)",
            id="list_of_pairs",
        ),
        pytest.param(
            "A",
            "'A' (str)",
            id="str",
        ),
        pytest.param(
            None,
            "None",
            id="None",
        ),
    ],
)
def test_require_node_mapping_rejects_and_reports_value(
    value,
    got,
):
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            f"'widths' must be a mapping of node name to int. Got {got}."
        ),
    ):
        require_node_mapping(
            value,
            "widths",
        )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(
            [1, 0],
            id="list",
        ),
        pytest.param(
            (1, 0),
            id="tuple",
        ),
        pytest.param(
            range(2),
            id="range",
        ),
        pytest.param(
            np.array([1, 0]),
            id="array",
        ),
        pytest.param(
            torch.tensor([1, 0]),
            id="tensor",
        ),
        pytest.param(
            {"a": 1, "b": 0}.values(),
            id="dict_values",
        ),
        pytest.param(
            (item for item in [1, 0]),
            id="generator",
        ),
        pytest.param(
            "ab",
            id="str_left_to_the_caller",
        ),
    ],
)
def test_reject_unordered_accepts_ordered_values(value):
    reject_unordered(
        value,
        "index",
        "a sequence of int",
    )


@pytest.mark.parametrize(
    ("value", "kind", "got"),
    [
        pytest.param(
            {1, 0},
            "a set",
            "{0, 1} (set)",
            id="set",
        ),
        pytest.param(
            frozenset({0}),
            "a set",
            "frozenset({0}) (frozenset)",
            id="frozenset",
        ),
        pytest.param(
            {"a": 1}.keys(),
            "a set",
            "dict_keys(['a']) (dict_keys)",
            id="dict_keys",
        ),
        pytest.param(
            {"a": 1},
            "a mapping",
            "{'a': 1} (dict)",
            id="dict",
        ),
        pytest.param(
            MappingProxyType({"a": 1}),
            "a mapping",
            "mappingproxy({'a': 1}) (mappingproxy)",
            id="mapping_proxy",
        ),
    ],
)
def test_reject_unordered_rejects_sets_and_mappings(
    value,
    kind,
    got,
):
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            f"'index' must be a sequence of int, not {kind}: its items "
            f"are matched by position. Got {got}. Pass a list or tuple "
            "in the intended order."
        ),
    ):
        reject_unordered(
            value,
            "index",
            "a sequence of int",
        )


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
def test_require_floating_accepts_floating_dtypes(dtype):
    require_floating(
        torch.zeros(
            2,
            dtype=dtype,
        ),
        "Layer input",
    )


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int64,
        torch.uint8,
        torch.bool,
        torch.complex128,
    ],
)
def test_require_floating_rejects_and_omits_values(dtype):
    tensor = torch.full(
        (2,),
        12345,
    ).to(dtype=dtype)

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "Layer input must be a floating-point tensor. Got Tensor of "
            f"shape (2,), dtype {dtype}. Convert it first, for example "
            "with .float()."
        ),
    ) as caught:
        require_floating(
            tensor,
            "Layer input",
        )
    assert "12345" not in str(caught.value)
