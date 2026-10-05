import numpy as np
import pytest
import torch

from kpnn2 import (
    Kpnn2Error,
    PackedLinear,
    PackedMultiheadAttention,
)

_SOURCE_BOUND = 3
_TARGET_BOUND = 2


def _packed_linear(
    source_index,
    target_index,
):
    return PackedLinear(
        source_index,
        target_index,
        _TARGET_BOUND,
        _SOURCE_BOUND,
    )


def _packed_attention(
    source_index,
    target_index,
):
    return PackedMultiheadAttention(
        source_index,
        target_index,
        _TARGET_BOUND,
        _SOURCE_BOUND,
        4,
        2,
    )


_LAYERS = [
    pytest.param(
        _packed_linear,
        "in_features",
        "out_features",
        "PackedLinear",
        id="PackedLinear",
    ),
    pytest.param(
        _packed_attention,
        "key_features",
        "query_features",
        "PackedMultiheadAttention",
        id="PackedMultiheadAttention",
    ),
]

_NOT_INDEX = "must be a 1-dimensional integer tensor or a sequence of int."
_UNORDERED = (
    "must be a 1-dimensional integer tensor or a sequence of int, not "
    "{kind}: its items are matched by position."
)

# Type name of a numpy scalar: ``bool`` on numpy 2, ``bool_`` on 1.x.
_NUMPY_BOOL = type(np.bool_(True)).__name__

_CASES = [
    pytest.param(
        torch.tensor([0.0, 1.0]),
        [0, 1],
        f"'source_index' {_NOT_INDEX} Got Tensor of shape (2,), "
        "dtype torch.float32.",
        id="float_tensor",
    ),
    pytest.param(
        [0, 1],
        [0, 1.0],
        f"'target_index' {_NOT_INDEX} Got 1.0 (float) at position 1.",
        id="float_sequence",
    ),
    pytest.param(
        torch.tensor([[0, 1]]),
        [0, 1],
        f"'source_index' {_NOT_INDEX} Got Tensor of shape (1, 2), "
        "dtype torch.int64.",
        id="two_dimensional",
    ),
    pytest.param(
        torch.tensor([True, False]),
        [0, 1],
        f"'source_index' {_NOT_INDEX} Got Tensor of shape (2,), "
        "dtype torch.bool.",
        id="bool_tensor",
    ),
    pytest.param(
        np.array([True, False]),
        [0, 1],
        f"'source_index' {_NOT_INDEX} Got ndarray of shape (2,), dtype bool.",
        id="numpy_bool_array",
    ),
    pytest.param(
        np.array([0.0, 1.0]),
        [0, 1],
        f"'source_index' {_NOT_INDEX} Got ndarray of shape (2,), "
        "dtype float64.",
        id="numpy_float_array",
    ),
    pytest.param(
        np.array(
            [[0, 1]],
            dtype=np.int64,
        ),
        [0, 1],
        f"'source_index' {_NOT_INDEX} Got ndarray of shape (1, 2), "
        "dtype int64.",
        id="numpy_two_dimensional",
    ),
    pytest.param(
        np.array(
            [0, 1],
            dtype=object,
        ),
        [0, 1],
        f"'source_index' {_NOT_INDEX} Got ndarray of shape (2,), dtype object.",
        id="numpy_object_array",
    ),
    pytest.param(
        np.array(
            0,
            dtype=np.int64,
        ),
        [0],
        f"'source_index' {_NOT_INDEX} Got ndarray of shape (), dtype int64.",
        id="numpy_zero_dim",
    ),
    pytest.param(
        [0, 1],
        [np.int64(0), np.bool_(True)],
        f"'target_index' {_NOT_INDEX} Got True ({_NUMPY_BOOL}) at position 1.",
        id="numpy_bool_item",
    ),
    pytest.param(
        [0, 1],
        [np.int64(0), np.float64(1.0)],
        f"'target_index' {_NOT_INDEX} Got 1.0 (float64) at position 1.",
        id="numpy_float_item",
    ),
    pytest.param(
        [0, 1, 2],
        [0, 1, "2"],
        f"'target_index' {_NOT_INDEX} Got '2' (str) at position 2.",
        id="str_item",
    ),
    pytest.param(
        5,
        [0],
        f"'source_index' {_NOT_INDEX} Got 5 (int).",
        id="not_iterable",
    ),
    pytest.param(
        torch.tensor([0j, 1j]),
        [0, 1],
        f"'source_index' {_NOT_INDEX} Got Tensor of shape (2,), "
        "dtype torch.complex64.",
        id="complex_tensor",
    ),
    pytest.param(
        {1, 0},
        [0, 1],
        f"'source_index' {_UNORDERED.format(kind='a set')} Got "
        "{{0, 1}} (set). Pass a list or tuple in the intended order.",
        id="set",
    ),
    pytest.param(
        [0, 1],
        frozenset({0, 1}),
        f"'target_index' {_UNORDERED.format(kind='a set')} Got "
        "frozenset({{0, 1}}) (frozenset). Pass a list or tuple in the "
        "intended order.",
        id="frozenset",
    ),
    pytest.param(
        {1: "b", 0: "a"},
        [0, 1],
        f"'source_index' {_UNORDERED.format(kind='a mapping')} Got "
        "{{1: 'b', 0: 'a'}} (dict). Pass a list or tuple in the "
        "intended order.",
        id="mapping",
    ),
    pytest.param(
        [0],
        "0",
        f"'target_index' {_NOT_INDEX} Got '0' (str).",
        id="str",
    ),
    pytest.param(
        [0, 1],
        [0],
        "'source_index' and 'target_index' must have the same length. "
        "Got 2 and 1.",
        id="length_mismatch",
    ),
    pytest.param(
        [],
        [],
        "'source_index' and 'target_index' must contain at least one index.",
        id="empty",
    ),
    pytest.param(
        [-1],
        [0],
        "'source_index' entries must satisfy "
        "0 <= source_index < {source_bound}. Got 1 entry out of range "
        f"with {{source_bound}}={_SOURCE_BOUND}; the first is -1 at "
        "position 0.",
        id="source_below",
    ),
    pytest.param(
        [_SOURCE_BOUND],
        [0],
        "'source_index' entries must satisfy "
        "0 <= source_index < {source_bound}. Got 1 entry out of range "
        f"with {{source_bound}}={_SOURCE_BOUND}; the first is "
        f"{_SOURCE_BOUND} at position 0.",
        id="source_above",
    ),
    pytest.param(
        [0, 1, 7, 2, -4],
        [0, 1, 0, 1, 0],
        "'source_index' entries must satisfy "
        "0 <= source_index < {source_bound}. Got 2 entries out of range "
        f"with {{source_bound}}={_SOURCE_BOUND}; the first is 7 at "
        "position 2.",
        id="source_several_out_of_range",
    ),
    pytest.param(
        [0],
        [-1],
        "'target_index' entries must satisfy "
        "0 <= target_index < {target_bound}. Got 1 entry out of range "
        f"with {{target_bound}}={_TARGET_BOUND}; the first is -1 at "
        "position 0.",
        id="target_below",
    ),
    pytest.param(
        [0],
        [_TARGET_BOUND],
        "'target_index' entries must satisfy "
        "0 <= target_index < {target_bound}. Got 1 entry out of range "
        f"with {{target_bound}}={_TARGET_BOUND}; the first is "
        f"{_TARGET_BOUND} at position 0.",
        id="target_above",
    ),
    pytest.param(
        [0, 0],
        [1, 1],
        "{owner} indices contain duplicate (source, target) pair(s). "
        "Got 1 duplicated pair: (0, 1).",
        id="duplicate_pair",
    ),
    pytest.param(
        [2, 0, 1, 2, 0, 0, 0, 1],
        [1, 1, 0, 1, 1, 0, 1, 1],
        "{owner} indices contain duplicate (source, target) pair(s). "
        "Got 2 duplicated pairs: (0, 1), (2, 1).",
        id="duplicate_pairs_sorted",
    ),
    pytest.param(
        [2, 2, 1, 1, 0, 0] * 2,
        [1, 0, 1, 0, 1, 0] * 2,
        "{owner} indices contain duplicate (source, target) pair(s). "
        "Got 6 duplicated pairs: (0, 0), (0, 1), (1, 0), (1, 1), (2, 0) "
        "(and 1 more).",
        id="duplicate_pairs_capped",
    ),
]


@pytest.mark.parametrize(
    (
        "build",
        "source_bound",
        "target_bound",
        "owner",
    ),
    _LAYERS,
)
@pytest.mark.parametrize(
    (
        "source_index",
        "target_index",
        "message",
    ),
    _CASES,
)
def test_packed_layers_share_index_errors(
    build,
    source_bound,
    target_bound,
    owner,
    source_index,
    target_index,
    message,
):
    expected = message.format(
        source_bound=source_bound,
        target_bound=target_bound,
        owner=owner,
    )
    with pytest.raises(Kpnn2Error) as excinfo:
        build(
            source_index,
            target_index,
        )
    assert str(excinfo.value) == expected


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            _packed_linear,
            id="PackedLinear",
        ),
        pytest.param(
            _packed_attention,
            id="PackedMultiheadAttention",
        ),
    ],
)
@pytest.mark.parametrize(
    "index",
    [
        pytest.param(
            torch.tensor([918273.0, 645546.0]),
            id="tensor",
        ),
        pytest.param(
            np.array([918273.0, 645546.0]),
            id="ndarray",
        ),
    ],
)
def test_index_errors_do_not_print_tensor_values(
    build,
    index,
):
    with pytest.raises(Kpnn2Error) as excinfo:
        build(
            index,
            [0, 1],
        )
    message = str(excinfo.value)
    assert "of shape (2,)" in message
    assert "918273" not in message
    assert "645546" not in message
