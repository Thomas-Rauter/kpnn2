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

_CASES = [
    pytest.param(
        torch.tensor([0.0, 1.0]),
        [0, 1],
        f"'source_index' {_NOT_INDEX}",
        id="float_tensor",
    ),
    pytest.param(
        [0, 1],
        [0, 1.0],
        f"'target_index' {_NOT_INDEX}",
        id="float_sequence",
    ),
    pytest.param(
        torch.tensor([[0, 1]]),
        [0, 1],
        f"'source_index' {_NOT_INDEX}",
        id="two_dimensional",
    ),
    pytest.param(
        np.array([True, False]),
        [0, 1],
        f"'source_index' {_NOT_INDEX}",
        id="numpy_bool_array",
    ),
    pytest.param(
        np.array([0.0, 1.0]),
        [0, 1],
        f"'source_index' {_NOT_INDEX}",
        id="numpy_float_array",
    ),
    pytest.param(
        np.array([[0, 1]]),
        [0, 1],
        f"'source_index' {_NOT_INDEX}",
        id="numpy_two_dimensional",
    ),
    pytest.param(
        np.array(
            [0, 1],
            dtype=object,
        ),
        [0, 1],
        f"'source_index' {_NOT_INDEX}",
        id="numpy_object_array",
    ),
    pytest.param(
        np.array(0),
        [0],
        f"'source_index' {_NOT_INDEX}",
        id="numpy_zero_dim",
    ),
    pytest.param(
        [0, 1],
        [np.int64(0), np.bool_(True)],
        f"'target_index' {_NOT_INDEX}",
        id="numpy_bool_item",
    ),
    pytest.param(
        [0, 1],
        [np.int64(0), np.float64(1.0)],
        f"'target_index' {_NOT_INDEX}",
        id="numpy_float_item",
    ),
    pytest.param(
        [0, 1],
        [0],
        "'source_index' and 'target_index' must have the same length.",
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
        "0 <= source_index < {source_bound}.",
        id="source_below",
    ),
    pytest.param(
        [_SOURCE_BOUND],
        [0],
        "'source_index' entries must satisfy "
        "0 <= source_index < {source_bound}.",
        id="source_above",
    ),
    pytest.param(
        [0],
        [-1],
        "'target_index' entries must satisfy "
        "0 <= target_index < {target_bound}.",
        id="target_below",
    ),
    pytest.param(
        [0],
        [_TARGET_BOUND],
        "'target_index' entries must satisfy "
        "0 <= target_index < {target_bound}.",
        id="target_above",
    ),
    pytest.param(
        [0, 0],
        [1, 1],
        "{owner} indices contain duplicate (source, target) pair(s).",
        id="duplicate_pair",
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
