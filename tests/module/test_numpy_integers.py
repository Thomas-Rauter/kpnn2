"""
numpy integers are accepted wherever an integer is expected.

Each public entry point takes a numpy integer scalar (or an
integer numpy array for packed indices) exactly like the Python
int, and keeps a Python ``int``: the result equals the int-built
one. ``bool``, ``numpy.bool_``, and floats stay rejected.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from kpnn2 import (
    Kpnn2Error,
    LayeredSpec,
    PackedLinear,
    PackedMultiheadAttention,
    map_node_attributions,
    parse_adjacency,
    parse_layered,
)

_NUMPY_INTEGER_TYPES = [
    pytest.param(
        np.int64,
        id="int64",
    ),
    pytest.param(
        np.int32,
        id="int32",
    ),
    pytest.param(
        np.uint8,
        id="uint8",
    ),
]

_NOT_INTEGERS = [
    pytest.param(
        True,
        id="bool",
    ),
    pytest.param(
        np.bool_(True),
        id="numpy_bool",
    ),
    pytest.param(
        np.float64(2.0),
        id="numpy_float64",
    ),
    pytest.param(
        2.0,
        id="float",
    ),
]


def _layered_edgelist():
    """
    A feeds H and C; H feeds D. Longest-path puts C at depth 1.
    """
    return pd.DataFrame(
        {
            "source": ["A", "A", "H"],
            "target": ["H", "C", "D"],
        }
    )


def _cycle_edgelist():
    return pd.DataFrame(
        {
            "source": ["x", "a", "b", "a"],
            "target": ["a", "b", "a", "y"],
        }
    )


def _assert_layered_ints(spec):
    for row in spec.layer_widths:
        for width in row:
            assert type(width) is int
    for dim in spec.layer_dims:
        assert type(dim) is int


@pytest.mark.parametrize(
    "integer_type",
    _NUMPY_INTEGER_TYPES,
)
def test_parse_layered_numpy_widths_and_ranks_match_int(integer_type):
    widths = {
        "H": 3,
        "C": 2,
    }
    ranks = {
        "A": 0,
        "H": 1,
        "C": 2,
        "D": 2,
    }
    expected = parse_layered(
        _layered_edgelist(),
        widths=widths,
        ranks=ranks,
    )

    spec = parse_layered(
        _layered_edgelist(),
        widths={name: integer_type(value) for name, value in widths.items()},
        ranks={name: integer_type(value) for name, value in ranks.items()},
    )

    assert spec == expected
    assert spec.fingerprint == expected.fingerprint
    payload = spec.to_dict()
    assert json.dumps(payload) == json.dumps(expected.to_dict())
    assert set(payload) >= {
        "widths",
        "ranks",
    }
    _assert_layered_ints(spec)


@pytest.mark.parametrize(
    "integer_type",
    _NUMPY_INTEGER_TYPES,
)
def test_parse_adjacency_numpy_widths_match_int(integer_type):
    expected = parse_adjacency(
        _cycle_edgelist(),
        widths={"a": 2},
    )

    spec = parse_adjacency(
        _cycle_edgelist(),
        widths={"a": integer_type(2)},
    )

    assert spec == expected
    assert spec.fingerprint == expected.fingerprint
    assert json.dumps(spec.to_dict()) == json.dumps(expected.to_dict())
    for width in spec.node_widths:
        assert type(width) is int
    assert type(spec.state_dim) is int


def test_from_dict_accepts_numpy_widths():
    expected = parse_layered(
        _layered_edgelist(),
        widths={"H": 3},
    )
    payload = expected.to_dict()
    payload["widths"] = {"H": np.int64(3)}

    spec = LayeredSpec.from_dict(payload)

    assert spec == expected
    _assert_layered_ints(spec)


def test_from_dict_schema_tag_stays_a_python_int():
    payload = parse_layered(_layered_edgelist()).to_dict()
    payload["kpnn2_spec"] = np.int64(1)

    with pytest.raises(
        Kpnn2Error,
        match="'kpnn2_spec' must be 1.",
    ):
        LayeredSpec.from_dict(payload)


@pytest.mark.parametrize(
    "value",
    _NOT_INTEGERS,
)
def test_non_integer_widths_still_raise(value):
    with pytest.raises(
        Kpnn2Error,
        match="must be a positive int",
    ):
        parse_layered(
            _layered_edgelist(),
            widths={"H": value},
        )
    with pytest.raises(
        Kpnn2Error,
        match="must be a positive int",
    ):
        parse_adjacency(
            _cycle_edgelist(),
            widths={"a": value},
        )


@pytest.mark.parametrize(
    "value",
    _NOT_INTEGERS,
)
def test_non_integer_ranks_still_raise(value):
    ranks = {
        "A": 0,
        "H": value,
        "C": 1,
        "D": 2,
    }
    with pytest.raises(
        Kpnn2Error,
        match="must be a non-negative int",
    ):
        parse_layered(
            _layered_edgelist(),
            ranks=ranks,
        )


def _assert_same_index_buffers(
    actual,
    expected,
):
    for name in (
        "source_index",
        "target_index",
    ):
        actual_index = getattr(
            actual,
            name,
        )
        assert actual_index.dtype == torch.int64
        assert torch.equal(
            actual_index,
            getattr(
                expected,
                name,
            ),
        )


def _assert_same_parameters(
    actual,
    expected,
):
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, parameter in expected_parameters.items():
        torch.testing.assert_close(
            actual_parameters[name],
            parameter,
        )


_PACKED_SOURCE = [0, 2, 1, 2]
_PACKED_TARGET = [0, 0, 1, 1]


def _packed_linear(
    source_index,
    target_index,
    out_features,
    in_features,
):
    return PackedLinear(
        source_index,
        target_index,
        out_features,
        in_features,
        generator=torch.Generator().manual_seed(42),
    )


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            np.int64,
            id="int64",
        ),
        pytest.param(
            np.int32,
            id="int32",
        ),
        pytest.param(
            np.uint8,
            id="uint8",
        ),
    ],
)
def test_packed_linear_from_numpy_matches_list(dtype):
    expected = _packed_linear(
        _PACKED_SOURCE,
        _PACKED_TARGET,
        2,
        3,
    )

    layer = _packed_linear(
        np.array(
            _PACKED_SOURCE,
            dtype=dtype,
        ),
        np.array(
            _PACKED_TARGET,
            dtype=dtype,
        ),
        dtype(2),
        dtype(3),
    )

    _assert_same_index_buffers(
        layer,
        expected,
    )
    assert type(layer.out_features) is int
    assert type(layer.in_features) is int
    assert layer.out_features == expected.out_features
    assert layer.in_features == expected.in_features
    assert torch.equal(
        layer.state_dict()["index_digest"],
        expected.state_dict()["index_digest"],
    )
    _assert_same_parameters(
        layer,
        expected,
    )


def test_packed_linear_from_numpy_scalar_sequence_matches_list():
    expected = _packed_linear(
        _PACKED_SOURCE,
        _PACKED_TARGET,
        2,
        3,
    )

    layer = _packed_linear(
        [np.int64(index) for index in _PACKED_SOURCE],
        [np.int32(index) for index in _PACKED_TARGET],
        2,
        3,
    )

    _assert_same_index_buffers(
        layer,
        expected,
    )


def test_packed_linear_copies_a_numpy_index():
    source_index = np.array(
        _PACKED_SOURCE,
        dtype=np.int64,
    )
    target_index = np.array(
        _PACKED_TARGET,
        dtype=np.int64,
    )
    before_source = source_index.copy()
    before_target = target_index.copy()

    layer = _packed_linear(
        source_index,
        target_index,
        2,
        3,
    )
    np.testing.assert_array_equal(
        source_index,
        before_source,
    )
    np.testing.assert_array_equal(
        target_index,
        before_target,
    )
    source_index[0] = 1
    target_index[0] = 1

    assert layer.source_index.tolist() == _PACKED_SOURCE
    assert layer.target_index.tolist() == _PACKED_TARGET


def test_packed_linear_accepts_strided_numpy_views():
    source_index = np.array(
        [2, 1, 2, 0],
        dtype=np.int64,
    )[::-1]
    target_index = np.array(
        [0, 9, 0, 9, 1, 9, 1],
        dtype=np.int32,
    )[::2]

    layer = _packed_linear(
        source_index,
        target_index,
        2,
        3,
    )

    assert layer.source_index.is_contiguous()
    assert layer.target_index.is_contiguous()
    assert layer.source_index.tolist() == _PACKED_SOURCE
    assert layer.target_index.tolist() == _PACKED_TARGET


@pytest.mark.parametrize(
    "name",
    [
        "out_features",
        "in_features",
    ],
)
@pytest.mark.parametrize(
    "value",
    _NOT_INTEGERS,
)
def test_packed_linear_non_integer_sizes_still_raise(
    name,
    value,
):
    sizes = {
        "out_features": 2,
        "in_features": 3,
    }
    sizes[name] = value
    with pytest.raises(
        Kpnn2Error,
        match=f"'{name}' must be a positive int.",
    ):
        PackedLinear(
            _PACKED_SOURCE,
            _PACKED_TARGET,
            **sizes,
        )


_ATTENTION_SOURCE = [0, 1, 1, 0]
_ATTENTION_TARGET = [1, 0, 1, 0]


def _attention(
    source_index,
    target_index,
    query_features,
    key_features,
    embed_dim,
    num_heads,
    **kwargs,
):
    return PackedMultiheadAttention(
        source_index,
        target_index,
        query_features,
        key_features,
        embed_dim,
        num_heads,
        generator=torch.Generator().manual_seed(42),
        **kwargs,
    )


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            np.int64,
            id="int64",
        ),
        pytest.param(
            np.int32,
            id="int32",
        ),
    ],
)
def test_packed_attention_from_numpy_matches_list(dtype):
    expected = _attention(
        _ATTENTION_SOURCE,
        _ATTENTION_TARGET,
        2,
        2,
        4,
        2,
        kdim=4,
        vdim=4,
        chunk_size=2,
    )

    layer = _attention(
        np.array(
            _ATTENTION_SOURCE,
            dtype=dtype,
        ),
        np.array(
            _ATTENTION_TARGET,
            dtype=dtype,
        ),
        dtype(2),
        dtype(2),
        dtype(4),
        dtype(2),
        kdim=dtype(4),
        vdim=dtype(4),
        chunk_size=dtype(2),
    )

    _assert_same_index_buffers(
        layer,
        expected,
    )
    for name in (
        "query_features",
        "key_features",
        "embed_dim",
        "num_heads",
        "head_dim",
        "chunk_size",
    ):
        value = getattr(
            layer,
            name,
        )
        assert type(value) is int
        assert value == getattr(
            expected,
            name,
        )
    assert torch.equal(
        layer.state_dict()["index_digest"],
        expected.state_dict()["index_digest"],
    )
    _assert_same_parameters(
        layer,
        expected,
    )


def test_packed_attention_chunk_size_numpy_is_stored_as_int():
    layer = _attention(
        _ATTENTION_SOURCE,
        _ATTENTION_TARGET,
        2,
        2,
        4,
        2,
        chunk_size=np.int64(2),
    )

    assert type(layer.chunk_size) is int
    assert layer.chunk_size == 2
    assert "chunk_size=2" in layer.extra_repr()


@pytest.mark.parametrize(
    "dropout",
    [
        np.int64(0),
        np.int64(1),
    ],
)
def test_packed_attention_numpy_integer_dropout_is_a_float(dropout):
    layer = _attention(
        _ATTENTION_SOURCE,
        _ATTENTION_TARGET,
        2,
        2,
        4,
        2,
        dropout=dropout,
    )

    assert type(layer.dropout) is float
    assert layer.dropout == int(dropout)


@pytest.mark.parametrize(
    "name",
    [
        "query_features",
        "key_features",
        "embed_dim",
        "num_heads",
    ],
)
@pytest.mark.parametrize(
    "value",
    _NOT_INTEGERS,
)
def test_packed_attention_non_integer_sizes_still_raise(
    name,
    value,
):
    sizes = {
        "query_features": 2,
        "key_features": 2,
        "embed_dim": 4,
        "num_heads": 2,
    }
    sizes[name] = value
    with pytest.raises(
        Kpnn2Error,
        match=f"'{name}' must be a positive int.",
    ):
        PackedMultiheadAttention(
            _ATTENTION_SOURCE,
            _ATTENTION_TARGET,
            **sizes,
        )


@pytest.mark.parametrize(
    "value",
    _NOT_INTEGERS,
)
def test_packed_attention_non_integer_chunk_size_still_raises(value):
    with pytest.raises(
        Kpnn2Error,
        match="'chunk_size' must be None or a positive int.",
    ):
        _attention(
            _ATTENTION_SOURCE,
            _ATTENTION_TARGET,
            2,
            2,
            4,
            2,
            chunk_size=value,
        )


@pytest.mark.parametrize(
    "name",
    [
        "kdim",
        "vdim",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(
            np.bool_(True),
            id="numpy_bool",
        ),
        pytest.param(
            np.float64(4.0),
            id="numpy_float64",
        ),
        pytest.param(
            np.int64(2),
            id="numpy_int_not_embed_dim",
        ),
    ],
)
def test_packed_attention_bad_numpy_kdim_vdim_still_raise(
    name,
    value,
):
    with pytest.raises(
        Kpnn2Error,
        match=f"'{name}' must be None or equal to embed_dim.",
    ):
        _attention(
            _ATTENTION_SOURCE,
            _ATTENTION_TARGET,
            2,
            2,
            4,
            2,
            **{name: value},
        )


def test_map_node_attributions_numpy_layer_matches_int():
    spec = parse_layered(
        _layered_edgelist(),
        widths={"H": 2},
    )
    attributions = torch.arange(
        2 * spec.layer_dims[1],
        dtype=torch.float32,
    ).reshape(
        2,
        spec.layer_dims[1],
    )
    expected = map_node_attributions(
        attributions,
        spec,
        1,
    )

    mapped = map_node_attributions(
        attributions,
        spec,
        np.int64(1),
    )

    xr.testing.assert_identical(
        mapped,
        expected,
    )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(
            True,
            id="bool",
        ),
        pytest.param(
            np.bool_(True),
            id="numpy_bool",
        ),
        pytest.param(
            np.float64(1.0),
            id="numpy_float64",
        ),
    ],
)
def test_map_node_attributions_non_integer_layer_still_raises(value):
    spec = parse_layered(_layered_edgelist())
    attributions = torch.zeros(spec.layer_dims[1])

    with pytest.raises(
        Kpnn2Error,
        match="'layer' must be an int.",
    ):
        map_node_attributions(
            attributions,
            spec,
            value,
        )
