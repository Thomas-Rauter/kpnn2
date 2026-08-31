import copy
import hashlib
import math
import struct

import pandas as pd
import pytest
import torch
from torch import nn

from kpnn2 import (
    AdjacencySpec,
    Kpnn2Error,
    MaskedLinear,
    PackedLinear,
    parse_adjacency,
)


def _cyclic_edgelist():
    return pd.DataFrame(
        {
            "source": ["x", "a", "b", "a"],
            "target": ["a", "b", "a", "y"],
        }
    )


def _packed_from_spec(
    spec,
    bias=True,
):
    n = len(spec.nodes)
    return PackedLinear(
        spec.source_index,
        spec.target_index,
        n,
        n,
        bias=bias,
    )


def _copy_dense_live_weights(
    packed,
    dense,
    spec,
):
    with torch.no_grad():
        for index, (source, target) in enumerate(
            zip(
                spec.source_index,
                spec.target_index,
            )
        ):
            packed.weight[index] = dense.weight[target, source]
        if packed.bias is not None:
            packed.bias.copy_(dense.bias)


def test_packed_linear_output_shape():
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 1, 2],
        [0, 0, 1],
        2,
        3,
    )
    x = torch.randn(
        4,
        3,
    )
    y = layer(x)
    assert y.shape == (4, 2)


def test_packed_linear_matches_masked_linear_on_parse_adjacency():
    torch.manual_seed(42)
    spec = parse_adjacency(_cyclic_edgelist())
    dense = MaskedLinear(spec.to_mask())
    packed = _packed_from_spec(spec)
    _copy_dense_live_weights(
        packed,
        dense,
        spec,
    )
    x = torch.randn(
        5,
        len(spec.nodes),
    )
    assert torch.allclose(
        packed(x),
        dense(x),
    )


def test_packed_linear_absent_edges_are_not_parameters():
    spec = parse_adjacency(_cyclic_edgelist())
    packed = _packed_from_spec(
        spec,
        bias=False,
    )
    n = len(spec.nodes)
    n_edges = len(spec.source_index)
    assert packed.weight.shape == (n_edges,)
    assert packed.nnz == n_edges
    assert n_edges < n * n
    mask = spec.to_mask()
    assert int(mask.sum().item()) == n_edges
    for source, target in zip(
        spec.source_index,
        spec.target_index,
    ):
        assert mask[target, source].item() == 1.0
    parameters = dict(packed.named_parameters())
    assert "weight" in parameters
    assert isinstance(packed.weight, nn.Parameter)
    assert packed.weight.shape == (n_edges,)


def test_packed_linear_gradients_match_dense_on_live_edges():
    torch.manual_seed(42)
    spec = parse_adjacency(_cyclic_edgelist())
    dense = MaskedLinear(
        spec.to_mask(),
        bias=True,
    )
    packed = _packed_from_spec(spec)
    _copy_dense_live_weights(
        packed,
        dense,
        spec,
    )
    x = torch.randn(
        4,
        len(spec.nodes),
    )
    dense(x).sum().backward()
    packed(x).sum().backward()
    original_grad = dense.parametrizations.weight.original.grad
    assert original_grad is not None
    assert packed.weight.grad is not None
    for index, (source, target) in enumerate(
        zip(
            spec.source_index,
            spec.target_index,
        )
    ):
        torch.testing.assert_close(
            packed.weight.grad[index],
            original_grad[target, source],
        )
    torch.testing.assert_close(
        packed.bias.grad,
        dense.bias.grad,
    )
    live = set(
        zip(
            spec.source_index,
            spec.target_index,
        )
    )
    n = len(spec.nodes)
    for target in range(n):
        for source in range(n):
            if (source, target) in live:
                continue
            assert original_grad[target, source].item() == 0.0


def test_packed_linear_zero_fan_in_bias_stays_zero():
    torch.manual_seed(42)
    spec = parse_adjacency(_cyclic_edgelist())
    packed = _packed_from_spec(spec)
    target_set = set(spec.target_index)
    for name in spec.input_nodes:
        row = spec.nodes.index(name)
        assert row not in target_set
        assert packed.bias[row].item() == 0.0


def test_packed_linear_rejects_duplicate_indices():
    with pytest.raises(
        Kpnn2Error,
        match="duplicate",
    ):
        PackedLinear(
            [0, 0],
            [1, 1],
            2,
            2,
        )


def test_packed_linear_rejects_out_of_range_index():
    with pytest.raises(
        Kpnn2Error,
        match="source_index",
    ):
        PackedLinear(
            [2],
            [0],
            2,
            2,
        )
    with pytest.raises(
        Kpnn2Error,
        match="target_index",
    ):
        PackedLinear(
            [0],
            [2],
            2,
            2,
        )
    with pytest.raises(
        Kpnn2Error,
        match="source_index",
    ):
        PackedLinear(
            [-1],
            [0],
            2,
            2,
        )


def test_packed_linear_rejects_empty_indices():
    with pytest.raises(
        Kpnn2Error,
        match="at least one",
    ):
        PackedLinear(
            [],
            [],
            2,
            2,
        )
    with pytest.raises(
        Kpnn2Error,
        match="at least one",
    ):
        PackedLinear(
            torch.tensor(
                [],
                dtype=torch.int64,
            ),
            torch.tensor(
                [],
                dtype=torch.int64,
            ),
            2,
            2,
        )


def test_packed_linear_rejects_length_mismatch_and_bad_ndim():
    with pytest.raises(
        Kpnn2Error,
        match="same length",
    ):
        PackedLinear(
            [0, 1],
            [0],
            2,
            2,
        )
    with pytest.raises(
        Kpnn2Error,
        match="1-dimensional",
    ):
        PackedLinear(
            torch.tensor(
                [[0, 1]],
                dtype=torch.int64,
            ),
            torch.tensor(
                [0, 1],
                dtype=torch.int64,
            ),
            2,
            2,
        )


def test_packed_linear_bias_false_has_no_bias_parameter():
    layer = PackedLinear(
        [0, 1],
        [1, 0],
        2,
        2,
        bias=False,
    )
    assert layer.bias is None
    parameters = dict(layer.named_parameters())
    assert "bias" not in parameters
    x = torch.ones(
        3,
        2,
    )
    y = layer(x)
    assert y.shape == (3, 2)


def test_packed_linear_dtype_cast_keeps_integer_indices():
    layer = PackedLinear(
        [0, 1],
        [1, 0],
        2,
        2,
    )
    layer = layer.half()
    assert layer.weight.dtype == torch.float16
    assert layer.bias.dtype == torch.float16
    assert layer.source_index.dtype == torch.int64
    assert layer.target_index.dtype == torch.int64
    x = torch.ones(
        2,
        2,
        dtype=torch.float16,
    )
    y = layer(x)
    assert y.dtype == torch.float16
    assert y.shape == (2, 2)


@pytest.mark.parametrize(
    "apply_cast",
    [
        pytest.param(
            lambda layer: layer.half(),
            id="half",
        ),
        pytest.param(
            lambda layer: layer.to(dtype=torch.bfloat16),
            id="bfloat16",
        ),
        pytest.param(
            lambda layer: layer.double(),
            id="double",
        ),
    ],
)
def test_packed_linear_module_dtype_cast_forward(apply_cast):
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 1],
        [1, 0],
        2,
        2,
    )
    layer = apply_cast(layer)
    assert layer.source_index.dtype == torch.int64
    assert layer.target_index.dtype == torch.int64
    x = torch.ones(
        2,
        2,
        dtype=layer.weight.dtype,
        device=layer.weight.device,
    )
    y = layer(x)
    assert y.dtype == layer.weight.dtype


def test_packed_linear_device_move_keeps_indices_integer():
    layer = PackedLinear(
        [0, 1],
        [1, 0],
        2,
        2,
    )
    layer = layer.to(
        device="cpu",
        dtype=torch.float32,
    )
    assert layer.source_index.dtype == torch.int64
    assert layer.target_index.dtype == torch.int64
    x = torch.ones(
        1,
        2,
    )
    y = layer(x)
    assert y.shape == (1, 2)


def test_packed_linear_state_dict_roundtrip():
    src = PackedLinear(
        [0, 1],
        [1, 0],
        2,
        2,
        bias=False,
    )
    dst = PackedLinear(
        [0, 1],
        [1, 0],
        2,
        2,
        bias=False,
    )
    with torch.no_grad():
        src.weight.fill_(0.5)
        dst.weight.fill_(0.25)
    keys = set(src.state_dict().keys())
    assert "weight" in keys
    assert "source_index" in keys
    assert "target_index" in keys
    assert "index_digest" in keys
    digest = src.state_dict()["index_digest"]
    assert digest.dtype == torch.uint8
    assert digest.shape == (32,)
    assert digest.device.type == "cpu"
    buffers = dict(src.named_buffers())
    assert not any("index_digest" in name for name in buffers)
    result = dst.load_state_dict(src.state_dict())
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    torch.testing.assert_close(
        dst.weight,
        src.weight,
    )
    assert torch.equal(
        dst.source_index,
        src.source_index,
    )


def test_packed_linear_load_rejects_foreign_indices():
    src = PackedLinear(
        [0],
        [0],
        2,
        2,
        bias=False,
    )
    dst = PackedLinear(
        [0],
        [1],
        2,
        2,
        bias=False,
    )
    with torch.no_grad():
        src.weight.fill_(0.5)
        dst.weight.fill_(0.25)
    before = dst.weight.clone()
    with pytest.raises(
        Kpnn2Error,
        match="checkpoint indices do not match",
    ):
        dst.load_state_dict(src.state_dict())
    torch.testing.assert_close(
        dst.weight,
        before,
    )


def test_packed_linear_load_without_index_digest():
    src = PackedLinear(
        [0, 1],
        [1, 0],
        2,
        2,
        bias=False,
    )
    dst = PackedLinear(
        [0, 1],
        [1, 0],
        2,
        2,
        bias=False,
    )
    with torch.no_grad():
        src.weight.fill_(0.5)
        dst.weight.fill_(0.25)
    state = src.state_dict()
    del state["index_digest"]
    result = dst.load_state_dict(
        state,
        strict=True,
    )
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    torch.testing.assert_close(
        dst.weight,
        src.weight,
    )


def test_packed_linear_index_digest_uses_live_indices():
    layer = PackedLinear(
        [0, 1],
        [1, 0],
        2,
        2,
        bias=False,
    )
    source_bytes = (
        layer.source_index.detach()
        .cpu()
        .contiguous()
        .to(torch.int64)
        .numpy()
        .tobytes()
    )
    target_bytes = (
        layer.target_index.detach()
        .cpu()
        .contiguous()
        .to(torch.int64)
        .numpy()
        .tobytes()
    )
    sizes = struct.pack(
        "<qq",
        2,
        2,
    )
    digest = hashlib.sha256(source_bytes + target_bytes + sizes).digest()
    expected = torch.tensor(
        tuple(digest),
        dtype=torch.uint8,
    )
    assert torch.equal(
        layer.state_dict()["index_digest"],
        expected,
    )


def test_packed_linear_deepcopy_independent_parameters():
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 1],
        [1, 0],
        2,
        2,
    )
    copied = copy.deepcopy(layer)
    assert copied is not layer
    assert copied.weight is not layer.weight
    assert copied.bias is not layer.bias
    assert copied.source_index is not layer.source_index
    assert copied.source_index.data_ptr() != layer.source_index.data_ptr()
    torch.testing.assert_close(
        copied.weight,
        layer.weight,
    )
    torch.testing.assert_close(
        copied.bias,
        layer.bias,
    )
    x = torch.ones(
        2,
        2,
    )
    torch.testing.assert_close(
        copied(x),
        layer(x),
    )
    with torch.no_grad():
        copied.weight.fill_(0.0)
    assert not torch.equal(
        copied.weight,
        layer.weight,
    )


def test_packed_linear_compiles_without_a_graph_break():
    dynamo = pytest.importorskip("torch._dynamo")
    if not dynamo.is_dynamo_supported():
        pytest.skip("torch.compile is not supported here")
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 2, 1, 3],
        [1, 0, 3, 2],
        4,
        4,
    )
    x = torch.randn(
        4,
        4,
    )
    expected = layer(x)
    dynamo.reset()
    compiled = torch.compile(
        layer,
        fullgraph=True,
        backend="eager",
    )
    torch.testing.assert_close(
        compiled(x),
        expected,
    )


def test_packed_construction_does_not_call_to_mask(monkeypatch):
    spec = parse_adjacency(_cyclic_edgelist())
    assert not hasattr(
        spec,
        "mask",
    )

    def boom(self):
        raise AssertionError("to_mask should not be called")

    monkeypatch.setattr(
        AdjacencySpec,
        "to_mask",
        boom,
    )
    packed = PackedLinear(
        spec.source_index,
        spec.target_index,
        len(spec.nodes),
        len(spec.nodes),
    )
    assert packed.nnz == len(spec.source_index)
    assert not hasattr(
        spec,
        "mask",
    )


def test_packed_linear_init_respects_degree_bound():
    torch.manual_seed(42)
    layer = PackedLinear(
        [0],
        [0],
        1,
        32,
    )
    fan_in = 1
    bound = 1.0 / math.sqrt(fan_in)
    eps = 1e-6
    assert torch.all(layer.weight.abs() <= bound + eps)
    assert abs(layer.bias[0].item()) <= bound + eps


def test_packed_linear_repr_reports_sizes():
    layer = PackedLinear(
        [0, 1],
        [1, 0],
        2,
        2,
        bias=False,
    )
    text = repr(layer)
    assert "in_features=2" in text
    assert "out_features=2" in text
    assert "nnz=2" in text
    assert "bias=False" in text
