import copy

import pandas as pd
import pytest
import torch

from kpnn2 import (
    AdjacencySpec,
    Kpnn2Error,
    PackedMultiheadAttention,
    parse_adjacency,
)


def _tiny_edgelist():
    return pd.DataFrame(
        {
            "source": ["x", "a", "b", "a", "a"],
            "target": ["a", "b", "a", "a", "y"],
        }
    )


def _attn_from_spec(
    spec,
    embed_dim=8,
    num_heads=2,
    add_self_loops=False,
    bias=True,
    batch_first=True,
):
    n = len(spec.nodes)
    return PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        n,
        n,
        embed_dim,
        num_heads,
        bias=bias,
        add_self_loops=add_self_loops,
        batch_first=batch_first,
    )


def _post_self_loop_pairs(
    source_index,
    target_index,
    n,
):
    pairs = list(
        zip(
            source_index,
            target_index,
        )
    )
    existing = set(pairs)
    for node in range(n):
        pair = (node, node)
        if pair not in existing:
            pairs.append(pair)
    return pairs


def test_forward_returns_tuple_matching_query_shape():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_edgelist())
    layer = _attn_from_spec(
        spec,
        batch_first=True,
    )
    n = len(spec.nodes)
    query = torch.randn(
        3,
        n,
        8,
    )
    result = layer(
        query,
        query,
        query,
        need_weights=False,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    attn_out, weights = result
    assert attn_out.shape == query.shape
    assert weights is None


def test_self_attention_is_finite():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_edgelist())
    layer = _attn_from_spec(spec)
    n = len(spec.nodes)
    x = torch.randn(
        2,
        n,
        8,
    )
    attn_out, weights = layer(
        x,
        x,
        x,
        need_weights=False,
    )
    assert weights is None
    assert torch.isfinite(attn_out).all()


def test_attributes_match_construction():
    spec = parse_adjacency(_tiny_edgelist())
    n = len(spec.nodes)
    embed_dim = 8
    num_heads = 2
    plain = PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        n,
        n,
        embed_dim,
        num_heads,
        add_self_loops=False,
    )
    assert plain.query_features == n
    assert plain.key_features == n
    assert plain.embed_dim == embed_dim
    assert plain.nnz == len(spec.source_index)
    assert plain.source_index.tolist() == list(spec.source_index)
    assert plain.target_index.tolist() == list(spec.target_index)

    source = [0, 1]
    target = [1, 0]
    raw = PackedMultiheadAttention(
        source,
        target,
        2,
        2,
        embed_dim,
        num_heads,
        add_self_loops=True,
    )
    expected = _post_self_loop_pairs(
        source,
        target,
        2,
    )
    assert raw.query_features == 2
    assert raw.key_features == 2
    assert raw.embed_dim == embed_dim
    assert raw.nnz == len(expected)
    assert (
        list(
            zip(
                raw.source_index.tolist(),
                raw.target_index.tolist(),
            )
        )
        == expected
    )


def test_add_self_loops_increases_nnz_without_mutating_caller():
    spec = parse_adjacency(_tiny_edgelist())
    n = len(spec.nodes)
    source_tuple = spec.source_index
    target_tuple = spec.target_index
    caller_source = list(source_tuple)
    caller_target = list(target_tuple)
    nnz_before = len(source_tuple)
    expected = _post_self_loop_pairs(
        source_tuple,
        target_tuple,
        n,
    )
    missing = len(expected) - nnz_before
    assert missing > 0
    layer = PackedMultiheadAttention(
        caller_source,
        caller_target,
        n,
        n,
        8,
        2,
        add_self_loops=True,
    )
    assert spec.source_index == source_tuple
    assert spec.target_index == target_tuple
    assert caller_source == list(source_tuple)
    assert caller_target == list(target_tuple)
    assert layer.nnz == nnz_before + missing
    assert (
        list(
            zip(
                layer.source_index.tolist(),
                layer.target_index.tolist(),
            )
        )
        == expected
    )


def test_isolated_query_output_is_finite():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_edgelist())
    layer = _attn_from_spec(
        spec,
        add_self_loops=False,
    )
    n = len(spec.nodes)
    x = torch.randn(
        2,
        n,
        8,
    )
    attn_out, _ = layer(
        x,
        x,
        x,
        need_weights=False,
    )
    isolated = [spec.nodes.index(name) for name in spec.input_nodes]
    assert isolated
    for index in isolated:
        slice_ = attn_out[
            :,
            index,
            :,
        ]
        assert torch.isfinite(slice_).all()
        assert not torch.isnan(slice_).any()


def test_rejects_duplicate_indices():
    with pytest.raises(
        Kpnn2Error,
        match="duplicate",
    ):
        PackedMultiheadAttention(
            [0, 0],
            [1, 1],
            2,
            2,
            8,
            2,
        )


def test_rejects_embed_dim_not_divisible_by_num_heads():
    with pytest.raises(
        Kpnn2Error,
        match="divisible",
    ):
        PackedMultiheadAttention(
            [0],
            [0],
            1,
            1,
            5,
            2,
        )


def test_rejects_empty_indices():
    with pytest.raises(
        Kpnn2Error,
        match="at least one",
    ):
        PackedMultiheadAttention(
            [],
            [],
            2,
            2,
            8,
            2,
        )
    with pytest.raises(
        Kpnn2Error,
        match="at least one",
    ):
        PackedMultiheadAttention(
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
            8,
            2,
        )


def test_rejects_length_mismatch_and_out_of_range_index():
    with pytest.raises(
        Kpnn2Error,
        match="same length",
    ):
        PackedMultiheadAttention(
            [0, 1],
            [0],
            2,
            2,
            8,
            2,
        )
    with pytest.raises(
        Kpnn2Error,
        match="source_index",
    ):
        PackedMultiheadAttention(
            [2],
            [0],
            2,
            2,
            8,
            2,
        )
    with pytest.raises(
        Kpnn2Error,
        match="target_index",
    ):
        PackedMultiheadAttention(
            [0],
            [2],
            2,
            2,
            8,
            2,
        )
    with pytest.raises(
        Kpnn2Error,
        match="source_index",
    ):
        PackedMultiheadAttention(
            [-1],
            [0],
            2,
            2,
            8,
            2,
        )


def test_need_weights_true_raises():
    torch.manual_seed(42)
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
    )
    x = torch.randn(
        2,
        2,
        8,
    )
    with pytest.raises(
        Kpnn2Error,
        match="need_weights",
    ):
        layer(
            x,
            x,
            x,
            need_weights=True,
        )


def test_attn_mask_not_none_raises():
    torch.manual_seed(42)
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
    )
    x = torch.randn(
        2,
        2,
        8,
    )
    attn_mask = torch.zeros(
        2,
        2,
    )
    with pytest.raises(
        Kpnn2Error,
        match="attn_mask",
    ):
        layer(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=attn_mask,
        )


def test_is_causal_true_raises():
    torch.manual_seed(42)
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
    )
    x = torch.randn(
        2,
        2,
        8,
    )
    with pytest.raises(
        Kpnn2Error,
        match="is_causal",
    ):
        layer(
            x,
            x,
            x,
            need_weights=False,
            is_causal=True,
        )


def test_kdim_or_vdim_not_embed_dim_raises():
    with pytest.raises(
        Kpnn2Error,
        match="kdim",
    ):
        PackedMultiheadAttention(
            [0],
            [0],
            1,
            1,
            8,
            2,
            kdim=4,
        )
    with pytest.raises(
        Kpnn2Error,
        match="vdim",
    ):
        PackedMultiheadAttention(
            [0],
            [0],
            1,
            1,
            8,
            2,
            vdim=4,
        )


def test_key_padding_mask_all_true_output_finite():
    torch.manual_seed(42)
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
    )
    query = torch.randn(
        3,
        2,
        8,
    )
    key_padding_mask = torch.ones(
        3,
        2,
        dtype=torch.bool,
    )
    attn_out, weights = layer(
        query,
        query,
        query,
        key_padding_mask=key_padding_mask,
        need_weights=False,
    )
    assert attn_out.shape == query.shape
    assert weights is None
    assert torch.isfinite(attn_out).all()


def test_key_padding_mask_float_raises():
    torch.manual_seed(42)
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
    )
    query = torch.randn(
        3,
        2,
        8,
    )
    with pytest.raises(
        Kpnn2Error,
        match="boolean",
    ):
        layer(
            query,
            query,
            query,
            key_padding_mask=torch.zeros(
                3,
                2,
            ),
            need_weights=False,
        )


def test_key_padding_mask_moves_to_scores_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(42)
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
    )
    layer = layer.to("cuda")
    query = torch.randn(
        3,
        2,
        8,
        device="cuda",
    )
    key_padding_mask = torch.zeros(
        3,
        2,
        dtype=torch.bool,
    )
    assert key_padding_mask.device.type == "cpu"
    attn_out, weights = layer(
        query,
        query,
        query,
        key_padding_mask=key_padding_mask,
        need_weights=False,
    )
    assert attn_out.device.type == "cuda"
    assert attn_out.shape == query.shape
    assert weights is None
    assert torch.isfinite(attn_out).all()


def test_bias_false_has_no_projection_bias():
    torch.manual_seed(42)
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
        bias=False,
    )
    assert layer.q_proj.bias is None
    assert layer.k_proj.bias is None
    assert layer.v_proj.bias is None
    assert layer.out_proj.bias is None
    x = torch.randn(
        2,
        2,
        8,
    )
    result = layer(
        x,
        x,
        x,
        need_weights=False,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    attn_out, weights = result
    assert attn_out.shape == x.shape
    assert weights is None


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
def test_module_dtype_cast_keeps_integer_indices(apply_cast):
    torch.manual_seed(42)
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
    )
    layer = apply_cast(layer)
    assert layer.source_index.dtype == torch.int64
    assert layer.target_index.dtype == torch.int64
    dtype = layer.q_proj.weight.dtype
    x = torch.ones(
        2,
        2,
        8,
        dtype=dtype,
        device=layer.q_proj.weight.device,
    )
    attn_out, weights = layer(
        x,
        x,
        x,
        need_weights=False,
    )
    assert attn_out.dtype == dtype
    assert weights is None


def test_state_dict_contains_weights_indices_and_digest():
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
        bias=False,
    )
    keys = set(layer.state_dict().keys())
    assert "q_proj.weight" in keys
    assert "k_proj.weight" in keys
    assert "v_proj.weight" in keys
    assert "out_proj.weight" in keys
    assert "source_index" in keys
    assert "target_index" in keys
    assert "index_digest" in keys
    digest = layer.state_dict()["index_digest"]
    assert digest.dtype == torch.uint8
    assert digest.shape == (32,)
    assert digest.device.type == "cpu"
    buffers = dict(layer.named_buffers())
    assert not any("index_digest" in name for name in buffers)


def test_load_rejects_foreign_indices():
    src = PackedMultiheadAttention(
        [0],
        [0],
        2,
        2,
        8,
        2,
        bias=False,
    )
    dst = PackedMultiheadAttention(
        [0],
        [1],
        2,
        2,
        8,
        2,
        bias=False,
    )
    with torch.no_grad():
        src.q_proj.weight.fill_(0.5)
        dst.q_proj.weight.fill_(0.25)
        src.k_proj.weight.fill_(0.5)
        dst.k_proj.weight.fill_(0.25)
        src.v_proj.weight.fill_(0.5)
        dst.v_proj.weight.fill_(0.25)
        src.out_proj.weight.fill_(0.5)
        dst.out_proj.weight.fill_(0.25)
    before = {
        name: tensor.detach().clone() for name, tensor in dst.named_parameters()
    }
    with pytest.raises(
        Kpnn2Error,
        match="checkpoint indices do not match",
    ):
        dst.load_state_dict(src.state_dict())
    for name, tensor in dst.named_parameters():
        torch.testing.assert_close(
            tensor,
            before[name],
        )


def test_load_without_index_digest():
    src = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
        bias=False,
    )
    dst = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
        bias=False,
    )
    with torch.no_grad():
        src.q_proj.weight.fill_(0.5)
        dst.q_proj.weight.fill_(0.25)
        src.k_proj.weight.fill_(0.5)
        dst.k_proj.weight.fill_(0.25)
        src.v_proj.weight.fill_(0.5)
        dst.v_proj.weight.fill_(0.25)
        src.out_proj.weight.fill_(0.5)
        dst.out_proj.weight.fill_(0.25)
    state = src.state_dict()
    del state["index_digest"]
    result = dst.load_state_dict(
        state,
        strict=True,
    )
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    torch.testing.assert_close(
        dst.q_proj.weight,
        src.q_proj.weight,
    )
    torch.testing.assert_close(
        dst.k_proj.weight,
        src.k_proj.weight,
    )
    torch.testing.assert_close(
        dst.v_proj.weight,
        src.v_proj.weight,
    )
    torch.testing.assert_close(
        dst.out_proj.weight,
        src.out_proj.weight,
    )


def test_deepcopy_independent_parameters():
    torch.manual_seed(42)
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
    )
    copied = copy.deepcopy(layer)
    assert copied is not layer
    assert copied.q_proj.weight is not layer.q_proj.weight
    assert copied.k_proj.weight is not layer.k_proj.weight
    assert copied.v_proj.weight is not layer.v_proj.weight
    assert copied.out_proj.weight is not layer.out_proj.weight
    assert copied.q_proj.weight.data_ptr() != layer.q_proj.weight.data_ptr()
    x = torch.randn(
        2,
        2,
        8,
    )
    out_orig, _ = layer(
        x,
        x,
        x,
        need_weights=False,
    )
    out_copy, _ = copied(
        x,
        x,
        x,
        need_weights=False,
    )
    torch.testing.assert_close(
        out_copy,
        out_orig,
    )
    with torch.no_grad():
        copied.q_proj.weight.fill_(0.0)
    assert not torch.equal(
        copied.q_proj.weight,
        layer.q_proj.weight,
    )


def test_repr_reports_sizes():
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
    )
    extra = layer.extra_repr()
    text = repr(layer)
    assert "embed_dim=8" in extra
    assert "num_heads=2" in extra
    assert "nnz=2" in extra
    assert "embed_dim=8" in text
    assert "num_heads=2" in text
    assert "nnz=2" in text


def test_construction_does_not_call_to_mask(monkeypatch):
    spec = parse_adjacency(_tiny_edgelist())
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
    layer = PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        len(spec.nodes),
        len(spec.nodes),
        8,
        2,
    )
    assert layer.nnz == len(spec.source_index)
    assert not hasattr(
        spec,
        "mask",
    )


def test_compiles_without_a_graph_break():
    dynamo = pytest.importorskip("torch._dynamo")
    if not dynamo.is_dynamo_supported():
        pytest.skip("torch.compile is not supported here")
    torch.manual_seed(42)
    layer = PackedMultiheadAttention(
        [0, 2, 1, 3],
        [1, 0, 3, 2],
        4,
        4,
        8,
        2,
    )
    x = torch.randn(
        2,
        4,
        8,
    )
    expected, _ = layer(
        x,
        x,
        x,
        need_weights=False,
    )
    dynamo.reset()
    compiled = torch.compile(
        layer,
        fullgraph=True,
        backend="eager",
    )
    got, got_weights = compiled(
        x,
        x,
        x,
        need_weights=False,
    )
    torch.testing.assert_close(
        got,
        expected,
    )
    assert got_weights is None


def test_unbatched_2d_forward():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_edgelist())
    layer = _attn_from_spec(spec)
    n = len(spec.nodes)
    x = torch.randn(
        n,
        8,
    )
    attn_out, weights = layer(
        x,
        x,
        x,
        need_weights=False,
    )
    assert attn_out.shape == x.shape
    assert weights is None
    assert torch.isfinite(attn_out).all()


def test_batch_first_false_returns_seq_major_layout():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_edgelist())
    layer = _attn_from_spec(
        spec,
        batch_first=False,
    )
    n = len(spec.nodes)
    x = torch.randn(
        n,
        3,
        8,
    )
    attn_out, weights = layer(
        x,
        x,
        x,
        need_weights=False,
    )
    assert attn_out.shape == x.shape
    assert weights is None
    assert torch.isfinite(attn_out).all()
