import copy
import hashlib
import struct

import pandas as pd
import pytest
import torch

from kpnn2 import (
    AdjacencySpec,
    Kpnn2Error,
    PackedMultiheadAttention,
    parse_adjacency,
)
from tests.helpers.packed_attention import (
    dense_packed_weights,
    pin_projections_identity,
    shape_heads,
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


def test_need_weights_true_returns_packed():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_edgelist())
    layer = _attn_from_spec(spec)
    layer.eval()
    n = len(spec.nodes)
    x = torch.randn(
        3,
        n,
        8,
    )
    out_off, none_w = layer(
        x,
        x,
        x,
        need_weights=False,
    )
    out_on, weights = layer(
        x,
        x,
        x,
        need_weights=True,
    )
    assert none_w is None
    assert weights is not None
    assert weights.shape == (3, layer.nnz)
    assert weights.shape != (3, n, n)
    torch.testing.assert_close(
        out_off,
        out_on,
    )


def test_average_attn_weights_false_keeps_heads():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_edgelist())
    layer = _attn_from_spec(spec)
    n = len(spec.nodes)
    x = torch.randn(
        3,
        n,
        8,
    )
    _, weights = layer(
        x,
        x,
        x,
        need_weights=True,
        average_attn_weights=False,
    )
    assert weights.shape == (
        3,
        layer.nnz,
        layer.num_heads,
    )
    _, averaged = layer(
        x,
        x,
        x,
        need_weights=True,
    )
    torch.testing.assert_close(
        averaged,
        weights.mean(dim=-1),
    )


def test_need_weights_false_ignores_average_flag():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_edgelist())
    layer = _attn_from_spec(spec)
    n = len(spec.nodes)
    x = torch.randn(
        2,
        n,
        8,
    )
    _, weights = layer(
        x,
        x,
        x,
        need_weights=False,
        average_attn_weights=False,
    )
    assert weights is None


def test_packed_weights_match_dense_live_pairs():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_edgelist())
    layer = _attn_from_spec(spec)
    pin_projections_identity(layer)
    layer.eval()
    n = len(spec.nodes)
    x = torch.randn(
        4,
        n,
        8,
    )
    _, packed = layer(
        x,
        x,
        x,
        need_weights=True,
        average_attn_weights=False,
    )
    headed = shape_heads(
        x,
        layer.num_heads,
    )
    expected = dense_packed_weights(
        headed,
        headed,
        layer.source_index,
        layer.target_index,
    )
    torch.testing.assert_close(
        packed,
        expected,
    )


def test_packed_weights_unbatched_and_seq_major():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_edgelist())
    layer = _attn_from_spec(spec)
    n = len(spec.nodes)
    x = torch.randn(
        n,
        8,
    )
    _, averaged = layer(
        x,
        x,
        x,
        need_weights=True,
    )
    assert averaged.shape == (layer.nnz,)
    _, per_head = layer(
        x,
        x,
        x,
        need_weights=True,
        average_attn_weights=False,
    )
    assert per_head.shape == (
        layer.nnz,
        layer.num_heads,
    )
    seq_layer = _attn_from_spec(
        spec,
        batch_first=False,
    )
    x_seq = torch.randn(
        n,
        3,
        8,
    )
    _, seq_avg = seq_layer(
        x_seq,
        x_seq,
        x_seq,
        need_weights=True,
    )
    assert seq_avg.shape == (seq_layer.nnz, 3)
    _, seq_heads = seq_layer(
        x_seq,
        x_seq,
        x_seq,
        need_weights=True,
        average_attn_weights=False,
    )
    assert seq_heads.shape == (
        seq_layer.nnz,
        3,
        seq_layer.num_heads,
    )


def test_packed_weights_length_matches_nnz_with_self_loops():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_edgelist())
    layer = _attn_from_spec(
        spec,
        add_self_loops=True,
    )
    assert layer.nnz > len(spec.source_index)
    n = len(spec.nodes)
    x = torch.randn(
        2,
        n,
        8,
    )
    _, weights = layer(
        x,
        x,
        x,
        need_weights=True,
    )
    assert weights.shape == (2, layer.nnz)
    edges = spec.to_edgelist()
    assert weights.shape[-1] != len(edges)


def test_packed_weights_align_with_to_edgelist():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_edgelist())
    layer = _attn_from_spec(
        spec,
        add_self_loops=False,
    )
    edges = spec.to_edgelist()
    n = len(spec.nodes)
    x = torch.randn(
        2,
        n,
        8,
    )
    _, weights = layer(
        x,
        x,
        x,
        need_weights=True,
    )
    assert weights.shape[-1] == len(edges)


def test_packed_weights_sum_to_one_per_live_query():
    torch.manual_seed(42)
    spec = parse_adjacency(_tiny_edgelist())
    layer = _attn_from_spec(spec)
    pin_projections_identity(layer)
    layer.eval()
    n = len(spec.nodes)
    x = torch.randn(
        3,
        n,
        8,
    )
    _, weights = layer(
        x,
        x,
        x,
        need_weights=True,
        average_attn_weights=False,
    )
    target = layer.target_index
    for query_pos in range(n):
        live = target == query_pos
        if not bool(live.any()):
            continue
        summed = weights[:, live, :].sum(dim=1)
        torch.testing.assert_close(
            summed,
            torch.ones_like(summed),
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


def test_module_autocast_keeps_parameter_dtype():
    torch.manual_seed(42)
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
    )
    x = torch.ones(
        2,
        2,
        8,
        dtype=torch.float32,
    )
    expected, _ = layer(
        x,
        x,
        x,
        need_weights=False,
    )
    with torch.autocast(
        device_type="cpu",
        dtype=torch.bfloat16,
    ):
        attn_out, weights = layer(
            x,
            x,
            x,
            need_weights=False,
        )
        attn_from_half, _ = layer(
            x.to(dtype=torch.float16),
            x.to(dtype=torch.float16),
            x.to(dtype=torch.float16),
            need_weights=False,
        )
    assert attn_out.dtype == torch.float32
    assert attn_from_half.dtype == torch.float32
    assert weights is None
    torch.testing.assert_close(
        attn_out,
        expected,
    )
    torch.testing.assert_close(
        attn_from_half,
        expected,
    )


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


def test_index_digest_includes_embed_dim_and_num_heads():
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
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
        "<qqqq",
        2,
        2,
        8,
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


def test_load_rejects_foreign_num_heads():
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
        4,
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
    before = dst.q_proj.weight.detach().clone()
    with pytest.raises(
        Kpnn2Error,
        match="checkpoint indices do not match",
    ):
        dst.load_state_dict(src.state_dict())
    torch.testing.assert_close(
        dst.q_proj.weight,
        before,
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


def test_identity_in_state_dict():
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
        bias=False,
        identity="abc",
    )
    assert layer.identity == "abc"
    saved = layer.state_dict()["identity"]
    assert saved.dtype == torch.uint8
    assert saved.device.type == "cpu"
    assert saved.tolist() == list(b"abc")
    buffers = dict(layer.named_buffers())
    assert not any("identity" in name for name in buffers)


def test_identity_omitted_from_state_dict():
    layer = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
        bias=False,
    )
    assert layer.identity is None
    assert "identity" not in layer.state_dict()


def _tiny_attn(**kwargs):
    return PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        4,
        2,
        bias=False,
        **kwargs,
    )


def _assert_same_projections(
    left,
    right,
):
    torch.testing.assert_close(
        left.q_proj.weight,
        right.q_proj.weight,
    )
    torch.testing.assert_close(
        left.k_proj.weight,
        right.k_proj.weight,
    )
    torch.testing.assert_close(
        left.v_proj.weight,
        right.v_proj.weight,
    )
    torch.testing.assert_close(
        left.out_proj.weight,
        right.out_proj.weight,
    )


def test_packed_mha_generator_is_keyword_only():
    with pytest.raises(TypeError):
        PackedMultiheadAttention(
            [0, 1],
            [1, 0],
            2,
            2,
            4,
            2,
            0.0,
            False,
            None,
            None,
            True,
            False,
            torch.Generator(),
        )


def test_packed_mha_generator_rejects_non_generator():
    with pytest.raises(
        Kpnn2Error,
        match="generator",
    ):
        _tiny_attn(generator=42)


def test_packed_mha_generator_none_matches_omitted():
    torch.manual_seed(42)
    omitted = _tiny_attn()
    torch.manual_seed(42)
    explicit = _tiny_attn(generator=None)
    _assert_same_projections(
        omitted,
        explicit,
    )


def test_packed_mha_generator_ignores_global_draws():
    torch.manual_seed(0)
    torch.randn(8)
    g = torch.Generator().manual_seed(42)
    polluted = _tiny_attn(generator=g)
    g2 = torch.Generator().manual_seed(42)
    clean = _tiny_attn(generator=g2)
    _assert_same_projections(
        polluted,
        clean,
    )
    assert not hasattr(
        polluted,
        "generator",
    )


def test_packed_mha_generator_does_not_shift_global_stream():
    torch.manual_seed(0)
    expected = torch.randn(3)
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(42)
    _tiny_attn(generator=g)
    got = torch.randn(3)
    torch.testing.assert_close(
        got,
        expected,
    )


def test_packed_mha_omitted_generator_shifts_global_stream():
    torch.manual_seed(0)
    expected = torch.randn(3)
    torch.manual_seed(0)
    _tiny_attn()
    got = torch.randn(3)
    assert not torch.equal(
        got,
        expected,
    )


def test_packed_mha_generator_skips_linear_kaiming_stream():
    torch.manual_seed(42)
    global_layer = _tiny_attn()
    isolated = _tiny_attn(
        generator=torch.Generator().manual_seed(42),
    )
    assert not torch.equal(
        global_layer.q_proj.weight,
        isolated.q_proj.weight,
    )


def test_load_rejects_foreign_identity():
    src = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
        bias=False,
        identity="left",
    )
    dst = PackedMultiheadAttention(
        [0, 1],
        [1, 0],
        2,
        2,
        8,
        2,
        bias=False,
        identity="right",
    )
    with torch.no_grad():
        src.q_proj.weight.fill_(0.5)
        dst.q_proj.weight.fill_(0.25)
    before = dst.q_proj.weight.detach().clone()
    with pytest.raises(
        Kpnn2Error,
        match="checkpoint identity does not match",
    ):
        dst.load_state_dict(src.state_dict())
    torch.testing.assert_close(
        dst.q_proj.weight,
        before,
    )


def test_load_rejects_renamed_nodes():
    original = parse_adjacency(
        pd.DataFrame(
            {
                "source": ["g1"],
                "target": ["g2"],
            }
        )
    )
    renamed = parse_adjacency(
        pd.DataFrame(
            {
                "source": ["g1b"],
                "target": ["g2"],
            }
        )
    )
    assert original.source_index == renamed.source_index
    assert original.target_index == renamed.target_index
    assert original.fingerprint != renamed.fingerprint
    n = len(original.nodes)
    src = PackedMultiheadAttention(
        original.source_index,
        original.target_index,
        n,
        n,
        8,
        2,
        bias=False,
        identity=original.fingerprint,
    )
    dst = PackedMultiheadAttention(
        renamed.source_index,
        renamed.target_index,
        n,
        n,
        8,
        2,
        bias=False,
        identity=renamed.fingerprint,
    )
    with torch.no_grad():
        src.q_proj.weight.fill_(0.5)
        dst.q_proj.weight.fill_(0.25)
    before = dst.q_proj.weight.detach().clone()
    with pytest.raises(
        Kpnn2Error,
        match="checkpoint identity does not match",
    ):
        dst.load_state_dict(src.state_dict())
    torch.testing.assert_close(
        dst.q_proj.weight,
        before,
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
