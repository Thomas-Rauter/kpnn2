import copy
import inspect

import pytest
import torch

from kpnn2 import (
    Kpnn2Error,
    PackedMultiheadAttention,
    parse_adjacency,
)
from kpnn2._packed_multihead_attention import (
    _packed_attention,
    _packed_attention_chunked,
    _padding_participate,
)
from tests.helpers.packed_attention import (
    cyclic_edgelist,
    dense_masked_attention,
    dense_packed_weights,
    rectangular_indices,
    shape_heads,
)


def _assert_close(
    actual,
    expected,
):
    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


def _index_tensor(index):
    return torch.tensor(
        index,
        dtype=torch.int64,
    )


def _cyclic_indices():
    spec = parse_adjacency(cyclic_edgelist())
    source_index = _index_tensor(spec.source_index)
    target_index = _index_tensor(spec.target_index)
    return source_index, target_index, len(spec.nodes)


def _shaped_qkv(
    batch,
    n_query,
    n_key,
    num_heads,
    head_dim,
):
    embed_dim = num_heads * head_dim
    query = shape_heads(
        torch.randn(
            batch,
            n_query,
            embed_dim,
        ),
        num_heads,
    )
    key = shape_heads(
        torch.randn(
            batch,
            n_key,
            embed_dim,
        ),
        num_heads,
    )
    value = shape_heads(
        torch.randn(
            batch,
            n_key,
            embed_dim,
        ),
        num_heads,
    )
    return query, key, value


def _tiny_attn(**kwargs):
    return PackedMultiheadAttention(
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        2,
        2,
        4,
        2,
        bias=False,
        **kwargs,
    )


def test_chunk_size_is_keyword_only():
    params = inspect.signature(PackedMultiheadAttention).parameters
    assert params["chunk_size"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["chunk_size"].default is None


def test_chunk_size_none_is_default():
    layer = _tiny_attn()
    assert layer.chunk_size is None
    explicit = _tiny_attn(chunk_size=None)
    assert explicit.chunk_size is None


def test_chunk_size_positive_int_is_stored():
    layer = _tiny_attn(chunk_size=1)
    assert layer.chunk_size == 1
    extra = layer.extra_repr()
    assert "chunk_size=1" in extra


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, 1.5, "4"],
)
def test_chunk_size_rejects_invalid(value):
    with pytest.raises(
        Kpnn2Error,
        match="chunk_size",
    ):
        _tiny_attn(chunk_size=value)


def test_chunked_kernel_matches_all_pairs_on_cyclic():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    query, key, value = _shaped_qkv(
        4,
        n,
        n,
        2,
        4,
    )
    mix, attn = _packed_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    for chunk_size in (1, 2, 3, 999):
        chunk_mix, chunk_attn = _packed_attention_chunked(
            query,
            key,
            value,
            source_index,
            target_index,
            dropout_p=0.0,
            training=False,
            chunk_size=chunk_size,
        )
        _assert_close(
            chunk_mix,
            mix,
        )
        _assert_close(
            chunk_attn,
            attn,
        )


def test_chunked_kernel_matches_dense_on_rectangular():
    torch.manual_seed(42)
    (
        source_list,
        target_list,
        query_features,
        key_features,
    ) = rectangular_indices()
    source_index = _index_tensor(source_list)
    target_index = _index_tensor(target_list)
    query, key, value = _shaped_qkv(
        3,
        query_features,
        key_features,
        2,
        4,
    )
    dense = dense_masked_attention(
        query,
        key,
        value,
        source_index,
        target_index,
    )
    mix, attn = _packed_attention_chunked(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
        chunk_size=1,
    )
    _assert_close(
        mix,
        dense,
    )
    packed = dense_packed_weights(
        query,
        key,
        source_index,
        target_index,
    )
    _assert_close(
        attn,
        packed,
    )


def test_chunked_kernel_matches_all_pairs_with_padding():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    batch = 4
    query, key, value = _shaped_qkv(
        batch,
        n,
        n,
        2,
        4,
    )
    key_padding_mask = torch.zeros(
        batch,
        n,
        dtype=torch.bool,
    )
    key_padding_mask[0, 1] = True
    key_padding_mask[2, 0] = True
    participate = _padding_participate(
        key_padding_mask,
        source_index,
        (batch,),
        n,
        query.device,
    )
    mix, attn = _packed_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
        participate=participate,
    )
    chunk_mix, chunk_attn = _packed_attention_chunked(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
        chunk_size=1,
        participate=participate,
    )
    _assert_close(
        chunk_mix,
        mix,
    )
    _assert_close(
        chunk_attn,
        attn,
    )


def test_chunked_eval_dropout_does_not_change_the_mix():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    query, key, value = _shaped_qkv(
        4,
        n,
        n,
        2,
        4,
    )
    mix_off, _ = _packed_attention_chunked(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
        chunk_size=1,
    )
    mix_eval, _ = _packed_attention_chunked(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.5,
        training=False,
        chunk_size=1,
    )
    _assert_close(
        mix_eval,
        mix_off,
    )


def _leaf_qkv(
    query,
    key,
    value,
):
    return (
        query.detach().clone().requires_grad_(True),
        key.detach().clone().requires_grad_(True),
        value.detach().clone().requires_grad_(True),
    )


def _grad_or_zeros(
    grad,
    like,
):
    if grad is None:
        return torch.zeros_like(like)
    return grad


def test_chunked_kernel_grads_match_all_pairs():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    query, key, value = _shaped_qkv(
        4,
        n,
        n,
        2,
        4,
    )
    q_a, k_a, v_a = _leaf_qkv(
        query,
        key,
        value,
    )
    mix_a, attn_a = _packed_attention(
        q_a,
        k_a,
        v_a,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    (mix_a.sum() + attn_a.sum()).backward()
    q_b, k_b, v_b = _leaf_qkv(
        query,
        key,
        value,
    )
    mix_b, attn_b = _packed_attention_chunked(
        q_b,
        k_b,
        v_b,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
        chunk_size=1,
    )
    (mix_b.sum() + attn_b.sum()).backward()
    _assert_close(
        _grad_or_zeros(
            q_b.grad,
            query,
        ),
        _grad_or_zeros(
            q_a.grad,
            query,
        ),
    )
    _assert_close(
        _grad_or_zeros(
            k_b.grad,
            key,
        ),
        _grad_or_zeros(
            k_a.grad,
            key,
        ),
    )
    _assert_close(
        _grad_or_zeros(
            v_b.grad,
            value,
        ),
        _grad_or_zeros(
            v_a.grad,
            value,
        ),
    )


def test_chunked_kernel_grads_match_dense():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    query, key, value = _shaped_qkv(
        4,
        n,
        n,
        2,
        4,
    )
    q_p, k_p, v_p = _leaf_qkv(
        query,
        key,
        value,
    )
    mix_p, _ = _packed_attention_chunked(
        q_p,
        k_p,
        v_p,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
        chunk_size=2,
    )
    mix_p.sum().backward()
    q_d, k_d, v_d = _leaf_qkv(
        query,
        key,
        value,
    )
    dense = dense_masked_attention(
        q_d,
        k_d,
        v_d,
        source_index,
        target_index,
    )
    dense.sum().backward()
    _assert_close(
        _grad_or_zeros(
            q_p.grad,
            query,
        ),
        _grad_or_zeros(
            q_d.grad,
            query,
        ),
    )
    _assert_close(
        _grad_or_zeros(
            k_p.grad,
            key,
        ),
        _grad_or_zeros(
            k_d.grad,
            key,
        ),
    )
    _assert_close(
        _grad_or_zeros(
            v_p.grad,
            value,
        ),
        _grad_or_zeros(
            v_d.grad,
            value,
        ),
    )


def test_public_module_chunk_size_matches_default_forward():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    full = PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        n,
        n,
        8,
        2,
        dropout=0.0,
        bias=True,
    )
    chunked = PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        n,
        n,
        8,
        2,
        dropout=0.0,
        bias=True,
        chunk_size=1,
    )
    chunked.load_state_dict(full.state_dict())
    x = torch.randn(
        3,
        n,
        8,
    )
    out_full, w_full = full(
        x,
        x,
        x,
        need_weights=True,
        average_attn_weights=False,
    )
    out_chunk, w_chunk = chunked(
        x,
        x,
        x,
        need_weights=True,
        average_attn_weights=False,
    )
    _assert_close(
        out_chunk,
        out_full,
    )
    _assert_close(
        w_chunk,
        w_full,
    )


def test_public_module_chunk_size_grads_match_default():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    full = PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        n,
        n,
        8,
        2,
        dropout=0.0,
    )
    chunked = PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        n,
        n,
        8,
        2,
        dropout=0.0,
        chunk_size=1,
    )
    chunked.load_state_dict(full.state_dict())
    x = torch.randn(
        3,
        n,
        8,
        requires_grad=True,
    )
    out_full, _ = full(
        x,
        x,
        x,
    )
    out_full.sum().backward()
    full_grad = x.grad.detach().clone()
    x.grad = None
    out_chunk, _ = chunked(
        x,
        x,
        x,
    )
    out_chunk.sum().backward()
    _assert_close(
        x.grad,
        full_grad,
    )
    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    ):
        _assert_close(
            getattr(chunked, name).weight.grad,
            getattr(full, name).weight.grad,
        )


def test_chunked_training_dropout_is_finite():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    layer = PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        n,
        n,
        8,
        2,
        dropout=0.5,
        chunk_size=1,
    )
    layer.train()
    x = torch.randn(
        2,
        n,
        8,
        requires_grad=True,
    )
    out, weights = layer(
        x,
        x,
        x,
        need_weights=True,
    )
    assert torch.isfinite(out).all()
    assert torch.isfinite(weights).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_chunked_kernel_dropout_matches_all_pairs_when_seeded():
    source_index, target_index, n = _cyclic_indices()
    query, key, value = _shaped_qkv(
        4,
        n,
        n,
        2,
        4,
    )
    torch.manual_seed(42)
    mix_a, attn_a = _packed_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.5,
        training=True,
    )
    torch.manual_seed(42)
    mix_b, attn_b = _packed_attention_chunked(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.5,
        training=True,
        chunk_size=1,
    )
    _assert_close(
        mix_b,
        mix_a,
    )
    _assert_close(
        attn_b,
        attn_a,
    )


def test_chunked_kernel_padding_grads_match_all_pairs():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    batch = 4
    query, key, value = _shaped_qkv(
        batch,
        n,
        n,
        2,
        4,
    )
    key_padding_mask = torch.zeros(
        batch,
        n,
        dtype=torch.bool,
    )
    key_padding_mask[0, 1] = True
    key_padding_mask[2, 0] = True
    participate = _padding_participate(
        key_padding_mask,
        source_index,
        (batch,),
        n,
        query.device,
    )
    q_a, k_a, v_a = _leaf_qkv(
        query,
        key,
        value,
    )
    mix_a, _ = _packed_attention(
        q_a,
        k_a,
        v_a,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
        participate=participate,
    )
    mix_a.sum().backward()
    q_b, k_b, v_b = _leaf_qkv(
        query,
        key,
        value,
    )
    mix_b, _ = _packed_attention_chunked(
        q_b,
        k_b,
        v_b,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
        chunk_size=1,
        participate=participate,
    )
    mix_b.sum().backward()
    _assert_close(
        _grad_or_zeros(
            q_b.grad,
            query,
        ),
        _grad_or_zeros(
            q_a.grad,
            query,
        ),
    )
    _assert_close(
        _grad_or_zeros(
            k_b.grad,
            key,
        ),
        _grad_or_zeros(
            k_a.grad,
            key,
        ),
    )
    _assert_close(
        _grad_or_zeros(
            v_b.grad,
            value,
        ),
        _grad_or_zeros(
            v_a.grad,
            value,
        ),
    )


def test_chunked_kernel_unbatched_matches_all_pairs():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    num_heads = 2
    head_dim = 4
    embed_dim = num_heads * head_dim
    query = shape_heads(
        torch.randn(
            n,
            embed_dim,
        ),
        num_heads,
    )
    key = shape_heads(
        torch.randn(
            n,
            embed_dim,
        ),
        num_heads,
    )
    value = shape_heads(
        torch.randn(
            n,
            embed_dim,
        ),
        num_heads,
    )
    mix, attn = _packed_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    chunk_mix, chunk_attn = _packed_attention_chunked(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
        chunk_size=1,
    )
    _assert_close(
        chunk_mix,
        mix,
    )
    _assert_close(
        chunk_attn,
        attn,
    )


def test_chunked_kernel_weight_only_grads_match_all_pairs():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    query, key, value = _shaped_qkv(
        4,
        n,
        n,
        2,
        4,
    )
    q_a, k_a, v_a = _leaf_qkv(
        query,
        key,
        value,
    )
    _, attn_a = _packed_attention(
        q_a,
        k_a,
        v_a,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    attn_a.sum().backward()
    q_b, k_b, v_b = _leaf_qkv(
        query,
        key,
        value,
    )
    _, attn_b = _packed_attention_chunked(
        q_b,
        k_b,
        v_b,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
        chunk_size=1,
    )
    attn_b.sum().backward()
    _assert_close(
        _grad_or_zeros(
            q_b.grad,
            query,
        ),
        _grad_or_zeros(
            q_a.grad,
            query,
        ),
    )
    _assert_close(
        _grad_or_zeros(
            k_b.grad,
            key,
        ),
        _grad_or_zeros(
            k_a.grad,
            key,
        ),
    )
    _assert_close(
        _grad_or_zeros(
            v_b.grad,
            value,
        ),
        _grad_or_zeros(
            v_a.grad,
            value,
        ),
    )


def test_chunk_size_is_not_in_state_dict():
    layer = _tiny_attn(chunk_size=2)
    assert "chunk_size" not in layer.state_dict()


def test_deepcopy_preserves_chunk_size():
    layer = _tiny_attn(chunk_size=2)
    copied = copy.deepcopy(layer)
    assert copied.chunk_size == 2
    torch.manual_seed(42)
    x = torch.randn(
        2,
        2,
        4,
    )
    out, _ = layer(
        x,
        x,
        x,
    )
    out_c, _ = copied(
        x,
        x,
        x,
    )
    _assert_close(
        out_c,
        out,
    )
