import torch

from kpnn2 import parse_adjacency
from kpnn2._packed_multihead_attention import (
    _packed_attention,
)
from tests.helpers.packed_attention import (
    allow_matrix,
    cyclic_edgelist,
    dense_masked_attention,
    rectangular_indices,
    shape_heads,
    three_cycle_indices,
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


def _not_close(
    actual,
    expected,
):
    return not torch.allclose(
        actual,
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


def test_packed_mix_matches_dense_on_cyclic_adjacency():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    batch = 4
    num_heads = 2
    head_dim = 4
    embed_dim = num_heads * head_dim
    query = shape_heads(
        torch.randn(batch, n, embed_dim),
        num_heads,
    )
    key = shape_heads(
        torch.randn(batch, n, embed_dim),
        num_heads,
    )
    value = shape_heads(
        torch.randn(batch, n, embed_dim),
        num_heads,
    )
    packed = _packed_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    dense = dense_masked_attention(
        query,
        key,
        value,
        source_index,
        target_index,
    )
    _assert_close(
        packed,
        dense,
    )


def test_packed_mix_matches_dense_on_three_cycle():
    torch.manual_seed(42)
    source_list, target_list, n = three_cycle_indices()
    source_index = _index_tensor(source_list)
    target_index = _index_tensor(target_list)
    allow = allow_matrix(
        source_index,
        target_index,
        n,
        n,
    )
    assert torch.all(allow.sum(dim=1) > 0)
    batch = 4
    num_heads = 2
    head_dim = 4
    query = torch.randn(batch, n, num_heads, head_dim)
    key = torch.randn(batch, n, num_heads, head_dim)
    value = torch.randn(batch, n, num_heads, head_dim)
    packed = _packed_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    dense = dense_masked_attention(
        query,
        key,
        value,
        source_index,
        target_index,
    )
    _assert_close(
        packed,
        dense,
    )


def test_live_packed_weights_sum_to_one_through_the_mix():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    num_heads = 2
    head_dim = 4
    allow = allow_matrix(
        source_index,
        target_index,
        n,
        n,
    )
    degree = allow.sum(dim=1)

    query = torch.randn(2, n, num_heads, head_dim)
    key = torch.randn(2, n, num_heads, head_dim)
    value = torch.randn(2, n, num_heads, head_dim)
    packed = _packed_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    dense = dense_masked_attention(
        query,
        key,
        value,
        source_index,
        target_index,
    )
    _assert_close(
        packed,
        dense,
    )
    deg1 = (degree == 1).nonzero(as_tuple=False).view(-1)
    assert deg1.numel() > 0
    q_one = int(deg1[0])
    k_one = int(allow[q_one].nonzero(as_tuple=False).view(-1)[0])
    _assert_close(
        packed[:, q_one],
        value[:, k_one],
    )

    multi = (degree >= 2).nonzero(as_tuple=False).view(-1)
    assert multi.numel() > 0
    q_multi = int(multi[0])
    live_keys = allow[q_multi].nonzero(as_tuple=False).view(-1)
    n_live = int(live_keys.numel())
    assert n_live <= head_dim
    query_oh = torch.randn(1, n, num_heads, head_dim)
    key_oh = torch.randn(1, n, num_heads, head_dim)
    value_oh = torch.zeros(1, n, num_heads, head_dim)
    for i, key_pos in enumerate(live_keys.tolist()):
        value_oh[0, key_pos, :, i] = 1.0
    packed_oh = _packed_attention(
        query_oh,
        key_oh,
        value_oh,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    dense_oh = dense_masked_attention(
        query_oh,
        key_oh,
        value_oh,
        source_index,
        target_index,
    )
    _assert_close(
        packed_oh,
        dense_oh,
    )
    mix_q = packed_oh[0, q_multi]
    _assert_close(
        mix_q.sum(dim=-1),
        torch.ones(num_heads),
    )
    if n_live < head_dim:
        _assert_close(
            mix_q[:, n_live:],
            torch.zeros_like(mix_q[:, n_live:]),
        )


def test_scale_is_one_over_sqrt_head_dim():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    batch = 4
    embed_dim = 8
    q_flat = torch.randn(batch, n, embed_dim)
    k_flat = torch.randn(batch, n, embed_dim)
    v_flat = torch.randn(batch, n, embed_dim)
    mixes = []
    for num_heads in (1, 2):
        query = shape_heads(
            q_flat,
            num_heads,
        )
        key = shape_heads(
            k_flat,
            num_heads,
        )
        value = shape_heads(
            v_flat,
            num_heads,
        )
        packed = _packed_attention(
            query,
            key,
            value,
            source_index,
            target_index,
            dropout_p=0.0,
            training=False,
        )
        dense = dense_masked_attention(
            query,
            key,
            value,
            source_index,
            target_index,
        )
        _assert_close(
            packed,
            dense,
        )
        mixes.append(
            packed.reshape(batch, n, embed_dim),
        )
    assert _not_close(
        mixes[0],
        mixes[1],
    )


def test_heads_do_not_mix_inside_the_kernel():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    batch = 3
    head_dim = 4
    query = torch.stack(
        [
            torch.randn(batch, n, head_dim),
            torch.randn(batch, n, head_dim),
        ],
        dim=-2,
    )
    key = torch.stack(
        [
            torch.randn(batch, n, head_dim),
            torch.randn(batch, n, head_dim),
        ],
        dim=-2,
    )
    value = torch.stack(
        [
            torch.randn(batch, n, head_dim),
            torch.randn(batch, n, head_dim),
        ],
        dim=-2,
    )
    packed = _packed_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    dense = dense_masked_attention(
        query,
        key,
        value,
        source_index,
        target_index,
    )
    _assert_close(
        packed,
        dense,
    )
    value_perturbed = value.clone()
    value_perturbed[..., 1, :] = value_perturbed[..., 1, :] + 10.0
    packed_p = _packed_attention(
        query,
        key,
        value_perturbed,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    dense_p = dense_masked_attention(
        query,
        key,
        value_perturbed,
        source_index,
        target_index,
    )
    _assert_close(
        packed_p,
        dense_p,
    )
    _assert_close(
        packed[..., 0, :],
        packed_p[..., 0, :],
    )
    _assert_close(
        dense[..., 0, :],
        dense_p[..., 0, :],
    )
    assert _not_close(
        packed[..., 1, :],
        packed_p[..., 1, :],
    )
    assert _not_close(
        dense[..., 1, :],
        dense_p[..., 1, :],
    )


def test_dead_pair_has_zero_influence_in_the_kernel():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    num_heads = 2
    head_dim = 4
    allow = allow_matrix(
        source_index,
        target_index,
        n,
        n,
    )
    query = torch.randn(2, n, num_heads, head_dim)
    key = torch.randn(2, n, num_heads, head_dim)
    value = torch.randn(2, n, num_heads, head_dim)
    packed = _packed_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    dense = dense_masked_attention(
        query,
        key,
        value,
        source_index,
        target_index,
    )
    _assert_close(
        packed,
        dense,
    )
    has_live = allow.sum(dim=1) > 0
    dead = (allow == 0) & has_live.unsqueeze(1)
    dead_pairs = dead.nonzero(as_tuple=False)
    assert dead_pairs.shape[0] > 0
    q_dead = int(dead_pairs[0, 0])
    k_dead = int(dead_pairs[0, 1])
    key_dead = key.clone()
    key_dead[:, k_dead] = key_dead[:, k_dead] + 50.0
    packed_dead = _packed_attention(
        query,
        key_dead,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    _assert_close(
        packed_dead[:, q_dead],
        packed[:, q_dead],
    )
    degree = allow.sum(dim=1)
    live_multi = (allow > 0) & (degree >= 2).unsqueeze(1)
    live_pairs = live_multi.nonzero(as_tuple=False)
    assert live_pairs.shape[0] > 0
    q_live = int(live_pairs[0, 0])
    k_live = int(live_pairs[0, 1])
    key_live = key.clone()
    key_live[:, k_live] = key_live[:, k_live] + 50.0
    packed_live = _packed_attention(
        query,
        key_live,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    assert _not_close(
        packed_live[:, q_live],
        packed[:, q_live],
    )


def test_eval_dropout_does_not_change_the_mix():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    batch = 4
    num_heads = 2
    head_dim = 4
    query = torch.randn(batch, n, num_heads, head_dim)
    key = torch.randn(batch, n, num_heads, head_dim)
    value = torch.randn(batch, n, num_heads, head_dim)
    packed_off = _packed_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    packed_eval = _packed_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.5,
        training=False,
    )
    _assert_close(
        packed_off,
        packed_eval,
    )


def test_rectangular_indices_match_dense_oracle():
    torch.manual_seed(42)
    (
        source_list,
        target_list,
        query_features,
        key_features,
    ) = rectangular_indices()
    source_index = _index_tensor(source_list)
    target_index = _index_tensor(target_list)
    assert query_features != key_features
    batch = 4
    num_heads = 2
    head_dim = 4
    query = torch.randn(
        batch,
        query_features,
        num_heads,
        head_dim,
    )
    key = torch.randn(
        batch,
        key_features,
        num_heads,
        head_dim,
    )
    value = torch.randn(
        batch,
        key_features,
        num_heads,
        head_dim,
    )
    packed = _packed_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
    )
    dense = dense_masked_attention(
        query,
        key,
        value,
        source_index,
        target_index,
    )
    _assert_close(
        packed,
        dense,
    )
