"""
Layout variants of PackedMultiheadAttention.

``add_self_loops=True`` ORs missing ``(i, i)`` pairs into the
module buffers. Rectangular cross-attention uses different
query and key sequence lengths. Isolated queries on the
query axis stay zeros after identity mix and ``out_proj``.
"""

import pytest
import torch

from kpnn2 import (
    Kpnn2Error,
    PackedMultiheadAttention,
    parse_adjacency,
)
from tests.helpers.packed_attention import (
    cyclic_edgelist,
    dense_masked_attention,
    pin_projections_identity,
    rectangular_indices,
    self_loop_edgelist,
    shape_heads,
)


def _layer_from_spec(
    spec,
    embed_dim=8,
    num_heads=2,
    bias=False,
    add_self_loops=False,
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
    )


def _attend(
    layer,
    query,
    key,
    value,
):
    output, _ = layer(
        query,
        key,
        value,
        need_weights=False,
    )
    return output


def _pair_set(
    source_index,
    target_index,
):
    return set(
        zip(
            source_index,
            target_index,
        )
    )


def _layer_pairs(layer):
    return _pair_set(
        layer.source_index.tolist(),
        layer.target_index.tolist(),
    )


def _oracle_from_layer(
    layer,
    query,
    key,
    value,
):
    q_h = shape_heads(
        layer.q_proj(query),
        layer.num_heads,
    )
    k_h = shape_heads(
        layer.k_proj(key),
        layer.num_heads,
    )
    v_h = shape_heads(
        layer.v_proj(value),
        layer.num_heads,
    )
    mixed = dense_masked_attention(
        q_h,
        k_h,
        v_h,
        layer.source_index,
        layer.target_index,
    )
    return layer.out_proj(
        mixed.reshape(
            *mixed.shape[:-2],
            layer.embed_dim,
        )
    )


def test_existing_self_loop_is_a_live_pair():
    torch.manual_seed(42)
    spec = parse_adjacency(self_loop_edgelist())
    n = len(spec.nodes)
    embed_dim = 8
    h_idx = spec.nodes.index("h")
    caller_pairs = _pair_set(
        spec.source_index,
        spec.target_index,
    )
    assert (h_idx, h_idx) in caller_pairs
    layer = _layer_from_spec(
        spec,
        embed_dim=embed_dim,
        num_heads=2,
        bias=False,
        add_self_loops=False,
    )
    pin_projections_identity(layer)
    assert (h_idx, h_idx) in _layer_pairs(layer)
    x = torch.randn(
        2,
        n,
        embed_dim,
    )
    base = _attend(
        layer,
        x,
        x,
        x,
    )
    key_pert = x.clone()
    key_pert[:, h_idx] = key_pert[:, h_idx] + 10.0
    out_pert = _attend(
        layer,
        x,
        key_pert,
        key_pert,
    )
    assert not torch.allclose(
        out_pert[:, h_idx],
        base[:, h_idx],
    )
    assert torch.isfinite(out_pert).all()
    assert torch.isfinite(base).all()


def test_add_self_loops_ors_missing_diagonals_for_isolated_queries():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    embed_dim = 8
    assert torch.equal(
        spec.to_mask().diag(),
        torch.zeros(n),
    )
    assert spec.input_index
    plain = _layer_from_spec(
        spec,
        embed_dim=embed_dim,
        num_heads=2,
        bias=False,
        add_self_loops=False,
    )
    looped = _layer_from_spec(
        spec,
        embed_dim=embed_dim,
        num_heads=2,
        bias=False,
        add_self_loops=True,
    )
    pin_projections_identity(plain)
    pin_projections_identity(looped)
    plain_targets = set(plain.target_index.tolist())
    looped_pairs = _layer_pairs(looped)
    for index in spec.input_index:
        assert index not in plain_targets
        assert (
            index,
            index,
        ) in looped_pairs
    x = torch.randn(
        2,
        n,
        embed_dim,
    )
    out_plain = _attend(
        plain,
        x,
        x,
        x,
    )
    out_looped = _attend(
        looped,
        x,
        x,
        x,
    )
    x_pert = x.clone()
    for index in spec.input_index:
        x_pert[:, index] = x_pert[:, index] + 10.0
    out_pert = _attend(
        looped,
        x_pert,
        x_pert,
        x_pert,
    )
    for index in spec.input_index:
        slice_plain = out_plain[
            :,
            index,
            :,
        ]
        torch.testing.assert_close(
            slice_plain,
            torch.zeros_like(slice_plain),
        )
        assert not torch.allclose(
            out_pert[:, index],
            out_looped[:, index],
        )


def test_oracle_uses_post_init_buffers_with_self_loops():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    embed_dim = 8
    num_heads = 2
    layer = _layer_from_spec(
        spec,
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=False,
        add_self_loops=True,
    )
    pin_projections_identity(layer)
    assert layer.nnz > len(spec.source_index)
    caller_pairs = _pair_set(
        spec.source_index,
        spec.target_index,
    )
    assert _layer_pairs(layer) != caller_pairs
    batch = 4
    query = torch.randn(
        batch,
        n,
        embed_dim,
    )
    key = torch.randn(
        batch,
        n,
        embed_dim,
    )
    value = torch.randn(
        batch,
        n,
        embed_dim,
    )
    expected = _oracle_from_layer(
        layer,
        query,
        key,
        value,
    )
    got = _attend(
        layer,
        query,
        key,
        value,
    )
    torch.testing.assert_close(
        got,
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


def test_add_self_loops_does_not_duplicate_existing_diagonal():
    spec = parse_adjacency(self_loop_edgelist())
    n = len(spec.nodes)
    existing = _pair_set(
        spec.source_index,
        spec.target_index,
    )
    h_idx = spec.nodes.index("h")
    assert (h_idx, h_idx) in existing
    missing = [node for node in range(n) if (node, node) not in existing]
    assert missing
    assert len(missing) < n
    original_nnz = len(spec.source_index)
    layer = _layer_from_spec(
        spec,
        add_self_loops=True,
    )
    assert layer.nnz == original_nnz + len(missing)
    assert layer.nnz != original_nnz + n
    listed = list(
        zip(
            layer.source_index.tolist(),
            layer.target_index.tolist(),
        )
    )
    assert len(listed) == len(set(listed))
    for node in range(n):
        assert listed.count((node, node)) == 1


def test_add_self_loops_leaves_spec_indices_unchanged():
    spec = parse_adjacency(cyclic_edgelist())
    source = spec.source_index
    target = spec.target_index
    _layer_from_spec(
        spec,
        add_self_loops=True,
    )
    assert spec.source_index == source
    assert spec.target_index == target


def test_rectangular_forbids_add_self_loops():
    (
        source_index,
        target_index,
        query_features,
        key_features,
    ) = rectangular_indices()
    assert query_features != key_features
    with pytest.raises(
        Kpnn2Error,
        match="query_features == key_features",
    ):
        PackedMultiheadAttention(
            source_index,
            target_index,
            query_features,
            key_features,
            8,
            2,
            add_self_loops=True,
        )


def test_rectangular_module_matches_oracle():
    torch.manual_seed(42)
    (
        source_index,
        target_index,
        query_features,
        key_features,
    ) = rectangular_indices()
    embed_dim = 8
    num_heads = 2
    layer = PackedMultiheadAttention(
        source_index,
        target_index,
        query_features,
        key_features,
        embed_dim,
        num_heads,
        bias=False,
    )
    pin_projections_identity(layer)
    batch = 4
    query = torch.randn(
        batch,
        query_features,
        embed_dim,
    )
    key = torch.randn(
        batch,
        key_features,
        embed_dim,
    )
    value = torch.randn(
        batch,
        key_features,
        embed_dim,
    )
    expected = _oracle_from_layer(
        layer,
        query,
        key,
        value,
    )
    got = _attend(
        layer,
        query,
        key,
        value,
    )
    torch.testing.assert_close(
        got,
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


def test_rectangular_query_neighborhoods():
    torch.manual_seed(42)
    (
        source_index,
        target_index,
        query_features,
        key_features,
    ) = rectangular_indices()
    assert source_index == [0, 1, 4]
    assert target_index == [0, 0, 2]
    assert 1 not in target_index
    embed_dim = 8
    layer = PackedMultiheadAttention(
        source_index,
        target_index,
        query_features,
        key_features,
        embed_dim,
        2,
        bias=False,
    )
    pin_projections_identity(layer)
    query = torch.randn(
        2,
        query_features,
        embed_dim,
    )
    kv = torch.randn(
        2,
        key_features,
        embed_dim,
    )
    base = _attend(
        layer,
        query,
        kv,
        kv,
    )
    isolated = base[
        :,
        1,
        :,
    ]
    torch.testing.assert_close(
        isolated,
        torch.zeros_like(isolated),
    )
    kv0 = kv.clone()
    kv0[:, 0] = kv0[:, 0] + 10.0
    out0 = _attend(
        layer,
        query,
        kv0,
        kv0,
    )
    assert not torch.allclose(
        out0[:, 0],
        base[:, 0],
    )
    kv1 = kv.clone()
    kv1[:, 1] = kv1[:, 1] + 10.0
    out1 = _attend(
        layer,
        query,
        kv1,
        kv1,
    )
    assert not torch.allclose(
        out1[:, 0],
        base[:, 0],
    )
    kv4 = kv.clone()
    kv4[:, 4] = kv4[:, 4] + 10.0
    out4 = _attend(
        layer,
        query,
        kv4,
        kv4,
    )
    torch.testing.assert_close(
        out4[:, 0],
        base[:, 0],
    )
    assert not torch.allclose(
        out4[:, 2],
        base[:, 2],
    )


def test_rectangular_rejects_query_on_key_axis():
    torch.manual_seed(42)
    (
        source_index,
        target_index,
        query_features,
        key_features,
    ) = rectangular_indices()
    assert query_features != key_features
    embed_dim = 8
    layer = PackedMultiheadAttention(
        source_index,
        target_index,
        query_features,
        key_features,
        embed_dim,
        2,
        bias=False,
    )
    query = torch.randn(
        2,
        key_features,
        embed_dim,
    )
    key = torch.randn(
        2,
        key_features,
        embed_dim,
    )
    value = torch.randn(
        2,
        key_features,
        embed_dim,
    )
    with pytest.raises(
        Kpnn2Error,
        match="query_features",
    ):
        layer(
            query,
            key,
            value,
            need_weights=False,
        )
