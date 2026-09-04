"""
Graph structure of PackedMultiheadAttention.

Attention is a contraction on the edgelist: direction is
source → target (query = target, key/value = source), isolated
queries stay zeros after the mix then go through out_proj, and
cycles are ordinary live pairs.
"""

import torch

from kpnn2 import (
    PackedMultiheadAttention,
    parse_adjacency,
)
from tests.helpers.packed_attention import (
    allow_matrix,
    cyclic_edgelist,
    dense_masked_attention,
    pin_projections_identity,
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


def _dead_and_live_pair(spec):
    n = len(spec.nodes)
    allow = allow_matrix(
        spec.source_index,
        spec.target_index,
        n,
        n,
    )
    assert torch.equal(
        allow,
        spec.to_mask(),
    )
    has_live = allow.sum(dim=-1) > 0
    dead = (allow == 0) & has_live.unsqueeze(-1)
    dead_pairs = dead.nonzero(as_tuple=False)
    assert dead_pairs.shape[0] > 0
    query_pos = int(dead_pairs[0, 0])
    dead_key = int(dead_pairs[0, 1])
    live_keys = allow[query_pos].nonzero(as_tuple=False).view(-1)
    assert live_keys.numel() > 0
    live_key = int(live_keys[0])
    return query_pos, dead_key, live_key


def test_direction_target_attends_to_source():
    torch.manual_seed(42)
    embed_dim = 8
    num_heads = 2
    layer = PackedMultiheadAttention(
        [0],
        [1],
        2,
        2,
        embed_dim,
        num_heads,
        bias=False,
    )
    pin_projections_identity(layer)
    x = torch.randn(
        1,
        2,
        embed_dim,
    )
    base = _attend(
        layer,
        x,
        x,
        x,
    )
    torch.testing.assert_close(
        base[:, 0],
        torch.zeros_like(base[:, 0]),
    )

    x_source = x.clone()
    x_source[:, 0] = x_source[:, 0] + 10.0
    out_source = _attend(
        layer,
        x_source,
        x_source,
        x_source,
    )
    torch.testing.assert_close(
        out_source[:, 0],
        torch.zeros_like(out_source[:, 0]),
    )
    assert not torch.allclose(
        out_source[:, 1],
        base[:, 1],
    )

    x_query = x.clone()
    x_query[:, 1] = x_query[:, 1] + 10.0
    out_query = _attend(
        layer,
        x_query,
        x_query,
        x_query,
    )
    torch.testing.assert_close(
        out_query[:, 0],
        torch.zeros_like(out_query[:, 0]),
    )
    torch.testing.assert_close(
        out_query[:, 1],
        base[:, 1],
    )


def test_absent_key_does_not_influence_query():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    embed_dim = 8
    layer = _layer_from_spec(
        spec,
        embed_dim=embed_dim,
        num_heads=2,
        bias=False,
    )
    pin_projections_identity(layer)
    query_pos, dead_key, _live_key = _dead_and_live_pair(spec)
    x = torch.randn(
        1,
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
    key_pert[:, dead_key] = key_pert[:, dead_key] + 10.0
    out_pert = _attend(
        layer,
        x,
        key_pert,
        key_pert,
    )
    torch.testing.assert_close(
        out_pert[:, query_pos],
        base[:, query_pos],
    )


def test_live_key_influences_query():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    embed_dim = 8
    layer = _layer_from_spec(
        spec,
        embed_dim=embed_dim,
        num_heads=2,
        bias=False,
    )
    pin_projections_identity(layer)
    query_pos, _dead_key, live_key = _dead_and_live_pair(spec)
    x = torch.randn(
        1,
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
    bump = torch.randn_like(
        key_pert[:, live_key],
    )
    key_pert[:, live_key] = key_pert[:, live_key] + bump
    out_pert = _attend(
        layer,
        x,
        key_pert,
        key_pert,
    )
    if torch.allclose(
        out_pert[:, query_pos],
        base[:, query_pos],
    ):
        torch.manual_seed(42)
        x = torch.randn(
            1,
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
        key_pert[:, live_key] = key_pert[:, live_key] + 10.0
        out_pert = _attend(
            layer,
            x,
            key_pert,
            key_pert,
        )
    assert not torch.allclose(
        out_pert[:, query_pos],
        base[:, query_pos],
    )


def test_isolated_query_is_zeros_not_nan():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    embed_dim = 8
    layer = _layer_from_spec(
        spec,
        embed_dim=embed_dim,
        num_heads=2,
        bias=False,
        add_self_loops=False,
    )
    pin_projections_identity(layer)
    torch.testing.assert_close(
        layer.out_proj.weight,
        torch.eye(
            embed_dim,
            dtype=layer.out_proj.weight.dtype,
        ),
    )
    assert layer.out_proj.bias is None
    x = torch.randn(
        2,
        n,
        embed_dim,
    )
    out = _attend(
        layer,
        x,
        x,
        x,
    )
    assert spec.input_index
    for index in spec.input_index:
        slice_ = out[
            :,
            index,
            :,
        ]
        torch.testing.assert_close(
            slice_,
            torch.zeros_like(slice_),
        )
        assert torch.isfinite(slice_).all()
        assert not torch.isnan(slice_).any()
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


def test_isolated_query_still_goes_through_out_proj():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    embed_dim = 8
    layer = _layer_from_spec(
        spec,
        embed_dim=embed_dim,
        num_heads=2,
        bias=True,
        add_self_loops=False,
    )
    with torch.no_grad():
        layer.out_proj.bias.copy_(
            torch.arange(
                embed_dim,
                dtype=layer.out_proj.bias.dtype,
            )
            + 1.0
        )
    x = torch.randn(
        2,
        n,
        embed_dim,
    )
    out = _attend(
        layer,
        x,
        x,
        x,
    )
    expected = layer.out_proj.bias.view(1, -1).expand(
        out.shape[0],
        embed_dim,
    )
    assert spec.input_index
    for index in spec.input_index:
        torch.testing.assert_close(
            out[:, index, :],
            expected,
        )


def test_cycle_pairs_are_live():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    embed_dim = 8
    a_idx = spec.nodes.index("a")
    b_idx = spec.nodes.index("b")
    pairs = set(
        zip(
            spec.source_index,
            spec.target_index,
        )
    )
    assert (a_idx, b_idx) in pairs
    assert (b_idx, a_idx) in pairs
    layer = _layer_from_spec(
        spec,
        embed_dim=embed_dim,
        num_heads=2,
        bias=False,
    )
    pin_projections_identity(layer)
    x = torch.randn(
        1,
        n,
        embed_dim,
    )
    base = _attend(
        layer,
        x,
        x,
        x,
    )
    x_a = x.clone()
    x_a[:, a_idx] = x_a[:, a_idx] + 10.0
    out_a = _attend(
        layer,
        x_a,
        x_a,
        x_a,
    )
    assert not torch.allclose(
        out_a[:, b_idx],
        base[:, b_idx],
    )
    x_b = x.clone()
    x_b[:, b_idx] = x_b[:, b_idx] + 10.0
    out_b = _attend(
        layer,
        x_b,
        x_b,
        x_b,
    )
    assert not torch.allclose(
        out_b[:, a_idx],
        base[:, a_idx],
    )


def test_no_n_by_n_score_parameter():
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    embed_dim = 8
    layer = _layer_from_spec(
        spec,
        embed_dim=embed_dim,
        num_heads=2,
    )
    for _, param in layer.named_parameters():
        assert list(param.shape) != [n, n]
    assert layer.nnz == len(spec.source_index)
    assert layer.nnz < n * n
    assert layer.q_proj.weight.shape == (embed_dim, embed_dim)


def test_module_mix_matches_oracle_after_projections():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    embed_dim = 8
    num_heads = 2
    layer = _layer_from_spec(
        spec,
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
    )
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
    q_h = shape_heads(
        layer.q_proj(query),
        num_heads,
    )
    k_h = shape_heads(
        layer.k_proj(key),
        num_heads,
    )
    v_h = shape_heads(
        layer.v_proj(value),
        num_heads,
    )
    mixed = dense_masked_attention(
        q_h,
        k_h,
        v_h,
        spec.source_index,
        spec.target_index,
    )
    expected = layer.out_proj(
        mixed.reshape(
            *mixed.shape[:-2],
            embed_dim,
        )
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
