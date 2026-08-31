import pandas as pd
import pytest
import torch

from kpnn2 import (
    Kpnn2Error,
    parse_adjacency,
)
from prototypes.attention import PackedMultiheadAttention
from prototypes.attention.packed_multihead_attention import (
    _packed_attention,
)


def _cyclic_edgelist():
    return pd.DataFrame(
        {
            "source": ["x", "a", "b", "a"],
            "target": ["a", "b", "a", "y"],
        }
    )


def _pathway_edgelist():
    return pd.DataFrame(
        {
            "source": [
                "sensor_1",
                "pathway_a",
                "pathway_b",
                "pathway_a",
            ],
            "target": [
                "pathway_a",
                "pathway_b",
                "pathway_a",
                "output_1",
            ],
        }
    )


def _self_loop_edgelist():
    return pd.DataFrame(
        {
            "source": ["in", "h", "h"],
            "target": ["h", "h", "out"],
        }
    )


def _attn_from_spec(
    spec,
    embed_dim=8,
    num_heads=2,
    add_self_loops=False,
    bias=True,
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


def _dense_masked_attention(
    query,
    key,
    value,
    allow,
    num_heads,
):
    batch, n_query, embed_dim = query.shape
    n_key = key.shape[1]
    head_dim = embed_dim // num_heads
    scale = head_dim**-0.5
    q = query.view(
        batch,
        n_query,
        num_heads,
        head_dim,
    )
    k = key.view(
        batch,
        n_key,
        num_heads,
        head_dim,
    )
    v = value.view(
        batch,
        n_key,
        num_heads,
        head_dim,
    )
    scores = (
        torch.einsum(
            "bqhd,bkhd->bhqk",
            q,
            k,
        )
        * scale
    )
    additive = torch.zeros(
        n_query,
        n_key,
        dtype=scores.dtype,
        device=scores.device,
    )
    additive = additive.masked_fill(
        allow <= 0,
        torch.finfo(scores.dtype).min,
    )
    has_key = allow.sum(dim=-1) > 0
    attn = torch.softmax(
        scores + additive,
        dim=-1,
    )
    attn = torch.nan_to_num(
        attn,
        nan=0.0,
    )
    attn = attn * has_key.view(1, 1, n_query, 1)
    mixed = torch.einsum(
        "bhqk,bkhd->bqhd",
        attn,
        v,
    )
    return mixed.reshape(
        batch,
        n_query,
        embed_dim,
    )


def test_parse_adjacency_packed_pairs_match_to_mask():
    spec = parse_adjacency(_cyclic_edgelist())
    mask = spec.to_mask()
    n = len(spec.nodes)
    assert mask.shape == (n, n)
    assert int(mask.sum().item()) == len(spec.source_index)
    reconstructed = torch.zeros_like(mask)
    for source, target in zip(
        spec.source_index,
        spec.target_index,
        strict=True,
    ):
        reconstructed[target, source] = 1.0
    assert torch.equal(
        reconstructed,
        mask,
    )


def test_packed_mha_builds_from_parse_adjacency():
    spec = parse_adjacency(_cyclic_edgelist())
    layer = _attn_from_spec(spec)
    assert layer.query_features == len(spec.nodes)
    assert layer.key_features == len(spec.nodes)
    assert layer.nnz == len(spec.source_index)
    assert list(layer.source_index.tolist()) == list(spec.source_index)
    assert list(layer.target_index.tolist()) == list(spec.target_index)


def test_packed_mha_matches_dense_masked_scores():
    torch.manual_seed(42)
    spec = parse_adjacency(_cyclic_edgelist())
    layer = _attn_from_spec(
        spec,
        embed_dim=8,
        num_heads=2,
        bias=False,
    )
    n = len(spec.nodes)
    x = torch.randn(
        4,
        n,
        8,
    )
    q = layer.q_proj(x)
    k = layer.k_proj(x)
    v = layer.v_proj(x)
    packed_mix = _packed_attention(
        layer._shape_heads(q),
        layer._shape_heads(k),
        layer._shape_heads(v),
        layer.source_index,
        layer.target_index,
        0.0,
        False,
    ).reshape(
        4,
        n,
        8,
    )
    dense_mix = _dense_masked_attention(
        q,
        k,
        v,
        spec.to_mask(),
        2,
    )
    torch.testing.assert_close(
        packed_mix,
        dense_mix,
        atol=1e-5,
        rtol=1e-5,
    )


def test_packed_mha_output_shape_and_self_attention_default():
    torch.manual_seed(42)
    spec = parse_adjacency(_pathway_edgelist())
    layer = _attn_from_spec(spec)
    n = len(spec.nodes)
    x = torch.randn(
        3,
        n,
        8,
    )
    y = layer(x)
    assert y.shape == (3, n, 8)
    y2 = layer(
        x,
        x,
        x,
    )
    torch.testing.assert_close(
        y,
        y2,
    )


def test_packed_mha_cycle_and_self_loop_are_legal_pairs():
    spec = parse_adjacency(_self_loop_edgelist())
    assert "h" in spec.hidden_nodes
    pairs = set(
        zip(
            spec.source_index,
            spec.target_index,
            strict=True,
        )
    )
    h = spec.nodes.index("h")
    assert (h, h) in pairs
    layer = _attn_from_spec(spec)
    x = torch.randn(
        2,
        len(spec.nodes),
        8,
    )
    y = layer(x)
    assert torch.isfinite(y).all()


def test_packed_mha_isolated_query_is_zero_without_self_loops():
    torch.manual_seed(42)
    spec = parse_adjacency(_cyclic_edgelist())
    layer = _attn_from_spec(
        spec,
        bias=False,
        add_self_loops=False,
    )
    n = len(spec.nodes)
    x = torch.randn(
        2,
        n,
        8,
    )
    y = layer(x)
    input_pos = spec.input_index[0]
    degrees = spec.to_mask().sum(dim=-1)
    assert degrees[input_pos].item() == 0.0
    assert torch.allclose(
        y[:, input_pos, :],
        torch.zeros_like(y[:, input_pos, :]),
        atol=1e-6,
    )
    assert not torch.isnan(y).any()


def test_packed_mha_add_self_loops_does_not_change_spec():
    spec = parse_adjacency(_cyclic_edgelist())
    n = len(spec.nodes)
    n_edges = len(spec.source_index)
    layer = _attn_from_spec(
        spec,
        add_self_loops=True,
    )
    assert layer.nnz == n_edges + n
    assert len(spec.source_index) == n_edges
    assert spec.to_mask().diag().sum().item() == 0.0


def test_packed_mha_forbidden_key_does_not_affect_query():
    torch.manual_seed(42)
    spec = parse_adjacency(_cyclic_edgelist())
    layer = _attn_from_spec(
        spec,
        bias=False,
    )
    n = len(spec.nodes)
    x = torch.randn(
        1,
        n,
        8,
    )
    mask = spec.to_mask()
    forbidden = (mask == 0).nonzero(as_tuple=False)
    query_pos = int(forbidden[0, 0].item())
    key_pos = int(forbidden[0, 1].item())
    x_pert = x.clone()
    x_pert[0, key_pos] = x_pert[0, key_pos] + 10.0
    q0 = layer.q_proj(x)
    k0 = layer.k_proj(x)
    v0 = layer.v_proj(x)
    k1 = layer.k_proj(x_pert)
    v1 = layer.v_proj(x_pert)
    mix0 = _packed_attention(
        layer._shape_heads(q0),
        layer._shape_heads(k0),
        layer._shape_heads(v0),
        layer.source_index,
        layer.target_index,
        0.0,
        False,
    )
    mix1 = _packed_attention(
        layer._shape_heads(q0),
        layer._shape_heads(k1),
        layer._shape_heads(v1),
        layer.source_index,
        layer.target_index,
        0.0,
        False,
    )
    torch.testing.assert_close(
        mix0[:, query_pos],
        mix1[:, query_pos],
        atol=1e-5,
        rtol=1e-5,
    )


def test_packed_mha_rejects_duplicate_pairs():
    with pytest.raises(Kpnn2Error, match="duplicate"):
        PackedMultiheadAttention(
            [0, 0],
            [1, 1],
            2,
            2,
            8,
            2,
        )


def test_packed_mha_rejects_undivisible_embed_dim():
    with pytest.raises(Kpnn2Error, match="divisible"):
        PackedMultiheadAttention(
            [0],
            [1],
            2,
            2,
            7,
            2,
        )


def test_packed_mha_no_n_by_n_score_parameter():
    spec = parse_adjacency(_cyclic_edgelist())
    n = len(spec.nodes)
    layer = _attn_from_spec(spec)
    for _, param in layer.named_parameters():
        assert list(param.shape) != [n, n]
    assert layer.nnz == len(spec.source_index)
    assert layer.nnz < n * n
    assert layer.q_proj.weight.shape == (8, 8)
