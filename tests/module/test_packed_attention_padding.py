"""
key_padding_mask is a data-axis ignore in packed space.

True means ignore that key. Applied on live packed pairs, not
as a second graph mask and not as a dense (L, S) additive
mask. After padding, a query with no remaining keys stays
zeros, not NaN.
"""

import pytest
import torch

from kpnn2 import (
    Kpnn2Error,
    PackedMultiheadAttention,
    parse_adjacency,
)
from tests.helpers.packed_attention import (
    allow_matrix,
    cyclic_edgelist,
    dense_masked_attention,
    pin_projections_identity,
    rectangular_indices,
    shape_heads,
    three_cycle_indices,
)


def _identity_layer(
    source_index,
    target_index,
    query_features,
    key_features,
    embed_dim=8,
    num_heads=2,
):
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
    return layer


def _attend(
    layer,
    query,
    key,
    value,
    key_padding_mask=None,
):
    output, _ = layer(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        need_weights=False,
    )
    return output


def _oracle(
    layer,
    query,
    key,
    value,
    key_padding_mask=None,
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
        key_padding_mask=key_padding_mask,
    )
    return layer.out_proj(
        mixed.reshape(
            *mixed.shape[:-2],
            layer.embed_dim,
        )
    )


def test_padding_live_key_leaves_the_softmax():
    torch.manual_seed(42)
    source_index, target_index, n = three_cycle_indices()
    query_pos = 0
    live_key = 2
    live_sources = [
        source
        for source, target in zip(
            source_index,
            target_index,
        )
        if target == query_pos
    ]
    assert live_sources == [live_key]
    embed_dim = 8
    layer = _identity_layer(
        source_index,
        target_index,
        n,
        n,
        embed_dim=embed_dim,
    )
    batch = 2
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
    base = _attend(
        layer,
        query,
        key,
        value,
    )
    torch.testing.assert_close(
        base[:, query_pos],
        value[:, live_key],
    )
    mask = torch.zeros(
        batch,
        n,
        dtype=torch.bool,
    )
    mask[:, live_key] = True
    padded = _attend(
        layer,
        query,
        key,
        value,
        key_padding_mask=mask,
    )
    zeros = torch.zeros_like(padded[:, query_pos])
    torch.testing.assert_close(
        padded[:, query_pos],
        zeros,
    )


def test_padding_dead_key_is_noop():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    allow = allow_matrix(
        spec.source_index,
        spec.target_index,
        n,
        n,
    )
    has_live = allow.sum(dim=-1) > 0
    dead = (allow == 0) & has_live.unsqueeze(-1)
    dead_pairs = dead.nonzero(as_tuple=False)
    assert dead_pairs.shape[0] > 0
    query_pos = int(dead_pairs[0, 0])
    dead_key = int(dead_pairs[0, 1])
    assert allow[query_pos, dead_key].item() == 0.0
    assert has_live[query_pos]
    embed_dim = 8
    layer = _identity_layer(
        spec.source_index,
        spec.target_index,
        n,
        n,
        embed_dim=embed_dim,
    )
    batch = 2
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
    base = _attend(
        layer,
        query,
        key,
        value,
    )
    mask = torch.zeros(
        batch,
        n,
        dtype=torch.bool,
    )
    mask[:, dead_key] = True
    padded = _attend(
        layer,
        query,
        key,
        value,
        key_padding_mask=mask,
    )
    torch.testing.assert_close(
        padded[:, query_pos],
        base[:, query_pos],
    )


def test_partial_padding_renormalizes_remaining_live_keys():
    torch.manual_seed(42)
    (
        source_index,
        target_index,
        query_features,
        key_features,
    ) = rectangular_indices()
    allow = allow_matrix(
        source_index,
        target_index,
        query_features,
        key_features,
    )
    live0 = (allow[0] > 0).nonzero(as_tuple=False).view(-1)
    assert live0.tolist() == [0, 1]
    live2 = (allow[2] > 0).nonzero(as_tuple=False).view(-1)
    assert live2.tolist() == [4]
    embed_dim = 8
    layer = _identity_layer(
        source_index,
        target_index,
        query_features,
        key_features,
        embed_dim=embed_dim,
    )
    batch = 2
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
    base = _attend(
        layer,
        query,
        key,
        value,
    )
    torch.testing.assert_close(
        base,
        _oracle(
            layer,
            query,
            key,
            value,
        ),
    )
    torch.testing.assert_close(
        base[:, 2],
        value[:, 4],
    )
    assert not torch.allclose(
        base[:, 0],
        value[:, 1],
    )
    mask = torch.zeros(
        batch,
        key_features,
        dtype=torch.bool,
    )
    mask[:, 0] = True
    padded = _attend(
        layer,
        query,
        key,
        value,
        key_padding_mask=mask,
    )
    torch.testing.assert_close(
        padded[:, 0],
        value[:, 1],
    )
    torch.testing.assert_close(
        padded[:, 2],
        value[:, 4],
    )
    torch.testing.assert_close(
        padded,
        _oracle(
            layer,
            query,
            key,
            value,
            key_padding_mask=mask,
        ),
    )


def test_all_remaining_keys_padded_are_zeros_not_nan():
    torch.manual_seed(42)
    source_index, target_index, n = three_cycle_indices()
    embed_dim = 8
    layer = _identity_layer(
        source_index,
        target_index,
        n,
        n,
        embed_dim=embed_dim,
    )
    batch = 3
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
    mask = torch.ones(
        batch,
        n,
        dtype=torch.bool,
    )
    padded = _attend(
        layer,
        query,
        key,
        value,
        key_padding_mask=mask,
    )
    zeros = torch.zeros_like(padded)
    torch.testing.assert_close(
        padded,
        zeros,
    )
    assert torch.isfinite(padded).all()
    assert not torch.isnan(padded).any()


def test_public_module_matches_oracle_with_same_mask():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    embed_dim = 8
    num_heads = 2
    layer = PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        n,
        n,
        embed_dim,
        num_heads,
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
    mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, False],
            [False, False, False, True],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )
    assert mask.any()
    assert (~mask).any()
    got = _attend(
        layer,
        query,
        key,
        value,
        key_padding_mask=mask,
    )
    expected = _oracle(
        layer,
        query,
        key,
        value,
        key_padding_mask=mask,
    )
    torch.testing.assert_close(
        got,
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


def test_unbatched_and_batched_padding_masks():
    torch.manual_seed(42)
    source_index, target_index, n = three_cycle_indices()
    embed_dim = 8
    layer = _identity_layer(
        source_index,
        target_index,
        n,
        n,
        embed_dim=embed_dim,
    )
    query_u = torch.randn(
        n,
        embed_dim,
    )
    key_u = torch.randn(
        n,
        embed_dim,
    )
    value_u = torch.randn(
        n,
        embed_dim,
    )
    mask_1d = torch.zeros(
        n,
        dtype=torch.bool,
    )
    mask_1d[2] = True
    got_u = _attend(
        layer,
        query_u,
        key_u,
        value_u,
        key_padding_mask=mask_1d,
    )
    expected_u = _oracle(
        layer,
        query_u,
        key_u,
        value_u,
        key_padding_mask=mask_1d,
    )
    torch.testing.assert_close(
        got_u,
        expected_u,
    )
    torch.testing.assert_close(
        got_u[0],
        torch.zeros_like(got_u[0]),
    )
    torch.testing.assert_close(
        got_u[1],
        value_u[0],
    )

    pad_a = 0
    pad_b = 1
    batch = 2
    query_b = torch.randn(
        batch,
        n,
        embed_dim,
    )
    key_b = torch.randn(
        batch,
        n,
        embed_dim,
    )
    value_b = torch.randn(
        batch,
        n,
        embed_dim,
    )
    mask_2d = torch.zeros(
        batch,
        n,
        dtype=torch.bool,
    )
    mask_2d[0, pad_a] = True
    mask_2d[1, pad_b] = True
    got_b = _attend(
        layer,
        query_b,
        key_b,
        value_b,
        key_padding_mask=mask_2d,
    )
    expected_b = _oracle(
        layer,
        query_b,
        key_b,
        value_b,
        key_padding_mask=mask_2d,
    )
    torch.testing.assert_close(
        got_b,
        expected_b,
    )
    assert not torch.allclose(
        got_b[0],
        got_b[1],
    )
    torch.testing.assert_close(
        got_b[0, 1],
        torch.zeros_like(got_b[0, 1]),
    )
    torch.testing.assert_close(
        got_b[0, 2],
        value_b[0, 1],
    )
    torch.testing.assert_close(
        got_b[1, 2],
        torch.zeros_like(got_b[1, 2]),
    )
    torch.testing.assert_close(
        got_b[1, 1],
        value_b[1, 0],
    )

    with pytest.raises(
        Kpnn2Error,
        match="key_padding_mask",
    ):
        layer(
            query_b,
            key_b,
            value_b,
            key_padding_mask=mask_1d,
            need_weights=False,
        )
