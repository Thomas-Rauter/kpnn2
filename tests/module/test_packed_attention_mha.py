"""
Same job as torch.nn.MultiheadAttention on live pairs.

Secondary oracle: copied weights plus an additive -inf
attn_mask on dead pairs. Every query has a live key
(three_cycle_indices). Kernel vs dense helper is
test_packed_attention_kernel.py; this file does not
duplicate that.

MHA is not defined the same way on empty neighborhoods
(softmax over all -inf is NaN). Isolated queries are out
of scope here.
"""

import pytest
import torch
from torch import nn

from kpnn2 import (
    Kpnn2Error,
    PackedMultiheadAttention,
    parse_adjacency,
)
from tests.helpers.packed_attention import (
    allow_matrix,
    cyclic_edgelist,
    three_cycle_indices,
)

_EMBED_DIM = 8
_NUM_HEADS = 2
_BATCH = 4


def _copy_packed_weights_to_mha(
    packed,
    mha,
):
    """
    Copy four separate Linears into fused MHA in_proj / out_proj.
    """
    with torch.no_grad():
        mha.in_proj_weight.copy_(
            torch.cat(
                (
                    packed.q_proj.weight,
                    packed.k_proj.weight,
                    packed.v_proj.weight,
                ),
                dim=0,
            )
        )
        if packed.q_proj.bias is not None:
            mha.in_proj_bias.copy_(
                torch.cat(
                    (
                        packed.q_proj.bias,
                        packed.k_proj.bias,
                        packed.v_proj.bias,
                    ),
                    dim=0,
                )
            )
        else:
            assert mha.in_proj_bias is None
        mha.out_proj.weight.copy_(
            packed.out_proj.weight,
        )
        if packed.out_proj.bias is not None:
            mha.out_proj.bias.copy_(
                packed.out_proj.bias,
            )
        else:
            assert mha.out_proj.bias is None


def _three_cycle():
    source_index, target_index, n = three_cycle_indices()
    allow = allow_matrix(
        source_index,
        target_index,
        n,
        n,
    )
    assert torch.all(allow.sum(dim=1) > 0)
    # allow is [query, key]; live 3-cycle pairs.
    assert allow[1, 0].item() == 1.0
    assert allow[2, 1].item() == 1.0
    assert allow[0, 2].item() == 1.0
    return source_index, target_index, n, allow


def _additive_attn_mask(allow):
    fill = torch.finfo(torch.float32).min
    mask = torch.zeros_like(allow)
    mask.masked_fill_(
        allow == 0,
        fill,
    )
    return mask


def _make_pair(
    source_index,
    target_index,
    n,
    bias=True,
    batch_first=True,
):
    packed = PackedMultiheadAttention(
        source_index,
        target_index,
        n,
        n,
        _EMBED_DIM,
        _NUM_HEADS,
        dropout=0.0,
        bias=bias,
        batch_first=batch_first,
    )
    mha = nn.MultiheadAttention(
        _EMBED_DIM,
        _NUM_HEADS,
        dropout=0.0,
        bias=bias,
        batch_first=batch_first,
    )
    _copy_packed_weights_to_mha(
        packed,
        mha,
    )
    packed.eval()
    mha.eval()
    return packed, mha


def _assert_close(
    actual,
    expected,
):
    torch.testing.assert_close(
        actual,
        expected,
    )


def _forward_both(
    packed,
    mha,
    query,
    key,
    value,
    attn_mask,
):
    packed_out, packed_weights = packed(
        query,
        key,
        value,
        need_weights=False,
    )
    mha_out, _ = mha(
        query,
        key,
        value,
        attn_mask=attn_mask,
        need_weights=False,
    )
    assert packed_weights is None
    _assert_close(
        packed_out,
        mha_out,
    )
    return packed_out, mha_out


def test_batched_batch_first_matches_mha():
    torch.manual_seed(42)
    source_index, target_index, n, allow = _three_cycle()
    packed, mha = _make_pair(
        source_index,
        target_index,
        n,
        bias=True,
        batch_first=True,
    )
    x = torch.randn(
        _BATCH,
        n,
        _EMBED_DIM,
    )
    packed_out, mha_out = _forward_both(
        packed,
        mha,
        x,
        x,
        x,
        _additive_attn_mask(allow),
    )
    assert packed_out.shape == (
        _BATCH,
        n,
        _EMBED_DIM,
    )
    assert mha_out.shape == packed_out.shape


def test_unbatched_2d_matches_mha():
    torch.manual_seed(42)
    source_index, target_index, n, allow = _three_cycle()
    packed, mha = _make_pair(
        source_index,
        target_index,
        n,
        bias=True,
        batch_first=True,
    )
    x = torch.randn(
        n,
        _EMBED_DIM,
    )
    packed_out, mha_out = _forward_both(
        packed,
        mha,
        x,
        x,
        x,
        _additive_attn_mask(allow),
    )
    assert packed_out.shape == (
        n,
        _EMBED_DIM,
    )
    assert mha_out.shape == packed_out.shape


def test_bias_false_batched_matches_mha():
    torch.manual_seed(42)
    source_index, target_index, n, allow = _three_cycle()
    packed, mha = _make_pair(
        source_index,
        target_index,
        n,
        bias=False,
        batch_first=True,
    )
    x = torch.randn(
        _BATCH,
        n,
        _EMBED_DIM,
    )
    packed_out, mha_out = _forward_both(
        packed,
        mha,
        x,
        x,
        x,
        _additive_attn_mask(allow),
    )
    assert packed_out.shape == (
        _BATCH,
        n,
        _EMBED_DIM,
    )
    assert mha_out.shape == packed_out.shape


def test_batch_first_false_matches_mha():
    torch.manual_seed(42)
    source_index, target_index, n, allow = _three_cycle()
    packed, mha = _make_pair(
        source_index,
        target_index,
        n,
        bias=True,
        batch_first=False,
    )
    x = torch.randn(
        n,
        _BATCH,
        _EMBED_DIM,
    )
    packed_out, mha_out = _forward_both(
        packed,
        mha,
        x,
        x,
        x,
        _additive_attn_mask(allow),
    )
    assert packed_out.shape == (
        n,
        _BATCH,
        _EMBED_DIM,
    )
    assert mha_out.shape == packed_out.shape


def test_distinct_qkv_matches_mha():
    torch.manual_seed(42)
    source_index, target_index, n, allow = _three_cycle()
    packed, mha = _make_pair(
        source_index,
        target_index,
        n,
        bias=True,
        batch_first=True,
    )
    query = torch.randn(
        _BATCH,
        n,
        _EMBED_DIM,
    )
    key = torch.randn(
        _BATCH,
        n,
        _EMBED_DIM,
    )
    value = torch.randn(
        _BATCH,
        n,
        _EMBED_DIM,
    )
    packed_out, mha_out = _forward_both(
        packed,
        mha,
        query,
        key,
        value,
        _additive_attn_mask(allow),
    )
    assert packed_out.shape == query.shape
    assert mha_out.shape == packed_out.shape


def test_mha_comparison_needs_a_query_with_a_live_key():
    # MHA softmax over all -inf is NaN. cyclic_edgelist has
    # isolated inputs, so it is not an MHA reference graph.
    # three_cycle_indices covers every query; that is the
    # graph the tests above compare to MHA.
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    cyclic_allow = allow_matrix(
        spec.source_index,
        spec.target_index,
        n,
        n,
    )
    assert (cyclic_allow.sum(dim=1) == 0).any()
    _source, _target, _n_cycle, cycle_allow = _three_cycle()
    assert torch.all(cycle_allow.sum(dim=1) > 0)


def test_packed_rejects_mha_attn_mask():
    torch.manual_seed(42)
    source_index, target_index, n, allow = _three_cycle()
    packed, _ = _make_pair(
        source_index,
        target_index,
        n,
    )
    square = _additive_attn_mask(allow)
    x = torch.randn(
        _BATCH,
        n,
        _EMBED_DIM,
    )
    with pytest.raises(
        Kpnn2Error,
        match="attn_mask",
    ):
        packed(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=square,
        )
