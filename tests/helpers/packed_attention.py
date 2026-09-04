"""
Test-only dense reference for packed multi-head attention.

Not part of the public kpnn2 API. May allocate an
``(n_query, n_key)`` score matrix. ``PackedMultiheadAttention``
must never do that.
"""

import pandas as pd
import torch
from torch import nn


def allow_matrix(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    query_features: int,
    key_features: int,
) -> torch.Tensor:
    """
    Dense 0/1 allow matrix; row is query (target), column is key
    (source). Same layout as ``AdjacencySpec.to_mask()``.
    """
    source_index = torch.as_tensor(
        source_index,
        dtype=torch.int64,
    ).cpu()
    target_index = torch.as_tensor(
        target_index,
        dtype=torch.int64,
    ).cpu()
    mask = torch.zeros(
        (query_features, key_features),
        dtype=torch.float32,
    )
    for source, target in zip(
        source_index.tolist(),
        target_index.tolist(),
    ):
        mask[target, source] = 1.0
    return mask


def shape_heads(
    projected: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """
    Split the last embed axis into ``(num_heads, head_dim)``.
    """
    *leading, seq, embed_dim = projected.shape
    if embed_dim % num_heads != 0:
        raise ValueError("'embed_dim' must be divisible by 'num_heads'.")
    head_dim = embed_dim // num_heads
    return projected.reshape(
        *leading,
        seq,
        num_heads,
        head_dim,
    )


def dense_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Dense scaled-dot attention on live packed pairs.

    ``query`` / ``key`` / ``value`` are
    ``(..., seq, heads, head_dim)``. Dead pairs (no packed
    edge, or a padded key) get additive
    ``torch.finfo(dtype).min``. Softmax is over the key axis.
    A query with no remaining live keys is zeros, not NaN.
    """
    n_query = query.shape[-3]
    n_key = key.shape[-3]
    head_dim = query.shape[-1]
    scale = head_dim**-0.5
    scores = (
        torch.einsum(
            "...qhd,...khd->...qhk",
            query,
            key,
        )
        * scale
    )
    allow = allow_matrix(
        source_index,
        target_index,
        n_query,
        n_key,
    )
    live = allow > 0
    live = live.to(device=query.device)
    if key_padding_mask is not None:
        if not isinstance(key_padding_mask, torch.Tensor):
            raise TypeError("'key_padding_mask' must be a boolean tensor.")
        if key_padding_mask.is_floating_point():
            raise TypeError(
                "Float padding masks are not supported; "
                "'key_padding_mask' must be boolean."
            )
        if key_padding_mask.dtype != torch.bool:
            raise TypeError("'key_padding_mask' must be a boolean tensor.")
        padded = key_padding_mask.to(device=query.device)
        if int(padded.shape[-1]) != n_key:
            raise ValueError(
                "'key_padding_mask' last dimension must equal "
                "the key sequence length."
            )
        pad_leading = tuple(padded.shape[:-1])
        batch_shape = tuple(query.shape[:-3])
        if pad_leading not in (
            (),
            batch_shape,
        ):
            raise ValueError(
                "'key_padding_mask' shape must be "
                "(key_features,) or match query batch dims "
                "plus (key_features,)."
            )
        live = live & ~padded.unsqueeze(-2)
    fill = torch.finfo(scores.dtype).min
    masked_scores = scores.masked_fill(
        ~live.unsqueeze(-2),
        fill,
    )
    attn = torch.softmax(
        masked_scores,
        dim=-1,
    )
    attn = torch.nan_to_num(
        attn,
        nan=0.0,
    )
    has_live = live.any(dim=-1)
    attn = attn * has_live.unsqueeze(-1).unsqueeze(-1).to(
        dtype=attn.dtype,
    )
    mixed = torch.einsum(
        "...qhk,...khd->...qhd",
        attn,
        value,
    )
    return mixed


def cyclic_edgelist() -> pd.DataFrame:
    """
    Inputs, outputs, a 2-cycle; isolated-query inputs after parse.
    """
    return pd.DataFrame(
        {
            "source": ["x", "a", "b", "a"],
            "target": ["a", "b", "a", "y"],
        }
    )


def self_loop_edgelist() -> pd.DataFrame:
    """
    One hidden self-loop between an input and an output.
    """
    return pd.DataFrame(
        {
            "source": ["in", "h", "h"],
            "target": ["h", "h", "out"],
        }
    )


def three_cycle_indices() -> tuple[list[int], list[int], int]:
    """
    3-cycle covering every node.

    Do not pass this through ``parse_adjacency``: that parser
    requires an in-degree-0 input and would reject a pure cycle.
    """
    source_index = [0, 1, 2]
    target_index = [1, 2, 0]
    n = 3
    return source_index, target_index, n


def rectangular_indices() -> tuple[list[int], list[int], int, int]:
    """
    Bipartite map: 3 queries, 5 keys; query 1 is isolated.
    """
    source_index = [0, 1, 4]
    target_index = [0, 0, 2]
    query_features = 3
    key_features = 5
    return (
        source_index,
        target_index,
        query_features,
        key_features,
    )


def pin_projections_identity(layer: nn.Module) -> None:
    """
    Set Q/K/V/out projections to identity; zero biases if present.
    """
    embed_dim = layer.embed_dim
    identity = torch.eye(
        embed_dim,
        dtype=layer.q_proj.weight.dtype,
        device=layer.q_proj.weight.device,
    )
    with torch.no_grad():
        for proj in (
            layer.q_proj,
            layer.k_proj,
            layer.v_proj,
            layer.out_proj,
        ):
            proj.weight.copy_(identity)
            if proj.bias is not None:
                proj.bias.zero_()
