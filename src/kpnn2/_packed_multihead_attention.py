"""
Packed multi-head attention: scores only on live edgelist pairs.
"""

import hashlib
import struct
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ._errors import Kpnn2Error
from ._generator import as_generator, run_preserving_default_rng
from ._identity import as_identity, check_identity, save_identity

_INDEX_DIGEST_KEY = "index_digest"


def _copy_index(
    value: object,
    name: str,
) -> torch.Tensor:
    """
    Copy ``value`` to a 1-D int64 tensor.

    Accepts a 1-D integer ``torch.Tensor`` or a sequence of
    ``int``. The result is contiguous and independent of
    ``value``.
    """
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise Kpnn2Error(
                f"'{name}' must be a 1-dimensional integer "
                "tensor or a sequence of int."
            )
        if value.is_floating_point() or value.dtype == torch.bool:
            raise Kpnn2Error(
                f"'{name}' must be a 1-dimensional integer "
                "tensor or a sequence of int."
            )
        return value.detach().to(dtype=torch.int64).contiguous().clone()

    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise Kpnn2Error(
            f"'{name}' must be a 1-dimensional integer tensor "
            "or a sequence of int."
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise Kpnn2Error(
            f"'{name}' must be a 1-dimensional integer tensor "
            "or a sequence of int."
        ) from exc
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            raise Kpnn2Error(
                f"'{name}' must be a 1-dimensional integer "
                "tensor or a sequence of int."
            )
    return torch.tensor(
        items,
        dtype=torch.int64,
    )


def _positive_int(
    value: object,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Kpnn2Error(f"'{name}' must be a positive int.")
    return value


def _index_digest(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    query_features: int,
    key_features: int,
    embed_dim: int,
    num_heads: int,
) -> torch.Tensor:
    """
    SHA-256 of packed indices plus layer sizes.

    Payload is the int64 C-contiguous bytes of ``source_index``,
    then ``target_index``, then ``query_features``,
    ``key_features``, ``embed_dim``, and ``num_heads`` as
    little-endian int64 so a reshape or a different head
    split cannot collide.
    """
    source_bytes = (
        source_index.detach()
        .cpu()
        .contiguous()
        .to(torch.int64)
        .numpy()
        .tobytes()
    )
    target_bytes = (
        target_index.detach()
        .cpu()
        .contiguous()
        .to(torch.int64)
        .numpy()
        .tobytes()
    )
    sizes = struct.pack(
        "<qqqq",
        int(query_features),
        int(key_features),
        int(embed_dim),
        int(num_heads),
    )
    digest = hashlib.sha256(source_bytes + target_bytes + sizes).digest()
    return torch.tensor(
        tuple(digest),
        dtype=torch.uint8,
    )


def _digest_matches(
    saved: object,
    current: torch.Tensor,
) -> bool:
    if not isinstance(saved, torch.Tensor):
        return False
    saved_flat = saved.detach().cpu().contiguous().reshape(-1)
    if saved_flat.shape != current.shape or saved_flat.dtype != current.dtype:
        return False
    return bool(
        torch.equal(
            saved_flat,
            current,
        )
    )


def _maybe_self_loops(
    source: torch.Tensor,
    target: torch.Tensor,
    query_features: int,
    key_features: int,
    add_self_loops: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not add_self_loops:
        return source, target
    if query_features != key_features:
        raise Kpnn2Error(
            "'add_self_loops' requires query_features == key_features."
        )
    existing = set(
        zip(
            source.tolist(),
            target.tolist(),
        )
    )
    extra_source: list[int] = []
    extra_target: list[int] = []
    for node in range(query_features):
        pair = (node, node)
        if pair not in existing:
            extra_source.append(node)
            extra_target.append(node)
    if not extra_source:
        return source, target
    extra_s = torch.tensor(
        extra_source,
        dtype=torch.int64,
        device=source.device,
    )
    extra_t = torch.tensor(
        extra_target,
        dtype=torch.int64,
        device=target.device,
    )
    return (
        torch.cat(
            [source, extra_s],
            dim=0,
        ),
        torch.cat(
            [target, extra_t],
            dim=0,
        ),
    )


def _expand_index(
    index: torch.Tensor,
    like: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    view_shape = [1] * like.ndim
    view_shape[dim] = -1
    return index.view(*view_shape).expand_as(like)


def _dropout_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Kpnn2Error("'dropout' must be a float >= 0.")
    if not (value >= 0):
        raise Kpnn2Error("'dropout' must be a float >= 0.")
    return float(value)


def _layout_to_batch_first(
    tensor: torch.Tensor,
    name: str,
    seq_len: int,
    seq_attr: str,
    embed_dim: int,
    batch_first: bool,
) -> tuple[torch.Tensor, bool]:
    if not isinstance(tensor, torch.Tensor):
        raise Kpnn2Error(f"'{name}' must be a torch.Tensor.")
    if tensor.ndim < 2:
        raise Kpnn2Error(f"'{name}' must be 2-dimensional or higher.")
    if tensor.shape[-1] != embed_dim:
        raise Kpnn2Error(f"'{name}' last dimension must equal embed_dim.")
    if tensor.ndim == 2:
        if tensor.shape[0] != seq_len:
            raise Kpnn2Error(f"{name} sequence length must equal {seq_attr}.")
        return tensor, False
    if batch_first:
        if tensor.shape[-2] != seq_len:
            raise Kpnn2Error(f"{name} sequence length must equal {seq_attr}.")
        return tensor, False
    if tensor.ndim != 3:
        raise Kpnn2Error(
            "With batch_first=False, inputs must be 2-D "
            "(seq, embed) or 3-D (seq, batch, embed)."
        )
    if tensor.shape[0] != seq_len:
        raise Kpnn2Error(f"{name} sequence length must equal {seq_attr}.")
    return tensor.transpose(0, 1), True


def _padding_participate(
    key_padding_mask: object,
    source_index: torch.Tensor,
    batch_shape: tuple[int, ...],
    key_features: int,
    device: torch.device,
) -> torch.Tensor | None:
    if key_padding_mask is None:
        return None
    if not isinstance(key_padding_mask, torch.Tensor):
        raise Kpnn2Error("'key_padding_mask' must be a boolean tensor.")
    if key_padding_mask.is_floating_point():
        raise Kpnn2Error(
            "Float padding masks are not supported; "
            "'key_padding_mask' must be boolean."
        )
    if key_padding_mask.dtype != torch.bool:
        raise Kpnn2Error("'key_padding_mask' must be a boolean tensor.")
    if key_padding_mask.ndim < 1:
        raise Kpnn2Error(
            "'key_padding_mask' shape must be (S,) unbatched or (N, S) batched."
        )
    if key_padding_mask.ndim == 1:
        if batch_shape != ():
            raise Kpnn2Error(
                "'key_padding_mask' shape must be (S,) "
                "unbatched or (N, S) batched."
            )
        if int(key_padding_mask.shape[0]) != key_features:
            raise Kpnn2Error(
                "'key_padding_mask' last dimension must equal key_features."
            )
    else:
        if int(key_padding_mask.shape[-1]) != key_features:
            raise Kpnn2Error(
                "'key_padding_mask' last dimension must equal key_features."
            )
        if tuple(key_padding_mask.shape[:-1]) != tuple(batch_shape):
            raise Kpnn2Error(
                "'key_padding_mask' shape must be (S,) "
                "unbatched or (N, S) batched."
            )
    mask = key_padding_mask.to(
        device=device,
    )
    index = source_index.to(
        device=device,
    )
    padded = mask[..., index]
    participate = ~padded
    return participate.unsqueeze(-1)


def _packed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    dropout_p: float,
    training: bool,
    participate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Softmax over live keys of each query; never an ``(n, n)`` score
    matrix. ``query`` / ``key`` / ``value`` are
    ``(..., seq, heads, head_dim)``. Returns the mixed values
    and packed weights ``(..., nnz, heads)``.
    """
    head_dim = query.shape[-1]
    scale = head_dim**-0.5
    q_live = query[..., target_index, :, :]
    k_live = key[..., source_index, :, :]
    scores = (q_live * k_live).sum(-1) * scale
    n_query = query.shape[-3]
    leading = query.shape[:-3]
    heads = query.shape[-2]
    fill = torch.finfo(scores.dtype).min
    max_buf = scores.new_full(
        (*leading, n_query, heads),
        fill,
    )
    score_index = _expand_index(
        target_index,
        scores,
        dim=-2,
    )
    scores_for_max = scores
    if participate is not None:
        scores_for_max = scores.masked_fill(
            ~participate,
            fill,
        )
    max_buf.scatter_reduce_(
        -2,
        score_index,
        scores_for_max,
        reduce="amax",
        include_self=True,
    )
    gathered_max = max_buf.gather(
        -2,
        score_index,
    )
    delta = scores - gathered_max
    if participate is not None:
        delta = delta.masked_fill(
            ~participate,
            0.0,
        )
    exp_scores = torch.exp(delta)
    if participate is not None:
        exp_scores = exp_scores * participate.to(
            dtype=exp_scores.dtype,
        )
    sum_buf = torch.zeros(
        (*leading, n_query, heads),
        dtype=scores.dtype,
        device=scores.device,
    )
    sum_buf.scatter_add_(
        -2,
        score_index,
        exp_scores,
    )
    denom = sum_buf.gather(
        -2,
        score_index,
    ).clamp_min(torch.finfo(scores.dtype).tiny)
    attn = exp_scores / denom
    if participate is not None:
        attn = attn * participate.to(
            dtype=attn.dtype,
        )
    if dropout_p > 0.0 and training:
        attn = F.dropout(
            attn,
            p=dropout_p,
        )
    v_live = value[..., source_index, :, :]
    weighted = attn.unsqueeze(-1) * v_live
    out = torch.zeros_like(query)
    value_index = _expand_index(
        target_index,
        weighted,
        dim=-3,
    )
    out.scatter_add_(
        -3,
        value_index,
        weighted,
    )
    return out, attn


def _optional_chunk_size(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Kpnn2Error("'chunk_size' must be None or a positive int.")
    return value


def _live_pair_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    q_live = query[..., target_index, :, :]
    k_live = key[..., source_index, :, :]
    return (q_live * k_live).sum(-1) * scale


def _apply_participate_to_scores(
    scores: torch.Tensor,
    participate: torch.Tensor | None,
    fill: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores_for_max = scores
    exp_delta = scores
    if participate is None:
        return scores_for_max, exp_delta
    scores_for_max = scores.masked_fill(
        ~participate,
        fill,
    )
    exp_delta = scores.masked_fill(
        ~participate,
        0.0,
    )
    return scores_for_max, exp_delta


class _PackedAttentionChunked(torch.autograd.Function):
    """
    Softmax and mix packed pairs in edge chunks.

    Forward never materializes ``(..., nnz, heads, head_dim)``
    pair gathers. Backward rematerializes those gathers from
    saved ``query`` / ``key`` / ``value`` and packed softmax.
    """

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        participate: torch.Tensor,
        dropout_p: float,
        training: bool,
        chunk_size: int,
        has_participate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        part = participate if has_participate else None
        head_dim = query.shape[-1]
        scale = head_dim**-0.5
        n_query = query.shape[-3]
        leading = query.shape[:-3]
        heads = query.shape[-2]
        nnz = int(source_index.shape[0])
        fill = torch.finfo(query.dtype).min
        max_buf = query.new_full(
            (*leading, n_query, heads),
            fill,
        )
        start = 0
        while start < nnz:
            end = min(start + chunk_size, nnz)
            src = source_index[start:end]
            tgt = target_index[start:end]
            scores = _live_pair_scores(
                query,
                key,
                src,
                tgt,
                scale,
            )
            part_chunk = None
            if part is not None:
                part_chunk = part[..., start:end, :]
            scores_for_max, _ = _apply_participate_to_scores(
                scores,
                part_chunk,
                fill,
            )
            score_index = _expand_index(
                tgt,
                scores,
                dim=-2,
            )
            max_buf.scatter_reduce_(
                -2,
                score_index,
                scores_for_max,
                reduce="amax",
                include_self=True,
            )
            start = end

        sum_buf = query.new_zeros(
            (*leading, n_query, heads),
        )
        softmax_attn = query.new_empty(
            (
                *leading,
                nnz,
                heads,
            ),
        )
        start = 0
        while start < nnz:
            end = min(start + chunk_size, nnz)
            src = source_index[start:end]
            tgt = target_index[start:end]
            scores = _live_pair_scores(
                query,
                key,
                src,
                tgt,
                scale,
            )
            part_chunk = None
            if part is not None:
                part_chunk = part[..., start:end, :]
            _, delta_scores = _apply_participate_to_scores(
                scores,
                part_chunk,
                fill,
            )
            score_index = _expand_index(
                tgt,
                scores,
                dim=-2,
            )
            gathered_max = max_buf.gather(
                -2,
                score_index,
            )
            delta = delta_scores - gathered_max
            if part_chunk is not None:
                delta = delta.masked_fill(
                    ~part_chunk,
                    0.0,
                )
            exp_scores = torch.exp(delta)
            if part_chunk is not None:
                exp_scores = exp_scores * part_chunk.to(
                    dtype=exp_scores.dtype,
                )
            sum_buf.scatter_add_(
                -2,
                score_index,
                exp_scores,
            )
            softmax_attn[..., start:end, :] = exp_scores
            start = end

        tiny = torch.finfo(query.dtype).tiny
        start = 0
        while start < nnz:
            end = min(start + chunk_size, nnz)
            tgt = target_index[start:end]
            exp_chunk = softmax_attn[..., start:end, :]
            score_index = _expand_index(
                tgt,
                exp_chunk,
                dim=-2,
            )
            denom = sum_buf.gather(
                -2,
                score_index,
            ).clamp_min(tiny)
            chunk_attn = exp_chunk / denom
            if part is not None:
                chunk_attn = chunk_attn * part[..., start:end, :].to(
                    dtype=chunk_attn.dtype,
                )
            softmax_attn[..., start:end, :] = chunk_attn
            start = end

        used_dropout = dropout_p > 0.0 and training
        dropped_attn = softmax_attn
        if used_dropout:
            dropped_attn = F.dropout(
                softmax_attn,
                p=dropout_p,
            )

        out = torch.zeros_like(query)
        start = 0
        while start < nnz:
            end = min(start + chunk_size, nnz)
            src = source_index[start:end]
            tgt = target_index[start:end]
            v_live = value[..., src, :, :]
            weighted = dropped_attn[..., start:end, :].unsqueeze(-1) * v_live
            value_index = _expand_index(
                tgt,
                weighted,
                dim=-3,
            )
            out.scatter_add_(
                -3,
                value_index,
                weighted,
            )
            start = end

        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.has_participate = has_participate
        ctx.used_dropout = used_dropout
        if used_dropout:
            ctx.save_for_backward(
                query,
                key,
                value,
                source_index,
                target_index,
                participate,
                softmax_attn,
                dropped_attn,
            )
        else:
            ctx.save_for_backward(
                query,
                key,
                value,
                source_index,
                target_index,
                participate,
                softmax_attn,
            )
        return out, dropped_attn

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor | None,
        grad_attn: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        need_q, need_k, need_v, *_ = ctx.needs_input_grad
        if not (need_q or need_k or need_v):
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        saved = ctx.saved_tensors
        query = saved[0]
        key = saved[1]
        value = saved[2]
        source_index = saved[3]
        target_index = saved[4]
        participate = saved[5]
        softmax_attn = saved[6]
        if ctx.used_dropout:
            dropped_attn = saved[7]
        else:
            dropped_attn = softmax_attn
        part = participate if ctx.has_participate else None
        scale = ctx.scale
        chunk_size = ctx.chunk_size
        n_query = query.shape[-3]
        leading = query.shape[:-3]
        heads = query.shape[-2]
        nnz = int(source_index.shape[0])

        grad_query = None
        grad_key = None
        grad_value = None
        if need_q:
            grad_query = torch.zeros_like(query)
        if need_k:
            grad_key = torch.zeros_like(key)
        if need_v:
            grad_value = torch.zeros_like(value)

        d_dropped = None
        if need_q or need_k:
            if grad_attn is None:
                d_dropped = torch.zeros_like(softmax_attn)
            else:
                d_dropped = grad_attn.clone()

        start = 0
        while start < nnz:
            end = min(start + chunk_size, nnz)
            src = source_index[start:end]
            tgt = target_index[start:end]
            dropped = dropped_attn[..., start:end, :]
            if grad_out is not None:
                d_out_live = grad_out[..., tgt, :, :]
                if need_v and grad_value is not None:
                    d_v_live = dropped.unsqueeze(-1) * d_out_live
                    v_index = _expand_index(
                        src,
                        d_v_live,
                        dim=-3,
                    )
                    grad_value.scatter_add_(
                        -3,
                        v_index,
                        d_v_live,
                    )
                if d_dropped is not None:
                    v_live = value[..., src, :, :]
                    d_dropped[..., start:end, :] = d_dropped[
                        ...,
                        start:end,
                        :,
                    ] + (d_out_live * v_live).sum(-1)
            start = end

        if d_dropped is not None:
            if ctx.used_dropout:
                drop_scale = torch.where(
                    softmax_attn != 0,
                    dropped_attn / softmax_attn,
                    torch.zeros_like(softmax_attn),
                )
                d_softmax = d_dropped * drop_scale
            else:
                d_softmax = d_dropped
            if part is not None:
                d_softmax = d_softmax * part.to(
                    dtype=d_softmax.dtype,
                )
            dot_buf = query.new_zeros(
                (*leading, n_query, heads),
            )
            start = 0
            while start < nnz:
                end = min(start + chunk_size, nnz)
                tgt = target_index[start:end]
                a_chunk = softmax_attn[..., start:end, :]
                da_chunk = d_softmax[..., start:end, :]
                prod = a_chunk * da_chunk
                score_index = _expand_index(
                    tgt,
                    prod,
                    dim=-2,
                )
                dot_buf.scatter_add_(
                    -2,
                    score_index,
                    prod,
                )
                start = end
            start = 0
            while start < nnz:
                end = min(start + chunk_size, nnz)
                src = source_index[start:end]
                tgt = target_index[start:end]
                a_chunk = softmax_attn[..., start:end, :]
                da_chunk = d_softmax[..., start:end, :]
                score_index = _expand_index(
                    tgt,
                    a_chunk,
                    dim=-2,
                )
                gathered_dot = dot_buf.gather(
                    -2,
                    score_index,
                )
                d_scores = a_chunk * (da_chunk - gathered_dot)
                q_live = query[..., tgt, :, :]
                k_live = key[..., src, :, :]
                if need_q and grad_query is not None:
                    d_q_live = d_scores.unsqueeze(-1) * k_live * scale
                    q_index = _expand_index(
                        tgt,
                        d_q_live,
                        dim=-3,
                    )
                    grad_query.scatter_add_(
                        -3,
                        q_index,
                        d_q_live,
                    )
                if need_k and grad_key is not None:
                    d_k_live = d_scores.unsqueeze(-1) * q_live * scale
                    k_index = _expand_index(
                        src,
                        d_k_live,
                        dim=-3,
                    )
                    grad_key.scatter_add_(
                        -3,
                        k_index,
                        d_k_live,
                    )
                start = end

        return (
            grad_query,
            grad_key,
            grad_value,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _packed_attention_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    dropout_p: float,
    training: bool,
    chunk_size: int,
    participate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the packed mix in edge chunks.

    Softmax is still over every live key of a query. Pair
    gathers of shape ``(..., chunk, heads, head_dim)`` are
    rematerialized in backward.
    """
    dummy = query.new_empty(0)
    part = dummy if participate is None else participate
    return _PackedAttentionChunked.apply(
        query,
        key,
        value,
        source_index,
        target_index,
        part,
        dropout_p,
        training,
        chunk_size,
        participate is not None,
    )


class PackedMultiheadAttention(nn.Module):
    """
    Multi-head attention restricted to the live edges of a named edgelist.

    Prior knowledge decides which node may attend to which: the
    query at an edge's target sees only the keys at its sources.
    Reach for it instead of masking ``nn.MultiheadAttention``,
    whose mask costs an ``(n, n)`` matrix. Build it from
    ``parse_adjacency`` indices and wrap it in your own encoder.
    Defaults differ from MHA: ``batch_first=True`` and
    ``need_weights=False``. ``True`` returns packed per-edge
    weights, not a dense ``(L, S)`` matrix.

    Parameters
    ----------
    source_index : torch.Tensor or sequence of int
        1-D integer indices of length ``nnz >= 1``. Entry ``i``
        is the key / value position of live edge ``i``, and must
        satisfy ``0 <= source_index < key_features``. Copied to
        an int64 buffer, so later writes to the argument do not
        reach this layer.
    target_index : torch.Tensor or sequence of int
        1-D integer indices of the same length. Entry ``i`` is
        the query position of live edge ``i``, and must satisfy
        ``0 <= target_index < query_features``. The two arrays
        are paired position by position and must not repeat a
        ``(source, target)`` pair.
    query_features : int
        Sequence length of ``query``, that is, how many query
        positions the packed indices address. Positive int.
    key_features : int
        Sequence length of ``key`` / ``value``. Positive int.
        Equal to ``query_features`` for a self-attention graph
        over one node set; smaller or larger for a bipartite
        query/key map.
    embed_dim : int
        Model width of ``query`` / ``key`` / ``value`` and of the
        output. Positive int, divisible by ``num_heads``.
    num_heads : int
        Number of attention heads. Each head attends over the
        same live pairs with ``embed_dim // num_heads`` channels.
    dropout : float, default=0.0
        Dropout probability applied to the packed attention
        weights in training mode only; ``0.0`` disables it.
        Must be ``>= 0``; integer ``0`` is accepted, ``bool``
        and negatives are rejected.
    bias : bool, default=True
        Whether the four projections learn a bias. ``False``
        makes them pure linear maps.
    kdim : int or None, default=None
        Key embed width. Must be ``None`` or equal to
        ``embed_dim``; kept for ``nn.MultiheadAttention``
        call-site parity, not to support a differing width.
    vdim : int or None, default=None
        Value embed width, under the same restriction as
        ``kdim``.
    batch_first : bool, default=True
        Layout of batched tensors: ``(..., seq, embed_dim)`` when
        ``True`` (kpnn2 sample-major), ``(seq, batch, embed_dim)``
        when ``False`` (the ``nn.MultiheadAttention`` layout).
        Unbatched 2-D ``(seq, embed)`` ignores this flag.
    add_self_loops : bool, default=False
        If ``True``, OR any missing ``(i, i)`` pair into the
        module buffers, so every query keeps its own token as a
        key. Existing self-loops are kept, not duplicated, and
        the caller's index objects are not mutated. Requires
        ``query_features == key_features``.
    identity : str or None, default=None
        Opaque checkpoint identity, typically
        ``spec.fingerprint``. Stored in ``state_dict`` next to
        ``index_digest`` as a 1-D CPU ``uint8`` tensor of the
        UTF-8 bytes. ``load_state_dict`` raises ``Kpnn2Error``
        when a present identity does not match this layer, and
        does not load the weights. A missing identity is not an
        error, even with ``strict=True``. ``None`` means this
        layer does not claim an identity.
    generator : torch.Generator or None, default=None
        Isolated RNG for Xavier-uniform on the four
        projections. ``None`` uses the default torch
        generator and keeps today's path: ``nn.Linear``
        kaiming-init then Xavier on that global stream.
        When set, those ``nn.Linear`` constructors do not
        advance the global stream. Not stored on the module;
        pass it again to ``reset_parameters`` to replay. Do
        not pass a seed integer.
    chunk_size : int or None, default=None
        How many live edges to gather at once in the packed
        softmax and mix. ``None`` gathers all live pairs at
        once. A positive int uses chunked live pairs: softmax
        is still over every live key of a query, but pair
        gathers are slices of that many edges and those
        gathers are rematerialized in backward. ``bool`` and
        values other than ``None`` or a positive ``int``
        raise ``Kpnn2Error``. Not stored on the module
        ``state_dict`` or in ``index_digest``.

    Attributes
    ----------
    query_features : int
        Sequence length of ``query``.
    key_features : int
        Sequence length of ``key`` / ``value``.
    embed_dim : int
        Model width.
    num_heads : int
        Number of attention heads.
    head_dim : int
        ``embed_dim // num_heads``.
    nnz : int
        Number of live edges, counting any pair added by
        ``add_self_loops``.
    dropout : float
        Dropout probability on packed attention weights.
    batch_first : bool
        Layout flag; default ``True``.
    add_self_loops : bool
        Whether missing self-loops were OR-ed at construction.
    source_index : torch.Tensor
        Int64 buffer of key / value positions, length ``nnz``.
        **Treat it as read-only:** like any PyTorch buffer it can
        be written to, and doing so rewires the layer without
        reinitializing it. Rebuild from the edgelist instead.
    target_index : torch.Tensor
        Int64 buffer of query positions, length ``nnz``,
        read-only in the same sense.
    q_proj, k_proj, v_proj, out_proj : nn.Linear
        Four separate ``embed_dim -> embed_dim`` projections, not
        a fused ``in_proj_weight``.
    identity : str | None
        The constructor ``identity``, or ``None``.
    chunk_size : int | None
        Edge-chunk length, or ``None`` for all live pairs.

    Raises
    ------
    Kpnn2Error
        At construction, if the indices are empty, not 1-D
        integers, mismatched in length, out of range, or
        duplicated as ``(source, target)`` pairs; if the sizes
        are not positive ints; if ``embed_dim`` is not divisible
        by ``num_heads``; if ``dropout`` is a ``bool`` or
        negative; if ``kdim`` / ``vdim`` are neither ``None`` nor
        ``embed_dim``; if ``add_self_loops`` is set when
        ``query_features != key_features``; if ``identity`` is
        neither a ``str`` nor ``None``; if ``generator`` is
        neither a ``torch.Generator`` nor ``None``; or if
        ``chunk_size`` is neither ``None`` nor a positive
        ``int``. From
        ``load_state_dict``, when the checkpoint carries an index
        digest or identity that does not match this layer, in
        which case the weights are not loaded. ``forward``
        documents its own rejected arguments.

    See Also
    --------
    PackedLinear : Same packed pairs when the update is one
        trainable scalar per edge rather than a contraction.
    MaskedLinear : Dense ``(out, in)`` weight; the default layer
        whenever that square fits.
    AdjacencySpec : Supplies ``source_index`` / ``target_index``;
        this layer takes those tuples, not the spec object.
    torch.nn.MultiheadAttention : Dense equivalent, and the
        reference for call shape, heads, and training.

    Notes
    -----
    The mix is a softmax over each query's live keys only, formed
    edge by edge, so no ``(n, n)`` or ``(L, S)`` score matrix is
    allocated and ``torch.sparse`` is not imported; cost scales
    with ``nnz``, not with ``query_features * key_features``.
    ``chunk_size=None`` (the default) gathers all live pairs at
    once. A positive ``chunk_size`` gathers chunked live pairs
    and rematerializes those ``(..., chunk, heads, head_dim)``
    tensors in backward so training does not save a full
    ``(..., nnz, heads, head_dim)`` gather. Packed softmax
    ``(..., nnz, heads)`` is still stored. A query with no live
    keys (and none added by ``add_self_loops``) mixes to zeros
    rather than NaN, then still goes through ``out_proj``. Input
    nodes of an ``AdjacencySpec`` are exactly that case.

    ``forward`` documents the accepted layouts, packed
    ``need_weights``, ``attn_mask`` / ``is_causal``, and
    ``key_padding_mask``.

    Index buffers stay integer after ``.half()`` / bfloat16 /
    ``.double()``; projection weights follow the module floating
    dtype like ``nn.Linear``. ``torch.autocast`` is unsupported:
    ``forward`` disables it and casts query, key, and value to
    the parameter dtype. Cast the module with
    ``.to(dtype=...)`` (or ``.half()`` / ``.double()``)
    instead. ``state_dict`` adds
    ``index_digest``, a 1-D CPU ``uint8`` tensor of length 32:
    the SHA-256 of the packed indices plus ``query_features``,
    ``key_features``, ``embed_dim``, and ``num_heads``, so a
    reshape or a different head split cannot collide. It is not
    a registered buffer, and a missing digest is not an error,
    even with ``strict=True``. ``identity`` is stored the same
    way when the constructor was given one: a rename that
    leaves the packed index pattern unchanged is caught when
    callers pass ``spec.fingerprint``. ``chunk_size`` is not
    part of ``state_dict`` or ``index_digest``. ``copy.deepcopy``
    works.

    Examples
    --------
    An input feeding a two-node feedback core plus one output,
    attended as one four-token sequence:

    >>> import pandas as pd
    >>> import torch
    >>> import kpnn2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["x", "a", "b", "a"],
    ...         "target": ["a", "b", "a", "y"],
    ...     }
    ... )
    >>> spec = kpnn2.parse_adjacency(edgelist)
    >>> n = len(spec.nodes)
    >>> attn = kpnn2.PackedMultiheadAttention(
    ...     spec.source_index,
    ...     spec.target_index,
    ...     n,
    ...     n,
    ...     embed_dim=8,
    ...     num_heads=2,
    ...     add_self_loops=True,
    ...     identity=spec.fingerprint,
    ... )
    >>> attn.nnz, attn.head_dim
    (8, 4)
    >>> tokens = torch.randn(2, n, 8)
    >>> output, weights = attn(tokens, tokens, tokens)
    >>> tuple(output.shape)
    (2, 4, 8)
    >>> weights is None
    True
    >>> _, packed = attn(
    ...     tokens,
    ...     tokens,
    ...     tokens,
    ...     need_weights=True,
    ... )
    >>> tuple(packed.shape)
    (2, 8)
    """

    source_index: torch.Tensor
    target_index: torch.Tensor
    identity: str | None

    def __init__(
        self,
        source_index: object,
        target_index: object,
        query_features: int,
        key_features: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = True,
        add_self_loops: bool = False,
        *,
        identity: str | None = None,
        generator: torch.Generator | None = None,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        query_features = _positive_int(
            query_features,
            "query_features",
        )
        key_features = _positive_int(
            key_features,
            "key_features",
        )
        embed_dim = _positive_int(
            embed_dim,
            "embed_dim",
        )
        num_heads = _positive_int(
            num_heads,
            "num_heads",
        )
        if embed_dim % num_heads != 0:
            raise Kpnn2Error("'embed_dim' must be divisible by 'num_heads'.")
        dropout = _dropout_value(dropout)
        if kdim is not None and kdim != embed_dim:
            raise Kpnn2Error("'kdim' must be None or equal to embed_dim.")
        if vdim is not None and vdim != embed_dim:
            raise Kpnn2Error("'vdim' must be None or equal to embed_dim.")
        source = _copy_index(
            source_index,
            "source_index",
        )
        target = _copy_index(
            target_index,
            "target_index",
        )
        if source.shape != target.shape:
            raise Kpnn2Error(
                "'source_index' and 'target_index' must have the same length."
            )
        nnz = int(source.numel())
        if nnz == 0:
            raise Kpnn2Error(
                "'source_index' and 'target_index' must contain "
                "at least one index."
            )
        if torch.any(source < 0) or torch.any(source >= key_features):
            raise Kpnn2Error(
                "'source_index' entries must satisfy "
                "0 <= source_index < key_features."
            )
        if torch.any(target < 0) or torch.any(target >= query_features):
            raise Kpnn2Error(
                "'target_index' entries must satisfy "
                "0 <= target_index < query_features."
            )
        pairs = list(
            zip(
                source.tolist(),
                target.tolist(),
            )
        )
        if len(set(pairs)) != len(pairs):
            raise Kpnn2Error(
                "PackedMultiheadAttention indices contain "
                "duplicate (source, target) pair(s)."
            )
        source, target = _maybe_self_loops(
            source,
            target,
            query_features,
            key_features,
            add_self_loops,
        )
        nnz = int(source.numel())

        self.query_features = query_features
        self.key_features = key_features
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.nnz = nnz
        self.dropout = dropout
        self.batch_first = batch_first
        self.add_self_loops = add_self_loops
        self.identity = as_identity(identity)
        self.chunk_size = _optional_chunk_size(chunk_size)
        generator = as_generator(generator)
        self.register_buffer(
            "source_index",
            source,
        )
        self.register_buffer(
            "target_index",
            target,
        )

        def _init_projections() -> None:
            self.q_proj = nn.Linear(
                embed_dim,
                embed_dim,
                bias=bias,
            )
            self.k_proj = nn.Linear(
                embed_dim,
                embed_dim,
                bias=bias,
            )
            self.v_proj = nn.Linear(
                embed_dim,
                embed_dim,
                bias=bias,
            )
            self.out_proj = nn.Linear(
                embed_dim,
                embed_dim,
                bias=bias,
            )

        if generator is None:
            _init_projections()
        else:
            run_preserving_default_rng(_init_projections)
        self.reset_parameters(generator)

    def reset_parameters(
        self,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        Xavier-uniform projections, like ``nn.MultiheadAttention``.

        Parameters
        ----------
        generator : torch.Generator or None, default=None
            Isolated RNG for the four Xavier draws. ``None``
            uses the default torch generator. Not stored on
            the module. Biases, when present, are zeroed and
            do not consume the generator.
        """
        generator = as_generator(generator)
        nn.init.xavier_uniform_(
            self.q_proj.weight,
            generator=generator,
        )
        nn.init.xavier_uniform_(
            self.k_proj.weight,
            generator=generator,
        )
        nn.init.xavier_uniform_(
            self.v_proj.weight,
            generator=generator,
        )
        nn.init.xavier_uniform_(
            self.out_proj.weight,
            generator=generator,
        )
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def extra_repr(self) -> str:
        """
        Sizes, live-edge count, dropout, layout, and chunk_size.
        """
        return (
            f"query_features={self.query_features}, "
            f"key_features={self.key_features}, "
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"nnz={self.nnz}, "
            f"dropout={self.dropout}, "
            f"batch_first={self.batch_first}, "
            f"add_self_loops={self.add_self_loops}, "
            f"chunk_size={self.chunk_size}"
        )

    def _save_to_state_dict(
        self,
        destination: dict[str, Any],
        prefix: str,
        keep_vars: bool,
    ) -> None:
        super()._save_to_state_dict(
            destination,
            prefix,
            keep_vars,
        )
        destination[prefix + _INDEX_DIGEST_KEY] = _index_digest(
            self.source_index,
            self.target_index,
            self.query_features,
            self.key_features,
            self.embed_dim,
            self.num_heads,
        )
        save_identity(
            destination,
            prefix,
            self.identity,
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        check_identity(
            state_dict,
            prefix,
            self.identity,
        )
        key = prefix + _INDEX_DIGEST_KEY
        saved = state_dict.pop(key, None)
        if saved is not None:
            current = _index_digest(
                self.source_index,
                self.target_index,
                self.query_features,
                self.key_features,
                self.embed_dim,
                self.num_heads,
            )
            if not _digest_matches(
                saved,
                current,
            ):
                raise Kpnn2Error(
                    "The checkpoint indices do not match this layer."
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _shape_heads(
        self,
        projected: torch.Tensor,
    ) -> torch.Tensor:
        *leading, seq, embed_dim = projected.shape
        if embed_dim != self.embed_dim:
            raise Kpnn2Error("Last dimension must equal embed_dim.")
        return projected.reshape(
            *leading,
            seq,
            self.num_heads,
            self.head_dim,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Packed multi-head attention; drop-in call shape vs MHA.

        ``query`` last dim is ``embed_dim``; sequence length is
        ``query_features``. ``key`` and ``value`` last dim is
        ``embed_dim``; sequence length is ``key_features``.

        With ``batch_first=True`` (the default), tensors are
        ``(..., seq, embed_dim)``, including unbatched 2-D
        ``(seq, embed)`` and batched 3-D
        ``(batch, seq, embed)``. With ``batch_first=False``,
        unbatched 2-D is still ``(seq, embed)``; batched 3-D is
        ``(seq, batch, embed)``, transposed to batch-first for
        the packed kernel and transposed back.

        Always returns a 2-tuple. ``need_weights`` defaults to
        ``False``, and then the second entry is ``None``. If
        ``need_weights`` is ``True``, the second entry is the
        packed per-edge softmax, aligned with
        ``source_index`` / ``target_index``, not MHA's dense
        ``(L, S)`` map. With ``average_attn_weights=True``
        (the default) the head axis is averaged, shape
        ``(..., nnz)``. With ``False``, shape
        ``(..., nnz, num_heads)``. Batch layout follows the
        output (including ``batch_first``). This path does
        not allocate ``(L, S)``.

        ``attn_mask`` must be ``None``. ``is_causal`` must be
        ``False``. ``key_padding_mask`` is ``None`` or a
        boolean mask: ``True`` means ignore that key. Shape
        ``(S,)`` unbatched or ``(N, S)`` batched. Applied in
        packed space; does not allocate ``(n, n)``. Copied
        onto the scores' device; stays boolean. Float
        padding masks raise ``Kpnn2Error``. After padding, a
        query with no remaining keys stays zeros, not NaN.
        """
        if attn_mask is not None:
            raise Kpnn2Error(
                "'attn_mask' must be None; the edgelist is the structural mask."
            )
        if is_causal is not False:
            raise Kpnn2Error("'is_causal' must be False.")
        query_bf, transposed = _layout_to_batch_first(
            query,
            "query",
            self.query_features,
            "query_features",
            self.embed_dim,
            self.batch_first,
        )
        key_bf, _ = _layout_to_batch_first(
            key,
            "key",
            self.key_features,
            "key_features",
            self.embed_dim,
            self.batch_first,
        )
        value_bf, _ = _layout_to_batch_first(
            value,
            "value",
            self.key_features,
            "key_features",
            self.embed_dim,
            self.batch_first,
        )
        if (
            query_bf.shape[:-2] != key_bf.shape[:-2]
            or key_bf.shape[:-2] != value_bf.shape[:-2]
        ):
            raise Kpnn2Error(
                "query, key, and value batch dimensions must match."
            )
        dtype = self.q_proj.weight.dtype
        with torch.autocast(
            device_type=query_bf.device.type,
            enabled=False,
        ):
            query_bf = query_bf.to(dtype=dtype)
            key_bf = key_bf.to(dtype=dtype)
            value_bf = value_bf.to(dtype=dtype)
            q = self._shape_heads(self.q_proj(query_bf))
            k = self._shape_heads(self.k_proj(key_bf))
            v = self._shape_heads(self.v_proj(value_bf))
            participate = _padding_participate(
                key_padding_mask,
                self.source_index,
                tuple(query_bf.shape[:-2]),
                self.key_features,
                query_bf.device,
            )
            if self.chunk_size is None:
                mixed, attn = _packed_attention(
                    q,
                    k,
                    v,
                    self.source_index,
                    self.target_index,
                    self.dropout,
                    self.training,
                    participate,
                )
            else:
                mixed, attn = _packed_attention_chunked(
                    q,
                    k,
                    v,
                    self.source_index,
                    self.target_index,
                    self.dropout,
                    self.training,
                    self.chunk_size,
                    participate,
                )
            output = self.out_proj(
                mixed.reshape(
                    *mixed.shape[:-2],
                    self.embed_dim,
                )
            )
            if transposed:
                output = output.transpose(0, 1)
            if not need_weights:
                return (output, None)
            weights = attn
            if average_attn_weights:
                weights = weights.mean(dim=-1)
            if transposed:
                weights = weights.transpose(0, 1)
            return (output, weights)
