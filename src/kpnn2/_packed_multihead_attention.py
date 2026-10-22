"""
Packed multi-head attention: scores only on live edgelist pairs.
"""

import hashlib
import struct
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ._adjacency_spec import AdjacencySpec, require_adjacency_spec
from ._errors import Kpnn2Error, internal_error
from ._generator import as_generator, run_preserving_default_rng
from ._identity import (
    as_identity,
    check_identity,
    save_identity,
    wiring_mismatch,
)
from ._packed_index import as_packed_pairs, digest_matches
from ._validate import (
    as_bool,
    as_positive_int,
    describe,
    is_integer,
    require_floating,
)

_INDEX_DIGEST_KEY = "index_digest"

_PADDING_LAYOUT = (
    "'key_padding_mask' shape must be (S,) unbatched or (N, S) batched."
)
_PADDING_WIDTH = "'key_padding_mask' last dimension must equal key_features."


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
            "'add_self_loops' requires query_features == key_features. "
            f"Got query_features={query_features} and "
            f"key_features={key_features}."
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
    """
    Return ``dropout`` as a float in [0, 1], the range torch accepts.

    Integers 0 and 1 count; ``bool``, NaN, and infinities do not.
    """
    number: int | float | None = None
    if is_integer(value):
        number = int(value)
    elif isinstance(value, float):
        number = value
    if number is None or not 0 <= number <= 1:
        raise Kpnn2Error(
            f"'dropout' must be a float in [0, 1]. Got {describe(value)}."
        )
    return float(number)


def _layout_to_batch_first(
    tensor: torch.Tensor,
    name: str,
    seq_len: int,
    seq_attr: str,
    embed_dim: int,
    batch_first: bool,
) -> tuple[torch.Tensor, bool]:
    if not isinstance(tensor, torch.Tensor):
        raise Kpnn2Error(
            f"'{name}' must be a torch.Tensor. Got {describe(tensor)}."
        )
    require_floating(
        tensor,
        f"'{name}'",
    )
    if tensor.ndim < 2:
        raise Kpnn2Error(
            f"'{name}' must be 2-dimensional or higher. Got shape "
            f"{tuple(tensor.shape)}."
        )
    if tensor.shape[-1] != embed_dim:
        raise Kpnn2Error(
            f"'{name}' last dimension must equal embed_dim. Got shape "
            f"{tuple(tensor.shape)} with embed_dim={embed_dim}."
        )
    if tensor.ndim == 2:
        if tensor.shape[0] != seq_len:
            raise _sequence_length_error(
                tensor,
                name,
                seq_len,
                seq_attr,
            )
        return tensor, False
    if batch_first:
        if tensor.shape[-2] != seq_len:
            raise _sequence_length_error(
                tensor,
                name,
                seq_len,
                seq_attr,
            )
        return tensor, False
    if tensor.ndim != 3:
        raise Kpnn2Error(
            "With batch_first=False, inputs must be 2-D "
            "(seq, embed) or 3-D (seq, batch, embed). Got "
            f"'{name}' of shape {tuple(tensor.shape)}."
        )
    if tensor.shape[0] != seq_len:
        raise _sequence_length_error(
            tensor,
            name,
            seq_len,
            seq_attr,
        )
    return tensor.transpose(0, 1), True


def _sequence_length_error(
    tensor: torch.Tensor,
    name: str,
    seq_len: int,
    seq_attr: str,
) -> Kpnn2Error:
    return Kpnn2Error(
        f"{name} sequence length must equal {seq_attr}. Got shape "
        f"{tuple(tensor.shape)} with {seq_attr}={seq_len}."
    )


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
        raise Kpnn2Error(
            "'key_padding_mask' must be a boolean tensor. Got "
            f"{describe(key_padding_mask)}."
        )
    if key_padding_mask.is_floating_point():
        raise Kpnn2Error(
            "Float padding masks are not supported; "
            "'key_padding_mask' must be boolean. Got "
            f"{describe(key_padding_mask)}."
        )
    if key_padding_mask.dtype != torch.bool:
        raise Kpnn2Error(
            "'key_padding_mask' must be a boolean tensor. Got "
            f"{describe(key_padding_mask)}."
        )
    if key_padding_mask.ndim < 1:
        raise _padding_shape_error(
            _PADDING_LAYOUT,
            key_padding_mask,
            batch_shape,
            key_features,
        )
    if key_padding_mask.ndim == 1:
        if batch_shape != ():
            raise _padding_shape_error(
                _PADDING_LAYOUT,
                key_padding_mask,
                batch_shape,
                key_features,
            )
        if int(key_padding_mask.shape[0]) != key_features:
            raise _padding_shape_error(
                _PADDING_WIDTH,
                key_padding_mask,
                batch_shape,
                key_features,
            )
    else:
        if int(key_padding_mask.shape[-1]) != key_features:
            raise _padding_shape_error(
                _PADDING_WIDTH,
                key_padding_mask,
                batch_shape,
                key_features,
            )
        if tuple(key_padding_mask.shape[:-1]) != tuple(batch_shape):
            raise _padding_shape_error(
                _PADDING_LAYOUT,
                key_padding_mask,
                batch_shape,
                key_features,
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


def _padding_shape_error(
    problem: str,
    key_padding_mask: torch.Tensor,
    batch_shape: tuple[int, ...],
    key_features: int,
) -> Kpnn2Error:
    expected = (*batch_shape, key_features)
    return Kpnn2Error(
        f"{problem} Expected shape {expected} for this input. Got shape "
        f"{tuple(key_padding_mask.shape)}."
    )


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
    if not is_integer(value) or value < 1:
        raise Kpnn2Error(
            "'chunk_size' must be None or a positive int. Got "
            f"{describe(value)}."
        )
    return int(value)


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


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.float64
    return torch.float32


def _query_degree(
    target_index: torch.Tensor,
    n_query: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Live keys per query as ``(n_query, 1)``, counted on the device.
    """
    degree = torch.bincount(
        target_index,
        minlength=n_query,
    )
    return degree.to(dtype=dtype).unsqueeze(-1)


def _rounding_eps(dtype: torch.dtype) -> float:
    acc_dtype = _accumulation_dtype(dtype)
    return torch.finfo(dtype).eps + torch.finfo(acc_dtype).eps


def _check_softmax_rows(
    row_sum: torch.Tensor,
    no_live_score: torch.Tensor,
    degree: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    # n positive terms sum to within (n - 1) * u of the computed sum
    # (u = eps / 2); with the divide and re-sum, |sum - 1| <= ~tol / 2.
    tol = (degree + 2) * _rounding_eps(dtype)
    # A row may sum to 0 only where no live key scored above the fill:
    # no key, every key padded, or every score -inf (overflow). A query
    # with a finite best score always holds a weight of exp(0) / sum.
    ok = (
        ((row_sum - 1.0).abs() <= tol)
        | (no_live_score & (row_sum.abs() <= tol))
        | ~torch.isfinite(row_sum)
    )
    if not bool(ok.all()):
        raise internal_error(
            "packed softmax weights of a query do not sum to 1"
        )


def _check_softmax_grad_rows(
    grad_sum: torch.Tensor,
    scale: torch.Tensor,
    degree: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    # dot, dot * (1 - sum a), each term, and the re-sum each err by at most
    # one row tolerance of scale; underflow adds <= tiny per rounding.
    bound = (degree + 2) * (
        4 * _rounding_eps(dtype) * scale + 8 * torch.finfo(dtype).tiny
    )
    ok = (grad_sum.abs() <= bound) | ~torch.isfinite(grad_sum)
    if not bool(ok.all()):
        raise internal_error(
            "packed softmax gradient of a query does not sum to 0"
        )


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
        acc_dtype = _accumulation_dtype(query.dtype)
        row_sum = query.new_zeros(
            (*leading, n_query, heads),
            dtype=acc_dtype,
        )
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
            row_sum.scatter_add_(
                -2,
                score_index,
                chunk_attn.to(dtype=acc_dtype),
            )
            start = end
        _check_softmax_rows(
            row_sum,
            max_buf == fill,
            _query_degree(
                target_index,
                n_query,
                acc_dtype,
            ),
            query.dtype,
        )

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
            acc_dtype = _accumulation_dtype(query.dtype)
            prod_scale = query.new_zeros(
                (*leading, n_query, heads),
                dtype=acc_dtype,
            )
            grad_sum = torch.zeros_like(prod_scale)
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
                prod_scale.scatter_add_(
                    -2,
                    score_index,
                    prod.abs().to(dtype=acc_dtype),
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
                grad_sum.scatter_add_(
                    -2,
                    score_index,
                    d_scores.to(dtype=acc_dtype),
                )
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
            # sum_j d_scores_j = dot * (1 - sum_j a_j), so each row is ~0.
            _check_softmax_grad_rows(
                grad_sum,
                prod_scale + dot_buf.to(dtype=acc_dtype).abs(),
                _query_degree(
                    target_index,
                    n_query,
                    acc_dtype,
                ),
                query.dtype,
            )

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


class PackedMultiheadAttention(nn.Module):  # numpydoc ignore=PR06
    """
    Multi-head attention in which each node attends only to its sources.

    Prior knowledge decides who may attend to whom; the data
    decides how much. Every node is one token, a vector of
    ``embed_dim`` numbers that stands for the node inside your
    model. For each node, the layer looks at the nodes with an
    edge into it (its sources), weighs them by how well their
    tokens match its own, and returns a weighted mix of what they
    pass on. The weights are recomputed for every sample: with a
    gene-regulatory prior, a target gene can weigh its regulators
    differently in each cell, depending on their state and its
    own. Two nodes without an edge exchange nothing in this layer.

    Information flows along the edge arrow, from source to target,
    as in ``PackedLinear``. In attention terms, the target asks and
    its sources answer. The target's token gives a *query* (what it
    looks for). Each source's token gives a *key* (what it offers
    to be matched against) and a *value* (what it passes on). Each
    query-key pair, that is, each live edge, gets a score. A
    softmax over the edges into one node turns their scores into
    weights that are non-negative and sum to 1, so the sources
    compete: a higher weight for one leaves less for the others.
    ``num_heads`` such weightings run side by side, each on its own
    slice of the channels, so different heads can weigh the same
    sources differently.

    Compare ``PackedLinear``, which learns one weight per edge,
    fixed after training and the same for every sample. Here
    nothing is learned per edge. The learned parameters are four
    ``embed_dim x embed_dim`` projections shared by every node and
    edge (``q_proj``, ``k_proj``, ``v_proj``, ``out_proj``), so
    their number does not depend on the graph. The graph only
    decides which pairs are scored.

    One call moves information one edge along the graph: a node's
    output depends only on its own token and those of its direct
    sources. Stack layers in your own encoder to reach nodes
    further upstream. This layer is that one building block, not a
    Transformer. You build the tokens (for example a projection of
    each measured value plus a learned embedding per node) and
    write the rest around the layer: normalization, residual
    connections, a feed-forward network, and the readout.

    Masking ``nn.MultiheadAttention`` with the graph computes the
    same mix for every node that has a source. But it scores every
    pair of tokens, an ``(n, n)`` matrix per head and sample (400
    million scores for 20,000 nodes), and then discards the pairs
    without an edge. This layer scores only the live edges, so its
    memory and time grow with their number, ``nnz``, rather than
    with ``n`` squared.

    Build it with ``from_adjacency`` on a ``parse_adjacency`` spec:
    one token per node (per unit, when ``widths=`` gives a node
    several). It reads the indices and sequence lengths off the
    spec, so the query and key sides cannot be swapped by hand. The
    constructor below takes packed indices directly, for pairs that
    come from no spec, such as a bipartite map in which pathway
    tokens (queries) attend to gene tokens (keys).

    Call it like ``nn.MultiheadAttention`` (it is not a subclass):
    ``layer(query, key, value)``, passing the same tokens three
    times for attention within one graph. Two defaults differ from
    MHA: ``batch_first=True`` and ``need_weights=False``. With
    ``need_weights=True`` the layer returns one weight per live
    edge, not a dense ``(L, S)`` matrix.

    Parameters
    ----------
    source_index : torch.Tensor, numpy.ndarray, or sequence of int
        Where each live edge starts. Entry ``i`` is the position in
        ``key`` / ``value`` of the token that edge ``i`` reads
        from, and must satisfy ``0 <= source_index < key_features``.
        1-D integer indices of length ``nnz >= 1``: an integer
        tensor, an integer numpy array, or a sequence of ``int`` or
        numpy integers. A set or a mapping is rejected, since
        entries pair up by position, and so is a complex tensor.
        Copied to an int64 buffer, so later writes to the argument
        do not reach this layer.
    target_index : torch.Tensor, numpy.ndarray, or sequence of int
        Where each live edge ends. Entry ``i`` is the position in
        ``query`` of the token that attends along edge ``i``, and
        must satisfy ``0 <= target_index < query_features``. Same
        length and forms as ``source_index``. The two arrays are
        paired position by position and must not repeat a
        ``(source, target)`` pair. Their shared order is the order
        of the per-edge weights ``forward`` returns; pairs added by
        ``add_self_loops`` follow at the end.
    query_features : int or numpy integer
        How many tokens attend: the sequence length of ``query``.
        Positive integer, stored as ``int``.
    key_features : int or numpy integer
        How many tokens can be attended to: the sequence length of
        ``key`` / ``value``. Positive integer, stored as ``int``.
        Equal to ``query_features`` for attention within one node
        set; smaller or larger for a bipartite query/key map.
    embed_dim : int or numpy integer
        How many numbers describe each token: the width of
        ``query``, ``key``, ``value``, and the output. A wider
        token can carry more about its node, and the four
        projections hold ``4 * embed_dim**2`` weights. Positive
        integer, divisible by ``num_heads``, stored as ``int``.
    num_heads : int or numpy integer
        How many attention patterns run side by side. Each head
        scores the same live edges from its own
        ``embed_dim // num_heads`` channels, so it can weigh a
        node's sources differently from the other heads. The heads'
        mixes are joined and passed through ``out_proj``. Positive
        integer, stored as ``int``.
    dropout : float, default=0.0
        Regularization on the edge weights, in training mode only:
        each weight is zeroed with this probability and the rest
        are scaled by ``1 / (1 - dropout)``, so the model cannot
        lean on any single edge. ``0.0`` disables it. Must be
        finite and in ``[0, 1]``, the range torch accepts; integers
        ``0`` and ``1`` are accepted, ``bool`` is rejected.
    bias : bool, default=True
        Whether the four projections learn an additive bias.
        ``False`` makes them pure linear maps. Must be a ``bool``.
    kdim : int, numpy integer, or None, default=None
        Exists so code written for ``nn.MultiheadAttention`` runs
        unchanged; keys of a different width are not supported.
        Must be ``None`` or an ``int`` or numpy integer equal to
        ``embed_dim`` (``True`` is rejected even when ``embed_dim``
        is 1).
    vdim : int, numpy integer, or None, default=None
        Same as ``kdim``, for the values.
    batch_first : bool, default=True
        Where the batch axis sits. ``True`` (kpnn2 sample-major):
        batched tensors are ``(..., seq, embed_dim)``. ``False``
        (the ``nn.MultiheadAttention`` layout): batched tensors are
        ``(seq, batch, embed_dim)``. Unbatched 2-D
        ``(seq, embed_dim)`` ignores this flag. Must be a ``bool``.
    add_self_loops : bool, default=False
        Let every token also attend to itself. Without a self-loop,
        a node's own token shapes how it weighs its sources
        (through its query) but is not part of the mix, and a node
        with no incoming edge, such as an input node of an
        ``AdjacencySpec``, mixes to zeros. With ``True``, any
        missing ``(i, i)`` pair is added to this layer's index
        buffers, after the given pairs. Existing self-loops are
        kept, not duplicated, and the caller's index objects are
        not mutated. The graph itself is not changed: other layers
        built from the same spec do not see these pairs. Requires
        ``query_features == key_features``. Must be a ``bool``.
    identity : str or None, default=None
        Labels the graph this layer belongs to, so
        ``load_state_dict`` refuses a checkpoint labeled for
        another graph, even one with the same index pattern, such
        as a graph with a renamed node. Typically
        ``spec.fingerprint``. An opaque string, stored in
        ``state_dict`` next to ``index_digest`` as a 1-D CPU
        ``uint8`` tensor of the UTF-8 bytes. ``load_state_dict``
        compares it with the checkpoint's identity:

        - the checkpoint has no identity: it loads, even with
          ``strict=True``
        - it has this layer's identity: it loads
        - it has another identity, or this layer's identity is
          ``None``: ``Kpnn2Error``, and no weights are loaded

        ``None`` means this layer does not claim an identity, so
        it loads only checkpoints saved without one.
    generator : torch.Generator or None, default=None
        A reproducible start without touching the global random
        state: the Xavier-uniform draws for the four projections
        use this generator. ``None`` uses the default torch
        generator and keeps today's path: ``nn.Linear``
        kaiming-init then Xavier on that global stream. When set,
        those ``nn.Linear`` constructors do not advance the global
        stream. Not stored on the module; pass it again to
        ``reset_parameters`` to replay. Do not pass a seed
        integer.
    chunk_size : int, numpy integer, or None, default=None
        Memory control for large graphs; it does not change what
        is computed. ``None`` gathers all live edges at once. A
        positive int processes them in slices of that many edges,
        so training holds less memory, at the cost of time: the
        softmax is still over every live key of a query, but the
        pair gathers are slices, and backward recomputes them. With
        a positive int, forward and backward also check on every
        call that each query's packed softmax sums to 1 (to 0 only
        when no live key scores above ``-inf``: none left after
        padding, or every score overflowed) and that its gradient
        sums to 0, and raise ``AssertionError`` (a kpnn2 bug) if
        not; each check waits for the device once. A numpy integer
        is stored as ``int``. ``bool`` and values other than
        ``None`` or a positive integer raise ``Kpnn2Error``. Not
        stored in ``state_dict`` or ``index_digest``, so a
        checkpoint loads into a layer with another ``chunk_size``.

    Attributes
    ----------
    query_features : int
        Number of tokens that attend (sequence length of
        ``query``).
    key_features : int
        Number of tokens that can be attended to (sequence length
        of ``key`` / ``value``).
    embed_dim : int
        Token width.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Channels per head, ``embed_dim // num_heads``.
    nnz : int
        Number of live edges scored, counting any pair added by
        ``add_self_loops``.
    dropout : float
        Dropout probability on the edge weights.
    batch_first : bool
        Layout flag; default ``True``.
    add_self_loops : bool
        Whether missing self-loops were added at construction.
    source_index : torch.Tensor
        Int64 buffer: the key / value position of each live edge,
        length ``nnz``. **Treat it as read-only:** like any
        PyTorch buffer it can be written to, and doing so rewires
        the layer without reinitializing it. Rebuild from the
        edgelist instead.
    target_index : torch.Tensor
        Int64 buffer: the query position of each live edge, length
        ``nnz``, read-only in the same sense.
    q_proj, k_proj, v_proj, out_proj : nn.Linear
        The four learned ``embed_dim -> embed_dim`` projections
        (query, key, value, and output): separate layers, not a
        fused ``in_proj_weight``.
    identity : str | None
        The constructor ``identity``, or ``None``.
    chunk_size : int | None
        Edge-chunk length, or ``None`` for all live edges at once.

    Raises
    ------
    Kpnn2Error
        At construction, if the indices are empty, not 1-D
        integers, mismatched in length, out of range, or
        duplicated as ``(source, target)`` pairs; if the sizes
        are not positive integers; if ``embed_dim`` is not divisible
        by ``num_heads``; if ``dropout`` is a ``bool``, not a
        number, NaN, or outside ``[0, 1]``; if ``bias``,
        ``batch_first``, or ``add_self_loops`` is not a
        ``bool``; if ``kdim`` / ``vdim`` are neither ``None``
        nor an integer equal to ``embed_dim``; if
        ``add_self_loops`` is set when
        ``query_features != key_features``; if ``identity`` is
        neither a ``str`` nor ``None``; if ``generator`` is
        neither a ``torch.Generator`` nor ``None``; or if
        ``chunk_size`` is neither ``None`` nor a positive
        integer. From
        ``load_state_dict``, when the checkpoint carries an index
        digest or identity that does not match this layer, in
        which case the weights are not loaded. ``forward``
        documents its own rejected arguments.

    See Also
    --------
    PackedLinear : Same packed pairs, but one trained weight per
        edge, the same for every sample, instead of attention
        weights computed per sample.
    MaskedLinear : Dense ``(out, in)`` weight; the default layer
        whenever that square fits.
    AdjacencySpec : Graph-wide packed edges; pass it to
        ``from_adjacency``.
    torch.nn.MultiheadAttention : Dense equivalent, and the
        reference for call shape, heads, and training.

    Notes
    -----
    One forward pass, for one sample and one head with
    ``d = embed_dim // num_heads`` channels:

    1. Project every token: ``q = q_proj(query)``,
       ``k = k_proj(key)``, ``v = v_proj(value)``, each split into
       ``num_heads`` slices of ``d`` channels.
    2. Score each live edge ``e`` from source ``s`` to target
       ``t``: ``score[e] = dot(q[t], k[s]) / sqrt(d)``.
    3. Turn the scores of the edges into each target into weights
       with a softmax over those edges only. Keys masked by
       ``key_padding_mask`` take no part. In training mode,
       ``dropout`` then zeroes weights at random and scales the
       rest by ``1 / (1 - dropout)``.
    4. Mix: ``mixed[t]`` is the sum of ``weight[e] * v[s]`` over
       the edges ``e`` into ``t``.
    5. Join the heads' slices of ``mixed`` back into ``embed_dim``
       channels and apply ``out_proj``.

    A query with no live keys, because it has no incoming edge and
    no self-loop or because ``key_padding_mask`` removed all of
    them, mixes to zeros rather than NaN, so its output is the bias
    of ``out_proj`` (zeros when ``bias=False``). Input nodes of an
    ``AdjacencySpec`` are exactly that case unless
    ``add_self_loops=True``. With it, an input node's only key is
    itself, so its softmax weight is 1 and it passes on its own
    value.

    No ``(n, n)`` or ``(L, S)`` score matrix is ever allocated and
    ``torch.sparse`` is not imported; cost scales with ``nnz``, not
    with ``query_features * key_features``. ``chunk_size=None``
    (the default) gathers all live pairs at once, and training
    keeps those ``(..., nnz, heads, head_dim)`` gathers for
    backward. A positive ``chunk_size`` gathers chunked live pairs
    and rematerializes those ``(..., chunk, heads, head_dim)``
    tensors in backward, so training does not save a full
    ``(..., nnz, heads, head_dim)`` gather. The packed softmax
    ``(..., nnz, heads)`` is still stored.

    ``forward`` documents the accepted layouts, packed
    ``need_weights``, ``attn_mask`` / ``is_causal``, and
    ``key_padding_mask``.

    Index buffers stay integer after ``.half()`` / bfloat16 /
    ``.double()``; projection weights follow the module floating
    dtype like ``nn.Linear``. ``torch.autocast`` is unsupported:
    ``forward`` disables it and casts query, key, and value to
    the parameter dtype. They must be floating-point tensors; an
    integer, ``bool``, or complex input raises instead of being
    cast. Cast the module with ``.to(dtype=...)`` (or ``.half()``
    / ``.double()``) instead.

    ``state_dict`` adds ``index_digest``, a 1-D CPU ``uint8``
    tensor of length 32: the SHA-256 of the packed indices plus
    ``query_features``, ``key_features``, ``embed_dim``, and
    ``num_heads``, so a reshape or a different head split cannot
    collide. It is not a registered buffer, and a missing digest
    is not an error, even with ``strict=True``. ``identity`` is
    stored the same way when the constructor was given one: a
    rename that leaves the packed index pattern unchanged is
    caught when callers pass ``spec.fingerprint``. ``chunk_size``
    is not part of ``state_dict`` or ``index_digest``.
    ``copy.deepcopy`` works.

    Examples
    --------
    An input ``x`` feeds a two-node cycle between ``a`` and
    ``b``, which feeds an output ``y``. All four nodes form one
    four-token sequence:

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
    >>> attn = kpnn2.PackedMultiheadAttention.from_adjacency(
    ...     spec,
    ...     embed_dim=8,
    ...     num_heads=2,
    ...     add_self_loops=True,
    ...     identity=spec.fingerprint,
    ... )
    >>> attn.nnz, attn.head_dim
    (8, 4)
    >>> tokens = torch.randn(2, spec.state_dim, 8)
    >>> output, weights = attn(tokens, tokens, tokens)
    >>> tuple(output.shape)
    (2, 4, 8)
    >>> weights is None
    True

    With ``need_weights=True`` each sample gets one weight per live
    edge: the four graph edges, then the four added self-loops.

    >>> _, packed = attn(
    ...     tokens,
    ...     tokens,
    ...     tokens,
    ...     need_weights=True,
    ... )
    >>> tuple(packed.shape)
    (2, 8)
    >>> for s, t in zip(
    ...     attn.source_index.tolist(),
    ...     attn.target_index.tolist(),
    ... ):
    ...     print(spec.nodes[s], "->", spec.nodes[t])
    a -> b
    a -> y
    b -> a
    x -> a
    a -> a
    b -> b
    x -> x
    y -> y
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
        query_features = as_positive_int(
            query_features,
            "query_features",
        )
        key_features = as_positive_int(
            key_features,
            "key_features",
        )
        embed_dim = as_positive_int(
            embed_dim,
            "embed_dim",
        )
        num_heads = as_positive_int(
            num_heads,
            "num_heads",
        )
        if embed_dim % num_heads != 0:
            raise Kpnn2Error(
                "'embed_dim' must be divisible by 'num_heads'. Got "
                f"embed_dim={embed_dim} and num_heads={num_heads}."
            )
        dropout = _dropout_value(dropout)
        bias = as_bool(
            bias,
            "bias",
        )
        if kdim is not None and not (
            is_integer(kdim) and int(kdim) == embed_dim
        ):
            raise Kpnn2Error(
                "'kdim' must be None or equal to embed_dim. Got "
                f"{describe(kdim)} with embed_dim={embed_dim}."
            )
        if vdim is not None and not (
            is_integer(vdim) and int(vdim) == embed_dim
        ):
            raise Kpnn2Error(
                "'vdim' must be None or equal to embed_dim. Got "
                f"{describe(vdim)} with embed_dim={embed_dim}."
            )
        batch_first = as_bool(
            batch_first,
            "batch_first",
        )
        add_self_loops = as_bool(
            add_self_loops,
            "add_self_loops",
        )
        source, target = as_packed_pairs(
            source_index,
            target_index,
            source_bound=key_features,
            target_bound=query_features,
            source_bound_name="key_features",
            target_bound_name="query_features",
            owner="PackedMultiheadAttention",
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

    @classmethod
    def from_adjacency(  # numpydoc ignore=PR06
        cls,
        spec: AdjacencySpec,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        add_self_loops: bool = False,
        identity: str | None = None,
        generator: torch.Generator | None = None,
        chunk_size: int | None = None,
    ) -> "PackedMultiheadAttention":
        """
        Build attention over the nodes of an ``AdjacencySpec``.

        This is the recommended way to build this layer on an
        ``AdjacencySpec``: one token per unit of the state vector,
        and the query at each edge's target attends to the key at
        its source. It reads the packed indices off ``spec`` and
        uses ``spec.state_dim`` as both sequence lengths, so the
        query and key sides cannot be swapped by hand. On this
        square layout a hand-written swap is always silent: the
        constructor accepts it and every node attends along
        reversed edges. The result is the layer the constructor
        builds from ``spec.source_index``, ``spec.target_index``,
        ``spec.state_dim``, ``spec.state_dim``, ``embed_dim``, and
        ``num_heads``, with the same init draws. It keeps no
        reference to ``spec``. Options after ``num_heads`` are
        keyword-only; ``kdim`` and ``vdim``, which can only equal
        ``embed_dim``, are left out.

        Parameters
        ----------
        spec : AdjacencySpec
            Spec from ``parse_adjacency``.
        embed_dim : int or numpy integer
            Same as the ``PackedMultiheadAttention`` constructor.
        num_heads : int or numpy integer
            Same as the ``PackedMultiheadAttention`` constructor.
        dropout : float, default=0.0
            Same as the ``PackedMultiheadAttention`` constructor.
        bias : bool, default=True
            Same as the ``PackedMultiheadAttention`` constructor.
        batch_first : bool, default=True
            Same as the ``PackedMultiheadAttention`` constructor.
        add_self_loops : bool, default=False
            Same as the ``PackedMultiheadAttention`` constructor.
        identity : str or None, default=None
            Same as the ``PackedMultiheadAttention`` constructor.
            Not filled in from ``spec``: pass
            ``spec.fingerprint`` yourself.
        generator : torch.Generator or None, default=None
            Same as the ``PackedMultiheadAttention`` constructor.
        chunk_size : int, numpy integer, or None, default=None
            Same as the ``PackedMultiheadAttention`` constructor.

        Returns
        -------
        PackedMultiheadAttention
            New layer with ``query_features == key_features ==
            spec.state_dim``.

        Raises
        ------
        Kpnn2Error
            If ``spec`` is not an ``AdjacencySpec`` (a
            ``LayeredSpec`` or a ``Hop`` included), or if the
            constructor rejects any other argument.

        See Also
        --------
        PackedLinear.from_adjacency : One trainable scalar per
            edge on the same packed pairs.

        Examples
        --------
        >>> import pandas as pd
        >>> import kpnn2
        >>> spec = kpnn2.parse_adjacency(
        ...     pd.DataFrame(
        ...         {
        ...             "source": ["x", "a", "b", "a"],
        ...             "target": ["a", "b", "a", "y"],
        ...         }
        ...     )
        ... )
        >>> attn = kpnn2.PackedMultiheadAttention.from_adjacency(
        ...     spec,
        ...     embed_dim=8,
        ...     num_heads=2,
        ...     identity=spec.fingerprint,
        ... )
        >>> attn.query_features, attn.key_features, attn.nnz
        (4, 4, 4)
        """
        require_adjacency_spec(
            spec,
            "Build one with parse_adjacency().",
        )
        return cls(
            spec.source_index,
            spec.target_index,
            spec.state_dim,
            spec.state_dim,
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            add_self_loops=add_self_loops,
            identity=identity,
            generator=generator,
            chunk_size=chunk_size,
        )

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
            if not digest_matches(
                saved,
                current,
            ):
                raise wiring_mismatch(
                    "The checkpoint indices do not match this layer.",
                    prefix,
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
            raise Kpnn2Error(
                "Last dimension must equal embed_dim. Got last dimension "
                f"{embed_dim} with embed_dim={self.embed_dim}."
            )
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
        Let each query attend to its live keys and mix their values.

        For attention within one graph, pass the same tokens as
        ``query``, ``key``, and ``value``: every node then attends
        to its sources. Pass different tensors when queries and
        keys are different token sets, as in a bipartite map built
        with the constructor. The arguments are those of
        ``nn.MultiheadAttention.forward``, so an encoder written
        for it can call this layer; the class docstring lists the
        differences.

        With ``batch_first=True`` (the default), ``query``,
        ``key``, and ``value`` are ``(..., seq, embed_dim)``:
        unbatched 2-D ``(seq, embed_dim)``, batched 3-D
        ``(batch, seq, embed_dim)``, or more leading axes. With
        ``batch_first=False``, unbatched 2-D is still
        ``(seq, embed_dim)``; batched 3-D is
        ``(seq, batch, embed_dim)``, transposed to batch-first for
        the packed kernel and transposed back. The batch axes of
        all three must match.

        Parameters
        ----------
        query : torch.Tensor
            The tokens that attend: ``query_features`` of them
            along the sequence axis, each ``embed_dim`` wide. Must
            be floating-point; an integer, ``bool``, or complex
            tensor raises ``Kpnn2Error`` rather than being cast to
            the parameter dtype.
        key : torch.Tensor
            The tokens matched against the queries:
            ``key_features`` of them along the sequence axis, each
            ``embed_dim`` wide. Floating-point, like ``query``.
            Required; it is not defaulted to ``query``.
        value : torch.Tensor
            What each key position passes on, in the shape of
            ``key`` and usually the same tensor. Floating-point,
            like ``query``.
        key_padding_mask : torch.Tensor or None, default=None
            Keys to leave out of a given sample, for example a
            feature that sample did not measure: ``True`` means
            ignore that key. Its edges drop out of the softmax, so
            each query's remaining sources share the whole weight.
            A boolean tensor of shape ``(S,)`` unbatched or
            ``(N, S)`` batched, where ``S = key_features``; with
            more leading axes, the same ones as the batch-first
            ``query``. The batch axes come first even when
            ``batch_first=False``. Applied in packed space, without
            allocating ``(n, n)``. Copied onto the scores' device
            and kept boolean. A float mask raises ``Kpnn2Error``.
        need_weights : bool, default=False
            Also return the weight of every live edge, for example
            to draw edge thickness on the prior graph. Must be a
            ``bool``.
        attn_mask : torch.Tensor or None, default=None
            Must be ``None``: the edgelist is the mask. Kept so that
            calls written for ``nn.MultiheadAttention`` still run.
        average_attn_weights : bool, default=True
            With ``need_weights=True``: ``True`` averages the
            weights over the heads, ``False`` keeps one per head.
            Must be a ``bool``.
        is_causal : bool, default=False
            Must be ``False``. Kept so that calls written for
            ``nn.MultiheadAttention`` still run.

        Returns
        -------
        output : torch.Tensor
            The updated tokens, one per query position, in the
            shape and layout of ``query``. A query with no live
            keys left mixes to zeros, not NaN, before
            ``out_proj``.
        weights : torch.Tensor or None
            ``None`` unless ``need_weights=True``. Then the
            softmax weight of every live edge, in the order of
            ``source_index`` / ``target_index``, not
            ``nn.MultiheadAttention``'s dense ``(L, S)`` map; this
            path does not allocate ``(L, S)``. Shape
            ``(..., nnz)`` averaged over heads, or
            ``(..., nnz, num_heads)`` per head, with the batch axes
            placed as in ``output``: ``(nnz, batch)`` or
            ``(nnz, batch, num_heads)`` when ``batch_first=False``
            and the input is batched. The weights into one query
            sum to 1, or to 0 when it has no live key. In training
            mode with ``dropout > 0`` they are the weights after
            dropout, as the mix used them; call ``eval()`` first to
            read the plain softmax. ``nnz`` counts the pairs added
            by ``add_self_loops``, so zip the weights with this
            layer's ``source_index`` / ``target_index``, not with
            ``to_edgelist()``.

        Raises
        ------
        Kpnn2Error
            If ``query``, ``key``, or ``value`` is not a
            floating-point tensor of at least 2 dimensions with
            last axis ``embed_dim`` and the sequence length above;
            if their batch axes differ; if a batched input is not
            3-D under ``batch_first=False``; if
            ``key_padding_mask`` is not a boolean tensor of the
            shape above; if ``need_weights`` or
            ``average_attn_weights`` is not a ``bool``; if
            ``attn_mask`` is not ``None``; or if ``is_causal`` is
            not ``False``.
        """
        need_weights = as_bool(
            need_weights,
            "need_weights",
        )
        average_attn_weights = as_bool(
            average_attn_weights,
            "average_attn_weights",
        )
        if attn_mask is not None:
            raise Kpnn2Error(
                "'attn_mask' must be None; the edgelist is the structural "
                f"mask. Got {describe(attn_mask)}."
            )
        if is_causal is not False:
            raise Kpnn2Error(
                f"'is_causal' must be False. Got {describe(is_causal)}."
            )
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
                "query, key, and value batch dimensions must match. Got "
                f"shapes {tuple(query.shape)}, {tuple(key.shape)}, and "
                f"{tuple(value.shape)}."
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
