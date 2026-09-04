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
) -> torch.Tensor:
    """
    Softmax over live keys of each query; never an ``(n, n)`` score
    matrix. ``query`` / ``key`` / ``value`` are
    ``(..., seq, heads, head_dim)``.
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
    return out


class PackedMultiheadAttention(nn.Module):
    """
    Multi-head attention whose allowed pairs are packed edges.

    Same job as ``torch.nn.MultiheadAttention`` (call as
    ``layer(query, key, value)``); not a subclass. Not a
    Transformer block and not a full model. Scores exist only
    for live ``(source, target)`` pairs: query = target, key /
    value = source. Forward never allocates an ``(n, n)`` or
    ``(L, S)`` score matrix and does not import
    ``torch.sparse``. This module does not take an
    ``AdjacencySpec``; pass packed indices (typically from
    ``parse_adjacency``).

    Parameters
    ----------
    source_index : torch.Tensor or sequence of int
        1-D integer indices of length ``nnz >= 1``. Entry ``i``
        is the key / value position of live edge ``i``. Copied
        to an int64 buffer.
    target_index : torch.Tensor or sequence of int
        1-D integer indices of the same length. Entry ``i`` is
        the query position of live edge ``i``. Copied to an
        int64 buffer.
    query_features : int
        Sequence length of ``query``. Must be a positive int.
    key_features : int
        Sequence length of ``key`` / ``value``. Must be a
        positive int.
    embed_dim : int
        Model width; must be a positive int divisible by
        ``num_heads``.
    num_heads : int
        Number of attention heads. Must be a positive int.
    dropout : float, default=0.0
        Dropout on packed attention weights. Stored as
        ``float >= 0``. Integer ``0`` is accepted. ``bool``
        and negative values are rejected.
    bias : bool, default=True
        Bias on the four ``nn.Linear`` projections.
    kdim : int or None, default=None
        Key embed width. Must be ``None`` or equal to
        ``embed_dim``. Other values raise ``Kpnn2Error``.
    vdim : int or None, default=None
        Value embed width. Must be ``None`` or equal to
        ``embed_dim``. Other values raise ``Kpnn2Error``.
    batch_first : bool, default=True
        If ``True``, batched tensors are
        ``(..., seq, embed_dim)``. If ``False``, batched
        tensors use the ``nn.MultiheadAttention`` layout
        ``(seq, batch, embed_dim)``. Unbatched 2-D
        ``(seq, embed)`` ignores this flag.
    add_self_loops : bool, default=False
        If ``True``, OR missing ``(i, i)`` pairs into the
        module buffers when ``query_features == key_features``.
        Caller index objects are not mutated. Existing
        self-loops are kept, not duplicated. If
        ``query_features != key_features``, raise
        ``Kpnn2Error``.

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
        Number of live edges after ``add_self_loops``.
    dropout : float
        Dropout probability on packed attention weights.
    batch_first : bool
        Layout flag; default ``True``.
    add_self_loops : bool
        Whether missing self-loops were OR-ed at init.
    source_index : torch.Tensor
        Int64 buffer of key / value positions, length ``nnz``.
    target_index : torch.Tensor
        Int64 buffer of query positions, length ``nnz``.
    q_proj, k_proj, v_proj, out_proj : nn.Linear
        Separate ``embed_dim → embed_dim`` projections. Not a
        fused ``in_proj_weight``.

    Raises
    ------
    Kpnn2Error
        If the indices are empty, not 1-D integers, mismatched
        in length, out of range, duplicated as
        ``(source, target)`` pairs, if sizes are not positive
        ints, if ``embed_dim`` is not divisible by
        ``num_heads``, if ``dropout`` is a ``bool`` or
        negative, if ``kdim`` / ``vdim`` are not ``None`` or
        ``embed_dim``, or if ``add_self_loops`` is set when
        ``query_features != key_features``.

    Notes
    -----
    ``batch_first`` defaults to ``True`` (kpnn2 sample-major).
    That differs from ``nn.MultiheadAttention``, whose default
    is sequence-major.

    ``need_weights`` defaults to ``False`` (MHA defaults
    ``True``). If ``need_weights`` is ``True``, ``forward``
    raises ``Kpnn2Error``: returning weights would allocate a
    dense ``(L, S)`` matrix. ``average_attn_weights`` is kept
    for call-site drop-in and has no effect while that raise
    stands.

    ``query``, ``key``, and ``value`` are required; ``key`` is
    not defaulted to ``query``. ``attn_mask`` must be ``None``
    (the edgelist is the structural mask). ``is_causal`` must
    be ``False``.

    Queries with no packed keys (and none added by
    self-loops) stay zeros after the mix, then still go
    through ``out_proj``. There is no NaN softmax. After
    ``key_padding_mask``, a query with no remaining keys also
    stays zeros. A boolean padding mask is copied onto the
    scores' device; it stays boolean. Float padding masks
    still raise ``Kpnn2Error``.

    Index buffers stay integer after ``.half()`` / bfloat16 /
    ``.double()``. ``state_dict`` includes ``index_digest``, a
    1-D CPU ``uint8`` tensor of length 32: SHA-256 of the
    packed indices plus ``query_features``, ``key_features``,
    ``embed_dim``, and ``num_heads``. It is not a registered
    persistent buffer. ``load_state_dict`` raises
    ``Kpnn2Error`` when a present digest does not match this
    layer, and does not load the weights. A missing digest is
    not an error, even with ``strict=True``.

    Typical construction::

        spec = parse_adjacency(edgelist)
        n = len(spec.nodes)
        attn = PackedMultiheadAttention(
            spec.source_index,
            spec.target_index,
            n,
            n,
            embed_dim,
            num_heads,
        )
    """

    source_index: torch.Tensor
    target_index: torch.Tensor

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
        self.register_buffer(
            "source_index",
            source,
        )
        self.register_buffer(
            "target_index",
            target,
        )
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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Xavier-uniform projections, like ``nn.MultiheadAttention``.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def extra_repr(self) -> str:
        """
        Sizes, live-edge count, dropout, and layout flags.
        """
        return (
            f"query_features={self.query_features}, "
            f"key_features={self.key_features}, "
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"nnz={self.nnz}, "
            f"dropout={self.dropout}, "
            f"batch_first={self.batch_first}, "
            f"add_self_loops={self.add_self_loops}"
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
        ``False``. If ``need_weights`` is ``True``, raise
        ``Kpnn2Error`` (a packed layer must not allocate a
        dense ``(L, S)`` weight matrix).
        ``average_attn_weights`` has no effect while that raise
        stands.

        ``attn_mask`` must be ``None``. ``is_causal`` must be
        ``False``. ``key_padding_mask`` is ``None`` or a
        boolean mask: ``True`` means ignore that key. Shape
        ``(S,)`` unbatched or ``(N, S)`` batched. Applied in
        packed space; does not allocate ``(n, n)``. Copied
        onto the scores' device; stays boolean. Float
        padding masks raise ``Kpnn2Error``. After padding, a
        query with no remaining keys stays zeros, not NaN.
        """
        if need_weights:
            raise Kpnn2Error(
                "PackedMultiheadAttention cannot return "
                "attention weights: that would allocate a dense "
                "(L, S) matrix. 'need_weights' must be False."
            )
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
        mixed = _packed_attention(
            q,
            k,
            v,
            self.source_index,
            self.target_index,
            self.dropout,
            self.training,
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
        return (output, None)
