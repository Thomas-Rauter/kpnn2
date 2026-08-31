"""
Packed multi-head attention: scores only on live edgelist pairs.
"""

from collections.abc import Iterable
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from kpnn2 import Kpnn2Error


def _copy_index(
    value: object,
    name: str,
) -> torch.Tensor:
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


def _expand_index(
    index: torch.Tensor,
    like: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    view_shape = [1] * like.ndim
    view_shape[dim] = -1
    return index.view(*view_shape).expand_as(like)


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


def _packed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """
    Softmax over live keys of each query; never an ``(n, n)`` score
    matrix. ``query`` / ``key`` / ``value`` are
    ``(..., n, heads, head_dim)``.
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
    max_buf.scatter_reduce_(
        -2,
        score_index,
        scores,
        reduce="amax",
        include_self=True,
    )
    gathered_max = max_buf.gather(
        -2,
        score_index,
    )
    exp_scores = torch.exp(scores - gathered_max)
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
    ``layer(query, key, value)``); not a subclass. Not a full
    Transformer block. Scores exist only for live
    ``(source, target)`` pairs: query = target, key = source.
    Forward never allocates an ``(n, n)`` score matrix and does
    not import ``torch.sparse``.

    Node names are domain-agnostic. Build indices from
    ``parse_adjacency`` (or any packed pair list). This module
    does not take an ``AdjacencySpec``.

    Parameters
    ----------
    source_index : torch.Tensor or sequence of int
        Key / value positions of live edges (information source).
    target_index : torch.Tensor or sequence of int
        Query positions of live edges (information target).
    query_features : int
        Sequence length of ``query`` (``n`` on an
        ``AdjacencySpec`` self-attention).
    key_features : int
        Sequence length of ``key`` / ``value``.
    embed_dim : int
        Model width; must be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads.
    dropout : float, default=0.0
        Dropout on packed attention weights.
    bias : bool, default=True
        Bias on Q/K/V and output projections.
    add_self_loops : bool, default=False
        If True, OR in missing ``(i, i)`` pairs when
        ``query_features == key_features``. The spec is not
        modified; isolated queries otherwise output zeros.

    Notes
    -----
    ``batch_first`` is True: tensors are ``(..., seq, embed_dim)``.
    That matches kpnn2 sample-major layouts, not the default
    ``nn.MultiheadAttention`` sequence-major layout.

    Queries with no packed keys stay zeros after the mix (no NaN
    softmax). That differs from dense masked MHA, where an
    all-``-inf`` row is undefined.

    Typical construction::

        spec = kpnn2.parse_adjacency(edgelist)
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
        if not isinstance(dropout, float) or dropout < 0.0:
            raise Kpnn2Error("'dropout' must be a float >= 0.")
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
        source, target = _maybe_self_loops(
            source,
            target,
            query_features,
            key_features,
            add_self_loops,
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

        self.query_features = query_features
        self.key_features = key_features
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.nnz = nnz
        self.dropout = dropout
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
        return (
            f"query_features={self.query_features}, "
            f"key_features={self.key_features}, "
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"nnz={self.nnz}, "
            f"dropout={self.dropout}, "
            f"add_self_loops={self.add_self_loops}"
        )

    def _shape_heads(
        self,
        projected: torch.Tensor,
    ) -> torch.Tensor:
        *leading, seq, embed_dim = projected.shape
        if embed_dim != self.embed_dim:
            raise Kpnn2Error("Last dimension must equal embed_dim.")
        shaped = projected.view(
            *leading,
            seq,
            self.num_heads,
            self.head_dim,
        )
        return shaped

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Prior-gated multi-head attention.

        ``query`` shape ``(..., query_features, embed_dim)``.
        If ``key`` is omitted, ``key`` and ``value`` default to
        ``query`` (self-attention). Output has the same shape as
        ``query``.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if query.shape[-2] != self.query_features:
            raise Kpnn2Error("query sequence length must equal query_features.")
        if key.shape[-2] != self.key_features:
            raise Kpnn2Error("key sequence length must equal key_features.")
        if value.shape[-2] != self.key_features:
            raise Kpnn2Error("value sequence length must equal key_features.")
        q = self._shape_heads(self.q_proj(query))
        k = self._shape_heads(self.k_proj(key))
        v = self._shape_heads(self.v_proj(value))
        mixed = _packed_attention(
            q,
            k,
            v,
            self.source_index,
            self.target_index,
            self.dropout,
            self.training,
        )
        *leading, seq, heads, head_dim = mixed.shape
        concat = mixed.reshape(
            *leading,
            seq,
            self.embed_dim,
        )
        return cast(
            torch.Tensor,
            self.out_proj(concat),
        )
