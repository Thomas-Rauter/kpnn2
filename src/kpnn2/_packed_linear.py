"""
Packed linear layer: one trainable scalar per live edge.
"""

import hashlib
import math
import struct
from typing import Any

import torch
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

    if isinstance(value, (str, bytes)):
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


def _index_digest(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """
    SHA-256 of packed indices plus layer sizes.

    Payload is the int64 C-contiguous bytes of ``source_index``,
    then ``target_index``, then ``out_features`` and
    ``in_features`` as little-endian int64 so a reshape cannot
    collide.
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
        "<qq",
        int(out_features),
        int(in_features),
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


def _positive_int(
    value: object,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Kpnn2Error(f"'{name}' must be a positive int.")
    return value


class PackedLinear(nn.Module):
    """
    Affine map with one trainable scalar per live edge.

    Packed 1-D weights, one per live edge; not ``torch.sparse``;
    forward is ``index_add``. This is not a subclass of
    ``Linear`` and not a full model. Dead edges are omitted: there
    is no dense ``(out_features, in_features)`` parameter and
    forward never allocates that square.

    Parameters
    ----------
    source_index : torch.Tensor or sequence of int
        1-D integer indices of length ``nnz >= 1``. Entry ``i``
        is the input column of live edge ``i``. Copied to an
        int64 buffer.
    target_index : torch.Tensor or sequence of int
        1-D integer indices of the same length. Entry ``i`` is
        the output row of live edge ``i``. Copied to an int64
        buffer.
    out_features : int
        Width of the output axis. Must be a positive int.
    in_features : int
        Width of the input axis. Must be a positive int.
    bias : bool, default=True
        If ``True``, learn a bias of shape ``(out_features,)``.
        If ``False``, there is no bias.

    Attributes
    ----------
    in_features : int
        Number of input columns.
    out_features : int
        Number of output columns.
    nnz : int
        Number of live edges, ``weight.shape[0]``.
    weight : nn.Parameter
        Trainable packed weights of shape ``(nnz,)``. One scalar
        per live edge, in the same order as the index buffers.
        This is an ordinary parameter; there is no
        ``parametrize`` and no dense ``(out, in)`` tensor.
    source_index : torch.Tensor
        Int64 buffer of input columns, length ``nnz``.
    target_index : torch.Tensor
        Int64 buffer of output rows, length ``nnz``.
    bias : nn.Parameter | None
        Trainable bias, or ``None`` when constructed with
        ``bias=False``.

    Raises
    ------
    Kpnn2Error
        If the indices are empty, not 1-D integers, mismatched
        in length, out of range, duplicated as
        ``(source, target)`` pairs, or if ``out_features`` /
        ``in_features`` are not positive ints.

    Notes
    -----
    Construct from packed indices, not from a dense mask and not
    from an ``AdjacencySpec``::

        PackedLinear(
            spec.source_index,
            spec.target_index,
            len(spec.nodes),
            len(spec.nodes),
        )

    ``0 <= source_index < in_features`` and
    ``0 <= target_index < out_features``. Duplicate pairs are
    rejected. Index buffers stay integer after ``.half()`` /
    bfloat16 / ``.double()``; ``weight`` and ``bias`` follow
    the module floating dtype like ``nn.Linear``.

    Forward gathers ``x[..., source_index]``, multiplies by
    ``weight``, and ``index_add``s into a zeros tensor of shape
    ``(..., out_features)``. It does not scatter into a dense
    ``(out, in)`` matrix and does not import ``torch.sparse``.

    ``reset_parameters`` uses per-row packed degree as
    ``fan_in``, counted with ``bincount`` over ``target_index``.
    Each live edge into row ``j`` is drawn uniformly from
    ``[-1 / sqrt(fan_in), 1 / sqrt(fan_in)]``. If
    ``fan_in == 0``, that row has no packed weights and
    ``bias[j]`` stays 0. Input nodes on an ``AdjacencySpec``
    have in-degree 0, so they have no packed incoming edges;
    this layer does not invent identity connections.

    ``state_dict`` keys are ``weight``, optional ``bias``,
    ``source_index``, ``target_index``, and ``index_digest``.
    ``index_digest`` is a 1-D CPU ``uint8`` tensor of length
    32: the SHA-256 of the live index buffers' int64
    C-contiguous bytes plus ``out_features`` and
    ``in_features`` as fixed-width integers, not a registered
    persistent buffer. ``load_state_dict`` raises ``Kpnn2Error``
    when a present digest does not match this layer, and does
    not load the weights. A missing digest is not an error,
    even with ``strict=True``. ``copy.deepcopy`` works.

    Typical construction for a large ``AdjacencySpec`` is this
    class; ``MaskedLinear(spec.to_mask())`` remains the dense
    path for small graphs.

    Examples
    --------
    Two crossed edges on a 2-wide state, no bias:

    >>> import torch
    >>> import kpnn2
    >>> layer = kpnn2.PackedLinear(
    ...     [0, 1],
    ...     [1, 0],
    ...     2,
    ...     2,
    ...     bias=False,
    ... )
    >>> layer.in_features, layer.out_features, layer.nnz
    (2, 2, 2)
    >>> x = torch.ones(3, 2)
    >>> y = layer(x)
    >>> tuple(y.shape)
    (3, 2)
    """

    source_index: torch.Tensor
    target_index: torch.Tensor
    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        source_index: object,
        target_index: object,
        out_features: int,
        in_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = _positive_int(
            out_features,
            "out_features",
        )
        in_features = _positive_int(
            in_features,
            "in_features",
        )
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
        if torch.any(source < 0) or torch.any(source >= in_features):
            raise Kpnn2Error(
                "'source_index' entries must satisfy "
                "0 <= source_index < in_features."
            )
        if torch.any(target < 0) or torch.any(target >= out_features):
            raise Kpnn2Error(
                "'target_index' entries must satisfy "
                "0 <= target_index < out_features."
            )
        pairs = list(
            zip(
                source.tolist(),
                target.tolist(),
            )
        )
        if len(set(pairs)) != len(pairs):
            raise Kpnn2Error(
                "PackedLinear indices contain duplicate "
                "(source, target) pair(s)."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.nnz = nnz
        self.register_buffer(
            "source_index",
            source,
        )
        self.register_buffer(
            "target_index",
            target,
        )
        self.weight = nn.Parameter(torch.empty(nnz))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter(
                "bias",
                None,
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize from per-row packed degree, not full width.

        For output row ``j``, ``fan_in`` is the number of packed
        edges with ``target_index == j`` (``bincount``,
        ``minlength=out_features``). Each live edge into that
        row, and ``bias[j]`` if present, is drawn uniformly from
        ``[-1 / sqrt(fan_in), 1 / sqrt(fan_in)]``. If
        ``fan_in == 0``, that row has no packed weights and
        ``bias[j]`` stays 0.
        """
        with torch.no_grad():
            self.weight.zero_()
            if self.bias is not None:
                self.bias.zero_()
            degrees = torch.bincount(
                self.target_index,
                minlength=self.out_features,
            ).tolist()
            target_list = self.target_index.tolist()
            row_edges: list[list[int]] = [[] for _ in range(self.out_features)]
            for index, row in enumerate(target_list):
                row_edges[row].append(index)
            for row, degree in enumerate(degrees):
                if degree <= 0:
                    continue
                bound = 1.0 / math.sqrt(degree)
                for index in row_edges[row]:
                    nn.init.uniform_(
                        self.weight[index : index + 1],
                        -bound,
                        bound,
                    )
                if self.bias is not None:
                    nn.init.uniform_(
                        self.bias[row : row + 1],
                        -bound,
                        bound,
                    )

    def extra_repr(self) -> str:
        """
        Sizes, live-edge count, and bias flag.
        """
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"nnz={self.nnz}, "
            f"bias={self.bias is not None}"
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
            self.out_features,
            self.in_features,
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
                self.out_features,
                self.in_features,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gather live inputs, scale by packed weights, ``index_add``.

        ``contrib = x[..., source_index] * weight``, then
        ``index_add`` into zeros of shape
        ``(..., out_features)``. Adds ``bias`` when present.
        Packed 1-D weights, one per live edge; not
        ``torch.sparse``; forward is ``index_add``.
        """
        contrib = x[..., self.source_index] * self.weight
        y = torch.zeros(
            (*x.shape[:-1], self.out_features),
            dtype=x.dtype,
            device=x.device,
        )
        y.index_add_(
            -1,
            self.target_index,
            contrib,
        )
        if self.bias is not None:
            y = y + self.bias
        return y
