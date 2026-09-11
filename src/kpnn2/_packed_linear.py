"""
Packed linear layer: one trainable scalar per live edge.
"""

import copy
import hashlib
import math
import struct
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from ._constraint import as_constraint, check_constraint_shape
from ._errors import Kpnn2Error
from ._generator import as_generator
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

    An ``nn.Linear``-style layer (call ``layer(x)``; not a
    subclass, not a full model) for a graph laid out by
    ``parse_layered`` or ``parse_adjacency``. On a hop it stores
    one weight per live edge of that hop instead of the dense
    ``(out, in)`` rectangle ``MaskedLinear(hop.to_mask())``
    would. On an ``AdjacencySpec`` it stores one weight per
    graph edge instead of an ``(n, n)`` square. Reach for it
    when that rectangle or square strains RAM. On an
    ``AdjacencySpec``, input nodes have no incoming edges, so
    writing inputs into the state each step is the caller's job.

    Parameters
    ----------
    source_index : torch.Tensor or sequence of int
        1-D integer indices of length ``nnz >= 1``. Entry ``i``
        is the input column of live edge ``i``, and must satisfy
        ``0 <= source_index < in_features``. Copied to an int64
        buffer, so later writes to the argument do not reach this
        layer.
    target_index : torch.Tensor or sequence of int
        1-D integer indices of the same length. Entry ``i`` is
        the output row of live edge ``i``, and must satisfy
        ``0 <= target_index < out_features``. The two arrays are
        paired position by position, must not repeat a
        ``(source, target)`` pair, and their shared order is also
        the order of ``weight``.
    out_features : int
        Width of the output axis. Must be a positive int. Note
        the order: ``out_features`` comes before ``in_features``,
        as in the ``(out, in)`` shape of a dense weight, not in
        the ``nn.Linear`` argument order.
    in_features : int
        Width of the input axis. Must be a positive int.
    bias : bool, default=True
        If ``True``, learn a bias of shape ``(out_features,)``.
        If ``False``, there is no bias.
    identity : str or None, default=None
        Opaque checkpoint identity, typically
        ``spec.fingerprint``. Stored in ``state_dict`` next to
        ``index_digest`` as a 1-D CPU ``uint8`` tensor of the
        UTF-8 bytes. ``load_state_dict`` raises ``Kpnn2Error``
        when a present identity does not match this layer, and
        does not load the weights. A missing identity is not an
        error, even with ``strict=True``. ``None`` means this
        layer does not claim an identity.
    constraint : torch.nn.Module or None, default=None
        Optional per-entry map on the packed ``weight``, applied
        in ``forward``. ``nn.Softplus()`` is the textbook
        non-negative edge reparametrization. The module must
        return a tensor of shape ``(nnz,)``. There are no
        absent edges here, so this map cannot resurrect a
        blocked cell. ``reset_parameters`` writes the
        unconstrained packed tensor; it does not invert this
        map. ``MaskedLinear`` takes the same argument.
        Mixed per-edge signs and frozen slots belong in this
        module, not in parse columns. A hard freeze is
        ``torch.where`` replacing those slots in ``forward``;
        a gradient hook that zeroes ``grad[i]`` is not a
        freeze (AdamW and SGD with momentum still move the
        stored parameter).
    generator : torch.Generator or None, default=None
        Isolated RNG for ``reset_parameters``. ``None`` uses
        the default torch generator, bit-identical to omitting
        the argument. Not stored on the module; pass it again
        to ``reset_parameters`` to replay. Do not pass a seed
        integer.

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
        When ``constraint`` is set, this tensor is
        unconstrained; ``forward`` uses
        ``constraint(weight)``.
    source_index : torch.Tensor
        Int64 buffer of input columns, length ``nnz``.
        **Treat it as read-only:** like any PyTorch buffer it can
        be written to, and doing so rewires the layer without
        reinitializing it. Rebuild from the edgelist instead.
    target_index : torch.Tensor
        Int64 buffer of output rows, length ``nnz``, read-only in
        the same sense.
    constraint : torch.nn.Module or None
        The constructor ``constraint`` module, or ``None``.
    bias : nn.Parameter | None
        Trainable bias, or ``None`` when constructed with
        ``bias=False``.
    identity : str | None
        The constructor ``identity``, or ``None``.

    Raises
    ------
    Kpnn2Error
        If the indices are empty, not 1-D integers, mismatched in
        length, out of range, or duplicated as
        ``(source, target)`` pairs; if ``out_features`` /
        ``in_features`` are not positive ints; if ``identity`` is
        neither a ``str`` nor ``None``; if ``constraint`` is
        neither an ``nn.Module`` nor ``None``, or does not
        preserve the packed weight shape; if ``generator`` is
        neither a ``torch.Generator`` nor ``None``; and from
        ``load_state_dict`` when the checkpoint carries an index
        digest or identity that does not match this layer, in
        which case the weights are not loaded.

    See Also
    --------
    MaskedLinear : Dense ``(out_features, in_features)`` weight;
        the GEMM hatch when that rectangle fits.
    Hop : Supplies per-layer ``source_index`` / ``target_index``
        on a ``LayeredSpec``.
    AdjacencySpec : Supplies graph-wide ``source_index`` /
        ``target_index``; this layer takes those tuples, not the
        spec object.
    PackedMultiheadAttention : Attention over the same packed
        pairs, when the update is a contraction rather than one
        scalar per edge.
    scatter_hop_outputs : Split a transposed hop's concatenated
        output back onto source layers.
    torch.nn.Linear : Dense equivalent, and the reference for
        shapes, ``bias``, calling the module, and training.

    Notes
    -----
    Forward gathers ``x[..., source_index]``, multiplies by
    ``constraint(weight)`` when ``constraint`` is set (else
    ``weight``), and ``index_add``s into zeros of shape
    ``(..., out_features)``, adding ``bias`` when present. ``x``
    is an ordinary dense activation tensor. Nothing scatters into
    a dense ``(out, in)`` matrix, nothing imports
    ``torch.sparse``, and no tensor subclass is involved, so
    ``torch.compile(layer, fullgraph=True)`` traces it. Index
    buffers stay integer after ``.half()`` / bfloat16 /
    ``.double()``; ``weight`` and ``bias`` follow the module
    floating dtype like ``nn.Linear``. ``torch.autocast`` is
    unsupported: ``forward`` disables it and casts ``x`` to
    the parameter dtype so AMP cannot mix Half into
    ``index_add``. Cast the module with ``.to(dtype=...)``
    (or ``.half()`` / ``.double()``) instead.

    ``reset_parameters`` uses per-row packed degree as
    ``fan_in``, not full ``in_features``. A row with
    ``fan_in == 0`` has no packed weights and its bias stays 0;
    input nodes of an ``AdjacencySpec`` are exactly that case,
    and this layer does not invent identity connections for them.
    Optional ``generator`` isolates those draws from other torch
    RNG consumers.

    ``state_dict`` keys are ``weight``, optional ``bias``,
    ``source_index``, ``target_index``, ``index_digest``, and
    ``identity`` when the constructor was given one.
    ``index_digest`` is a 1-D CPU ``uint8`` tensor of length 32:
    the SHA-256 of the live index buffers' int64 C-contiguous
    bytes plus ``out_features`` and ``in_features`` as
    fixed-width integers, not a registered buffer. A missing
    digest is not an error, even with ``strict=True``. The
    digest catches same-shape rewiring. A rename that leaves
    the packed index pattern unchanged is caught by
    ``identity`` when callers pass ``spec.fingerprint``.
    ``copy.deepcopy`` works.

    ``transpose()`` is the tied-autoencoder helper: it swaps
    the index buffers and the feature sizes so packed slot
    ``i`` is still the same live edge, then shares or copies
    ``weight``. Bias is never tied. Do not reparse a reversed
    edgelist and assign ``dec.weight = enc.weight``: that
    permutes slots. On a hop whose source axis is several
    layers, split the transposed output with
    ``scatter_hop_outputs``.

    Examples
    --------
    An input feeding a two-node feedback core plus one output,
    stepped once over the shared state vector:

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
    >>> core = kpnn2.PackedLinear(
    ...     spec.source_index,
    ...     spec.target_index,
    ...     n,
    ...     n,
    ...     identity=spec.fingerprint,
    ... )
    >>> core.in_features, core.out_features, core.nnz
    (4, 4, 4)
    >>> state = torch.zeros(2, n)
    >>> state[:, spec.input_index] = torch.ones(2, 1)
    >>> state = torch.relu(core(state))
    >>> tuple(state.shape)
    (2, 4)
    >>> tuple(state[:, spec.output_index].shape)
    (2, 1)

    A layered hop uses the same constructor on that hop's
    packed indices:

    >>> layered = kpnn2.parse_layered(
    ...     pd.DataFrame(
    ...         {
    ...             "source": ["A", "H", "A"],
    ...             "target": ["H", "C", "C"],
    ...         }
    ...     )
    ... )
    >>> hop = layered.hops[1]
    >>> layer = kpnn2.PackedLinear(
    ...     hop.source_index,
    ...     hop.target_index,
    ...     hop.out_features,
    ...     hop.in_features,
    ... )
    >>> layer.nnz
    2
    """

    source_index: torch.Tensor
    target_index: torch.Tensor
    weight: nn.Parameter
    bias: nn.Parameter | None
    identity: str | None
    constraint: nn.Module | None

    def __init__(
        self,
        source_index: object,
        target_index: object,
        out_features: int,
        in_features: int,
        bias: bool = True,
        *,
        identity: str | None = None,
        constraint: nn.Module | None = None,
        generator: torch.Generator | None = None,
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
        self.identity = as_identity(identity)
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
        constraint_module = as_constraint(constraint)
        if constraint_module is not None:
            check_constraint_shape(
                constraint_module,
                self.weight,
            )
        self.constraint = constraint_module
        self.reset_parameters(generator)

    def reset_parameters(
        self,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        Initialize from per-row packed degree, not full width.

        For output row ``j``, ``fan_in`` is the number of packed
        edges with ``target_index == j`` (``bincount``,
        ``minlength=out_features``). Each live edge into that
        row, and ``bias[j]`` if present, is drawn uniformly from
        ``[-1 / sqrt(fan_in), 1 / sqrt(fan_in)]``. If
        ``fan_in == 0``, that row has no packed weights and
        ``bias[j]`` stays 0.

        Parameters
        ----------
        generator : torch.Generator or None, default=None
            Isolated RNG for these draws. ``None`` uses the
            default torch generator. Not stored on the module.
        """
        generator = as_generator(generator)
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
                        generator=generator,
                    )
                if self.bias is not None:
                    nn.init.uniform_(
                        self.bias[row : row + 1],
                        -bound,
                        bound,
                        generator=generator,
                    )

    def transpose(
        self,
        bias: bool = True,
        *,
        tie: bool = True,
        identity: str | None = None,
        generator: torch.Generator | None = None,
    ) -> "PackedLinear":
        """
        Return a packed layer that applies the same edges backwards.

        Packed slot ``i`` stays the same live edge: the 1-D
        ``weight`` is not permuted. The new layer reads the
        former output axis and writes the former input axis.
        That is the packed analogue of ``enc.weight.T`` for a
        tied autoencoder. Bias is never shared.

        Parameters
        ----------
        bias : bool, default=True
            If ``True``, the new layer gets its own bias of
            shape ``(in_features,)``, the original input
            width, initialized from the transposed packed
            degree. If ``False``, there is no bias. This
            layer's bias is not copied.
        tie : bool, default=True
            If ``True``, the returned layer's ``weight`` is
            this layer's ``weight`` ``nn.Parameter``. Gradients
            from both forwards accumulate there. If
            ``False``, copy the current values into a new
            Parameter.
        identity : str or None, default=None
            Checkpoint identity for the new layer, typically
            ``spec.fingerprint``. This layer's identity is
            not copied. ``None`` means the new layer does
            not claim an identity.
        generator : torch.Generator or None, default=None
            Forwarded to the inner ``PackedLinear``
            constructor. ``None`` uses the default torch
            generator. A full init still runs, then
            ``weight`` is replaced; discarded weight draws
            still advance this generator.

        Returns
        -------
        PackedLinear
            New module: ``source_index`` and ``target_index``
            swapped, ``in_features`` and ``out_features``
            swapped, same ``nnz``. ``constraint`` is a
            deepcopy of this layer's constraint, or ``None``.

        Raises
        ------
        Kpnn2Error
            If ``tie`` is not a ``bool``, or if constructing
            the new layer fails (see the constructor).

        Notes
        -----
        Do not parse a reversed edgelist and assign
        ``dec.weight = enc.weight``. Packed order is
        lexicographic by ``(source name, target name)`` of
        each spec, so those slots do not line up. This
        method keeps slot ``i`` as the same edge.

        On a layered hop, the transposed output is
        ``hop.in_features`` wide. Split it with
        ``scatter_hop_outputs`` when the hop reads several
        source layers; add those pieces into the caller's
        ``saved`` dict, because two reversed hops may write
        the same earlier layer. An ``AdjacencySpec`` map is
        already ``n``-wide; no gather or scatter.

        ``MaskedLinear`` has no packed slots: use
        ``F.linear(h, layer.weight.T, dec_bias)``.

        Examples
        --------
        Slot ``i`` is the same edge after a transpose, so
        sharing ``weight`` is the tied map:

        >>> import torch
        >>> import kpnn2
        >>> layer = kpnn2.PackedLinear(
        ...     [0, 1],
        ...     [0, 0],
        ...     1,
        ...     2,
        ...     bias=False,
        ... )
        >>> with torch.no_grad():
        ...     layer.weight[:] = torch.tensor([2.0, 3.0])
        >>> mirrored = layer.transpose(bias=False)
        >>> mirrored.weight is layer.weight
        True
        >>> mirrored.in_features, mirrored.out_features
        (1, 2)
        >>> layer(torch.tensor([[1.0, 4.0]])).tolist()
        [[14.0]]
        >>> mirrored(torch.tensor([[1.0]])).tolist()
        [[2.0, 3.0]]
        """
        if not isinstance(tie, bool):
            raise Kpnn2Error("'tie' must be True or False.")
        constraint = None
        if self.constraint is not None:
            constraint = copy.deepcopy(self.constraint)
        mirrored = PackedLinear(
            self.target_index,
            self.source_index,
            self.in_features,
            self.out_features,
            bias,
            identity=identity,
            constraint=constraint,
            generator=generator,
        )
        mirrored.to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        if tie:
            mirrored.weight = self.weight
        else:
            with torch.no_grad():
                mirrored.weight.copy_(self.weight)
        return mirrored

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

        ``contrib = x[..., source_index] * live_weight``, then
        ``index_add`` into zeros of shape
        ``(..., out_features)``. ``live_weight`` is
        ``constraint(weight)`` when ``constraint`` is set,
        otherwise ``weight``. Adds ``bias`` when present.
        Packed 1-D weights, one per live edge; not
        ``torch.sparse``; forward is ``index_add``.
        ``torch.autocast`` is unsupported: this path
        disables it and casts ``x`` to the parameter dtype.
        """
        with torch.autocast(
            device_type=x.device.type,
            enabled=False,
        ):
            weight = self.weight
            if self.constraint is not None:
                weight = self.constraint(weight)
            x = x.to(dtype=weight.dtype)
            contrib = x[..., self.source_index] * weight
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
