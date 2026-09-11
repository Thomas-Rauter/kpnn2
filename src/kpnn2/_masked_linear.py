"""
Masked linear layer for edgelist-defined connectivity.
"""

import hashlib
import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize

from ._constraint import as_constraint, check_constraint_shape
from ._errors import Kpnn2Error
from ._generator import as_generator
from ._identity import as_identity, check_identity, save_identity
from ._mask_tensor import as_mask_tensor

_MASK_DIGEST_KEY = "mask_digest"


def _mask_digest(mask: torch.Tensor) -> torch.Tensor:
    payload = mask.detach().cpu().contiguous().numpy().tobytes()
    digest = hashlib.sha256(payload).digest()
    return torch.tensor(
        tuple(digest),
        dtype=torch.uint8,
    )


def _mask_digest_matches(
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


class _MaskParametrization(nn.Module):
    """
    Connectivity buffer behind ``MaskedLinear.weight``.

    Registered first on ``MaskedLinear`` with
    ``torch.nn.utils.parametrize.register_parametrization``.
    ``forward`` is the identity: later maps must not run after a
    mask multiply, or a non-zero-preserving map (for example
    ``softplus``) would resurrect blocked edges.
    ``_MaskedParametrizationList`` multiplies by this mask after
    every registered map, so ``layer.weight`` stays masked.

    ``right_inverse`` is the identity on a copy: assigning
    ``layer.weight = w`` stores ``w`` unchanged in ``original``,
    where the mask hides the entries it zeroes. It is not a true
    inverse, because the product is not surjective.
    """

    mask: torch.Tensor

    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer(
            "mask",
            mask,
            persistent=False,
        )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Return ``weight`` unchanged. The list applies the mask.
        """
        return weight

    def right_inverse(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Return an independent copy of ``weight``.
        """
        return weight.clone()

    def _apply(
        self,
        fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> "_MaskParametrization":
        out = super()._apply(
            fn,
            *args,
            **kwargs,
        )
        stored = self._buffers.get("mask")
        if stored is not None and stored.dtype != torch.float32:
            self._buffers["mask"] = stored.to(dtype=torch.float32)
        return cast("_MaskParametrization", out)


class _ConstraintParametrization(nn.Module):
    """
    Optional per-entry map applied before the connectivity mask.

    ``right_inverse`` is a clone, like ``_MaskParametrization``:
    assigning ``layer.weight = w`` writes ``w`` into
    ``original``, and the constraint is applied on read.
    """

    def __init__(self, constraint: nn.Module) -> None:
        super().__init__()
        self.constraint = constraint

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Apply the stored ``constraint`` module.
        """
        return self.constraint(weight)

    def right_inverse(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Return an independent copy of ``weight``.
        """
        return weight.clone()


class _MaskedParametrizationList(parametrize.ParametrizationList):
    """
    Parametrization list that always remasks last.

    ``register_parametrization`` appends. Connectivity is not a
    composable inner map: a blocked entry must stay zero in
    ``layer.weight`` no matter what else is stacked. Index 0
    holds the mask; this ``forward`` multiplies by it after every
    other registered map.
    """

    def forward(self) -> torch.Tensor:
        """
        Apply registered maps, then the connectivity mask.
        """
        weight = super().forward()
        mask = cast(
            _MaskParametrization,
            self[0],
        ).mask
        return weight * mask.to(
            dtype=weight.dtype,
            device=weight.device,
        )


class MaskedLinear(nn.Module):
    """
    Affine hop whose connectivity is fixed by an edgelist mask.

    One hop of a knowledge-primed network: an ``nn.Linear``-style
    layer (call ``layer(x)``; not a subclass) in which only edges
    present in the prior-knowledge graph can carry weight, so
    absent edges need no hand-zeroing after every optimizer step.
    Build one from ``spec.hops[i].to_mask()`` or
    ``spec.to_mask()``, fed by ``gather_hop_inputs`` on a
    layered hop. The large-n path on the same indices is
    ``PackedLinear``. The masked ``weight`` is recomputed rather
    than stored, and initialization scales by per-row mask
    degree.

    Parameters
    ----------
    mask : torch.Tensor
        Connectivity of shape ``(out_features, in_features)``,
        usually ``spec.hops[i].to_mask()`` or ``spec.to_mask()``. A
        nonzero entry ``[j, k]`` lets input column ``k`` reach
        output row ``j``; a zero blocks it for the life of the
        layer. Stored as an independent float32 copy, so later
        writes to the tensor passed in do not reach this layer.
        Non-finite values are not special-cased: the stored
        tensor is multiplied with the trainable weight as it is.
    bias : bool, default=True
        If ``True``, learn a bias of shape ``(out_features,)``.
        If ``False``, there is no bias.
    identity : str or None, default=None
        Opaque checkpoint identity, typically
        ``spec.fingerprint``. Stored in ``state_dict`` next to
        ``mask_digest`` as a 1-D CPU ``uint8`` tensor of the
        UTF-8 bytes. ``load_state_dict`` raises ``Kpnn2Error``
        when a present identity does not match this layer, and
        does not load the weights. A missing identity is not an
        error, even with ``strict=True``. ``None`` means this
        layer does not claim an identity.
    constraint : torch.nn.Module or None, default=None
        Optional per-entry map on the unconstrained weight,
        applied **before** the mask. ``nn.Softplus()`` is the
        textbook non-negative edge reparametrization. The
        module must return a tensor of the same shape as the
        unconstrained weight. ``reset_parameters`` writes that
        unconstrained tensor; it does not invert this map.
        ``PackedLinear`` takes the same argument. Mixed
        per-edge signs and frozen slots belong in this
        module, not in parse columns. A hard freeze is
        ``torch.where`` replacing those slots; a gradient
        hook that zeroes a slot is not a freeze (AdamW and
        SGD with momentum still move the stored tensor).
    generator : torch.Generator or None, default=None
        Isolated RNG for ``reset_parameters``. ``None`` uses
        the default torch generator, bit-identical to omitting
        the argument. Not stored on the module; pass it again
        to ``reset_parameters`` to replay. Do not pass a seed
        integer.

    Attributes
    ----------
    in_features : int
        Number of input columns, ``mask.shape[1]``.
    out_features : int
        Number of output columns, ``mask.shape[0]``.
    weight : torch.Tensor
        Effective weight: the constructor ``constraint``, if
        any, then any later parametrizations, then a ``mask``
        cast to that tensor's dtype and device. Recomputed on
        every access, so it is **not** a parameter: in-place
        writes to it are discarded. Assigning
        (``layer.weight = w``, under ``torch.no_grad()``)
        copies ``w`` into ``original``; the mask (and
        ``constraint``, if set) are applied on read.
    parametrizations : nn.ModuleDict
        Holds ``parametrizations.weight.original``, the trainable
        ``nn.Parameter`` of shape
        ``(out_features, in_features)``, the mask module, and
        the constructor ``constraint`` when one was given.
        Masked-out entries can be nonzero in ``original``; they
        never reach the output. ``model.parameters()`` includes
        that tensor. A param-group filter that uses
        ``"weight" in name`` matches it;
        ``name.endswith(".weight")`` does not.
    mask : torch.Tensor
        Float32 buffer, same shape as the constructor ``mask``.
        Not trained and not saved in ``state_dict``. Stays
        float32 after ``.half()`` / bfloat16 / ``.double()``.
        **Treat it as read-only:** like any PyTorch buffer it can
        be written to, and doing so silently rewires the layer.
        Rebuild from the edgelist instead.
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
        If ``mask`` is not a ``torch.Tensor`` or is not 2-D; if
        ``identity`` is neither a ``str`` nor ``None``; if
        ``constraint`` is neither an ``nn.Module`` nor
        ``None``, or does not preserve the weight shape; if
        ``generator`` is neither a ``torch.Generator`` nor
        ``None``; and from ``load_state_dict`` when the
        checkpoint carries a mask digest or identity that
        does not match this layer; the weights are then not
        loaded.

    See Also
    --------
    PackedLinear : One trainable scalar per live edge, for graphs
        whose dense ``(out_features, in_features)`` weight would
        not fit in RAM. Tied decode is
        ``PackedLinear.transpose``; on this layer use
        ``F.linear(h, weight.T, dec_bias)``.
    gather_hop_inputs : Assembles the input tensor of a hop that
        reads more than one saved layer.
    torch.nn.Linear : Dense equivalent, and the reference for
        shapes, ``bias``, calling the module, and training.

    Notes
    -----
    Sizes come from ``mask.shape``; there are no separate size
    arguments. Forward is ``Y = F.linear(X, weight, bias)``
    with ``weight`` the effective tensor above, equivalently
    ``Y = X @ (C(W) ⊙ M).T + b`` when ``constraint`` is ``C``
    (the identity when omitted), so ``.half()``, bfloat16, and
    ``.double()`` work as on ``nn.Linear``. ``torch.autocast``
    is unsupported: ``forward`` disables it and casts ``x`` to
    the parameter dtype. The multiply is dense on purpose, and
    ``X`` is an ordinary dense activation tensor. Nothing in
    the forward path is a tensor subclass, so
    ``torch.compile(layer, fullgraph=True)`` traces it,
    parametrization included.

    The mask is the outermost factor of ``layer.weight``, even
    if the caller later
    ``register_parametrization``s another map. A blocked entry
    stays zero there and in the forward pass. ``constraint`` is
    the supported inner map for sign-constrained edges; do not
    stack ``softplus`` after the mask yourself.

    The unconstrained tensor is stored with
    ``torch.nn.utils.parametrize.register_parametrization``, the
    PyTorch mechanism for "the effective weight is a function of
    a stored parameter", and it comes with that machinery's
    conventions:

    - ``state_dict`` keys are ``parametrizations.weight.original``,
      optional ``bias``, ``mask_digest``, and ``identity`` when
      the constructor was given one; ``mask`` stays out of it.
      ``mask_digest`` is a 1-D CPU ``uint8`` tensor of length 32:
      the SHA-256 of the live mask's float32 C-contiguous bytes
      at save time, not a registered buffer. A missing digest is
      not an error, even with ``strict=True``. The digest
      catches same-shape rewiring. A rename that leaves the 0/1
      pattern unchanged is caught by ``identity`` when callers
      pass ``spec.fingerprint``.
    - ``repr`` reports ``ParametrizedMaskedLinear``, because
      PyTorch swaps in a subclass to install the ``weight``
      property. ``isinstance(layer, MaskedLinear)`` is still
      ``True``.
    - ``copy.deepcopy`` works; pickling the module object does
      not, here as for any parametrized module. Save
      ``state_dict``, not the module.
    - Utilities that need ``weight`` to be a raw
      ``nn.Parameter``, such as ``torch.nn.utils.prune``, reject
      a parametrized ``weight`` here exactly as they do on a
      parametrized ``nn.Linear``. Point them at
      ``layer.parametrizations.weight`` under the name
      ``original``; their mask then composes with the
      connectivity mask.
    - Do not call ``parametrize.remove_parametrizations`` on
      ``weight``: that drops the mask and leaves a dense layer.

    ``reset_parameters`` uses per-row mask degree as ``fan_in``,
    not full ``in_features``. Because a hop carries every parent
    of its target, including skip parents, that per-row degree
    is the unit's real fan-in. Optional ``generator`` isolates
    those draws from other torch RNG consumers.

    Examples
    --------
    One hop whose second output is connected only to the second
    input:

    >>> import torch
    >>> import kpnn2
    >>> mask = torch.tensor(
    ...     [
    ...         [1.0, 1.0],
    ...         [0.0, 1.0],
    ...     ]
    ... )
    >>> layer = kpnn2.MaskedLinear(
    ...     mask,
    ...     bias=False,
    ... )
    >>> layer.in_features, layer.out_features
    (2, 2)
    >>> y = layer(torch.ones(3, 2))
    >>> tuple(y.shape)
    (3, 2)

    ``layer.weight`` is the masked product; the trainable tensor
    is one level down:

    >>> bool(layer.weight[1, 0] == 0.0)
    True
    >>> tuple(layer.parametrizations.weight.original.shape)
    (2, 2)

    ``constraint`` is applied before the mask, so blocked
    entries stay zero:

    >>> layer = kpnn2.MaskedLinear(
    ...     mask,
    ...     bias=False,
    ...     constraint=torch.nn.Softplus(),
    ... )
    >>> bool(layer.weight[1, 0] == 0.0)
    True
    """

    weight: torch.Tensor
    bias: nn.Parameter | None
    identity: str | None

    def __init__(
        self,
        mask: torch.Tensor,
        bias: bool = True,
        *,
        identity: str | None = None,
        constraint: nn.Module | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(mask, torch.Tensor):
            raise Kpnn2Error("'mask' must be a torch.Tensor.")
        if mask.ndim != 2:
            raise Kpnn2Error(
                "'mask' must be a 2-dimensional tensor of shape "
                "(out_features, in_features)."
            )
        constraint = as_constraint(constraint)

        out_features, in_features = mask.shape
        self.in_features = in_features
        self.out_features = out_features
        self.identity = as_identity(identity)

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter(
                "bias",
                None,
            )

        parametrize.register_parametrization(
            self,
            "weight",
            _MaskParametrization(as_mask_tensor(mask)),
        )
        self._weight_parametrizations().__class__ = _MaskedParametrizationList
        if constraint is not None:
            check_constraint_shape(
                constraint,
                self._original_weight,
            )
            parametrize.register_parametrization(
                self,
                "weight",
                _ConstraintParametrization(constraint),
            )
        self.reset_parameters(generator)

    def _weight_parametrizations(self) -> parametrize.ParametrizationList:
        holders = cast(
            nn.ModuleDict,
            self.parametrizations,
        )
        return cast(
            parametrize.ParametrizationList,
            holders["weight"],
        )

    def _mask_module(self) -> _MaskParametrization:
        return cast(
            _MaskParametrization,
            self._weight_parametrizations()[0],
        )

    @property
    def mask(self) -> torch.Tensor:
        """
        The float32 connectivity buffer, read-only by contract.
        """
        return self._mask_module().mask

    @mask.setter
    def mask(self, value: torch.Tensor) -> None:
        self._mask_module().mask = value

    @property
    def constraint(self) -> nn.Module | None:
        """
        The constructor ``constraint`` module, or ``None``.
        """
        holder = self._weight_parametrizations()
        if len(holder) < 2:
            return None
        extra = holder[1]
        if not isinstance(extra, _ConstraintParametrization):
            return None
        return extra.constraint

    @property
    def _original_weight(self) -> torch.Tensor:
        return self._weight_parametrizations().original

    def reset_parameters(
        self,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        Initialize from per-row mask degree, not full width.

        This is the difference from
        ``torch.nn.Linear.reset_parameters``. For output row
        ``j``, ``fan_in`` is the number of ones in ``mask[j]``.
        That row of ``parametrizations.weight.original`` (and
        ``bias[j]``, if present) is drawn uniformly from
        ``[-1 / sqrt(fan_in), 1 / sqrt(fan_in)]``. If
        ``fan_in == 0``, the row and bias entry stay 0.

        Degrees are counted in one pass over ``mask``, so no
        per-row device synchronization happens here. Rows are
        then drawn one at a time, which writes straight into
        the trainable tensor without a full-size temporary.

        Parameters
        ----------
        generator : torch.Generator or None, default=None
            Isolated RNG for these draws. ``None`` uses the
            default torch generator. Not stored on the module.
        """
        generator = as_generator(generator)
        original = self._original_weight
        with torch.no_grad():
            original.zero_()
            if self.bias is not None:
                self.bias.zero_()
            degrees = self.mask.sum(dim=1).trunc().tolist()
            for row, degree in enumerate(degrees):
                if degree <= 0:
                    continue
                bound = 1.0 / math.sqrt(degree)
                nn.init.uniform_(
                    original[row],
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

    def extra_repr(self) -> str:
        """
        Sizes and bias flag, worded as in ``nn.Linear``.
        """
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
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
        destination[prefix + _MASK_DIGEST_KEY] = _mask_digest(self.mask)
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
        key = prefix + _MASK_DIGEST_KEY
        saved = state_dict.pop(key, None)
        if saved is not None:
            current = _mask_digest(self.mask)
            if not _mask_digest_matches(
                saved,
                current,
            ):
                raise Kpnn2Error(
                    "The checkpoint mask does not match this layer."
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
        ``F.linear`` of ``x`` with the effective ``weight``.

        ``self.weight`` is the effective tensor: constructor
        ``constraint`` (if any), then any later parametrizations,
        then a ``mask`` cast to the trainable tensor's dtype and
        device. The stored ``mask`` remains float32, so
        ``.half()``, bfloat16, and ``.double()`` match
        ``nn.Linear``. ``torch.autocast`` is unsupported:
        this path disables it and casts ``x`` to the parameter
        dtype.
        """
        with torch.autocast(
            device_type=x.device.type,
            enabled=False,
        ):
            weight = self.weight
            x = x.to(dtype=weight.dtype)
            return F.linear(
                x,
                weight,
                self.bias,
            )
