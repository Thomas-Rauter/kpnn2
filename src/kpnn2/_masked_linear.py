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
# kpnn2 0.1 stored the trainable tensor through torch parametrize.
_LEGACY_WEIGHT_KEY = "parametrizations.weight.original"


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
    ``PackedLinear``, which stores its weight the same way:
    ``weight`` is the trainable tensor and ``effective_weight()``
    is what ``forward`` uses. Initialization scales by per-row
    mask degree.

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
        tensor is multiplied with the weight as it is.
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
        Optional per-entry map on ``weight``, applied in
        ``forward`` **before** the mask. ``nn.Softplus()`` is the
        textbook non-negative edge reparametrization. The module
        must return a tensor of the same shape as ``weight``.
        ``reset_parameters`` writes ``weight``; it does not
        invert this map. ``PackedLinear`` takes the same
        argument. Mixed per-edge signs and frozen slots belong
        in this module, not in parse columns. A hard freeze is
        ``torch.where`` replacing those slots; a gradient hook
        that zeroes a slot is not a freeze (AdamW and SGD with
        momentum still move the stored tensor).
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
    weight : nn.Parameter
        The trainable tensor, shape
        ``(out_features, in_features)``, stored under the name
        ``weight`` as on ``nn.Linear``. Masked-out entries start
        at exactly 0; their gradient is always 0, so SGD,
        momentum, Adam, and weight decay keep them at 0. When
        ``constraint`` is set, this tensor is unconstrained.
        Read ``effective_weight()`` for the map ``forward``
        uses.
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
        ``F.linear(h, layer.effective_weight().T, dec_bias)``.
    gather_hop_inputs : Assembles the input tensor of a hop that
        reads more than one saved layer.
    torch.nn.Linear : Dense equivalent, and the reference for
        shapes, ``bias``, calling the module, and training.

    Notes
    -----
    Sizes come from ``mask.shape``; there are no separate size
    arguments. Forward is
    ``Y = F.linear(X, effective_weight(), bias)``, that is
    ``Y = X @ (C(W) ⊙ M).T + b`` with ``W`` the ``weight``
    parameter, ``C`` the ``constraint`` (the identity when
    omitted), and ``M`` the mask cast to ``W``'s dtype and
    device, so ``.half()``, bfloat16, and ``.double()`` work as
    on ``nn.Linear``. ``torch.autocast`` is unsupported:
    ``forward`` disables it and casts ``x`` to the parameter
    dtype. The multiply is dense on purpose, and ``X`` is an
    ordinary dense activation tensor. Nothing in the forward
    path is a tensor subclass, so
    ``torch.compile(layer, fullgraph=True)`` traces it.

    The mask is always applied last. A map you
    ``register_parametrization`` on ``weight`` yourself runs when
    ``weight`` is read, before ``constraint`` and before the
    mask, so it cannot resurrect a blocked entry. ``constraint``
    is the supported per-entry map for sign-constrained edges.

    ``weight`` is a plain ``nn.Parameter``, as on ``nn.Linear``
    and ``PackedLinear``:

    - ``state_dict`` keys are ``weight``, optional ``bias``,
      ``mask_digest``, and ``identity`` when the constructor was
      given one; ``mask`` stays out of it. ``mask_digest`` is a
      1-D CPU ``uint8`` tensor of length 32: the SHA-256 of the
      live mask's float32 C-contiguous bytes at save time, not a
      registered buffer. A missing digest is not an error, even
      with ``strict=True``. The digest catches same-shape
      rewiring. A rename that leaves the 0/1 pattern unchanged
      is caught by ``identity`` when callers pass
      ``spec.fingerprint``. A kpnn2 0.1 checkpoint, which stored
      the trainable tensor as ``parametrizations.weight.original``,
      still loads; its masked-out entries are set to 0 on load,
      which does not change the output.
    - Param-group filters match ``weight`` the same way on this
      layer, ``PackedLinear``, and ``nn.Linear``.
    - ``copy.deepcopy`` and pickling (``torch.save(model)``)
      work.
    - ``torch.nn.utils.prune`` works on ``weight``; its mask
      composes with the connectivity mask.

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

    ``weight`` is the trainable parameter; a blocked entry starts
    at 0 there and is 0 in the map ``forward`` uses:

    >>> [name for name, _ in layer.named_parameters()]
    ['weight']
    >>> bool(layer.weight[1, 0] == 0.0)
    True
    >>> bool(layer.effective_weight()[1, 0] == 0.0)
    True

    ``constraint`` is applied before the mask, so blocked
    entries stay zero even though ``softplus(0) > 0``:

    >>> layer = kpnn2.MaskedLinear(
    ...     mask,
    ...     bias=False,
    ...     constraint=torch.nn.Softplus(),
    ... )
    >>> bool(layer.effective_weight()[1, 0] == 0.0)
    True
    """

    weight: nn.Parameter
    bias: nn.Parameter | None
    identity: str | None
    constraint: nn.Module | None
    mask: torch.Tensor

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
        self.register_buffer(
            "mask",
            as_mask_tensor(mask),
            persistent=False,
        )
        if constraint is not None:
            check_constraint_shape(
                constraint,
                self.weight,
            )
        self.constraint = constraint
        self.reset_parameters(generator)

    def _trainable_weight(self) -> torch.Tensor:
        """
        Return the stored tensor behind ``weight``.

        ``weight`` itself, or the ``original`` of a
        parametrization the caller registered on it.
        """
        if parametrize.is_parametrized(
            self,
            "weight",
        ):
            holders = cast(
                nn.ModuleDict,
                self.parametrizations,
            )
            return cast(
                parametrize.ParametrizationList,
                holders["weight"],
            ).original
        return self.weight

    def effective_weight(self) -> torch.Tensor:
        """
        Return the ``(out_features, in_features)`` map ``forward`` uses.

        ``constraint(weight)`` when ``constraint`` is set (else
        ``weight``), times the mask cast to that tensor's dtype
        and device. Recomputed on every call and differentiable,
        so it is the tensor to read, export, or penalize as the
        layer's live edge weights. ``PackedLinear`` has the same
        method.

        Returns
        -------
        torch.Tensor
            The effective weight. Blocked entries are 0.
        """
        weight = self.weight
        if self.constraint is not None:
            weight = self.constraint(weight)
        return weight * self.mask.to(
            dtype=weight.dtype,
            device=weight.device,
        )

    def reset_parameters(
        self,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        Initialize from per-row mask degree, not full width.

        This is the difference from
        ``torch.nn.Linear.reset_parameters``. For output row
        ``j``, ``fan_in`` is the number of ones in ``mask[j]``.
        The live entries of that row of ``weight`` (and
        ``bias[j]``, if present) are drawn uniformly from
        ``[-1 / sqrt(fan_in), 1 / sqrt(fan_in)]``. Masked-out
        entries are 0. If ``fan_in == 0``, the row and bias
        entry stay 0.

        Rows are drawn one at a time, each as a full row that is
        then masked, so the draws match earlier releases and no
        full-size temporary is allocated.

        Parameters
        ----------
        generator : torch.Generator or None, default=None
            Isolated RNG for these draws. ``None`` uses the
            default torch generator. Not stored on the module.
        """
        generator = as_generator(generator)
        weight = self._trainable_weight()
        with torch.no_grad():
            weight.zero_()
            if self.bias is not None:
                self.bias.zero_()
            degrees = self.mask.sum(dim=1).trunc().tolist()
            for row, degree in enumerate(degrees):
                if degree <= 0:
                    continue
                bound = 1.0 / math.sqrt(degree)
                nn.init.uniform_(
                    weight[row],
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
            weight.masked_fill_(
                self.mask == 0,
                0.0,
            )

    def _apply(
        self,
        fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> "MaskedLinear":
        out = super()._apply(
            fn,
            *args,
            **kwargs,
        )
        stored = self._buffers.get("mask")
        if stored is not None and stored.dtype != torch.float32:
            self._buffers["mask"] = stored.to(dtype=torch.float32)
        return out

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
        legacy = prefix + _LEGACY_WEIGHT_KEY
        migrated = (
            legacy in state_dict
            and prefix + "weight" not in state_dict
            and not parametrize.is_parametrized(
                self,
                "weight",
            )
        )
        if migrated:
            state_dict[prefix + "weight"] = state_dict.pop(legacy)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if migrated:
            with torch.no_grad():
                self.weight.masked_fill_(
                    self.mask == 0,
                    0.0,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ``F.linear`` of ``x`` with ``effective_weight()``.

        The constraint (if any) and then the mask are applied to
        ``weight`` here, on every call. The stored ``mask``
        remains float32 and is cast to the parameter dtype, so
        ``.half()``, bfloat16, and ``.double()`` match
        ``nn.Linear``. ``torch.autocast`` is unsupported: this
        path disables it and casts ``x`` to the parameter dtype.
        """
        with torch.autocast(
            device_type=x.device.type,
            enabled=False,
        ):
            weight = self.effective_weight()
            x = x.to(dtype=weight.dtype)
            return F.linear(
                x,
                weight,
                self.bias,
            )
