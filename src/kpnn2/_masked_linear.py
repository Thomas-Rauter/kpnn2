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

from ._errors import Kpnn2Error
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
    Connectivity factor behind ``MaskedLinear.weight``.

    Registered on ``MaskedLinear`` with
    ``torch.nn.utils.parametrize.register_parametrization``, so
    ``layer.weight`` is ``original * mask`` and the trainable
    tensor stays at ``layer.parametrizations.weight.original``.

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
        Multiply ``weight`` by a dtype/device-cast ``mask``.
        """
        return weight * self.mask.to(
            dtype=weight.dtype,
            device=weight.device,
        )

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


class MaskedLinear(nn.Module):
    """
    Affine hop with a fixed connectivity mask.

    Same job as ``torch.nn.Linear``: call ``layer(x)`` in an
    ``nn.Module``. This is not a subclass of ``Linear``. For
    shapes, ``bias``, calling the module, and training, see the
    PyTorch docs for ``torch.nn.Linear``.

    The mask is applied with
    ``torch.nn.utils.parametrize.register_parametrization``, so
    ``layer.weight`` is the **effective** masked weight
    (recomputed, not an ``nn.Parameter``). The trainable
    tensor lives at
    ``layer.parametrizations.weight.original``.
    ``model.parameters()`` includes that tensor. A
    param-group filter that uses ``"weight" in name``
    matches it; ``name.endswith(".weight")`` does not.

    Parameters
    ----------
    mask : torch.Tensor
        Connectivity of shape ``(out_features, in_features)``.
        Non-finite values are not special-cased: the tensor is
        stored as float32 and multiplied with the trainable
        weight.
    bias : bool, default=True
        If ``True``, learn a bias of shape ``(out_features,)``.
        If ``False``, there is no bias.

    Attributes
    ----------
    in_features : int
        Number of input columns, ``mask.shape[1]``.
    out_features : int
        Number of output columns, ``mask.shape[0]``.
    weight : torch.Tensor
        Effective weight, the product of
        ``parametrizations.weight.original`` and a ``mask`` cast
        to that tensor's dtype and device. Recomputed on every
        access, so it is **not** a parameter: in-place writes to
        it are discarded. Assigning
        (``layer.weight = w``, under ``torch.no_grad()``) copies
        ``w`` into ``original``, where the mask hides the entries
        it zeroes.
    parametrizations : nn.ModuleDict
        Holds ``parametrizations.weight.original``, the trainable
        ``nn.Parameter`` of shape
        ``(out_features, in_features)``, and the mask module.
        Masked-out entries can be nonzero in ``original``; they
        never reach the output.
    mask : torch.Tensor
        Float32 buffer, same shape as the constructor ``mask``.
        Not trained and not saved in ``state_dict``. An
        independent copy of the constructor tensor (including
        ``spec.hops[i].mask``), so later edits to that tensor do
        not reach this layer. Stays float32 after ``.half()`` /
        bfloat16 / ``.double()``. **Treat it as read-only:**
        like any PyTorch buffer it can be written to, and doing
        so silently rewires the layer. Rebuild from the edgelist
        instead.
    bias : nn.Parameter | None
        Trainable bias, or ``None`` when constructed with
        ``bias=False``.

    Raises
    ------
    Kpnn2Error
        If ``mask`` is not a ``torch.Tensor``, or is not 2-D.

    Notes
    -----
    Construct with ``mask``; sizes come from ``mask.shape``.
    ``mask`` is an ordinary float32 buffer: not trained, omitted
    from ``state_dict``, and not a tensor subclass, so nothing
    custom runs per operation and ``torch.compile`` sees plain
    tensors. Nothing prevents writing to it; treat it as
    read-only and rebuild from the edgelist to change wiring.
    Module dtype casts do not change the stored mask dtype.

    Forward is ``Y = F.linear(X, self.weight, self.bias)``, and
    ``self.weight`` is ``original * mask.to(dtype=original.dtype,
    device=original.device)``, so ``.half()``, bfloat16, and
    ``.double()`` work like ``nn.Linear``. There are no extra
    edge-weight constraints beyond the mask.

    Registering a parametrization is what PyTorch does for
    "the effective weight is a function of a stored parameter",
    and it comes with that machinery's conventions:

    - ``state_dict`` keys are ``parametrizations.weight.original``,
      optional ``bias``, and ``mask_digest``; ``mask`` stays out
      of it. ``"weight" in name`` matches that parameter key;
      ``name.endswith(".weight")`` misses it. Default
      ``Adam(model.parameters())`` needs no filter.
      ``mask_digest`` is a 1-D CPU ``uint8`` tensor of
      length 32: the SHA-256 of the live mask's float32
      C-contiguous bytes at save time, not a registered buffer.
      ``load_state_dict`` raises ``Kpnn2Error`` when a present
      digest does not match this layer's mask, and does not
      load the weights. A missing digest is not an error, even
      with ``strict=True``. The digest catches same-shape
      rewiring, not a rename that leaves the 0/1 pattern
      unchanged (that is ``spec.fingerprint``).
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
    not full ``in_features``. Typical construction:
    ``MaskedLinear(spec.hops[i].mask)``. Because a hop mask
    carries every parent of its target, including skip parents,
    that per-row degree is the unit's real fan-in.

    Examples
    --------
    One hop whose second output is connected only to the second
    input:

    >>> import torch
    >>> import kpnn2 as k2
    >>> mask = torch.tensor(
    ...     [
    ...         [1.0, 1.0],
    ...         [0.0, 1.0],
    ...     ]
    ... )
    >>> layer = k2.MaskedLinear(
    ...     mask,
    ...     bias=False,
    ... )
    >>> layer.in_features, layer.out_features
    (2, 2)
    >>> tuple(layer.mask.shape)
    (2, 2)
    >>> x = torch.ones(3, 2)
    >>> y = layer(x)
    >>> tuple(y.shape)
    (3, 2)

    ``layer.weight`` is the masked product; the trainable tensor
    is one level down:

    >>> bool(layer.weight[1, 0] == 0.0)
    True
    >>> tuple(layer.parametrizations.weight.original.shape)
    (2, 2)

    A zero in the mask blocks that input column:

    >>> mask = torch.tensor([[1.0, 0.0]])
    >>> layer = k2.MaskedLinear(
    ...     mask,
    ...     bias=False,
    ... )
    >>> a = layer(torch.tensor([[1.0, 0.0]]))
    >>> b = layer(torch.tensor([[1.0, 99.0]]))
    >>> torch.equal(a, b)
    True
    """

    weight: torch.Tensor
    bias: nn.Parameter | None

    def __init__(
        self,
        mask: torch.Tensor,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(mask, torch.Tensor):
            raise Kpnn2Error("'mask' must be a torch.Tensor.")
        if mask.ndim != 2:
            raise Kpnn2Error(
                "'mask' must be a 2-dimensional tensor of shape "
                "(out_features, in_features)."
            )

        out_features, in_features = mask.shape
        self.in_features = in_features
        self.out_features = out_features

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
        self.reset_parameters()

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
    def _original_weight(self) -> torch.Tensor:
        return self._weight_parametrizations().original

    def reset_parameters(self) -> None:
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
        """
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
                )
                if self.bias is not None:
                    nn.init.uniform_(
                        self.bias[row : row + 1],
                        -bound,
                        bound,
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

        ``self.weight`` comes from the mask parametrization: the
        trainable tensor times a ``mask`` cast to its dtype and
        device. The stored ``mask`` remains float32, so
        ``.half()``, bfloat16, and ``.double()`` match
        ``nn.Linear``.
        """
        return F.linear(
            x,
            self.weight,
            self.bias,
        )
