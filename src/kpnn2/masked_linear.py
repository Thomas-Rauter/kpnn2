"""
Masked linear layer for edgelist-defined connectivity.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ._frozen_mask import freeze_mask
from .errors import Kpnn2Error


class MaskedLinear(nn.Module):
    """
    Affine hop with a frozen connectivity mask.

    Same job as ``torch.nn.Linear``: call ``layer(x)`` in an
    ``nn.Module``. This is not a subclass of ``Linear``. For
    shapes, ``bias``, calling the module, and training, see the
    PyTorch docs for ``torch.nn.Linear``.

    Parameters
    ----------
    mask : torch.Tensor
        Connectivity of shape ``(out_features, in_features)``.
        Non-finite values are not special-cased: the tensor is
        stored as float32 and multiplied with ``raw_weight``.
    bias : bool, default=True
        If ``True``, learn a bias of shape ``(out_features,)``.
        If ``False``, there is no bias.

    Attributes
    ----------
    in_features : int
        Number of input columns, ``mask.shape[1]``.
    out_features : int
        Number of output columns, ``mask.shape[0]``.
    mask : torch.Tensor
        Frozen float32 buffer, same shape as the constructor
        ``mask``. Not trained, not saved in ``state_dict``, and
        not writable in place. Connectivity comes only from the
        tensor passed to the constructor (typically
        ``spec.masks[i]``).
    raw_weight : nn.Parameter
        Trainable weight of shape ``(out_features, in_features)``.
        Masked-out entries can be nonzero in this tensor; they are
        zeroed in the forward product ``raw_weight * mask``.
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
    ``mask`` is a float32 buffer, is not trained, and is omitted
    from ``state_dict``. In-place writes and replacement are
    rejected. The trainable tensor is ``raw_weight``. Forward is
    ``Y = F.linear(X, raw_weight * mask, bias)``. There are no
    extra edge-weight constraints beyond the mask.

    ``reset_parameters`` uses per-row mask degree as ``fan_in``,
    not full ``in_features``. Typical construction:
    ``MaskedLinear(spec.masks[i])``.

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

    mask: torch.Tensor
    raw_weight: nn.Parameter
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

        self.raw_weight = nn.Parameter(
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
            freeze_mask(mask),
            persistent=False,
        )
        self._mask_locked = True
        self.reset_parameters()

    def __setattr__(
        self,
        name: str,
        value: Any,
    ) -> None:
        if name == "mask" and getattr(self, "_mask_locked", False):
            raise Kpnn2Error(
                "MaskedLinear.mask is read-only. Rebuild the "
                "layer from the edgelist GraphSpec."
            )
        super().__setattr__(
            name,
            value,
        )

    def _apply(
        self,
        fn,
        *args,
        **kwargs,
    ):
        out = super()._apply(
            fn,
            *args,
            **kwargs,
        )
        stored = self._buffers.get("mask")
        if stored is not None:
            self._buffers["mask"] = freeze_mask(stored)
        return out

    def reset_parameters(self) -> None:
        """
        Initialize from per-row mask degree, not full width.

        This is the difference from
        ``torch.nn.Linear.reset_parameters``. For output row
        ``j``, ``fan_in`` is the number of ones in ``mask[j]``.
        That row of ``raw_weight`` (and ``bias[j]``, if present)
        is drawn uniformly from
        ``[-1 / sqrt(fan_in), 1 / sqrt(fan_in)]``. If
        ``fan_in == 0``, the row and bias entry stay 0.
        """
        with torch.no_grad():
            self.raw_weight.zero_()
            if self.bias is not None:
                self.bias.zero_()
            for row in range(self.out_features):
                fan_in = int(self.mask[row].sum().item())
                if fan_in == 0:
                    continue
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(
                    self.raw_weight[row],
                    -bound,
                    bound,
                )
                if self.bias is not None:
                    nn.init.uniform_(
                        self.bias[row : row + 1],
                        -bound,
                        bound,
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ``F.linear(x, raw_weight * mask, bias)``.
        """
        return F.linear(
            x,
            self.raw_weight * self.mask,
            self.bias,
        )
