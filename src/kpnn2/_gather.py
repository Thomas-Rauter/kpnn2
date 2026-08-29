"""
Source axis assembly for one hop.
"""

from collections.abc import Mapping

import torch

from ._errors import Kpnn2Error
from ._spec import Hop


def gather_hop_inputs(
    saved: Mapping[int, torch.Tensor],
    hop: Hop,
) -> torch.Tensor:
    """
    Concatenate the layer tensors one hop reads, in mask-column
    order.

    Call this in ``forward()`` just before
    ``MaskedLinear(hop.mask)``. It sits between hops. It does
    not inject values into the previous layer, does not pick
    skip nodes by name, and holds no weights.

    A hop mask's columns are **whole** source layers laid side
    by side (``hop.source_layers``). This function builds that
    tensor from ``saved``:

    - Adjacent hop (no skips): one source, the layer below.
      That saved tensor is returned as-is, with no copy.
    - Hop with skips: the previous layer plus older layers,
      concatenated on the last axis. The hop mask, not this
      gather, zeros columns that are not edges. Example: skip
      ``A → C`` with ``[A, B]`` then ``[H]`` yields
      ``[A, B, H]``.

    Store every layer you produce in ``saved``. A missing
    source **layer** (not a missing node) raises
    ``Kpnn2Error`` instead of silently dropping those edges.
    Unused keys are ignored; saved tensors are not modified.

    Parameters
    ----------
    saved : mapping of int to torch.Tensor
        Layer index to that layer's activation. Width of
        ``saved[i]`` must be ``layer_dims[i]``. Only the layers
        in ``hop.source_layers`` are read.
    hop : Hop
        The hop about to be applied, from ``spec.hops``.

    Returns
    -------
    torch.Tensor
        Shape ``(..., hop.mask.shape[1])``, ready for
        ``MaskedLinear(hop.mask)``.

    Raises
    ------
    Kpnn2Error
        If ``saved`` is not a mapping or ``hop`` is not a
        ``Hop``; a needed layer is missing from ``saved`` or is
        not a tensor; a saved tensor has the wrong number of
        units; or the parts disagree on dtype or device.

    Examples
    --------
    The hop into ``C`` reads layers 0 and 1, so its input is
    two columns wide:

    >>> import pandas as pd
    >>> import torch
    >>> import kpnn2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H", "A"],
    ...         "target": ["H", "C", "C"],
    ...     }
    ... )
    >>> spec = kpnn2.parse_layered(edgelist)
    >>> saved = {
    ...     0: torch.tensor([[2.0]]),
    ...     1: torch.tensor([[5.0]]),
    ... }
    >>> x = kpnn2.gather_hop_inputs(
    ...     saved,
    ...     spec.hops[1],
    ... )
    >>> x.tolist()
    [[2.0, 5.0]]

    Forgetting to store a layer raises instead of silently
    dropping the edges that read it:

    >>> kpnn2.gather_hop_inputs(
    ...     {1: torch.tensor([[5.0]])},
    ...     spec.hops[1],
    ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    Kpnn2Error: saved is missing layer 0. ...
    """
    if not isinstance(hop, Hop):
        raise Kpnn2Error("'hop' must be a Hop from spec.hops.")
    if not isinstance(saved, Mapping):
        raise Kpnn2Error("'saved' must be a mapping of layer index to tensor.")

    parts: list[torch.Tensor] = []
    for layer, n_units in zip(
        hop.source_layers,
        hop.source_dims,
    ):
        if layer not in saved:
            raise Kpnn2Error(
                f"saved is missing layer {layer}. The hop into "
                f"layer {hop.target_layer} reads layers "
                f"{list(hop.source_layers)}."
            )
        tensor = saved[layer]
        if not isinstance(tensor, torch.Tensor):
            raise Kpnn2Error(f"saved[{layer}] must be a torch.Tensor.")
        if tensor.ndim < 1:
            raise Kpnn2Error(
                f"saved[{layer}] has the wrong number of units. "
                f"Expected {n_units}, got a 0-dimensional tensor."
            )
        if tensor.shape[-1] != n_units:
            raise Kpnn2Error(
                f"saved[{layer}] has the wrong number of units. "
                f"Expected {n_units}, got {tensor.shape[-1]}."
            )
        parts.append(tensor)

    first = parts[0]
    for layer, tensor in zip(
        hop.source_layers,
        parts,
    ):
        if tensor.dtype != first.dtype:
            raise Kpnn2Error(
                "Saved layers must share a dtype. "
                f"saved[{hop.source_layers[0]}] is {first.dtype} "
                f"and saved[{layer}] is {tensor.dtype}."
            )
        if tensor.device != first.device:
            raise Kpnn2Error(
                "Saved layers must share a device. "
                f"saved[{hop.source_layers[0]}] is on "
                f"{first.device} and saved[{layer}] is on "
                f"{tensor.device}."
            )

    if len(parts) == 1:
        return first
    return torch.cat(
        parts,
        dim=-1,
    )
