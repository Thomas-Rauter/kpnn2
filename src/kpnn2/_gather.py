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
    Concatenate the layer activations one hop reads.

    A hop mask spans every layer that feeds its target, so its
    input is those layers side by side in ``hop.source_layers``
    order. This builds that tensor and checks it against
    ``hop``, which is the step that makes a missing activation
    an error instead of a quietly dropped edge.

    Parameters
    ----------
    saved : mapping of int to torch.Tensor
        Layer index to that layer's activation. Width of
        ``saved[i]`` must be ``layer_dims[i]``. Only the layers
        in ``hop.source_layers`` are read, and they are not
        modified.
    hop : Hop
        The hop about to be applied, from ``spec.hops``.

    Returns
    -------
    torch.Tensor
        Shape ``(..., hop.mask.shape[1])``, ready for
        ``MaskedLinear(hop.mask)``. An adjacent-only hop reads
        one layer, and then the saved tensor itself is returned
        without a copy.

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
    >>> import kpnn2 as k2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H", "A"],
    ...         "target": ["H", "C", "C"],
    ...     }
    ... )
    >>> spec = k2.parse_layered(edgelist)
    >>> saved = {
    ...     0: torch.tensor([[2.0]]),
    ...     1: torch.tensor([[5.0]]),
    ... }
    >>> x = k2.gather_hop_inputs(
    ...     saved,
    ...     spec.hops[1],
    ... )
    >>> x.tolist()
    [[2.0, 5.0]]

    Forgetting to store a layer raises instead of silently
    dropping the edges that read it:

    >>> k2.gather_hop_inputs(
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
