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
    Concatenate the saved layer tensors one hop reads, in source-column order.

    A hop's source columns are whole source layers laid side by
    side, so the tensor feeding ``PackedLinear`` or
    ``MaskedLinear`` on that hop is those layers concatenated.
    Call it between hops, keeping every layer you produce in
    ``saved``; a forgotten layer raises rather than dropping the
    edges that read it. Inputs are never modified, and a
    single-source hop returns the saved tensor itself.

    Parameters
    ----------
    saved : mapping of int to torch.Tensor
        Layer depth to that layer's activation, as far as
        ``forward()`` has produced them. ``saved[i]`` is shaped
        ``(..., layer_dims[i])``; the leading dimensions are the
        caller's, typically a batch. Only the layers in
        ``hop.source_layers`` are read, so extra keys are
        ignored, and those layers must share a dtype and a
        device. Neither the mapping nor its tensors are copied
        or modified.
    hop : Hop
        The hop about to be applied, one entry of ``spec.hops``.
        Its ``source_layers`` and ``source_dims`` decide which
        keys are read, in which order, and how wide each one
        must be; the packed indices are not used here.

    Returns
    -------
    torch.Tensor
        The hop's source axis, shape
        ``(..., hop.in_features)``, columns in concatenated
        source-unit order, dtype and device of the saved layers.
        ``source_nodes`` is one name per node, so it is shorter
        than this axis when a source node is wider than 1. When
        the hop reads a single layer, which is every hop with no
        skip parents and ``hops[0]`` always, this is that saved
        tensor itself rather than a copy, so writing into it
        writes into the saved activation.

    Raises
    ------
    Kpnn2Error
        If ``saved`` is not a mapping or ``hop`` is not a
        ``Hop``; a layer in ``hop.source_layers`` is absent from
        ``saved`` or is not a tensor; a saved tensor is
        0-dimensional or is the wrong number of units wide; or
        the source layers disagree on dtype or device.
    RuntimeError
        Propagated from ``torch.cat`` when two source layers
        disagree in a dimension other than the last, such as a
        batch size.

    See Also
    --------
    PackedLinear : Applies the hop to the tensor returned here.
    MaskedLinear : Dense hatch via ``hop.to_mask()``.
    Hop : The record that fixes the source layers and the
        concatenated column order this follows.
    align_inputs : Builds the layer-0 tensor that seeds
        ``saved``.

    Notes
    -----
    Nothing here is graph-aware: this holds no weights, does not
    inject values into the previous layer, and does not pick
    skip sources by name. Columns that are not edges stay in the
    concatenated tensor; ``PackedLinear`` never reads them, and
    ``MaskedLinear(hop.to_mask())`` zeros them. With the skip
    ``A -> C`` reaching past layer 1, layer 0 ``[A, B]`` and
    layer 1 ``[H]`` gather to ``[A, B, H]``, and the ``A -> C``
    weight is the packed pair whose source column is ``A``.

    An ``AdjacencySpec`` has no hops and is not accepted; in
    that layout every edge is already a packed index pair.

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
